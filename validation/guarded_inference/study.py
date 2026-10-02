"""Fixed-index paired calibration study. Development runs cannot pass confirmation gates.

Each worker owns explicit indexes in its append-only SQLite ledger. Data and permutation
seeds depend on study cell/index, never scheduling or completion order. No failures are replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import time
import traceback
import importlib.metadata
import platform
import numpy as np
import pandas as pd
from scipy import stats
from survival_toolkit.marker_evaluation import prepare_marker_cohort
from survival_toolkit.marker_diagnostics import diagnose_markers
from survival_toolkit.marker_screen import CoxScoreScreen, MaxTAccumulator, fit_cox_null, residualize

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=Path(__file__).with_name("protocol.json")
SOURCE_FILES=["src/survival_toolkit/clinical_basis.py", "src/survival_toolkit/marker_diagnostics.py",
              "src/survival_toolkit/marker_evaluation.py", "src/survival_toolkit/marker_screen.py",
              "src/survival_toolkit/encoding.py", "src/survival_toolkit/analysis.py",
              "validation/guarded_inference/study.py", "validation/guarded_inference/protocol.json"]


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def dataset(condition,n,p,seed,index):
    names=json.loads(PROTOCOL.read_text())["conditions"]
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,p,names.index(condition),index,0]))
    z=rng.normal(size=n); noise=rng.normal(size=(n,p))
    x=.8*z[:,None]+noise
    if condition=="independent": x=noise.copy()
    if condition in {"nonlinear_marker","nonph_nonlinear","nonlinear_clinical_risk"}: x=z[:,None]**2+noise
    if condition in {"heteroskedastic","nonph_heteroskedastic"}: x=.8*z[:,None]+np.exp(.8*z[:,None])*noise
    if condition=="correlated": x=.8*z[:,None]+np.sqrt(.7)*rng.normal(size=(n,1))+np.sqrt(.3)*noise
    eta=.8*z
    if condition=="nonlinear_clinical_risk": eta+=.4*(z*z-1)
    signal=np.zeros(p,dtype=bool)
    if condition in {"partial_weak","partial_strong"}:
        signal[:5]=True; eta+=(.2 if condition=="partial_weak" else .4)*noise[:,:5].sum(axis=1)
    exponential=rng.exponential(size=n)
    if condition.startswith("nonph_"):
        first=.06*np.exp(.8*z); second=.06*np.exp(-.8*z)
        t=np.where(exponential<=10*first,exponential/first,10+(exponential-10*first)/second)
    else: t=exponential/(.06*np.exp(eta))
    rate=.04*np.exp(.8*z) if condition=="z_dependent_censoring" else .04
    c=rng.exponential(size=n)/rate
    frame=pd.DataFrame({"time":np.minimum(t,c),"event":(t<=c).astype(int),"Z":z})
    marker_names=[f"X{j:04d}" for j in range(p)]
    frame=pd.concat([frame,pd.DataFrame(x,columns=marker_names)],axis=1)
    perm_seed=np.random.SeedSequence([seed,n,p,names.index(condition),index,1])
    return frame,marker_names,signal,perm_seed


def calculation(cohort,signal,perm_seed,permutations):
    z=cohort.clinical
    if cohort.clinical_error or z is None or z.shape[1]==0:
        raise ValueError("requested clinical basis unavailable; no marginal fallback")
    null=fit_cox_null(cohort.time,cohort.event,z)
    if not null.converged or not np.isfinite(null.beta).all() or not np.isfinite(null.covariance).all():
        raise ValueError("clinical calculation nonconvergence or nonfinite/aliased coefficients")
    block=cohort.markers
    screen=CoxScoreScreen(cohort.time,cohort.event,null=null,Z=z)
    observed=screen.statistics(block).chi2
    if not np.isfinite(observed).all(): raise ValueError("unestimable marker score")
    accumulator=MaxTAccumulator(observed)
    sources=residualize(block,z)
    rng=np.random.default_rng(perm_seed)
    for start in range(0,permutations,16):
        perms=[rng.permutation(len(block)) for _ in range(min(16,permutations-start))]
        accumulator.update(screen.permuted_chi2(sources,perms))
    p=accumulator.result().p_step_down
    rejected=p<=.05
    return {"null_rejected":bool(rejected[~signal].any()),
            "true_rejected":int(rejected[signal].sum()), "n_true":int(signal.sum()),
            "minimum_null_p":float(np.min(p[~signal])), "failure":False}


def paired(condition,n,p,seed,index,permutations):
    frame,names,signal,perm_seed=dataset(condition,n,p,seed,index)
    outputs=[]
    linear=None
    for basis,method in (("linear","guarded_linear"),("restricted_cubic_spline","guarded_spline")):
        try:
            cohort=prepare_marker_cohort(frame,time_column="time",event_column="event",marker_columns=names,
                                          clinical_columns=["Z"],clinical_basis=basis)
            calculated=calculation(cohort,signal,perm_seed,permutations)
            if basis=="linear": linear=dict(calculated)
            # A failed diagnostic must not erase an otherwise estimable raw calculation.
            # It withholds inference and remains in both paired denominators.
            try:
                diagnostic=diagnose_markers(cohort,cohort.markers)
                diagnostic_failed=False
            except Exception as exc:
                diagnostic={"allowed":False,"reasons":["diagnostic_failed: "+type(exc).__name__+": "+str(exc)]}
                diagnostic_failed=True
            outputs.append({"method":method,"allowed":diagnostic["allowed"],
                            "diagnostic_failed":diagnostic_failed,"reasons":diagnostic["reasons"],
                            "raw_null_rejected":calculated["null_rejected"],"raw_true_rejected":calculated["true_rejected"],
                            **calculated,"null_rejected":bool(diagnostic["allowed"] and calculated["null_rejected"]),
                            "true_rejected":calculated["true_rejected"] if diagnostic["allowed"] else 0})
        except Exception as exc:
            # The fixed replicate remains present, including code/numerical failures. Never silently retry it.
            error={"failure":True,"error":type(exc).__name__+": "+str(exc),"traceback":traceback.format_exc(),
                   "allowed":False,"null_rejected":False,"true_rejected":0,"n_true":int(signal.sum())}
            outputs.append({"method":method,**error})
            if basis=="linear": linear=dict(error)
    return [{"method":"legacy_linear","allowed":not linear["failure"],**linear},*outputs]


def verify_freeze(path):
    manifest=json.loads(Path(path).read_text())
    if manifest.get("source_hashes")!=hashes(): raise ValueError("Frozen study sources changed; confirmation is refused.")
    reference=manifest.get("reference_verification")
    if not reference or not reference.get("passed"): raise ValueError("Independent R reference must pass before confirmation.")
    if manifest.get("development_review_complete") is not True: raise ValueError("Development review was not sealed.")
    current_versions={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","statsmodels")}
    if manifest.get("package_versions") != current_versions or manifest.get("python_version") != platform.python_version():
        raise ValueError("Confirmation runtime differs from the frozen numerical environment")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    protocol=json.loads(PROTOCOL.read_text()); condition=args.condition
    if condition not in protocol["conditions"]: raise ValueError("Unspecified condition")
    if args.start<0 or args.stop<=args.start or args.workers<1 or not 0<=args.worker<args.workers:
        raise ValueError("Invalid fixed-index ownership")
    configuration=dict(stage=args.stage,condition=condition,n=args.n,p=args.p,permutations=args.permutations)
    if args.stage=="development": seed=protocol["development_seed"]; freeze_hash=None
    else:
        if not args.freeze: raise ValueError("A frozen manifest is required for confirmation")
        freeze_hash=verify_freeze(args.freeze)
        if args.permutations!=protocol["permutations"]: raise ValueError("Confirmation permutation count cannot be changed")
        if args.stage=="main":
            cell=protocol["main"]; seed=protocol["main_seed"]
        elif args.stage=="extension":
            cells=[c for c in protocol["extensions"] if (c["n"],c["p"])==(args.n,args.p)]
            if not cells: raise ValueError("Unspecified extension cell")
            cell=cells[0];seed=protocol["extension_seed"]
        else:
            cell=protocol["large_marker"];seed=protocol["extension_seed"]
            if condition not in cell["conditions"]: raise ValueError("Unspecified large-marker condition")
        if (args.n,args.p)!=(cell["n"],cell["p"]) or args.stop>cell["replicates_per_condition"]:
            raise ValueError("Confirmation cell/index outside frozen specification")
    configuration.update(seed=seed,freeze_hash=freeze_hash,source_hashes=hashes())
    configuration["package_versions"]={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","statsmodels")}
    configuration["python_version"]=platform.python_version()
    args.output=Path(args.output);args.output.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(args.output)
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS replicates (idx INTEGER PRIMARY KEY, host TEXT, started REAL, elapsed REAL, result TEXT)")
    serialized=json.dumps(configuration,sort_keys=True)
    existing=db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()
    if existing and existing[0]!=serialized: raise ValueError("Ledger configuration differs; no overwrite is allowed")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('configuration', ?)",(serialized,));db.commit()
    for index in range(args.start,args.stop):
        if index%args.workers!=args.worker: continue
        if db.execute("SELECT 1 FROM replicates WHERE idx=?",(index,)).fetchone(): continue
        started=time.time(); results=paired(condition,args.n,args.p,seed,index,args.permutations)
        db.execute("INSERT INTO replicates VALUES (?,?,?,?,?)",(index,socket.gethostname(),started,time.time()-started,json.dumps(results)))
        db.commit()
    count=db.execute("SELECT count(*) FROM replicates").fetchone()[0]
    print(json.dumps({"ledger":str(args.output),"records":count,"worker":args.worker,"condition":condition}))
    db.close()


def cp_interval(hits,total):
    if total==0: return [None,None]
    return [0. if hits==0 else float(stats.beta.ppf(.025,hits,total-hits+1)),
            1. if hits==total else float(stats.beta.ppf(.975,hits+1,total-hits))]


def summary(paths,output):
    records={}; configs={}
    for path in paths:
        db=sqlite3.connect(path);config=json.loads(db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()[0]);db.close()
        cell=(config["stage"],config["condition"],config["n"],config["p"])
        if cell in configs and configs[cell]!=config: raise ValueError("Inconsistent configurations for one cell")
        configs[cell]=config
        db=sqlite3.connect(path)
        for idx,result in db.execute("SELECT idx,result FROM replicates"):
            if (cell,idx) in records: raise ValueError("Duplicate index ownership across ledgers")
            records[(cell,idx)]=json.loads(result)
        db.close()
    rows=[];protocol=json.loads(PROTOCOL.read_text())
    for cell,config in configs.items():
        stage,condition,n,p=cell;planned=(protocol["main"]["replicates_per_condition"] if stage=="main" else
             protocol["large_marker"]["replicates_per_condition"] if stage=="large" else
             protocol["extensions"][0]["replicates_per_condition"] if stage=="extension" else sum(1 for c,i in records if c==cell))
        for method in protocol["methods"]:
            data=[next(r for r in result if r["method"]==method) for (c,i),result in records.items() if c==cell]
            missing=planned-len(data);failed=sum(d["failure"] for d in data);fail=missing+failed
            allowed=[d for d in data if d["allowed"] and not d["failure"]]
            hits=sum(d["null_rejected"] for d in data);raw_hits=sum(d.get("raw_null_rejected",d["null_rejected"]) for d in data)
            conditional=sum(d["null_rejected"] for d in allowed)
            def upper(h,N): return None if N==0 else 1. if h==N else float(stats.beta.ppf(.95,h+1,N-h))
            power=np.array([d["true_rejected"]/d["n_true"] if d["n_true"] else 0. for d in data])
            power_mean=float(power.sum()/planned) if condition.startswith("partial_") else None
            half=1.96*float(np.std(power,ddof=1))/np.sqrt(planned) if len(power)>1 else None
            rows.append(dict(stage=stage,condition=condition,n=n,p=p,method=method,planned=planned,completed=len(data),
                failures=failed,diagnostic_exceptions=sum(d.get("diagnostic_failed",False) for d in data),
                missing=missing,allowed=len(allowed),allowed_fraction=len(allowed)/planned,
                fwer=hits/planned,fwer_raw_lower=raw_hits/planned,fwer_raw_upper=(raw_hits+fail)/planned,
                raw_fwer_mc95=cp_interval(raw_hits,planned),raw_fwer_worst_case_mc95=cp_interval(raw_hits+fail,planned),
                fwer_failure_bounds=[hits/planned,(hits+fail)/planned],fwer_mc95=cp_interval(hits,planned),
                fwer_worst_case_upper95=upper(hits+fail,planned),
                conditional_fwer=None if not allowed else conditional/len(allowed),conditional_mc95=cp_interval(conditional,len(allowed)),
                conditional_upper95=upper(conditional,len(allowed)),withhold_fraction=(planned-len(allowed))/planned,
                healthy_false_alarm_fraction=(len(data)-len(allowed)-failed)/planned if condition in {"independent","linear","correlated","z_dependent_censoring"} else None,
                power=power_mean,power_mc95=None if power_mean is None or half is None else [max(0.,power_mean-half),min(1.,power_mean+half)],
                power_failure_bounds=None if power_mean is None else [power_mean,min(1.,power_mean+fail/planned)],
                source_hashes=config["source_hashes"],freeze_hash=config["freeze_hash"]))
    # Never label a development pilot a successful confirmation study.
    for row in rows:
        supported=row["condition"] in protocol["supported"].get(row["method"],[])
        row["engineering_status"]="not_applicable" if not supported else "pending"
        if supported and row["stage"]=="main" and not row["missing"]:
            baseline=next(r for r in rows if all(r[k]==row[k] for k in ("stage","condition","n","p")) and r["method"]=="legacy_linear")
            power_gate=(row["power"] is None or row["power"]>=.9*(baseline["power_failure_bounds"][1]))
            availability=(row["power"] is not None or row["allowed_fraction"]>=.8)
            passed=(row["fwer_worst_case_upper95"]<=.06 and row["conditional_upper95"] is not None and
                    row["conditional_upper95"]<=.06 and availability and power_gate)
            row["engineering_status"]="passed" if passed else "failed_exploratory_only"
    expected_cells={("main",c,180,30) for c in protocol["conditions"]}
    expected_cells.update(("extension",c,e["n"],e["p"]) for c in protocol["conditions"] for e in protocol["extensions"])
    expected_cells.update(("large",c,180,3000) for c in protocol["large_marker"]["conditions"])
    absent_cells=[list(c) for c in sorted(expected_cells-set(configs))]
    release_status={}
    for method,conditions in protocol["supported"].items():
        required=[r for r in rows if r["stage"]=="main" and r["method"]==method and r["condition"] in conditions]
        release_status[method]=("failed_exploratory_only" if any(r["engineering_status"]=="failed_exploratory_only" for r in required)
                               else "passed_supported_conditions_only" if len(required)==len(conditions) and all(r["engineering_status"]=="passed" for r in required)
                               else "pending")
    result={"protocol":protocol,"summaries":rows,
            "model_release_status":release_status,
            "planned_confirmation_datasets":85500, "absent_confirmation_cells":absent_cells,
            "complete":not absent_cells and all(not r["missing"] for r in rows),
            "note":"Development intervals are descriptive; confirmation remains pending until every prespecified cell/index is accounted for."}
    Path(output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"cells":len(configs),"records":len(records),"output":str(output)}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest="command",required=True)
    worker=commands.add_parser("run");worker.add_argument("--stage",choices=["development","main","extension","large"],required=True)
    worker.add_argument("--condition",required=True);worker.add_argument("--n",type=int,default=180);worker.add_argument("--p",type=int,default=30)
    worker.add_argument("--permutations",type=int,default=999);worker.add_argument("--start",type=int,default=0);worker.add_argument("--stop",type=int,required=True)
    worker.add_argument("--workers",type=int,default=1);worker.add_argument("--worker",type=int,default=0);worker.add_argument("--output",required=True);worker.add_argument("--freeze")
    aggregate=commands.add_parser("summarize");aggregate.add_argument("ledgers",nargs="+");aggregate.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="run":
        if args.workers<1 or not 0<=args.worker<args.workers or not 0<=args.start<args.stop: parser.error("Invalid fixed-index ownership")
        run(args)
    else: summary(args.ledgers,args.output)

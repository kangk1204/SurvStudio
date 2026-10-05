"""Fixed-index v3 development and confirmation. No replacement or promotion by launch."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import sqlite3
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
spec = importlib.util.spec_from_file_location("v2_fixed_kernel", ROOT / "validation/guarded_inference/study.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)
from survival_toolkit.marker_evaluation import prepare_marker_cohort
from survival_toolkit.marker_diagnostics import diagnose_markers as diagnose_v2
from survival_toolkit.marker_bootstrap import diagnose_markers as diagnose_v3


def protocol():
    return json.loads(PROTOCOL.read_text())


def hashes():
    files = [*kernel.SOURCE_FILES, "src/survival_toolkit/marker_bootstrap.py",
        "src/survival_toolkit/marker_evaluation_v3.py", "src/survival_toolkit/data/marker_bootstrap_policy.json",
        "validation/publication_v3/study.py", "validation/publication_v3/protocol.json",
        "validation/publication_v3/reference.R", "validation/publication_v3/verify_reference.py",
        "validation/publication_v3/control.py", "validation/publication_v3/execute.py",
        "validation/publication_v3/confirmation_audit.py", "validation/publication_v3/verify_reference_full.py",
        "validation/publication_v3/verify_aggregate_reference.py", "validation/publication_v3/aggregate_reference.R",
        "src/survival_toolkit/marker_qualification.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(set(files))}


def environment():
    return {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n)
        for n in ("numpy", "pandas", "scipy", "statsmodels")}}


def dataset(condition, n, p, seed, index):
    cfg = protocol()
    if condition in cfg["conditions"]:
        frame, names, signal, perm = kernel.dataset(condition, n, p, seed, index)
        return frame, names, signal, perm, ["Z"]
    number = len(cfg["conditions"]) + cfg["stress_conditions"].index(condition)
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, p, number, index, 0]))
    z = rng.normal(size=n)
    noise = rng.normal(size=(n, p))
    clinical = ["Z"]
    frame = pd.DataFrame({"Z": z})
    mean = .8*z[:, None]
    eta = .8*z
    def lognormal():
        return (np.exp(rng.normal(size=(n,p))) - np.exp(.5)) / np.sqrt(np.exp(2)-np.exp(1))
    if condition == "t5_residual": noise = rng.standard_t(5, size=(n,p))*np.sqrt(3/5)
    elif condition == "lognormal_residual": noise = lognormal()
    elif condition == "uniform_clinical":
        z = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n)
        frame["Z"] = z; mean = .8*z[:,None]; eta=.8*z
    elif condition == "block_correlated":
        common = rng.normal(size=(n,(p+9)//10))
        noise = np.sqrt(.85)*common[:,np.arange(p)//10] + np.sqrt(.15)*noise
    elif condition == "two_clinical":
        z2 = .3*z + np.sqrt(1-.3**2)*rng.normal(size=n)
        frame = pd.DataFrame({"Z1":z,"Z2":z2}); clinical=["Z1","Z2"]
        mean=(.8*z+.4*z2)[:,None]; eta=.8*z-.5*z2
    elif condition == "sine_mean": mean=mean+.7*np.sin(z[:,None])
    elif condition == "step_variance": noise*=np.where(z<=0,1.,2.)[:,None]
    elif condition == "conditional_shape": noise=np.where(z[:,None]<=0,noise,lognormal())
    else: raise ValueError("Unknown stress condition")
    t=rng.exponential(size=n)/(.06*np.exp(eta));c=rng.exponential(size=n)/.04
    frame["time"]=np.minimum(t,c);frame["event"]=(t<=c).astype(int)
    names=[f"X{j:04d}" for j in range(p)]
    frame=pd.concat([frame,pd.DataFrame(mean+noise,columns=names)],axis=1)
    return frame,names,np.zeros(p,bool),np.random.SeedSequence([seed,n,p,number,index,1]),clinical


def cells(stage):
    cfg=protocol()
    if stage=="cost":
        return [dict(n=n,p=p,replicates=cfg[stage]["replicates_per_cell"],conditions=[cfg[stage]["condition"]]) for n,p in cfg[stage]["cells"]]
    if stage in {"screen","selection"}:
        return [dict(n=n,p=p,replicates=cfg[stage]["replicates_per_condition"],conditions=cfg["conditions"]) for n,p in cfg[stage]["cells"]]
    return [cell for cell in cfg["confirmation"] if cell["stage"]==stage]


def paired(stage, condition, n, p, seed, index, selected=None):
    cfg=protocol(); frame,names,signal,perm,clinical=dataset(condition,n,p,seed,index)
    is_development=stage in {"cost","screen","selection"}
    candidates=cfg["candidates"] if is_development else [selected]
    draws=cfg["screen_draws"] if stage=="screen" else cfg["diagnostic_draws"]
    condition_index=(cfg["conditions"]+cfg["stress_conditions"]).index(condition)
    diagnostic_seed=int(np.random.SeedSequence([seed,n,p,condition_index,index,2]).generate_state(1)[0])
    results=[]
    for basis,suffix in (("linear","linear"),("restricted_cubic_spline","spline")):
        method_names=["v2_"+suffix]+(["candidate_A_"+suffix,"candidate_B_"+suffix] if is_development else ["v3_"+suffix])
        try:
            prepared=prepare_marker_cohort(frame,time_column="time",event_column="event",marker_columns=names,
                clinical_columns=clinical,clinical_basis=basis)
            begin=time.perf_counter();raw=kernel.calculation(prepared,signal,perm,cfg["permutations"])
            raw_seconds=time.perf_counter()-begin
        except Exception as exc:
            for name in (["legacy_linear"] if suffix=="linear" else [])+method_names:
                results.append(dict(method=name,failure=True,allowed=False,null_rejected=False,true_rejected=0,
                    n_true=int(signal.sum()),error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc()))
            continue
        if suffix=="linear": results.append(dict(method="legacy_linear",allowed=True,raw_seconds=raw_seconds,**raw))
        for name,candidate in zip(method_names,[None,*candidates]):
            begin=time.perf_counter()
            try:
                diagnostic=(diagnose_v2(prepared,prepared.markers) if candidate is None else
                    diagnose_v3(prepared,prepared.markers,candidate=candidate,bootstrap_seed=diagnostic_seed,draws=draws))
                allowed=bool(diagnostic["allowed"])
                reasons=Counter(r.split(":",1)[0] for r in diagnostic["reasons"])
                failed=any("failed" in key or "unavailable" in key or "rank_deficient" in key for key in reasons)
                results.append(dict(method=name,failure=False,diagnostic_failure=failed,allowed=allowed,
                    null_rejected=bool(allowed and raw["null_rejected"]),true_rejected=raw["true_rejected"] if allowed else 0,
                    raw_null_rejected=raw["null_rejected"],raw_true_rejected=raw["true_rejected"],n_true=raw["n_true"],
                    reasons=dict(reasons),bootstrap=diagnostic.get("bootstrap"),raw_seconds=raw_seconds,
                    diagnostic_seconds=time.perf_counter()-begin))
            except Exception as exc:
                results.append(dict(method=name,failure=False,diagnostic_failure=True,allowed=False,
                    null_rejected=False,true_rejected=0,raw_null_rejected=raw["null_rejected"],raw_true_rejected=raw["true_rejected"],
                    n_true=raw["n_true"],reasons={"diagnostic_exception":1},error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc()))
    return results


def run(args):
    cfg=protocol();confirmation=args.stage not in {"cost","screen","selection"}
    if not 0 <= args.owner < args.owners: raise ValueError("Invalid fixed-index owner")
    selected=None; freeze_sha=None
    if confirmation:
        if not args.freeze: raise ValueError("Confirmation requires a sealed source/reference/selection manifest")
        manifest=json.loads(Path(args.freeze).read_text())
        sealed=datetime.fromisoformat(manifest["sealed_at_utc"])
        if sealed.tzinfo is None or sealed.utcoffset().total_seconds()!=0:
            raise ValueError("Seal requires absolute UTC time")
        if sealed.astimezone(timezone(timedelta(hours=9))).date().isoformat()>cfg["freeze_deadline"]:
            raise ValueError("Late source seal is not eligible for confirmation")
        if manifest.get("freeze_deadline")!=cfg["freeze_deadline"] or manifest.get("source_ci_passed") is not True:
            raise ValueError("Dated, source-bound CI seal required")
        if manifest["source_hashes"]!=hashes() or manifest["environment"]!=environment() or manifest.get("reference_passed") is not True or manifest.get("selection_eligible") is not True:
            raise ValueError("Confirmation freeze/source/environment/eligibility mismatch")
        selected=manifest["candidate"];freeze_sha=hashlib.sha256(Path(args.freeze).read_bytes()).hexdigest()
    seed=cfg[("extension" if args.stage=="large" else args.stage)+"_seed"]
    config=dict(stage=args.stage,seed=seed,owner=args.owner,owners=args.owners,host=socket.gethostname(),
        source_hashes=hashes(),environment=environment(),freeze_sha256=freeze_sha,candidate=selected)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(output) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS replicates(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,elapsed REAL,result TEXT,PRIMARY KEY(condition,n,p,idx))")
        db.execute("CREATE TABLE IF NOT EXISTS attempts(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,started_utc TEXT NOT NULL,ended_utc TEXT,status TEXT NOT NULL,error TEXT,PRIMARY KEY(condition,n,p,idx))")
        existing=db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()
        encoded=json.dumps(config,sort_keys=True)
        if existing and existing[0]!=encoded: raise ValueError("Immutable ledger source/configuration changed")
        db.execute("INSERT OR IGNORE INTO metadata VALUES('configuration',?)",(encoded,));db.commit()
        for cell in cells(args.stage):
            for condition in cell["conditions"]:
                for index in range(cell["replicates"]):
                    if index % args.owners != args.owner: continue
                    key=(condition,cell["n"],cell["p"],index)
                    if db.execute("SELECT 1 FROM replicates WHERE condition=? AND n=? AND p=? AND idx=?",key).fetchone(): continue
                    if db.execute("SELECT 1 FROM attempts WHERE condition=? AND n=? AND p=? AND idx=?",key).fetchone():
                        raise ValueError("Unresolved prior attempt; no automatic replacement")
                    db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,NULL,'started',NULL)",(*key,datetime.now(timezone.utc).isoformat()))
                    db.commit()
                    started=time.perf_counter()
                    try:
                        result=paired(args.stage,condition,cell["n"],cell["p"],seed,index,selected)
                        db.execute("INSERT INTO replicates VALUES(?,?,?,?,?,?)",(*key,time.perf_counter()-started,json.dumps(result)))
                        state="completed_with_scientific_failure" if any(r.get("failure") or r.get("diagnostic_failure") for r in result) else "completed"
                        db.execute("UPDATE attempts SET ended_utc=?,status=? WHERE condition=? AND n=? AND p=? AND idx=?",
                                   (datetime.now(timezone.utc).isoformat(),state,*key))
                    except BaseException as exc:
                        db.rollback()
                        db.execute("UPDATE attempts SET ended_utc=?,status='interrupted',error=? WHERE condition=? AND n=? AND p=? AND idx=?",
                                   (datetime.now(timezone.utc).isoformat(),type(exc).__name__+": "+str(exc),*key))
                        db.commit()
                        raise
                    db.commit()
        print(json.dumps({"stage":args.stage,"owner":args.owner,"records":db.execute("SELECT count(*) FROM replicates").fetchone()[0]}))


def binomial(hits,total,side):
    if not total: return None
    if side=="upper": return 1. if hits==total else float(stats.beta.ppf(.95,hits+1,total-hits))
    return 0. if hits==0 else float(stats.beta.ppf(.05,hits,total-hits+1))


def paired_ratio(values,baseline,seed):
    a=np.asarray(values,float);b=np.asarray(baseline,float)
    if not len(a) or b.mean()==0: return None
    rng=np.random.default_rng(seed);ratios=[];undefined=0
    for begin in range(0,9999,128):
        indices=rng.integers(0,len(a),size=(min(128,9999-begin),len(a)))
        denominator=b[indices].mean(axis=1)
        undefined+=int(np.sum(denominator==0))
        ratios.extend(np.divide(a[indices].mean(axis=1),denominator,out=np.zeros(len(indices)),where=denominator>0))
    return {"point":float(a.mean()/b.mean()),"lower95":float(np.quantile(ratios,.05)),
        "mc95":[float(np.quantile(ratios,.025)),None if undefined else float(np.quantile(ratios,.975))],
        "undefined_denominator_draws":undefined,"zero_denominator_policy":"conservative lower bound; upper interval undefined" if undefined else "not needed"}


def paired_differences(data,baseline,planned,seed):
    """Descriptive paired intervals; unresolved pairs retain full [-1,1] bounds."""
    pairs=[(a,b) for a,b in zip(data,baseline) if not a["failure"] and not b["failure"]]
    if not pairs: return {"complete_pairs":0,"unresolved_pairs":planned,"differences":{}}
    def values(row):
        return [float(row["null_rejected"]),float(row["allowed"]),row["true_rejected"]/row["n_true"] if row["n_true"] else 0.]
    differences=np.asarray([np.subtract(values(a),values(b)) for a,b in pairs])
    rng=np.random.default_rng(seed);draws=[]
    for begin in range(0,9999,128):
        indices=rng.integers(0,len(pairs),size=(min(128,9999-begin),len(pairs)))
        draws.extend(differences[indices].mean(axis=1))
    draws=np.asarray(draws);unresolved=planned-len(pairs);result={}
    for j,name in enumerate(("fwer","allowed_fraction","power")):
        mean=float(differences[:,j].mean());weight=len(pairs)/planned;radius=unresolved/planned
        result[name]={"point_complete_pairs":mean,"mc95_complete_pairs":np.quantile(draws[:,j],[.025,.975]).tolist(),
            "all_planned_failure_bounds":[max(-1.,weight*mean-radius),min(1.,weight*mean+radius)]}
    return {"complete_pairs":len(pairs),"unresolved_pairs":unresolved,"differences":result}


def summarize(args):
    cfg=protocol();records={};metadata=[];attempt_counts=Counter();attempt_keys=set();utc_logged=0;unresolved_attempts=0
    for path in args.ledgers:
        with sqlite3.connect(path) as db:
            configuration=json.loads(db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()[0]);metadata.append(configuration)
            local_records={}
            for condition,n,p,index,elapsed,value in db.execute("SELECT condition,n,p,idx,elapsed,result FROM replicates"):
                key=(condition,n,p,index)
                if key in records: raise ValueError("Duplicate fixed index")
                if index%configuration["owners"]!=configuration["owner"]: raise ValueError("Wrong owner for index")
                records[key]=(json.loads(value),elapsed)
                local_records[key]=True
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='attempts'").fetchone():
                for condition,n,p,index,started,ended,status,error in db.execute("SELECT * FROM attempts"):
                    key=(condition,n,p,index)
                    if key in attempt_keys: raise ValueError("Duplicate fixed-index attempt")
                    attempt_keys.add(key)
                    if index%configuration["owners"]!=configuration["owner"]: raise ValueError("Wrong attempt owner")
                    attempt_counts[status]+=1
                    a=datetime.fromisoformat(started);b=None if ended is None else datetime.fromisoformat(ended)
                    if a.tzinfo is None or a.utcoffset().total_seconds()!=0 or b is not None and (b.tzinfo is None or b.utcoffset().total_seconds()!=0 or b<a):
                        raise ValueError("Invalid absolute UTC attempt timestamps")
                    if key in local_records:
                        if status not in {"completed","completed_with_scientific_failure"} or b is None: raise ValueError("Committed replicate has no completed attempt")
                        utc_logged+=1
                    else:
                        if status not in {"started","interrupted"}: raise ValueError("Completed attempt has no replicate")
                        if status=="started" and b is not None or status=="interrupted" and b is None:
                            raise ValueError("Unresolved attempt timestamps contradict its status")
                        unresolved_attempts+=1
    if not metadata: raise ValueError("No ledgers")
    first=metadata[0];stage=first["stage"]
    for item in metadata:
        for key in ("stage","seed","owners","source_hashes","environment","freeze_sha256","candidate"):
            if item[key]!=first[key]: raise ValueError("Mixed ledger configurations")
    methods=cfg["development_methods"] if stage in {"cost","screen","selection"} else cfg["methods"]
    rows=[];planned=0;expected=set()
    for cell in cells(stage):
        for condition in cell["conditions"]:
            count=cell["replicates"];planned+=count
            for index in range(count): expected.add((condition,cell["n"],cell["p"],index))
            for method_index,method in enumerate(methods):
                data=[];baseline=[];seconds=[]
                for index in range(count):
                    entry=records.get((condition,cell["n"],cell["p"],index))
                    if entry:
                        by_method={r["method"]:r for r in entry[0]}
                        if set(by_method)!=set(methods) or len(entry[0])!=len(methods): raise ValueError("Missing/duplicate method")
                        data.append(by_method[method]);baseline.append(by_method["legacy_linear"]);seconds.append(entry[1])
                missing=count-len(data);failed=sum(r["failure"] for r in data);uncertain=missing+failed
                allowed=[r for r in data if r["allowed"] and not r["failure"]]
                hits=sum(r["null_rejected"] for r in data);raw=sum(r.get("raw_null_rejected",r["null_rejected"]) for r in data)
                conditional=sum(r["null_rejected"] for r in allowed)
                reasons=Counter();reason_datasets=Counter()
                for r in data:
                    reasons.update(r.get("reasons",{}));reason_datasets.update(r.get("reasons",{}).keys())
                power=[r["true_rejected"]/r["n_true"] if r["n_true"] else 0. for r in data]+[0.]*missing
                # For adoption, unresolved baseline power is set to its upper bound;
                # unresolved candidate power is zero. Failures cannot inflate retention.
                reference=[1. if r["failure"] else r["true_rejected"]/r["n_true"] if r["n_true"] else 0. for r in baseline]+[1.]*missing
                ratio=paired_ratio(power,reference,np.random.SeedSequence([cfg["mc_seed"],cell["n"],cell["p"],(cfg["conditions"]+cfg["stress_conditions"]).index(condition),method_index])) if condition.startswith("partial_") else None
                differences=paired_differences(data,baseline,count,np.random.SeedSequence([cfg["mc_seed"],cell["n"],cell["p"],(cfg["conditions"]+cfg["stress_conditions"]).index(condition),method_index,11]))
                rows.append(dict(stage=stage,condition=condition,n=cell["n"],p=cell["p"],method=method,planned=count,completed=len(data),missing=missing,failures=failed,
                    diagnostic_failures=sum(r.get("diagnostic_failure",False) for r in data),allowed=len(allowed),allowed_fraction=len(allowed)/count,
                    allowed_lower95=binomial(len(allowed),count,"lower"),fwer=hits/count,raw_fwer_bounds=[raw/count,(raw+uncertain)/count],
                    fwer_failure_bounds=[hits/count,(hits+uncertain)/count],fwer_upper95=binomial(hits+uncertain,count,"upper"),fwer_lower95=binomial(hits,count,"lower"),
                    conditional_fwer=None if not allowed else conditional/len(allowed),conditional_upper95=binomial(conditional,len(allowed),"upper"),conditional_lower95=binomial(conditional,len(allowed),"lower"),
                    withhold_fraction=1-len(allowed)/count,withhold_reasons=dict(reasons),power=None if ratio is None else float(np.mean(power)),power_ratio=ratio,
                    withhold_reason_dataset_counts=dict(reason_datasets),withhold_reason_fractions={key:hits/count for key,hits in reason_datasets.items()},
                    healthy_diagnostic_false_alarm_fraction=(sum(not r["allowed"] and not r["failure"] and not r.get("diagnostic_failure",False) for r in data)/count) if condition in cfg["healthy_original"]+cfg["healthy_stress"] else None,
                    paired_difference_from_legacy=differences,
                    mean_diagnostic_seconds=float(np.mean([r.get("diagnostic_seconds",0.) for r in data])) if data else None,
                    elapsed_p95_seconds=float(np.quantile(seconds,.95)) if seconds else None))
    if set(records)-expected: raise ValueError("Unexpected fixed index")
    if attempt_keys-expected: raise ValueError("Unexpected fixed-index attempt")
    if stage not in {"cost","screen","selection"} and utc_logged!=len(records):
        raise ValueError("Confirmation requires original per-replicate UTC records")
    result=dict(stage=stage,complete=set(records)==expected,planned_datasets=planned,completed_datasets=len(records),
        methods=methods,summaries=rows,configuration=first,attempt_status_counts=dict(attempt_counts),
        original_utc_completed_replicates=utc_logged,original_utc_coverage_complete=utc_logged==len(records),
        unresolved_attempts=unresolved_attempts,historical_missing_utc_policy="Unavailable original timestamps are not reconstructed",
        note="Development is descriptive and cannot qualify a release" if stage in {"cost","screen","selection"} else "Frozen finite-simulation engineering evidence")
    out=Path(args.output)
    if out.exists(): raise ValueError("Summary is immutable")
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("stage","complete","planned_datasets","completed_datasets")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    worker=commands.add_parser("run");worker.add_argument("--stage",choices=["cost","screen","selection","main","extension","large","stress"],required=True)
    worker.add_argument("--owners",type=int,default=48);worker.add_argument("--owner",type=int,required=True);worker.add_argument("--output",required=True);worker.add_argument("--freeze")
    summary=commands.add_parser("summarize");summary.add_argument("ledgers",nargs="+");summary.add_argument("--output",required=True)
    args=parser.parse_args();run(args) if args.command=="run" else summarize(args)

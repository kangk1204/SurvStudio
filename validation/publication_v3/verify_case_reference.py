"""Verify locked case computations with an independent R fit on raw rows only.

Actual external input is read only after a case lock and its completed aggregate
evaluation exist. --self-test uses generated synthetic rows, never the real case.
All raw rows, predictions and actual recipes stay in the requested private directory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
from survival_toolkit.clinical_basis import transform_clinical_encoder

DIRECTORY=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("prespecified_public_case",DIRECTORY/"public_case.py")
case=importlib.util.module_from_spec(spec);spec.loader.exec_module(case)


def synthetic_inputs(output):
    """Fixtures include tied times, missing covariates and external extrapolation."""
    data=output/"synthetic-data";data.mkdir()
    rng=np.random.default_rng(2026100343)
    for development,n in ((True,360),(False,240)):
        age=rng.uniform(35,80,n);nodes=rng.integers(1,26,n)
        meno=(age>50).astype(int);hormon=rng.integers(0,2,n)
        size=rng.choice([15,35,65],n);er=np.exp(rng.normal(3,1,n));pgr=np.exp(rng.normal(3,.8,n))
        eta=.025*(age-55)+.04*nodes+.3*(size>20)-.2*hormon-.12*np.log1p(pgr)
        time=np.maximum(1,np.round(rng.exponential(1800,n)*np.exp(-eta)))
        censor=np.maximum(1,np.round(rng.exponential(3500,n)))
        d=pd.DataFrame(dict(age=age,meno=meno,size=size,nodes=nodes,er=er,pgr=pgr,hormon=hormon))
        if development:
            d["size"]=d["size"].map({15:"<=20",35:"20-50",65:">50"})
            d["recur"]=(time<=censor).astype(int);d["rtime"]=np.minimum(time,censor)
            d["death"]=rng.integers(0,2,n);d["dtime"]=d.rtime+rng.integers(0,1000,n)
            # A known censored follow-up gap and a node-negative excluded subject.
            d.loc[0,["recur","death","rtime","dtime"]]=[0,1,100,800]
            d.loc[1,"nodes"]=0
            d.loc[2,"age"]=np.nan;d.loc[3,"pgr"]=np.nan
            name="rotterdam.csv"
        else:
            d["rfstime"]=np.minimum(time,censor);d["status"]=(time<=censor).astype(int)
            d.loc[0,"age"]=95;d.loc[1,"age"]=20;d.loc[2,"age"]=np.nan
            d.loc[3,"pgr"]=np.nan;d.loc[4,"size"]=np.nan
            name="gbsg-sealed.csv"
        d.to_csv(data/name,index=False)
    (data/"data-manifest.json").write_text(json.dumps({p.name:case.digest(p) for p in data.glob("*.csv")},indent=2)+"\n")
    selection=output/"synthetic-only-fallback.json"
    selection.write_text(json.dumps({"status":"v2_fallback_no_eligible_candidate","synthetic_fixture_only":True})+"\n")
    recipes=output/"synthetic-recipes";evaluation=output/"synthetic-evaluation"
    case.lock(SimpleNamespace(data=data,selection=selection,output=recipes))
    case.evaluate(SimpleNamespace(data=data,recipes=recipes,output=evaluation))
    aggregate_path=evaluation/"external-aggregate.json"
    aggregate=json.loads(aggregate_path.read_text())
    aggregate.update(known_public_benchmark=False,synthetic_fixture_only=True)
    aggregate_path.write_text(json.dumps(aggregate,indent=2,allow_nan=False)+"\n")
    return data,recipes,evaluation


def verify(data,recipes,evaluation,output,r_library,synthetic=False):
    case.check_export(data)
    cfg=json.loads(case.CONFIG.read_text());seal=json.loads((recipes/"lock.json").read_text())
    if seal["external_source_sha256"]!=case.digest(data/"gbsg-sealed.csv") or seal["protocol_sha256"]!=case.digest(case.CONFIG):
        raise ValueError("Case source/protocol seal changed")
    if seal["source_hashes"].get("public_case.py")!=case.digest(DIRECTORY/"public_case.py"):
        raise ValueError("Locked case numerical source changed")
    aggregate=json.loads((evaluation/"external-aggregate.json").read_text())
    if aggregate.get("external_evaluated_after_lock") is not True:
        raise ValueError("Completed locked external evaluation required")
    reference=output/"raw-reference";reference.mkdir()
    for name in ("rotterdam.csv","gbsg-sealed.csv"): shutil.copyfile(data/name,reference/name)
    development=pd.read_csv(data/"rotterdam.csv");external=pd.read_csv(data/"gbsg-sealed.csv")
    draws=np.random.default_rng(cfg["bootstrap_seed"]).integers(0,len(external),size=(cfg["bootstrap"],len(external)))
    pd.DataFrame(draws).to_csv(reference/"bootstrap-rows.csv",index=False)
    settings=[];locked={}
    for item in seal["recipes"]:
        path=recipes/item["name"]
        if case.digest(path)!=item["sha256"]: raise ValueError("Recipe changed after lock")
        recipe=json.loads(path.read_text());name=item["name"][:-5]
        endpoint=name.split("-",1)[0];basis=recipe["clinical"]["basis"]
        settings.append(dict(name=name,endpoint=endpoint,basis=basis,limit=cfg["administrative_censor_days"],
            primary=cfg["primary_horizon_days"],secondary1=cfg["secondary_horizons_days"][0],secondary2=cfg["secondary_horizons_days"][1]))
        locked[name]=recipe
    if not settings: raise ValueError("No fitted locked model available; retain case fit failures")
    pd.DataFrame(settings).to_csv(reference/"reference-settings.csv",index=False)
    command=["Rscript",str(DIRECTORY/"case_reference.R"),str(reference)]
    if r_library: command.append(r_library)
    subprocess.run(command,check=True)
    checks=[]
    def check(name,quantity,actual,expected,exact=False):
        a=np.asarray(actual);b=np.asarray(expected)
        matched=a.shape==b.shape
        if matched and exact:
            passed=bool(np.array_equal(a,b));difference=None
        elif matched:
            a=a.astype(float);b=b.astype(float)
            finite=np.isfinite(a)&np.isfinite(b)
            matched=bool(np.array_equal(np.isfinite(a),np.isfinite(b)))
            difference=float(np.max(np.abs(a[finite]-b[finite]))) if np.any(finite) else 0.
            passed=matched and difference<=1e-6
        else: passed=False;difference=None
        checks.append(dict(case=name,quantity=quantity,exact=exact,maximum_difference=difference,passed=passed))
    results={r["case"][:-5]:r for r in aggregate["results"]}
    for setting in settings:
        name=setting["name"];recipe=locked[name]
        if results[name]["recipe_hash"]!=recipe["recipe_hash"]: raise ValueError("Aggregate recipe binding differs")
        dev=case.prepare_frame(development,True,setting["endpoint"]);ext=case.prepare_frame(external,False,setting["endpoint"])
        encoder=recipe["clinical"]["encoder"]
        xd=transform_clinical_encoder(dev,encoder,output="dataframe")
        xe=transform_clinical_encoder(ext,encoder,output="dataframe")
        xd["logPGR"]=dev.logPGR.fillna(recipe["marker_medians"]["logPGR"])
        xe["logPGR"]=ext.logPGR.fillna(recipe["marker_medians"]["logPGR"])
        for kind,frame,design in (("development",dev,xd),("external",ext,xe)):
            rd=pd.read_csv(reference/(name+"-r-"+kind+".csv"),dtype={"size":str})
            check(name,kind+"_time",frame.time,rd.time,exact=True)
            check(name,kind+"_event",frame.event,rd.event,exact=True)
            check(name,kind+"_size",frame["size"].fillna("missing"),rd["size"].fillna("missing"),exact=True)
            check(name,kind+"_receptors",frame[["logER","logPGR"]],rd[["logER","logPGR"]])
            rdesign=pd.read_csv(reference/(name+"-r-"+kind+"-design.csv"))
            check(name,kind+"_design_terms",design.columns,rdesign.columns,exact=True)
            check(name,kind+"_fixed_design",design,rdesign)
        transforms={name+"_median":value for name,value in encoder["numeric_impute_values"].items()}
        transforms["logPGR_median"]=recipe["marker_medians"]["logPGR"]
        for column,spline in encoder["spline_specifications"].items():
            for key in ("knots","means","scales"):
                transforms.update({f"{column}_{dict(knots='knot',means='mean',scales='scale')[key]}{j+1}":v for j,v in enumerate(spline[key])})
        rt=pd.read_csv(reference/(name+"-r-transform.csv")).set_index("quantity").value
        check(name,"training_transform_parameters",[transforms[k] for k in rt.index],rt)
        metrics=pd.read_csv(reference/(name+"-r-metrics.csv"));predictions=pd.read_csv(reference/(name+"-r-predictions.csv"))
        calibration=pd.read_csv(reference/(name+"-r-calibration.csv"));bootstrap=pd.read_csv(reference/(name+"-r-bootstrap.csv"))
        for kind,model,bounds in (("clinical_only",recipe["clinical_only_model"],recipe["clinical_only_model"]["calibration_lp_boundaries"]),
                                 ("clinical_plus_PGR",recipe["model"],recipe["calibration_lp_boundaries"])):
            rc=pd.read_csv(reference/(name+"-"+kind+"-r-coefficients.csv"))
            check(name,kind+"_coefficient_terms",model["terms"],rc.term,exact=True)
            check(name,kind+"_coefficients",model["coefficients"],rc.coefficient)
            lp=xd[model["terms"]].to_numpy()@np.asarray(model["coefficients"])
            external_lp=xe[model["terms"]].to_numpy()@np.asarray(model["coefficients"])
            check(name,kind+"_training_lp",lp,pd.read_csv(reference/(name+"-"+kind+"-r-development-lp.csv")).lp)
            check(name,kind+"_lp_group_boundaries",bounds,pd.read_csv(reference/(name+"-"+kind+"-r-groups.csv")).boundary)
            baseline=pd.read_csv(reference/(name+"-"+kind+"-r-baseline.csv"))
            check(name,kind+"_baseline_times",model["baseline"]["times"],baseline.time,exact=True)
            check(name,kind+"_baseline_log_hazard",model["baseline"]["log_cumulative_hazard"],baseline.log_cumulative_hazard)
            check(name,kind+"_baseline_center",[model["baseline"]["lp_center"]],metrics[metrics.model==kind].lp_center.iloc[:1])
            for horizon in results[name]["horizons"]:
                at=horizon["horizon_days"]
                rp=predictions[(predictions.model==kind)&(predictions.horizon==at)]
                module=case.importlib.import_module("survival_toolkit.marker_evaluation")
                survival=module._baseline_survival_at(model["baseline"],external_lp,at)
                check(name,kind+f"_external_lp_{at}",external_lp,rp.lp)
                check(name,kind+f"_survival_{at}",survival,rp.survival)
                rm=metrics[(metrics.model==kind)&(metrics.horizon==at)].iloc[0]
                current=horizon["model_calibration"][kind]
                for quantity in ("calibration_slope","expected_risk"):
                    check(name,kind+f"_{quantity}_{at}",[current[quantity]],[rm[quantity]])
                c=horizon["metrics"]["c_index" if kind=="clinical_plus_PGR" else "clinical_only_c_index"]
                check(name,kind+f"_C_{at}",[c],[rm.c_index])
                pc=pd.DataFrame(current["fixed_group_calibration"])
                rc=calibration[(calibration.model==kind)&(calibration.horizon==at)]
                check(name,kind+f"_calibration_group_n_{at}",pc.n,rc.n,exact=True)
                check(name,kind+f"_absolute_risk_calibration_{at}",pc[["predicted_risk","observed_KM_risk"]],rc[["predicted_risk","observed_KM_risk"]])
            main=results[name]["horizons"][0]["metrics"]
            key="c_index_ci" if kind=="clinical_plus_PGR" else "clinical_only_c_index_ci"
            values=bootstrap[kind].dropna().to_numpy()
            check(name,kind+"_bootstrap_C_interval",main[key],np.quantile(values,[.025,.975]))
        main=results[name]["horizons"][0]["metrics"]
        rmain=metrics[metrics.horizon==cfg["primary_horizon_days"]].set_index("model")
        check(name,"paired_delta_C",[main["delta_c_index"]],[rmain.loc["clinical_plus_PGR","c_index"]-rmain.loc["clinical_only","c_index"]])
        check(name,"paired_delta_C_interval",main["delta_c_index_ci"],np.quantile(bootstrap.delta.dropna(),[.025,.975]))
    sources={p.name:case.digest(p) for p in (Path(__file__),DIRECTORY/"case_reference.R",DIRECTORY/"public_case.py",case.CONFIG)}
    report=dict(passed=all(c["passed"] for c in checks),checks=checks,tolerance=1e-6,synthetic_fixture_only=synthetic,
        source_hashes=sources,data_manifest_sha256=case.digest(data/"data-manifest.json"),
        lock_sha256=case.digest(recipes/"lock.json"),aggregate_sha256=case.digest(evaluation/"external-aggregate.json"),
        reference_hashes={p.name:case.digest(p) for p in reference.iterdir() if p.is_file()},
        boundary="R reads raw data, the prespecified model and shared bootstrap rows only; fits transforms and both Cox models independently. This verifies numerics, not prediction superiority or diagnostics validity.")
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(passed=report["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]])))
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path);p.add_argument("--recipes",type=Path);p.add_argument("--evaluation",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--r-library");p.add_argument("--self-test",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs=synthetic_inputs(args.output) if args.self_test else (args.data,args.recipes,args.evaluation)
    if any(x is None for x in inputs): p.error("Actual verification requires --data, --recipes and --evaluation")
    verify(*inputs,args.output,args.r_library,synthetic=args.self_test)

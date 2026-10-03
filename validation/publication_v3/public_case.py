"""Prepare a prespecified model; refuse external evaluation without its sealed recipes."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from survival_toolkit.marker_qualification import qualification, _withhold

CONFIG=Path(__file__).with_name("case_protocol.json")


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_frame(frame,development,endpoint):
    if endpoint not in {"conservative_RFS","liberal_RFS"}: raise ValueError("Unknown prespecified endpoint")
    d=frame.copy()
    if development: d=d.loc[d.nodes>0].copy()
    if (d[["er","pgr"]]<0).any().any(): raise ValueError("Negative receptor measurement")
    d["logER"]=np.log1p(d.er);d["logPGR"]=np.log1p(d.pgr)
    if not development:
        size=pd.to_numeric(d["size"],errors="raise")
        if (size.dropna()<=0).any(): raise ValueError("Invalid tumor size")
        d["size"]=pd.Series(np.where(size<=20,"<=20",np.where(size<=50,"20-50",">50")),index=d.index).where(size.notna())
    if not set(d["size"].dropna())<={"<=20","20-50",">50"}: raise ValueError("Unknown size category")
    d["size"]=d["size"].map({"<=20":"0","20-50":"1",">50":"2"})
    if development:
        ignore=(d.recur==0)&(d.death==1)&(d.rtime<d.dtime)
        if endpoint=="conservative_RFS":
            d["event"]=np.where((d.recur==1)|ignore,d.recur,d.death)
            d["time"]=np.where((d.recur==1)|ignore,d.rtime,d.dtime)
        else:
            d["event"]=np.maximum(d.recur,d.death)
            d["time"]=np.where(d.recur==1,d.rtime,d.dtime)
    else: d["time"]=d.rfstime;d["event"]=d.status
    if not np.isfinite(d.time).all() or (d.time<=0).any() or not set(d.event)<={0,1}: raise ValueError("Invalid RFS endpoint")
    limit=json.loads(CONFIG.read_text())["administrative_censor_days"]
    d.loc[d.time>limit,"event"]=0;d["time"]=np.minimum(d.time,limit)
    return d


def check_export(data):
    manifest=json.loads((data/"data-manifest.json").read_text())
    for name in ("rotterdam.csv","gbsg-sealed.csv"):
        if manifest.get(name)!=digest(data/name): raise ValueError("Preserved public source export changed")


def export(args):
    args.output.mkdir(parents=True,exist_ok=False)
    code='''args<-commandArgs(TRUE); .libPaths(c(args[2],.libPaths())); library(survival); write.csv(rotterdam,file.path(args[1],"rotterdam.csv"),row.names=FALSE); write.csv(gbsg,file.path(args[1],"gbsg-sealed.csv"),row.names=FALSE); capture.output(sessionInfo(),file=file.path(args[1],"data-session.txt"))'''
    script=args.output/"export.R";script.write_text(code)
    subprocess.run(["Rscript",str(script),str(args.output),args.r_library or ""],check=True)
    (args.output/"data-manifest.json").write_text(json.dumps({p.name:digest(p) for p in args.output.iterdir() if p.is_file()},indent=2)+"\n")
    print("Public source data exported privately; no GBSG model evaluation performed")


def lock(args):
    check_export(args.data)
    cfg=json.loads(CONFIG.read_text());selection=json.loads(args.selection.read_text())
    if selection.get("status") not in {"selected","v2_fallback_no_eligible_candidate","v2_fallback_resource_limit"}: raise ValueError("Candidate/fallback decision required before model lock")
    method="marker-inference/3" if selection.get("status")=="selected" else "marker-inference/2"
    module=importlib.import_module("survival_toolkit.marker_evaluation_v3" if method.endswith("/3") else "survival_toolkit.marker_evaluation")
    args.output.mkdir(parents=True,exist_ok=False);original=pd.read_csv(args.data/"rotterdam.csv")
    from survival_toolkit.marker_bootstrap import diagnose_markers as v3_diagnose
    from survival_toolkit.marker_diagnostics import diagnose_markers as v2_diagnose
    recipes=[]; failures=[]
    for endpoint in (cfg["endpoint_primary"],cfg["endpoint_sensitivity"]):
        d=prepare_frame(original,True,endpoint)
        for basis in (cfg["basis_primary"],cfg["basis_sensitivity"]):
            settings=module.MarkerSettings(n_permutations=999,n_resamples=0,clinical_basis=basis,random_seed=cfg["bootstrap_seed"])
            cohort=module.prepare_marker_cohort(d,time_column="time",event_column="event",marker_columns=["logPGR"],clinical_columns=cfg["clinical"],categorical_clinical=cfg["categorical"],clinical_basis=basis)
            if cohort.clinical_error:
                failures.append({"endpoint":endpoint,"basis":basis,"reason":cohort.clinical_error})
                continue
            full=module.run_procedure(cohort,np.arange(len(cohort.time)),settings)
            full=full._replace(selected={**full.selected,"added_value":np.ones(len(cohort.marker_names),bool)})
            signature=module._fit_signature(cohort,np.arange(len(cohort.time)),full,"added_value",1,"efron")
            if signature is None:
                failures.append({"endpoint":endpoint,"basis":basis,"reason":"Prespecified prediction model cannot be fitted"})
                continue
            diagnostic=(v3_diagnose(cohort,cohort.markers,candidate=selection["candidate"],bootstrap_seed=cfg["bootstrap_seed"]) if method.endswith("/3") else v2_diagnose(cohort,cohort.markers))
            diagnostic=_withhold(diagnostic,qualification(basis,method))
            recipe=module.freeze_recipe(cohort,full,signature,"added_value",time_column="time",event_column="event",event_positive_value=1,categorical_clinical=cfg["categorical"],inference=diagnostic)
            recipe["model_selection"]=cfg["model_selection"]
            recipe["preprocessing_manifest"]={"config_sha256":digest(CONFIG),"transform_definitions":cfg["transforms"],"development_source_sha256":digest(args.data/"rotterdam.csv")}
            lp=module._signature_risk(cohort,np.arange(len(cohort.time)),signature)
            recipe["calibration_lp_boundaries"]=np.quantile(lp,[.2,.4,.6,.8]).tolist()
            from survival_toolkit.clinical_basis import transform_clinical_encoder
            encoded=transform_clinical_encoder(cohort.clinical_frame,recipe["clinical"]["encoder"],output="dataframe")
            clinical_model=recipe["clinical_only_model"]
            clinical_lp=encoded[clinical_model["terms"]].to_numpy()@np.asarray(clinical_model["coefficients"])
            clinical_model["baseline"]=module._centred_baseline(cohort.time,cohort.event,clinical_lp,"efron")
            clinical_model["calibration_lp_boundaries"]=np.quantile(clinical_lp,[.2,.4,.6,.8]).tolist()
            recipe["model"]["default_horizon"]=cfg["primary_horizon_days"]
            recipe=module._json_ready(recipe)
            recipe["recipe_hash"]=module.recipe_hash(recipe)
            name=endpoint+"-"+basis+".json";(args.output/name).write_text(json.dumps(recipe,indent=2,allow_nan=False)+"\n")
            recipes.append({"name":name,"sha256":digest(args.output/name),"recipe_hash":recipe["recipe_hash"]})
    manifest=dict(protocol_sha256=digest(CONFIG),selection_sha256=digest(args.selection),external_source_sha256=digest(args.data/"gbsg-sealed.csv"),recipes=recipes,failures=failures,external_evaluated=False,
        source_hashes={Path(__file__).name:digest(__file__)})
    (args.output/"lock.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"locked_recipes":len(recipes),"external_evaluated":False}))


def evaluate(args):
    check_export(args.data)
    cfg=json.loads(CONFIG.read_text());seal=json.loads((args.recipes/"lock.json").read_text())
    if seal["protocol_sha256"]!=digest(CONFIG) or seal["external_source_sha256"]!=digest(args.data/"gbsg-sealed.csv"): raise ValueError("Locked case inputs changed")
    if seal["source_hashes"].get(Path(__file__).name)!=digest(__file__): raise ValueError("Locked case analysis source changed")
    args.output.mkdir(parents=True,exist_ok=False)
    original=pd.read_csv(args.data/"gbsg-sealed.csv");results=[]
    for item in seal["recipes"]:
        path=args.recipes/item["name"]
        if digest(path)!=item["sha256"]: raise ValueError("Locked recipe changed")
        recipe=json.loads(path.read_text());endpoint=item["name"].split("-",1)[0]
        d=prepare_frame(original,False,endpoint)
        module=importlib.import_module("survival_toolkit.marker_evaluation_v3" if recipe["inference"]["method_version"]=="marker-inference/3" else "survival_toolkit.marker_evaluation")
        horizons=[]
        for horizon in [cfg["primary_horizon_days"],*cfg["secondary_horizons_days"]]:
            result=module.validate_locked_recipe(d,recipe,horizon=horizon,n_bootstrap=cfg["bootstrap"] if horizon==cfg["primary_horizon_days"] else 0,random_seed=cfg["bootstrap_seed"],marker_scaling="as_measured")
            from survival_toolkit.clinical_basis import transform_clinical_encoder
            from survival_toolkit.analysis import compute_km_analysis
            encoded=transform_clinical_encoder(d,recipe["clinical"]["encoder"],output="dataframe")
            for marker in recipe["markers"]:
                encoded[marker]=d[marker].fillna(recipe["marker_medians"][marker])
            lp=encoded[recipe["model"]["terms"]].to_numpy() @ np.asarray(recipe["model"]["coefficients"])
            model_calibration={}
            for name,model,bounds in (("clinical_plus_PGR",recipe["model"],recipe["calibration_lp_boundaries"]),("clinical_only",recipe["clinical_only_model"],recipe["clinical_only_model"]["calibration_lp_boundaries"])):
                predictor=encoded[model["terms"]].to_numpy()@np.asarray(model["coefficients"])
                risk=1-module._baseline_survival_at(model["baseline"],predictor,horizon)
                groups=np.searchsorted(bounds,predictor,side="right")
                km=compute_km_analysis(d.assign(calibration_group=groups),"time","event","calibration_group",event_positive_value=1)
                observed={}
                for curve in km["curves"]:
                    index=max(0,np.searchsorted(curve["timeline"],horizon,side="right")-1)
                    observed[int(curve["group"])]=1-float(curve["survival"][index])
                calibration=[{"group":g,"n":int(np.sum(groups==g)),"predicted_risk":float(np.mean(risk[groups==g])) if np.any(groups==g) else None,"observed_KM_risk":observed.get(g)} for g in range(5)]
                slope=module._external_cox(d.time.to_numpy(),d.event.to_numpy(),predictor[:,None],None,"efron")
                model_calibration[name]={"calibration_slope":None if slope is None else slope["log_hr"],"expected_risk":float(np.mean(risk)),"fixed_group_calibration":calibration}
            horizons.append({"horizon_days":horizon,"metrics":result["metrics"],"inference":result["inference"],"model_calibration":model_calibration})
        results.append({"case":item["name"],"recipe_hash":recipe["recipe_hash"],"horizons":horizons})
    (args.output/"external-aggregate.json").write_text(json.dumps({"known_public_benchmark":True,"external_evaluated_after_lock":True,"results":results,"independent_R_verified":False},indent=2,allow_nan=False)+"\n")
    print("External aggregates calculated; independent R verification still required")


parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
p=sub.add_parser("export");p.add_argument("--output",type=Path,required=True);p.add_argument("--r-library")
p=sub.add_parser("lock");p.add_argument("--data",type=Path,required=True);p.add_argument("--selection",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
p=sub.add_parser("evaluate");p.add_argument("--data",type=Path,required=True);p.add_argument("--recipes",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
if __name__=="__main__":
    args=parser.parse_args();{"export":export,"lock":lock,"evaluate":evaluate}[args.command](args)

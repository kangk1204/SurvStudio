"""Synthetic comparison fixtures and explicit human web execution records."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from survival_toolkit.analysis import compute_km_analysis
from survival_toolkit.clinical_basis import fit_clinical_encoder,transform_clinical_encoder
from survival_toolkit.marker_screen import fit_cox

ROOT=Path(__file__).resolve().parents[2]
TASKS=("event_coding","KM_estimation","adjusted_HR_reference","misspecification",
    "duplicates_outcome_leakage","selection_internal_evaluation","locked_external_validation","endpoint_mixing")


def fixtures(output):
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/"validation/guarded_inference"))
    spec=importlib.util.spec_from_file_location("fixed_tool_fixtures",ROOT/"validation/guarded_inference/compare_tools.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.fixtures(output)
    records=[dict(tool=tool,task=task,status="pending_human_execution",date="",operator="",url="",terms_checked="",input_sha256="",settings="",actions="",numeric_outputs="",screenshot="",downloaded_artifacts="",limitation="") for tool in ("KM_Plotter_custom_data","surviveR") for task in TASKS]
    pd.DataFrame(records).to_csv(output/"human-execution-records.csv",index=False)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.csv") if p.name!="human-execution-records.csv"}
    (output/"fixture-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"HUMAN_EXECUTION.md").write_text('''# Manual web comparison

Use synthetic data only. These are researcher task executions, not a participant study.
Check access and current terms yourself. Never automate KM Plotter or bypass login.
Record exact settings, steps, screenshots and downloaded outputs in the CSV.

1. A.csv: time and event_code2, positive event=2. Check event counts against event.
2. A.csv: KM by grade, fixed survival values at 5, 10 and 15 time units.
3. A.csv: adjusted continuous m0 + z + binary + grade; grade A reference. If the tool only supports grouped or unadjusted HR, record non-comparable rather than numerical disagreement.
4. misspecified.csv: clinical z and m0..m29. Record diagnostics and how any failed inference is displayed/exported.
5. duplicate.csv: inspect repeated patient ID; A.csv: try assigning event as a predictor. Record warnings/rejections and avoid treating successful upload as protection.
6. A.csv: inner_train identifies the training subset. Select using this subset only; evaluate untouched remaining rows. Record whether training transforms/selection are carried into evaluation.
7. Derive a model with A.csv and apply unchanged to B.csv. Record what coefficients, transforms and diagnostic status can be exported and reapplied. Manual external reconstruction is a separate workflow.
8. A.csv: pair rfs_months with dmfs_event. Record endpoint declaration/checking behaviour; this synthetic pair is intentionally inconsistently named.

Allowed statuses: executed_numeric_match, executed_numeric_difference, executed_workflow,
verified_unsupported, access_limited, not_comparable, pending_human_execution.
Absence from documentation does not establish an unsupported feature. Do not report
human time saved, usability, or interpretation error reduction.
''')


def lifelines(inputs,output):
    from lifelines import KaplanMeierFitter,CoxPHFitter
    a=pd.read_csv(inputs/"A.csv");results=[]
    km=compute_km_analysis(a,"time","event_code2","grade",event_positive_value=2)
    for curve in km["curves"]:
        group=curve["group"];d=a[a.grade==group]
        fit=KaplanMeierFitter().fit(d.time,event_observed=d.event)
        for at in (5,10,15):
            ix=max(np.searchsorted(curve["timeline"],at,side="right")-1,0)
            actual=curve["survival"][ix];expected=float(fit.predict(at))
            results.append(dict(quantity=f"KM_{group}_{at}",SurvStudio=actual,lifelines=expected,absolute_difference=abs(actual-expected),tolerance=1e-6))
    encoder=fit_clinical_encoder(a,["z","grade","binary"],["grade"],basis="linear")
    x=transform_clinical_encoder(a,encoder);x=np.column_stack([x,a.m0])
    fit=fit_cox(a.time.to_numpy(),a.event.to_numpy(),x)
    columns=[f"x{i}" for i in range(x.shape[1])];d=pd.DataFrame(x,columns=columns);d["time"]=a.time;d["event"]=a.event
    reference=CoxPHFitter().fit(d,"time","event",fit_options={"precision":1e-10,"max_steps":100})
    for i,column in enumerate(columns):
        expected=float(reference.params_[column]);actual=float(fit.beta[i])
        results.append(dict(quantity="coefficient_"+column,SurvStudio=actual,lifelines=expected,absolute_difference=abs(actual-expected),tolerance=1e-6))
    results.append(dict(quantity="loglik",SurvStudio=fit.loglik,lifelines=reference.log_likelihood_,absolute_difference=abs(fit.loglik-reference.log_likelihood_),tolerance=1e-6))
    output.mkdir(parents=True,exist_ok=False);table=pd.DataFrame(results);table["passed"]=table.absolute_difference<=table.tolerance
    table.to_csv(output/"lifelines-comparison.csv",index=False)
    report=dict(passed=bool(table.passed.all()),lifelines_version=importlib.metadata.version("lifelines"),
        quantities=len(table),maximum_difference=float(table.absolute_difference.max()),
        boundary="Common KM and Cox numeric tasks only; no claim that library workflows are unavailable")
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["fixtures","lifelines"]);p.add_argument("--inputs",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();fixtures(args.output) if args.command=="fixtures" else lifelines(args.inputs,args.output)

"""Expanded R oracle using 9999 draws for every basis/design fixture.

Kept separate from the immutable development runner reference. Additional marker
family sizes use the same independently reconstructed raw-input calculation.
"""
import argparse
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from survival_toolkit.marker_evaluation import prepare_marker_cohort
from survival_toolkit.marker_bootstrap import diagnose_markers,joint_maxima,uniform_stream

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",required=True,type=Path);parser.add_argument("--r-library")
parser.add_argument("--markers",type=int,default=2)
args=parser.parse_args()
if args.markers<2: parser.error("At least two markers required")
args.output.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(921);n=90;z=rng.normal(size=n)
t=rng.exponential(size=n)/(.06*np.exp(.8*z));c=rng.exponential(size=n)/.04
frame=pd.DataFrame(dict(time=np.minimum(t,c),event=(t<=c).astype(int),Z=z,X0=.8*z+rng.normal(size=n),X1=.8*z+rng.normal(size=n)))
names=[f"X{j}" for j in range(args.markers)]
for j in range(2,args.markers): frame[f"X{j}"]=.8*z+rng.normal(size=n)
fixtures=[];python={}
for basis,stratified,multiple in (("linear",False,False),("restricted_cubic_spline",False,False),("linear",True,False),("linear",False,True),("restricted_cubic_spline",False,True)):
    input_frame=frame.copy()
    if stratified: input_frame["stratum"]=np.arange(n)%2
    clinical=["Z"]
    if multiple:
        input_frame["Z2"]=.3*z+np.sqrt(1-.3**2)*np.random.default_rng(924).normal(size=n);clinical.append("Z2")
        input_frame[names]=input_frame[names].add(.4*input_frame.Z2,axis=0)
    cohort=prepare_marker_cohort(input_frame,time_column="time",event_column="event",marker_columns=names,clinical_columns=clinical,clinical_basis=basis,strata_columns=["stratum"] if stratified else [])
    for candidate in ("residual_vector","restricted_wild"):
        name=basis+"-"+candidate+("-stratified" if stratified else "")+("-two-clinical" if multiple else "");draws=9999
        u=uniform_stream(33,draws,n)
        input_frame.to_csv(args.output/(name+"-input.csv"),index=False)
        pd.DataFrame(u).to_csv(args.output/(name+"-uniforms.csv"),index=False)
        result=diagnose_markers(cohort,cohort.markers,candidate=candidate,bootstrap_seed=33,draws=draws,uniforms=u)
        if result["bootstrap"]["status"]!="calculated": raise ValueError("Python bootstrap reference fixture failed")
        tests=result["residual_tests"]
        observed=np.array([test["statistic"]/test["df"] for kind in ("nonlinear_mean","variance") for test in tests if test["diagnostic"]==kind])
        python[name]=(joint_maxima(cohort,cohort.markers,candidate=candidate,uniforms=u),observed,result["bootstrap"])
        fixtures.append(dict(name=name,basis=basis,candidate=candidate))
pd.DataFrame(fixtures).to_csv(args.output/"fixtures.csv",index=False)
command=["Rscript",str(Path(__file__).with_name("reference.R")),str(args.output)]
if args.r_library: command.append(args.r_library)
subprocess.run(command,check=True)
checks=[]
for fixture in fixtures:
    name=fixture["name"];maximum,observed,boot=python[name]
    rm=pd.read_csv(args.output/(name+"-r-maxima.csv"))["maximum"].to_numpy()
    ro=pd.read_csv(args.output/(name+"-r-observed.csv"))["statistic"].to_numpy()
    rd=pd.read_csv(args.output/(name+"-r-decision.csv")).iloc[0]
    differences={"maximum":float(np.max(np.abs(maximum-rm))),"observed":float(np.max(np.abs(observed-ro))),
        "global_p":abs(boot["global_p"]-rd["p"]),"mc_interval":float(np.max(np.abs(np.asarray(boot["global_mc95"])-rd[["lower","upper"]].to_numpy(float))))}
    exact=int(rd["hits"])==boot["global_exceedances"]
    checks.append(dict(name=name,draws=len(maximum),markers=args.markers,differences=differences,exceedance_count_exact=exact,passed=exact and max(differences.values())<=1e-6))
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
root=Path(__file__).resolve().parents[2]
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
    (Path(__file__),Path(__file__).with_name("reference.R"),root/"src/survival_toolkit/marker_bootstrap.py")}
from study import hashes as numerical_hashes
sources={**numerical_hashes(),**sources}
report=dict(passed=all(c["passed"] for c in checks),checks=checks,environment={"python":platform.python_version(),"packages":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","statsmodels")}},input_and_output_hashes=files,source_hashes=sources,
    diagnostic_draws=9999,markers=args.markers,full_draw_coverage=True,
    boundary="Raw inputs and uniforms shared; R independently builds basis projections, HC3 statistics, resamples, counts and intervals")
(args.output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"checks":checks}))
if not report["passed"]: raise SystemExit(1)

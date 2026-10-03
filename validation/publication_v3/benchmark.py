"""Same Linux server: common Cox fit, three warmups and ten measured repeats.

Inputs and imports are outside the operation timer. Peak RSS covers each entire
worker process and cannot be interpreted as the Cox operation's memory increment.
No web latency or human usability measurement is made.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def worker(args):
    d=pd.read_csv(args.inputs/"A.csv")
    columns=["grade_B","grade_C","z","binary","m0"]
    x=pd.DataFrame(dict(grade_B=(d.grade=="B").astype(float),grade_C=(d.grade=="C").astype(float),z=d.z,binary=d.binary,m0=d.m0))
    if args.engine=="lifelines":
        from lifelines import CoxPHFitter
        frame=x.assign(time=d.time,event=d.event)
        def fit():
            model=CoxPHFitter().fit(frame,"time","event",fit_options={"precision":1e-10,"max_steps":100})
            return model.params_.to_numpy(),model.log_likelihood_
    else:
        from survival_toolkit.marker_screen import fit_cox
        times=d.time.to_numpy();events=d.event.to_numpy();matrix=x.to_numpy()
        def fit():
            model=fit_cox(times,events,matrix)
            if not model.converged or not np.isfinite(model.beta).all(): raise ValueError("Common Cox task failed")
            return model.beta,model.loglik
    elapsed=[]
    for index in range(13):
        start=time.perf_counter();coefficients,loglik=fit();duration=time.perf_counter()-start
        if index>=3: elapsed.append(duration)
    pd.DataFrame(dict(run=range(1,11),seconds=elapsed)).to_csv(args.output,index=False)
    pd.DataFrame(dict(term=columns,coefficient=coefficients)).to_csv(str(args.output)+".coefficients.csv",index=False)
    Path(str(args.output)+".loglik.json").write_text(json.dumps(float(loglik))+"\n")


def run(args):
    if platform.system()!="Linux" or not Path("/usr/bin/time").exists(): raise ValueError("Run this common benchmark on the same Linux server")
    manifest=json.loads((args.inputs/"fixture-manifest.json").read_text())
    if manifest["A.csv"]!=hashlib.sha256((args.inputs/"A.csv").read_bytes()).hexdigest(): raise ValueError("Comparison fixture changed")
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    if args.r_library: env["R_LIBS_USER"]=args.r_library
    commands={engine:[sys.executable,str(Path(__file__).resolve()),"--worker","--engine",engine,"--inputs",str(args.inputs),"--output",str(args.output/(engine+".csv"))] for engine in ("SurvStudio","lifelines")}
    commands["R_survival"]=["Rscript",str(ROOT/"validation/guarded_inference/benchmark_reference.R"),str(args.inputs/"A.csv"),str(args.output/"R_survival.csv")]
    load_before=os.getloadavg();order=[]
    # Prespecified order; do not choose it using observed durations.
    rng=np.random.default_rng(2026100344)
    for engine in rng.permutation(list(commands)):
        order.append(str(engine))
        subprocess.run(["/usr/bin/time","-f","%M","-o",str(args.output/(engine+"-peak-rss-kb.txt")),*commands[engine]],check=True,env=env)
    reference=pd.read_csv(args.output/"R_survival.csv.coefficients.csv").coefficient.to_numpy();rows=[];checks=[]
    for engine in commands:
        seconds=pd.read_csv(args.output/(engine+".csv")).seconds.to_numpy()
        if len(seconds)!=10 or not np.isfinite(seconds).all() or (seconds<0).any(): raise ValueError("Incomplete timing repeats")
        coefficients=pd.read_csv(args.output/(engine+".csv.coefficients.csv")).coefficient.to_numpy()
        difference=float(np.max(np.abs(coefficients-reference)))
        checks.append(dict(engine=engine,coefficient_maximum_difference=difference,passed=difference<=1e-6))
        rows.append(dict(engine=engine,n=len(pd.read_csv(args.inputs/"A.csv")),terms=5,warmups=3,measured_runs=10,
            median_seconds=float(np.median(seconds)),q25_seconds=float(np.quantile(seconds,.25)),q75_seconds=float(np.quantile(seconds,.75)),
            minimum_seconds=float(seconds.min()),maximum_seconds=float(seconds.max()),process_peak_rss_kb=int((args.output/(engine+"-peak-rss-kb.txt")).read_text())))
    pd.DataFrame(rows).to_csv(args.output/"timing-memory-summary.csv",index=False)
    report=dict(passed=all(c["passed"] for c in checks),checks=checks,summary=rows,host=platform.node(),platform=platform.platform(),load_before=load_before,load_after=os.getloadavg(),engine_order=order,
        task="Unstratified Efron Cox, grade A reference, z/binary/m0; common input preparation excluded",
        input_sha256=manifest["A.csv"],source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/"validation/guarded_inference/benchmark_reference.R")},
        python_packages={n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","statsmodels","lifelines")},
        limitation="Shared server load and R timer resolution affect timing. Process peak RSS includes runtime/imports. No web latency, usability or interpretation-error claim.")
    (args.output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker",action="store_true");p.add_argument("--engine",choices=["SurvStudio","lifelines"])
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--r-library")
    args=p.parse_args();worker(args) if args.worker else run(args)

"""Common numerical tasks, with process memory separated from fit timing."""
import argparse
from datetime import datetime, timezone
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
    d=pd.read_csv(args.inputs/'A.csv')
    x=pd.DataFrame(dict(grade_B=(d.grade=='B').astype(float),grade_C=(d.grade=='C').astype(float),z=d.z,binary=d.binary.astype(float),m0=d.m0))
    if args.task=='Cox':
        if args.engine=='SurvStudio':
            from survival_toolkit.marker_screen import fit_cox
            def fit():return fit_cox(d.time.to_numpy(),d.event.to_numpy(),x.to_numpy()).beta
        elif args.engine=='lifelines':
            from lifelines import CoxPHFitter
            frame=x.assign(time=d.time,event=d.event)
            def fit():return CoxPHFitter().fit(frame,'time','event',fit_options={'precision':1e-10,'max_steps':100}).params_.to_numpy()
        else:
            from mlsurv.models import CoxPHModel
            from sksurv.util import Surv
            y=Surv.from_arrays(d.event==1,d.time)
            def fit():return CoxPHModel(alpha=0).fit(x,y).model_.coef_
    elif args.engine=='SurvStudio':
        from survival_toolkit.analysis import compute_km_analysis
        def fit():
            result=compute_km_analysis(d,'time','event','grade',event_positive_value=1)
            values=[]
            for curve in result['curves']:
                for at in (5,10,15):
                    i=max(np.searchsorted(curve['timeline'],at,side='right')-1,0)
                    values.append(curve['survival'][i])
            return values
    else:
        from lifelines import KaplanMeierFitter
        def fit():
            values=[]
            for group in ('A','B','C'):
                part=d[d.grade==group];m=KaplanMeierFitter().fit(part.time,part.event)
                values.extend(float(m.predict(at)) for at in (5,10,15))
            return values
    durations=[]
    for i in range(13):
        start=time.perf_counter();values=fit();duration=time.perf_counter()-start
        if i>=3:durations.append(duration)
    pd.DataFrame({'run':range(1,11),'seconds':durations}).to_csv(args.output,index=False)
    pd.DataFrame({'value':values}).to_csv(str(args.output)+'.values.csv',index=False)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.inputs/'fixture-manifest.json').read_text())
    if hashlib.sha256((args.inputs/'A.csv').read_bytes()).hexdigest()!=manifest['A.csv']:raise ValueError('Fixed input changed')
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    jobs=[(task,engine) for task,engines in [('Cox',['SurvStudio','lifelines','mlsurv_Cox_model','R_survival','R_rms']),('KM',['SurvStudio','lifelines','R_survival'])] for engine in engines]
    order=np.random.default_rng(2026100906).permutation(len(jobs)).tolist()
    lock=dict(locked_utc=datetime.now(timezone.utc).isoformat(),input_sha256=manifest['A.csv'],warmups=3,measured_runs=10,
              jobs=jobs,order=order,order_seed=2026100906,
              source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_suffix('.R')]})
    (args.output/'benchmark-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    load_before=os.getloadavg()
    for i in order:
        task,engine=jobs[i];out=args.output/(task+'-'+engine+'.csv')
        command=([sys.executable,str(Path(__file__)),'--worker','--engine',engine,'--task',task,'--inputs',str(args.inputs),'--output',str(out)] if not engine.startswith('R_') else
                 ['Rscript',str(Path(__file__).with_suffix('.R')),str(args.inputs/'A.csv'),str(out),args.r_library,args.hmisc_library,task,engine])
        with (args.output/(task+'-'+engine+'.log')).open('xb') as log:
            subprocess.run(['/usr/bin/time','-f','%M','-o',str(out)+'.rss-kb',*command],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows=[]
    for task,engine in jobs:
        file=args.output/(task+'-'+engine+'.csv')
        durations=pd.read_csv(file).seconds.to_numpy()
        reference=pd.read_csv(args.output/(task+'-R_survival.csv.values.csv')).value.to_numpy()
        actual=pd.read_csv(str(file)+'.values.csv').value.to_numpy()
        difference=float(np.max(np.abs(actual-reference)))
        rows.append(dict(task=task,engine=engine,warmups=3,measured_runs=len(durations),median_seconds=float(np.median(durations)),
            q25_seconds=float(np.quantile(durations,.25)),q75_seconds=float(np.quantile(durations,.75)),
            process_peak_rss_kb=int(Path(str(file)+'.rss-kb').read_text()),numeric_maximum_difference=difference,numeric_passed=difference<=1e-6))
    pd.DataFrame(rows).to_csv(args.output/'timing-memory-summary.csv',index=False)
    report=dict(passed=all(r['numeric_passed'] and r['measured_runs']==10 for r in rows),rows=rows,host=platform.node(),
                load_before=load_before,load_after=os.getloadavg(),lock_sha256=hashlib.sha256((args.output/'benchmark-lock.json').read_bytes()).hexdigest(),
                packages={n:importlib.metadata.version(n) for n in ('numpy','pandas','scipy','statsmodels','lifelines','mlsurv')},
                limitations=['Whole-process peak RSS includes runtime/imports; not the operation memory increment.',
                'KM wrappers produce different ancillary outputs; timing compares their declared common numerical task.',
                'mlsurv Cox model timing is not its full CV/diagnostics/report pipeline.',
                'Untied continuous times make Efron and Breslow estimates equivalent in this fixture.',
                'No web latency or human usability measurement.'])
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'tasks':len(rows)}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--engine');p.add_argument('--task',choices=['KM','Cox']);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--r-library');p.add_argument('--hmisc-library')
    a=p.parse_args();worker(a) if a.worker else run(a)

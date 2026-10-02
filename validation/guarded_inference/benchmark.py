"""Same-server common Cox numerical task: 3 warmups, 10 measured fits, process peak RSS.

Operation time excludes imports/input preparation for both engines. Peak RSS includes
the whole worker process; neither measurement is a human-usability assessment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from survival_toolkit.marker_screen import fit_cox


def worker(data,output):
    d=pd.read_csv(data);x=np.column_stack([(d.grade=='B').astype(float),(d.grade=='C').astype(float),d.z,d.binary,d.m0])
    times=[]
    for i in range(13):
        start=time.perf_counter();fit=fit_cox(d.time.to_numpy(),d.event.to_numpy(),x);elapsed=time.perf_counter()-start
        if i>=3: times.append(elapsed)
    pd.DataFrame({'run':range(1,11),'seconds':times}).to_csv(output,index=False)
    pd.DataFrame({'coefficient':fit.beta}).to_csv(str(output)+'.coefficients.csv',index=False)


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    scripts={'SurvStudio':[sys.executable,str(Path(__file__).resolve()),'--worker','--data',a.data,'--output',str(out/'SurvStudio.csv')],
             'R_survival':['Rscript',str(Path(__file__).with_name('benchmark_reference.R')),a.data,str(out/'R_survival.csv')]}
    load_before=os.getloadavg() if hasattr(os,'getloadavg') else None
    for engine,command in scripts.items():
        subprocess.run(['/usr/bin/time','-f','%M','-o',str(out/(engine+'-peak-rss-kb.txt')),*command],check=True,env=env)
    rows=[]
    for engine in scripts:
        seconds=pd.read_csv(out/(engine+'.csv')).seconds.to_numpy()
        rows.append({'engine':engine,'n':260,'terms':5,'warmups':3,'measured_runs':10,'median_seconds':float(np.median(seconds)),
                     'q25_seconds':float(np.quantile(seconds,.25)),'q75_seconds':float(np.quantile(seconds,.75)),
                     'minimum_seconds':float(seconds.min()),'maximum_seconds':float(seconds.max()),
                     'process_peak_rss_kb':int((out/(engine+'-peak-rss-kb.txt')).read_text())})
    pd.DataFrame(rows).to_csv(out/'timing-memory-summary.csv',index=False)
    x=pd.read_csv(out/'SurvStudio.csv.coefficients.csv').coefficient.to_numpy();y=pd.read_csv(out/'R_survival.csv.coefficients.csv').coefficient.to_numpy()
    report={'host':platform.node(),'platform':platform.platform(),'load_before':load_before,
            'load_after':os.getloadavg() if hasattr(os,'getloadavg') else None,'prespecified_common_task':'unstratified Efron Cox fit, grade A reference, z,binary,m0',
            'coefficient_maximum_absolute_difference':float(np.max(np.abs(x-y))),'coefficient_tolerance':1e-6,
            'passed':bool(np.max(np.abs(x-y))<=1e-6),'measurement_scope':'operation time after common input preparation; peak RSS includes whole worker process',
            'limitation':'Shared server load and R millisecond timer resolution constrain timing interpretation. No web latency or human-usability inference.',
            'input_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'summary':rows}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();worker(a.data,Path(a.output)) if a.worker else run(a)

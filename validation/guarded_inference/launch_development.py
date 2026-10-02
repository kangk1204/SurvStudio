"""Bounded development pilot, deliberately using only the development seed."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--replicates',type=int,default=50);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
script=Path(__file__).with_name('study.py')
conditions=json.loads(script.with_name('protocol.json').read_text())['conditions']
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
def run(condition):
    command=[sys.executable,str(script),'run','--stage','development','--condition',condition,'--stop',str(a.replicates),'--output',str(out/(condition+'.sqlite'))]
    with (out/(condition+'.log')).open('w') as log: subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
with ThreadPoolExecutor(max_workers=a.workers) as pool: list(pool.map(run,conditions))
subprocess.run([sys.executable,str(script),'summarize',*[str(out/(c+'.sqlite')) for c in conditions],'--output',str(out/'summary.json')],check=True,env=env)

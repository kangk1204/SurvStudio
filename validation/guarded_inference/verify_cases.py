"""Verify fixed real/synthetic external predictions and endpoint-specific pooling in R."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import pandas as pd
from reanalyse_cases import save


def run(out,rlib=None):
    out=Path(out);recipe=json.loads((out/'recipe.json').read_text())
    if recipe is None:
        save(out/'independent-R-verification.json',{'status':'not_applicable_prediction_not_estimable','passed':False})
        return
    command=['Rscript',str(Path(__file__).with_name('case_reference.R')),str(out)]
    if rlib: command.append(rlib)
    subprocess.run(command,check=True)
    reference=pd.read_csv(out/'independent-R-values.csv');comparisons=[]
    for row in reference.itertuples(index=False):
        report=json.loads((out/(row.file+'-report.json')).read_text())
        found=float(row.value)
        expected=0. if row.quantity=='maximum_prediction_difference' else report['metrics']['c_index' if row.quantity=='C' else 'calibration_slope']
        comparisons.append({'file':row.file,'quantity':row.quantity,'R':found,'Python':expected,
                            'absolute_difference':None if expected is None else abs(found-expected),'tolerance':1e-6,
                            'status':'unavailable_python' if expected is None else 'passed' if abs(found-expected)<=1e-6 else 'failed'})
    if (out/'independent-R-pooling.csv').exists():
        pooled=json.loads((out/'pooled-aggregate.json').read_text())
        for row in pd.read_csv(out/'independent-R-pooling.csv').itertuples(index=False):
            expected=pooled[row.group]['quantities'][row.quantity]['pooled'][row.field]
            comparisons.append({'file':row.group,'quantity':row.quantity+':'+row.field,'R':row.value,'Python':expected,
                                'absolute_difference':abs(row.value-expected),'tolerance':1e-6,'status':'passed' if abs(row.value-expected)<=1e-6 else 'failed'})
    pd.DataFrame(comparisons).to_csv(out/'independent-R-comparison.csv',index=False)
    verification={'status':'passed' if all(r['status']=='passed' for r in comparisons) else 'incomplete_or_failed',
                  'passed':bool(comparisons and all(r['status']=='passed' for r in comparisons)),
                  'quantities':len(comparisons),'unavailable':sum(r['status']=='unavailable_python' for r in comparisons),
                  'maximum_absolute_difference':max((r['absolute_difference'] for r in comparisons if r['absolute_difference'] is not None),default=None),
                  'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file() and p.name!='independent-R-verification.json'},
                  'reference_script_sha256':hashlib.sha256(Path(__file__).with_name('case_reference.R').read_bytes()).hexdigest()}
    save(out/'independent-R-verification.json',verification)
    print(json.dumps({k:v for k,v in verification.items() if k!='hashes'}))
    if not verification['passed']: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rlib');a=p.parse_args();run(a.output,a.rlib)

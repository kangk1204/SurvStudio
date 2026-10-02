"""Audit coverage, immutable study sources and independent R operating characteristics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from reanalyse_cases import save


def run(root,with_r=False):
    root=Path(root).resolve();source=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'aggregate-manifest.json').read_text());freeze=json.loads((root/'freeze.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise ValueError('Evidence file changed: '+name)
    for name,digest in freeze['source_hashes'].items():
        raw=subprocess.check_output(['git','show',freeze['git_revision']+':'+name],cwd=source)
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('Frozen commit differs: '+name)
    rows=pd.read_csv(root/'replicate-outcomes.csv.gz',dtype={'error':'string'},low_memory=False);cell=pd.read_csv(root/'confirmation-cells.csv')
    if len(rows)!=256500 or rows.duplicated(['stage','condition','n','p','index','method']).any(): raise ValueError('Fixed index coverage differs')
    sizes=rows.groupby(['stage','condition','n','p','method']).size()
    for r in cell.itertuples(index=False):
        if sizes.loc[r.stage,r.condition,r.n,r.p,r.method]!=r.planned: raise ValueError('Missing fixed repetitions')
    comparisons=[]
    if with_r:
        subprocess.run(['Rscript',str(Path(__file__).with_name('aggregate_reference.R')),str(root)],check=True)
    reference=root/'independent-R-aggregate-values.csv'
    if reference.exists():
        grouped=cell.set_index(['stage','condition','n','p','method'])
        for r in pd.read_csv(reference).itertuples(index=False):
            actual=getattr(r,'value');expected=grouped.loc[(r.stage,r.condition,r.n,r.p,r.method),r.quantity]
            both_missing=bool(pd.isna(actual) and pd.isna(expected))
            integer=r.quantity in ('completed','failures','diagnostic_exceptions','missing','allowed')
            tolerance=0. if integer else 1e-12
            difference=None if both_missing else abs(actual-expected)
            comparisons.append({'stage':r.stage,'condition':r.condition,'n':r.n,'p':r.p,'method':r.method,'quantity':r.quantity,
                                'R':None if pd.isna(actual) else actual,'Python':None if pd.isna(expected) else expected,
                                'absolute_difference':difference,'tolerance':tolerance,'passed':both_missing or bool(difference<=tolerance)})
        pd.DataFrame(comparisons).to_csv(root/'independent-R-aggregate-comparison.csv',index=False)
    report={'fixed_datasets':len(rows.drop_duplicates(['stage','condition','n','p','index'])),'paired_method_records':len(rows),
            'calculation_failures':int(rows.calculation_failed.sum()),'frozen_source_hashes_verified':True,
            'independent_R_quantities':len(comparisons),'independent_R_passed':bool(comparisons and all(r['passed'] for r in comparisons)),
            'maximum_absolute_difference':max((r['absolute_difference'] for r in comparisons if r['absolute_difference'] is not None),default=None),
            'manifest_sha256':hashlib.sha256((root/'aggregate-manifest.json').read_bytes()).hexdigest(),
            'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'R_script_sha256':hashlib.sha256(Path(__file__).with_name('aggregate_reference.R').read_bytes()).hexdigest()}
    report['passed']=bool(report['fixed_datasets']==85500 and report['paired_method_records']==256500 and report['independent_R_passed'])
    save(root/'aggregate-audit.json',report);print(json.dumps(report))
    if with_r and not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--with-r',action='store_true');a=p.parse_args();run(a.results,a.with_r)

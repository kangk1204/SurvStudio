"""Prespecified default-linear and fixed-spline sensitivity reanalyses of existing cohorts.

All outputs, including any patient-level reference inputs, belong outside the repository.
Never choose a clinical basis or signature from external results. Preserve original QC
eligibility and endpoint mapping; this is existing-data reanalysis, not unseen validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper/scripts'))


def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(np.integer,np.bool_)): return value.item()
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run(a):
    os.environ['SURVSTUDIO_DATA']=str(Path(a.data).resolve())
    from common import (development_data,evaluate_development,GEO_COHORTS,geo_cohort,screen_validation_cohorts,
                        validation_row,locked_parts,component_row,pooled_validation,DEVELOPMENT)
    from survival_toolkit.marker_evaluation import MarkerSettings,validate_locked_recipe
    out=Path(a.output).resolve()
    if out.is_relative_to(ROOT): raise ValueError('Patient-level evidence/manuscript must remain outside Git')
    out.mkdir(parents=True,exist_ok=True)
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [Path(__file__),ROOT/'paper/scripts/common.py',ROOT/'src/survival_toolkit/clinical_basis.py',
                    ROOT/'src/survival_toolkit/marker_diagnostics.py',ROOT/'src/survival_toolkit/marker_evaluation.py']}
    configuration={'case':a.case,'clinical_basis':a.basis,'seed':20260926,'permutations':1000,'resamples':200,
                   'external_bootstraps':2000,'source_hashes':source_hashes,
                   'original_qc_eligibility_retained':True,'existing_external_data_reanalysis':True,
                   'primary_endpoint':'RFS' if a.case=='V' else 'OS','secondary_endpoint':'DMFS' if a.case=='V' else None,
                   'external_basis_or_signature_selection':False}
    if (out/'complete.json').exists():
        old=json.loads((out/'complete.json').read_text())
        if old['configuration']!=configuration: raise ValueError('Completed reanalysis source/settings differ')
        return
    if (out/'configuration.json').exists() and json.loads((out/'configuration.json').read_text())!=configuration:
        raise ValueError('Reanalysis source/settings differ; preserve the existing run and use a new directory')
    save(out/'configuration.json',configuration)
    frame,genes=development_data(a.case)
    # Hash each supplied data file; the original data tree remains unchanged.
    data_root=Path(a.data).resolve()
    manifest={str(p.relative_to(data_root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(data_root.rglob('*')) if p.is_file()}
    save(out/'private-input-hashes.json',manifest)
    analysis=out/'development-analysis.json'
    if analysis.exists(): result=json.loads(analysis.read_text())
    else:
        settings=MarkerSettings(n_permutations=1000,n_resamples=200,random_seed=20260926,clinical_basis=a.basis)
        result=evaluate_development(a.case,frame,genes,settings)
        save(analysis,result)
    recipe=result['locked_recipe'];save(out/'recipe.json',recipe)
    if recipe is None:
        save(out/'complete.json',{'configuration':configuration,'inference':result['inference'],
                                 'prediction_status':'not_estimable','completed':time.time()})
        return
    if a.case=='I':
        external={name:geo_cohort(name,recipe['markers']) for name in GEO_COHORTS}
        entries={name:{'endpoint':'os'} for name in external};screened=list(entries.values())
    else:
        screened,external=screen_validation_cohorts(a.case,recipe)
        entries={r['cohort']:r for r in screened}
    save(out/'eligibility.json',screened)
    rows=[]
    for name,data in external.items():
        for scaling in (('as_measured','within_cohort') if a.case=='I' else ('within_cohort',)):
            prefix=out/f'{name}-{scaling}'
            report_path=Path(str(prefix)+'-report.json')
            if report_path.exists(): report=json.loads(report_path.read_text())
            else:
                report=validate_locked_recipe(data,recipe,marker_scaling=scaling,n_bootstrap=2000,random_seed=20260926)
                save(report_path,report)
            parts=locked_parts(data,recipe,report,scaling)
            row={'cohort':name,'endpoint':entries[name]['endpoint'],'scaling':scaling,
                 'inference_status':report['inference']['status'],**validation_row(report),**component_row(parts)}
            rows.append(row)
            # Patient-level scores are private numerical verification inputs, never release files.
            pd.DataFrame({k:v for k,v in parts.items() if isinstance(v,np.ndarray)}).to_csv(Path(str(prefix)+'-private-scores.csv'),index=False)
            data[[*DEVELOPMENT[a.case]['covariates'],*recipe['markers'],recipe['outcome']['time_column'],
                  recipe['outcome']['event_column']]].to_csv(Path(str(prefix)+'-private-input.csv'),index=False)
            print(name,scaling,report['inference']['status'],flush=True)
    cohorts=pd.DataFrame(rows);cohorts.to_csv(out/'external-aggregate.csv',index=False)
    pooled={}
    for (endpoint,scaling),part in (cohorts.groupby(['endpoint','scaling']) if not cohorts.empty else []):
        pooled[f'{endpoint}:{scaling}']={'n':int(part.n.sum()),'events':int(part.events.sum()),
                                       'k':len(part),'inference_statuses':sorted(part.inference_status.unique()),
                                       **pooled_validation(part)}
    # In particular, Case V never pools RFS and DMFS into one gain.
    save(out/'pooled-aggregate.json',pooled)
    save(out/'complete.json',{'configuration':configuration,'development_inference':result['inference'],
                             'prediction_status':'estimable_exploratory' if not result['inference']['allowed'] else 'estimable_assumption_dependent',
                             'cohorts':len(external),'pooled':pooled,'completed':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['I','IV','V'],required=True)
    p.add_argument('--basis',choices=['linear','restricted_cubic_spline'],required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);run(p.parse_args())

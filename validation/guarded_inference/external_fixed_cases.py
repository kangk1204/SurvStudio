"""Queued external reanalysis: frozen development transforms primary, rescaling sensitivity.

The development signature/basis is already fixed before any external output is read.
Patient-level reference inputs and every manuscript artifact stay outside Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from reanalyse_cases import save

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper/scripts'))


def run(a):
    development=Path(a.development)
    while not development.exists():
        if not a.wait: raise FileNotFoundError(development)
        time.sleep(30)
    os.environ['SURVSTUDIO_DATA']=a.data
    from common import GEO_COHORTS,geo_cohort,screen_validation_cohorts,validation_row,locked_parts,component_row,pool_interval
    from survival_toolkit.marker_evaluation import validate_locked_recipe
    from survival_toolkit.marker_qualification import qualify_marker_result,qualify_external_result
    from survival_toolkit.analysis import _cohort_frame
    out=Path(a.output).resolve()
    if out.is_relative_to(ROOT): raise ValueError('Real-case evidence must stay outside Git')
    out.mkdir(parents=True,exist_ok=True)
    result=qualify_marker_result(json.loads(development.read_text()));recipe=result['locked_recipe']
    config={'case':a.case,'development_sha256':hashlib.sha256(development.read_bytes()).hexdigest(),
            'clinical_basis':result['clinical_basis'],'primary_marker_scaling':'as_measured',
            'sensitivity_marker_scaling':'within_cohort','clinical_transform':'frozen_development_only',
            'primary_endpoint':'rfs' if a.case=='V' else 'os','secondary_endpoint':'dmfs' if a.case=='V' else None,
            'external_basis_or_signature_selection':False,'existing_external_data_reanalysis':True,
            'external_bootstraps':2000,'seed':20260926,
            'engineering_qualification':result['inference'].get('engineering_qualification'),
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path(__file__).with_name('case_reference.R'),Path(__file__).with_name('verify_cases.py'),
                 ROOT/'paper/scripts/common.py',*[ROOT/'src/survival_toolkit'/f'{n}.py' for n in
                    ('clinical_basis','marker_diagnostics','marker_evaluation','marker_screen','analysis','marker_qualification')]]}}
    if (out/'configuration.json').exists() and json.loads((out/'configuration.json').read_text())!=config:
        raise ValueError('External reanalysis configuration changed; preserve this output')
    if (out/'complete.json').exists(): return
    save(out/'configuration.json',config);save(out/'recipe.json',recipe)
    if recipe is None:
        save(out/'complete.json',{'configuration':config,'prediction_status':'not_estimable','inference':result['inference']})
        return
    if a.case=='I':
        frames={name:geo_cohort(name,recipe['markers']) for name in GEO_COHORTS}
        entries={name:{'endpoint':'os'} for name in frames};screened=[{'cohort':k,**v} for k,v in entries.items()]
    else:
        screened,frames=screen_validation_cohorts(a.case,recipe);entries={x['cohort']:x for x in screened}
    save(out/'eligibility.json',screened)
    rows=[]
    for name,data in frames.items():
        for scaling in ('as_measured','within_cohort'):
            prefix=out/f'{name}-{scaling}';report_file=Path(str(prefix)+'-report.json')
            if report_file.exists(): report=json.loads(report_file.read_text())
            else:
                report=qualify_external_result(validate_locked_recipe(data,recipe,marker_scaling=scaling,n_bootstrap=2000,random_seed=20260926),recipe)
                save(report_file,report)
            parts=locked_parts(data,recipe,report,scaling)
            rows.append({'cohort':name,'endpoint':entries[name]['endpoint'],'scaling':scaling,
                         'role':'primary' if scaling=='as_measured' else 'sensitivity',
                         'inference_status':report['inference']['status'],**validation_row(report),**component_row(parts)})
            pd.DataFrame({k:v for k,v in parts.items() if isinstance(v,np.ndarray)}).to_csv(Path(str(prefix)+'-private-scores.csv'),index=False)
            outcome=recipe['outcome'];selected=_cohort_frame(data,time_column=outcome['time_column'],event_column=outcome['event_column'],
                                                            event_positive_value=outcome['event_positive_value'])
            columns=[*recipe['clinical']['columns'],*[m for m in recipe['markers'] if m in data],outcome['time_column'],outcome['event_column']]
            data.loc[list(selected.attrs['source_row_index']),columns].to_csv(Path(str(prefix)+'-private-input.csv'),index=False)
            print(name,scaling,'done',flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/'external-aggregate.csv',index=False)
    pooled={}
    for (endpoint,scaling),part in (table.groupby(['endpoint','scaling']) if not table.empty else []):
        group={'n':int(part.n.sum()),'events':int(part.events.sum()),'k':len(part),'endpoint':endpoint,
               'role':'primary' if scaling=='as_measured' else 'sensitivity','quantities':{}}
        for key,lower,upper in [('c','c_lower','c_upper'),('clinical_c','clinical_lower','clinical_upper'),
                                ('delta_c','delta_lower','delta_upper'),('calibration_slope','calibration_lower','calibration_upper')]:
            eligible=part[[key,lower,upper]].notna().all(axis=1)&(part[upper]>part[lower])
            group['quantities'][key]={'eligible_cohorts':int(eligible.sum()),'unavailable_cohorts':part.loc[~eligible,'cohort'].tolist(),
                                     'pooled':pool_interval(part.loc[eligible],key,lower,upper) if eligible.sum()>=2 else None}
        pooled[f'{endpoint}:{scaling}']=group
    save(out/'pooled-aggregate.json',pooled)
    save(out/'complete.json',{'configuration':config,'prediction_status':'estimable','development_inference':result['inference'],
                             'cohorts':len(frames),'pooled':pooled,'completed':time.time(),
                             'independent_R_status':'pending'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['I','IV','V'],required=True);p.add_argument('--development',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--wait',action='store_true');run(p.parse_args())

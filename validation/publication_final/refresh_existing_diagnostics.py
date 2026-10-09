"""Reevaluate development diagnostics while retaining existing fixed predictions.

Patient-level inputs and actual recipes must remain outside the Git checkout.
This is a diagnostic reanalysis of known data, not a new prediction study.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def run(args):
    out=args.output.resolve()
    if out.is_relative_to(ROOT) or args.original.resolve().is_relative_to(ROOT):raise ValueError('Keep actual-case evidence private')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['SURVSTUDIO_DATA']=str(args.data.resolve());sys.path.insert(0,str(ROOT/'paper/scripts'))
    from common import development_data,DEVELOPMENT
    from survival_toolkit.marker_evaluation import prepare_marker_cohort,recipe_hash
    from survival_toolkit.marker_bootstrap import diagnose_markers
    from survival_toolkit.marker_qualification import qualify_marker_result,qualify_external_result
    def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
    stem='case-'+args.case+'-'+args.basis
    original=args.original/'cases_final'/stem;analysis=args.original/'cases'/stem/'development-analysis.json'
    old=json.loads(analysis.read_text());old_recipe=json.loads((original/'recipe.json').read_text())
    seed=int(np.random.SeedSequence([2026100908,('I','IV','V').index(args.case),int(args.basis!='linear')]).generate_state(1)[0])
    save(out/'lock.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        locked_utc=datetime.now(timezone.utc).isoformat(),original_analysis_sha256=hashlib.sha256(analysis.read_bytes()).hexdigest(),
        original_recipe_sha256=hashlib.sha256((original/'recipe.json').read_bytes()).hexdigest(),
        diagnostic_draws=9999,seed=seed,seed_family=2026100908,candidate='restricted_wild',
        complete_original_marker_family=True,predictions_and_metrics_recomputed=False,
        external_diagnostic_boundary='Previous R-verified external diagnostics retained; new v3 development withholding and qualification propagated. No new external v3 diagnostic certification.'))
    started=time.perf_counter();frame,genes=development_data(args.case);spec=DEVELOPMENT[args.case]
    cohort=prepare_marker_cohort(frame,time_column=spec['time'],event_column=spec['event'],marker_columns=genes,
        clinical_columns=spec['covariates'],categorical_clinical=spec['categorical'],clinical_basis=args.basis,event_positive_value=1)
    if len(cohort.time)!=old['cohort']['n'] or cohort.row_mask_hash!=old['cohort']['row_mask_hash'] or set(cohort.marker_names)!={r['marker'] for r in old['marker_table']} or len(cohort.marker_names)!=len(old['marker_table']):
        raise ValueError('Reconstructed original development cohort/family changed')
    if cohort.clinical_encoder!=old_recipe['clinical']['encoder']:raise ValueError('Original frozen clinical encoder changed')
    medians=np.nanmedian(cohort.markers,axis=0);block=np.where(np.isfinite(cohort.markers),cohort.markers,medians)
    diagnostic=diagnose_markers(cohort,block,candidate='restricted_wild',bootstrap_seed=seed,draws=9999)
    diagnostic.setdefault('provenance',{})['existing_data_diagnostic_reanalysis']=True
    diagnostic['provenance']['original_raw_calculation_method_version']=old['method_version']
    recipe=copy.deepcopy(old_recipe);recipe['inference']=copy.deepcopy(diagnostic);recipe['recipe_hash']=recipe_hash(recipe)
    result=dict(primary_lens='added_value',method_version='marker-inference/3',clinical_basis=args.basis,inference=diagnostic,
        marker_table=[],tier_counts={},locked_recipe=recipe)
    qualified=qualify_marker_result(result);recipe=qualified['locked_recipe']
    for key in ('model','clinical','markers','marker_medians','outcome'):
        if recipe[key]!=old_recipe[key]:raise ValueError('Diagnostic refresh changed fixed prediction '+key)
    save(out/'diagnostics.json',diagnostic);save(out/'recipe.json',recipe)
    reports=[]
    for path in sorted(original.glob('*-report.json')):
        report=json.loads(path.read_text());metrics=copy.deepcopy(report['metrics']);previous=copy.deepcopy(report['inference'])
        report['previous_external_inference']=previous;report['inference']=copy.deepcopy(qualified['inference'])
        report['inference']['reasons']+=['retained external v2: '+reason for reason in previous.get('reasons',[])]
        report=qualify_external_result(report,recipe);report['recipe_hash']=recipe['recipe_hash']
        if report['metrics']!=metrics:raise ValueError('State refresh changed fixed external metrics')
        save(out/path.name,report);reports.append(dict(file=path.name,metrics_sha256=hashlib.sha256(json.dumps(metrics,sort_keys=True).encode()).hexdigest(),withheld=not report['inference']['allowed']))
    save(out/'summary.json',dict(case=args.case,basis=args.basis,n=len(cohort.time),markers=len(cohort.marker_names),
        diagnostic_status=diagnostic['status'],product_inference=qualified['inference'],bootstrap=diagnostic.get('bootstrap'),
        fixed_prediction_components_unchanged=True,external_metrics_unchanged=True,external_reports=reports,
        elapsed_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Known-data diagnostic reevaluation; unchanged external predictions retain their original evidence dates.'))
    print(json.dumps(dict(case=args.case,basis=args.basis,status='completed',seconds=time.perf_counter()-started)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['I','IV','V'],required=True);p.add_argument('--basis',choices=['linear','restricted_cubic_spline'],required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

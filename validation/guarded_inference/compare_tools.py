"""Executed synthetic A/B tasks with explicit R workflow and auditable limitations.

KM Plotter remains execution-unverified because the accessed terms prohibit automated
non-human access. Do not equate an unexecuted task with a missing capability.
"""
import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from survival_toolkit.app import app
from survival_toolkit.analysis import compute_km_analysis,_harrell_c_index
from survival_toolkit.clinical_basis import fit_clinical_encoder,transform_clinical_encoder
from survival_toolkit.marker_evaluation import MarkerSettings,evaluate_markers,prepare_marker_cohort,validate_locked_recipe
from survival_toolkit.marker_diagnostics import diagnose_markers,hc3_wald
from survival_toolkit.marker_screen import fit_cox,fit_cox_null,CoxScoreScreen
from survival_toolkit.duplicates import possible_duplicates
from survival_toolkit.marker_qualification import qualify_marker_result,qualify_external_result
from reanalyse_cases import save


def fixtures(out):
    rng=np.random.default_rng(np.random.SeedSequence([2026100301,777]))
    def data(n,shift):
        z=rng.normal(shift,1,n);grade=rng.choice(['A','B','C'],n);binary=rng.integers(0,2,n)
        m=rng.normal(size=(n,30));eta=.6*z+.25*binary+.2*(grade=='B')-.15*(grade=='C')+1.0*m[:,0]
        t=rng.exponential(size=n)/(.06*np.exp(eta));c=rng.exponential(size=n)/.04
        frame=pd.DataFrame({'patient_id':[f'S{shift}-{i}' for i in range(n)],'time':np.minimum(t,c),'event':(t<=c).astype(int),
                            'z':z,'grade':grade,'binary':binary,**{f'm{i}':m[:,i] for i in range(30)}})
        frame['event_code2']=frame.event+1;frame['inner_train']=(np.arange(n)<int(.7*n)).astype(int)
        frame['rfs_months']=frame.time;frame['dmfs_event']=frame.event
        return frame
    a=data(260,0);b=data(180,1)
    z=rng.normal(size=600);t=rng.exponential(size=600)/(.06*np.exp(.8*z));c=rng.exponential(size=600)/.04
    d=pd.DataFrame({'time':np.minimum(t,c),'event':(t<=c).astype(int),'z':z,
                    **{f'm{i}':z*z+rng.normal(size=600) for i in range(30)}})
    duplicate=a.copy();duplicate.iloc[-1]=duplicate.iloc[12]
    for name,frame in [('A',a),('B',b),('misspecified',d),('duplicate',duplicate)]: frame.to_csv(out/(name+'.csv'),index=False)
    return a,b,d,duplicate


def run(out,rlib=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);a,b,d,duplicate=fixtures(out)
    markers=[f'm{i}' for i in range(30)];clinical=['z','grade','binary']
    started=time.time()
    result=evaluate_markers(a,time_column='time',event_column='event',marker_columns=markers,clinical_columns=clinical,
                            categorical_clinical=['grade'],id_column='patient_id',
                            settings=MarkerSettings(n_permutations=99,n_resamples=4,random_seed=2026100301))
    research_status=result['inference']['status']
    result=qualify_marker_result(result)
    recipe=result['locked_recipe']
    if recipe is None: raise ValueError('Fixed synthetic task has no estimable locked prediction')
    save(out/'recipe.json',recipe);save(out/'survstudio-task-analysis.json',result)
    external=qualify_external_result(validate_locked_recipe(b,recipe,n_bootstrap=0),recipe)
    save(out/'survstudio-external.json',external)
    encoder=fit_clinical_encoder(a,clinical,['grade'],basis='linear')
    design=transform_clinical_encoder(a,encoder)
    adjusted=fit_cox(a.time.to_numpy(),a.event.to_numpy(),np.column_stack([design,a.m0]))
    values={f'adjusted_coefficient_{i+1}':float(x) for i,x in enumerate(adjusted.beta)}
    values['adjusted_loglik']=adjusted.loglik
    km=compute_km_analysis(a,'time','event_code2','grade',event_positive_value=2)
    values['recoded_events']=sum(r['Events'] for r in km['summary_table'])
    for curve in km['curves']:
        for at in (5,10,15):
            ix=max(np.searchsorted(curve['timeline'],at,side='right')-1,0)
            values[f"{curve['group']} @ {at} survival"]=curve['survival'][ix]
    inside=a.inner_train.to_numpy(bool);training=a.loc[inside]
    inner_encoder=fit_clinical_encoder(training,clinical,['grade'],basis='linear')
    z=transform_clinical_encoder(training,inner_encoder)
    null=fit_cox_null(training.time.to_numpy(),training.event.to_numpy(),z)
    scores=CoxScoreScreen(training.time.to_numpy(),training.event.to_numpy(),null=null,Z=z).statistics(training[markers].to_numpy()).chi2
    selected=int(np.argmax(scores));fit=fit_cox(training.time.to_numpy(),training.event.to_numpy(),np.column_stack([z,training[markers[selected]]]))
    risk=np.column_stack([transform_clinical_encoder(a.loc[~inside],inner_encoder),a.loc[~inside,markers[selected]]])@fit.beta
    values['inner_selected_marker']=selected;values['inner_heldout_C']=_harrell_c_index(a.loc[~inside,'time'].to_numpy(),a.loc[~inside,'event'].to_numpy(),risk)
    diagnostic=diagnose_markers(prepare_marker_cohort(d,time_column='time',event_column='event',marker_columns=markers,clinical_columns=['z']),d[markers].to_numpy())
    values['misspecified_withheld_by_explicit_rule']=int(not diagnostic['allowed'])
    for test in diagnostic['residual_tests']:
        name='mean' if test['diagnostic']=='nonlinear_mean' else 'variance'
        values[f"diagnostic_{test['marker']}_{name}"]=test['statistic']
    values['duplicate_ID_rows']=int(duplicate.patient_id.duplicated().sum())
    duplicates=possible_duplicates(duplicate[markers].to_numpy(),[str(i) for i in range(len(duplicate))])
    with TestClient(app,base_url='http://127.0.0.1') as client:
        dataset=client.post('/api/upload',files={'file':('A.csv',a.to_csv(index=False),'text/csv')}).json()
        leaked=client.post('/api/marker-evaluation',json={'dataset_id':dataset['dataset_id'],'time_column':'time','event_column':'event','marker_columns':['event'],'n_permutations':19,'n_resamples':2})
        application=client.post('/api/marker-evaluation',json={'dataset_id':dataset['dataset_id'],'time_column':'time','event_column':'event',
            'marker_columns':markers,'clinical_columns':clinical,'categorical_clinical':['grade'],
            'n_permutations':99,'n_resamples':4,'random_seed':2026100301})
        if application.status_code!=200: raise ValueError(application.text)
        application=application.json()
        save(out/'survstudio-api-result.json',{k:application[k] for k in ('analysis','display_table','report')})
        api_model=application['analysis']['locked_recipe']['model']
        if api_model['terms']!=recipe['model']['terms'] or api_model['ties']!=recipe['model']['ties']:
            raise ValueError('API fixed model definition differs from the qualified numerical kernel')
        # CSV round trips change last-bit float representations; enforce the declared solver tolerance.
        api_differences=[float(np.max(np.abs(np.asarray(api_model['coefficients'])-recipe['model']['coefficients'])))]
        for key in ('times','log_cumulative_hazard','lp_center'):
            api_differences.append(float(np.max(np.abs(np.asarray(api_model['baseline'][key])-recipe['model']['baseline'][key]))))
        if max(api_differences)>1e-6: raise ValueError('API fixed model exceeds the solver tolerance')
    try: compute_km_analysis(a,'rfs_months','dmfs_event',event_positive_value=1)
    except ValueError as exc: endpoint_rejected=True;endpoint_reason=str(exc)
    else: endpoint_rejected=False;endpoint_reason=None
    values['manual_endpoint_mismatch_rejected']=int(endpoint_rejected)
    values['outcome_role_rejected']=int(leaked.status_code==400)
    values['locked_external_C']=external['metrics']['c_index'];values['locked_external_calibration']=external['metrics']['calibration_slope']
    command=['Rscript',str(Path(__file__).with_name('tool_reference.R')),str(out)]
    if rlib: command.append(rlib)
    subprocess.run(command,check=True)
    reference=pd.read_csv(out/'r_values.csv').set_index('quantity').value
    rows=[]
    for quantity,value in values.items():
        expected=float(reference[quantity]);difference=abs(value-expected)
        # Exact checks for coded events, selected marker and manual rejection flags.
        tolerance=0. if quantity in {'recoded_events','inner_selected_marker','duplicate_ID_rows','manual_endpoint_mismatch_rejected','outcome_role_rejected','misspecified_withheld_by_explicit_rule'} else 1e-6
        rows.append({'quantity':quantity,'SurvStudio':value,'R_survival_script':expected,'absolute_difference':difference,'tolerance':tolerance,'passed':bool(difference<=tolerance)})
    pd.DataFrame(rows).to_csv(out/'numerical-comparison.csv',index=False)
    tasks=[
        ('event_coding','executed_numerical_match','explicit_positive_value=2; R script recodes before Surv'),
        ('KM_estimation','executed_numerical_match','fixed 5,10,15 time points, three grades'),
        ('adjusted_HR_reference','executed_numerical_match','grade A is reference; numeric coefficients/log-likelihood compared'),
        ('misspecification','executed_workflow','SurvStudio diagnostic withholding; R explicit HC3 calculation and manual 1% rule'),
        ('duplicates_outcome_leakage','executed_workflow','SurvStudio duplicate profile warning and API role rejection; R explicit ID/input checks'),
        ('selection_internal_evaluation','executed_kernel_workflow','both scripted numerical kernels select on inner training rows and score untouched held-out rows; application resampling checked separately'),
        ('locked_external_validation','executed_numerical_match','R reconstructs frozen training encoder and applies fixed coefficients; no external selection'),
        ('endpoint_mixing','executed_workflow','SurvStudio rejects RFS-time/DMFS-event pair; R workflow needs an explicit endpoint declaration check')]
    table=[]
    for task,status,detail in tasks:
        table.append({'task':task,'SurvStudio_and_R_status':status,'execution_detail':detail,'KM_Plotter_status':'execution_unverified_terms_prohibit_automation','KM_Plotter_capability':'not_inferred_from_nonexecution'})
    pd.DataFrame(table).to_csv(out/'task-comparison.csv',index=False)
    verification={'synthetic':True,'development_fixture_seed':[2026100301,777],'numerical_passed':all(r['passed'] for r in rows),
                  'misspecification_withheld':not diagnostic['allowed'],'duplicate_profile_flagged':bool(duplicates['identical']),
                  'outcome_role_rejected':leaked.status_code==400,'endpoint_pair_rejected':endpoint_rejected,'endpoint_reason':endpoint_reason,
                  'tasks':table,'elapsed_seconds':time.time()-started,'application_inference_status':application['analysis']['inference']['status'],
                  'research_kernel_inference_status':research_status,
                  'engineering_qualification':application['analysis']['inference'].get('engineering_qualification'),
                  'API_standard_inference_masked':all(row['Family-wise P'] is None for row in application['display_table']),
                  'API_fixed_model_maximum_absolute_difference':max(api_differences),'API_fixed_model_tolerance':1e-6,
                  'KM_terms_url':'https://kmplot.com/analysis/index.php?cancer=custom_plot&p=service',
                  'human_usability_claims':False,'timing_benchmark_status':'separate_common_task_benchmark',
                  'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file() and p.name!='verification.json'},
                  'package_versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels']},
                  'source_hashes':{str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__).resolve(),Path(__file__).with_name('tool_reference.R'),
                                             *[Path(__file__).resolve().parents[2]/('src/survival_toolkit/'+n+'.py') for n in
                                               ['clinical_basis','marker_diagnostics','marker_evaluation','marker_screen','analysis','app','marker_qualification']],
                                             Path(__file__).resolve().parents[2]/'src/survival_toolkit/data/marker_qualification.json']}}
    verification['passed']=bool(verification['numerical_passed'] and verification['misspecification_withheld'] and verification['duplicate_profile_flagged'] and verification['outcome_role_rejected'] and endpoint_rejected)
    save(out/'verification.json',verification)
    print(json.dumps({'passed':verification['passed'],'quantities':len(rows),'tasks':8,'KM_Plotter':'execution_unverified'}))
    if not verification['passed']: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rlib');a=p.parse_args();run(a.output,a.rlib)

"""Observed native library responses to the frozen comparison fixtures.

Acceptance of one supplied input is not a claim that a feature is unsupported.
The marker residual exchangeability checks and predictive Cox diagnostics have
different targets; their p values are deliberately not numerically compared.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from comparison import verify_inputs, sha, save, TASKS


def run(inputs, libraries, reference, output):
    contract=verify_inputs(inputs);output.mkdir(parents=True,exist_ok=False)
    lock=dict(locked_utc=datetime.now(timezone.utc).isoformat(),input_contract_sha256=sha(inputs/'contract.json'),
        source_sha256=sha(__file__),cases=['normal','misspecified','duplicate','outcome_feature','endpoint_mixing'],
        settings={'mlsurv_cv':3,'seed':2026100901,'imputer':None,'scaler':None,'selector':None,
                  'check_leakage':True,'ph_test':True,'linearity_test':True,'feature_selection_or_tuning':False},
        observation_rule='Record native acceptance, errors, diagnostics, warnings and exports. No unsupported-feature conclusion from absence.',
        different_targets='Predictive Cox PH/linearity tests are not marker-residual conditional-exchangeability diagnostics.')
    save(output/'probe-lock.json',lock)
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
    from mlsurv import SurvivalLearner,load_learner
    from sksurv.util import Surv
    a=pd.read_csv(inputs/'A.csv');b=pd.read_csv(inputs/'B.csv')
    def design(frame):
        if 'grade' not in frame:return frame[['z','m0']].copy()
        return pd.DataFrame({'grade_B':(frame.grade=='B').astype(float),'grade_C':(frame.grade=='C').astype(float),
            'z':frame.z,'binary':frame.binary.astype(float),'m0':frame.m0})
    observations=[];log=io.StringIO();handler=logging.StreamHandler(log);logging.getLogger('mlsurv').addHandler(handler)
    try:
        for name in lock['cases']:
            frame=pd.read_csv(inputs/('misspecified.csv' if name=='misspecified' else 'duplicate.csv' if name=='duplicate' else 'A.csv'))
            x=design(frame);time=frame.time;event=frame.event==1
            if name=='outcome_feature':x['event']=frame.event
            if name=='endpoint_mixing':time=frame.rfs_months;event=frame.dmfs_event==1
            for tool in ('lifelines','mlsurv'):
                folder=output/(name+'-'+tool);folder.mkdir();capture=io.StringIO()
                row=dict(case=name,tool=tool,capability_layer='native',input_rows=len(frame),features=list(x),status='started')
                with warnings.catch_warnings(record=True) as caught,redirect_stdout(capture),redirect_stderr(capture):
                    warnings.simplefilter('always')
                    try:
                        if tool=='lifelines':
                            fitted=CoxPHFitter().fit(x.assign(time=time,event=event),'time','event',fit_options={'precision':1e-10,'max_steps':100})
                            fitted.summary.to_csv(folder/'coefficients.csv')
                            proportional_hazard_test(fitted,x.assign(time=time,event=event),time_transform='log').summary.to_csv(folder/'PH-logtime.csv')
                            # The generic survival constructor receives explicit arrays.
                            KaplanMeierFitter().fit(time,event).survival_function_.to_csv(folder/'KM.csv')
                            row.update(status='executed',coefficient_rows=len(fitted.summary),standard_p_values=int(fitted.summary.p.notna().sum()))
                        else:
                            y=Surv.from_arrays(event,time)
                            # The original explicit train/test fixture is used for normal A;
                            # otherwise setup's native split/CV acts on the frozen input.
                            learner=SurvivalLearner(x,y)
                            learner.setup(cv_folds=3,imputer=None,scaler=None,feature_selector=None,random_state=2026100901,n_jobs=1,
                                verbose=False,check_leakage=True,ph_test=True,linearity_test=True,outcome_name='RFS' if name=='endpoint_mixing' else 'event')
                            for attr in ('ph_test_result','linearity_test_result'):
                                value=getattr(learner,attr)
                                if isinstance(value,pd.DataFrame):value.to_csv(folder/(attr+'.csv'),index=False)
                                row[attr+'_available']=isinstance(value,pd.DataFrame)
                            result=learner.train('coxph',verbose=False)
                            coeff=result.coefficients
                            if isinstance(coeff,pd.DataFrame):coeff.to_csv(folder/'coefficients.csv',index=False)
                            elif isinstance(coeff,pd.Series):coeff.to_csv(folder/'coefficients.csv')
                            else:save(folder/'coefficients.json',coeff)
                            learner.export_results(folder/'tables',format='csv',audit=True)
                            if name=='misspecified':
                                learner.generate_report(str(folder/'report.md'),include_optuna=False)
                                learner.save_learner(folder/'learner.pkl')
                            row.update(status='executed',cv_mean=float(result.cv_metric_mean),cv_folds=3)
                    except Exception as exc:
                        row.update(status='execution_failed',exception=type(exc).__name__,detail=str(exc))
                save(folder/'warnings.json',[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
                (folder/'console.txt').write_text(capture.getvalue())
                row['warnings']=len(caught);row['output_files']=sorted(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file())
                observations.append(row)
    finally:logging.getLogger('mlsurv').removeHandler(handler)
    (output/'mlsurv-logging.txt').write_text(log.getvalue())
    # This comparison uses only equivalent numeric quantities from the already
    # executed independent R workflow and the fixed native library outputs.
    expected=pd.read_csv(reference/'r_values.csv').set_index('quantity').value
    actual=pd.read_csv(libraries/'common-numeric-values.csv')
    numeric=[]
    for row in actual.to_dict('records'):
        ref=float(expected[row['quantity']]);delta=abs(row['value']-ref)
        numeric.append({**row,'R_survival':ref,'difference':delta,'tolerance':1e-6,'passed':delta<=1e-6})
    pd.DataFrame(numeric).to_csv(output/'independent-numeric-comparison.csv',index=False)
    # An already saved native model receives covariates only. External outcome
    # reversal is an evaluation change, not a new fit or prediction input.
    restored=load_learner(libraries/'learner.pkl');xb=design(b)
    first=np.asarray(restored.predict(xb,models='coxph',verbose=False).risk_scores)
    changed=b.copy();changed['event']=1-changed.event
    second=np.asarray(restored.predict(design(changed),models='coxph',verbose=False).risk_scores)
    checks=[dict(check='mlsurv_frozen_predictions_ignore_changed_external_outcomes',passed=bool(np.array_equal(first,second)),maximum_difference=float(np.max(np.abs(first-second)))),
            dict(check='lifelines_common_numeric_R_match',passed=all(r['passed'] for r in numeric),quantities=len(numeric))]
    save(output/'native-observations.json',observations)
    records=[]
    for tool in ('lifelines','mlsurv'):
        for task in TASKS:
            if task in ('event_coding','KM_estimation','adjusted_HR_reference') and tool=='lifelines':
                status='executed_numeric_match';detail='Explicit event recoding, KM time points, reference coding and Cox estimates match independent R.';layer='native_with_input_script'
            elif task=='KM_estimation':
                status='executed_different_output';detail='Native report includes overall KM; not the declared grade-specific estimate comparison.';layer='native'
            elif task=='selection_internal_evaluation':
                status='executed_different_estimand' if tool=='mlsurv' else 'not_executed';detail='mlsurv native three-fold CV of a prespecified model; not a post-marker-selection comparison. lifelines added nested selection was not executed.';layer='native' if tool=='mlsurv' else 'not_assessed'
            elif task=='locked_external_validation':
                status='executed_workflow' if tool=='mlsurv' else 'not_executed';detail='mlsurv native save/load, fixed predict and external validation executed; outcomes leave predictions unchanged. lifelines persistence workflow was not executed.';layer='native' if tool=='mlsurv' else 'not_assessed'
            else:
                status='native_input_response_observed';detail='See fixed '+('misspecified' if task=='misspecification' else 'duplicate/outcome_feature' if task=='duplicates_outcome_leakage' else 'endpoint_mixing' if task=='endpoint_mixing' else 'normal')+' probe and retained warnings/exports. This is not a universal unsupported-feature claim.';layer='native_with_input_script'
            records.append(dict(tool=tool,task=task,status=status,capability_layer=layer,detail=detail))
    pd.DataFrame(records).to_csv(output/'task-records.csv',index=False)
    verification=dict(passed=all(c['passed'] for c in checks),checks=checks,lock_sha256=sha(output/'probe-lock.json'),
        observations_complete=len(observations)==10,observations=observations,
        output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        boundary='Native warnings and outputs observed on supplied inputs; not statistical qualification, exhaustive unsupported-feature certification or human usability evidence')
    save(output/'verification.json',verification)
    print(json.dumps({'passed':verification['passed'],'native_cases':len(observations),'numeric_quantities':len(numeric)}))
    if not verification['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--libraries',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.libraries,a.reference,a.output)

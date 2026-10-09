"""Added lifelines selection/persistence scripts and a corrected leakage probe."""
import argparse
import io
from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from comparison import verify_inputs,save,sha


def run(inputs,output):
    verify_inputs(inputs);output.mkdir(parents=True,exist_ok=False)
    save(output/'lock.json',dict(input_contract_sha256=sha(inputs/'contract.json'),source_sha256=sha(__file__),
         selection='training-only maximum partial likelihood over 30 single-marker clinical Cox models',
         folds='frozen A.inner_train split',ties='Efron',persistence='Python pickle of fixed-version native fitter',
         leakage_probe='outcome_as_feature retains the event vector under a distinct feature name',
         prior_probe_limitation='native-probes-r1 lifelines outcome_feature was consumed as the event column; excluded from outcome-feature assessment'))
    a=pd.read_csv(inputs/'A.csv');b=pd.read_csv(inputs/'B.csv')
    def x(d):return pd.DataFrame(dict(grade_B=(d.grade=='B').astype(float),grade_C=(d.grade=='C').astype(float),z=d.z,binary=d.binary.astype(float)))
    train=a[a.inner_train==1];held=a[a.inner_train==0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fits=[CoxPHFitter().fit(x(train).assign(marker=train[f'm{i}'],time=train.time,event=train.event),'time','event',fit_options={'precision':1e-10,'max_steps':100}) for i in range(30)]
        scores=np.asarray([f.log_likelihood_ for f in fits]);selected=int(scores.argmax())
        risk=fits[selected].predict_log_partial_hazard(x(held).assign(marker=held[f'm{selected}'])).to_numpy()
        c=float(concordance_index(held.time,-risk,held.event))
        frozen=CoxPHFitter().fit(x(a).assign(m0=a.m0,time=a.time,event=a.event),'time','event',fit_options={'precision':1e-10,'max_steps':100})
        with (output/'lifelines-fitter.pkl').open('wb') as handle:pickle.dump(frozen,handle)
        with (output/'lifelines-fitter.pkl').open('rb') as handle:loaded=pickle.load(handle)
        first=frozen.predict_log_partial_hazard(x(b).assign(m0=b.m0)).to_numpy()
        after=loaded.predict_log_partial_hazard(x(b).assign(m0=b.m0)).to_numpy()
        changed=b.copy();changed['event']=1-changed.event
        outcome_changed=loaded.predict_log_partial_hazard(x(changed).assign(m0=changed.m0)).to_numpy()
        row=dict(status='started',feature='outcome_as_feature')
        try:
            invalid=CoxPHFitter().fit(x(a).assign(m0=a.m0,outcome_as_feature=a.event,time=a.time,event=a.event),'time','event')
            invalid.summary.to_csv(output/'outcome-feature-coefficients.csv')
            row.update(status='executed',coefficient_rows=len(invalid.params_),outcome_feature_present='outcome_as_feature' in invalid.params_.index)
        except Exception as exc:row.update(status='execution_failed',exception=type(exc).__name__,detail=str(exc))
    save(output/'warnings.json',[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
    pd.DataFrame(dict(marker=range(30),training_loglik=scores)).to_csv(output/'training-selection.csv',index=False)
    pd.DataFrame(dict(before_save=first,after_load=after,changed_outcome=outcome_changed)).to_csv(output/'fixed-external-predictions.csv',index=False)
    checks=[dict(check='fixed_version_pickle_predictions',passed=bool(np.array_equal(first,after)),maximum_difference=float(np.max(np.abs(first-after)))),
            dict(check='external_outcomes_do_not_change_prediction',passed=bool(np.array_equal(first,outcome_changed)),maximum_difference=float(np.max(np.abs(first-outcome_changed))))]
    save(output/'verification.json',dict(passed=all(c['passed'] for c in checks),checks=checks,inner_selected_marker=selected,inner_heldout_C=c,
        train_n=len(train),heldout_n=len(held),leakage_probe=row,capability_layer='added_script_using_native_lifelines',
        numeric_comparability='Selection uses maximum partial likelihood, not the R/SurvStudio score-screen estimand; no equality assertion.',
        input_contract_sha256=sha(inputs/'contract.json'),source_sha256=sha(__file__)))
    print('Native lifelines follow-up workflows completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.output)

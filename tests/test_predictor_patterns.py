from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import pytest
from survival_toolkit.errors import UserInputError
from survival_toolkit.predictor_patterns import compare_predictor_sets, describe_pattern, _paired_intervals, _fit
from survival_toolkit.sample_data import make_molecular_demo
from survival_toolkit.evaluation import stratified_holdout_indices


def test_added_outcome_copy_pauses_before_either_survival_model_fits(monkeypatch):
    from survival_toolkit.errors import OutcomeProxyError
    import survival_toolkit.predictor_patterns as patterns
    frame = make_molecular_demo(n_patients=240)
    frame['gene_A'] = 13 - 7 * frame.os_event
    fitted = []
    monkeypatch.setattr(patterns, '_fit', lambda *args, **kwargs: fitted.append(True))
    with pytest.raises(OutcomeProxyError) as caught:
        patterns.compare_predictor_sets(frame, **config())
    assert not fitted
    assert caught.value.audit['status'] == 'review_required'
    assert caught.value.audit['training_rows'] == 168


def test_predictor_comparison_retains_review_and_findings_in_saved_rows():
    pytest.importorskip('sksurv')
    frame = make_molecular_demo(n_patients=120)
    # A deterministic baseline predictor is indistinguishable from a copied event
    # column by association alone; a recorded provenance review must stay visible.
    frame['gene_A'] = (frame.gene_A > 0).astype(int)
    frame['os_event'] = frame.gene_A
    frame['os_months'] = np.where(frame.os_event == 1, 1., 10.)
    reason = 'Baseline gene assay measured at enrollment; source and timing checked.'
    result = compare_predictor_sets(frame, **config(), outcome_proxy_review=reason)
    audit = result['input_audit']['outcome_proxy']
    assert audit['status'] == 'reviewed_by_user'
    assert any(p['findings'] for p in audit['partitions'])
    assert audit['review_reason'] == reason
    assert all(row['outcome_proxy_review'] == reason for row in result['comparison']['rows'])
    assert all(row['outcome_proxy_status'] == 'reviewed_by_user' for row in result['comparison']['rows'])
    assert result['risk_display']['rows']


def test_pattern_api_keeps_review_separate_from_ordinary_predictor_checkbox():
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app, store
    frame = make_molecular_demo(n_patients=240)
    frame['gene_A'] = frame.os_event + .03 * frame.gene_A
    stored = store.create(frame, filename='test.csv', source='upload')
    response = TestClient(app, base_url='http://127.0.0.1').post(
        '/api/predictor-patterns', json={'dataset_id': stored.dataset_id, **config()})
    assert response.status_code == 400
    detail = response.json()['detail']
    assert detail['code'] == 'outcome_proxy_review_required'
    assert detail['outcome_proxy']['findings']


def config(**updates):
    return dict(time_column='os_months', event_column='os_event', event_positive_value=1,
                clinical_features=['age', 'stage'], added_features=['gene_A', 'gene_B'],
                categorical_features=['stage'], target_feature='gene_A', group_feature='stage',
                model_type='gbs', random_seed=17, n_estimators=10, max_depth=2, epochs=10,
                n_bootstrap=100, **updates)


def test_all_event_input_keeps_real_holdout_but_all_censored_does_not():
    train, test, mode = stratified_holdout_indices(np.ones(100), random_state=42)
    assert mode == 'holdout'
    assert len(train) == 70 and len(test) == 30
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(100))
    assert stratified_holdout_indices(np.zeros(100), random_state=42)[2] == 'apparent'
    assert stratified_holdout_indices(np.ones(10), random_state=42)[2] == 'apparent'


def test_all_event_baseline_inputs_are_not_rejected_for_absent_censoring():
    pytest.importorskip('sksurv')
    frame = make_molecular_demo(n_patients=120)
    frame['os_event'] = 1
    result = compare_predictor_sets(frame, **config())
    assert result['evaluation_mode'] == 'holdout'
    assert result['test_patients'] == result['test_events'] == 36
    assert result['input_audit']['outcome_proxy']['status'] == 'partially_assessed'


@pytest.mark.parametrize('model', ['gbs', 'rsf', 'deepsurv'])
def test_real_models_share_test_rows_and_keep_private_fit_out_of_response(model):
    pytest.importorskip('sksurv' if model != 'deepsurv' else 'torch')
    frame = make_molecular_demo(n_patients=100)
    kwargs = config(); kwargs['model_type'] = model
    result = compare_predictor_sets(frame, **kwargs)
    train, test, mode = stratified_holdout_indices(frame.os_event, random_state=17)
    assert mode == result['evaluation_mode'] == 'holdout'
    assert result['_test_positions'] == test.tolist()
    assert result['test_predictions']['row_ids'] == [str(i) for i in test]
    assert len(result['test_predictions']['risks']['Clinical only']) == len(test)
    assert result['comparison']['n_bootstrap_requested'] == 100
    assert result['comparison']['n_bootstrap_valid'] + result['comparison']['n_bootstrap_undefined'] == 100
    assert result['pattern'] is not None
    assert result['pattern']['model_refitted_for_curve'] is False
    assert set(result['_training_positions']).isdisjoint(test)
    public = {k:v for k,v in result.items() if not k.startswith('_')}
    import json
    json.dumps(public, allow_nan=False)


def test_training_transform_and_curve_do_not_see_changed_test_predictors():
    pytest.importorskip('sksurv')
    from survival_toolkit.analysis import _cohort_frame
    frame = _cohort_frame(make_molecular_demo(n_patients=120), "os_months", "os_event", event_positive_value=1, extra_columns=["age", "stage", "gene_A", "gene_B"], drop_missing_extra_columns=False).reset_index(drop=True)
    train, test, _ = stratified_holdout_indices(frame.os_event, random_state=17)
    args = dict(time_column='os_months', event_column='os_event', features=['age','stage','gene_A','gene_B'],
                categorical_features=['stage'], model_type='gbs', train_idx=train, test_idx=test,
                seed=17, n_estimators=10, max_depth=2, epochs=10)
    fitted = _fit(frame, **args)
    changed = frame.copy(); changed.loc[test, 'gene_A'] = 10000; changed.loc[test,'stage'] = 'UNSEEN'
    other = _fit(changed, **args)
    assert fitted['transform_hash'] == other['transform_hash']
    p1 = describe_pattern(fitted, target_feature='gene_A', group_feature='stage', seed=17)
    p2 = describe_pattern(other, target_feature='gene_A', group_feature='stage', seed=17)
    assert p1 == p2
    pd.testing.assert_frame_equal(frame.iloc[train], other['training'])


def test_common_background_reveals_nonparallel_predictions_and_masks_missing_support():
    frame = pd.DataFrame({'stage': [1]*30+[2]*30, 'gene': np.r_[np.linspace(-1,1,30),np.linspace(0,2,30)], 'other': np.arange(60)})
    calls = []
    def predict(raw):
        calls.append(raw.copy())
        return pd.to_numeric(raw.stage).to_numpy()*raw.gene.to_numpy() + raw.other.to_numpy()/100
    fitted = {'training':frame,'encoder':{'categorical_features':['stage']},'predict':predict}
    p = describe_pattern(fitted, target_feature='gene', group_feature='stage', seed=2, n_points=9)
    assert {r['group'] for r in p['rows']} == {'1','2'}
    assert any(r['risk_score'] is None and not r['supported'] for r in p['rows'])
    supported = [r for r in p['rows'] if r['supported']]
    assert all(np.isfinite(r['risk_score']) for r in supported)
    for raw in calls[1:]: pd.testing.assert_series_equal(raw.other, calls[0].other)
    assert p['model_refitted_for_curve'] is False
    assert fitted['training'].equals(frame)


def test_categorical_curve_retains_absent_combinations_as_null():
    frame = pd.DataFrame({'stage':['I']*20+['II']*20,'gene':['low']*20+['high']*20})
    fitted={'training':frame,'encoder':{'categorical_features':['stage','gene']},
            'predict':lambda raw: (raw.gene=='high').to_numpy(dtype=float)+(raw.stage=='II').to_numpy(dtype=float)}
    pattern=describe_pattern(fitted,target_feature='gene',group_feature='stage',seed=1)
    assert len(pattern['rows'])==4
    assert sum(r['supported'] for r in pattern['rows'])==2
    assert all(r['risk_score'] is None for r in pattern['rows'] if not r['supported'])
    from survival_toolkit.plots import build_predictor_pattern_figure
    fig=build_predictor_pattern_figure(pattern)
    assert sum(y is None for t in fig['data'] for y in t['y'])==2


@pytest.mark.parametrize('problem', ['negative','duplicate','small','overlap'])
def test_invalid_inputs_never_fit_or_fall_back(monkeypatch,problem):
    frame=make_molecular_demo(n_patients=100); kwargs=config()
    if problem=='negative': frame.loc[0,'os_months']=-1
    if problem=='duplicate': frame.loc[1,'patient_id']=frame.loc[0,'patient_id']
    if problem=='small': frame=frame.iloc[:8]
    if problem=='overlap': kwargs['added_features']=['age']
    monkeypatch.setattr('survival_toolkit.predictor_patterns._fit',lambda *a,**k: pytest.fail('invalid input reached fitting'))
    with pytest.raises(UserInputError): compare_predictor_sets(frame,**kwargs)


def test_missing_outcomes_and_nondefault_event_code_preserve_source_alignment():
    pytest.importorskip('sksurv')
    frame=make_molecular_demo(n_patients=100)
    frame.loc[3,'os_months']=np.nan
    frame['os_event']=frame.os_event.replace({1:2})
    kwargs=config(); kwargs['event_positive_value']=2
    r=compare_predictor_sets(frame,**kwargs)
    assert r['excluded_missing_outcome_rows']==1
    assert '3' not in r['test_predictions']['row_ids']
    assert set(r['test_predictions']['event']) <= {0,1}
    assert {x['source_row'] for x in r['risk_display']['rows']} == set(r['test_predictions']['row_ids'])


def test_constant_target_has_explicit_curve_error_and_keeps_completed_comparison():
    pytest.importorskip('sksurv')
    frame=make_molecular_demo(n_patients=100); frame['gene_A']=1
    r=compare_predictor_sets(frame,**config())
    assert r['pattern'] is None and r['pattern_error']
    assert np.isfinite(r['comparison']['delta_c'])


def test_all_bootstrap_indices_are_scheduled_without_replacing_undefined_draws():
    r=_paired_intervals(np.array([1.,2.]),np.array([1,0]),np.array([[2.,3.],[1.,0.]]),n_bootstrap=100,seed=12)
    rows=np.random.default_rng(12).integers(0,2,size=(100,2))
    expected=int(np.sum(rows[:,0]!=rows[:,1]))
    assert r['n_bootstrap_valid']==expected
    assert len(r['_bootstrap'])==100
    assert np.isnan(r['_bootstrap']).any()


def test_api_rejects_outcome_features_and_boolean_settings():
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app
    client=TestClient(app,base_url='http://127.0.0.1')
    loaded=client.post('/api/load-molecular-demo')
    assert loaded.status_code==200,loaded.text
    body=config(); body['dataset_id']=loaded.json()['dataset_id']
    body['random_seed']=True
    assert client.post('/api/predictor-patterns',json=body).status_code==422
    body['random_seed']=17; body['added_features']=['os_event']; body['target_feature']='os_event'
    refused=client.post('/api/predictor-patterns',json=body)
    assert refused.status_code==400 and 'outcome' in refused.text.lower(),refused.text


def test_api_completed_comparison_keeps_curves_settings_and_hash_without_private_objects():
    pytest.importorskip('sksurv')
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app
    client=TestClient(app,base_url='http://127.0.0.1')
    dataset=client.post('/api/load-molecular-demo').json()
    body=config();body['dataset_id']=dataset['dataset_id'];body['time_unit_label']='Years'
    response=client.post('/api/predictor-patterns',json=body)
    assert response.status_code==200,response.text
    result=response.json()
    assert result['request_config']['clinical_features']==['age','stage']
    assert result['analysis']['method_version']=='predictor-patterns/1'
    assert result['analysis']['test_patients']==144
    assert result['comparison_figure']['data'] and result['pattern_figure']['data']
    assert result['risk_group_figure']['data']
    assert 'Years' in str(result['risk_group_figure']['layout'])
    assert len(result['analysis']['risk_display']['rows']) == result['analysis']['test_patients']
    assert all(not k.startswith('_') for k in result['analysis'])
    assert len(result['analysis']['transform_hashes'])==2

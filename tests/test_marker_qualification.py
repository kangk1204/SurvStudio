import copy
import pytest
from test_guarded_marker_inference import cohort,evaluate
from survival_toolkit.marker_evaluation import validate_locked_recipe
import survival_toolkit.marker_qualification as policy


@pytest.mark.parametrize('method,status', [('marker-inference/1','failed_exploratory_only'), ('marker-inference/2','not_evaluated')])
def test_failed_or_unevaluated_profile_masks_inference_without_altering_fixed_prediction(method,status):
    raw=evaluate(cohort())
    raw['method_version']=method
    raw['inference']['method_version']=method
    prediction=copy.deepcopy(raw['locked_recipe']['model'])
    result=policy.qualify_marker_result(raw)
    assert result['inference']['engineering_qualification']['status']==status
    assert result['inference']['status']=='withheld' and not result['inference']['allowed']
    assert result['locked_recipe']['model']==prediction
    assert result['signature']['optimism_corrected_c'] is None
    assert 'exploratory_signature' in result
    for row in result['marker_table']:
        assert row['tier']=='inference withheld' and row['added_value']['p_fwer'] is None
        assert row['exploratory']['added_value']['p_value'] is not None
    outside=validate_locked_recipe(cohort(5),result['locked_recipe'],n_bootstrap=0)
    assert outside['inference']['status']=='withheld'
    assert all(row['replication_p_holm'] is None for row in outside['markers'])


def test_missing_or_version_mismatched_evidence_cannot_enable_inference(monkeypatch,tmp_path):
    assert policy.qualification('linear','unknown-method')['status']=='not_evaluated'
    monkeypatch.setattr(policy,'REGISTRY',tmp_path/'missing.json')
    assert policy.qualification('linear','marker-inference/1')['status']=='not_evaluated'
    assert policy.qualify_marker_result(evaluate(cohort()))['inference']['status']=='withheld'


def test_passed_profile_cannot_override_dataset_diagnostic_failure(monkeypatch):
    monkeypatch.setattr(policy,'qualification',lambda *a:{'status':'passed_supported_conditions_only'})
    raw=evaluate(cohort(n=500,nonlinear=True))
    result=policy.qualify_marker_result(raw)
    assert result['inference']['status']=='withheld' and not result['inference']['allowed']

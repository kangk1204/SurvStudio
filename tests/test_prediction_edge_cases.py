"""Input, review scope and descriptive-output edge cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.errors import UserInputError
from survival_toolkit.event_guard import event_coding_warning
from survival_toolkit.evaluation import c_index_intervals
from survival_toolkit.prediction_guard import guarded_prediction_inputs, check_training_outcome_proxies
from survival_toolkit.predictor_patterns import compare_predictor_sets, describe_pattern
from survival_toolkit.sample_data import make_molecular_demo


@pytest.mark.parametrize("column", ["OSCENSOR", "RFSCENSOR", "PFSCENS", "DFSCENSORED", "DSSCENSORING", "CENSOROS", "CENSORSTATUS", "OSCENSORFLAG"])
def test_compact_censor_names_require_source_recoding(column):
    frame = pd.DataFrame({column: [0, 1] * 12})
    assert "recoded column" in event_coding_warning(frame, column)
    frame[column] = ["Dead", "Alive"] * 12
    assert event_coding_warning(frame, column) is None


@pytest.mark.parametrize("column", ["censorship", "mycensorprotein", "CENSORIN", "OS_censor_event", "death_from_censor", "vital_status"])
def test_censor_substrings_and_explicit_event_names_are_not_rejected(column):
    assert event_coding_warning(pd.DataFrame({column: [0, 1]}), column) is None


def test_selected_lasso_curvature_failure_is_already_wrapped(monkeypatch):
    pytest.importorskip("sksurv")
    from survival_toolkit import ml_models as ml
    rng = np.random.default_rng(40)
    x = rng.normal(size=(80, 3))
    y = np.array([(bool(i % 2), float(i + 1)) for i in range(80)], dtype=[("event", bool), ("time", float)])
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular selected information")
    monkeypatch.setattr(ml, "polish_lasso_coefficients", singular)
    with pytest.raises(ml._LassoNumericalError, match="stable numerical coefficients") as caught:
        ml._fit_selected_lasso(x, y, {"alpha": .1, "alpha_path": [.2, .1]})
    assert isinstance(caught.value.__cause__, np.linalg.LinAlgError)


def test_group_type_guidance_names_the_selected_feature():
    fitted = {"training": pd.DataFrame({"treatment": [1, 2]}), "encoder": {"categorical_features": []}}
    with pytest.raises(UserInputError, match="mark 'treatment' as categorical") as caught:
        describe_pattern(fitted, target_feature="gene", group_feature="treatment", seed=1)
    assert "stage" not in str(caught.value)


def test_comparison_audits_original_ids_once_and_preserves_scores(monkeypatch):
    pytest.importorskip("sksurv")
    import survival_toolkit.prediction_guard as guard
    from test_predictor_patterns import config
    frame = make_molecular_demo(n_patients=100)
    frame["patient_id"] = [f"P{i}" for i in range(len(frame))]
    before = frame.copy(deep=True)
    original = guard.prediction_input_audit
    calls = []
    def record(df, *args, **kwargs):
        calls.append(df)
        return original(df, *args, **kwargs)
    monkeypatch.setattr(guard, "prediction_input_audit", record)
    result = compare_predictor_sets(frame, **config())
    assert len(calls) == 1 and calls[0] is frame
    assert result["input_audit"]["identifier_checks"][0]["column"] == "patient_id"
    assert result["input_audit"]["identical_input_rows"]["reason"] == "recorded_identifier_available"
    assert len(result["input_audit"]["outcome_proxy"]["partitions"]) == 2
    pd.testing.assert_frame_equal(frame, before)


def test_child_review_is_honored_without_clearing_parent_or_sibling():
    from survival_toolkit.errors import OutcomeProxyError
    from survival_toolkit.prediction_guard import _PROXY_CONTEXT
    frame = make_molecular_demo(n_patients=100)
    frame["gene_A"] = frame.os_event
    reason = "Baseline assay at enrollment; timing and source independently checked."
    @guarded_prediction_inputs
    def child(df, time_column, event_column, features, event_positive_value=1, outcome_proxy_review=None):
        check_training_outcome_proxies(df, time_column, event_column, features)
        return {"comparison_table": [{"model": "test"}]}
    @guarded_prediction_inputs
    def parent(df, time_column, event_column, features, event_positive_value=1):
        reviewed = child(df, time_column, event_column, features, outcome_proxy_review=reason)
        assert _PROXY_CONTEXT.get()["review"] is None
        with pytest.raises(OutcomeProxyError):
            child(df, time_column, event_column, features)
        return {"child": reviewed}
    result = parent(frame, "os_months", "os_event", ["gene_A"])
    assert result["child"]["input_audit"]["outcome_proxy"]["review_reason"] == reason
    assert result["child"]["comparison_table"][0]["outcome_proxy_review"] == reason
    assert result["input_audit"]["outcome_proxy"]["review_reason"] is None
    assert _PROXY_CONTEXT.get() is None


def test_preflight_reuse_is_bound_to_exact_frame_and_audited_features():
    from survival_toolkit.prediction_guard import checked_prediction_cohort
    frame = make_molecular_demo(n_patients=100)
    @guarded_prediction_inputs
    def child(df, time_column, event_column, features, event_positive_value=1):
        return {}
    @guarded_prediction_inputs
    def parent(df, time_column, event_column, features, event_positive_value=1):
        prepared = df[[time_column, event_column, *features]].copy()
        with checked_prediction_cohort(prepared, time_column, event_column, features):
            assert child(prepared, time_column, event_column, features)["input_audit"]["input_rows"] == len(df)
            other = prepared.copy()
            other.loc[0, time_column] = -1
            with pytest.raises(ValueError, match="negative follow-up"):
                child(other, time_column, event_column, features)
            with pytest.raises(ValueError, match="outcome columns cannot"):
                child(prepared, time_column, event_column, [event_column])
        with pytest.raises(ValueError, match="passed the raw input audit"):
            with checked_prediction_cohort(prepared, time_column, event_column, ["new_predictor"]):
                pass
        return {}
    parent(frame, "os_months", "os_event", ["gene_A"])


def test_nearly_constant_training_risk_keeps_scores_without_arbitrary_groups():
    from survival_toolkit.risk_display import describe_heldout_risk
    result = describe_heldout_risk(training_risk=[1., 1. + 1e-14], test_risk=[1. - 1e-14, 1. + 1e-14],
                                  time=[1., 2.], event=[1, 0], source_rows=[1, 2])
    assert result["km"] is None and result["cutoff"] is None
    assert "constant" in result["curve_error"]
    assert {row["predicted_risk_group"] for row in result["rows"]} == {"Risk group not estimable"}
    assert len(result["rows"]) == 2


def test_bootstrap_metadata_retains_undefined_draws_without_replacement():
    result = c_index_intervals([1., 2.], [1, 0], {"Cox PH": [2., 1.]}, n_bootstrap=100, random_seed=12)
    scheduled = np.random.default_rng(12).integers(0, 2, size=(100, 2))
    valid = int(np.sum(scheduled[:, 0] != scheduled[:, 1]))
    assert result["n_bootstrap_requested"] == 100
    assert result["n_bootstrap_valid"] == result["n_bootstrap"] == valid
    assert result["n_bootstrap_undefined"] == 100 - valid
    assert result["rows"][0]["c_index_ci"] == [1., 1.]


def test_no_comparable_bootstrap_pairs_have_null_intervals_and_zero_valid_draws():
    result = c_index_intervals([1., 2.], [0, 1], {"Cox PH": [2., 1.]}, n_bootstrap=100, random_seed=12)
    assert result["n_bootstrap_valid"] == result["n_bootstrap"] == 0
    assert result["n_bootstrap_undefined"] == 100
    assert result["rows"][0]["c_index"] is None
    assert result["rows"][0]["c_index_ci"] == [None, None]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.analysis import detect_duplicate_identifier_columns
from survival_toolkit.ml_models import _drop_constant_train_columns, _drop_rank_deficient_train_columns
from survival_toolkit.prediction_guard import guarded_prediction_inputs, prediction_input_audit


def cohort():
    rng = np.random.default_rng(2026101016)
    return pd.DataFrame({"patient_id": [f"P{i}" for i in range(120)],
                         "time": rng.exponential(20, 120), "event": np.arange(120) % 2,
                         "marker": rng.normal(size=120)})


@pytest.mark.parametrize("column", ["time", "event"])
def test_selected_outcome_is_rejected(column):
    with pytest.raises(ValueError, match="outcome columns cannot"):
        prediction_input_audit(cohort(), "time", "event", [column], 1)


def test_negative_time_is_rejected_and_zero_is_retained():
    df = cohort()
    df.loc[0, "time"] = 0
    assert prediction_input_audit(df, "time", "event", ["marker"], 1)["analyzed_rows"] == 120
    df.loc[1, "time"] = -3
    with pytest.raises(ValueError, match="1 negative follow-up time"):
        prediction_input_audit(df, "time", "event", ["marker"], 1)


def test_repeated_ids_block_even_when_many_rows_per_patient():
    df = cohort()
    df["patient_id"] = [f"P{i // 4}" for i in range(120)]
    with pytest.raises(ValueError, match="one row per patient"):
        prediction_input_audit(df, "time", "event", ["marker"], 1)
    # A binary attribute called patient_group is not treated as an identifier.
    assert not detect_duplicate_identifier_columns(pd.DataFrame({"patient_group": [0, 1] * 60}))


def test_missing_outcomes_count_once_and_input_is_preserved():
    df = cohort()
    df.loc[0, ["time", "event"]] = np.nan
    df.loc[1, "event"] = np.nan
    df.loc[2, "time"] = np.inf
    before = df.copy(deep=True)
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    assert audit["input_rows"] == 120 and audit["analyzed_rows"] == 117
    assert audit["excluded_outcome_rows"] == 3
    assert "not imputed" in audit["notes"][0]
    pd.testing.assert_frame_equal(df, before)


def test_guard_runs_before_fitting_and_notes_reach_export_tables():
    fits = []

    @guarded_prediction_inputs
    def model(df, time_column, event_column, features, event_positive_value=1):
        fits.append(True)
        return {"comparison_table": [{"model": "Cox PH", "c_index": None,
                "removed_features": [{"column": "copy", "reason": "redundant in Cox training design"}]}],
                "scientific_summary": {"cautions": []}, "manuscript_tables": {"table_notes": []}}

    df = cohort()
    result = model(df, "time", "event", ["marker"])
    assert result["comparison_table"][0]["c_index"] is None
    assert "copy" in result["comparison_table"][0]["removed_feature_notes"]
    assert any("copy" in note for note in result["manuscript_tables"]["table_notes"])
    assert "hidden outcome proxy" in result["comparison_table"][0]["input_notes"]
    df.loc[0, "patient_id"] = df.loc[1, "patient_id"]
    with pytest.raises(ValueError, match="Repeated patient IDs"):
        model(df, "time", "event", ["marker"])
    assert fits == [True]


def test_training_only_removal_records_do_not_depend_on_test_values():
    train = pd.DataFrame({"marker": [-1., 0., 1.], "copy": [-1., 0., 1.], "constant": [1., 1., 1.]})
    test = pd.DataFrame({"marker": [4., 5.], "copy": [8., 9.], "constant": [2., 3.]})
    reduced, evaluation = _drop_constant_train_columns(train, test)
    reduced, evaluation = _drop_rank_deficient_train_columns(reduced, evaluation)
    reasons = reduced.attrs["removed_features"]
    assert len(reasons) == 2
    assert {row["reason"] for row in reasons} == {"constant in training rows", "redundant in Cox training design"}
    changed = test * -100
    repeated, _ = _drop_constant_train_columns(train, changed)
    repeated, _ = _drop_rank_deficient_train_columns(repeated, changed[repeated.columns])
    assert list(repeated.columns) == list(reduced.columns)
    assert repeated.attrs == reduced.attrs


def test_hidden_proxy_is_not_certified_or_silently_selected():
    df = cohort()
    df["marker"] = df.event + 0.05 * df.marker
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    assert audit["predictor_availability"] == "user_review_required"
    assert audit["analyzed_rows"] == 120


@pytest.mark.parametrize("route", ["ml", "cv", "deep"])
def test_public_model_routes_stop_before_training_on_duplicate_ids(route):
    from survival_toolkit.ml_models import compare_survival_models, cross_validate_survival_models
    from survival_toolkit.deep_models import compare_deep_survival_models
    df = cohort()
    df.loc[0, "patient_id"] = df.loc[1, "patient_id"]
    function = {"ml": compare_survival_models, "cv": cross_validate_survival_models,
                "deep": compare_deep_survival_models}[route]
    with pytest.raises(ValueError, match="Repeated patient IDs"):
        function(df, "time", "event", ["marker"])


def test_api_rejects_duplicates_and_records_availability_review():
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app, store
    client = TestClient(app, base_url="http://127.0.0.1")
    df = cohort()
    stored = store.create(df, filename="cohort.csv", source="upload")
    request = {"dataset_id": stored.dataset_id, "time_column": "time", "event_column": "event",
               "features": ["marker"], "model_type": "compare", "n_estimators": 10}
    assert client.post("/api/ml-model", json={**request, "predictor_availability_confirmed": False}).status_code == 422
    response = client.post("/api/ml-model", json={**request, "predictor_availability_confirmed": True})
    assert response.status_code == 200, response.text
    result = response.json()["analysis"]
    assert result["input_audit"]["predictor_availability"] == "user_confirmed"
    assert any("user confirmed" in text for text in result["manuscript_tables"]["table_notes"])
    df.loc[0, "patient_id"] = df.loc[1, "patient_id"]
    duplicate = store.create(df, filename="duplicate.csv", source="upload")
    response = client.post("/api/ml-model", json={**request, "dataset_id": duplicate.dataset_id})
    assert response.status_code == 400 and "Repeated patient IDs" in response.text

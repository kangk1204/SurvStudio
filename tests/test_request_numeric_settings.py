"""JSON booleans must not silently change numeric analysis settings."""
from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
import survival_toolkit.app as app_module

COMMON = {"dataset_id": "numeric-settings", "time_column": "time", "event_column": "event"}
ML = {**COMMON, "features": ["x"], "model_type": "rsf"}
DL = {**COMMON, "features": ["x"], "model_type": "deepsurv"}


@pytest.mark.parametrize("model,payload,field,value", [
    (app_module.KaplanMeierRequest, COMMON, "max_time", True),
    (app_module.KaplanMeierRequest, COMMON, "fh_p", True),
    (app_module.KaplanMeierRequest, COMMON, "fh_p", False),
    (app_module.MLModelRequest, ML, "random_state", True),
    (app_module.MLModelRequest, ML, "random_state", False),
    (app_module.MLModelRequest, ML, "cv_repeats", True),
    (app_module.MLModelRequest, ML, "max_depth", True),
    (app_module.MLModelRequest, ML, "learning_rate", True),
    (app_module.DeepModelRequest, DL, "random_seed", True),
    (app_module.DeepModelRequest, DL, "dropout", False),
    (app_module.DeepModelRequest, DL, "hidden_layers", [True]),
    (app_module.DeepModelRequest, DL, "parallel_jobs", True),
    (app_module.DeepModelRequest, DL, "n_layers", True),
    (app_module.DeepModelRequest, DL, "early_stopping_patience", True),
    (app_module.DeepModelRequest, DL, "early_stopping_min_delta", False),
    (app_module.MarkerEvaluationRequest, {**COMMON, "marker_columns": ["x"]}, "n_permutations", False),
    (app_module.MarkerEvaluationRequest, {**COMMON, "marker_columns": ["x"]}, "n_resamples", False),
    (app_module.MarkerEvaluationRequest, {**COMMON, "marker_columns": ["x"]}, "max_missing_fraction", False),
    (app_module.SignatureSearchRequest, {**COMMON, "candidate_columns": ["x"]}, "bootstrap_iterations", False),
    (app_module.DeriveGroupRequest, {"dataset_id": "numeric-settings", "source_column": "x", "method": "median_split"}, "random_seed", False),
    (app_module.ModelComparisonIntervalsRequest, {"predictions": [{"row_ids": ["a", "b"], "time": [1, 2],
        "event": [1, 0], "risk": {"Cox PH": [.2, .1]}}]}, "random_seed", True),
    (app_module.DesignAuditCohort, {"n": 100}, "events", True),
])
def test_numeric_settings_refuse_json_booleans(model, payload, field, value):
    with pytest.raises(ValidationError):
        model.model_validate({**payload, field: value})


@pytest.mark.parametrize("route,payload,field,value", [
    ("/api/kaplan-meier", COMMON, "max_time", True),
    ("/api/ml-model", ML, "random_state", False),
    ("/api/deep-model", DL, "hidden_layers", [True]),
    ("/api/marker-evaluation", {**COMMON, "marker_columns": ["x"]}, "n_permutations", False),
])
def test_invalid_numeric_settings_stop_before_dataset_or_training(monkeypatch, route, payload, field, value):
    calls = []
    def must_not_access_dataset(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Request validation must run before dataset access")
    monkeypatch.setattr(app_module, "_get_stored_dataset", must_not_access_dataset)
    client = TestClient(app_module.app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    response = client.post(route, json={**payload, field: value})
    assert response.status_code == 422, response.text
    assert calls == []


def test_numeric_zero_optional_none_and_actual_boolean_flags_keep_their_meaning():
    ml = app_module.MLModelRequest.model_validate({**ML, "random_state": 0, "max_depth": None,
        "compute_shap": False, "shap_safe_mode": True, "predictor_availability_confirmed": True,
        "event_positive_value": True, "n_estimators": "10", "learning_rate": "0.1"})
    assert ml.random_state == 0 and ml.max_depth is None
    assert ml.compute_shap is False and ml.shap_safe_mode is True
    assert ml.predictor_availability_confirmed is True and ml.event_positive_value is True
    assert ml.n_estimators == 10 and ml.learning_rate == .1
    dl = app_module.DeepModelRequest.model_validate({**DL, "dropout": 0, "early_stopping_patience": None,
                                                   "hidden_layers": [1, 8]})
    assert dl.dropout == 0 and dl.early_stopping_patience is None and dl.hidden_layers == [1, 8]
    marker = app_module.MarkerEvaluationRequest.model_validate({**COMMON, "marker_columns": ["x"],
        "n_permutations": 0, "n_resamples": 0, "max_missing_fraction": 0})
    assert marker.n_permutations == 0 and marker.n_resamples == 0 and marker.max_missing_fraction == 0

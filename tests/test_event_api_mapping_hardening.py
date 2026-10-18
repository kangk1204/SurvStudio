"""Product APIs must not guess which of two nonstandard numeric codes is the event."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from survival_toolkit.app import app, store


_ROUTES = [
    ("/api/kaplan-meier", {}),
    ("/api/cox", {"covariates": ["age"]}),
    ("/api/cox-preview", {"covariates": ["age"]}),
    ("/api/cohort-table", {"variables": ["age"]}),
    ("/api/derive-group", {"source_column": "age", "method": "optimal_cutpoint", "permutation_iterations": 0}),
    ("/api/optimal-cutpoint", {"variable": "age", "permutation_iterations": 0}),
    ("/api/discover-signature", {"candidate_columns": ["age"], "bootstrap_iterations": 0}),
    ("/api/ml-model", {"features": ["age"], "model_type": "rsf", "predictor_availability_confirmed": True}),
    ("/api/deep-model", {"features": ["age"], "model_type": "deepsurv", "predictor_availability_confirmed": True}),
    ("/api/marker-evaluation", {"marker_columns": ["gene"], "n_permutations": 0, "n_resamples": 0}),
    ("/api/time-dependent-importance", {"features": ["age"]}),
    ("/api/counterfactual", {"features": ["age"], "target_feature": "age", "counterfactual_value": 65}),
    ("/api/pdp", {"features": ["age"], "target_feature": "age"}),
    ("/api/predictor-patterns", {"clinical_features": ["stage"], "added_features": ["gene"],
                                "categorical_features": ["stage"], "target_feature": "gene", "group_feature": "stage"}),
]


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(4721)
    return pd.DataFrame({
        "time": rng.uniform(1, 100, 150),
        "status": [1] * 39 + [2] * 111,
        "age": rng.normal(55, 8, 150),
        "gene": rng.normal(size=150),
        "stage": np.tile(["I", "II", "III"], 50),
    })


@pytest.mark.parametrize("route,settings", _ROUTES)
def test_omitted_event_mapping_rejects_ambiguous_one_two_codes(route, settings):
    frame = _cohort()
    original = frame.copy(deep=True)
    stored = store.create(frame, filename="one_two.csv", source="upload")
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post(route, json={"dataset_id": stored.dataset_id,
                "time_column": "time", "event_column": "status", **settings})
        assert response.status_code == 400, response.text
        assert "event coding" in response.json()["detail"].lower(), response.text
        pd.testing.assert_frame_equal(store.get(stored.dataset_id).dataframe, original)
    finally:
        store.delete(stored.dataset_id)


@pytest.mark.parametrize("positive,expected_events", [(1, 39), (2, 111)])
def test_explicit_one_or_two_mapping_keeps_the_selected_event(positive, expected_events):
    frame = _cohort()
    stored = store.create(frame, filename="one_two.csv", source="upload")
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post("/api/kaplan-meier", json={"dataset_id": stored.dataset_id,
                "time_column": "time", "event_column": "status", "event_positive_value": positive})
        assert response.status_code == 200, response.text
        analysis = response.json()["analysis"]
        assert analysis["cohort"]["n"] == 150
        assert analysis["cohort"]["events"] == expected_events
        assert response.json()["request_config"]["event_positive_value"] == positive
    finally:
        store.delete(stored.dataset_id)


@pytest.mark.parametrize("labels", [(0, 1), ("Alive", "Dead"), ("No recurrence", "Recurrence")])
def test_omitted_mapping_preserves_recognized_labels_and_missing_events(labels):
    frame = _cohort()
    frame["status"] = pd.Series([labels[1]] * 39 + [labels[0]] * 110 + [None], dtype=object)
    stored = store.create(frame, filename="recognized.csv", source="upload")
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post("/api/kaplan-meier", json={"dataset_id": stored.dataset_id,
                "time_column": "time", "event_column": "status"})
        assert response.status_code == 200, response.text
        cohort = response.json()["analysis"]["cohort"]
        assert cohort["n"] == 149
        assert cohort["events"] == 39
        assert response.json()["request_config"]["event_positive_value"] is None
    finally:
        store.delete(stored.dataset_id)


def test_all_api_event_fields_require_inference_or_explicit_selection():
    # Cover every schema, including auxiliary/external routes, without running expensive fits.
    schemas = app.openapi()["components"]["schemas"]
    event_schemas = [schema for schema in schemas.values() if "event_positive_value" in schema.get("properties", {})]
    assert len(event_schemas) >= 14
    for schema in event_schemas:
        assert schema["properties"]["event_positive_value"].get("default") is None, schema["title"]


@pytest.mark.parametrize("positive", [None, 1, 2])
def test_external_recipe_does_not_determine_ambiguous_external_event_codes(positive):
    module = importlib.import_module("survival_toolkit.app")
    frame = _cohort()
    stored = store.create(frame, filename="external.csv", source="upload")
    # Only the role/outcome metadata used by the public preflight is needed here; numeric
    # recipe integrity and prediction are covered by locked-recipe tests separately.
    recipe = {"outcome": {"time_column": "time", "event_column": "status", "event_positive_value": 1},
              "clinical": {"columns": []}, "markers": ["gene"], "strata_columns": []}
    try:
        if positive is None:
            with pytest.raises(ValueError, match="external event column.*explicit event-positive"):
                module._reject_leaky_validation_columns(stored, recipe, {}, positive)
        else:
            module._reject_leaky_validation_columns(stored, recipe, {}, positive)
    finally:
        store.delete(stored.dataset_id)

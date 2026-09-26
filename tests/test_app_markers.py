from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from survival_toolkit.app import app

# The local request guard only answers loopback Host headers.
client = TestClient(app, base_url="http://127.0.0.1")

_FAST = {"n_permutations": 49, "n_resamples": 8, "random_seed": 7}


def _marker_request(dataset_id: str, **changes) -> dict:
    return {
        "dataset_id": dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "marker_columns": ["biomarker_score", "immune_index"],
        "clinical_columns": ["age", "stage"],
        "categorical_clinical": ["stage"],
        **_FAST,
        **changes,
    }


def test_marker_evaluation_returns_tiers_figures_and_a_locked_recipe():
    dataset = client.post("/api/load-example").json()
    response = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"]))

    assert response.status_code == 200, response.text
    payload = response.json()
    analysis = payload["analysis"]
    assert analysis["primary_lens"] == "added_value"
    assert {row["Marker"] for row in payload["display_table"]} == {"biomarker_score", "immune_index"}
    assert {"Tier", "Family-wise P", "Selection frequency", "LR test P"} <= set(payload["display_table"][0])
    assert payload["stability_figure"]["data"] and "layout" in payload["rank_figure"]
    assert analysis["locked_recipe"]["recipe_hash"]
    assert payload["request_config"]["n_permutations"] == 49
    assert payload["dataset_hash"]


def test_marker_evaluation_rejects_outcome_columns_as_markers():
    dataset = client.post("/api/load-example").json()
    response = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"], marker_columns=["os_months"]))

    assert response.status_code == 400
    assert "outcome" in response.json()["detail"].lower()


def test_marker_evaluation_requires_categorical_columns_to_be_clinical():
    dataset = client.post("/api/load-example").json()
    response = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"], categorical_clinical=["sex"]))

    assert response.status_code == 422


def test_marker_validation_applies_the_locked_recipe_to_another_dataset():
    development = client.post("/api/load-example").json()
    recipe = client.post("/api/marker-evaluation", json=_marker_request(development["dataset_id"])).json()["analysis"]["locked_recipe"]
    external = client.post("/api/load-example").json()

    response = client.post("/api/marker-validation", json={"dataset_id": external["dataset_id"], "recipe": recipe, "n_bootstrap": 30})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["recipe_hash"] == recipe["recipe_hash"]
    assert 0.5 < payload["validation"]["metrics"]["c_index"] < 1.0
    assert "recipe" not in payload["request_config"]
    assert "data" in payload["figure"]


def test_marker_validation_rejects_an_edited_recipe():
    development = client.post("/api/load-example").json()
    recipe = client.post("/api/marker-evaluation", json=_marker_request(development["dataset_id"])).json()["analysis"]["locked_recipe"]
    edited = copy.deepcopy(recipe)
    edited["model"]["coefficients"][0] += 0.1

    response = client.post("/api/marker-validation", json={"dataset_id": development["dataset_id"], "recipe": edited})

    assert response.status_code == 400
    assert "hash" in response.json()["detail"]


def test_design_audit_flags_a_single_cohort_best_of_101_design():
    response = client.post(
        "/api/design-audit",
        json={
            "candidate_models": 101,
            "gene_only": True,
            "selection_cohorts": [{"name": "GSE72094", "n": 398, "events": 122}],
            "training_in_selection": True,
            "headline": "average including training",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    codes = [flag["code"] for flag in payload["flags"]]
    assert {"NO_SEALED_COHORT", "TRAINING_IN_SELECTION", "SINGLE_SELECTION_COHORT", "MANY_CANDIDATES"} <= set(codes)
    assert payload["high_risk_flags"] >= 3
    assert payload["expected_optimism"]["value"] > 0.05
    assert payload["request_config"]["selection_cohorts"][0]["name"] == "GSE72094"


def test_design_audit_validates_its_input():
    base = {"candidate_models": 12, "gene_only": True, "selection_cohorts": [{"n": 150}]}
    assert client.post("/api/design-audit", json=base).status_code == 200
    for changes in ({"headline": "best"}, {"candidate_models": 1}, {"selection_cohorts": []}, {"winner": "RSF"}):
        assert client.post("/api/design-audit", json={**base, **changes}).status_code == 422
    sealed_headline = client.post("/api/design-audit", json={**base, "headline": "sealed"})
    assert sealed_headline.status_code == 400
    assert "sealed" in sealed_headline.json()["detail"]

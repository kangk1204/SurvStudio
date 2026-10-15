import copy

import pandas as pd
import pytest

from survival_toolkit.analysis import _cohort_frame
from survival_toolkit.marker_review import review_external_horizon


def fixture(horizon=5):
    frame = pd.DataFrame({"time": [1., 2., 4., 100.], "event": [1., 0., 0., float("nan")]})
    scored = _cohort_frame(frame, time_column="time", event_column="event", event_positive_value=1)
    recipe = {"recipe_version": 3, "outcome": {"time_column": "time", "event_column": "event", "event_positive_value": 1},
              "clinical": {"columns": [], "encoder": {"numeric_features": [], "categorical_features": [], "feature_names": []}},
              "model": {"terms": []}}
    validation = {"cohort": {"n": len(scored), "row_mask_hash": scored.attrs["row_mask_hash"]},
                  "inference": {"status": "assumption_dependent", "allowed": True},
                  "markers": [{"replication_p_holm": .02}],
                  "metrics": {"horizon": horizon, "observed_risk": .6, "expected_risk": .7,
                              "observed_expected_ratio": .6/.7, "brier": .05, "brier_skill": .4,
                              "c_index": .63, "calibration_slope": .91}}
    return frame, recipe, validation


def test_followup_guard_uses_the_scored_rows_and_preserves_raw_calculations():
    frame, recipe, validation = fixture()
    before = copy.deepcopy(validation)
    result = review_external_horizon(validation, frame, recipe)
    assert validation == before
    assert result["absolute_risk_review"]["maximum_followup"] == 4
    assert result["absolute_risk_review"]["n_at_or_beyond_horizon"] == 0
    assert result["exploratory_horizon_metrics"]["brier"] == .05
    assert all(result["metrics"][key] is None for key in result["exploratory_horizon_metrics"])
    assert result["metrics"]["c_index"] == .63 and result["metrics"]["calibration_slope"] == .91
    assert result["inference"] == before["inference"] and result["markers"] == before["markers"]


def test_supported_boundary_mapped_columns_and_outcome_coding_are_preserved():
    frame, recipe, validation = fixture(horizon=4)
    frame = frame.rename(columns={"time": "followup", "event": "status"})
    frame["status"] = frame.status.map({1.: "Dead", 0.: "Alive"})
    result = review_external_horizon(validation, frame, recipe, column_mapping={"time": "followup", "event": "status"},
                                     event_positive_value="Dead")
    assert result["absolute_risk_review"]["status"] == "within_observed_followup"
    assert result["metrics"] == validation["metrics"]
    assert "exploratory_horizon_metrics" not in result


def test_mismatched_cohort_fails_instead_of_reviewing_other_patients():
    frame, recipe, validation = fixture()
    validation["cohort"]["row_mask_hash"] = "another cohort"
    with pytest.raises(RuntimeError, match="scored validation cohort"):
        review_external_horizon(validation, frame, recipe)


def test_api_withholds_unobserved_horizon_without_changing_the_locked_recipe():
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app
    from test_marker_review import _gene_cohort, _gene_recipe

    recipe = _gene_recipe(0.0)
    before = copy.deepcopy(recipe)
    external = _gene_cohort(2)
    external["event"] = (external.event.eq(1) & external.time.le(.5)).astype(int)
    external["time"] = external.time.clip(upper=.5)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        uploaded = client.post("/api/upload", files={"file": ("cohort.csv", external.to_csv(index=False), "text/csv")}).json()
        response = client.post("/api/marker-validation", json={"dataset_id": uploaded["dataset_id"], "recipe": recipe,
            "event_positive_value": 1, "horizon": 5, "n_bootstrap": 0})
        assert response.status_code == 200, response.text
        result = response.json()["validation"]
    assert recipe == before
    assert result["absolute_risk_review"]["status"] == "beyond_validation_followup"
    assert result["metrics"]["brier"] is None and result["metrics"]["observed_expected_ratio"] is None
    assert result["metrics"]["c_index"] is not None
    assert "exploratory_horizon_metrics" in result

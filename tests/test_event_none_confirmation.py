"""A literal None label needs an explicit binary mapping, never silent NA or censoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from survival_toolkit.analysis import coerce_event, compute_cox_analysis, compute_km_analysis
from survival_toolkit.app import app, store


@pytest.mark.parametrize("positive", ["Relapse", "Recurrence", "Dead", "Event"])
def test_explicit_observed_event_maps_only_the_other_literal_none_label(positive):
    original = pd.Series([positive, "None", None, np.nan, pd.NA, "None", positive], name="status", dtype=object)
    before = original.copy(deep=True)
    coded = coerce_event(original, event_positive_value=positive)
    assert coded.iloc[[0, 1, 5, 6]].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert coded.iloc[2:5].isna().all()
    assert coded.attrs["event_coding"]["event_positive_value"] == positive
    assert coded.attrs["event_coding"]["censored_labels"] == ["none"]
    assert coded.attrs["event_coding"]["missing_values_remain_missing"]
    pd.testing.assert_series_equal(original, before)
    with pytest.raises(ValueError, match="literal 'None'.*not assumed"):
        coerce_event(original)


@pytest.mark.parametrize("labels,positive", [
    (["Relapse", "None"], "None"),
    (["Relapse", "None"], 1),  # A selected label must actually occur in the column.
    (["Relapse", "None"], "Recurrence"),
    (["Relapse", "None", "Alive"], "Relapse"),
    (["Relapse", "None", "Relaspe"], "Relapse"),
    (["Relapse", "None", "Dead"], "Relapse"),
    (["Relaspe", "None"], "Relaspe"),
    (["custom", "None"], "custom"),
    (["None", "None"], "Relapse"),
    (["None", "None"], "None"),
])
def test_none_is_not_a_general_censor_token_or_a_way_to_collapse_other_labels(labels, positive):
    with pytest.raises(ValueError, match="literal label 'None'|more than one recognized event"):
        coerce_event(pd.Series(labels, dtype=object), event_positive_value=positive)


def test_real_missing_values_do_not_invent_censored_patients():
    coded = coerce_event(pd.Series(["Relapse", None, "Relapse"], dtype=object), "Relapse")
    assert coded.tolist()[::2] == [1.0, 1.0]
    assert pd.isna(coded.iloc[1])
    assert "event_coding" not in coded.attrs


def test_numeric_event_mapping_is_unchanged_and_multivalued_input_has_clear_guidance():
    assert coerce_event(pd.Series([0, 1, 0])).tolist() == [0.0, 1.0, 0.0]
    assert coerce_event(pd.Series([1, 2, 1]), 2).tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="more than two numeric values.*event/status column"):
        coerce_event(pd.Series([42.5, 61.0, 73.0], name="age"))


def _source_frame():
    rng = np.random.default_rng(1319)
    return pd.DataFrame({
        "time": rng.uniform(1, 80, 122),
        "status": ["Relapse"] * 40 + ["None"] * 80 + [None, None],
        "age": rng.normal(60, 8, 122),
    })


@pytest.mark.parametrize("route,settings", [
    ("/api/kaplan-meier", {}),
    ("/api/cox", {"covariates": ["age"]}),
])
def test_uploaded_none_label_requires_confirmation_and_is_recorded_in_results_and_export(route, settings):
    source = _source_frame()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        uploaded = client.post("/api/upload", files={"file": ("relapse_none.csv", source.to_csv(index=False), "text/csv")})
        assert uploaded.status_code == 200, uploaded.text
        dataset_id = uploaded.json()["dataset_id"]
        try:
            request = {"dataset_id": dataset_id, "time_column": "time", "event_column": "status", **settings}
            for unconfirmed in ({}, {"event_positive_value": None}, {"event_positive_value": "None"}):
                refused = client.post(route, json={**request, **unconfirmed})
                assert refused.status_code == 400, refused.text
                assert "None" in refused.json()["detail"]
                assert "recode" in refused.json()["detail"]
            response = client.post(route, json={**request, "event_positive_value": "Relapse"})
            assert response.status_code == 200, response.text
            analysis = response.json()["analysis"]
            metadata = analysis["cohort"] if route.endswith("kaplan-meier") else analysis["model_stats"]
            assert metadata["n"] == 120
            assert metadata["events"] == 40
            if route.endswith("kaplan-meier"):
                assert metadata["censored"] == 80
                assert metadata["dropped_missing_rows"] == 2
            coding = metadata["event_coding"]
            assert coding["source"] == "explicit_two_label_selection"
            assert coding["event_positive_value"] == "Relapse"
            note = coding["note"]
            assert note in analysis["scientific_summary"]["cautions"]
            rows = analysis["summary_table"] if route.endswith("kaplan-meier") else analysis["results_table"]
            exported = client.post("/api/export-table", json={"format": "csv", "rows": rows,
                "notes": analysis["scientific_summary"]["cautions"]})
            assert exported.status_code == 200
            assert "literal 'None' = censored (0)" in exported.text
            frame = store.get(dataset_id).dataframe
            assert frame["status"].eq("None").sum() == 80
            assert frame["status"].isna().sum() == 2
        finally:
            store.delete(dataset_id)


def test_confirmed_mapping_matches_an_explicit_binary_recode_numerically():
    frame = _source_frame()
    recoded = frame.copy()
    recoded["status"] = recoded["status"].map({"Relapse": 1, "None": 0})
    actual_km = compute_km_analysis(frame, "time", "status", event_positive_value="Relapse")
    expected_km = compute_km_analysis(recoded, "time", "status", event_positive_value=1)
    assert actual_km["curves"] == expected_km["curves"]
    assert actual_km["summary_table"] == expected_km["summary_table"]
    actual_cox = compute_cox_analysis(frame, "time", "status", covariates=["age"], event_positive_value="Relapse")
    expected_cox = compute_cox_analysis(recoded, "time", "status", covariates=["age"], event_positive_value=1)
    assert actual_cox["results_table"] == expected_cox["results_table"]
    assert actual_cox["diagnostics_table"] == expected_cox["diagnostics_table"]

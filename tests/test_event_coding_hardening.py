from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from survival_toolkit.analysis import compute_km_analysis, suggest_columns
from survival_toolkit.app import app, store


def cohort(column, values):
    return pd.DataFrame({"OS_months": np.arange(1, 25, dtype=float), column: values * 12})


@pytest.mark.parametrize("column", ["OS_censor_status", "CensorStatus", "censored_vital_status", "PFS.CENS.STATUS"])
@pytest.mark.parametrize("values", [[0, 1], [False, True], ["No", "Yes"]])
def test_generic_status_word_does_not_bypass_censor_veto(column, values):
    frame = cohort(column, values)
    before = frame.copy(deep=True)
    assert column not in suggest_columns(frame)["event_columns"]
    # Use a generic time name for the PFS test to isolate the coding guard.
    frame = frame.rename(columns={"OS_months": "time"})
    with pytest.raises(ValueError, match="censoring indicator.*recoded column"):
        compute_km_analysis(frame, "time", column, event_positive_value=values[-1])
    pd.testing.assert_frame_equal(before, frame.rename(columns={"time": "OS_months"}))


def test_explicit_outcome_labels_and_recoded_event_keep_the_same_curve():
    frame = cohort("OS_censor_status", [0, 1])
    before = frame.copy(deep=True)
    frame["event"] = 1 - frame["OS_censor_status"]
    expected = compute_km_analysis(frame, "OS_months", "event", event_positive_value=1)
    frame["OS_censor_status"] = np.where(frame["event"], "Dead", "Alive")
    actual = compute_km_analysis(frame, "OS_months", "OS_censor_status", event_positive_value="Dead")
    assert expected["cohort"]["n"] == actual["cohort"]["n"] == 24
    assert expected["curves"] == actual["curves"]
    assert before["OS_censor_status"].tolist() == [0, 1] * 12


@pytest.mark.parametrize("value", [0, 1])
def test_api_cannot_override_censor_semantics_with_event_value(value):
    frame = cohort("OS_censor_status", [0, 1])
    stored = store.create(frame, filename="source.csv", source="upload")
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            result = client.post("/api/kaplan-meier", json={"dataset_id": stored.dataset_id,
                "time_column": "OS_months", "event_column": "OS_censor_status", "event_positive_value": value})
        assert result.status_code == 400
        assert "censoring indicator" in result.text
        pd.testing.assert_frame_equal(store.get(stored.dataset_id).dataframe, frame)
    finally:
        store.delete(stored.dataset_id)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for browser-script checks")
def test_browser_censor_status_rule_matches_backend(tmp_path):
    from test_frontend_review import _run_page

    result = _run_page(tmp_path, r"""
      return page.run("['OS_censor_status', 'CensorStatus', 'censored_vital_status', 'death_from_censor', 'vital_status'].map(looksLikeCensoringIndicatorName)");
    """)
    assert result == [True, True, True, False, False]

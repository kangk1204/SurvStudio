from __future__ import annotations

import numpy as np
import pytest

from survival_toolkit.evaluation import c_index_intervals, merge_prediction_blocks, prediction_block
from survival_toolkit.marker_screen import harrell_c_many
from survival_toolkit.sample_data import make_example_dataset


def _sksurv_available() -> bool:
    try:
        import sksurv  # noqa: F401
    except ImportError:
        return False
    return True


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _outcomes(n: int = 150, seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-signal))
    censor = rng.exponential(1.5, size=n)
    return np.minimum(event_time, censor), (event_time <= censor).astype(int), signal


def test_intervals_pair_the_models_on_the_same_resampled_patients() -> None:
    time, event, signal = _outcomes()
    rng = np.random.default_rng(9)
    risks = {
        "Cox PH": signal + rng.normal(scale=1.0, size=signal.size),
        "Copy of Cox PH": None,
        "Noise": rng.normal(size=signal.size),
    }
    risks["Copy of Cox PH"] = risks["Cox PH"].copy()

    result = c_index_intervals(time, event, risks, n_bootstrap=400, random_seed=3)

    rows = {row["model"]: row for row in result["rows"]}
    assert result["reference"] == "Cox PH" and result["n"] == time.size and result["n_bootstrap"] == 400
    assert rows["Cox PH"]["c_index"] == pytest.approx(harrell_c_many(time, event, risks["Cox PH"])[0])
    low, high = rows["Cox PH"]["c_index_ci"]
    assert low < rows["Cox PH"]["c_index"] < high
    # The same model paired with itself differs by exactly zero in every draw.
    assert rows["Copy of Cox PH"]["delta_vs_reference"] == 0.0 and rows["Copy of Cox PH"]["delta_ci"] == [0.0, 0.0]
    # Noise is clearly worse than a model with signal, and the paired interval says so.
    assert rows["Noise"]["delta_ci"][1] < 0.0
    assert "delta_vs_reference" not in rows["Cox PH"]


def test_intervals_without_the_reference_model_report_intervals_only() -> None:
    time, event, signal = _outcomes()
    result = c_index_intervals(time, event, {"RSF": signal}, n_bootstrap=100)
    assert result["reference"] is None and "delta_ci" not in result["rows"][0]
    with pytest.raises(ValueError, match="finite risk score"):
        c_index_intervals(time, event, {"RSF": np.full(time.size, np.nan)})


def test_prediction_blocks_keep_the_patients_every_model_scored_and_merge_by_row_label() -> None:
    block = prediction_block(
        ["a", "b", "c", "d"],
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 0],
        {"Cox PH": (None, [0.1, 0.2, 0.3, 0.4]), "RSF": ([0, 2, 3], [5.0, 6.0, 7.0])},
    )
    assert block["row_ids"] == ["a", "c", "d"]
    assert block["risk"] == {"Cox PH": [0.1, 0.3, 0.4], "RSF": [5.0, 6.0, 7.0]}

    other = {"row_ids": ["d", "c", "x"], "time": [4.0, 3.0, 9.0], "event": [0, 1, 1], "risk": {"DeepSurv": [0.9, 0.8, 0.7]}}
    time, event, risks, rows = merge_prediction_blocks([block, other])
    assert rows == ["c", "d"] and time.tolist() == [3.0, 4.0] and event.tolist() == [1, 0]
    assert risks["DeepSurv"].tolist() == [0.8, 0.9] and risks["RSF"].tolist() == [6.0, 7.0]

    with pytest.raises(ValueError, match="disagree on the outcome"):
        merge_prediction_blocks([block, {**other, "event": [1, 1, 1]}])
    with pytest.raises(ValueError, match="share no test patients"):
        merge_prediction_blocks([block, {**other, "row_ids": ["x", "y", "z"]}])


@pytest.mark.skipif(not _sksurv_available(), reason="scikit-survival not installed")
def test_ml_comparison_returns_test_predictions_that_reproduce_each_c_index() -> None:
    from survival_toolkit.ml_models import compare_survival_models, cross_validate_survival_models

    df = make_example_dataset(seed=30, n_patients=200)
    result = compare_survival_models(df, time_column="os_months", event_column="os_event", features=["age", "biomarker_score", "sex"],
                                     categorical_features=["sex"])

    block = result["test_predictions"]
    assert result["evaluation_mode"] == "holdout"
    assert len(block["row_ids"]) == result["n_evaluation_patients"]
    recomputed = c_index_intervals(block["time"], block["event"], block["risk"], n_bootstrap=100)
    reported = {row["model"]: row["c_index"] for row in result["comparison_table"]}
    for row in recomputed["rows"]:
        assert row["c_index"] == pytest.approx(reported[row["model"]], abs=1e-12)

    locked = cross_validate_survival_models(df, time_column="os_months", event_column="os_event", features=["age", "biomarker_score"],
                                            cv_folds=3, cv_repeats=1, locked_test_fraction=0.25)
    locked_block = locked["locked_test_predictions"]
    assert len(locked_block["row_ids"]) == locked["n_locked_test_patients"]
    reported = {row["model"]: row["locked_test_c_index"] for row in locked["comparison_table"]}
    for row in c_index_intervals(locked_block["time"], locked_block["event"], locked_block["risk"], n_bootstrap=100)["rows"]:
        assert row["c_index"] == pytest.approx(reported[row["model"]], abs=1e-12)
    assert "test_risk" not in str(result["comparison_table"]) and "test_risk" not in str(locked["comparison_table"])


@pytest.mark.skipif(not (_torch_available() and _sksurv_available()), reason="torch or scikit-survival not installed")
def test_deep_and_classical_comparisons_pair_on_the_same_test_patients() -> None:
    from survival_toolkit.deep_models import compare_deep_survival_models, evaluate_single_deep_survival_model
    from survival_toolkit.ml_models import compare_survival_models

    df = make_example_dataset(seed=14, n_patients=120)
    features = ["age", "biomarker_score", "immune_index"]
    deep = compare_deep_survival_models(df, time_column="os_months", event_column="os_event", features=features, hidden_layers=[8], epochs=2,
                                        batch_size=16, num_time_bins=6, d_model=16, n_heads=4, n_layers=1, latent_dim=4, n_clusters=3,
                                        random_seed=42, included_models=["DeepSurv", "Neural MTLR"])
    classical = compare_survival_models(df, time_column="os_months", event_column="os_event", features=features, random_state=42)

    reported = {row["model"]: row["c_index"] for row in deep["comparison_table"]}
    block = deep["test_predictions"]
    for row in c_index_intervals(block["time"], block["event"], block["risk"], n_bootstrap=100)["rows"]:
        assert row["c_index"] == pytest.approx(reported[row["model"]], abs=1e-9)
    assert sorted(block["row_ids"]) == sorted(classical["test_predictions"]["row_ids"])

    time, event, risks, rows = merge_prediction_blocks([classical["test_predictions"], block])
    assert len(rows) == classical["n_evaluation_patients"]
    result = c_index_intervals(time, event, risks, n_bootstrap=100)
    assert {row["model"] for row in result["rows"] if "delta_ci" in row} >= {"DeepSurv", "Neural MTLR", "Random Survival Forest"}

    single = evaluate_single_deep_survival_model("deepsurv", df=df, time_column="os_months", event_column="os_event",
                                                 features=features, hidden_layers=[8], epochs=2, batch_size=16, random_seed=42)
    assert "holdout_risk" not in single


def test_interval_endpoint_merges_blocks_and_reports_paired_differences() -> None:
    from fastapi.testclient import TestClient

    from survival_toolkit.app import app

    time, event, signal = _outcomes(n=80)
    rows = [f"r{index}" for index in range(time.size)]
    classical = {"row_ids": rows, "time": time.tolist(), "event": event.tolist(), "risk": {"Cox PH": signal.tolist()}}
    deep = {"row_ids": rows[::-1], "time": time[::-1].tolist(), "event": event[::-1].tolist(), "risk": {"DeepSurv": (-signal[::-1]).tolist()}}
    client = TestClient(app, base_url="http://127.0.0.1")

    response = client.post("/api/model-comparison-intervals", json={"predictions": [classical, deep], "n_bootstrap": 200})

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_shared"] == 80 and payload["reference"] == "Cox PH"
    deepsurv = next(row for row in payload["rows"] if row["model"] == "DeepSurv")
    assert deepsurv["delta_vs_reference"] < 0 and deepsurv["delta_ci"][1] < 0
    mismatch = {**deep, "event": [1] * 80}
    rejected = client.post("/api/model-comparison-intervals", json={"predictions": [classical, mismatch]})
    assert rejected.status_code == 400
    assert "disagree on the outcome of shared test patients" in rejected.json()["detail"]

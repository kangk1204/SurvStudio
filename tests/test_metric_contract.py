"""Regressions for metric validity and reproducibility across optional extras."""

import numpy as np
import pandas as pd
import pytest

from survival_toolkit import ml_models as ml
from survival_toolkit.analysis import _harrell_c_index
from survival_toolkit.errors import UserInputError
from survival_toolkit.marker_screen import harrell_c_many


@pytest.mark.parametrize("risk,expected", [([0.01, 0.01 + 5e-9], 0.0), ([0.01, 0.01], 0.5), ([0.02, 0.01], 1.0)])
def test_optional_ml_c_index_uses_the_same_score_ties(monkeypatch, risk, expected):
    pytest.importorskip("sksurv")
    time = np.array([1.0, 2.0])
    event = np.array([1, 0])
    outcome = np.array([(True, 1.0), (False, 2.0)], dtype=[("event", bool), ("time", float)])
    assert _harrell_c_index(time, event, risk) == expected
    assert harrell_c_many(time, event, risk)[0] == expected
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", True)
    assert ml._sksurv_c_index(outcome, risk) == expected
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", False)
    assert ml._sksurv_c_index(outcome, risk) == expected


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_optional_deep_c_index_has_no_dependency_on_the_metric_extra(monkeypatch, dtype_name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sksurv")
    from survival_toolkit import deep_models as deep

    dtype = getattr(torch, dtype_name)
    risk = torch.tensor([0.01, 0.01 + 5e-9], dtype=dtype)
    time, event = torch.tensor([1.0, 2.0]), torch.tensor([1, 0])
    monkeypatch.setattr(deep, "_SKSURV_METRICS_AVAILABLE", True)
    assert deep._compute_c_index_torch(risk, time, event) == 0.0
    monkeypatch.setattr(deep, "_SKSURV_METRICS_AVAILABLE", False)
    assert deep._compute_c_index_torch(risk, time, event) == 0.0


@pytest.mark.parametrize("bad_probability", [-0.01, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("metric", ["brier", "calibration"])
def test_prediction_metrics_reject_invalid_survival_probabilities(bad_probability, metric):
    time, event = np.arange(1.0, 5.0), np.array([1, 1, 0, 0])
    pred = np.array([bad_probability, 0.6, 0.9, 0.8])
    with pytest.raises(UserInputError, match="finite|between 0 and 1"):
        if metric == "brier":
            ml.compute_integrated_brier_score(time, event, lambda grid: np.broadcast_to(pred[:, None], (4, len(grid))), eval_times=[1.0, 2.0])
        else:
            ml.compute_calibration_data(time, event, pred, t=2.0, n_bins=2)


@pytest.mark.parametrize("time,event,message", [
    ([1, 2], [1, 2], "only 0"),
    ([1, 2], [1, -1], "only 0"),
    ([1, 2], [1, np.nan], "finite"),
    ([1, np.inf], [1, 0], "finite"),
    ([-1, 2], [1, 0], "nonnegative"),
    ([], [], "nonempty"),
    ([[1], [2]], [1, 0], "one-dimensional"),
])
@pytest.mark.parametrize("metric", ["brier", "calibration"])
def test_prediction_metrics_reject_invalid_outcomes(time, event, message, metric):
    with pytest.raises(UserInputError, match=message):
        if metric == "brier":
            ml.compute_integrated_brier_score(time, event, lambda grid: np.full((len(time), len(grid)), 0.5), eval_times=[1.0])
        else:
            ml.compute_calibration_data(time, event, np.full(len(time), 0.5), t=1.0)


def test_brier_support_outcomes_are_checked_separately():
    with pytest.raises(UserInputError, match="support_events.*only 0"):
        ml.compute_integrated_brier_score([1, 2], [1, 0], lambda grid: np.full((2, len(grid)), 0.5),
                                          eval_times=[1], support_times=[1, 2], support_events=[1, 2])


@pytest.mark.parametrize("grid", [[1, np.nan], [[1], [2]], []])
def test_brier_rejects_malformed_requested_time_grids(grid):
    with pytest.raises(UserInputError, match="eval_times"):
        ml.compute_integrated_brier_score([1, 2], [1, 0], lambda grid: np.full((2, len(grid)), 0.5), eval_times=grid)


@pytest.mark.parametrize("horizon", [np.nan, np.inf, -1])
def test_calibration_rejects_invalid_horizons(horizon):
    with pytest.raises(UserInputError, match="finite, nonnegative"):
        ml.compute_calibration_data([1, 2], [1, 0], [0.5, 0.5], t=horizon)


@pytest.mark.parametrize("bins", [0, -1, 2.5, True])
def test_calibration_requires_a_positive_integer_bin_count(bins):
    with pytest.raises(UserInputError, match="positive integer"):
        ml.compute_calibration_data([1, 2], [1, 0], [0.5, 0.5], t=1, n_bins=bins)


def test_valid_probability_boundaries_and_hand_computed_metrics():
    time, event = np.arange(1.0, 5.0), np.array([1, 1, 0, 0])
    # No censoring before t=2: constant S(t)=0.5 has squared error 0.25 for every patient.
    result = ml.compute_integrated_brier_score(time, event, lambda grid: np.full((4, len(grid)), 0.5), eval_times=[1, 2])
    assert result["ibs"] == pytest.approx(0.25)
    calibration = ml.compute_calibration_data(time, event, [0.0, 1.0, 0.5, 1.0], t=2.0, n_bins=1)
    assert calibration["bins"][0]["count"] == 4
    assert calibration["bins"][0]["predicted_mean"] == 0.625
    assert calibration["bins"][0]["observed"] == pytest.approx(0.5)


def test_lasso_inner_selection_excludes_validation_rows_from_imputation_and_levels(monkeypatch):
    pytest.importorskip("sksurv")
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(71)
    n = 100
    frame = pd.DataFrame({"t": rng.exponential(10, n) + 0.1, "e": np.arange(n) % 2,
                          "x": rng.normal(size=n), "category": np.where(np.arange(n) % 3, "common", "other"),
                          "row_id": np.arange(n)})
    frame.loc[::7, "x"] = np.nan
    folds = list(StratifiedKFold(5, shuffle=True, random_state=11).split(frame, frame.e))
    # A category present only in the first validation fold must not become a learned level.
    frame.loc[folds[0][1], "category"] = "validation_only"
    frame.loc[folds[0][1], "x"] += 20
    features = ["x", "category"]
    encoded, _, _ = ml._encode_train_test_features(frame, frame, features, ["category"])
    original = ml._fit_feature_encoder
    seen = []

    def capture(training, selected, categorical=None):
        encoder = original(training, selected, categorical)
        transformed = ml._transform_feature_encoder(training, encoder)
        seen.append((training.copy(), encoder, transformed))
        return encoder

    monkeypatch.setattr(ml, "_fit_feature_encoder", capture)
    result = ml._select_lasso_alpha(frame, encoded, features=features, categorical_features=["category"],
                                    time_column="t", event_column="e", random_state=11)
    assert result["selection_mode"] == "inner_cv"
    assert len(seen) == 5
    for (training, encoder, transformed), (train_rows, eval_rows) in zip(seen, folds):
        assert training.row_id.tolist() == train_rows.tolist()
        assert set(training.row_id).isdisjoint(eval_rows)
        missing = training.x.isna()
        assert missing.any()
        assert transformed.loc[missing, "x"].tolist() == pytest.approx([training.x.median()] * int(missing.sum()))
        assert set(encoder["categorical_mappings"]["category"]["all_levels"]) == set(training.category)
    assert "validation_only" not in seen[0][1]["categorical_mappings"]["category"]["all_levels"]
    assert seen[0][0].x.median() != pytest.approx(frame.x.median())

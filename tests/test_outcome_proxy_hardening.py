from __future__ import annotations

import importlib.util
import threading

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.concurrency import cancellation_scope
from survival_toolkit.errors import JobCancelledError, OutcomeProxyError, must_propagate
from survival_toolkit.outcome_proxy import screen_training_outcomes, training_fingerprint
from survival_toolkit.prediction_guard import check_training_outcome_proxies


def training_frame(n=40):
    rng = np.random.default_rng(2026101407)
    return pd.DataFrame({
        "time": rng.exponential(10, n),
        "event": np.arange(n) % 2,
        "baseline": rng.normal(size=n),
    })


@pytest.mark.parametrize("encoding", ["numeric", "category"])
def test_supported_small_exact_event_patterns_require_source_review(encoding):
    frame = training_frame(24)
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    before = frame.copy(deep=True)
    audit = screen_training_outcomes(frame, "time", "event", ["renamed"])
    finding, = audit["findings"]
    assert audit["status"] == "review_required"
    assert audit["coverage"] == "limited" and audit["scores"] == []
    assert finding["target"] == "event" and finding["rows_checked"] == 24
    assert finding["training_rows"] == 24 and finding["missing_predictor_rows"] == 0
    expected = "exact_affine_outcome_pattern" if encoding == "numeric" else "exact_categorical_event_pattern"
    assert finding["kind"] == expected
    assert any("learned screening" in note for note in audit["notes"])
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("encoding", ["numeric", "category"])
@pytest.mark.parametrize("n", [12, 23])
def test_too_few_rows_do_not_claim_exact_pattern_assessment(encoding, n):
    frame = training_frame(n)
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    audit = screen_training_outcomes(frame, "time", "event", ["renamed"])
    assert audit["status"] == "not_assessed" and audit["findings"] == []
    assert audit["coverage"] == "limited" and audit["scores"] == []


@pytest.mark.parametrize("encoding", ["numeric", "category"])
@pytest.mark.parametrize("event_count", [1, 5])
def test_rare_event_indicator_requires_support_before_exact_event_flag(encoding, event_count):
    frame = training_frame()
    frame["event"] = 0
    frame.loc[:event_count - 1, "event"] = 1
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    audit = screen_training_outcomes(frame, "time", "event", ["renamed"])
    assert audit["status"] == "not_assessed" and audit["findings"] == []


@pytest.mark.parametrize("encoding", ["numeric", "category"])
@pytest.mark.parametrize("event_count", [6, 18])
def test_exact_event_pattern_accepts_the_support_boundary_in_both_states(encoding, event_count):
    frame = training_frame(24)
    frame["event"] = 0
    frame.loc[:event_count - 1, "event"] = 1
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    audit = screen_training_outcomes(frame, "time", "event", ["renamed"])
    assert audit["status"] == "review_required"
    assert audit["findings"][0]["target"] == "event"


@pytest.mark.parametrize("encoding", ["numeric", "category"])
def test_partial_missing_exact_pattern_reports_its_actual_support(encoding):
    frame = training_frame()
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    frame["renamed"] = frame["renamed"].astype(float if encoding == "numeric" else object)
    frame.loc[24:, "renamed"] = np.nan
    audit = screen_training_outcomes(frame, "time", "event", ["renamed"])
    finding, = audit["findings"]
    assert audit["status"] == "review_required" and audit["coverage"] == "limited"
    assert finding["rows_checked"] == 24 and finding["training_rows"] == 40
    assert finding["missing_predictor_rows"] == 16


@pytest.mark.parametrize("encoding", ["numeric", "category"])
def test_insufficient_complete_rows_do_not_receive_an_exact_event_flag(encoding):
    frame = training_frame()
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    frame["renamed"] = frame["renamed"].astype(float if encoding == "numeric" else object)
    frame.loc[23:, "renamed"] = np.nan
    assert screen_training_outcomes(frame, "time", "event", ["renamed"])["findings"] == []


@pytest.mark.parametrize("encoding", ["numeric", "category"])
def test_event_support_is_measured_on_complete_predictor_rows(encoding):
    frame = training_frame()
    frame["renamed"] = (13 - 7 * frame.event if encoding == "numeric" else
                        np.where(frame.event, "red", "blue"))
    frame["renamed"] = frame["renamed"].astype(float if encoding == "numeric" else object)
    # Twenty censored and four event rows have values; the full cohort is balanced.
    incomplete = frame.index[frame.event == 1][4:]
    frame.loc[incomplete, "renamed"] = np.nan
    assert frame.renamed.notna().sum() == 24
    assert screen_training_outcomes(frame, "time", "event", ["renamed"])["findings"] == []


def test_wide_exact_pattern_is_checked_beyond_the_learned_feature_budget():
    frame = training_frame()
    rng = np.random.default_rng(2026101408)
    features = [f"measurement_{index}" for index in range(350)]
    frame = pd.concat([frame, pd.DataFrame(rng.normal(size=(40, 350)), columns=features)], axis=1)
    frame[features[-1]] = 13 - 7 * frame.event
    audit = screen_training_outcomes(frame, "time", "event", features)
    finding, = audit["findings"]
    assert audit["status"] == "review_required" and finding["features"] == [features[-1]]
    assert audit["scores"] == []


@pytest.mark.parametrize("n", [24, 40])
def test_ordinary_small_numeric_and_nonnumeric_baseline_inputs_are_not_paused(n):
    frame = training_frame(n)
    frame["stage"] = np.resize(["I", "I", "II", "II"], n)
    audit = screen_training_outcomes(frame, "time", "event", ["baseline", "stage"], ["stage"])
    assert audit["status"] == "not_assessed" and audit["findings"] == []
    assert audit["coverage"] == "limited"


def test_fingerprint_distinguishes_category_order_levels_and_declared_encoding():
    frame = training_frame()
    frame["group"] = pd.Categorical(np.resize(["a", "b"], len(frame)), categories=["a", "b"])
    reordered = frame.copy()
    reordered["group"] = reordered.group.cat.reorder_categories(["b", "a"])
    ordered = frame.copy()
    ordered["group"] = ordered.group.cat.as_ordered()
    additional = frame.copy()
    additional["group"] = additional.group.cat.add_categories(["unused"])
    fingerprints = {
        training_fingerprint(frame), training_fingerprint(reordered),
        training_fingerprint(ordered), training_fingerprint(additional),
        training_fingerprint(frame, ["group"]),
    }
    assert len(fingerprints) == 5
    assert training_fingerprint(frame) == training_fingerprint(frame.copy())
    changed_dtype = frame.copy()
    changed_dtype["baseline"] = changed_dtype.baseline.astype("float32")
    assert training_fingerprint(changed_dtype) != training_fingerprint(frame)


def test_structural_scan_honors_cancellation_before_learned_screening():
    cancelled = threading.Event()
    cancelled.set()
    with cancellation_scope(cancelled), pytest.raises(JobCancelledError) as caught:
        screen_training_outcomes(training_frame(), "time", "event", ["baseline"])
    assert must_propagate(caught.value)


def test_wide_structural_scan_polls_cancellation_during_feature_iteration():
    class CancelOnSecondPoll(threading.Event):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def is_set(self):
            self.polls += 1
            return self.polls >= 2

    frame = training_frame()
    features = [f"measurement_{index}" for index in range(65)]
    rng = np.random.default_rng(2026101410)
    frame = pd.concat([frame, pd.DataFrame(rng.normal(size=(40, 65)), columns=features)], axis=1)
    cancelled = CancelOnSecondPoll()
    with cancellation_scope(cancelled), pytest.raises(JobCancelledError):
        screen_training_outcomes(frame, "time", "event", features)
    assert cancelled.polls == 2


def test_existing_learned_thresholds_seed_and_budget_remain_fixed():
    audit = screen_training_outcomes(training_frame(), "time", "event", ["baseline"])
    settings = audit["settings"]
    assert {key: settings[key] for key in (
        "seed", "inner_folds", "min_rows", "max_rows", "event_auc",
        "event_balanced_accuracy", "log_follow_up_r2", "tree_count", "tree_depth",
        "neural_layers", "neural_max_iter", "neural_max_features", "joint_max_features",
    )} == {
        "seed": 2026101101, "inner_folds": 3, "min_rows": 48, "max_rows": 1200,
        "event_auc": 0.995, "event_balanced_accuracy": 0.98, "log_follow_up_r2": 0.995,
        "tree_count": 48, "tree_depth": 6, "neural_layers": [16, 8], "neural_max_iter": 150,
        "neural_max_features": 64, "joint_max_features": 256,
    }


def test_new_small_category_finding_retains_typed_whole_request_stop():
    frame = training_frame(24)
    frame["renamed"] = np.where(frame.event, "red", "blue")
    with pytest.raises(OutcomeProxyError) as caught:
        check_training_outcome_proxies(frame, "time", "event", ["renamed"], ["renamed"])
    assert must_propagate(caught.value)
    assert caught.value.audit["status"] == "review_required"
    assert caught.value.audit["findings"][0]["kind"] == "exact_categorical_event_pattern"
    assert "strong baseline predictor" in str(caught.value)


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="learned screen needs scikit-learn")
def test_strong_partial_baseline_signal_is_not_treated_as_observed_outcome_copy():
    rng = np.random.default_rng(2026101409)
    n = 240
    baseline = rng.normal(size=n)
    failure = rng.exponential(10 * np.exp(-1.5 * baseline))
    censor = rng.exponential(20, n)
    frame = pd.DataFrame({"time": np.minimum(failure, censor),
                          "event": (failure <= censor).astype(int), "baseline": baseline})
    audit = screen_training_outcomes(frame, "time", "event", ["baseline"])
    assert audit["status"] == "no_pattern_detected" and audit["findings"] == []

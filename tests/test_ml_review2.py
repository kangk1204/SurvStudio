"""Regression tests for the second review of the ML models (ml_models.py)."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.errors import InternalAnalysisError, JobCancelledError, UserInputError
from survival_toolkit.sample_data import make_example_dataset


def _sksurv_available() -> bool:
    try:
        import sksurv  # noqa: F401
    except ImportError:
        return False
    return True


requires_sksurv = pytest.mark.skipif(not _sksurv_available(), reason="scikit-survival not installed")

_FIT_FUNCTIONS = ("_fit_evaluate_cox_split", "_fit_evaluate_lasso_cox_split", "_fit_evaluate_rsf_split", "_fit_evaluate_gbs_split")


def _fake_fit(c_index: float | None = 0.6):
    """A fold fit that returns fixed metrics without fitting anything."""

    def _fit(train_frame, test_frame, **kwargs):
        event_column = kwargs["event_column"]
        return {
            "c_index": c_index,
            "n_features": len(kwargs["features"]),
            "training_time_ms": 1.0,
            "train_n": len(train_frame),
            "test_n": len(test_frame),
            "train_events": int(train_frame[event_column].sum()),
            "test_events": int(test_frame[event_column].sum()),
        }

    return _fit


def _patch_fits(monkeypatch, ml, **overrides) -> None:
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", True)
    for name in _FIT_FUNCTIONS:
        monkeypatch.setattr(ml, name, overrides.get(name, _fake_fit()))


def _survstudio_bug(ml):
    """A function whose TypeError is raised inside SurvStudio code, as a coding error would be."""
    namespace: dict[str, object] = {}
    exec(compile("def _bug(*args, **kwargs):\n    return None + 1\n", "injected_bug", "exec"), ml.__dict__, namespace)
    return namespace["_bug"]


# ── Exception handling of the per-model fallbacks and the Brier helpers ─────


def test_brier_helpers_let_index_and_zero_division_errors_propagate(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1, 0, 1, 0])

    def failing(exc):
        def _raise(*args, **kwargs):
            raise exc
        return _raise

    for exc in (IndexError("bug"), ZeroDivisionError("bug")):
        monkeypatch.setattr(ml, "compute_integrated_brier_score", failing(exc))
        with pytest.raises(type(exc)):
            ml._maybe_compute_brier_metrics(times, events, lambda grid: None)
        monkeypatch.setattr(ml, "_sksurv_survival_predictor", failing(exc))
        with pytest.raises(type(exc)):
            ml._maybe_compute_sksurv_brier_metrics(times, events, types.SimpleNamespace(predict_survival_function=len),
                                                   pd.DataFrame({"x": times}))
    # Failures that describe the data still only blank the Brier metrics.
    for exc in (ValueError("no support"), np.linalg.LinAlgError("singular"), FloatingPointError("overflow")):
        monkeypatch.setattr(ml, "compute_integrated_brier_score", failing(exc))
        with pytest.warns(RuntimeWarning, match="could not be computed"):
            assert ml._maybe_compute_brier_metrics(times, events, lambda grid: None) is None


def test_model_fallbacks_propagate_memory_errors(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    features = ["age", "biomarker_score"]

    def _oom(train_frame, test_frame, **kwargs):
        raise MemoryError()

    _patch_fits(monkeypatch, ml, _fit_evaluate_rsf_split=_oom)
    with pytest.raises(MemoryError):
        ml.compare_survival_models(df, "os_months", "os_event", features)
    with pytest.raises(MemoryError):
        ml.cross_validate_survival_models(df, "os_months", "os_event", features, cv_folds=3, cv_repeats=1)

    from survival_toolkit.evaluation import locked_test_split

    dev_positions, _ = locked_test_split(df["os_event"].to_numpy(), random_state=11, test_fraction=0.3)

    def _oom_on_locked_refit(train_frame, test_frame, **kwargs):
        if len(train_frame) == len(dev_positions):
            raise MemoryError()
        return _fake_fit()(train_frame, test_frame, **kwargs)

    _patch_fits(monkeypatch, ml, _fit_evaluate_gbs_split=_oom_on_locked_refit)
    with pytest.raises(MemoryError):
        ml.cross_validate_survival_models(df, "os_months", "os_event", features, cv_folds=3, cv_repeats=1,
                                          random_state=11, locked_test_fraction=0.3)


def test_model_fallbacks_propagate_a_bug_the_brier_helper_reraised(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=30, n_patients=150)
    # A TypeError inside the IBS code (wrapped in an InternalAnalysisError by its public boundary)
    # is a coding error: the Cox fit's Brier helper re-raises it, and so must the comparison.
    monkeypatch.setattr(ml, "_ipcw_brier_weights", _survstudio_bug(ml))
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", False)
    for run in (
        lambda: ml.compare_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"]),
        lambda: ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=2,
                                                  cv_repeats=1),
    ):
        with pytest.raises(InternalAnalysisError) as caught:
            run()
        assert isinstance(caught.value.__cause__, TypeError)


def test_failures_with_an_empty_message_are_recorded_by_exception_type(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)

    def _silent_failure(train_frame, test_frame, **kwargs):
        raise ValueError()

    _patch_fits(monkeypatch, ml, _fit_evaluate_rsf_split=_silent_failure)
    comparison = ml.compare_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"])
    assert comparison["errors"] == [{"model": "Random Survival Forest", "error": "ValueError"}]
    cv = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=2,
                                           cv_repeats=1)
    assert {error["error"] for error in cv["errors"]} == {"ValueError"}


def test_a_cox_baseline_failure_only_blanks_the_cox_brier_metrics(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)

    def _no_baseline(*args, **kwargs):
        raise ValueError("baseline hazard unavailable")

    monkeypatch.setattr(ml, "_cox_ph_survival_predictor", _no_baseline)
    _patch_fits(monkeypatch, ml, _fit_evaluate_cox_split=ml._fit_evaluate_cox_split)
    with pytest.warns(RuntimeWarning, match="baseline hazard unavailable"):
        comparison = ml.compare_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"])
    cox = next(row for row in comparison["comparison_table"] if row["model"] == "Cox PH")
    assert cox["c_index"] is not None and cox["ibs"] is None
    assert comparison["errors"] == []


# ── Repeated cross-validation ───────────────────────────────────────────────


def _late_event_cohort(n: int = 40, n_events: int = 6) -> pd.DataFrame:
    """Events only at the longest follow-up times: a test fold with a single event has no comparable pair."""
    rng = np.random.default_rng(5)
    time = np.arange(1.0, n + 1.0)
    event = np.zeros(n, dtype=int)
    event[-n_events:] = 1
    return pd.DataFrame({"time": time, "event": event, "x": rng.normal(size=n), "z": rng.normal(size=n)})


def _c_index_fit(seen: list[pd.DataFrame] | None = None):
    """A fold fit whose C-index is Harrell's C of the feature x on the test fold (None without a comparable pair)."""
    from survival_toolkit.analysis import _harrell_c_index

    def _fit(train_frame, test_frame, **kwargs):
        if seen is not None:
            seen.append(test_frame)
        result = _fake_fit()(train_frame, test_frame, **kwargs)
        result["c_index"] = _harrell_c_index(test_frame["time"].to_numpy(dtype=float),
                                             test_frame["event"].to_numpy(dtype=float), test_frame["x"].to_numpy(dtype=float))
        return result

    return _fit


def test_repeated_cv_skips_folds_without_a_comparable_pair_for_every_model_alike(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = _late_event_cohort()
    seen: list[pd.DataFrame] = []
    _patch_fits(monkeypatch, ml, **{name: _c_index_fit(seen) for name in _FIT_FUNCTIONS})
    result = ml.cross_validate_survival_models(df, "time", "event", ["x", "z"], cv_folds=5, cv_repeats=3, random_state=42)

    # Six events in five stratified folds: four folds per repeat hold a single event, which no
    # patient in its fold outlives, so the C-index is undefined there for every model.
    assert result["n_skipped_folds"] == 12 and len(result["skipped_folds"]) == 12
    assert all(ml._has_comparable_pair(fold["time"], fold["event"]) for fold in seen)
    assert result["evaluation_mode"] == "repeated_cv"
    assert result["errors"] == [] and result["excluded_models"] == [] and result["ranking_complete"] is True
    for row in result["comparison_table"]:
        assert row["c_index"] is not None and row["n_evaluations"] == 3 and row["evaluation_mode"] == "repeated_cv"
    assert [row["Rank"] for row in result["manuscript_tables"]["model_performance_table"]] == [1, 2, 3, 4]
    cautions = result["scientific_summary"]["cautions"]
    assert any(text.startswith("12 of 15 cross-validation test folds had no comparable pair") for text in cautions)
    assert not any("failed or fell back" in text for text in cautions)
    assert any("12 test fold(s) had no comparable pair" in note for note in result["manuscript_tables"]["table_notes"])


def test_repeated_cv_refuses_when_no_fold_has_a_comparable_pair(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    _patch_fits(monkeypatch, ml, **{name: _c_index_fit() for name in _FIT_FUNCTIONS})
    # Five events in five folds: every test fold holds exactly one event and nobody outlives it.
    with pytest.raises(UserInputError, match="No cross-validation test fold had a comparable pair.*fewer folds or a cohort with more events"):
        ml.cross_validate_survival_models(_late_event_cohort(n_events=5), "time", "event", ["x", "z"], cv_folds=5,
                                          cv_repeats=2)


def test_locked_test_set_without_a_comparable_pair_is_explained(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml
    from survival_toolkit.evaluation import locked_test_split

    df = _late_event_cohort(n=40, n_events=5)
    events = df["event"].to_numpy()
    # A seed whose locked test set holds a single event, which none of its patients outlives.
    seed = next(
        seed for seed in range(100)
        if int(events[locked_test_split(events, random_state=seed, test_fraction=0.25)[1]].sum()) == 1
    )
    _patch_fits(monkeypatch, ml, **{name: _c_index_fit() for name in _FIT_FUNCTIONS})
    result = ml.cross_validate_survival_models(df, "time", "event", ["x", "z"], cv_folds=2, cv_repeats=1,
                                               random_state=seed, locked_test_fraction=0.25)
    assert all(row["locked_test_c_index"] is None for row in result["comparison_table"])
    summary = result["scientific_summary"]
    assert any(text.startswith("The locked test set has no comparable pair of patients") for text in summary["cautions"])
    assert not any("reached a locked-test C-index" in text for text in summary["strengths"])


def test_repeated_cv_excludes_a_model_that_lacks_a_fold_the_others_have(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    calls = {"n": 0}

    def _nan_scores_once(train_frame, test_frame, **kwargs):
        calls["n"] += 1
        return _fake_fit(None if calls["n"] == 2 else 0.6)(train_frame, test_frame, **kwargs)

    _patch_fits(monkeypatch, ml, _fit_evaluate_gbs_split=_nan_scores_once)
    result = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=3,
                                               cv_repeats=1)
    gbs = next(row for row in result["comparison_table"] if row["model"] == "Gradient Boosted Survival")
    assert gbs["c_index"] is None and gbs["evaluation_mode"] == "repeated_cv_incomplete" and gbs["n_failures"] == 1
    assert result["excluded_models"] == ["Gradient Boosted Survival"] and result["ranking_complete"] is False
    assert [error["model"] for error in result["errors"]] == ["Gradient Boosted Survival"]
    assert "not all finite" in result["errors"][0]["error"]
    others = [row for row in result["comparison_table"] if row["model"] != "Gradient Boosted Survival"]
    assert all(row["c_index"] is not None for row in others)


def test_repeated_cv_accepts_the_largest_seed(monkeypatch) -> None:
    from sklearn.model_selection import StratifiedKFold

    from survival_toolkit import ml_models as ml
    from survival_toolkit.analysis import _cohort_frame

    df = make_example_dataset(seed=27, n_patients=150)
    features = ["age", "biomarker_score"]
    seeds: list[int] = []
    test_ages: list[list[float]] = []

    def _record_seed(train_frame, test_frame, **kwargs):
        seeds.append(int(kwargs["random_state"]))
        test_ages.append(sorted(test_frame["age"].tolist()))
        return _fake_fit()(train_frame, test_frame, **kwargs)

    _patch_fits(monkeypatch, ml, _fit_evaluate_cox_split=_record_seed)
    result = ml.cross_validate_survival_models(df, "os_months", "os_event", features, cv_folds=3, cv_repeats=2,
                                               random_state=2**32 - 1)
    # The second repeat's seed wraps to 0 instead of overflowing scikit-learn's seed range.
    assert seeds == [2**32 - 1] * 3 + [0] * 3
    assert result["evaluation_mode"] == "repeated_cv" and result["errors"] == []
    frame = _cohort_frame(df, "os_months", "os_event", extra_columns=features, drop_missing_extra_columns=False)
    events = frame["os_event"].astype(int).to_numpy()
    expected = [
        sorted(frame["age"].to_numpy()[test].tolist())
        for seed in (2**32 - 1, 0)
        for _, test in StratifiedKFold(3, shuffle=True, random_state=seed).split(frame, events)
    ]
    assert test_ages == expected
    # Ordinary seeds keep their folds: the derived seed is the plain sum when it does not overflow.
    assert ml._derived_seed(42, 2) == 44


def test_repeated_cv_summary_describes_fold_means_and_models_that_could_not_run(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", False)
    monkeypatch.setattr(ml, "_fit_evaluate_cox_split", _fake_fit())
    result = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=5,
                                               cv_repeats=3)
    summary = result["scientific_summary"]
    # Models skipped because scikit-survival is missing are not fold failures.
    assert result["evaluation_mode"] == "repeated_cv"
    assert not any("fold-level fit" in text for text in summary["cautions"])
    assert any(text.startswith("3 model(s) were not evaluated because scikit-survival is not installed") for text in summary["cautions"])
    assert result["excluded_models"] == ["Gradient Boosted Survival", "LASSO-Cox", "Random Survival Forest"]
    assert result["ranking_complete"] is False
    # The counts are mean fold sizes, not patients.
    mean_test_events = int(round(np.mean([row["test_events"] for row in result["fold_results"]])))
    assert summary["strengths"][1] == (
        f"Repeated-CV mean C-index was estimated on test folds of about 30 patients ({mean_test_events} events) each."
    )
    assert "training folds of about 120 patients" in summary["strengths"][0]


# ── Partial dependence ──────────────────────────────────────────────────────


@requires_sksurv
def test_partial_dependence_counts_categories_as_the_encoder_reads_them() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=200).copy()
    df["dose"] = np.random.default_rng(3).integers(1, 4, size=len(df)).astype(float)
    features = ["age", "biomarker_score", "dose"]
    # One held-out row holds 2.5, so the whole column reads as decimals ("2.0") while the
    # encoder, fitted on whole-number training rows, stores the levels "1", "2", "3".
    df.loc[_holdout_rows(df, features, 1), "dose"] = 2.5
    fitted = ml.train_random_survival_forest(df, "os_months", "os_event", features, categorical_features=["dose"],
                                             n_estimators=5, random_state=42, compute_importance=False, compute_brier=False)
    assert fitted["_feature_encoder"]["categorical_mappings"]["dose"]["all_levels"] == ["1", "2", "3"]
    pdp = ml.compute_partial_dependence(fitted["_model"], fitted["_X_encoded"], "dose", categorical_features=["dose"],
                                        feature_encoder=fitted["_feature_encoder"], analysis_frame=fitted["_analysis_frame"])
    observed = fitted["_analysis_frame"]["dose"].value_counts()
    assert pdp["category_counts"] == {"1": int(observed[1.0]), "2": int(observed[2.0]), "3": int(observed[3.0])}


@requires_sksurv
def test_partial_dependence_refuses_a_column_the_model_never_used() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=21, n_patients=160)
    fitted = ml.train_random_survival_forest(df, "os_months", "os_event", ["age", "biomarker_score"], n_estimators=5,
                                             random_state=1, compute_importance=False, compute_brier=False)
    with pytest.raises(UserInputError, match="'os_months' is not an input of the fitted model. Use one of: 'age', 'biomarker_score'"):
        ml.compute_partial_dependence(fitted["_model"], fitted["_X_encoded"], "os_months", n_points=5,
                                      feature_encoder=fitted["_feature_encoder"], analysis_frame=fitted["_analysis_frame"])


def test_partial_dependence_refuses_non_finite_predictions_instead_of_leaving_gaps() -> None:
    from survival_toolkit import ml_models as ml

    class _NanModel:
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return np.where(X[:, 0] > 25.0, np.nan, X[:, 0])

    frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
    encoder = ml._fit_feature_encoder(frame, ["age"])
    with pytest.raises(UserInputError, match="non-finite risk scores"):
        ml.compute_partial_dependence(_NanModel(), frame, "age", n_points=3, feature_encoder=encoder, analysis_frame=frame)
    with pytest.raises(UserInputError, match="non-finite risk scores"):
        ml.compute_partial_dependence(_NanModel(), frame, "age", n_points=3)


# ── Numerical helpers ───────────────────────────────────────────────────────


@pytest.mark.parametrize("sksurv_path", [True, False])
def test_c_index_is_undefined_for_non_finite_risk_scores_or_times(monkeypatch, sksurv_path: bool) -> None:
    from survival_toolkit import ml_models as ml

    if sksurv_path and not ml.SKSURV_AVAILABLE:
        pytest.skip("scikit-survival not installed")
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", sksurv_path)
    y = np.empty(6, dtype=[("event", bool), ("time", float)])
    y["event"] = [True, True, False, True, False, True]
    y["time"] = [1, 2, 3, 4, 5, 6]
    assert ml._sksurv_c_index(y, np.array([6.0, 5, 4, 3, 2, 1])) == pytest.approx(1.0)
    for risk in ([np.nan, 5, 4, 3, 2, 1.0], [6, 5, 4, 3, 2, np.nan], [np.inf, 5, 4, 3, 2, 1.0]):
        assert ml._sksurv_c_index(y, np.array(risk)) is None
    y_inf = y.copy()
    y_inf["time"][5] = np.inf
    assert ml._sksurv_c_index(y_inf, np.array([6.0, 5, 4, 3, 2, 1])) is None


def _loop_breslow(times, status, linear_predictor):
    """The per-event-time loop the vectorised Breslow estimator replaced."""
    times = np.asarray(times, dtype=float)
    status = np.asarray(status, dtype=float) > 0
    risk = np.exp(np.clip(np.asarray(linear_predictor, dtype=float), -50.0, 50.0))
    event_times = np.unique(times[status])
    order = np.argsort(times, kind="mergesort")
    reverse_cumsum = np.cumsum(risk[order][::-1])[::-1]
    risk_set_sums = reverse_cumsum[np.searchsorted(times[order], event_times, side="left")]
    deaths = np.array([np.sum(status & (times == value)) for value in event_times], dtype=float)
    return event_times, np.exp(-np.cumsum(deaths / risk_set_sums))


def test_breslow_baseline_counts_tied_deaths_exactly_as_the_loop_did() -> None:
    from survival_toolkit.ml_models import _breslow_baseline_survival

    rng = np.random.default_rng(0)
    for n in (5, 200, 3000):
        times = np.round(rng.exponential(20.0, n))  # many tied times
        status = (rng.random(n) < 0.6).astype(float)
        linear_predictor = rng.normal(size=n)
        expected_times, expected_survival = _loop_breslow(times, status, linear_predictor)
        event_times, survival = _breslow_baseline_survival(times, status, linear_predictor)
        assert np.array_equal(event_times, expected_times)
        assert np.array_equal(survival, expected_survival)


def test_repeat_with_a_single_scored_fold_reports_no_spread() -> None:
    from survival_toolkit.ml_models import _summarize_repeated_cv_rows

    common = {"n_features": 3, "train_n": 80, "test_n": 20, "training_time_ms": 5.0}
    summary = _summarize_repeated_cv_rows([
        {"repeat": 1, "c_index": 0.70, **common},
        {"repeat": 2, "c_index": 0.60, **common},
        {"repeat": 2, "c_index": 0.64, **common},
    ])
    assert summary["repeat_results"][0]["c_index_std"] is None
    assert summary["repeat_results"][1]["c_index_std"] == pytest.approx(float(np.std([0.60, 0.64], ddof=1)))


@requires_sksurv
def test_lasso_inner_cross_validation_stops_when_its_request_is_cancelled() -> None:
    import threading

    from survival_toolkit import ml_models as ml
    from survival_toolkit.concurrency import cancellation_scope

    df = make_example_dataset(seed=24, n_patients=200)
    encoded, _, _ = ml._encode_train_test_features(df, df, ["age", "biomarker_score", "immune_index"])
    cancelled = threading.Event()
    cancelled.set()
    with cancellation_scope(cancelled), pytest.raises(JobCancelledError):
        ml._select_lasso_alpha(df.reset_index(drop=True), encoded.reset_index(drop=True), time_column="os_months",
                               event_column="os_event", random_state=11, features=["age", "biomarker_score", "immune_index"])


# ── Summaries and descriptions ──────────────────────────────────────────────


@requires_sksurv
def test_gradient_boosting_summary_says_it_is_a_boosted_cox_model() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=30, n_patients=120)
    result = ml.train_gradient_boosted_survival(df, "os_months", "os_event", ["age", "biomarker_score"], n_estimators=5,
                                                compute_importance=False, compute_brier=False)
    strengths = result["scientific_summary"]["strengths"]
    assert not any("no proportional-hazards assumption" in text for text in strengths)
    assert any(text.startswith("Boosted Cox model") and "assumed to be proportional" in text for text in strengths)


@requires_sksurv
def test_lasso_summary_does_not_claim_unused_evaluation_rows_in_apparent_mode() -> None:
    from survival_toolkit import ml_models as ml

    rng = np.random.default_rng(1)
    n = 200
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    event = np.ones(n, dtype=int)
    event[:3] = 0  # three censored patients: too few for a holdout
    df = pd.DataFrame({"t": rng.exponential(np.exp(-0.7 * x1)) * 10 + 0.1, "e": event, "x1": x1, "x2": x2})
    result = ml.train_lasso_cox(df, "t", "e", ["x1", "x2"], random_state=3)
    assert result["model_stats"]["evaluation_mode"] == "apparent"
    assert result["model_stats"]["alpha_selection_mode"] == "inner_cv"
    summary = result["scientific_summary"]
    assert not any("evaluation rows were never used" in text for text in summary["cautions"])
    assert any("also the evaluation rows" in text and "optimistic" in text for text in summary["cautions"])
    assert any("inner cross-validation on the fitting rows" in text for text in summary["strengths"])


@requires_sksurv
def test_lasso_penalty_selection_reports_the_folds_split_and_the_folds_scored(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=24, n_patients=200)
    features = ["age", "biomarker_score", "immune_index"]
    encoded, _, _ = ml._encode_train_test_features(df, df, features)
    original = ml._drop_constant_train_columns
    calls = {"n": 0}

    def _second_inner_fold_fails(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # the first call is the full path, the next ones the inner folds
            raise ValueError("degenerate inner fold")
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "_drop_constant_train_columns", _second_inner_fold_fails)
    meta = ml._select_lasso_alpha(df.reset_index(drop=True), encoded.reset_index(drop=True), time_column="os_months",
                                  event_column="os_event", random_state=11, features=features)
    assert meta["selection_mode"] == "inner_cv"
    assert meta["inner_cv_folds"] == 5 and meta["inner_cv_scored_folds"] == 4

    monkeypatch.setattr(ml, "_drop_constant_train_columns", original)
    monkeypatch.setattr(ml, "_select_lasso_alpha", lambda *args, **kwargs: meta)
    fitted = ml.train_lasso_cox(df, "os_months", "os_event", features)
    assert any(
        "5-fold stratified inner cross-validation on the training split (4 of the 5 folds could be scored)" in text
        for text in fitted["scientific_summary"]["strengths"]
    )


def test_headlines_use_the_right_article_and_keep_the_metric_capitalisation() -> None:
    from survival_toolkit import ml_models as ml

    def headline(mode):
        return ml._scientific_summary_ml(model_name="RSF", c_index=0.7, n_patients=100, n_events=40, n_features=3,
                                         evaluation_mode=mode)["headline"]

    assert headline("apparent") == "RSF estimated an apparent C-index of 0.700 on the current evaluation path."
    assert headline("holdout") == "RSF estimated a holdout C-index of 0.700 on the current evaluation path."
    assert headline("repeated_cv") == "RSF estimated a repeated-CV mean C-index of 0.700 on the current evaluation path."


@requires_sksurv
def test_permutation_importance_method_states_its_repeats_subsample_and_sample() -> None:
    from survival_toolkit import ml_models as ml

    holdout = ml.train_random_survival_forest(make_example_dataset(seed=20, n_patients=200), "os_months", "os_event",
                                              ["age", "biomarker_score"], n_estimators=5, compute_brier=False)
    assert "averaged over 5 shuffles" in holdout["importance_method"]
    assert "random subsample of 300 rows" in holdout["importance_method"]
    assert "in-sample" not in holdout["importance_method"]
    apparent = ml.train_random_survival_forest(make_example_dataset(seed=35, n_patients=16), "os_months", "os_event",
                                               ["age", "biomarker_score"], n_estimators=5, compute_brier=False)
    assert apparent["model_stats"]["evaluation_mode"] == "apparent"
    assert "this importance is in-sample" in apparent["importance_method"]
    assert not hasattr(ml, "train_test_split")


def test_counterfactual_strength_describes_the_median_per_patient_ratio(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    class _LinearAgeModel:
        def predict(self, X):
            return np.asarray(X)[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
    trained = {"_model": _LinearAgeModel(), "_X_encoded": analysis_frame.copy(), "_analysis_frame": analysis_frame}
    df = make_example_dataset(seed=63, n_patients=40)

    def run(original, value):
        return ml.counterfactual_survival(df, "os_months", "os_event", ["age"], target_feature="age",
                                          original_value=original, counterfactual_value=value, trained_result=trained)

    tripled = run(5.0, 15.0)["scientific_summary"]
    assert tripled["strengths"][1] == (
        "The median per-patient relative risk score (cumulative hazard summed over the training event times) "
        "increases by 200.0% (higher risk)."
    )
    nearly_equal = run(10.0, 10.2)["scientific_summary"]
    assert nearly_equal["strengths"][1].endswith("changes by 2.0% (similar risk).")
    assert "changes by 2.0%." in nearly_equal["headline"]


def test_calibration_reports_the_bins_it_formed() -> None:
    from survival_toolkit.ml_models import compute_calibration_data

    predicted = np.repeat([0.2, 0.5, 0.8], [14, 13, 13])
    result = compute_calibration_data(np.arange(1, 41, dtype=float), np.tile([1, 0], 20), predicted, t=10.0, n_bins=10)
    # Tied predictions collapse the ten requested quantile bins.
    formed = len(result["bins"])
    assert formed < 10 and result["n_bins"] == formed and result["n_bins_requested"] == 10
    summary = result["scientific_summary"]
    assert f"across {formed} bin(s) (10 requested;" in summary["strengths"][0]
    assert {"label": "Bins", "value": formed} in summary["metrics"]


# ── Random Survival Forest memory and settings ──────────────────────────────


@requires_sksurv
def test_very_deep_tree_limits_are_estimated_and_absurd_ones_refused() -> None:
    from survival_toolkit import ml_models as ml

    unlimited = ml.rsf_memory_estimate_bytes(1000, 500, n_estimators=10, min_samples_leaf=6)
    assert ml.rsf_memory_estimate_bytes(1000, 500, n_estimators=10, min_samples_leaf=6, max_depth=1100) == unlimited
    assert ml.rsf_memory_estimate_bytes(1000, 500, n_estimators=10, min_samples_leaf=6, max_depth=2**31 - 1) == unlimited
    df = make_example_dataset(seed=10, n_patients=120)
    fitted = ml.train_random_survival_forest(df, "os_months", "os_event", ["age", "biomarker_score"], n_estimators=5,
                                             max_depth=1100, compute_importance=False, compute_brier=False)
    assert fitted["model_stats"]["max_depth"] == 1100
    with pytest.raises(UserInputError, match="max_depth must be at most 2147483647"):
        ml.train_random_survival_forest(df, "os_months", "os_event", ["age"], n_estimators=5, max_depth=10**30)


def test_rsf_memory_budget_treats_inf_as_no_limit_and_ignores_unusable_values(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    times = np.arange(1.0, 2001.0)
    monkeypatch.setenv("SURVSTUDIO_RSF_MEMORY_BUDGET_GB", "inf")
    assert ml._rsf_memory_budget_bytes() == float("inf")
    ml.check_rsf_memory(times, n_estimators=10**6, min_samples_leaf=1)
    for unusable in ("nan", "-3", "0", "lots"):
        monkeypatch.setenv("SURVSTUDIO_RSF_MEMORY_BUDGET_GB", unusable)
        assert ml._rsf_memory_budget_bytes() == 2e9
    monkeypatch.setenv("SURVSTUDIO_RSF_MEMORY_BUDGET_GB", "1.5")
    assert ml._rsf_memory_budget_bytes() == 1.5e9
    with pytest.raises(ValueError) as refused:
        ml.check_rsf_memory(times, n_estimators=10**6, min_samples_leaf=1)
    # The web app has no min_samples_leaf setting, so that advice is marked as the package's.
    assert "raise min_samples_leaf" not in str(refused.value)
    assert "in the Python package a larger min_samples_leaf" in str(refused.value)


# ── Optimal cutpoint scan ───────────────────────────────────────────────────


def test_cutpoint_scan_refuses_a_numeric_marker_stored_as_text_with_stray_values() -> None:
    from survival_toolkit.ml_models import find_optimal_cutpoint

    df = make_example_dataset(seed=10, n_patients=200).copy()
    marker = df["biomarker_score"].round(2).astype(object)
    marker.loc[df["biomarker_score"].nsmallest(30).index] = "<0.1"
    df["marker_txt"] = marker
    with pytest.raises(UserInputError, match='Marker "marker_txt" looks numeric but contains 30 non-numeric value.*"<0.1"'):
        find_optimal_cutpoint(df, "os_months", "os_event", "marker_txt", event_positive_value=1, permutation_iterations=0)


def test_cutpoint_scan_counts_and_reports_the_marker_values_it_leaves_out() -> None:
    from survival_toolkit.ml_models import find_optimal_cutpoint

    df = make_example_dataset(seed=10, n_patients=200).copy()
    clean = find_optimal_cutpoint(df, "os_months", "os_event", "biomarker_score", permutation_iterations=0)
    assert clean["n_marker_values_non_numeric"] == 0 and clean["n_marker_values_non_finite"] == 0
    assert clean["input_notes"] == []

    # Infinite values of a numeric marker (for example log of 0) are left out and counted.
    with_inf = df.assign(marker=df["biomarker_score"])
    with_inf.loc[with_inf.index[:3], "marker"] = np.inf
    result = find_optimal_cutpoint(with_inf, "os_months", "os_event", "marker", permutation_iterations=0)
    assert result["n_marker_values_non_finite"] == 3
    assert result["n_above_cutpoint"] + result["n_below_cutpoint"] == len(df) - 3
    assert result["input_notes"] == ["3 infinite value(s) in marker were treated as missing and left out of the scan."]

    # A text marker that is mostly, but not clearly, numbers: its text and "inf" entries are left out and counted.
    text = df["biomarker_score"].round(3).astype(str).astype(object)
    text.iloc[:60] = "not measured"
    text.iloc[60:62] = "inf"
    result = find_optimal_cutpoint(df.assign(marker=text), "os_months", "os_event", "marker", permutation_iterations=0)
    assert result["n_marker_values_non_numeric"] == 60 and result["n_marker_values_non_finite"] == 2
    assert result["n_above_cutpoint"] + result["n_below_cutpoint"] == len(df) - 62
    assert np.isfinite(result["optimal_cutpoint"])
    assert any(note.startswith("60 non-numeric value(s) in marker") for note in result["input_notes"])


@pytest.mark.parametrize("iterations", [-1, -5, 2.5, True])
def test_cutpoint_scan_refuses_an_invalid_permutation_count(iterations) -> None:
    from survival_toolkit.ml_models import find_optimal_cutpoint

    df = make_example_dataset(seed=10, n_patients=120)
    with pytest.raises(UserInputError, match="permutation_iterations must be a whole number of at least 0"):
        find_optimal_cutpoint(df, "os_months", "os_event", "biomarker_score", permutation_iterations=iterations)


def test_cutpoint_scan_records_count_marker_groups_and_the_result_counts_risk_groups() -> None:
    from survival_toolkit.ml_models import find_optimal_cutpoint

    rng = np.random.default_rng(1)
    n = 200
    protective = rng.normal(size=n)
    event_time = rng.exponential(8.0 / np.exp(-0.8 * protective))
    censor_time = rng.uniform(0.0, 3.0, size=n)
    df = pd.DataFrame({"t": np.minimum(event_time, censor_time), "e": (event_time <= censor_time).astype(int), "m": protective})
    result = find_optimal_cutpoint(df, "t", "e", "m", event_positive_value=1, permutation_iterations=0)
    # A protective marker: the patients above the cutpoint are the lower-risk group.
    assert result["label_above_cutpoint"] == "Low"
    best = next(record for record in result["scan_data"] if record["cutpoint"] == result["optimal_cutpoint"])
    assert set(best) == {"cutpoint", "statistic", "p_value", "n_above_cutpoint", "n_below_cutpoint"}
    above = int((df["m"] > result["optimal_cutpoint"]).sum())
    assert best["n_above_cutpoint"] == result["n_above_cutpoint"] == above
    assert result["n_low"] == above and result["n_high"] == n - above


# ── Reference-level scoring and median imputation ───────────────────────────


def _holdout_rows(df: pd.DataFrame, features: list[str], n_rows: int) -> list:
    from survival_toolkit import ml_models as ml
    from survival_toolkit.analysis import _cohort_frame

    frame = _cohort_frame(df, "os_months", "os_event", extra_columns=features, drop_missing_extra_columns=False)
    _, eval_positions, mode = ml._split_train_test_positions(frame, "os_event", random_state=42)
    assert mode == "holdout"
    return [frame.attrs["source_row_index"][position] for position in eval_positions[:n_rows]]


def test_missing_categorical_values_only_in_evaluation_rows_are_cautioned(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=260).copy()
    features = ["age", "biomarker_score", "stage"]
    df["stage"] = df["stage"].astype(object)
    df.loc[_holdout_rows(df, features, 6), "stage"] = None
    _patch_fits(monkeypatch, ml)
    result = ml.compare_survival_models(df, "os_months", "os_event", features, categorical_features=["stage"])
    cautions = [text for text in result["scientific_summary"]["cautions"] if "never occurs" in text]
    assert cautions and cautions[0].startswith("6 evaluation row(s) had a categorical level that never occurs")
    assert "missing categorical value" in cautions[0]


@requires_sksurv
def test_median_imputed_numeric_values_are_counted_and_reported(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=21, n_patients=160).copy()
    features = ["age", "biomarker_score", "stage"]
    df.loc[df.index[:5], "age"] = np.nan
    df.loc[df.index[5:7], "biomarker_score"] = np.inf
    expected = {"age": 5, "biomarker_score": 2}

    single = ml.train_gradient_boosted_survival(df, "os_months", "os_event", features, n_estimators=5,
                                                compute_importance=False, compute_brier=False)
    _patch_fits(monkeypatch, ml)
    comparison = ml.compare_survival_models(df, "os_months", "os_event", features)
    cv = ml.cross_validate_survival_models(df, "os_months", "os_event", features, cv_folds=2, cv_repeats=1)
    for result in (single, comparison, cv):
        assert result["imputed_numeric_values"] == expected
        assert any(
            text.startswith("7 missing numeric value(s) were replaced by the median of the training rows before modeling: "
                            "age (5), biomarker_score (2).")
            for text in result["scientific_summary"]["cautions"]
        )


@requires_sksurv
def test_results_name_the_features_coded_as_categorical(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=21, n_patients=160)
    features = ["age", "biomarker_score", "stage"]
    # "stage" is text, so it is reference-coded although the request declares no categorical feature.
    single = ml.train_gradient_boosted_survival(df, "os_months", "os_event", features, categorical_features=[],
                                                n_estimators=5, compute_importance=False, compute_brier=False)
    _patch_fits(monkeypatch, ml)
    comparison = ml.compare_survival_models(df, "os_months", "os_event", features, categorical_features=[])
    cv = ml.cross_validate_survival_models(df, "os_months", "os_event", features, categorical_features=[], cv_folds=2,
                                           cv_repeats=1)
    for result in (single, comparison, cv):
        assert result["categorical_features"] == ["stage"]


@requires_sksurv
def test_splits_keep_the_feature_types_decided_on_the_whole_cohort(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=200).copy()
    code = pd.Series(np.random.default_rng(4).integers(1, 13, size=len(df)).astype(str), index=df.index, dtype=object)
    held_out = set(_holdout_rows(df, ["age"], 43))
    code.loc[sorted(held_out)] = "x"
    code.loc[[index for index in df.index if index not in held_out][:2]] = "x"
    df["code"] = code
    # On the whole cohort "code" is a text feature with 13 levels, 22.5% of them "x", so it is
    # categorical; the training split alone would look like numbers with two stray entries.
    fitted = ml.train_gradient_boosted_survival(df, "os_months", "os_event", ["age", "code"], n_estimators=5,
                                                compute_importance=False, compute_brier=False)
    assert fitted["categorical_features"] == ["code"]
    assert fitted["_feature_encoder"]["categorical_features"] == ["code"]
    _patch_fits(monkeypatch, ml, _fit_evaluate_gbs_split=ml._fit_evaluate_gbs_split)
    comparison = ml.compare_survival_models(df, "os_months", "os_event", ["age", "code"], n_estimators=5)
    assert comparison["errors"] == []
    assert next(row for row in comparison["comparison_table"] if row["model"] == "Gradient Boosted Survival")["c_index"] is not None


@requires_sksurv
def test_a_text_level_only_in_the_evaluation_rows_keeps_the_feature_categorical_in_every_split(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=200).copy()
    features = ["age", "biomarker_score", "grade"]
    grade = pd.Series(np.random.default_rng(6).integers(1, 4, size=len(df)).astype(str), index=df.index, dtype=object)
    grade.loc[_holdout_rows(df, ["age"], 1)] = "Unknown"
    df["grade"] = grade
    # On the whole cohort "grade" is text with a non-numeric level, so it is categorical; the
    # training split alone holds only "1", "2" and "3", which must not make it numeric there.
    comparison = ml.compare_survival_models(df, "os_months", "os_event", features, n_estimators=5)
    assert comparison["errors"] == [] and len(comparison["comparison_table"]) == 4
    assert comparison["categorical_features"] == ["grade"]
    # In cross-validation the "Unknown" row is a training row in some folds and a test row in
    # another; boosting (unlike an unpenalized Cox fit) copes with a one-row level.
    _patch_fits(monkeypatch, ml, _fit_evaluate_gbs_split=ml._fit_evaluate_gbs_split)
    cv = ml.cross_validate_survival_models(df, "os_months", "os_event", features, n_estimators=5, cv_folds=3,
                                           cv_repeats=1)
    assert cv["errors"] == [] and cv["evaluation_mode"] == "repeated_cv"


@requires_sksurv
def test_counterfactual_without_a_fitted_encoder_keeps_the_categorical_target_levels() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=200)
    features = ["age", "biomarker_score", "stage"]
    fitted = ml.train_random_survival_forest(df, "os_months", "os_event", features, categorical_features=["stage"],
                                             n_estimators=10, random_state=1, compute_importance=False, compute_brier=False)
    without_encoder = {key: value for key, value in fitted.items() if key != "_feature_encoder"}

    def run(trained, value):
        return ml.counterfactual_survival(df, "os_months", "os_event", features, categorical_features=["stage"],
                                          target_feature="stage", original_value="I", counterfactual_value=value,
                                          trained_result=trained)["risk_change_pct"]

    # Encoding a one-level scenario frame dropped the stage columns, so every level scored as the reference level.
    assert run(without_encoder, "IV") != pytest.approx(run(without_encoder, "II"))
    assert run(without_encoder, "IV") == pytest.approx(run(fitted, "IV"))


# ── Rankings without a C-index ──────────────────────────────────────────────


def test_comparison_without_any_c_index_names_no_top_model_and_ranks_nothing(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml
    from survival_toolkit.analysis import _cohort_frame

    df = _late_event_cohort(n=40, n_events=4)
    frame = _cohort_frame(df, "time", "event", extra_columns=["x", "z"], drop_missing_extra_columns=False)
    _, eval_positions, mode = ml._split_train_test_positions(frame, "event", random_state=42)
    holdout = frame.iloc[eval_positions]
    # The holdout holds one event that no holdout patient outlives, so no model has a C-index.
    assert mode == "holdout" and not ml._has_comparable_pair(holdout["time"], holdout["event"])

    _patch_fits(monkeypatch, ml, **{name: _c_index_fit() for name in _FIT_FUNCTIONS})
    result = ml.compare_survival_models(df, "time", "event", ["x", "z"])
    assert all(row["c_index"] is None and row["rank"] is None for row in result["comparison_table"])
    assert result["ranking_complete"] is False
    assert result["excluded_models"] == sorted(row["model"] for row in result["comparison_table"])
    summary = result["scientific_summary"]
    assert "top:" not in summary["headline"]
    assert any("no comparable pair" in text and "not ranked" in text for text in summary["cautions"])
    assert not any("top-ranked model was selected" in text for text in summary["cautions"])
    assert {row["Rank"] for row in result["manuscript_tables"]["model_performance_table"]} == {"Not ranked"}


def test_repeated_cv_without_any_complete_model_names_no_cv_selected_model(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    first_fold: dict[str, tuple] = {}

    def _fail_first_fold(train_frame, test_frame, **kwargs):
        key = (len(train_frame), float(train_frame["os_months"].sum()))
        first_fold.setdefault("key", key)
        if key == first_fold["key"]:
            raise ValueError("No non-constant encoded features remain")
        return _fake_fit()(train_frame, test_frame, **kwargs)

    _patch_fits(monkeypatch, ml, **{name: _fail_first_fold for name in _FIT_FUNCTIONS})
    result = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=3,
                                               cv_repeats=1, random_state=11, locked_test_fraction=0.3)
    assert all(row["c_index"] is None and row["rank"] is None for row in result["comparison_table"])
    summary = result["scientific_summary"]
    assert "top:" not in summary["headline"]
    assert not any("CV-selected model (" in text for text in summary["strengths"] + summary["cautions"])
    assert summary["cautions"][0].startswith("No model was scored on every cross-validation fold")
    tables = result["manuscript_tables"]
    assert {row["Rank"] for row in tables["model_performance_table"]} == {"Not ranked"}
    assert not any("CV-selected (rank 1) model" in note for note in tables["table_notes"])
    assert any(note.startswith("No model has a cross-validated C-index") for note in tables["table_notes"])
    assert result["ranking_complete"] is False


def test_repeated_cv_manuscript_ranks_come_from_the_rows() -> None:
    from survival_toolkit.ml_models import build_manuscript_result_tables

    common = {"cv_folds": 5, "cv_repeats": 3, "n_repeats": 3, "evaluation_mode": "repeated_cv"}
    tables = build_manuscript_result_tables({
        "evaluation_mode": "repeated_cv_incomplete",
        "comparison_table": [
            {"model": "DeepSurv", "c_index": 0.71, "rank": 2, **common},
            {"model": "Neural MTLR", "c_index": 0.73, "rank": 1, **common},
            # Left unranked by the comparison, or without a cross-validated C-index.
            {"model": "DeepHit", "c_index": 0.75, "rank": None, **common},
            {"model": "Survival VAE", "c_index": None, "rank": 3, **{**common, "evaluation_mode": "repeated_cv_incomplete"}},
        ],
    })
    assert [row["Rank"] for row in tables["model_performance_table"]] == [2, 1, "Not ranked", "Not ranked"]


@pytest.mark.parametrize("fraction", [0, 0.0, -0.2, float("nan"), float("inf"), 0.6, True, "0.3"])
def test_repeated_cv_refuses_a_locked_test_fraction_outside_the_form_bounds(monkeypatch, fraction) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    _patch_fits(monkeypatch, ml)
    with pytest.raises(UserInputError, match=r"locked_test_fraction must be None \(no locked test set\) or a fraction between 0.05 and 0.5"):
        ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=2,
                                          cv_repeats=1, locked_test_fraction=fraction)


def test_repeated_cv_accepts_the_locked_test_fraction_bounds_and_none(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    _patch_fits(monkeypatch, ml)
    for accepted in (0.05, 0.5, None):
        result = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=2,
                                                   cv_repeats=1, locked_test_fraction=accepted)
        assert result["locked_test_fraction"] == accepted
        assert (result["n_locked_test_patients"] is None) == (accepted is None)


def test_comparisons_flag_missing_brier_metrics_of_the_top_model(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)
    _patch_fits(monkeypatch, ml)  # the fake fits report a C-index but no Brier metrics
    comparison = ml.compare_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"])
    cv = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=2, cv_repeats=1)
    for result in (comparison, cv):
        assert result["comparison_table"][0]["ibs"] is None
        assert any("IBS / Brier Skill Score could not be computed" in text for text in result["scientific_summary"]["cautions"])
    summary = ml._augment_scientific_summary_with_brier(
        {"status": "robust", "headline": "h", "strengths": [], "cautions": [], "next_steps": [], "metrics": []},
        {"ibs": None, "null_ibs": None, "brier_skill_score": None},
    )
    assert summary["status"] == "review" and summary["cautions"]


# ── SHAP ────────────────────────────────────────────────────────────────────


def test_kernel_shap_estimates_every_feature_and_draws_from_its_own_random_state(monkeypatch) -> None:
    shap = pytest.importorskip("shap")
    from survival_toolkit import ml_models as ml

    class _Unsupported:
        def __init__(self, model):
            raise ValueError("Model type not yet supported by TreeExplainer")

    monkeypatch.setattr(ml, "shap", types.SimpleNamespace(TreeExplainer=_Unsupported, KernelExplainer=shap.KernelExplainer))
    seen_states: list[np.ndarray] = []

    class _Linear:
        random_state = 7

        def predict(self, X):
            # What any other code drawing from NumPy's global random state would see meanwhile.
            seen_states.append(np.random.get_state()[1][:8].copy())
            X = np.asarray(X, dtype=float)
            return X @ np.arange(1.0, X.shape[1] + 1.0)

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 12)), columns=[f"f{i:02d}" for i in range(12)])
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    first = ml.compute_shap_values(_Linear(), X)
    # The global state is neither reseeded during the run nor changed after it.
    assert all(np.array_equal(state, before[:8]) for state in seen_states)
    assert np.array_equal(np.random.get_state()[1], before)
    assert first["method"] == "kernel" and first["random_state"] == 7
    # Every one of the 12 features gets an attribution (shap's default l1_reg kept at most 10 per row).
    values = np.array([row["shap_values"] for row in first["shap_summary"]], dtype=float).T
    assert int((np.abs(values) > 1e-9).sum(axis=1).max()) == 12
    second = ml.compute_shap_values(_Linear(), X)
    for one, two in zip(first["shap_summary"], second["shap_summary"], strict=True):
        np.testing.assert_allclose(one["shap_values"], two["shap_values"], rtol=0, atol=0)


def test_shap_tree_explainer_memory_errors_are_not_an_unsupported_model(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    class _OOMTree:
        def __init__(self, model):
            raise MemoryError("tree explainer ran out of memory")

    class _Kernel:
        def __init__(self, fn, background):
            pass

        def shap_values(self, X, **kwargs):
            return np.ones_like(X, dtype=float)

    monkeypatch.setattr(ml, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(ml, "shap", types.SimpleNamespace(TreeExplainer=_OOMTree, KernelExplainer=_Kernel), raising=False)
    model = types.SimpleNamespace(predict=lambda matrix: np.asarray(matrix).sum(axis=1), random_state=3)
    X = pd.DataFrame(np.arange(30.0).reshape(10, 3), columns=list("abc"))
    with pytest.raises(MemoryError):
        ml.compute_shap_values(model, X)

    class _UnsupportedTree:
        def __init__(self, model):
            raise ValueError("Model type not yet supported by TreeExplainer")

    monkeypatch.setattr(ml, "shap", types.SimpleNamespace(TreeExplainer=_UnsupportedTree, KernelExplainer=_Kernel), raising=False)
    assert ml.compute_shap_values(model, X)["method"] == "kernel"

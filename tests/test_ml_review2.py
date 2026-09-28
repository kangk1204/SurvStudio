"""Regression tests for the second review of the ML models (ml_models.py)."""

from __future__ import annotations

import types
import warnings

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

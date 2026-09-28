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

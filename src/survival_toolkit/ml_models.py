"""Machine-learning survival models and cutpoint optimisation utilities.

This module provides penalized and tree-based survival models
(LASSO-Cox, Random Survival Forest, Gradient Boosted Survival),
optimal cutpoint scanning, SHAP explanations, and partial-dependence
computation.  Every public function returns a plain ``dict`` that is
JSON-serialisable so it can be served directly by the FastAPI backend.

Optional heavy dependencies (scikit-survival, shap) are imported lazily
behind availability flags so the rest of the toolkit keeps working when
they are not installed.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from survival_toolkit.prediction_guard import guarded_prediction_inputs
from scipy import stats
from scipy.linalg import qr as scipy_qr
from statsmodels.duration.hazard_regression import PHReg

from survival_toolkit.analysis import (
    ThresholdLogrankScan,
    duplicate_identifier_caution,
    fit_phreg,
    _cohort_frame,
    _harrell_c_index,
    _safe_float,
)
from survival_toolkit.encoding import (
    _stored_level_values,
    canonical_category_values,
    coerce_feature_subset,
    fit_feature_encoder as _fit_shared_feature_encoder,
    numeric_text_contamination,
    ordered_category_values as _ordered_category_values,
    reject_numeric_text_features,
    transform_feature_encoder as _transform_shared_feature_encoder,
)
from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.errors import InternalAnalysisError, must_propagate, user_input_boundary
from survival_toolkit.evaluation import (
    DEFAULT_HOLDOUT_FRACTION,
    evaluation_split_fingerprint,
    locked_test_split,
    prediction_block,
    stratified_holdout_indices,
)
from survival_toolkit.evaluation import metric_name_for_evaluation as _metric_name_for_evaluation

try:
    from sklearn.model_selection import StratifiedKFold

    _SKLEARN_AVAILABLE = True
except ImportError:
    StratifiedKFold = None
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from sksurv.ensemble import (
        GradientBoostingSurvivalAnalysis,
        RandomSurvivalForest,
    )
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    SKSURV_AVAILABLE = True
except ImportError:
    GradientBoostingSurvivalAnalysis = None
    RandomSurvivalForest = None
    CoxnetSurvivalAnalysis = None
    concordance_index_censored = None
    SKSURV_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TREE_N_JOBS = -1
# Gradient boosting relies on shallow base learners; "auto" (None) resolves to
# this depth instead of fully grown trees.
_GBS_DEFAULT_MAX_DEPTH = 3
# Seeds derived as seed + offset wrap into the range numpy and scikit-learn accept.
_SEED_MODULUS = 2**32


def _derived_seed(base_seed: int, offset: int = 0) -> int:
    """``base_seed + offset`` wrapped into [0, 2**32), as in the deep-learning models.

    Equal to the plain sum whenever it does not overflow, so ordinary seeds keep their fold
    assignments and training seeds.
    """
    return int((int(base_seed) + int(offset)) % _SEED_MODULUS)


def _resolve_gbs_max_depth(max_depth: int | None) -> int:
    return _GBS_DEFAULT_MAX_DEPTH if max_depth is None else int(max_depth)
_SKLEARN_INSTALL_MSG = (
    "scikit-learn is required for ML model splitting and time-dependent importance. "
    "Install with: pip install 'survstudio[ml]'"
)


def _prepare_sksurv_data(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
) -> np.ndarray:
    """Convert time/event columns into the structured array that *sksurv*
    expects: ``dtype=[('event', bool), ('time', float)]``.
    """
    event_values = df[event_column].to_numpy(dtype=bool)
    time_values = df[time_column].to_numpy(dtype=float)
    y = np.empty(len(df), dtype=[("event", bool), ("time", float)])
    y["event"] = event_values
    y["time"] = time_values
    return y


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(_SKLEARN_INSTALL_MSG)


# scikit-learn stores a tree depth as a C integer and uses this value for "no limit".
_MAX_TREE_DEPTH = 2**31 - 1


def _validate_max_depth(max_depth: Any) -> int | None:
    """``max_depth`` of a tree model: None (automatic) or a whole number of at least 1."""
    if max_depth is None:
        return None
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or int(max_depth) < 1:
        raise ValueError(
            f"max_depth must be a whole number of at least 1 (or empty for the automatic depth); got {max_depth!r}."
        )
    if int(max_depth) > _MAX_TREE_DEPTH:
        raise ValueError(
            f"max_depth must be at most {_MAX_TREE_DEPTH} (or empty for the automatic depth); got {max_depth!r}."
        )
    return int(max_depth)


# The web form's bounds for the share of the cohort reserved as a locked test set.
_LOCKED_TEST_FRACTION_BOUNDS = (0.05, 0.5)


def _validated_locked_test_fraction(value: Any) -> float | None:
    """``None`` (no locked test set) or a fraction within the web form's bounds, as for the deep models.

    0, a negative value, or NaN used to leave the run without a locked test set silently.
    """
    if value is None:
        return None
    low, high = _LOCKED_TEST_FRACTION_BOUNDS
    valid_type = not isinstance(value, bool) and isinstance(value, (int, np.integer, float, np.floating))
    if not valid_type or not (low <= float(value) <= high):
        raise ValueError(
            f"locked_test_fraction must be None (no locked test set) or a fraction between {low} and {high} "
            f"(got {value!r})."
        )
    return float(value)


# A fitted Random Survival Forest keeps, in every node of every tree, the survival and
# cumulative hazard functions at each distinct training time (2 x 8 bytes), so its size
# grows with trees x nodes x distinct times, roughly quadratically in the training rows.
_RSF_MEMORY_BUDGET_ENV_VAR = "SURVSTUDIO_RSF_MEMORY_BUDGET_GB"
_RSF_DEFAULT_MEMORY_BUDGET_BYTES = 2_000_000_000
_RSF_BYTES_PER_NODE_AND_TIME = 16
# While predicting, each tree running on a worker thread materialises a
# (rows x distinct training times x 2) array (and, for risk scores, a copy of its event-time
# columns); forest predictions run in row chunks of at most this many bytes of such arrays.
_RSF_PREDICTION_CHUNK_BYTES = 256 * 1024**2


def _rsf_memory_budget_bytes() -> float:
    """The forest memory budget in bytes: the environment variable in GB ("inf" for no limit), else 2 GB."""
    try:
        configured = float(os.environ.get(_RSF_MEMORY_BUDGET_ENV_VAR, "") or 0.0)
    except ValueError:
        configured = 0.0
    if configured == math.inf:
        return math.inf
    if not math.isfinite(configured) or configured <= 0.0:
        return float(_RSF_DEFAULT_MEMORY_BUDGET_BYTES)
    return configured * 1e9


def rsf_memory_estimate_bytes(
    n_rows: int,
    n_unique_times: int,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None = None,
) -> float:
    """Approximate size of a fitted scikit-survival Random Survival Forest.

    A fully grown tree on a bootstrap sample has about ``n_rows / min_samples_leaf`` nodes
    (measured 0.93-0.98 x for leaves of 3-15 rows; about 0.4 x ``n_rows`` for leaves of one
    row, where the default ``min_samples_split=6`` stops the splitting), and at most
    ``2**(max_depth + 1) - 1`` nodes with a depth limit.
    """
    nodes = float(n_rows) / max(float(min_samples_leaf), 2.5)
    if max_depth is not None:
        # Computed in floating point: 2**(max_depth + 1) of a very deep limit exceeds a float.
        nodes = min(nodes, math.ldexp(1.0, min(int(max_depth) + 1, 1023)) - 1.0)
    return float(n_estimators) * max(nodes, 1.0) * float(n_unique_times) * _RSF_BYTES_PER_NODE_AND_TIME


def check_rsf_memory(
    times: np.ndarray,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None = None,
    model_label: str = "Random Survival Forest",
) -> None:
    """Refuse a Random Survival Forest whose fitted trees would not fit the memory budget."""
    time_values = np.asarray(times, dtype=float).reshape(-1)
    n_rows = int(time_values.shape[0])
    n_unique = int(np.unique(time_values).size)
    estimate = rsf_memory_estimate_bytes(
        n_rows,
        n_unique,
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_depth=max_depth,
    )
    budget = _rsf_memory_budget_bytes()
    if estimate > budget:
        raise ValueError(
            f"{model_label} would need about {estimate / 1e9:.3g} GB of memory: {int(n_estimators)} trees on "
            f"{n_rows} training rows with {n_unique} distinct follow-up times (limit {budget / 1e9:.3g} GB). "
            "Use fewer trees, set a maximum tree depth, or round the follow-up times (for example to whole days "
            "or months); in the Python package a larger min_samples_leaf also shrinks the forest. "
            f"{_RSF_MEMORY_BUDGET_ENV_VAR} changes the limit."
        )


def _is_random_survival_forest(model: Any) -> bool:
    return RandomSurvivalForest is not None and isinstance(model, RandomSurvivalForest)


def _prediction_row_chunk(model: Any) -> int | None:
    """Rows per predict call for a Random Survival Forest (None: predict all rows at once)."""
    if not _is_random_survival_forest(model):
        return None
    unique_times = getattr(model, "unique_times_", None)
    if unique_times is None:
        return None
    n_unique = max(int(np.asarray(unique_times).size), 1)
    is_event_time = getattr(model, "is_event_time_", None)
    n_event_times = int(np.count_nonzero(is_event_time)) if is_event_time is not None else n_unique
    n_trees = max(len(getattr(model, "estimators_", []) or []), 1)
    try:
        from joblib import effective_n_jobs

        threads = int(effective_n_jobs(getattr(model, "n_jobs", None)))
    except (ImportError, ValueError):
        threads = os.cpu_count() or 1
    threads = max(1, min(threads, n_trees))
    # Per row: for each concurrent tree one (distinct times x 2) float64 array and, when
    # predicting risk scores, a copy of its event-time columns; plus the output row (a
    # survival curve when predicting survival functions). No floor: the byte bound holds.
    per_row = (threads * (2 * n_unique + n_event_times) + n_unique) * 8
    return max(1, int(_RSF_PREDICTION_CHUNK_BYTES // per_row))


def _predict_risk_scores(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """``model.predict`` on every row, in row chunks for a Random Survival Forest.

    Each row's score does not depend on the other rows, so chunking changes only the
    memory a forest's prediction takes, not the scores.
    """
    X_array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    chunk = _prediction_row_chunk(model)
    if chunk is None or X_array.shape[0] <= chunk:
        return np.asarray(model.predict(X_array), dtype=float)
    parts = []
    for start in range(0, X_array.shape[0], chunk):
        raise_if_cancelled()
        parts.append(np.asarray(model.predict(X_array[start : start + chunk]), dtype=float))
    return np.concatenate(parts)


def _validate_model_feature_columns(
    features: Sequence[str],
    *,
    time_column: str,
    event_column: str,
) -> None:
    offenders = sorted({str(feature) for feature in features if str(feature) in {str(time_column), str(event_column)}})
    if offenders:
        raise ValueError(
            "Survival outcome columns cannot be used as ML model features: "
            + ", ".join(offenders)
            + "."
        )


def _fit_feature_encoder(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _fit_shared_feature_encoder(df, features, categorical_features)


def _transform_feature_encoder(
    df: pd.DataFrame,
    encoder: dict[str, Any],
) -> pd.DataFrame:
    return _transform_shared_feature_encoder(df, encoder, output="dataframe")


def _sksurv_c_index(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
) -> float | None:
    """Compute Harrell's C-index via *sksurv* if available, else fall back
    to the pure-Python implementation from ``analysis.py``.

    None when no pair is comparable, or when a risk score or time is not finite: such a
    value has no rank, and scoring it anyway gives a meaningless C-index.
    """
    events = y_true["event"].astype(bool)
    times = y_true["time"].astype(float)
    risk = np.asarray(risk_scores, dtype=float).reshape(-1)
    if not (np.isfinite(risk).all() and np.isfinite(times).all()):
        return None

    if SKSURV_AVAILABLE:
        try:
            # Exact score ties match the shared estimator and R. sksurv's default
            # 1e-8 tolerance otherwise changes the result when this extra is installed.
            c_index, _, _, _, _ = concordance_index_censored(events, times, risk, tied_tol=0.0)
            return _safe_float(c_index)
        except (ValueError, ZeroDivisionError):
            # No comparable pairs or all censored; fall through to the shared estimator.
            pass

    return _harrell_c_index(
        times,
        events.astype(int).astype(float),
        risk,
    )


def _has_comparable_pair(times: Any, events: Any) -> bool:
    """Whether Harrell's C-index is defined on these patients at all, for any model.

    A pair is comparable when an event is followed by a longer follow-up (or by a censoring
    at the same time); without one, every model's C-index is undefined.
    """
    time_values = np.asarray(times, dtype=float).reshape(-1)
    event_values = np.asarray(events, dtype=float).reshape(-1)
    return _harrell_c_index(time_values, event_values, np.zeros(time_values.size, dtype=float)) is not None


_PERMUTATION_IMPORTANCE_MAX_ROWS = 300
PERMUTATION_IMPORTANCE_METHOD = (
    "Permutation importance: mean drop in Harrell's C-index on the evaluation rows when a raw feature is "
    "shuffled (all one-hot columns of a categorical feature are shuffled together), averaged over 5 shuffles "
    "(3 with more than 20 features, 2 with more than 60); an evaluation set of more than "
    f"{_PERMUTATION_IMPORTANCE_MAX_ROWS} rows is scored on a random subsample of {_PERMUTATION_IMPORTANCE_MAX_ROWS} rows."
)


def _permutation_importance_method(evaluation_mode: str) -> str:
    """How the permutation importance was computed; in-sample without a holdout."""
    if evaluation_mode == "holdout":
        return PERMUTATION_IMPORTANCE_METHOD
    return (
        PERMUTATION_IMPORTANCE_METHOD
        + " The cohort was too small for a holdout, so the evaluation rows are the rows the model was fitted on and "
        "this importance is in-sample."
    )


def encoded_feature_groups(encoded_columns: Sequence[str], feature_encoder: dict[str, Any] | None) -> dict[str, list[int]]:
    """Positions of the encoded columns that belong to each raw input feature.

    Columns the encoder does not describe (for example hand-built matrices) form their
    own group, so every encoded column is covered exactly once.
    """
    column_positions = {str(name): index for index, name in enumerate(encoded_columns)}
    groups: dict[str, list[int]] = {}
    covered: set[int] = set()
    if isinstance(feature_encoder, dict):
        for column in feature_encoder.get("numeric_features", []):
            position = column_positions.get(str(column))
            if position is not None:
                groups.setdefault(str(column), []).append(position)
                covered.add(position)
        mappings = feature_encoder.get("categorical_mappings", {}) or {}
        for column in feature_encoder.get("categorical_features", []):
            meta = mappings.get(column, {}) or {}
            level_columns = meta.get("level_columns") or {}
            names = [level_columns.get(level, f"{column}_{level}") for level in meta.get("retained_levels", [])]
            names.extend(name for name in (meta.get("unknown_column"), meta.get("missing_column")) if name)
            positions = [column_positions[str(name)] for name in names if str(name) in column_positions]
            if positions:
                groups[str(column)] = positions
                covered.update(positions)
    for name, position in column_positions.items():
        if position not in covered:
            groups.setdefault(name, []).append(position)
    return groups


def _grouped_permutation_importance(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    feature_encoder: dict[str, Any] | None,
    *,
    random_state: int,
) -> list[dict[str, Any]]:
    """Permutation importance per raw feature of a fitted survival model on the evaluation rows
    (out of sample when they are a holdout).

    Shuffling one-hot columns one at a time would create impossible rows (two levels at
    once) and split a feature's importance across its levels, so each raw feature's
    columns are permuted together with one shared row permutation.
    """
    matrix = np.asarray(X_eval.to_numpy(dtype=float), dtype=float)
    y_values = np.asarray(y_eval)
    rng = np.random.default_rng(int(random_state))
    if matrix.shape[0] > _PERMUTATION_IMPORTANCE_MAX_ROWS:
        rows = np.sort(rng.choice(matrix.shape[0], size=_PERMUTATION_IMPORTANCE_MAX_ROWS, replace=False))
        matrix = matrix[rows]
        y_values = y_values[rows]
    groups = encoded_feature_groups(list(X_eval.columns), feature_encoder)
    n_repeats = 5 if len(groups) <= 20 else (3 if len(groups) <= 60 else 2)
    baseline = _sksurv_c_index(y_values, _predict_risk_scores(model, matrix))
    records: list[dict[str, Any]] = []
    for feature, positions in groups.items():
        raise_if_cancelled()
        drops: list[float] = []
        if baseline is not None and matrix.shape[0] > 1:
            for _ in range(n_repeats):
                permutation = rng.permutation(matrix.shape[0])
                shuffled = matrix.copy()
                shuffled[:, positions] = matrix[permutation][:, positions]
                score = _sksurv_c_index(y_values, _predict_risk_scores(model, shuffled))
                if score is not None:
                    drops.append(float(baseline) - float(score))
        records.append(
            {
                "feature": feature,
                "importance": _safe_float(float(np.mean(drops))) if drops else None,
                "importance_std": _safe_float(float(np.std(drops, ddof=1))) if len(drops) > 1 else None,
                "encoded_columns": [str(X_eval.columns[position]) for position in positions],
            }
        )
    records.sort(key=lambda row: row["importance"] if row["importance"] is not None else float("-inf"), reverse=True)
    return records


def _representative_subsample_indices(values: np.ndarray, n_samples: int) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    total_n = int(values_arr.shape[0])
    if total_n <= n_samples:
        return np.arange(total_n, dtype=int)

    order = np.argsort(values_arr, kind="mergesort")
    target_positions = np.linspace(0, total_n - 1, num=n_samples)
    chosen = order[np.round(target_positions).astype(int)]
    chosen = np.unique(chosen)
    if chosen.shape[0] < n_samples:
        seen = set(int(idx) for idx in chosen.tolist())
        for idx in order:
            idx_int = int(idx)
            if idx_int in seen:
                continue
            chosen = np.append(chosen, idx_int)
            seen.add(idx_int)
            if chosen.shape[0] >= n_samples:
                break
    return np.sort(chosen[:n_samples].astype(int))


def _scientific_summary_ml(
    *,
    model_name: str,
    c_index: float | None,
    n_patients: int,
    n_events: int,
    n_features: int | None,
    evaluation_mode: str = "apparent",
    n_evaluation_patients: int | None = None,
    n_evaluation_events: int | None = None,
    n_fit_patients: int | None = None,
    n_fit_events: int | None = None,
    extra_strengths: Sequence[str] | None = None,
    extra_cautions: Sequence[str] | None = None,
    counts_are_fold_means: bool = False,
) -> dict[str, Any]:
    """Generate an insight-board payload (headline, strengths, cautions,
    next_steps, metrics, status) following the same shape as
    ``_km_scientific_summary`` in *analysis.py*.

    With ``counts_are_fold_means`` the fitting and evaluation counts are the mean sizes of
    the cross-validation training and test folds, and the text says so.
    """
    metric_name = _metric_name_for_evaluation(evaluation_mode)
    eval_n = int(n_evaluation_patients if n_evaluation_patients is not None else n_patients)
    eval_events = int(n_evaluation_events if n_evaluation_events is not None else n_events)
    fit_n = int(n_fit_patients if n_fit_patients is not None else n_patients)
    fit_events = int(n_fit_events if n_fit_events is not None else n_events)
    feature_text = "an unknown number of" if n_features is None else str(int(n_features))
    if counts_are_fold_means:
        strengths: list[str] = [
            f"{model_name} was trained with {feature_text} feature(s) on training folds of about {fit_n} patients "
            f"({fit_events} events) each.",
            f"{metric_name} was estimated on test folds of about {eval_n} patients ({eval_events} events) each.",
        ]
    else:
        strengths = [
            f"{model_name} was trained with {feature_text} feature(s) on {fit_n} patients ({fit_events} events).",
            f"{metric_name} was estimated on {eval_n} patients ({eval_events} events).",
        ]
    cautions: list[str] = []
    next_steps: list[str] = []

    if extra_strengths:
        strengths.extend([item for item in extra_strengths if item])
    if extra_cautions:
        cautions.extend([item for item in extra_cautions if item])

    epv = fit_events / max(int(n_features or 0), 1)
    if n_features is not None and epv < 10:
        cautions.append(
            f"Event-to-feature ratio is {epv:.1f}; screening models can overfit when events are sparse relative to selected features."
        )
        next_steps.append(
            "Reduce the feature set or use repeated resampling before treating this screening result as stable."
        )

    if fit_events < 20:
        cautions.append("Fewer than 20 total events limits model reliability.")

    if evaluation_mode == "holdout":
        strengths.append("Discrimination was estimated on a deterministic holdout split.")
        cautions.append(
            "A deterministic holdout scores one split, and the point estimate alone does not show its uncertainty; "
            "judge it by the bootstrap intervals over the test patients of a model comparison or by repeated cross-validation."
        )
    elif evaluation_mode == "repeated_cv":
        strengths.append("Discrimination was estimated with repeated stratified cross-validation.")
    elif evaluation_mode == "repeated_cv_incomplete":
        strengths.append("Discrimination was estimated with repeated stratified cross-validation.")
        cautions.append(
            "One or more repeated-CV folds failed or fell back, so the reported mean excludes incomplete fold-level estimates."
        )
    else:
        cautions.append(
            "Discrimination was estimated on the training cohort because a stable holdout split was not feasible; this apparent C-index is optimistic."
        )

    if c_index is not None:
        strengths.append(f"{metric_name} = {c_index:.3f}.")
        if c_index >= 0.70:
            strengths.append(
                "C-index is well above chance-level ranking (0.50), which can be useful for screening if independent validation agrees."
            )
        if c_index < 0.60:
            cautions.append(f"{metric_name} is below 0.60, so discrimination is limited.")
        if c_index < 0.55:
            cautions.append("C-index is close to chance-level ranking (0.50).")

    # Headline ("an apparent C-index", "a holdout C-index", "a repeated-CV mean C-index")
    metric_phrase = metric_name[:1].lower() + metric_name[1:]
    if c_index is not None:
        article = "an" if metric_phrase[:1] in "aeiou" else "a"
        headline = f"{model_name} estimated {article} {metric_phrase} of {c_index:.3f} on the current evaluation path."
    else:
        headline = f"{model_name} training completed but the {metric_phrase} could not be computed."

    next_steps.append(
        "Validate on an independent cohort or via cross-validation before drawing clinical conclusions."
    )

    status = "robust"
    if cautions:
        status = "review"
    if (c_index is not None and c_index < 0.55) or fit_events < 10:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": n_patients},
            {"label": "Events", "value": n_events},
            {"label": "Mean training-fold patients" if counts_are_fold_means else "Training patients", "value": fit_n},
            {"label": "Mean training-fold events" if counts_are_fold_means else "Training events", "value": fit_events},
            {"label": "Features", "value": n_features},
            {"label": metric_name, "value": _safe_float(c_index)},
            {"label": "Evaluation mode", "value": evaluation_mode},
        ],
    }


def _manuscript_validation_strategy_label(evaluation_mode: str) -> str:
    if evaluation_mode == "holdout":
        return "Deterministic holdout"
    if evaluation_mode == "apparent":
        return "Apparent (resubstitution)"
    if evaluation_mode == "holdout_fallback_apparent":
        return "Apparent fallback after holdout failure"
    if evaluation_mode == "mixed_holdout_apparent":
        return "Mixed holdout/apparent"
    if evaluation_mode == "repeated_cv":
        return "Repeated stratified CV"
    if evaluation_mode == "repeated_cv_incomplete":
        return "Repeated stratified CV (incomplete)"
    return str(evaluation_mode or "unknown").replace("_", " ")


def _summarize_repeated_cv_rows(
    model_rows: Sequence[dict[str, Any]],
    *,
    train_n_key: str = "train_n",
    test_n_key: str = "test_n",
    train_events_key: str | None = "train_events",
    test_events_key: str | None = "test_events",
    n_features_key: str = "n_features",
) -> dict[str, Any]:
    def _mean_of(row_items: list[dict[str, Any]], key: str | None) -> float | None:
        if key is None:
            return None
        values = [float(item[key]) for item in row_items if item.get(key) is not None]
        return float(np.mean(values)) if values else None

    repeat_groups: dict[int, list[dict[str, Any]]] = {}
    for row in model_rows:
        repeat = int(row.get("repeat", 1))
        repeat_groups.setdefault(repeat, []).append(row)

    repeat_results: list[dict[str, Any]] = []
    for repeat in sorted(repeat_groups):
        rows = repeat_groups[repeat]
        c_values = [float(item["c_index"]) for item in rows if item.get("c_index") is not None]
        if not c_values:
            continue
        train_times = [float(item["training_time_ms"]) for item in rows if item.get("training_time_ms") is not None]
        ibs_values = [float(item["ibs"]) for item in rows if item.get("ibs") is not None]
        null_ibs_values = [float(item["null_ibs"]) for item in rows if item.get("null_ibs") is not None]
        # Pooled like the top-level score (1 - mean IBS / mean null IBS over the same folds),
        # not an average of per-fold ratios, so repeat rows agree with the aggregate.
        paired_brier = [
            (float(item["ibs"]), float(item["null_ibs"]))
            for item in rows
            if item.get("ibs") is not None and item.get("null_ibs") is not None
        ]
        repeat_bss: float | None = None
        if paired_brier:
            mean_repeat_null = float(np.mean([pair[1] for pair in paired_brier]))
            if mean_repeat_null > 0.0:
                repeat_bss = 1.0 - float(np.mean([pair[0] for pair in paired_brier])) / mean_repeat_null
        repeat_results.append({
            "repeat": repeat,
            "c_index": float(np.mean(c_values)),
            # A single scored fold has no spread to report.
            "c_index_std": float(np.std(c_values, ddof=1)) if len(c_values) > 1 else None,
            "c_index_median": float(np.median(c_values)),
            "ibs": float(np.mean(ibs_values)) if ibs_values else None,
            "null_ibs": float(np.mean(null_ibs_values)) if null_ibs_values else None,
            "brier_skill_score": _safe_float(repeat_bss),
            "training_time_ms": float(np.mean(train_times)) if train_times else None,
            "n_folds": len(c_values),
            "n_features": int(round(np.mean([float(item[n_features_key]) for item in rows if item.get(n_features_key) is not None]))),
            "train_n": int(round(_mean_of(rows, train_n_key))) if _mean_of(rows, train_n_key) is not None else None,
            "test_n": int(round(_mean_of(rows, test_n_key))) if _mean_of(rows, test_n_key) is not None else None,
            "train_events": int(round(_mean_of(rows, train_events_key))) if _mean_of(rows, train_events_key) is not None else None,
            "test_events": int(round(_mean_of(rows, test_events_key))) if _mean_of(rows, test_events_key) is not None else None,
        })

    if not repeat_results:
        raise ValueError("No repeated-CV repeat summaries could be computed.")

    # Fold-level estimates carry the real variability; the SD of repeat means
    # only reflects partition noise and collapses to zero with one repeat.
    fold_c_values = np.array(
        [float(item["c_index"]) for item in model_rows if item.get("c_index") is not None],
        dtype=float,
    )
    repeat_c_values = np.array([float(item["c_index"]) for item in repeat_results], dtype=float)
    mean_c = float(np.mean(fold_c_values))
    std_c = float(np.std(fold_c_values, ddof=1)) if len(fold_c_values) > 1 else None
    repeat_std_c = float(np.std(repeat_c_values, ddof=1)) if len(repeat_c_values) > 1 else None
    interval_lower: float | None = None
    interval_upper: float | None = None
    if len(fold_c_values) > 1:
        interval_lower = float(np.quantile(fold_c_values, 0.025))
        interval_upper = float(np.quantile(fold_c_values, 0.975))

    def _mean_repeat_field(field: str) -> int | None:
        values = [float(item[field]) for item in repeat_results if item.get(field) is not None]
        if not values:
            return None
        return int(round(float(np.mean(values))))

    # Brier metrics: only report them when every counted fold produced them,
    # and pool the skill score so it agrees with the mean IBS columns.
    brier_rows = [
        item
        for item in model_rows
        if item.get("c_index") is not None and item.get("ibs") is not None and item.get("null_ibs") is not None
    ]
    mean_ibs: float | None = None
    mean_null_ibs: float | None = None
    pooled_bss: float | None = None
    if brier_rows and len(brier_rows) == len(fold_c_values):
        mean_ibs = float(np.mean([float(item["ibs"]) for item in brier_rows]))
        mean_null_ibs = float(np.mean([float(item["null_ibs"]) for item in brier_rows]))
        if mean_null_ibs > 0.0:
            pooled_bss = 1.0 - mean_ibs / mean_null_ibs

    return {
        "repeat_results": repeat_results,
        "c_index": mean_c,
        "c_index_std": _safe_float(std_c),
        "c_index_std_label": "SD across fold-level estimates",
        "c_index_repeat_std": _safe_float(repeat_std_c),
        "c_index_median": float(np.median(fold_c_values)),
        "c_index_interval_lower": _safe_float(interval_lower),
        "c_index_interval_upper": _safe_float(interval_upper),
        "c_index_interval_label": "Fold-level 2.5th-97.5th percentile range",
        "ibs": _safe_float(mean_ibs),
        "null_ibs": _safe_float(mean_null_ibs),
        "brier_skill_score": _safe_float(pooled_bss),
        "brier_evaluations": int(len(brier_rows)),
        "n_repeats": int(len(repeat_results)),
        "n_evaluations": int(sum(item["n_folds"] for item in repeat_results)),
        "training_time_ms": float(
            np.mean([float(item["training_time_ms"]) for item in repeat_results if item["training_time_ms"] is not None])
        ),
        "n_features": _mean_repeat_field("n_features"),
        "train_n": _mean_repeat_field("train_n"),
        "test_n": _mean_repeat_field("test_n"),
        "train_events": _mean_repeat_field("train_events"),
        "test_events": _mean_repeat_field("test_events"),
    }


def repeated_cv_row_fields(summary: dict[str, Any] | None, *, incomplete: bool) -> dict[str, Any]:
    """Comparison-table fields shared by the ML and deep-learning repeated-CV results.

    Aggregate metrics are withheld (None) when any fold failed or fell back, so an
    incomplete run can never be ranked as if it were complete.
    """

    def _metric(key: str) -> float | None:
        return None if incomplete or summary is None else _safe_float(summary.get(key))

    def _count(key: str) -> int | None:
        return None if summary is None or summary.get(key) is None else int(summary[key])

    return {
        "c_index": _metric("c_index"),
        "c_index_std": _metric("c_index_std"),
        "c_index_std_label": None if summary is None else summary.get("c_index_std_label"),
        "c_index_repeat_std": _metric("c_index_repeat_std"),
        "c_index_median": _metric("c_index_median"),
        "c_index_interval_lower": _metric("c_index_interval_lower"),
        "c_index_interval_upper": _metric("c_index_interval_upper"),
        "c_index_interval_label": None if summary is None else summary.get("c_index_interval_label"),
        "ibs": _metric("ibs"),
        "null_ibs": _metric("null_ibs"),
        "brier_skill_score": _metric("brier_skill_score"),
        "n_features": _count("n_features"),
        "training_time_ms": None if summary is None else _safe_float(summary.get("training_time_ms")),
        "n_repeats": 0 if summary is None else int(summary["n_repeats"]),
        "evaluation_mode": "repeated_cv_incomplete" if incomplete else "repeated_cv",
        "repeat_results": [] if summary is None else summary["repeat_results"],
        "training_samples": _count("train_n"),
        "evaluation_samples": _count("test_n"),
        "train_events": _count("train_events"),
        "test_events": _count("test_events"),
    }


def _time_integral_mean(values: np.ndarray, times: np.ndarray) -> float:
    values_arr = np.asarray(values, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    if values_arr.size == 0 or times_arr.size == 0:
        return 0.0
    if values_arr.size == 1:
        return float(values_arr[0])
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        area = float(trapezoid(values_arr, times_arr))
    else:
        area = float(np.sum(np.diff(times_arr) * (values_arr[:-1] + values_arr[1:]) * 0.5))
    range_ = float(times_arr[-1] - times_arr[0])
    if range_ <= 0.0:
        return float(np.mean(values_arr))
    return area / range_


def _step_survival_lookup(event_times: np.ndarray, survival: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    event_times_arr = np.asarray(event_times, dtype=float).reshape(-1)
    survival_arr = np.asarray(survival, dtype=float).reshape(-1)
    query_arr = np.asarray(query_times, dtype=float).reshape(-1)
    output = np.ones_like(query_arr, dtype=float)
    if event_times_arr.size == 0:
        return output
    indices = np.searchsorted(event_times_arr, query_arr, side="right") - 1
    valid = indices >= 0
    output[valid] = survival_arr[indices[valid]]
    return np.clip(output, 0.0, 1.0)


def _step_function_matrix(step_functions: Any, eval_times: np.ndarray) -> np.ndarray:
    eval_times_arr = np.asarray(eval_times, dtype=float).reshape(-1)
    functions_arr = np.asarray(step_functions, dtype=object).reshape(-1)
    matrix = np.empty((functions_arr.shape[0], eval_times_arr.shape[0]), dtype=float)
    for row_idx, fn in enumerate(functions_arr):
        values = np.asarray(fn(eval_times_arr), dtype=float).reshape(-1)
        if values.shape[0] != eval_times_arr.shape[0]:
            raise ValueError("Survival step function returned an unexpected number of evaluation points.")
        matrix[row_idx, :] = np.clip(values, 0.0, 1.0)
    return matrix


def _sksurv_step_positions(step_times: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    """Positions a scikit-survival ``StepFunction`` on ``step_times`` reads at ``eval_times``.

    Same rule and domain check as ``StepFunction.__call__`` with its default domain
    ``[0, step_times[-1]]``: the value at ``t`` is the one of the last step time at or
    before ``t``, and the first value before the first step time.
    """
    grid = np.asarray(step_times, dtype=float).reshape(-1)
    query = np.atleast_1d(np.asarray(eval_times, dtype=float)).reshape(-1)
    if query.size == 0:
        return np.zeros(0, dtype=int)
    if not np.isfinite(query).all():
        raise ValueError("x must be finite")
    if np.min(query) < 0.0 or np.max(query) > grid[-1]:
        raise ValueError(f"x must be within [0.000000; {grid[-1]:f}]")
    query = np.clip(query, a_min=grid[0], a_max=None)
    positions = np.searchsorted(grid, query, side="left")
    not_exact = grid[positions] != query
    positions[not_exact] -= 1
    return positions


def _breslow_baseline_survival_function(model: Any, alpha: float | None) -> Any:
    """Breslow baseline survival step function of a fitted Cox-type scikit-survival model.

    Gradient boosting with the Cox loss and Coxnet predict ``S0(t) ** exp(risk)``; None for
    other models or when the fitted model has no baseline.
    """
    getter = getattr(model, "_get_baseline_model", None)
    if not callable(getter):
        return None
    is_coxnet = CoxnetSurvivalAnalysis is not None and isinstance(model, CoxnetSurvivalAnalysis)
    is_boosting = GradientBoostingSurvivalAnalysis is not None and isinstance(model, GradientBoostingSurvivalAnalysis)
    if not (is_coxnet or is_boosting):
        return None
    try:
        baseline_model = getter(alpha) if is_coxnet else getter()
    except (TypeError, ValueError):
        return None
    baseline_survival = getattr(baseline_model, "baseline_survival_", None)
    if getattr(baseline_survival, "x", None) is None or getattr(baseline_survival, "y", None) is None:
        return None
    return baseline_survival


def _sksurv_survival_predictor(model: Any, X: pd.DataFrame, *, alpha: float | None = None) -> Any:
    """Predicted survival of every row of ``X`` at requested times, as an (n_rows, n_times) matrix.

    Only the requested times are kept, never a curve over every training time per row:
    Cox-type models raise the Breslow baseline at those times to ``exp(risk)`` (as their
    ``predict_survival_function`` does), and a Random Survival Forest predicts in row chunks.
    """
    X_array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    baseline_survival = _breslow_baseline_survival_function(model, alpha)
    if baseline_survival is not None:
        if CoxnetSurvivalAnalysis is not None and isinstance(model, CoxnetSurvivalAnalysis):
            linear_predictor = model.predict(X_array, alpha=alpha)
        else:
            linear_predictor = model.predict(X_array)
        # As BreslowEstimator.get_survival_function: S0(t) ** exp(linear predictor), unclipped.
        risk = np.exp(np.asarray(linear_predictor, dtype=float).reshape(-1))
        baseline_times = np.asarray(baseline_survival.x, dtype=float)
        baseline_values = np.asarray(baseline_survival.y, dtype=float)

        def _predict_cox(eval_times: np.ndarray) -> np.ndarray:
            positions = _sksurv_step_positions(baseline_times, eval_times)
            survival = np.power(baseline_values[positions][np.newaxis, :], risk[:, np.newaxis])
            return np.clip(survival, 0.0, 1.0)

        return _predict_cox

    if _is_random_survival_forest(model) and getattr(model, "unique_times_", None) is not None:
        forest_times = np.asarray(model.unique_times_, dtype=float)

        def _predict_forest(eval_times: np.ndarray) -> np.ndarray:
            positions = _sksurv_step_positions(forest_times, eval_times)
            n_rows = int(X_array.shape[0])
            chunk = _prediction_row_chunk(model) or max(n_rows, 1)
            survival = np.empty((n_rows, positions.size), dtype=float)
            for start in range(0, n_rows, chunk):
                raise_if_cancelled()
                block = model.predict_survival_function(X_array[start : start + chunk], return_array=True)
                survival[start : start + chunk] = np.clip(np.asarray(block, dtype=float)[:, positions], 0.0, 1.0)
            return survival

        return _predict_forest

    predict_kwargs = {"return_array": False}
    if alpha is not None:
        predict_kwargs["alpha"] = alpha
    step_functions = model.predict_survival_function(X_array, **predict_kwargs)

    def _predict(eval_times: np.ndarray) -> np.ndarray:
        return _step_function_matrix(step_functions, np.asarray(eval_times, dtype=float))

    return _predict


def _breslow_baseline_survival(
    times: np.ndarray,
    status: np.ndarray,
    linear_predictor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow baseline survival S0(t) = exp(-H0(t)) at unique event times.

    ``statsmodels``' ``baseline_cumulative_hazard`` reports H0(t-) (it
    subtracts the jump at each event time), which lags survival predictions by
    one event time; this helper returns the right-continuous H0(t).
    """
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    status_arr = np.asarray(status, dtype=float).reshape(-1) > 0
    risk = np.exp(np.clip(np.asarray(linear_predictor, dtype=float).reshape(-1), -50.0, 50.0))
    # Distinct event times and the number of events at each, in one sort.
    event_times, death_counts = np.unique(times_arr[status_arr], return_counts=True)
    if event_times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(times_arr, kind="mergesort")
    sorted_times = times_arr[order]
    # Sum of exp(lp) over subjects with time >= t (risk set), via reverse cumsum.
    reverse_cumsum = np.cumsum(risk[order][::-1])[::-1]
    first_at_or_after = np.searchsorted(sorted_times, event_times, side="left")
    risk_set_sums = reverse_cumsum[first_at_or_after]
    deaths = death_counts.astype(float)
    cumulative_hazard = np.cumsum(deaths / risk_set_sums)
    return event_times, np.exp(-cumulative_hazard)


def _cox_ph_survival_predictor(results: Any, exog: pd.DataFrame | np.ndarray) -> Any:
    baseline_hazard = getattr(results, "baseline_cumulative_hazard", None)
    if not baseline_hazard or len(baseline_hazard) != 1:
        raise ValueError("Cox PH Brier-score support currently requires a single-stratum baseline hazard.")
    params = np.asarray(results.params, dtype=float)
    model = results.model
    baseline_times, baseline_survival = _breslow_baseline_survival(
        np.asarray(model.endog, dtype=float),
        np.asarray(model.status, dtype=float),
        np.asarray(model.exog, dtype=float) @ params,
    )
    linear_predictor = np.asarray(exog, dtype=float) @ params
    hazard_ratio = np.exp(np.clip(linear_predictor, -50.0, 50.0))

    def _predict(eval_times: np.ndarray) -> np.ndarray:
        eval_times_arr = np.asarray(eval_times, dtype=float).reshape(-1)
        baseline_values = _step_survival_lookup(baseline_times, baseline_survival, eval_times_arr)
        return np.power(baseline_values[np.newaxis, :], hazard_ratio[:, np.newaxis], dtype=float)

    return _predict


def _brier_failure_must_propagate(exc: BaseException) -> bool:
    """Failures a per-model fallback or the optional Brier metrics must not swallow: bugs,
    cancellation, exhausted memory.

    ``compute_integrated_brier_score`` wraps its own ``TypeError``s in an
    ``InternalAnalysisError``, so the original exception is checked too.
    """
    if isinstance(exc, MemoryError) or must_propagate(exc):
        return True
    cause = exc.__cause__
    return isinstance(exc, InternalAnalysisError) and cause is not None and (
        isinstance(cause, MemoryError) or must_propagate(cause)
    )


def _failure_message(exc: BaseException) -> str:
    """The message recorded for a failed model or fold; the exception type when the message is empty."""
    return str(exc).strip() or type(exc).__name__


# Failures of the optional Brier metrics that come from the data (no IPCW support, a
# singular or overflowing baseline); anything else, such as an IndexError or a
# ZeroDivisionError, is a bug and propagates.
_EXPECTED_BRIER_FAILURES = (ValueError, np.linalg.LinAlgError, FloatingPointError)


def _maybe_compute_brier_metrics(
    times: np.ndarray,
    events: np.ndarray,
    predicted_survival_fn: Any,
    *,
    support_times: np.ndarray | None = None,
    support_events: np.ndarray | None = None,
) -> dict[str, Any] | None:
    try:
        return compute_integrated_brier_score(
            times,
            events,
            predicted_survival_fn,
            support_times=support_times,
            support_events=support_events,
        )
    except _EXPECTED_BRIER_FAILURES as exc:
        if _brier_failure_must_propagate(exc):
            raise
        warnings.warn(
            f"Integrated Brier Score / Brier Skill Score could not be computed: {exc}",
            RuntimeWarning,
        )
        return None


def _survival_predictor_or_none(build: Any) -> Any:
    """``build()``, the survival-function predictor of a fitted model; None (with a warning) when
    the data do not allow one, so only the Brier metrics are lost, never the model."""
    try:
        return build()
    except _EXPECTED_BRIER_FAILURES as exc:
        if _brier_failure_must_propagate(exc):
            raise
        warnings.warn(
            f"Integrated Brier Score / Brier Skill Score could not be computed because survival-function predictions were unavailable: {exc}",
            RuntimeWarning,
        )
        return None


def _maybe_compute_sksurv_brier_metrics(
    times: np.ndarray,
    events: np.ndarray,
    model: Any,
    X: pd.DataFrame,
    *,
    alpha: float | None = None,
    support_times: np.ndarray | None = None,
    support_events: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if not callable(getattr(model, "predict_survival_function", None)):
        warnings.warn(
            "Integrated Brier Score / Brier Skill Score could not be computed because survival-function predictions "
            f"were unavailable: {type(model).__name__} has no predict_survival_function().",
            RuntimeWarning,
        )
        return None
    predicted_survival_fn = _survival_predictor_or_none(lambda: _sksurv_survival_predictor(model, X, alpha=alpha))
    if predicted_survival_fn is None:
        return None
    return _maybe_compute_brier_metrics(
        times,
        events,
        predicted_survival_fn,
        support_times=support_times,
        support_events=support_events,
    )


def _augment_scientific_summary_with_brier(
    scientific_summary: dict[str, Any],
    brier_result: dict[str, Any] | None,
) -> dict[str, Any]:
    scientific_summary = copy.deepcopy(scientific_summary)
    # A comparison passes the top model's Brier fields, which are all None when they failed.
    if not brier_result or _safe_float(brier_result.get("ibs")) is None:
        scientific_summary["cautions"].append(
            "IBS / Brier Skill Score could not be computed for this run, so calibration-error interpretation remains incomplete."
        )
        if scientific_summary.get("status") == "robust":
            scientific_summary["status"] = "review"
        return scientific_summary

    ibs = _safe_float(brier_result.get("ibs"))
    null_ibs = _safe_float(brier_result.get("null_ibs"))
    brier_skill_score = _safe_float(brier_result.get("brier_skill_score"))
    metrics = list(scientific_summary.get("metrics", []))
    existing_labels = {str(item.get("label")) for item in metrics}
    brier_metrics = [
        ("IBS", ibs),
        ("Null-model IBS", null_ibs),
        ("Brier Skill Score", brier_skill_score),
    ]
    for label, value in brier_metrics:
        if label not in existing_labels:
            metrics.append({"label": label, "value": value})
    scientific_summary["metrics"] = metrics

    if ibs is not None and null_ibs is not None and brier_skill_score is not None:
        if brier_skill_score > 0.0:
            scientific_summary["strengths"].append(
                f"IBS = {ibs:.4f} versus a Kaplan-Meier null-model IBS of {null_ibs:.4f}, giving a Brier Skill Score of {brier_skill_score:.3f}."
            )
        else:
            scientific_summary["cautions"].append(
                f"IBS = {ibs:.4f} versus a Kaplan-Meier null-model IBS of {null_ibs:.4f}, yielding a Brier Skill Score of {brier_skill_score:.3f}; this run did not improve on the null benchmark."
            )
            if scientific_summary.get("status") == "robust":
                scientific_summary["status"] = "review"
            if brier_skill_score < -0.05:
                scientific_summary["status"] = "caution"
    elif ibs is not None:
        scientific_summary["cautions"].append(
            f"IBS = {ibs:.4f}, but the Kaplan-Meier null benchmark was not stable enough to compute a Brier Skill Score."
        )
        if scientific_summary.get("status") == "robust":
            scientific_summary["status"] = "review"

    scientific_summary["next_steps"].append(
        "Use Brier Skill Score versus the Kaplan-Meier null benchmark, not raw IBS alone, when arguing that a model meaningfully improves survival prediction error."
    )
    return scientific_summary


def _require_predict_callable(model: Any, *, context: str) -> None:
    predict_fn = getattr(model, "predict", None)
    if not callable(predict_fn):
        raise ValueError(
            f"{context} requires a fitted model with a callable predict() method."
        )


def _split_train_test_positions(
    frame: pd.DataFrame,
    event_column: str,
    random_state: int = 42,
    test_size: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Positions of the shared stratified holdout used by every model family."""
    _require_sklearn()
    return stratified_holdout_indices(
        frame[event_column].astype(int).to_numpy(),
        random_state=random_state,
        test_size=test_size,
    )


def _split_train_test(
    frame: pd.DataFrame,
    event_column: str,
    random_state: int = 42,
    test_size: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Create a deterministic train/test split for model comparison.

    Returns the split frames and a short evaluation-mode label.
    """
    train_positions, eval_positions, mode = _split_train_test_positions(
        frame,
        event_column,
        random_state=random_state,
        test_size=test_size,
    )
    if mode != "holdout":
        return frame.copy().reset_index(drop=True), frame.copy().reset_index(drop=True), "apparent"
    return (
        frame.iloc[train_positions].reset_index(drop=True),
        frame.iloc[eval_positions].reset_index(drop=True),
        "holdout",
    )


def _frame_source_rows(frame: pd.DataFrame) -> list[Any] | None:
    source_rows = frame.attrs.get("source_row_index")
    if source_rows is None or len(source_rows) != len(frame):
        return None
    return list(source_rows)


def _unseen_category_rows(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    categorical_features: Sequence[str],
) -> int:
    """Evaluation rows that the encoder fitted on ``train_frame`` scores as the reference level
    although their categorical value is not the reference level.

    The shared encoder has no unknown-level column, so a level never seen in training is
    scored as the reference level. It has a missing-value column only when the training rows
    contain missing values, so without one a missing evaluation value is scored as the
    reference level too.
    """
    affected = pd.Series(False, index=eval_frame.index)
    for feature in categorical_features:
        if feature not in train_frame.columns or feature not in eval_frame.columns:
            continue
        train_values = train_frame[feature]
        train_levels = set(train_values.dropna().astype(str))
        eval_values = eval_frame[feature]
        affected |= eval_values.notna() & ~eval_values.astype(str).isin(train_levels)
        if not bool(train_values.isna().any()):
            affected |= eval_values.isna()
    return int(affected.sum())


def _resolved_categorical_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None,
) -> list[str]:
    """Features the shared encoder one-hot encodes: the declared ones plus every text column."""
    present = [feature for feature in features if feature in frame.columns]
    if not present:
        return []
    _, resolved, _ = coerce_feature_subset(frame, present, categorical_features)
    return list(resolved)


def _encoder_categorical_features(
    feature_encoder: dict[str, Any] | None,
    requested: Sequence[str] | None,
) -> list[str]:
    """Categorical features of a fitted encoder; the requested list when there is no encoder."""
    if isinstance(feature_encoder, dict) and feature_encoder.get("categorical_features") is not None:
        return [str(feature) for feature in feature_encoder.get("categorical_features") or []]
    return [str(feature) for feature in requested or []]


def _resolve_category_level(feature_encoder: dict[str, Any] | None, feature: str, value: Any) -> str:
    """The encoder level that a requested value of a categorical feature names.

    A value the model never saw would be encoded as the reference level, so it is refused.
    Numbers and numeric text match a level of the same number (3, 3.0 and "3.0" match the
    canonical level "3" of a whole-number code column).
    """
    mapping = {}
    if isinstance(feature_encoder, dict):
        mapping = (feature_encoder.get("categorical_mappings") or {}).get(feature) or {}
    levels = [str(level) for level in mapping.get("all_levels") or []]
    text = str(value)
    if not levels:
        return text
    for candidate in (text, text.strip()):
        if candidate in levels:
            return candidate
    number = pd.to_numeric(pd.Series([text.strip()]), errors="coerce").iloc[0]
    if pd.notna(number):
        level_numbers = pd.to_numeric(pd.Series(levels), errors="coerce")
        matches = [
            level
            for level, level_number in zip(levels, level_numbers)
            if pd.notna(level_number) and float(level_number) == float(number)
        ]
        if len(matches) == 1:
            return matches[0]
    shown = ", ".join(f"'{level}'" for level in levels[:20]) + (", ..." if len(levels) > 20 else "")
    raise ValueError(
        f"'{value}' is not a level of '{feature}' that the model was trained on, so it would be scored as the "
        f"reference level. Use one of: {shown}."
    )


def _unseen_category_caution(n_rows: int, scope: str) -> str | None:
    if n_rows <= 0:
        return None
    return (
        f"{n_rows} {scope} row(s) had a categorical level that never occurs in the corresponding training data "
        "(or a missing categorical value where the training data have none); the models score those rows as the "
        "reference level. Merge rare levels and recode missing values before modeling."
    )


def _median_imputed_counts(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None,
) -> dict[str, int]:
    """Rows per numeric feature whose value is missing or infinite, which the shared encoder
    replaces by the median of the training rows."""
    present = [feature for feature in features if feature in frame.columns]
    if not present:
        return {}
    selected, _, numeric_features = coerce_feature_subset(frame, present, categorical_features)
    counts: dict[str, int] = {}
    for column in numeric_features:
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        n_missing = int((~np.isfinite(values)).sum())
        if n_missing:
            counts[str(column)] = n_missing
    return counts


def _median_imputation_caution(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    shown = ", ".join(f"{feature} ({n_rows})" for feature, n_rows in list(counts.items())[:10])
    if len(counts) > 10:
        shown += f", and {len(counts) - 10} more feature(s)"
    return (
        f"{sum(counts.values())} missing numeric value(s) were replaced by the median of the training rows before "
        f"modeling: {shown}. Median imputation assumes the values are missing at random."
    )


def _encode_train_test_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encode train/test feature frames with aligned columns."""
    encoder = _fit_feature_encoder(train_frame, features, categorical_features)
    train_encoded = _transform_feature_encoder(train_frame, encoder)
    test_encoded = _transform_feature_encoder(test_frame, encoder)
    return train_encoded, test_encoded, encoder


def _drop_constant_train_columns(
    train_encoded: pd.DataFrame,
    eval_encoded: pd.DataFrame,
    *,
    model_label: str = "Cox PH",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop encoded columns that are constant in the training split.

    Cox PH can fail with singular matrices when the train split contains
    all-zero unknown buckets or other constant encoded features. The
    evaluation matrix is reduced to the same column set.
    """
    varying_columns = [
        column
        for column in train_encoded.columns
        if train_encoded[column].nunique(dropna=False) > 1
    ]
    if not varying_columns:
        raise ValueError(f"No non-constant encoded features remain for {model_label}.")
    reduced_train = train_encoded.loc[:, varying_columns].copy()
    reduced_eval = eval_encoded.loc[:, varying_columns].copy()
    removed = list(train_encoded.attrs.get("removed_features", [])) + [
        {"column": str(column), "reason": "constant in training rows"}
        for column in train_encoded.columns if column not in varying_columns
    ]
    reduced_train.attrs["removed_features"] = removed
    reduced_eval.attrs["removed_features"] = removed
    return reduced_train, reduced_eval


def _drop_constant_train_columns_with_full(
    train_encoded: pd.DataFrame,
    eval_encoded: pd.DataFrame,
    full_encoded: pd.DataFrame | None = None,
    *,
    model_label: str = "Cox PH",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    reduced_train, reduced_eval = _drop_constant_train_columns(train_encoded, eval_encoded, model_label=model_label)
    if full_encoded is None:
        return reduced_train, reduced_eval, None
    return reduced_train, reduced_eval, full_encoded.loc[:, reduced_train.columns].copy()


def _effective_tree_min_samples_leaf(
    min_samples_leaf: int,
    n_training_rows: int,
) -> int:
    requested = max(1, int(min_samples_leaf))
    available_rows = int(n_training_rows)
    if available_rows < 1:
        raise ValueError("Tree-based survival models need at least one training row.")
    return min(requested, available_rows)


def _drop_rank_deficient_train_columns(
    train_encoded: pd.DataFrame,
    eval_encoded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop encoded columns that make the Cox training design rank-deficient.

    Shared predictive feature lists can legitimately contain overlapping
    categorical representations that tree and deep models tolerate, but
    unpenalized Cox PH on a shared holdout path cannot. Keep a full-rank
    subset so the Cox screening baseline remains comparable instead of
    failing with a singular matrix.
    """
    train_matrix = train_encoded.to_numpy(dtype=float)
    if train_matrix.ndim != 2 or train_matrix.shape[1] <= 1:
        return train_encoded.copy(), eval_encoded.copy()

    _q, r_factor, pivots = scipy_qr(train_matrix, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r_factor))
    max_diagonal = float(diagonal.max()) if diagonal.size else 0.0
    tolerance = np.finfo(float).eps * max(train_matrix.shape) * max(1.0, max_diagonal)
    rank = int(np.sum(diagonal > tolerance))
    if rank <= 0:
        raise ValueError("No full-rank encoded features remain for Cox PH.")
    if rank >= train_encoded.shape[1]:
        return train_encoded.copy(), eval_encoded.copy()

    keep_indices = sorted(int(index) for index in pivots[:rank])
    keep_columns = [train_encoded.columns[index] for index in keep_indices]
    reduced_train = train_encoded.loc[:, keep_columns].copy()
    reduced_eval = eval_encoded.loc[:, keep_columns].copy()
    removed = list(train_encoded.attrs.get("removed_features", [])) + [
        {"column": str(column), "reason": "redundant in Cox training design"}
        for column in train_encoded.columns if column not in keep_columns
    ]
    reduced_train.attrs["removed_features"] = removed
    reduced_eval.attrs["removed_features"] = removed
    return reduced_train, reduced_eval


def _standardize_encoded_matrices(
    train_encoded: pd.DataFrame,
    eval_encoded: pd.DataFrame,
    full_encoded: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, pd.Series]]:
    """Standardize encoded matrices using training-split moments only."""
    train_float = train_encoded.astype(float).copy()
    eval_float = eval_encoded.astype(float).copy()
    full_float = full_encoded.astype(float).copy() if full_encoded is not None else None

    means = train_float.mean(axis=0).astype(float)
    scales = train_float.std(axis=0, ddof=0).astype(float).replace(0.0, 1.0)

    def _apply(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        scaled = (frame - means) / scales
        scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return scaled.astype(float)

    return _apply(train_float), _apply(eval_float), _apply(full_float), {
        "mean": means,
        "scale": scales,
    }


def _make_lasso_coxnet_model(*, alpha: float | None = None) -> CoxnetSurvivalAnalysis:
    kwargs: dict[str, Any] = {
        "l1_ratio": 1.0,
        "normalize": False,
        "tol": 1e-7,
        "max_iter": 100000,
        "fit_baseline_model": True,
    }
    if alpha is None:
        kwargs.update({"n_alphas": 40, "alpha_min_ratio": "auto"})
    else:
        kwargs["alphas"] = [float(alpha)]
    return CoxnetSurvivalAnalysis(**kwargs)


def _coerce_coxnet_coef_vector(model: CoxnetSurvivalAnalysis) -> np.ndarray:
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 1:
        return coef.astype(float)
    if coef.ndim == 2 and coef.shape[1] >= 1:
        return coef[:, -1].astype(float)
    raise ValueError("LASSO-Cox did not produce a usable coefficient vector.")


def _select_lasso_alpha(
    train_frame: pd.DataFrame,
    train_encoded: pd.DataFrame,
    *,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    time_column: str,
    event_column: str,
    random_state: int,
    inner_folds: int = 5,
) -> dict[str, Any]:
    """Choose an L1 penalty by inner stratified K-fold CV on the training split.

    The alpha grid comes from a Coxnet path fit on the training split; each
    inner fold fits imputation, category encoding, constant-column filtering,
    standardization and the path on its own training part, and scores Harrell's C
    on its held-out part. The alpha
    with the highest mean inner C-index among non-empty models is selected
    (ties -> sparser model), analogous to glmnet's ``lambda.min``.
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is not installed.")
    _require_sklearn()

    n_train = int(train_frame.shape[0])
    train_events = train_frame[event_column].astype(int).to_numpy()
    y_train = _prepare_sksurv_data(train_frame, time_column, event_column)

    full_encoded, _ = _drop_constant_train_columns(train_encoded, train_encoded, model_label="LASSO-Cox")
    full_scaled, _, _, _ = _standardize_encoded_matrices(full_encoded, full_encoded)
    path_model = _make_lasso_coxnet_model()
    path_model.fit(full_scaled.to_numpy(), y_train)
    alphas = np.asarray(path_model.alphas_, dtype=float)
    coef_path = np.asarray(path_model.coef_, dtype=float)
    if coef_path.ndim == 1:
        coef_path = coef_path[:, np.newaxis]
    n_nonzero_by_alpha = [
        int(np.count_nonzero(np.abs(coef_path[:, min(idx, coef_path.shape[1] - 1)]) > 1e-10))
        for idx in range(alphas.size)
    ]

    _, stratum_counts = np.unique(train_events, return_counts=True)
    min_stratum = int(stratum_counts.min()) if stratum_counts.size else 0
    n_splits = min(int(inner_folds), min_stratum)
    scores_by_alpha: list[list[float]] = [[] for _ in range(alphas.size)]
    selection_mode = "inner_cv"
    if n_train >= 30 and stratum_counts.size >= 2 and n_splits >= 2:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for inner_train_idx, inner_eval_idx in splitter.split(train_encoded, train_events):
            # Each inner fold fits a whole Coxnet path, so a cancelled request stops between folds.
            raise_if_cancelled()
            try:
                inner_train_encoded, inner_eval_encoded, inner_train_frame, inner_eval_frame = _encoded_fold_matrices(
                    train_frame.iloc[inner_train_idx].reset_index(drop=True),
                    train_frame.iloc[inner_eval_idx].reset_index(drop=True),
                    features=features,
                    categorical_features=categorical_features,
                    model_label="LASSO-Cox",
                )
                inner_train_scaled, inner_eval_scaled, _, _ = _standardize_encoded_matrices(
                    inner_train_encoded,
                    inner_eval_encoded,
                )
                fold_model = CoxnetSurvivalAnalysis(
                    alphas=alphas,
                    l1_ratio=1.0,
                    normalize=False,
                    tol=1e-7,
                    max_iter=100000,
                )
                fold_model.fit(
                    inner_train_scaled.to_numpy(),
                    _prepare_sksurv_data(
                        inner_train_frame,
                        time_column,
                        event_column,
                    ),
                )
            except (ValueError, ArithmeticError):
                # Degenerate inner fold (for example a singular design or too few events).
                continue
            y_inner_eval = _prepare_sksurv_data(
                inner_eval_frame,
                time_column,
                event_column,
            )
            for idx, alpha in enumerate(alphas.tolist()):
                try:
                    risk_scores = np.asarray(
                        fold_model.predict(inner_eval_scaled.to_numpy(), alpha=float(alpha)),
                        dtype=float,
                    )
                except (ValueError, ArithmeticError):
                    continue
                c_index = _sksurv_c_index(y_inner_eval, risk_scores)
                if c_index is not None and np.isfinite(float(c_index)):
                    scores_by_alpha[idx].append(float(c_index))
    if not any(scores_by_alpha):
        # Too small for inner CV: fall back to apparent training concordance.
        selection_mode = "apparent"
        for idx, alpha in enumerate(alphas.tolist()):
            try:
                risk_scores = np.asarray(path_model.predict(full_scaled.to_numpy(), alpha=float(alpha)), dtype=float)
            except (ValueError, ArithmeticError):
                continue
            c_index = _sksurv_c_index(y_train, risk_scores)
            if c_index is not None and np.isfinite(float(c_index)):
                scores_by_alpha[idx].append(float(c_index))

    candidate_rows = [
        {
            "alpha": float(alphas[idx]),
            "c_index": float(np.mean(scores)),
            "c_index_se": float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else None,
            "n_scored_folds": len(scores),
            "n_nonzero_features": n_nonzero_by_alpha[idx],
        }
        for idx, scores in enumerate(scores_by_alpha)
        if scores
    ]
    if not candidate_rows:
        raise ValueError("LASSO-Cox could not find a valid penalty along the fitted Coxnet path.")
    max_folds = max(row["n_scored_folds"] for row in candidate_rows)
    complete_rows = [row for row in candidate_rows if row["n_scored_folds"] == max_folds]
    nonzero_rows = [row for row in complete_rows if row["n_nonzero_features"] > 0]
    search_rows = nonzero_rows or complete_rows
    best = max(
        search_rows,
        key=lambda row: (round(float(row["c_index"]), 12), float(row["alpha"])),
    )
    return {
        "alpha": float(best["alpha"]),
        "selection_mode": selection_mode,
        "selection_rule": "inner_cv_max_mean_c_index" if selection_mode == "inner_cv" else "apparent_max_c_index",
        "selection_threshold_c_index": float(best["c_index"]),
        "inner_selection_c_index": float(best["c_index"]),
        "inner_selection_c_index_se": _safe_float(best.get("c_index_se")),
        # k of the k-fold inner split, and how many of its folds could be scored.
        "inner_cv_folds": int(n_splits) if selection_mode == "inner_cv" else 0,
        "inner_cv_scored_folds": int(max_folds) if selection_mode == "inner_cv" else 0,
        "n_nonzero_features": int(best["n_nonzero_features"]),
        "n_alpha_candidates": int(len(candidate_rows)),
    }


def _prepare_model_evaluation_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Prepare aligned train/evaluation/full encoded matrices.

    The shared stratified holdout is used whenever the cohort allows one; a cohort too small
    for a holdout is scored on the rows it is fitted on (apparent evaluation). The encoder
    keeps every row (missing values are imputed from the training rows), so encoding never
    removes the evaluation rows. Feature types are decided on the whole ``frame``, so the
    training split never decides a feature's type anew.
    """
    categorical_features = _resolved_categorical_features(frame, features, categorical_features)
    train_frame, eval_frame, evaluation_mode = _split_train_test(
        frame,
        event_column,
        random_state=random_state,
    )
    train_encoded, eval_encoded, encoder = _encode_train_test_features(
        train_frame,
        eval_frame,
        features,
        categorical_features,
    )
    if train_encoded.isna().any().any() or eval_encoded.isna().any().any():
        raise ValueError("Feature encoding produced non-finite values after imputation.")
    if train_encoded.empty or eval_encoded.empty:
        raise ValueError("No valid rows remain after encoding features for model evaluation.")
    train_encoded = train_encoded.reset_index(drop=True)
    eval_encoded = eval_encoded.reset_index(drop=True)
    train_frame = train_frame.reset_index(drop=True)
    eval_frame = eval_frame.reset_index(drop=True)

    full_encoded = _transform_feature_encoder(frame, encoder).reset_index(drop=True)
    full_frame = frame.reset_index(drop=True)
    if full_encoded.isna().any().any():
        raise ValueError("Encoded full feature matrix contains non-finite values after imputation.")

    return {
        "train_frame": train_frame,
        "eval_frame": eval_frame,
        "full_frame": full_frame,
        "train_encoded": train_encoded,
        "eval_encoded": eval_encoded,
        "full_encoded": full_encoded,
        "evaluation_mode": evaluation_mode,
        "metric_name": _metric_name_for_evaluation(evaluation_mode),
        "feature_encoder": encoder,
    }


# ===================================================================
# 1. Optimal cutpoint scanning
# ===================================================================

def _logrank_observed_expected(
    time_values: np.ndarray,
    event_values: np.ndarray,
    group_mask: np.ndarray,
) -> tuple[float, float]:
    """Observed and log-rank expected event counts for the rows in ``group_mask``.

    Risk-set sizes come from sorted times (O(n log n)); the per-event-time terms are
    added in increasing time order, as a sequential scan over the event times would.
    """
    times = np.asarray(time_values, dtype=float).reshape(-1)
    events = np.asarray(event_values, dtype=float).reshape(-1) > 0
    mask = np.asarray(group_mask, dtype=bool).reshape(-1)
    observed = float(events[mask].sum())
    event_times, deaths = np.unique(times[events], return_counts=True)
    if event_times.size == 0:
        return observed, 0.0
    sorted_times = np.sort(times)
    sorted_group_times = np.sort(times[mask])
    n_at_risk = (times.size - np.searchsorted(sorted_times, event_times, side="left")).astype(float)
    group_at_risk = (sorted_group_times.size - np.searchsorted(sorted_group_times, event_times, side="left")).astype(float)
    terms = deaths.astype(float) * group_at_risk / n_at_risk
    expected = 0.0
    for term in terms.tolist():
        expected += term
    return observed, expected


@user_input_boundary
def find_optimal_cutpoint(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    variable: str,
    event_positive_value: Any = None,
    min_group_fraction: float = 0.1,
    lower_label: str = "Low",
    upper_label: str = "High",
    permutation_iterations: int = 500,
    random_seed: int = 20260311,
    include_split_series: bool = False,
) -> dict[str, Any]:
    """Scan all unique values of *variable* and return the cutpoint that
    maximises the log-rank chi-square statistic.

    Returns
    -------
    dict
        Keys: ``optimal_cutpoint``, ``p_value`` (selection-adjusted when
        permutations are enabled), ``raw_p_value`` (unadjusted), ``n_high`` and
        ``n_low`` (sizes of the higher- and lower-risk groups),
        ``n_above_cutpoint`` and ``n_below_cutpoint`` (sizes of the groups above and
        below the cutpoint), ``scan_data`` (list of per-cutpoint records, each with
        ``n_above_cutpoint`` and ``n_below_cutpoint``), ``split_column`` (name of the
        derived grouping column), and ``n_marker_values_non_numeric`` /
        ``n_marker_values_non_finite`` with ``input_notes`` for marker values that
        were left out of the scan.
    """
    if str(variable) in {str(time_column), str(event_column)}:
        raise ValueError(
            f"'{variable}' is the survival {'time' if str(variable) == str(time_column) else 'event'} column; "
            "choose a marker other than the outcome columns for the cutpoint scan."
        )
    if (
        isinstance(permutation_iterations, bool)
        or not isinstance(permutation_iterations, (int, np.integer))
        or int(permutation_iterations) < 0
    ):
        raise ValueError(
            f"permutation_iterations must be a whole number of at least 0; got {permutation_iterations!r}."
        )
    permutation_iterations = int(permutation_iterations)
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=[variable],
    )
    # Numbers stored as text with a few stray entries (such as "<0.1" below a detection
    # limit) are refused by the rule the model features use, instead of silently scanning
    # only the patients whose entry parses.
    contamination = numeric_text_contamination(frame[variable])
    if contamination is not None:
        examples = ", ".join(f'"{value}"' for value in contamination["examples"])
        raise ValueError(
            f'Marker "{variable}" looks numeric but contains {contamination["n_non_numeric"]} non-numeric value(s) '
            f"such as {examples}, and the cutpoint scan would leave those patients out. Recode those values as "
            "numbers or as missing (blank cells)."
        )

    numeric_values = pd.to_numeric(frame[variable], errors="coerce")
    numeric_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
    source_rows = _frame_source_rows(frame)
    if include_split_series and source_rows is None:
        # Falling back to positional labels would shift the split onto other patients.
        raise ValueError("The analysed rows could not be mapped back to the source dataset rows.")
    # Missing markers were already dropped with the cohort, so a value that does not parse is
    # text; infinite values in a numeric column were dropped there as missing and are counted too.
    n_non_numeric = int(np.isnan(numeric_array).sum())
    n_non_finite = int(np.isinf(numeric_array).sum()) + int(frame.attrs.get("rows_with_infinite_extra_values", 0) or 0)
    keep_mask = np.isfinite(numeric_array)
    frame = frame.loc[keep_mask].reset_index(drop=True)
    # Original row labels of the analysed rows, so the derived split can be
    # mapped back onto the uploaded dataset row-for-row.
    kept_source_rows = (
        [label for label, keep in zip(source_rows, keep_mask) if keep]
        if source_rows is not None
        else None
    )

    if frame.empty:
        raise ValueError(
            f"No valid numeric values in '{variable}' after cleaning"
            + (f" ({n_non_numeric} value(s) are not numbers)." if n_non_numeric else ".")
        )
    input_notes: list[str] = []
    if n_non_numeric:
        input_notes.append(f"{n_non_numeric} non-numeric value(s) in {variable} were treated as missing and left out of the scan.")
    if n_non_finite:
        input_notes.append(f"{n_non_finite} infinite value(s) in {variable} were treated as missing and left out of the scan.")

    time_values = frame[time_column].to_numpy(dtype=float)
    event_values = frame[event_column].to_numpy(dtype=float)
    var_values = numeric_array[keep_mask]
    n_total = len(frame)
    min_size = max(int(math.ceil(n_total * min_group_fraction)), 1)

    unique_vals = np.unique(var_values)
    if len(unique_vals) < 2:
        raise ValueError(
            f"'{variable}' has fewer than 2 unique values; a cutpoint split is not possible."
        )

    # Candidate cutpoints: midpoints between consecutive sorted unique values. Group
    # sizes depend only on the marker's values, which a permutation merely reorders, so
    # size-infeasible cutpoints are dropped once for the observed scan and every
    # permutation.
    all_candidates = 0.5 * (unique_vals[:-1] + unique_vals[1:])
    all_n_high = n_total - np.searchsorted(np.sort(var_values), all_candidates, side="right")
    size_feasible = (all_n_high >= min_size) & ((n_total - all_n_high) >= min_size)
    candidates = all_candidates[size_feasible]
    if candidates.size == 0:
        raise ValueError(
            f"No valid cutpoint found for '{variable}'. "
            "Ensure min_group_fraction allows at least one feasible split."
        )
    scanner = ThresholdLogrankScan(time_values, event_values)
    candidates, candidate_grid = _limit_cutpoint_candidates(
        candidates,
        n_event_times=int(scanner.event_times.size),
        permutation_iterations=int(permutation_iterations),
    )

    def _scan(marker: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scanned = scanner.scan(marker, candidates)
        n_high_values = scanned["n_high"]
        events_high = scanned["events_high"]
        # At least one event in each group, as for the observed split.
        eligible = (events_high > 0) & ((scanner.total_events - events_high) > 0)
        return scanned["statistic"], n_high_values, eligible

    statistics, n_high_values, eligible = _scan(var_values)
    if not bool(np.any(eligible)):
        raise ValueError(
            f"No valid cutpoint found for '{variable}'. "
            "Ensure min_group_fraction allows at least one feasible split."
        )
    # Group sizes by marker value (above / below the cutpoint); the result's n_high and
    # n_low are the higher- and lower-risk groups, which may be the other way round.
    scan_data: list[dict[str, Any]] = [
        {
            "cutpoint": _safe_float(cutpoint),
            "statistic": _safe_float(statistic),
            "p_value": _safe_float(float(stats.chi2.sf(statistic, df=1))),
            "n_above_cutpoint": int(n_above),
            "n_below_cutpoint": int(n_total - n_above),
        }
        for cutpoint, statistic, n_above, keep in zip(candidates, statistics, n_high_values, eligible, strict=True)
        if keep
    ]
    # First maximum in cutpoint order, as a sequential scan with a strict ">" would pick.
    best_position = int(np.argmax(np.where(eligible, statistics, -np.inf)))
    best_stat = float(statistics[best_position])
    best_record = next(
        record for record in scan_data if record["cutpoint"] == _safe_float(candidates[best_position])
    )

    # Build the split — assign labels based on actual survival direction
    optimal_cp = best_record["cutpoint"]
    high_mask = var_values > optimal_cp

    # Orient labels by the log-rank observed-minus-expected events of the
    # above-cutpoint group: more events than expected under the null means
    # higher risk. Median comparisons fail when medians are not reached.
    above_observed, above_expected = _logrank_observed_expected(time_values, event_values, high_mask)
    swap = above_observed < above_expected

    if swap:
        label_above, label_below = lower_label, upper_label
    else:
        label_above, label_below = upper_label, lower_label

    split_col_name = f"{variable}_group"
    split_series = None
    if include_split_series:
        split_series = pd.Series(
            np.where(high_mask, label_above, label_below),
            index=pd.Index(kept_source_rows),
            dtype="string",
        )

    selection_adjusted_p_value = None
    perm_valid = 0
    if permutation_iterations > 0:
        rng = np.random.default_rng(int(random_seed))
        extreme = 0
        # Identical arithmetic for the observed and permuted scans, so a tie with the
        # observed maximum is counted; the tolerance only absorbs summation order.
        threshold = best_stat - 1e-9 * max(1.0, abs(best_stat))
        for _ in range(int(permutation_iterations)):
            raise_if_cancelled()
            permuted_statistics, _, permuted_eligible = _scan(rng.permutation(var_values))
            if not bool(np.any(permuted_eligible)):
                continue
            perm_valid += 1
            if float(np.max(permuted_statistics[permuted_eligible])) >= threshold:
                extreme += 1
        selection_adjusted_p_value = float((extreme + 1) / (perm_valid + 1)) if perm_valid else None

    raw_p_value = best_record["p_value"]
    primary_p_value = _safe_float(selection_adjusted_p_value)
    if primary_p_value is None:
        primary_p_value = raw_p_value

    result = {
        "optimal_cutpoint": best_record["cutpoint"],
        "statistic": best_record["statistic"],
        "p_value": primary_p_value,
        "p_value_label": (
            "selection_adjusted_p_value" if selection_adjusted_p_value is not None else "raw_p_value"
        ),
        "raw_p_value": raw_p_value,
        "selection_adjusted_p_value": _safe_float(selection_adjusted_p_value),
        "selection_adjustment": {
            "method": "permutation_max_statistic" if permutation_iterations > 0 else None,
            "permutation_iterations": int(permutation_iterations),
            "permutation_valid_resamples": int(perm_valid),
            "random_seed": int(random_seed),
            "note": (
                "The raw_p_value is unadjusted for cutpoint selection. "
                "When available, p_value is the selection-adjusted value."
            ),
        },
        "n_above_cutpoint": best_record["n_above_cutpoint"],
        "n_below_cutpoint": best_record["n_below_cutpoint"],
        "n_high": best_record["n_below_cutpoint"] if swap else best_record["n_above_cutpoint"],
        "n_low": best_record["n_above_cutpoint"] if swap else best_record["n_below_cutpoint"],
        "risk_direction_rule": "log-rank observed vs expected events in the above-cutpoint group",
        "above_cutpoint_observed_events": _safe_float(above_observed),
        "above_cutpoint_expected_events": _safe_float(above_expected),
        "label_above_cutpoint": label_above,
        "label_below_cutpoint": label_below,
        "scan_data": scan_data,
        "split_column": split_col_name,
        "candidate_grid": candidate_grid,
        "n_marker_values_non_numeric": n_non_numeric,
        "n_marker_values_non_finite": n_non_finite,
        "input_notes": input_notes,
    }
    if split_series is not None:
        result["split_series"] = split_series
    return result


# Event times x cutpoints x (permutations + 1) cells one cutpoint scan may evaluate
# (~15 s of vectorised work). Larger problems scan a quantile-spaced subset of cutpoints.
_CUTPOINT_SCAN_CELL_BUDGET = 1_000_000_000
_MIN_CUTPOINT_GRID = 50


def _limit_cutpoint_candidates(
    candidates: np.ndarray,
    *,
    n_event_times: int,
    permutation_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep every feasible cutpoint unless the scan would exceed the work budget.

    The observed scan and every permutation use the same cutpoint set, so a thinned grid
    still gives a valid selection-adjusted p-value for the procedure actually run.
    """
    n_candidates = int(candidates.size)
    per_scan_cells = max(int(n_event_times), 1) * (int(permutation_iterations) + 1)
    max_candidates = max(_MIN_CUTPOINT_GRID, _CUTPOINT_SCAN_CELL_BUDGET // per_scan_cells)
    if n_candidates <= max_candidates:
        return candidates, {
            "method": "all_midpoints",
            "n_candidates": n_candidates,
            "n_feasible_candidates": n_candidates,
            "note": None,
        }
    positions = np.unique(np.round(np.linspace(0, n_candidates - 1, int(max_candidates))).astype(int))
    reduced = candidates[positions]
    return reduced, {
        "method": "quantile_spaced_subset",
        "n_candidates": int(reduced.size),
        "n_feasible_candidates": n_candidates,
        "note": (
            f"{int(reduced.size)} of {n_candidates} feasible cutpoints (evenly spaced over the marker's ranks) were "
            "scanned to keep the scan and its permutation null tractable; the permutation p-value is exact for "
            "this scanned set."
        ),
    }


def _prepare_training_matrices(
    df: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None,
    event_positive_value: Any,
    random_state: int,
    internal_evaluation: bool,
    model_label: str,
) -> dict[str, Any]:
    """Clean the cohort and build aligned train/evaluation/full encoded matrices.

    Shared by the RSF, GBS, and LASSO-Cox trainers. With ``internal_evaluation`` the
    shared stratified holdout is used (falling back to apparent evaluation for small
    cohorts); otherwise every row is used for fitting and scoring. Encoded columns that
    are constant in the training rows are dropped from all three matrices.
    """
    _validate_model_feature_columns(features, time_column=time_column, event_column=event_column)
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )
    reject_numeric_text_features(frame, features, list(categorical_features or []))
    # Feature types are decided once on the whole cohort (declared categorical features plus the
    # text features the encoder one-hot codes), so a split never decides a feature's type anew.
    resolved_categorical = _resolved_categorical_features(frame, features, categorical_features)

    if internal_evaluation:
        split = _prepare_model_evaluation_split(
            frame,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=resolved_categorical,
            random_state=random_state,
        )
        matrices = {
            key: split[key]
            for key in (
                "train_frame",
                "eval_frame",
                "full_frame",
                "train_encoded",
                "eval_encoded",
                "full_encoded",
                "evaluation_mode",
                "metric_name",
                "feature_encoder",
            )
        }
    else:
        feature_encoder = _fit_feature_encoder(frame, features, resolved_categorical)
        full_encoded = _transform_feature_encoder(frame, feature_encoder).reset_index(drop=True)
        full_frame = frame.reset_index(drop=True)
        if full_encoded.empty:
            raise ValueError(f"No analyzable rows remain after encoding features for {model_label}.")
        matrices = {
            "train_frame": full_frame,
            "eval_frame": full_frame,
            "full_frame": full_frame,
            "train_encoded": full_encoded,
            "eval_encoded": full_encoded,
            "full_encoded": full_encoded,
            "evaluation_mode": "apparent",
            "metric_name": _metric_name_for_evaluation("apparent"),
            "feature_encoder": feature_encoder,
        }
    (
        matrices["train_encoded"],
        matrices["eval_encoded"],
        matrices["full_encoded"],
    ) = _drop_constant_train_columns_with_full(
        matrices["train_encoded"],
        matrices["eval_encoded"],
        matrices["full_encoded"],
        model_label=model_label,
    )
    matrices["y_train"] = _prepare_sksurv_data(matrices["train_frame"], time_column, event_column)
    matrices["y_eval"] = _prepare_sksurv_data(matrices["eval_frame"], time_column, event_column)
    matrices["y_full"] = _prepare_sksurv_data(matrices["full_frame"], time_column, event_column)
    matrices["imputed_numeric_counts"] = _median_imputed_counts(frame, features, resolved_categorical)
    matrices["categorical_features"] = resolved_categorical
    matrices["unseen_category_rows"] = (
        _unseen_category_rows(matrices["train_frame"], matrices["eval_frame"], matrices["categorical_features"])
        if matrices["evaluation_mode"] == "holdout"
        else 0
    )
    return matrices


def _cohort_counts(matrices: dict[str, Any], event_column: str) -> dict[str, int]:
    full_frame = matrices["full_frame"]
    eval_frame = matrices["eval_frame"]
    train_frame = matrices["train_frame"]
    return {
        "n_patients": int(full_frame.shape[0]),
        "n_events": int(full_frame[event_column].sum()),
        "n_evaluation_patients": int(eval_frame.shape[0]),
        "n_evaluation_events": int(eval_frame[event_column].sum()),
        "n_fit_patients": int(train_frame.shape[0]),
        "n_fit_events": int(train_frame[event_column].sum()),
    }


def _holdout_brier_metrics(
    model: Any,
    matrices: dict[str, Any],
    *,
    time_column: str,
    event_column: str,
    alpha: float | None = None,
) -> dict[str, Any] | None:
    """IPCW Brier metrics on the evaluation rows, with censoring weights from the fitting rows."""
    eval_frame = matrices["eval_frame"]
    support_frame = matrices["train_frame"] if matrices["evaluation_mode"] == "holdout" else matrices["full_frame"]
    return _maybe_compute_sksurv_brier_metrics(
        eval_frame[time_column].to_numpy(dtype=float),
        eval_frame[event_column].to_numpy(dtype=int),
        model,
        matrices["eval_encoded"],
        alpha=alpha,
        support_times=support_frame[time_column].to_numpy(dtype=float),
        support_events=support_frame[event_column].to_numpy(dtype=int),
    )


def _fitted_model_result(
    *,
    model_type: str,
    model_name: str,
    model: Any,
    matrices: dict[str, Any],
    event_column: str,
    evaluation_risk_scores: np.ndarray,
    risk_scores: np.ndarray,
    importance_records: list[dict[str, Any]],
    importance_method: str,
    brier_result: dict[str, Any] | None,
    model_settings: dict[str, Any],
    training_time_ms: float,
    extra_strengths: Sequence[str] | None = None,
    extra_cautions: Sequence[str] | None = None,
    extra_payload: dict[str, Any] | None = None,
    brier_computed: bool = True,
) -> dict[str, Any]:
    """Assemble the shared result payload of a fitted RSF, GBS, or LASSO-Cox model.

    With ``brier_computed=False`` the caller skipped the Brier metrics on purpose, so the
    summary does not report them as having failed.
    """
    counts = _cohort_counts(matrices, event_column)
    feature_names = list(matrices["train_encoded"].columns)
    c_index = _sksurv_c_index(matrices["y_eval"], np.asarray(evaluation_risk_scores, dtype=float))
    imputed_counts = dict(matrices.get("imputed_numeric_counts") or {})
    scientific_summary = _scientific_summary_ml(
        model_name=model_name,
        c_index=c_index,
        n_patients=counts["n_patients"],
        n_events=counts["n_events"],
        n_features=len(feature_names),
        evaluation_mode=matrices["evaluation_mode"],
        n_evaluation_patients=counts["n_evaluation_patients"],
        n_evaluation_events=counts["n_evaluation_events"],
        n_fit_patients=counts["n_fit_patients"],
        n_fit_events=counts["n_fit_events"],
        extra_strengths=extra_strengths,
        extra_cautions=[
            *(extra_cautions or []),
            _unseen_category_caution(int(matrices.get("unseen_category_rows") or 0), "evaluation"),
            _median_imputation_caution(imputed_counts),
        ],
    )
    if brier_computed:
        scientific_summary = _augment_scientific_summary_with_brier(scientific_summary, brier_result)
    result = {
        "model_type": model_type,
        "model_stats": {
            "c_index": _safe_float(c_index),
            "ibs": None if brier_result is None else _safe_float(brier_result.get("ibs")),
            "null_ibs": None if brier_result is None else _safe_float(brier_result.get("null_ibs")),
            "brier_skill_score": None if brier_result is None else _safe_float(brier_result.get("brier_skill_score")),
            "metric_name": matrices["metric_name"],
            "evaluation_mode": matrices["evaluation_mode"],
            **model_settings,
            "n_patients": counts["n_patients"],
            "n_events": counts["n_events"],
            "n_evaluation_patients": counts["n_evaluation_patients"],
            "n_evaluation_events": counts["n_evaluation_events"],
            "n_features": len(feature_names),
            "training_time_ms": training_time_ms,
        },
        "feature_importance": importance_records,
        "importance_method": importance_method,
        "predicted_risk_scores": [_safe_float(v) for v in risk_scores],
        "evaluation_risk_scores": [_safe_float(v) for v in evaluation_risk_scores],
        "calibration_metrics": brier_result,
        "feature_names": feature_names,
        "removed_features": list(matrices["train_encoded"].attrs.get("removed_features", [])),
        # The features coded as categorical (declared ones plus text features), reference-coded.
        "categorical_features": list(matrices.get("categorical_features") or []),
        # Missing (or infinite) values per numeric feature, replaced by the training rows' median.
        "imputed_numeric_values": imputed_counts,
        "scientific_summary": scientific_summary,
        "_model": model,
        "_X_encoded": matrices["full_encoded"],
        "_X_eval_encoded": matrices["eval_encoded"],
        "_feature_encoder": matrices["feature_encoder"],
        "_analysis_frame": matrices["full_frame"],
        "_analysis_eval_frame": matrices["eval_frame"],
        "_analysis_train_frame": matrices["train_frame"],
        "_y": matrices["y_full"],
        "_y_eval": matrices["y_eval"],
    }
    if extra_payload:
        result.update(extra_payload)
    return result


# ===================================================================
# 2. Random Survival Forest
# ===================================================================


@user_input_boundary
@guarded_prediction_inputs
def train_random_survival_forest(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 6,
    random_state: int = 42,
    internal_evaluation: bool = True,
    *,
    compute_importance: bool = True,
    compute_brier: bool = True,
) -> dict[str, Any]:
    """Train a Random Survival Forest and return model statistics,
    feature importances, predicted risk scores, and a scientific summary.

    Internal callers that only need the fitted model can skip the permutation importance
    (``compute_importance=False``) and the IPCW Brier metrics (``compute_brier=False``).
    """
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for Random Survival Forest. "
            "Install it with: pip install scikit-survival"
        )
    max_depth = _validate_max_depth(max_depth)
    matrices = _prepare_training_matrices(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_state=random_state,
        internal_evaluation=internal_evaluation,
        model_label="Random Survival Forest",
    )
    effective_min_samples_leaf = _effective_tree_min_samples_leaf(
        min_samples_leaf, int(matrices["train_encoded"].shape[0])
    )
    check_rsf_memory(
        matrices["y_train"]["time"],
        n_estimators=n_estimators,
        min_samples_leaf=effective_min_samples_leaf,
        max_depth=max_depth,
    )

    t_start = time.monotonic()
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=effective_min_samples_leaf,
        random_state=random_state,
        n_jobs=_TREE_N_JOBS,
    )
    model.fit(matrices["train_encoded"].to_numpy(), matrices["y_train"])
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)

    return _fitted_model_result(
        model_type="RandomSurvivalForest",
        model_name="Random Survival Forest",
        model=model,
        matrices=matrices,
        event_column=event_column,
        evaluation_risk_scores=_predict_risk_scores(model, matrices["eval_encoded"]),
        risk_scores=_predict_risk_scores(model, matrices["full_encoded"]),
        # Permutation importance per raw feature on the evaluation rows (out of sample with a
        # holdout), computed the same way for RSF and GBS so their rankings are comparable
        # (impurity importance is in-sample, and sksurv >= 0.24 no longer provides it for RSF).
        importance_records=(
            _grouped_permutation_importance(
                model,
                matrices["eval_encoded"],
                matrices["y_eval"],
                matrices["feature_encoder"],
                random_state=random_state,
            )
            if compute_importance
            else []
        ),
        importance_method=(
            _permutation_importance_method(matrices["evaluation_mode"])
            if compute_importance
            else "Not computed for this run."
        ),
        brier_result=(
            _holdout_brier_metrics(model, matrices, time_column=time_column, event_column=event_column)
            if compute_brier
            else None
        ),
        brier_computed=compute_brier,
        model_settings={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": effective_min_samples_leaf,
        },
        training_time_ms=training_time_ms,
        extra_strengths=[
            f"Ensemble of {n_estimators} trees with min_samples_leaf={effective_min_samples_leaf}.",
            "Non-parametric model; no proportional-hazards assumption required.",
        ],
    )


# ===================================================================
# 3. Gradient Boosted Survival Analysis
# ===================================================================


@user_input_boundary
@guarded_prediction_inputs
def train_gradient_boosted_survival(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int | None = 3,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    internal_evaluation: bool = True,
    *,
    compute_importance: bool = True,
    compute_brier: bool = True,
) -> dict[str, Any]:
    """Train a Gradient Boosted Survival model and return model statistics,
    feature importances, predicted risk scores, and a scientific summary.

    Internal callers that only need the fitted model can skip the permutation importance
    (``compute_importance=False``) and the IPCW Brier metrics (``compute_brier=False``).
    """
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for Gradient Boosted Survival. "
            "Install it with: pip install scikit-survival"
        )
    max_depth = _validate_max_depth(max_depth)
    matrices = _prepare_training_matrices(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_state=random_state,
        internal_evaluation=internal_evaluation,
        model_label="Gradient Boosted Survival",
    )
    effective_min_samples_leaf = _effective_tree_min_samples_leaf(
        min_samples_leaf, int(matrices["train_encoded"].shape[0])
    )
    resolved_max_depth = _resolve_gbs_max_depth(max_depth)

    t_start = time.monotonic()
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=resolved_max_depth,
        min_samples_leaf=effective_min_samples_leaf,
        random_state=random_state,
    )
    model.fit(matrices["train_encoded"].to_numpy(), matrices["y_train"])
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)

    return _fitted_model_result(
        model_type="GradientBoostingSurvivalAnalysis",
        model_name="Gradient Boosted Survival",
        model=model,
        matrices=matrices,
        event_column=event_column,
        evaluation_risk_scores=model.predict(matrices["eval_encoded"].to_numpy()),
        risk_scores=model.predict(matrices["full_encoded"].to_numpy()),
        importance_records=(
            _grouped_permutation_importance(
                model,
                matrices["eval_encoded"],
                matrices["y_eval"],
                matrices["feature_encoder"],
                random_state=random_state,
            )
            if compute_importance
            else []
        ),
        importance_method=(
            _permutation_importance_method(matrices["evaluation_mode"])
            if compute_importance
            else "Not computed for this run."
        ),
        brier_result=(
            _holdout_brier_metrics(model, matrices, time_column=time_column, event_column=event_column)
            if compute_brier
            else None
        ),
        brier_computed=compute_brier,
        model_settings={
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": resolved_max_depth,
            "min_samples_leaf": effective_min_samples_leaf,
        },
        training_time_ms=training_time_ms,
        extra_strengths=[
            f"Boosted ensemble with {n_estimators} stages, learning_rate={learning_rate}, max_depth={resolved_max_depth}, min_samples_leaf={effective_min_samples_leaf}.",
            "Boosted Cox model (Cox partial-likelihood loss): the trees allow non-linear effects and interactions, "
            "but the hazards are assumed to be proportional over time.",
        ],
    )


# ===================================================================
# 4. LASSO-Cox
# ===================================================================


@user_input_boundary
@guarded_prediction_inputs
def train_lasso_cox(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    random_state: int = 42,
    internal_evaluation: bool = True,
) -> dict[str, Any]:
    """Train an L1-penalized Cox model for high-dimensional screening."""
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for LASSO-Cox. "
            "Install it with: pip install scikit-survival"
        )
    matrices = _prepare_training_matrices(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_state=random_state,
        internal_evaluation=internal_evaluation,
        model_label="LASSO-Cox",
    )
    alpha_meta = _select_lasso_alpha(
        matrices["train_frame"],
        matrices["train_encoded"],
        features=features,
        categorical_features=matrices["categorical_features"],
        time_column=time_column,
        event_column=event_column,
        random_state=random_state,
    )
    (
        matrices["train_encoded"],
        matrices["eval_encoded"],
        matrices["full_encoded"],
        scaler,
    ) = _standardize_encoded_matrices(
        matrices["train_encoded"],
        matrices["eval_encoded"],
        matrices["full_encoded"],
    )

    t_start = time.monotonic()
    model = _make_lasso_coxnet_model(alpha=alpha_meta["alpha"])
    model.fit(matrices["train_encoded"].to_numpy(), matrices["y_train"])
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)

    coef_vector = _coerce_coxnet_coef_vector(model)
    feature_names = list(matrices["train_encoded"].columns)
    n_active_features = int(np.count_nonzero(np.abs(coef_vector) > 1e-10))
    importance_records = sorted(
        [
            {
                "feature": name,
                "importance": _safe_float(abs(coef)),
                "coefficient": _safe_float(coef),
            }
            for name, coef in zip(feature_names, coef_vector, strict=False)
        ],
        key=lambda row: row["importance"] if row["importance"] is not None else 0.0,
        reverse=True,
    )
    # Without a holdout the rows that chose the penalty are also the evaluation rows.
    holdout = matrices["evaluation_mode"] == "holdout"
    fitting_rows = "the training split" if holdout else "the fitting rows (every analysed patient)"
    inner_folds = int(alpha_meta.get("inner_cv_folds", 0) or 0)
    scored_folds = int(alpha_meta.get("inner_cv_scored_folds", inner_folds) or 0)
    if alpha_meta.get("selection_rule") == "inner_cv_max_mean_c_index":
        selection_strength = (
            f"Penalty selection used {inner_folds}-fold stratified inner cross-validation on {fitting_rows}"
            + (f" ({scored_folds} of the {inner_folds} folds could be scored)" if scored_folds < inner_folds else "")
            + " and kept the alpha with the highest mean inner C-index (ties go to the sparser model)."
        )
    else:
        selection_strength = (
            f"Penalty selection used apparent concordance on {fitting_rows} because they were too few for inner "
            "cross-validation."
        )
    if alpha_meta["selection_mode"] != "inner_cv":
        selection_caution = (
            "Penalty selection fell back to apparent training performance because inner cross-validation was not feasible."
        )
    elif holdout:
        selection_caution = (
            "Penalty selection used only the training split (inner cross-validation); evaluation rows were never used."
        )
    else:
        selection_caution = (
            "Penalty selection used inner cross-validation on the fitting rows, which are also the evaluation rows "
            "here (apparent evaluation), so the reported C-index is optimistic."
        )
    return _fitted_model_result(
        model_type="LassoCox",
        model_name="LASSO-Cox",
        model=model,
        matrices=matrices,
        event_column=event_column,
        evaluation_risk_scores=np.asarray(model.predict(matrices["eval_encoded"].to_numpy()), dtype=float),
        risk_scores=np.asarray(model.predict(matrices["full_encoded"].to_numpy()), dtype=float),
        importance_records=importance_records,
        importance_method="Absolute LASSO-Cox coefficient on the standardized encoded design (one row per encoded column).",
        brier_result=_holdout_brier_metrics(
            model,
            matrices,
            time_column=time_column,
            event_column=event_column,
            alpha=float(alpha_meta["alpha"]),
        ),
        model_settings={
            "alpha": _safe_float(alpha_meta["alpha"]),
            "alpha_selection_mode": alpha_meta["selection_mode"],
            "alpha_selection_rule": alpha_meta.get("selection_rule"),
            "alpha_selection_c_index": _safe_float(alpha_meta["inner_selection_c_index"]),
            "alpha_selection_c_index_se": _safe_float(alpha_meta.get("inner_selection_c_index_se")),
            "alpha_selection_threshold_c_index": _safe_float(alpha_meta.get("selection_threshold_c_index")),
            "n_alpha_candidates": int(alpha_meta["n_alpha_candidates"]),
            "alpha_selection_inner_cv_folds": inner_folds,
            "alpha_selection_inner_cv_scored_folds": scored_folds,
            "n_active_features": n_active_features,
        },
        training_time_ms=training_time_ms,
        extra_strengths=[
            (
                f"L1-penalized Coxnet selected alpha={alpha_meta['alpha']:.4g} "
                f"with {n_active_features} non-zero coefficient(s)."
            ),
            selection_strength,
        ],
        extra_cautions=[
            (
                "This penalized Cox path is intended for predictive screening in wide feature sets. "
                "Do not interpret its shrunk coefficients like inferential Cox PH hazard ratios."
            ),
            selection_caution,
        ],
        extra_payload={"_feature_scaler": scaler},
    )


# ===================================================================
# 5. Model comparison
# ===================================================================


def _rank_by_c_index(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Sort comparison rows best C-index first and rank those that have a C-index.

    Rows without a C-index come last with ``rank`` None (not ranked). Returns the top-ranked
    row, or None when no row has a C-index, so no model can be named the best.
    """
    rows.sort(key=lambda row: row["c_index"] if row.get("c_index") is not None else -1.0, reverse=True)
    rank = 0
    for row in rows:
        if row.get("c_index") is None:
            row["rank"] = None
        else:
            rank += 1
            row["rank"] = rank
    return rows[0] if rows and rows[0].get("c_index") is not None else None


@user_input_boundary
@guarded_prediction_inputs
def compare_survival_models(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train Cox PH, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival
    on a deterministic train/test split and return a comparison table with
    holdout discrimination/error metrics, feature count, and training time for each model.
    """
    categorical_features = list(categorical_features or [])
    _validate_model_feature_columns(features, time_column=time_column, event_column=event_column)
    # An invalid depth would otherwise only drop the tree models from the table.
    max_depth = _validate_max_depth(max_depth)

    # Prepare a common clean frame
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )
    reject_numeric_text_features(frame, features, list(categorical_features or []))
    # Feature types are decided once on the whole cohort, so no split decides a feature's type anew.
    resolved_categorical = _resolved_categorical_features(frame, features, categorical_features)

    n_patients = int(frame.shape[0])
    n_events = int(frame[event_column].sum())
    comparison: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    predictions: dict[str, tuple[list[int], list[float]]] = {}
    train_positions, eval_positions, evaluation_mode = _split_train_test_positions(
        frame,
        event_column,
        random_state=random_state,
    )
    train_frame = frame.iloc[train_positions].reset_index(drop=True)
    test_frame = frame.iloc[eval_positions].reset_index(drop=True)
    split_fingerprint = evaluation_split_fingerprint(
        _frame_source_rows(frame),
        [(train_positions, eval_positions)],
        kind=evaluation_mode,
    )

    model_specs = _ml_model_specs(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    if not SKSURV_AVAILABLE:
        errors.extend(
            [
                {"model": "LASSO-Cox", "error": "scikit-survival is not installed."},
                {"model": "Random Survival Forest", "error": "scikit-survival is not installed."},
                {"model": "Gradient Boosted Survival", "error": "scikit-survival is not installed."},
            ]
        )

    for model_name, fit_fn, extra_kwargs in model_specs:
        raise_if_cancelled()
        try:
            result = fit_fn(
                train_frame,
                test_frame,
                time_column=time_column,
                event_column=event_column,
                features=features,
                categorical_features=resolved_categorical,
                random_state=random_state,
                **extra_kwargs,
            )
            if result.get("test_risk") is not None:
                predictions[model_name] = (result["test_positions"], result["test_risk"])
            comparison.append({
                "model": model_name,
                "c_index": _safe_float(result["c_index"]),
                "ibs": _safe_float(result.get("ibs")),
                "null_ibs": _safe_float(result.get("null_ibs")),
                "brier_skill_score": _safe_float(result.get("brier_skill_score")),
                "n_features": result["n_features"],
                "removed_features": result.get("removed_features", []),
                "n_active_features": result.get("n_active_features"),
                "training_time_ms": result["training_time_ms"],
                "evaluation_mode": evaluation_mode,
                "training_samples": result.get("train_n"),
                "evaluation_samples": result.get("test_n"),
                "train_events": result.get("train_events"),
                "test_events": result.get("test_events"),
            })
        except Exception as exc:
            if _brier_failure_must_propagate(exc):
                raise
            errors.append({"model": model_name, "error": _failure_message(exc)})

    if not comparison:
        raise ValueError(
            "All models failed to train. Errors: "
            + "; ".join(f"{e['model']}: {e['error']}" for e in errors)
        )

    best = _rank_by_c_index(comparison)
    if best is not None:
        ranking_caution = (
            "The top-ranked model was selected and scored on the same evaluation split; treat this as screening "
            "rather than final external validation."
        )
    else:
        ranking_caution = (
            "No model has a C-index on the evaluation split because "
            + (
                "its patients include no comparable pair (no event followed by a longer follow-up)"
                if not _has_comparable_pair(test_frame[time_column], test_frame[event_column])
                else "no model returned finite risk scores for it"
            )
            + ", so the models are not ranked."
        )
    scientific_summary = _scientific_summary_ml(
        model_name=(
            f"Model Comparison Screening (top: {best['model']})" if best is not None else "Model Comparison Screening"
        ),
        c_index=None if best is None else best["c_index"],
        n_patients=n_patients,
        n_events=n_events,
        n_features=(
            best["n_features"] if best is not None else max(int(row.get("n_features") or 0) for row in comparison) or None
        ),
        evaluation_mode=evaluation_mode,
        n_evaluation_patients=int(test_frame.shape[0]),
        n_evaluation_events=int(test_frame[event_column].sum()),
        n_fit_patients=int(train_frame.shape[0]),
        n_fit_events=int(train_frame[event_column].sum()),
        extra_strengths=[
            f"{len(comparison)} model(s) trained and compared with {evaluation_mode} evaluation.",
        ],
        extra_cautions=[
            ranking_caution,
            f"{len(errors)} model(s) failed to train." if errors else None,
        ],
    )
    if best is not None:
        scientific_summary = _augment_scientific_summary_with_brier(
            scientific_summary,
            {
                "ibs": best.get("ibs"),
                "null_ibs": best.get("null_ibs"),
                "brier_skill_score": best.get("brier_skill_score"),
            },
        )
    duplicate_caution = duplicate_identifier_caution(df)
    if duplicate_caution:
        scientific_summary["cautions"].insert(0, duplicate_caution)
    unseen_caution = _unseen_category_caution(
        _unseen_category_rows(train_frame, test_frame, resolved_categorical) if evaluation_mode == "holdout" else 0,
        "evaluation",
    )
    if unseen_caution:
        scientific_summary["cautions"].append(unseen_caution)
    imputed_counts = _median_imputed_counts(frame, features, resolved_categorical)
    imputation_caution = _median_imputation_caution(imputed_counts)
    if imputation_caution:
        scientific_summary["cautions"].append(imputation_caution)

    result = {
        "comparison_table": comparison,
        "errors": errors,
        # The features coded as categorical (declared ones plus text features), reference-coded.
        "categorical_features": resolved_categorical,
        "ranking_complete": (
            len(errors) == 0
            and len(comparison) == len(model_specs)
            and all(row["c_index"] is not None for row in comparison)
        ),
        # Models left out of the ranking: failed fits and models without a C-index.
        "excluded_models": sorted(
            {str(error["model"]) for error in errors} | {str(row["model"]) for row in comparison if row["c_index"] is None}
        ),
        "n_patients": n_patients,
        "n_events": n_events,
        "n_fit_patients": int(train_frame.shape[0]),
        "n_fit_events": int(train_frame[event_column].sum()),
        "n_evaluation_patients": int(test_frame.shape[0]),
        "n_evaluation_events": int(test_frame[event_column].sum()),
        "evaluation_mode": evaluation_mode,
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION if evaluation_mode == "holdout" else None,
        "split_seed": int(random_state),
        "evaluation_split_fingerprint": split_fingerprint,
        "imputed_numeric_values": imputed_counts,
        "scientific_summary": scientific_summary,
        "test_predictions": (
            _test_prediction_block(frame, eval_positions, test_frame, time_column, event_column, predictions)
            if evaluation_mode == "holdout"
            else None
        ),
    }
    result["manuscript_tables"] = build_manuscript_result_tables(result)
    return result


def _test_prediction_block(
    frame: pd.DataFrame,
    positions: np.ndarray,
    test_frame: pd.DataFrame,
    time_column: str,
    event_column: str,
    predictions: dict[str, tuple[list[int], list[float]]],
) -> dict[str, Any] | None:
    """The models' risk scores on the test patients all of them scored, keyed by stored row label."""
    if not predictions:
        return None
    source_rows = _frame_source_rows(frame)
    row_ids = [source_rows[int(position)] if source_rows is not None else int(position) for position in positions]
    return prediction_block(
        row_ids,
        test_frame[time_column].to_numpy(dtype=float),
        test_frame[event_column].astype(int).to_numpy(),
        predictions,
    )


def _encoded_fold_matrices(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    features: Sequence[str],
    categorical_features: Sequence[str] | None,
    model_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode a train/evaluation split with training-only preprocessing.

    Returns the encoded matrices (constant training columns dropped) and the matching
    outcome frames, all re-indexed from zero.
    """
    train_encoded, test_encoded, _ = _encode_train_test_features(
        train_frame,
        test_frame,
        features,
        categorical_features,
    )
    train_encoded, test_encoded = _drop_constant_train_columns(train_encoded, test_encoded, model_label=model_label)
    train_eval = train_frame.loc[train_encoded.index].reset_index(drop=True)
    test_eval = test_frame.loc[test_encoded.index].reset_index(drop=True)
    # Positions of the scored rows in ``test_frame``, so test-set predictions can be matched to patients.
    test_eval.attrs["test_positions"] = [int(label) for label in test_encoded.index]
    if train_encoded.empty or test_encoded.empty:
        raise ValueError(f"No valid rows remain after encoding features for {model_label}.")
    return train_encoded, test_encoded, train_eval, test_eval


def _fold_result(
    *,
    model: str,
    c_index: float | None,
    brier_result: dict[str, Any] | None,
    n_features: int,
    training_time_ms: float,
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    event_column: str,
    extra: dict[str, Any] | None = None,
    risk_score: np.ndarray | None = None,
    removed_features: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Metrics of one model fitted on a training split and scored on its evaluation split.

    With ``risk_score`` the result also carries the evaluation rows' risk scores and their
    positions in the evaluation frame (``test_risk``, ``test_positions``); callers copy only
    the fields they report, so these stay internal unless a comparison uses them.
    """
    predictions = {}
    if risk_score is not None:
        predictions = {
            "test_risk": np.asarray(risk_score, dtype=float).reshape(-1).tolist(),
            "test_positions": list(test_eval.attrs.get("test_positions", range(test_eval.shape[0]))),
        }
    return {
        **predictions,
        "model": model,
        "c_index": _safe_float(c_index),
        "ibs": None if brier_result is None else _safe_float(brier_result.get("ibs")),
        "null_ibs": None if brier_result is None else _safe_float(brier_result.get("null_ibs")),
        "brier_skill_score": None if brier_result is None else _safe_float(brier_result.get("brier_skill_score")),
        "n_features": int(n_features),
        "removed_features": list(removed_features or []),
        **(extra or {}),
        "training_time_ms": training_time_ms,
        "train_n": int(train_eval.shape[0]),
        "test_n": int(test_eval.shape[0]),
        "train_events": int(train_eval[event_column].sum()),
        "test_events": int(test_eval[event_column].sum()),
    }


def _fit_evaluate_cox_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    del random_state
    train_encoded, test_encoded, train_eval, test_eval = _encoded_fold_matrices(
        train_frame,
        test_frame,
        features=features,
        categorical_features=categorical_features,
        model_label="Cox PH",
    )
    # Rank is judged on the training-standardised design: on raw columns the QR tolerance
    # scales with the largest column and drops informative tiny-scale features, and
    # centring exposes combinations that are constant (which the partial likelihood,
    # having no intercept, cannot identify). Standardising per column first leaves the
    # kept columns exactly as standardising them after the pruning would.
    train_encoded, test_encoded, _, _ = _standardize_encoded_matrices(train_encoded, test_encoded)
    train_encoded, test_encoded = _drop_rank_deficient_train_columns(train_encoded, test_encoded)

    train_times = train_eval[time_column].to_numpy(dtype=float)
    train_status = train_eval[event_column].astype(int).to_numpy()
    test_times = test_eval[time_column].to_numpy(dtype=float)
    test_status = test_eval[event_column].astype(int).to_numpy()

    t_start = time.monotonic()
    model = PHReg(train_times, train_encoded.to_numpy(dtype=float), status=train_status, ties="efron")
    results, converged = fit_phreg(model)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    if not converged:
        # Same rule as the inferential Cox workflow: a non-converged
        # partial-likelihood fit is not a valid benchmark entry.
        raise ValueError(
            "Cox PH did not converge on this training split (too many covariates for the number of events, "
            "or separation). Use LASSO-Cox or reduce the feature set."
        )
    param_vector = np.asarray(results.params, dtype=float)
    risk_score = (test_encoded.to_numpy(dtype=float) @ param_vector).astype(float)
    fit_components = [param_vector, risk_score]
    llf_value = float(results.llf) if getattr(results, "llf", None) is not None else np.nan
    if (not np.isfinite(llf_value)) or any(not np.isfinite(component).all() for component in fit_components):
        raise ValueError(
            "Cox PH fit produced non-finite estimates on the shared evaluation path. "
            "This usually means redundant covariates, sparse categories, or quasi-complete separation."
        )

    y_test = _prepare_sksurv_data(test_eval, time_column, event_column)
    # As for the scikit-survival models, a baseline that cannot be built only blanks the Brier metrics.
    predicted_survival_fn = _survival_predictor_or_none(
        lambda: _cox_ph_survival_predictor(results, test_encoded.to_numpy(dtype=float))
    )
    brier_result = (
        None
        if predicted_survival_fn is None
        else _maybe_compute_brier_metrics(
            test_times,
            test_status,
            predicted_survival_fn,
            support_times=train_times,
            support_events=train_status,
        )
    )
    return _fold_result(
        removed_features=list(train_encoded.attrs.get("removed_features", [])),
        model="Cox PH",
        c_index=_sksurv_c_index(y_test, risk_score),
        risk_score=risk_score,
        brier_result=brier_result,
        n_features=len(param_vector),
        training_time_ms=training_time_ms,
        train_eval=train_eval,
        test_eval=test_eval,
        event_column=event_column,
    )


def _fit_evaluate_lasso_cox_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is not installed.")

    train_encoded, test_encoded, train_eval, test_eval = _encoded_fold_matrices(
        train_frame,
        test_frame,
        features=features,
        categorical_features=categorical_features,
        model_label="LASSO-Cox",
    )

    alpha_meta = _select_lasso_alpha(
        train_eval,
        train_encoded,
        features=features,
        categorical_features=categorical_features,
        time_column=time_column,
        event_column=event_column,
        random_state=random_state,
    )
    train_encoded, test_encoded, _, _ = _standardize_encoded_matrices(train_encoded, test_encoded)
    y_train = _prepare_sksurv_data(train_eval, time_column, event_column)

    t_start = time.monotonic()
    model = _make_lasso_coxnet_model(alpha=alpha_meta["alpha"])
    model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    risk_score = np.asarray(model.predict(test_encoded.to_numpy()), dtype=float)
    coef_vector = _coerce_coxnet_coef_vector(model)

    y_test = _prepare_sksurv_data(test_eval, time_column, event_column)
    brier_result = _maybe_compute_sksurv_brier_metrics(
        test_eval[time_column].to_numpy(dtype=float),
        test_eval[event_column].to_numpy(dtype=int),
        model,
        test_encoded,
        alpha=float(alpha_meta["alpha"]),
        support_times=train_eval[time_column].to_numpy(dtype=float),
        support_events=train_eval[event_column].to_numpy(dtype=int),
    )
    return _fold_result(
        removed_features=list(train_encoded.attrs.get("removed_features", [])),
        model="LASSO-Cox",
        c_index=_sksurv_c_index(y_test, risk_score),
        risk_score=risk_score,
        brier_result=brier_result,
        n_features=int(train_encoded.shape[1]),
        training_time_ms=training_time_ms,
        train_eval=train_eval,
        test_eval=test_eval,
        event_column=event_column,
        extra={
            "n_active_features": int(np.count_nonzero(np.abs(coef_vector) > 1e-10)),
            "alpha_selection_rule": alpha_meta.get("selection_rule"),
        },
    )


def _fit_evaluate_rsf_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 6,
    random_state: int = 42,
) -> dict[str, Any]:
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is not installed.")
    max_depth = _validate_max_depth(max_depth)

    train_encoded, test_encoded, train_eval, test_eval = _encoded_fold_matrices(
        train_frame,
        test_frame,
        features=features,
        categorical_features=categorical_features,
        model_label="Random Survival Forest",
    )
    effective_min_samples_leaf = _effective_tree_min_samples_leaf(min_samples_leaf, int(train_encoded.shape[0]))

    y_train = _prepare_sksurv_data(train_eval, time_column, event_column)
    check_rsf_memory(
        y_train["time"],
        n_estimators=n_estimators,
        min_samples_leaf=effective_min_samples_leaf,
        max_depth=max_depth,
    )
    t_start = time.monotonic()
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=effective_min_samples_leaf,
        random_state=random_state,
        n_jobs=_TREE_N_JOBS,
    )
    model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    risk_score = _predict_risk_scores(model, test_encoded)

    y_test = _prepare_sksurv_data(test_eval, time_column, event_column)
    brier_result = _maybe_compute_sksurv_brier_metrics(
        test_eval[time_column].to_numpy(dtype=float),
        test_eval[event_column].to_numpy(dtype=int),
        model,
        test_encoded,
        support_times=train_eval[time_column].to_numpy(dtype=float),
        support_events=train_eval[event_column].to_numpy(dtype=int),
    )
    return _fold_result(
        removed_features=list(train_encoded.attrs.get("removed_features", [])),
        model="Random Survival Forest",
        c_index=_sksurv_c_index(y_test, risk_score),
        risk_score=risk_score,
        brier_result=brier_result,
        n_features=int(train_encoded.shape[1]),
        training_time_ms=training_time_ms,
        train_eval=train_eval,
        test_eval=test_eval,
        event_column=event_column,
    )


def _fit_evaluate_gbs_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int | None = 3,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is not installed.")
    max_depth = _validate_max_depth(max_depth)

    train_encoded, test_encoded, train_eval, test_eval = _encoded_fold_matrices(
        train_frame,
        test_frame,
        features=features,
        categorical_features=categorical_features,
        model_label="Gradient Boosted Survival",
    )
    effective_min_samples_leaf = _effective_tree_min_samples_leaf(min_samples_leaf, int(train_encoded.shape[0]))

    y_train = _prepare_sksurv_data(train_eval, time_column, event_column)
    t_start = time.monotonic()
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=_resolve_gbs_max_depth(max_depth),
        min_samples_leaf=effective_min_samples_leaf,
        random_state=random_state,
    )
    model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    risk_score = model.predict(test_encoded.to_numpy())

    y_test = _prepare_sksurv_data(test_eval, time_column, event_column)
    brier_result = _maybe_compute_sksurv_brier_metrics(
        test_eval[time_column].to_numpy(dtype=float),
        test_eval[event_column].to_numpy(dtype=int),
        model,
        test_encoded,
        support_times=train_eval[time_column].to_numpy(dtype=float),
        support_events=train_eval[event_column].to_numpy(dtype=int),
    )
    return _fold_result(
        removed_features=list(train_encoded.attrs.get("removed_features", [])),
        model="Gradient Boosted Survival",
        c_index=_sksurv_c_index(y_test, risk_score),
        risk_score=risk_score,
        brier_result=brier_result,
        n_features=int(train_encoded.shape[1]),
        training_time_ms=training_time_ms,
        train_eval=train_eval,
        test_eval=test_eval,
        event_column=event_column,
    )


def build_manuscript_result_tables(result: dict[str, Any]) -> dict[str, Any]:
    """Build manuscript-oriented model performance tables from a comparison result."""
    evaluation_mode = str(result.get("evaluation_mode", "unknown"))
    rows: list[dict[str, Any]] = []
    comparison_table = list(result.get("comparison_table", []))
    repeated_cv_mode = evaluation_mode in {"repeated_cv", "repeated_cv_incomplete"}
    include_brier_metrics = any(
        row.get("ibs") is not None or row.get("null_ibs") is not None or row.get("brier_skill_score") is not None
        for row in comparison_table
    )

    def _format_seed_values(values: Any) -> str | None:
        if not values:
            return None
        try:
            ints = [int(value) for value in values]
        except (TypeError, ValueError):
            return None
        return ", ".join(str(value) for value in ints) if ints else None

    include_locked_test = any("locked_test_c_index" in row for row in comparison_table)
    include_active_features = any(row.get("n_active_features") is not None for row in comparison_table)

    def _features_cell(row: dict[str, Any]) -> Any:
        active = row.get("n_active_features")
        if active is None:
            return row.get("n_features")
        return f"{int(active)} active of {row.get('n_features')}"

    def _rank_cell(row: dict[str, Any], position: int) -> Any:
        """The row's rank (its position when the row carries none); a row without a C-index is not ranked."""
        rank_value = row.get("rank", position)
        if rank_value is None or _safe_float(row.get("c_index")) is None:
            return "Not ranked"
        return rank_value

    any_c_index = any(_safe_float(row.get("c_index")) is not None for row in comparison_table)

    if repeated_cv_mode:
        interval_label = "Fold-level 2.5th-97.5th percentile range"
        include_provenance = any(
            row.get("training_seeds") or row.get("split_seeds") or row.get("monitor_seeds")
            for row in comparison_table
        )
        include_fallback_counts = any(int(row.get("n_apparent_fallbacks", 0) or 0) > 0 for row in comparison_table)
        for rank, row in enumerate(comparison_table, start=1):
            interval_lower = _safe_float(row.get("c_index_interval_lower"))
            interval_upper = _safe_float(row.get("c_index_interval_upper"))
            row_mode = str(row.get("evaluation_mode", evaluation_mode))
            manuscript_row = {
                "Rank": _rank_cell(row, rank),
                "Model": row["model"],
                "Validation Strategy": (
                    f"{row.get('cv_repeats', 1)}x{row.get('cv_folds', 1)} repeated stratified CV"
                    if row_mode == "repeated_cv"
                    else f"{row.get('cv_repeats', 1)}x{row.get('cv_folds', 1)} repeated stratified CV (incomplete)"
                ),
                "Mean C-index": _safe_float(row.get("c_index")),
                "SD (fold-level)": _safe_float(row.get("c_index_std")),
                "Repeats, n": row.get("n_repeats"),
                interval_label: (
                    f"{interval_lower:.3f} to {interval_upper:.3f}"
                    if interval_lower is not None and interval_upper is not None
                    else None
                ),
                "Evaluations, n": row.get("n_evaluations"),
                "Failures, n": row.get("n_failures"),
                "Features, n": _features_cell(row) if include_active_features else row.get("n_features"),
                "Patients, n": result.get("n_patients"),
                "Events, n": result.get("n_events"),
                "Mean Training Patients, n": row.get("training_samples"),
                "Mean Training Events, n": row.get("train_events"),
                "Mean Evaluation Patients, n": row.get("evaluation_samples"),
                "Mean Evaluation Events, n": row.get("test_events"),
                "Mean Training Time, ms": _safe_float(row.get("training_time_ms")),
            }
            if include_brier_metrics:
                manuscript_row["Mean IBS"] = _safe_float(row.get("ibs"))
                manuscript_row["Mean Null IBS"] = _safe_float(row.get("null_ibs"))
                manuscript_row["Pooled Brier Skill Score"] = _safe_float(row.get("brier_skill_score"))
            if include_locked_test:
                manuscript_row["Locked-test C-index"] = _safe_float(row.get("locked_test_c_index"))
                if include_brier_metrics:
                    manuscript_row["Locked-test IBS"] = _safe_float(row.get("locked_test_ibs"))
                    manuscript_row["Locked-test Brier Skill Score"] = _safe_float(row.get("locked_test_brier_skill_score"))
                manuscript_row["Locked-test Patients, n"] = row.get("locked_test_samples")
                manuscript_row["Locked-test Events, n"] = row.get("locked_test_events")
            if include_fallback_counts:
                manuscript_row["Apparent fallback folds, n"] = int(row.get("n_apparent_fallbacks", 0) or 0)
            if include_provenance:
                manuscript_row["Training seeds"] = _format_seed_values(row.get("training_seeds"))
                manuscript_row["Split seeds"] = _format_seed_values(row.get("split_seeds"))
                manuscript_row["Monitor seeds"] = _format_seed_values(row.get("monitor_seeds"))
            rows.append(manuscript_row)
        table_notes = [
            "Mean C-index averages all fold-level estimates from repeated stratified cross-validation; SD is the standard deviation across those fold-level estimates.",
            "The fold-level percentile range reports the 2.5th to 97.5th percentiles of fold C-index values. Folds share training data, so neither the SD nor the range is a formal confidence interval.",
            "Blank C-index fields indicate incomplete repeated-CV evaluation because one or more folds failed or fell back to apparent evaluation.",
        ]
        if include_locked_test:
            table_notes.insert(
                0,
                str(
                    result.get("locked_test_note")
                    or "A stratified locked test set was reserved before any fitting; cross-validation used only the development set."
                ),
            )
            table_notes.append(
                "Models are ranked by development-set cross-validation. Report the locked-test C-index of the CV-selected (rank 1) model as the independent test performance."
                if any_c_index
                else "No model has a cross-validated C-index, so the models are not ranked and there is no CV-selected model whose locked-test C-index could be reported."
            )
        if evaluation_mode == "repeated_cv_incomplete":
            table_notes.append(
                "This comparison is labeled repeated-CV incomplete because one or more apparent-fallback or failed folds were excluded from the aggregate."
            )
        n_skipped_folds = int(result.get("n_skipped_folds") or 0)
        if n_skipped_folds:
            table_notes.append(
                f"{n_skipped_folds} test fold(s) had no comparable pair of patients, so no model has a C-index there; "
                "they were skipped for every model alike, and Evaluations, n counts the folds that were scored."
            )
        if include_provenance:
            table_notes.append(
                "Training, split, and monitor seed columns record the exact repeated-CV partitioning used for replay under the same model settings."
            )
        if include_fallback_counts:
            table_notes.append(
                "Apparent fallback fold counts report folds that were excluded from the repeated-CV aggregate because a clean holdout concordance estimate could not be retained."
            )
        if include_brier_metrics:
            table_notes.append(
                "IBS columns summarize IPCW integrated Brier error; the pooled Brier Skill Score is 1 - mean IBS / mean null IBS, with the Kaplan-Meier null model fit on each training partition."
            )
    else:
        row_modes = {str(row.get("evaluation_mode", evaluation_mode)) for row in comparison_table}
        for rank, row in enumerate(comparison_table, start=1):
            manuscript_row = {
                "Rank": _rank_cell(row, rank),
                "Model": row["model"],
                "Validation Strategy": _manuscript_validation_strategy_label(str(row.get("evaluation_mode", evaluation_mode))),
                "C-index": _safe_float(row.get("c_index")),
                "Features, n": _features_cell(row) if include_active_features else row.get("n_features"),
                "Patients, n": result.get("n_patients"),
                "Events, n": result.get("n_events"),
                "Evaluation Patients, n": row.get("evaluation_samples", result.get("n_evaluation_patients")),
                "Evaluation Events, n": row.get("test_events", result.get("n_evaluation_events")),
                "Training Time, ms": _safe_float(row.get("training_time_ms")),
            }
            if include_brier_metrics:
                manuscript_row["IBS"] = _safe_float(row.get("ibs"))
                manuscript_row["Null IBS"] = _safe_float(row.get("null_ibs"))
                manuscript_row["Brier Skill Score"] = _safe_float(row.get("brier_skill_score"))
            rows.append(manuscript_row)
        table_notes = [
            "C-index values come from a single deterministic holdout split unless evaluation_mode is apparent.",
            "Apparent evaluation indicates resubstitution on the analyzable cohort and should not be interpreted as external validation.",
        ]
        if include_brier_metrics:
            table_notes.append(
                "IBS and Brier Skill Score use IPCW error on the evaluation cohort, with Brier Skill Score defined relative to a Kaplan-Meier null model from the training cohort."
            )
        if len(row_modes) > 1:
            table_notes.append(
                "Rows mix deterministic holdout and apparent fallback evaluation; apparent-fallback rows are shown for transparency but should not be ranked against holdout rows."
            )
        elif row_modes and row_modes <= {"apparent", "holdout_fallback_apparent"}:
            table_notes[0] = (
                "C-index values come from apparent evaluation because a stable holdout estimate was not available."
            )

    return {
        "model_performance_table": rows,
        "table_notes": table_notes,
        "caption": (
            "Table 1. Performance of survival models: repeated stratified cross-validation on the development set and a single evaluation on a locked independent test set."
            if include_locked_test and evaluation_mode == "repeated_cv"
            else "Table 1. Performance of survival models under repeated stratified cross-validation."
            if evaluation_mode == "repeated_cv"
            else (
                "Table 1. Performance of survival models under incomplete repeated stratified cross-validation."
                if evaluation_mode == "repeated_cv_incomplete"
                else (
                    "Table 1. Performance of survival models under apparent evaluation."
                    if comparison_table and all(str(row.get("evaluation_mode", evaluation_mode)) in {"apparent", "holdout_fallback_apparent"} for row in comparison_table)
                    else (
                        "Table 1. Performance of survival models under mixed holdout/apparent evaluation."
                        if any(str(row.get("evaluation_mode", evaluation_mode)) != "holdout" for row in comparison_table)
                        else "Table 1. Performance of survival models under deterministic holdout evaluation."
                    )
                )
            )
        ),
    }


def _ml_model_specs(
    *,
    n_estimators: int,
    max_depth: int | None,
    learning_rate: float,
) -> list[tuple[str, Any, dict[str, Any]]]:
    specs: list[tuple[str, Any, dict[str, Any]]] = [("Cox PH", _fit_evaluate_cox_split, {})]
    if SKSURV_AVAILABLE:
        specs.extend(
            [
                ("LASSO-Cox", _fit_evaluate_lasso_cox_split, {}),
                (
                    "Random Survival Forest",
                    _fit_evaluate_rsf_split,
                    {"n_estimators": n_estimators, "max_depth": max_depth},
                ),
                (
                    "Gradient Boosted Survival",
                    _fit_evaluate_gbs_split,
                    {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": _resolve_gbs_max_depth(max_depth),
                    },
                ),
            ]
        )
    return specs


def _locked_test_note(n_dev: int, n_dev_events: int, n_test: int, n_test_events: int, fraction: float) -> str:
    return (
        f"A stratified locked test set ({n_test} patients, {n_test_events} events; {fraction:.0%} of the cohort) "
        f"was reserved before any fitting. Repeated CV, preprocessing, and tuning used only the development set "
        f"({n_dev} patients, {n_dev_events} events); each model was then refit once on the full development set "
        "and scored once on the locked test set."
    )


@user_input_boundary
@guarded_prediction_inputs
def cross_validate_survival_models(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    cv_folds: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42,
    locked_test_fraction: float | None = None,
) -> dict[str, Any]:
    """Evaluate Cox PH, LASSO-Cox, RSF, and GBS with repeated stratified cross-validation.

    With ``locked_test_fraction`` set (a fraction from 0.05 to 0.5), a stratified test
    set is reserved first; repeated CV runs on the remaining development set only, and
    every model is refit on the development set and scored once on the untouched test set.
    """
    _require_sklearn()
    categorical_features = list(categorical_features or [])
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if cv_repeats < 1:
        raise ValueError("cv_repeats must be at least 1.")
    if cv_folds * cv_repeats > 200:
        raise ValueError("cv_folds * cv_repeats must not exceed 200 total evaluations.")
    _validate_model_feature_columns(features, time_column=time_column, event_column=event_column)
    # An invalid depth would otherwise only drop the tree models from the table.
    max_depth = _validate_max_depth(max_depth)
    locked_test_fraction = _validated_locked_test_fraction(locked_test_fraction)

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )
    reject_numeric_text_features(frame, features, list(categorical_features or []))
    # Feature types are decided once on the whole cohort, so no fold decides a feature's type anew.
    resolved_categorical = _resolved_categorical_features(frame, features, categorical_features)
    imputed_counts = _median_imputed_counts(frame, features, resolved_categorical)
    n_patients = int(frame.shape[0])
    n_events = int(frame[event_column].sum())
    source_rows = _frame_source_rows(frame)
    all_events = frame[event_column].astype(int).to_numpy()

    use_locked_test = locked_test_fraction is not None
    if use_locked_test:
        dev_positions, test_positions = locked_test_split(
            all_events,
            random_state=random_state,
            test_fraction=float(locked_test_fraction),
        )
    else:
        dev_positions = np.arange(n_patients, dtype=int)
        test_positions = np.array([], dtype=int)
    dev_frame = frame.iloc[dev_positions].reset_index(drop=True)
    locked_test_frame = frame.iloc[test_positions].reset_index(drop=True) if use_locked_test else None

    events = dev_frame[event_column].astype(int).to_numpy()
    unique, counts = np.unique(events, return_counts=True)
    if len(unique) < 2 or counts.min() < cv_folds:
        raise ValueError(
            f"Repeated CV requires at least {cv_folds} samples in each event stratum after cleaning"
            + (" in the development set." if use_locked_test else ".")
        )

    model_specs = _ml_model_specs(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    # Models that could not run at all are listed apart from fold-level failures.
    unavailable_errors: list[dict[str, Any]] = []
    if not SKSURV_AVAILABLE:
        unavailable_errors = [
            {"model": "LASSO-Cox", "error": "scikit-survival is not installed."},
            {"model": "Random Survival Forest", "error": "scikit-survival is not installed."},
            {"model": "Gradient Boosted Survival", "error": "scikit-survival is not installed."},
        ]
    errors: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    design_splits: list[tuple[np.ndarray, np.ndarray]] = []
    skipped_folds: list[dict[str, int]] = []
    unseen_fold_rows = 0

    for repeat_idx in range(cv_repeats):
        repeat_seed = _derived_seed(random_state, repeat_idx)
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dev_frame, events), start=1):
            design_splits.append((dev_positions[train_idx], dev_positions[test_idx]))
            train_frame = dev_frame.iloc[train_idx].reset_index(drop=True)
            test_frame = dev_frame.iloc[test_idx].reset_index(drop=True)
            if not _has_comparable_pair(test_frame[time_column], test_frame[event_column]):
                # No event in this test fold is followed by a longer follow-up, so the C-index is
                # undefined for every model alike: the fold is skipped for all of them instead of
                # leaving every model incomplete.
                skipped_folds.append({"repeat": repeat_idx + 1, "fold": fold_idx})
                continue
            unseen_fold_rows += _unseen_category_rows(train_frame, test_frame, resolved_categorical)
            for model_name, fit_fn, extra_kwargs in model_specs:
                raise_if_cancelled()
                try:
                    result = fit_fn(
                        train_frame,
                        test_frame,
                        time_column=time_column,
                        event_column=event_column,
                        features=features,
                        categorical_features=resolved_categorical,
                        random_state=repeat_seed,
                        **extra_kwargs,
                    )
                    if result["c_index"] is None:
                        # The fold has comparable pairs, so only this model's scores are at fault.
                        errors.append({
                            "model": model_name,
                            "repeat": repeat_idx + 1,
                            "fold": fold_idx,
                            "error": "The C-index could not be computed on this test fold because the model's risk "
                            "scores were not all finite.",
                        })
                    fold_results.append({
                        "model": model_name,
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx,
                        "c_index": result["c_index"],
                        "ibs": result.get("ibs"),
                        "null_ibs": result.get("null_ibs"),
                        "brier_skill_score": result.get("brier_skill_score"),
                        "n_features": result["n_features"],
                        "removed_features": result.get("removed_features", []),
                        "n_active_features": result.get("n_active_features"),
                        "training_time_ms": result["training_time_ms"],
                        "train_n": result["train_n"],
                        "test_n": result["test_n"],
                        "train_events": result["train_events"],
                        "test_events": result["test_events"],
                    })
                except Exception as exc:
                    if _brier_failure_must_propagate(exc):
                        raise
                    errors.append({
                        "model": model_name,
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx,
                        "error": _failure_message(exc),
                    })

    n_total_folds = cv_folds * cv_repeats
    n_scored_folds = n_total_folds - len(skipped_folds)
    if n_scored_folds == 0:
        raise ValueError(
            "No cross-validation test fold had a comparable pair of patients (an event followed by a longer "
            "follow-up), so the C-index is undefined in every fold. Use fewer folds or a cohort with more events."
        )

    locked_results: dict[str, dict[str, Any]] = {}
    if use_locked_test and locked_test_frame is not None:
        design_splits.append((dev_positions, test_positions))
        for model_name, fit_fn, extra_kwargs in model_specs:
            raise_if_cancelled()
            try:
                locked_results[model_name] = fit_fn(
                    dev_frame,
                    locked_test_frame,
                    time_column=time_column,
                    event_column=event_column,
                    features=features,
                    categorical_features=resolved_categorical,
                    random_state=random_state,
                    **extra_kwargs,
                )
            except Exception as exc:
                if _brier_failure_must_propagate(exc):
                    raise
                locked_results[model_name] = {"error": _failure_message(exc)}

    comparison: list[dict[str, Any]] = []
    for model_name, _, _ in model_specs:
        model_rows = [row for row in fold_results if row["model"] == model_name and row["c_index"] is not None]
        n_failures = sum(1 for err in errors if err["model"] == model_name)
        summary = _summarize_repeated_cv_rows(model_rows) if model_rows else None
        # Complete means scored on every fold that had a comparable pair; folds skipped for
        # every model alike do not make a model incomplete.
        incomplete = len(model_rows) < n_scored_folds or n_failures > 0
        if summary is None and n_failures == 0:
            continue
        active_counts = [float(row["n_active_features"]) for row in model_rows if row.get("n_active_features") is not None]
        row_payload = {
            "model": model_name,
            **repeated_cv_row_fields(summary, incomplete=incomplete),
            "n_active_features": int(round(float(np.mean(active_counts)))) if active_counts else None,
            "removed_features": [dict(column=column, reason=reason) for column, reason in sorted({
                (item["column"], item["reason"]) for fit in model_rows for item in fit.get("removed_features", [])
            })],
            "n_evaluations": len(model_rows),
            "n_failures": n_failures,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
        }
        if use_locked_test:
            locked = locked_results.get(model_name) or {}
            row_payload.update({
                "locked_test_c_index": _safe_float(locked.get("c_index")),
                "locked_test_ibs": _safe_float(locked.get("ibs")),
                "locked_test_null_ibs": _safe_float(locked.get("null_ibs")),
                "locked_test_brier_skill_score": _safe_float(locked.get("brier_skill_score")),
                "locked_test_samples": locked.get("test_n"),
                "locked_test_events": locked.get("test_events"),
                "locked_test_training_samples": locked.get("train_n"),
                "locked_test_error": locked.get("error"),
            })
        comparison.append(row_payload)

    if not comparison:
        raise ValueError(
            "All repeated-CV model fits failed. Errors: "
            + "; ".join(
                f"{e['model']} r{e.get('repeat', '?')}f{e.get('fold', '?')}: {e['error']}" for e in errors
            )
        )

    # None when no model was scored on every fold: then no model is named the top or CV-selected one.
    best = _rank_by_c_index(comparison)
    mean_train_n = int(round(np.mean([row["train_n"] for row in fold_results]))) if fold_results else n_patients
    mean_test_n = int(round(np.mean([row["test_n"] for row in fold_results]))) if fold_results else n_patients
    mean_train_events = int(round(np.mean([row["train_events"] for row in fold_results]))) if fold_results else n_events
    mean_test_events = int(round(np.mean([row["test_events"] for row in fold_results]))) if fold_results else n_events
    # Only fold-level failures make the cross-validation incomplete; models that could not run
    # at all are reported apart.
    aggregate_mode = (
        "repeated_cv_incomplete"
        if errors or any(str(row.get("evaluation_mode")) == "repeated_cv_incomplete" for row in comparison)
        else "repeated_cv"
    )
    extra_strengths = [
        f"{len(comparison)} model(s) evaluated across {cv_repeats} repeat(s) of {cv_folds}-fold stratified CV"
        + (" on the development set." if use_locked_test else "."),
    ]
    locked_note = None
    if use_locked_test and locked_test_frame is not None:
        locked_note = _locked_test_note(
            int(dev_frame.shape[0]),
            int(dev_frame[event_column].sum()),
            int(locked_test_frame.shape[0]),
            int(locked_test_frame[event_column].sum()),
            float(locked_test_fraction),
        )
        extra_strengths.append(locked_note)
        best_locked = None if best is None else best.get("locked_test_c_index")
        if best is not None and best_locked is not None:
            extra_strengths.append(
                f"The CV-selected model ({best['model']}) reached a locked-test C-index of {best_locked:.3f}; "
                "this single untouched-test estimate is the performance to report."
            )
    # A model that fails when refit on the development set has no locked-test estimate; the
    # failure is an error of the run, but it does not make the cross-validation incomplete.
    locked_errors = [
        {"model": model_name, "stage": "locked_test", "error": str(locked_results[model_name]["error"])}
        for model_name, _, _ in model_specs
        if (locked_results.get(model_name) or {}).get("error") is not None
    ]
    extra_cautions: list[str] = [f"{len(errors)} fold-level fit(s) failed."] if errors else []
    if unavailable_errors:
        extra_cautions.append(
            f"{len(unavailable_errors)} model(s) were not evaluated because scikit-survival is not installed ("
            + ", ".join(str(error["model"]) for error in unavailable_errors)
            + ")."
        )
    if skipped_folds:
        extra_cautions.append(
            f"{len(skipped_folds)} of {n_total_folds} cross-validation test folds had no comparable pair of patients "
            "(no event followed by a longer follow-up), so the C-index is undefined there for every model; those folds "
            f"were skipped for all models alike, and the cross-validated means use the other {n_scored_folds} folds."
        )
    if locked_test_frame is not None and not _has_comparable_pair(
        locked_test_frame[time_column], locked_test_frame[event_column]
    ):
        extra_cautions.append(
            "The locked test set has no comparable pair of patients (no event followed by a longer follow-up), so no "
            "model has a locked-test C-index; use a larger locked test fraction or a cohort with more events."
        )
    if locked_errors:
        failed_names = ", ".join(error["model"] for error in locked_errors)
        extra_cautions.append(
            f"{len(locked_errors)} model(s) failed when refit on the development set and scored on the locked test set "
            f"({failed_names}); their locked-test C-index is blank"
            + (
                f", including the CV-selected model ({best['model']}), so this run has no untouched-test estimate to report."
                if best is not None and any(error["model"] == best["model"] for error in locked_errors)
                else "."
            )
        )
    scientific_summary = _scientific_summary_ml(
        model_name=(
            f"Repeated-CV Model Comparison Screening (top: {best['model']})"
            if best is not None
            else "Repeated-CV Model Comparison Screening"
        ),
        c_index=None if best is None else best["c_index"],
        n_patients=n_patients,
        n_events=n_events,
        n_features=(
            best.get("n_features")
            if best is not None
            else max(int(row.get("n_features") or 0) for row in comparison) or None
        ),
        evaluation_mode=aggregate_mode,
        n_evaluation_patients=mean_test_n,
        n_evaluation_events=mean_test_events,
        n_fit_patients=mean_train_n,
        n_fit_events=mean_train_events,
        extra_strengths=extra_strengths,
        extra_cautions=extra_cautions or None,
        counts_are_fold_means=True,
    )
    if best is not None:
        scientific_summary = _augment_scientific_summary_with_brier(
            scientific_summary,
            {
                "ibs": best.get("ibs"),
                "null_ibs": best.get("null_ibs"),
                "brier_skill_score": best.get("brier_skill_score"),
            },
        )
    if best is None:
        scientific_summary["cautions"].insert(
            0,
            "No model was scored on every cross-validation fold, so the models are not ranked and there is no "
            + ("CV-selected model whose locked-test C-index could be reported." if use_locked_test else "top-ranked model."),
        )
    elif use_locked_test:
        scientific_summary["cautions"].insert(
            0,
            "Models were ranked by development-set repeated CV; report the locked-test C-index of the CV-selected model as the independent performance estimate, not the best locked-test value across models.",
        )
    else:
        scientific_summary["cautions"].insert(
            0,
            "The top-ranked model was selected and scored within the same repeated-CV screening run; treat this as model screening rather than final external validation. Reserve a locked test set or use an external cohort for the performance you report.",
        )
    duplicate_caution = duplicate_identifier_caution(df)
    if duplicate_caution:
        scientific_summary["cautions"].insert(0, duplicate_caution)
    for caution in (
        _unseen_category_caution(unseen_fold_rows, "cross-validation evaluation (summed over folds)"),
        _unseen_category_caution(
            _unseen_category_rows(dev_frame, locked_test_frame, resolved_categorical)
            if locked_test_frame is not None
            else 0,
            "locked-test",
        ),
        _median_imputation_caution(imputed_counts),
    ):
        if caution:
            scientific_summary["cautions"].append(caution)

    all_errors = [*unavailable_errors, *errors, *locked_errors]
    fingerprint_kind = "repeated_cv+locked_test" if use_locked_test else "repeated_cv"
    result = {
        "comparison_table": comparison,
        "fold_results": fold_results,
        "repeat_results": [row["repeat_results"] for row in comparison],
        "errors": all_errors,
        # The features coded as categorical (declared ones plus text features), reference-coded.
        "categorical_features": list(resolved_categorical),
        "ranking_complete": not all_errors and all(row.get("c_index") is not None for row in comparison),
        # Models left out of (or incomplete in) the cross-validation ranking: models that could
        # not run, failed on a fold, or lack a fold the other models were scored on. A model
        # whose locked-test refit failed is still ranked and appears only in ``errors``.
        "excluded_models": sorted(
            {str(error["model"]) for error in [*unavailable_errors, *errors]}
            | {str(row["model"]) for row in comparison if row.get("evaluation_mode") == "repeated_cv_incomplete"}
        ),
        "n_patients": n_patients,
        "n_events": n_events,
        "evaluation_mode": aggregate_mode,
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        # Test folds without a comparable pair of patients, skipped for every model alike.
        "n_skipped_folds": len(skipped_folds),
        "skipped_folds": skipped_folds,
        "imputed_numeric_values": imputed_counts,
        "split_seed": int(random_state),
        "locked_test_fraction": float(locked_test_fraction) if use_locked_test else None,
        "n_development_patients": int(dev_frame.shape[0]),
        "n_development_events": int(dev_frame[event_column].sum()),
        "n_locked_test_patients": int(locked_test_frame.shape[0]) if locked_test_frame is not None else None,
        "n_locked_test_events": int(locked_test_frame[event_column].sum()) if locked_test_frame is not None else None,
        "locked_test_note": locked_note,
        "evaluation_split_fingerprint": evaluation_split_fingerprint(source_rows, design_splits, kind=fingerprint_kind),
        "scientific_summary": scientific_summary,
        "locked_test_predictions": (
            _test_prediction_block(
                frame,
                test_positions,
                locked_test_frame,
                time_column,
                event_column,
                {
                    name: (locked["test_positions"], locked["test_risk"])
                    for name, locked in locked_results.items()
                    if locked.get("test_risk") is not None
                },
            )
            if use_locked_test and locked_test_frame is not None
            else None
        ),
    }
    result["manuscript_tables"] = build_manuscript_result_tables(result)
    return result


# ===================================================================
# 5. SHAP values
# ===================================================================


# Kernel SHAP samples its feature coalitions with ``np.random.choice`` and
# ``np.random.permutation``, the process-wide random state. A run gives the explainer's
# module a view of NumPy whose ``random`` is a seeded local RandomState instead, so the
# attributions are reproducible and other code's random state is neither reseeded nor
# consumed. Runs hold this lock because they share that module.
_KERNEL_SHAP_RNG_LOCK = threading.Lock()
_DEFAULT_SHAP_SEED = 42


class _NumpyWithLocalRandom:
    """``numpy`` as seen by the Kernel SHAP module, with ``numpy.random`` one seeded RandomState."""

    def __init__(self, seed: int) -> None:
        # A RandomState draws the same numbers as the global functions after ``np.random.seed(seed)``.
        self.random = np.random.RandomState(int(seed) % _SEED_MODULUS)

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


@contextmanager
def _kernel_shap_local_rng(explainer_class: Any, seed: int) -> Iterator[None]:
    module = sys.modules.get(str(getattr(explainer_class, "__module__", "")))
    if module is None or not module.__name__.startswith("shap."):
        # A stand-in explainer, not shap's: nothing there samples from NumPy's random state.
        yield
        return
    with _KERNEL_SHAP_RNG_LOCK:
        original = module.__dict__.get("np")
        if original is not np:
            # This shap version does not sample through a module-level ``np``.
            yield
            return
        module.np = _NumpyWithLocalRandom(seed)
        try:
            yield
        finally:
            module.np = original


def _shap_seed(model: Any, random_state: int | None) -> int:
    """The explicit seed, else the fitted model's integer ``random_state``, else a fixed default."""
    if random_state is not None:
        return int(random_state)
    model_seed = getattr(model, "random_state", None)
    if isinstance(model_seed, (int, np.integer)) and not isinstance(model_seed, bool):
        return int(model_seed)
    return _DEFAULT_SHAP_SEED


@user_input_boundary
def compute_shap_values(
    model: Any,
    X_encoded: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    *,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Compute SHAP values for a fitted sklearn-compatible survival model.

    Parameters
    ----------
    model
        A fitted tree-based model (RSF or GBS from scikit-survival).
    X_encoded
        The encoded feature matrix used for training (or a subset thereof).
    feature_names
        Optional explicit feature names; defaults to ``X_encoded.columns``.
    random_state
        Seed of the approximate Kernel SHAP fallback, which samples feature coalitions;
        defaults to the fitted model's ``random_state``. The same seed gives the same
        attributions, and the seed used is reported as ``random_state``.

    Returns
    -------
    dict
        ``feature_importance`` (mean |SHAP| per feature, sorted descending)
        and ``shap_summary`` (per-instance SHAP values for the top features).
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "shap is required for SHAP explanations. "
            "Install it with: pip install shap"
        )

    if feature_names is None:
        feature_names = list(X_encoded.columns)

    _require_predict_callable(model, context="SHAP explanations")
    X_array = X_encoded.to_numpy() if isinstance(X_encoded, pd.DataFrame) else np.asarray(X_encoded)

    shap_method = "tree"
    stability = "native_tree"
    usage_note = "TreeExplainer ran on the encoded evaluation matrix."
    X_eval = X_array
    background_samples = None
    seed = _shap_seed(model, random_state)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_array)
    except Exception as exc:
        # Exhausted memory, cancellation and bugs are not an unsupported model.
        if _brier_failure_must_propagate(exc):
            raise
        # scikit-survival estimators are often unsupported by TreeExplainer.
        # Fall back to a capped KernelExplainer run only for moderate feature
        # counts. High-dimensional Kernel SHAP is too unstable for reporting.
        shap_method = "kernel"
        stability = "approximate_screening_only"
        n = int(X_array.shape[0])
        encoded_feature_count = int(X_array.shape[1])
        if encoded_feature_count > 80:
            raise ValueError(
                "TreeExplainer is unavailable for this fitted model, and approximate Kernel SHAP "
                f"is disabled for high-dimensional inputs ({encoded_feature_count} encoded features). "
                "Reduce the ML feature set or rely on the model's built-in importance ranking instead."
            )
        risk_scores = _predict_risk_scores(model, X_array).reshape(-1)
        bg_n = min(40, n)
        eval_n = min(60, n)
        bg_idx = _representative_subsample_indices(risk_scores, bg_n)
        eval_idx = _representative_subsample_indices(risk_scores, eval_n)
        X_bg = X_array[bg_idx]
        X_eval = X_array[eval_idx]
        background_samples = int(X_bg.shape[0])
        usage_note = (
            "Kernel SHAP was approximated on representative subsamples of the encoded evaluation matrix "
            f"(coalitions sampled with seed {seed}). Use these attributions for screening rather than manuscript claims."
        )

        def _predict_fn(x: np.ndarray) -> np.ndarray:
            return _predict_risk_scores(model, x)

        kernel_nsamples = min(160, max(40, X_bg.shape[1] * 6))
        with _kernel_shap_local_rng(shap.KernelExplainer, seed):
            explainer = shap.KernelExplainer(_predict_fn, X_bg)
            # shap's default l1_reg ("num_features(10)") would keep at most ten non-zero
            # attributions per patient; every encoded feature is estimated instead.
            try:
                shap_values = explainer.shap_values(
                    X_eval,
                    nsamples=kernel_nsamples,
                    l1_reg=False,
                    silent=True,
                )
            except TypeError:
                shap_values = explainer.shap_values(X_eval, nsamples=kernel_nsamples, l1_reg=False)

    # shap_values may be 2-D (n_samples, n_features) or 3-D for multi-output
    if isinstance(shap_values, (list, tuple)):
        shap_values = np.asarray(shap_values[-1] if len(shap_values) else shap_values, dtype=float)
    else:
        shap_values = np.asarray(shap_values, dtype=float)

    if shap_values.ndim == 3:
        # Use the last output (typically risk) or average across outputs
        shap_values = shap_values[:, :, -1]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_records = sorted(
        [
            {"feature": name, "mean_abs_shap": _safe_float(val)}
            for name, val in zip(feature_names, mean_abs_shap, strict=False)
        ],
        key=lambda r: r["mean_abs_shap"] if r["mean_abs_shap"] is not None else 0.0,
        reverse=True,
    )

    # Summary data for beeswarm-style plot (top 20 features)
    top_n = min(20, len(feature_names))
    top_features = [r["feature"] for r in importance_records[:top_n]]
    top_indices = [feature_names.index(f) for f in top_features if f in feature_names]

    shap_summary: list[dict[str, Any]] = []
    for idx in top_indices:
        fname = feature_names[idx]
        shap_summary.append({
            "feature": fname,
            "shap_values": [_safe_float(v) for v in shap_values[:, idx]],
            "feature_values": [_safe_float(v) for v in X_eval[:, idx]],
            "mean_abs_shap": _safe_float(mean_abs_shap[idx]),
        })

    return {
        "method": shap_method,
        "stability": stability,
        "usage_note": usage_note,
        "feature_importance": importance_records,
        "shap_summary": shap_summary,
        "n_samples": int(X_eval.shape[0]),
        "background_samples": background_samples,
        "n_features": int(X_array.shape[1]),
        "random_state": seed if shap_method == "kernel" else None,
    }


# ===================================================================
# 6. Partial dependence
# ===================================================================


def _apply_feature_scaler(encoded: pd.DataFrame, scaler: dict[str, Any] | None) -> pd.DataFrame:
    """Re-apply training standardization (LASSO-Cox stores its design standardized)."""
    if not scaler:
        return encoded
    means = pd.Series(scaler.get("mean")).reindex(encoded.columns).fillna(0.0).astype(float)
    scales = pd.Series(scaler.get("scale")).reindex(encoded.columns).fillna(1.0).astype(float).replace(0.0, 1.0)
    scaled = (encoded.astype(float) - means) / scales
    return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _model_risk_scale(model: Any) -> str:
    """How ``model.predict`` scores risk.

    ``"log_hazard"``: a log partial hazard (Cox-type models). ``"summed_cumulative_hazard"``:
    a positive score such as a Random Survival Forest's, the predicted cumulative hazard
    summed over the training event times (the expected number of events).
    """
    if GradientBoostingSurvivalAnalysis is not None and isinstance(model, GradientBoostingSurvivalAnalysis):
        return "log_hazard"
    if CoxnetSurvivalAnalysis is not None and isinstance(model, CoxnetSurvivalAnalysis):
        return "log_hazard"
    return "summed_cumulative_hazard"


@user_input_boundary
def compute_partial_dependence(
    model: Any,
    X_encoded: pd.DataFrame,
    feature_name: str,
    n_points: int = 50,
    categorical_features: Sequence[str] | None = None,
    feature_encoder: dict[str, Any] | None = None,
    analysis_frame: pd.DataFrame | None = None,
    feature_scaler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute partial dependence of the model's risk score on a single
    feature.

    Numeric features are evaluated on an evenly spaced grid. Categorical
    features are evaluated category by category by rebuilding the encoded
    design matrix from the raw analyzable frame.

    Returns
    -------
    dict
        ``feature``, ``feature_type``, ``values`` (grid or categories),
        ``mean_risk`` (averaged predicted risk at each value).
    """
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2 for partial dependence.")
    _require_predict_callable(model, context="Partial dependence")
    n_points = int(n_points)
    # The fitted encoder decides how the model saw a feature: text columns are one-hot
    # encoded even when the request did not list them as categorical.
    categorical_features = _encoder_categorical_features(feature_encoder, categorical_features)
    if isinstance(feature_encoder, dict) and feature_encoder.get("features") is not None:
        model_inputs = [str(feature) for feature in feature_encoder.get("features") or []]
        if str(feature_name) not in model_inputs:
            # Varying a column the model never used would draw a flat, meaningless curve.
            raise ValueError(
                f"Feature '{feature_name}' is not an input of the fitted model. Use one of: "
                + ", ".join(f"'{feature}'" for feature in model_inputs[:20])
                + (", ..." if len(model_inputs) > 20 else "")
                + "."
            )
    computation_errors: list[str] = []

    def _predict_mean_for_variant(frame_variant: pd.DataFrame) -> float | None:
        if feature_encoder is not None:
            encoded = _transform_feature_encoder(frame_variant, feature_encoder)
            encoded = encoded.reindex(columns=X_encoded.columns, fill_value=0.0).fillna(0.0)
            X_variant = _apply_feature_scaler(encoded, feature_scaler).to_numpy(dtype=float)
        else:
            if feature_name not in X_encoded.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found in the encoded data. "
                    f"Available: {list(X_encoded.columns)}"
                )
            X_variant = X_encoded.to_numpy(dtype=float, copy=True)
            col_idx = list(X_encoded.columns).index(feature_name)
            replacement = pd.to_numeric(frame_variant[feature_name], errors="coerce").to_numpy(dtype=float)
            if feature_scaler:
                mean = float(pd.Series(feature_scaler.get("mean")).get(feature_name, 0.0))
                scale = float(pd.Series(feature_scaler.get("scale")).get(feature_name, 1.0)) or 1.0
                replacement = (replacement - mean) / scale
            X_variant[:, col_idx] = replacement
        preds = _predict_risk_scores(model, X_variant)
        mean_risk = _safe_float(float(np.mean(preds)))
        if mean_risk is None:
            # A silent gap in the curve would read as a missing category or grid value.
            raise ValueError("the model returned non-finite risk scores")
        return mean_risk

    if analysis_frame is not None and feature_name in analysis_frame.columns:
        if feature_name in categorical_features:
            if feature_encoder is None:
                # Without the encoder a category cannot be placed on the model's encoded columns.
                raise ValueError(
                    f"Partial dependence of the categorical feature '{feature_name}' needs the fitted feature "
                    "encoder, which maps each category onto the model's encoded columns."
                )
            encoder_levels = [
                str(level)
                for level in (
                    feature_encoder.get("categorical_mappings", {})
                    .get(feature_name, {})
                    .get("all_levels", [])
                )
            ]
            category_values = encoder_levels or _ordered_category_values(analysis_frame[feature_name])
            if len(category_values) < 2:
                raise ValueError(
                    f"Feature '{feature_name}' has fewer than two observed categories. "
                    "Partial dependence requires at least two distinct values."
                )

            mean_risks: list[float | None] = []
            # Count the rows of each level as the encoder reads them: canonical labels matched to
            # the stored levels, so "2.0" counts for level "2" even when the whole column holds 2.5.
            category_counts = _stored_level_values(
                canonical_category_values(analysis_frame[feature_name]), category_values
            ).value_counts(dropna=True)
            for category in category_values:
                raise_if_cancelled()
                frame_variant = analysis_frame.copy()
                frame_variant[feature_name] = category
                try:
                    mean_risks.append(_predict_mean_for_variant(frame_variant))
                except Exception as exc:
                    if must_propagate(exc) or isinstance(exc, MemoryError):
                        raise
                    computation_errors.append(f"{feature_name}={category}: {type(exc).__name__}: {exc}")
                    mean_risks.append(None)

            if computation_errors:
                raise ValueError(
                    "Partial dependence failed for one or more category levels: "
                    + "; ".join(computation_errors[:3])
                )

            return {
                "feature": feature_name,
                "feature_type": "categorical",
                "values": category_values,
                "mean_risk": mean_risks,
                "n_grid_points": len(category_values),
                "category_counts": {
                    category: int(category_counts.get(category, 0)) for category in category_values
                },
            }

        raw_values = pd.to_numeric(analysis_frame[feature_name], errors="coerce")
        valid_values = raw_values.dropna()
        if valid_values.empty:
            raise ValueError(
                f"Feature '{feature_name}' could not be converted to numeric values for partial dependence."
            )

        col_min = float(valid_values.min())
        col_max = float(valid_values.max())
        if col_min == col_max:
            raise ValueError(
                f"Feature '{feature_name}' has no variation (min == max == {col_min}). "
                "Partial dependence requires at least two distinct values."
            )

        grid = np.linspace(col_min, col_max, n_points)
        mean_risks: list[float | None] = []
        for grid_val in grid:
            raise_if_cancelled()
            frame_variant = analysis_frame.copy()
            frame_variant[feature_name] = float(grid_val)
            try:
                mean_risks.append(_predict_mean_for_variant(frame_variant))
            except Exception as exc:
                if must_propagate(exc) or isinstance(exc, MemoryError):
                    raise
                computation_errors.append(f"{feature_name}={float(grid_val):.6g}: {type(exc).__name__}: {exc}")
                mean_risks.append(None)

        if computation_errors:
            raise ValueError(
                "Partial dependence failed for one or more grid values: "
                + "; ".join(computation_errors[:3])
            )

        return {
            "feature": feature_name,
            "feature_type": "numeric",
            "values": [_safe_float(float(v)) for v in grid],
            "mean_risk": mean_risks,
            "n_grid_points": n_points,
            "feature_range": {"min": _safe_float(col_min), "max": _safe_float(col_max)},
        }

    if feature_name not in X_encoded.columns:
        raise ValueError(
            f"Feature '{feature_name}' was not found in the analyzable feature set. "
            "Use a feature from the selected model inputs."
        )

    X_array = X_encoded.to_numpy(dtype=float)
    col_idx = list(X_encoded.columns).index(feature_name)
    col_values = X_array[:, col_idx]

    col_min = float(np.nanmin(col_values))
    col_max = float(np.nanmax(col_values))

    if col_min == col_max:
        raise ValueError(
            f"Feature '{feature_name}' has no variation (min == max == {col_min}). "
            "Partial dependence requires at least two distinct values."
        )

    grid = np.linspace(col_min, col_max, n_points)
    mean_risks: list[float | None] = []

    for grid_val in grid:
        raise_if_cancelled()
        X_modified = X_array.copy()
        X_modified[:, col_idx] = grid_val
        try:
            preds = _predict_risk_scores(model, X_modified)
            mean_risk = _safe_float(float(np.mean(preds)))
            if mean_risk is None:
                raise ValueError("the model returned non-finite risk scores")
            mean_risks.append(mean_risk)
        except Exception as exc:
            if must_propagate(exc) or isinstance(exc, MemoryError):
                raise
            computation_errors.append(f"{feature_name}={float(grid_val):.6g}: {type(exc).__name__}: {exc}")
            mean_risks.append(None)

    if computation_errors:
        raise ValueError(
            "Partial dependence failed for one or more grid values: "
            + "; ".join(computation_errors[:3])
        )

    return {
        "feature": feature_name,
        "feature_type": "numeric",
        "values": [_safe_float(float(v)) for v in grid],
        "mean_risk": mean_risks,
        "n_grid_points": n_points,
        "feature_range": {"min": _safe_float(col_min), "max": _safe_float(col_max)},
    }


# ===================================================================
# 7. Integrated Brier Score (XAI)
# ===================================================================


def _censoring_survival_function(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse Kaplan-Meier estimate of the censoring survival G(t) = P(C > t).

    Events precede censorings at tied times, so the subjects who had the event at a
    time are not at risk of being censored then: hazard_C(u) = c_u / (n_u - d_u).
    Returns the distinct observed times and G at each of them (right-continuous).
    """
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    event_mask = np.asarray(events, dtype=float).reshape(-1) > 0
    unique_times, inverse = np.unique(times_arr, return_inverse=True)
    exits = np.bincount(inverse, minlength=unique_times.size).astype(float)
    deaths = np.bincount(inverse, weights=event_mask.astype(float), minlength=unique_times.size)
    censored = exits - deaths
    at_risk = exits[::-1].cumsum()[::-1]
    denominator = at_risk - deaths
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.where(denominator > 0.0, censored / denominator, 0.0)
    return unique_times, np.cumprod(1.0 - hazard)


def _censoring_survival_zero_time(times: np.ndarray, events: np.ndarray) -> float:
    """First time at which the censoring survival G(t) of the support cohort reaches zero (inf if never).

    From then on inverse-probability-of-censoring weights 1 / G(t) are undefined; this
    happens when the last observed time carries a censoring (for example tied with an event).
    """
    censor_times, censor_survival = _censoring_survival_function(times, events)
    exhausted = np.flatnonzero(censor_survival <= 0.0)
    return float(censor_times[exhausted[0]]) if exhausted.size else float("inf")


def _ipcw_brier_weights(
    times: np.ndarray,
    events: np.ndarray,
    eval_times: np.ndarray,
    *,
    support_times: np.ndarray,
    support_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """IPCW weights and "still at risk" indicators for Brier scores at ``eval_times``.

    Graf et al. (1999) weights with the convention that an event precedes a censoring
    at the same time, as the Kaplan-Meier estimator does:

    * ``t_i > t``: weight ``1 / G(t)`` (still under observation at t);
    * ``t_i <= t`` and an event: weight ``1 / G(t_i-)``, because the event is observed
      exactly when ``C >= t_i`` (Gerds & Schumacher, 2006; pec, riskRegression);
    * ``t_i <= t`` and censored: weight 0.

    ``G`` is the reverse Kaplan-Meier estimate from the support cohort whose censoring
    hazard at a tied time excludes the subjects who had the event then (as sksurv's
    reverse estimator does). Returns ``(weights, alive)``, both ``(n, len(eval_times))``.
    """
    eps = 1e-12
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    events_arr = np.asarray(events, dtype=float).reshape(-1)
    eval_arr = np.asarray(eval_times, dtype=float).reshape(-1)
    censor_times, censor_surv = _censoring_survival_function(support_times, support_events)

    def _lookup(query: np.ndarray, *, left_limit: bool) -> np.ndarray:
        out = np.ones_like(query, dtype=float)
        if censor_times.size == 0:
            return out
        idx = np.searchsorted(censor_times, query, side="left" if left_limit else "right") - 1
        valid = idx >= 0
        out[valid] = censor_surv[idx[valid]]
        return out

    g_at_eval = np.maximum(_lookup(eval_arr, left_limit=False), eps)
    g_before_subject = np.maximum(_lookup(times_arr, left_limit=True), eps)
    alive = times_arr[:, np.newaxis] > eval_arr[np.newaxis, :]
    event_before = (~alive) & (events_arr[:, np.newaxis] == 1)
    weights = np.where(alive, 1.0 / g_at_eval[np.newaxis, :], 0.0)
    weights = np.where(event_before, (1.0 / g_before_subject)[:, np.newaxis], weights)
    return weights, alive.astype(float)


def _brier_scores_from_weights(weights: np.ndarray, alive: np.ndarray, survival: np.ndarray) -> np.ndarray:
    """Brier score at each evaluation time from precomputed IPCW weights."""
    return np.mean(weights * (np.asarray(survival, dtype=float) - alive) ** 2, axis=0)


def _metric_vector(values: Any, name: str) -> np.ndarray:
    """A nonempty, finite vector for public prediction-metric helpers."""
    numbers = np.asarray(values, dtype=float)
    if numbers.ndim != 1 or not numbers.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.isfinite(numbers).all():
        raise ValueError(f"{name} must contain only finite numbers.")
    return numbers


def _metric_outcomes(times: Any, events: Any, *, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    time = _metric_vector(times, prefix + "times")
    event = _metric_vector(events, prefix + "events")
    if time.size != event.size:
        raise ValueError(f"{prefix}times and {prefix}events must have the same length.")
    if (time < 0.0).any():
        raise ValueError(f"{prefix}times must be nonnegative follow-up durations.")
    if not np.isin(event, (0.0, 1.0)).all():
        raise ValueError(f"{prefix}events must contain only 0 (censored) and 1 (event).")
    return time, event


def _check_survival_probabilities(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted values contain non-finite survival probabilities.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Predicted survival probabilities must be between 0 and 1.")


@user_input_boundary
def compute_integrated_brier_score(
    times: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[int],
    predicted_survival_fn: Any,
    eval_times: np.ndarray | Sequence[float] | None = None,
    *,
    support_times: np.ndarray | Sequence[float] | None = None,
    support_events: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute IBS and Brier Skill Score for survival-prediction error assessment.

    IBS reflects time-integrated survival-prediction error and is influenced
    by both discrimination and calibration.  This helper also computes a
    Kaplan-Meier null-model IBS and the corresponding Brier Skill Score
    ``1 - IBS_model / IBS_null`` so that absolute error can be contextualized
    against a no-covariate reference model.

    Parameters
    ----------
    times
        Observed follow-up times for each patient.
    events
        Event indicators (1 = event, 0 = censored) for each patient.
    predicted_survival_fn
        A callable ``f(eval_times) -> np.ndarray`` of shape
        ``(n_samples, n_eval_times)`` returning predicted survival
        probabilities for every patient at the requested time points.
    eval_times
        Optional array of time points at which to evaluate the Brier score.
        If *None*, 100 equally-spaced points between the 10th and 90th
        percentiles of the evaluation-cohort times are used (capped at the
        last event time in the IPCW support set).
    support_times, support_events
        Optional follow-up times and event indicators used to define the
        IPCW censoring weights and the default evaluation-time support.
        When omitted, the evaluation cohort is reused.

    Returns
    -------
    dict
        ``ibs`` (float), ``null_ibs`` (float), ``brier_skill_score`` (float),
        ``brier_scores`` / ``null_brier_scores`` (lists of per-time-point
        records), ``eval_times`` (list), and ``scientific_summary``.
    """
    times_arr, events_arr = _metric_outcomes(times, events)
    support_times_arr, support_events_arr = _metric_outcomes(
        support_times if support_times is not None else times_arr,
        support_events if support_events is not None else events_arr,
        prefix="support_",
    )

    n_samples = len(times_arr)
    support_event_times = support_times_arr[support_events_arr == 1]
    support_time_upper = (
        float(np.max(support_event_times))
        if support_event_times.size
        else float(np.max(support_times_arr))
    )

    # Default evaluation grid: the 10th-90th percentile range of the observed
    # evaluation-cohort times (the usual sksurv practice). It stays inside the
    # evaluation follow-up, where the censoring distribution G(t) is still
    # well estimated, and skips t ~ 0 where every model is trivially right.
    eval_upper = min(support_time_upper, float(np.max(times_arr)) if times_arr.size else support_time_upper)
    # The IPCW weights 1 / G(t) are undefined once the censoring survival reaches zero,
    # so every evaluation time lies strictly before that point.
    censoring_zero_time = _censoring_survival_zero_time(support_times_arr, support_events_arr)
    if eval_times is None:
        grid_lower, grid_upper = (
            np.percentile(times_arr, [10.0, 90.0]) if times_arr.size else (0.0, support_time_upper)
        )
        grid_upper = min(float(grid_upper), eval_upper)
        if grid_upper >= censoring_zero_time:
            earlier_support_times = np.unique(support_times_arr[support_times_arr < censoring_zero_time])
            if earlier_support_times.size == 0:
                raise ValueError(
                    "The censoring survival of the support cohort is zero from its first follow-up time, "
                    "so IPCW Brier scores are undefined."
                )
            grid_upper = float(earlier_support_times[-1])
        grid_lower = min(float(grid_lower), grid_upper)
        eval_times_arr = np.linspace(grid_lower, grid_upper, 100)
    else:
        eval_times_arr = _metric_vector(eval_times, "eval_times")

    # Keep evaluation times within the IPCW support window and the evaluation
    # follow-up to avoid extrapolation beyond observed data.
    eval_times_arr = eval_times_arr[
        (eval_times_arr >= 0.0) & (eval_times_arr <= eval_upper) & (eval_times_arr < censoring_zero_time)
    ]
    if len(eval_times_arr) == 0:
        raise ValueError(
            "No evaluation time points fall within the IPCW support range"
            + (
                f" (the censoring survival of the support cohort reaches zero at t={censoring_zero_time:.4g})."
                if np.isfinite(censoring_zero_time)
                else "."
            )
        )
    eval_times_arr = np.unique(np.asarray(eval_times_arr, dtype=float))
    single_time_support = eval_times_arr.size == 1 or float(eval_times_arr[-1] - eval_times_arr[0]) <= 0.0

    # Predicted survival matrix: (n_samples, n_eval_times)
    surv_matrix = np.asarray(predicted_survival_fn(eval_times_arr), dtype=float)
    if surv_matrix.shape != (n_samples, len(eval_times_arr)):
        raise ValueError(
            f"predicted_survival_fn must return shape ({n_samples}, {len(eval_times_arr)}), "
            f"got {surv_matrix.shape}."
        )
    _check_survival_probabilities(surv_matrix)

    from statsmodels.duration.survfunc import SurvfuncRight

    eps = 1e-12
    weights, alive_indicator = _ipcw_brier_weights(
        times_arr,
        events_arr,
        eval_times_arr,
        support_times=support_times_arr,
        support_events=support_events_arr,
    )
    event_sf = SurvfuncRight(support_times_arr, support_events_arr)
    event_times = event_sf.surv_times.astype(float)
    event_surv = event_sf.surv_prob.astype(float)
    null_survival = _step_survival_lookup(event_times, event_surv, eval_times_arr)

    bs_values = _brier_scores_from_weights(weights, alive_indicator, surv_matrix)
    null_bs_values = _brier_scores_from_weights(
        weights,
        alive_indicator,
        np.broadcast_to(null_survival[np.newaxis, :], surv_matrix.shape),
    )
    brier_scores: list[dict[str, Any]] = [
        {"time": _safe_float(float(t)), "score": _safe_float(float(score))}
        for t, score in zip(eval_times_arr, bs_values, strict=True)
    ]
    null_brier_scores: list[dict[str, Any]] = [
        {"time": _safe_float(float(t)), "score": _safe_float(float(score))}
        for t, score in zip(eval_times_arr, null_bs_values, strict=True)
    ]

    # Integrated Brier Score via trapezoidal rule
    bs_arr = np.array(bs_values, dtype=float)
    null_bs_arr = np.array(null_bs_values, dtype=float)
    ibs = _time_integral_mean(bs_arr, eval_times_arr)
    null_ibs = _time_integral_mean(null_bs_arr, eval_times_arr)
    brier_skill_score: float | None = None
    if np.isfinite(null_ibs) and float(null_ibs) > eps:
        brier_skill_score = float(1.0 - ibs / float(null_ibs))

    # Scientific summary
    n_events = int(np.sum(events_arr))
    grid_description = (
        "the 10th-90th percentile range of evaluation-cohort follow-up"
        if eval_times is None
        else "the requested evaluation times"
    )
    censoring_text = (
        f", ending before the censoring survival reaches zero at {censoring_zero_time:.4g}"
        if censoring_zero_time <= eval_upper
        else ""
    )
    window_text = (
        f"[{float(eval_times_arr[0]):.4g}, {float(eval_times_arr[-1]):.4g}] ({grid_description}, capped at the last "
        f"support event time {support_time_upper:.4g} and the last evaluation follow-up time{censoring_text})"
    )
    strengths: list[str] = (
        [
            f"Brier score computed at a single IPCW-supported time point for {n_samples} patients ({n_events} events).",
            "The Kaplan-Meier null-model reference was evaluated at the same time point to contextualize absolute prediction error.",
            f"The evaluation window {window_text} collapsed to a single supported time point.",
        ]
        if single_time_support
        else [
            f"IBS computed over {len(eval_times_arr)} time points for {n_samples} patients ({n_events} events).",
            "IBS summarizes predicted-vs-observed survival error across time, and Brier Skill Score contextualizes that error versus a Kaplan-Meier null model.",
            f"Evaluation times span {window_text}, which avoids late-time extrapolation where the censoring distribution is poorly estimated.",
        ]
    )
    cautions: list[str] = []
    next_steps: list[str] = []
    if single_time_support:
        cautions.append(
            "The IPCW support window collapsed to a single time point, so the reported IBS and null-model IBS reduce to pointwise Brier scores rather than a time-integrated summary."
        )

    score_label = "Pointwise Brier score" if single_time_support else "IBS"
    if ibs < 0.1:
        strengths.append(f"{score_label} = {ibs:.4f}; smaller values imply closer agreement between predictions and observations.")
    elif ibs < 0.2:
        strengths.append(f"{score_label} = {ibs:.4f}; smaller values imply closer agreement between predictions and observations.")
    elif ibs < 0.25:
        cautions.append(f"{score_label} = {ibs:.4f}; the prediction error is moderate, so review the survival curves.")
    else:
        cautions.append(f"{score_label} = {ibs:.4f}; the prediction error is large, so the model needs review.")

    if brier_skill_score is None:
        cautions.append(
            "Brier Skill Score could not be computed because the Kaplan-Meier null-model IBS was not stably positive over the evaluation window."
        )
    elif brier_skill_score > 0.10:
        strengths.append(
            f"Brier Skill Score = {brier_skill_score:.3f} versus a Kaplan-Meier null-model IBS of {null_ibs:.4f}, indicating clear improvement over the no-covariate reference."
        )
    elif brier_skill_score > 0.0:
        strengths.append(
            f"Brier Skill Score = {brier_skill_score:.3f} versus a Kaplan-Meier null-model IBS of {null_ibs:.4f}, indicating modest improvement over the no-covariate reference."
        )
    else:
        cautions.append(
            f"Brier Skill Score = {brier_skill_score:.3f} versus a Kaplan-Meier null-model IBS of {null_ibs:.4f}; this model did not improve on the no-covariate reference."
        )

    cautions.append(
        "This implementation uses IPCW Brier scores; if the censoring survival "
        "probability approaches 0 at late times, estimates can become unstable."
    )
    if support_times is not None or support_events is not None:
        strengths.append("IPCW censoring weights were estimated from the supplied support cohort rather than the evaluation cohort.")
    next_steps.append(
        "Compare both IBS and Brier Skill Score across models; Brier Skill Score should stay above 0 to justify improvement over the Kaplan-Meier null reference."
    )
    if single_time_support:
        next_steps.append(
            "Use follow-up support beyond a single time point if you need a true integrated Brier score across time."
        )

    status = "robust"
    if any("moderate" in c for c in cautions):
        status = "review"
    if ibs >= 0.25 or n_events < 10 or (brier_skill_score is not None and brier_skill_score <= 0.0):
        status = "caution"
    elif brier_skill_score is None or single_time_support:
        status = "review"

    headline_prefix = (
        f"Pointwise Brier score = {ibs:.4f} at t={_safe_float(eval_times_arr[0])}"
        if single_time_support
        else f"Integrated Brier Score = {ibs:.4f} over [{_safe_float(eval_times_arr[0])}, {_safe_float(eval_times_arr[-1])}]"
    )
    scientific_summary = {
        "status": status,
        "headline": (
            headline_prefix
            + (
                f"; Brier Skill Score = {brier_skill_score:.3f} versus the Kaplan-Meier null model."
                if brier_skill_score is not None
                else "; Kaplan-Meier null-model Brier Skill Score was unavailable."
            )
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "IBS", "value": _safe_float(ibs)},
            {"label": "Null-model IBS", "value": _safe_float(null_ibs)},
            {"label": "Brier Skill Score", "value": _safe_float(brier_skill_score)},
            {"label": "Patients", "value": n_samples},
            {"label": "Events", "value": n_events},
            {"label": "Eval time points", "value": len(eval_times_arr)},
        ],
    }

    return {
        "ibs": _safe_float(ibs),
        "null_ibs": _safe_float(null_ibs),
        "brier_skill_score": _safe_float(brier_skill_score),
        "brier_scores": brier_scores,
        "null_brier_scores": null_brier_scores,
        "eval_times": [_safe_float(float(t)) for t in eval_times_arr],
        "scientific_summary": scientific_summary,
    }


# ===================================================================
# 8. Calibration curve data (XAI)
# ===================================================================


@user_input_boundary
def compute_calibration_data(
    times: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[int],
    predicted_survival_at_t: np.ndarray | Sequence[float],
    t: float | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curve data (predicted vs observed survival).

    Inspired by the SurvBoard finding (Wissel et al., 2025) that
    statistical models consistently outperform deep-learning models in
    calibration.  This function bins patients by their predicted survival
    probability at a fixed time *t*, then estimates the actual survival in
    each bin using the Kaplan-Meier estimator (``SurvfuncRight``).

    Parameters
    ----------
    times
        Observed follow-up times for each patient.
    events
        Event indicators (1 = event, 0 = censored).
    predicted_survival_at_t
        Predicted :math:`\\hat{S}(t)` for each patient at time *t*.
    t
        The time point corresponding to ``predicted_survival_at_t``.
        Used only for labelling; if *None* a placeholder is used.
    n_bins
        Number of bins to group patients into (default 10).

    Returns
    -------
    dict
        ``time_point``, ``bins`` (list of {predicted_mean, observed, count}),
        and ``scientific_summary``.
    """
    from statsmodels.duration.survfunc import SurvfuncRight

    times_arr, events_arr = _metric_outcomes(times, events)
    pred_arr = _metric_vector(predicted_survival_at_t, "predicted_survival_at_t")
    _check_survival_probabilities(pred_arr)

    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if t is not None:
        t = float(t)
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("t must be a finite, nonnegative follow-up time.")

    if not (len(times_arr) == len(events_arr) == len(pred_arr)):
        raise ValueError("times, events, and predicted_survival_at_t must have the same length.")

    n_samples = len(times_arr)
    n_events = int(np.sum(events_arr))
    time_label = _safe_float(t) if t is not None else "unspecified"

    # Quantile binning (duplicates dropped) keeps bins balanced even when
    # predictions are concentrated in a narrow range.
    bins_result: list[dict[str, Any]] = []
    pred_series = pd.Series(pred_arr)
    try:
        bin_assign = pd.qcut(pred_series, q=n_bins, duplicates="drop")
        bin_categories = list(bin_assign.cat.categories)
    except (ValueError, IndexError):
        # Too few distinct predictions to form quantile bins.
        bin_assign = None
        bin_categories = []

    if bin_assign is None or not bin_categories:
        # Fallback: single-bin summary
        bin_categories = [None]
    # Tied predictions merge quantile bins, so fewer bins than requested may be formed.
    n_bins_formed = len(bin_categories)
    bins_text = f"{n_bins_formed} bin(s)" + (
        f" ({n_bins} requested; tied predicted values left fewer distinct quantile bins)"
        if n_bins_formed != n_bins
        else ""
    )

    non_estimable_bins = 0
    for cat in bin_categories:
        mask = np.ones_like(pred_arr, dtype=bool) if cat is None else (bin_assign == cat).to_numpy()

        count = int(np.sum(mask))
        if count == 0:
            bins_result.append({"predicted_mean": None, "observed": None, "count": 0})
            continue

        predicted_mean = float(np.mean(pred_arr[mask]))

        # Kaplan-Meier estimate of survival at time t in this bin
        bin_times = times_arr[mask]
        bin_events = events_arr[mask]

        observed_survival: float | None = None
        if t is not None:
            last_follow_up = float(np.max(bin_times))
            if np.sum(bin_events) > 0:
                try:
                    sf = SurvfuncRight(bin_times, bin_events)
                    km_times = sf.surv_times
                    km_surv = sf.surv_prob
                    idx = int(np.searchsorted(km_times, t, side="right")) - 1
                    observed_survival = 1.0 if idx < 0 else float(km_surv[idx])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    observed_survival = None
                # Beyond the bin's last follow-up the Kaplan-Meier curve is undefined
                # unless it already reached zero.
                if observed_survival is not None and t > last_follow_up and observed_survival > 0.0:
                    observed_survival = None
            elif last_follow_up >= t:
                # No events and some patients followed to t: no failures observed.
                observed_survival = 1.0
            # Otherwise everyone was censored before t: survival at t is not estimable.
            if observed_survival is None:
                non_estimable_bins += 1

        bins_result.append({
            "predicted_mean": _safe_float(predicted_mean),
            "observed": _safe_float(observed_survival),
            "count": count,
        })

    # Scientific summary
    non_empty_bins = [b for b in bins_result if b["observed"] is not None and b["predicted_mean"] is not None]
    if len(non_empty_bins) >= 2:
        pred_vals = np.array([b["predicted_mean"] for b in non_empty_bins])
        obs_vals = np.array([b["observed"] for b in non_empty_bins])
        mean_abs_diff = float(np.mean(np.abs(pred_vals - obs_vals)))
    else:
        mean_abs_diff = None

    strengths: list[str] = [
        f"Calibration assessed at t={time_label} across {bins_text} for {n_samples} patients ({n_events} events); this is a descriptive binning-based check, not a formal calibration test.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    if mean_abs_diff is not None and mean_abs_diff < 0.05:
        strengths.append(f"Mean |predicted - observed| = {mean_abs_diff:.4f}; smaller values indicate closer agreement.")
    elif mean_abs_diff is not None and mean_abs_diff < 0.10:
        strengths.append(f"Mean |predicted - observed| = {mean_abs_diff:.4f}; smaller values indicate closer agreement.")
    elif mean_abs_diff is not None:
        cautions.append(f"Mean |predicted - observed| = {mean_abs_diff:.4f}; larger values indicate weaker agreement.")

    cautions.append(
        "Observed survival is estimated with within-bin Kaplan-Meier curves and is not IPCW-adjusted; treat this as a descriptive calibration check, not a formal calibration estimate."
    )

    empty_bins = sum(1 for b in bins_result if b["count"] == 0)
    if empty_bins > 0:
        cautions.append(
            f"{empty_bins} of {n_bins_formed} bins had no patients; "
            "consider reducing n_bins or using a larger dataset."
        )
    if non_estimable_bins > 0:
        cautions.append(
            f"{non_estimable_bins} bin(s) had no follow-up reaching t={time_label}, so observed survival there is not "
            "estimable and those bins are left out of the agreement summary."
        )

    next_steps.append(
        "Plot predicted_mean vs observed for a visual calibration curve; points near the diagonal indicate closer agreement."
    )
    next_steps.append(
        "If agreement is poor, consider recalibrating with Platt scaling or isotonic regression."
    )

    status = "robust"
    if cautions:
        status = "review"
    if n_events < 10 or (mean_abs_diff is not None and mean_abs_diff >= 0.15):
        status = "caution"

    scientific_summary = {
        "status": status,
        "headline": (
            f"Heuristic calibration check at t={time_label}: "
            + (f"mean absolute deviation = {mean_abs_diff:.4f}." if mean_abs_diff is not None else "insufficient data for a descriptive check.")
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": n_samples},
            {"label": "Events", "value": n_events},
            {"label": "Bins", "value": n_bins_formed},
            {"label": "Mean |pred - obs|", "value": _safe_float(mean_abs_diff)},
        ],
    }

    predicted_points = [b["predicted_mean"] for b in bins_result if b["predicted_mean"] is not None and b["observed"] is not None]
    observed_points = [b["observed"] for b in bins_result if b["predicted_mean"] is not None and b["observed"] is not None]

    return {
        "time_point": time_label,
        "n_bins": n_bins_formed,
        "n_bins_requested": n_bins,
        "bins": bins_result,
        "predicted": predicted_points,
        "observed": observed_points,
        "scientific_summary": scientific_summary,
    }


# ===================================================================
# 9. Time-dependent feature importance (XAI)
# ===================================================================


@user_input_boundary
def compute_time_dependent_importance(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    eval_times: Sequence[float] | None = None,
    event_positive_value: Any = None,
    model_type: str = "rsf",
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Time-dependent permutation importance of a fitted survival model.

    A Random Survival Forest (``model_type="rsf"``) or Gradient Boosted Survival model
    (``"gbs"``) is fitted on the shared stratified training split. For each evaluation
    time ``t`` the importance of a raw feature is the increase in the IPCW Brier score at
    ``t`` on the evaluation rows when that feature is shuffled (all one-hot columns of a
    categorical feature together), averaged over repeats. IPCW weights keep patients
    censored before ``t`` represented instead of dropping them, and nothing is scored on
    the rows the model was fitted on unless the cohort is too small for a holdout.

    Returns
    -------
    dict
        ``eval_times``, ``features``, ``importance_matrix`` (times x features),
        ``importance_matrix_feature_major`` (features x times),
        ``importance_matrix_orientation``, ``baseline_brier_scores``,
        ``dominant_feature_per_time``, ``evaluable_patients_per_time``,
        ``skipped_time_points``, and ``scientific_summary``.
    """
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for time-dependent importance. "
            "Install it with: pip install scikit-survival"
        )
    if model_type not in {"rsf", "gbs"}:
        raise ValueError(f"Unsupported time-dependent importance model_type '{model_type}'. Expected 'rsf' or 'gbs'.")
    _validate_model_feature_columns(features, time_column=time_column, event_column=event_column)

    trainer = train_random_survival_forest if model_type == "rsf" else train_gradient_boosted_survival
    trainer_kwargs: dict[str, Any] = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
    if model_type == "gbs":
        trainer_kwargs["learning_rate"] = learning_rate
    # Only the fitted model is needed here, not the trainer's own importance and Brier metrics.
    fitted = trainer(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        compute_importance=False,
        compute_brier=False,
        **trainer_kwargs,
    )
    model = fitted["_model"]
    evaluation_mode = str(fitted["model_stats"]["evaluation_mode"])
    eval_frame: pd.DataFrame = fitted["_analysis_eval_frame"]
    X_eval: pd.DataFrame = fitted["_X_eval_encoded"]
    # Censoring weights come from the fitting rows, as for the holdout IBS.
    support_frame: pd.DataFrame = fitted["_analysis_train_frame"]
    time_values = eval_frame[time_column].to_numpy(dtype=float)
    event_values = eval_frame[event_column].to_numpy(dtype=float)
    support_times = support_frame[time_column].to_numpy(dtype=float)
    support_events = support_frame[event_column].to_numpy(dtype=float)

    rng = np.random.default_rng(int(random_state))
    matrix = X_eval.to_numpy(dtype=float)
    if matrix.shape[0] > _PERMUTATION_IMPORTANCE_MAX_ROWS:
        rows = np.sort(rng.choice(matrix.shape[0], size=_PERMUTATION_IMPORTANCE_MAX_ROWS, replace=False))
        matrix = matrix[rows]
        time_values = time_values[rows]
        event_values = event_values[rows]

    support_event_times = support_times[support_events == 1]
    horizon = min(
        float(np.max(support_event_times)) if support_event_times.size else float(np.max(support_times)),
        float(np.max(time_values)),
    )
    if eval_times is None:
        event_times = time_values[event_values == 1]
        if event_times.size < 5:
            eval_times_arr = np.unique(event_times)
        else:
            eval_times_arr = np.quantile(event_times, [0.2, 0.4, 0.5, 0.6, 0.8])
    else:
        eval_times_arr = np.asarray(eval_times, dtype=float)
    requested_times = np.sort(np.unique(eval_times_arr[np.isfinite(eval_times_arr)]))
    in_support = (requested_times > 0.0) & (requested_times <= horizon)
    # Where the censoring survival of the fitting rows reaches zero, IPCW weights are undefined.
    censoring_zero_time = _censoring_survival_zero_time(support_times, support_events)
    censoring_defined = requested_times < censoring_zero_time
    eval_times_arr = requested_times[in_support & censoring_defined]
    skipped_time_points: list[dict[str, Any]] = [
        {
            "time": _safe_float(float(t)),
            "reason": "outside_follow_up_support" if not supported else "censoring_survival_zero",
            "evaluable_patients": None,
        }
        for t, supported in zip(requested_times, in_support)
        if not (supported and t < censoring_zero_time)
    ]
    if eval_times_arr.size == 0:
        raise ValueError(
            "No evaluation time points fall inside the follow-up support of the evaluation rows "
            f"(0 < t <= {horizon:.4g}"
            + (f", before the censoring survival reaches zero at {censoring_zero_time:.4g}" if np.isfinite(censoring_zero_time) else "")
            + ")."
        )

    weights, alive = _ipcw_brier_weights(
        time_values,
        event_values,
        eval_times_arr,
        support_times=support_times,
        support_events=support_events,
    )

    def _survival_matrix(design: np.ndarray) -> np.ndarray:
        return _sksurv_survival_predictor(model, design)(eval_times_arr)

    baseline_scores = _brier_scores_from_weights(weights, alive, _survival_matrix(matrix))
    groups = encoded_feature_groups(list(X_eval.columns), fitted.get("_feature_encoder"))
    feature_names = list(groups)
    n_repeats = 5 if len(groups) <= 20 else (3 if len(groups) <= 60 else 2)
    importance_by_feature = np.zeros((len(feature_names), eval_times_arr.size), dtype=float)
    for feature_index, feature in enumerate(feature_names):
        raise_if_cancelled()
        positions = groups[feature]
        increases = np.zeros(eval_times_arr.size, dtype=float)
        for _ in range(n_repeats):
            permutation = rng.permutation(matrix.shape[0])
            shuffled = matrix.copy()
            shuffled[:, positions] = matrix[permutation][:, positions]
            increases += _brier_scores_from_weights(weights, alive, _survival_matrix(shuffled)) - baseline_scores
        importance_by_feature[feature_index] = increases / n_repeats

    evaluable_patients_per_time = [
        int(np.sum((time_values > t) | ((time_values <= t) & (event_values == 1)))) for t in eval_times_arr
    ]
    importance_matrix_time_major: list[list[float | None]] = [
        [_safe_float(float(value)) for value in importance_by_feature[:, time_index]]
        for time_index in range(eval_times_arr.size)
    ]
    dominant_per_time: list[str | None] = [
        feature_names[int(np.argmax(importance_by_feature[:, time_index]))] if feature_names else None
        for time_index in range(eval_times_arr.size)
    ]
    importance_matrix_feature_major = [
        [_safe_float(float(value)) for value in importance_by_feature[feature_index]]
        for feature_index in range(len(feature_names))
    ]

    n_patients = int(fitted["model_stats"]["n_patients"])
    n_events = int(fitted["model_stats"]["n_events"])
    n_eval_events = int(np.sum(event_values))
    varying_features: list[str] = []
    if eval_times_arr.size >= 2 and feature_names:
        spread = np.std(importance_by_feature, axis=1)
        varying_features = [feature_names[index] for index in np.argsort(spread)[::-1][:3]]

    model_label = "Random Survival Forest" if model_type == "rsf" else "Gradient Boosted Survival"
    strengths: list[str] = [
        f"Importance computed at {eval_times_arr.size} time point(s) for {len(feature_names)} feature(s) from a "
        f"{model_label} fitted on the training split.",
        "Each value is the increase in the IPCW Brier score at that time when the feature is shuffled on the "
        f"evaluation rows ({matrix.shape[0]} patients, {n_eval_events} events), averaged over {n_repeats} shuffles.",
        "Inverse-probability-of-censoring weights keep patients censored before a time point represented instead of dropping them.",
    ]
    cautions: list[str] = [
        "Permutation importance measures how much the fitted model relies on a feature; correlated features share and can mask each other's importance.",
        "This is model-based permutation importance, not SurvSHAP(t) or a causal effect.",
    ]
    if evaluation_mode != "holdout":
        cautions.append(
            "The cohort was too small for a holdout split, so importance was measured on the rows the model was fitted on and is optimistic."
        )
    if varying_features:
        strengths.append(f"Features with the greatest change across time: {', '.join(varying_features)}.")
    if n_eval_events < 20:
        cautions.append("Fewer than 20 events on the evaluation rows make per-time-point importance noisy.")
    if skipped_time_points:
        cautions.append(
            f"{len(skipped_time_points)} requested time point(s) outside the evaluation follow-up support "
            f"(0 < t <= {horizon:.4g}"
            + (
                f", and before the censoring survival of the fitting rows reaches zero at {censoring_zero_time:.4g}"
                if np.isfinite(censoring_zero_time)
                else ""
            )
            + ") were skipped."
        )
    next_steps = [
        "Compare early and late time points to see which features drive short- versus long-term risk predictions.",
        "Check stability by rerunning with another seed before interpreting small differences.",
    ]

    unique_dominant = [feature for feature in dominant_per_time if feature is not None]
    headline_feature = max(set(unique_dominant), key=unique_dominant.count) if unique_dominant else "N/A"
    status = "robust"
    if n_eval_events < 20 or evaluation_mode != "holdout":
        status = "review"
    if n_eval_events < 10:
        status = "caution"

    scientific_summary = {
        "status": status,
        "headline": (
            f"Time-dependent permutation importance ({model_label}); most frequently dominant feature: {headline_feature}."
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": n_patients},
            {"label": "Events", "value": n_events},
            {"label": "Evaluation patients", "value": int(matrix.shape[0])},
            {"label": "Time points", "value": int(eval_times_arr.size)},
            {
                "label": "Evaluable patients, min",
                "value": int(min(evaluable_patients_per_time)) if evaluable_patients_per_time else 0,
            },
            {"label": "Features", "value": len(feature_names)},
        ],
    }

    return {
        "eval_times": [_safe_float(float(t)) for t in eval_times_arr],
        "features": feature_names,
        "importance_matrix": importance_matrix_time_major,
        "importance_matrix_time_major": importance_matrix_time_major,
        "importance_matrix_feature_major": importance_matrix_feature_major,
        "importance_matrix_orientation": "time_major",
        "importance_label": "Brier score increase",
        "importance_method": (
            "Increase in the IPCW Brier score at each time when a raw feature is shuffled on the evaluation rows."
        ),
        "baseline_brier_scores": [_safe_float(float(value)) for value in baseline_scores],
        "model_type": model_type,
        "evaluation_mode": evaluation_mode,
        "random_state": int(random_state),
        "dominant_feature_per_time": dominant_per_time,
        "evaluable_patients_per_time": evaluable_patients_per_time,
        "skipped_time_points": skipped_time_points,
        "scientific_summary": scientific_summary,
    }


# ===================================================================
# 10. Counterfactual survival curves (XAI)
# ===================================================================


@user_input_boundary
def counterfactual_survival(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    target_feature: str = "",
    original_value: Any = None,
    counterfactual_value: Any = None,
    event_positive_value: Any = None,
    model_type: str = "rsf",
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    random_state: int = 42,
    trained_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate counterfactual survival analysis: 'What if this feature were different?'

    Inspired by Tier 3-2 of the survey: interactive counterfactual survival
    curves. A model is trained on the observed data, then risk scores are
    predicted under two cohort-level scenarios:
    - the target feature set to ``original_value`` for every patient
      (or the observed values when ``original_value`` is *None*)
    - the target feature set to ``counterfactual_value`` for every patient
    The shift in median risk score summarizes a model-based association under
    this cohort-level perturbation; it is descriptive and not causal.

    Parameters
    ----------
    df
        Source DataFrame.
    time_column, event_column
        Column names for time and event indicator.
    features
        Feature column names.
    categorical_features
        Subset of *features* that are categorical.
    target_feature
        The feature to modify for the counterfactual scenario.
    original_value
        The baseline value to substitute for *target_feature* across the
        analyzable cohort. If *None*, the observed feature values are used
        as-is for the baseline prediction.
    counterfactual_value
        The value to substitute for *target_feature* in the counterfactual
        scenario.
    event_positive_value
        Passed through to ``_cohort_frame`` for event coercion.
    model_type
        ``"rsf"`` (default) trains a Random Survival Forest; ``"gbs"``
        trains a Gradient Boosted Survival model.

    Returns
    -------
    dict
        ``target_feature``, ``original_value``, ``counterfactual_value``,
        ``original_median_risk``, ``counterfactual_median_risk``,
        ``risk_change_pct``, and ``scientific_summary``.
    """
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for counterfactual survival analysis. "
            "Install it with: pip install scikit-survival"
        )
    _validate_model_feature_columns(features, time_column=time_column, event_column=event_column)

    if not target_feature:
        raise ValueError("target_feature must be specified.")
    if target_feature not in features:
        raise ValueError(
            f"target_feature '{target_feature}' must be in the features list."
        )

    cat_feats = list(categorical_features or [])
    if trained_result is None:
        frame = _cohort_frame(
            df,
            time_column=time_column,
            event_column=event_column,
            event_positive_value=event_positive_value,
            extra_columns=list(features),
            drop_missing_extra_columns=False,
        )
        # Only the fitted model is needed, not the trainer's own importance and Brier metrics.
        if model_type == "gbs":
            result = train_gradient_boosted_survival(
                frame,
                time_column=time_column,
                event_column=event_column,
                features=features,
                categorical_features=cat_feats,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                compute_importance=False,
                compute_brier=False,
            )
        elif model_type == "rsf":
            result = train_random_survival_forest(
                frame,
                time_column=time_column,
                event_column=event_column,
                features=features,
                categorical_features=cat_feats,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                compute_importance=False,
                compute_brier=False,
            )
        else:
            raise ValueError(f"Unsupported counterfactual model_type '{model_type}'. Expected 'rsf' or 'gbs'.")
    else:
        result = trained_result
        frame = result.get("_analysis_frame")
        required_frame_columns = {time_column, event_column, *features}
        if frame is None or not required_frame_columns.issubset(set(frame.columns)):
            frame = _cohort_frame(
                df,
                time_column=time_column,
                event_column=event_column,
                event_positive_value=event_positive_value,
                extra_columns=list(features),
                drop_missing_extra_columns=False,
            )

    model = result["_model"]
    _require_predict_callable(model, context="Counterfactual survival analysis")
    X_original = result["_X_encoded"]
    encoder = result.get("_feature_encoder")
    feature_scaler = result.get("_feature_scaler")
    analysis_frame = result.get("_analysis_frame")
    if analysis_frame is None or not set(features).issubset(set(analysis_frame.columns)):
        analysis_frame = frame
    if encoder is None:
        # A trained result without its fitted encoder: rebuild it once from the observed rows,
        # never from a scenario frame, where a categorical target holds a single value and
        # would lose its indicator columns (every patient scored as the reference level).
        encoder = _fit_feature_encoder(analysis_frame, features, cat_feats)
    # The fitted encoder decides how the model saw the target: text columns are one-hot
    # encoded even when the request did not list them as categorical.
    target_is_categorical = target_feature in _encoder_categorical_features(encoder, cat_feats)

    def _scenario_value(value: Any) -> Any:
        """The value the scenario applies: a level the model knows for a categorical target."""
        if value is None or not target_is_categorical:
            return value
        return _resolve_category_level(encoder, target_feature, value)

    original_level = _scenario_value(original_value)
    counterfactual_level = _scenario_value(counterfactual_value)

    def _scenario_label(value: Any) -> str:
        return "observed values" if value is None else str(value)

    def _build_scenario_matrix(value: Any) -> pd.DataFrame:
        if value is None:
            return X_original

        frame_variant = analysis_frame.copy()
        if target_is_categorical:
            frame_variant[target_feature] = str(value)
        else:
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                raise ValueError(
                    f"Counterfactual value '{value}' for numeric feature '{target_feature}' "
                    "could not be converted to a number."
                )
            frame_variant[target_feature] = float(numeric_value)

        encoded = _transform_feature_encoder(frame_variant, encoder)
        return _apply_feature_scaler(
            encoded.reindex(columns=X_original.columns, fill_value=0.0).fillna(0.0),
            feature_scaler,
        )

    original_label = _scenario_label(original_level)
    counterfactual_label = _scenario_label(counterfactual_level)

    # Original risk scores
    X_baseline = _build_scenario_matrix(original_level)
    original_risk = _predict_risk_scores(model, X_baseline)
    original_median_risk = float(np.median(original_risk))

    # Build counterfactual feature matrix
    X_cf = _build_scenario_matrix(counterfactual_level)
    cf_risk = _predict_risk_scores(model, X_cf)
    cf_median_risk = float(np.median(cf_risk))

    # Relative change on a ratio scale. Cox-type scores (GBS, LASSO-Cox) are
    # log partial hazards, so a percentage of the raw score is meaningless;
    # compare per-patient hazard ratios exp(cf - original) instead. RSF scores
    # are cumulative hazards summed over the training event times (positive), so
    # per-patient ratios apply directly.
    risk_scale = _model_risk_scale(model)
    original_arr = np.asarray(original_risk, dtype=float)
    cf_arr = np.asarray(cf_risk, dtype=float)
    if risk_scale == "log_hazard":
        per_patient_ratio = np.exp(np.clip(cf_arr - original_arr, -50.0, 50.0))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            per_patient_ratio = np.where(original_arr > 0.0, cf_arr / original_arr, np.nan)
    finite_ratio = per_patient_ratio[np.isfinite(per_patient_ratio)]
    if finite_ratio.size:
        risk_change_pct = (float(np.median(finite_ratio)) - 1.0) * 100.0
    else:
        risk_change_pct = 0.0 if np.allclose(cf_arr, original_arr) else None

    n_patients = int(frame.shape[0])
    n_events = int(frame[event_column].sum())

    # Interpret direction
    if risk_change_pct is None:
        direction = "changes"
        direction_label = "changed risk"
    elif risk_change_pct > 5.0:
        direction = "increases"
        direction_label = "higher risk"
    elif risk_change_pct < -5.0:
        direction = "decreases"
        direction_label = "lower risk"
    else:
        direction = "changes"
        direction_label = "similar risk"

    if risk_change_pct is None:
        effect_sentence = (
            "The relative change in predicted risk is undefined because the baseline predicted risk is zero."
        )
        headline_effect = (
            f"Under a model-based scenario that sets '{target_feature}' from {original_label} to {counterfactual_label}, "
            "predicted risk changed, but the relative change is undefined because the baseline predicted risk is zero."
        )
    else:
        relative_quantity = (
            "hazard"
            if risk_scale == "log_hazard"
            else "risk score (cumulative hazard summed over the training event times)"
        )
        # The reported change is the median of the per-patient ratios, not a change of the median risk.
        effect_sentence = (
            f"The median per-patient relative {relative_quantity} {direction} by {abs(risk_change_pct):.1f}% "
            f"({direction_label})."
        )
        headline_effect = (
            f"Under a model-based scenario that sets '{target_feature}' from {original_label} to {counterfactual_label}, "
            f"the median per-patient relative {relative_quantity} {direction} by {abs(risk_change_pct):.1f}%."
        )

    strengths: list[str] = [
        f"Counterfactual scenario analysis set '{target_feature}' from "
        f"{original_label} to {counterfactual_label} across {n_patients} patients.",
        effect_sentence,
    ]
    cautions: list[str] = [
        "Counterfactual analysis assumes independent feature manipulation; "
        "correlated features may invalidate the 'all else equal' assumption.",
        "This is a descriptive model-based perturbation on observational data, not a causal estimate.",
    ]
    next_steps: list[str] = [
        "Validate counterfactual findings with domain expertise before clinical interpretation.",
        "Consider testing multiple counterfactual values to map the dose-response curve.",
    ]

    if n_events < 20:
        cautions.append("Fewer than 20 events limits the reliability of risk estimates.")

    status = "robust"
    if n_events < 20:
        status = "review"
    if n_events < 10:
        status = "caution"

    scientific_summary = {
        "status": status,
        "headline": headline_effect,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": n_patients},
            {"label": "Events", "value": n_events},
            {"label": "Original median risk", "value": _safe_float(original_median_risk)},
            {"label": "Counterfactual median risk", "value": _safe_float(cf_median_risk)},
            {"label": "Risk change (%)", "value": _safe_float(risk_change_pct)},
        ],
    }

    return {
        "target_feature": target_feature,
        "original_value": original_value,
        "counterfactual_value": counterfactual_value,
        "original_median_risk": _safe_float(original_median_risk),
        "counterfactual_median_risk": _safe_float(cf_median_risk),
        "risk_change_pct": _safe_float(risk_change_pct),
        "risk_change_method": (
            "median per-patient hazard ratio exp(counterfactual - original) on the log partial hazard scale"
            if risk_scale == "log_hazard"
            else "median per-patient ratio of risk scores, each the predicted cumulative hazard summed over the training event times"
        ),
        "model_type": model_type,
        "scientific_summary": scientific_summary,
    }

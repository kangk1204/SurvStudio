"""Machine-learning survival models and cutpoint optimisation utilities.

This module provides tree-based survival models (Random Survival Forest,
Gradient Boosted Survival), optimal cutpoint scanning, SHAP explanations,
and partial-dependence computation.  Every public function returns a plain
``dict`` that is JSON-serialisable so it can be served directly by the
FastAPI backend.

Optional heavy dependencies (scikit-survival, shap) are imported lazily
behind availability flags so the rest of the toolkit keeps working when
they are not installed.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff

from survival_toolkit.analysis import (
    _cohort_frame,
    _harrell_c_index,
    _safe_float,
    coerce_event,
)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from sksurv.ensemble import (
        GradientBoostingSurvivalAnalysis,
        RandomSurvivalForest,
    )
    from sksurv.metrics import concordance_index_censored

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXPECTED_CUTPOINT_SCAN_ERRORS = (
    ValueError,
    ZeroDivisionError,
    FloatingPointError,
    OverflowError,
    np.linalg.LinAlgError,
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


def _coerce_feature_subset(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Coerce a feature subset into numeric/categorical columns."""
    categorical_features = list(categorical_features or [])
    selected = df[list(features)].copy()
    resolved_categorical: list[str] = []
    for col in features:
        if col in categorical_features:
            selected[col] = selected[col].astype("string")
            resolved_categorical.append(col)
        elif is_numeric_dtype(selected[col]):
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
        else:
            selected[col] = selected[col].astype("string")
            resolved_categorical.append(col)
    numeric_features = [col for col in features if col not in resolved_categorical]
    return selected, resolved_categorical, numeric_features


def _ordered_category_values(series: pd.Series) -> list[str]:
    """Return observed category values in a deterministic, semantically stable order."""
    non_missing = series.dropna()
    if non_missing.empty:
        return []
    if isinstance(series.dtype, pd.CategoricalDtype) and getattr(series.dtype, "ordered", False):
        return [str(value) for value in series.dtype.categories.tolist() if pd.notna(value)]
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    if numeric_values.notna().all():
        ordered_numeric = np.sort(numeric_values.unique().astype(float))
        return [str(int(value)) if float(value).is_integer() else str(value) for value in ordered_numeric]
    return list(dict.fromkeys(non_missing.astype("string").tolist()))


def _fit_feature_encoder(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit a deterministic feature encoder with explicit unknown buckets."""
    selected, resolved_categorical, numeric_features = _coerce_feature_subset(
        df,
        features,
        categorical_features,
    )
    categorical_mappings: dict[str, dict[str, Any]] = {}
    feature_names: list[str] = []
    for col in resolved_categorical:
        levels = sorted(
            str(level) for level in selected[col].dropna().astype("string").unique().tolist()
        )
        baseline = levels[0] if levels else None
        retained_levels = levels[1:] if len(levels) > 1 else []
        unknown_column = f"{col}__unknown"
        missing_column = f"{col}__missing"
        categorical_mappings[col] = {
            "all_levels": levels,
            "baseline_level": baseline,
            "retained_levels": retained_levels,
            "unknown_column": unknown_column,
            "missing_column": missing_column,
        }
        feature_names.extend([f"{col}_{level}" for level in retained_levels])
        feature_names.append(unknown_column)
        feature_names.append(missing_column)
    numeric_impute_values: dict[str, float] = {}
    for col in numeric_features:
        numeric_series = pd.to_numeric(selected[col], errors="coerce")
        median_value = numeric_series.median(skipna=True)
        numeric_impute_values[col] = 0.0 if pd.isna(median_value) else float(median_value)
    feature_names.extend(numeric_features)
    return {
        "features": list(features),
        "categorical_features": resolved_categorical,
        "numeric_features": numeric_features,
        "categorical_mappings": categorical_mappings,
        "numeric_impute_values": numeric_impute_values,
        "feature_names": feature_names,
    }


def _transform_feature_encoder(
    df: pd.DataFrame,
    encoder: dict[str, Any],
) -> pd.DataFrame:
    """Transform features with a fitted encoder and explicit unknown buckets."""
    selected, _, _ = _coerce_feature_subset(
        df,
        encoder["features"],
        encoder["categorical_features"],
    )
    encoded_columns: dict[str, pd.Series] = {}

    for col in encoder["categorical_features"]:
        mapping = encoder["categorical_mappings"][col]
        values = selected[col].astype("string")
        all_levels = pd.Index(mapping["all_levels"], dtype="string")
        for level in mapping["retained_levels"]:
            encoded_columns[f"{col}_{level}"] = values.eq(level).fillna(False).astype(float)
        missing_mask = values.isna()
        unknown_mask = values.notna() & ~values.isin(all_levels)
        encoded_columns[mapping["unknown_column"]] = unknown_mask.astype(float)
        encoded_columns[mapping["missing_column"]] = missing_mask.astype(float)

    for col in encoder["numeric_features"]:
        impute_value = float(encoder["numeric_impute_values"].get(col, 0.0))
        numeric_series = pd.to_numeric(selected[col], errors="coerce")
        encoded_columns[col] = numeric_series.fillna(impute_value).astype(float)

    encoded = pd.DataFrame(encoded_columns, index=selected.index)
    encoded = encoded.reindex(columns=encoder["feature_names"], fill_value=0.0)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    for col in encoder["numeric_features"]:
        encoded[col] = encoded[col].fillna(float(encoder["numeric_impute_values"].get(col, 0.0)))
    encoded = encoded.fillna(0.0)
    return encoded


def _encode_features(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode categorical features, leave numeric ones intact.

    Returns a new DataFrame suitable for sklearn-style estimators.
    """
    encoder = _fit_feature_encoder(df, features, categorical_features)
    return _transform_feature_encoder(df, encoder)


def _sksurv_c_index(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
) -> float | None:
    """Compute Harrell's C-index via *sksurv* if available, else fall back
    to the pure-Python implementation from ``analysis.py``.
    """
    events = y_true["event"].astype(bool)
    times = y_true["time"].astype(float)

    if SKSURV_AVAILABLE:
        try:
            c_index, _, _, _, _ = concordance_index_censored(events, times, risk_scores)
            return _safe_float(c_index)
        except (MemoryError, KeyboardInterrupt):
            raise
        except Exception:
            pass

    return _harrell_c_index(
        times,
        events.astype(int).astype(float),
        risk_scores.astype(float),
    )


def _scientific_summary_ml(
    *,
    model_name: str,
    c_index: float | None,
    n_patients: int,
    n_events: int,
    n_features: int,
    evaluation_mode: str = "apparent",
    n_evaluation_patients: int | None = None,
    n_evaluation_events: int | None = None,
    n_fit_patients: int | None = None,
    n_fit_events: int | None = None,
    extra_strengths: Sequence[str] | None = None,
    extra_cautions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate an insight-board payload (headline, strengths, cautions,
    next_steps, metrics, status) following the same shape as
    ``_km_scientific_summary`` in *analysis.py*.
    """
    metric_name = _metric_name_for_evaluation(evaluation_mode)
    eval_n = int(n_evaluation_patients if n_evaluation_patients is not None else n_patients)
    eval_events = int(n_evaluation_events if n_evaluation_events is not None else n_events)
    fit_n = int(n_fit_patients if n_fit_patients is not None else n_patients)
    fit_events = int(n_fit_events if n_fit_events is not None else n_events)
    strengths: list[str] = [
        f"{model_name} was trained with {n_features} feature(s) on {fit_n} patients ({fit_events} events).",
        f"{metric_name} was estimated on {eval_n} patients ({eval_events} events).",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    if extra_strengths:
        strengths.extend(extra_strengths)
    if extra_cautions:
        cautions.extend(extra_cautions)

    epv = fit_events / max(n_features, 1)
    if epv < 10:
        cautions.append(
            f"Events per feature is {epv:.1f} (< 10); results may be overfit."
        )
        next_steps.append(
            "Reduce the number of features or increase the cohort size before trusting these estimates."
        )

    if fit_events < 20:
        cautions.append("Fewer than 20 total events limits model reliability.")

    if evaluation_mode == "holdout":
        strengths.append("Discrimination was estimated on a deterministic holdout split.")
    elif evaluation_mode == "repeated_cv":
        strengths.append("Discrimination was estimated with repeated stratified cross-validation.")
    else:
        cautions.append(
            "Discrimination was estimated on the training cohort because a stable holdout split was not feasible; this apparent C-index is optimistic."
        )

    if c_index is not None:
        strengths.append(f"{metric_name} = {c_index:.3f}.")
        if c_index < 0.60:
            cautions.append(f"{metric_name} is below 0.60, so discrimination is limited.")

    # Headline
    if c_index is not None:
        headline = (
            f"{model_name} achieved a {metric_name.lower()} of {c_index:.3f} on the current evaluation path."
        )
    else:
        headline = f"{model_name} training completed but the {metric_name.lower()} could not be computed."

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
            {"label": "Training patients", "value": fit_n},
            {"label": "Training events", "value": fit_events},
            {"label": "Features", "value": n_features},
            {"label": metric_name, "value": _safe_float(c_index)},
            {"label": "Evaluation mode", "value": evaluation_mode},
        ],
    }


def _metric_name_for_evaluation(evaluation_mode: str) -> str:
    if evaluation_mode == "holdout":
        return "Holdout C-index"
    if evaluation_mode == "repeated_cv":
        return "Repeated-CV mean C-index"
    return "Apparent C-index"


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
        repeat_results.append({
            "repeat": repeat,
            "c_index": float(np.mean(c_values)),
            "c_index_std": float(np.std(c_values, ddof=1)) if len(c_values) > 1 else 0.0,
            "c_index_median": float(np.median(c_values)),
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

    repeat_c_values = np.array([float(item["c_index"]) for item in repeat_results], dtype=float)
    mean_c = float(np.mean(repeat_c_values))
    std_c = float(np.std(repeat_c_values, ddof=1)) if len(repeat_c_values) > 1 else 0.0
    interval_lower: float | None = None
    interval_upper: float | None = None
    if len(repeat_c_values) > 1:
        interval_lower = float(np.quantile(repeat_c_values, 0.025))
        interval_upper = float(np.quantile(repeat_c_values, 0.975))

    def _mean_repeat_field(field: str) -> int | None:
        values = [float(item[field]) for item in repeat_results if item.get(field) is not None]
        if not values:
            return None
        return int(round(float(np.mean(values))))

    return {
        "repeat_results": repeat_results,
        "c_index": mean_c,
        "c_index_std": std_c,
        "c_index_median": float(np.median(repeat_c_values)),
        "c_index_interval_lower": _safe_float(interval_lower),
        "c_index_interval_upper": _safe_float(interval_upper),
        "c_index_interval_label": "Empirical repeat interval (repeat means)",
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


def _split_train_test(
    frame: pd.DataFrame,
    event_column: str,
    random_state: int = 42,
    test_size: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Create a deterministic train/test split for model comparison.

    Returns the split frames and a short evaluation-mode label.
    """
    if len(frame) < 20:
        return frame.copy().reset_index(drop=True), frame.copy().reset_index(drop=True), "apparent"

    events = frame[event_column].astype(int)
    if events.nunique() < 2 or events.value_counts().min() < 4:
        return frame.copy().reset_index(drop=True), frame.copy().reset_index(drop=True), "apparent"

    train_idx, test_idx = train_test_split(
        frame.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=events.to_numpy(),
    )
    train_frame = frame.loc[train_idx].reset_index(drop=True)
    test_frame = frame.loc[test_idx].reset_index(drop=True)
    return train_frame, test_frame, "holdout"


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
        raise ValueError("No non-constant encoded features remain for Cox PH.")
    return (
        train_encoded.loc[:, varying_columns].copy(),
        eval_encoded.loc[:, varying_columns].copy(),
    )


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

    Falls back to apparent evaluation if a holdout split becomes unusable after
    encoding or missing-value filtering.
    """
    categorical_features = list(categorical_features or [])

    def _encode_split(
        train_frame: pd.DataFrame,
        eval_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        train_encoded, eval_encoded, encoder = _encode_train_test_features(
            train_frame,
            eval_frame,
            features,
            categorical_features,
        )
        if train_encoded.isna().any().any() or eval_encoded.isna().any().any():
            raise ValueError("Feature encoding produced non-finite values after imputation.")
        train_encoded = train_encoded.reset_index(drop=True)
        eval_encoded = eval_encoded.reset_index(drop=True)
        train_clean = train_frame.reset_index(drop=True)
        eval_clean = eval_frame.reset_index(drop=True)
        return train_clean, eval_clean, train_encoded, eval_encoded, encoder

    train_frame, eval_frame, evaluation_mode = _split_train_test(
        frame,
        event_column,
        random_state=random_state,
    )
    train_frame, eval_frame, train_encoded, eval_encoded, encoder = _encode_split(train_frame, eval_frame)

    if train_encoded.empty or eval_encoded.empty:
        evaluation_mode = "apparent"
        base_frame = frame.copy().reset_index(drop=True)
        train_frame, eval_frame, train_encoded, eval_encoded, encoder = _encode_split(base_frame, base_frame)

    if train_encoded.empty or eval_encoded.empty:
        raise ValueError("No valid rows remain after encoding features for model evaluation.")

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


def find_optimal_cutpoint(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    variable: str,
    event_positive_value: Any = None,
    min_group_fraction: float = 0.1,
    lower_label: str = "Low",
    upper_label: str = "High",
    permutation_iterations: int = 100,
    random_seed: int = 20260311,
) -> dict[str, Any]:
    """Scan all unique values of *variable* and return the cutpoint that
    maximises the log-rank chi-square statistic.

    Returns
    -------
    dict
        Keys: ``optimal_cutpoint``, ``p_value`` (selection-adjusted when
        permutations are enabled), ``raw_p_value`` (unadjusted), ``n_high``,
        ``n_low``, ``scan_data`` (list of per-cutpoint records), and
        ``split_column`` (name of the derived grouping column).
    """
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=[variable],
    )

    numeric_values = pd.to_numeric(frame[variable], errors="coerce")
    frame = frame.loc[numeric_values.notna()].reset_index(drop=True)
    numeric_values = pd.to_numeric(frame[variable], errors="coerce")

    if frame.empty:
        raise ValueError(f"No valid numeric values in '{variable}' after cleaning.")

    time_values = frame[time_column].to_numpy(dtype=float)
    event_values = frame[event_column].to_numpy(dtype=float)
    var_values = numeric_values.to_numpy(dtype=float)
    n_total = len(frame)
    min_size = max(int(math.ceil(n_total * min_group_fraction)), 1)

    unique_vals = np.unique(var_values)
    if len(unique_vals) < 2:
        raise ValueError(
            f"'{variable}' has fewer than 2 unique values; a cutpoint split is not possible."
        )

    # Candidate cutpoints: midpoints between consecutive sorted unique values
    candidates = 0.5 * (unique_vals[:-1] + unique_vals[1:])

    scan_data: list[dict[str, Any]] = []
    best_stat = -1.0
    best_record: dict[str, Any] | None = None

    for cutpoint in candidates:
        high_mask = var_values > cutpoint
        n_high = int(high_mask.sum())
        n_low = n_total - n_high

        if n_high < min_size or n_low < min_size:
            continue

        # Require at least one event in each group
        if event_values[high_mask].sum() == 0 or event_values[~high_mask].sum() == 0:
            continue

        try:
            groups = np.where(high_mask, upper_label, lower_label)
            chisq, p_value = survdiff(time_values, event_values, groups)
        except _EXPECTED_CUTPOINT_SCAN_ERRORS:
            continue

        record = {
            "cutpoint": _safe_float(cutpoint),
            "statistic": _safe_float(chisq),
            "p_value": _safe_float(p_value),
            "n_high": n_high,
            "n_low": n_low,
        }
        scan_data.append(record)

        if chisq > best_stat:
            best_stat = chisq
            best_record = record

    if best_record is None:
        raise ValueError(
            f"No valid cutpoint found for '{variable}'. "
            "Ensure min_group_fraction allows at least one feasible split."
        )

    # Build the split — assign labels based on actual survival direction
    optimal_cp = best_record["cutpoint"]
    high_mask = var_values > optimal_cp

    def _rmst(times: np.ndarray, events: np.ndarray, horizon: float) -> float | None:
        """Restricted mean survival time from a KM step curve."""
        from statsmodels.duration.survfunc import SurvfuncRight as _SFR

        try:
            sf = _SFR(times, events)
        except Exception:
            return None
        event_times = sf.surv_times.astype(float)
        step_timeline = np.concatenate(([0.0], event_times))
        step_survival = np.concatenate(([1.0], sf.surv_prob.astype(float)))
        horizon = float(max(horizon, 1e-9))
        area = 0.0
        clipped = np.clip(step_timeline, 0, horizon)
        for i in range(len(clipped) - 1):
            left = clipped[i]
            right = clipped[i + 1]
            if right > left:
                area += float(step_survival[i]) * float(right - left)
        if clipped[-1] < horizon:
            area += float(step_survival[-1]) * float(horizon - clipped[-1])
        return float(area)

    # Compare median survival (fallback to RMST if medians are not estimable).
    from statsmodels.duration.survfunc import SurvfuncRight as _SFR

    median_above = None
    median_below = None
    try:
        sf_above = _SFR(time_values[high_mask], event_values[high_mask])
        median_above = float(sf_above.quantile(0.5))
    except Exception:
        pass
    try:
        sf_below = _SFR(time_values[~high_mask], event_values[~high_mask])
        median_below = float(sf_below.quantile(0.5))
    except Exception:
        pass

    # If values > cutpoint have LONGER survival, swap labels
    swap = False
    if median_above is not None and median_below is not None:
        swap = median_above > median_below
    elif median_above is None and median_below is not None:
        swap = True  # above group never reaches median → longer survival
    elif median_above is not None and median_below is None:
        swap = False  # below group lives longer (never reaches median), above = High is correct
    elif median_above is None and median_below is None:
        horizon = float(np.max(time_values))
        rmst_above = _rmst(time_values[high_mask], event_values[high_mask], horizon=horizon)
        rmst_below = _rmst(time_values[~high_mask], event_values[~high_mask], horizon=horizon)
        if rmst_above is not None and rmst_below is not None:
            swap = rmst_above > rmst_below

    if swap:
        label_above, label_below = lower_label, upper_label
    else:
        label_above, label_below = upper_label, lower_label

    split_col_name = f"{variable}_group"
    split_series = pd.Series(
        np.where(high_mask, label_above, label_below),
        index=frame.index,
        dtype="string",
    )

    selection_adjusted_p_value = None
    perm_valid = 0
    if permutation_iterations > 0:
        rng = np.random.default_rng(int(random_seed))
        extreme = 0
        for _ in range(int(permutation_iterations)):
            perm_valid += 1
            permuted_var = rng.permutation(var_values)
            perm_best = -1.0
            for cutpoint in candidates:
                high_mask_perm = permuted_var > cutpoint
                n_high = int(high_mask_perm.sum())
                n_low = n_total - n_high
                if n_high < min_size or n_low < min_size:
                    continue
                if event_values[high_mask_perm].sum() == 0 or event_values[~high_mask_perm].sum() == 0:
                    continue
                try:
                    chisq_perm, _ = survdiff(time_values, event_values, np.where(high_mask_perm, "A", "B"))
                except _EXPECTED_CUTPOINT_SCAN_ERRORS:
                    continue
                if float(chisq_perm) > perm_best:
                    perm_best = float(chisq_perm)
            if perm_best >= float(best_stat):
                extreme += 1
        selection_adjusted_p_value = float((extreme + 1) / (perm_valid + 1)) if perm_valid else None

    raw_p_value = best_record["p_value"]
    primary_p_value = _safe_float(selection_adjusted_p_value)
    if primary_p_value is None:
        primary_p_value = raw_p_value

    return {
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
        "n_high": best_record["n_high"],
        "n_low": best_record["n_low"],
        "label_above_cutpoint": label_above,
        "label_below_cutpoint": label_below,
        "scan_data": scan_data,
        "split_column": split_col_name,
    }


# ===================================================================
# 2. Random Survival Forest
# ===================================================================


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
) -> dict[str, Any]:
    """Train a Random Survival Forest and return model statistics,
    feature importances, predicted risk scores, and a scientific summary.
    """
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for Random Survival Forest. "
            "Install it with: pip install scikit-survival"
        )

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )

    if internal_evaluation:
        split = _prepare_model_evaluation_split(
            frame,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        train_frame = split["train_frame"]
        eval_frame = split["eval_frame"]
        full_frame = split["full_frame"]
        train_encoded = split["train_encoded"]
        eval_encoded = split["eval_encoded"]
        full_encoded = split["full_encoded"]
        evaluation_mode = split["evaluation_mode"]
        metric_name = split["metric_name"]
        feature_encoder = split["feature_encoder"]
    else:
        feature_encoder = _fit_feature_encoder(frame, features, categorical_features)
        full_encoded = _transform_feature_encoder(frame, feature_encoder).reset_index(drop=True)
        full_frame = frame.reset_index(drop=True)
        if full_encoded.empty:
            raise ValueError("No analyzable rows remain after encoding features for RSF.")
        train_frame = eval_frame = full_frame
        train_encoded = eval_encoded = full_encoded
        evaluation_mode = "apparent"
        metric_name = _metric_name_for_evaluation(evaluation_mode)
    y_train = _prepare_sksurv_data(train_frame, time_column, event_column)
    y_eval = _prepare_sksurv_data(eval_frame, time_column, event_column)
    y_full = _prepare_sksurv_data(full_frame, time_column, event_column)

    t_start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)

    evaluation_risk_scores = model.predict(eval_encoded.to_numpy())
    c_index = _sksurv_c_index(y_eval, evaluation_risk_scores)
    risk_scores = model.predict(full_encoded.to_numpy())

    # Feature importance — try built-in, fall back to permutation
    feature_names = list(train_encoded.columns)
    try:
        importances = model.feature_importances_
    except NotImplementedError:
        # sksurv >=0.24 removed built-in importances for RSF
        from sklearn.inspection import permutation_importance as _perm_imp

        perm_eval_encoded = eval_encoded
        perm_y_eval = y_eval
        if int(eval_encoded.shape[0]) > 120:
            perm_rng = np.random.default_rng(random_state)
            sampled_idx = np.sort(perm_rng.choice(eval_encoded.shape[0], size=120, replace=False))
            perm_eval_encoded = eval_encoded.iloc[sampled_idx].reset_index(drop=True)
            perm_y_eval = y_eval[sampled_idx]
        perm_repeats = 3 if int(perm_eval_encoded.shape[1]) <= 20 else 2
        perm_result = _perm_imp(
            model,
            perm_eval_encoded.to_numpy(),
            perm_y_eval,
            n_repeats=perm_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        importances = perm_result.importances_mean
    importance_records = sorted(
        [
            {"feature": name, "importance": _safe_float(imp)}
            for name, imp in zip(feature_names, importances, strict=False)
        ],
        key=lambda r: r["importance"] if r["importance"] is not None else 0.0,
        reverse=True,
    )

    n_patients = int(full_frame.shape[0])
    n_events = int(full_frame[event_column].sum())
    n_eval_patients = int(eval_frame.shape[0])
    n_eval_events = int(eval_frame[event_column].sum())

    scientific_summary = _scientific_summary_ml(
        model_name="Random Survival Forest",
        c_index=c_index,
        n_patients=n_patients,
        n_events=n_events,
        n_features=len(feature_names),
        evaluation_mode=evaluation_mode,
        n_evaluation_patients=n_eval_patients,
        n_evaluation_events=n_eval_events,
        n_fit_patients=int(train_frame.shape[0]),
        n_fit_events=int(train_frame[event_column].sum()),
        extra_strengths=[
            f"Ensemble of {n_estimators} trees with min_samples_leaf={min_samples_leaf}.",
            "Non-parametric model; no proportional-hazards assumption required.",
        ],
    )

    return {
        "model_type": "RandomSurvivalForest",
        "model_stats": {
            "c_index": _safe_float(c_index),
            "metric_name": metric_name,
            "evaluation_mode": evaluation_mode,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_patients": n_patients,
            "n_events": n_events,
            "n_evaluation_patients": n_eval_patients,
            "n_evaluation_events": n_eval_events,
            "n_features": len(feature_names),
            "training_time_ms": training_time_ms,
        },
        "feature_importance": importance_records,
        "predicted_risk_scores": [_safe_float(v) for v in risk_scores],
        "evaluation_risk_scores": [_safe_float(v) for v in evaluation_risk_scores],
        "feature_names": feature_names,
        "scientific_summary": scientific_summary,
        "_model": model,
        "_X_encoded": full_encoded,
        "_X_eval_encoded": eval_encoded,
        "_feature_encoder": feature_encoder,
        "_analysis_frame": full_frame,
        "_analysis_eval_frame": eval_frame,
        "_y": y_full,
        "_y_eval": y_eval,
    }


# ===================================================================
# 3. Gradient Boosted Survival Analysis
# ===================================================================


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
) -> dict[str, Any]:
    """Train a Gradient Boosted Survival model and return model statistics,
    feature importances, predicted risk scores, and a scientific summary.
    """
    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for Gradient Boosted Survival. "
            "Install it with: pip install scikit-survival"
        )

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )

    if internal_evaluation:
        split = _prepare_model_evaluation_split(
            frame,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        train_frame = split["train_frame"]
        eval_frame = split["eval_frame"]
        full_frame = split["full_frame"]
        train_encoded = split["train_encoded"]
        eval_encoded = split["eval_encoded"]
        full_encoded = split["full_encoded"]
        evaluation_mode = split["evaluation_mode"]
        metric_name = split["metric_name"]
        feature_encoder = split["feature_encoder"]
    else:
        feature_encoder = _fit_feature_encoder(frame, features, categorical_features)
        full_encoded = _transform_feature_encoder(frame, feature_encoder).reset_index(drop=True)
        full_frame = frame.reset_index(drop=True)
        if full_encoded.empty:
            raise ValueError("No analyzable rows remain after encoding features for GBS.")
        train_frame = eval_frame = full_frame
        train_encoded = eval_encoded = full_encoded
        evaluation_mode = "apparent"
        metric_name = _metric_name_for_evaluation(evaluation_mode)
    y_train = _prepare_sksurv_data(train_frame, time_column, event_column)
    y_eval = _prepare_sksurv_data(eval_frame, time_column, event_column)
    y_full = _prepare_sksurv_data(full_frame, time_column, event_column)

    t_start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth if max_depth is not None else 3,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)

    evaluation_risk_scores = model.predict(eval_encoded.to_numpy())
    c_index = _sksurv_c_index(y_eval, evaluation_risk_scores)
    risk_scores = model.predict(full_encoded.to_numpy())

    importances = model.feature_importances_
    feature_names = list(train_encoded.columns)
    importance_records = sorted(
        [
            {"feature": name, "importance": _safe_float(imp)}
            for name, imp in zip(feature_names, importances, strict=False)
        ],
        key=lambda r: r["importance"] if r["importance"] is not None else 0.0,
        reverse=True,
    )

    n_patients = int(full_frame.shape[0])
    n_events = int(full_frame[event_column].sum())
    n_eval_patients = int(eval_frame.shape[0])
    n_eval_events = int(eval_frame[event_column].sum())

    scientific_summary = _scientific_summary_ml(
        model_name="Gradient Boosted Survival",
        c_index=c_index,
        n_patients=n_patients,
        n_events=n_events,
        n_features=len(feature_names),
        evaluation_mode=evaluation_mode,
        n_evaluation_patients=n_eval_patients,
        n_evaluation_events=n_eval_events,
        n_fit_patients=int(train_frame.shape[0]),
        n_fit_events=int(train_frame[event_column].sum()),
        extra_strengths=[
            f"Boosted ensemble with {n_estimators} stages, learning_rate={learning_rate}, max_depth={max_depth}.",
            "Non-parametric model; no proportional-hazards assumption required.",
        ],
    )

    return {
        "model_type": "GradientBoostingSurvivalAnalysis",
        "model_stats": {
            "c_index": _safe_float(c_index),
            "metric_name": metric_name,
            "evaluation_mode": evaluation_mode,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_patients": n_patients,
            "n_events": n_events,
            "n_evaluation_patients": n_eval_patients,
            "n_evaluation_events": n_eval_events,
            "n_features": len(feature_names),
            "training_time_ms": training_time_ms,
        },
        "feature_importance": importance_records,
        "predicted_risk_scores": [_safe_float(v) for v in risk_scores],
        "evaluation_risk_scores": [_safe_float(v) for v in evaluation_risk_scores],
        "feature_names": feature_names,
        "scientific_summary": scientific_summary,
        "_model": model,
        "_X_encoded": full_encoded,
        "_X_eval_encoded": eval_encoded,
        "_feature_encoder": feature_encoder,
        "_analysis_frame": full_frame,
        "_analysis_eval_frame": eval_frame,
        "_y": y_full,
        "_y_eval": y_eval,
    }


# ===================================================================
# 4. Model comparison
# ===================================================================


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
    """Train Cox PH, Random Survival Forest, and Gradient Boosted Survival
    on a deterministic train/test split and return a comparison table with
    holdout C-index, feature count, and training time for each model.
    """
    categorical_features = list(categorical_features or [])

    # Prepare a common clean frame
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )

    n_patients = int(frame.shape[0])
    n_events = int(frame[event_column].sum())
    comparison: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    train_frame, test_frame, evaluation_mode = _split_train_test(
        frame,
        event_column,
        random_state=random_state,
    )

    model_specs: list[tuple[str, Any, dict[str, Any]]] = [
        ("Cox PH", _fit_evaluate_cox_split, {}),
    ]
    if SKSURV_AVAILABLE:
        model_specs.extend(
            [
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
                        "max_depth": max_depth,
                    },
                ),
            ]
        )
    else:
        errors.extend(
            [
                {"model": "Random Survival Forest", "error": "scikit-survival is not installed."},
                {"model": "Gradient Boosted Survival", "error": "scikit-survival is not installed."},
            ]
        )

    for model_name, fit_fn, extra_kwargs in model_specs:
        try:
            result = fit_fn(
                train_frame,
                test_frame,
                time_column=time_column,
                event_column=event_column,
                features=features,
                categorical_features=categorical_features,
                random_state=random_state,
                **extra_kwargs,
            )
            comparison.append({
                "model": model_name,
                "c_index": _safe_float(result["c_index"]),
                "n_features": result["n_features"],
                "training_time_ms": result["training_time_ms"],
                "evaluation_mode": evaluation_mode,
            })
        except Exception as exc:
            errors.append({"model": model_name, "error": str(exc)})

    if not comparison:
        raise ValueError(
            "All models failed to train. Errors: "
            + "; ".join(f"{e['model']}: {e['error']}" for e in errors)
        )

    # Sort by C-index descending (None last)
    comparison.sort(
        key=lambda r: r["c_index"] if r["c_index"] is not None else -1.0,
        reverse=True,
    )

    best = comparison[0]
    scientific_summary = _scientific_summary_ml(
        model_name=f"Model Comparison Screening (top: {best['model']})",
        c_index=best["c_index"],
        n_patients=n_patients,
        n_events=n_events,
        n_features=best["n_features"],
        evaluation_mode=evaluation_mode,
        n_evaluation_patients=int(test_frame.shape[0]),
        n_evaluation_events=int(test_frame[event_column].sum()),
        n_fit_patients=int(train_frame.shape[0]),
        n_fit_events=int(train_frame[event_column].sum()),
        extra_strengths=[
            f"{len(comparison)} model(s) trained and compared with {evaluation_mode} evaluation.",
        ],
        extra_cautions=[
            "The top-ranked model was selected and scored on the same evaluation split; treat this as screening rather than final external validation.",
            f"{len(errors)} model(s) failed to train." if errors else None,
        ],
    )

    result = {
        "comparison_table": comparison,
        "errors": errors,
        "ranking_complete": len(errors) == 0 and len(comparison) == len(model_specs),
        "excluded_models": sorted({str(error["model"]) for error in errors}),
        "n_patients": n_patients,
        "n_events": n_events,
        "evaluation_mode": evaluation_mode,
        "scientific_summary": scientific_summary,
    }
    result["manuscript_tables"] = build_manuscript_result_tables(result)
    return result


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
    train_encoded, test_encoded, _ = _encode_train_test_features(
        train_frame,
        test_frame,
        features,
        categorical_features,
    )
    train_encoded, test_encoded = _drop_constant_train_columns(train_encoded, test_encoded)
    train_eval = train_frame.loc[train_encoded.index].reset_index(drop=True)
    test_eval = test_frame.loc[test_encoded.index].reset_index(drop=True)
    if train_encoded.empty or test_encoded.empty:
        raise ValueError("No valid rows remain after encoding features for Cox PH.")

    train_times = train_eval[time_column].to_numpy(dtype=float)
    train_status = train_eval[event_column].astype(int).to_numpy()
    test_times = test_eval[time_column].to_numpy(dtype=float)
    test_status = test_eval[event_column].astype(int).to_numpy()

    t_start = time.monotonic()
    model = PHReg(train_times, train_encoded.to_numpy(dtype=float), status=train_status, ties="efron")
    results = model.fit(disp=False)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    param_vector = np.asarray(results.params, dtype=float)
    risk_score = (test_encoded.to_numpy(dtype=float) @ param_vector).astype(float)
    fit_components = [param_vector, risk_score]
    llf_value = float(results.llf) if getattr(results, "llf", None) is not None else np.nan
    if (not np.isfinite(llf_value)) or any(not np.isfinite(component).all() for component in fit_components):
        raise ValueError(
            "Cox PH fit produced non-finite estimates on the shared evaluation path. "
            "This usually means redundant covariates, sparse categories, or quasi-complete separation."
        )

    return {
        "model": "Cox PH",
        "c_index": _safe_float(_harrell_c_index(test_times, test_status.astype(float), risk_score)),
        "n_features": len(param_vector),
        "training_time_ms": training_time_ms,
        "train_n": int(train_eval.shape[0]),
        "test_n": int(test_eval.shape[0]),
        "train_events": int(train_eval[event_column].sum()),
        "test_events": int(test_eval[event_column].sum()),
    }


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

    train_encoded, test_encoded, _ = _encode_train_test_features(
        train_frame,
        test_frame,
        features,
        categorical_features,
    )
    train_eval = train_frame.loc[train_encoded.index].reset_index(drop=True)
    test_eval = test_frame.loc[test_encoded.index].reset_index(drop=True)
    if train_encoded.empty or test_encoded.empty:
        raise ValueError("No valid rows remain after encoding features for Random Survival Forest.")

    y_train = _prepare_sksurv_data(train_eval, time_column, event_column)
    t_start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    risk_score = model.predict(test_encoded.to_numpy())

    return {
        "model": "Random Survival Forest",
        "c_index": _safe_float(
            _harrell_c_index(
                test_eval[time_column].to_numpy(dtype=float),
                test_eval[event_column].astype(int).to_numpy().astype(float),
                risk_score,
            )
        ),
        "n_features": int(train_encoded.shape[1]),
        "training_time_ms": training_time_ms,
        "train_n": int(train_eval.shape[0]),
        "test_n": int(test_eval.shape[0]),
        "train_events": int(train_eval[event_column].sum()),
        "test_events": int(test_eval[event_column].sum()),
    }


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
    max_depth: int = 3,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is not installed.")

    train_encoded, test_encoded, _ = _encode_train_test_features(
        train_frame,
        test_frame,
        features,
        categorical_features,
    )
    train_eval = train_frame.loc[train_encoded.index].reset_index(drop=True)
    test_eval = test_frame.loc[test_encoded.index].reset_index(drop=True)
    if train_encoded.empty or test_encoded.empty:
        raise ValueError("No valid rows remain after encoding features for Gradient Boosted Survival.")

    y_train = _prepare_sksurv_data(train_eval, time_column, event_column)
    t_start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth if max_depth is not None else 3,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(train_encoded.to_numpy(), y_train)
    training_time_ms = round((time.monotonic() - t_start) * 1000, 1)
    risk_score = model.predict(test_encoded.to_numpy())

    return {
        "model": "Gradient Boosted Survival",
        "c_index": _safe_float(
            _harrell_c_index(
                test_eval[time_column].to_numpy(dtype=float),
                test_eval[event_column].astype(int).to_numpy().astype(float),
                risk_score,
            )
        ),
        "n_features": int(train_encoded.shape[1]),
        "training_time_ms": training_time_ms,
        "train_n": int(train_eval.shape[0]),
        "test_n": int(test_eval.shape[0]),
        "train_events": int(train_eval[event_column].sum()),
        "test_events": int(test_eval[event_column].sum()),
    }


def build_manuscript_result_tables(result: dict[str, Any]) -> dict[str, Any]:
    """Build manuscript-oriented model performance tables from a comparison result."""
    evaluation_mode = str(result.get("evaluation_mode", "unknown"))
    rows: list[dict[str, Any]] = []
    comparison_table = list(result.get("comparison_table", []))
    repeated_cv_mode = evaluation_mode in {"repeated_cv", "repeated_cv_incomplete"}

    def _format_seed_values(values: Any) -> str | None:
        if not values:
            return None
        try:
            ints = [int(value) for value in values]
        except Exception:
            return None
        return ", ".join(str(value) for value in ints) if ints else None

    if repeated_cv_mode:
        interval_label = "Empirical repeat interval (repeat means)"
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
                "Rank": rank,
                "Model": row["model"],
                "Validation Strategy": (
                    f"{row.get('cv_repeats', 1)}x{row.get('cv_folds', 1)} repeated stratified CV"
                    if row_mode == "repeated_cv"
                    else f"{row.get('cv_repeats', 1)}x{row.get('cv_folds', 1)} repeated stratified CV (incomplete)"
                ),
                "Mean C-index": _safe_float(row.get("c_index")),
                "SD": _safe_float(row.get("c_index_std")),
                "Repeat means, n": row.get("n_repeats"),
                interval_label: (
                    f"{interval_lower:.3f} to {interval_upper:.3f}"
                    if interval_lower is not None and interval_upper is not None
                    else None
                ),
                "Evaluations, n": row.get("n_evaluations"),
                "Failures, n": row.get("n_failures"),
                "Features, n": row.get("n_features"),
                "Mean Training Time, ms": _safe_float(row.get("training_time_ms")),
            }
            if include_fallback_counts:
                manuscript_row["Apparent fallback folds, n"] = int(row.get("n_apparent_fallbacks", 0) or 0)
            if include_provenance:
                manuscript_row["Training seeds"] = _format_seed_values(row.get("training_seeds"))
                manuscript_row["Split seeds"] = _format_seed_values(row.get("split_seeds"))
                manuscript_row["Monitor seeds"] = _format_seed_values(row.get("monitor_seeds"))
            rows.append(manuscript_row)
        table_notes = [
            "C-index values summarize repeat-level means from repeated stratified cross-validation.",
            "The empirical repeat interval reports the 2.5th to 97.5th percentiles of repeat mean C-index values.",
            "The repeat interval is descriptive across repeats and should not be interpreted as a formal confidence interval.",
            "Blank C-index fields indicate incomplete repeated-CV evaluation because one or more folds failed or fell back to apparent evaluation.",
        ]
        if evaluation_mode == "repeated_cv_incomplete":
            table_notes.append(
                "This comparison is labeled repeated-CV incomplete because one or more apparent-fallback or failed folds were excluded from the aggregate."
            )
        if include_provenance:
            table_notes.append(
                "Training, split, and monitor seed columns record the exact repeated-CV partitioning used for replay under the same model settings."
            )
        if include_fallback_counts:
            table_notes.append(
                "Apparent fallback fold counts report folds that were excluded from the repeated-CV aggregate because a clean holdout concordance estimate could not be retained."
            )
    else:
        row_modes = {str(row.get("evaluation_mode", evaluation_mode)) for row in comparison_table}
        for rank, row in enumerate(comparison_table, start=1):
            rank_value = row.get("rank", rank)
            rows.append({
                "Rank": rank_value if rank_value is not None else "Not ranked",
                "Model": row["model"],
                "Validation Strategy": row.get("evaluation_mode", evaluation_mode),
                "C-index": _safe_float(row.get("c_index")),
                "Features, n": row.get("n_features"),
                "Training Time, ms": _safe_float(row.get("training_time_ms")),
            })
        table_notes = [
            "C-index values come from a single deterministic holdout split unless evaluation_mode is apparent.",
            "Apparent evaluation indicates resubstitution on the analyzable cohort and should not be interpreted as external validation.",
        ]
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
            "Table 1. Discrimination performance of survival models under repeated stratified cross-validation."
            if evaluation_mode == "repeated_cv"
            else (
                "Table 1. Discrimination performance of survival models under incomplete repeated stratified cross-validation."
                if evaluation_mode == "repeated_cv_incomplete"
                else (
                    "Table 1. Discrimination performance of survival models under apparent evaluation."
                    if comparison_table and all(str(row.get("evaluation_mode", evaluation_mode)) in {"apparent", "holdout_fallback_apparent"} for row in comparison_table)
                    else (
                        "Table 1. Discrimination performance of survival models under mixed holdout/apparent evaluation."
                        if any(str(row.get("evaluation_mode", evaluation_mode)) != "holdout" for row in comparison_table)
                        else "Table 1. Discrimination performance of survival models under deterministic holdout evaluation."
                    )
                )
            )
        ),
    }


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
) -> dict[str, Any]:
    """Evaluate Cox PH, RSF, and GBS with repeated stratified cross-validation."""
    categorical_features = list(categorical_features or [])
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if cv_repeats < 1:
        raise ValueError("cv_repeats must be at least 1.")

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )
    n_patients = int(frame.shape[0])
    n_events = int(frame[event_column].sum())
    events = frame[event_column].astype(int).to_numpy()
    unique, counts = np.unique(events, return_counts=True)
    if len(unique) < 2 or counts.min() < cv_folds:
        raise ValueError(
            f"Repeated CV requires at least {cv_folds} samples in each event stratum after cleaning."
        )

    model_specs: list[tuple[str, Any, dict[str, Any]]] = [("Cox PH", _fit_evaluate_cox_split, {})]
    if SKSURV_AVAILABLE:
        model_specs.extend(
            [
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
                        "max_depth": max_depth,
                    },
                ),
            ]
        )
    else:
        errors = [
            {"model": "Random Survival Forest", "error": "scikit-survival is not installed."},
            {"model": "Gradient Boosted Survival", "error": "scikit-survival is not installed."},
        ]
    fold_results: list[dict[str, Any]] = []
    if SKSURV_AVAILABLE:
        errors = []

    for repeat_idx in range(cv_repeats):
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state + repeat_idx,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(frame, events), start=1):
            train_frame = frame.iloc[train_idx].reset_index(drop=True)
            test_frame = frame.iloc[test_idx].reset_index(drop=True)
            for model_name, fit_fn, extra_kwargs in model_specs:
                try:
                    result = fit_fn(
                        train_frame,
                        test_frame,
                        time_column=time_column,
                        event_column=event_column,
                        features=features,
                        categorical_features=categorical_features,
                        random_state=random_state + repeat_idx,
                        **extra_kwargs,
                    )
                    fold_results.append({
                        "model": model_name,
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx,
                        "c_index": result["c_index"],
                        "n_features": result["n_features"],
                        "training_time_ms": result["training_time_ms"],
                        "train_n": result["train_n"],
                        "test_n": result["test_n"],
                        "train_events": result["train_events"],
                        "test_events": result["test_events"],
                    })
                except Exception as exc:
                    errors.append({
                        "model": model_name,
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx,
                        "error": str(exc),
                    })

    comparison: list[dict[str, Any]] = []
    for model_name, _, _ in model_specs:
        model_rows = [row for row in fold_results if row["model"] == model_name and row["c_index"] is not None]
        n_failures = sum(1 for err in errors if err["model"] == model_name)
        expected_evaluations = cv_folds * cv_repeats
        summary = _summarize_repeated_cv_rows(model_rows) if model_rows else None
        incomplete = (len(model_rows) + n_failures) < expected_evaluations or n_failures > 0
        if summary is None and n_failures == 0:
            continue
        comparison.append({
            "model": model_name,
            "c_index": None if incomplete or summary is None else _safe_float(summary["c_index"]),
            "c_index_std": None if incomplete or summary is None else _safe_float(summary["c_index_std"]),
            "c_index_median": None if incomplete or summary is None else _safe_float(summary["c_index_median"]),
            "c_index_interval_lower": None if incomplete or summary is None else _safe_float(summary["c_index_interval_lower"]),
            "c_index_interval_upper": None if incomplete or summary is None else _safe_float(summary["c_index_interval_upper"]),
            "c_index_interval_label": None if summary is None else summary["c_index_interval_label"],
            "n_features": None if summary is None else int(summary["n_features"]),
            "training_time_ms": None if summary is None else _safe_float(summary["training_time_ms"]),
            "n_evaluations": len(model_rows),
            "n_repeats": 0 if summary is None else int(summary["n_repeats"]),
            "n_failures": n_failures,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "evaluation_mode": "repeated_cv_incomplete" if incomplete else "repeated_cv",
            "repeat_results": [] if summary is None else summary["repeat_results"],
            "training_samples": None if summary is None else int(summary["train_n"]),
            "evaluation_samples": None if summary is None else int(summary["test_n"]),
            "train_events": None if summary is None else int(summary["train_events"]),
            "test_events": None if summary is None else int(summary["test_events"]),
        })

    if not comparison:
        raise ValueError(
            "All repeated-CV model fits failed. Errors: "
            + "; ".join(f"{e['model']} r{e['repeat']}f{e['fold']}: {e['error']}" for e in errors)
        )

    comparison.sort(key=lambda row: row["c_index"] if row["c_index"] is not None else -1.0, reverse=True)
    mean_train_n = int(round(np.mean([row["train_n"] for row in fold_results]))) if fold_results else n_patients
    mean_test_n = int(round(np.mean([row["test_n"] for row in fold_results]))) if fold_results else n_patients
    mean_train_events = int(round(np.mean([row["train_events"] for row in fold_results]))) if fold_results else n_events
    mean_test_events = int(round(np.mean([row["test_events"] for row in fold_results]))) if fold_results else n_events
    best = comparison[0]
    aggregate_mode = (
        "repeated_cv_incomplete"
        if errors or any(str(row.get("evaluation_mode")) == "repeated_cv_incomplete" for row in comparison)
        else "repeated_cv"
    )
    scientific_summary = _scientific_summary_ml(
        model_name=f"Repeated-CV Model Comparison Screening (top: {best['model']})",
        c_index=best["c_index"],
        n_patients=n_patients,
        n_events=n_events,
        n_features=best["n_features"],
        evaluation_mode=aggregate_mode,
        n_evaluation_patients=mean_test_n,
        n_evaluation_events=mean_test_events,
        n_fit_patients=mean_train_n,
        n_fit_events=mean_train_events,
        extra_strengths=[
            f"{len(comparison)} model(s) evaluated across {cv_repeats} repeat(s) of {cv_folds}-fold stratified CV.",
        ],
        extra_cautions=[f"{len(errors)} fold-level fit(s) failed."] if errors else None,
    )
    scientific_summary["cautions"].insert(
        0,
        "The top-ranked model was selected and scored within the same repeated-CV screening run; treat this as model screening rather than final external validation.",
    )

    result = {
        "comparison_table": comparison,
        "fold_results": fold_results,
        "repeat_results": [row["repeat_results"] for row in comparison],
        "errors": errors,
        "ranking_complete": not errors and all(row.get("c_index") is not None for row in comparison),
        "excluded_models": sorted({str(error["model"]) for error in errors}),
        "n_patients": n_patients,
        "n_events": n_events,
        "evaluation_mode": aggregate_mode,
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        "scientific_summary": scientific_summary,
    }
    result["manuscript_tables"] = build_manuscript_result_tables(result)
    return result


# ===================================================================
# 5. SHAP values
# ===================================================================


def compute_shap_values(
    model: Any,
    X_encoded: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
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

    X_array = X_encoded.to_numpy() if isinstance(X_encoded, pd.DataFrame) else np.asarray(X_encoded)

    shap_method = "tree"
    X_eval = X_array
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_array)
    except Exception:
        # scikit-survival estimators are often unsupported by TreeExplainer.
        # Fall back to a tightly capped KernelExplainer run so single-model
        # training does not appear hung on larger cohorts.
        shap_method = "kernel"
        rng = np.random.default_rng(20260311)
        n = int(X_array.shape[0])
        bg_n = min(20, n)
        eval_n = min(20, n)
        bg_idx = rng.choice(n, size=bg_n, replace=False) if n > bg_n else np.arange(n)
        X_bg = X_array[bg_idx]
        X_eval = X_array[:eval_n]

        def _predict_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(x), dtype=float)

        explainer = shap.KernelExplainer(_predict_fn, X_bg)
        kernel_nsamples = min(60, max(20, X_bg.shape[1] * 8))
        try:
            shap_values = explainer.shap_values(
                X_eval,
                nsamples=kernel_nsamples,
                silent=True,
            )
        except TypeError:
            shap_values = explainer.shap_values(X_eval, nsamples=kernel_nsamples)

    # shap_values may be 2-D (n_samples, n_features) or 3-D for multi-output
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
        "feature_importance": importance_records,
        "shap_summary": shap_summary,
        "n_samples": int(X_eval.shape[0]),
        "n_features": int(X_array.shape[1]),
    }


# ===================================================================
# 6. Partial dependence
# ===================================================================


def compute_partial_dependence(
    model: Any,
    X_encoded: pd.DataFrame,
    feature_name: str,
    n_points: int = 50,
    categorical_features: Sequence[str] | None = None,
    feature_encoder: dict[str, Any] | None = None,
    analysis_frame: pd.DataFrame | None = None,
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
    categorical_features = list(categorical_features or [])
    computation_errors: list[str] = []

    def _predict_mean_for_variant(frame_variant: pd.DataFrame) -> float | None:
        if feature_encoder is not None:
            encoded = _transform_feature_encoder(frame_variant, feature_encoder)
            encoded = encoded.reindex(columns=X_encoded.columns, fill_value=0.0).fillna(0.0)
            X_variant = encoded.to_numpy(dtype=float)
        else:
            if feature_name not in X_encoded.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found in the encoded data. "
                    f"Available: {list(X_encoded.columns)}"
                )
            X_variant = X_encoded.to_numpy(dtype=float)
            col_idx = list(X_encoded.columns).index(feature_name)
            replacement = pd.to_numeric(frame_variant[feature_name], errors="coerce").to_numpy(dtype=float)
            X_variant[:, col_idx] = replacement
        preds = model.predict(X_variant)
        return _safe_float(float(np.mean(preds)))

    if analysis_frame is not None and feature_name in analysis_frame.columns:
        if feature_name in categorical_features:
            encoder_levels: list[str] = []
            if feature_encoder is not None:
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
            category_counts = analysis_frame[feature_name].astype("string").value_counts(dropna=True)
            for category in category_values:
                frame_variant = analysis_frame.copy()
                frame_variant[feature_name] = category
                try:
                    mean_risks.append(_predict_mean_for_variant(frame_variant))
                except Exception as exc:
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
            frame_variant = analysis_frame.copy()
            frame_variant[feature_name] = float(grid_val)
            try:
                mean_risks.append(_predict_mean_for_variant(frame_variant))
            except Exception as exc:
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
        X_modified = X_array.copy()
        X_modified[:, col_idx] = grid_val
        try:
            preds = model.predict(X_modified)
            mean_risks.append(_safe_float(float(np.mean(preds))))
        except Exception as exc:
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


def compute_integrated_brier_score(
    times: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[int],
    predicted_survival_fn: Any,
    eval_times: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute Integrated Brier Score (IBS) for model calibration assessment.

    IBS is often used as a descriptive survival-prediction error score.
    It reflects time-integrated prediction error and is influenced by both
    discrimination and calibration, but it should not be over-interpreted
    as a stand-alone manuscript-grade validation result.

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
        If *None*, 100 equally-spaced points from 0 to the largest
        observed time are used.

    Returns
    -------
    dict
        ``ibs`` (float), ``brier_scores`` (list of per-time-point records),
        ``eval_times`` (list), and ``scientific_summary``.
    """
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=float)

    if len(times_arr) != len(events_arr):
        raise ValueError("times and events must have the same length.")

    n_samples = len(times_arr)

    # Default evaluation grid
    if eval_times is None:
        t_max = float(np.max(times_arr))
        eval_times_arr = np.linspace(0, t_max, 100)
    else:
        eval_times_arr = np.asarray(eval_times, dtype=float)

    # Remove eval times beyond the last observed time to avoid
    # extrapolation artefacts
    t_max_obs = float(np.max(times_arr))
    eval_times_arr = eval_times_arr[eval_times_arr <= t_max_obs]
    if len(eval_times_arr) == 0:
        raise ValueError(
            "No evaluation time points fall within the observed follow-up range."
        )

    # Predicted survival matrix: (n_samples, n_eval_times)
    surv_matrix = np.asarray(predicted_survival_fn(eval_times_arr), dtype=float)
    if surv_matrix.shape != (n_samples, len(eval_times_arr)):
        raise ValueError(
            f"predicted_survival_fn must return shape ({n_samples}, {len(eval_times_arr)}), "
            f"got {surv_matrix.shape}."
        )

    # Compute Brier Score at each evaluation time with IPCW to handle censoring.
    # Reference weight scheme (Graf et al., 1999; commonly used in survival packages):
    # - If t_i > t: weight = 1 / G_hat(t)
    # - If t_i <= t and event_i == 1: weight = 1 / G_hat(t_i)
    # - If t_i <= t and event_i == 0: weight = 0
    from statsmodels.duration.survfunc import SurvfuncRight

    censor_status = (1.0 - events_arr).astype(float)
    censor_sf = SurvfuncRight(times_arr, censor_status)
    censor_times = censor_sf.surv_times.astype(float)
    censor_surv = censor_sf.surv_prob.astype(float)

    def _G_hat(query_times: np.ndarray | float) -> np.ndarray:
        qt = np.asarray(query_times, dtype=float)
        if censor_times.size == 0:
            return np.ones_like(qt, dtype=float)
        idx = np.searchsorted(censor_times, qt, side="right") - 1
        out = np.ones_like(qt, dtype=float)
        valid_idx = idx >= 0
        out[valid_idx] = censor_surv[idx[valid_idx]]
        return out

    eps = 1e-12
    brier_scores: list[dict[str, Any]] = []
    bs_values: list[float] = []

    for j, t in enumerate(eval_times_arr):
        # Indicator: patient experienced event before or at time t
        # I(T > t) = 1 if the patient is still alive at t
        alive_indicator = (times_arr > t).astype(float)
        s_hat = surv_matrix[:, j]

        g_t = float(max(_G_hat(float(t)).item(), eps))
        g_ti = np.maximum(_G_hat(times_arr), eps)
        weights = np.zeros(n_samples, dtype=float)
        weights[times_arr > t] = 1.0 / g_t
        event_mask = (times_arr <= t) & (events_arr == 1)
        weights[event_mask] = 1.0 / g_ti[event_mask]

        bs = float(np.mean(weights * (s_hat - alive_indicator) ** 2))
        bs_values.append(bs)
        brier_scores.append({
            "time": _safe_float(float(t)),
            "score": _safe_float(bs),
        })

    # Integrated Brier Score via trapezoidal rule
    bs_arr = np.array(bs_values, dtype=float)
    if len(eval_times_arr) >= 2:
        ibs = float(np.trapz(bs_arr, eval_times_arr)) / (
            eval_times_arr[-1] - eval_times_arr[0]
        )
    else:
        ibs = float(bs_arr[0])

    # Scientific summary
    n_events = int(np.sum(events_arr))
    strengths: list[str] = [
        f"IBS computed over {len(eval_times_arr)} time points for {n_samples} patients ({n_events} events).",
        "IBS is a descriptive score that summarizes predicted-vs-observed survival error across time.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    if ibs < 0.1:
        strengths.append(f"IBS = {ibs:.4f}; smaller values imply closer agreement between predictions and observations.")
    elif ibs < 0.2:
        strengths.append(f"IBS = {ibs:.4f}; smaller values imply closer agreement between predictions and observations.")
    elif ibs < 0.25:
        cautions.append(f"IBS = {ibs:.4f}; the prediction error is moderate, so review the survival curves.")
    else:
        cautions.append(f"IBS = {ibs:.4f}; the prediction error is large, so the model needs review.")

    cautions.append(
        "This implementation uses IPCW Brier scores; if the censoring survival "
        "probability approaches 0 at late times, estimates can become unstable."
    )
    next_steps.append(
        "Compare IBS across models to select the best-calibrated predictor (lower is better)."
    )
    next_steps.append(
        "Use compute_calibration_data() for a descriptive visual check of calibration agreement."
    )

    status = "robust"
    if any("moderate" in c for c in cautions):
        status = "review"
    if ibs >= 0.25 or n_events < 10:
        status = "caution"

    scientific_summary = {
        "status": status,
        "headline": f"Integrated Brier Score = {ibs:.4f} over [{_safe_float(eval_times_arr[0])}, {_safe_float(eval_times_arr[-1])}] as a descriptive summary.",
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "IBS", "value": _safe_float(ibs)},
            {"label": "Patients", "value": n_samples},
            {"label": "Events", "value": n_events},
            {"label": "Eval time points", "value": len(eval_times_arr)},
        ],
    }

    return {
        "ibs": _safe_float(ibs),
        "brier_scores": brier_scores,
        "eval_times": [_safe_float(float(t)) for t in eval_times_arr],
        "scientific_summary": scientific_summary,
    }


# ===================================================================
# 8. Calibration curve data (XAI)
# ===================================================================


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

    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=float)
    pred_arr = np.asarray(predicted_survival_at_t, dtype=float)

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
    except Exception:
        bin_assign = None
        bin_categories = []

    if bin_assign is None or not bin_categories:
        # Fallback: single-bin summary
        bin_categories = [None]

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
        if t is not None and np.sum(bin_events) > 0:
            try:
                sf = SurvfuncRight(bin_times, bin_events)
                # Evaluate survival at the target time
                km_times = sf.surv_times
                km_surv = sf.surv_prob
                if len(km_times) > 0:
                    idx = np.searchsorted(km_times, t, side="right") - 1
                    if idx < 0:
                        observed_survival = 1.0
                    else:
                        observed_survival = float(km_surv[idx])
            except Exception:
                observed_survival = None
        elif t is not None:
            # No events in this bin: observed survival is 1.0 (no failures)
            observed_survival = 1.0

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
        f"Calibration assessed at t={time_label} across {n_bins} bins for {n_samples} patients ({n_events} events); this is a descriptive binning-based check, not a formal calibration test.",
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
            f"{empty_bins} of {n_bins} bins had no patients; "
            "consider reducing n_bins or using a larger dataset."
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
            {"label": "Bins", "value": n_bins},
            {"label": "Mean |pred - obs|", "value": _safe_float(mean_abs_diff)},
        ],
    }

    predicted_points = [b["predicted_mean"] for b in bins_result if b["predicted_mean"] is not None and b["observed"] is not None]
    observed_points = [b["observed"] for b in bins_result if b["predicted_mean"] is not None and b["observed"] is not None]

    return {
        "time_point": time_label,
        "bins": bins_result,
        "predicted": predicted_points,
        "observed": observed_points,
        "scientific_summary": scientific_summary,
    }


# ===================================================================
# 9. Time-dependent feature importance (XAI)
# ===================================================================


def compute_time_dependent_importance(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    eval_times: Sequence[float] | None = None,
    event_positive_value: Any = None,
    model_type: str = "rsf",
) -> dict[str, Any]:
    """Compute how feature importance changes over time.

    This is a proxy time-slice importance analysis, not a formal
    survival-specific attribution method. For each evaluation time point
    *t*, patients censored before *t* are excluded, the remaining known
    status labels are binarised (event before *t* → 1, else → 0), and a
    simple tree-based classifier is fitted. Feature importances from each
    classifier form rows of the importance matrix.

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
    eval_times
        Time points at which to evaluate importance.  If *None*, five
        equally-spaced quantiles of the observed event times are used.
    event_positive_value
        Passed through to ``_cohort_frame`` for event coercion.
    model_type
        Retained for API compatibility. The current implementation uses
        time-slice ``RandomForestClassifier`` models regardless of value.

    Returns
    -------
    dict
        ``eval_times``, ``features``, ``importance_matrix`` (times × features),
        ``importance_matrix_feature_major`` (features × times),
        ``importance_matrix_orientation``, ``dominant_feature_per_time``,
        and ``scientific_summary``.
    """
    _ = model_type

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=list(features),
        drop_missing_extra_columns=False,
    )
    X_encoded = _encode_features(frame, features, categorical_features).reset_index(drop=True)
    frame = frame.reset_index(drop=True)

    if X_encoded.empty:
        raise ValueError("No analyzable rows remain after encoding features.")

    time_values = frame[time_column].to_numpy(dtype=float)
    event_values = frame[event_column].to_numpy(dtype=float)
    feature_names = list(X_encoded.columns)

    # Default eval_times: quintiles of observed event times
    if eval_times is None:
        event_times = time_values[event_values == 1]
        if len(event_times) < 5:
            eval_times_arr = np.unique(event_times)
        else:
            eval_times_arr = np.quantile(event_times, [0.2, 0.4, 0.5, 0.6, 0.8])
    else:
        eval_times_arr = np.asarray(eval_times, dtype=float)

    # Remove duplicates and sort
    eval_times_arr = np.sort(np.unique(eval_times_arr))
    if len(eval_times_arr) == 0:
        raise ValueError("No valid evaluation time points.")

    importance_matrix_time_major: list[list[float | None]] = []
    dominant_per_time: list[str | None] = []
    evaluable_patients_per_time: list[int] = []
    skipped_time_points: list[dict[str, Any]] = []
    computation_errors: list[str] = []

    X_array = X_encoded.to_numpy(dtype=float)

    for t in eval_times_arr:
        # Binary outcome: event occurred at or before time t.
        # Subjects censored before t are excluded because their status at t is unknown.
        evaluable_mask = (time_values > t) | ((time_values <= t) & (event_values == 1))
        binary_outcome = ((time_values <= t) & (event_values == 1)).astype(int)
        X_t = X_array[evaluable_mask]
        y_t = binary_outcome[evaluable_mask]

        evaluable_patients = int(np.sum(evaluable_mask))
        evaluable_patients_per_time.append(evaluable_patients)

        # Need at least two classes among patients with known status at time t.
        if evaluable_patients < 2 or len(np.unique(y_t)) < 2:
            importance_matrix_time_major.append([None] * len(feature_names))
            dominant_per_time.append(None)
            skipped_time_points.append({
                "time": _safe_float(float(t)),
                "reason": "not_enough_evaluable_patients_or_classes",
                "evaluable_patients": evaluable_patients,
            })
            continue

        try:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=1,
            )
            clf.fit(X_t, y_t)
            importances = clf.feature_importances_
            row = [_safe_float(float(v)) for v in importances]
            importance_matrix_time_major.append(row)

            best_idx = int(np.argmax(importances))
            dominant_per_time.append(feature_names[best_idx])
        except Exception as exc:
            importance_matrix_time_major.append([None] * len(feature_names))
            dominant_per_time.append(None)
            computation_errors.append(f"t={float(t):.6g}: {type(exc).__name__}: {exc}")

    if computation_errors:
        raise ValueError(
            "Time-dependent importance failed for one or more evaluation times: "
            + "; ".join(computation_errors[:3])
        )

    n_patients = int(frame.shape[0])
    n_events = int(np.sum(event_values))

    # Identify features whose importance varies most across time
    valid_rows = [
        row for row in importance_matrix_time_major
        if all(v is not None for v in row)
    ]
    varying_features: list[str] = []
    if len(valid_rows) >= 2:
        imp_arr = np.array(valid_rows, dtype=float)
        stds = np.std(imp_arr, axis=0)
        top_varying_idx = np.argsort(stds)[::-1][:3]
        varying_features = [feature_names[i] for i in top_varying_idx]

    strengths: list[str] = [
        f"Importance computed at {len(eval_times_arr)} time points for {len(feature_names)} features.",
        "Provides a proxy view of how feature rankings change across early vs late follow-up.",
        "Patients censored before a given time point are excluded from that time-specific classifier.",
    ]
    cautions: list[str] = [
        "This is a proxy time-slice analysis and should not be described as formal SurvSHAP(t) or a survival-specific attribution method.",
    ]
    next_steps: list[str] = []

    if varying_features:
        strengths.append(
            f"Features with greatest temporal variation: {', '.join(varying_features)}."
        )
    if n_events < 20:
        cautions.append("Fewer than 20 events may make per-time-point estimates unreliable.")
    next_steps.append(
        "Examine the importance_matrix to identify which features dominate at early vs late time points."
    )
    next_steps.append(
        "Consider dedicated SurvSHAP(t) implementations for more precise time-dependent explanations."
    )

    unique_dominant = [f for f in dominant_per_time if f is not None]
    headline_feature = max(set(unique_dominant), key=unique_dominant.count) if unique_dominant else "N/A"

    status = "robust"
    if n_events < 20:
        status = "review"
    if n_events < 10:
        status = "caution"

    scientific_summary = {
        "status": status,
        "headline": (
            f"Proxy time-slice importance summary computed with censoring-aware slices; most frequently dominant feature: {headline_feature}."
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": n_patients},
            {"label": "Events", "value": n_events},
            {"label": "Time points", "value": len(eval_times_arr)},
            {"label": "Evaluable patients, min", "value": int(min(evaluable_patients_per_time) if evaluable_patients_per_time else 0)},
            {"label": "Features (encoded)", "value": len(feature_names)},
        ],
    }

    importance_matrix_feature_major: list[list[float | None]] = []
    for feat_idx in range(len(feature_names)):
        column: list[float | None] = []
        for row in importance_matrix_time_major:
            column.append(row[feat_idx] if feat_idx < len(row) else None)
        importance_matrix_feature_major.append(column)

    return {
        "eval_times": [_safe_float(float(t)) for t in eval_times_arr],
        "features": feature_names,
        "importance_matrix": importance_matrix_time_major,
        "importance_matrix_time_major": importance_matrix_time_major,
        "importance_matrix_feature_major": importance_matrix_feature_major,
        "importance_matrix_orientation": "time_major",
        "dominant_feature_per_time": dominant_per_time,
        "evaluable_patients_per_time": evaluable_patients_per_time,
        "skipped_time_points": skipped_time_points,
        "scientific_summary": scientific_summary,
    }


# ===================================================================
# 10. Counterfactual survival curves (XAI)
# ===================================================================


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
    X_original = result["_X_encoded"]
    encoder = result.get("_feature_encoder")
    analysis_frame = result.get("_analysis_frame")
    if analysis_frame is None or not set(features).issubset(set(analysis_frame.columns)):
        analysis_frame = frame

    def _scenario_label(value: Any) -> str:
        return "observed values" if value is None else str(value)

    def _build_scenario_matrix(value: Any) -> pd.DataFrame:
        if value is None:
            return X_original

        frame_variant = analysis_frame.copy()
        if target_feature in cat_feats:
            frame_variant[target_feature] = str(value)
        else:
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                raise ValueError(
                    f"Counterfactual value '{value}' for numeric feature '{target_feature}' "
                    "could not be converted to a number."
                )
            frame_variant[target_feature] = float(numeric_value)

        if encoder is not None:
            encoded = _transform_feature_encoder(frame_variant, encoder)
            return encoded.reindex(columns=X_original.columns, fill_value=0.0).fillna(0.0)

        encoded = _encode_features(frame_variant, features, cat_feats)
        return encoded.reindex(columns=X_original.columns, fill_value=0.0).fillna(0.0)

    original_label = _scenario_label(original_value)
    counterfactual_label = _scenario_label(counterfactual_value)

    # Original risk scores
    X_baseline = _build_scenario_matrix(original_value)
    original_risk = model.predict(X_baseline.to_numpy())
    original_median_risk = float(np.median(original_risk))

    # Build counterfactual feature matrix
    X_cf = _build_scenario_matrix(counterfactual_value)
    cf_risk = model.predict(X_cf.to_numpy())
    cf_median_risk = float(np.median(cf_risk))

    # Percentage change
    if original_median_risk != 0.0:
        risk_change_pct = ((cf_median_risk - original_median_risk) / abs(original_median_risk)) * 100.0
    else:
        risk_change_pct = 0.0 if cf_median_risk == 0.0 else None

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
        direction = "has minimal effect on"
        direction_label = "similar risk"

    if risk_change_pct is None:
        risk_change_text = "is not well-defined because the baseline median risk is zero"
        headline_effect = (
            f"Under a model-based scenario that sets '{target_feature}' from {original_label} to {counterfactual_label}, "
            "median predicted risk changed, but the relative percentage change is undefined because the baseline median risk is zero."
        )
    else:
        risk_change_text = f"{direction} by {abs(risk_change_pct):.1f}% ({direction_label})"
        headline_effect = (
            f"Under a model-based scenario that sets '{target_feature}' from {original_label} to {counterfactual_label}, "
            f"median predicted risk {direction} by {abs(risk_change_pct):.1f}%."
        )

    strengths: list[str] = [
        f"Counterfactual scenario analysis set '{target_feature}' from "
        f"{original_label} to {counterfactual_label} across {n_patients} patients.",
        f"Median predicted risk {risk_change_text}.",
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
        "model_type": model_type,
        "scientific_summary": scientific_summary,
    }

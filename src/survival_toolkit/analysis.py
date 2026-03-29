from __future__ import annotations

import io
import itertools
import math
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy import stats
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

TRUE_TOKENS = {
    "1",
    "true",
    "yes",
    "y",
    "event",
    "dead",
    "death",
    "deceased",
    "progressed",
    "progression",
    "relapse",
    "failure",
}
FALSE_TOKENS = {
    "0",
    "false",
    "no",
    "n",
    "censored",
    "alive",
    "diseasefree",
    "disease-free",
    "nonevent",
    "non-event",
}
KM_WEIGHT_MAP = {
    "logrank": None,
    "gehan_breslow": "gb",
    "tarone_ware": "tw",
    "fleming_harrington": "fh",
}
TERM_CATEGORICAL_PATTERN = re.compile(r'C\(Q\("(?P<var>.+)"\)\)\[T\.(?P<level>.+)\]')
TERM_NUMERIC_PATTERN = re.compile(r'Q\("(?P<var>.+)"\)')
_MAX_EXP_INPUT = math.log(np.finfo(float).max)
_MIN_EXP_INPUT = math.log(np.finfo(float).tiny)


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for raw_name in columns:
        name = str(raw_name).strip() or "unnamed"
        counter = seen.get(name, 0)
        if counter:
            unique_name = f"{name}_{counter + 1}"
        else:
            unique_name = name
        seen[name] = counter + 1
        output.append(unique_name)
    return output


def _read_csv_with_fallback(source: io.BytesIO | str | Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return pd.read_csv(source, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file. Supported encodings: UTF-8, Latin-1.")


def _load_dataframe_source(source: io.BytesIO | str | Path, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        df = _read_csv_with_fallback(source)
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(source)
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file: {exc}") from exc
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(source)
        except Exception as exc:
            raise ValueError(f"Failed to read Parquet file: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported input file extension '{suffix or '<none>'}' for '{filename}'. "
            "Supported formats are CSV, TSV, XLSX, XLS, and Parquet."
        )

    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")
    df.columns = make_unique_columns(df.columns)
    return df


def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return _load_dataframe_source(io.BytesIO(file_bytes), filename)


def load_dataframe_from_path(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"Input path is not a file: {path_obj}")
    return _load_dataframe_source(path_obj, path_obj.name)


def serialize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def preview_rows(df: pd.DataFrame, n_rows: int = 12) -> list[dict[str, Any]]:
    preview = df.head(n_rows).copy()
    return [
        {column: serialize_value(value) for column, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]


def _column_kind(series: pd.Series) -> str:
    unique_count = int(series.nunique(dropna=True))
    if unique_count <= 2 and not is_datetime64_any_dtype(series):
        return "binary"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _column_keywords(columns: Sequence[str], tokens: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        lowered = column.lower()
        if any(token in lowered for token in tokens):
            matches.append(column)
    return matches


def _normalize_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return str(int(value))
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value) in (0.0, 1.0):
            return str(int(value))
        return None
    return str(value).strip().lower()


def coerce_event(series: pd.Series, event_positive_value: Any = None) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()
    if not valid.any():
        raise ValueError("The event column contains only missing values.")

    if event_positive_value is not None and event_positive_value != "":
        target = event_positive_value
        numeric_series = pd.to_numeric(series, errors="coerce")
        try:
            target_numeric = float(target)
        except (TypeError, ValueError):
            target_numeric = None

        # If the column is fully numeric-coercible, only allow truly binary
        # coding. Multi-state numeric status columns must be rejected rather
        # than silently collapsing extra states into censoring.
        if target_numeric is not None and numeric_series.notna().sum() == valid.sum():
            observed_numeric = sorted({float(value) for value in numeric_series.loc[valid].astype(float).tolist()})
            if target_numeric not in observed_numeric:
                raise ValueError(
                    f"The selected event-positive value '{target}' is not present in the numeric event column."
                )
            if len(observed_numeric) > 2:
                raise ValueError(
                    "The numeric event column has more than two distinct states. "
                    "Provide a pre-binarized event indicator before survival analysis."
                )
            out.loc[valid] = (numeric_series.loc[valid] == target_numeric).astype(float)
            return out

        def _explicit_token(value: Any) -> str | None:
            if pd.isna(value):
                return None
            if isinstance(value, (bool, np.bool_)):
                return "1" if bool(value) else "0"
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    return None
                float_value = float(value)
                if float_value.is_integer():
                    return str(int(float_value))
                return str(float_value).strip().lower()
            return str(value).strip().lower()

        target_token = _explicit_token(target)
        if target_token is None:
            raise ValueError("The selected event-positive value could not be parsed.")

        value_tokens = series.map(_explicit_token)
        if value_tokens.loc[valid].isna().any():
            raise ValueError(
                "The event column contains non-missing values that cannot be normalized for event coding."
            )

        observed_tokens = sorted(set(value_tokens.loc[valid].astype(str).tolist()))

        # If the target is a known event/censor token, decode using the full token vocabulary.
        if target_token in TRUE_TOKENS:
            allowed = TRUE_TOKENS | FALSE_TOKENS
            unknown = [tok for tok in observed_tokens if tok not in allowed]
            if unknown:
                raise ValueError(
                    "Event coding contains unrecognized tokens alongside standard event/censor labels: "
                    + ", ".join(unknown[:6])
                    + (" ..." if len(unknown) > 6 else "")
                )
            event_tokens = TRUE_TOKENS
        elif target_token in FALSE_TOKENS:
            allowed = TRUE_TOKENS | FALSE_TOKENS
            unknown = [tok for tok in observed_tokens if tok not in allowed]
            if unknown:
                raise ValueError(
                    "Event coding contains unrecognized tokens alongside standard event/censor labels: "
                    + ", ".join(unknown[:6])
                    + (" ..." if len(unknown) > 6 else "")
                )
            event_tokens = FALSE_TOKENS
        else:
            # Custom label: allow mapping only for true binary columns to avoid masking typos.
            if target_token not in observed_tokens:
                raise ValueError(
                    f"The selected event-positive value '{target}' is not present in the event column."
                )
            if len(observed_tokens) > 2:
                raise ValueError(
                    "The event column has more than two distinct values after normalization. "
                    "For multi-class status columns, please recode to a binary event indicator."
                )
            event_tokens = {target_token}

        out.loc[valid] = value_tokens.loc[valid].isin(event_tokens).astype(float)
        return out

    if is_bool_dtype(series):
        out.loc[valid] = series.loc[valid].astype(int).astype(float)
        return out

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() == valid.sum():
        unique_floats = set(numeric_series.loc[valid].unique().tolist())
        if unique_floats.issubset({0.0, 1.0}):
            out.loc[valid] = numeric_series.loc[valid].astype(float)
            return out

    normalized = series.map(_normalize_token)
    mapped = pd.Series(np.nan, index=series.index, dtype=float)
    mapped.loc[normalized.isin(TRUE_TOKENS)] = 1.0
    mapped.loc[normalized.isin(FALSE_TOKENS)] = 0.0
    if mapped.loc[valid].notna().sum() == valid.sum():
        return mapped

    raise ValueError(
        "Could not infer event coding. Select the value that represents the event in the dashboard."
    )


def looks_binary(series: pd.Series) -> bool:
    try:
        coerced = coerce_event(series)
    except ValueError:
        valid = series.dropna()
        if valid.empty:
            return False

        numeric_series = pd.to_numeric(valid, errors="coerce")
        if numeric_series.notna().sum() == len(valid):
            observed_numeric = {float(value) for value in numeric_series.astype(float).tolist()}
            return len(observed_numeric) == 2

        normalized = valid.map(_normalize_token)
        normalized_non_missing = normalized.dropna()
        if len(normalized_non_missing) == len(valid):
            return len(set(normalized_non_missing.tolist())) == 2

        return int(valid.nunique(dropna=True)) == 2
    non_missing = coerced.dropna()
    return not non_missing.empty and set(non_missing.unique()).issubset({0.0, 1.0})


def suggest_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(df.columns)
    event_tokens = ("event", "status", "death", "progress", "relapse", "censor")
    time_raw = _column_keywords(columns, ("time", "month", "day", "week", "year", "os", "pfs", "dfs"))
    time_columns = [c for c in time_raw if not any(tok in c.lower() for tok in event_tokens)]
    suggestions = {
        "time_columns": time_columns,
        "event_columns": _column_keywords(columns, event_tokens),
        "group_columns": _column_keywords(columns, ("group", "arm", "treatment", "stage", "sex", "risk", "cluster")),
    }
    return suggestions


def profile_dataframe(df: pd.DataFrame, dataset_id: str, filename: str) -> dict[str, Any]:
    column_profiles: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    binary_candidate_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        kind = _column_kind(series)
        unique_preview = [serialize_value(value) for value in series.dropna().unique()[:8]]
        profile = {
            "name": column,
            "kind": kind,
            "missing": int(series.isna().sum()),
            "non_missing": int(series.notna().sum()),
            "n_unique": int(series.nunique(dropna=True)),
            "unique_preview": unique_preview,
        }
        if is_numeric_dtype(series):
            numeric_columns.append(column)
            profile["min"] = serialize_value(series.min())
            profile["max"] = serialize_value(series.max())
        else:
            categorical_columns.append(column)
        if looks_binary(series):
            binary_candidate_columns.append(column)
        if kind == "binary" and column not in categorical_columns and column not in numeric_columns:
            categorical_columns.append(column)
        column_profiles.append(profile)

    suggestions = suggest_columns(df)

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": column_profiles,
        "preview": preview_rows(df),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_candidate_columns": binary_candidate_columns,
        "suggestions": suggestions,
    }


def quote_name(name: str) -> str:
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{escaped}")'


def _next_available_column_name(existing_columns: Iterable[Any], base_name: str) -> str:
    used = {str(column) for column in existing_columns}
    if base_name not in used:
        return base_name
    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in used:
            return candidate
        suffix += 1


def _ensure_positive_times(time_values: pd.Series) -> pd.Series:
    positive = time_values > 0
    if positive.sum() == 0:
        raise ValueError("The survival time column must contain positive values.")
    return positive


def _cohort_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    event_positive_value: Any = None,
    extra_columns: Sequence[str] | None = None,
    drop_missing_extra_columns: bool = True,
) -> pd.DataFrame:
    extra_columns = list(extra_columns or [])
    required_columns = [time_column, event_column, *extra_columns]
    frame = df[required_columns].copy()
    frame[time_column] = pd.to_numeric(frame[time_column], errors="coerce")
    frame[event_column] = coerce_event(frame[event_column], event_positive_value=event_positive_value)
    for column in extra_columns:
        if not is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("string")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if drop_missing_extra_columns:
        frame = frame.dropna()
    else:
        frame = frame.dropna(subset=[time_column, event_column])
    frame = frame.loc[_ensure_positive_times(frame[time_column])].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No analyzable rows remain after removing missing values.")
    if frame[event_column].sum() == 0:
        raise ValueError("No events were found after preprocessing the event column.")
    return frame


def _sorted_group_labels(series: pd.Series) -> list[str]:
    labels = series.astype("string").dropna().unique().tolist()
    return sorted(labels, key=lambda value: (len(str(value)), str(value)))


def _ordered_level_strings(series: pd.Series) -> list[str]:
    non_missing = series.dropna()
    if non_missing.empty:
        return []
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    if numeric_values.notna().all():
        ordered_numeric = np.sort(numeric_values.unique().astype(float))
        return [str(int(value)) if float(value).is_integer() else str(value) for value in ordered_numeric]
    return list(dict.fromkeys(non_missing.astype("string").tolist()))


def _is_binary_numeric_series(series: pd.Series) -> bool:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_values.empty:
        return False
    return int(numeric_values.nunique()) == 2


def _pointwise_km_ci(survival: np.ndarray, se: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    z_value = norm.ppf(1 - alpha / 2)
    lower = survival.copy()
    upper = survival.copy()

    for idx, (s_value, se_value) in enumerate(zip(survival, se, strict=False)):
        if not np.isfinite(s_value):
            lower[idx] = np.nan
            upper[idx] = np.nan
            continue
        if s_value <= 0:
            lower[idx] = 0.0
            upper[idx] = 0.0
            continue
        log_s = math.log(s_value)
        denominator = s_value * log_s
        if (
            s_value >= 1 - 1e-12
            or not np.isfinite(se_value)
            or se_value <= 0
            or not np.isfinite(log_s)
            or abs(denominator) <= 1e-12
        ):
            lower[idx] = s_value
            upper[idx] = s_value
            continue
        transformed = np.log(-np.log(s_value))
        transformed_se = abs(se_value / denominator)
        if not np.isfinite(transformed_se):
            lower[idx] = s_value
            upper[idx] = s_value
            continue
        low = np.exp(-np.exp(transformed + z_value * transformed_se))
        high = np.exp(-np.exp(transformed - z_value * transformed_se))
        lower[idx] = np.clip(low, 0.0, 1.0)
        upper[idx] = np.clip(high, 0.0, 1.0)
    return lower, upper


def _step_values(event_times: np.ndarray, survival: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(event_times, query_times, side="right") - 1
    output = np.ones_like(query_times, dtype=float)
    valid = indices >= 0
    output[valid] = survival[indices[valid]]
    return output


def _restricted_mean_survival_time(timeline: np.ndarray, survival: np.ndarray, horizon: float) -> float:
    clipped_timeline = np.clip(timeline, 0, horizon)
    area = 0.0
    for idx in range(len(clipped_timeline) - 1):
        left = clipped_timeline[idx]
        right = clipped_timeline[idx + 1]
        if right <= left:
            continue
        area += survival[idx] * (right - left)
    if clipped_timeline[-1] < horizon:
        area += survival[-1] * (horizon - clipped_timeline[-1])
    return float(area)


def _safe_float(value: Any) -> float | None:
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(float_value):
        return None
    return float_value


def _safe_exp(value: Any) -> float:
    exponent = float(value)
    if exponent >= _MAX_EXP_INPUT:
        return float(np.finfo(float).max)
    if exponent <= _MIN_EXP_INPUT:
        return float(np.finfo(float).tiny)
    return math.exp(exponent)


def _summarize_labels(labels: Sequence[str], max_items: int = 3) -> str:
    cleaned = [str(label) for label in labels if label]
    if not cleaned:
        return "none"
    if len(cleaned) <= max_items:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:max_items])} +{len(cleaned) - max_items} more"


def _weighted_test_label(test_name: str | None, *, fh_p: float | None = None) -> str:
    label_map = {
        "logrank": "log-rank",
        "gehan_breslow": "Gehan-Breslow",
        "tarone_ware": "Tarone-Ware",
        "fleming_harrington": (
            f"Fleming-Harrington (fh_p={fh_p:g})"
            if fh_p is not None and np.isfinite(float(fh_p))
            else "Fleming-Harrington (fh_p-only)"
        ),
    }
    return label_map.get(str(test_name), str(test_name))


def _km_scientific_summary(
    summary_rows: Sequence[dict[str, Any]],
    cohort_summary: dict[str, Any],
    group_column: str | None,
    test_payload: dict[str, Any] | None,
    pairwise_rows: Sequence[dict[str, Any]],
    *,
    fh_p: float | None = None,
) -> dict[str, Any]:
    group_count = len(summary_rows)
    min_group_n = min(int(row["N"]) for row in summary_rows)
    min_group_events = min(int(row["Events"]) for row in summary_rows)
    total_events = int(cohort_summary["events"])

    strengths = [
        "Kaplan-Meier estimation uses Greenwood standard errors with log-log confidence intervals.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    if group_column:
        test_name = _weighted_test_label(test_payload["test"], fh_p=fh_p) if test_payload else "weighted"
        strengths.append(f"Global {test_name} comparison was run across {group_count} groups.")
        if pairwise_rows:
            strengths.append("Pairwise group comparisons include Benjamini-Hochberg adjusted p-values.")
    else:
        strengths.append("This run estimates a single cohort without between-group multiple testing.")

    if total_events < 20:
        cautions.append("Fewer than 20 total events limits precision of survival contrasts.")
    if min_group_n < 15:
        cautions.append("At least one group has fewer than 15 patients.")
    if min_group_events < 5:
        cautions.append("At least one group has fewer than 5 events, so median survival and p-values may be unstable.")
    if cohort_summary["median_follow_up"] is None:
        cautions.append("Median follow-up could not be estimated from the censoring distribution.")

    if group_column and test_payload:
        if float(test_payload["p_value"]) < 0.05:
            headline = f"Global {_weighted_test_label(test_payload['test'], fh_p=fh_p)} testing suggests survival differs across groups."
            next_steps.append("Inspect groupwise medians, RMST, and adjusted pairwise comparisons before making a manuscript claim.")
        else:
            headline = f"Global {_weighted_test_label(test_payload['test'], fh_p=fh_p)} testing does not show clear survival separation across groups."
            next_steps.append("Do not treat visual curve separation alone as evidence if the global test is not significant.")
    else:
        headline = "Single-cohort survival was estimated without a between-group hypothesis test."
        next_steps.append("Use a grouping variable to compare subcohorts formally.")

    if min_group_events < 5 or min_group_n < 15:
        next_steps.append("Avoid anchoring on median survival in sparse groups; emphasize confidence intervals and follow-up maturity.")

    status = "robust"
    if cautions:
        status = "review"
    if total_events < 10 or min_group_events < 3:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": int(cohort_summary["n"])},
            {"label": "Events", "value": total_events},
            {"label": "Groups", "value": group_count},
            {"label": "Median follow-up", "value": _safe_float(cohort_summary["median_follow_up"])},
        ],
    }


def _cox_scientific_summary(
    model_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
    model_stats: dict[str, Any],
) -> dict[str, Any]:
    significant_terms = [
        row["Label"]
        for row in model_rows
        if row["P value"] is not None
        and row["CI lower"] is not None
        and row["CI upper"] is not None
        and float(row["P value"]) < 0.05
        and not (float(row["CI lower"]) <= 1.0 <= float(row["CI upper"]))
    ]
    ph_alert_terms = [
        str(row["Term"])
        for row in diagnostic_rows
        if row["P value"] is not None and float(row["P value"]) < 0.05
    ]

    strengths = [
        "Cox regression was fit with the Efron tie method.",
        f"Model estimates use the analyzable cohort after dropping rows with missing selected covariates (N = {int(model_stats['n'])}).",
        "Proportional-hazards diagnostics were evaluated from rank-based Spearman correlations between Schoenfeld residuals and log time.",
        "The reported discrimination metric is an apparent C-index computed on the fitted cohort.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    epv = _safe_float(model_stats.get("events_per_parameter"))
    c_index = _safe_float(model_stats.get("c_index"))
    if epv is not None and epv < 10:
        cautions.append("Events per parameter is below 10, so coefficients may be unstable or overfit.")
        next_steps.append("Reduce model complexity or increase the event count before treating estimates as final.")
    cautions.append("Changing the covariate set can change the analyzable cohort because Cox fitting uses complete-case rows for the selected covariates.")
    if ph_alert_terms:
        cautions.append(
            f"Possible proportional-hazards violations detected for: {_summarize_labels(ph_alert_terms)}."
        )
        next_steps.append("Consider stratification or time-varying effects for PH-violating terms.")
    if c_index is not None and c_index < 0.6:
        cautions.append("Apparent model discrimination is modest (C-index below 0.60).")
    cautions.append("The Cox C-index is apparent, so it is optimistic and should not be treated as external validation.")
    cautions.append(
        "The current dashboard does not yet provide a built-in external-cohort apply workflow for Cox validation; validate the final specification on a separate cohort outside this run."
    )
    if not significant_terms:
        cautions.append("No model term shows clear nominal evidence at p < 0.05.")

    if significant_terms:
        headline = (
            f"Model fit identified {len(significant_terms)} term(s) with nominal hazard association: "
            f"{_summarize_labels(significant_terms)}."
        )
        next_steps.append("Interpret hazard ratios together with confidence intervals, not p-values alone.")
    else:
        headline = "Model fit completed, but no term shows clear nominal hazard association under the current specification."
        next_steps.append("Revisit covariate selection, encoding, and cohort size before forcing interpretation.")

    status = "robust"
    if cautions:
        status = "review"
    if (epv is not None and epv < 5) or len(ph_alert_terms) >= 2:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Events", "value": int(model_stats["events"])},
            {"label": "Parameters", "value": int(model_stats["parameters"])},
            {"label": "EPV", "value": epv},
            {"label": "Apparent C-index", "value": c_index},
        ],
    }


def _signature_scientific_summary(
    best_split: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    strengths = [
        "Signature ranking uses Benjamini-Hochberg adjusted p-values across tested combinations (within this search run).",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    if search_space["permutation_iterations"] > 0:
        strengths.append("Permutation-based empirical filtering was enabled for top-ranked candidates.")
    if search_space["validation_iterations"] > 0:
        strengths.append("Repeated subsample holdout checks were enabled for top-ranked candidates.")

    support = _safe_float(best_split.get("Bootstrap support (p<0.05)"))
    direction_consistency = _safe_float(best_split.get("Bootstrap HR direction consistency"))
    validation_support = _safe_float(best_split.get("Validation support (p<alpha)"))
    permutation_p = _safe_float(best_split.get("Permutation p"))
    signature_n = int(best_split["N signature+"])
    is_significant = bool(best_split["Statistically significant"])
    alpha = float(search_space["significance_level"])

    if search_space["truncated"]:
        cautions.append("Search space hit the internal combination cap, so discovery was not exhaustive.")
        cautions.append("Adjusted p-values only account for the tested subset of combinations under the cap.")
    analyzable_n = search_space.get("n_rows_analyzed")
    if analyzable_n is not None:
        cautions.append(
            f"Signature discovery used {int(analyzable_n)} analyzable rows after dropping missing candidate values; changing the candidate set can change the search cohort."
        )
    else:
        cautions.append(
            "Signature discovery uses the analyzable subset after dropping missing candidate values; changing the candidate set can change the search cohort."
        )
    if search_space["permutation_iterations"] == 0 and search_space["validation_iterations"] == 0:
        cautions.append(
            "Because the signature is selected from many tested rules in the same cohort, "
            "reported p-values (including BH-adjusted) can be optimistic without permutation or holdout checks."
        )
    if not is_significant:
        cautions.append("Top-ranked signature remains exploratory under the current significance rules.")
        next_steps.append("Do not lock this signature for biological interpretation without further validation.")
    if signature_n < max(15, int(search_space["min_group_size"])):
        cautions.append("The signature-positive subgroup is small, which can inflate apparent separation.")
    if support is not None and support < 0.7:
        cautions.append("Bootstrap support is below 0.70, so the signal may be unstable.")
    if direction_consistency is not None and direction_consistency < 0.75:
        cautions.append("Bootstrap hazard-ratio direction is not consistently preserved.")
    if validation_support is not None and validation_support < 0.5:
        cautions.append("Internal validation support is below 0.50.")
    if permutation_p is not None and permutation_p > alpha:
        cautions.append("Permutation testing does not confirm the observed ranking at the configured alpha.")

    has_internal_confirmation = (
        search_space["permutation_iterations"] > 0 or search_space["validation_iterations"] > 0
    )
    if is_significant and has_internal_confirmation:
        headline = (
            "Top-ranked signature remains an internally supported screening result "
            "and still needs external validation before it is framed as a biomarker claim."
        )
        next_steps.append("Validate the locked signature on an external cohort before presenting it as a biomarker claim.")
    elif is_significant:
        headline = (
            "Top-ranked signature clears the current ranking rules but remains exploratory "
            "without permutation or holdout confirmation."
        )
        next_steps.append("Run permutation or holdout validation before framing this signature as a robust biomarker claim.")
    else:
        headline = "Top-ranked signature is still exploratory and should be treated as hypothesis-generating."
        next_steps.append("Narrow the candidate list or increase sample size before claiming a stable signature.")

    status = "robust"
    if cautions:
        status = "review"
    if not is_significant or signature_n < 10:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Tested combos", "value": int(search_space["tested_combinations"])},
            {"label": "Significant combos", "value": int(search_space["significant_signatures"])},
            {"label": "Signature+ N", "value": signature_n},
            {"label": "Permutation p", "value": permutation_p},
        ],
    }


def _median_follow_up(time_values: pd.Series, event_values: pd.Series) -> float | None:
    censor_status = 1 - event_values.astype(int)
    if censor_status.sum() == 0:
        return None
    sf = SurvfuncRight(time_values.to_numpy(), censor_status.to_numpy())
    return _safe_float(sf.quantile(0.5))


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    p_array = np.asarray(p_values, dtype=float)
    if p_array.size == 0:
        return []
    order = np.argsort(p_array)
    adjusted = np.empty_like(p_array)
    n_tests = len(p_array)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n_tests - rank + 1
        candidate = min(running, p_array[idx] * n_tests / original_rank)
        running = candidate
        adjusted[idx] = candidate
    return [float(min(1.0, value)) for value in adjusted]


def _coerce_numeric_if_needed(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series


def derive_group_column(
    df: pd.DataFrame,
    source_column: str,
    method: str,
    new_column_name: str | None = None,
    cutoff: float | None = None,
    lower_label: str = "Low",
    upper_label: str = "High",
    time_column: str | None = None,
    event_column: str | None = None,
    event_positive_value: Any = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    numeric_series = pd.to_numeric(df[source_column], errors="coerce")
    usable = numeric_series.dropna()
    if usable.empty:
        raise ValueError(f"{source_column} does not contain numeric values that can be split.")

    def quantile_split(n_bins: int, prefix: str, method_name: str) -> tuple[pd.Series, dict[str, Any]]:
        split_raw = pd.qcut(numeric_series, n_bins, duplicates="drop")
        if not hasattr(split_raw, "cat"):
            raise ValueError(f"{source_column} cannot be split with {method_name}.")
        categories = split_raw.cat.categories
        n_groups = int(len(categories))
        if n_groups < 2:
            raise ValueError(
                f"{source_column} does not have enough unique values for {method_name}."
            )
        labels = [f"{prefix}{idx}" for idx in range(1, n_groups + 1)]
        codes = split_raw.cat.codes.to_numpy()
        label_values = np.array(labels, dtype=object)
        mapped = np.where(codes >= 0, label_values[codes], pd.NA)
        split = pd.Series(mapped, index=numeric_series.index, dtype="string")
        cutoffs = [float(interval.right) for interval in categories[:-1]]
        return split, {
            "method": method_name,
            "cutoffs": cutoffs,
            "n_groups": n_groups,
        }

    outcome_informed = False

    if method == "optimal_cutpoint":
        if not time_column or not event_column:
            raise ValueError("optimal_cutpoint requires time_column and event_column.")
        from survival_toolkit.ml_models import find_optimal_cutpoint

        result = find_optimal_cutpoint(
            df,
            time_column=time_column,
            event_column=event_column,
            variable=source_column,
            event_positive_value=event_positive_value,
            min_group_fraction=0.1,
            lower_label=lower_label,
            upper_label=upper_label,
            permutation_iterations=100,
            random_seed=20260311,
        )
        split_point = result["optimal_cutpoint"]
        lbl_above = result["label_above_cutpoint"]
        lbl_below = result["label_below_cutpoint"]
        labels = np.where(numeric_series > split_point, lbl_above, lbl_below)
        summary = {
            "method": method,
            "cutoff": split_point,
            "label_above_cutpoint": lbl_above,
            "label_below_cutpoint": lbl_below,
            "assignment_rule": f"{source_column} > cutoff -> {lbl_above}, else -> {lbl_below}",
            "statistic": result["statistic"],
            "p_value": result["p_value"],
            "p_value_label": result.get("p_value_label"),
            "raw_p_value": result.get("raw_p_value"),
            "selection_adjusted_p_value": result.get("selection_adjusted_p_value"),
            "selection_adjustment": result.get("selection_adjustment"),
            "scan_data": result["scan_data"],
        }
        outcome_informed = True
    elif method == "median_split":
        split_point = float(usable.median())
        labels = np.where(numeric_series <= split_point, lower_label, upper_label)
        summary = {"method": method, "cutoff": split_point}
    elif method == "tertile_split":
        labels, summary = quantile_split(n_bins=3, prefix="T", method_name=method)
    elif method == "quartile_split":
        labels, summary = quantile_split(n_bins=4, prefix="Q", method_name=method)
    elif method == "custom_cutoff":
        if cutoff is None:
            raise ValueError("A numeric cutoff is required for a custom split.")
        split_point = float(cutoff)
        labels = np.where(numeric_series <= split_point, lower_label, upper_label)
        summary = {"method": method, "cutoff": split_point}
    else:
        raise ValueError(f"Unsupported derive-group method: {method}")

    label_series = pd.Series(labels, index=df.index, dtype="string")
    label_series.loc[numeric_series.isna()] = pd.NA

    requested_name = (new_column_name or "").strip() or None
    if requested_name is not None:
        column_name = requested_name
    else:
        column_name = _next_available_column_name(df.columns, f"{source_column}__{method}")
    if requested_name is not None and column_name in df.columns:
        raise ValueError(
            f'"{column_name}" already exists. Choose a new derived-column name instead of overwriting an existing field.'
        )
    updated = df.copy()
    updated[column_name] = label_series

    counts = (
        updated[column_name]
        .fillna("Missing")
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("group")
        .reset_index(name="n")
        .to_dict(orient="records")
    )
    summary["column_name"] = column_name
    summary["outcome_informed"] = outcome_informed
    summary["counts"] = counts
    return updated, column_name, summary


def _build_candidate_indicators(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
    min_group_size: int,
) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []
    for column in candidate_columns:
        series = frame[column]
        if is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            quantile_candidates = (0.25, 0.5, 0.75)
            seen_thresholds: set[float] = set()
            for quantile in quantile_candidates:
                cutoff = float(numeric.quantile(quantile))
                if not np.isfinite(cutoff):
                    continue
                threshold_key = round(cutoff, 10)
                if threshold_key in seen_thresholds:
                    continue
                seen_thresholds.add(threshold_key)
                mask = numeric > cutoff
                n_positive = int(mask.sum())
                if min_group_size <= n_positive <= (len(mask) - min_group_size):
                    indicators.append(
                        {
                            "column": column,
                            "kind": "numeric_gt",
                            "threshold": cutoff,
                            "quantile": float(quantile),
                            "label": f"{column} > Q{int(quantile * 100)}({cutoff:.3f})",
                        }
                    )
            continue

        values = series.astype("string")
        counts = values.value_counts(dropna=True)
        if counts.empty:
            continue
        if int(counts.shape[0]) > 8:
            continue
        reference = str(counts.idxmax())
        for level in counts.index.tolist():
            level_text = str(level)
            if level_text == reference:
                continue
            mask = values == level_text
            n_positive = int(mask.sum())
            if min_group_size <= n_positive <= (len(mask) - min_group_size):
                indicators.append(
                    {
                        "column": column,
                        "kind": "categorical_level",
                        "level": level_text,
                        "reference": reference,
                        "label": f'{column} == "{level_text}"',
                    }
                )
    return indicators


def _evaluate_indicator(df: pd.DataFrame, indicator: dict[str, Any]) -> pd.Series:
    column = indicator["column"]
    if indicator["kind"] == "numeric_gt":
        numeric = pd.to_numeric(df[column], errors="coerce")
        out = pd.Series(pd.NA, index=df.index, dtype="boolean")
        valid = numeric.notna()
        out.loc[valid] = (numeric.loc[valid] > float(indicator["threshold"])).astype(bool)
        return out
    if indicator["kind"] == "categorical_level":
        values = df[column].astype("string")
        out = pd.Series(pd.NA, index=df.index, dtype="boolean")
        valid = values.notna()
        out.loc[valid] = (values.loc[valid] == str(indicator["level"])).astype(bool)
        return out
    raise ValueError(f"Unsupported indicator kind: {indicator['kind']}")


def _signature_mask(
    frame: pd.DataFrame, combo: Sequence[dict[str, Any]], operator: str = "AND"
) -> np.ndarray:
    if not combo:
        return np.zeros(frame.shape[0], dtype=bool)
    if operator not in {"AND", "OR"}:
        raise ValueError(f"Unsupported signature operator: {operator}")

    values = [
        _evaluate_indicator(frame, indicator).fillna(False).to_numpy(dtype=bool)
        for indicator in combo
    ]
    if operator == "AND":
        mask = np.logical_and.reduce(values)
    else:
        mask = np.logical_or.reduce(values)
    return np.asarray(mask, dtype=bool)


def _stability_score(row: dict[str, Any]) -> float:
    # Composite score balancing significance, robustness, effect size, and parsimony.
    bh_p = max(float(row["BH adjusted p"]), 1e-12)
    effect = abs(math.log(max(float(row["Hazard ratio (signature+ vs -)"]), 1e-12)))
    support = row["Bootstrap support (p<0.05)"]
    support_value = float(support) if support is not None else 0.0
    direction_consistency = row.get("Bootstrap HR direction consistency")
    direction_consistency_value = (
        float(direction_consistency) if direction_consistency is not None else 0.0
    )
    validation_support = row.get("Validation support (p<alpha)")
    validation_support_value = (
        float(validation_support) if validation_support is not None else 0.0
    )
    permutation_p = row["Permutation p"]
    permutation_penalty = 0.0
    if permutation_p is not None:
        permutation_penalty = max(float(permutation_p) - 0.05, 0.0)
    complexity_penalty = 0.18 * max(int(row["Rule count"]) - 1, 0)
    return float(
        (-math.log10(bh_p))
        + (0.35 * effect)
        + (0.85 * support_value)
        + (0.55 * direction_consistency_value)
        + (1.05 * validation_support_value)
        - complexity_penalty
        - (0.65 * permutation_penalty)
    )


def _signature_is_significant(
    row: dict[str, Any],
    significance_level: float,
    require_permutation: bool,
    require_validation: bool,
    min_validation_support: float,
    require_bootstrap_consistency: bool,
    min_bootstrap_consistency: float,
) -> bool:
    bh_p = float(row["BH adjusted p"])
    if bh_p > significance_level:
        return False

    permutation_p = row.get("Permutation p")
    if require_permutation and permutation_p is None:
        return False
    if permutation_p is not None and float(permutation_p) > significance_level:
        return False

    ci_low = _safe_float(row.get("HR CI lower"))
    ci_high = _safe_float(row.get("HR CI upper"))
    if ci_low is None or ci_high is None:
        return False
    if ci_low <= 1.0 <= ci_high:
        return False
    if require_bootstrap_consistency:
        direction_consistency = _safe_float(row.get("Bootstrap HR direction consistency"))
        if direction_consistency is None:
            return False
        if direction_consistency < min_bootstrap_consistency:
            return False
    if require_validation:
        validation_support = _safe_float(row.get("Validation support (p<alpha)"))
        if validation_support is None:
            return False
        if validation_support < min_validation_support:
            return False
    return True


def _bootstrap_signature_metrics(
    frame: pd.DataFrame,
    time_column: str,
    event_column: str,
    combo: Sequence[dict[str, Any]],
    combo_operator: str,
    min_group_size: int,
    n_iterations: int,
    sample_fraction: float,
    random_seed: int,
) -> dict[str, float | int | None]:
    if n_iterations <= 0:
        return {
            "Bootstrap support (p<0.05)": None,
            "Bootstrap median HR": None,
            "Bootstrap median p": None,
            "Bootstrap HR direction consistency": None,
            "Bootstrap valid resamples": 0,
        }

    n_obs = int(frame.shape[0])
    sample_size = int(math.ceil(n_obs * sample_fraction))
    sample_size = min(max(sample_size, min_group_size * 2), n_obs)
    rng = np.random.default_rng(random_seed)
    significant_count = 0
    valid_resamples = 0
    p_values: list[float] = []
    hazard_ratios: list[float] = []

    for _ in range(n_iterations):
        sampled_idx = rng.integers(0, n_obs, size=sample_size)
        sampled = frame.iloc[sampled_idx].reset_index(drop=True)

        mask_values = _signature_mask(sampled, combo, operator=combo_operator)
        n_high = int(mask_values.sum())
        n_low = sample_size - n_high
        if n_high < min_group_size or n_low < min_group_size:
            continue

        events = sampled[event_column].to_numpy(dtype=int)
        times = sampled[time_column].to_numpy(dtype=float)
        if events[mask_values].sum() == 0 or events[~mask_values].sum() == 0:
            continue

        try:
            _, p_value = survdiff(times, events, np.where(mask_values, "Signature+", "Signature-"))
            cox_frame = pd.DataFrame(
                {
                    "__time": times,
                    "__event": events,
                    "__signature": mask_values.astype(float),
                }
            )
            cox_model = PHReg.from_formula(
                'Q("__time") ~ Q("__signature")',
                data=cox_frame,
                status=cox_frame["__event"],
                ties="efron",
            )
            cox_results = cox_model.fit(disp=False)
        except Exception:
            continue

        valid_resamples += 1
        p_float = float(p_value)
        p_values.append(p_float)
        hazard_ratios.append(_safe_exp(cox_results.params[0]))
        if p_float < 0.05:
            significant_count += 1

    if valid_resamples == 0:
        return {
            "Bootstrap support (p<0.05)": None,
            "Bootstrap median HR": None,
            "Bootstrap median p": None,
            "Bootstrap HR direction consistency": None,
            "Bootstrap valid resamples": 0,
        }

    hr_array = np.asarray(hazard_ratios, dtype=float)
    direction_consistency = float(
        max(float(np.mean(hr_array >= 1.0)), float(np.mean(hr_array < 1.0)))
    )

    return {
        "Bootstrap support (p<0.05)": float(significant_count / valid_resamples),
        "Bootstrap median HR": float(np.median(hazard_ratios)),
        "Bootstrap median p": float(np.median(p_values)),
        "Bootstrap HR direction consistency": direction_consistency,
        "Bootstrap valid resamples": int(valid_resamples),
    }


def _permutation_p_value(
    times: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
    observed_p: float,
    n_iterations: int,
    random_seed: int,
) -> tuple[float | None, int]:
    if n_iterations <= 0:
        return None, 0

    rng = np.random.default_rng(random_seed)
    valid = 0
    extreme = 0

    for _ in range(n_iterations):
        perm_mask = rng.permutation(mask)
        if events[perm_mask].sum() == 0 or events[~perm_mask].sum() == 0:
            continue
        try:
            _, perm_p = survdiff(times, events, np.where(perm_mask, "Signature+", "Signature-"))
        except Exception:
            continue
        valid += 1
        if float(perm_p) <= observed_p:
            extreme += 1

    if valid == 0:
        return None, 0
    empirical = float((extreme + 1) / (valid + 1))
    return empirical, int(valid)


def _validation_signature_metrics(
    frame: pd.DataFrame,
    time_column: str,
    event_column: str,
    combo: Sequence[dict[str, Any]],
    combo_operator: str,
    min_group_size: int,
    min_events_per_group: int,
    n_iterations: int,
    validation_fraction: float,
    significance_level: float,
    random_seed: int,
) -> dict[str, float | int | None]:
    if n_iterations <= 0:
        return {
            "Validation support (p<alpha)": None,
            "Validation median HR": None,
            "Validation median p": None,
            "Validation valid folds": 0,
        }

    n_obs = int(frame.shape[0])
    min_holdout = min_group_size * 2
    max_holdout = n_obs - 1
    if max_holdout < min_holdout:
        return {
            "Validation support (p<alpha)": None,
            "Validation median HR": None,
            "Validation median p": None,
            "Validation valid folds": 0,
        }
    holdout_size = int(math.ceil(n_obs * validation_fraction))
    holdout_size = min(max(holdout_size, min_holdout), max_holdout)
    rng = np.random.default_rng(random_seed)
    significant_count = 0
    valid_folds = 0
    p_values: list[float] = []
    hazard_ratios: list[float] = []

    for _ in range(n_iterations):
        holdout_idx = rng.choice(n_obs, size=holdout_size, replace=False)
        sampled = frame.iloc[holdout_idx].reset_index(drop=True)
        mask_values = _signature_mask(sampled, combo, operator=combo_operator)
        n_high = int(mask_values.sum())
        n_low = holdout_size - n_high
        if n_high < min_group_size or n_low < min_group_size:
            continue

        events = sampled[event_column].to_numpy(dtype=int)
        times = sampled[time_column].to_numpy(dtype=float)
        if events[mask_values].sum() < min_events_per_group or events[~mask_values].sum() < min_events_per_group:
            continue

        try:
            _, p_value = survdiff(times, events, np.where(mask_values, "Signature+", "Signature-"))
            cox_frame = pd.DataFrame(
                {
                    "__time": times,
                    "__event": events,
                    "__signature": mask_values.astype(float),
                }
            )
            cox_model = PHReg.from_formula(
                'Q("__time") ~ Q("__signature")',
                data=cox_frame,
                status=cox_frame["__event"],
                ties="efron",
            )
            cox_results = cox_model.fit(disp=False)
            conf_int = cox_results.conf_int()
        except Exception:
            continue

        ci_low = _safe_exp(conf_int[0, 0])
        ci_high = _safe_exp(conf_int[0, 1])
        valid_folds += 1
        p_float = float(p_value)
        p_values.append(p_float)
        hazard_ratios.append(_safe_exp(cox_results.params[0]))
        if p_float <= significance_level and not (ci_low <= 1.0 <= ci_high):
            significant_count += 1

    if valid_folds == 0:
        return {
            "Validation support (p<alpha)": None,
            "Validation median HR": None,
            "Validation median p": None,
            "Validation valid folds": 0,
        }

    return {
        "Validation support (p<alpha)": float(significant_count / valid_folds),
        "Validation median HR": float(np.median(hazard_ratios)),
        "Validation median p": float(np.median(p_values)),
        "Validation valid folds": int(valid_folds),
    }


def discover_feature_signature(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    candidate_columns: Sequence[str],
    event_positive_value: Any = None,
    max_combination_size: int = 3,
    top_k: int = 15,
    min_group_fraction: float = 0.1,
    bootstrap_iterations: int = 30,
    bootstrap_sample_fraction: float = 0.8,
    permutation_iterations: int = 0,
    validation_iterations: int = 0,
    validation_fraction: float = 0.35,
    significance_level: float = 0.05,
    combination_operator: str = "mixed",
    random_seed: int = 20260311,
    new_column_name: str | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    max_tested_combinations = 5000
    unique_candidates = list(dict.fromkeys(candidate_columns))
    if not unique_candidates:
        raise ValueError("Select at least one candidate feature for signature discovery.")
    if max_combination_size < 1:
        raise ValueError("max_combination_size must be at least 1.")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be at least 0.")
    if bootstrap_sample_fraction < 0.4 or bootstrap_sample_fraction > 1.0:
        raise ValueError("bootstrap_sample_fraction must be between 0.4 and 1.0.")
    if permutation_iterations < 0:
        raise ValueError("permutation_iterations must be at least 0.")
    if validation_iterations < 0:
        raise ValueError("validation_iterations must be at least 0.")
    if validation_fraction < 0.2 or validation_fraction > 0.6:
        raise ValueError("validation_fraction must be between 0.2 and 0.6.")
    if significance_level <= 0.0 or significance_level > 0.2:
        raise ValueError("significance_level must be within (0, 0.2].")
    normalized_operator = str(combination_operator).strip().lower()
    if normalized_operator not in {"and", "or", "mixed"}:
        raise ValueError("combination_operator must be one of: and, or, mixed.")
    if random_seed < 0:
        raise ValueError("random_seed must be >= 0.")

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=unique_candidates,
    )

    n_obs = int(frame.shape[0])
    min_group_size = max(8, int(math.ceil(n_obs * min_group_fraction)))
    min_events_per_group = max(3, int(math.ceil(n_obs * 0.03)))
    indicators = _build_candidate_indicators(frame, unique_candidates, min_group_size=min_group_size)
    if not indicators:
        raise ValueError("No valid binary indicators could be generated from the selected features.")

    max_size = min(max_combination_size, len(indicators))
    rows: list[dict[str, Any]] = []
    valid_combinations: list[dict[str, Any]] = []
    truncated = False
    operators_for_size = {
        "and": ["AND"],
        "or": ["OR"],
        "mixed": ["AND", "OR"],
    }
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(indicators, size):
            if len(rows) >= max_tested_combinations:
                truncated = True
                break
            if len({part["column"] for part in combo}) != len(combo):
                continue
            operator_list = operators_for_size[normalized_operator]
            if size == 1 and normalized_operator == "mixed":
                operator_list = ["AND"]
            for combo_operator in operator_list:
                if len(rows) >= max_tested_combinations:
                    truncated = True
                    break
                mask = _signature_mask(frame, combo, operator=combo_operator)
                n_high = int(mask.sum())
                n_low = n_obs - n_high
                if n_high < min_group_size or n_low < min_group_size:
                    continue

                events = frame[event_column].to_numpy(dtype=int)
                times = frame[time_column].to_numpy(dtype=float)
                if events[mask].sum() < min_events_per_group or events[~mask].sum() < min_events_per_group:
                    continue
                try:
                    chisq, p_value = survdiff(times, events, np.where(mask, "Signature+", "Signature-"))
                except Exception:
                    continue

                cox_frame = pd.DataFrame(
                    {
                        "__time": times,
                        "__event": events,
                        "__signature": mask.astype(float),
                    }
                )
                cox_model = PHReg.from_formula(
                    'Q("__time") ~ Q("__signature")',
                    data=cox_frame,
                    status=cox_frame["__event"],
                    ties="efron",
                )
                try:
                    cox_results = cox_model.fit(disp=False)
                except Exception:
                    continue
                conf_int = cox_results.conf_int()
                hr = _safe_exp(cox_results.params[0])
                ci_low = _safe_exp(conf_int[0, 0])
                ci_high = _safe_exp(conf_int[0, 1])

                sf_high = SurvfuncRight(times[mask], events[mask])
                sf_low = SurvfuncRight(times[~mask], events[~mask])
                median_high = _safe_float(sf_high.quantile(0.5))
                median_low = _safe_float(sf_low.quantile(0.5))

                valid_combinations.append({"combo": combo, "operator": combo_operator})
                rows.append(
                    {
                        "Signature": f" {combo_operator} ".join(part["label"] for part in combo),
                        "Combination operator": combo_operator,
                        "Features": [part["column"] for part in combo],
                        "Rule count": int(len(combo)),
                        "N signature+": n_high,
                        "N signature-": n_low,
                        "Events signature+": int(events[mask].sum()),
                        "Events signature-": int(events[~mask].sum()),
                        "Chi-square": float(chisq),
                        "P value": float(p_value),
                        "Hazard ratio (signature+ vs -)": hr,
                        "HR CI lower": ci_low,
                        "HR CI upper": ci_high,
                        "Median signature+": median_high,
                        "Median signature-": median_low,
                        "Bootstrap support (p<0.05)": None,
                        "Bootstrap median HR": None,
                        "Bootstrap median p": None,
                        "Bootstrap HR direction consistency": None,
                        "Bootstrap valid resamples": 0,
                        "Permutation p": None,
                        "Permutation valid resamples": 0,
                        "Validation support (p<alpha)": None,
                        "Validation median HR": None,
                        "Validation median p": None,
                        "Validation valid folds": 0,
                        "Stability score": None,
                        "Statistically significant": False,
                    }
                )
            if truncated:
                break
        if truncated:
            break

    if not rows:
        raise ValueError("No analyzable feature combinations passed minimum group/event requirements.")

    adjusted = _bh_adjust([row["P value"] for row in rows])
    for row, adj in zip(rows, adjusted, strict=False):
        row["BH adjusted p"] = adj

    primary_ranked_idx = sorted(
        range(len(rows)),
        key=lambda idx: (
            rows[idx]["BH adjusted p"],
            rows[idx]["P value"],
            -abs(math.log(max(rows[idx]["Hazard ratio (signature+ vs -)"], 1e-8))),
        ),
    )
    # Compute expensive robustness metrics for (a) output candidates and (b)
    # any combination that could plausibly meet the configured significance rules.
    metric_candidate_idx: set[int] = set(primary_ranked_idx[: min(len(primary_ranked_idx), max(80, top_k * 4))])
    metric_candidate_idx.update(
        idx for idx, row in enumerate(rows)
        if float(row.get("BH adjusted p", 1.0)) <= float(significance_level)
        or float(row.get("P value", 1.0)) <= float(significance_level)
    )

    if bootstrap_iterations > 0:
        for idx in sorted(metric_candidate_idx):
            bootstrap_metrics = _bootstrap_signature_metrics(
                frame=frame,
                time_column=time_column,
                event_column=event_column,
                combo=valid_combinations[idx]["combo"],
                combo_operator=valid_combinations[idx]["operator"],
                min_group_size=min_group_size,
                n_iterations=bootstrap_iterations,
                sample_fraction=bootstrap_sample_fraction,
                random_seed=random_seed + 1000 + idx,
            )
            rows[idx].update(bootstrap_metrics)

    times = frame[time_column].to_numpy(dtype=float)
    events = frame[event_column].to_numpy(dtype=int)
    if permutation_iterations > 0:
        for idx in sorted(metric_candidate_idx):
            combo = valid_combinations[idx]
            mask = _signature_mask(frame, combo["combo"], operator=combo["operator"])
            empirical_p, valid_perm = _permutation_p_value(
                times=times,
                events=events,
                mask=mask,
                observed_p=float(rows[idx]["P value"]),
                n_iterations=permutation_iterations,
                random_seed=random_seed + 2000 + idx,
            )
            rows[idx]["Permutation p"] = empirical_p
            rows[idx]["Permutation valid resamples"] = valid_perm

    if validation_iterations > 0:
        for idx in sorted(metric_candidate_idx):
            validation_metrics = _validation_signature_metrics(
                frame=frame,
                time_column=time_column,
                event_column=event_column,
                combo=valid_combinations[idx]["combo"],
                combo_operator=valid_combinations[idx]["operator"],
                min_group_size=min_group_size,
                min_events_per_group=min_events_per_group,
                n_iterations=validation_iterations,
                validation_fraction=validation_fraction,
                significance_level=significance_level,
                random_seed=random_seed + 3000 + idx,
            )
            rows[idx].update(validation_metrics)

    for row in rows:
        row["Stability score"] = _stability_score(row)
        row["Statistically significant"] = _signature_is_significant(
            row,
            significance_level=significance_level,
            require_permutation=permutation_iterations > 0,
            require_validation=validation_iterations > 0,
            min_validation_support=0.5,
            require_bootstrap_consistency=bootstrap_iterations > 0,
            min_bootstrap_consistency=0.6,
        )

    ranked_idx = sorted(
        range(len(rows)),
        key=lambda idx: (
            not rows[idx]["Statistically significant"],
            -rows[idx]["Stability score"],
            rows[idx]["BH adjusted p"],
            rows[idx]["P value"],
        ),
    )
    ranked_rows = [rows[idx] for idx in ranked_idx][:top_k]
    best_idx = ranked_idx[0]
    best_combo = valid_combinations[best_idx]["combo"]
    best_operator = valid_combinations[best_idx]["operator"]

    output_df = df.copy()
    requested_name = (new_column_name or "").strip() or None
    if requested_name is not None:
        best_column_name = requested_name
    else:
        best_column_name = _next_available_column_name(output_df.columns, "auto_signature_group")
    if requested_name is not None and best_column_name in output_df.columns:
        raise ValueError(
            f'"{best_column_name}" already exists. Choose a new derived-column name instead of overwriting an existing field.'
        )
    indicator_values_list = [_evaluate_indicator(output_df, indicator) for indicator in best_combo]
    missing = pd.Series(False, index=output_df.index, dtype=bool)
    for indicator_values in indicator_values_list:
        missing = missing | indicator_values.isna().to_numpy(dtype=bool)
    bool_arrays = [item.fillna(False).to_numpy(dtype=bool) for item in indicator_values_list]
    if best_operator == "OR":
        combined = np.logical_or.reduce(bool_arrays)
    else:
        combined = np.logical_and.reduce(bool_arrays)
    labels = pd.Series(np.where(combined, "Signature+", "Signature-"), index=output_df.index, dtype="string")
    labels.loc[missing] = pd.NA
    output_df[best_column_name] = labels

    counts = (
        output_df[best_column_name]
        .fillna("Missing")
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("group")
        .reset_index(name="n")
        .to_dict(orient="records")
    )
    search_space = {
        "n_rows_analyzed": n_obs,
        "candidate_columns": unique_candidates,
        "generated_indicators": int(len(indicators)),
        "tested_combinations": int(len(rows)),
        "truncated": truncated,
        "min_group_size": int(min_group_size),
        "min_events_per_group": int(min_events_per_group),
        "max_combination_size": int(max_size),
        "combination_operator": normalized_operator,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_sample_fraction": float(bootstrap_sample_fraction),
        "bootstrap_scored_signatures": int(
            sum(1 for row in rows if row["Bootstrap valid resamples"] > 0)
        ),
        "permutation_iterations": int(permutation_iterations),
        "permutation_scored_signatures": int(
            sum(1 for row in rows if row["Permutation valid resamples"] > 0)
        ),
        "validation_iterations": int(validation_iterations),
        "validation_fraction": float(validation_fraction),
        "validation_scored_signatures": int(
            sum(1 for row in rows if row["Validation valid folds"] > 0)
        ),
        "significance_level": float(significance_level),
        "permutation_required_for_significance": bool(permutation_iterations > 0),
        "validation_required_for_significance": bool(validation_iterations > 0),
        "validation_min_support": 0.5,
        "bootstrap_min_direction_consistency": 0.6,
        "random_seed": int(random_seed),
        "significant_signatures": int(sum(1 for row in rows if row["Statistically significant"])),
    }
    scientific_summary = _signature_scientific_summary(
        best_split=rows[best_idx],
        search_space=search_space,
    )

    payload = {
        "results_table": ranked_rows,
        "best_split": rows[best_idx],
        "search_space": search_space,
        "derived_group": {
            "column_name": best_column_name,
            "counts": counts,
            "outcome_informed": True,
        },
        "scientific_summary": scientific_summary,
    }
    return output_df, best_column_name, payload


def compute_km_analysis(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    group_column: str | None = None,
    event_positive_value: Any = None,
    confidence_level: float = 0.95,
    max_time: float | None = None,
    risk_table_points: int = 6,
    logrank_weight: str = "logrank",
    fh_p: float = 1.0,
) -> dict[str, Any]:
    extra_columns = [group_column] if group_column else []
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=extra_columns,
    )
    if group_column:
        frame[group_column] = frame[group_column].astype("string")
        frame = frame.dropna(subset=[group_column]).reset_index(drop=True)
        group_labels = _sorted_group_labels(frame[group_column])
    else:
        group_labels = ["Overall"]

    if len(group_labels) == 0:
        raise ValueError("No groups remain after removing missing values.")

    time_values = frame[time_column].to_numpy(dtype=float)
    event_values = frame[event_column].to_numpy(dtype=int)
    alpha = 1.0 - confidence_level
    display_horizon = float(np.nanmax(time_values) if max_time is None else min(max_time, np.nanmax(time_values)))
    display_horizon = max(display_horizon, 1e-6)
    risk_ticks = np.round(np.linspace(0, display_horizon, risk_table_points), 2).tolist()

    summary_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    curve_payloads: list[dict[str, Any]] = []

    for label in group_labels:
        group_frame = frame if label == "Overall" and not group_column else frame.loc[frame[group_column] == label].copy()
        t_group = group_frame[time_column].to_numpy(dtype=float)
        e_group = group_frame[event_column].to_numpy(dtype=int)
        sf = SurvfuncRight(t_group, e_group)

        event_times = sf.surv_times.astype(float)
        step_timeline = np.concatenate(([0.0], event_times))
        step_survival = np.concatenate(([1.0], sf.surv_prob.astype(float)))
        lower, upper = _pointwise_km_ci(sf.surv_prob.astype(float), sf.surv_prob_se.astype(float), alpha=alpha)
        lower = np.concatenate(([1.0], lower))
        upper = np.concatenate(([1.0], upper))
        censor_times = group_frame.loc[group_frame[event_column] == 0, time_column].to_numpy(dtype=float)
        censor_survival = _step_values(event_times, sf.surv_prob.astype(float), censor_times) if censor_times.size else np.array([])

        median_survival = _safe_float(sf.quantile(0.5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_ci = sf.quantile_ci(0.5)
        median_ci_low = _safe_float(median_ci[0]) if isinstance(median_ci, tuple) else None
        median_ci_high = _safe_float(median_ci[1]) if isinstance(median_ci, tuple) else None
        rmst = _restricted_mean_survival_time(step_timeline, step_survival, display_horizon)

        summary_rows.append(
            {
                "Group": label,
                "N": int(group_frame.shape[0]),
                "Events": int(group_frame[event_column].sum()),
                "Censored": int((1 - group_frame[event_column]).sum()),
                "Median survival": median_survival,
                "Median CI lower": median_ci_low,
                "Median CI upper": median_ci_high,
                "RMST": rmst,
            }
        )
        risk_row = OrderedDict({"Group": label})
        for tick in risk_ticks:
            risk_row[f"{tick:g}"] = int((group_frame[time_column] >= tick).sum())
        risk_rows.append(dict(risk_row))
        curve_payloads.append(
            {
                "group": label,
                "timeline": step_timeline.tolist(),
                "survival": step_survival.tolist(),
                "ci_lower": lower.tolist(),
                "ci_upper": upper.tolist(),
                "censor_times": censor_times.tolist(),
                "censor_survival": censor_survival.tolist(),
            }
        )

    test_payload = None
    pairwise_rows: list[dict[str, Any]] = []
    weight_type = KM_WEIGHT_MAP.get(logrank_weight)
    if group_column and len(group_labels) >= 2:
        kwargs: dict[str, Any] = {}
        if weight_type == "fh":
            kwargs["fh_p"] = fh_p
        chisq, p_value = survdiff(time_values, event_values, frame[group_column].astype(str).to_numpy(), weight_type=weight_type, **kwargs)
        test_payload = {
            "test": logrank_weight,
            "chisq": float(chisq),
            "p_value": float(p_value),
        }
        raw_p_values: list[float] = []
        pair_ids: list[tuple[str, str]] = []
        for left, right in itertools.combinations(group_labels, 2):
            mask = frame[group_column].isin([left, right]).to_numpy()
            chisq_pair, p_pair = survdiff(
                time_values[mask],
                event_values[mask],
                frame.loc[mask, group_column].astype(str).to_numpy(),
                weight_type=weight_type,
                **kwargs,
            )
            raw_p_values.append(float(p_pair))
            pair_ids.append((left, right))
            pairwise_rows.append(
                {
                    "Comparison": f"{left} vs {right}",
                    "Chi-square": float(chisq_pair),
                    "P value": float(p_pair),
                }
            )
        adjusted = _bh_adjust(raw_p_values)
        for row, adjusted_p in zip(pairwise_rows, adjusted, strict=False):
            row["BH adjusted p"] = adjusted_p

    cohort_summary = {
        "n": int(frame.shape[0]),
        "events": int(frame[event_column].sum()),
        "censored": int((1 - frame[event_column]).sum()),
        "median_follow_up": _median_follow_up(frame[time_column], frame[event_column]),
        "time_max": float(np.nanmax(frame[time_column])),
    }

    scientific_summary = _km_scientific_summary(
        summary_rows=summary_rows,
        cohort_summary=cohort_summary,
        group_column=group_column,
        test_payload=test_payload,
        pairwise_rows=pairwise_rows,
        fh_p=fh_p,
    )

    test_label = (
        None
        if test_payload is None
        else _weighted_test_label(str(test_payload["test"]), fh_p=fh_p)
    )

    return {
        "curves": curve_payloads,
        "summary_table": summary_rows,
        "groups": int(len(summary_rows)),
        "logrank_p": (
            None
            if test_payload is None or logrank_weight != "logrank"
            else float(test_payload["p_value"])
        ),
        "risk_table": {
            "columns": ["Group", *[f"{tick:g}" for tick in risk_ticks]],
            "rows": risk_rows,
        },
        "pairwise_table": pairwise_rows,
        "test": test_payload,
        "test_p_value": None if test_payload is None else float(test_payload["p_value"]),
        "test_p_value_label": test_label,
        "cohort": cohort_summary,
        "display_horizon": display_horizon,
        "group_column": group_column,
        "scientific_summary": scientific_summary,
    }


def _categorical_candidates(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    candidates: list[str] = []
    for column in columns:
        series = df[column]
        if not is_numeric_dtype(series):
            candidates.append(column)
    return candidates


def _prepare_cox_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
    event_positive_value: Any = None,
) -> pd.DataFrame:
    required_columns = [*covariates]
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=required_columns,
    )
    for column in covariates:
        if column in categorical_covariates:
            string_values = frame[column].astype("string")
            categories = _ordered_reference_categories(string_values.dropna().unique().tolist(), column)
            frame[column] = pd.Categorical(string_values, categories=categories)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after removing missing values for the Cox model.")
    return frame


def _build_cox_formula(time_column: str, covariates: Sequence[str], categorical_covariates: Sequence[str]) -> str:
    if not covariates:
        raise ValueError("Select at least one covariate for the Cox PH model.")
    terms: list[str] = []
    for column in covariates:
        if column in categorical_covariates:
            terms.append(f"C({quote_name(column)})")
        else:
            terms.append(quote_name(column))
    return f"{quote_name(time_column)} ~ {' + '.join(terms)}"


_STAGE_ORDER = [
    "0",
    "i",
    "ia",
    "ib",
    "ic",
    "ii",
    "iia",
    "iib",
    "iic",
    "iii",
    "iiia",
    "iiib",
    "iiic",
    "iv",
    "iva",
    "ivb",
    "ivc",
]
_STAGE_ORDER_INDEX = {stage: idx for idx, stage in enumerate(_STAGE_ORDER)}


def _normalize_category_text(value: Any) -> tuple[str, str]:
    text = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    compact = normalized.replace(" ", "")
    return normalized, compact


def _stage_level_key(value: Any) -> tuple[int, int, str] | None:
    normalized, compact = _normalize_category_text(value)
    stage_token = compact
    if stage_token.startswith("stage"):
        stage_token = stage_token[5:]
    if not stage_token:
        return None
    if "unknown" in normalized:
        return (1, len(_STAGE_ORDER_INDEX) + 1, normalized)
    stage_index = _STAGE_ORDER_INDEX.get(stage_token)
    if stage_index is None:
        return None
    return (0, stage_index, normalized)


def _category_reference_sort_key(column: str, value: Any) -> tuple[Any, ...]:
    normalized, compact = _normalize_category_text(value)
    column_text, column_compact = _normalize_category_text(column)
    unknown_like = {"unknown", "missing", "not available", "na", "n a"}
    if normalized in unknown_like or "unknown" in normalized or "missing" in normalized:
        return (9, normalized)

    smoking_context = "smok" in column_compact or "smoker" in normalized or "nonsmoker" in compact
    if smoking_context:
        if "lifelong non smoker" in normalized or "never smoker" in normalized or "non smoker" in normalized or "nonsmoker" in compact:
            return (0, 0, normalized)
        if "former smoker" in normalized or normalized.startswith("former "):
            return (0, 1, normalized)
        if "current smoker" in normalized or normalized.startswith("current "):
            return (0, 2, normalized)

    stage_key = _stage_level_key(value)
    if stage_key is not None:
        return (1, *stage_key)

    if "wildtype" in compact or "wt" == compact:
        return (2, 0, normalized)
    if "mutated" in normalized or "mutant" in normalized:
        return (2, 1, normalized)

    if normalized in {"negative", "neg", "no", "absent", "control", "reference", "baseline", "normal", "none"}:
        return (3, 0, normalized)
    if normalized in {"positive", "pos", "yes", "present", "case", "abnormal"}:
        return (3, 1, normalized)

    if "sex" in column_text or "gender" in column_text:
        if normalized == "female":
            return (4, 0, normalized)
        if normalized == "male":
            return (4, 1, normalized)

    return (8, normalized)


def _ordered_reference_categories(values: Sequence[Any], column: str) -> list[str]:
    unique_values = [str(item) for item in values]
    return sorted(unique_values, key=lambda item: _category_reference_sort_key(column, item))


def _reference_levels(frame: pd.DataFrame, categorical_covariates: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for column in categorical_covariates:
        categories = [str(item) for item in frame[column].cat.categories]
        if categories:
            output[column] = categories[0]
    return output


def _clean_term(term: str, reference_levels: dict[str, str]) -> tuple[str, str, str | None]:
    categorical_match = TERM_CATEGORICAL_PATTERN.match(term)
    if categorical_match:
        variable = categorical_match.group("var")
        level = categorical_match.group("level")
        reference = reference_levels.get(variable)
        label = f"{variable}: {level} vs {reference}" if reference else f"{variable}: {level}"
        return variable, label, reference
    numeric_match = TERM_NUMERIC_PATTERN.match(term)
    if numeric_match:
        variable = numeric_match.group("var")
        return variable, variable, None
    return term, term, None


def _harrell_c_index(time_values: np.ndarray, event_values: np.ndarray, risk_score: np.ndarray) -> float | None:
    event_mask = event_values == 1
    event_idx = np.where(event_mask)[0]
    if len(event_idx) == 0:
        return None
    t_event = time_values[event_idx]
    r_event = risk_score[event_idx]
    t_all = time_values
    r_all = risk_score
    # Broadcast: compare each event subject against all subjects with longer time
    later_mask = t_all[np.newaxis, :] > t_event[:, np.newaxis]  # (n_event, n_all)
    if not later_mask.any():
        return None
    r_diff = r_event[:, np.newaxis] - r_all[np.newaxis, :]  # (n_event, n_all)
    concordant = float(np.sum((r_diff > 0) & later_mask))
    concordant += 0.5 * float(np.sum((r_diff == 0) & later_mask))
    comparable = float(np.sum(later_mask))
    if comparable == 0:
        return None
    return concordant / comparable


def compute_cox_analysis(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> dict[str, Any]:
    categorical_covariates = list(dict.fromkeys(categorical_covariates or _categorical_candidates(df, covariates)))
    frame = _prepare_cox_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        covariates=covariates,
        categorical_covariates=categorical_covariates,
        event_positive_value=event_positive_value,
    )
    formula = _build_cox_formula(time_column, covariates, categorical_covariates)
    status = frame[event_column].astype(int).to_numpy()
    model = PHReg.from_formula(formula, data=frame, status=status, ties="efron")
    results = model.fit(disp=False)

    reference_levels = _reference_levels(frame, categorical_covariates)
    risk_score = results.model.exog @ results.params
    conf_int = results.conf_int()
    model_rows: list[dict[str, Any]] = []
    for idx, term in enumerate(results.model.exog_names):
        variable, label, reference = _clean_term(term, reference_levels)
        beta = float(results.params[idx])
        hr = _safe_exp(beta)
        ci_low = _safe_exp(conf_int[idx, 0])
        ci_high = _safe_exp(conf_int[idx, 1])
        model_rows.append(
            {
                "Variable": variable,
                "Label": label,
                "Reference": reference,
                "Beta": beta,
                "Hazard ratio": hr,
                "CI lower": ci_low,
                "CI upper": ci_high,
                "SE": float(results.bse[idx]),
                "Z": float(results.tvalues[idx]),
                "P value": float(results.pvalues[idx]),
            }
        )

    schoenfeld = results.schoenfeld_residuals
    log_time = np.log(frame[time_column].to_numpy(dtype=float))
    diagnostic_rows: list[dict[str, Any]] = []
    for idx, term in enumerate(results.model.exog_names):
        valid = np.isfinite(schoenfeld[:, idx]) & np.isfinite(log_time)
        if valid.sum() < 4:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = stats.spearmanr(log_time[valid], schoenfeld[valid, idx])
        _, label, _ = _clean_term(term, reference_levels)
        diagnostic_rows.append(
            {
                "Term": label,
                "Schoenfeld rho": _safe_float(rho),
                "P value": _safe_float(p_value),
            }
        )

    n_obs = int(frame.shape[0])
    n_events = int(frame[event_column].sum())
    k_params = len(results.params)
    c_index = _harrell_c_index(
        frame[time_column].to_numpy(dtype=float),
        frame[event_column].to_numpy(dtype=int),
        risk_score.astype(float),
    )

    scientific_summary = _cox_scientific_summary(
        model_rows=model_rows,
        diagnostic_rows=diagnostic_rows,
        model_stats={
            "n": n_obs,
            "events": n_events,
            "parameters": k_params,
            "events_per_parameter": float(n_events / k_params) if k_params else None,
            "partial_log_likelihood": _safe_float(results.llf),
            "aic": _safe_float(-2 * results.llf + 2 * k_params),
            "bic": _safe_float(-2 * results.llf + k_params * np.log(max(n_obs, 1))),
            "c_index": _safe_float(c_index),
            "tie_method": "efron",
        },
    )

    return {
        "formula": formula,
        "results_table": model_rows,
        "diagnostics_table": diagnostic_rows,
        "model_stats": {
            "n": n_obs,
            "events": n_events,
            "parameters": k_params,
            "events_per_parameter": float(n_events / k_params) if k_params else None,
            "partial_log_likelihood": _safe_float(results.llf),
            "aic": _safe_float(-2 * results.llf + 2 * k_params),
            "bic": _safe_float(-2 * results.llf + k_params * np.log(max(n_obs, 1))),
            "c_index": _safe_float(c_index),
            "apparent_c_index": _safe_float(c_index),
            "c_index_label": "Apparent C-index",
            "evaluation_mode": "apparent",
            "tie_method": "efron",
        },
        "categorical_covariates": categorical_covariates,
        "scientific_summary": scientific_summary,
    }


def compute_cohort_table(df: pd.DataFrame, variables: Sequence[str], group_column: str | None = None) -> dict[str, Any]:
    variables = [variable for variable in variables if variable != group_column]
    if not variables:
        raise ValueError("Select at least one variable for the cohort summary table.")
    columns = [*variables]
    if group_column:
        columns.append(group_column)
    frame = df[columns].copy()

    group_frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
    group_frames["Overall"] = frame
    if group_column:
        string_group = frame[group_column].astype("string")
        group_labels = _sorted_group_labels(string_group)
        for label in group_labels:
            group_frames[label] = frame.loc[string_group == label]
    else:
        group_labels = []

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "Variable": "Cohort size",
            "Statistic": "N",
            **{label: int(group_frame.shape[0]) for label, group_frame in group_frames.items()},
        }
    )

    for variable in variables:
        series = frame[variable]
        is_binary_numeric = is_numeric_dtype(series) and _is_binary_numeric_series(series)
        if is_numeric_dtype(series) and not is_binary_numeric:
            row = {"Variable": variable, "Statistic": "Mean ± SD | Median [IQR]"}
            for label, group_frame in group_frames.items():
                values = pd.to_numeric(group_frame[variable], errors="coerce").dropna()
                if values.empty:
                    row[label] = "NA"
                    continue
                sd_value = values.std(ddof=1)
                if not np.isfinite(sd_value):
                    sd_value = 0.0
                row[label] = (
                    f"{values.mean():.2f} ± {sd_value:.2f} | "
                    f"{values.median():.2f} [{values.quantile(0.25):.2f}, {values.quantile(0.75):.2f}]"
                )
            rows.append(row)
            rows.append(
                {
                    "Variable": variable,
                    "Statistic": "Missing",
                    **{
                        label: int(group_frame[variable].isna().sum())
                        for label, group_frame in group_frames.items()
                    },
                }
            )
            continue

        source_series = pd.to_numeric(series, errors="coerce") if is_binary_numeric else series.astype("string")
        levels = _ordered_level_strings(source_series)
        for level in levels:
            row = {"Variable": variable, "Statistic": str(level)}
            for label, group_frame in group_frames.items():
                if is_binary_numeric:
                    group_values = pd.to_numeric(group_frame[variable], errors="coerce")
                    denominator = int(group_values.notna().sum())
                    numerator = int((group_values == float(level)).sum())
                else:
                    group_values = group_frame[variable].astype("string")
                    denominator = int(group_values.notna().sum())
                    numerator = int((group_values == level).sum())
                proportion = (100.0 * numerator / denominator) if denominator else 0.0
                row[label] = f"{numerator} ({proportion:.1f}%)"
            rows.append(row)
        rows.append(
            {
                "Variable": variable,
                "Statistic": "Missing",
                **{
                    label: int(group_frame[variable].isna().sum())
                    for label, group_frame in group_frames.items()
                },
            }
        )

    return {
        "columns": ["Variable", "Statistic", "Overall", *group_labels],
        "rows": rows,
    }

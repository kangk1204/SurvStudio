from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# A text column is treated as a contaminated number when at least this share of its
# non-missing values parse as numbers and the numeric part has more than
# ``_NUMERIC_TEXT_MIN_LEVELS`` distinct values (a small code set such as 1/2/3/"unknown"
# stays a legitimate categorical variable).
_NUMERIC_TEXT_MIN_SHARE = 0.8
_NUMERIC_TEXT_MIN_LEVELS = 10


def numeric_text_contamination(series: pd.Series) -> dict[str, Any] | None:
    """Describe stray text in a column that otherwise holds continuous numbers.

    Returns ``None`` for numeric, boolean, and genuinely categorical columns. A column such
    as ``[12.5, 13.1, ..., "unknown"]`` is read as text, so without this check it would be
    encoded as a categorical variable with one level per distinct number.
    """

    if is_numeric_dtype(series) or is_bool_dtype(series):
        return None
    non_missing = series.dropna()
    if non_missing.empty:
        return None
    text = non_missing.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    numeric_mask = numeric.notna()
    n_numeric = int(numeric_mask.sum())
    n_text = int((~numeric_mask).sum())
    if n_text == 0 or n_numeric < _NUMERIC_TEXT_MIN_SHARE * len(text):
        return None
    if int(numeric[numeric_mask].nunique()) <= _NUMERIC_TEXT_MIN_LEVELS:
        return None
    examples = list(dict.fromkeys(text[~numeric_mask].tolist()))[:3]
    return {"n_non_numeric": n_text, "examples": examples}


def reject_numeric_text_features(frame: pd.DataFrame, features: Sequence[str]) -> None:
    """Refuse model features that are continuous numbers with a few stray text values.

    ML, deep-learning, and Cox paths share this rule so the same column cannot be a
    hundreds-of-levels categorical in one module and an error in another.
    """

    for feature in features:
        if feature not in frame.columns:
            continue
        details = numeric_text_contamination(frame[feature])
        if details is None:
            continue
        examples = ", ".join(f'"{value}"' for value in details["examples"])
        raise ValueError(
            f'Feature "{feature}" looks numeric but contains {details["n_non_numeric"]} non-numeric '
            f"value(s) such as {examples}. Recode those values as missing (blank cells) so the column "
            "is used as a number, or recode the whole column into a small set of categories."
        )


def ordered_level_labels(values: pd.Series, column_name: str | None = None) -> list[str]:
    """Observed labels in the order used for reference levels.

    Numeric-looking labels sort numerically ("2" before "10"); other labels follow the
    clinical reference ordering used by the Cox workflow (for example stage I < II < III,
    never smoker before current smoker, wild type before mutant).
    """

    from survival_toolkit.analysis import _ordered_unique_level_strings

    return _ordered_unique_level_strings(values.dropna().astype("string"), column_name)


def coerce_feature_subset(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Resolve feature dtypes into deterministic categorical/numeric subsets."""

    categorical_features = [col for col in list(categorical_features or []) if col in features]
    selected = df.loc[:, list(features)].copy()
    resolved_categorical: list[str] = []
    for column in features:
        if column in categorical_features:
            selected[column] = selected[column].astype("string")
            resolved_categorical.append(column)
            continue
        if is_numeric_dtype(selected[column]):
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
            continue
        selected[column] = selected[column].astype("string")
        resolved_categorical.append(column)
    numeric_features = [column for column in features if column not in resolved_categorical]
    return selected, resolved_categorical, numeric_features


def ordered_category_values(series: pd.Series) -> list[str]:
    """Return observed category values in a deterministic, semantically stable order."""

    non_missing = series.dropna()
    if non_missing.empty:
        return []
    if isinstance(series.dtype, pd.CategoricalDtype) and getattr(series.dtype, "ordered", False):
        return [str(value) for value in series.dtype.categories.tolist() if pd.notna(value)]
    return ordered_level_labels(non_missing, str(series.name) if series.name is not None else None)


def fit_feature_encoder(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    *,
    standardize_numeric: bool = False,
) -> dict[str, Any]:
    """Fit a deterministic tabular encoder shared by ML and DL paths."""

    if not list(features):
        raise ValueError("Select at least one feature before fitting the encoder.")
    reject_numeric_text_features(df, features)

    selected, resolved_categorical, numeric_features = coerce_feature_subset(
        df,
        features,
        categorical_features,
    )

    categorical_mappings: dict[str, dict[str, Any]] = {}
    categorical_levels: dict[str, list[str]] = {}
    categorical_all_levels: dict[str, list[str]] = {}
    categorical_unknown_columns: dict[str, str | None] = {}
    categorical_missing_columns: dict[str, str | None] = {}
    feature_names: list[str] = []
    categorical_feature_indices: list[int] = []
    # Numeric features keep their raw names, so generated dummy names must avoid them (and each
    # other): a categorical "tgrade" with level "III" next to a numeric "tgrade_III" column would
    # otherwise produce two encoded columns with the same name.
    used_names: set[str] = {str(column) for column in numeric_features}

    def _allocate_name(base_name: str) -> str:
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}__{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    for column in resolved_categorical:
        # The first level is the dropped baseline, so it follows the same reference ordering
        # as the Cox workflow instead of plain string order ("10" before "2").
        levels = ordered_level_labels(selected[column], str(column))
        retained_levels = levels[1:] if len(levels) > 1 else []
        level_columns = {level: _allocate_name(f"{column}_{level}") for level in retained_levels}
        # Indicators that are constant in the fitting data carry no information for any
        # model: an unseen level can never occur while fitting, and a missing-value
        # indicator only varies when the fitting data contain missing values. Unseen or
        # missing values at transform time are scored as the baseline level.
        missing_column = (
            _allocate_name(f"{column}__missing") if bool(selected[column].isna().any()) else None
        )
        categorical_mappings[column] = {
            "all_levels": levels,
            "baseline_level": levels[0] if levels else None,
            "retained_levels": retained_levels,
            "level_columns": level_columns,
            "unknown_column": None,
            "missing_column": missing_column,
        }
        categorical_levels[column] = retained_levels
        categorical_all_levels[column] = levels
        categorical_unknown_columns[column] = None
        categorical_missing_columns[column] = missing_column
        start_index = len(feature_names)
        feature_names.extend(level_columns[level] for level in retained_levels)
        if missing_column is not None:
            feature_names.append(missing_column)
        categorical_feature_indices.extend(range(start_index, len(feature_names)))

    numeric_impute_values: dict[str, float] = {}
    scaler_params: dict[str, dict[str, float]] = {}
    numeric_feature_indices: list[int] = []
    if numeric_features:
        # Treat +/-inf as missing so a single infinite value cannot make the mean infinite and the
        # standard deviation NaN (which would zero the whole feature after scaling).
        numeric_frame = (
            selected[numeric_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        medians = numeric_frame.median(skipna=True).fillna(0.0)
        numeric_array = numeric_frame.fillna(medians).to_numpy(dtype=np.float64)
        means = np.mean(numeric_array, axis=0) if standardize_numeric else np.zeros(len(numeric_features), dtype=float)
        stds = np.std(numeric_array, axis=0) if standardize_numeric else np.ones(len(numeric_features), dtype=float)
        stds[stds < 1e-12] = 1.0
        for index, column in enumerate(numeric_features):
            numeric_impute_values[column] = float(medians[column])
            scaler_params[column] = {
                "mean": float(means[index]),
                "std": float(stds[index]),
                "impute_value": float(medians[column]),
            }
        start_index = len(feature_names)
        feature_names.extend(numeric_features)
        numeric_feature_indices.extend(range(start_index, len(feature_names)))

    if not feature_names:
        raise ValueError(
            "No usable features remain after encoding. "
            "This can happen when all categorical features have only one level."
        )

    return {
        "features": list(features),
        "categorical_features": resolved_categorical,
        "numeric_features": numeric_features,
        "categorical_mappings": categorical_mappings,
        "categorical_levels": categorical_levels,
        "categorical_all_levels": categorical_all_levels,
        "categorical_unknown_columns": categorical_unknown_columns,
        "categorical_missing_columns": categorical_missing_columns,
        "numeric_impute_values": numeric_impute_values,
        "scaler_params": scaler_params,
        "feature_names": feature_names,
        "encoded_columns": list(feature_names),
        "categorical_feature_indices": categorical_feature_indices,
        "numeric_feature_indices": numeric_feature_indices,
        "standardize_numeric": bool(standardize_numeric),
    }


def transform_feature_encoder(
    df: pd.DataFrame,
    encoder: dict[str, Any],
    *,
    output: Literal["dataframe", "numpy"] = "dataframe",
) -> pd.DataFrame | np.ndarray:
    """Transform features with a fitted shared encoder."""

    selected, _, _ = coerce_feature_subset(
        df,
        encoder["features"],
        encoder.get("categorical_features"),
    )
    encoded_columns: dict[str, pd.Series] = {}

    for column in encoder.get("categorical_features", []):
        mapping = encoder["categorical_mappings"][column]
        values = selected[column].astype("string")
        all_levels = pd.Index(mapping["all_levels"], dtype="string")
        level_columns = mapping.get("level_columns") or {}
        for level in mapping["retained_levels"]:
            encoded_name = level_columns.get(level, f"{column}_{level}")
            encoded_columns[encoded_name] = values.eq(level).fillna(False).astype(float)
        # Hand-built encoder dicts may still name an unknown-level column.
        if mapping.get("unknown_column"):
            unknown_mask = values.notna() & ~values.isin(all_levels)
            encoded_columns[mapping["unknown_column"]] = unknown_mask.astype(float)
        if mapping.get("missing_column"):
            encoded_columns[mapping["missing_column"]] = values.isna().astype(float)

    for column in encoder.get("numeric_features", []):
        numeric_series = pd.to_numeric(selected[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        impute_value = float(encoder.get("numeric_impute_values", {}).get(column, 0.0))
        numeric_series = numeric_series.fillna(impute_value).astype(float)
        if encoder.get("standardize_numeric"):
            params = encoder.get("scaler_params", {}).get(column, {})
            numeric_series = (numeric_series - float(params.get("mean", 0.0))) / float(params.get("std", 1.0))
        encoded_columns[column] = numeric_series

    encoded = pd.DataFrame(encoded_columns, index=selected.index)
    encoded = encoded.reindex(columns=encoder["feature_names"], fill_value=0.0)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)

    for column in encoder.get("numeric_features", []):
        params = encoder.get("scaler_params", {}).get(column, {})
        if encoder.get("standardize_numeric"):
            fill_value = (
                float(params.get("impute_value", 0.0)) - float(params.get("mean", 0.0))
            ) / float(params.get("std", 1.0))
        else:
            fill_value = float(encoder.get("numeric_impute_values", {}).get(column, 0.0))
        encoded[column] = encoded[column].fillna(fill_value).astype(float)
    encoded = encoded.fillna(0.0)

    if output == "numpy":
        return encoded.to_numpy(dtype=np.float64, copy=False)
    return encoded

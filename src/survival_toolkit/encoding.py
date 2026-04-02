from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    if numeric_values.notna().all():
        ordered_numeric = np.sort(numeric_values.unique().astype(float))
        return [str(int(value)) if float(value).is_integer() else str(value) for value in ordered_numeric]
    return list(dict.fromkeys(non_missing.astype("string").tolist()))


def fit_feature_encoder(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    *,
    standardize_numeric: bool = False,
) -> dict[str, Any]:
    """Fit a deterministic tabular encoder shared by ML and DL paths."""

    selected, resolved_categorical, numeric_features = coerce_feature_subset(
        df,
        features,
        categorical_features,
    )

    categorical_mappings: dict[str, dict[str, Any]] = {}
    categorical_levels: dict[str, list[str]] = {}
    categorical_all_levels: dict[str, list[str]] = {}
    categorical_unknown_columns: dict[str, str] = {}
    categorical_missing_columns: dict[str, str] = {}
    feature_names: list[str] = []
    categorical_feature_indices: list[int] = []

    for column in resolved_categorical:
        levels = sorted(
            str(level)
            for level in selected[column].dropna().astype("string").unique().tolist()
        )
        retained_levels = levels[1:] if len(levels) > 1 else []
        unknown_column = f"{column}__unknown"
        missing_column = f"{column}__missing"
        categorical_mappings[column] = {
            "all_levels": levels,
            "baseline_level": levels[0] if levels else None,
            "retained_levels": retained_levels,
            "unknown_column": unknown_column,
            "missing_column": missing_column,
        }
        categorical_levels[column] = retained_levels
        categorical_all_levels[column] = levels
        categorical_unknown_columns[column] = unknown_column
        categorical_missing_columns[column] = missing_column
        start_index = len(feature_names)
        feature_names.extend([f"{column}_{level}" for level in retained_levels])
        feature_names.append(unknown_column)
        feature_names.append(missing_column)
        categorical_feature_indices.extend(range(start_index, len(feature_names)))

    numeric_impute_values: dict[str, float] = {}
    scaler_params: dict[str, dict[str, float]] = {}
    numeric_feature_indices: list[int] = []
    if numeric_features:
        numeric_frame = selected[numeric_features].apply(pd.to_numeric, errors="coerce")
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
        for level in mapping["retained_levels"]:
            encoded_columns[f"{column}_{level}"] = values.eq(level).fillna(False).astype(float)
        missing_mask = values.isna()
        unknown_mask = values.notna() & ~values.isin(all_levels)
        encoded_columns[mapping["unknown_column"]] = unknown_mask.astype(float)
        encoded_columns[mapping["missing_column"]] = missing_mask.astype(float)

    for column in encoder.get("numeric_features", []):
        numeric_series = pd.to_numeric(selected[column], errors="coerce")
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

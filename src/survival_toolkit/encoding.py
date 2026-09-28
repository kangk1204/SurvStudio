from __future__ import annotations

from collections import Counter
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
# A text feature that was not marked categorical is refused when it has more distinct values
# than this: it is almost always numbers stored as text (values below a detection limit such as
# "<0.1", decimal commas) or an identifier, and one-hot encoding would add a column per value.
MAX_TEXT_LEVELS = 50


def canonical_category_values(series: pd.Series) -> pd.Series:
    """Text labels of a categorical feature that do not depend on how the column was read.

    Integer-valued numbers are written without a decimal part, so a code column read as whole
    numbers (1, 2, 3) and the same column read as decimals because one cell is blank (1.0, 2.0,
    NaN) give the same levels "1", "2", "3"; booleans read "True"/"False"; text keeps its value.
    """

    from survival_toolkit.analysis import _canonical_level_strings

    return _canonical_level_strings(series)


def _float_values(series: pd.Series) -> pd.Series:
    """A numeric feature as float64 with NaN for missing values.

    Nullable integer and boolean columns (from Parquet or pandas' nullable dtypes) cannot hold a
    fractional median, so they are converted before any imputation.
    """

    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric.to_numpy(dtype="float64", na_value=np.nan), index=series.index, name=series.name)


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


def text_level_overflow(series: pd.Series) -> dict[str, Any] | None:
    """Describe a text column with more than ``MAX_TEXT_LEVELS`` distinct values.

    Returns ``None`` for numeric and boolean columns and for text columns with few enough levels.
    """

    if is_numeric_dtype(series) or is_bool_dtype(series):
        return None
    non_missing = series.dropna()
    if non_missing.empty:
        return None
    n_levels = int(non_missing.astype("string").nunique())
    if n_levels <= MAX_TEXT_LEVELS:
        return None
    text = non_missing.astype(str).str.strip()
    numeric_mask = pd.to_numeric(text, errors="coerce").notna()
    with_points = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce").notna()
    return {
        "n_levels": n_levels,
        "n_values": int(len(text)),
        "n_numeric": int(numeric_mask.sum()),
        "decimal_comma": bool(with_points.mean() >= _NUMERIC_TEXT_MIN_SHARE and text.str.contains(",", regex=False).any()),
        "examples": list(dict.fromkeys(text[~numeric_mask].tolist()))[:3],
    }


def _text_level_message(feature: Any, details: dict[str, Any], *, declarations: bool) -> str:
    examples = ", ".join(f'"{value}"' for value in details["examples"])
    n_levels = details["n_levels"]
    if details["decimal_comma"]:
        message = (
            f'Feature "{feature}" looks like numbers written with commas (such as {examples}); as text it would become a '
            f"categorical feature with {n_levels} levels. Write the numbers with a decimal point and without thousands "
            "separators so the column is read as numbers."
        )
    elif details["n_numeric"] >= 0.5 * details["n_values"]:
        n_text = details["n_values"] - details["n_numeric"]
        message = (
            f'Feature "{feature}" is stored as text: {details["n_numeric"]} of {details["n_values"]} values are numbers but '
            f"{n_text} {'is' if n_text == 1 else 'are'} not (such as {examples}), so it would become a categorical feature "
            f"with {n_levels} levels. Recode those entries as numbers or blank cells so the column is used as a number."
        )
    else:
        message = (
            f'Feature "{feature}" has {n_levels} distinct text values, so it would become a categorical feature with '
            f"{n_levels - 1} indicator columns. Recode it into at most {MAX_TEXT_LEVELS} categories, or leave it out if it "
            "identifies patients or samples."
        )
    if declarations:
        message += " To keep every level, mark it as categorical."
    return message


def reject_numeric_text_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> None:
    """Refuse model features that are continuous numbers with a few stray text values, and
    text features with more than ``MAX_TEXT_LEVELS`` distinct values.

    ML, deep-learning, and Cox paths share this rule so the same column cannot be a
    hundreds-of-levels categorical in one module and an error in another. Features listed in
    ``categorical_features`` were marked categorical by the user and are used as they are;
    callers that do not pass the list get both checks for every feature.
    """

    declared = {str(column) for column in categorical_features or []}
    for feature in features:
        if feature not in frame.columns or str(feature) in declared:
            continue
        series = frame[feature]
        if isinstance(series, pd.DataFrame):
            # Duplicated column labels: the encoder reports them with a clear message.
            continue
        details = numeric_text_contamination(series)
        if details is not None:
            examples = ", ".join(f'"{value}"' for value in details["examples"])
            raise ValueError(
                f'Feature "{feature}" looks numeric but contains {details["n_non_numeric"]} non-numeric '
                f"value(s) such as {examples}. Recode those values as missing (blank cells) so the column "
                "is used as a number, or recode the whole column into a small set of categories."
            )
        overflow = text_level_overflow(series)
        if overflow is not None:
            raise ValueError(_text_level_message(feature, overflow, declarations=categorical_features is not None))


def ordered_level_labels(values: pd.Series, column_name: str | None = None) -> list[str]:
    """Observed labels in the order used for reference levels.

    Numeric-looking labels sort numerically ("2" before "10"); other labels follow the
    clinical reference ordering used by the Cox workflow (for example stage I < II < III,
    never smoker before current smoker, wild type before mutant).
    """

    from survival_toolkit.analysis import _ordered_unique_level_strings

    return _ordered_unique_level_strings(canonical_category_values(values.dropna()), column_name)


def _checked_features(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """The feature list, refused with a clear message when it repeats a name or names a column the data lack."""

    features = list(features)
    repeated = [str(name) for name, count in Counter(features).items() if count > 1]
    if repeated:
        raise ValueError(f"Each feature can be used once; listed more than once: {', '.join(repeated[:5])}.")
    missing = [str(name) for name in features if name not in df.columns]
    if missing:
        raise ValueError(
            f"The data lack the feature column(s) {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}."
        )
    duplicated_columns = {str(name) for name in df.columns[df.columns.duplicated()]}
    ambiguous = [str(name) for name in features if str(name) in duplicated_columns]
    if ambiguous:
        raise ValueError(
            f"The data have more than one column named {', '.join(ambiguous[:5])}; rename them before modelling."
        )
    return features


def coerce_feature_subset(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Resolve feature dtypes into deterministic categorical/numeric subsets.

    Categorical features become canonical text labels (``canonical_category_values``), so their
    levels do not depend on whether a code column was read as whole numbers or as decimals;
    numeric features become float64 with NaN for missing values.
    """

    features = _checked_features(df, features)
    categorical_features = [col for col in list(categorical_features or []) if col in features]
    selected = df.loc[:, features].copy()
    resolved_categorical: list[str] = []
    for column in features:
        if column in categorical_features or not is_numeric_dtype(selected[column]):
            selected[column] = canonical_category_values(selected[column])
            resolved_categorical.append(column)
            continue
        selected[column] = _float_values(selected[column])
    numeric_features = [column for column in features if column not in resolved_categorical]
    return selected, resolved_categorical, numeric_features


def _stored_level_values(values: pd.Series, levels: Sequence[Any]) -> pd.Series:
    """Canonical category values mapped onto an encoder's stored levels.

    A value that is not a stored level but has the same numeric value as one ("1" against "1.0")
    takes that level. This keeps encoders fitted before the labels were canonical ("1.0"-style
    levels from a column read as decimals) working, and matches "2.0" in a column that also holds
    2.5 against a stored "2".
    """

    values = values.astype("string")
    stored = [str(level) for level in levels]
    stored_set = set(stored)
    unmatched = [value for value in values.dropna().unique().tolist() if value not in stored_set]
    if not unmatched:
        return values
    stored_numbers = pd.to_numeric(pd.Series(stored, dtype="object"), errors="coerce").tolist()
    by_number: dict[float, str] = {}
    for level, number in zip(stored, stored_numbers):
        if pd.notna(number) and np.isfinite(float(number)):
            by_number.setdefault(float(number), level)
    if not by_number:
        return values
    unmatched_numbers = pd.to_numeric(pd.Series(unmatched, dtype="object"), errors="coerce").tolist()
    translation = {
        value: by_number[float(number)]
        for value, number in zip(unmatched, unmatched_numbers)
        if pd.notna(number) and float(number) in by_number
    }
    return values.replace(translation) if translation else values


def unseen_category_counts(df: pd.DataFrame, encoder: dict[str, Any]) -> dict[str, int]:
    """Rows per categorical feature whose value matches no level seen when the encoder was fitted.

    The encoder scores those rows as the reference level. Values are compared the way
    ``transform_feature_encoder`` compares them, so 1.0 and 1 count as the same level.
    """

    selected, _, _ = coerce_feature_subset(df, encoder["features"], encoder.get("categorical_features"))
    counts: dict[str, int] = {}
    for column in encoder.get("categorical_features", []):
        levels = [str(level) for level in encoder["categorical_mappings"][column]["all_levels"]]
        values = _stored_level_values(selected[column], levels).dropna()
        counts[column] = int((~values.isin(levels)).sum())
    return counts


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
    features = _checked_features(df, features)
    # Columns the user marked categorical are used as categorical even when they look numeric.
    reject_numeric_text_features(df, features, list(categorical_features or []))

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
        values = _stored_level_values(selected[column], mapping["all_levels"])
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

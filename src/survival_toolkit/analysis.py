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
from scipy.special import ndtri
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from survival_toolkit.errors import ColumnNotFoundError, user_input_boundary

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
    "living",
    "diseasefree",
    "disease-free",
    "nonevent",
    "non-event",
}
_GENERIC_TRUE_TOKENS = {"1", "true", "yes", "y", "event"}
_DEATH_TRUE_TOKENS = {"dead", "death", "deceased"}
_PROGRESSION_TRUE_TOKENS = {"progressed", "progression", "relapse", "failure"}
_EVENT_TOKEN_FAMILIES = {
    **{token: "event_generic" for token in _GENERIC_TRUE_TOKENS},
    **{token: "event_death" for token in _DEATH_TRUE_TOKENS},
    **{token: "event_progression" for token in _PROGRESSION_TRUE_TOKENS},
    **{token: "censor" for token in FALSE_TOKENS},
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
MAX_MODEL_FEATURE_CANDIDATES = 1000
_MODEL_FEATURE_ID_PATTERN = re.compile(r"^(patient_id|sample_id|subject_id|id|barcode|uuid|cohort)$", re.IGNORECASE)
_EVENT_NAME_PATTERNS = (
    re.compile(r"event"),
    re.compile(r"death"),
    re.compile(r"mort"),
    re.compile(r"status$"),
    re.compile(r"vital_status"),
    re.compile(r"survival_status"),
    re.compile(r"outcome_status"),
    re.compile(r"relapse"),
    re.compile(r"recur"),
    re.compile(r"progress"),
    re.compile(r"failure"),
    re.compile(r"censor"),
)
_SURVIVAL_ENDPOINT_ABBREVIATIONS = {"os", "pfs", "dfs", "rfs", "efs", "tts", "tte", "dss", "css", "mfs", "dmfs"}
_TIME_UNIT_TOKENS = {"day", "days", "week", "weeks", "month", "months", "year", "years"}
_TIME_CONTEXT_TOKENS = {
    "time",
    "duration",
    "survival",
    "follow",
    "followup",
    "follow_up",
    "fup",
}
_OUTCOME_CONTEXT_TOKENS = {
    "event",
    "death",
    "mort",
    "status",
    "progress",
    "progression",
    "relapse",
    "recur",
    "recurrence",
    "failure",
    "censor",
}
_TIME_NAME_TOKENS = (
    "time",
    "os",
    "pfs",
    "dfs",
    "rfs",
    "efs",
    "tts",
    "tte",
    "dss",
    "css",
    "follow",
    "followup",
    "follow_up",
    "fup",
    "survival",
)
_OUTCOME_STATUS_VALUE_FAMILIES = {
    "alive": "censor",
    "alivewithoutdisease": "censor",
    "censored": "censor",
    "diseasefree": "censor",
    "living": "censor",
    "ned": "censor",
    "noevidenceofdisease": "censor",
    "nonevent": "censor",
    "awd": "event_progression",
    "alivewithdisease": "event_progression",
    "deceased": "event_death",
    "dead": "event_death",
    "deadofdisease": "event_death",
    "death": "event_death",
    "deathofdisease": "event_death",
    "died": "event_death",
    "doc": "event_death",
    "dod": "event_death",
    "failure": "event_progression",
    "progressed": "event_progression",
    "progression": "event_progression",
    "recurrence": "event_progression",
    "recurred": "event_progression",
    "relapse": "event_progression",
    "relapsed": "event_progression",
}
_BASELINE_STATUS_PATTERNS = (
    re.compile(r"egfr"),
    re.compile(r"kras"),
    re.compile(r"braf"),
    re.compile(r"alk"),
    re.compile(r"ros1"),
    re.compile(r"erbb2"),
    re.compile(r"mutation"),
    re.compile(r"mutated"),
    re.compile(r"wildtype"),
    re.compile(r"sex"),
    re.compile(r"gender"),
    re.compile(r"stage"),
    re.compile(r"grade"),
    re.compile(r"treat"),
    re.compile(r"therapy"),
    re.compile(r"drug"),
    re.compile(r"smok"),
    re.compile(r"histolog"),
    re.compile(r"subtype"),
    re.compile(r"cluster"),
    re.compile(r"group"),
    re.compile(r"arm"),
    re.compile(r"cohort"),
    re.compile(r"horth"),
)


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    used: set[str] = set()
    output: list[str] = []
    for raw_name in columns:
        name = str(raw_name).strip() or "unnamed"
        counter = seen.get(name, 0)
        unique_name = name if counter == 0 else f"{name}_{counter + 1}"
        while unique_name in used:
            counter += 1
            unique_name = f"{name}_{counter + 1}"
        seen[name] = counter + 1
        used.add(unique_name)
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


@user_input_boundary
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


def _column_matches_keyword(column: str, token: str) -> bool:
    lowered = str(column).lower()
    normalized_token = str(token).lower()
    if not lowered or not normalized_token:
        return False
    if len(normalized_token) >= 4:
        return normalized_token in lowered
    pluralizable_time_units = {"day", "week", "month", "year"}
    suffix = "s?" if normalized_token in pluralizable_time_units else ""
    pattern = rf"(^|[^a-z0-9]){re.escape(normalized_token)}{suffix}([^a-z0-9]|$)"
    return re.search(pattern, lowered) is not None


def _column_keywords(columns: Sequence[str], tokens: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        if any(_column_matches_keyword(column, token) for token in tokens):
            matches.append(column)
    return matches


def _normalize_column_label(name: str) -> str:
    return str(name or "").strip().lower()


def _token_variants(token: str | None) -> tuple[str, ...]:
    raw = str(token or "").strip().lower()
    if not raw:
        return ()
    variants: list[str] = []

    def _add(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    _add(raw)
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    _add(compact)
    for part in re.split(r"[^a-z0-9]+", raw):
        _add(part)
    return tuple(variants)


def _column_name_tokens(name: str) -> list[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name or ""))
    return [token for token in re.split(r"[^a-z0-9]+", expanded.lower()) if token]


def _tokens_contain_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not tokens or not phrase or len(tokens) < len(phrase):
        return False
    width = len(phrase)
    return any(tuple(tokens[idx:idx + width]) == tuple(phrase) for idx in range(len(tokens) - width + 1))


def _endpoint_family_from_column_name(name: str) -> str | None:
    tokens = _column_name_tokens(name)
    if not tokens:
        return None
    token_set = set(tokens)
    phrase_families = (
        ("os", ("overall", "survival")),
        ("pfs", ("progression", "free")),
        ("dfs", ("disease", "free")),
        ("rfs", ("recurrence", "free")),
        ("rfs", ("relapse", "free")),
        ("efs", ("event", "free")),
        ("dss", ("disease", "specific")),
        ("css", ("cancer", "specific")),
    )
    for family, phrase in phrase_families:
        if _tokens_contain_phrase(tokens, phrase):
            return family
    for family in ("os", "pfs", "dfs", "rfs", "efs", "tts", "tte", "dss", "css"):
        if family in token_set:
            return family
    return None


def _validate_endpoint_family_pair(time_column: str, event_column: str) -> None:
    time_family = _endpoint_family_from_column_name(time_column)
    event_family = _endpoint_family_from_column_name(event_column)
    if time_family and event_family and time_family != event_family:
        raise ValueError(
            f'"{time_column}" and "{event_column}" look like different survival endpoints. '
            "Choose a matched time/event pair from the same endpoint family."
        )


def _looks_like_survival_time_column_name(name: str) -> bool:
    normalized = _normalize_column_label(name)
    tokens = _column_name_tokens(name)
    if not normalized or not tokens:
        return False

    token_set = set(tokens)
    has_time_units = bool(token_set & _TIME_UNIT_TOKENS)
    has_abbreviation = bool(token_set & _SURVIVAL_ENDPOINT_ABBREVIATIONS)
    has_followup = (
        "followup" in token_set
        or _tokens_contain_phrase(tokens, ("follow", "up"))
        or "fu" in token_set
        or "fup" in token_set
    )
    has_survival_context = has_abbreviation or "survival" in token_set or "surv" in token_set or has_followup
    has_event_context = bool(token_set & _OUTCOME_CONTEXT_TOKENS)
    has_time_keyword = "time" in token_set
    has_duration_keyword = "duration" in token_set
    has_generic_time_context = has_time_keyword or has_duration_keyword

    if normalized in {"time", "survival_time", "event_time", "time_to_event", "followup_time"}:
        return True
    if _looks_like_baseline_status_column(name) and not (has_survival_context or has_event_context):
        return False
    if has_duration_keyword and has_time_units and not (has_survival_context or has_event_context):
        return False
    if has_survival_context and (has_time_units or has_generic_time_context or not has_event_context):
        return True
    if has_generic_time_context and (len(tokens) == 1 or has_survival_context or has_event_context):
        return True
    if has_time_units and (has_survival_context or has_event_context or has_time_keyword):
        return True
    return False


def _is_event_like_column_name(name: str) -> bool:
    normalized = _normalize_column_label(name)
    if not normalized:
        return False
    if normalized in {"event", "status"}:
        return True
    return any(pattern.search(normalized) for pattern in _EVENT_NAME_PATTERNS)


def _looks_like_baseline_status_column(name: str) -> bool:
    normalized = _normalize_column_label(name)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _BASELINE_STATUS_PATTERNS)


def _outcome_status_value_family(value: Any) -> str | None:
    normalized = _normalize_token(value)
    if normalized is None:
        return None
    for candidate in _token_variants(normalized):
        family = _OUTCOME_STATUS_VALUE_FAMILIES.get(candidate)
        if family is not None:
            return family
    for candidate in _token_variants(normalized):
        if candidate in _EVENT_TOKEN_FAMILIES:
            return _EVENT_TOKEN_FAMILIES[candidate]
    return None


def _looks_like_event_outcome_column(name: str, series: pd.Series) -> bool:
    if not _is_event_like_column_name(name) or _looks_like_baseline_status_column(name):
        return False
    if looks_binary(series):
        return True
    families = {
        family
        for family in (_outcome_status_value_family(value) for value in series.dropna().unique().tolist())
        if family is not None
    }
    return bool(families)


def _has_recognizable_event_coding(series: pd.Series) -> bool:
    valid = series.dropna()
    if valid.empty:
        return False

    numeric_series = pd.to_numeric(valid, errors="coerce")
    if numeric_series.notna().sum() == len(valid):
        observed_numeric = {float(value) for value in numeric_series.astype(float).unique().tolist()}
        if observed_numeric in ({0.0, 1.0}, {1.0, 2.0}):
            return True
        if len(observed_numeric) == 1 and next(iter(observed_numeric)) in {0.0, 1.0, 2.0}:
            return True

    families = {
        family
        for family in (_outcome_status_value_family(value) for value in valid.unique().tolist())
        if family is not None
    }
    return bool(families - {"censor"})


def _model_feature_candidate_columns_from_metadata(
    columns: Sequence[str],
    *,
    suggested_time_columns: Sequence[str],
    binary_candidate_columns: Sequence[str],
) -> list[str]:
    suggested_time_set = set(suggested_time_columns)
    binary_set = set(binary_candidate_columns)
    candidates: list[str] = []
    for column in columns:
        if _MODEL_FEATURE_ID_PATTERN.fullmatch(column):
            continue
        if column in suggested_time_set:
            continue
        if column in binary_set and _is_event_like_column_name(column) and not _looks_like_baseline_status_column(column):
            continue
        candidates.append(column)
    return candidates


def _survival_outcome_like_columns(df: pd.DataFrame) -> set[str]:
    likely_time_columns = {column for column in df.columns if _looks_like_survival_time_column_name(column)}
    likely_event_columns = {
        column for column in df.columns if _looks_like_event_outcome_column(column, df[column])
    }
    return likely_time_columns | likely_event_columns


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
            target_token = _normalize_token(target)
            if target_token in FALSE_TOKENS:
                raise ValueError(
                    f"The selected event-positive value '{target}' maps to censoring, not the event."
                )
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
        observed_family_map = {
            token: _outcome_status_value_family(token)
            for token in observed_tokens
        }
        observed_families = {family for family in observed_family_map.values() if family is not None}

        def _raise_if_multistate_family(target_family: str) -> None:
            disallowed = sorted(observed_families - {target_family, "censor"})
            if disallowed:
                raise ValueError(
                    "The event column contains more than one recognized event state. "
                    "Recode it to a binary event indicator before survival analysis."
                )

        # If the target is a known event/censor token, decode using the full token vocabulary.
        if target_token in TRUE_TOKENS:
            unknown = [tok for tok, family in observed_family_map.items() if family is None]
            if unknown:
                raise ValueError(
                    "Event coding contains unrecognized tokens alongside standard event/censor labels: "
                    + ", ".join(unknown[:6])
                    + (" ..." if len(unknown) > 6 else "")
                )
            target_family = _outcome_status_value_family(target_token)
            if target_family is None:
                raise ValueError("The selected event-positive value could not be mapped to a supported event family.")
            if target_family == "event_generic":
                concrete_event_families = sorted(observed_families - {"censor"})
                if len(concrete_event_families) == 1:
                    target_family = concrete_event_families[0]
            _raise_if_multistate_family(target_family)
            event_tokens = {
                token
                for token, family in observed_family_map.items()
                if family == target_family
            }
        elif target_token in FALSE_TOKENS:
            unknown = [tok for tok, family in observed_family_map.items() if family is None]
            if unknown:
                raise ValueError(
                    "Event coding contains unrecognized tokens alongside standard event/censor labels: "
                    + ", ".join(unknown[:6])
                    + (" ..." if len(unknown) > 6 else "")
                )
            raise ValueError(
                f"The selected event-positive value '{target}' maps to censoring, not the event. "
                "Choose the value that means the event happened."
            )
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

    inferred, inference_error = _try_coerce_binary_event(series)
    if inferred is not None:
        return inferred
    if inference_error == "multistate":
        raise ValueError(
            "The event column contains more than one recognized event state. "
            "Recode it to a binary event indicator before survival analysis."
        )

    raise ValueError(
        "Could not infer event coding. Select the value that represents the event in the dashboard."
    )


def looks_binary(series: pd.Series) -> bool:
    valid = series.dropna()
    if valid.empty:
        return False

    inferred, inference_error = _try_coerce_binary_event(series)
    if inferred is not None:
        non_missing = inferred.dropna()
        return not non_missing.empty and set(non_missing.unique()).issubset({0.0, 1.0})
    if inference_error == "multistate":
        return False

    numeric_series = pd.to_numeric(valid, errors="coerce")
    if numeric_series.notna().sum() == len(valid):
        observed_numeric = {float(value) for value in numeric_series.astype(float).tolist()}
        return len(observed_numeric) == 2

    normalized = valid.map(_normalize_token)
    normalized_non_missing = normalized.dropna()
    if len(normalized_non_missing) == len(valid):
        observed_families = {
            family
            for family in (_outcome_status_value_family(token) for token in normalized_non_missing.tolist())
            if family is not None
        }
        if not observed_families:
            observed_tokens = set(normalized_non_missing.tolist())
            return len(observed_tokens) == 2
        event_families = observed_families - {"censor"}
        return len(event_families) == 1 and bool(event_families)

    return int(valid.nunique(dropna=True)) == 2


def _try_coerce_binary_event(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    valid = series.notna()
    if not valid.any():
        return pd.Series(np.nan, index=series.index, dtype=float), None

    out = pd.Series(np.nan, index=series.index, dtype=float)
    if is_bool_dtype(series):
        out.loc[valid] = series.loc[valid].astype(int).astype(float)
        return out, None

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() == int(valid.sum()):
        unique_floats = set(numeric_series.loc[valid].unique().tolist())
        if unique_floats.issubset({0.0, 1.0}):
            out.loc[valid] = numeric_series.loc[valid].astype(float)
            return out, None

    normalized_families = series.map(_outcome_status_value_family)
    mapped = pd.Series(np.nan, index=series.index, dtype=float)
    mapped.loc[normalized_families.eq("censor")] = 0.0
    mapped.loc[normalized_families.notna() & normalized_families.ne("censor")] = 1.0
    if mapped.loc[valid].notna().sum() == int(valid.sum()):
        observed_families = {
            family
            for family in normalized_families.loc[valid].dropna().astype(str).tolist()
        }
        event_families = observed_families - {"censor"}
        if len(event_families) > 1:
            return None, "multistate"
        return mapped, None

    return None, "unrecognized"


@user_input_boundary
def find_event_equivalent_columns(
    df: pd.DataFrame,
    event_column: str,
    event_positive_value: Any = None,
) -> set[str]:
    if event_column not in df.columns:
        return set()
    try:
        reference = coerce_event(df[event_column], event_positive_value=event_positive_value)
    except ValueError:
        return set()

    reference_valid = reference.notna()
    if not reference_valid.any():
        return set()

    equivalents: set[str] = set()
    reference_values = reference.to_numpy(dtype=float, na_value=np.nan)
    for column in df.columns:
        if column == event_column or _looks_like_baseline_status_column(column):
            continue
        series = df[column]
        if not looks_binary(series) and not _has_recognizable_event_coding(series):
            continue
        try:
            candidate = coerce_event(series)
        except ValueError:
            continue
        overlap = reference_valid & candidate.notna()
        if int(overlap.sum()) < 3:
            continue
        candidate_values = candidate.to_numpy(dtype=float, na_value=np.nan)
        if np.array_equal(candidate_values[overlap.to_numpy(dtype=bool)], reference_values[overlap.to_numpy(dtype=bool)]):
            equivalents.add(str(column))
    return equivalents


def suggest_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(df.columns)
    event_tokens = ("event", "status", "death", "progress", "relapse", "censor")
    time_columns = [
        column
        for column in columns
        if _looks_like_survival_time_column_name(column)
    ]
    suggestions = {
        "time_columns": time_columns,
        "event_columns": _column_keywords(columns, event_tokens),
        "group_columns": _column_keywords(columns, ("group", "arm", "treatment", "stage", "sex", "risk", "cluster")),
    }
    return suggestions


def _profile_dataframe_column(column: str, series: pd.Series) -> tuple[dict[str, Any], bool, bool]:
    kind = _column_kind(series)
    profile = {
        "name": str(column),
        "kind": kind,
        "missing": int(series.isna().sum()),
        "non_missing": int(series.notna().sum()),
        "n_unique": int(series.nunique(dropna=True)),
        "unique_preview": [serialize_value(value) for value in series.dropna().unique()[:8]],
    }
    is_numeric = bool(is_numeric_dtype(series))
    if is_numeric:
        profile["min"] = serialize_value(series.min())
        profile["max"] = serialize_value(series.max())
    is_binary = looks_binary(series)
    return profile, is_numeric, is_binary


def _require_dataframe_columns(df: pd.DataFrame, columns: Sequence[str | None]) -> None:
    requested: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column is None:
            continue
        name = str(column)
        if not name or name in seen:
            continue
        requested.append(name)
        seen.add(name)
    missing = [name for name in requested if name not in df.columns]
    if not missing:
        return
    if len(missing) == 1:
        raise ColumnNotFoundError(f'Column not found in dataset: "{missing[0]}".')
    quoted = ", ".join(f'"{name}"' for name in missing)
    raise ColumnNotFoundError(f"Columns not found in dataset: {quoted}.")


@user_input_boundary
def profile_dataframe(df: pd.DataFrame, dataset_id: str, filename: str) -> dict[str, Any]:
    column_profiles: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    binary_candidate_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        profile, is_numeric, is_binary = _profile_dataframe_column(column, series)
        if is_numeric:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
        if is_binary:
            binary_candidate_columns.append(column)
        if profile["kind"] == "binary" and column not in categorical_columns and column not in numeric_columns:
            categorical_columns.append(column)
        column_profiles.append(profile)

    suggestions = suggest_columns(df)
    model_feature_candidates = _model_feature_candidate_columns_from_metadata(
        list(df.columns),
        suggested_time_columns=suggestions.get("time_columns", []),
        binary_candidate_columns=binary_candidate_columns,
    )

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
        "model_feature_candidate_count": len(model_feature_candidates),
        "suggestions": suggestions,
    }


def model_feature_candidate_columns(df: pd.DataFrame) -> list[str]:
    suggestions = suggest_columns(df)
    binary_candidate_columns = [column for column in df.columns if looks_binary(df[column])]
    return _model_feature_candidate_columns_from_metadata(
        list(df.columns),
        suggested_time_columns=suggestions.get("time_columns", []),
        binary_candidate_columns=binary_candidate_columns,
    )


@user_input_boundary
def ensure_model_feature_candidate_limit(
    df: pd.DataFrame,
    *,
    max_features: int = MAX_MODEL_FEATURE_CANDIDATES,
) -> int:
    candidate_count = len(model_feature_candidate_columns(df))
    if candidate_count > max_features:
        raise ValueError(
            f"Dataset exposes {candidate_count} model feature candidates after excluding likely survival endpoint columns. "
            f"SurvStudio supports at most {max_features} model features per dataset upload."
        )
    return candidate_count


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


def _validate_time_column_choice(df: pd.DataFrame, time_column: str) -> None:
    _require_dataframe_columns(df, [time_column])
    likely_time_columns = suggest_columns(df).get("time_columns", [])
    if likely_time_columns and time_column not in likely_time_columns:
        examples = ", ".join(likely_time_columns[:3])
        raise ValueError(
            f'"{time_column}" does not look like a survival follow-up time column. '
            f"Choose one of the likely time columns instead: {examples}."
        )


def _validate_event_column_choice(df: pd.DataFrame, event_column: str) -> None:
    _require_dataframe_columns(df, [event_column])
    series = df[event_column]
    likely_event_columns = [
        column for column in df.columns if _looks_like_event_outcome_column(column, df[column])
    ]
    if event_column in likely_event_columns:
        return

    if not looks_binary(series):
        if _is_event_like_column_name(event_column):
            return
        raise ValueError(
            f'"{event_column}" is not a binary event column. '
            "Choose a 0/1-style event column or recode it before survival analysis."
        )

    if _looks_like_baseline_status_column(event_column):
        examples = ", ".join(column for column in likely_event_columns[:3] if column != event_column)
        hint = f" Choose one of the likely event columns instead: {examples}." if examples else ""
        raise ValueError(
            f'"{event_column}" looks more like a baseline characteristic than a survival event column.{hint}'
        )

    if _has_recognizable_event_coding(series):
        return

    if _is_event_like_column_name(event_column):
        return

    examples = ", ".join(column for column in likely_event_columns[:3] if column != event_column)
    if examples:
        raise ValueError(
            f'"{event_column}" does not look like a survival event column. '
            f"Choose one of the likely event columns instead: {examples}."
        )
    raise ValueError(
        f'"{event_column}" does not look like a survival event column. '
        "Use a true event indicator or recode the dataset before survival analysis."
    )


def _cohort_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    event_positive_value: Any = None,
    extra_columns: Sequence[str] | None = None,
    drop_missing_extra_columns: bool = True,
) -> pd.DataFrame:
    if time_column == event_column:
        raise ValueError("The survival time column and event column must be different.")
    _validate_endpoint_family_pair(time_column, event_column)
    extra_columns = list(extra_columns or [])
    required_columns = [time_column, event_column, *extra_columns]
    _require_dataframe_columns(df, required_columns)
    _validate_time_column_choice(df, time_column)
    _validate_event_column_choice(df, event_column)
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


def _ordered_unique_level_strings(series: pd.Series, column_name: str | None = None) -> list[str]:
    non_missing = series.dropna()
    if non_missing.empty:
        return []
    level_strings = [str(value) for value in non_missing.unique().tolist()]
    numeric_values = pd.to_numeric(pd.Series(level_strings, dtype="string"), errors="coerce")
    if numeric_values.notna().all():
        ordered_numeric = np.sort(numeric_values.astype(float).unique())
        return [str(int(value)) if float(value).is_integer() else str(value) for value in ordered_numeric]
    return _ordered_reference_categories(level_strings, column_name or str(series.name or ""))


def _sorted_group_labels(series: pd.Series, column_name: str | None = None) -> list[str]:
    return _ordered_unique_level_strings(series.astype("string"), column_name)


def _ordered_level_strings(series: pd.Series, column_name: str | None = None) -> list[str]:
    return _ordered_unique_level_strings(series, column_name)


def _is_binary_numeric_series(series: pd.Series) -> bool:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_values.empty:
        return False
    return int(numeric_values.nunique()) == 2


def _pointwise_km_ci(survival: np.ndarray, se: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    survival = np.asarray(survival, dtype=float)
    se = np.asarray(se, dtype=float)
    z_value = float(ndtri(1 - alpha / 2))
    lower = survival.copy()
    upper = survival.copy()

    finite_mask = np.isfinite(survival)
    lower[~finite_mask] = np.nan
    upper[~finite_mask] = np.nan

    zero_mask = finite_mask & (survival <= 0.0)
    lower[zero_mask] = 0.0
    upper[zero_mask] = 0.0
    one_mask = finite_mask & (survival >= 1.0 - 1e-12)
    lower[one_mask] = np.clip(survival[one_mask], 0.0, 1.0)
    upper[one_mask] = np.clip(survival[one_mask], 0.0, 1.0)

    log_s = np.full_like(survival, np.nan, dtype=float)
    candidate_mask = finite_mask & ~zero_mask & ~one_mask
    log_s[candidate_mask] = np.log(survival[candidate_mask])
    denominator = survival * log_s
    valid_mask = (
        candidate_mask
        & (survival < 1 - 1e-12)
        & np.isfinite(se)
        & (se > 0)
        & np.isfinite(log_s)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )

    transformed = np.full_like(survival, np.nan, dtype=float)
    transformed[valid_mask] = np.log(-log_s[valid_mask])
    transformed_se = np.full_like(survival, np.nan, dtype=float)
    transformed_se[valid_mask] = np.abs(se[valid_mask] / denominator[valid_mask])
    valid_mask &= np.isfinite(transformed_se)

    low = np.exp(-np.exp(transformed[valid_mask] + z_value * transformed_se[valid_mask]))
    high = np.exp(-np.exp(transformed[valid_mask] - z_value * transformed_se[valid_mask]))
    lower[valid_mask] = np.clip(low, 0.0, 1.0)
    upper[valid_mask] = np.clip(high, 0.0, 1.0)
    return lower, upper


def _step_values(event_times: np.ndarray, survival: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(event_times, query_times, side="right") - 1
    output = np.ones_like(query_times, dtype=float)
    valid = indices >= 0
    output[valid] = survival[indices[valid]]
    return output


def _restricted_mean_survival_time(timeline: np.ndarray, survival: np.ndarray, horizon: float) -> float:
    timeline = np.asarray(timeline, dtype=float)
    survival = np.asarray(survival, dtype=float)
    if timeline.size == 0 or survival.size == 0:
        return 0.0
    horizon = float(max(horizon, 0.0))
    clipped_timeline = np.clip(timeline, 0.0, horizon)
    widths = np.maximum(np.diff(clipped_timeline), 0.0)
    area = float(np.dot(survival[:-1], widths)) if widths.size else 0.0
    tail_width = max(horizon - float(clipped_timeline[-1]), 0.0)
    if tail_width > 0.0:
        area += float(survival[-1]) * tail_width
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
    """Exponentiate with saturation, always returning a finite float."""
    exponent = float(value)
    if exponent >= _MAX_EXP_INPUT:
        return float(np.finfo(float).max)
    if exponent <= _MIN_EXP_INPUT:
        return float(np.finfo(float).tiny)
    return math.exp(exponent)


def _safe_exp_or_none(value: Any) -> float | None:
    """Exponentiate only when safely representable; otherwise return None."""
    try:
        exponent = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(exponent):
        return None
    if exponent >= _MAX_EXP_INPUT or exponent <= _MIN_EXP_INPUT:
        return None
    return math.exp(exponent)


def _hazard_ratio_effect_size(value: Any) -> float:
    hr = _safe_float(value)
    if hr is None or hr <= 0.0:
        return 0.0
    return abs(math.log(max(hr, 1e-12)))


def _cox_estimated_parameter_count(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> int:
    estimated_parameters = 0
    categorical_set = {str(column) for column in categorical_covariates}
    for column in covariates:
        if column in categorical_set:
            observed_levels = int(frame[column].dropna().nunique()) if column in frame.columns else 0
            estimated_parameters += max(observed_levels - 1, 0)
        else:
            estimated_parameters += 1
    return int(estimated_parameters)


def _summarize_labels(labels: Sequence[str], max_items: int = 3) -> str:
    cleaned = [str(label) for label in labels if label]
    if not cleaned:
        return "none"
    if len(cleaned) <= max_items:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:max_items])} +{len(cleaned) - max_items} more"


def _cox_stability_snapshot(
    frame: pd.DataFrame,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> dict[str, Any]:
    estimated_parameters = _cox_estimated_parameter_count(frame, covariates, categorical_covariates)
    event_count = int(frame[event_column].sum())
    events_per_parameter = (
        float(event_count / estimated_parameters)
        if estimated_parameters > 0
        else None
    )
    stability_warnings: list[str] = []
    risky_levels: list[dict[str, Any]] = []
    for column in categorical_covariates:
        if column not in frame.columns:
            continue
        grouped = frame.groupby(column, dropna=True, observed=False)[event_column].agg(["size", "sum"])
        categories = list(frame[column].cat.categories) if hasattr(frame[column], "cat") else []
        if categories:
            reference_level = str(categories[0])
            if reference_level in grouped.index:
                reference_row = grouped.loc[reference_level]
                reference_count = int(reference_row["size"])
                reference_events = int(reference_row["sum"])
                reference_censored = reference_count - reference_events
                if reference_count < 5:
                    risky_levels.append(
                        {
                            "column": str(column),
                            "level": reference_level,
                            "rows": reference_count,
                            "events": reference_events,
                            "censored": reference_censored,
                            "issue": "small_reference_level",
                        }
                    )
                    stability_warnings.append(
                        f'{column} uses "{reference_level}" as the Cox reference level, but only {reference_count} row'
                        f'{"s" if reference_count != 1 else ""} remain after missing-value filtering. Comparisons against this reference can look unstable.'
                    )
        one_sided_levels: list[str] = []
        for level, row in grouped.iterrows():
            total_count = int(row["size"])
            level_event_count = int(row["sum"])
            censored_count = total_count - level_event_count
            if level_event_count == 0 or censored_count == 0:
                risky_levels.append(
                    {
                        "column": str(column),
                        "level": str(level),
                        "rows": total_count,
                        "events": level_event_count,
                        "censored": censored_count,
                        "issue": "one_sided_outcome",
                    }
                )
                one_sided_levels.append(f"{level} ({total_count} rows)")
        if one_sided_levels:
            preview_levels = ", ".join(one_sided_levels[:3])
            stability_warnings.append(
                f'{column} has level(s) with only events or only censored rows after missing-value filtering: {preview_levels}'
                f'{" ..." if len(one_sided_levels) > 3 else ""}. This can produce non-finite Cox estimates.'
            )
    if events_per_parameter is not None:
        if events_per_parameter < 5:
            stability_warnings.append(
                f"Events per parameter is {events_per_parameter:.2f}, which is extremely low for Cox regression. Reduce covariates or add more events before treating the fit as stable."
            )
        elif events_per_parameter < 10:
            stability_warnings.append(
                f"Events per parameter is {events_per_parameter:.2f}, so coefficient estimates may still be unstable."
            )
    return {
        "events": event_count,
        "estimated_parameters": int(estimated_parameters),
        "events_per_parameter": events_per_parameter,
        "stability_warnings": stability_warnings,
        "risky_levels": risky_levels,
    }


def _cox_nonfinite_estimate_message(stability_snapshot: dict[str, Any]) -> str:
    details: list[str] = []
    risky_levels = list(stability_snapshot.get("risky_levels") or [])
    epv = _safe_float(stability_snapshot.get("events_per_parameter"))
    if epv is not None and epv < 10:
        details.append(f"EPV={epv:.2f}")

    small_reference_examples = [
        f'{item["column"]}="{item["level"]}" (n={int(item["rows"])})'
        for item in sorted(
            (level for level in risky_levels if level.get("issue") == "small_reference_level"),
            key=lambda level: (
                int(level.get("rows", 0)),
                str(level.get("column", "")),
                str(level.get("level", "")),
            ),
        )
    ]
    if small_reference_examples:
        details.append(
            f"sparse reference levels such as {_summarize_labels(small_reference_examples, max_items=3)}"
        )

    one_sided_examples = [
        f'{item["column"]}="{item["level"]}" (n={int(item["rows"])})'
        for item in sorted(
            (level for level in risky_levels if level.get("issue") == "one_sided_outcome"),
            key=lambda level: (
                int(level.get("rows", 0)),
                str(level.get("column", "")),
                str(level.get("level", "")),
            ),
        )
    ]
    if one_sided_examples:
        details.append(
            f'levels with only events or only censored rows such as {_summarize_labels(one_sided_examples, max_items=3)}'
        )

    detail_text = f" Problem signals in the analyzable cohort: {'; '.join(details)}." if details else ""
    return (
        "Cox PH fit produced non-finite estimates. This usually means redundant covariates, sparse categories, "
        "or quasi-complete separation. Remove overlapping variables or collapse sparse levels."
        f"{detail_text}"
    )


def _cox_fit_failure_message(exc: Exception, stability_snapshot: dict[str, Any]) -> str:
    raw = str(exc).strip()
    lowered = raw.lower()
    if "singular matrix" in lowered:
        detail = _cox_nonfinite_estimate_message(stability_snapshot)
        return (
            "Cox PH fit failed because the design matrix is singular. This usually means redundant covariates, "
            "overlapping encodings of the same signal, or sparse categorical levels. "
            "Remove one of the overlapping variables or collapse sparse levels. "
            f"{detail}"
        )
    return _cox_nonfinite_estimate_message(stability_snapshot)


def _cox_categorical_stability_alerts(
    frame: pd.DataFrame,
    categorical_covariates: Sequence[str],
    *,
    min_level_n: int = 5,
) -> dict[str, list[str]]:
    reference_alerts: list[str] = []
    sparse_level_alerts: list[str] = []
    for column in categorical_covariates:
        if column not in frame.columns:
            continue
        series = frame[column]
        counts = series.value_counts(dropna=True)
        if counts.empty:
            continue
        if hasattr(series, "cat") and len(series.cat.categories) > 0:
            reference_level = str(series.cat.categories[0])
        else:
            reference_level = str(counts.index[0])
        reference_count = int(counts.get(reference_level, 0))
        if 0 < reference_count < min_level_n:
            reference_alerts.append(f'{column} reference "{reference_level}" (n={reference_count})')
        for level, count in counts.items():
            if 0 < int(count) < min_level_n:
                sparse_level_alerts.append(f"{column}={level} (n={int(count)})")
    return {
        "reference_levels": reference_alerts,
        "sparse_levels": sparse_level_alerts,
    }


def _cox_wide_ci_alert_terms(
    model_rows: Sequence[dict[str, Any]],
    *,
    width_ratio_threshold: float = 8.0,
) -> list[str]:
    wide_terms: list[str] = []
    for row in model_rows:
        ci_low = _safe_float(row.get("CI lower"))
        ci_high = _safe_float(row.get("CI upper"))
        if ci_low is None or ci_high is None or ci_low <= 0:
            continue
        if (ci_high / ci_low) >= width_ratio_threshold:
            wide_terms.append(str(row.get("Label", row.get("Variable", ""))))
    return wide_terms


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
    outcome_informed_group: bool = False,
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

    if group_column and not outcome_informed_group:
        test_name = _weighted_test_label(test_payload["test"], fh_p=fh_p) if test_payload else "weighted"
        strengths.append(f"Global {test_name} comparison was run across {group_count} groups.")
        if pairwise_rows:
            strengths.append("Pairwise group comparisons include Benjamini-Hochberg adjusted p-values.")
    elif group_column and outcome_informed_group:
        strengths.append("Outcome-informed groups were visualized descriptively without a fresh between-group hypothesis test.")
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

    if group_column and outcome_informed_group:
        headline = "Outcome-informed groups were visualized descriptively; a fresh log-rank p-value is not reported on the same selected split."
        cautions.append("This grouping was derived using outcome information, so treat the KM figure as exploratory rather than confirmatory.")
        next_steps.append("Report the selection procedure and any selection-adjusted p-value from the cutpoint or signature workflow instead of a fresh raw log-rank test.")
    elif group_column and test_payload:
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
    *,
    categorical_alerts: dict[str, list[str]] | None = None,
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
    categorical_alerts = categorical_alerts or {"reference_levels": [], "sparse_levels": []}
    reference_alerts = [str(item) for item in categorical_alerts.get("reference_levels", []) if item]
    sparse_level_alerts = [str(item) for item in categorical_alerts.get("sparse_levels", []) if item]
    wide_ci_terms = _cox_wide_ci_alert_terms(model_rows)
    non_estimable_terms = [
        str(row.get("Label", row.get("Variable", "")))
        for row in model_rows
        if row.get("Hazard ratio") is None or row.get("CI lower") is None or row.get("CI upper") is None
    ]

    complete_case_n = int(model_stats["n"])
    outcome_rows_raw = model_stats.get("outcome_rows")
    dropped_rows_raw = model_stats.get("dropped_rows")
    outcome_rows = int(outcome_rows_raw) if outcome_rows_raw is not None else None
    dropped_rows = int(dropped_rows_raw) if dropped_rows_raw is not None else None
    dropped_fraction = None
    if outcome_rows is not None and outcome_rows > 0 and dropped_rows is not None:
        dropped_fraction = float(dropped_rows / outcome_rows)

    cohort_statement = f"Model estimates use the analyzable cohort after dropping rows with missing selected covariates (N = {complete_case_n})."
    if outcome_rows is not None:
        cohort_statement = (
            "Model estimates use the analyzable cohort after dropping rows with missing selected covariates "
            f"(N = {complete_case_n} of {int(outcome_rows)} outcome-valid rows)."
        )

    strengths = [
        "Cox regression was fit with the Efron tie method.",
        cohort_statement,
        "Proportional-hazards screening used rank-based Spearman correlations between Schoenfeld residuals and log time; this is approximate and not a formal Grambsch-Therneau test.",
        "The reported discrimination metric is an apparent C-index on the fitted cohort, so it reflects training-cohort ranking only.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []

    epv = _safe_float(model_stats.get("events_per_parameter"))
    c_index = _safe_float(model_stats.get("c_index"))
    if epv is not None and epv < 10:
        cautions.append("Events per parameter is below 10, so coefficients may be unstable or overfit.")
        next_steps.append("Reduce model complexity or increase the event count before treating estimates as final.")
    cautions.append("Changing the covariate set can change the analyzable cohort because Cox fitting uses complete-case rows for the selected covariates.")
    if dropped_rows:
        drop_message = f"{int(dropped_rows)} outcome-valid rows were excluded because at least one selected covariate was missing."
        if dropped_fraction is not None:
            drop_message = (
                f"{int(dropped_rows)} outcome-valid rows ({dropped_fraction:.1%}) were excluded "
                "because at least one selected covariate was missing."
            )
        cautions.append(drop_message)
        next_steps.append("Review missingness patterns or use an imputation strategy before treating the fitted cohort as representative.")
    if ph_alert_terms:
        cautions.append(
            f"Possible proportional-hazards violations detected for: {', '.join(ph_alert_terms)}."
        )
        next_steps.append("Consider stratification or time-varying effects for PH-violating terms.")
    if reference_alerts:
        cautions.append(
            f"Some Cox reference levels are very small after missing-value filtering: {_summarize_labels(reference_alerts, max_items=4)}."
        )
        next_steps.append("Use a more stable reference level or collapse sparse categories before interpreting reference-based contrasts.")
    if sparse_level_alerts:
        cautions.append(
            f"Sparse categorical levels remain in the analyzable cohort: {_summarize_labels(sparse_level_alerts, max_items=4)}."
        )
        next_steps.append("Collapse rare categorical levels before treating term-specific hazard ratios as stable.")
    if wide_ci_terms:
        cautions.append(
            f"Some hazard-ratio intervals are very wide, which suggests unstable estimates: {_summarize_labels(wide_ci_terms, max_items=4)}."
        )
        next_steps.append("Treat wide-interval terms as unstable unless the category encoding or cohort size is improved.")
    if non_estimable_terms:
        cautions.append(
            f"Some Cox contrasts produced non-estimable hazard ratios or confidence intervals: {_summarize_labels(non_estimable_terms, max_items=4)}."
        )
        next_steps.append("Collapse sparse categories or remove quasi-separated terms before interpreting those contrasts.")
    if c_index is not None and c_index < 0.6:
        cautions.append("Apparent model discrimination is modest (C-index below 0.60).")
    cautions.append("The Cox C-index is apparent, so it is optimistic and should not be treated as external validation.")
    if c_index is not None:
        strengths.append(
            f"A C-index of {c_index:.3f} means the fitted model ranks about {c_index * 100:.1f}% of comparable patient pairs in the observed risk order."
        )
    cautions.append(
        "The current dashboard does not yet provide a built-in external-cohort apply workflow for Cox validation; validate the final specification on a separate cohort outside this run."
    )
    if not significant_terms:
        cautions.append("No model term shows clear nominal evidence at p < 0.05.")

    structural_instability = bool(
        (epv is not None and epv < 10)
        or reference_alerts
        or sparse_level_alerts
        or wide_ci_terms
        or non_estimable_terms
    )

    if significant_terms:
        if structural_instability:
            headline = (
                f"Model fit shows {len(significant_terms)} term(s) with nominal hazard association, "
                f"but some estimates appear unstable: {_summarize_labels(significant_terms)}."
            )
        elif ph_alert_terms:
            headline = (
                f"Model fit shows {len(significant_terms)} term(s) with nominal hazard association, "
                f"but some terms need closer proportional-hazards review: {_summarize_labels(ph_alert_terms)}."
            )
        else:
            headline = (
                f"Model fit identified {len(significant_terms)} term(s) with nominal hazard association: "
                f"{_summarize_labels(significant_terms)}."
            )
        next_steps.append("Interpret hazard ratios together with confidence intervals, not p-values alone.")
    else:
        if structural_instability:
            headline = (
                "Model fit completed, but no term shows clear nominal hazard association and some estimates remain unstable under the current specification."
            )
        elif ph_alert_terms:
            headline = (
                "Model fit completed, but no term shows clear nominal hazard association and some terms still need closer proportional-hazards review under the current specification."
            )
        else:
            headline = "Model fit completed, but no term shows clear nominal hazard association under the current specification."
        next_steps.append("Revisit covariate selection, encoding, and cohort size before forcing interpretation.")

    status = "robust"
    if cautions:
        status = "review"
    if (epv is not None and epv < 5) or len(ph_alert_terms) >= 2 or reference_alerts or sparse_level_alerts:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Outcome-valid rows", "value": outcome_rows},
            {"label": "Dropped for missing covariates", "value": dropped_rows},
            {"label": "Events", "value": int(model_stats["events"])},
            {"label": "Parameters", "value": int(model_stats["parameters"])},
            {"label": "EPV", "value": epv},
            {"label": "Apparent C-index (training cohort)", "value": c_index},
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
    cautions.append(
        "Stability score is a composite heuristic that mixes significance, effect size, replication support, and parsimony; its weights are expert-set and not independently validated."
    )

    if search_space["permutation_iterations"] > 0:
        strengths.append("Permutation-based empirical filtering was enabled for top-ranked candidates.")
    if search_space["validation_iterations"] > 0:
        strengths.append("Repeated subsample holdout checks were enabled for top-ranked candidates.")

    support = _safe_float(best_split.get("Bootstrap support (p<alpha)"))
    direction_consistency = _safe_float(best_split.get("Bootstrap HR direction consistency"))
    validation_support = _safe_float(best_split.get("Validation support (p<alpha)"))
    permutation_p = _safe_float(best_split.get("Permutation p"))
    signature_n = int(best_split["N signature+"])
    is_significant = bool(best_split["Statistically significant"])
    alpha = float(search_space["significance_level"])

    if search_space["truncated"]:
        cautions.append("Search space hit the internal combination cap, so discovery was not exhaustive.")
        cautions.append("Adjusted p-values only account for the tested subset of combinations under the cap.")
        cautions.append(
            "When the cap is hit, the retained rules depend on deterministic candidate order, so earlier input columns can be overrepresented."
        )
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


def _format_percent_value(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _parse_percentile_values(cutoff: str | float | None, *, mode: str) -> list[float]:
    if cutoff is None:
        raise ValueError("Enter percentile value(s) before creating a grouped column.")
    if isinstance(cutoff, (int, float, np.integer, np.floating)):
        tokens = [str(float(cutoff))]
    else:
        tokens = [token.strip() for token in str(cutoff).split(",") if token.strip()]
    if not tokens:
        raise ValueError("Enter percentile value(s) before creating a grouped column.")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Percentile values must be numeric, for example 25 or 25,25.") from exc
    if any((not math.isfinite(value)) or value <= 0 or value >= 100 for value in values):
        raise ValueError("Percentile values must be greater than 0 and less than 100.")
    if mode == "percentile_split":
        if len(values) not in {1, 2}:
            raise ValueError("Percentile split accepts one value (25) or two values (25,25).")
        if len(values) == 2 and sum(values) >= 100:
            raise ValueError("Two percentile values must sum to less than 100 so a middle group remains.")
    elif mode == "extreme_split":
        if len(values) != 1:
            raise ValueError("Extreme split accepts one value, for example 25.")
        if values[0] >= 50:
            raise ValueError("Extreme split percentile must be less than 50 so a middle range remains excluded.")
    return values


def _append_realized_group_share_note(
    summary: dict[str, Any],
    counts: Sequence[dict[str, Any]],
) -> None:
    non_missing = [row for row in counts if str(row.get("group")) != "Missing"]
    total = sum(int(row.get("n", 0) or 0) for row in non_missing)
    if total <= 0:
        return
    realized = [
        {
            "group": str(row.get("group")),
            "n": int(row.get("n", 0) or 0),
            "fraction": float((int(row.get("n", 0) or 0) / total) * 100.0),
        }
        for row in non_missing
    ]
    summary["realized_group_shares"] = realized
    realized_text = ", ".join(
        f"{item['group']} = {item['fraction']:.1f}%"
        for item in realized
    )
    summary["assignment_rule"] = (
        f"{summary.get('assignment_rule', '')} "
        f"Realized non-missing shares: {realized_text}. Ties at the threshold can shift these from the nominal target."
    ).strip()


def _percentile_threshold_label(percentile: float, direction: str) -> str:
    value = _format_percent_value(percentile)
    if direction == "above":
        return f"At/above {value}th percentile threshold"
    if direction == "below":
        return f"At/below {value}th percentile threshold"
    if direction == "strict_above":
        return f"Above {value}th percentile threshold"
    if direction == "strict_below":
        return f"Below {value}th percentile threshold"
    raise ValueError(f"Unsupported percentile-threshold direction: {direction}")


@user_input_boundary
def derive_group_column(
    df: pd.DataFrame,
    source_column: str,
    method: str,
    new_column_name: str | None = None,
    cutoff: str | float | None = None,
    lower_label: str = "Low",
    upper_label: str = "High",
    time_column: str | None = None,
    event_column: str | None = None,
    event_positive_value: Any = None,
    min_group_fraction: float = 0.1,
    permutation_iterations: int = 500,
    random_seed: int = 20260311,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    _require_dataframe_columns(df, [source_column])
    if source_column in _survival_outcome_like_columns(df):
        raise ValueError(
            f'"{source_column}" looks like a survival endpoint column. '
            "Do not derive groups from survival time or event indicators."
        )
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
            min_group_fraction=min_group_fraction,
            lower_label=lower_label,
            upper_label=upper_label,
            permutation_iterations=permutation_iterations,
            random_seed=random_seed,
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
            "min_group_fraction": float(min_group_fraction),
            "permutation_iterations": int(permutation_iterations),
            "random_seed": int(random_seed),
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
    elif method == "percentile_split":
        percentiles = _parse_percentile_values(cutoff, mode=method)
        cutoff_spec = ",".join(_format_percent_value(value) for value in percentiles)
        if len(percentiles) == 1:
            top_percent = percentiles[0]
            quantile = 1 - (top_percent / 100.0)
            split_point = float(usable.quantile(quantile))
            threshold_percent = 100.0 - top_percent
            rest_label = "Rest"
            # Match median_split at 50th percentile so ties at the threshold stay in the lower/rest group.
            if math.isclose(top_percent, 50.0):
                top_label = _percentile_threshold_label(threshold_percent, "strict_above")
                rest_label = _percentile_threshold_label(threshold_percent, "below")
                labels = np.where(numeric_series <= split_point, rest_label, top_label)
                assignment_rule = (
                    f"{source_column} > percentile threshold ({split_point:.3f}) -> {top_label}, else -> {rest_label}"
                )
            else:
                top_label = _percentile_threshold_label(threshold_percent, "above")
                labels = np.where(numeric_series >= split_point, top_label, rest_label)
                assignment_rule = (
                    f"{source_column} >= percentile threshold ({split_point:.3f}) -> {top_label}, else -> {rest_label}"
                )
            summary = {
                "method": method,
                "cutoff_spec": cutoff_spec,
                "percentiles": percentiles,
                "cutoffs": [split_point],
                "n_groups": 2,
                "assignment_rule": assignment_rule,
            }
        else:
            bottom_percent, top_percent = percentiles
            middle_percent = 100.0 - bottom_percent - top_percent
            low_threshold = float(usable.quantile(bottom_percent / 100.0))
            high_threshold = float(usable.quantile(1 - (top_percent / 100.0)))
            if not low_threshold < high_threshold:
                raise ValueError("Percentile split thresholds overlap. Choose less aggressive percentiles or a variable with more distinct values.")
            bottom_label = _percentile_threshold_label(bottom_percent, "below")
            middle_label = "Between percentile thresholds"
            top_label = _percentile_threshold_label(100.0 - top_percent, "above")
            labels = np.where(
                numeric_series <= low_threshold,
                bottom_label,
                np.where(numeric_series >= high_threshold, top_label, middle_label),
            )
            summary = {
                "method": method,
                "cutoff_spec": cutoff_spec,
                "percentiles": percentiles,
                "cutoffs": [low_threshold, high_threshold],
                "n_groups": 3,
                "assignment_rule": (
                    f"{source_column} <= lower percentile threshold ({low_threshold:.3f}) -> {bottom_label}; "
                    f"{source_column} >= upper percentile threshold ({high_threshold:.3f}) -> {top_label}; "
                    f"else -> {middle_label}"
                ),
            }
    elif method == "extreme_split":
        percentiles = _parse_percentile_values(cutoff, mode=method)
        tail_percent = percentiles[0]
        cutoff_spec = _format_percent_value(tail_percent)
        low_threshold = float(usable.quantile(tail_percent / 100.0))
        high_threshold = float(usable.quantile(1 - (tail_percent / 100.0)))
        if not low_threshold < high_threshold:
            raise ValueError("Extreme split thresholds overlap. Choose a smaller percentile or a variable with more distinct values.")
        bottom_label = _percentile_threshold_label(tail_percent, "below")
        top_label = _percentile_threshold_label(100.0 - tail_percent, "above")
        labels = np.full(len(numeric_series), pd.NA, dtype=object)
        low_mask = (numeric_series <= low_threshold).fillna(False).to_numpy()
        high_mask = (numeric_series >= high_threshold).fillna(False).to_numpy()
        labels[low_mask] = bottom_label
        labels[high_mask] = top_label
        excluded_middle_count = int((numeric_series.notna() & ~pd.Series(low_mask | high_mask, index=numeric_series.index)).sum())
        summary = {
            "method": method,
            "cutoff_spec": cutoff_spec,
            "percentiles": percentiles,
            "cutoffs": [low_threshold, high_threshold],
            "n_groups": 2,
            "excluded_count": excluded_middle_count,
            "assignment_rule": (
                f"{source_column} <= lower percentile threshold ({low_threshold:.3f}) -> {bottom_label}; "
                f"{source_column} >= upper percentile threshold ({high_threshold:.3f}) -> {top_label}; "
                "else -> excluded middle range"
            ),
        }
    else:
        raise ValueError(f"Unsupported derive-group method: {method}")

    label_series = pd.Series(labels, index=df.index, dtype="string")
    label_series.loc[numeric_series.isna()] = pd.NA
    observed_groups = int(label_series.dropna().nunique())
    expected_groups = summary.get("n_groups")
    if method == "percentile_split" and expected_groups == 3 and observed_groups < 3:
        raise ValueError("Percentile split did not produce three distinct groups. Choose a less aggressive percentile setting or another variable.")
    if method in {"percentile_split", "extreme_split"} and observed_groups < 2:
        raise ValueError("Selected percentile thresholds did not produce at least two non-empty groups.")

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
    if method in {"percentile_split", "extreme_split"}:
        _append_realized_group_share_note(summary, counts)
    summary["recipe"] = {
        "source_column": source_column,
        "column_name": column_name,
        "method": method,
        "cutoff": summary.get("cutoff"),
        "cutoff_spec": summary.get("cutoff_spec"),
        "cutoffs": list(summary.get("cutoffs", [])),
        "percentiles": list(summary.get("percentiles", [])),
        "lower_label": lower_label,
        "upper_label": upper_label,
        "time_column": time_column,
        "event_column": event_column,
        "event_positive_value": event_positive_value,
        "min_group_fraction": summary.get("min_group_fraction"),
        "permutation_iterations": summary.get("permutation_iterations"),
        "random_seed": summary.get("random_seed"),
        "outcome_informed": outcome_informed,
    }
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


def _signature_cox_metrics(
    times: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    cox_frame = pd.DataFrame(
        {
            "__time": np.asarray(times, dtype=float),
            "__event": np.asarray(events, dtype=int),
            "__signature": np.asarray(mask, dtype=bool).astype(float),
        }
    )
    cox_model = PHReg.from_formula(
        'Q("__time") ~ Q("__signature")',
        data=cox_frame,
        status=cox_frame["__event"],
        ties="efron",
    )
    cox_results = cox_model.fit(disp=False)
    conf_int = np.asarray(cox_results.conf_int(), dtype=float)
    return {
        "Hazard ratio (signature+ vs -)": _safe_exp_or_none(cox_results.params[0]),
        "HR CI lower": _safe_exp_or_none(conf_int[0, 0]),
        "HR CI upper": _safe_exp_or_none(conf_int[0, 1]),
    }


def _stability_score(row: dict[str, Any]) -> float:
    # Composite score balancing significance, robustness, effect size, and parsimony.
    bh_p = max(float(row["BH adjusted p"]), 1e-12)
    significance_score = min(-math.log10(bh_p), 10.0)
    effect = _hazard_ratio_effect_size(row.get("Hazard ratio (signature+ vs -)"))
    support = row["Bootstrap support (p<alpha)"]
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
        significance_score
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
    significance_level: float,
) -> dict[str, float | int | None]:
    if n_iterations <= 0:
        return {
            "Bootstrap support (p<alpha)": None,
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
        hr = _safe_exp_or_none(cox_results.params[0])
        if hr is not None:
            hazard_ratios.append(hr)
        if p_float < significance_level:
            significant_count += 1

    if valid_resamples == 0:
        return {
            "Bootstrap support (p<alpha)": None,
            "Bootstrap median HR": None,
            "Bootstrap median p": None,
            "Bootstrap HR direction consistency": None,
            "Bootstrap valid resamples": 0,
        }

    direction_consistency = None
    if hazard_ratios:
        hr_array = np.asarray(hazard_ratios, dtype=float)
        direction_consistency = float(
            max(float(np.mean(hr_array >= 1.0)), float(np.mean(hr_array < 1.0)))
        )

    return {
        "Bootstrap support (p<alpha)": float(significant_count / valid_resamples),
        "Bootstrap median HR": float(np.median(hazard_ratios)) if hazard_ratios else None,
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

        ci_low = _safe_exp_or_none(conf_int[0, 0])
        ci_high = _safe_exp_or_none(conf_int[0, 1])
        valid_folds += 1
        p_float = float(p_value)
        p_values.append(p_float)
        hr = _safe_exp_or_none(cox_results.params[0])
        if hr is not None:
            hazard_ratios.append(hr)
        if p_float <= significance_level and ci_low is not None and ci_high is not None and not (ci_low <= 1.0 <= ci_high):
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
        "Validation median HR": float(np.median(hazard_ratios)) if hazard_ratios else None,
        "Validation median p": float(np.median(p_values)),
        "Validation valid folds": int(valid_folds),
    }


@user_input_boundary
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
                        "Hazard ratio (signature+ vs -)": None,
                        "HR CI lower": None,
                        "HR CI upper": None,
                        "Median signature+": median_high,
                        "Median signature-": median_low,
                        "Bootstrap support (p<alpha)": None,
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
    for row, adj in zip(rows, adjusted, strict=True):
        row["BH adjusted p"] = adj

    primary_ranked_idx = sorted(
        range(len(rows)),
        key=lambda idx: (
            rows[idx]["BH adjusted p"],
            rows[idx]["P value"],
            -float(rows[idx]["Chi-square"]),
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
    times = frame[time_column].to_numpy(dtype=float)
    events = frame[event_column].to_numpy(dtype=int)

    for idx in sorted(metric_candidate_idx):
        combo = valid_combinations[idx]
        mask = _signature_mask(frame, combo["combo"], operator=combo["operator"])
        try:
            rows[idx].update(_signature_cox_metrics(times, events, mask))
        except Exception:
            continue

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
                significance_level=significance_level,
            )
            rows[idx].update(bootstrap_metrics)

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
    best_row = rows[best_idx]
    signature_recipe = {
        "column_name": best_column_name,
        "operator": str(best_operator),
        "features": list(best_row.get("Features", [])),
        "signature": str(best_row.get("Signature", "")),
        "positive_label": "Signature+",
        "negative_label": "Signature-",
        "statistically_significant": bool(best_row.get("Statistically significant")),
        "outcome_informed": True,
        "random_seed": int(random_seed),
    }

    payload = {
        "results_table": ranked_rows,
        "best_split": best_row,
        "search_space": search_space,
        "signature_recipe": signature_recipe,
        "derived_group": {
            "column_name": best_column_name,
            "counts": counts,
            "outcome_informed": True,
            "auto_apply_recommended": bool(best_row.get("Statistically significant")),
            "recipe": signature_recipe,
        },
        "scientific_summary": scientific_summary,
    }
    return output_df, best_column_name, payload


@user_input_boundary
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
    suppress_group_inference: bool = False,
    outcome_informed_group: bool = False,
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
        group_labels = _sorted_group_labels(frame[group_column], group_column)
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

        if step_timeline[-1] < display_horizon:
            step_timeline = np.concatenate((step_timeline, [display_horizon]))
            step_survival = np.concatenate((step_survival, [step_survival[-1]]))
            lower = np.concatenate((lower, [lower[-1]]))
            upper = np.concatenate((upper, [upper[-1]]))

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
    if group_column and len(group_labels) >= 2 and not suppress_group_inference:
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
        for row, adjusted_p in zip(pairwise_rows, adjusted, strict=True):
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
        outcome_informed_group=outcome_informed_group,
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
        "outcome_informed_group": bool(outcome_informed_group),
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
    *,
    drop_missing_covariates: bool = True,
) -> pd.DataFrame:
    required_columns = [*covariates]
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=required_columns,
        drop_missing_extra_columns=drop_missing_covariates,
    )
    for column in covariates:
        if column in categorical_covariates:
            string_values = frame[column].astype("string")
            categories = _ordered_reference_categories(string_values.dropna().unique().tolist(), column)
            frame[column] = pd.Categorical(string_values, categories=categories)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if drop_missing_covariates:
        frame = frame.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after removing missing values for the Cox model.")
    return frame


def _validate_cox_covariates(covariates: Sequence[str]) -> None:
    covariate_set = {str(column) for column in covariates}
    overlapping_stage_sets = [
        {"stage", "stage_group"},
        {"pathologic_stage", "stage_group"},
        {"stage", "pathologic_stage"},
    ]
    for overlap in overlapping_stage_sets:
        if overlap <= covariate_set:
            joined = ", ".join(sorted(overlap))
            raise ValueError(
                f"Cox PH cannot fit overlapping stage representations together ({joined}). "
                "Select one stage variable to avoid redundant encoding."
            )


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
    if int(np.sum(event_mask)) == 0:
        return None

    order = np.argsort(time_values, kind="mergesort")
    unique_risks, inverse = np.unique(risk_score.astype(float), return_inverse=True)
    ranks = inverse + 1  # Fenwick tree is 1-indexed.
    tree = np.zeros(len(unique_risks) + 1, dtype=np.int64)

    def _fenwick_add(index: int, delta: int) -> None:
        while index < tree.size:
            tree[index] += delta
            index += index & -index

    def _fenwick_prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    concordant = 0.0
    comparable = 0.0
    later_count = 0
    cursor = len(order) - 1

    while cursor >= 0:
        time_value = time_values[order[cursor]]
        group_end = cursor
        while cursor >= 0 and time_values[order[cursor]] == time_value:
            cursor -= 1
        group_indices = order[cursor + 1:group_end + 1]

        if later_count:
            for idx in group_indices:
                if event_values[idx] != 1:
                    continue
                rank = int(ranks[idx])
                lower = _fenwick_prefix(rank - 1)
                equal = _fenwick_prefix(rank) - lower
                concordant += float(lower) + 0.5 * float(equal)
                comparable += float(later_count)

        for idx in group_indices:
            _fenwick_add(int(ranks[idx]), 1)
            later_count += 1

    if comparable <= 0.0:
        return None
    return concordant / comparable


@user_input_boundary
def compute_cox_analysis(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> dict[str, Any]:
    _validate_cox_covariates(covariates)
    categorical_covariates = list(dict.fromkeys(categorical_covariates or _categorical_candidates(df, covariates)))
    preview_frame = _prepare_cox_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        covariates=covariates,
        categorical_covariates=categorical_covariates,
        event_positive_value=event_positive_value,
        drop_missing_covariates=False,
    )
    frame = preview_frame.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after removing missing values for the Cox model.")
    for column in categorical_covariates:
        string_values = frame[column].astype("string")
        categories = _ordered_reference_categories(string_values.dropna().unique().tolist(), column)
        frame[column] = pd.Categorical(string_values, categories=categories)
    formula = _build_cox_formula(time_column, covariates, categorical_covariates)
    status = frame[event_column].astype(int).to_numpy()
    stability_snapshot = _cox_stability_snapshot(frame, event_column, covariates, categorical_covariates)
    model = PHReg.from_formula(formula, data=frame, status=status, ties="efron")
    try:
        results = model.fit(disp=False)
    except Exception as exc:
        raise ValueError(_cox_fit_failure_message(exc, stability_snapshot)) from exc

    conf_int = np.asarray(results.conf_int(), dtype=float)
    param_vector = np.asarray(results.params, dtype=float)
    bse_vector = np.asarray(results.bse, dtype=float)
    z_vector = np.asarray(results.tvalues, dtype=float)
    p_vector = np.asarray(results.pvalues, dtype=float)
    llf_value = float(results.llf) if results.llf is not None else np.nan

    reference_levels = _reference_levels(frame, categorical_covariates)
    risk_score = np.asarray(results.model.exog @ results.params, dtype=float)
    fit_components = [param_vector, conf_int.reshape(-1), bse_vector, z_vector, p_vector, risk_score]
    if (not np.isfinite(llf_value)) or any(not np.isfinite(component).all() for component in fit_components):
        raise ValueError(_cox_nonfinite_estimate_message(stability_snapshot))

    model_rows: list[dict[str, Any]] = []
    for idx, term in enumerate(results.model.exog_names):
        variable, label, reference = _clean_term(term, reference_levels)
        beta = float(param_vector[idx])
        hr = _safe_exp_or_none(beta)
        ci_low = _safe_exp_or_none(conf_int[idx, 0])
        ci_high = _safe_exp_or_none(conf_int[idx, 1])
        model_rows.append(
            {
                "Variable": variable,
                "Label": label,
                "Reference": reference,
                "Beta": beta,
                "Hazard ratio": hr,
                "CI lower": ci_low,
                "CI upper": ci_high,
                "SE": float(bse_vector[idx]),
                "Z": float(z_vector[idx]),
                "P value": float(p_vector[idx]),
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
    outcome_rows = int(preview_frame.shape[0])
    dropped_rows = int(outcome_rows - n_obs)
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
            "outcome_rows": outcome_rows,
            "dropped_rows": dropped_rows,
            "events": n_events,
            "parameters": k_params,
            "events_per_parameter": float(n_events / k_params) if k_params else None,
            "partial_log_likelihood": _safe_float(llf_value),
            "aic": _safe_float(-2 * llf_value + 2 * k_params),
            "bic": _safe_float(-2 * llf_value + k_params * np.log(max(n_obs, 1))),
            "c_index": _safe_float(c_index),
            "tie_method": "efron",
        },
        categorical_alerts=_cox_categorical_stability_alerts(frame, categorical_covariates),
    )

    return {
        "formula": formula,
        "results_table": model_rows,
        "diagnostics_table": diagnostic_rows,
        "model_stats": {
            "n": n_obs,
            "outcome_rows": outcome_rows,
            "dropped_rows": dropped_rows,
            "events": n_events,
            "parameters": k_params,
            "events_per_parameter": float(n_events / k_params) if k_params else None,
            "partial_log_likelihood": _safe_float(llf_value),
            "aic": _safe_float(-2 * llf_value + 2 * k_params),
            "bic": _safe_float(-2 * llf_value + k_params * np.log(max(n_obs, 1))),
            "c_index": _safe_float(c_index),
            "apparent_c_index": _safe_float(c_index),
            "c_index_label": "Apparent C-index (training cohort)",
            "evaluation_mode": "apparent",
            "tie_method": "efron",
        },
        "categorical_covariates": categorical_covariates,
        "scientific_summary": scientific_summary,
    }


@user_input_boundary
def preview_cox_analysis_inputs(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> dict[str, Any]:
    if not covariates:
        raise ValueError("Select at least one covariate for the Cox model.")
    _validate_cox_covariates(covariates)
    categorical_covariates = list(dict.fromkeys(categorical_covariates or _categorical_candidates(df, covariates)))
    preview_frame = _prepare_cox_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        covariates=covariates,
        categorical_covariates=categorical_covariates,
        event_positive_value=event_positive_value,
        drop_missing_covariates=False,
    )
    missing_by_covariate: list[dict[str, Any]] = []
    for column in covariates:
        missing_count = int(preview_frame[column].isna().sum())
        if missing_count > 0:
            missing_by_covariate.append({"column": column, "missing_rows": missing_count})
    complete_case = preview_frame.dropna(subset=list(covariates)).reset_index(drop=True)
    if complete_case.empty:
        raise ValueError("No rows remain after removing missing values for the Cox model.")
    stability_snapshot = _cox_stability_snapshot(complete_case, event_column, covariates, categorical_covariates)
    missing_by_covariate.sort(key=lambda item: (-int(item["missing_rows"]), str(item["column"])))
    return {
        "outcome_rows": int(preview_frame.shape[0]),
        "analyzable_rows": int(complete_case.shape[0]),
        "dropped_rows": int(preview_frame.shape[0] - complete_case.shape[0]),
        "events": int(stability_snapshot["events"]),
        "estimated_parameters": int(stability_snapshot["estimated_parameters"]),
        "events_per_parameter": stability_snapshot["events_per_parameter"],
        "covariates": list(covariates),
        "categorical_covariates": list(categorical_covariates),
        "missing_by_covariate": missing_by_covariate,
        "stability_warnings": list(stability_snapshot["stability_warnings"]),
        "risky_levels": list(stability_snapshot["risky_levels"]),
    }


@user_input_boundary
def compute_cohort_table(df: pd.DataFrame, variables: Sequence[str], group_column: str | None = None) -> dict[str, Any]:
    variables = [variable for variable in variables if variable != group_column]
    if not variables:
        raise ValueError("Select at least one variable for the cohort summary table.")
    overall_label = "Overall (grouped subset)" if group_column else "Overall"
    columns = [*variables]
    if group_column:
        columns.append(group_column)
    _require_dataframe_columns(df, columns)
    frame = df[columns].copy()

    group_frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
    if group_column:
        string_group = frame[group_column].astype("string")
        frame = frame.loc[string_group.notna()].copy()
        string_group = frame[group_column].astype("string")
        group_frames[overall_label] = frame
        group_labels = _sorted_group_labels(string_group, group_column)
        for label in group_labels:
            group_frames[label] = frame.loc[string_group == label]
    else:
        group_frames[overall_label] = frame
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
        levels = _ordered_level_strings(source_series, variable)
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
        "columns": ["Variable", "Statistic", overall_label, *group_labels],
        "rows": rows,
        "overall_scope": (
            "Overall summarizes the non-missing grouped subset."
            if group_column
            else "Overall summarizes the full analyzable table cohort."
        ),
    }

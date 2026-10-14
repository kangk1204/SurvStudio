"""Input safeguards and transparent preprocessing notes for predictive models."""
from __future__ import annotations

from functools import wraps
from contextvars import ContextVar
import inspect
from typing import Any

import numpy as np
import pandas as pd

GUARD_VERSION = "prediction-input/3"
_PROXY_CONTEXT: ContextVar[dict | None] = ContextVar("survstudio_proxy_context", default=None)
AVAILABILITY_NOTE = (
    "Use only variables available at the time you intend to predict risk. "
    "Remove measurements collected after the outcome or derived from follow-up. "
    "Automatic checks cannot identify every hidden outcome proxy."
)


def check_training_outcome_proxies(frame, time_column, event_column, features, categorical_features=()):
    """Called at the actual training boundary, before any survival model is fitted."""
    from survival_toolkit.errors import OutcomeProxyError, must_propagate
    from survival_toolkit.outcome_proxy import screen_training_outcomes, training_fingerprint, PROXY_VERSION

    context = _PROXY_CONTEXT.get()
    key = training_fingerprint(frame[[time_column, event_column, *features]], categorical_features or ())
    cache = context["partitions"] if context is not None else {}
    if key not in cache:
        try:
            cache[key] = screen_training_outcomes(frame, time_column, event_column, features, categorical_features or ())
        except Exception as exc:
            if must_propagate(exc):
                raise
            audit = {"version": PROXY_VERSION, "status": "screen_failed", "training_hash": key,
                     "training_rows": int(len(frame)), "scope": "training_partition_only", "findings": [],
                     "failure_type": type(exc).__name__,
                     "notes": ["Outcome pattern screening could not finish. Model evaluation was stopped; no completed result is reported."]}
            raise OutcomeProxyError(audit["notes"][0], audit) from exc
    audit = cache[key]
    if audit["findings"]:
        reason = context.get("review") if context is not None else None
        if not reason:
            finding = audit["findings"][0]
            labels = ", ".join(finding["features"][:6])
            if len(finding["features"]) > 6:
                labels += ", … (combined predictors)"
            raise OutcomeProxyError(
                f"Model training paused: '{labels}' can reconstruct the observed {finding['target'].replace('_', ' ')} "
                "on the checked training rows. "
                "This may be hidden outcome information or a strong baseline predictor. Remove the suspect input, "
                "or review when it was measured and record its source before continuing. "
                "The usual predictor checkbox does not clear this finding.", audit)
        audit["status"] = "reviewed_by_user"
        audit["review_reason"] = reason
        audit["notes"] = [note for note in audit["notes"] if not note.startswith("Outcome-pattern findings were retained")]
        audit["notes"].append("Outcome-pattern findings were retained; the user reviewed predictor timing and source: " + reason)
    return audit


def _proxy_summary(context):
    partitions = list(context["partitions"].values())
    statuses = {item["status"] for item in partitions}
    status = ("reviewed_by_user" if "reviewed_by_user" in statuses else
              "partially_assessed" if statuses & {"not_assessed", "partially_assessed"} else
              "no_pattern_detected" if partitions else "not_assessed")
    notes = list(dict.fromkeys(note for item in partitions for note in item["notes"]))
    if not partitions:
        notes.append("Outcome-pattern screening was not assessed on a training partition.")
    return {"status": status, "scope": "training_partitions_only", "partitions": partitions,
            "coverage": "limited" if not partitions or any(item.get("coverage") == "limited" for item in partitions) else "screened",
            "review_reason": context.get("review"), "notes": notes}


def _predictive_identifier_checks(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Compare recognizable IDs without changing the uploaded values.

    Only surrounding whitespace is removed from text keys. Case, internal
    whitespace, leading zeros, and numeric/text distinctions remain unchanged.
    Missing markers do not identify a patient and are reported separately.
    Explicit ID names retain the stricter one-row-per-patient predictive rule;
    entity-only names retain the existing mostly-unique heuristic.
    """
    from survival_toolkit.analysis import (
        _column_name_tokens, _IDENTIFIER_HEAD_TOKENS, _is_identifier_like_name,
        _MISSING_VALUE_TOKENS,
    )

    duplicates: list[dict] = []
    checks: list[dict] = []
    for column in df.columns:
        if not _is_identifier_like_name(str(column)):
            continue
        source = df[column]
        if pd.api.types.is_float_dtype(source.dtype):
            numeric = source.dropna().to_numpy(dtype=float)
            if not bool(np.all(np.isfinite(numeric))) or not bool(np.all(np.mod(numeric, 1) == 0)):
                # A decimal-valued measurement is not an identifier.
                continue
        text_mask = source.map(lambda value: isinstance(value, str))
        stripped = source[text_mask].astype("string").str.strip()
        trimmed = int((stripped != source[text_mask]).sum())
        # Positional mapping also works for a directly supplied frame with a
        # repeated index; label-based assignment could affect unrelated rows.
        def comparison_key(value):
            if isinstance(value, str):
                key = value.strip()
                return pd.NA if key.lower() in _MISSING_VALUE_TOKENS else key
            return value

        keys = source.map(comparison_key)
        valid = keys.dropna()
        counts = valid.value_counts()
        explicit = any(token in _IDENTIFIER_HEAD_TOKENS for token in _column_name_tokens(str(column)))
        repeated = counts[counts > 1]
        finding = {
            "column": str(column), "n_rows": int(len(valid)), "n_unique": int(len(counts)),
            "n_repeated_ids": int(len(repeated)), "n_extra_rows": int((repeated - 1).sum()),
            "missing_identifier_rows": int(keys.isna().sum()), "trimmed_identifier_rows": trimmed,
            "comparison": "surrounding_whitespace_only", "raw_values_changed": False,
        }
        checks.append(finding)
        if len(repeated) and (explicit or len(counts) >= 0.5 * len(valid)):
            duplicates.append(finding)
    return duplicates, checks


def _recognizable_outcome_predictor_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Recognize explicit outcome roles, without treating generic status as proof.

    Shared survival-role predicates protect endpoint aliases and baseline-marker
    exceptions. A role word is additionally required so a bare endpoint
    abbreviation or an outcome-like substring in a gene name is insufficient.
    Ambiguous status codes remain a provenance question for the data owner and
    the existing training-only pattern screen.
    """
    from survival_toolkit.analysis import (
        _column_name_tokens, _COMPACT_TIME_COLUMN_NAMES, _looks_like_event_outcome_column,
        _looks_like_survival_time_column_name, _TIME_UNIT_TOKENS, _values_name_the_outcome,
        _without_missing_event_markers, _SURVIVAL_ENDPOINT_ABBREVIATIONS,
    )

    event_roles = {
        "event", "status", "vital", "death", "dead", "died", "deceased", "censor", "censored",
        "censoring", "cens", "alive", "living", "survived", "progression", "progressed",
        "recurrence", "recurred", "relapse", "relapsed", "failure", "free",
    }
    time_roles = {"time", "duration", "followup", "fup", "futime", *_TIME_UNIT_TOKENS}
    compact_roles = {
        endpoint + role: endpoint + "_" + role
        for endpoint in _SURVIVAL_ENDPOINT_ABBREVIATIONS
        for role in event_roles | time_roles
    }
    offenders: list[str] = []
    for column in dict.fromkeys(features):
        # Acronym + capitalized role ("RFSStatus", "OSMonths") has no
        # lower-to-upper boundary at the acronym. Match exact role compounds,
        # without splitting arbitrary gene symbols or changing column labels.
        role_name = compact_roles.get("".join(_column_name_tokens(column)), column)
        tokens = _column_name_tokens(role_name)
        token_set = set(tokens)
        compact = "".join(tokens)
        explicit_time = bool(token_set & time_roles) or compact in _COMPACT_TIME_COLUMN_NAMES
        if explicit_time and _looks_like_survival_time_column_name(role_name):
            offenders.append(column)
            continue
        if not token_set & event_roles:
            continue
        values = _without_missing_event_markers(df[column])
        bare_status = all(token == "status" or token.isdigit() for token in tokens)
        if bare_status and not _values_name_the_outcome(values):
            continue
        if _looks_like_event_outcome_column(role_name, values):
            offenders.append(column)
    return offenders


def _identical_input_row_audit(df: pd.DataFrame, identifier_checks: list[dict]) -> dict:
    """Bounded, non-blocking check of complete rows when recorded IDs are absent.

    Equal rows are a reason to review provenance, not proof of a shared patient.
    Never compare only the selected predictors or delete matching observations.
    Wide omics tables are outside this optional check's resource scope.
    """
    if any(check["n_rows"] for check in identifier_checks):
        return {"status": "not_assessed", "reason": "recorded_identifier_available"}
    if df.shape[1] > 256 or df.size > 2_000_000:
        return {"status": "not_assessed", "reason": "full_row_check_limit"}
    try:
        all_copies = df.duplicated(keep=False)
        extra_copies = df.duplicated(keep="first")
    except TypeError:
        return {"status": "not_assessed", "reason": "non_scalar_input_cells"}
    return {"status": "review_required" if bool(all_copies.any()) else "no_identical_rows",
            "scope": "all_input_columns", "matching_rows": int(all_copies.sum()),
            "extra_rows": int(extra_copies.sum()), "rows_removed": 0}


def prediction_input_audit(df: pd.DataFrame, time_column: str, event_column: str,
                           features: list[str], event_positive_value: Any = None) -> dict[str, Any]:
    from survival_toolkit.analysis import (
        _coerce_survival_time_values, _cohort_frame, _require_dataframe_columns,
    )

    _require_dataframe_columns(df, [time_column, event_column, *features])
    overlapping = sorted(set(features) & {time_column, event_column})
    if overlapping:
        raise ValueError("Survival outcome columns cannot be used as ML/DL model features: "
                         + ", ".join(overlapping) + ".")
    other_outcomes = _recognizable_outcome_predictor_columns(df, features)
    if other_outcomes:
        raise ValueError(
            "Survival outcome columns cannot be used as ML/DL model features: "
            + ", ".join(other_outcomes)
            + ". Outcomes from another endpoint or follow-up are not baseline predictors. "
            "Remove these columns from the predictor selection and verify measurement timing and source. "
            "Raw data have not been changed."
        )
    duplicates, identifier_checks = _predictive_identifier_checks(df)
    if duplicates:
        finding = duplicates[0]
        raise ValueError(
            f"Repeated patient IDs in '{finding['column']}': {finding['n_repeated_ids']} repeated ID(s), "
            f"{finding['n_extra_rows']} extra row(s). ML/DL evaluation requires one row per patient "
            "so the same patient cannot enter both training and evaluation. Resolve repeated records "
            "in the source data first. ID comparison ignores surrounding whitespace and missing markers; "
            "raw values and rows have not been automatically changed."
        )
    times = _coerce_survival_time_values(df[time_column], time_column)
    negative = int((times < 0).sum())
    if negative:
        raise ValueError(
            f"'{time_column}' contains {negative} negative follow-up time(s). "
            "Correct the time origin or duration in the source data before ML/DL evaluation. "
            "Zero follow-up time is allowed; no rows have been silently removed."
        )
    frame = _cohort_frame(df, time_column, event_column, event_positive_value,
                          extra_columns=features, drop_missing_extra_columns=False)
    excluded = int(df.shape[0] - frame.shape[0])
    notes = [AVAILABILITY_NOTE]
    identical_rows = _identical_input_row_audit(df, identifier_checks)
    if identical_rows["status"] == "review_required":
        notes.append(
            f"No recorded patient IDs were available, and {identical_rows['matching_rows']} input rows "
            f"match across every column ({identical_rows['extra_rows']} extra matching row(s)). "
            "Review whether these are repeated records or distinct patients. This is not proof of patient "
            "duplication; no rows were removed and analysis was not automatically blocked."
        )
    elif identical_rows.get("reason") == "full_row_check_limit":
        notes.append("No recorded patient IDs were available. Complete-row matching was not assessed "
                     "for this table size; confirm patient independence from source records.")
    for check in identifier_checks:
        if check["missing_identifier_rows"]:
            notes.append(
                f"'{check['column']}' has {check['missing_identifier_rows']} missing ID value(s). "
                "Duplicate checking covers only recorded IDs; confirm that these rows represent distinct patients. "
                "Missing IDs were not treated as a shared patient and no rows were removed."
            )
        if check["trimmed_identifier_rows"]:
            notes.append(
                f"'{check['column']}': surrounding whitespace was ignored only when comparing IDs "
                "for repeated patients. Uploaded values were preserved."
            )
    if excluded:
        notes.insert(0, f"Excluded {excluded} of {len(df)} input rows because follow-up time or event "
                     "was missing or non-finite. These outcomes were not imputed. "
                     f"{len(frame)} rows remain; check the exclusions before interpreting results.")
    return {
        "version": GUARD_VERSION, "input_rows": int(len(df)), "analyzed_rows": int(len(frame)),
        "excluded_outcome_rows": excluded,
        "exclusion_reason": "missing_or_nonfinite_outcome" if excluded else None,
        "predictor_availability": "user_review_required",
        "outcome_role_check": "recognizable_names_and_codes",
        "duplicate_check": "identifier_name_heuristic",
        "identifier_checks": identifier_checks,
        "identical_input_rows": identical_rows,
        "notes": notes,
    }


def guarded_prediction_inputs(function):
    """Check raw input before fitting; carry the audit into results and table exports.

    Pattern screening runs at each actual training boundary. Predictor measurement
    time and provenance remain questions for the data owner.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def run(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        review = values.get("outcome_proxy_review")
        if review is not None and (not isinstance(review, str) or not 20 <= len(review.strip()) <= 1000):
            raise ValueError("Outcome-pattern review needs 20–1000 characters explaining predictor timing and source.")
        parent = _PROXY_CONTEXT.get()
        context = parent if parent is not None else {"partitions": {}, "review": review.strip() if review else None}
        token = _PROXY_CONTEXT.set(context)
        try:
            return _run_checked(args, kwargs, values, context)
        finally:
            _PROXY_CONTEXT.reset(token)

    def _run_checked(args, kwargs, values, context):
        if values.get("prepared_data") is not None or values.get("evaluation_split") is not None:
            # Internal deep-model folds receive matrices prepared from an already
            # checked cohort. Do not reinterpret those matrices as raw input.
            result = function(*args, **kwargs)
            result["input_audit"] = {
                "version": GUARD_VERSION, "input_rows": None, "analyzed_rows": None,
                "excluded_outcome_rows": None, "predictor_availability": "user_review_required",
                "source": "prepared_matrices", "raw_input_rechecked": False,
                "notes": ["Preprocessed matrices were supplied; raw patient IDs and outcomes were not rechecked.",
                          AVAILABILITY_NOTE],
            }
            audit = result["input_audit"]
        else:
            features = values.get("features")
            if features is None:
                features = [*values.get("clinical_features", []), *values.get("added_features", [])]
            audit = prediction_input_audit(values["df"], values["time_column"], values["event_column"],
                                       list(features), values.get("event_positive_value"))
            result = function(*args, **kwargs)
        audit["outcome_proxy"] = _proxy_summary(context)
        audit["notes"] = list(dict.fromkeys([*audit["notes"], *audit["outcome_proxy"]["notes"]]))
        result["input_audit"] = audit
        notes = list(audit["notes"])
        for row in [*result.get("comparison_table", []), *result.get("comparison", {}).get("rows", [])]:
            removed = row.get("removed_features") or []
            if removed:
                labels = ", ".join(f"{item['column']} ({item['reason']})" for item in removed)
                notes.append(f"{row['model']}: training-only preprocessing removed {labels}.")
            # The plain comparison CSV retains these notes as well as the JSON result.
            row["input_rows"] = audit["input_rows"]
            row["excluded_outcome_rows"] = audit["excluded_outcome_rows"]
            row["input_notes"] = " ".join(audit["notes"])
            row["outcome_proxy_status"] = audit["outcome_proxy"]["status"]
            row["outcome_proxy_review"] = audit["outcome_proxy"]["review_reason"]
            row["removed_feature_notes"] = "; ".join(
                f"{item['column']}: {item['reason']}" for item in removed)
        if result.get("removed_features"):
            labels = ", ".join(f"{item['column']} ({item['reason']})" for item in result["removed_features"])
            notes.append(f"Training-only preprocessing removed {labels}.")
        for key in ("scientific_summary", "insight_board"):
            summary = result.get(key)
            if isinstance(summary, dict):
                summary.setdefault("cautions", []).extend(note for note in notes if note not in summary.get("cautions", []))
        tables = result.get("manuscript_tables")
        if isinstance(tables, dict):
            tables.setdefault("table_notes", []).extend(note for note in notes if note not in tables.get("table_notes", []))
        return result

    return run

"""Input safeguards and transparent preprocessing notes for predictive models."""
from __future__ import annotations

from functools import wraps
from contextvars import ContextVar
import inspect
from typing import Any

import numpy as np
import pandas as pd

GUARD_VERSION = "prediction-input/2"
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
    key = training_fingerprint(frame[[time_column, event_column, *features]])
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
                f"Model training paused: '{labels}' can reconstruct the observed {finding['target'].replace('_', ' ')}. "
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


def prediction_input_audit(df: pd.DataFrame, time_column: str, event_column: str,
                           features: list[str], event_positive_value: Any = None) -> dict[str, Any]:
    from survival_toolkit.analysis import (
        _coerce_survival_time_values, _cohort_frame, _require_dataframe_columns,
        _column_name_tokens, _IDENTIFIER_HEAD_TOKENS, _is_identifier_like_name,
        detect_duplicate_identifier_columns,
    )

    _require_dataframe_columns(df, [time_column, event_column, *features])
    overlapping = sorted(set(features) & {time_column, event_column})
    if overlapping:
        raise ValueError("Survival outcome columns cannot be used as ML/DL model features: "
                         + ", ".join(overlapping) + ".")
    duplicates = detect_duplicate_identifier_columns(df)
    # Explicit ID columns may contain many records per patient. Apply the
    # stricter predictive rule here, without changing the frozen analysis kernel.
    found = {item["column"] for item in duplicates}
    for column in df.columns:
        if str(column) in found or not _is_identifier_like_name(str(column)):
            continue
        if not any(token in _IDENTIFIER_HEAD_TOKENS for token in _column_name_tokens(str(column))):
            continue
        values = df[column].dropna()
        if pd.api.types.is_float_dtype(values.dtype) and not bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0)):
            continue
        counts = values.value_counts()
        repeated = counts[counts > 1]
        if len(repeated):
            duplicates.append({"column": str(column), "n_repeated_ids": int(len(repeated)),
                               "n_extra_rows": int((repeated - 1).sum())})
    if duplicates:
        finding = duplicates[0]
        raise ValueError(
            f"Repeated patient IDs in '{finding['column']}': {finding['n_repeated_ids']} repeated ID(s), "
            f"{finding['n_extra_rows']} extra row(s). ML/DL evaluation requires one row per patient "
            "so the same patient cannot enter both training and evaluation. Resolve repeated records "
            "in the source data first; rows have not been automatically deleted."
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
    if excluded:
        notes.insert(0, f"Excluded {excluded} of {len(df)} input rows because follow-up time or event "
                     "was missing or non-finite. These outcomes were not imputed. "
                     f"{len(frame)} rows remain; check the exclusions before interpreting results.")
    return {
        "version": GUARD_VERSION, "input_rows": int(len(df)), "analyzed_rows": int(len(frame)),
        "excluded_outcome_rows": excluded,
        "exclusion_reason": "missing_or_nonfinite_outcome" if excluded else None,
        "predictor_availability": "user_review_required",
        "duplicate_check": "identifier_name_heuristic",
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
            audit = prediction_input_audit(values["df"], values["time_column"], values["event_column"],
                                       list(values["features"]), values.get("event_positive_value"))
            result = function(*args, **kwargs)
        audit["outcome_proxy"] = _proxy_summary(context)
        audit["notes"] = list(dict.fromkeys([*audit["notes"], *audit["outcome_proxy"]["notes"]]))
        result["input_audit"] = audit
        notes = list(audit["notes"])
        for row in result.get("comparison_table", []):
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

"""Input safeguards and transparent preprocessing notes for predictive models."""
from __future__ import annotations

from functools import wraps
import inspect
from typing import Any

import pandas as pd

GUARD_VERSION = "prediction-input/1"
AVAILABILITY_NOTE = (
    "Use only variables available at the time you intend to predict risk. "
    "Remove measurements collected after the outcome or derived from follow-up. "
    "Automatic checks cannot identify every hidden outcome proxy."
)


def prediction_input_audit(df: pd.DataFrame, time_column: str, event_column: str,
                           features: list[str], event_positive_value: Any = None) -> dict[str, Any]:
    from survival_toolkit.analysis import (
        _coerce_survival_time_values, _cohort_frame, _require_dataframe_columns,
        detect_duplicate_identifier_columns,
    )

    _require_dataframe_columns(df, [time_column, event_column, *features])
    overlapping = sorted(set(features) & {time_column, event_column})
    if overlapping:
        raise ValueError("Survival outcome columns cannot be used as ML/DL model features: "
                         + ", ".join(overlapping) + ".")
    duplicates = detect_duplicate_identifier_columns(df)
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

    No association-based feature screening is performed. The patient's predictor
    measurement time remains a question for the data owner.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def run(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
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
            return result
        audit = prediction_input_audit(values["df"], values["time_column"], values["event_column"],
                                       list(values["features"]), values.get("event_positive_value"))
        result = function(*args, **kwargs)
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
            row["removed_feature_notes"] = "; ".join(
                f"{item['column']}: {item['reason']}" for item in removed)
        if result.get("removed_features"):
            labels = ", ".join(f"{item['column']} ({item['reason']})" for item in result["removed_features"])
            notes.append(f"Training-only preprocessing removed {labels}.")
        summary = result.get("scientific_summary")
        if isinstance(summary, dict):
            summary.setdefault("cautions", []).extend(note for note in notes if note not in summary.get("cautions", []))
        tables = result.get("manuscript_tables")
        if isinstance(tables, dict):
            tables.setdefault("table_notes", []).extend(note for note in notes if note not in tables.get("table_notes", []))
        return result

    return run

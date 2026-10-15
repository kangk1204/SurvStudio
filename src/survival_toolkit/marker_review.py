"""Plain-language explanations of existing marker decisions.

This presentation layer never runs a test, changes a threshold, excludes a
marker, or releases a hold. Keep it outside the qualified numerical kernel.
"""
from __future__ import annotations

from typing import Any

import copy
import numpy as np


def marker_diagnostic_review(result: dict[str, Any]) -> dict[str, Any]:
    inference = result.get("inference") or {}
    reasons = [str(reason) for reason in inference.get("reasons") or []]
    variance: list[str] = []
    mean: list[str] = []
    messages: list[str] = []
    unexplained: list[str] = []
    failed = False
    clinical = False
    for reason in reasons:
        if reason.startswith("residual_misspecification: "):
            marker, separator, kind = reason.removeprefix("residual_misspecification: ").rpartition("/")
            if separator and kind in {"variance", "nonlinear_mean"}:
                target = variance if kind == "variance" else mean
                if marker not in target:
                    target.append(marker)
                continue
        if reason.startswith("clinical_misspecification: "):
            clinical = True
            continue
        if reason.startswith(("clinical_diagnostic_failed:", "residual_diagnostic_failed:")):
            failed = True
            continue
        unexplained.append(reason)
    if variance:
        count = len(variance)
        messages.append(f"Residual spread changes with clinical information for {count} marker{'s' if count != 1 else ''}. "
                        "Adjusting average expression does not remove unequal spread.")
    if mean:
        count = len(mean)
        messages.append(f"For {count} marker{'s' if count != 1 else ''}, the clinical adjustment leaves a nonlinear expression pattern.")
    if clinical:
        messages.append("The clinical survival model shows a time-effect or functional-form problem; review its detailed diagnostics.")
    if failed:
        messages.append("A required diagnostic could not be calculated, so added-value inference cannot be assessed.")
    messages.extend(unexplained)
    withheld = inference.get("status") == "withheld" or inference.get("allowed") is False
    return {
        "version": "marker-review/1",
        "status": inference.get("status", "not_assessed"),
        "messages": messages,
        "flagged_variance_markers": variance,
        "flagged_mean_markers": mean,
        "raw_reasons": reasons,
        "interpretation": (
            "These findings concern the analysis assumptions; they do not prove that the input is wrong or a gene has no prognostic value."
            if variance or mean or clinical else "Diagnostic checks do not certify the analysis assumptions."
        ),
        "next_steps": [
            "Review source coding, preprocessing and clinical groups. Record any revised analysis before examining its marker results.",
            "Do not drop flagged genes or change cutoffs only to obtain a pass. Any revised candidate needs independent validation.",
        ] if withheld else [],
        "decision_changed": False,
    }


def review_external_horizon(validation: dict[str, Any], frame: Any, recipe: dict[str, Any], *,
                            column_mapping: dict[str, str] | None = None,
                            event_positive_value: Any = None) -> dict[str, Any]:
    """Withhold horizon-specific evaluation beyond the scored cohort's follow-up.

    A model can predict beyond these observations, but that prediction cannot be
    evaluated here. Ranking, the all-time calibration slope and marker inference
    are unchanged. The frozen numerical calculation and recipe remain untouched.
    """
    from survival_toolkit.analysis import _cohort_frame
    from survival_toolkit.marker_evaluation import _clinical_columns_behind, _unique_rows

    result = copy.deepcopy(validation)
    metrics = result.get("metrics") or {}
    horizon = metrics.get("horizon")
    if horizon is None:
        return result
    mapping = column_mapping or {}
    def external(name: str) -> str:
        return mapping.get(name, name)
    outcome = recipe["outcome"]
    time_column, event_column = external(outcome["time_column"]), external(outcome["event_column"])
    positive = outcome["event_positive_value"] if event_positive_value is None else event_positive_value
    columns = recipe["clinical"]["columns"]
    terms = {*recipe["model"]["terms"], *((recipe.get("clinical_only_model") or {}).get("terms") or [])}
    used = _clinical_columns_behind(recipe["clinical"]["encoder"], columns, terms)
    strata = recipe.get("strata_columns") or []
    extra = strata if recipe.get("recipe_version", 1) >= 3 else [*used, *strata]
    scored = _cohort_frame(_unique_rows(frame), time_column=time_column, event_column=event_column,
                           event_positive_value=positive, extra_columns=[external(c) for c in extra])
    cohort = result.get("cohort") or {}
    if len(scored) != cohort.get("n") or scored.attrs.get("row_mask_hash") != cohort.get("row_mask_hash"):
        raise RuntimeError("The follow-up review does not match the scored validation cohort.")
    times = scored[time_column].to_numpy(dtype=float)
    maximum = float(np.max(times))
    supported = float(horizon) <= maximum
    result["absolute_risk_review"] = {
        "status": "within_observed_followup" if supported else "beyond_validation_followup",
        "horizon": horizon, "maximum_followup": maximum,
        "n_at_or_beyond_horizon": int(np.sum(times >= float(horizon))),
        "interpretation": "Follow-up coverage is necessary but does not establish calibration or reliable censoring weights.",
    }
    if not supported:
        keys = ("observed_risk", "expected_risk", "observed_expected_ratio", "brier", "brier_skill")
        result["exploratory_horizon_metrics"] = {key: metrics[key] for key in keys if key in metrics}
        for key in keys:
            if key in metrics:
                metrics[key] = None
        result.setdefault("notes", []).append(
            f"The requested horizon ({float(horizon):g}) exceeds the longest validation follow-up ({maximum:g}). "
            "Absolute-risk calibration and Brier evaluation at this horizon are unavailable. "
            "Raw calculations are retained separately; C-index and the all-time calibration slope are unchanged."
        )
    return result

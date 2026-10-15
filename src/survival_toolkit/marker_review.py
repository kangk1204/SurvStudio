"""Plain-language explanations of existing marker decisions.

This presentation layer never runs a test, changes a threshold, excludes a
marker, or releases a hold. Keep it outside the qualified numerical kernel.
"""
from __future__ import annotations

from typing import Any


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

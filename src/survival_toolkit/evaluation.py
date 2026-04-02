from __future__ import annotations


def metric_name_for_evaluation(evaluation_mode: str) -> str:
    if evaluation_mode == "holdout":
        return "Holdout C-index"
    if evaluation_mode == "holdout_fallback_apparent":
        return "Apparent fallback C-index"
    if evaluation_mode == "repeated_cv":
        return "Repeated-CV mean C-index"
    if evaluation_mode == "repeated_cv_incomplete":
        return "Repeated-CV mean C-index (incomplete)"
    return "Apparent C-index"

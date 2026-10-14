"""Describe fixed-model risk scores and held-out groups, without selecting a cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd
from survival_toolkit.analysis import compute_km_analysis
from survival_toolkit.errors import UserInputError


def describe_heldout_risk(*, training_risk, test_risk, time, event, source_rows):
    """Learn a median risk cutoff on training rows; never use test outcomes to set it."""
    training = np.asarray(training_risk, dtype=float).reshape(-1)
    risk = np.asarray(test_risk, dtype=float).reshape(-1)
    if not len(training) or not np.isfinite(training).all() or not np.isfinite(risk).all():
        raise UserInputError("Finite training and test risk scores are required for this display.")
    if not (len(risk) == len(time) == len(event) == len(source_rows)):
        raise UserInputError("Every test score must match one held-out patient and outcome.")
    cutoff = float(np.median(training))
    groups = np.where(risk >= cutoff, "Higher predicted risk", "Lower predicted risk")
    ranks = pd.Series(risk).rank(method="min", ascending=False).astype(int).to_numpy()
    rows = [{"source_row": str(source_rows[i]), "risk_score": float(risk[i]),
             "test_risk_rank": int(ranks[i]), "predicted_risk_group": str(groups[i])}
            for i in np.argsort(-risk, kind="stable")]
    result = {"display_version": "risk-display/1", "cutoff": cutoff,
              "cutoff_source": "Median fitted-model risk on training patients only",
              "cutoff_rule": "At or above training median = higher predicted risk",
              "cutoff_optimized_on_outcomes": False, "rows": rows,
              "risk_score_interpretation": "Relative model score; higher means higher predicted risk, not a survival probability. Units differ between models.",
              "group_interpretation": "Observed survival in fixed, model-defined groups on held-out patients. These groups are descriptive and are not validated clinical decision thresholds.",
              "logrank_reported": False,
              "logrank_policy": "Use held-out C-index and paired feature-set intervals to assess prediction; group curves are descriptive.",
              "km": None, "curve_error": None}
    if len(np.unique(groups)) < 2:
        result["curve_error"] = "Only one predicted risk group occurs in the test patients; two-group curves are unavailable."
        return result
    frame = pd.DataFrame({"followup_time": np.asarray(time, dtype=float),
                          "event": np.asarray(event, dtype=int), "predicted_risk_group": groups})
    try:
        full = compute_km_analysis(frame, "followup_time", "event", "predicted_risk_group",
                                   event_positive_value=1, suppress_group_inference=True)
        # Carry descriptive curves and counts only, not auxiliary association tests.
        keys = ["curves", "risk_table", "cohort", "groups", "display_horizon", "confidence_level", "group_column", "summary_table"]
        result["km"] = {key: full[key] for key in keys}
        result["km"]["test"] = None
        result["km"]["logrank_p"] = None
    except (UserInputError, ValueError) as exc:
        result["curve_error"] = str(exc)
    return result

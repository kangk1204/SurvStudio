"""Regression tests for the second September 2026 review of the figures and the checklist text: every figure and
sentence has to show the quantity the engine computed, and give the reason the engine had when it skipped a step."""

from __future__ import annotations

import math

from survival_toolkit.plots import build_marker_replication_figure


def _fit(hazard_ratio: float, low: float, high: float) -> dict:
    return {"log_hr": math.log(hazard_ratio), "hazard_ratio": hazard_ratio, "ci_lower": low, "ci_upper": high, "wald_p": 0.01}


def _drawn(figure: dict) -> dict:
    """Marker -> plotted hazard ratio, over the forest's named traces."""
    return {label: x for trace in figure["data"] if trace.get("name") for x, label in zip(trace["x"], trace["y"])}


def _annotation_text(figure: dict) -> str:
    return " ".join(str(annotation.get("text", "")) for annotation in figure["layout"].get("annotations", []))


# ── Replication figure ───────────────────────────────────────────


def test_replication_forest_draws_the_hazard_ratio_of_the_fit_each_replication_test_used() -> None:
    validation = {
        "markers": [
            # Added value was tested: the adjusted hazard ratio is drawn, not the marginal one.
            {"marker": "adj", "marginal": _fit(1.9, 1.5, 2.4), "adjusted": _fit(1.3, 1.1, 1.6), "tested": "added_value",
             "same_direction": True, "replicated": True, "replication_p_holm": 0.01},
            # A locked marginal claim is tested without adjustment even when an adjusted fit exists.
            {"marker": "marg", "marginal": _fit(0.7, 0.55, 0.9), "adjusted": _fit(0.95, 0.8, 1.1), "tested": "marginal",
             "same_direction": True, "replicated": False, "replication_p_holm": 0.08},
            # The added-value fit was not estimable, so nothing was tested: no marginal HR drawn as "opposite direction".
            {"marker": "unfit", "marginal": _fit(1.5, 1.2, 1.9), "adjusted": None, "tested": None,
             "same_direction": False, "replicated": False, "replication_p_holm": None},
            {"marker": "missing", "marginal": None, "adjusted": None, "tested": None, "absent": True,
             "same_direction": False, "replicated": False, "replication_p_holm": None},
            {"marker": "wide", "marginal": _fit(1.2, 0.9, 1.6), "adjusted": {**_fit(3.0, 0.5, 1.0), "ci_upper": None}, "tested": "added_value",
             "same_direction": True, "replicated": False, "replication_p_holm": 0.4},
        ],
        "metrics": {"c_index": 0.68, "clinical_only_c_index": 0.64, "delta_c_index": 0.04},
    }

    figure = build_marker_replication_figure(validation)

    assert _drawn(figure) == {"adj": 1.3, "marg": 0.7}
    assert {trace["name"] for trace in figure["data"] if trace.get("name")} == {"replicated", "same direction, not significant"}
    # The recipe's order, first at the top.
    assert figure["layout"]["yaxis2"]["categoryarray"] == ["marg", "adj"]
    note = _annotation_text(figure)
    assert "Not drawn: unfit (not estimable), missing (not measured), wide (interval not finite)" in note
    assert figure["layout"]["margin"]["b"] > 110


def test_replication_forest_of_an_older_result_without_the_tested_fit_keeps_the_adjusted_one() -> None:
    older = {
        "markers": [
            {"marker": "old", "marginal": _fit(1.9, 1.5, 2.4), "adjusted": _fit(1.3, 1.1, 1.6), "same_direction": True,
             "replicated": True, "replication_p_holm": 0.01},
            {"marker": "plain", "marginal": _fit(0.8, 0.7, 0.95), "same_direction": False, "replicated": False, "replication_p_holm": 0.9},
        ],
        "metrics": {},
    }

    figure = build_marker_replication_figure(older)

    assert _drawn(figure) == {"old": 1.3, "plain": 0.8}
    assert "Not drawn" not in _annotation_text(figure)
    assert figure["layout"]["margin"]["b"] == 110

"""Regression tests for the second September 2026 review of the figures and the checklist text: every figure and
sentence has to show the quantity the engine computed, and give the reason the engine had when it skipped a step."""

from __future__ import annotations

import math

from survival_toolkit.plots import (
    build_marker_replication_figure,
    build_marker_stability_figure,
    build_marker_summary_figure,
    marker_evidence_funnel,
)
from survival_toolkit.reporting import marker_results_paragraph, remark_checklist

_REQUEST = {"time_column": "os_time", "event_column": "os_event", "event_positive_value": 1}


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


# ── Evidence funnel ──────────────────────────────────────────────


def _screen_result(**changes) -> dict:
    """A marginal screen of five markers, as evaluate_markers returns it without permutations."""
    result = {
        "primary_lens": "marginal",
        "marker_table": [
            {"marker": f"m{index}", "tier": "not supported",
             "marginal": {"p_value": 0.001, "q_bh": 0.004, "p_fwer": None, "q_perm": None,
                          "selection_frequency": 0.9, "direction_consistency": 1.0}}
            for index in range(5)
        ],
        "tier_counts": {"robust": 0, "suggestive": 0, "marginal only": 0, "not supported": 5},
        "cohort": {"n_markers_evaluated": 5, "dropped_markers": []},
        "settings": {"alpha": 0.05, "robust_frequency": 0.5, "robust_direction": 0.9},
        "null": {"n_permutations": 0},
        "resampling": {"n_valid": 20, "n_failed": 0, "stability_assessed": True},
        "signature": {},
    }
    result.update(changes)
    return result


def _funnel_text(figure: dict) -> dict:
    """Funnel label -> the text printed beside its bar."""
    text = next(trace for trace in figure["data"] if trace.get("mode") == "text")
    return dict(zip(text["y"], text["text"]))


def test_evidence_funnel_does_not_count_family_wise_bars_that_were_never_computed() -> None:
    result = _screen_result()

    stages = {stage["label"]: stage for stage in marker_evidence_funnel(result)}
    figure = build_marker_summary_figure(result)

    assert stages["FDR q ≤ 0.05"]["count"] == 5
    for label in ("Family-wise p ≤ 0.05", "Robust"):
        assert stages[label]["count"] is None and stages[label]["note"] == "not computed (no permutations)"
    printed = _funnel_text(figure)
    assert printed["FDR q ≤ 0.05"] == "<b>5</b>"
    assert printed["Family-wise p ≤ 0.05"] == printed["Robust"] == "not computed (no permutations)"
    bars = next(trace for trace in figure["data"] if trace["type"] == "bar")
    assert dict(zip(bars["y"], bars["x"]))["Robust"] == 0.0
    # The stability figure's robust rule says it could not apply.
    stability = build_marker_stability_figure(result)
    assert "No permutations were run, so no marker could be robust" in _annotation_text(stability)


def test_evidence_funnel_marks_the_robust_bar_not_assessed_without_subsamples() -> None:
    table = [
        {"marker": f"m{index}", "tier": "suggestive" if index < 2 else "not supported",
         "marginal": {"p_value": 0.001, "q_bh": 0.004, "p_fwer": 0.01 if index < 2 else 0.2, "q_perm": 0.01}}
        for index in range(5)
    ]
    result = _screen_result(
        marker_table=table, null={"n_permutations": 99}, resampling={"n_valid": 0, "n_failed": 0, "stability_assessed": False}
    )

    stages = {stage["label"]: stage for stage in marker_evidence_funnel(result)}

    assert stages["Family-wise p ≤ 0.05"]["count"] == 2
    assert stages["Robust"]["count"] is None and stages["Robust"]["note"] == "not assessed (no subsamples)"
    printed = _funnel_text(build_marker_summary_figure(result))
    assert printed["Family-wise p ≤ 0.05"] == "<b>2</b>" and printed["Robust"] == "not assessed (no subsamples)"


# ── The model of the whole procedure ─────────────────────────────


def _added_value_result(**changes) -> dict:
    """evaluate_markers output whose full-cohort model did not converge while the subsamples were scored."""
    result = {
        "primary_lens": "added_value",
        "marker_table": [],
        "tier_counts": {"robust": 0, "suggestive": 2, "marginal only": 0, "not supported": 6},
        "cohort": {"n": 300, "events": 120, "n_markers_evaluated": 8, "clinical_columns": ["age"], "strata_columns": [], "dropped_markers": []},
        "settings": {"alpha": 0.05, "fdr_level": 0.1, "shortlist_size": 50, "max_signature_markers": 10},
        "null": {"n_permutations": 1000, "lens2_null": "smith"},
        "resampling": {"n_valid": 200, "n_failed": 0, "fraction": 0.632, "stability_assessed": True},
        "signature": {
            "markers": [], "clinical_only": False, "apparent_c": None, "optimism_corrected_c": None,
            "signature_c_left_out": 0.65, "clinical_c_left_out": 0.62, "signature_gain_left_out": 0.03,
            "n_signature_replicates": 150, "n_clinical_replicates": 150,
        },
        "duplicates": {},
    }
    result.update(changes)
    return result


def _ladder(figure: dict) -> list:
    return next(trace for trace in figure["data"] if trace.get("mode") == "markers+text")["y"]


def test_a_full_cohort_model_that_could_not_be_fitted_is_named_in_the_text_and_the_summary_figure() -> None:
    result = _added_value_result()

    text = marker_results_paragraph(result)
    figure = build_marker_summary_figure(result)
    items = {entry["item"]: entry for entry in remark_checklist(result, request=_REQUEST)["items"]}

    assert "it reached" not in text
    assert "The final model could not be fitted in the full cohort, so it has no apparent or optimism-corrected C-index." in text
    assert "In the patients left out of each of 150 subsamples, the whole procedure reached a mean C-index of 0.650 against 0.620" in text
    assert "the whole procedure reached a mean C-index" in items["18"]["text"]
    titles = [annotation["text"] for annotation in figure["layout"]["annotations"]]
    assert "C-index (full-cohort model not fitted)" in titles and "C-index of the selected-marker model" not in titles
    assert _ladder(figure) == ["Whole procedure<br>(left-out)", "Clinical only<br>(left-out)"]

    # Without a scored subsample the panel gives that reason, not "No marker entered the model."
    bare = build_marker_summary_figure(_added_value_result(signature={"markers": [], "clinical_only": False, "apparent_c": None}))
    notes = _annotation_text(bare)
    assert "The model could not be fitted in the full cohort." in notes and "No marker entered the model." not in notes


def test_the_left_out_rung_of_a_clinical_only_model_is_the_whole_procedure() -> None:
    signature = {"markers": [], "clinical_only": True, "apparent_c": 0.70, "optimism_corrected_c": 0.69,
                 "signature_c_left_out": 0.66, "clinical_c_left_out": 0.655}

    clinical_only = build_marker_summary_figure(_added_value_result(signature=signature))
    selected = build_marker_summary_figure(_added_value_result(signature={**signature, "markers": ["m1"], "clinical_only": False}))

    assert _ladder(clinical_only) == ["Apparent", "Optimism-corrected", "Whole procedure<br>(left-out)", "Clinical only<br>(left-out)"]
    assert _ladder(selected) == ["Apparent", "Optimism-corrected", "Left-out patients", "Clinical only<br>(left-out)"]


def test_a_screen_without_clinical_covariates_says_whether_it_selected_a_marker_or_failed_to_fit_one() -> None:
    no_model = {"markers": [], "clinical_only": False, "apparent_c": None, "optimism_corrected_c": None}
    # Every marker of the screen has q = 0.004, so the procedure selected markers and failed to fit them.
    selected = _screen_result(signature=no_model)
    nothing_selected = _screen_result(signature=no_model)
    nothing_selected["marker_table"] = [
        {**row, "marginal": {**row["marginal"], "q_bh": 0.4}} for row in nothing_selected["marker_table"]
    ]

    none_figure = build_marker_summary_figure(nothing_selected)
    failed_figure = build_marker_summary_figure(selected)

    assert "No marker was selected, so there is no model." in _annotation_text(none_figure)
    assert "C-index (no marker selected)" in _annotation_text(none_figure)
    assert "The model could not be fitted in the full cohort." in _annotation_text(failed_figure)

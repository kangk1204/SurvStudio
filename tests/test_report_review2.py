"""Regression tests for the second September 2026 review of the figures and the checklist text: every figure and
sentence has to show the quantity the engine computed, and give the reason the engine had when it skipped a step."""

from __future__ import annotations

import math

from survival_toolkit.plots import (
    build_cox_forest_figure,
    build_km_figure,
    build_loss_curve_figure,
    build_marker_replication_figure,
    build_marker_stability_figure,
    build_marker_summary_figure,
    build_time_dependent_importance_figure,
    marker_evidence_funnel,
)
from survival_toolkit.reporting import marker_results_paragraph, remark_checklist, tripod_ai_checklist

_REQUEST = {"time_column": "os_time", "event_column": "os_event", "event_positive_value": 1}


def _fit(hazard_ratio: float, low: float, high: float) -> dict:
    return {"log_hr": math.log(hazard_ratio), "hazard_ratio": hazard_ratio, "ci_lower": low, "ci_upper": high, "wald_p": 0.01}


def _drawn(figure: dict) -> dict:
    """Marker -> plotted hazard ratio, over the forest's named traces."""
    return {label: x for trace in figure["data"] if trace.get("name") for x, label in zip(trace["x"], trace["y"])}


def _annotation_text(figure: dict) -> str:
    return " ".join(str(annotation.get("text", "")) for annotation in figure["layout"].get("annotations", []))


def _items(report: dict) -> dict:
    return {entry["item"]: entry for entry in report["items"]}


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
    assert "The final model could not be fitted in the full cohort, so it has no apparent or subsample gap-adjusted C-index." in text
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

    assert _ladder(clinical_only) == ["Apparent", "Subsample gap-adjusted", "Whole procedure<br>(left-out)", "Clinical only<br>(left-out)"]
    assert _ladder(selected) == ["Apparent", "Subsample gap-adjusted", "Left-out patients", "Clinical only<br>(left-out)"]


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


# ── REMARK text ──────────────────────────────────────────────────


def _exact(hazard_ratio: float | None = 1.2) -> dict | None:
    return None if hazard_ratio is None else {"hazard_ratio": hazard_ratio, "ci_lower": 1.0, "ci_upper": 1.5, "wald_p": 0.04}


def _remark_result(n_markers: int, shortlisted: int, *, unfitted: int = 0, **changes) -> dict:
    """An added-value evaluation whose ``shortlisted`` strongest markers got exact Cox fits, the last ``unfitted``
    of them without an estimate on either lens."""
    table = []
    for index in range(n_markers):
        exact = None
        if index < shortlisted:
            estimable = index < shortlisted - unfitted
            exact = {"marginal": _exact() if estimable else None, "adjusted": _exact() if estimable else None}
        table.append({"marker": f"m{index}", "tier": "not supported", "exact": exact,
                      "added_value": {"q_bh": 0.5}, "marginal": {"p_value": 0.2}})
    result = _added_value_result(
        marker_table=table,
        tier_counts={"robust": 0, "suggestive": 0, "marginal only": 0, "not supported": n_markers},
        cohort={"n": 300, "events": 120, "n_markers_evaluated": n_markers, "clinical_columns": ["age"], "strata_columns": [],
                "dropped_markers": []},
        signature={"markers": ["m0"], "clinical_only": False, "apparent_c": 0.74, "optimism_corrected_c": 0.71},
    )
    result.update(changes)
    return result


def test_remark_items_count_the_markers_with_an_exact_fit_against_the_shortlist() -> None:
    whole_panel = _items(remark_checklist(_remark_result(8, 8, unfitted=1), request=_REQUEST))
    shortlist = _items(remark_checklist(_remark_result(60, 52, unfitted=1), request=_REQUEST))
    complete = _items(remark_checklist(_remark_result(60, 52), request=_REQUEST))

    assert "Adjusted hazard ratios with confidence intervals are given for 7 of the 8 markers (1 marker had no estimate)," in whole_panel["17"]["text"]
    assert "the 50 strongest" not in whole_panel["16"]["text"]
    assert "for 51 markers (the 50 strongest and every supported one; 1 marker had no estimate)" in shortlist["17"]["text"]
    assert "for 52 markers (the 50 strongest and every supported one)," in complete["16"]["text"]


def test_remark_univariable_item_promises_no_interval_the_table_does_not_give() -> None:
    added_value = _items(remark_checklist(_remark_result(8, 8), request=_REQUEST))["15"]["text"]
    marginal = _items(
        remark_checklist(_remark_result(8, 8, primary_lens="marginal", cohort={"n": 300, "events": 120, "n_markers_evaluated": 8}),
                         request=_REQUEST)
    )["15"]["text"]

    # With clinical covariates the exported table has the unadjusted HR as a point estimate only.
    assert "confidence interval from an unadjusted" not in added_value
    assert "all 8 markers, the hazard ratio from an unadjusted Cox model as a point estimate (Unadjusted HR)" in added_value
    assert "the table's confidence intervals belong to the adjusted hazard ratios" in added_value
    # Without them the table's HR and interval are the unadjusted ones.
    assert "the hazard ratio with a 95% confidence interval from an unadjusted Cox model" in marginal


def test_remark_counts_use_singular_nouns() -> None:
    result = _remark_result(
        8, 8,
        cohort={"n": 300, "events": 120, "n_markers_evaluated": 8, "clinical_columns": ["age"], "strata_columns": [],
                "dropped_markers": [{"marker": "flat", "reason": "constant"}]},
        duplicates={"checked": True, "identical_checked": True, "markers_used": 5000, "n_pairs": 1, "n_identical": 1},
    )

    report = remark_checklist(result, request=_REQUEST)

    assert "1 marker was excluded before the analysis (flat)." in _items(report)["8"]["text"]
    assert "flagged 1 pair of patients with near-identical profiles and 1 group of patients with identical values." in report["results"]
    assert "(s)" not in " ".join([report["methods"], report["results"], *(entry["text"] for entry in report["items"])])


def test_remark_duplicate_screen_gives_the_patient_cap_as_the_reason_near_identical_profiles_were_not_compared() -> None:
    capped = {"checked": False, "identical_checked": True, "n_pairs": 0, "n_identical": 0,
              "note": "Near-identical profiles are checked for up to 6,000 patients."}
    cohort = {"n": 7000, "events": 3000, "n_markers_evaluated": 5000, "clinical_columns": ["age"], "strata_columns": [], "dropped_markers": []}

    methods = remark_checklist(_remark_result(8, 8, duplicates=capped, cohort=cohort), request=_REQUEST)["methods"]
    # A result without the note: a panel of 5,000 markers was large enough, so the cohort was too large.
    legacy = remark_checklist(
        _remark_result(8, 8, duplicates={key: value for key, value in capped.items() if key != "note"}, cohort=cohort), request=_REQUEST
    )["methods"]
    small = remark_checklist(
        _remark_result(8, 8, duplicates={**capped, "note": "Near-identical profiles are checked on panels of at least 200 markers."}),
        request=_REQUEST,
    )["methods"]

    expected = "near-identical profiles are compared only in cohorts of up to 6,000 patients."
    assert expected in methods and expected in legacy
    assert "panels of at least 200 markers" not in methods and "panels of at least 200 markers" not in legacy
    assert "near-identical profiles are checked only on panels of at least 200 markers." in small


def test_remark_gives_the_reason_the_optimism_could_not_be_corrected_when_every_subsample_failed() -> None:
    failed = _remark_result(8, 8, resampling={"n_valid": 0, "n_failed": 12, "fraction": 0.632, "stability_assessed": False})
    failed["signature"] = {"markers": ["m0"], "clinical_only": False, "apparent_c": 0.7, "optimism_corrected_c": None}

    methods = remark_checklist(failed, request=_REQUEST)["methods"]

    assert "could not be adjusted for the subsampling gap because every subsample failed." in methods
    assert "no subsample was available" not in methods


def test_remark_describes_the_shrinkage_of_each_subsamples_strongest_marker() -> None:
    result = _remark_result(8, 8)
    result["signature"] = {**result["signature"], "top_marker_shrinkage": 0.62}

    text = _items(remark_checklist(result, request=_REQUEST))["18"]["text"]

    assert "subsamples that selected it" not in text
    assert (
        "in the patients left out, the log hazard ratio of each subsample's strongest marker (the one with the largest "
        "score statistic, whether or not it was selected) was on average 62% of its value in the subsample"
    ) in text


# ── TRIPOD+AI text ───────────────────────────────────────────────


def _ml(rows: list[dict], *, mode: str = "holdout", **changes) -> dict:
    result = {
        "family": "ml",
        "comparison_table": rows,
        "errors": [],
        "n_patients": 400,
        "n_events": 150,
        "evaluation_mode": mode,
        "evaluation_split_fingerprint": "fp",
        "request_config": {"time_column": "t", "event_column": "e", "event_positive_value": 1, "features": ["age", "grade"],
                           "categorical_features": ["grade"], "n_estimators": 100, "max_depth": None, "learning_rate": 0.1},
    }
    result.update(changes)
    return result


def _dl_mixed() -> dict:
    return {
        "family": "dl",
        "evaluation_mode": "mixed_holdout_apparent",
        "n_patients": 400,
        "n_events": 150,
        "evaluation_split_fingerprint": "fp",
        "comparison_table": [
            {"model": "DeepSurv", "c_index": 0.66, "evaluation_mode": "holdout", "comparable_for_ranking": True},
            {"model": "DeepHit", "c_index": 0.91, "evaluation_mode": "apparent", "comparable_for_ranking": False},
        ],
        "errors": [],
        "request_config": {"time_column": "t", "event_column": "e", "event_positive_value": 1, "features": ["age"], "categorical_features": [],
                           "epochs": 100, "learning_rate": 0.001, "hidden_layers": [64], "dropout": 0.1, "batch_size": 64,
                           "early_stopping_patience": 10},
    }


def test_tripod_limitations_say_when_performance_is_apparent() -> None:
    apparent = _ml([{"model": "Cox PH", "c_index": 0.74, "evaluation_mode": "apparent"},
                    {"model": "Random Survival Forest", "c_index": 0.91, "evaluation_mode": "apparent"}], mode="apparent")
    holdout = _ml([{"model": "Cox PH", "c_index": 0.70, "evaluation_mode": "holdout"}])

    apparent_items = _items(tripod_ai_checklist([apparent]))
    only_apparent = apparent_items["26"]["text"]
    mixed = _items(tripod_ai_checklist([_dl_mixed()]))["26"]["text"]
    internal = _items(tripod_ai_checklist([holdout]))["26"]["text"]

    assert only_apparent.startswith(
        "Performance is apparent: every model was scored on the patients used for fitting, which is optimistic."
    )
    assert "comes from internal validation" not in only_apparent
    assert apparent_items["16"]["text"] == (
        "Performance was estimated by scoring the models on the patients used for fitting (apparent performance, which is optimistic)."
    )
    assert mixed.startswith(
        "Performance comes from internal validation in one data set, except for DeepHit, which was scored on the patients "
        "used for fitting (apparent performance, which is optimistic); an external cohort is needed to judge transportability."
    )
    assert internal.startswith("Performance comes from internal validation in one data set; an external cohort is needed")


def test_tripod_predictors_say_text_predictors_were_coded_as_categorical() -> None:
    declared_only = _items(tripod_ai_checklist([_ml([{"model": "Cox PH", "c_index": 0.7}])]))["9"]["text"]
    resolved = _items(tripod_ai_checklist([_ml([{"model": "Cox PH", "c_index": 0.7}], categorical_features=["grade", "site"])]))["9"]["text"]

    assert declared_only.startswith("2 predictors: age, grade; categorical: grade. Any predictor stored as text was also reference-coded")
    # A result that names the categorical predictors it resolved is quoted as it is.
    assert resolved.startswith("2 predictors: age, grade; categorical: grade, site. Describe")


def test_tripod_hyperparameters_describe_the_models_that_ran_with_their_actual_settings() -> None:
    cox_only = _ml(
        [{"model": "Cox PH", "c_index": 0.7}],
        errors=[{"model": name, "error": "scikit-survival is not installed."}
                for name in ("LASSO-Cox", "Random Survival Forest", "Gradient Boosted Survival")],
    )
    trees = _ml([{"model": "Random Survival Forest", "c_index": 0.72}, {"model": "Gradient Boosted Survival", "c_index": 0.71}])
    deep = _ml([{"model": "Random Survival Forest", "c_index": 0.72}, {"model": "Gradient Boosted Survival", "c_index": 0.71}],
               request_config={**trees["request_config"], "max_depth": 4})

    cox_methods = tripod_ai_checklist([cox_only])["methods"]
    tree_methods = tripod_ai_checklist([trees])["methods"]
    dl_methods = tripod_ai_checklist([_dl_mixed()])["methods"]

    assert "Random survival forests" not in cox_methods and "gradient boosting" not in cox_methods and "LASSO-Cox penalty" not in cox_methods
    # Without a depth the forests grow unlimited trees and boosting uses its default depth of 3.
    assert "Random survival forests used 100 trees without a depth limit; gradient boosting used 100 trees of maximum depth 3 with learning rate 0.1." in tree_methods
    assert "LASSO-Cox" not in tree_methods
    assert "Random survival forests used 100 trees of maximum depth 4; gradient boosting used 100 trees of maximum depth 4" in tripod_ai_checklist([deep])["methods"]
    assert (
        "stopping early after 10 epochs without improvement on a monitoring subset drawn from each training partition; each "
        "network was then refit on the whole training partition for the number of epochs early stopping selected."
    ) in dl_methods


# ── Other figures ────────────────────────────────────────────────


def _km_curve(group: str) -> dict:
    return {
        "group": group,
        "timeline": [0.0, 10.0, 20.0],
        "survival": [1.0, 0.8, 0.6],
        "ci_lower": [1.0, 0.7, 0.5],
        "ci_upper": [1.0, 0.9, 0.7],
        "censor_times": [],
        "censor_survival": [],
    }


def test_km_curves_keep_a_distinct_style_for_each_of_up_to_fifty_groups() -> None:
    plain = [f"G{index:02d}" for index in range(50)]
    with_high_and_low = ["High", "Low", *plain[:48]]

    for groups in (plain, with_high_and_low):
        figure = build_km_figure({"curves": [_km_curve(group) for group in groups], "test": None, "display_horizon": 20.0})
        curves = [trace for trace in figure["data"] if trace.get("name") in set(groups)]
        styles = {(trace["line"]["color"], trace["line"]["dash"], (trace.get("marker") or {}).get("symbol")) for trace in curves}
        assert len(curves) == len(styles) == len(groups)


def test_km_note_names_the_test_the_analysis_ran() -> None:
    from survival_toolkit.analysis import compute_km_analysis
    from survival_toolkit.sample_data import make_example_dataset

    analysis = compute_km_analysis(
        make_example_dataset(), "os_months", "os_event", group_column="treatment", event_positive_value=1,
        logrank_weight="fleming_harrington", fh_p=0.5,
    )
    weighted = _annotation_text(build_km_figure(analysis))
    # A result without the analysis's label keeps a readable name for the test.
    plain = _annotation_text(build_km_figure({"curves": [_km_curve("A")], "test": {"test": "logrank", "p_value": 0.04}, "display_horizon": 20.0}))

    assert "Fleming-Harrington (fh_p=0.5) test: p" in weighted
    assert "Log-rank test: p = 0.040" in plain


def test_time_dependent_heatmap_puts_the_most_important_feature_at_the_top() -> None:
    figure = build_time_dependent_importance_figure(
        {"features": ["age", "stage", "biomarker"], "eval_times": [12.0, 24.0], "importance_matrix": [[0.1, 0.2, 0.5], [0.2, 0.1, 0.4]]},
        top_n=3,
    )

    # Rows are listed most important first and the axis runs top-down, so the first row is drawn at the top.
    assert figure["data"][0]["y"] == ["biomarker", "age", "stage"]
    assert figure["layout"]["yaxis"]["autorange"] == "reversed"


def test_stability_figure_draws_the_coloured_tiers_over_the_grey_points_and_lists_them_first() -> None:
    def row(name: str, tier: str) -> dict:
        return {"marker": name, "tier": tier, "marginal": {"selection_frequency": 0.9, "direction_consistency": 1.0, "p_fwer": 0.01}}

    figure = build_marker_stability_figure(
        {
            "primary_lens": "marginal",
            "settings": {"alpha": 0.05, "robust_frequency": 0.5, "robust_direction": 0.9},
            "null": {"n_permutations": 99},
            "marker_table": [row("R1", "robust"), row("S1", "suggestive"), row("N1", "not supported")],
        }
    )

    # Plotly draws later traces on top; the legend follows the ranks.
    assert [trace["name"] for trace in figure["data"]] == ["not supported", "suggestive", "robust"]
    ranks = {trace["name"]: trace["legendrank"] for trace in figure["data"]}
    assert ranks["robust"] < ranks["suggestive"] < ranks["not supported"]


def test_cox_forest_names_the_terms_it_cannot_draw() -> None:
    figure = build_cox_forest_figure(
        {
            "results_table": [
                {"Label": "age", "Hazard ratio": 1.1, "CI lower": 1.01, "CI upper": 1.2, "P value": 0.01},
                {"Label": "stage: IV vs I", "Hazard ratio": float("inf"), "CI lower": 0.9, "CI upper": float("inf"), "P value": 0.02},
                {"Label": "grade", "Hazard ratio": None, "CI lower": None, "CI upper": None, "P value": None},
            ]
        }
    )

    assert figure["data"][0]["y"] == ["age"]
    assert "Not drawn (no finite hazard ratio or interval): stage: IV vs I, grade" in _annotation_text(figure)
    assert figure["layout"]["margin"]["b"] > 70


def test_loss_curve_best_epoch_skips_missing_monitor_values() -> None:
    nothing = build_loss_curve_figure([1.0, 0.9, 0.8], monitor_loss_history=[math.nan] * 3, monitor_label="Monitor C-index", monitor_goal="max")
    gap_min = build_loss_curve_figure([1.0, 0.9, 0.8], monitor_loss_history=[math.nan, 0.9, 0.8])
    gap_max = build_loss_curve_figure([1.0, 0.9, 0.8], monitor_loss_history=[math.nan, 0.61, 0.58], monitor_goal="max")

    assert "Best monitor epoch" not in _annotation_text(nothing) and not nothing["layout"].get("shapes")
    assert "Best monitor epoch: 3" in _annotation_text(gap_min)
    assert "Best monitor epoch: 2" in _annotation_text(gap_max)


def test_loss_curve_status_counts_epochs_in_the_right_number() -> None:
    assert "Trained for 1 epoch" == _annotation_text(build_loss_curve_figure([1.0], epochs_trained=1))
    assert "Trained for 3 epochs" == _annotation_text(build_loss_curve_figure([1.0, 0.9, 0.8], epochs_trained=3))

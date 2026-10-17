"""Explanations must never change diagnostic decisions or exported inference."""
import copy

from survival_toolkit.marker_review import marker_diagnostic_review
from survival_toolkit.reporting import marker_results_paragraph


def test_variance_explanation_preserves_the_whole_result_and_unknown_reasons():
    result = {"inference": {"status": "withheld", "allowed": False, "threshold": 0.01,
        "reasons": ["residual_misspecification: gene_A/variance", "residual_misspecification: gene_B/variance",
                    "residual_misspecification: gene_C/nonlinear_mean", "future_diagnostic: needs review"]},
        "marker_table": [{"added_value": {"p_value": None, "q_perm": None}, "tier": "inference withheld"}]}
    before = copy.deepcopy(result)
    review = marker_diagnostic_review(result)
    assert result == before
    assert review["flagged_variance_markers"] == ["gene_A", "gene_B"]
    assert review["flagged_mean_markers"] == ["gene_C"]
    assert "for 2 markers" in review["messages"][0]
    assert "future_diagnostic: needs review" in review["messages"]
    assert "do not prove" in review["interpretation"]
    assert not review["decision_changed"]
    text = marker_results_paragraph(result)
    assert "unequal spread" in text and "raw computations" in text
    assert all(reason in text for reason in result["inference"]["reasons"])


def test_failed_diagnostic_is_not_reported_as_detected_biological_misspecification():
    review = marker_diagnostic_review({"inference": {"status": "withheld", "allowed": False,
        "reasons": ["residual_diagnostic_failed: gene/variance", "clinical_diagnostic_failed: PH"]}})
    assert review["flagged_variance_markers"] == []
    assert review["flagged_mean_markers"] == []
    assert len(review["messages"]) == 1 and "could not be calculated" in review["messages"][0]


def test_passed_or_legacy_result_is_never_given_a_new_hold_or_certification():
    for inference in ({}, {"status": "assumption_dependent", "allowed": True, "reasons": []}):
        result = {"inference": inference}
        before = copy.deepcopy(result)
        review = marker_diagnostic_review(result)
        assert result == before and review["messages"] == [] and review["next_steps"] == []
        assert "do not certify" in review["interpretation"]


def test_family_hold_distinguishes_triggering_markers_from_held_selections():
    result = {"inference": {"status": "withheld", "allowed": False, "threshold": 0.01,
        "reasons": ["residual_misspecification: gene_A/variance",
                    "residual_misspecification: gene_A/nonlinear_mean",
                    "residual_misspecification: gene_B/variance"]},
        "cohort": {"n_markers_evaluated": 100},
        "resampling": {"n_valid": 20, "n_failed": 0, "n_withheld": 20}}
    before = copy.deepcopy(result)
    review = marker_diagnostic_review(result)
    assert result == before
    assert review["n_markers_with_residual_findings"] == 2
    assert review["n_markers_evaluated"] == 100
    assert review["hold_scope"] == "entire_marker_family"
    assert review["selection_interpretation"] == "all_training_subsamples_withheld"
    assert "every marker failed" in " ".join(review["messages"])
    assert "All 20 usable training subsamples" in marker_results_paragraph(result)
    assert "not evidence" in marker_results_paragraph(result)


def test_partial_or_absent_subsamples_are_not_described_as_all_held():
    for resampling in ({}, {"n_valid": 0, "n_withheld": 0}, {"n_valid": 20, "n_withheld": 19}):
        result = {"inference": {"status": "withheld", "allowed": False,
            "reasons": ["residual_diagnostic_failed: gene/variance"]}, "resampling": resampling}
        review = marker_diagnostic_review(result)
        assert review["selection_interpretation"] == "see_resampling"
        assert review["n_markers_with_residual_findings"] == 0
        assert not any("All " in text for text in review["messages"])


def test_csv_rows_keep_family_hold_and_zero_selection_interpretation():
    from survival_toolkit.app import _marker_display_rows

    stats = {"rank_interval": [1, 2], "p_value": None, "p_fwer": None, "q_perm": None,
        "selection_frequency": 0.0, "direction_consistency": 0.95, "median_rank": 1.0}
    result = {"primary_lens": "added_value", "inference": {"status": "withheld", "allowed": False,
        "reasons": ["residual_misspecification: gene_A/variance"]},
        "resampling": {"n_valid": 20, "n_withheld": 20},
        "marker_table": [{"marker": name, "inference_status": "withheld", "tier": "inference withheld",
            "pattern": "unavailable", "direction": "+", "added_value": copy.deepcopy(stats),
            "marginal": {"p_value": 0.01}} for name in ["gene_A", "gene_without_residual_finding"]]}
    before = copy.deepcopy(result)
    rows = _marker_display_rows(result)
    assert result == before
    assert len(rows) == 2
    for row in rows:
        assert row["Diagnostic hold scope"] == "entire_marker_family"
        assert row["Selection interpretation"] == "all_training_subsamples_withheld"
        assert row["Family-wise P"] is None and row["Selection frequency"] == 0.0


def test_exploratory_score_rank_is_not_the_subsample_median_rank():
    from survival_toolkit.app import _marker_display_rows
    stats = {"rank_interval": [1, 20], "p_value": None, "p_fwer": None, "q_perm": None,
        "selection_frequency": 0.0, "direction_consistency": 0.95, "median_rank": 7.0}
    result = {"primary_lens": "added_value", "inference": {"status": "withheld", "allowed": False},
        "marker_table": [{"marker": name, "inference_status": "withheld", "tier": "inference withheld",
            "pattern": "unavailable", "direction": "+", "added_value": {**stats, "chi2": score},
            "marginal": {"p_value": 0.01}} for name, score in [("B", 3.0), ("C", None), ("A", 3.0)]]}
    before = copy.deepcopy(result)
    rows = _marker_display_rows(result)
    assert result == before
    assert [row["Marker"] for row in rows] == ["A", "B", "C"]
    assert [row["Full-cohort score rank"] for row in rows] == [1, 2, None]
    assert all(row["Median rank"] == 7.0 and row["Family-wise P"] is None for row in rows)

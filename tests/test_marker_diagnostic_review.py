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

"""The short marker interpretation must retain uncertainty and safety stops."""
from pathlib import Path
import json
import shutil

import pytest

from test_frontend_review import _run_page
from survival_toolkit.marker_review import marker_diagnostic_review

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")


def test_family_hold_overview_separates_residual_findings_from_exploratory_ranks(tmp_path: Path):
    analysis = {"primary_lens": "added_value", "inference": {"status": "withheld", "allowed": False,
        "reasons": ["residual_misspecification: gene_A/variance", "residual_misspecification: gene_B/variance"]},
        "cohort": {"n": 80, "n_markers_evaluated": 100},
        "resampling": {"n_valid": 20, "n_withheld": 20, "stability_assessed": True}}
    analysis["diagnostic_review"] = marker_diagnostic_review(analysis)
    script = "const payload = " + json.dumps({"analysis": analysis}) + ";" + r"""
      page.context.__payload = payload;
      const before = JSON.stringify(payload);
      const overview = page.run('markerOverviewSummary(__payload)');
      return {overview, unchanged: before === JSON.stringify(payload)};
    """
    result = _run_page(tmp_path, script)
    overview = result["overview"]
    assert result["unchanged"]
    assert "family" in overview["headline"] and "exploratory" in overview["headline"]
    metrics = {v["label"]: v["value"] for v in overview["metrics"]}
    assert metrics["Markers with residual findings"] == "2 of 100"
    assert metrics["Stable candidates"] == "Not assessed"
    assert "not evidence" in " ".join(overview["cautions"])
    assert not overview["strengths"]


def test_wide_diagnostic_table_prioritizes_findings_without_truncating_saved_tests(tmp_path: Path):
    result = _run_page(tmp_path, r"""
      const inference = { threshold: 0.01, clinical_tests: [{name: 'PH', p_value: 0.2, p_holm: 0.5}],
        residual_tests: Array.from({length: 600}, (_, i) => ({marker: `gene_${i}`, diagnostic: 'variance',
          status: i === 599 ? 'failed' : 'calculated', p_value: i >= 579 ? 0.000001 : 0.8,
          p_holm: i >= 579 ? 0.001 : 1.0})) };
      page.context.__inference = inference;
      const before = JSON.stringify(inference);
      const selected = page.run('markerDiagnosticTableTests(__inference)');
      return {selected, unchanged: before === JSON.stringify(inference)};
    """)
    assert result["unchanged"]
    selected = result["selected"]
    assert selected["total"] == 601 and len(selected["visible"]) == 300
    names = {row.get("marker") for row in selected["visible"]}
    assert {f"gene_{i}" for i in range(579, 600)} <= names
    assert any(row.get("name") == "PH" for row in selected["visible"])


_PAYLOAD = r"""
  const payload = { analysis: {
    primary_lens: 'added_value', inference: {status: 'assumption_dependent', allowed: true},
    cohort: {n: 360, events: 259, n_markers_evaluated: 2},
    tier_counts: {robust: 1, suggestive: 1},
    null: {n_permutations: 1000}, resampling: {n_valid: 200, stability_assessed: true},
    signature: {markers: ['marker_A'], apparent_c: 0.673, optimism_corrected_c: 0.651,
      signature_c_left_out: 0.656, clinical_c_left_out: 0.629,
      signature_gain_left_out: 0.027, signature_gain_left_out_ci: [0.001, 0.052]},
  }, display_table: [{Marker: 'marker_A', Tier: 'robust'}] };
"""


def test_overview_compares_paired_scores_and_keeps_the_payload_unchanged(tmp_path: Path):
    result = _run_page(tmp_path, _PAYLOAD + r"""
      page.context.__payload = payload;
      const before = JSON.stringify(payload);
      const overview = page.run('markerOverviewSummary(__payload)');
      return {overview, unchanged: before === JSON.stringify(payload)};
    """)
    summary = result["overview"]
    metrics = {row["label"]: row["value"] for row in summary["metrics"]}
    assert metrics == {"Stable candidates (internal)": "1 of 2", "Clinical only · C-index": "0.629",
                       "Marker procedure · C-index": "0.656", "Change in C-index": "+0.027"}
    assert "marker_A" in " ".join(summary["cautions"])
    assert "95% CI 0.001 to 0.052" in " ".join(summary["cautions"])
    assert "Independent external validation is needed" in " ".join(summary["cautions"])
    assert result["unchanged"]


def test_withheld_results_never_promote_raw_candidates_or_internal_scores(tmp_path: Path):
    result = _run_page(tmp_path, _PAYLOAD + r"""
      payload.analysis.inference = {status: 'withheld', allowed: false,
        reasons: ['Residual diagnostic failed'], interpretation: 'Do not interpret added-value tests.'};
      page.context.__payload = payload;
      return page.run('markerOverviewSummary(__payload)');
    """)
    assert "inference is withheld" in result["headline"]
    assert [row["value"] for row in result["metrics"]] == ["Not assessed", "Unavailable"]
    assert "Residual diagnostic failed" in result["cautions"]
    assert "0.656" not in str(result) and "marker_A" not in str(result)


def test_missing_scores_and_zero_scores_remain_distinct(tmp_path: Path):
    result = _run_page(tmp_path, _PAYLOAD + r"""
      page.context.__payload = payload;
      payload.analysis.signature.signature_c_left_out = null;
      const missing = page.run('markerOverviewSummary(__payload)');
      payload.analysis.signature.signature_c_left_out = 0;
      payload.analysis.signature.clinical_c_left_out = 0;
      payload.analysis.signature.signature_gain_left_out = 0;
      payload.analysis.signature.signature_gain_left_out_ci = null;
      const zero = page.run('markerOverviewSummary(__payload)');
      return {missing, zero};
    """)
    assert result["missing"]["metrics"][-1]["value"] == "Unavailable"
    assert [row["value"] for row in result["zero"]["metrics"]][1:] == ["0.000", "0.000", "+0.000"]
    assert "No uncertainty interval" in " ".join(result["zero"]["cautions"])


def test_withheld_review_explains_spread_without_hiding_unknown_reasons(tmp_path: Path):
    result = _run_page(tmp_path, _PAYLOAD + r"""
      payload.analysis.inference = {status: 'withheld', allowed: false,
        reasons: ['residual_misspecification: gene_A/variance', 'unmapped finding'],
        interpretation: 'Original interpretation.'};
      payload.analysis.diagnostic_review = {
        raw_reasons: ['residual_misspecification: gene_A/variance'],
        messages: ['Residual spread changes with clinical information for 1 marker.'],
        interpretation: 'This does not prove that the input is wrong.',
        next_steps: ['Review the source and record the revised analysis first.'],
      };
      page.context.__payload = payload;
      const before = JSON.stringify(payload);
      const overview = page.run('markerOverviewSummary(__payload)');
      return {overview, unchanged: before === JSON.stringify(payload)};
    """)
    overview = result["overview"]
    assert result["unchanged"]
    assert [row["value"] for row in overview["metrics"]] == ["Not assessed", "Unavailable"]
    assert "unmapped finding" in overview["cautions"]
    assert "residual_misspecification: gene_A/variance" not in overview["cautions"]
    assert "does not prove" in " ".join(overview["cautions"])
    assert overview["next_steps"] == ["Review the source and record the revised analysis first."]


def test_unassessed_stability_and_marginal_results_are_not_promoted(tmp_path: Path):
    result = _run_page(tmp_path, _PAYLOAD + r"""
      page.context.__payload = payload;
      payload.analysis.resampling = {n_valid: 0, stability_assessed: false};
      const stability = page.run('markerOverviewSummary(__payload)');
      payload.analysis.null.n_permutations = 0;
      const untested = page.run('markerOverviewSummary(__payload)');
      payload.analysis.primary_lens = 'marginal';
      const marginal = page.run('markerOverviewSummary(__payload)');
      return {stability, untested, marginal};
    """)
    assert result["stability"]["metrics"][0]["value"] == "Not assessed"
    assert "No permutation tests" in " ".join(result["untested"]["cautions"])
    assert result["marginal"]["metrics"][-1]["value"] == "Unavailable"
    assert "improvement over clinical information was not assessed" in " ".join(result["marginal"]["cautions"])


def test_external_horizon_warning_never_promotes_raw_calibration_scores(tmp_path: Path):
    rows = _run_page(tmp_path, r"""
      page.context.__validation = {metrics: {c_index: 0.63, calibration_slope: 0.91,
        horizon: 5, observed_expected_ratio: 0.9, brier_skill: 0.4},
        absolute_risk_review: {status: 'beyond_validation_followup'}};
      return page.run('markerValidationMetrics(__validation)');
    """)
    assert any(row["label"] == "Absolute-risk evaluation" and "Unavailable" in row["value"] for row in rows)
    assert any(row["label"] == "C-index" for row in rows)
    assert any(row["label"] == "Calibration slope" for row in rows)
    assert not any("Observed/expected" in row["label"] or "Brier" in row["label"] for row in rows)


def test_uncertainty_repeated_patients_and_legacy_diagnostics_stay_visible(tmp_path: Path):
    result = _run_page(tmp_path, _PAYLOAD + r"""
      payload.analysis.signature.signature_gain_left_out_ci = [-0.01, 0.052];
      payload.analysis.duplicates = {n_identical: 1, identical: [['P1', 'P2']]};
      delete payload.analysis.inference;
      page.context.__payload = payload;
      return page.run('markerOverviewSummary(__payload)');
    """)
    text = " ".join(result["cautions"])
    assert result["status"] == "review"
    assert "Possible repeated patients" in text
    assert "Prediction improvement is uncertain" in text
    assert "Diagnostics were not assessed" in text

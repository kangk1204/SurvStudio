"""The short marker interpretation must retain uncertainty and safety stops."""
from pathlib import Path
import shutil

import pytest

from test_frontend_review import _run_page

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")


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

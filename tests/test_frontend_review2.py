"""Behaviour tests of the front end, second review round.

The real index.html and app scripts run in Node with the small DOM of tests/test_frontend_review.py
(imported from there), and each test drives the page the way a user would.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from test_frontend_review import (  # noqa: F401 (pytest fixtures used by name)
    _compare_all_script,
    _locked_test_compare,
    _run_page,
    compare_payloads,
    example_dataset,
    marker_payloads,
)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")


# ── Prediction leaderboard ──────────────────────────────────────


def _stale_board_script(extra: str) -> str:
    """Compare All on both families, then a DL-only setting changes: the board shows the last snapshot."""
    return _compare_all_script(r"""
      page.change("#dlLearningRate", "0.002");
      await page.settle(10);
      page.run("renderBenchmarkBoard()");
      await page.settle(10);
      const rows = () => page.run(`refs.benchmarkComparisonShell.querySelectorAll('tbody tr')
        .map((tr) => [tr.children[0].textContent.trim(), tr.children[2].textContent.trim()])`);
    """ + extra)


def test_stale_reference_board_is_in_screen_rank_order(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R15-1: the stale snapshot is ranked like a current board (DL's DeepHit leads), not listed ML first."""
    result = _run_page(tmp_path, _stale_board_script(r"""
      return {
        stale: page.run("benchmarkBoardState().showingStaleBoard"),
        rows: rows(),
        ranking: page.run("benchmarkBoardState().rankingRows.map((row) => row.model)"),
        order: page.run("refs.benchmarkComparisonPlot.layout.yaxis.categoryarray"),
      };
    """), dataset=example_dataset, compare=compare_payloads)

    assert result["stale"] is True
    assert result["rows"] == [
        ["1", "DeepHit"],
        ["2", "Random Survival Forest"],
        ["3", "DeepSurv"],
        ["4", "LASSO-Cox"],
        ["5", "Cox PH"],
    ]
    assert result["ranking"] == ["DeepHit", "Random Survival Forest", "DeepSurv", "LASSO-Cox", "Cox PH"]
    # Plotly lists the first category at the bottom: rank 1 is last, i.e. on top, and marked.
    assert result["order"][-1] == "DeepHit (DL) · rank 1"
    assert result["order"][0] == "Cox PH (ML)"
    assert sum("rank 1" in label for label in result["order"]) == 1


def test_stale_locked_test_board_notes_the_rank_one_model_it_ranks_first(
    tmp_path: Path, example_dataset: dict, compare_payloads: dict
) -> None:
    """R15-1: the lead note about the rank-1 model's failed locked-test refit follows the ranked order."""
    compare = _locked_test_compare(compare_payloads, failed=False)
    deephit = compare["dl"]["comparison_table"][0]
    deephit.update({"c_index": 0.76, "locked_test_c_index": None, "locked_test_error": "the refit did not converge"})
    result = _run_page(tmp_path, _stale_board_script(r"""
      return {
        stale: page.run("benchmarkBoardState().showingStaleBoard"),
        rows: rows(),
        note: page.run("refs.benchmarkTableNote.textContent"),
      };
    """), dataset=example_dataset, compare=compare)

    assert result["stale"] is True
    assert result["rows"][0] == ["1", "DeepHit"]
    assert "The locked-test refit of the rank-1 model (DeepHit) failed" in result["note"]


# ── Result mode after a failed run ──────────────────────────────


_DL_SINGLE_HANDLER = r"""
  const dlSingle = (request) => ({ status: 200, body: {
    figures: { importance: { data: [{ type: "bar", x: [0.3], y: ["age"] }], layout: { height: 360 } },
               loss: { data: [{ x: [1, 2], y: [1, 0.5] }], layout: { height: 360 } } },
    analysis: { c_index: 0.7, evaluation_mode: "holdout", epochs_trained: 20 },
    request_config: request.json(),
  } });
  const mlSingle = (request) => ({ status: 200, body: {
    importance_figure: { data: [{ type: "bar", x: [0.3], y: ["age"] }], layout: { height: 360 } },
    analysis: { model_stats: { c_index: 0.7, evaluation_mode: "holdout", n_patients: 360, n_features: 4 } },
    request_config: request.json(),
  } });
"""


def test_a_failed_compare_all_phase_keeps_the_single_result_current_and_shown(
    tmp_path: Path, example_dataset: dict, compare_payloads: dict
) -> None:
    """R13-3/R15-3: after the DL phase fails, the restored DeepSurv result is current, not stale, and shown."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _DL_SINGLE_HANDLER + r"""
      page.fetchHandler = (request) => (request.url.endsWith("/api/deep-model") ? dlSingle(request) : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('benchmark'); reviewBenchmarkModel('deepsurv', 'single'); refs.runPredictiveWorkbenchButton.click()");
      await page.settle();
      page.run("closePredictiveWorkbench()");
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/ml-model")) return { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } };
        if (request.url.endsWith("/api/deep-model")) return { status: 400, body: { detail: "All deep models failed to train." } };
        if (request.url.endsWith("/api/model-comparison-intervals")) return { status: 200, body: fixtures.compare.intervals };
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("refs.runPredictiveCompareAllButton.click()");
      await page.settle(60);
      const after = page.run(`({ single: panelModeForPayload(state.dl) === "single", mode: preferredResultMode("dl"),
        current: Boolean(currentGoalResult("dl")), stale: refs.dlImportancePlot.classList.contains("plot-stale") })`);
      page.run("reviewBenchmarkModel('deepsurv', 'single')");
      return {
        after,
        shown: page.run("Boolean(selectedPredictiveSingleResult('dl'))"),
        gridHidden: page.run("refs.dlImportancePlot.closest('.ml-plots-grid').classList.contains('hidden')"),
        plotHidden: page.run("refs.dlImportancePlot.classList.contains('result-hidden')"),
      };
    """, dataset=example_dataset, compare=compare_payloads)

    assert result == {
        "after": {"single": True, "mode": "single", "current": True, "stale": False},
        "shown": True,
        "gridHidden": False,
        "plotHidden": False,
    }


def test_a_failed_run_of_one_mode_keeps_the_result_of_the_other_mode(
    tmp_path: Path, example_dataset: dict, compare_payloads: dict
) -> None:
    """R13-3: a failed ML comparison leaves the RSF result current; a failed RSF run leaves the comparison current."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _DL_SINGLE_HANDLER + r"""
      const snapshot = () => page.run(`({ mode: preferredResultMode("ml"), current: Boolean(currentGoalResult("ml")),
        stale: refs.mlImportancePlot.classList.contains("plot-stale"), hidden: refs.mlImportancePlot.classList.contains("result-hidden") })`);
      page.fetchHandler = (request) => (request.url.endsWith("/api/ml-model") ? mlSingle(request) : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('benchmark'); reviewBenchmarkModel('rsf', 'single'); refs.runPredictiveWorkbenchButton.click()");
      await page.settle();
      const trained = snapshot();
      page.fetchHandler = () => ({ status: 500, body: { detail: "compare failed on the server" } });
      await page.run("withLoading(refs.runCompareButton, runCompareModels)");
      await page.settle();
      const afterFailedCompare = snapshot();
      page.fetchHandler = (request) => (request.url.endsWith("/api/ml-model")
        ? { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } }
        : { status: 500, body: { detail: "unexpected" } });
      await page.run("withLoading(refs.runCompareButton, runCompareModels)");
      await page.settle();
      page.fetchHandler = () => ({ status: 500, body: { detail: "the forest failed" } });
      await page.run("withLoading(refs.runMlButton, runMlModel)");
      await page.settle();
      return {
        trained,
        afterFailedCompare,
        afterFailedSingle: page.run(`({ mode: preferredResultMode("ml"), current: Boolean(currentGoalResult("ml")),
          comparison: Boolean(currentGoalResult("ml")?.analysis?.comparison_table) })`),
      };
    """, dataset=example_dataset, compare=compare_payloads)

    assert result["trained"] == {"mode": "single", "current": True, "stale": False, "hidden": False}
    assert result["afterFailedCompare"] == result["trained"]
    assert result["afterFailedSingle"] == {"mode": "compare", "current": True, "comparison": True}


# ── Runtime banner ──────────────────────────────────────────────


def test_the_compare_all_banner_stays_up_through_both_phases(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R13-5/R15-10: the phases (each started through withLoading) do not clear or replace the Compare All banner."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const mlHold = deferred();
      const dlHold = deferred();
      page.fetchHandler = (request) => {
        const body = request.json();
        if (request.url.endsWith("/api/ml-model")) return mlHold.promise.then(() => ({ status: 200, body: { analysis: fixtures.compare.ml, request_config: body } }));
        if (request.url.endsWith("/api/deep-model")) return dlHold.promise.then(() => ({ status: 200, body: { analysis: fixtures.compare.dl, request_config: body } }));
        if (request.url.endsWith("/api/model-comparison-intervals")) return { status: 200, body: fixtures.compare.intervals };
        return { status: 500, body: { detail: "unexpected" } };
      };
      const banner = () => page.run("refs.runtimeBanner.textContent");
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(3);
      const duringMl = banner();
      mlHold.resolve();
      await page.settle(10);
      const duringDl = banner();
      dlHold.resolve();
      await page.settle(40);
      return { duringMl, duringDl, after: page.run("refs.runtimeBanner.className") };
    """, dataset=example_dataset, compare=compare_payloads)

    assert result["duringMl"].startswith("Comparing the full predictive stack")
    assert result["duringDl"] == result["duringMl"]
    assert result["after"] == "runtime-banner hidden"


def test_starting_a_run_keeps_the_banner_of_a_run_in_flight_and_clears_a_notice(
    tmp_path: Path, example_dataset: dict, marker_payloads: dict
) -> None:
    """R13-5/R15-10: withLoading clears a finished load's notice, never the progress banner of another run."""
    result = _run_page(tmp_path, r"""
      const km = (request) => ({ status: 200, body: {
        analysis: { summary_table: [], risk_table: { rows: [], columns: [] }, pairwise_table: [], cohort: { n: 360 } },
        figure: { data: [{ x: [0], y: [1] }], layout: {} }, request_config: request.json() } });
      page.run("refs.datasetFile.files = [{ name: 'cohort.csv' }]");
      page.fetchHandler = (request) => (request.url.endsWith("/api/upload") ? { status: 200, body: fixtures.dataset } : km(request));
      page.change("#datasetFile");
      await page.settle();
      const notice = page.run("refs.runtimeBanner.textContent");
      page.run("refs.runKmButton.click()");
      await page.settle();
      const afterKm = page.run("refs.runtimeBanner.className");
      const hold = deferred();
      page.fetchHandler = (request) => (request.url.endsWith("/api/marker-evaluation")
        ? hold.promise.then(() => ({ status: 200, body: { ...fixtures.markers.added, request_config: request.json() } }))
        : km(request));
      page.run("activateTab('markers'); refs.runMarkersButton.click()");
      await page.settle(3);
      const during = page.run("refs.runtimeBanner.textContent");
      page.run("activateTab('km'); refs.runKmButton.click()");
      await page.settle();
      const duringAfterKm = page.run("refs.runtimeBanner.textContent");
      hold.resolve();
      await page.settle();
      return { notice, afterKm, during, duringAfterKm, after: page.run("refs.runtimeBanner.className") };
    """, dataset=example_dataset, markers=marker_payloads)

    assert " loaded (360 rows" in result["notice"]
    assert result["afterKm"] == "runtime-banner hidden"
    assert result["during"].startswith("Evaluating 2 marker(s)")
    assert result["duringAfterKm"] == result["during"]
    assert result["after"] == "runtime-banner hidden"

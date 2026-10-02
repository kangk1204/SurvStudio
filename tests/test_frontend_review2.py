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
    _signature_script,
    client,
    compare_payloads,
    example_dataset,
    marker_payloads,
)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")


def _synthetic_dataset() -> dict:
    """A small dataset payload with a second 0/1 column ("status_code") that is not a standard event name."""
    n = 40
    rows = [
        {
            "patient_id": f"P{index + 1}",
            "os_months": index % 17 + 1.5,
            "os_event": 0 if index % 3 == 0 else 1,
            "age": 40 + index % 30,
            "biomarker": (index % 11) / 10,
            "sex": "M" if index % 2 else "F",
            "status_code": 0 if index % 4 == 0 else 1,
        }
        for index in range(n)
    ]

    def column(name: str, kind: str, unique: int, preview: list) -> dict:
        return {"name": name, "kind": kind, "n_unique": unique, "unique_preview": preview, "missing": 0, "non_missing": n}

    return {
        "dataset_id": "synthetic-1",
        "filename": "synthetic.csv",
        "n_rows": n,
        "n_columns": 7,
        "dataset_hash": "hash-1",
        "columns": [
            column("patient_id", "categorical", n, ["P1", "P2", "P3"]),
            column("os_months", "numeric", 17, [1.5, 2.5, 3.5]),
            column("os_event", "binary", 2, [0, 1]),
            column("age", "numeric", 30, [40, 41, 42]),
            column("biomarker", "numeric", 11, [0, 0.1, 0.2]),
            column("sex", "categorical", 2, ["F", "M"]),
            column("status_code", "binary", 2, [0, 1]),
        ],
        "numeric_columns": ["os_months", "os_event", "age", "biomarker", "status_code"],
        "binary_candidate_columns": ["os_event", "sex", "status_code"],
        "suggestions": {"time_columns": ["os_months"], "event_columns": ["os_event"]},
        "preview": rows[:5],
        "derived_column_provenance": {},
    }


_KM_HANDLER = r"""
  const km = (request) => ({ status: 200, body: {
    analysis: { summary_table: [], risk_table: { rows: [], columns: [] }, pairwise_table: [], cohort: { n: 360 } },
    figure: { data: [{ x: [0], y: [1] }], layout: {} }, request_config: request.json() } });
"""


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


def test_a_single_model_run_does_not_hide_the_leaderboard(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R15-7: only Compare All puts the board in its "in progress" state; a DeepSurv run leaves it shown."""
    result = _run_page(tmp_path, _compare_all_script(r"""
      const hold = deferred();
      const board = () => page.run(`({ rows: refs.benchmarkComparisonShell.querySelectorAll("tbody tr").length,
        summary: refs.benchmarkSummaryGrid.textContent.replace(/\s+/g, " ").trim(), plotHidden: refs.benchmarkComparisonPlot.classList.contains("hidden") })`);
      const before = board();
      page.fetchHandler = (request) => (request.url.endsWith("/api/deep-model")
        ? hold.promise.then(() => ({ status: 500, body: { detail: "stopped" } }))
        : { status: 500, body: { detail: "unexpected" } });
      page.run("reviewBenchmarkModel('deepsurv', 'single'); refs.runPredictiveWorkbenchButton.click()");
      await page.settle(3);
      page.run("closePredictiveWorkbench()");
      await page.settle(3);
      const during = board();
      hold.resolve();
      await page.settle();
      return { before, during };
    """), dataset=example_dataset, compare=compare_payloads)

    assert result["before"]["rows"] == 5
    assert result["during"]["rows"] == 5
    assert result["during"]["plotHidden"] is False
    assert "still running" not in result["during"]["summary"]
    assert "in progress" not in result["during"]["summary"]


def test_a_board_without_cox_ph_shows_no_delta_c_column(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R15-8: the ΔC column and its note follow the intervals' reference; a DL-only board has none."""
    dl_only = client.post("/api/model-comparison-intervals", json={"predictions": [compare_payloads["dl"]["test_predictions"]]})
    assert dl_only.status_code == 200, dl_only.text
    assert dl_only.json()["reference"] is None
    # The server says why there is no reference (and the note is data: it must stay inert text).
    reference_note = "Cox PH is not among these <models>, so no difference to a reference is reported."
    dl_only_payload = {**dl_only.json(), "reference_note": reference_note}
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/ml-model")) return { status: 400, body: { detail: "All models failed to train." } };
        if (request.url.endsWith("/api/deep-model")) return { status: 200, body: { analysis: fixtures.compare.dl, request_config: request.json() } };
        if (request.url.endsWith("/api/model-comparison-intervals")) return { status: 200, body: fixtures.dlOnly };
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(60);
      return {
        headers: page.run("refs.benchmarkComparisonShell.querySelectorAll('th').map((th) => th.textContent.trim())"),
        cells: page.run("refs.benchmarkComparisonShell.querySelectorAll('tbody tr')[0].children.length"),
        note: page.run("refs.benchmarkTableNote.textContent"),
        noteHtml: page.run("refs.benchmarkTableNote.innerHTML"),
        shapes: page.run("(refs.benchmarkComparisonPlot.layout?.shapes || []).length"),
      };
    """, dataset=example_dataset, compare=compare_payloads, dlOnly=dl_only_payload)

    assert "95% CI" in result["headers"]
    assert not any("ΔC" in header for header in result["headers"])
    assert result["cells"] == len(result["headers"])
    assert "ΔC vs Cox PH is paired" not in result["note"]
    assert reference_note in result["note"]
    assert "&lt;models&gt;" in result["noteHtml"]

    with_cox = _run_page(tmp_path, _compare_all_script(r"""
      return {
        headers: headers(),
        cells: page.run("refs.benchmarkComparisonShell.querySelectorAll('tbody tr')[0].children.length"),
        note: page.run("refs.benchmarkTableNote.textContent"),
      };
    """), dataset=example_dataset, compare=compare_payloads)
    assert "ΔC vs Cox PH (95% CI)" in with_cox["headers"]
    assert with_cox["cells"] == len(with_cox["headers"])
    assert "ΔC vs Cox PH is paired" in with_cox["note"]


def _interval_failure_script(failure: str) -> str:
    """Compare All whose first interval request fails as `failure` returns; later ones succeed."""
    return r"""
      await loadDataset(page, fixtures.dataset);
      let intervalCalls = 0;
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/ml-model")) return { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } };
        if (request.url.endsWith("/api/deep-model")) return { status: 200, body: { analysis: fixtures.compare.dl, request_config: request.json() } };
        if (request.url.endsWith("/api/model-comparison-intervals")) {
          intervalCalls += 1;
          return intervalCalls === 1 ? """ + failure + r""" : { status: 200, body: fixtures.compare.intervals };
        }
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(60);
      const afterFailure = page.run("runtime.benchmarkIntervals.status");
      // Past any retry pause: the board is rendered again.
      page.run("runtime.benchmarkIntervals.retryAt = 0; renderBenchmarkBoard();");
      await page.settle();
      return { afterFailure, after: page.run("runtime.benchmarkIntervals.status"), calls: intervalCalls };
    """


@pytest.mark.parametrize(
    ("failure", "retried"),
    [
        ('{ status: 422, body: { detail: "Every test patient needs one event indicator of 0 or 1." } }', False),
        ('Promise.reject(new TypeError("Failed to fetch"))', True),
        ('{ status: 502, body: { detail: "bad gateway" } }', True),
    ],
)
def test_interval_requests_are_retried_only_after_network_or_server_errors(
    tmp_path: Path, example_dataset: dict, compare_payloads: dict, failure: str, retried: bool
) -> None:
    """R15-9: a request the server refused (4xx) would be refused again, so it is not retried."""
    result = _run_page(tmp_path, _interval_failure_script(failure), dataset=example_dataset, compare=compare_payloads)

    assert result["afterFailure"] == "error"
    assert result == {"afterFailure": "error", "after": "ready" if retried else "error", "calls": 2 if retried else 1}


def test_the_locked_test_ranking_note_is_the_same_in_summary_and_table(
    tmp_path: Path, example_dataset: dict, compare_payloads: dict
) -> None:
    """R15-14: with the cross-family ranking withheld, both notes speak of each family's rank-1 model."""
    compare = _locked_test_compare(compare_payloads, failed=False)
    compare["dl"]["evaluation_split_fingerprint"] = "another-split"
    result = _run_page(tmp_path, _compare_all_script(r"""
      return {
        withheld: page.run("benchmarkBoardState().withholdCrossFamilyRanking"),
        summary: page.run("refs.benchmarkSummaryGrid.textContent"),
        note: page.run("refs.benchmarkTableNote.textContent"),
      };
    """), dataset=example_dataset, compare=compare)

    assert result["withheld"] is True
    for text in (result["summary"], result["note"]):
        assert "the locked-test C-index of the rank-1 model is the independent estimate" not in text
        assert "Within each family, ranking uses development-set cross-validation" in text


def test_an_unranked_locked_test_comparison_names_no_rank_one_model(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """Without a complete CV aggregate no model is ranked, so neither the note nor the banner names rows[0] as rank 1."""
    compare = dict(compare_payloads)
    ml = dict(compare_payloads["ml"])
    ml.pop("test_predictions")
    ml.update({"evaluation_mode": "repeated_cv_incomplete", "n_development_patients": 250, "n_locked_test_patients": 110, "n_locked_test_events": 50})
    ml["comparison_table"] = [
        {"model": model, "c_index": None, "rank": None, "comparable_for_ranking": False, "locked_test_c_index": test,
         "evaluation_mode": "repeated_cv_incomplete"}
        for model, test in (("Random Survival Forest", 0.66), ("Cox PH", 0.64))
    ]
    compare["ml"] = ml
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = (request) => (request.url.endsWith("/api/ml-model")
        ? { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } }
        : { status: 500, body: { detail: "unexpected" } });
      await page.run("withLoading(refs.runCompareButton, runCompareModels)");
      await page.settle();
      return {
        note: page.run("refs.mlComparisonShell.querySelector('.comparison-table-note').textContent"),
        banner: page.run("refs.mlMetaBanner.textContent"),
      };
    """, dataset=example_dataset, compare=compare)

    assert "rank-1 model (Random Survival Forest" not in result["note"]
    assert "No model had a complete cross-validation aggregate, so none was selected" in result["note"]
    assert "Random Survival Forest" not in result["banner"]
    assert result["banner"].startswith("No model was CV-selected (no complete cross-validation aggregate)")
    assert "rank-1" not in result["banner"]


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


def test_a_finished_ml_run_does_not_pull_the_workbench_away_from_deep_learning(tmp_path: Path, example_dataset: dict) -> None:
    """R13-6: the user moved on to DeepSurv's controls while RSF trained; the view stays there and a toast tells."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _DL_SINGLE_HANDLER + r"""
      const hold = deferred();
      page.fetchHandler = (request) => (request.url.endsWith("/api/ml-model") ? hold.promise.then(() => mlSingle(request)) : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('benchmark'); reviewBenchmarkModel('rsf', 'single'); refs.runPredictiveWorkbenchButton.click()");
      await page.settle(3);
      page.run("reviewBenchmarkModel('deepsurv', 'single')");
      await page.settle(3);
      hold.resolve();
      await page.settle();
      return {
        family: page.run("runtime.predictiveFamily"),
        selector: page.run("refs.predictiveModelSelector.value"),
        tab: page.run("activeTabName()"),
        trained: page.run("panelModeForPayload(state.ml)"),
        toasts: page.toasts(),
      };
    """, dataset=example_dataset)

    assert result["family"] == "dl"
    assert result["selector"] == "deepsurv"
    assert result["tab"] == "benchmark"
    assert result["trained"] == "single"
    assert any("Random Survival Forest model finished in the background" in toast for toast in result["toasts"]), result["toasts"]


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


# ── Compare All ─────────────────────────────────────────────────


_FAMILY_RUN_BUTTONS = "['runMlButton', 'runCompareButton', 'runCompareInlineButton', 'runDlButton', 'runDlCompareButton', 'runDlCompareInlineButton']"


def test_family_run_buttons_stay_off_while_compare_all_runs(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R15-4: the ML and DL panels' Run buttons cannot start a run that would take over a Compare All phase."""
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
      const enabled = () => page.run(`""" + _FAMILY_RUN_BUTTONS + r""".filter((key) => !refs[key].disabled)`);
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(3);
      const duringMl = enabled();
      mlHold.resolve();
      await page.settle(10);
      const duringDl = enabled();
      dlHold.resolve();
      await page.settle(40);
      return { duringMl, duringDl, after: enabled() };
    """, dataset=example_dataset, compare=compare_payloads)

    assert result["duringMl"] == []
    assert result["duringDl"] == []
    assert len(result["after"]) == 6


def test_a_family_busy_with_another_run_is_skipped_not_reported_as_failed(
    tmp_path: Path, example_dataset: dict, compare_payloads: dict
) -> None:
    """R15-4: when the DL scope is busy, Compare All leaves that run's result alone and says why DL is missing."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _DL_SINGLE_HANDLER + r"""
      const mlHold = deferred();
      const dlHold = deferred();
      page.fetchHandler = (request) => {
        const body = request.json();
        if (request.url.endsWith("/api/ml-model")) return mlHold.promise.then(() => ({ status: 200, body: { analysis: fixtures.compare.ml, request_config: body } }));
        if (request.url.endsWith("/api/deep-model")) return body.model_type === "compare"
          ? { status: 200, body: { analysis: fixtures.compare.dl, request_config: body } }
          : dlHold.promise.then(() => dlSingle(request));
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(3);
      // A DeepSurv run started while the ML phase runs (its button is off; this starts it directly).
      page.run("setPredictiveModel('deepsurv', { syncHistory: false }); withLoading(refs.runDlButton, runDlModel)");
      await page.settle(3);
      mlHold.resolve();
      await page.settle(40);
      dlHold.resolve();
      await page.settle(40);
      return {
        deepRequests: page.requests.filter((request) => request.url.endsWith("/api/deep-model")).map((request) => request.json().model_type),
        dlMode: page.run("panelModeForPayload(state.dl)"),
        toasts: page.toasts(),
      };
    """, dataset=example_dataset, compare=compare_payloads)

    assert result["deepRequests"] == ["deepsurv"]
    assert result["dlMode"] == "single"
    assert any("skipped Deep Learning" in toast for toast in result["toasts"]), result["toasts"]
    assert not any("only one model family returned comparison rows" in toast for toast in result["toasts"])


def test_compare_all_checks_every_setting_before_the_first_phase(tmp_path: Path, example_dataset: dict) -> None:
    """R15-5: an invalid DL setting or an empty feature list stops Compare All before any model runs."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = () => ({ status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('benchmark'); refs.dlHiddenLayers.value = '64,,64'");
      page.run("refs.runPredictiveCompareAllButton.click()");
      await page.settle();
      const invalidDl = { requests: page.requests.filter((request) => /ml-model|deep-model/.test(request.url)).length, toasts: page.toasts() };
      page.run("refs.dlHiddenLayers.value = '64,64'; setSharedModelFeatureSelection([])");
      await page.run("withLoading(refs.runPredictiveCompareAllButton, runUnifiedPredictiveComparison, 'predictive')");
      await page.settle();
      return { invalidDl, noFeatures: page.toasts().slice(invalidDl.toasts.length) };
    """, dataset=example_dataset)

    assert result["invalidDl"]["requests"] == 0
    assert len(result["invalidDl"]["toasts"]) == 1
    assert result["invalidDl"]["toasts"][0].startswith("Hidden layers must be a comma-separated list of positive integers.")
    assert result["noFeatures"] == ["Select at least one ML/DL model feature."]


# ── Endpoint changes ────────────────────────────────────────────


def _optimal_derive_script(change_endpoint: str) -> str:
    """Start an optimal-cutpoint Create, change the endpoint while it scans, then let the scan answer."""
    return r"""
      await loadDataset(page, fixtures.dataset);
    """ + _KM_HANDLER + r"""
      const hold = deferred();
      page.fetchHandler = (request) => (request.url.endsWith("/api/derive-group") ? hold.promise : km(request));
      page.run("activateTab('km'); refs.derivePanel.classList.remove('hidden'); syncDeriveToggleButton(); refs.deriveSource.value = 'biomarker_score'");
      page.change("#deriveMethod", "optimal_cutpoint");
      page.run("refs.deriveButton.click()");
      await page.settle(3);
      const derive = page.requests.find((request) => request.url.endsWith("/api/derive-group"));
      const sent = derive.json();
      const scanning = page.run("refs.deriveStatus.textContent");
    """ + change_endpoint + r"""
      await page.settle(3);
      const columns = [...fixtures.dataset.columns, { name: "biomarker_score__optimal_cutpoint", kind: "categorical", n_unique: 2, unique_preview: ["Low", "High"], missing: 0, non_missing: 360 }];
      hold.resolve({ status: 200, body: { ...fixtures.dataset, dataset_id: "derived-snapshot", columns, derived_column: "biomarker_score__optimal_cutpoint",
        derive_summary: { method: "optimal_cutpoint", outcome_informed: true, p_value: 0.001, p_value_label: "selection_adjusted_p_value", cutoff: 2.5,
          counts: [{ group: "Low", n: 200 }, { group: "High", n: 160 }], recipe: { method: "optimal_cutpoint", source_column: "biomarker_score" } },
        cutpoint_figure: { data: [{ x: [1, 2], y: [1, 2] }], layout: {} } } });
      await page.settle(40);
      return {
        sentEvent: sent.event_column,
        scanning,
        aborted: derive.signal.aborted,
        dataset: page.run("state.dataset.dataset_id"),
        group: page.run("refs.groupColumn.value"),
        summaryShown: page.run("!refs.deriveSummary.classList.contains('hidden')"),
        scanPlot: page.run("Boolean(refs.cutpointPlot.data)"),
        kmGroups: page.requests.filter((request) => request.url.endsWith("/api/kaplan-meier")).map((request) => request.json().group_column),
        status: page.run("refs.deriveStatus.textContent"),
        busy: page.run("isScopeBusy('derive')"),
        toasts: page.toasts(),
      };
    """


def test_changing_the_endpoint_cancels_an_optimal_cutpoint_scan(tmp_path: Path, example_dataset: dict) -> None:
    """R15-2: a grouping optimised for the old endpoint is never applied under the new one."""
    result = _run_page(tmp_path, _optimal_derive_script(r"""
      page.change("#eventColumn", "pfs_event");
    """), dataset=example_dataset)

    assert result["sentEvent"] == "os_event"
    assert result["scanning"] == "Scanning a new grouping column..."
    assert result["aborted"] is True
    assert result["dataset"] == example_dataset["dataset_id"]
    assert result["group"] == ""
    assert result["summaryShown"] is False
    assert result["scanPlot"] is False
    assert result["kmGroups"] == []
    assert result["status"] == ""
    assert result["busy"] is False


def test_an_optimal_cutpoint_for_another_endpoint_is_discarded(tmp_path: Path, example_dataset: dict) -> None:
    """R15-2: when the endpoint changed without the change handlers (a restored page), the answer is still checked."""
    result = _run_page(tmp_path, _optimal_derive_script(r"""
      page.run("refs.eventColumn.value = 'pfs_event'");
    """), dataset=example_dataset)

    assert result["aborted"] is False
    assert result["dataset"] == example_dataset["dataset_id"]
    assert result["group"] == ""
    assert result["summaryShown"] is False
    assert result["kmGroups"] == []
    assert result["status"] == ""
    assert any("The endpoint changed while the optimal cutpoint was being scanned" in toast for toast in result["toasts"])


def test_without_a_suggested_event_column_event_stays_blank(tmp_path: Path) -> None:
    """R14-20: the dataset's second column is not preselected as the event when nothing suggests one."""
    dataset = _synthetic_dataset()
    flag = {"name": "flag", "kind": "binary", "n_unique": 2, "unique_preview": [0, 1], "missing": 0, "non_missing": 40}
    dataset["columns"] = [dataset["columns"][0], flag, *[column for column in dataset["columns"][1:] if column["name"] != "os_event"]]
    dataset["numeric_columns"] = ["flag", "os_months", "age", "biomarker", "status_code"]
    dataset["binary_candidate_columns"] = ["flag", "sex", "status_code"]
    dataset["suggestions"] = {"time_columns": ["os_months"], "event_columns": []}
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      return { event: page.run("refs.eventColumn.value"), options: page.run("refs.eventColumn.options.map((option) => option.value)") };
    """, dataset=dataset)

    assert result["event"] == ""
    assert "flag" in result["options"]


def test_unticking_all_event_columns_is_an_endpoint_change(tmp_path: Path) -> None:
    """R13-2: when unticking "All columns" resets Event, results are cleared and the Cox preview is refreshed."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _KM_HANDLER + r"""
      page.fetchHandler = (request) => {
        if (!request.url.endsWith("/api/cox-preview")) return km(request);
        const events = request.json().event_column === "status_code" ? 30 : 26;
        return { status: 200, body: { preview: { analyzable_rows: 40, outcome_rows: 40, events, estimated_parameters: 2, events_per_parameter: events / 2 } } };
      };
      const snapshot = () => page.run(`({ event: refs.eventColumn.value, line: refs.coxPreviewLine.textContent, km: Boolean(state.km) })`);
      page.run("activateTab('cox')");
      page.change("#showAllEventColumns", true);
      page.change("#eventColumn", "status_code");
      await page.settle();
      page.run("refs.runKmButton.click()");
      await page.settle();
      const ticked = snapshot();
      page.change("#showAllEventColumns", false);
      await page.settle();
      const unticked = snapshot();
      // A non-standard event column chosen without "All columns" is blocked; ticking the box unblocks it.
      page.run(`{ const option = document.createElement("option"); option.value = "status_code"; option.textContent = "status_code";
        refs.eventColumn.appendChild(option); refs.eventColumn.value = "status_code"; refreshCoxPreview({ force: true }); }`);
      await page.settle();
      const blocked = page.run("refs.coxPreviewLine.textContent");
      page.change("#showAllEventColumns", true);
      await page.settle();
      return { ticked, unticked, blocked, afterTick: page.run("refs.coxPreviewLine.textContent") };
    """, dataset=_synthetic_dataset())

    assert result["ticked"]["event"] == "status_code" and result["ticked"]["km"] is True
    assert "30 events" in result["ticked"]["line"]
    assert result["unticked"]["event"] == "os_event"
    assert result["unticked"]["km"] is False
    assert "26 events" in result["unticked"]["line"]
    assert "not a standard event column name" in result["blocked"]
    assert "30 events" in result["afterTick"]


# ── Exports ─────────────────────────────────────────────────────


def test_comparison_exports_send_scalar_cells_only(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R13-9: per-repeat and per-fold detail never reaches the CSV (the server would write it as a Python literal)."""
    compare = {key: value for key, value in compare_payloads.items()}
    ml = dict(compare_payloads["ml"])
    ml["comparison_table"] = [
        {**row, "repeat_results": [{"repeat": 1, "c_index": 0.7}], "fold_results": [{"fold": 1, "c_index": 0.7}], "extra": {"a": 1}}
        for row in compare_payloads["ml"]["comparison_table"]
    ]
    compare["ml"] = ml
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/ml-model")) return { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } };
        if (request.url.endsWith("/api/export-table")) return { status: 200, body: "csv" };
        return { status: 500, body: { detail: "unexpected" } };
      };
      await page.run("withLoading(refs.runCompareButton, runCompareModels)");
      await page.settle();
      page.run("refs.downloadMlComparisonButton.click()");
      await page.settle();
      const sent = page.requests.find((request) => request.url.endsWith("/api/export-table")).json();
      return { columns: sent.columns, keys: [...new Set(sent.rows.flatMap((row) => Object.keys(row)))], rows: sent.rows.length };
    """, dataset=example_dataset, compare=compare)

    assert result["rows"] == 3
    for name in ("repeat_results", "fold_results", "extra"):
        assert name not in result["columns"] and name not in result["keys"]
    assert {"model", "c_index", "rank"} <= set(result["columns"])


def test_the_table_one_csv_is_exported_by_the_server_with_its_provenance(tmp_path: Path, example_dataset: dict) -> None:
    """R13-10: the CSV goes through /api/export-table like the XLSX, so it carries the dataset's provenance."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const table = { columns: ["Variable", "Statistic", "Overall", "Test positive"],
        rows: [{ Variable: "Cohort size", Statistic: "N", Overall: 360, "Test positive": 45 }], row_mask_hash: "rows" };
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/cohort-table")) return { status: 200, body: { analysis: table, request_config: request.json() } };
        if (request.url.endsWith("/api/export-table")) return { status: 200, body: "csv" };
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('tables'); refs.runCohortTableButton.click()");
      await page.settle();
      page.run("refs.downloadCohortTableButton.click()");
      await page.settle();
      const request = page.requests.find((item) => item.url.endsWith("/api/export-table"));
      return { sent: request ? request.json() : null, downloads: page.downloads.length };
    """, dataset=example_dataset)

    sent = result["sent"]
    assert sent is not None
    assert sent["format"] == "csv" and sent["style"] == "plain"
    assert sent["columns"] == ["Variable", "Statistic", "Overall", "Test positive"]
    assert sent["rows"] == [{"Variable": "Cohort size", "Statistic": "N", "Overall": 360, "Test positive": 45}]
    assert sent["provenance"]["dataset_hash"] == example_dataset["dataset_hash"]
    assert result["downloads"] == 1


def test_a_refused_checklist_export_names_the_field_the_server_rejected(tmp_path: Path) -> None:
    """R13-19: the validation detail of a 422 (a list) reaches the user instead of a generic message."""
    result = _run_page(tmp_path, r"""
      page.fetchHandler = () => ({ status: 422, body: { detail: [{ loc: ["body", "format"], msg: "Input should be 'docx' or 'markdown'" }] } });
      const refused = await page.run("downloadChecklist({ guideline: 'REMARK' }, 'pdf', 'remark').then(() => 'ok', (error) => error.message)");
      page.fetchHandler = () => ({ status: 500, body: "<html>proxy error</html>" });
      const unreadable = await page.run("downloadChecklist({ guideline: 'REMARK' }, 'docx', 'remark').then(() => 'ok', (error) => error.message)");
      return { refused, unreadable };
    """)

    assert result == {"refused": "format: Input should be 'docx' or 'markdown'", "unreadable": "Checklist export failed."}


def test_discover_has_its_own_busy_scope_and_its_csv_says_why_it_waits(tmp_path: Path, example_dataset: dict) -> None:
    """R13-11/R13-12: while a cut-point search runs, KM stays usable and the signature CSV asks to wait for the run."""
    result = _run_page(tmp_path, _signature_script(r"""
      const firstRun = { csvEnabled: !page.run("refs.downloadSignatureButton.disabled") };
      const hold = deferred();
      const answer = page.fetchHandler;
      page.fetchHandler = (request) => (request.url.endsWith("/api/discover-signature") ? hold.promise.then(() => answer(request)) : answer(request));
      page.run("refs.runSignatureSearchButton.click()");
      await page.settle(3);
      page.run("refs.downloadSignatureButton.disabled = false; refs.downloadSignatureButton.click()");
      const during = {
        toast: page.toasts().pop(),
        kmButtonDisabled: page.run("refs.runKmButton.disabled"),
        kmStatus: page.run("document.querySelector('[data-run-status=km]').textContent"),
        discoverBusy: page.run("refs.runSignatureSearchButton.disabled"),
      };
      hold.resolve();
      await page.settle();
      return { firstRun, during, after: { busy: page.run("isScopeBusy('signature') || isScopeBusy('km')"), discoverDisabled: page.run("refs.runSignatureSearchButton.disabled") } };
    """), dataset=example_dataset)

    assert result["firstRun"] == {"csvEnabled": True}
    assert result["during"] == {
        "toast": "Wait for the current run to finish before exporting this result.",
        "kmButtonDisabled": False,
        "kmStatus": "",
        "discoverBusy": True,
    }
    assert result["after"] == {"busy": False, "discoverDisabled": False}


def test_manuscript_tables_show_the_headers_they_export(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """R13-21: "Training Time, ms" is shown as exported, not rewritten to "Training Time, Ms"."""
    compare = dict(compare_payloads)
    ml = dict(compare_payloads["ml"])
    ml["manuscript_tables"] = {"model_performance_table": [
        {"Rank": 1, "Model": "Random Survival Forest", "C-index": 0.714, "Features, n": 4, "Training Time, ms": 812.5},
    ]}
    compare["ml"] = ml
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = (request) => (request.url.endsWith("/api/ml-model")
        ? { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } }
        : { status: 500, body: { detail: "unexpected" } });
      await page.run("withLoading(refs.runCompareButton, runCompareModels)");
      await page.settle();
      return page.run("refs.mlManuscriptShell.querySelectorAll('th').map((th) => th.textContent)");
    """, dataset=example_dataset, compare=compare)

    assert result == ["Rank", "Model", "C-index", "Features, n", "Training Time, ms"]


def test_the_data_preview_shows_values_as_they_are_in_the_file(tmp_path: Path, example_dataset: dict) -> None:
    """R13-22: a 13-digit ID and a 6-decimal value are not rewritten to "1.70e+12" and "52.123"."""
    dataset = {**example_dataset, "preview": [dict(row) for row in example_dataset["preview"]]}
    dataset["columns"] = [*example_dataset["columns"], {"name": "record", "kind": "numeric", "n_unique": 360, "unique_preview": [], "missing": 0, "non_missing": 360}]
    dataset["preview"][0].update({"record": 1700000000000, "age": 52.123456})
    dataset["preview"][1].update({"record": None})
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const headers = page.run("refs.datasetPreviewShell.querySelectorAll('th').map((th) => th.textContent)");
      const cell = (row, name) => page.run(`refs.datasetPreviewShell.querySelectorAll('tbody tr')[${row}].children[${headers.indexOf(name)}].textContent`);
      return { record: cell(0, "record"), age: cell(0, "age"), missing: cell(1, "record") };
    """, dataset=dataset)

    assert result == {"record": "1700000000000", "age": "52.123456", "missing": "NA"}


# ── Markers ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def untested_marker_payloads(example_dataset) -> dict:
    """Real marker evaluations run with Subsamples = 0 and with Permutations = 0."""
    request = {
        "dataset_id": example_dataset["dataset_id"],
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "marker_columns": ["biomarker_score", "immune_index"],
        "clinical_columns": ["age", "stage"],
        "categorical_clinical": ["stage"],
        "n_permutations": 49,
        "n_resamples": 8,
        "random_seed": 7,
    }
    payloads = {}
    for key, changes in (("noSubsamples", {"n_resamples": 0}), ("noPermutations", {"n_permutations": 0})):
        response = client.post("/api/marker-evaluation", json={**request, **changes})
        assert response.status_code == 200, response.text
        payloads[key] = response.json()
    return payloads


def test_marker_summary_does_not_claim_what_was_not_tested(tmp_path: Path, untested_marker_payloads: dict) -> None:
    """R15-6: with no subsamples, stability is untested; with no permutations, there are no family-wise p-values."""
    result = _run_page(tmp_path, r"""
      page.context.__markers = fixtures.markers;
      const summarize = (key) => {
        const summary = page.run(`markerSummary(__markers.${key})`);
        return { headline: summary.headline, strengths: summary.strengths.join(" | "), cautions: summary.cautions.join(" | "), next: summary.next_steps.join(" | ") };
      };
      return { noSubsamples: summarize("noSubsamples"), noPermutations: summarize("noPermutations") };
    """, markers=untested_marker_payloads)

    no_subsamples = result["noSubsamples"]
    analysis = untested_marker_payloads["noSubsamples"]["analysis"]
    assert analysis["resampling"]["n_valid"] == 0
    assert not analysis["resampling"]["stability_assessed"]
    assert analysis["tier_counts"]["suggestive"] == 0
    assert analysis["tier_counts"]["robust"] == 0
    assert analysis["signature"]["optimism_corrected_c"] is None
    assert analysis["inference"]["status"] == "withheld"
    assert "withheld" in no_subsamples["headline"].lower()
    assert "does not hold up across subsamples" not in no_subsamples["headline"]
    assert "repeated on 0 subsamples" not in no_subsamples["strengths"]
    assert "exploratory" in no_subsamples["cautions"].lower()
    assert "optimism-corrected" not in no_subsamples["next"]

    no_permutations = result["noPermutations"]
    analysis = untested_marker_payloads["noPermutations"]["analysis"]
    assert analysis["null"]["n_permutations"] == 0
    assert analysis["inference"]["status"] == "withheld"
    assert all(row["added_value"]["p_fwer"] is None for row in analysis["marker_table"])
    assert "after family-wise error control" not in no_permutations["headline"]
    assert "withheld" in no_permutations["headline"].lower()
    assert "0 permutations" not in no_permutations["strengths"]
    assert "Westfall-Young" not in no_permutations["strengths"]
    assert "exploratory" in no_permutations["cautions"].lower()


def test_the_left_out_c_index_of_an_unfitted_model_is_the_whole_procedures(tmp_path: Path) -> None:
    """With no full-cohort model (its fit failed), the left-out C-index is the selection procedure's, as on the server."""
    result = _run_page(tmp_path, r"""
      const analysis = {
        primary_lens: "added_value",
        tier_counts: { robust: 0, suggestive: 0 },
        cohort: { n: 300, events: 120, n_markers_evaluated: 12 },
        settings: { alpha: 0.05, robust_frequency: 0.5, robust_direction: 0.9 },
        resampling: { n_valid: 5, fraction: 0.632, stability_assessed: true },
        null: { n_permutations: 199, lens2_null: "smith" },
      };
      const summarize = (signature) => {
        page.context.__payload = { analysis: { ...analysis, signature } };
        const summary = page.run("markerSummary(__payload)");
        return [...summary.cautions, ...summary.strengths].join(" | ");
      };
      const leftOut = { signature_c_left_out: 0.645, clinical_c_left_out: 0.64, signature_gain_left_out: 0.004, n_clinical_replicates: 5, signature_optimism: 0.03 };
      return {
        unfitted: summarize({ ...leftOut, markers: [], clinical_only: false, apparent_c: null, optimism_corrected_c: null }),
        fitted: summarize({ ...leftOut, markers: ["GENE_A"], clinical_only: false, apparent_c: 0.7, optimism_corrected_c: 0.67 }),
      };
    """)

    assert "In the patients left out of each of 5 subsamples, the whole selection procedure reached C 0.645 against 0.64" in result["unfitted"]
    assert "selected-marker" not in result["unfitted"]
    assert "The selected markers add little" not in result["unfitted"]
    assert "apparent C-index is optimistic" not in result["unfitted"]
    assert "In the patients left out of each of 5 subsamples, the whole selection procedure reached C 0.645" in result["fitted"]
    assert "No interval was available to assess the gain's uncertainty" in result["fitted"]
    assert "The selection procedure's mean subsample-to-left-out C-index gap is 0.030" in result["fitted"]
    assert "heuristic that includes training-size effects" in result["fitted"]


def test_dropped_markers_are_grouped_by_their_reason_and_cohort_notes_are_shown(tmp_path: Path) -> None:
    """Markers left out for infinite values get their own group, an unknown reason a generic one; the engine's
    cohort notes are cautions, and data-derived text in them stays inert."""
    result = _run_page(tmp_path, r"""
      page.context.__payload = { analysis: {
        primary_lens: "added_value",
        tier_counts: { robust: 0, suggestive: 0 },
        settings: { alpha: 0.05, robust_frequency: 0.5, robust_direction: 0.9 },
        resampling: { n_valid: 5, fraction: 0.632, stability_assessed: true },
        null: { n_permutations: 99, lens2_null: "smith" },
        signature: {},
        cohort: {
          n: 300, events: 120, n_markers_evaluated: 12,
          dropped_markers: [
            { marker: "log_a", reason: "infinite values", n_infinite: 3 },
            { marker: "log_b", reason: "infinite values", n_infinite: 1 },
            { marker: "rare", reason: "near-constant (95% at one value)" },
            { marker: "flat", reason: "constant" },
            { marker: "sparse", reason: "40% missing" },
            { marker: "odd", reason: "a reason added later" },
          ],
          notes: ["age_<b>x</b> was left out of the clinical model: collinear.", "No clinical covariate could be estimated, so the markers were evaluated without clinical adjustment."],
        },
      } };
      const summary = page.run("markerSummary(__payload)");
      page.run("renderInsightBoard(refs.markersInsightBoard, markerSummary(__payload), '')");
      return { cautions: summary.cautions, html: page.run("refs.markersInsightBoard.innerHTML"), bold: page.run("refs.markersInsightBoard.querySelectorAll('b').length") };
    """)

    dropped = next(caution for caution in result["cautions"] if "left out before testing" in caution)
    assert dropped.startswith("6 marker(s) were left out before testing: ")
    assert "1 near-constant (most patients at one value, as for genes expressed in few patients)" in dropped
    assert "2 constant or mostly missing" in dropped
    assert "2 with infinite values (for example the log of 0; use log(x + 1) instead)" in dropped
    assert "1 for another reason" in dropped
    assert "age_<b>x</b> was left out of the clinical model: collinear." in result["cautions"]
    assert "No clinical covariate could be estimated, so the markers were evaluated without clinical adjustment." in result["cautions"]
    assert result["bold"] == 0 and "age_&lt;b&gt;x&lt;/b&gt;" in result["html"]


def test_blank_marker_settings_use_the_shared_defaults(tmp_path: Path, example_dataset: dict) -> None:
    """Blank Permutations, Subsamples and Seed send the workspace's MARKER_NUMERIC_DEFAULTS, the values result
    currency compares against."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.run("refs.markerPermutations.value = ''; refs.markerResamples.value = ''; refs.markerRandomSeed.value = '';");
      const sent = page.run("(({ n_permutations, n_resamples, random_seed }) => ({ n_permutations, n_resamples, random_seed }))(markerRequestFields())");
      const shared = page.run("({ ...MARKER_NUMERIC_DEFAULTS })");
      return { sent, shared };
    """, dataset=example_dataset)

    assert result["sent"] == result["shared"] == {"n_permutations": 1000, "n_resamples": 200, "random_seed": 20260926}


_MATRIX = "{ matrix_id: 'm1', filename: 'expr.tsv', n_markers: 20, n_matched: 300, n_patients: 360, id_column: 'patient_id' }"


def test_a_marker_matrix_is_freed_when_another_dataset_replaces_its_own(tmp_path: Path, example_dataset: dict) -> None:
    """R15-11: a derived snapshot keeps the attached matrix; another dataset deletes it on the server."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.run("state.markerMatrix = """ + _MATRIX + r"""; renderMarkerMatrixState();");
      const columns = [...fixtures.dataset.columns, { name: "age__median_split", kind: "categorical", n_unique: 2, unique_preview: ["Low", "High"], missing: 0, non_missing: 360 }];
      page.fetchHandler = (request) => {
        if (request.method === "DELETE") return { status: 200, body: { status: "deleted" } };
        if (request.url.endsWith("/api/derive-group")) return { status: 200, body: { ...fixtures.dataset, dataset_id: "derived-snapshot", columns,
          derived_column: "age__median_split", derive_summary: { method: "median_split", counts: [] } } };
        return { status: 200, body: { preview: { analyzable_rows: 360, outcome_rows: 360, events: 250, estimated_parameters: 4, events_per_parameter: 62 } } };
      };
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); refs.deriveButton.click()");
      await page.settle();
      const afterDerive = { matrix: page.run("state.markerMatrix?.matrix_id || null"), deletes: page.requests.filter((request) => request.method === "DELETE").length };
      await loadDataset(page, { ...fixtures.dataset, dataset_id: "another-cohort" });
      return { afterDerive, matrix: page.run("state.markerMatrix"), deletes: page.requests.filter((request) => request.method === "DELETE").map((request) => request.url) };
    """, dataset=example_dataset)

    assert result["afterDerive"] == {"matrix": "m1", "deletes": 0}
    assert result["matrix"] is None
    assert result["deletes"] == ["/api/marker-matrix/m1"]


def test_a_marker_file_attached_to_a_replaced_dataset_is_deleted(tmp_path: Path, example_dataset: dict) -> None:
    """R15-11: the attach answer for the old dataset is not used, and its matrix is freed on the server."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const hold = deferred();
      page.fetchHandler = (request) => {
        if (request.method === "DELETE") return { status: 200, body: { status: "deleted" } };
        if (request.url.endsWith("/api/marker-matrix")) return hold.promise;
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('markers'); refs.markerMatrixFile.files = [{ name: 'expr.tsv' }]; refs.attachMarkerMatrixButton.click()");
      await page.settle(3);
      await loadDataset(page, { ...fixtures.dataset, dataset_id: "another-cohort" });
      hold.resolve({ status: 200, body: { matrix_id: "m2", filename: "expr.tsv", n_markers: 20, n_matched: 300, n_patients: 360, id_column: "patient_id" } });
      await page.settle();
      return {
        matrix: page.run("state.markerMatrix"),
        deletes: page.requests.filter((request) => request.method === "DELETE").map((request) => request.url),
        toasts: page.toasts(),
      };
    """, dataset=example_dataset)

    assert result["matrix"] is None
    assert result["deletes"] == ["/api/marker-matrix/m2"]
    assert any("was not attached" in toast for toast in result["toasts"]), result["toasts"]


def test_a_validation_upload_cancelled_in_flight_is_still_deleted(tmp_path: Path, example_dataset: dict, marker_payloads: dict) -> None:
    """R15-11: the upload is let finish, so the external cohort it stored can be deleted again."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = (request) => (request.url.endsWith("/api/marker-evaluation")
        ? { status: 200, body: { ...fixtures.markers.added, request_config: request.json() } }
        : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('markers'); refs.runMarkersButton.click()");
      await page.settle();
      const upload = deferred();
      page.fetchHandler = (request) => {
        if (request.method === "DELETE") return { status: 200, body: { status: "deleted" } };
        if (request.url.endsWith("/api/upload")) return upload.promise;
        if (request.url.endsWith("/api/marker-validation")) return deferred().promise;
        if (request.url.endsWith("/api/marker-evaluation")) return { status: 200, body: { ...fixtures.markers.added, request_config: request.json() } };
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("refs.markerValidationFile.files = [{ name: 'external.csv' }]; refs.runMarkerValidationButton.click()");
      await page.settle(3);
      // A new evaluation makes this validation obsolete while the file is still uploading.
      page.run("runMarkerEvaluation()");
      await page.settle();
      upload.resolve({ status: 200, body: { dataset_id: "external-3", filename: "external.csv" } });
      await page.settle();
      return {
        deletes: page.requests.filter((request) => request.method === "DELETE").map((request) => request.url),
        validations: page.requests.filter((request) => request.url.endsWith("/api/marker-validation")).length,
        busy: page.run("isScopeBusy('markers')"),
      };
    """, dataset=example_dataset, markers=marker_payloads)

    assert result == {"deletes": ["/api/dataset/external-3"], "validations": 0, "busy": False}


def test_a_validation_without_a_figure_clears_the_previous_replication_plot(tmp_path: Path) -> None:
    """R15-13: the plot of an earlier validation does not stay under a later one."""
    result = _run_page(tmp_path, r"""
      const validation = { cohort: { n: 200, events: 90 }, metrics: { c_index: 0.7 }, markers: [], notes: [] };
      page.context.__validation = validation;
      await page.run(`renderMarkerValidation({ external_filename: "first.csv", validation: __validation,
        figure: { data: [{ x: [1, 2], y: [1, 2] }], layout: {} } })`);
      const first = page.run("({ data: Boolean(refs.markerValidationPlot.data), hidden: refs.markerValidationPlot.classList.contains('hidden') })");
      await page.run(`renderMarkerValidation({ external_filename: "second.csv", validation: __validation, figure: null })`);
      return { first, second: page.run("({ data: Boolean(refs.markerValidationPlot.data), hidden: refs.markerValidationPlot.classList.contains('hidden') })") };
    """)

    assert result == {"first": {"data": True, "hidden": False}, "second": {"data": False, "hidden": True}}


# ── Cox ─────────────────────────────────────────────────────────


_COX_PREVIEW = r"""
  const coxPreview = { status: 200, body: { preview: { analyzable_rows: 350, outcome_rows: 360, dropped_rows: 10, events: 250, estimated_parameters: 4, events_per_parameter: 62.5 } } };
"""


def test_cox_diagnostics_that_are_unavailable_say_why(tmp_path: Path, example_dataset: dict) -> None:
    """R13-13: the "unavailable" explanations of the diagnostics and martingale panels are shown, not hidden."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _COX_PREVIEW + r"""
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/cox-preview")) return coxPreview;
        if (!request.url.endsWith("/api/cox")) return { status: 500, body: { detail: "unexpected" } };
        return { status: 200, body: {
          analysis: { results_table: [{ Variable: "stage", "Hazard ratio": 1.4 }], diagnostics_table: [],
            model_stats: { n: 360, events: 250, parameters: 3, martingale_note: "No continuous covariates to screen for functional form." } },
          figure: { data: [{ x: [1], y: [1] }], layout: {} }, request_config: request.json() } };
      };
      page.run("activateTab('cox'); refs.runCoxButton.click()");
      await page.settle();
      const panel = (name) => page.run(`({ hidden: refs.${name}.classList.contains("result-hidden"),
        cardHidden: refs.${name}.closest(".table-card").classList.contains("result-hidden"), text: refs.${name}.textContent.trim() })`);
      return { diagnostics: panel("coxDiagnosticsPlot"), martingale: panel("coxMartingalePlot") };
    """, dataset=example_dataset)

    assert result["diagnostics"] == {"hidden": False, "cardHidden": False, "text": "Scaled Schoenfeld residual screening was unavailable for this fit."}
    assert result["martingale"] == {"hidden": False, "cardHidden": False, "text": "No continuous covariates to screen for functional form."}


def test_a_cox_preview_of_other_settings_is_never_shown(tmp_path: Path, example_dataset: dict) -> None:
    """R13-2/R14-11: once the covariates change, the old preview leaves the line at once, not after the debounce."""
    result = _run_page(tmp_path, r"""
      page.fetchHandler = (request) => {
        if (!request.url.endsWith("/api/cox-preview")) return { status: 500, body: { detail: "unexpected" } };
        const parameters = request.json().covariates.length;
        return { status: 200, body: { preview: { analyzable_rows: 360, outcome_rows: 360, events: 250, estimated_parameters: parameters, events_per_parameter: 250 / parameters } } };
      };
      await loadDataset(page, fixtures.dataset);
      await page.settle();
      const line = () => page.run("refs.coxPreviewLine.textContent");
      const before = line();
      const extra = page.run(`allCheckboxValues(refs.covariateChecklist).find((value) => !/stage/.test(value)
        && !selectedCheckboxValues(refs.covariateChecklist).includes(value))`);
      page.change(`#covariateChecklist input[value="${extra}"]`, true);
      const immediately = line();
      await page.settle();
      return { before, immediately, after: line(), selected: page.run("selectedCheckboxValues(refs.covariateChecklist).length") };
    """, dataset=example_dataset)

    assert f"{result['selected'] - 1} parameters" in result["before"]
    assert result["immediately"] == ""
    assert f"{result['selected']} parameters" in result["after"]


def test_the_cox_preview_is_shown_as_soon_as_a_dataset_loads(tmp_path: Path, example_dataset: dict) -> None:
    """R13-14: the default Cox covariates get their usable-row preview without a click."""
    result = _run_page(tmp_path, _COX_PREVIEW + r"""
      page.fetchHandler = (request) => (request.url.endsWith("/api/cox-preview") ? coxPreview : { status: 500, body: { detail: "unexpected" } });
      await loadDataset(page, fixtures.dataset);
      await page.settle();
      return {
        previews: page.requests.filter((request) => request.url.endsWith("/api/cox-preview")).length,
        line: page.run("refs.coxPreviewLine.textContent"),
      };
    """, dataset=example_dataset)

    assert result["previews"] == 1
    assert result["line"].startswith("350 of 360 patients usable (10 dropped for missing values)")


# ── Navigation, uploads and errors ──────────────────────────────


def test_arrow_keys_move_between_shown_tabs_without_new_history_entries(tmp_path: Path, example_dataset: dict) -> None:
    """R13-17/R14-15: the ML and DL buttons (both the Prediction tab) are skipped, and keys replace history like clicks."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      let pushes = 0;
      const history = page.context.history;
      const push = history.pushState.bind(history);
      history.pushState = (...args) => { pushes += 1; return push(...args); };
      page.run("activateTab('benchmark', { focusTabButton: true })");
      const strip = page.document.querySelector(".tab-strip");
      const press = (key) => {
        strip.dispatchEvent({ type: "keydown", key, target: page.document.activeElement, bubbles: true, defaultPrevented: false,
          preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} });
        return page.run("activeTabName()");
      };
      const keys = ["ArrowLeft", "ArrowRight", "ArrowRight", "Home", "End", "ArrowRight"];
      return { tabs: keys.map(press), pushes, focused: page.document.activeElement?.dataset?.tab };
    """, dataset=example_dataset)

    assert result == {"tabs": ["markers", "benchmark", "tables", "km", "data", "km"], "pushes": 0, "focused": "km"}


def test_plot_failures_in_event_handlers_are_caught(tmp_path: Path, example_dataset: dict) -> None:
    """R13-18: a failed martingale redraw is reported; a failed or throwing reset-axes relayout is contained."""
    result = _run_page(tmp_path, r"""
      let unhandled = 0;
      process.on("unhandledRejection", () => { unhandled += 1; });
      await loadDataset(page, fixtures.dataset);
      page.run(`state.cox = { analysis: { martingale_plot_data: [
        { term: "age", value: [1, 2, 3], residual: [0.1, -0.2, 0.3] }, { term: "bmi", value: [1, 2, 3], residual: [0.2, 0.1, -0.3] }] } };
        syncCoxMartingaleSelector(coxMartingalePanels(), "age");`);
      const newPlot = page.plotly.newPlot;
      page.plotly.newPlot = () => Promise.reject(new Error("the chart could not be drawn"));
      page.change("#coxMartingaleVariableSelect", "bmi");
      await page.settle();
      page.plotly.newPlot = newPlot;
      page.run(`refs.coxPlot._fullLayout = { xaxis: { range: [0, 2] }, yaxis: { range: [0, 3] }, height: 400 };
        refs.coxPlot.__handlers = {};
        refs.coxPlot.on = (name, handler) => { refs.coxPlot.__handlers[name] = handler; };
        stabilizeCoxPlotResetAxes(refs.coxPlot);`);
      page.plotly.relayout = () => Promise.reject(new Error("relayout failed"));
      page.run("refs.coxPlot.__handlers.plotly_relayout({ 'xaxis.autorange': true })");
      await page.settle();
      const afterRejected = page.run("refs.coxPlot.__stableResetAxesState.applying");
      page.plotly.relayout = () => { throw new Error("relayout threw"); };
      let threw = false;
      try {
        page.run("refs.coxPlot.__handlers.plotly_relayout({ 'yaxis.autorange': true })");
      } catch {
        threw = true;
      }
      await page.settle();
      await new Promise((resolve) => setTimeout(resolve, 20));
      return { unhandled, toasts: page.toasts(), afterRejected, threw, afterThrown: page.run("refs.coxPlot.__stableResetAxesState.applying") };
    """, dataset=example_dataset)

    assert result["unhandled"] == 0
    assert any("the chart could not be drawn" in toast for toast in result["toasts"]), result["toasts"]
    assert result["afterRejected"] is False
    assert result["threw"] is False
    assert result["afterThrown"] is False


def test_the_upload_button_stays_busy_until_the_last_of_overlapping_uploads_ends(tmp_path: Path, example_dataset: dict) -> None:
    """R13-20: a second file (dropped while one uploads) replaces the first; the button waits for the second."""
    result = _run_page(tmp_path, r"""
      const uploads = [deferred(), deferred()];
      let calls = 0;
      page.fetchHandler = (request) => (request.url.endsWith("/api/upload") ? uploads[calls++].promise : { status: 500, body: { detail: "unexpected" } });
      const button = () => page.run("({ disabled: refs.uploadButton.disabled, loading: refs.uploadButton.classList.contains('is-loading') })");
      page.run("refs.datasetFile.files = [{ name: 'first.csv' }]");
      page.change("#datasetFile");
      await page.settle(3);
      page.run("refs.datasetFile.files = [{ name: 'second.csv' }]");
      page.change("#datasetFile");
      await page.settle(3);
      const firstAborted = page.requests[0].signal.aborted;
      const whileSecond = button();
      uploads[1].resolve({ status: 200, body: fixtures.dataset });
      await page.settle();
      return { calls, firstAborted, whileSecond, after: button(), dataset: page.run("state.dataset?.dataset_id") };
    """, dataset=example_dataset)

    assert result["calls"] == 2
    assert result["firstAborted"] is True
    assert result["whileSecond"] == {"disabled": True, "loading": True}
    assert result["after"] == {"disabled": False, "loading": False}
    assert result["dataset"] == example_dataset["dataset_id"]


def test_dead_code_of_the_second_review_is_gone() -> None:
    """R13-16: withLoading has no swallowErrors option, required refs need no guards, the workbench has no fallback."""
    static = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static"
    app_js = (static / "app.js").read_text(encoding="utf-8")
    sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted(static.glob("app*.js")))
    assert "swallowErrors" not in sources
    assert "clearCohortTableOutput" not in sources
    assert "shouldClearTableOutput" not in sources
    for name in ("downloadCoxPngButton", "downloadCoxSvgButton", "downloadCohortTableXlsxButton", "downloadMlComparisonPngButton",
                 "downloadMlComparisonSvgButton", "downloadDlComparisonPngButton", "downloadDlComparisonSvgButton",
                 "downloadKmPngButton", "downloadKmSvgButton"):
        assert f"if (refs.{name}) refs.{name}.addEventListener" not in app_js, name
    start = app_js.index("function reviewBenchmarkSourceTab(")
    body = app_js[start:app_js.index("\n}\n", start)]
    assert "scrollToAnalysisResult" not in body and "WorkspaceCard" not in body


# ── Variable selections ─────────────────────────────────────────


def test_selections_start_from_defaults_for_a_new_dataset_and_stay_as_chosen_otherwise(tmp_path: Path, example_dataset: dict) -> None:
    """R14 note: another dataset starts from the defaults; an endpoint change or a derived snapshot keeps the choices,
    including lists the user emptied (no silent re-ticking of defaults)."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const lists = () => page.run(`({ cox: selectedCheckboxValues(refs.covariateChecklist), features: selectedCheckboxValues(refs.modelFeatureChecklist),
        dlFeatures: selectedCheckboxValues(refs.dlModelFeatureChecklist), mlCategoricals: selectedCheckboxValues(refs.modelCategoricalChecklist),
        dlCategoricals: selectedCheckboxValues(refs.dlModelCategoricalChecklist), markers: selectedCheckboxValues(refs.markerChecklist),
        table: selectedCheckboxValues(refs.cohortVariableChecklist) })`);
      const defaults = lists();
      page.run("setCheckedValues(refs.covariateChecklist, ['age']); setSharedModelFeatureSelection(['age']); setCheckedValues(refs.cohortVariableChecklist, ['age']);");
      await loadDataset(page, { ...fixtures.dataset, dataset_id: "another-cohort" });
      const anotherDataset = lists();
      // Emptied lists stay empty through an endpoint change.
      page.run("setSharedModelFeatureSelection(['age', 'stage']); setCheckedValues(refs.modelCategoricalChecklist, []); setCheckedValues(refs.dlModelCategoricalChecklist, []);");
      page.run("refs.clearMarkersButton.click()");
      page.change("#eventColumn", "pfs_event");
      await page.settle();
      const afterEndpoint = lists();
      page.run("setSharedModelFeatureSelection([])");
      page.change("#eventColumn", "os_event");
      await page.settle();
      const emptiedFeatures = lists();
      // A derived snapshot of the same data keeps the choices too.
      page.run("setSharedModelFeatureSelection(['age', 'stage']); setCheckedValues(refs.modelCategoricalChecklist, []); setCheckedValues(refs.dlModelCategoricalChecklist, []);");
      const columns = [...fixtures.dataset.columns, { name: "age__median_split", kind: "categorical", n_unique: 2, unique_preview: ["Low", "High"], missing: 0, non_missing: 360 }];
      page.fetchHandler = (request) => (request.url.endsWith("/api/derive-group")
        ? { status: 200, body: { ...fixtures.dataset, dataset_id: "derived-snapshot", columns, derived_column: "age__median_split", derive_summary: { method: "median_split", counts: [] } } }
        : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); refs.deriveButton.click()");
      await page.settle();
      const afterDerive = lists();
      // Create refreshes the lists once more after the snapshot (applyControlSnapshot's own re-ticking of
      // categorical flags is the workspace part's): emptied flags stay empty there.
      page.run("setCheckedValues(refs.modelCategoricalChecklist, []); setCheckedValues(refs.dlModelCategoricalChecklist, []); refreshVariableSelections();");
      return { defaults, anotherDataset, afterEndpoint, emptiedFeatures, afterDerive, afterRefresh: lists(), dataset: page.run("state.dataset.dataset_id") };
    """, dataset=example_dataset)

    defaults = result["defaults"]
    assert len(defaults["cox"]) == 4 and len(defaults["features"]) > 1 and len(defaults["table"]) > 1 and defaults["markers"]
    assert result["anotherDataset"] == defaults
    assert result["afterEndpoint"]["features"] == ["age", "stage"]
    assert result["afterEndpoint"]["mlCategoricals"] == [] and result["afterEndpoint"]["dlCategoricals"] == []
    assert result["afterEndpoint"]["markers"] == []
    assert result["emptiedFeatures"]["features"] == [] and result["emptiedFeatures"]["dlFeatures"] == []
    assert result["dataset"] == "derived-snapshot"
    assert result["afterDerive"]["features"] == ["age", "stage"]
    assert result["afterRefresh"]["features"] == ["age", "stage"]
    assert result["afterRefresh"]["mlCategoricals"] == [] and result["afterRefresh"]["dlCategoricals"] == []


def test_select_all_with_a_search_filter_adds_the_shown_items(tmp_path: Path, example_dataset: dict) -> None:
    """R14-5: "Select all" adds what the search shows to the selection; it never drops the hidden choices."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const pick = (search, checklist, input, button, start, query) => {
        page.run(`setCheckedValues(refs.${checklist}, ${JSON.stringify(start)})`);
        page.change(`#${input}`, query);
        page.run(`refs.${button}.click()`);
        const selected = page.run(`selectedCheckboxValues(refs.${checklist})`);
        page.change(`#${input}`, "");
        return selected;
      };
      return {
        cox: pick("cox", "covariateChecklist", "covariateSearchInput", "selectAllCoxCovariatesButton", ["age"], "stage"),
        strata: pick("strata", "strataChecklist", "strataSearchInput", "selectAllCoxStrataButton", ["treatment"], "sex"),
        table: pick("table", "cohortVariableChecklist", "cohortVariableSearchInput", "selectAllCohortVariablesButton", ["age"], "sex"),
        markers: pick("markers", "markerChecklist", "markerSearchInput", "selectAllMarkersButton", ["biomarker_score"], "immune"),
      };
    """, dataset=example_dataset)

    assert result["cox"] == ["age", "stage"]
    assert result["strata"] == ["sex", "treatment"]
    assert result["table"] == ["age", "sex"]
    assert result["markers"] == ["biomarker_score", "immune_index"]


# ── Behaviour behind source-text tests (R13-8) ─────────────────


def test_km_banner_and_tables_format_p_values_as_p_values(tmp_path: Path, example_dataset: dict) -> None:
    """test_app.py checks the formatter's source text; here the page shows "p<0.001", "p=0.012" and "<0.001"."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      let pValue = 0.0004;
      page.fetchHandler = (request) => ({ status: 200, body: {
        analysis: { summary_table: [{ Group: "All", N: 360 }], risk_table: { rows: [], columns: [] },
          pairwise_table: [{ Comparison: "A vs B", "P value": pValue, "Hazard ratio": 0.0004 }], cohort: { n: 360, events: 250 },
          test: { test: "Log-rank", p_value: pValue } },
        figure: { data: [{ x: [0], y: [1] }], layout: {} }, request_config: request.json() } });
      page.run("refs.runKmButton.click()");
      await page.settle();
      const small = { banner: page.run("refs.kmMetaBanner.textContent"), cells: page.run("refs.kmPairwiseShell.querySelectorAll('td').map((td) => td.textContent)") };
      pValue = 0.0123;
      page.run("refs.runKmButton.click()");
      await page.settle();
      return { small, larger: page.run("refs.kmMetaBanner.textContent") };
    """, dataset=example_dataset)

    assert result["small"]["banner"].endswith("Log-rank p<0.001")
    assert result["small"]["cells"] == ["A vs B", "<0.001", "4.00e-4"]
    assert result["larger"].endswith("Log-rank p=0.012")


def test_a_run_cancelled_by_an_endpoint_change_never_lands(tmp_path: Path, example_dataset: dict) -> None:
    """test_app.py checks the request-token source text; here a KM run in flight is aborted and leaves no result."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const hold = deferred();
      page.fetchHandler = (request) => (request.url.endsWith("/api/kaplan-meier") ? hold.promise : { status: 500, body: { detail: "unexpected" } });
      page.run("refs.runKmButton.click()");
      await page.settle(3);
      const km = page.requests.find((request) => request.url.endsWith("/api/kaplan-meier"));
      page.change("#eventColumn", "pfs_event");
      hold.resolve({ status: 200, body: { analysis: { summary_table: [], risk_table: { rows: [], columns: [] }, pairwise_table: [], cohort: { n: 1 } },
        figure: { data: [{ x: [0], y: [1] }], layout: {} }, request_config: km.json() } });
      await page.settle();
      return { aborted: km.signal.aborted, km: page.run("state.km"), plotted: page.run("Boolean(refs.kmPlot.data)"), busy: page.run("isScopeBusy('km')"), toasts: page.toasts() };
    """, dataset=example_dataset)

    assert result == {"aborted": True, "km": None, "plotted": False, "busy": False, "toasts": []}


# ── Derived groupings ───────────────────────────────────────────


def test_only_an_optimal_cutpoint_labels_its_groups_as_risk(tmp_path: Path) -> None:
    """R13-1: a median split's "High"/"Low" are source values, not risk groups."""
    result = _run_page(tmp_path, r"""
      const pills = (method) => {
        page.context.__method = method;
        page.run(`renderDerivedGroupSummary("biomarker__" + __method, { method: __method, counts: [{ group: "Low", n: 22 }, { group: "High", n: 18 }] })`);
        return page.run("refs.deriveSummary.querySelectorAll('.count-pill span').map((span) => span.textContent)");
      };
      return { median: pills("median_split"), optimal: pills("optimal_cutpoint") };
    """)

    assert result == {"median": ["Low", "High"], "optimal": ["Low risk", "High risk"]}


def _derive_script(controls: str, recipe: str) -> str:
    return r"""
      await loadDataset(page, fixtures.dataset);
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); syncDeriveToggleButton();");
      page.change("#deriveSource", "biomarker");
    """ + controls + r"""
      const recipe = """ + recipe + r""";
      const columns = [...fixtures.dataset.columns, { name: recipe.column_name, kind: "categorical", n_unique: 2, unique_preview: ["Low", "High"], missing: 0, non_missing: 40 }];
      page.fetchHandler = (request) => (request.url.endsWith("/api/derive-group")
        ? { status: 200, body: { ...fixtures.dataset, dataset_id: "synthetic-2", columns, derived_column: recipe.column_name,
            derive_summary: { method: recipe.method, cutoff_spec: recipe.cutoff_spec || null, counts: [{ group: "Low", n: 22 }, { group: "High", n: 18 }], recipe } } }
        : { status: 200, body: { preview: { analyzable_rows: 40, outcome_rows: 40, events: 26, estimated_parameters: 1, events_per_parameter: 26 } } });
      page.run("refs.deriveButton.click()");
      await page.settle();
      const notes = () => page.run("refs.deriveSummary.querySelectorAll('.note-box').map((note) => note.textContent).join(' | ')");
      const created = { group: page.run("refs.groupColumn.value"), notes: notes() };
      page.change("#deriveMethod", "tertile_split");
      return { created, afterEdit: notes() };
    """


def test_a_blank_column_name_describes_the_automatically_named_grouping(tmp_path: Path) -> None:
    """R13-4: right after Create with a blank name, the card does not say the controls "do not describe" it."""
    result = _run_page(tmp_path, _derive_script(
        r"""page.change("#deriveMethod", "median_split"); page.change("#deriveColumnName", "");""",
        r"""{ source_column: "biomarker", column_name: "biomarker__median_split_2", method: "median_split" }""",
    ), dataset=_synthetic_dataset())

    assert result["created"]["group"] == "biomarker__median_split_2"
    assert "do not describe" not in result["created"]["notes"]
    # A real edit of the draft still says so.
    assert "They do not describe biomarker__median_split_2." in result["afterEdit"]


def test_percentile_specs_compare_as_numbers(tmp_path: Path) -> None:
    """R13-4: "25, 25" typed in the form is the server's "25,25"."""
    result = _run_page(tmp_path, _derive_script(
        r"""page.change("#deriveMethod", "percentile_split"); page.change("#deriveCutoff", "25, 25.0"); page.change("#deriveColumnName", "tails");""",
        r"""{ source_column: "biomarker", column_name: "tails", method: "percentile_split", cutoff_spec: "25,25" }""",
    ), dataset=_synthetic_dataset())

    assert result["created"]["group"] == "tails"
    assert "do not describe" not in result["created"]["notes"]
    assert "They do not describe tails." in result["afterEdit"]

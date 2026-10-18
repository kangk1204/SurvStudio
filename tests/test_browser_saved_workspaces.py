from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_browser_e2e import _free_port, _launch_browser, _wait_for_server


@pytest.fixture
def durable_browser_server(tmp_path):
    root = Path(__file__).resolve().parents[1]
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "SURVSTUDIO_STATE_DIR": str(tmp_path / "archive"), "SURVSTUDIO_WORKSPACE_OWNER": "synthetic-browser-owner"}
    process = None
    def start():
        nonlocal process
        launcher = """
import importlib, sys
from fastapi import Request
import uvicorn
module = importlib.import_module('survival_toolkit.app')
@module.app.post('/_test/store-result')
async def fixture_result(request: Request):
    body = await request.json()
    metadata = module.app.state.durable_workspace.save_result(body['dataset_id'], body['route'], body['payload'], source_version='synthetic-browser-fixture', controls=body['payload']['request_config'])
    return metadata
uvicorn.run(module.app, host='127.0.0.1', port=int(sys.argv[1]), log_level='error')
"""
        process = subprocess.Popen([sys.executable, "-c", launcher, str(port)], cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_for_server(base_url)
    def stop():
        nonlocal process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
            process = None
    start()
    try:
        yield base_url, stop, start
    finally:
        stop()


def _saved(page):
    page.wait_for_function("() => savedWorkState.enabled && !savedWorkState.pending && !savedWorkState.error")
    page.wait_for_timeout(450)
    page.wait_for_function("() => !savedWorkState.pending && !savedWorkState.error")


def test_browser_reload_and_server_restart_restore_exact_km_and_ml_without_fitting(durable_browser_server):
    playwright = pytest.importorskip("playwright.sync_api")
    url, stop, start = durable_browser_server
    with playwright.sync_playwright() as api:
        browser = _launch_browser(api)
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        analyses = []
        page.on("request", lambda request: analyses.append(request.url) if request.method == "POST" and any(path in request.url for path in ("/api/kaplan-meier", "/api/ml-model", "/api/deep-model")) else None)
        page.goto(url)
        page.wait_for_function("() => savedWorkState.enabled")
        page.locator("#loadExampleButton").click()
        page.locator("#workspace").wait_for(state="visible")
        page.locator("#runKmButton").click()
        page.wait_for_function("() => Boolean(state.km)")
        _saved(page)
        before = page.evaluate("() => ({dataset: state.dataset.dataset_id, hash: state.dataset.dataset_hash, km: state.km, controls: captureControlSnapshot()})")
        assert len(analyses) == 1
        page.reload()
        page.wait_for_function("() => Boolean(state.km) && !savedWorkState.restoring")
        assert page.evaluate("Object.fromEntries(Object.entries(state.km).filter(([key]) => key !== 'saved_result'))") == {key: value for key, value in before["km"].items() if key != "saved_result"}
        assert page.evaluate("state.km.saved_result.result_id") == before["km"]["saved_result"]["result_id"]
        assert page.evaluate("state.dataset.dataset_id") == before["dataset"]
        assert len(analyses) == 1
        # A fixed response fixture exercises the complete single-model renderer and screen restore;
        # no model fitting occurs in this browser test.
        ml = page.evaluate("""async () => {
          const config = { dataset_id: state.dataset.dataset_id, time_column: refs.timeColumn.value,
            event_column: refs.eventColumn.value, event_positive_value: refs.eventPositiveValue.value,
            model_type: 'rsf', features: ['age'], categorical_features: [], random_state: 42 };
          const payload = { dataset_hash: state.dataset.dataset_hash, request_config: config,
            analysis: { model_stats: { c_index: 0, n_patients: 360, n_features: 1, evaluation_mode: 'holdout' },
              scientific_summary: null }, importance_figure: null, shap_figure: null,
              exploratory: { withheld_p: null, zero: 0 } };
          await renderMlModelResult(payload);
          queueSavedWorkspace();
          await saveCurrentWorkspace();
          return payload;
        }""")
        _saved(page)
        stop(); start()
        page.reload()
        page.wait_for_function("() => Boolean(state.ml) && Boolean(state.km) && !savedWorkState.restoring")
        assert page.evaluate("Object.fromEntries(Object.entries(state.km).filter(([key]) => key !== 'saved_result'))") == {key: value for key, value in before["km"].items() if key != "saved_result"}
        assert page.evaluate("state.km.saved_result.result_id") == before["km"]["saved_result"]["result_id"]
        assert page.evaluate("state.ml") == ml
        assert page.evaluate("state.ml.exploratory.withheld_p") is None
        assert page.evaluate("state.ml.analysis.model_stats.c_index") == 0
        assert len(analyses) == 1
        page.locator("#savedWorkButton").click()
        page.locator("#savedWorkList .saved-work-item").first.wait_for()
        assert "cohort" in page.locator("#savedWorkList").inner_text().lower()
        browser.close()


def test_browser_save_failure_is_visible_and_does_not_claim_saved(durable_browser_server):
    playwright = pytest.importorskip("playwright.sync_api")
    url, _, _ = durable_browser_server
    with playwright.sync_playwright() as api:
        browser = _launch_browser(api)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("() => savedWorkState.enabled")
        page.locator("#loadExampleButton").click()
        page.locator("#workspace").wait_for(state="visible")
        _saved(page)
        page.route("**/api/saved-workspaces/*/snapshot", lambda route: route.fulfill(status=503, content_type="application/json", body='{"detail":"Storage unavailable"}'))
        page.evaluate("() => { queueSavedWorkspace(); return saveCurrentWorkspace(); }")
        assert page.locator("#autosaveStatus").get_attribute("data-tone") == "error"
        assert "not saved" in page.locator("#autosaveStatus").inner_text().lower()
        assert page.evaluate("savedWorkState.error") is True
        dataset_id = page.evaluate("state.dataset.dataset_id")
        with pytest.raises(Exception, match="could not be saved"):
            page.evaluate("async (id) => await reopenSavedWorkspace(id)", dataset_id)
        assert page.evaluate("state.dataset.dataset_id") == dataset_id
        browser.close()


def test_browser_reopen_is_serialized_and_history_results_remain_accessible(durable_browser_server):
    playwright = pytest.importorskip("playwright.sync_api")
    url, _, _ = durable_browser_server
    with playwright.sync_playwright() as api:
        browser = _launch_browser(api)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("() => savedWorkState.enabled")
        page.locator("#loadExampleButton").click()
        page.locator("#workspace").wait_for(state="visible")
        page.locator("#runKmButton").click()
        page.wait_for_function("() => Boolean(state.km)")
        _saved(page)
        outcome = page.evaluate("""async () => {
          const id = state.dataset.dataset_id;
          const outcomes = await Promise.allSettled([reopenSavedWorkspace(id), reopenSavedWorkspace(id)]);
          return outcomes.map((item) => item.status);
        }""")
        assert outcome == ["fulfilled", "rejected"]
        page.locator("#savedWorkButton").click()
        page.locator("#savedWorkList details summary").first.click()
        page.locator("#savedWorkList details button").first.wait_for(state="visible")
        assert "kaplan-meier" in page.locator("#savedWorkList details button").first.inner_text()
        browser.close()


def test_browser_history_restores_nondefault_model_and_seed_settings(durable_browser_server):
    playwright = pytest.importorskip("playwright.sync_api")
    url, _, _ = durable_browser_server
    with playwright.sync_playwright() as api:
        browser = _launch_browser(api)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("() => savedWorkState.enabled")
        page.locator("#loadExampleButton").click()
        page.locator("#workspace").wait_for(state="visible")
        _saved(page)
        dataset = page.evaluate("state.dataset")
        endpoint = page.evaluate("() => ({time_column: refs.timeColumn.value, event_column: refs.eventColumn.value, event_positive_value: refs.eventPositiveValue.value})")
        config = {"dataset_id": dataset["dataset_id"], **endpoint, "model_type": "lasso_cox", "features": ["age"], "categorical_features": [], "random_state": 73, "evaluation_strategy": "holdout", "n_estimators": 170, "learning_rate": 0.07, "compute_shap": False, "shap_safe_mode": True}
        # Endpoint exists only in this test launcher; it saves a fixed response, not a fitted model.
        payload = {"dataset_hash": dataset["dataset_hash"], "request_config": config, "analysis": {"model_stats": {"c_index": 0.6123456789, "evaluation_mode": "holdout", "n_patients": 360, "n_features": 1}}, "importance_figure": None, "shap_figure": None}
        response = page.request.post(url + "/_test/store-result", data={"dataset_id": dataset["dataset_id"], "route": "/api/ml-model", "payload": payload})
        assert response.ok, response.text()
        record = response.json()
        page.evaluate("async (args) => await reopenSavedWorkspace(args.id, {resultId: args.result})", {"id": dataset["dataset_id"], "result": record["result_id"]})
        assert page.locator("#mlModelType").input_value() == "lasso_cox"
        assert page.locator("#mlRandomSeed").input_value() == "73"
        assert page.locator("#dlRandomSeed").input_value() == "73"
        assert page.evaluate("Boolean(currentGoalResult('ml'))") is True
        assert page.evaluate("state.ml.analysis.model_stats.c_index") == 0.6123456789
        browser.close()


def test_browser_serialization_and_old_save_failure_keep_truthful_status(durable_browser_server):
    playwright = pytest.importorskip("playwright.sync_api")
    url, _, _ = durable_browser_server
    with playwright.sync_playwright() as api:
        browser = _launch_browser(api)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("() => savedWorkState.enabled")
        page.locator("#loadExampleButton").click()
        page.locator("#workspace").wait_for(state="visible")
        _saved(page)
        outcome = page.evaluate("""async () => {
          const previous = state.km;
          const cyclic = {}; cyclic.self = cyclic; state.km = cyclic;
          const saved = await saveCurrentWorkspace();
          const failed = {saved, error: savedWorkState.error, status: document.getElementById('autosaveStatus').textContent};
          state.km = previous;
          await saveCurrentWorkspace();
          return failed;
        }""")
        assert outcome["saved"] is False and outcome["error"] is True
        assert "not saved" in outcome["status"].lower()
        outcome = page.evaluate("""async () => {
          const originalFetch = fetchJSON;
          const originalDataset = state.dataset;
          let rejectOld;
          fetchJSON = (path, options) => path.includes(originalDataset.dataset_id)
            ? new Promise((resolve, reject) => { rejectOld = reject; }) : Promise.resolve({status:'saved'});
          savedWorkState.generation += 1;
          const first = saveCurrentWorkspace();
          while (!rejectOld) await Promise.resolve();
          state.dataset = {...originalDataset, dataset_id: 'another-synthetic-cohort'};
          savedWorkState.generation += 1;
          const second = saveCurrentWorkspace();
          rejectOld(new Error('old dataset failed'));
          const results = await Promise.all([first, second]);
          const final = {results, error: savedWorkState.error, pending: savedWorkState.pending,
            status: document.getElementById('autosaveStatus').textContent};
          fetchJSON = originalFetch; state.dataset = originalDataset;
          return final;
        }""")
        assert outcome["results"] == [False, True]
        assert outcome["error"] is False and outcome["pending"] is False
        assert outcome["status"] == "Saved automatically"
        browser.close()


def test_browser_missing_saved_matrix_does_not_restore_a_valid_attachment(durable_browser_server, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    url, _, _ = durable_browser_server
    clinical = tmp_path / "synthetic-clinical.csv"
    clinical.write_text("patient_id,time,event,age\n" + "\n".join(f"P{i},{i + 1},{int(i % 3 != 0)},{40 + i}" for i in range(24)))
    expression = tmp_path / "synthetic-expression.csv"
    expression.write_text("sample,GeneA,GeneB\n" + "\n".join(f"P{i},{i * .1},{i * .2 + 1}" for i in range(24)))
    with playwright.sync_playwright() as api:
        browser = _launch_browser(api)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("() => savedWorkState.enabled")
        page.locator("#datasetFile").set_input_files(str(clinical))
        page.locator("#workspace").wait_for(state="visible")
        page.evaluate("() => { activateTab('markers'); document.getElementById('markerMatrixDetails').open = true; }")
        page.locator("#markerMatrixIdColumn").select_option("patient_id")
        page.locator("#markerMatrixFile").set_input_files(str(expression))
        page.locator("#attachMarkerMatrixButton").click()
        page.wait_for_function("() => Boolean(state.markerMatrix?.matrix_id)")
        _saved(page)
        matrix_id = page.evaluate("state.markerMatrix.matrix_id")
        dataset_id = page.evaluate("state.dataset.dataset_id")
        deleted = page.request.delete(url + f"/api/marker-matrix/{matrix_id}?dataset_id={dataset_id}")
        assert deleted.ok
        page.reload()
        page.wait_for_function("() => state.dataset && !savedWorkState.restoring")
        assert page.evaluate("state.markerMatrix") is None
        assert "reattach" in page.locator("#runtimeBanner").inner_text().lower()
        browser.close()

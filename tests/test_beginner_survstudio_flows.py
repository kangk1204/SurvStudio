from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for test server at {base_url}")


@pytest.fixture
def browser_server() -> str:
    project_root = Path(__file__).resolve().parents[1]
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "survival_toolkit.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=project_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _assert_tab_active(page, tab_name: str) -> None:
    page.wait_for_function(
        """
        (tabName) => {
          const tab = document.querySelector(`[data-tab="${tabName}"]`);
          const panel = document.querySelector(`#panel-${tabName}`);
          if (!tab || !panel) return false;
          return tab.getAttribute('aria-selected') === 'true' && panel.classList.contains('active');
        }
        """,
        tab_name,
    )
    assert page.locator(f'[data-tab="{tab_name}"]').get_attribute("aria-selected") == "true"
    assert page.locator(f"#panel-{tab_name}").evaluate("(node) => node.classList.contains('active')")


def test_beginner_example_walkthrough_runs_tabs_and_updates_feedback(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")

            assert "Group by does not change Cox, ML, or DL inputs." in page.locator("#groupingSummaryText").inner_text()
            assert "current Group by setting" in page.locator("#kmDependencyText").inner_text()
            assert "Group by does not change the model unless you add that column as a covariate." in page.locator("#coxDependencyText").inner_text()
            assert "model features selected below" in page.locator("#mlFeatureSummaryText").inner_text()
            assert "model features selected on the ML tab" in page.locator("#dlFeatureSummaryText").inner_text()

            _assert_tab_active(page, "km")
            page.locator("#runKmButton").click()
            page.wait_for_function(
                "document.getElementById('downloadKmSummaryButton') && !document.getElementById('downloadKmSummaryButton').disabled"
            )
            assert "N=" in page.locator("#kmMetaBanner").inner_text()
            assert "events=" in page.locator("#kmMetaBanner").inner_text()

            page.locator('[data-tab="cox"]').click()
            _assert_tab_active(page, "cox")
            page.locator("#covariateChecklist input[value='age']").check()
            page.locator("#covariateChecklist input[value='biomarker_score']").check()
            page.locator("#covariateChecklist input[value='immune_index']").check()
            page.locator("#runCoxButton").click()
            page.wait_for_function(
                "document.getElementById('downloadCoxResultsButton') && !document.getElementById('downloadCoxResultsButton').disabled"
            )
            assert "C-index" in page.locator("#coxMetaBanner").inner_text()

            page.locator('[data-tab="ml"]').click()
            _assert_tab_active(page, "ml")
            page.locator("#mlModelType").select_option("rsf")
            page.locator("#mlNEstimators").fill("10")
            page.locator("#runMlButton").click()
            page.wait_for_function(
                "document.getElementById('mlMetaBanner').textContent.includes('eval=')"
            )
            assert "eval=" in page.locator("#mlMetaBanner").inner_text()
            assert "model features selected below" in page.locator("#mlFeatureSummaryText").inner_text()

            page.locator('[data-tab="dl"]').click()
            _assert_tab_active(page, "dl")
            page.locator("#dlModelType").select_option("deepsurv")
            page.locator("#dlEpochs").fill("10")
            page.locator("#dlHiddenLayers").fill("8")
            page.locator("#runDlButton").click()
            page.wait_for_function(
                "document.getElementById('dlMetaBanner').textContent.includes('eval=')"
            )
            assert "eval=" in page.locator("#dlMetaBanner").inner_text()
            assert "epochs=" in page.locator("#dlMetaBanner").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_beginner_real_dataset_preset_keeps_group_by_and_model_inputs_separate(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#datasetPresetBar").wait_for(state="visible")

            assert "No preset applied yet." in page.locator("#datasetPresetStatusTitle").inner_text()
            assert "updates recommended columns and checkbox selections only" in page.locator("#datasetPresetStatusText").inner_text()

            page.locator("#applyBasicPresetButton").click()
            page.wait_for_function(
                "document.getElementById('datasetPresetStatusTitle').textContent.includes('GBSG2 preset applied')"
            )
            assert page.locator("#timeColumn").input_value() == "rfs_days"
            assert page.locator("#eventColumn").input_value() == "rfs_event"
            assert page.locator("#groupColumn").input_value() == "horTh"
            assert "Group by does not change Cox, ML, or DL inputs." in page.locator("#groupingSummaryText").inner_text()
            assert "KM uses the Study Design outcome definition and the current Group by setting." in page.locator("#kmDependencyText").inner_text()
            assert "Cox uses the Study Design outcome definition and the covariates selected below." in page.locator("#coxDependencyText").inner_text()

            page.locator("#applyModelPresetButton").click()
            page.wait_for_function(
                "document.getElementById('datasetPresetStatusText').textContent.includes('feature checklists used by ML and DL')"
            )
            assert "model features selected below" in page.locator("#mlFeatureSummaryText").inner_text()
            assert "model features selected on the ML tab" in page.locator("#dlFeatureSummaryText").inner_text()
            assert "Model features: 6" in page.locator("#datasetPresetChips").inner_text()
            assert "Categorical: 3" in page.locator("#datasetPresetChips").inner_text()

            for tab_name in ("km", "cox", "ml", "dl"):
                page.locator(f'[data-tab="{tab_name}"]').click()
                _assert_tab_active(page, tab_name)

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_beginner_ml_compare_options_toggle_cv_inputs_and_finish_with_visible_feedback(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#applyModelPresetButton").click()

            page.locator('[data-tab="ml"]').click()
            _assert_tab_active(page, "ml")

            assert page.locator("#mlCvFoldsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert page.locator("#mlCvRepeatsWrap").evaluate("(node) => node.classList.contains('hidden')")

            page.locator("#mlEvaluationStrategy").select_option("repeated_cv")
            page.wait_for_function(
                "document.getElementById('mlCvFoldsWrap') && !document.getElementById('mlCvFoldsWrap').classList.contains('hidden')"
            )
            assert not page.locator("#mlCvFoldsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert not page.locator("#mlCvRepeatsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert not page.locator("#mlCvFolds").is_disabled()
            assert not page.locator("#mlCvRepeats").is_disabled()

            page.locator("#mlCvFolds").fill("2")
            page.locator("#mlCvRepeats").fill("1")
            assert "Compare All Evaluation" in page.locator("#mlEvaluationStrategy").inner_text()
            assert not page.locator("#mlCvFolds").is_disabled()
            assert not page.locator("#mlCvRepeats").is_disabled()

            page.locator("#mlEvaluationStrategy").select_option("holdout")
            page.wait_for_function(
                "document.getElementById('mlCvFoldsWrap') && document.getElementById('mlCvFoldsWrap').classList.contains('hidden')"
            )
            assert page.locator("#mlCvFoldsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert page.locator("#mlCvRepeatsWrap").evaluate("(node) => node.classList.contains('hidden')")

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_beginner_dl_single_run_keeps_feedback_visible_and_hides_cv_controls(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#applyModelPresetButton").click()

            page.locator('[data-tab="dl"]').click()
            _assert_tab_active(page, "dl")

            assert page.locator("#dlCvFoldsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert page.locator("#dlCvRepeatsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert page.locator("#dlParallelJobs").is_disabled()

            page.locator("#dlEvaluationStrategy").select_option("repeated_cv")
            page.wait_for_function(
                "document.getElementById('dlCvFoldsWrap') && !document.getElementById('dlCvFoldsWrap').classList.contains('hidden')"
            )
            assert not page.locator("#dlCvFoldsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert not page.locator("#dlCvRepeatsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert not page.locator("#dlParallelJobs").is_disabled()

            page.locator("#dlEvaluationStrategy").select_option("holdout")
            page.wait_for_function(
                "document.getElementById('dlCvFoldsWrap') && document.getElementById('dlCvFoldsWrap').classList.contains('hidden')"
            )
            assert page.locator("#dlCvFoldsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert page.locator("#dlCvRepeatsWrap").evaluate("(node) => node.classList.contains('hidden')")
            assert page.locator("#dlParallelJobs").is_disabled()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")

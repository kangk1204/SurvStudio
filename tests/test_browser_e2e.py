from __future__ import annotations

import json
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
        try:
            sock.bind(("127.0.0.1", 0))
        except PermissionError as exc:
            pytest.skip(f"Localhost port binding is unavailable in this environment: {exc}")
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


def _is_playwright_environment_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "executable doesn't exist",
        "download new browsers",
        "host system is missing dependencies",
        "looks like playwright was just installed",
        "please run the following command to download new browsers",
    )
    return any(marker in message for marker in markers)


def _launch_browser(api):
    try:
        return api.chromium.launch(headless=True)
    except Exception as exc:
        if not _is_playwright_environment_error(exc):
            raise
        last_exc = exc
        for candidate in (
            os.environ.get("CHROME_BIN"),
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
        ):
            if not candidate or not Path(candidate).exists():
                continue
            try:
                return api.chromium.launch(headless=True, executable_path=candidate)
            except Exception as chrome_exc:
                last_exc = chrome_exc
        raise last_exc


def _switch_to_expert(page) -> None:
    page.locator("#workspace").wait_for(state="visible")
    if page.locator("#guidedShell").is_visible():
        page.evaluate("() => document.getElementById('expertModeButton')?.click()")
        page.wait_for_function("document.body.dataset.uiMode === 'expert'")


def _assert_tab_active(page, tab_name: str) -> None:
    page.wait_for_function(
        """
        (name) => {
          const tab = document.querySelector(`[data-tab="${name}"]`);
          const panel = document.querySelector(`#panel-${name}`);
          return tab && panel
            && tab.getAttribute('aria-selected') === 'true'
            && panel.classList.contains('active');
        }
        """,
        arg=tab_name,
    )


def _open_predictive_workbench(page, model_key: str | None = None) -> None:
    page.locator('[data-tab="benchmark"]').click()
    _assert_tab_active(page, "benchmark")
    if page.locator("#benchmarkWorkbench").is_hidden():
        page.locator("#benchmarkComparisonShell [data-benchmark-model]").wait_for(state="visible")
        page.locator("#benchmarkComparisonShell [data-benchmark-model]").first.click()
        page.wait_for_function(
            "() => document.getElementById('benchmarkWorkbench') && !document.getElementById('benchmarkWorkbench').classList.contains('hidden')"
        )
    if model_key is not None:
        page.locator("#predictiveModelSelector").select_option(model_key)
        page.wait_for_function(
            "(value) => document.getElementById('predictiveModelSelector')?.value === value",
            arg=model_key,
        )


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


def test_browser_downloads_km_summary_csv_and_png(browser_server: str, tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            page.locator("#runKmButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadKmSummaryButton').disabled && !document.getElementById('downloadKmPngButton').disabled"
            )

            with page.expect_download() as summary_info:
                page.locator("#downloadKmSummaryButton").click()
            summary_download = summary_info.value
            summary_path = tmp_path / (summary_download.suggested_filename or "km_summary.csv")
            summary_download.save_as(summary_path)
            assert summary_path.exists()
            assert summary_path.stat().st_size > 0

            with page.expect_download() as png_info:
                page.locator("#downloadKmPngButton").click()
            png_download = png_info.value
            png_path = tmp_path / (png_download.suggested_filename or "km_curve.png")
            png_download.save_as(png_path)
            assert png_path.exists()
            assert png_path.stat().st_size > 0

            context.close()
            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_preset_application_shows_visible_feedback(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _switch_to_expert(page)
            page.locator("#datasetPresetBar").wait_for(state="visible")
            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")

            assert page.locator("#datasetPresetStatusTitle").inner_text() == "No preset applied yet."
            assert "does not run an analysis" in page.locator("#datasetPresetStatusText").inner_text()

            page.locator("#applyBasicPresetButton").click()
            page.wait_for_function(
                "document.getElementById('datasetPresetStatusTitle').textContent.includes('GBSG2 preset applied')"
            )
            assert page.locator("#timeColumn").input_value() == "rfs_days"
            assert page.locator("#eventColumn").input_value() == "rfs_event"
            assert page.locator("#groupColumn").input_value() == "horTh"
            assert "Cox covariates: 6" in page.locator("#datasetPresetChips").inner_text()

            page.locator("#applyModelPresetButton").click()
            page.wait_for_function(
                "document.getElementById('dlFeatureSummaryText').textContent.includes('Training inputs come only from the shared ML/DL model feature selections')"
            )
            selected_feature_count = page.eval_on_selector_all(
                "#dlModelFeatureChecklist input",
                "els => els.filter(e => e.checked).length",
            )
            selected_categorical_count = page.eval_on_selector_all(
                "#dlModelCategoricalChecklist input",
                "els => els.filter(e => e.checked).length",
            )
            assert page.locator('[data-tab="benchmark"]').get_attribute("aria-selected") == "true"
            assert "feature checklists used by ML and DL" in page.locator("#datasetPresetStatusText").inner_text()
            assert f"Model features: {selected_feature_count}" in page.locator("#datasetPresetChips").inner_text()
            assert f"Model features: {selected_feature_count}" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Grouping only: horTh" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert f"Categorical: {selected_categorical_count}" in page.locator("#dlFeatureSummaryChips").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_benchmark_tab_combines_latest_ml_and_dl_compare_outputs(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _mock_ml_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "analysis": {
                    "comparison_table": [
                        {"model": "Random Survival Forest", "c_index": 0.714, "evaluation_mode": "holdout", "n_features": 6, "training_time_ms": 121.5, "rank": 1},
                        {"model": "LASSO-Cox", "c_index": 0.681, "evaluation_mode": "holdout", "n_features": 4, "training_time_ms": 39.4, "rank": 2},
                    ],
                    "evaluation_mode": "holdout",
                    "scientific_summary": {
                        "status": "review",
                        "headline": "ML comparison complete.",
                        "strengths": [],
                        "cautions": [],
                        "next_steps": [],
                    },
                    "manuscript_tables": {"model_performance_table": []},
                },
                "request_config": body,
            }),
        )

    def _mock_dl_requests(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        if body.get("model_type") != "compare":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "analysis": {
                        "model": "DeepHit",
                        "c_index": 0.726,
                        "evaluation_mode": "holdout",
                        "epochs_trained": 3,
                        "best_monitor_epoch": 3,
                        "stopped_early": False,
                        "max_epochs_requested": 40,
                        "n_features": 6,
                        "loss_history": [1.12, 0.94, 0.81],
                        "monitor_history": [1.08, 0.91, 0.79],
                        "scientific_summary": {
                            "status": "review",
                            "headline": "DeepHit single-model training complete.",
                            "strengths": ["Loss decreased over epochs."],
                            "cautions": [],
                            "next_steps": [],
                        },
                    },
                    "figures": {
                        "loss": {
                            "data": [
                                {
                                    "type": "scatter",
                                    "mode": "lines+markers",
                                    "x": [1, 2, 3],
                                    "y": [1.12, 0.94, 0.81],
                                    "name": "Training loss",
                                },
                            ],
                            "layout": {"title": {"text": "DeepHit loss"}},
                        },
                    },
                    "request_config": body,
                }),
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "analysis": {
                    "comparison_table": [
                        {"model": "DeepHit", "c_index": 0.731, "evaluation_mode": "holdout", "epochs_trained": 40, "n_features": 6, "training_time_ms": 442.7, "rank": 1},
                        {"model": "DeepSurv", "c_index": 0.703, "evaluation_mode": "holdout", "epochs_trained": 40, "n_features": 6, "training_time_ms": 388.1, "rank": 2},
                    ],
                    "evaluation_mode": "holdout",
                    "scientific_summary": {
                        "status": "review",
                        "headline": "DL comparison complete.",
                        "strengths": [],
                        "cautions": [],
                        "next_steps": [],
                    },
                    "manuscript_tables": {"model_performance_table": []},
                },
                "request_config": body,
            }),
        )

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.route("**/api/ml-model", _mock_ml_compare)
            page.route("**/api/deep-model", _mock_dl_requests)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")
            assert page.locator("#benchmarkWorkbench").is_hidden()
            page.locator("#runPredictiveCompareAllButton").click()
            page.wait_for_function(
                "document.getElementById('mlMetaBanner').textContent.includes('Screening top model=')"
            )
            page.wait_for_function(
                "document.getElementById('dlMetaBanner').textContent.includes('Screening top model=')"
            )

            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")

            assert "Predictive Overview" in page.locator("#benchmarkSummaryGrid").inner_text()
            assert "BOARD READY" in page.locator("#benchmarkSummaryGrid").inner_text()
            page.wait_for_function(
                "() => { const plot = document.getElementById('benchmarkComparisonPlot'); return plot && !plot.classList.contains('hidden') && Array.isArray(plot.data) && plot.data.length === 1 && plot.data[0].x.length >= 4; }"
            )
            families = page.eval_on_selector(
                "#benchmarkComparisonPlot",
                "el => Array.from(new Set((el.data?.[0]?.customdata || []).map(row => row[1]))).sort()"
            )
            assert families == ["Classical ML", "Deep Learning"]
            assert not page.locator("#mlComparisonPlot").is_visible()
            assert not page.locator("#dlComparisonPlot").is_visible()
            assert "Random Survival Forest" in page.locator("#benchmarkComparisonShell").inner_text()
            assert "DeepHit" in page.locator("#benchmarkComparisonShell").inner_text()
            assert "current screening rows from the latest ML and DL comparison outputs" in page.locator("#benchmarkTableNote").inner_text()

            first_row_text = page.locator("#benchmarkComparisonShell tbody tr").nth(0).inner_text()
            assert "DeepHit" in first_row_text
            assert "Deep Learning" in first_row_text

            page.locator('#benchmarkComparisonShell tbody tr').nth(0).locator('[data-benchmark-model]').click()
            _assert_tab_active(page, "benchmark")
            page.wait_for_function(
                "document.getElementById('predictiveModelSelector').value === 'deephit'"
            )
            assert page.locator("#benchmarkWorkbench").is_visible()
            assert not page.locator("#benchmarkMlMount").is_visible()
            assert page.locator("#benchmarkDlMount").is_visible()
            assert page.locator("#benchmarkDlMount .model-choice-field").is_hidden()
            assert page.locator("#benchmarkDlMount #runDlCompareButton").is_hidden()
            assert page.locator("#benchmarkDlMount #runDlCompareInlineButton").is_hidden()
            page.locator("#runPredictiveWorkbenchButton").click()
            page.wait_for_function(
                "document.getElementById('dlMetaBanner').textContent.includes('DEEPHIT: Holdout C-index=0.726')"
            )
            page.wait_for_function(
                "() => { const plot = document.getElementById('dlLossPlot'); return plot && Array.isArray(plot.data) && plot.data.length === 1; }"
            )
            page.locator("#closePredictiveWorkbenchButton").click()
            page.wait_for_function(
                "() => document.getElementById('benchmarkWorkbench').classList.contains('hidden')"
            )
            page.wait_for_function(
                "() => document.querySelectorAll('#benchmarkComparisonShell tbody tr').length === 4"
            )
            assert "Both model families are currently represented." in page.locator("#benchmarkSummaryGrid").inner_text()
            assert "current screening rows from the latest ML and DL comparison outputs" in page.locator("#benchmarkTableNote").inner_text()
            assert "Random Survival Forest" in page.locator("#benchmarkComparisonShell").inner_text()
            assert "DeepHit" in page.locator("#benchmarkComparisonShell").inner_text()
            assert "Deep Learning Survival Models" in page.locator("#benchmarkDlMount").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_benchmark_hides_partial_board_until_unified_compare_finishes(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _mock_ml_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "analysis": {
                    "comparison_table": [
                        {"model": "Random Survival Forest", "c_index": 0.714, "evaluation_mode": "holdout", "n_features": 6, "training_time_ms": 121.5, "rank": 1},
                    ],
                    "evaluation_mode": "holdout",
                    "scientific_summary": {"status": "review", "headline": "ML comparison complete.", "strengths": [], "cautions": [], "next_steps": []},
                    "manuscript_tables": {"model_performance_table": []},
                },
                "request_config": body,
            }),
        )

    def _mock_dl_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        time.sleep(2.5)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "analysis": {
                    "comparison_table": [
                        {"model": "DeepHit", "c_index": 0.731, "evaluation_mode": "holdout", "epochs_trained": 40, "n_features": 6, "training_time_ms": 442.7, "rank": 1},
                    ],
                    "evaluation_mode": "holdout",
                    "scientific_summary": {"status": "review", "headline": "DL comparison complete.", "strengths": [], "cautions": [], "next_steps": []},
                    "manuscript_tables": {"model_performance_table": []},
                },
                "request_config": body,
            }),
        )

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.route("**/api/ml-model", _mock_ml_compare)
            page.route("**/api/deep-model", _mock_dl_compare)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")

            page.locator("#runPredictiveCompareAllButton").click()
            page.wait_for_function(
                "document.getElementById('benchmarkComparisonShell').textContent.includes('Partial leaderboard rows stay hidden until both model families finish.')"
            )

            page.wait_for_function(
                "document.getElementById('mlMetaBanner').textContent.includes('Screening top model=')"
            )
            page.wait_for_function(
                "document.getElementById('dlMetaBanner').textContent.includes('Screening top model=')"
            )
            page.wait_for_function(
                "() => { const plot = document.getElementById('benchmarkComparisonPlot'); return plot && !plot.classList.contains('hidden') && Array.isArray(plot.data) && plot.data[0].x.length === 2; }"
            )
            assert "RUNNING" not in page.locator("#benchmarkSummaryGrid").inner_text()
            assert "Random Survival Forest" in page.locator("#benchmarkComparisonShell").inner_text()
            assert "DeepHit" in page.locator("#benchmarkComparisonShell").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_benchmark_hides_unified_chart_for_mixed_evaluation_modes(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _mock_ml_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "analysis": {
                    "comparison_table": [
                        {"model": "Random Survival Forest", "c_index": 0.714, "evaluation_mode": "holdout", "n_features": 6, "training_time_ms": 121.5, "rank": 1},
                        {"model": "LASSO-Cox", "c_index": 0.681, "evaluation_mode": "holdout", "n_features": 4, "training_time_ms": 39.4, "rank": 2},
                    ],
                    "evaluation_mode": "holdout",
                    "scientific_summary": {"status": "review", "headline": "ML comparison complete.", "strengths": [], "cautions": [], "next_steps": []},
                    "manuscript_tables": {"model_performance_table": []},
                },
                "request_config": body,
            }),
        )

    def _mock_dl_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "analysis": {
                    "comparison_table": [
                        {"model": "DeepHit", "c_index": 0.731, "evaluation_mode": "repeated_cv", "epochs_trained": 40, "n_features": 6, "training_time_ms": 442.7, "rank": 1},
                        {"model": "DeepSurv", "c_index": 0.703, "evaluation_mode": "repeated_cv", "epochs_trained": 40, "n_features": 6, "training_time_ms": 388.1, "rank": 2},
                    ],
                    "evaluation_mode": "repeated_cv",
                    "scientific_summary": {"status": "review", "headline": "DL comparison complete.", "strengths": [], "cautions": [], "next_steps": []},
                    "manuscript_tables": {"model_performance_table": []},
                },
                "request_config": body,
            }),
        )

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.route("**/api/ml-model", _mock_ml_compare)
            page.route("**/api/deep-model", _mock_dl_compare)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")
            page.locator("#runPredictiveCompareAllButton").click()
            page.wait_for_function(
                "() => document.getElementById('benchmarkSummaryGrid').textContent.includes('Needs alignment') || document.getElementById('benchmarkSummaryGrid').textContent.includes('NEEDS ALIGNMENT')"
            )
            page.wait_for_function(
                "() => document.getElementById('benchmarkComparisonPlot').classList.contains('hidden')"
            )

            assert "same evaluation mode" in page.locator("#benchmarkSummaryGrid").inner_text().lower()
            assert "mixed evaluation paths" in page.locator("#benchmarkPlotNote").inner_text().lower()
            assert "no cross-family ranking is published" in page.locator("#benchmarkTableNote").inner_text().lower()
            assert "family rank" in page.locator("#benchmarkComparisonShell").inner_text().lower()
            rows_text = page.locator("#benchmarkComparisonShell").inner_text()
            assert "Random Survival Forest" in rows_text
            assert "DeepHit" in rows_text

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_predictive_compare_all_runs_ml_and_dl(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="predictive"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")
            assert page.locator("#panel-benchmark .benchmark-action-card").is_hidden()
            assert page.locator("#benchmarkWorkbench").is_hidden()
            assert page.locator("text=Open model controls").count() == 0
            assert "Open model controls" not in page.locator("#benchmarkSummaryGrid").inner_text()
            assert "Open model controls" not in page.locator("#benchmarkComparisonShell").inner_text()

            page.locator('[data-guided-action="run-predictive-compare-all"]').click()
            page.wait_for_function("document.getElementById('runPredictiveCompareAllButton').disabled === true")
            page.wait_for_function("document.body.dataset.guidedStep === '5'")
            page.wait_for_function(
                "document.getElementById('benchmarkSummaryGrid').textContent.includes('ML rows ready:')"
            )
            page.wait_for_function(
                "() => document.getElementById('benchmarkSummaryGrid').textContent.includes('ML rows ready: 4') || document.getElementById('benchmarkSummaryGrid').textContent.includes('ML rows ready: 1')"
            )
            page.wait_for_function(
                "() => { const text = document.getElementById('benchmarkSummaryGrid').textContent; const ml = /ML rows ready:\\s*(\\d+)/.exec(text); const dl = /DL rows ready:\\s*(\\d+)/.exec(text); return ml && dl && Number(ml[1]) > 0 && Number(dl[1]) >= 0; }"
            )
            assert "Selected model:" not in page.locator("#benchmarkSummaryGrid").inner_text()
            assert "Show selected controls" not in page.locator("#benchmarkSummaryGrid").inner_text()
            assert "Compare all models" in page.locator("#guidedRailActions").inner_text()
            assert "Review shared features" in page.locator("#guidedRailActions").inner_text()
            assert "Back" in page.locator("#guidedRailActions").inner_text()

            benchmark_text = page.locator("#benchmarkComparisonShell").inner_text()
            assert "Random Survival Forest" in benchmark_text
            assert "DeepHit" in benchmark_text
            assert "current screening rows from the latest ML and DL comparison outputs" in page.locator("#benchmarkTableNote").inner_text()

            page.locator('#guidedRailActions [data-guided-action="review-shared-features"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")
            page.wait_for_function("document.getElementById('benchmarkWorkbench').classList.contains('hidden') === false")
            page.wait_for_function("document.getElementById('runPredictiveWorkbenchButton').classList.contains('hidden')")
            page.wait_for_function("document.querySelectorAll('#guidedPanel [data-guided-action=\"run-predictive-selected\"]').length === 0")
            page.locator('#guidedPanel [data-guided-action="close-predictive-workbench"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '5'")
            page.wait_for_function("document.getElementById('benchmarkWorkbench').classList.contains('hidden')")

            page.wait_for_function("document.querySelectorAll('#benchmarkComparisonShell [data-benchmark-model]').length > 0")
            page.locator("#benchmarkComparisonShell [data-benchmark-model]").first.click()
            page.wait_for_function("document.getElementById('benchmarkWorkbench').classList.contains('hidden') === false")
            page.wait_for_function("document.body.dataset.guidedStep === '4'")
            page.wait_for_function("document.getElementById('runPredictiveWorkbenchButton') && !document.getElementById('runPredictiveWorkbenchButton').classList.contains('hidden')")
            page.wait_for_function("document.querySelectorAll('#guidedPanel [data-guided-action=\"run-predictive-selected\"]').length === 0")
            guided_panel_text = page.locator("#guidedPanel").inner_text()
            assert "Compare all" not in guided_panel_text
            assert "Back to leaderboard" in guided_panel_text
            assert "\nBack\n" not in f"\n{guided_panel_text}\n"
            if page.locator("#benchmarkMlMount").is_visible():
                assert page.locator("#benchmarkMlMount .model-choice-field").is_hidden()
                assert page.locator("#benchmarkMlMount #runMlButton").is_hidden()
                assert page.locator("#benchmarkMlMount #runCompareButton").is_hidden()
                assert page.locator("#benchmarkMlMount #runCompareInlineButton").is_hidden()
            else:
                assert page.locator("#benchmarkDlMount .model-choice-field").is_hidden()
                assert page.locator("#benchmarkDlMount #runDlButton").is_hidden()
                assert page.locator("#benchmarkDlMount #runDlCompareButton").is_hidden()
                assert page.locator("#benchmarkDlMount #runDlCompareInlineButton").is_hidden()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_predictive_single_model_tuning_returns_to_stale_leaderboard(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _mock_ml_payload(body: dict) -> dict:
        return {
            "request_config": body,
            "analysis": {
                "comparison_table": [
                    {"model": "Random Survival Forest", "c_index": 0.712, "evaluation_mode": "holdout", "n_features": len(body.get("features", [])), "training_time_ms": 120.0, "rank": 1},
                    {"model": "Gradient Boosted Survival", "c_index": 0.701, "evaluation_mode": "holdout", "n_features": len(body.get("features", [])), "training_time_ms": 150.0, "rank": 2},
                ],
                "evaluation_mode": "holdout",
                "scientific_summary": {
                    "status": "review",
                    "headline": "ML comparison complete.",
                    "strengths": [],
                    "cautions": [],
                    "next_steps": [],
                },
            },
        }

    def _mock_dl_compare_payload(body: dict) -> dict:
        return {
            "request_config": body,
            "analysis": {
                "comparison_table": [
                    {"model": "DeepSurv", "c_index": 0.676, "evaluation_mode": "holdout", "epochs_trained": 48, "n_features": len(body.get("features", [])), "training_time_ms": 420.0, "rank": 1},
                    {"model": "DeepHit", "c_index": 0.651, "evaluation_mode": "holdout", "epochs_trained": 44, "n_features": len(body.get("features", [])), "training_time_ms": 510.0, "rank": 2},
                ],
                "evaluation_mode": "holdout",
                "scientific_summary": {
                    "status": "review",
                    "headline": "DL comparison complete.",
                    "strengths": [],
                    "cautions": [],
                    "next_steps": [],
                },
            },
        }

    def _mock_dl_single_payload(body: dict) -> dict:
        return {
            "request_config": body,
            "analysis": {
                "c_index": 0.633,
                "evaluation_mode": "holdout",
                "epochs_trained": 36,
                "n_features": len(body.get("features", [])),
                "training_seed": 42,
                "scientific_summary": {
                    "status": "review",
                    "headline": "DeepSurv single fit complete.",
                    "strengths": [],
                    "cautions": [],
                    "next_steps": [],
                },
            },
            "figures": {},
        }

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            def route_ml_model(route) -> None:
                body = json.loads(route.request.post_data or "{}")
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(_mock_ml_payload(body)),
                )

            def route_dl_model(route) -> None:
                body = json.loads(route.request.post_data or "{}")
                payload = _mock_dl_compare_payload(body) if body.get("model_type") == "compare" else _mock_dl_single_payload(body)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(payload),
                )

            page.route("**/api/ml-model", route_ml_model)
            page.route("**/api/deep-model", route_dl_model)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="predictive"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.locator('[data-guided-action="run-predictive-compare-all"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '5'")
            page.wait_for_function("document.getElementById('benchmarkComparisonShell').textContent.includes('DeepSurv')")

            page.locator('#benchmarkComparisonShell [data-benchmark-model="deepsurv"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")
            page.wait_for_function("document.getElementById('benchmarkWorkbench').classList.contains('hidden') === false")
            page.locator("#dlHiddenLayers").fill("64")
            page.wait_for_function("document.getElementById('dlHiddenLayers').value === '64'")
            page.locator("#runPredictiveWorkbenchButton").click()
            page.wait_for_function("document.body.dataset.guidedStep === '5'")
            page.wait_for_function("document.getElementById('dlMetaBanner').textContent.includes('DEEPSURV')")
            page.wait_for_function("document.getElementById('closePredictiveWorkbenchButton') && !document.getElementById('closePredictiveWorkbenchButton').classList.contains('hidden')")
            page.locator("#closePredictiveWorkbenchButton").click()
            page.wait_for_function("document.body.dataset.guidedStep === '5'")
            page.wait_for_function("document.querySelector('[data-tab=\"benchmark\"]').getAttribute('aria-selected') === 'true'")
            page.wait_for_function("document.getElementById('benchmarkWorkbench').classList.contains('hidden')")
            page.wait_for_function("document.getElementById('benchmarkComparisonShell').textContent.includes('DeepSurv')")

            summary_text = page.locator("#benchmarkSummaryGrid").inner_text().lower()
            table_text = page.locator("#benchmarkComparisonShell").inner_text()
            assert "stale reference" in summary_text or "current settings no longer match" in summary_text
            assert "DeepSurv" in table_text
            assert "DeepHit" in table_text

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_predictive_failed_single_model_rerun_stays_on_step4(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _mock_ml_payload(body: dict) -> dict:
        return {
            "request_config": body,
            "analysis": {
                "comparison_table": [
                    {"model": "Random Survival Forest", "c_index": 0.712, "evaluation_mode": "holdout", "n_features": len(body.get("features", [])), "training_time_ms": 120.0, "rank": 1},
                    {"model": "Gradient Boosted Survival", "c_index": 0.701, "evaluation_mode": "holdout", "n_features": len(body.get("features", [])), "training_time_ms": 150.0, "rank": 2},
                ],
                "evaluation_mode": "holdout",
                "scientific_summary": {
                    "status": "review",
                    "headline": "ML comparison complete.",
                    "strengths": [],
                    "cautions": [],
                    "next_steps": [],
                },
            },
        }

    def _mock_dl_compare_payload(body: dict) -> dict:
        return {
            "request_config": body,
            "analysis": {
                "comparison_table": [
                    {"model": "DeepSurv", "c_index": 0.676, "evaluation_mode": "holdout", "epochs_trained": 48, "n_features": len(body.get("features", [])), "training_time_ms": 420.0, "rank": 1},
                    {"model": "DeepHit", "c_index": 0.651, "evaluation_mode": "holdout", "epochs_trained": 44, "n_features": len(body.get("features", [])), "training_time_ms": 510.0, "rank": 2},
                ],
                "evaluation_mode": "holdout",
                "scientific_summary": {
                    "status": "review",
                    "headline": "DL comparison complete.",
                    "strengths": [],
                    "cautions": [],
                    "next_steps": [],
                },
            },
        }

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            def route_ml_model(route) -> None:
                body = json.loads(route.request.post_data or "{}")
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(_mock_ml_payload(body)),
                )

            def route_dl_model(route) -> None:
                body = json.loads(route.request.post_data or "{}")
                if body.get("model_type") == "compare":
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(_mock_dl_compare_payload(body)),
                    )
                    return
                route.fulfill(
                    status=500,
                    content_type="application/json",
                    body=json.dumps({"detail": "forced single-model failure"}),
                )

            page.route("**/api/ml-model", route_ml_model)
            page.route("**/api/deep-model", route_dl_model)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="predictive"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.locator('[data-guided-action="run-predictive-compare-all"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '5'")
            page.wait_for_function("document.getElementById('benchmarkComparisonShell').textContent.includes('DeepSurv')")

            page.locator('#benchmarkComparisonShell [data-benchmark-model="deepsurv"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")
            page.wait_for_function("document.getElementById('benchmarkWorkbench').classList.contains('hidden') === false")
            page.locator("#dlHiddenLayers").fill("64")
            page.locator("#runPredictiveWorkbenchButton").click()
            page.wait_for_function("document.querySelector('#toastContainer .toast-error') !== null")

            assert page.locator("body").get_attribute("data-guided-step") == "4"
            assert page.locator("#benchmarkWorkbench").is_visible()
            assert page.locator("#runPredictiveWorkbenchButton").is_visible()
            assert "DeepSurv" in page.locator("#benchmarkWorkbench").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_change_analysis_returns_to_step3_with_valid_tab(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="predictive"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.locator('[data-guided-action="previous-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            active_tab = page.locator('[data-tab][aria-selected="true"]')
            active_tab_name = active_tab.get_attribute("data-tab")
            assert active_tab_name in {"benchmark", "km", "cox", "ml", "dl", "tables"}
            assert page.locator(f"#panel-{active_tab_name}").evaluate("(el) => el.classList.contains('active')") is True

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_predictive_compare_partial_failure_stays_on_step4(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            def fail_deep_compare(route) -> None:
                post_data = route.request.post_data or ""
                if route.request.method == "POST" and '"model_type":"compare"' in post_data:
                    time.sleep(0.6)
                    route.fulfill(
                        status=500,
                        content_type="application/json",
                        body=json.dumps({"detail": "forced deep compare failure"}),
                    )
                    return
                route.continue_()

            page.route("**/api/deep-model", fail_deep_compare)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="predictive"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")
            page.locator('[data-guided-action="run-predictive-compare-all"]').click()
            page.wait_for_function(
                "() => { const text = document.querySelector('#benchmarkSummaryGrid')?.innerText ?? ''; return text.includes('Unified predictive board is incomplete') || text.includes('Incomplete compare'); }",
                timeout=60000,
            )

            assert page.locator("body").get_attribute("data-guided-step") == "4"
            summary_text = page.locator("#benchmarkSummaryGrid").inner_text()
            assert "Unified predictive board is incomplete" in summary_text
            assert "both ml and dl comparison rows are current" in page.locator("#benchmarkPlotNote").inner_text().lower()
            assert "both ml and dl comparison rows are current" in page.locator("#benchmarkTableNote").inner_text().lower()
            assert "must finish with both model families" in page.locator("#benchmarkComparisonShell").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_cox_results_table_stays_within_card(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            _switch_to_expert(page)
            page.locator('[data-tab="cox"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"cox\"]').getAttribute('aria-selected') === 'true'"
            )
            page.locator("#runCoxButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadCoxResultsButton').disabled"
            )

            shell_box = page.locator("#coxResultsShell").bounding_box()
            card_box = page.locator("#coxResultsShell").locator("xpath=ancestor::div[contains(@class,'table-card')]").bounding_box()
            assert shell_box is not None
            assert card_box is not None
            shell_right = shell_box["x"] + shell_box["width"]
            card_right = card_box["x"] + card_box["width"]
            assert shell_right <= card_right + 1.0

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_back_button_returns_to_home_not_blank(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            page.locator("#timeUnitLabel").fill("Days")
            page.locator("#maxTime").fill("24")
            page.locator("#groupColumn").select_option("stage")
            page.locator("#tab-cox").click()
            page.locator("#covariateChecklist").wait_for(state="visible")
            page.locator("#covariateChecklist input[value='immune_index']").check(force=True)
            page.locator("#covariateChecklist input[value='age']").uncheck()
            page.locator("#tab-km").click()
            page.locator("#runKmButton").wait_for(state="visible")
            page.locator("#runKmButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadKmSummaryButton').disabled"
            )
            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")

            page.go_back(wait_until="networkidle")
            page.locator("#workspace").wait_for(state="visible")
            page.wait_for_function("document.body.dataset.uiMode === 'guided'")

            page.go_back(wait_until="networkidle")
            page.locator("#landing").wait_for(state="visible")
            assert page.locator("#workspace").is_hidden()
            assert "Drop a file here or click to browse" in page.locator("#landing").inner_text()

            page.go_forward(wait_until="networkidle")
            page.locator("#workspace").wait_for(state="visible")
            page.wait_for_function("document.body.dataset.uiMode === 'guided'")

            page.go_forward(wait_until="networkidle")
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#configStrip").wait_for(state="visible")
            page.wait_for_function("document.body.dataset.uiMode === 'expert'")
            assert page.locator('[data-tab="benchmark"]').get_attribute("aria-selected") == "true"
            assert page.locator("#timeUnitLabel").input_value() == "Days"
            assert page.locator("#maxTime").input_value() == "24"
            assert page.locator("#groupColumn").input_value() == "stage"
            assert page.locator("#covariateChecklist input[value='immune_index']").is_checked()
            assert not page.locator("#covariateChecklist input[value='age']").is_checked()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_study_design_collapses_grouping_controls_outside_km(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            page.locator('[data-tab="km"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"km\"]').getAttribute('aria-selected') === 'true'"
            )

            assert page.locator("#groupingDetails").evaluate("(el) => el.open") is True
            assert "current group by: overall only" in page.locator("#groupingSummaryText").inner_text().lower()

            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")
            assert page.locator("#groupingDetails").evaluate("(el) => el.open") is False
            assert "Grouping only:" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Training inputs come only from the shared ML/DL model feature selections" in page.locator("#dlFeatureSummaryText").inner_text()

            page.locator('[data-tab="tables"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"tables\"]').getAttribute('aria-selected') === 'true'"
            )
            assert page.locator("#groupingDetails").evaluate("(el) => el.open") is True

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_mode_goal_cards_and_mode_toggle(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")

            assert "Confirm outcome" in page.locator("#guidedSummaryTitle").inner_text()
            assert page.locator("#configStrip").is_visible()
            assert page.locator("#configStrip").evaluate("(el) => el.parentElement && el.parentElement.id") == "guidedRailPanelMount"
            assert page.locator("#tabStrip").is_hidden()
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function(
                "document.getElementById('guidedSummaryTitle').textContent.includes('Choose analysis')"
            )
            assert "Choose analysis" in page.locator("#guidedSummaryTitle").inner_text()
            assert "Outcome: os_months / os_event = 1" in page.locator("#guidedSummaryChips").inner_text()
            assert page.locator("#configStrip").is_hidden()
            assert page.locator("#configStrip").evaluate("(el) => el.parentElement && el.parentElement.id") == "guidedConfigMount"

            page.locator('[data-guided-action="choose-goal"][data-goal="cox"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"cox\"]').getAttribute('aria-selected') === 'true'"
            )
            assert "Run Cox PH" in page.locator("#guidedPanel").inner_text()
            assert "Analysis: Cox PH" in page.locator("#guidedSummaryChips").inner_text()
            assert page.locator("#tabStrip").is_hidden()
            assert page.locator("#configStrip").is_hidden()
            assert page.locator('#panel-cox').is_visible()
            assert page.locator('#panel-km').is_hidden()

            page.evaluate("() => document.getElementById('expertModeButton')?.click()")
            page.wait_for_function(
                "document.body.dataset.uiMode === 'expert' && document.getElementById('guidedShell').classList.contains('hidden')"
            )

            page.locator("#guidedModeButton").click()
            page.wait_for_function(
                "document.body.dataset.uiMode === 'guided' && !document.getElementById('guidedShell').classList.contains('hidden')"
            )
            assert "Run Cox PH" in page.locator("#guidedPanel").inner_text()
            assert page.locator("#tabStrip").is_hidden()
            assert page.locator('#panel-cox').is_visible()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_model_features_stay_separate_from_cox_and_keep_non_endpoint_inputs(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            page.locator("#groupColumn").select_option("sex")

            model_features = page.eval_on_selector_all(
                "#modelFeatureChecklist input",
                "els => els.filter(e => e.checked).map(e => e.value)",
            )
            assert "pfs_months" not in model_features
            assert "pfs_event" not in model_features
            assert "biomarker_score" in model_features
            assert "immune_index" in model_features

            page.locator('[data-tab="cox"]').click()
            page.locator("#covariateChecklist input[value='age']").uncheck()
            cox_covariates = page.eval_on_selector_all(
                "#covariateChecklist input",
                "els => els.filter(e => e.checked).map(e => e.value)",
            )
            assert "age" not in cox_covariates
            assert page.eval_on_selector_all(
                "#modelFeatureChecklist input",
                "els => els.filter(e => e.checked).map(e => e.value)",
            ) == model_features

            page.locator('[data-tab="km"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"km\"]').getAttribute('aria-selected') === 'true'"
            )
            page.locator("#deriveToggle").click()
            assert page.locator("#deriveSource").is_disabled()
            assert page.locator("#deriveButton").is_disabled()
            assert "locked while Group by uses sex" in page.locator("#deriveStatus").inner_text()
            assert page.locator("#groupColumn").input_value() == "sex"

            updated_model_features = page.eval_on_selector_all(
                "#modelFeatureChecklist input",
                "els => els.filter(e => e.checked).map(e => e.value)",
            )
            assert updated_model_features == model_features

            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")
            assert "Model features: 7" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Grouping only: sex" in page.locator("#dlFeatureSummaryChips").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_event_column_defaults_to_event_like_fields_and_advanced_toggle_reveals_all(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            _switch_to_expert(page)

            default_options = page.locator("#eventColumn option").evaluate_all(
                "(options) => options.map((option) => option.value)"
            )
            assert "os_event" in default_options
            assert "egfr_status" not in default_options
            assert "kras_status" not in default_options
            assert "Showing likely event columns only" in page.locator("#eventColumnHelp").inner_text()

            page.locator("#showAllEventColumns").check()
            page.wait_for_function(
                "Array.from(document.querySelectorAll('#eventColumn option')).some((option) => option.value === 'egfr_status')"
            )
            advanced_options = page.locator("#eventColumn option").evaluate_all(
                "(options) => options.map((option) => option.value)"
            )
            assert "egfr_status" in advanced_options
            assert "Showing all columns." in page.locator("#eventColumnHelp").inner_text()

            page.locator("#eventColumn").select_option("egfr_status")
            page.wait_for_function(
                "document.getElementById('eventColumnWarning').textContent.includes('baseline characteristic')"
            )
            assert "Use it as Group by or as a model feature instead." in page.locator("#eventColumnWarning").inner_text()
            assert page.locator("#eventPositiveValue").input_value() == ""
            assert page.locator("#eventValueWarning").is_hidden()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_event_value_requires_explicit_choice_for_ambiguous_binary_codes(browser_server: str, tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            upload_path = tmp_path / "ambiguous_event.csv"
            upload_path.write_text(
                "\n".join(
                    [
                        "os_months,os_status,age",
                        "10,1,61",
                        "12,2,63",
                        "18,1,67",
                        "20,2,70",
                    ]
                ),
                encoding="utf-8",
            )

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#datasetFile").set_input_files(str(upload_path))
            page.locator("#workspace").wait_for(state="visible")
            page.wait_for_function(
                "document.getElementById('eventColumn').value === 'os_status'"
            )
            page.wait_for_function(
                "document.getElementById('eventValueWarning').textContent.includes('TCGA-style 1/2 coding')"
            )
            assert page.locator("#eventColumn").input_value() == "os_status"
            assert page.locator("#eventColumnWarning").is_hidden()
            assert page.locator("#eventValueWarning").is_hidden()
            assert page.locator("#eventPositiveValue").input_value() == ""
            assert "Choose which value means event" in page.locator("#guidedPanel").inner_text()
            assert page.locator('[data-guided-action="next-step"]').is_disabled()

            page.locator("#eventPositiveValue").select_option("1")
            page.wait_for_function(
                "() => !document.querySelector('[data-guided-action=\"next-step\"]').disabled"
            )
            assert page.locator('[data-guided-action="next-step"]').is_enabled()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_event_column_blocks_binary_baseline_covariates_in_guided_step_two(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#showAllEventColumns").check()
            page.locator("#eventColumn").select_option("menostat")
            page.wait_for_function(
                "document.getElementById('guidedPanel').textContent.includes('does not look like a survival event column')"
            )

            assert page.locator("#eventColumnWarning").is_hidden()
            assert "does not look like a survival event column" in page.locator("#guidedPanel").inner_text()
            assert page.locator('[data-guided-action="next-step"]').is_disabled()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_km_derive_defaults_to_group_when_current_group_is_overall_only(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            page.locator("#runKmButton").click()
            page.wait_for_function(
                "document.getElementById('kmMetaBanner').textContent.includes('N=')"
            )
            initial_banner = page.locator("#kmMetaBanner").inner_text()

            assert page.locator("#groupColumn").input_value() == ""
            page.locator("#deriveToggle").click()

            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveMethod").select_option("median_split")
            page.locator("#deriveColumnName").fill("age_guided_split")
            page.locator("#deriveButton").click(force=True)
            page.wait_for_function(
                "document.getElementById('groupColumn').value === 'age_guided_split'"
            )
            page.wait_for_function(
                "(previousText) => document.getElementById('kmMetaBanner').textContent !== previousText",
                arg=initial_banner,
            )

            assert page.locator("#groupColumn").input_value() == "age_guided_split"
            assert "Current grouping now uses age_guided_split" in page.locator("#deriveSummary").inner_text()
            assert page.locator("#kmMetaBanner").inner_text() != initial_banner

            page.locator("#groupColumn").select_option("sex")
            page.wait_for_function(
                "document.getElementById('deriveSummary').textContent.includes('Current grouping remains sex')"
            )
            assert "Derived column age_guided_split is available." in page.locator("#deriveSummary").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_km_derive_preserves_existing_group_until_user_reruns(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            _switch_to_expert(page)

            page.locator("#groupColumn").select_option("stage_group")
            page.locator("#runKmButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadKmSummaryButton').disabled"
            )
            initial_banner = page.locator("#kmMetaBanner").inner_text()

            page.locator("#deriveToggle").click()
            assert page.locator("#deriveSource").is_disabled()
            assert page.locator("#deriveButton").is_disabled()
            assert "locked while Group by uses stage_group" in page.locator("#deriveStatus").inner_text()

            assert page.locator("#groupColumn").input_value() == "stage_group"
            assert page.locator("#kmMetaBanner").inner_text() == initial_banner

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_km_run_creates_pending_derived_group_without_separate_create_button(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            page.locator('[data-guided-action="next-step"]').click()
            page.locator('[data-guided-action="choose-goal"][data-goal="km"]').click()
            page.wait_for_function("document.body.dataset.guidedGoal === 'km'")

            assert page.locator("#deriveToggle").is_hidden()
            assert page.locator("#deriveButton").is_hidden()

            page.locator("#deriveColumnName").fill("age_guided_run")
            page.locator('#guidedPanel [data-guided-action="run-km"]').click()
            page.wait_for_function(
                "document.body.dataset.guidedStep === '5' && document.getElementById('groupColumn').value === 'age_guided_run'"
            )

            assert page.locator("#groupColumn").input_value() == "age_guided_run"
            assert "Current grouping now uses age_guided_run" in page.locator("#deriveSummary").inner_text()
            assert "N=" in page.locator("#kmMetaBanner").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_km_run_keeps_existing_group_when_derive_draft_is_pending(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            page.locator('[data-guided-action="next-step"]').click()
            page.locator('[data-guided-action="choose-goal"][data-goal="km"]').click()
            page.wait_for_function("document.body.dataset.guidedGoal === 'km'")

            page.locator("#groupColumn").select_option("stage_group")
            assert page.locator("#deriveColumnName").is_disabled()
            assert page.locator("#deriveButton").is_disabled()
            assert "locked while Group by uses stage_group" in page.locator("#deriveStatus").inner_text()
            page.locator('#guidedPanel [data-guided-action="run-km"]').click()
            page.wait_for_function(
                "document.body.dataset.guidedStep === '5' && document.getElementById('groupColumn').value === 'stage_group'"
            )

            assert page.locator("#groupColumn").input_value() == "stage_group"
            assert "Derived column" not in page.locator("#deriveSummary").inner_text()
            assert "N=" in page.locator("#kmMetaBanner").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_dataset_entry_resets_scroll_to_top(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 720})

            page.goto(browser_server, wait_until="networkidle")
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")

            page.evaluate("document.getElementById('loadExampleButton').click()")
            page.locator("#workspace").wait_for(state="visible")
            page.wait_for_timeout(450)

            config_box = page.locator("#configStrip").bounding_box()
            assert config_box is not None
            assert config_box["y"] < 340

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_mode_back_button_walks_previous_steps(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")

            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="km"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.go_back(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            assert "Choose what you want to do next" in page.locator("#guidedPanel").inner_text()
            assert "ML/DL Models" in page.locator("#guidedPanel").inner_text()
            assert "Deep Learning" not in page.locator("#guidedPanel").inner_text()

            page.go_back(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            assert "Tell SurvStudio what counts as survival time and event" in page.locator("#guidedPanel").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_step_rail_allows_navigation_to_reached_steps(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")

            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="km"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.locator('#stepIndicator .step[data-step="3"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            assert "Choose what you want to do next" in page.locator("#guidedPanel").inner_text()

            page.locator('#stepIndicator .step[data-step="2"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            assert "Tell SurvStudio what counts as survival time and event" in page.locator("#guidedPanel").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_guided_and_expert_mode_back_forward_restores_mode_and_step(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")

            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="predictive"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.evaluate("() => document.getElementById('expertModeButton')?.click()")
            page.wait_for_function("document.body.dataset.uiMode === 'expert'")
            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")

            page.go_back(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.uiMode === 'guided' && document.body.dataset.guidedStep === '4'")
            assert "Run ML/DL Models" in page.locator("#guidedPanel").inner_text()
            assert "Compare all models" in page.locator("#guidedPanel").inner_text()

            page.go_forward(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.uiMode === 'expert'")
            assert page.locator('[data-tab="benchmark"]').get_attribute("aria-selected") == "true"

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_optimal_cutpoint_summary_explains_risk_labels(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _switch_to_expert(page)
            page.locator('[data-tab="km"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"km\"]').getAttribute('aria-selected') === 'true'"
            )

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveMethod").select_option("optimal_cutpoint")
            page.locator("#deriveButton").click(force=True)
            page.wait_for_function(
                "document.getElementById('deriveSummary').textContent.includes('Assignment rule')"
            )

            derive_text = page.locator("#deriveSummary").inner_text()
            assert "High/Low indicate risk direction" in derive_text
            assert "Assignment rule" in derive_text
            assert ("Selection-adjusted p-value" in derive_text) or ("Raw p-value" in derive_text)
            assert "Current grouping now uses age__optimal_cutpoint" in derive_text

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_optimal_cutpoint_summary_wraps_long_derived_column(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            _switch_to_expert(page)
            page.locator('[data-tab="km"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"km\"]').getAttribute('aria-selected') === 'true'"
            )

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("pack_years_smoked")
            page.locator("#deriveMethod").select_option("optimal_cutpoint")
            page.locator("#deriveButton").click(force=True)
            page.wait_for_function(
                "document.getElementById('deriveSummary').textContent.includes('Derived column')"
            )

            grid_box = page.locator("#deriveSummary .signature-summary-grid").bounding_box()
            derived_box = page.locator("#deriveSummary .signature-summary-grid > div").nth(0).bounding_box()
            assert grid_box is not None
            assert derived_box is not None
            assert derived_box["x"] + derived_box["width"] <= grid_box["x"] + grid_box["width"] + 1.0

            overflow = page.locator("#deriveSummary .signature-summary-grid").evaluate(
                "(el) => ({ scrollWidth: el.scrollWidth, clientWidth: el.clientWidth })"
            )
            assert overflow["scrollWidth"] <= overflow["clientWidth"] + 1

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_ml_importance_plot_stays_inside_its_section(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            _open_predictive_workbench(page)
            page.locator("#mlNEstimators").fill("10")
            page.locator("#runMlButton").click()
            page.wait_for_function(
                "document.getElementById('mlMetaBanner').textContent.includes('eval=')"
            )

            importance_box = page.locator("#mlImportancePlot").bounding_box()
            shap_box = page.locator("#mlShapPlot").bounding_box()
            banner_box = page.locator("#mlMetaBanner").bounding_box()
            assert importance_box is not None
            assert shap_box is not None
            assert banner_box is not None
            assert importance_box["y"] + importance_box["height"] <= banner_box["y"] + 1.0
            assert shap_box["y"] + shap_box["height"] <= banner_box["y"] + 1.0

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_risk_table_ticks_change_columns_and_flash_table(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            page.locator("#runKmButton").click()
            page.wait_for_function("!document.getElementById('downloadKmSummaryButton').disabled")
            default_columns = page.locator("#kmRiskShell thead th").count()
            assert default_columns == 7

            page.locator("#riskTablePoints").fill("10")
            page.locator("#runKmButton").click()
            page.wait_for_function(
                "document.querySelectorAll('#kmRiskShell thead th').length === 11"
            )
            page.wait_for_function(
                "document.getElementById('kmRiskShell').classList.contains('preset-applied-flash')"
            )
            updated_columns = page.locator("#kmRiskShell thead th").count()
            assert updated_columns == 11

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_dl_epoch_validation_message_is_human_readable(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            _open_predictive_workbench(page, "deepsurv")
            page.locator("#dlEpochs").fill("1001")
            page.locator("#runDlButton").click()
            page.wait_for_function(
                "document.getElementById('toastContainer').textContent.includes('Epochs must be between 10 and 1000')"
            )
            assert "Epochs must be between 10 and 1000" in page.locator("#toastContainer").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise

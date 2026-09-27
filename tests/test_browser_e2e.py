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


def _wait_for_workspace(page) -> None:
    page.locator("#workspace").wait_for(state="visible")


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
        page.locator("#openPredictiveWorkbenchButton").click()
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
            _wait_for_workspace(page)

            page.locator("#runKmButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadKmSummaryButton').disabled && !document.getElementById('downloadKmPngButton').disabled"
            )

            page.locator("#panel-km .export-menu > summary").click()
            with page.expect_download() as summary_info:
                page.locator("#downloadKmSummaryButton").click()
            page.wait_for_function("!document.querySelector('#panel-km .export-menu').open")
            summary_download = summary_info.value
            summary_path = tmp_path / (summary_download.suggested_filename or "km_summary.csv")
            summary_download.save_as(summary_path)
            assert summary_path.exists()
            assert summary_path.stat().st_size > 0

            page.locator("#panel-km .export-menu > summary").click()
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


def test_browser_benchmark_tab_combines_latest_ml_and_dl_compare_outputs(browser_server: str, tmp_path: Path) -> None:
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
                    "evaluation_split_fingerprint": "holdout-seed42-shared",
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
                    "evaluation_split_fingerprint": "holdout-seed42-shared",
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
            _wait_for_workspace(page)

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

            leaderboard_card = page.locator("#benchmarkComparisonShell").locator("xpath=ancestor::div[contains(@class,'table-card')]")
            leaderboard_card.locator(".export-menu > summary").click()
            with page.expect_download() as tripod_info:
                page.locator("#downloadTripodMarkdownButton").click()
            tripod_path = tmp_path / (tripod_info.value.suggested_filename or "tripod_ai_checklist.md")
            tripod_info.value.save_as(tripod_path)
            tripod_text = tripod_path.read_text(encoding="utf-8")
            assert tripod_text.startswith("# TRIPOD+AI checklist")
            assert "Random Survival Forest" in tripod_text and "DeepHit" in tripod_text
            assert "split fingerprint holdout-seed42-shared" in tripod_text

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
                    "evaluation_split_fingerprint": "holdout-seed42-shared",
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
                    "evaluation_split_fingerprint": "holdout-seed42-shared",
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
            _wait_for_workspace(page)
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
                    "evaluation_split_fingerprint": "holdout-seed42-shared",
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
            _wait_for_workspace(page)

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


def test_browser_cox_results_table_stays_within_card(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaUploadReadyButton").click()
            _wait_for_workspace(page)
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
            _wait_for_workspace(page)
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
            page.locator("#landing").wait_for(state="visible")
            assert page.locator("#workspace").is_hidden()
            assert "Drop a file here or click to browse" in page.locator("#landing").inner_text()

            page.go_forward(wait_until="networkidle")
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#configStrip").wait_for(state="visible")
            page.wait_for_function("document.querySelector('[data-tab=\"benchmark\"]').getAttribute('aria-selected') === 'true'")
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


def test_browser_grouping_controls_show_only_on_curves_and_table_1(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _wait_for_workspace(page)
            page.locator('[data-tab="km"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"km\"]').getAttribute('aria-selected') === 'true'"
            )

            assert page.locator("#groupingDetails").evaluate("(el) => el.open") is True
            assert "choose a column to compare groups" in page.locator("#groupingSummaryText").inner_text().lower()

            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")
            assert page.locator("#groupingConfigBlock").is_hidden()
            assert "Grouping only:" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Training inputs come only from the shared ML/DL model feature selections" in page.locator("#dlFeatureSummaryText").inner_text()

            page.locator('[data-tab="tables"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"tables\"]').getAttribute('aria-selected') === 'true'"
            )
            assert page.locator("#groupingConfigBlock").is_visible()
            assert page.locator("#groupingDetails").evaluate("(el) => el.open") is True

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
            _wait_for_workspace(page)
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
            page.locator("#loadTcgaUploadReadyButton").click()
            _wait_for_workspace(page)

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
            assert page.locator("#eventValueWarning").is_visible()
            assert page.locator("#eventPositiveValue").input_value() == ""
            assert page.locator("#runKmButton").is_disabled()

            page.locator("#eventPositiveValue").select_option("1")
            page.wait_for_function("() => !document.getElementById('runKmButton').disabled")
            assert page.locator("#runKmButton").is_enabled()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_event_column_blocks_binary_baseline_covariates(browser_server: str) -> None:
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
                "document.getElementById('eventColumnWarning').textContent.includes('does not look like a survival event column')"
            )

            assert page.locator("#eventColumnWarning").is_visible()
            assert page.locator("#runKmButton").is_disabled()

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
            _wait_for_workspace(page)

            page.locator("#runKmButton").click()
            page.wait_for_function(
                "document.getElementById('kmMetaBanner').textContent.includes('N=')"
            )
            initial_banner = page.locator("#kmMetaBanner").inner_text()

            assert page.locator("#groupColumn").input_value() == ""
            page.locator("#deriveToggle").click()

            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveMethod").select_option("median_split")
            page.locator("#deriveColumnName").fill("age_median_group")
            page.locator("#deriveButton").click(force=True)
            page.wait_for_function(
                "document.getElementById('groupColumn').value === 'age_median_group'"
            )
            page.wait_for_function(
                "(previousText) => document.getElementById('kmMetaBanner').textContent !== previousText",
                arg=initial_banner,
            )

            assert page.locator("#groupColumn").input_value() == "age_median_group"
            assert "Current grouping now uses age_median_group" in page.locator("#deriveSummary").inner_text()
            assert page.locator("#kmMetaBanner").inner_text() != initial_banner

            page.locator("#groupColumn").select_option("sex")
            page.wait_for_function(
                "document.getElementById('deriveSummary').textContent.includes('Current grouping remains sex')"
            )
            derive_text = page.locator("#deriveSummary").inner_text()
            assert "Derived column age_median_group is available." in derive_text
            assert "The counts and method details below describe age_median_group, not sex." in derive_text
            assert "STORED DERIVED GROUPING" in derive_text

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
            page.locator("#loadTcgaUploadReadyButton").click()
            _wait_for_workspace(page)

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


def test_browser_km_derive_summary_marks_stored_result_vs_locked_draft(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaUploadReadyButton").click()
            _wait_for_workspace(page)
            # The sample opens grouped by stage; derived groups start from Overall only.
            page.wait_for_function("document.getElementById('groupColumn').value === 'stage_group'")
            page.locator("#groupColumn").select_option("")

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("pack_years_smoked")
            page.locator("#deriveMethod").select_option("median_split")
            page.locator("#deriveButton").click(force=True)
            page.wait_for_function(
                "document.getElementById('groupColumn').value === 'pack_years_smoked__median_split'"
            )

            page.locator("#groupColumn").select_option("")
            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveColumnName").fill("biomarker_group")
            page.locator("#groupColumn").select_option("egfr_status")
            page.wait_for_function(
                "document.getElementById('deriveSummary').textContent.includes('Current grouping remains egfr_status')"
            )

            derive_text = page.locator("#deriveSummary").inner_text()
            status_text = page.locator("#deriveStatus").inner_text()
            assert "STORED DERIVED GROUPING" in derive_text
            assert "Derived column pack_years_smoked__median_split is available." in derive_text
            assert "The counts and method details below describe pack_years_smoked__median_split, not egfr_status." in derive_text
            assert "The disabled Source variable / Method / Column name controls above are draft settings" in derive_text
            assert "They do not describe pack_years_smoked__median_split." in derive_text
            assert "The card below describes pack_years_smoked__median_split, not the current Group by egfr_status." in status_text

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


def test_browser_optimal_cutpoint_summary_explains_risk_labels(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _wait_for_workspace(page)
            # The sample opens grouped by hormone therapy; derived groups start from Overall only.
            page.wait_for_function("document.getElementById('groupColumn').value === 'horTh'")
            page.locator("#groupColumn").select_option("")
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
            page.locator("#loadTcgaUploadReadyButton").click()
            _wait_for_workspace(page)
            # The sample opens grouped by stage; derived groups start from Overall only.
            page.wait_for_function("document.getElementById('groupColumn').value === 'stage_group'")
            page.locator("#groupColumn").select_option("")
            page.locator('[data-tab="km"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"km\"]').getAttribute('aria-selected') === 'true'"
            )

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("pack_years_smoked")
            page.locator("#deriveMethod").select_option("optimal_cutpoint")
            # The workspace scrolls smoothly after loading, so click the element itself rather than a screen position.
            page.evaluate("() => document.getElementById('deriveButton').click()")
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
            _wait_for_workspace(page)
            _open_predictive_workbench(page)
            page.locator("#mlNEstimators").fill("10")
            page.locator("#runMlButton").click()
            page.wait_for_function(
                "document.getElementById('mlMetaBanner').textContent.includes('eval=')"
            )

            # The page scrolls smoothly to the new result, so separate bounding_box() calls can
            # catch the plots and the banner at different scroll positions; read all three
            # rectangles together and let the layout settle.
            page.wait_for_function(
                """() => {
                    const box = (id) => document.getElementById(id).getBoundingClientRect();
                    const banner = box("mlMetaBanner");
                    return box("mlImportancePlot").bottom <= banner.top + 1 && box("mlShapPlot").bottom <= banner.top + 1;
                }""",
                timeout=10000,
            )

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_predictive_workbench_plots_resize_with_viewport(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    long_labels = [
        "histology_Lung Micropapillary Adenocarcinoma",
        "smoking_status_Former smoker (duration unknown)",
        "histology_Mucinous (Colloid) carcinoma",
        "histology_Lung Solid Pattern Predominant Adenocarcinoma",
        "pathologic_stage_Stage IIIB",
        "stage_group_Stage IV",
        "expression_subtype_Squamoid",
        "pathologic_stage_Stage IA",
    ]

    def _mock_dl_single(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "request_config": body,
                "analysis": {
                    "c_index": 0.739,
                    "evaluation_mode": "holdout",
                    "epochs_trained": 100,
                    "best_monitor_epoch": 100,
                    "max_epochs_requested": 100,
                    "training_seed": 42,
                    "scientific_summary": {
                        "status": "review",
                        "headline": "MTLR run complete.",
                        "strengths": [],
                        "cautions": [],
                        "next_steps": [],
                    },
                },
                "figures": {
                    "importance": {
                        "data": [
                            {
                                "type": "bar",
                                "orientation": "h",
                                "x": [62, 52, 51, 45, 40, 36, 31, 28],
                                "y": long_labels,
                                "marker": {"color": "rgba(47, 101, 217, 0.92)"},
                            },
                        ],
                        "layout": {
                            "title": {"text": "MTLR Gradient-Based Feature Salience"},
                            "height": 720,
                            "margin": {"l": 300, "r": 24, "t": 54, "b": 48},
                        },
                    },
                    "loss": {
                        "data": [
                            {
                                "type": "scatter",
                                "mode": "lines",
                                "x": [1, 20, 40, 60, 80, 100],
                                "y": [1.48, 1.21, 0.96, 0.82, 0.74, 0.67],
                                "name": "Training loss",
                            },
                            {
                                "type": "scatter",
                                "mode": "lines",
                                "x": [1, 20, 40, 60, 80, 100],
                                "y": [1.45, 1.19, 0.9, 0.77, 0.64, 0.55],
                                "name": "Monitor loss",
                            },
                        ],
                        "layout": {
                            "title": {"text": "MTLR Training Loss and Monitor loss"},
                            "height": 360,
                        },
                    },
                },
            }),
        )

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.route("**/api/deep-model", _mock_dl_single)

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _wait_for_workspace(page)
            _open_predictive_workbench(page)

            page.locator("#predictiveModelSelector").select_option("mtlr")
            page.wait_for_function(
                "() => document.getElementById('benchmarkWorkbench')?.dataset.activeFamily === 'dl'"
            )
            page.locator("#runPredictiveWorkbenchButton").click()
            page.wait_for_function(
                "document.getElementById('dlMetaBanner').textContent.includes('MTLR:')"
            )
            page.wait_for_function(
                "() => document.getElementById('dlImportancePlot')?.classList.contains('js-plotly-plot') === true"
            )

            page.set_viewport_size({"width": 980, "height": 1400})
            page.wait_for_timeout(350)

            overflow = page.locator("#dlImportancePlot").evaluate(
                """(el) => ({
                  scrollWidth: el.scrollWidth,
                  clientWidth: el.clientWidth,
                  offsetWidth: el.offsetWidth,
                  plotWidth: el.classList.contains('js-plotly-plot')
                    ? el.getBoundingClientRect().width
                    : (el.querySelector('.js-plotly-plot')?.getBoundingClientRect().width || 0),
                })"""
            )
            loss_overflow = page.locator("#dlLossPlot").evaluate(
                """(el) => ({
                  scrollWidth: el.scrollWidth,
                  clientWidth: el.clientWidth,
                  offsetWidth: el.offsetWidth,
                  plotWidth: el.classList.contains('js-plotly-plot')
                    ? el.getBoundingClientRect().width
                    : (el.querySelector('.js-plotly-plot')?.getBoundingClientRect().width || 0),
                })"""
            )

            assert overflow["scrollWidth"] <= overflow["clientWidth"] + 1
            assert loss_overflow["scrollWidth"] <= loss_overflow["clientWidth"] + 1
            assert overflow["plotWidth"] <= overflow["offsetWidth"] + 1
            assert loss_overflow["plotWidth"] <= loss_overflow["offsetWidth"] + 1

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
            _wait_for_workspace(page)

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
            _wait_for_workspace(page)
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


def _compare_summary() -> dict:
    return {"status": "review", "headline": "Comparison complete.", "strengths": [], "cautions": [], "next_steps": []}


def test_browser_unified_board_requires_matching_split_fingerprints_and_reports_locked_test(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    sent: dict[str, list[dict]] = {"ml": [], "dl": []}
    fingerprints = {"ml": "rcv-seed7-shared", "dl": "rcv-seed7-shared"}

    def _mock_ml_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        sent["ml"].append(body)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "request_config": body,
                "analysis": {
                    "comparison_table": [
                        {"model": "Random Survival Forest", "c_index": 0.714, "c_index_std": 0.031, "evaluation_mode": "repeated_cv", "rank": 1, "locked_test_c_index": 0.688, "locked_test_samples": 108, "locked_test_events": 40},
                        {"model": "LASSO-Cox", "c_index": 0.681, "c_index_std": 0.024, "evaluation_mode": "repeated_cv", "rank": 2, "locked_test_c_index": 0.702, "locked_test_samples": 108, "locked_test_events": 40},
                    ],
                    "evaluation_mode": "repeated_cv",
                    "cv_folds": 5,
                    "cv_repeats": 3,
                    "evaluation_split_fingerprint": fingerprints["ml"],
                    "locked_test_fraction": 0.3,
                    "n_development_patients": 252,
                    "n_locked_test_patients": 108,
                    "n_locked_test_events": 40,
                    "scientific_summary": _compare_summary(),
                    "manuscript_tables": {"model_performance_table": []},
                },
            }),
        )

    def _mock_dl_compare(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        sent["dl"].append(body)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "request_config": body,
                "analysis": {
                    "comparison_table": [
                        {"model": "DeepHit", "c_index": 0.731, "evaluation_mode": "repeated_cv", "rank": 1, "locked_test_c_index": 0.651, "locked_test_samples": 108, "locked_test_events": 40},
                        {"model": "Survival VAE", "c_index": 0.604, "evaluation_mode": "repeated_cv", "rank": None, "comparable_for_ranking": False},
                    ],
                    "evaluation_mode": "repeated_cv",
                    "cv_folds": 5,
                    "cv_repeats": 3,
                    "evaluation_split_fingerprint": fingerprints["dl"],
                    "scientific_summary": _compare_summary(),
                    "manuscript_tables": {"model_performance_table": []},
                },
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
            _wait_for_workspace(page)
            _open_predictive_workbench(page)

            page.locator("#mlEvaluationStrategy").select_option("repeated_cv")
            page.locator("#mlRandomSeed").fill("7")
            # Evaluation settings are shared, so DL follows the ML controls.
            page.wait_for_function("() => document.getElementById('dlEvaluationStrategy').value === 'repeated_cv' && document.getElementById('dlRandomSeed').value === '7'")
            assert page.locator("#mlLockedTestToggle").is_checked()
            assert page.locator("#mlLockedTestFraction").input_value() == "30"

            page.locator("#closePredictiveWorkbenchButton").click()
            page.locator("#runPredictiveCompareAllButton").click()
            page.wait_for_function(
                "() => { const plot = document.getElementById('benchmarkComparisonPlot'); return plot && !plot.classList.contains('hidden') && Array.isArray(plot.data) && plot.data.length === 1; }",
                timeout=60000,
            )
            ml_body, dl_body = sent["ml"][-1], sent["dl"][-1]
            assert ml_body["random_state"] == 7 and dl_body["random_seed"] == 7
            assert ml_body["evaluation_strategy"] == dl_body["evaluation_strategy"] == "repeated_cv"
            assert (ml_body["cv_folds"], ml_body["cv_repeats"]) == (dl_body["cv_folds"], dl_body["cv_repeats"])
            assert ml_body["locked_test_fraction"] == pytest.approx(0.3)
            assert dl_body["locked_test_fraction"] == pytest.approx(0.3)

            table_text = page.locator("#benchmarkComparisonShell").inner_text()
            assert "LOCKED-TEST C-INDEX" in table_text.upper()
            assert "Not ranked" in table_text
            assert "0 \tDeep" not in table_text
            assert "locked-test c-index of the rank-1 model" in page.locator("#benchmarkTableNote").inner_text().lower()
            assert "SD (folds)" in page.locator("#mlComparisonShell").inner_text()

            # Toggling the locked test set changes the request, so the ML compare result goes stale.
            page.evaluate("() => document.getElementById('mlLockedTestToggle').click()")
            page.wait_for_function("() => currentCompareGoalPayload('ml') === null")
            assert not page.locator("#dlLockedTestToggle").is_checked()
            page.evaluate("() => document.getElementById('mlLockedTestToggle').click()")
            page.wait_for_function("() => currentCompareGoalPayload('ml') !== null")

            fingerprints["dl"] = "rcv-seed7-different"
            page.locator("#runPredictiveCompareAllButton").click()
            page.wait_for_function(
                "() => document.getElementById('benchmarkSummaryGrid').textContent.includes('different row partitions')",
                timeout=60000,
            )
            page.wait_for_function("() => document.getElementById('benchmarkComparisonPlot').classList.contains('hidden')")
            assert "split fingerprints differ" in page.locator("#benchmarkPlotNote").inner_text()
            assert "family rank" in page.locator("#benchmarkComparisonShell").inner_text().lower()
            assert "no cross-family ranking is published" in page.locator("#benchmarkTableNote").inner_text().lower()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_late_derive_response_does_not_replace_newer_dataset(browser_server: str, tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    upload = tmp_path / "second_cohort.csv"
    lines = ["subject,os_months,os_event,age,arm"]
    for index in range(80):
        lines.append(f"S{index},{5 + (index * 7) % 60},{index % 3 == 0 and 1 or 0},{40 + index % 30},{'AB'[index % 2]}")
    upload.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _wait_for_workspace(page)
            # The sample opens grouped by hormone therapy; derived groups start from Overall only.
            page.wait_for_function("document.getElementById('groupColumn').value === 'horTh'")
            page.locator("#groupColumn").select_option("")
            page.evaluate(
                """() => {
                  const originalFetch = window.fetch;
                  window.__deriveDelayed = 0;
                  window.fetch = async (url, options) => {
                    const response = await originalFetch(url, options);
                    if (String(url).includes('/api/derive-group')) {
                      window.__deriveDelayed += 1;
                      await new Promise((resolve) => setTimeout(resolve, 2500));
                    }
                    return response;
                  };
                }"""
            )
            page.locator('[data-tab="km"]').click()
            page.evaluate("() => document.getElementById('deriveToggle').click()")
            page.select_option("#deriveMethod", "median_split")
            page.evaluate("() => document.getElementById('deriveButton').click()")
            page.wait_for_function("() => window.__deriveDelayed === 1")
            page.set_input_files("#datasetFile", str(upload))
            page.wait_for_function("() => state.dataset && state.dataset.filename === 'second_cohort.csv'")
            page.wait_for_timeout(3200)

            assert page.evaluate("() => state.dataset.filename") == "second_cohort.csv"
            assert "second_cohort.csv" in page.locator("#datasetBadge").inner_text()
            assert page.locator("#timeUnitLabel").input_value() == "Months"
            assert page.locator("#maxTime").input_value() == ""

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_comparison_image_export_disabled_after_single_model_run(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _mock_ml(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        if body.get("model_type") == "compare":
            payload = {
                "request_config": body,
                "analysis": {
                    "comparison_table": [
                        {"model": "Random Survival Forest", "c_index": 0.714, "evaluation_mode": "holdout", "rank": 1},
                        {"model": "Cox PH", "c_index": 0.69, "evaluation_mode": "holdout", "rank": 2},
                    ],
                    "evaluation_mode": "holdout",
                    "evaluation_split_fingerprint": "holdout-seed42-shared",
                    "scientific_summary": _compare_summary(),
                    "manuscript_tables": {"model_performance_table": []},
                },
                "figure": {"data": [{"type": "bar", "x": ["Random Survival Forest", "Cox PH"], "y": [0.714, 0.69]}], "layout": {"title": {"text": "Model Comparison"}}},
            }
        else:
            payload = {
                "request_config": body,
                "analysis": {
                    "model_stats": {"c_index": 0.7, "evaluation_mode": "holdout", "n_patients": 360, "n_features": 4},
                    "scientific_summary": _compare_summary(),
                },
                "importance_figure": {"data": [{"type": "bar", "orientation": "h", "x": [0.3, 0.2], "y": ["age", "stage"]}], "layout": {"height": 360}},
            }
        route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.route("**/api/ml-model", _mock_ml)
            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _wait_for_workspace(page)
            page.locator('[data-tab="benchmark"]').click()
            _assert_tab_active(page, "benchmark")

            page.evaluate("() => document.getElementById('runCompareButton').click()")
            page.wait_for_function("() => !document.getElementById('downloadMlComparisonPngButton').disabled")

            _open_predictive_workbench(page, "rsf")
            page.locator("#runPredictiveWorkbenchButton").click()
            page.wait_for_function("document.getElementById('mlMetaBanner').textContent.includes('Holdout C-index=0.7')")
            page.evaluate("() => renderSharedFeatureSummary()")

            assert page.locator("#downloadMlComparisonPngButton").is_disabled()
            assert page.locator("#downloadMlComparisonSvgButton").is_disabled()
            assert page.evaluate("() => !(document.getElementById('mlComparisonPlot').data || []).length")

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_duplicate_identifier_warning_is_persistent_and_escaped(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    def _with_duplicates(route) -> None:
        response = route.fetch()
        payload = response.json()
        payload["duplicate_identifier_columns"] = [
            {"column": "patient_id<img src=x onerror=window.__xss=1>", "n_rows": 360, "n_unique": 300, "n_repeated_ids": 55, "n_extra_rows": 60},
        ]
        route.fulfill(response=response, body=json.dumps(payload))

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.route("**/api/load-example", _with_duplicates)
            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#datasetIntegrityWarning").wait_for(state="visible")
            warning = page.locator("#datasetIntegrityWarning").inner_text()
            assert "repeats 55 IDs" in warning
            assert "360 rows vs 300 unique" in warning
            assert "double-count" in warning
            assert page.evaluate("() => window.__xss || 0") == 0
            assert page.locator("#datasetIntegrityWarning img").count() == 0

            _wait_for_workspace(page)
            assert page.locator("#datasetIntegrityWarning").is_visible()
            page.locator("#brandHome").click()
            page.locator("#landing").wait_for(state="visible")
            assert page.locator("#datasetIntegrityWarning").is_hidden()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_cohort_table_sends_outcome_columns_and_shows_notes(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    sent: list[dict] = []

    def _mock_cohort_table(route) -> None:
        body = json.loads(route.request.post_data or "{}")
        sent.append(body)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "request_config": body,
                "analysis": {
                    "rows": [{"Variable": "age", "Level": "", "Overall": "53.1 (10.1)"}],
                    "columns": ["Variable", "Level", "Overall"],
                    "notes": ["Restricted to 686 rows with a valid survival outcome (same rows as Kaplan-Meier and Cox)."],
                },
            }),
        )

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.route("**/api/cohort-table", _mock_cohort_table)
            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _wait_for_workspace(page)
            page.locator('[data-tab="tables"]').click()
            _assert_tab_active(page, "tables")
            page.evaluate("() => setCheckedValues(refs.cohortVariableChecklist, ['age'])")
            page.evaluate("() => renderSharedFeatureSummary()")
            page.locator("#runCohortTableButton").click()
            page.wait_for_function("() => !document.getElementById('downloadCohortTableButton').disabled")

            body = sent[-1]
            assert body["time_column"] == "rfs_days"
            assert body["event_column"] == "rfs_event"
            assert str(body["event_positive_value"]) == "1"
            assert "same rows as Kaplan-Meier and Cox" in page.locator("#cohortTableShell").inner_text()
            assert page.locator("#runCohortTableButtonLabel").inner_text() == "Build Table"
            assert page.evaluate("() => currentCohortTableOutputState().isCurrent") is True

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_markers_tab_evaluates_the_example_markers(browser_server: str, tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _wait_for_workspace(page)
            page.locator('[data-tab="markers"]').click()
            _assert_tab_active(page, "markers")
            assert page.locator("#markerChecklist input[value='biomarker_score']").is_checked()
            assert page.locator("#markerClinicalChecklist input[value='age']").is_checked()
            assert "judged on added value" in page.locator("#markerSelectionLine").inner_text()
            assert page.locator("#markersTableShell").locator("xpath=ancestor::div[contains(@class,'table-card')]").is_hidden()

            page.locator("#panel-markers .options-details > summary").click()
            page.locator("#markerPermutations").fill("99")
            page.locator("#markerResamples").fill("10")
            page.locator("#runMarkersButton").click()
            page.wait_for_function("!document.getElementById('downloadMarkersCsvButton').disabled", timeout=120000)

            assert "biomarker_score" in page.locator("#markersTableShell").inner_text()
            assert page.locator('[data-run-status="markers"]').inner_text() == "Up to date"
            assert page.locator("#markerValidationSection").is_visible()
            assert page.locator("#runMarkerValidationButton").is_enabled()

            page.locator("#panel-markers .export-menu > summary").click()
            with page.expect_download() as remark_info:
                page.locator("#downloadMarkerRemarkMarkdownButton").click()
            remark_path = tmp_path / (remark_info.value.suggested_filename or "remark_checklist.md")
            remark_info.value.save_as(remark_path)
            remark_text = remark_path.read_text(encoding="utf-8")
            assert remark_path.suffix == ".md"
            assert remark_text.startswith("# REMARK checklist")
            assert "biomarker_score" in remark_text and "99 permutations" in remark_text

            page.locator("#markerChecklist input[value='immune_index']").uncheck()
            page.wait_for_function("document.querySelector('[data-run-status=\"markers\"]').textContent === 'Settings changed'")
            assert page.locator("#downloadMarkersCsvButton").is_disabled()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_markers_tab_evaluates_an_attached_marker_matrix(browser_server: str, tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    import numpy as np
    import pandas as pd

    from survival_toolkit.sample_data import make_example_dataset

    frame = make_example_dataset()
    rng = np.random.default_rng(4)
    genes = pd.DataFrame(rng.normal(size=(20, len(frame))), index=[f"GENE{index:02d}" for index in range(20)], columns=frame["patient_id"])
    genes.loc["SIGNAL"] = frame["biomarker_score"].to_numpy()
    matrix_path = tmp_path / "expression.tsv"
    genes.to_csv(matrix_path, sep="	", index_label="gene")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _wait_for_workspace(page)
            page.locator('[data-tab="markers"]').click()
            _assert_tab_active(page, "markers")

            page.locator("#markerMatrixDetails > summary").click()
            assert page.locator("#markerMatrixIdColumn").input_value() == "patient_id"
            page.locator("#markerMatrixFile").set_input_files(str(matrix_path))
            page.locator("#attachMarkerMatrixButton").click()
            page.locator("#markerMatrixStatus").wait_for(state="visible")
            assert "21 markers, 360 of 360 patients matched by patient_id" in page.locator("#markerMatrixSummary").inner_text()
            assert "21 markers from expression.tsv" in page.locator("#markerSelectionLine").inner_text()
            assert page.locator("#selectAllMarkersButton").is_disabled()
            assert page.locator("#attachMarkerMatrixButton").is_disabled()

            page.locator("#panel-markers .options-details > summary").click()
            page.locator("#markerPermutations").fill("99")
            page.locator("#markerResamples").fill("10")
            page.locator("#runMarkersButton").click()
            page.wait_for_function("!document.getElementById('downloadMarkersCsvButton').disabled", timeout=120000)

            assert "markers from expression.tsv" in page.locator("#markersMetaBanner").inner_text()
            assert "SIGNAL" in page.locator("#markersTableShell tbody tr").nth(0).inner_text()

            page.locator("#removeMarkerMatrixButton").click()
            assert page.locator("#markerMatrixStatus").is_hidden()
            assert page.locator("#selectAllMarkersButton").is_enabled()
            page.wait_for_function("document.querySelector('[data-run-status=\"markers\"]').textContent === 'Settings changed'")

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


def test_browser_design_check_page_flags_a_single_cohort_best_of_101_design(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = _launch_browser(api)
            page = browser.new_page(viewport={"width": 1280, "height": 1000})

            page.goto(f"{browser_server}/design-check", wait_until="networkidle")
            page.locator("#trainingInSelection").check()
            page.locator("#checkDesignButton").click()
            page.locator("#designResult").wait_for(state="visible")

            flags = page.locator("#designFlags").inner_text()
            assert "No cohort was kept out of the choice" in flags
            assert "The training cohort's apparent C-index entered the choice" in flags
            assert page.locator("#optimismValue").inner_text().startswith("+0.")

            page.locator("[data-remove-cohort]").click()
            page.locator("#checkDesignButton").click()
            page.wait_for_function("document.getElementById('designError').textContent.includes('Add at least one cohort')")

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        if _is_playwright_environment_error(exc):
            pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")
        raise


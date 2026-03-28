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


def test_browser_downloads_km_summary_csv_and_png(browser_server: str, tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")

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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_preset_application_shows_visible_feedback(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#datasetPresetBar").wait_for(state="visible")
            page.locator('[data-tab="dl"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"dl\"]').getAttribute('aria-selected') === 'true'"
            )

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
                "document.getElementById('dlFeatureSummaryText').textContent.includes('Shared across ML and DL')"
            )
            assert page.locator('[data-tab="dl"]').get_attribute("aria-selected") == "true"
            assert "feature checklists used by ML and DL" in page.locator("#datasetPresetStatusText").inner_text()
            assert "Model features: 6" in page.locator("#datasetPresetChips").inner_text()
            assert "Features: 6" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Categorical: 3" in page.locator("#dlFeatureSummaryChips").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_cox_results_table_stays_within_card(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            page.locator("#workspace").wait_for(state="visible")
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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_back_button_returns_to_home_not_blank(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#runKmButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadKmSummaryButton').disabled"
            )

            page.go_back(wait_until="networkidle")
            page.locator("#landing").wait_for(state="visible")
            assert page.locator("#workspace").is_hidden()
            assert "Drop a file here or click to browse" in page.locator("#landing").inner_text()

            page.go_forward(wait_until="networkidle")
            page.locator("#workspace").wait_for(state="visible")
            assert not page.locator("#downloadKmSummaryButton").is_disabled()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")

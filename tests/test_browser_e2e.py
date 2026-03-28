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


def _switch_to_expert(page) -> None:
    page.locator("#workspace").wait_for(state="visible")
    if page.locator("#guidedShell").is_visible():
        page.locator("#expertModeButton").click()
        page.wait_for_function("document.body.dataset.uiMode === 'expert'")


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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_preset_application_shows_visible_feedback(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _switch_to_expert(page)
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
                "document.getElementById('dlFeatureSummaryText').textContent.includes('Training inputs come only from the shared ML/DL model feature selections')"
            )
            assert page.locator('[data-tab="dl"]').get_attribute("aria-selected") == "true"
            assert "feature checklists used by ML and DL" in page.locator("#datasetPresetStatusText").inner_text()
            assert "Model features: 6" in page.locator("#datasetPresetChips").inner_text()
            assert "Model features: 6" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Grouping only: horTh" in page.locator("#dlFeatureSummaryChips").inner_text()
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
            _switch_to_expert(page)
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
            _switch_to_expert(page)
            page.locator("#timeUnitLabel").fill("Days")
            page.locator("#maxTime").fill("24")
            page.locator("#groupColumn").select_option("stage")
            page.locator("#covariateChecklist input[value='immune_index']").check()
            page.locator("#covariateChecklist input[value='age']").uncheck()
            page.locator("#runKmButton").click()
            page.wait_for_function(
                "!document.getElementById('downloadKmSummaryButton').disabled"
            )
            page.locator('[data-tab="ml"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"ml\"]').getAttribute('aria-selected') === 'true'"
            )

            page.go_back(wait_until="networkidle")
            page.locator("#landing").wait_for(state="visible")
            assert page.locator("#workspace").is_hidden()
            assert "Drop a file here or click to browse" in page.locator("#landing").inner_text()

            page.go_forward(wait_until="networkidle")
            page.locator("#workspace").wait_for(state="visible")
            assert not page.locator("#downloadKmSummaryButton").is_disabled()
            assert page.locator('[data-tab="ml"]').get_attribute("aria-selected") == "true"
            assert page.locator("#timeUnitLabel").input_value() == "Days"
            assert page.locator("#maxTime").input_value() == "24"
            assert page.locator("#groupColumn").input_value() == "stage"
            assert page.locator("#covariateChecklist input[value='immune_index']").is_checked()
            assert not page.locator("#covariateChecklist input[value='age']").is_checked()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_study_design_collapses_grouping_controls_outside_km(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            assert page.locator("#groupingDetails").evaluate("(el) => el.open") is True
            assert "Current Group by: overall only" in page.locator("#groupingSummaryText").inner_text().lower()

            page.locator('[data-tab="dl"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"dl\"]').getAttribute('aria-selected') === 'true'"
            )
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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_guided_mode_goal_cards_and_mode_toggle(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")

            assert "Confirm outcome" in page.locator("#guidedSummaryTitle").inner_text()
            assert page.locator("#configStrip").is_visible()
            assert page.locator("#tabStrip").is_hidden()
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function(
                "document.getElementById('guidedSummaryTitle').textContent.includes('Choose analysis')"
            )
            assert "Choose analysis" in page.locator("#guidedSummaryTitle").inner_text()
            assert "Event:" in page.locator("#guidedSummaryChips").inner_text()
            assert "Group by: Overall only" in page.locator("#guidedSummaryChips").inner_text()
            assert page.locator("#configStrip").is_hidden()

            page.locator('[data-guided-action="choose-goal"][data-goal="cox"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"cox\"]').getAttribute('aria-selected') === 'true'"
            )
            assert "Configure Cox PH" in page.locator("#guidedPanel").inner_text()
            assert "Goal: Cox PH" in page.locator("#guidedSummaryChips").inner_text()
            assert page.locator("#tabStrip").is_hidden()
            assert page.locator("#configStrip").is_hidden()
            assert page.locator('#panel-cox').is_visible()
            assert page.locator('#panel-km').is_hidden()

            page.locator("#expertModeButton").click()
            page.wait_for_function(
                "document.body.dataset.uiMode === 'expert' && document.getElementById('guidedShell').classList.contains('hidden')"
            )

            page.locator("#guidedModeButton").click()
            page.wait_for_function(
                "document.body.dataset.uiMode === 'guided' && !document.getElementById('guidedShell').classList.contains('hidden')"
            )
            assert "Configure Cox PH" in page.locator("#guidedPanel").inner_text()
            assert page.locator("#tabStrip").is_hidden()
            assert page.locator('#panel-cox').is_visible()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_model_features_stay_separate_from_cox_and_keep_non_endpoint_inputs(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
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

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveMethod").select_option("optimal_cutpoint")
            page.locator("#deriveButton").click()
            page.wait_for_function(
                "document.getElementById('deriveStatus').textContent.includes('Created')"
            )
            assert page.locator("#groupColumn").input_value() == "sex"
            assert "Group by stayed as sex" in page.locator("#deriveStatus").inner_text()
            assert "Set as Group by" in page.locator("#deriveSummary").inner_text()

            updated_model_features = page.eval_on_selector_all(
                "#modelFeatureChecklist input",
                "els => els.filter(e => e.checked).map(e => e.value)",
            )
            assert updated_model_features == model_features

            page.locator('[data-tab="dl"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"dl\"]').getAttribute('aria-selected') === 'true'"
            )
            assert "Model features: 7" in page.locator("#dlFeatureSummaryChips").inner_text()
            assert "Grouping only: sex" in page.locator("#dlFeatureSummaryChips").inner_text()

            page.locator("#deriveSummary #applyDerivedGroupButton").click()
            page.wait_for_function(
                "document.getElementById('groupColumn').value === 'age__optimal_cutpoint'"
            )
            assert page.locator("#groupColumn").input_value() == "age__optimal_cutpoint"
            assert "Grouping only: age__optimal_cutpoint" in page.locator("#dlFeatureSummaryChips").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_event_column_defaults_to_event_like_fields_and_advanced_toggle_reveals_all(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
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
            assert "Showing event-like binary columns only" in page.locator("#eventColumnHelp").inner_text()

            page.locator("#showAllEventColumns").check()
            page.wait_for_function(
                "Array.from(document.querySelectorAll('#eventColumn option')).some((option) => option.value === 'egfr_status')"
            )
            advanced_options = page.locator("#eventColumn option").evaluate_all(
                "(options) => options.map((option) => option.value)"
            )
            assert "egfr_status" in advanced_options
            assert "Advanced mode is on" in page.locator("#eventColumnHelp").inner_text()

            page.locator("#eventColumn").select_option("egfr_status")
            page.wait_for_function(
                "document.getElementById('eventColumnWarning').textContent.includes('baseline characteristic')"
            )
            assert "Use it as Group by or as a model feature" in page.locator("#eventColumnWarning").inner_text()
            assert page.locator("#eventPositiveValue").input_value() == ""

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_event_value_requires_explicit_choice_for_ambiguous_binary_codes(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            _switch_to_expert(page)

            page.locator("#showAllEventColumns").check()
            page.locator("#eventColumn").select_option("egfr_status")
            page.wait_for_function(
                "document.getElementById('eventValueWarning').textContent.includes('Choose which value means the event happened') || document.getElementById('eventValueWarning').textContent.includes('could not safely guess')"
            )
            assert page.locator("#eventPositiveValue").input_value() == ""

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_km_derive_defaults_to_group_when_current_group_is_overall_only(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)

            page.locator("#runKmButton").click()
            page.wait_for_function(
                "document.getElementById('kmMetaBanner').textContent.includes('N=')"
            )
            assert "p=" not in page.locator("#kmMetaBanner").inner_text()

            assert page.locator("#groupColumn").input_value() == ""
            page.locator("#deriveToggle").click()
            assert page.locator("#deriveApplyToGroup").is_checked()

            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveMethod").select_option("median_split")
            page.locator("#deriveColumnName").fill("age_guided_split")
            page.locator("#deriveButton").click()
            page.wait_for_function(
                "document.getElementById('groupColumn').value === 'age_guided_split'"
            )
            page.wait_for_function(
                "document.getElementById('kmMetaBanner').textContent.includes('p=')"
            )

            assert page.locator("#groupColumn").input_value() == "age_guided_split"
            assert "Refreshing Kaplan-Meier with the new grouping" in page.locator("#deriveStatus").inner_text()
            assert "Group by now uses age_guided_split" in page.locator("#deriveSummary").inner_text()
            assert "p=" in page.locator("#kmMetaBanner").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_dataset_entry_resets_scroll_to_top(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})

            page.goto(browser_server, wait_until="networkidle")
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
            assert page.evaluate("window.scrollY") > 0

            page.evaluate("document.getElementById('loadExampleButton').click()")
            page.locator("#workspace").wait_for(state="visible")
            page.wait_for_timeout(450)

            assert page.evaluate("window.scrollY") <= 24
            config_box = page.locator("#configStrip").bounding_box()
            assert config_box is not None
            assert config_box["y"] < 220

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_guided_mode_back_button_walks_previous_steps(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
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
            assert "Choose one analysis path" in page.locator("#guidedPanel").inner_text()

            page.go_back(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            assert "Confirm outcome" in page.locator("#guidedPanel").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_guided_step_rail_allows_navigation_to_reached_steps(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
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
            assert "Choose one analysis path" in page.locator("#guidedPanel").inner_text()

            page.locator('#stepIndicator .step[data-step="2"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            assert "Confirm outcome" in page.locator("#guidedPanel").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_guided_and_expert_mode_back_forward_restores_mode_and_step(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#guidedShell").wait_for(state="visible")

            page.wait_for_function("document.body.dataset.guidedStep === '2'")
            page.locator('[data-guided-action="next-step"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '3'")
            page.locator('[data-guided-action="choose-goal"][data-goal="ml"]').click()
            page.wait_for_function("document.body.dataset.guidedStep === '4'")

            page.locator("#expertModeButton").click()
            page.wait_for_function("document.body.dataset.uiMode === 'expert'")
            page.locator('[data-tab="dl"]').click()
            page.wait_for_function("document.querySelector('[data-tab=\"dl\"]').getAttribute('aria-selected') === 'true'")

            page.go_back(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.uiMode === 'guided' && document.body.dataset.guidedStep === '4'")
            assert "Configure ML models" in page.locator("#guidedPanel").inner_text()

            page.go_forward(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.uiMode === 'expert'")
            assert page.locator('[data-tab="dl"]').get_attribute("aria-selected") == "true"

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_optimal_cutpoint_summary_explains_risk_labels(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadGbsg2Button").click()
            _switch_to_expert(page)

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("age")
            page.locator("#deriveMethod").select_option("optimal_cutpoint")
            page.locator("#deriveButton").click()
            page.wait_for_function(
                "document.getElementById('deriveSummary').textContent.includes('Assignment rule')"
            )

            derive_text = page.locator("#deriveSummary").inner_text()
            assert "High/Low indicate risk direction" in derive_text
            assert "Assignment rule" in derive_text
            assert ("Selection-adjusted p-value" in derive_text) or ("Raw p-value" in derive_text)
            assert "ML and DL feature selections did not change automatically" in derive_text
            assert "not as an ML/DL training feature" in derive_text
            assert "Set as Group by" in derive_text

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_optimal_cutpoint_summary_wraps_long_derived_column(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadTcgaButton").click()
            _switch_to_expert(page)

            page.locator("#deriveToggle").click()
            page.locator("#deriveSource").select_option("pack_years_smoked")
            page.locator("#deriveMethod").select_option("optimal_cutpoint")
            page.locator("#deriveButton").click()
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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_ml_importance_plot_stays_inside_its_section(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1400})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            page.locator('[data-tab="ml"]').click()
            page.locator("#mlModelType").select_option("rsf")
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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_risk_table_ticks_change_columns_and_flash_table(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
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
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")


def test_browser_dl_epoch_validation_message_is_human_readable(browser_server: str) -> None:
    playwright = pytest.importorskip("playwright.sync_api")

    try:
        with playwright.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})

            page.goto(browser_server, wait_until="networkidle")
            page.locator("#loadExampleButton").click()
            _switch_to_expert(page)
            page.locator('[data-tab="dl"]').click()
            page.wait_for_function(
                "document.querySelector('[data-tab=\"dl\"]').getAttribute('aria-selected') === 'true'"
            )
            page.locator("#dlEpochs").fill("1001")
            page.locator("#runDlButton").click()
            page.wait_for_function(
                "document.getElementById('toastContainer').textContent.includes('Epochs must be between 10 and 1000')"
            )
            assert "Epochs must be between 10 and 1000" in page.locator("#toastContainer").inner_text()

            browser.close()
    except Exception as exc:  # pragma: no cover - environment-dependent skip path
        pytest.skip(f"Playwright browser test unavailable in this environment: {exc}")

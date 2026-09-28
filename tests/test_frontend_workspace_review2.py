"""Behaviour tests of the outcome pickers, column lists, history, workspace chrome and the design check page.

They run the real templates and scripts in Node with the small DOM of test_frontend_review.py. Dataset payloads
come from the server's own profiler, so the column kinds and suggestions are those a real upload gets.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest
from test_frontend_review import _HARNESS_JS, _ROOT, _STATIC_DIR, _TEMPLATE, _run_page

from survival_toolkit.analysis import profile_dataframe

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")

_DESIGN_TEMPLATE = _ROOT / "templates" / "design_check.html"
_SCRIPT_LOOP = "for (const file of SCRIPT_ORDER) {"
# Evaluated in the page: "ready" when a run could start, otherwise the reason it cannot.
_ENDPOINT_STATUS = "(() => { try { currentBaseConfig(); return 'ready'; } catch (error) { return error.message; } })()"


def _profile(dataset_id: str, columns: dict) -> dict:
    """A dataset payload as the server builds it for an upload."""
    return json.loads(json.dumps(profile_dataframe(pd.DataFrame(columns), dataset_id, f"{dataset_id}.csv")))


def _veteran() -> dict:
    return _profile("veteran", {
        "trt": [1, 2] * 6,
        "time": [72, 411, 228, 126, 118, 10, 82, 110, 314, 100, 42, 8],
        "status": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        "age": [69, 64, 38, 63, 65, 49, 69, 68, 43, 70, 81, 63],
    })


def _lung(dataset_id: str = "lung") -> dict:
    # survival::lung codes status 1 = censored, 2 = dead.
    return _profile(dataset_id, {
        "inst": [3, 3, 3, 5, 1, 12, 7, 11, 1, 7, 6, 16],
        "time": [306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654],
        "status": [2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2],
        "relapse": [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        "age": [74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68],
        "sex": [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2],
    })


# ── Event column and event value ───────────────────────────────


def test_event_value_never_carries_over_to_another_dataset(tmp_path: Path) -> None:
    """#1: "status" means 1 = event in veteran (0/1) but 1 = censored in lung (1/2)."""
    derived = _lung("lung-derived")
    derived["columns"].append({"name": "age_group", "kind": "categorical", "missing": 0, "non_missing": 12, "n_unique": 2, "unique_preview": ["Low", "High"]})
    derived["derived_column"] = "age_group"
    derived["derive_summary"] = {"method": "median_split", "counts": []}
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.veteran);
      const veteran = page.run("({ event: refs.eventColumn.value, value: refs.eventPositiveValue.value })");
      const veteranState = page.run("currentHistoryState()");
      await loadDataset(page, fixtures.lung);
      const lung = page.run(`({
        event: refs.eventColumn.value,
        value: refs.eventPositiveValue.value,
        warning: refs.eventValueWarning.textContent,
      })`);
      lung.runnable = page.run(fixtures.status);
      page.fetchHandler = () => ({ status: 500, body: { detail: "no request expected" } });
      page.run("refs.runKmButton.click()");
      await page.settle();
      const kmRequests = page.requests.filter((request) => request.url.endsWith("/api/kaplan-meier")).length;
      // The user confirms that 2 means the event; a derived group of the same data keeps that choice.
      page.change("#eventPositiveValue", "2");
      page.fetchHandler = (request) => (request.url.endsWith("/api/derive-group") ? { status: 200, body: fixtures.derived } : { status: 500, body: { detail: "x" } });
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); refs.deriveSource.value = 'age'; refs.deriveButton.click()");
      await page.settle();
      const derived = page.run("({ dataset: state.dataset.dataset_id, value: refs.eventPositiveValue.value })");
      // Back to the veteran page restores its own value.
      page.fetchHandler = (request) => (request.url.endsWith("/api/dataset/veteran") ? { status: 200, body: fixtures.veteran } : { status: 500, body: { detail: "x" } });
      page.context.__state = veteranState;
      await page.run("restoreHistoryState(__state)");
      await page.settle();
      const back = page.run("({ dataset: state.dataset.dataset_id, value: refs.eventPositiveValue.value })");
      return { veteran, lung, kmRequests, derived, back };
    """, veteran=_veteran(), lung=_lung(), derived=derived, status=_ENDPOINT_STATUS)

    assert result["veteran"] == {"event": "status", "value": "1"}
    assert result["lung"]["event"] == "status"
    assert result["lung"]["value"] == ""
    assert "TCGA-style 1/2 coding" in result["lung"]["warning"]
    assert result["lung"]["runnable"] == 'Choose the Event Value for "status" before running an analysis.'
    assert result["kmRequests"] == 0
    assert result["derived"] == {"dataset": "lung-derived", "value": "2"}
    assert result["back"] == {"dataset": "veteran", "value": "1"}


def test_event_columns_that_name_the_event_pass_the_baseline_check(tmp_path: Path) -> None:
    """#3: the server accepts "progression_on_therapy" and "treatment_failure" coded 0/1; "treatment_arm" stays out."""
    dataset = _profile("pot", {
        "pfs_months": [5, 8, 12, 20, 25, 30, 33, 40, 41, 50, 52, 60],
        "progression_on_therapy": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "treatment_failure": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        "treatment_arm": ["A", "B"] * 6,
        "age": [50, 61, 72, 45, 66, 58, 70, 49, 63, 55, 68, 59],
    })
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const load = page.run(`({ event: refs.eventColumn.value, value: refs.eventPositiveValue.value,
        options: refs.eventColumn.options.map((option) => option.value), warning: refs.eventColumnWarning.textContent,
        covariates: allCheckboxValues(refs.covariateChecklist) })`);
      load.ready = page.run(fixtures.status);
      page.change("#eventColumn", "treatment_failure");
      const failure = page.run(`({ value: refs.eventPositiveValue.value, warning: refs.eventColumnWarning.textContent })`);
      failure.ready = page.run(fixtures.status);
      page.change("#showAllEventColumns", true);
      page.change("#eventColumn", "treatment_arm");
      const arm = page.run(`({ warning: refs.eventColumnWarning.textContent })`);
      arm.ready = page.run(fixtures.status);
      return { load, failure, arm };
    """, dataset=dataset, status=_ENDPOINT_STATUS)

    assert result["load"]["event"] == "progression_on_therapy"
    assert result["load"]["value"] == "1"
    assert result["load"]["options"] == ["progression_on_therapy", "treatment_failure"]
    assert result["load"]["warning"] == ""
    assert result["load"]["ready"] == "ready"
    # Event columns are outcomes, not covariates (the server refuses them as model inputs too).
    assert result["load"]["covariates"] == ["treatment_arm", "age"]
    assert result["failure"] == {"value": "1", "warning": "", "ready": "ready"}
    assert "looks more like a baseline characteristic" in result["arm"]["warning"]
    assert "looks more like a baseline characteristic" in result["arm"]["ready"]


def test_server_suggested_event_columns_are_offered_and_kept_out_of_covariates(tmp_path: Path) -> None:
    """#4: the TCGA-CDR 0/1 endpoints OS, PFI and DSS are event columns even next to vital_status."""
    dataset = _profile("cdr", {
        "gender": ["MALE", "FEMALE"] * 6,
        "vital_status": ["Alive", "Dead", "Alive", "Dead", "Alive", "Alive", "Dead", "Alive", "Alive", "Dead", "Alive", "Alive"],
        "OS": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "OS.time": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        "PFI": [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        "PFI.time": [50, 80, 300, 90, 500, 100, 120, 800, 900, 60, 1100, 70],
        "marital_status": ["Married", "Single"] * 6,
        "age": [50, 61, 72, 45, 66, 58, 70, 49, 63, 55, 68, 59],
    })
    assert {"OS", "PFI", "marital_status"} <= set(dataset["suggestions"]["event_columns"])
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const load = page.run(`({ event: refs.eventColumn.value, value: refs.eventPositiveValue.value,
        options: refs.eventColumn.options.map((option) => option.value), help: refs.eventColumnHelp.textContent,
        covariates: allCheckboxValues(refs.covariateChecklist), features: allCheckboxValues(refs.modelFeatureChecklist),
        tableVariables: allCheckboxValues(refs.cohortVariableChecklist), deriveSources: refs.deriveSource.options.map((option) => option.value) })`);
      page.change("#timeColumn", "PFI.time");
      page.change("#eventColumn", "PFI");
      const pfi = page.run(`({ value: refs.eventPositiveValue.value, warning: refs.eventColumnWarning.textContent })`);
      pfi.ready = page.run(fixtures.status);
      return { load, pfi };
    """, dataset=dataset, status=_ENDPOINT_STATUS)

    load = result["load"]
    assert load["event"] == "vital_status"
    assert load["value"] == "Dead"
    # A suggestion whose values are no event coding (Married/Single) stays a covariate.
    assert load["options"] == ["vital_status", "OS", "PFI"]
    assert load["help"] == "Showing likely event columns only."
    for listed in (load["covariates"], load["features"], load["tableVariables"]):
        assert not {"OS", "PFI", "vital_status", "OS.time", "PFI.time"} & set(listed)
        assert {"gender", "marital_status", "age"} <= set(listed)
    assert load["deriveSources"] == ["age"]
    assert result["pfi"] == {"value": "1", "warning": "", "ready": "ready"}


def test_a_censoring_flag_is_neither_suggested_nor_accepted(tmp_path: Path) -> None:
    """#9: "censored" coded 0/1 usually means 1 = censored; the server refuses it as the event column."""
    dataset = _profile("cens", {
        "time": [5, 8, 12, 20, 25, 30, 33, 40, 41, 50, 52, 60],
        "censored": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "age": [50, 61, 72, 45, 66, 58, 70, 49, 63, 55, 68, 59],
        "treatment": ["A", "B"] * 6,
    })
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const load = page.run(`({ event: refs.eventColumn.value, options: refs.eventColumn.options.map((option) => option.value),
        help: refs.eventColumnHelp.textContent, covariates: allCheckboxValues(refs.covariateChecklist) })`);
      page.change("#eventColumn", "censored");
      const chosen = page.run(`({ warning: refs.eventColumnWarning.textContent, tone: refs.eventColumnWarning.className })`);
      chosen.ready = page.run(fixtures.status);
      page.fetchHandler = () => ({ status: 500, body: { detail: "no request expected" } });
      page.run("refs.runKmButton.click()");
      await page.settle();
      chosen.kmRequests = page.requests.filter((request) => request.url.endsWith("/api/kaplan-meier")).length;
      return { load, chosen };
    """, dataset=dataset, status=_ENDPOINT_STATUS)

    assert result["load"]["event"] == ""
    assert result["load"]["options"] == ["", "censored", "treatment"]
    assert result["load"]["help"] == "No clear event column name was found; showing binary columns."
    assert "censored" not in result["load"]["covariates"]
    assert "looks like a censoring indicator" in result["chosen"]["warning"]
    assert "event = 1 - censored" in result["chosen"]["warning"]
    assert "event-warning-error" in result["chosen"]["tone"]
    assert result["chosen"]["ready"] == result["chosen"]["warning"]
    assert result["chosen"]["kmRequests"] == 0


def test_without_a_likely_event_column_nothing_is_preselected(tmp_path: Path) -> None:
    """#20: the dataset's second column ("sex") is no event guess, and the help text says why the list is broad."""
    dataset = _profile("ovarian", {
        "id": [f"p{index}" for index in range(12)],
        "sex": ["M", "F"] * 6,
        "age": [50, 61, 72, 45, 66, 58, 70, 49, 63, 55, 68, 59],
        "futime": [59, 115, 156, 421, 431, 448, 464, 475, 477, 563, 638, 744],
        "fustat": [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    })
    assert dataset["suggestions"]["event_columns"] == []
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const load = page.run(`({ event: refs.eventColumn.value, options: refs.eventColumn.options.map((option) => option.value),
        help: refs.eventColumnHelp.textContent, warning: refs.eventColumnWarning.textContent })`);
      page.change("#eventColumn", "fustat");
      load.chosen = page.run("refs.eventColumn.value");
      return load;
    """, dataset=dataset)

    assert result["event"] == ""
    assert result["options"] == ["", "sex", "fustat"]
    assert result["help"] == "No clear event column name was found; showing binary columns."
    assert result["warning"] == ""
    assert result["chosen"] == "fustat"


def test_an_event_column_without_values_replaces_the_previous_warning(tmp_path: Path) -> None:
    """#21: choosing an all-missing column showed the warning of the column chosen before."""
    dataset = _profile("missing", {
        "os_months": [5, 8, 12, 20, 25, 30, 33, 40, 41, 50, 52, 60],
        "os_event": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "sex": ["M", "F"] * 6,
        "empty_flag": [None] * 12,
    })
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.change("#showAllEventColumns", true);
      page.change("#eventColumn", "sex");
      const before = page.run("refs.eventColumnWarning.textContent");
      page.change("#eventColumn", "empty_flag");
      return { before, after: page.run("refs.eventColumnWarning.textContent"), value: page.run("refs.eventValueWarning.textContent") };
    """, dataset=dataset)

    assert "baseline characteristic" in result["before"]
    assert result["after"] == '"empty_flag" is not a binary event column. Choose a 0/1-style event column or recode it.'
    assert "No non-missing values" in result["value"]


def test_the_time_menu_lists_only_numeric_suggestions(tmp_path: Path) -> None:
    """#10: the server suggests "long_term_survival" (yes/no) by name; it is no follow-up time."""
    dataset = _profile("lts", {
        "long_term_survival": ["yes", "no", "no", "yes", "no", "no", "yes", "no", "no", "no", "yes", "no"],
        "os_months": [50, 8, 12, 60, 25, 30, 70, 40, 41, 50, 52, 60],
        "os_event": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "age": [50, 61, 72, 45, 66, 58, 70, 49, 63, 55, 68, 59],
    })
    assert dataset["suggestions"]["time_columns"][0] == "long_term_survival"
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      return page.run(`({ time: refs.timeColumn.value, options: refs.timeColumn.options.map((option) => option.value),
        warning: refs.timeColumnWarning.textContent, ready: endpointIsReady() })`);
    """, dataset=dataset)

    assert result == {"time": "os_months", "options": ["os_months"], "warning": "", "ready": True}


# ── Snapshots and history ──────────────────────────────────────


def _categorical_dataset() -> dict:
    return _profile("cats", {
        "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "event": [0, 1] * 6,
        "age": [50, 60, 70, 80, 55, 65, 75, 85, 52, 62, 72, 82],
        "stage": [1, 2, 3, 4] * 3,
        "sex": ["M", "F"] * 6,
        "bmi": [20, 25, 30, 35, 21, 22, 23, 24, 26, 27, 28, 29],
    })


def test_restoring_a_snapshot_keeps_the_categorical_flags_the_user_unticked(tmp_path: Path) -> None:
    """#2: Back/Forward and a new derived group re-ticked "Treat as categorical" flags."""
    derived = _categorical_dataset()
    derived["dataset_id"] = "cats-derived"
    derived["columns"].append({"name": "age_group", "kind": "categorical", "missing": 0, "non_missing": 12, "n_unique": 2, "unique_preview": ["Low", "High"]})
    derived["derived_column"] = "age_group"
    derived["derive_summary"] = {"method": "median_split", "counts": []}
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const flags = () => page.run("({ ml: selectedCheckboxValues(refs.modelCategoricalChecklist), dl: selectedCheckboxValues(refs.dlModelCategoricalChecklist) })");
      const initial = flags();
      for (const list of ["#modelCategoricalChecklist", "#dlModelCategoricalChecklist"]) {
        page.run(`document.querySelector("${list} input[value='stage']").checked = false`);
        page.change(`${list} input[value='stage']`);
      }
      const unticked = flags();
      page.context.__snapshot = page.run("captureControlSnapshot()");
      page.run("applyControlSnapshot(__snapshot)");
      const restored = flags();
      page.fetchHandler = (request) => (request.url.endsWith("/api/derive-group") ? { status: 200, body: fixtures.derived } : { status: 500, body: { detail: "x" } });
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); refs.deriveSource.value = 'age'; refs.deriveButton.click()");
      await page.settle();
      return { initial, unticked, restored, derived: flags(), dataset: page.run("state.dataset.dataset_id") };
    """, dataset=_categorical_dataset(), derived=derived)

    assert result["initial"] == {"ml": ["stage", "sex"], "dl": ["stage", "sex"]}
    assert result["unticked"] == {"ml": ["sex"], "dl": ["sex"]}
    assert result["restored"] == {"ml": ["sex"], "dl": ["sex"]}
    assert result["dataset"] == "cats-derived"
    assert result["derived"] == {"ml": ["sex"], "dl": ["sex"]}


def test_restoring_a_snapshot_updates_the_ml_model_controls(tmp_path: Path) -> None:
    """#13: Back to a Gradient Boosted Survival page left the learning rate disabled as for RSF."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.change("#mlModelType", "gbs");
      page.context.__state = page.run("currentHistoryState()");
      page.change("#mlModelType", "rsf");
      const rsf = page.run("refs.mlLearningRate.disabled");
      await page.run("restoreHistoryState(__state)");
      await page.settle();
      return { rsf, model: page.run("refs.mlModelType.value"), disabled: page.run("refs.mlLearningRate.disabled") };
    """, dataset=_categorical_dataset())

    assert result == {"rsf": True, "model": "gbs", "disabled": False}


def test_a_restored_endpoint_gets_its_own_warnings(tmp_path: Path) -> None:
    """#22: the "confirm the 1/2 coding" note stayed although the restored page had 2 chosen."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.change("#eventPositiveValue", "2");
      page.context.__state = page.run("currentHistoryState()");
      page.change("#eventColumn", "relapse");
      const other = page.run("({ event: refs.eventColumn.value, value: refs.eventPositiveValue.value })");
      await page.run("restoreHistoryState(__state)");
      await page.settle();
      return { other, event: page.run("refs.eventColumn.value"), value: page.run("refs.eventPositiveValue.value"),
        warning: page.run("refs.eventValueWarning.textContent"), ready: page.run("endpointIsReady()") };
    """, dataset=_lung())

    assert result["other"] == {"event": "relapse", "value": "1"}
    assert result["event"] == "status"
    assert result["value"] == "2"
    assert result["warning"] == ""
    assert result["ready"] is True


def test_a_history_entry_without_a_state_keeps_the_workspace_and_the_skip_link_moves_focus(tmp_path: Path) -> None:
    """#8: following "Skip to workspace" (#workspace) made an entry whose popstate took the page home."""
    result = _run_page(tmp_path, r"""
      const click = (selector) => page.run(`(() => {
        const event = { type: "click", bubbles: true, target: null, defaultPrevented: false,
          preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
        document.querySelector("${selector}").dispatchEvent(event);
        return event.defaultPrevented;
      })()`);
      const landingSkip = { prevented: click(".skip-link"), focused: page.run("document.activeElement?.id || null") };
      await loadDataset(page, fixtures.dataset);
      const skip = { prevented: click(".skip-link"), focused: page.run("document.activeElement?.id || null"),
        tabindex: page.run("refs.workspace.getAttribute('tabindex')") };
      page.run("refs.workspace.dispatchEvent({ type: 'blur', bubbles: false, preventDefault() {}, stopPropagation() {} })");
      skip.tabindexAfterBlur = page.run("refs.workspace.getAttribute('tabindex')");
      page.run("history.state = null");
      await page.run("restoreHistoryState(null)");
      await page.settle();
      return { landingSkip, skip, dataset: page.run("state.dataset?.dataset_id || null"),
        workspaceHidden: page.run("refs.workspace.classList.contains('hidden')"), entry: page.run("history.state?.view || null") };
    """, dataset=_categorical_dataset())

    assert result["landingSkip"] == {"prevented": True, "focused": "landing"}
    assert result["skip"] == {"prevented": True, "focused": "workspace", "tabindex": "-1", "tabindexAfterBlur": None}
    assert result["dataset"] == "cats"
    assert result["workspaceHidden"] is False
    assert result["entry"] == "workspace"


def test_history_updates_the_browser_refuses_do_not_break_the_page(tmp_path: Path) -> None:
    """#17: Safari and Firefox throw a SecurityError when a page updates its history too often."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.run(`history.replaceState = () => { throw new Error("SecurityError: too many calls"); };
        history.pushState = history.replaceState;`);
      const tab = page.run("(() => { try { activateTab('cox', { historyMode: 'push' }); return activeTabName(); } catch (error) { return error.message; } })()");
      page.change("#groupColumn", "sex");
      await page.settle();
      return { tab, group: page.run("refs.groupColumn.value") };
    """, dataset=_categorical_dataset())

    assert result == {"tab": "cox", "group": "sex"}


def test_leaving_the_page_with_results_asks_first_and_the_design_link_opens_a_new_tab(tmp_path: Path) -> None:
    """#6: the Markers tab's design-check link replaced the workspace, and Back could not bring it back."""
    html = _TEMPLATE.read_text(encoding="utf-8")
    markers_panel = html[html.index('id="panel-markers"'):html.index("<!-- Cohort Table Panel -->")]
    link = re.search(r'<a href="/design-check"[^>]*>', markers_panel)
    assert link and 'target="_blank"' in link.group(0) and 'rel="noopener"' in link.group(0)

    result = _run_page(tmp_path, r"""
      const leave = () => page.run(`(() => {
        const event = { type: "beforeunload", defaultPrevented: false, returnValue: undefined, preventDefault() { this.defaultPrevented = true; } };
        confirmLeavingWithResults(event);
        return { asked: event.defaultPrevented, returnValue: event.returnValue ?? null };
      })()`);
      const home = leave();
      await loadDataset(page, fixtures.dataset);
      const noResults = leave();
      page.fetchHandler = (request) => ({ status: 200, body: {
        analysis: { summary_table: [{ Group: "Overall", N: 12 }], risk_table: { rows: [], columns: [] }, pairwise_table: [], cohort: { n: 12 } },
        figure: { data: [{ x: [0, 1], y: [1, 0.5] }], layout: {} },
        request_config: request.json(),
      } });
      page.run("refs.runKmButton.click()");
      await page.settle();
      return { home, noResults, withResults: leave(), km: page.run("Boolean(state.km)") };
    """, dataset=_categorical_dataset())

    assert result["home"] == {"asked": False, "returnValue": None}
    assert result["noResults"] == {"asked": False, "returnValue": None}
    assert result["km"] is True
    assert result["withResults"] == {"asked": True, "returnValue": ""}


# ── Column lists ───────────────────────────────────────────────


def test_select_all_with_a_search_filter_adds_to_the_selection(tmp_path: Path) -> None:
    """#5: "Select all" while searching replaced the selection with the matching items."""
    dataset = _profile("genes", {
        "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "event": [0, 1] * 6,
        "age": [50, 60, 70, 80, 55, 65, 75, 85, 52, 62, 72, 82],
        "sex": ["M", "F"] * 6,
        "gene_a": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 1.2, 2.3, 3.4],
        "gene_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    })
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.run("setCheckedValues(refs.covariateChecklist, ['age', 'sex']); setCheckedValues(refs.cohortVariableChecklist, ['age'])");
      page.change("#covariateSearchInput", "gene");
      page.run("refs.selectAllCoxCovariatesButton.click()");
      page.change("#cohortVariableSearchInput", "gene");
      page.run("refs.selectAllCohortVariablesButton.click()");
      return { covariates: page.run("selectedCheckboxValues(refs.covariateChecklist)"), table: page.run("selectedCheckboxValues(refs.cohortVariableChecklist)") };
    """, dataset=dataset)

    assert result == {"covariates": ["age", "sex", "gene_a", "gene_b"], "table": ["age", "gene_a", "gene_b"]}


def test_a_cox_covariate_ticked_later_is_categorical_by_the_ml_rule(tmp_path: Path) -> None:
    """#12: "stage" coded 1-4 was categorical only when it happened to be among the first covariates."""
    dataset = _profile("order", {
        "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "event": [0, 1] * 6,
        "age": [50, 60, 70, 80, 55, 65, 75, 85, 52, 62, 72, 82],
        "bmi": [20, 25, 30, 35, 21, 22, 23, 24, 26, 27, 28, 29],
        "x1": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 1.2, 2.3, 3.4],
        "x2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "stage": [1, 2, 3, 4] * 3,
    })
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const load = page.run("({ covariates: currentCoxSelections().covariates, ml: selectedCheckboxValues(refs.modelCategoricalChecklist) })");
      page.run("refs.covariateChecklist.querySelector('input[value=\"stage\"]').checked = true");
      page.change("#covariateChecklist input[value='stage']");
      return { load, cox: page.run("currentCoxSelections().categoricalCovariates") };
    """, dataset=dataset)

    assert result["load"] == {"covariates": ["age", "bmi", "x1", "x2"], "ml": ["stage"]}
    assert result["cox"] == ["stage"]


def test_bulk_list_buttons_wait_for_their_run(tmp_path: Path) -> None:
    """#25: the Cox strata and Table 1 list buttons stayed enabled while their analysis ran."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const states = () => page.run(`[refs.selectAllCoxStrataButton, refs.clearCoxStrataButton,
        refs.selectAllCohortVariablesButton, refs.clearCohortVariablesButton].map((button) => button.disabled)`);
      page.run("setScopeBusy('cox', true, refs.runCoxButton); setScopeBusy('tables', true, refs.runCohortTableButton)");
      const busy = states();
      page.run("setScopeBusy('cox', false, refs.runCoxButton); setScopeBusy('tables', false, refs.runCohortTableButton)");
      return { busy, idle: states() };
    """, dataset=_categorical_dataset())

    assert result == {"busy": [True, True, True, True], "idle": [False, False, False, False]}


# ── Workspace chrome ───────────────────────────────────────────


# ── Markup and styles ──────────────────────────────────────────


class _LabelScanner(HTMLParser):
    """Labels that contain a button, with their for= target and the ids of the controls inside them."""

    def __init__(self) -> None:
        super().__init__()
        self.open: list[dict] = []
        self.labels: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        values = dict(attrs)
        if tag == "label":
            self.open.append({"for": values.get("for"), "button": False, "controls": []})
        elif self.open and tag == "button":
            self.open[-1]["button"] = True
        elif self.open and tag in {"input", "select", "textarea"}:
            self.open[-1]["controls"].append(values.get("id"))

    def handle_endtag(self, tag: str) -> None:
        if tag == "label" and self.open:
            label = self.open.pop()
            if label["button"]:
                self.labels.append(label)


def test_a_label_with_a_help_button_names_its_control() -> None:
    """#14: without for=, a label names its first labelable descendant, the "?" button, not the field."""
    scanner = _LabelScanner()
    scanner.feed(_TEMPLATE.read_text(encoding="utf-8"))

    assert len(scanner.labels) == 11
    for label in scanner.labels:
        assert label["for"] and label["for"] in label["controls"], label


def test_the_collapsed_derive_panel_is_out_of_the_tab_order() -> None:
    """#24: the collapsed panel stays laid out (display: flex) for its transition, so it needs visibility: hidden."""
    styles = (_STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    block = styles[styles.index(".derive-panel.hidden {"):]
    block = block[:block.index("}")]

    assert "visibility: hidden;" in block
    base = styles[styles.index(".derive-panel {"):]
    assert "visibility var(--duration-slow)" in base[:base.index("}")]


# ── Design check page ──────────────────────────────────────────


def _run_design_page(tmp_path: Path, script: str, **fixtures) -> object:
    """Like _run_page, for design_check.html and its script."""
    assert _SCRIPT_LOOP in _HARNESS_JS
    harness = tmp_path / "design_harness.js"
    harness.write_text(_HARNESS_JS.replace(_SCRIPT_LOOP, 'for (const file of ["design_check.js"]) {'), encoding="utf-8")
    data = tmp_path / "design_fixtures.json"
    data.write_text(
        json.dumps({"staticDir": str(_STATIC_DIR), "template": str(_DESIGN_TEMPLATE), "script": script, **fixtures}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(harness), str(data)], capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout.rsplit("@@RESULT@@", 1)[1])


_DESIGN_HELPERS = r"""
  const submit = async () => {
    page.document.getElementById("designForm").dispatchEvent({ type: "submit", bubbles: false, target: null,
      defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} });
    await page.settle();
    return {
      error: page.document.getElementById("designError").textContent,
      resultHidden: page.document.getElementById("designResult").classList.contains("hidden"),
      requests: page.requests.filter((request) => request.url.endsWith("/api/design-audit")).length,
    };
  };
  const audit = { expected_optimism: { value: 0.05, range: [0.03, 0.07] }, expected_regret: { value: 0.01, range: [0, 0.02] },
    flags: [{ severity: "high", message: "No cohort was kept out of the choice.", remedy: "Seal one." }],
    placement: { selection_cohorts: 1, median_cohort_size: 200, size_unit: "patients", candidate_models: 101, features: "genes only" },
    map: { version: "test" } };
"""


def test_a_failed_design_recheck_hides_the_previous_result(tmp_path: Path) -> None:
    """#16: the result of the previous design stayed on screen under the error of the new one."""
    result = _run_design_page(tmp_path, _DESIGN_HELPERS + r"""
      page.fetchHandler = () => ({ status: 200, body: audit });
      const first = await submit();
      page.fetchHandler = () => ({ status: 422, body: { detail: "candidate_models must be at least 2." } });
      return { first, second: await submit() };
    """)

    assert result["first"] == {"error": "", "resultHidden": False, "requests": 1}
    assert result["second"] == {"error": "candidate_models must be at least 2.", "resultHidden": True, "requests": 2}


def test_design_check_names_the_failure_that_happened(tmp_path: Path) -> None:
    """#27: a display error was reported as "the server did not answer"."""
    result = _run_design_page(tmp_path, _DESIGN_HELPERS + r"""
      page.fetchHandler = () => Promise.reject(new TypeError("Failed to fetch"));
      const offline = await submit();
      page.fetchHandler = () => ({ status: 200, body: { ...audit, flags: [null] } });
      return { offline, broken: await submit() };
    """)

    assert result["offline"]["error"] == "The local SurvStudio server did not answer."
    assert result["broken"]["error"].startswith("The design was checked, but the result could not be shown: ")
    assert result["broken"]["resultHidden"] is True


def test_design_check_flags_incomplete_cohort_rows_instead_of_dropping_them(tmp_path: Path) -> None:
    """#27: an unnamed row whose patient count could not be read silently left the design."""
    result = _run_design_page(tmp_path, _DESIGN_HELPERS + r"""
      page.fetchHandler = () => ({ status: 200, body: audit });
      const addRow = () => {
        page.document.querySelector('[data-add-cohort="selectionCohorts"]').click();
        const rows = page.document.querySelectorAll("#selectionCohorts .cohort-row");
        return rows[rows.length - 1];
      };
      const unreadable = addRow().querySelector('[data-cohort-field="n"]');
      unreadable.validity = { badInput: true };
      const badNumber = await submit();
      unreadable.closest(".cohort-row").remove();
      addRow().querySelector('[data-cohort-field="name"]').value = "GSE72094";
      const missingCount = await submit();
      return { badNumber, missingCount };
    """)

    assert result["badNumber"] == {
        "error": "The patients or events of the selection cohort in row 2 could not be read as numbers. Enter whole numbers, or remove that row.",
        "resultHidden": True,
        "requests": 0,
    }
    assert result["missingCount"]["error"] == 'Enter the number of patients for the selection cohort "GSE72094", or remove that row.'
    assert result["missingCount"]["requests"] == 0

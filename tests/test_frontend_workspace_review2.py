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


# ── Column lists ───────────────────────────────────────────────


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

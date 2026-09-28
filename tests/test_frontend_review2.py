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

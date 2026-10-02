"""Publication workflow checks: missing inputs and failed fits must stay visible."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_script(monkeypatch, name):
    directory = Path(__file__).resolve().parents[1] / "paper" / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("publication_" + name, directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_figures_report_missing_inputs_but_refuse_stale_results(monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    module = load_script(monkeypatch, "figures")
    def missing():
        raise FileNotFoundError(2, "absent", "other_case.csv")
    monkeypatch.setattr(module, "figure_workflow", missing)
    module.main({"workflow"}, available_only=True)
    assert "pending: workflow" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        module.main({"workflow"})
    def stale():
        raise module.MixedResults(["changed after computation"])
    monkeypatch.setattr(module, "figure_workflow", stale)
    with pytest.raises(SystemExit, match="mixed, stale"):
        module.main({"workflow"}, available_only=True)


def test_failed_competitor_draws_remain_in_rate_bounds(monkeypatch, tmp_path):
    module = load_script(monkeypatch, "18_competitors_null")
    assert module.rate(pd.Series([True, False]), 4)["failure_bounds"] == [0.25, 0.75]
    assert module.rate(pd.Series([], dtype=bool), 4)["failure_bounds"] == [0, 1]
    with pytest.raises(ValueError):
        module.rate(pd.Series([True, False]), 1)
    (tmp_path / "settings.json").write_text(json.dumps({"replicates": 2, "mime_replicates": 2}))
    monkeypatch.setattr(module, "NULL", tmp_path)
    monkeypatch.setattr(module, "read_result", lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    failed = pd.DataFrame({"replicate": [0, 1], "p1_failed": [True, True]})
    summary = module.null_summary(failed, pd.DataFrame())
    assert summary["failure_reporting"]["P1"] == {"planned": 2, "completed": 0, "failed_or_missing_ids": [0, 1]}
    assert "P1" not in summary  # no success-conditioned rate is fabricated when every fit fails


@pytest.mark.parametrize("commit,code,valid", [
    ("3e0c4af1", "0123456789abcdef", True),
    ("unknown", "0123456789abcdef", False),
    ("3e0c4af1-dirty", "0123456789abcdef", False),
    ("3e0c4af1", None, False),
])
def test_figures_require_a_known_clean_source_and_analysis_hash(monkeypatch, tmp_path, commit, code, valid):
    module = load_script(monkeypatch, "common")
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    result = tmp_path / "result.json"
    result.write_text('{"value": 0.65}\n')
    (tmp_path / "stamps").mkdir()
    stamp = {"sha256": module.sha256_file(result), "survstudio": {"commit": commit}, "analysis_code": code}
    (tmp_path / "stamps/result.json.json").write_text(json.dumps(stamp))
    problems = module.result_problems([result.name])
    assert (not problems) == valid

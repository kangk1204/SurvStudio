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


def test_comparison_table_retains_missing_marker_and_gain_outcomes(monkeypatch):
    module = load_script(monkeypatch, "18_competitors_table")
    rows = module.experiment_2({"SurvStudio": {
        "fwer": 0.5, "replicates": 2, "planned_replicates": 4,
        "fwer_failure_bounds": [0.25, 0.75],
        "null_with_subsamples": {
            "replicates": 3, "planned_replicates": 4,
            "gain_interval_replicates": 2, "verdict_adds": 0.5,
            "gain_coverage": 0.5,
        },
    }})
    assert rows[0]["claim_rate_failure_bounds"] == [0.25, 0.75]
    assert rows[1]["replicates"] == 2  # only two gain intervals were available, despite three successful runs
    assert rows[1]["planned_replicates"] == 4
    assert rows[1]["claim_rate_failure_bounds"] == [0.25, 0.75]
    assert "not selection-procedure coverage" in rows[1]["notes"]


def test_figure_guard_reserves_space_for_the_legend(monkeypatch):
    pytest.importorskip("matplotlib")
    module = load_script(monkeypatch, "figures")
    fig, ax = module.plt.subplots(figsize=(5, 10))
    ax.plot([0, 1], [0, 1])
    try:
        _, _, problems = module.print_size(fig)
        assert any("high" in problem and "legend" in problem for problem in problems)
    finally:
        module.plt.close(fig)


def test_comparator_external_intervals_match_the_paper_bootstrap(monkeypatch):
    module = load_script(monkeypatch, "competitors")
    import common
    import survival_toolkit.marker_evaluation as markers
    # The API default is intentionally smaller than the paper's explicit setting.
    def validate(frame, recipe, marker_scaling="as_measured", n_bootstrap=200, random_seed=0):
        return {"draws": n_bootstrap, "seed": random_seed}
    monkeypatch.setattr(markers, "validate_locked_recipe", validate)
    monkeypatch.setattr(common, "validation_row", lambda value: value)
    patients = pd.DataFrame({name: [1] for name in ["patient_id", "os_months", "os_event", *common.COVARIATES]})
    result = module.external_gain({}, patients, [0.2])
    assert result["draws"] == common.BOOTSTRAP_DRAWS == 2000
    assert result["seed"] == 20260926


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

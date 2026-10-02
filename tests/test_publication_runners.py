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


def test_saved_figure_exports_are_deterministic_and_svg_has_no_dtd(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import hashlib
    import xml.etree.ElementTree as ET
    module = load_script(monkeypatch, "figures")
    monkeypatch.setattr(module, "FIGURES", tmp_path)
    hashes = []
    for _ in range(2):
        fig, ax = module.plt.subplots(figsize=(2, 1.2))
        ax.plot([0, 1], [0.6, 0.7])
        module.save(fig, "export_check")
        hashes.append({extension: hashlib.sha256((tmp_path / f"export_check.{extension}").read_bytes()).hexdigest()
                       for extension in ("png", "pdf", "svg")})
    assert hashes[0] == hashes[1]
    svg = (tmp_path / "export_check.svg").read_text()
    assert "<!DOCTYPE" not in svg and "<!ENTITY" not in svg
    assert ET.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"
    assert json.loads((tmp_path / "export_check.provenance.json").read_text())["outputs"] == hashes[-1]


def test_rendering_archive_does_not_inherit_an_unrelated_parent_commit(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import subprocess
    module = load_script(monkeypatch, "figures")
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init", str(parent)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(parent), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--allow-empty", "-m", "Unrelated parent"], check=True, capture_output=True)
    archive = parent / "extracted-package"
    archive.mkdir()
    assert module.rendering_source(archive) == {"rendering_commit": "unknown", "rendering_git_head": "unknown"}
    parent_head = subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip()
    assert module.rendering_source(parent)["rendering_git_head"] == parent_head
    plain = tmp_path / "plain-archive"
    plain.mkdir()
    assert module.rendering_source(plain)["rendering_git_head"] == "unknown"


def test_numerical_and_shell_sources_reject_an_unrelated_parent_checkout(monkeypatch, tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    module = load_script(monkeypatch, "common")
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init", str(parent)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(parent), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--allow-empty", "-m", "Unrelated source"], check=True, capture_output=True)
    archive = parent / "archive"
    scripts = archive / "paper/scripts"
    scripts.mkdir(parents=True)
    assert module.git_commit(archive) == "unknown"
    assert module.git_commit(parent) != "unknown"
    if os.name == "nt" or not shutil.which("bash"):
        pytest.skip("POSIX shell checks require POSIX bash; numerical source checks already completed")
    directory = Path(__file__).resolve().parents[1] / "paper"
    for name in ("run_step.sh", "run_all.sh"):
        shutil.copy2(directory / name, archive / "paper" / name)
    (scripts / "zz_probe.py").write_text("import os\nprint('SOURCE=' + os.environ['SURVSTUDIO_COMMIT'])\n")
    environment = {**os.environ, "SURVSTUDIO_SRC": str(archive), "ANALYSIS_PYTHON": sys.executable}
    for name, arguments in (("run_step.sh", ["zz_probe.py"]), ("run_all.sh", ["zz"])):
        run = subprocess.run(["bash", str(archive / "paper" / name), *arguments], env=environment,
                             capture_output=True, text=True, check=True)
        assert "SOURCE=unknown" in run.stdout and "cannot be tied to a commit" in run.stderr
        valid = subprocess.run(["bash", str(archive / "paper" / name), *arguments],
                               env={**environment, "SURVSTUDIO_SRC": str(parent)},
                               capture_output=True, text=True, check=True)
        assert "SOURCE=" + module.git_commit(parent) in valid.stdout
        assert "cannot be tied to a commit" not in valid.stderr


def _tier_logistic_data():
    import numpy as np
    rng = np.random.default_rng(731)
    return pd.DataFrame({"abs_z": rng.normal(2, 0.6, 128), "cohorts": rng.integers(2, 8, 128),
                         "mean_expression": rng.normal(4, 0.8, 128),
                         "tier": ["robust"] * 8 + ["not supported"] * 120,
                         "replicated": [True] * 8 + list(rng.random(120) < 0.35)})


def test_quasi_separated_tier_regression_has_no_inferential_verdict(monkeypatch):
    pytest.importorskip("statsmodels")
    module = load_script(monkeypatch, "15b_tier_replication_checks")
    result = module.logistic(_tier_logistic_data())
    assert not result["inference_available"]
    assert result["coefficients"] == {} and result["likelihood_ratio_p"] is None
    assert result["tiers_add_nothing_beyond_z"] is None
    assert "No logistic tier inference" in result["statement"]
    json.dumps(result, allow_nan=False)


def test_estimable_tier_regression_keeps_descriptive_dependence_limits(monkeypatch):
    pytest.importorskip("statsmodels")
    module = load_script(monkeypatch, "15b_tier_replication_checks")
    table = _tier_logistic_data()
    table.loc[:7, "replicated"] = [True, False] * 4
    result = module.logistic(table)
    assert result["converged"] and result["inference_available"] and result["descriptive_only"]
    assert "gene dependence" in result["statement"].lower()
    assert "assume independent genes" in result["inference_note"]
    json.dumps(result, allow_nan=False)


def test_constant_replication_outcome_is_reported_without_a_fit(monkeypatch):
    pytest.importorskip("statsmodels")
    module = load_script(monkeypatch, "15b_tier_replication_checks")
    table = _tier_logistic_data()
    table["replicated"] = True
    result = module.logistic(table)
    assert not result["inference_available"] and result["status"] == "replication outcome lacks variation"
    json.dumps(result, allow_nan=False)


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


@pytest.mark.parametrize("successes,total", [(0, 200), (200, 200), (11, 400), (35, 50)])
def test_binary_monte_carlo_intervals_match_independent_wilson_reference(monkeypatch, successes, total):
    pytest.importorskip("matplotlib")
    proportion = pytest.importorskip("statsmodels.stats.proportion")
    module = load_script(monkeypatch, "figures")
    bounds = module.binary_rate_interval(successes/total, total)
    reference = proportion.proportion_confint(successes, total, method="wilson")
    assert bounds == pytest.approx(reference, abs=1e-14)
    assert bounds[1] > bounds[0]  # 0/200 and 200/200 must not look certain


def test_split_claim_interval_counts_replicates_and_keeps_boundary_uncertainty(monkeypatch):
    pytest.importorskip("matplotlib")
    import math
    module = load_script(monkeypatch, "figures")
    row = {"approach": "P1 Mime", "design": "3 selection + 4 sealed", "claim_rate": 1.0,
           "replicates": 50, "cases": 1750, "claim_rate_mcse": 0.0}
    interval = module.comparison_rate_interval(row)
    assert interval["independent_replicates"] == 50
    assert interval["lower"] < 0.85 and interval["upper"] == 1.0
    # Evaluate the two-sided probability bound independently of the interval formula.
    radius = 1-interval["lower"]
    assert 2*math.exp(-2*50*radius*radius) == pytest.approx(0.05)
    binary = module.comparison_rate_interval({**row, "design": "all seven", "cases": 50})
    assert binary["lower"] > interval["lower"]  # different units warrant different methods


@pytest.mark.parametrize("rate,total", [(0.125, 3), (1.1, 200), (float("nan"), 200), (0.5, 0), (0.5, 2.5)])
def test_binary_monte_carlo_intervals_refuse_invalid_denominators(monkeypatch, rate, total):
    pytest.importorskip("matplotlib")
    module = load_script(monkeypatch, "figures")
    with pytest.raises(ValueError):
        module.binary_rate_interval(rate, total)

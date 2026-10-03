"""Regression cases for the post-confirmation diagnostic investigation."""
import numpy as np
import pytest
import importlib.util
from pathlib import Path
from scipy import stats

from survival_toolkit.marker_screen import fit_cox
from survival_toolkit.marker_diagnostics import METHOD_VERSION
from survival_toolkit.marker_qualification import qualification


def test_finite_fit_with_cancelling_large_coefficients_is_not_separation():
    rng = np.random.default_rng(10482)
    n = 400
    z = rng.normal(size=n)
    perturbation = rng.normal(size=n)
    design = np.column_stack([z, z + 0.01 * perturbation])
    time = rng.exponential(size=n) / np.exp(0.8 * z + 0.4 * perturbation)
    event = np.ones(n, dtype=int)
    original = fit_cox(time, event, design)
    # This is a finite maximum with coefficient cancellation, not monotone likelihood.
    assert original.converged
    assert np.max(np.abs(original.beta) * design.std(axis=0)) > 10
    assert not original.separated.any()
    reparameterized = fit_cox(time, event, np.column_stack([z, perturbation]))
    assert reparameterized.converged and not reparameterized.separated.any()
    assert design @ original.beta == pytest.approx(
        np.column_stack([z, perturbation]) @ reparameterized.beta, abs=1e-7)
    assert original.loglik == pytest.approx(reparameterized.loglik, abs=1e-8)


def test_corrected_diagnostics_have_separate_audited_qualification():
    assert METHOD_VERSION == "marker-inference/2"
    for basis in ("linear", "restricted_cubic_spline"):
        gate = qualification(basis, METHOD_VERSION)
        assert gate["status"] == ("passed_supported_conditions_only" if basis == "linear" else "failed_exploratory_only")
        assert gate["evaluated_method_version"] == "marker-inference/2"
        assert gate["qualification_dimensions"] == {"n": 180, "markers": 30, "permutations": 999, "datasets_per_condition": 5000}
        assert "not universal" in gate["claim_boundary"]
        assert qualification(basis, "marker-inference/1")["status"] == "failed_exploratory_only"


def test_old_v3_recipe_keeps_predictions_and_original_inference_version():
    import copy
    from test_guarded_marker_inference import cohort, evaluate
    from survival_toolkit.marker_evaluation import validate_locked_recipe, recipe_hash
    from survival_toolkit.marker_qualification import qualify_external_result
    recipe = evaluate(cohort())["locked_recipe"]
    current = validate_locked_recipe(cohort(5), recipe, n_bootstrap=0)
    original = copy.deepcopy(recipe)
    original["inference"]["method_version"] = "marker-inference/1"
    original["recipe_hash"] = recipe_hash(original)
    outside = validate_locked_recipe(cohort(5), original, n_bootstrap=0)
    assert outside["metrics"] == current["metrics"]
    assert outside["inference"]["method_version"] == "marker-inference/1"
    gated = qualify_external_result(outside, original)
    assert gated["inference"]["engineering_qualification"]["status"] == "failed_exploratory_only"
    assert not gated["inference"]["allowed"]


def helper():
    path = Path(__file__).parents[1] / "validation/diagnostic_followup/development.py"
    spec = importlib.util.spec_from_file_location("diagnostic_development", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_is_explicitly_approximate_and_preserves_clinical_failures():
    module = helper()
    original = {"threshold": .01, "allowed": False, "reasons": ["clinical_diagnostic_failed: PH", "residual_misspecification: x/variance"],
                "residual_tests": [{"marker": "x", "diagnostic": "variance", "status": "calculated",
                                    "statistic": 14., "df": 4, "df_residual": 175, "p_value": .001}]}
    result = module.f_sensitivity(original)
    assert not result["allowed"]
    assert "clinical_diagnostic_failed: PH" in result["reasons"]
    assert result["residual_tests"][0]["p_value"] == pytest.approx(stats.f.sf(3.5, 4, 175))
    assert original["residual_tests"][0]["p_value"] == .001
    assert result["method_version"] == "candidate-HC3-F/1"


def test_old_separation_reconstruction_restores_process_state_after_exception():
    module = helper()
    current = module.screen_module._runaway_coefficients
    with pytest.raises(RuntimeError):
        with module.original_separation():
            assert module.screen_module._runaway_coefficients is not current
            raise RuntimeError("synthetic diagnostic failure")
    assert module.screen_module._runaway_coefficients is current


def test_development_reuses_same_raw_calculation_across_diagnostic_variants():
    module = helper()
    protocol = {"n": 180, "p": 8, "permutations": 9, "development_seed": 2026100311}
    rows = module.paired("partial_strong", 0, protocol)
    assert len(rows) == 7 and not any(r["failure"] for r in rows)
    for suffix in ("linear", "spline"):
        selected = [r for r in rows if r["method"].endswith(suffix) and "raw_null_rejected" in r]
        assert len({r["raw_null_rejected"] for r in selected}) == 1
        assert len({r["raw_true_rejected"] for r in selected}) == 1


def test_new_confirmation_preserves_scenarios_and_gates_but_uses_unused_seeds():
    import json
    root = Path(__file__).parents[1]
    previous = json.loads((root / "validation/guarded_inference/protocol.json").read_text())
    current = json.loads((root / "validation/diagnostic_followup/confirmation_protocol.json").read_text())
    for field in ("conditions", "methods", "main", "extensions", "large_marker", "supported", "engineering_gates", "diagnostics"):
        assert current[field] == previous[field]
    assert current["method_version"] == METHOD_VERSION
    old_seeds = {previous[f"{stage}_seed"] for stage in ("development", "main", "extension")}
    new_seeds = {current[f"{stage}_seed"] for stage in ("development", "main", "extension")}
    assert len(new_seeds) == 3 and not new_seeds & old_seeds


def test_confirmation_refuses_unsealed_or_changed_sources(tmp_path):
    import json
    import sys
    path = Path(__file__).parents[1] / "validation/diagnostic_followup/confirmation.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("new_confirmation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "unsealed.json"
    manifest.write_text(json.dumps({"source_hashes": module.hashes(), "reference_verification": {"passed": True}}))
    with pytest.raises(ValueError, match="review was not sealed"):
        module.verify_freeze(manifest)
    manifest.write_text(json.dumps({"source_hashes": {}}))
    with pytest.raises(ValueError, match="sources changed"):
        module.verify_freeze(manifest)


def test_confirmation_exception_retains_array_context(monkeypatch):
    import sys
    path = Path(__file__).parents[1] / "validation/diagnostic_followup/confirmation.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("context_confirmation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def failing_calculation():
        block = np.zeros((3, 64))
        row = np.array([0.1])
        return block[row]

    monkeypatch.setattr(module, "original_calculation", failing_calculation)
    with pytest.raises(IndexError) as info:
        module.calculation_with_context()
    note = info.value.__notes__[0]
    assert 'float64' in note and 'failing_calculation' in note and 'shape' in note

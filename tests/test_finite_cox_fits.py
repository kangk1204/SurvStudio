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


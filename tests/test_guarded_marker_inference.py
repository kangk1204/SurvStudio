from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import survival_toolkit.marker_evaluation as engine
from survival_toolkit.clinical_basis import ClinicalBasisError, fit_clinical_encoder, rcs_basis, transform_clinical_encoder
from survival_toolkit.errors import UserInputError
from survival_toolkit.marker_diagnostics import hc3_wald, holm
from survival_toolkit.plots import marker_evidence_funnel
from survival_toolkit.reporting import marker_results_paragraph


def cohort(seed=33, n=300, nonlinear=False):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = (z ** 2 if nonlinear else 0.8 * z)[:, None] + rng.normal(size=(n, 3))
    event_time = rng.exponential(size=n) / (0.06 * np.exp(0.8 * z))
    censor = rng.exponential(size=n) / 0.04
    return pd.DataFrame({"time": np.minimum(event_time, censor), "event": (event_time <= censor).astype(int),
                         "z": z, **{f"m{i}": x[:, i] for i in range(3)}})


def evaluate(frame, basis="linear", **overrides):
    options = dict(n_permutations=19, n_resamples=2, shortlist_size=3, random_seed=42, clinical_basis=basis)
    options.update(overrides)
    return engine.evaluate_markers(frame, time_column="time", event_column="event", clinical_columns=["z"],
                                   marker_columns=["m0", "m1", "m2"], settings=engine.MarkerSettings(**options))


def test_explicit_rcs_formula_and_linear_tails():
    knots = [-2., -1., 0., 1., 2.]
    x = np.array([-5., -3., 0., 2., 4., 6.])
    raw = rcs_basis(x, knots)
    expected = np.column_stack([x, *[
        (np.maximum(x - k, 0) ** 3 - np.maximum(x - 1, 0) ** 3 * (2 - k)
         + np.maximum(x - 2, 0) ** 3 * (1 - k)) / 16 for k in knots[:3]]])
    assert raw == pytest.approx(expected, abs=1e-12)
    assert raw[-1] - 2 * raw[-2] + raw[-3] == pytest.approx(np.zeros(4), abs=1e-12)
    with pytest.raises(ClinicalBasisError):
        rcs_basis(x, [-2, -1, -1, 1, 2])


def test_hc3_matches_independent_statsmodels_covariance():
    rng = np.random.default_rng(34)
    x = rng.normal(size=(200, 2))
    d = np.column_stack([np.ones(200), x])
    y = x[:, 0] + rng.normal(size=200) * np.exp(x[:, 1] / 2)
    ours = hc3_wald(y, d, 1)
    reference = sm.OLS(y, d).fit(cov_type="HC3").wald_test(np.eye(3)[1:], use_f=False, scalar=True)
    assert ours["statistic"] == pytest.approx(float(reference.statistic), abs=1e-9)
    assert ours["p_value"] == pytest.approx(float(reference.pvalue), abs=1e-12)
    assert holm([0.001, None, 0.02]) == pytest.approx([0.003, None, 0.04])


def test_training_knots_encoding_and_prediction_do_not_use_held_out_rows():
    frame = cohort()
    frame["grade"] = np.where(np.arange(len(frame)) < 200, "A", "B")
    prepared = engine.prepare_marker_cohort(frame, time_column="time", event_column="event", marker_columns=["m0"],
                                            clinical_columns=["z", "grade"], categorical_clinical=["grade"],
                                            clinical_basis="restricted_cubic_spline")
    train = np.arange(200)
    design, encoder, names = engine._training_clinical(prepared, train)
    knots = encoder["spline_specifications"]["z"]["knots"]
    assert knots == pytest.approx(np.quantile(frame.z[:200], [0.05, 0.275, 0.5, 0.725, 0.95]))
    assert encoder["categorical_mappings"]["grade"]["all_levels"] == ["A"]
    changed = prepared.clinical_frame.copy()
    changed.loc[200:, "z"] = 10000
    changed.loc[200:, "grade"] = "C"
    again, frozen, again_names = engine._training_clinical(prepared._replace(clinical_frame=changed), train)
    assert frozen == encoder and names == again_names
    assert again == pytest.approx(design)
    outside = transform_clinical_encoder(changed.iloc[200:], encoder)
    assert np.isfinite(outside).all()
    assert encoder["spline_specifications"]["z"]["knots"] == knots


def test_binary_covariate_is_not_expanded_after_median_imputation():
    frame = pd.DataFrame({"binary": [0., 0., 1., 1., np.nan]})
    encoder = fit_clinical_encoder(frame, ["binary"], basis="restricted_cubic_spline")
    assert encoder["numeric_kinds"]["binary"] == "binary_or_constant"
    assert encoder["spline_specifications"] == {}
    assert transform_clinical_encoder(frame, encoder)[-1, 0] == 0.5


def test_withheld_state_nulls_and_exploratory_values_propagate():
    result = evaluate(cohort(n=500, nonlinear=True))
    assert result["inference"]["status"] == "withheld"
    assert any("nonlinear_mean" in reason for reason in result["inference"]["reasons"])
    for row in result["marker_table"]:
        assert row["tier"] == "inference withheld"
        assert all(row["added_value"][k] is None for k in ("p_value", "q_bh", "p_fwer", "q_perm"))
        assert row["exploratory"]["added_value"]["p_value"] is not None
    stages = marker_evidence_funnel(result)
    assert all(stage["count"] is None for stage in stages if stage["label"] not in {"Supplied", "Tested"})
    assert "withheld" in marker_results_paragraph(result)
    recipe = result["locked_recipe"]
    assert recipe["recipe_version"] == 3 and recipe["inference"]["status"] == "withheld"
    external = engine.validate_locked_recipe(cohort(44), recipe, n_bootstrap=0)
    assert external["inference"]["status"] == "withheld"
    assert np.isfinite(external["metrics"]["c_index"])
    assert all(row["replication_p_holm"] is None and not row["replicated"] for row in external["markers"])


def test_duplicate_knots_withhold_without_linear_fallback():
    frame = cohort()
    frame["z"] = np.r_[np.zeros(280), np.ones(10), np.full(10, 2.)]
    result = evaluate(frame, "restricted_cubic_spline")
    assert result["primary_lens"] == "added_value" and result["inference"]["status"] == "withheld"
    assert result["clinical_basis"] == "restricted_cubic_spline"
    assert result["locked_recipe"] is None
    assert all(row["added_value"]["chi2"] is None for row in result["marker_table"])
    assert all(row["marginal"]["chi2"] is not None for row in result["marker_table"])


def test_required_diagnostic_failure_withholds(monkeypatch):
    import survival_toolkit.marker_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, "_cox_grambsch_therneau_test", lambda *args: {
        "p_value": None, "statistic": None, "df": None, "term_p_values": [None], "term_statistics": [None]})
    result = evaluate(cohort())
    assert result["inference"]["status"] == "withheld"
    assert any("diagnostic_failed" in reason for reason in result["inference"]["reasons"])


def test_missing_clinical_values_use_frozen_training_median():
    frame = cohort()
    frame.loc[:9, "z"] = np.nan
    recipe = evaluate(frame)["locked_recipe"]
    assert recipe["development"]["n"] == 300
    external = cohort(2)
    external.loc[:19, "z"] = np.nan
    filled = external.fillna({"z": recipe["clinical"]["encoder"]["numeric_impute_values"]["z"]})
    a = engine.validate_locked_recipe(external, recipe, n_bootstrap=0)
    b = engine.validate_locked_recipe(filled, recipe, n_bootstrap=0)
    assert a["cohort"]["n"] == 300 and a["metrics"] == b["metrics"]


def test_v3_tampering_and_legacy_prediction_compatibility():
    recipe = evaluate(cohort(), "restricted_cubic_spline")["locked_recipe"]
    changed = copy.deepcopy(recipe)
    changed["clinical"]["encoder"]["spline_specifications"]["z"]["knots"][0] -= 1
    with pytest.raises(UserInputError, match="hash"):
        engine.validate_locked_recipe(cohort(4), changed, n_bootstrap=0)
    changed["recipe_hash"] = engine.recipe_hash(changed)
    changed["clinical"]["encoder"]["spline_specifications"]["z"]["knots"][1] = changed["clinical"]["encoder"]["spline_specifications"]["z"]["knots"][0]
    changed["recipe_hash"] = engine.recipe_hash(changed)
    with pytest.raises(UserInputError, match="knots"):
        engine.validate_locked_recipe(cohort(4), changed, n_bootstrap=0)
    linear = evaluate(cohort())["locked_recipe"]
    baseline = engine.validate_locked_recipe(cohort(4), linear, n_bootstrap=0)["metrics"]
    for version in (1, 2):
        legacy = copy.deepcopy(linear)
        legacy["recipe_version"] = version
        legacy.pop("inference")
        legacy["clinical"]["encoder"] = legacy["clinical"]["encoder"]["base_encoder"]
        if version == 1:
            b = legacy["model"]["baseline"]
            b["survival"] = np.exp(-np.exp(np.array(b.pop("log_cumulative_hazard")) - b.pop("lp_center"))).tolist()
        legacy["recipe_hash"] = engine._legacy_recipe_hash(legacy) if version == 1 else engine.recipe_hash(legacy)
        report = engine.validate_locked_recipe(cohort(4), legacy, n_bootstrap=0)
        assert report["metrics"] == pytest.approx(baseline)
        assert report["inference"]["status"] == "not_assessed"

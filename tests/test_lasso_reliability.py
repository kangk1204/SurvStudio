"""Regressions for selected-path fitting and numerical failure propagation."""

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")
from sklearn.exceptions import ConvergenceWarning

from survival_toolkit import ml_models as ml
from survival_toolkit.errors import UserInputError
from survival_toolkit.sample_data import make_example_dataset


def _wide_design():
    rng = np.random.default_rng(521)
    x = rng.normal(size=(90, 180))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    failure = rng.exponential(np.exp(-0.8*x[:, 0] + 0.5*x[:, 1]))
    censor = rng.exponential(2, len(x))
    y = np.array(list(zip(failure <= censor, np.minimum(failure, censor) + 0.01)),
                 dtype=[("event", bool), ("time", float)])
    return x, y


def test_wide_selected_fit_preserves_reference_path_before_precision_refinement(monkeypatch):
    x, y = _wide_design()
    reference = ml._make_lasso_coxnet_model().set_params(
        alpha_min_ratio=.15, tol=ml._LASSO_REFIT_TOL, max_iter=ml._LASSO_REFIT_MAX_ITER).fit(x, y)
    index = 18
    alpha = float(reference.alphas_[index])
    original = ml.polish_lasso_coefficients
    initial = []
    def record_initial(*args, **kwargs):
        initial.append(np.asarray(args[2]).copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(ml, "polish_lasso_coefficients", record_initial)
    selected = ml._fit_selected_lasso(x, y, {"alpha": alpha, "alpha_path": reference.alphas_[:index+1].tolist()})
    assert selected.alphas_[-1] == alpha
    np.testing.assert_allclose(initial[0], reference.coef_[:, index], atol=1e-12, rtol=0)
    np.testing.assert_allclose(ml._coerce_coxnet_coef_vector(selected), reference.coef_[:, index], atol=1e-6, rtol=0)
    np.testing.assert_allclose(selected.predict(x), reference.predict(x, alpha=alpha), atol=1e-6, rtol=0)
    np.testing.assert_allclose(selected.predict_survival_function(x, return_array=True),
                               reference.predict_survival_function(x, alpha=alpha, return_array=True), atol=1e-6, rtol=0)
    assert selected._survstudio_kkt_residual <= 1e-12


def test_training_only_alpha_rule_and_prefix_are_preserved():
    df = make_example_dataset(seed=24, n_patients=150)
    features = ["age", "biomarker_score", "immune_index"]
    encoded, _, _ = ml._encode_train_test_features(df, df, features)
    meta = ml._select_lasso_alpha(df, encoded, features=features, time_column="os_months",
                                  event_column="os_event", random_state=11)
    scaled, _, _, _ = ml._standardize_encoded_matrices(encoded, encoded)
    reference = ml._make_lasso_coxnet_model().fit(scaled.to_numpy(), ml._prepare_sksurv_data(df, "os_months", "os_event"))
    idx = int(np.flatnonzero(reference.alphas_ == meta["alpha"])[0])
    assert meta["selection_rule"] == "inner_cv_max_mean_c_index"
    assert meta["inner_cv_folds"] == meta["inner_cv_scored_folds"] == 5
    np.testing.assert_array_equal(meta["alpha_path"], reference.alphas_[:idx+1])


def test_penalty_selection_does_not_compute_unused_survival_baselines(monkeypatch):
    df = make_example_dataset(seed=24, n_patients=120)
    features = ["age", "biomarker_score"]
    encoded, _, _ = ml._encode_train_test_features(df, df, features)
    original_fit = ml.CoxnetSurvivalAnalysis.fit
    seen = []
    def fit(model, *args, **kwargs):
        seen.append(model.fit_baseline_model)
        assert not model.fit_baseline_model
        return original_fit(model, *args, **kwargs)
    monkeypatch.setattr(ml.CoxnetSurvivalAnalysis, "fit", fit)
    result = ml._select_lasso_alpha(df, encoded, features=features, time_column="os_months",
                                    event_column="os_event", random_state=11)
    assert result["selection_mode"] == "inner_cv" and len(seen) == 6


def test_convergence_warning_rejects_the_fit_before_results(monkeypatch):
    model = ml._make_lasso_coxnet_model(alpha=0.1)
    def incomplete(*args):
        warnings.warn("iteration limit", ConvergenceWarning)
        return model
    monkeypatch.setattr(model, "fit", incomplete)
    x, y = _wide_design()
    with pytest.raises(ml._LassoNumericalError, match="did not converge"):
        ml._fit_checked_lasso(model, x, y)


def _fitted_stub(monkeypatch):
    model = ml._make_lasso_coxnet_model(alpha=0.1)
    model.coef_ = np.zeros((2, 1))
    model.offset_ = np.array([0.0])
    model.deviance_ratio_ = np.array([0.2])
    model.alphas_ = np.array([0.1])
    baseline = SimpleNamespace(x=np.array([1., 2.]), y=np.array([0.9, 0.8]))
    monkeypatch.setattr(model, "fit", lambda *args: model)
    monkeypatch.setattr(model, "predict", lambda x: np.zeros(len(x)))
    monkeypatch.setattr(model, "_get_baseline_model", lambda alpha: SimpleNamespace(baseline_survival_=baseline))
    return model, baseline


@pytest.mark.parametrize("attribute", ["coef_", "offset_", "deviance_ratio_", "alphas_"])
def test_nonfinite_fit_attributes_are_rejected(monkeypatch, attribute):
    model, _ = _fitted_stub(monkeypatch)
    getattr(model, attribute).flat[0] = np.nan
    with pytest.raises(ml._LassoNumericalError, match="non-finite fitted"):
        ml._fit_checked_lasso(model, np.zeros((4, 2)), np.empty(4))


@pytest.mark.parametrize("values", [[np.nan, .8], [np.inf, .8], [-.1, .8], [1.1, .8], [.5, .8]])
def test_invalid_baseline_survival_is_rejected(monkeypatch, values):
    model, baseline = _fitted_stub(monkeypatch)
    baseline.y = np.asarray(values)
    with pytest.raises(ml._LassoNumericalError, match="invalid baseline"):
        ml._fit_checked_lasso(model, np.zeros((4, 2)), np.empty(4))


@pytest.mark.parametrize("risk,message", [([np.nan], "non-finite"), ([np.inf], "non-finite"), ([1000.], "too large")])
def test_nonfinite_or_overflowing_risks_are_rejected(monkeypatch, risk, message):
    model, _ = _fitted_stub(monkeypatch)
    monkeypatch.setattr(model, "predict", lambda x: np.asarray(risk))
    with pytest.raises(ml._LassoNumericalError, match=message):
        ml._checked_lasso_risk(model, np.zeros((1, 2)))


def test_strong_negative_risk_underflow_is_not_a_fit_failure(monkeypatch):
    model, _ = _fitted_stub(monkeypatch)
    monkeypatch.setattr(model, "predict", lambda x: np.full(len(x), -1000.))
    np.testing.assert_array_equal(ml._checked_lasso_risk(model, np.zeros((2, 2))), [-1000., -1000.])


@pytest.mark.parametrize("failure", [ConvergenceWarning("iteration limit"), ArithmeticError("weights are too large"),
                                     ValueError("invalid solver state")])
def test_unstable_inner_fold_cannot_trigger_apparent_selection_fallback(monkeypatch, failure):
    df = make_example_dataset(seed=24, n_patients=120)
    features = ["age", "biomarker_score"]
    encoded, _, _ = ml._encode_train_test_features(df, df, features)
    original_fit = ml.CoxnetSurvivalAnalysis.fit
    calls = {"n": 0}
    def fit(model, x, y):
        calls["n"] += 1
        if calls["n"] == 2:  # full training path, then the first inner fold
            if isinstance(failure, Warning):
                warnings.warn(str(failure), type(failure))
            else:
                raise failure
        return original_fit(model, x, y)
    monkeypatch.setattr(ml.CoxnetSurvivalAnalysis, "fit", fit)
    with pytest.raises(ml._LassoNumericalError, match="did not converge|numerically valid"):
        ml._select_lasso_alpha(df, encoded, features=features, time_column="os_months", event_column="os_event", random_state=11)
    assert calls["n"] == 2


def test_selected_baseline_is_not_an_earlier_nearly_equal_alpha():
    from sksurv.functions import StepFunction

    model = ml._make_lasso_coxnet_model(alpha=.00000001)
    model.alphas_ = np.array([.00000002, .00000001])
    model.coef_ = np.zeros((2, 2)); model.offset_ = np.zeros(2)
    curves = [StepFunction(np.array([1., 2.]), np.array(values)) for values in [[.9, .8], [.7, .6]]]
    model._baseline_models = [SimpleNamespace(baseline_survival_=curve) for curve in curves]
    assert model._get_baseline_model(.00000001).baseline_survival_ is curves[0]
    assert ml._breslow_baseline_survival_function(model, .00000001) is curves[1]
    predictor = ml._sksurv_survival_predictor(model, pd.DataFrame(np.zeros((3, 2))), alpha=.00000001)
    np.testing.assert_allclose(predictor(np.array([1., 2.])), np.tile([.7, .6], (3, 1)))


def test_final_path_cannot_silently_stop_before_selected_alpha(monkeypatch):
    model, _ = _fitted_stub(monkeypatch)
    monkeypatch.setattr(ml, "_make_lasso_coxnet_model", lambda **kwargs: model)
    with pytest.raises(ml._LassoNumericalError, match="stopped before"):
        ml._fit_selected_lasso(np.zeros((4, 2)), np.empty(4), {"alpha": .05, "alpha_path": [.1, .05]})


def test_public_training_rejects_numerical_failure_and_comparison_excludes_it(monkeypatch):
    df = make_example_dataset(seed=24, n_patients=120)
    def incomplete(*args, **kwargs):
        warnings.warn("iteration limit", ConvergenceWarning)
    monkeypatch.setattr(ml.CoxnetSurvivalAnalysis, "fit", incomplete)
    with pytest.raises(UserInputError, match="did not converge"):
        ml.train_lasso_cox(df, "os_months", "os_event", ["age", "biomarker_score"])
    result = ml.compare_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], n_estimators=5)
    assert "LASSO-Cox" not in {x["model"] for x in result["comparison_table"]}
    assert any(x["model"] == "LASSO-Cox" and "did not converge" in x["error"] for x in result["errors"])
    assert "LASSO-Cox" not in result["test_predictions"]["risk"]


def test_api_rejects_unconverged_lasso_without_storing_an_artifact(monkeypatch):
    from fastapi.testclient import TestClient
    import survival_toolkit.app as app_module

    stored = app_module.store.create(make_example_dataset(seed=24, n_patients=120),
                                      filename="normal.csv", copy_dataframe=False)
    def incomplete(*args, **kwargs):
        warnings.warn("iteration limit", ConvergenceWarning)
    monkeypatch.setattr(ml.CoxnetSurvivalAnalysis, "fit", incomplete)
    artifacts = []
    monkeypatch.setattr(app_module, "_remember_ml_artifact", lambda *args: artifacts.append(args))
    with TestClient(app_module.app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/ml-model", json={"dataset_id": stored.dataset_id,
            "time_column": "os_months", "event_column": "os_event", "event_positive_value": 1,
            "features": ["age", "biomarker_score"], "model_type": "lasso_cox", "compute_shap": False})
    assert response.status_code == 400
    assert "did not converge" in response.json()["detail"]
    assert not artifacts


@pytest.mark.parametrize("x,events,times,gradient", [([1., 2., 1.], [True, True, False], [1., 2., 1.], 1/9),
                                                     ([1., 2., 4.], [True, True, False], [1., 1., 1.], 5/9)])
def test_kkt_check_matches_hand_calculated_breslow_ties(x, events, times, gradient):
    # At beta=0, t=1 has an event and censor together in the risk set.
    # Mean risk x=4/3 then2; event sum=3, so gradient=(4/3+2-3)/3=1/9.
    x = np.asarray(x)[:, None]
    y = np.array(list(zip(events, times)), dtype=[("event", bool), ("time", float)])
    assert ml._lasso_kkt_residual(x, y, np.array([0.]), .05) == pytest.approx(gradient-.05)
    assert ml._lasso_kkt_residual(x, y, np.array([0.]), 1.) == 0


def test_warning_free_but_inaccurate_selected_fit_is_rejected(monkeypatch):
    x, y = _wide_design()
    model = ml._make_lasso_coxnet_model(alpha=.1).fit(x, y)
    monkeypatch.setattr(ml, "_make_lasso_coxnet_model", lambda **kwargs: model)
    monkeypatch.setattr(ml, "_fit_checked_lasso", lambda *args: model)
    monkeypatch.setattr(ml, "_lasso_kkt_residual", lambda *args: .001)
    with pytest.raises(ml._LassoNumericalError, match="numerical convergence check"):
        ml._fit_selected_lasso(x, y, {"alpha": .1, "alpha_path": [.1]})


def test_kkt_rejects_underflowed_event_risk_sets():
    x = np.array([[0.], [-1000.]])
    y = np.array([(True, 1.), (True, 2.)], dtype=[("event", bool), ("time", float)])
    with pytest.raises(ml._LassoNumericalError, match="risk weights"):
        ml._lasso_kkt_residual(x, y, np.array([1.]), .1)

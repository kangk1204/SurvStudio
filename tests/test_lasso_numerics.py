"""Independent derivative, precision and baseline checks on synthetic data."""
import numpy as np
import pytest

from survival_toolkit import lasso_numerics as numerics


def _tied_design():
    x = np.random.default_rng(72).normal(size=(9, 4))
    times = np.array([0., 1., 1., 2., 2., 3., 4., 4., 5.])
    events = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
    return x, times, events


def _direct_loss(x, times, events, beta):
    lp = x @ beta
    result = -lp[events].sum()
    for time in np.unique(times[events]):
        risk = lp[times >= time]
        peak = risk.max()
        result += np.count_nonzero(events & (times == time)) * (peak + np.log(np.exp(risk-peak).sum()))
    return result / len(x)


def test_score_and_information_match_independent_finite_differences_with_ties():
    x, times, events = _tied_design()
    beta = np.array([.2, -.1, .3, .15])
    state = numerics._breslow_state(x, times, events, beta, np.arange(4))
    epsilon = 1e-5
    gradient = np.empty(4)
    hessian = np.empty((4, 4))
    for j in range(4):
        direction = np.eye(4)[j] * epsilon
        gradient[j] = (_direct_loss(x, times, events, beta+direction)
                       - _direct_loss(x, times, events, beta-direction)) / (2*epsilon)
        plus = numerics._breslow_state(x, times, events, beta+direction, np.arange(4))
        minus = numerics._breslow_state(x, times, events, beta-direction, np.arange(4))
        hessian[:, j] = (plus.gradient-minus.gradient) / (2*epsilon)
    assert state.loss == pytest.approx(_direct_loss(x, times, events, beta), abs=1e-13)
    np.testing.assert_allclose(state.gradient, gradient, atol=1e-9, rtol=0)
    analytic = np.column_stack([state.curvature(v) for v in np.eye(4)])
    np.testing.assert_allclose(analytic, hessian, atol=1e-9, rtol=0)


def test_matrix_free_and_dense_newton_steps_agree(monkeypatch):
    x, times, events = _tied_design()
    state = numerics._breslow_state(x, times, events, np.ones(4)*.1, np.arange(4))
    dense = state.newton_step(state.gradient)
    monkeypatch.setattr(numerics, "_DENSE_LIMIT", 1)
    np.testing.assert_allclose(state.newton_step(state.gradient), dense, atol=1e-9, rtol=0)


def _survival_design():
    rng = np.random.default_rng(749)
    x = rng.normal(size=(150, 5)) + np.arange(5)
    failure = rng.exponential(np.exp(-.8*x[:, 0]+.4*x[:, 1]))
    censor = rng.exponential(2, len(x))
    y = np.array(list(zip(failure<=censor, np.minimum(failure, censor)+.01)),
                 dtype=[("event", bool), ("time", float)])
    return x, y


def test_precision_refines_coefficients_without_mutating_design_or_input():
    pytest.importorskip("sksurv")
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    x, y = _survival_design()
    initial = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[.05], tol=1e-12, max_iter=100000).fit(x, y).coef_[:, -1]
    x_before, before = x.copy(), initial.copy()
    perturbed = initial.copy()
    perturbed[perturbed != 0] += 1e-5
    result, metadata = numerics.polish_lasso_coefficients(x, y, perturbed, .05)
    active = np.flatnonzero(result)
    state = numerics._breslow_state(x-x.mean(axis=0), y["time"], y["event"], result, active)
    residual = np.maximum(np.abs(state.gradient)-.05, 0)
    residual[active] = np.abs(state.gradient[active]+.05*np.sign(result[active]))
    assert residual.max() <= 1e-12
    assert metadata["iterations"] > 0
    assert np.array_equal(result == 0, initial == 0)
    assert np.array_equal(np.sign(result), np.sign(initial))
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(initial, before)


def test_valid_zero_fit_stays_zero_but_inactive_violation_is_rejected():
    x, times, events = _tied_design()
    y = np.array(list(zip(events, times)), dtype=[("event", bool), ("time", float)])
    result, meta = numerics.polish_lasso_coefficients(x, y, np.zeros(4), 10.)
    np.testing.assert_array_equal(result, np.zeros(4))
    assert meta["iterations"] == 0
    with pytest.raises(ValueError, match="different active set"):
        numerics.polish_lasso_coefficients(x, y, np.zeros(4), 1e-12)


def test_singular_selected_information_cannot_certify_a_coefficient_vector():
    y = np.array([(True, 1.), (False, 2.), (True, 3.)], dtype=[("event", bool), ("time", float)])
    with pytest.raises(np.linalg.LinAlgError):
        numerics.polish_lasso_coefficients(np.ones((3, 2)), y, np.ones(2), .1)


def test_non_descent_information_step_is_rejected(monkeypatch):
    x, y = _survival_design()
    monkeypatch.setattr(numerics._BreslowState, "newton_step", lambda state, gradient: -gradient)
    with pytest.raises(ValueError, match="descent step"):
        numerics.polish_lasso_coefficients(x, y, np.ones(5)*.1, .05)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 0.])
def test_invalid_penalty_is_rejected(bad):
    x, y = _survival_design()
    with pytest.raises(ValueError, match="invalid training fit"):
        numerics.polish_lasso_coefficients(x, y, np.zeros(5), bad)


def test_refined_model_offset_baseline_and_deviance_use_the_same_coefficients():
    pytest.importorskip("sksurv")
    from survival_toolkit import ml_models as ml
    x, y = _survival_design()
    model = ml._fit_selected_lasso(x, y, {"alpha": .05, "alpha_path": [.1, .05]})
    lp = model.predict(x)
    np.testing.assert_allclose(lp, (x-x.mean(axis=0)) @ model.coef_[:, -1], atol=1e-12, rtol=0)
    functions = model.predict_survival_function(x)
    for horizon in [1., 2.]:
        hazard = sum(np.count_nonzero(y["event"] & (y["time"] == t)) / np.exp(lp[y["time"] >= t]).sum()
                     for t in np.unique(y["time"][y["event"] & (y["time"] <= horizon)]))
        np.testing.assert_allclose([f(horizon) for f in functions], np.exp(-hazard*np.exp(lp)), atol=1e-10, rtol=0)
    null = np.log(np.arange(1, len(y)+1))[y["event"][np.argsort(-y["time"], kind="stable")]].sum()
    expected = 1-_direct_loss(x, y["time"], y["event"], model.coef_[:, -1])*len(y)/null
    assert model.deviance_ratio_[-1] == pytest.approx(expected, abs=1e-12)
    assert model._survstudio_kkt_residual <= 1e-12

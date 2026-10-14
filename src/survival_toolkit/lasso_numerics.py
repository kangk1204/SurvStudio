"""Training-only precision refinement of a selected Breslow L1 Cox fit."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg

from survival_toolkit.concurrency import raise_if_cancelled

GRADIENT_TOLERANCE = 1e-12
STEP_TOLERANCE = 1e-9
_MAX_STEPS = 12
_DENSE_LIMIT = 512


@dataclass
class _BreslowState:
    loss: float
    gradient: np.ndarray
    active_design: np.ndarray
    risk_means: np.ndarray
    row_mass: np.ndarray
    deaths: np.ndarray

    def curvature(self, vector: np.ndarray) -> np.ndarray:
        x, means = self.active_design, self.risk_means
        return (x.T @ (self.row_mass * (x @ vector))
                - means.T @ (self.deaths * (means @ vector))) / len(x)

    def newton_step(self, gradient: np.ndarray) -> np.ndarray:
        size = len(gradient)
        if size <= _DENSE_LIMIT:
            hessian = (self.active_design.T @ (self.row_mass[:, None] * self.active_design)
                       - self.risk_means.T @ (self.deaths[:, None] * self.risk_means)) / len(self.active_design)
            hessian = (hessian + hessian.T) / 2
            return cho_solve(cho_factor(hessian, check_finite=True), gradient)
        # Avoid a quadratic allocation for large selected models. The operator
        # is the same observed information as the dense calculation above.
        diagonal = ((self.active_design ** 2).T @ self.row_mass
                    - (self.risk_means ** 2).T @ self.deaths) / len(self.active_design)
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise ValueError("LASSO-Cox has undefined active curvature.")
        scale = float(np.max(np.abs(gradient)))
        if not scale:
            return np.zeros_like(gradient)
        def matvec(vector):
            raise_if_cancelled()
            return self.curvature(vector)
        operator = LinearOperator((size, size), matvec=matvec, dtype=float)
        preconditioner = LinearOperator((size, size), matvec=lambda v: v / diagonal, dtype=float)
        step, status = cg(operator, gradient / scale, M=preconditioner, rtol=1e-10,
                          atol=0, maxiter=max(100, min(5 * size, 2000)))
        if status != 0:
            raise ValueError("LASSO-Cox active curvature did not converge.")
        return step * scale


def _breslow_state(x, times, events, beta, active) -> _BreslowState:
    order = np.argsort(times, kind="stable")
    t, e, design = times[order], events[order], x[order]
    lp = design @ beta
    peak = float(lp.max())
    with np.errstate(under="ignore"):
        weights = np.exp(lp - peak)
    _, starts = np.unique(t, return_index=True)
    deaths = np.add.reduceat(e.astype(float), starts)
    starts, deaths = starts[deaths > 0], deaths[deaths > 0]
    denominators = np.cumsum(weights[::-1])[::-1][starts]
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise ValueError("LASSO-Cox event risk sets cannot be evaluated reliably.")
    gradient = np.empty(len(beta))
    for first in range(0, len(beta), 128):
        raise_if_cancelled()
        chunk = design[:, first:first+128]
        risk_totals = np.cumsum((chunk * weights[:, None])[::-1], axis=0)[::-1]
        gradient[first:first+128] = (np.sum(risk_totals[starts] * (deaths / denominators)[:, None], axis=0)
                                   - np.sum(chunk[e], axis=0)) / len(x)
    # Each patient's weight appears in all event risk sets up to that time.
    contributions = np.zeros(len(x))
    contributions[starts] = deaths / denominators
    row_mass = weights * np.cumsum(contributions)
    selected = design[:, active]
    totals = np.cumsum((selected * weights[:, None])[::-1], axis=0)[::-1]
    means = totals[starts] / denominators[:, None]
    loss = math.fsum((deaths * (np.log(denominators) + peak)).tolist() + (-lp[e]).tolist()) / len(x)
    if (not np.isfinite(loss) or not np.isfinite(gradient).all()
            or not np.isfinite(row_mass).all() or not np.isfinite(means).all()):
        raise ValueError("LASSO-Cox produced non-finite precision calculations.")
    return _BreslowState(loss, gradient, selected, means, row_mass, deaths)


def polish_lasso_coefficients(design, outcome, coefficients, alpha, penalty_factor=None):
    """Refine the current active orthant without re-selecting alpha or variables.

    Inactive KKT inequalities are checked over the complete training design.
    An orthant change, nonpositive curvature or unsuccessful refinement fails
    explicitly; it cannot silently substitute a different model. This numerical
    precision calculation does not certify prediction or biomarker inference.
    """
    x = np.asarray(design, dtype=float)
    beta = np.asarray(coefficients, dtype=float).copy()
    times, events = np.asarray(outcome["time"], dtype=float), np.asarray(outcome["event"], dtype=bool)
    factors = np.ones(len(beta)) if penalty_factor is None else np.asarray(penalty_factor, dtype=float)
    if (x.ndim != 2 or min(x.shape) < 1 or beta.shape != (x.shape[1],)
            or factors.shape != beta.shape or times.shape != (len(x),) or events.shape != times.shape
            or not events.any() or not np.isfinite(x).all() or not np.isfinite(beta).all()
            or not np.isfinite(times).all() or not np.isfinite(factors).all() or np.any(factors < 0)
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("LASSO-Cox cannot refine an invalid training fit.")
    x = x - x.mean(axis=0)
    penalties = float(alpha) * factors
    active = np.flatnonzero((beta != 0) | (factors == 0))
    signs = np.sign(beta[active])
    iterations = 0
    for iteration in range(_MAX_STEPS):
        raise_if_cancelled()
        state = _breslow_state(x, times, events, beta, active)
        residual = np.maximum(np.abs(state.gradient) - penalties, 0)
        adjusted = state.gradient[active] + penalties[active] * signs
        residual[active] = np.abs(adjusted)
        if not len(active):
            if residual.max() > GRADIENT_TOLERANCE:
                raise ValueError("LASSO-Cox precision check requires a different active set.")
            break
        delta = state.newton_step(adjusted)
        if not np.isfinite(delta).all():
            raise ValueError("LASSO-Cox produced a non-finite precision step.")
        if residual.max() <= GRADIENT_TOLERANCE and np.abs(delta).max() <= STEP_TOLERANCE:
            break
        descent = float(np.dot(adjusted, delta))
        if not np.isfinite(descent) or descent <= 0:
            raise ValueError("LASSO-Cox active curvature did not produce a descent step.")
        # A nearly converged coefficient vector should stay in the same orthant.
        # Backtracking controls the exact original penalized likelihood.
        objective = state.loss + float(np.dot(penalties, np.abs(beta)))
        for exponent in range(24):
            step = 0.5 ** exponent
            candidate = beta.copy()
            candidate[active] -= step * delta
            penalized = penalties[active] > 0
            if np.any(np.sign(candidate[active][penalized]) != signs[penalized]):
                continue
            trial = _breslow_state(x, times, events, candidate, active)
            target = trial.loss + float(np.dot(penalties, np.abs(candidate)))
            allowance = 32 * np.finfo(float).eps * max(1., abs(objective))
            if target <= objective - 1e-4 * step * descent + allowance:
                beta = candidate
                iterations = iteration + 1
                break
        else:
            raise ValueError("LASSO-Cox could not refine its active coefficients reliably.")
    else:
        raise ValueError("LASSO-Cox precision refinement did not converge.")
    final = _breslow_state(x, times, events, beta, active)
    residual = np.maximum(np.abs(final.gradient) - penalties, 0)
    residual[active] = np.abs(final.gradient[active] + penalties[active] * signs)
    if residual.max() > GRADIENT_TOLERANCE:
        raise ValueError("LASSO-Cox precision refinement did not satisfy the full training KKT check.")
    return beta, {"iterations": iterations, "gradient_tolerance": GRADIENT_TOLERANCE,
                  "kkt_max_residual": float(residual.max()), "partial_likelihood_loss": final.loss,
                  "penalized_objective": final.loss + float(np.dot(penalties, np.abs(beta)))}

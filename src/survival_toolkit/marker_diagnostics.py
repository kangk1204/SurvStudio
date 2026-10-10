"""Predeclared diagnostics and withholding policy for conditional marker inference.

Passing these finite diagnostics does not establish exchangeability, proportional
hazards, subset pivotality, or universal error control. Tests describe a working model.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

from survival_toolkit.analysis import _cox_grambsch_therneau_test, _efron_schoenfeld_residuals
from survival_toolkit.clinical_basis import fit_clinical_encoder, transform_clinical_encoder
from survival_toolkit.marker_screen import fit_cox, residualize

METHOD_VERSION = "marker-inference/1"
WITHHOLD_ALPHA = 0.01


def holm(values: Sequence[float | None]) -> list[float | None]:
    """Holm over the entire prespecified family, including unavailable tests."""
    p = np.asarray([1.0 if x is None else x for x in values], dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm needs probabilities or explicit unavailable tests.")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(p.size)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(p.size, 0, -1)))
    return [None if value is None else float(adjusted[i]) for i, value in enumerate(values)]


def hc3_wald(y: np.ndarray, design: np.ndarray, test_start: int) -> dict[str, Any]:
    """OLS Wald chi-square for the indicated coefficients, using HC3 covariance."""
    try:
        d = np.asarray(design, float)
        y = np.asarray(y, float)
        n, p = d.shape
        q = p - test_start
        if q == 0:
            return {"status": "not_applicable", "p_value": None, "df": 0}
        if n <= p + 2 or not np.isfinite(d).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(d) != p:
            raise ValueError("insufficient observations, non-finite values or deficient diagnostic rank")
        inverse = np.linalg.inv(d.T @ d)
        beta = inverse @ d.T @ y
        residual = y - d @ beta
        leverage = np.sum((d @ inverse) * d, axis=1)
        if np.any(leverage >= 1 - 1e-10):
            raise ValueError("diagnostic leverage is too close to one")
        covariance = inverse @ (d.T @ ((residual / (1 - leverage))[:, None] ** 2 * d)) @ inverse
        selected = covariance[test_start:, test_start:]
        if np.linalg.matrix_rank(selected) != q:
            raise ValueError("singular HC3 covariance")
        statistic = float(beta[test_start:] @ np.linalg.solve(selected, beta[test_start:]))
        if not np.isfinite(statistic) or statistic < 0:
            raise ValueError("non-finite HC3 Wald statistic")
        return {"status": "calculated", "statistic": statistic, "df": q,
                "p_value": float(stats.chi2.sf(statistic, q)), "method": "OLS HC3 Wald chi-square"}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"status": "failed", "p_value": None, "reason": str(exc), "method": "OLS HC3 Wald chi-square"}


def _orthogonal_columns(d: np.ndarray) -> np.ndarray:
    if not d.size:
        return np.zeros((d.shape[0], 0))
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    rank = int(np.sum(s > max(d.shape) * np.finfo(float).eps * s[0])) if s.size else 0
    return u[:, :rank]


def _usable(fit: Any) -> bool:
    return bool(fit.converged and np.isfinite(fit.beta).all() and np.isfinite(fit.covariance).all()
                and np.isfinite(fit.loglik) and not np.any(fit.separated))


def diagnose_markers(cohort: Any, block: np.ndarray, *, ties: str = "efron",
                     functional_form_encoder: dict[str, Any] | None = None,
                     frozen_transform: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"method_version": METHOD_VERSION, "threshold": WITHHOLD_ALPHA,
        "status": "assumption_dependent", "allowed": True, "reasons": [],
        "clinical_basis": cohort.clinical_basis, "clinical_tests": [], "residual_tests": [],
        "families": {"clinical": "global and per-term classic GT log-time PH, plus linear-versus-fixed-RCS LR",
                     "residual": "HC3 nonlinear mean and variance tests across all retained markers"},
        "interpretation": "Inference depends on model and permutation assumptions; diagnostics do not certify them."}
    if not cohort.clinical_columns:
        result["clinical_status"] = "not_applicable"
        result["interpretation"] = "Marginal inference depends on null exchangeability and subset pivotality; clinical diagnostics are not applicable."
        return result
    reasons = result["reasons"]
    if cohort.clinical_error:
        reasons.append("clinical_basis_failed: " + cohort.clinical_error)
    if cohort.dropped_clinical:
        reasons.append("clinical_rank_deficient: requested clinical terms were not all estimable")
    design = cohort.clinical
    if design is None or design.shape[1] == 0:
        reasons.append("clinical_basis_unavailable")
    elif not np.isfinite(design).all() or np.linalg.matrix_rank(design - design.mean(axis=0)) != design.shape[1]:
        reasons.append("clinical_rank_or_finiteness_failed")
    if reasons:
        result.update(status="withheld", allowed=False, clinical_status="failed")
        return result
    fit = fit_cox(cohort.time, cohort.event, design, cohort.strata, ties)
    result["clinical_fit"] = {"converged": bool(fit.converged), "separated": bool(np.any(fit.separated)),
                              "finite": bool(np.isfinite(fit.beta).all() and np.isfinite(fit.covariance).all()),
                              "loglik": float(fit.loglik), "iterations": int(fit.iterations)}
    if not _usable(fit):
        reasons.append("clinical_fit_failed: nonconvergence, separation, aliasing or non-finite fit")
        result.update(status="withheld", allowed=False, clinical_status="failed")
        return result
    tests = result["clinical_tests"]
    try:
        if ties != "efron" or np.any(cohort.time[cohort.event == 1] <= 0):
            raise ValueError("validated log-time PH diagnostic requires Efron ties and positive event times")
        residual = _efron_schoenfeld_residuals(design, cohort.time, cohort.event, fit.beta, cohort.strata)
        transformed = np.log(np.maximum(cohort.time, np.finfo(float).tiny))
        ph = _cox_grambsch_therneau_test(residual, fit.covariance, transformed)
        tests.append({"name": "PH_global", "method": "classic GT log-time (pre-survival 3.0)",
                      "p_value": ph["p_value"], "statistic": ph["statistic"], "df": ph["df"]})
        tests.extend({"name": "PH_" + name, "method": "classic GT log-time (pre-survival 3.0)",
                      "p_value": p, "statistic": statistic, "df": 1}
                     for name, p, statistic in zip(cohort.clinical_names, ph["term_p_values"], ph["term_statistics"]))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        tests.append({"name": "PH", "p_value": None, "reason": str(exc)})
    if cohort.clinical_basis == "linear":
        try:
            if frozen_transform:
                if functional_form_encoder is None:
                    raise ValueError("No frozen functional-form diagnostic transform is available")
                expanded_encoder = functional_form_encoder
            else:
                expanded_encoder = fit_clinical_encoder(cohort.clinical_frame, cohort.clinical_columns,
                                                        cohort.categorical_clinical, basis="restricted_cubic_spline")
            result["functional_form_encoder"] = expanded_encoder
            expanded = transform_clinical_encoder(cohort.clinical_frame, expanded_encoder)
            df = expanded.shape[1] - design.shape[1]
            if df > 0:
                augmented = fit_cox(cohort.time, cohort.event, expanded, cohort.strata, ties)
                if not _usable(augmented):
                    raise ValueError("fixed spline expansion failed to converge or has separation/rank problems")
                lr = 2 * (augmented.loglik - fit.loglik)
                if lr < -1e-6:
                    raise ValueError("nested spline likelihood is lower than the linear model")
                tests.append({"name": "functional_form_LR", "statistic": max(lr, 0), "df": df,
                              "method": "linear versus fixed five-knot RCS Cox LR",
                              "p_value": float(stats.chi2.sf(max(lr, 0), df))})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            tests.append({"name": "functional_form_LR", "p_value": None, "reason": str(exc)})
    for test, corrected in zip(tests, holm([t["p_value"] for t in tests])):
        test["p_holm"] = corrected
        if corrected is None:
            reasons.append("clinical_diagnostic_failed: " + test["name"])
        elif corrected <= WITHHOLD_ALPHA:
            reasons.append("clinical_misspecification: " + test["name"])

    # Mean: fixed quadratic/cubic raw numeric directions beyond the fitted clinical span.
    # Variance: all clinical basis directions, conditional on stratum intercepts.
    codes = np.zeros(len(block), int) if cohort.strata is None else np.unique(cohort.strata, return_inverse=True)[1]
    intercepts = np.eye(int(codes.max()) + 1)[codes]
    q0 = _orthogonal_columns(intercepts)
    clinical_directions = _orthogonal_columns(design - q0 @ (q0.T @ design))
    base = np.column_stack([q0, clinical_directions])
    polynomial = []
    encoder = cohort.clinical_encoder
    raw = transform_clinical_encoder(cohort.clinical_frame, encoder.get("base_encoder", encoder), output="dataframe")
    for name in encoder["numeric_features"]:
        x = raw[name].to_numpy(float)
        if encoder.get("numeric_kinds", {}).get(name, "continuous" if np.unique(x).size > 2 else "binary_or_constant") == "continuous":
            z = (x - x.mean()) / x.std()
            polynomial.extend([z ** 2, z ** 3])
    extra = np.column_stack(polynomial) if polynomial else np.zeros((len(block), 0))
    extra = _orthogonal_columns(extra - base @ (base.T @ extra))
    mean_design = np.column_stack([base, extra])
    remaining = residualize(block, design, cohort.strata)
    # Under homoscedastic errors E[e_i^2 | Z] = (1-h_i)*sigma^2.
    # Remove this training-projection effect before testing variance changes.
    leverage = np.sum(base * base, axis=1)
    for i, name in enumerate(cohort.marker_names):
        for kind, y, d, start in (("nonlinear_mean", remaining[:, i], mean_design, base.shape[1]),
                                  ("variance", remaining[:, i] ** 2 / np.maximum(1 - leverage, 1e-12), base, q0.shape[1])):
            test = hc3_wald(y, d, start)
            result["residual_tests"].append({"marker": name, "diagnostic": kind, **test})
    residual_tests = result["residual_tests"]
    for test, corrected in zip(residual_tests, holm([t["p_value"] for t in residual_tests])):
        test["p_holm"] = corrected
        if test["status"] == "failed":
            reasons.append("residual_diagnostic_failed: " + test["marker"] + "/" + test["diagnostic"])
        elif corrected is not None and corrected <= WITHHOLD_ALPHA:
            reasons.append("residual_misspecification: " + test["marker"] + "/" + test["diagnostic"])
    result.update(status="withheld" if reasons else "assumption_dependent", allowed=not reasons,
                  clinical_status="calculated")
    result["provenance"] = {"row_mask_hash": cohort.row_mask_hash,
                            "n_training": int(len(cohort.time)), "ties": ties,
                            "basis_method_version": cohort.clinical_encoder.get("basis_method_version"),
                            "mean_directions": "standardized raw numeric squared/cubed beyond clinical span",
                            "variance_directions": "clinical basis conditional on stratum intercepts; squared residual/(1-h) response"}
    return result

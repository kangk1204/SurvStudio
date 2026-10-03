"""Experimental joint calibration of the unchanged HC3 residual diagnostics.

The v2 numerical files stay byte-identical to their qualified release. These
finite bootstrap diagnostics do not establish residual exchangeability.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from survival_toolkit import marker_diagnostics as legacy
from survival_toolkit.clinical_basis import transform_clinical_encoder
from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.marker_screen import residualize

METHOD_VERSION = "marker-inference/3"
N_BOOTSTRAP = 9999
CANDIDATES = ("residual_vector", "restricted_wild")


def policy() -> dict[str, Any]:
    import json
    path = Path(__file__).with_name("data") / "marker_bootstrap_policy.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("candidate") not in CANDIDATES or value.get("draws") != N_BOOTSTRAP:
        raise ValueError("Unknown or altered joint diagnostic policy")
    return {**value, "sha256": hashlib.sha256(raw).hexdigest()}


def designs(cohort: Any, block: np.ndarray):
    codes = np.zeros(len(block), int) if cohort.strata is None else np.unique(cohort.strata, return_inverse=True)[1]
    q0 = legacy._orthogonal_columns(np.eye(int(codes.max()) + 1)[codes])
    clinical = legacy._orthogonal_columns(cohort.clinical - q0 @ (q0.T @ cohort.clinical))
    base = np.column_stack([q0, clinical])
    encoder = cohort.clinical_encoder
    raw = transform_clinical_encoder(cohort.clinical_frame, encoder.get("base_encoder", encoder), output="dataframe")
    polynomial = []
    for name in encoder["numeric_features"]:
        x = raw[name].to_numpy(float)
        kind = encoder.get("numeric_kinds", {}).get(name, "continuous" if np.unique(x).size > 2 else "binary_or_constant")
        if kind == "continuous":
            z = (x - x.mean()) / x.std()
            polynomial.extend([z ** 2, z ** 3])
    extra = np.column_stack(polynomial) if polynomial else np.zeros((len(block), 0))
    extra = legacy._orthogonal_columns(extra - base @ (base.T @ extra))
    return codes, q0, base, np.column_stack([base, extra])


class WaldDesign:
    """Batched OLS/HC3, with fixed design and one result per response/draw."""
    def __init__(self, design: np.ndarray, start: int):
        self.d = np.asarray(design, float)
        n, p = self.d.shape
        self.df = p - start
        if n <= p + 2 or not np.isfinite(self.d).all() or np.linalg.matrix_rank(self.d) != p:
            raise ValueError("insufficient observations or deficient diagnostic rank")
        self.inverse = np.linalg.inv(self.d.T @ self.d)
        self.pinv = self.inverse @ self.d.T
        self.h = np.sum((self.d @ self.inverse) * self.d, axis=1)
        if np.any(self.h >= 1 - 1e-10):
            raise ValueError("diagnostic leverage is too close to one")
        self.influence = (self.d @ self.inverse)[:, start:]
        self.start = start

    def statistic(self, responses: np.ndarray) -> np.ndarray:
        # responses: draw, marker, observation. Marker chunks do not alter draws.
        y = np.asarray(responses, float)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite diagnostic response")
        if self.df == 0:
            return np.full(y.shape[:-1], -np.inf)
        beta = y @ self.pinv.T
        residual = y - beta @ self.d.T
        square = (residual / (1 - self.h)) ** 2
        covariance = np.einsum("bmn,nq,nr->bmqr", square, self.influence, self.influence, optimize=True)
        eigen = np.linalg.eigvalsh(covariance)
        if np.any(eigen[..., 0] <= self.df * np.finfo(float).eps * eigen[..., -1]):
            raise ValueError("singular bootstrap HC3 covariance")
        selected = beta[..., self.start:]
        solved = np.linalg.solve(covariance, selected[..., None])[..., 0]
        statistic = np.sum(selected * solved, axis=-1)
        if not np.isfinite(statistic).all() or np.any(statistic < -1e-10):
            raise ValueError("nonfinite bootstrap Wald statistic")
        return np.maximum(statistic, 0) / self.df


def uniform_stream(seed: int, draws: int, n: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0xD1A6, 3])).random((draws, n))


def plus_one_interval(hits: int, draws: int) -> list[float]:
    lo = 0. if hits == 0 else float(stats.beta.ppf(.025, hits, draws - hits + 1))
    hi = 1. if hits == draws else float(stats.beta.ppf(.975, hits + 1, draws - hits))
    return [(1 + draws * lo) / (draws + 1), (1 + draws * hi) / (draws + 1)]


def joint_maxima(cohort: Any, block: np.ndarray, *, candidate: str, uniforms: np.ndarray,
                 draw_chunk: int = 32, marker_chunk: int = 128) -> np.ndarray:
    if candidate not in CANDIDATES or draw_chunk < 1 or marker_chunk < 1:
        raise ValueError("Invalid joint bootstrap candidate or chunk size")
    block = np.asarray(block, float)
    uniforms = np.asarray(uniforms, float)
    if uniforms.ndim != 2 or uniforms.shape[1] != len(block) or not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError("Invalid common bootstrap stream")
    codes, q0, base, mean_design = designs(cohort, block)
    mean_test = WaldDesign(mean_design, base.shape[1])
    var_test = WaldDesign(base, q0.shape[1])
    remaining = residualize(block, cohort.clinical, cohort.strata)
    h = np.sum(base * base, axis=1)
    variance = remaining ** 2 / (1 - h[:, None])
    strata = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    adjusted = remaining / np.sqrt(1 - h[:, None])
    for rows in strata:
        adjusted[rows] -= adjusted[rows].mean(axis=0)
    variance_fit = q0 @ (q0.T @ variance)
    mean_fit = base @ (base.T @ remaining)
    mean_error = (remaining - mean_fit) / (1 - h[:, None])
    variance_error = (variance - variance_fit) / (1 - np.sum(q0 * q0, axis=1)[:, None])
    maxima = np.full(len(uniforms), -np.inf)
    for begin in range(0, len(uniforms), draw_chunk):
        raise_if_cancelled()
        u = uniforms[begin:begin + draw_chunk]
        indices = np.zeros(u.shape, dtype=int)
        if candidate == "residual_vector":
            for rows in strata:
                indices[:, rows] = rows[np.floor(u[:, rows] * len(rows)).astype(int)]
        multipliers = np.where(u >= .5, 1., -1.)[:, None, :]
        for column in range(0, block.shape[1], marker_chunk):
            stop = min(column + marker_chunk, block.shape[1])
            if candidate == "residual_vector":
                sample = adjusted[indices, column:stop].transpose(0, 2, 1)
                mean_y = sample - (sample @ base) @ base.T
                var_y = mean_y ** 2 / (1 - h)
            else:
                mean_y = mean_fit[:, column:stop].T[None] + mean_error[:, column:stop].T[None] * multipliers
                var_y = variance_fit[:, column:stop].T[None] + variance_error[:, column:stop].T[None] * multipliers
            for test, response in ((mean_test, mean_y), (var_test, var_y)):
                if test.df:
                    values = test.statistic(response)
                    maxima[begin:begin + len(u)] = np.maximum(maxima[begin:begin + len(u)], values.max(axis=1))
    if not np.isfinite(maxima).all():
        raise ValueError("joint diagnostic family has no estimable statistic")
    return maxima


def diagnose_markers(cohort: Any, block: np.ndarray, *, ties: str = "efron",
                     functional_form_encoder=None, frozen_transform=False,
                     candidate: str | None = None, bootstrap_seed: int = 2026100321,
                     draws: int = N_BOOTSTRAP, draw_chunk: int = 32,
                     marker_chunk: int = 128, uniforms: np.ndarray | None = None) -> dict[str, Any]:
    if not isinstance(draws, int) or draws < 1 or draws > N_BOOTSTRAP:
        raise ValueError("Invalid diagnostic bootstrap count")
    declared = policy()
    candidate = declared["candidate"] if candidate is None else candidate
    if candidate not in CANDIDATES:
        raise ValueError("Unknown joint diagnostic candidate")
    result = copy.deepcopy(legacy.diagnose_markers(cohort, block, ties=ties,
        functional_form_encoder=functional_form_encoder, frozen_transform=frozen_transform))
    result.update(method_version=METHOD_VERSION, diagnostic_policy="v3_joint_bootstrap")
    result["bootstrap"] = {"candidate": candidate, "draws": draws, "seed": int(bootstrap_seed),
        "method": "joint single-step maxT of HC3 Wald/df", "policy_sha256": declared["sha256"],
        "selection_status": declared["selection_status"], "status": "not_applicable",
        "stream_version": "numpy-seedsequence-uniform/1", "mc_interval_level": .95}
    result["families"]["residual"] = "joint bootstrap maxT across all applicable marker mean and variance diagnostics"
    if not cohort.clinical_columns or result.get("clinical_status") == "failed":
        return result
    tests = result["residual_tests"]
    result["reasons"] = [reason for reason in result["reasons"] if not reason.startswith("residual_")]
    for test in tests:
        test["p_holm_v2"] = test.pop("p_holm", None)
        test["p_maxT"] = None
    usable = [test for test in tests if test.get("status") == "calculated"]
    try:
        if any(test.get("status") == "failed" for test in tests):
            raise ValueError("A required observed residual diagnostic failed")
        if usable:
            u = uniform_stream(bootstrap_seed, draws, len(block)) if uniforms is None else uniforms
            if len(u) != draws:
                raise ValueError("Bootstrap stream length differs from declared count")
            maxima = joint_maxima(cohort, block, candidate=candidate, uniforms=u,
                                  draw_chunk=draw_chunk, marker_chunk=marker_chunk)
            observed = np.array([test["statistic"] / test["df"] for test in usable])
            # Sorted maxima avoid allocating a draws by tests comparison matrix.
            ordered = np.sort(maxima)
            counts = draws - np.searchsorted(ordered, observed, side="left")
            for test, hits in zip(usable, counts):
                test["bootstrap_exceedances"] = int(hits)
                test["p_maxT"] = float((1 + hits) / (draws + 1))
                if test["p_maxT"] <= legacy.WITHHOLD_ALPHA:
                    result["reasons"].append("residual_misspecification: " + test["marker"] + "/" + test["diagnostic"])
            hits = int(counts.min())
            interval = plus_one_interval(hits, draws)
            result["bootstrap"].update(status="calculated", global_p=(hits + 1) / (draws + 1),
                global_exceedances=hits, global_mc95=interval, completed_draws=draws,
                uniforms_sha256=hashlib.sha256(np.asarray(u, dtype="<f8").tobytes()).hexdigest())
            if interval[0] <= legacy.WITHHOLD_ALPHA <= interval[1]:
                result["reasons"].append("diagnostic_mc_uncertain: joint residual family crosses the 1% boundary")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        result["bootstrap"].update(status="failed", reason=str(exc))
        result["reasons"].append("residual_diagnostic_failed: " + str(exc))
    result.update(status="withheld" if result["reasons"] else "assumption_dependent", allowed=not result["reasons"])
    digest = hashlib.sha256()
    for array in (cohort.clinical, block):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    digest.update(str(cohort.marker_names).encode())
    result.setdefault("provenance", {}).update(diagnostic_input_sha256=digest.hexdigest(),
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation_scope="Experimental finite calibration; no exchangeability or universal error-control certification")
    return result

"""Vectorized Cox score screening of many candidate markers.

The functions here work on plain arrays, so the web app and headless benchmark
scripts share them. For every marker column x, the screen evaluates the Cox
partial-likelihood score at beta_x = 0 with the clinical part of the model held at
its fitted null value:

* score U = x' M, where M holds the martingale residuals of the null model;
* efficient information I = x'Wx - (x'WZ)(Z'WZ)^-1 (Z'Wx), from the risk-set sums.

U^2 / I is the score test that R's ``coxph`` reports with
``init = c(coef(null_model), 0)`` and ``iter.max = 0``, for Efron or Breslow ties,
with or without strata. Without clinical covariates it is the univariate Cox score
test. It equals the log-rank test only when no event times are tied.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence

import numpy as np
from scipy import sparse, special

from survival_toolkit.analysis import _bh_adjust, _efron_tie_groups

TIES_METHODS = ("efron", "breslow")
# Columns processed together. Blocks of a few hundred columns keep the n x block working
# arrays in cache; 512 ran about three times faster than 2048 on 500 rows.
_COLUMN_BLOCK = 512


class CoxNull(NamedTuple):
    """The fitted null model: clinical covariates only (or no covariates at all)."""

    eta: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    loglik: float
    converged: bool


class CoxFit(NamedTuple):
    """A Cox model fitted by Newton-Raphson on the partial likelihood."""

    beta: np.ndarray
    covariance: np.ndarray
    loglik: float
    converged: bool
    iterations: int


class ScoreStats(NamedTuple):
    """Per-marker score statistics at beta_marker = 0."""

    score: np.ndarray
    information: np.ndarray
    chi2: np.ndarray
    z: np.ndarray
    p_value: np.ndarray
    beta_one_step: np.ndarray
    collinear: np.ndarray


class MaxT(NamedTuple):
    """Westfall–Young max-statistic adjusted p-values."""

    p_single_step: np.ndarray
    p_step_down: np.ndarray
    n_permutations: int


class _StratumTerms(NamedTuple):
    order: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    expected: np.ndarray
    martingale: np.ndarray
    # exp(eta)-weighted sums of the tied events' rows by tie group (over all sorted rows).
    tie_sum: sparse.csr_matrix
    # exp(eta)-weighted sums of the sorted rows from each event time's first position up to the next one's.
    segment_sum: sparse.csr_matrix
    # Whether the Efron tie terms (b, c) are non-zero; they vanish without ties and for Breslow.
    tied: bool


def _as_strata(strata: np.ndarray | None, n: int) -> np.ndarray:
    if strata is None:
        return np.zeros(n, dtype=np.int64)
    codes = np.asarray(strata).reshape(-1)
    if codes.shape[0] != n:
        raise ValueError("strata must have one entry per row.")
    return codes


def _check_ties(ties: str) -> str:
    if ties not in TIES_METHODS:
        raise ValueError(f"ties must be one of {TIES_METHODS}, not {ties!r}.")
    return ties


def _stratum_terms(
    rows: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
    ties: str,
) -> _StratumTerms | None:
    """Risk-set bookkeeping of one stratum for the score and information sums.

    For d events tied at time t, event k = 0..d-1 uses the denominator
    S0 - (k/d) S0_tied for Efron ties and S0 for Breslow ties. Per tie group,
    a, b and c sum 1/den^2 weighted by (k/d)^0, (k/d)^1 and (k/d)^2. The expected
    counts E and martingale residuals M = event - E follow R's residuals.coxph.
    """
    groups = _efron_tie_groups(rows, time, event, eta)
    if groups is None:
        return None
    tie_index = groups.tie_index
    if ties == "breslow":
        tied_s0 = np.bincount(tie_index, weights=groups.risk[groups.dead_pos])
        fraction = np.zeros_like(groups.fraction)
        denominator = groups.denominator + groups.fraction * tied_s0[tie_index]
    else:
        fraction = groups.fraction
        denominator = groups.denominator
    inverse = 1.0 / denominator
    inverse_sq = inverse * inverse
    hazard = np.bincount(tie_index, weights=inverse)
    shared = np.bincount(tie_index, weights=fraction * inverse)
    a = np.bincount(tie_index, weights=inverse_sq)
    b = np.bincount(tie_index, weights=fraction * inverse_sq)
    c = np.bincount(tie_index, weights=fraction * fraction * inverse_sq)
    cumulative = np.cumsum(hazard)
    event_times = groups.sorted_time[groups.tie_starts]
    n_passed = np.searchsorted(event_times, groups.sorted_time, side="right")
    cumulative_hazard = np.where(n_passed > 0, cumulative[np.maximum(n_passed - 1, 0)], 0.0)
    # Each tied event carries only its Efron share (1 - k/d) of its own jump.
    cumulative_hazard[groups.dead_pos] -= shared[tie_index]
    expected = groups.risk * cumulative_hazard
    died = np.zeros(groups.order.size, dtype=float)
    died[groups.dead_pos] = 1.0
    n_groups, n_rows = groups.tie_starts.size, groups.order.size
    tie_sum = sparse.csr_matrix(
        (groups.risk[groups.dead_pos], (tie_index, groups.dead_pos)),
        shape=(n_groups, n_rows),
    )
    # Rows censored before the first event time belong to no risk set.
    segment = np.searchsorted(groups.tie_starts, np.arange(n_rows), side="right") - 1
    in_risk = np.flatnonzero(segment >= 0)
    segment_sum = sparse.csr_matrix(
        (groups.risk[in_risk], (segment[in_risk], in_risk)),
        shape=(n_groups, n_rows),
    )
    return _StratumTerms(
        order=groups.order,
        a=a,
        b=b,
        c=c,
        expected=expected,
        martingale=died - expected,
        tie_sum=tie_sum,
        segment_sum=segment_sum,
        tied=bool(np.any(b != 0.0) or np.any(c != 0.0)),
    )


def _reverse_cumsum_rows(block: np.ndarray) -> np.ndarray:
    """Cumulative sums from the last row upwards.

    np.cumsum along axis 0 walks each column with a row-sized stride, which is slow for
    blocks thousands of columns wide; adding whole rows keeps the memory access contiguous.
    """
    if block.shape[1] < 64:
        return np.cumsum(block[::-1], axis=0)[::-1]
    out = np.empty_like(block)
    running = np.zeros(block.shape[1], dtype=block.dtype)
    for row in range(block.shape[0] - 1, -1, -1):
        running += block[row]
        out[row] = running
    return out


def _risk_set_sums(terms: _StratumTerms, sorted_block: np.ndarray, *, tied_sums: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Risk-set sums S1 and tied-event sums T1 of exp(eta) * x at each event time.

    The risk set of an event time holds every row from its first sorted position on, so S1
    is the reverse cumulative sum of the rows summed between consecutive event times.
    """
    s1 = _reverse_cumsum_rows(np.asarray(terms.segment_sum @ sorted_block))
    if not tied_sums:
        return s1, None
    return s1, np.asarray(terms.tie_sum @ sorted_block)


class _WeightedSide(NamedTuple):
    """One block's sums pre-multiplied by the per-row and per-group information weights."""

    expected: np.ndarray
    a_s1: np.ndarray
    b_t1: np.ndarray
    b_s1: np.ndarray
    c_t1: np.ndarray


def _weighted_side(terms: _StratumTerms, sorted_block: np.ndarray, sums: tuple[np.ndarray, np.ndarray]) -> _WeightedSide:
    s1, t1 = sums
    return _WeightedSide(
        expected=terms.expected[:, None] * sorted_block,
        a_s1=terms.a[:, None] * s1,
        b_t1=terms.b[:, None] * t1,
        b_s1=terms.b[:, None] * s1,
        c_t1=terms.c[:, None] * t1,
    )


def _cross_information(
    left: np.ndarray,
    left_sums: tuple[np.ndarray, np.ndarray],
    right: _WeightedSide,
) -> np.ndarray:
    """Information between two sorted blocks: sum E x y' - sum_g [a S1x S1y' - b(...) + c T1x T1y'].

    The weights sit on the right-hand block (the few clinical covariates), so a wide
    marker block is only read once.
    """
    s1_left, t1_left = left_sums
    return (
        left.T @ right.expected
        - s1_left.T @ right.a_s1
        + s1_left.T @ right.b_t1
        + t1_left.T @ right.b_s1
        - t1_left.T @ right.c_t1
    )


def cox_partial_loglik(
    time: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
    strata: np.ndarray | None = None,
    ties: str = "efron",
) -> float:
    """Cox partial log-likelihood of a fixed linear predictor (Efron or Breslow ties)."""
    ties = _check_ties(ties)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(bool)
    eta = np.asarray(eta, dtype=float).reshape(-1)
    codes = _as_strata(strata, time.shape[0])
    loglik = 0.0
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        groups = _efron_tie_groups(rows, time, event, eta)
        if groups is None:
            continue
        shift = float(eta[rows].max())
        denominator = groups.denominator
        if ties == "breslow":
            tied_s0 = np.bincount(groups.tie_index, weights=groups.risk[groups.dead_pos])
            denominator = denominator + groups.fraction * tied_s0[groups.tie_index]
        dead_eta = eta[groups.order[groups.dead_pos]] - shift
        loglik += float(np.sum(dead_eta) - np.sum(np.log(denominator)))
    return loglik


def _score_and_information_at(
    time: np.ndarray,
    event: np.ndarray,
    design: np.ndarray,
    codes: np.ndarray,
    eta: np.ndarray,
    ties: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial-likelihood score vector and information matrix of ``design`` at the linear predictor ``eta``."""
    k = design.shape[1]
    score = np.zeros(k, dtype=float)
    information = np.zeros((k, k), dtype=float)
    for code in np.unique(codes):
        terms = _stratum_terms(np.flatnonzero(codes == code), time, event, eta, ties)
        if terms is None:
            continue
        # Centring changes neither sum (the martingale residuals of a stratum sum to zero)
        # and keeps the information from cancelling for large-valued columns.
        sorted_x = design[terms.order]
        sorted_x = sorted_x - sorted_x.mean(axis=0, keepdims=True)
        sums = _risk_set_sums(terms, sorted_x)
        score += terms.martingale @ sorted_x
        information += _cross_information(sorted_x, sums, _weighted_side(terms, sorted_x, sums))
    return score, information


def fit_cox(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    strata: np.ndarray | None = None,
    ties: str = "efron",
    *,
    max_iterations: int = 50,
) -> CoxFit:
    """Maximum partial-likelihood Cox fit by Newton-Raphson with step halving (Efron or Breslow ties, strata).

    Uses the screen's risk-set sums, so a fit costs a few passes over the rows; it replaces
    statsmodels' PHReg, whose Efron fit loops over event times in Python. Convergence
    follows R's coxph: the partial log-likelihood changes by less than 1e-9 relative.
    """
    ties = _check_ties(ties)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(bool)
    design = np.asarray(X, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.shape[0] != time.shape[0]:
        raise ValueError("X must have one row per subject.")
    codes = _as_strata(strata, time.shape[0])
    beta = np.zeros(design.shape[1], dtype=float)
    loglik = cox_partial_loglik(time, event, design @ beta, codes, ties)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        score, information = _score_and_information_at(time, event, design, codes, design @ beta, ties)
        # A diverging fit (monotone likelihood, as for a heavy-tailed marker that separates the
        # events) can leave the information non-finite; it then ends as not converged instead of
        # failing the whole marker screen.
        if not (np.all(np.isfinite(score)) and np.all(np.isfinite(information))):
            break
        try:
            step = np.linalg.lstsq(information, score, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        for _ in range(40):
            candidate = beta + step
            candidate_loglik = cox_partial_loglik(time, event, design @ candidate, codes, ties)
            if np.isfinite(candidate_loglik) and candidate_loglik >= loglik - 1e-12 * max(abs(loglik), 1.0):
                break
            step = step / 2.0
        else:
            break
        change = candidate_loglik - loglik
        beta, loglik = candidate, candidate_loglik
        if abs(change) <= 1e-9 * max(abs(loglik), 1.0):
            converged = True
            break
    _, information = _score_and_information_at(time, event, design, codes, design @ beta, ties)
    if not np.all(np.isfinite(information)):
        covariance = np.full_like(information, np.nan)
        converged = False
    else:
        try:
            covariance = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            try:
                covariance = np.linalg.pinv(information)
            except np.linalg.LinAlgError:
                covariance = np.full_like(information, np.nan)
            converged = False
    return CoxFit(beta=beta, covariance=covariance, loglik=float(loglik), converged=converged, iterations=iterations)


def fit_cox_null(
    time: np.ndarray,
    event: np.ndarray,
    Z: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    ties: str = "efron",
) -> CoxNull:
    """Fit the clinical-only Cox model that every marker is screened against."""
    ties = _check_ties(ties)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(int)
    n = time.shape[0]
    if Z is None or np.asarray(Z).size == 0:
        eta = np.zeros(n, dtype=float)
        return CoxNull(
            eta=eta,
            beta=np.zeros(0, dtype=float),
            covariance=np.zeros((0, 0), dtype=float),
            loglik=cox_partial_loglik(time, event, eta, strata, ties),
            converged=True,
        )
    design = np.asarray(Z, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.shape[0] != n:
        raise ValueError("Z must have one row per subject.")
    fit = fit_cox(time, event, design, strata, ties)
    return CoxNull(
        eta=design @ fit.beta,
        beta=fit.beta,
        covariance=np.atleast_2d(fit.covariance),
        loglik=fit.loglik,
        converged=fit.converged,
    )


def residualize(X: np.ndarray, Z: np.ndarray | None = None, strata: np.ndarray | None = None) -> np.ndarray:
    """Residuals of each marker column after least-squares projection on strata and Z.

    The screen's statistics do not change when a combination of Z or a stratum
    constant is added to a marker, so permuting these residuals gives the
    Freedman–Lane null for "no added value beyond the clinical covariates" while
    keeping each marker's relationship with the clinical covariates.
    """
    block = np.asarray(X, dtype=float)
    n = block.shape[0]
    codes = _as_strata(strata, n)
    _, stratum_index = np.unique(codes, return_inverse=True)
    columns = [np.eye(int(stratum_index.max()) + 1)[stratum_index]]
    if Z is not None and np.asarray(Z).size:
        design_z = np.asarray(Z, dtype=float)
        columns.append(design_z.reshape(n, -1))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, block, rcond=None)
    return block - design @ coefficients


class CoxScoreScreen:
    """Score statistics of many markers against one fitted null Cox model.

    Parameters
    ----------
    time, event
        Follow-up times and 0/1 event indicators.
    null
        The fitted null model from :func:`fit_cox_null` (its linear predictor fixes
        the risk-set weights).
    Z
        The clinical covariates of the null model (None when it has none). They
        enter only through the efficient information.
    strata, ties
        Stratum codes and the tie method; they must match the null fit.
    """

    def __init__(
        self,
        time: np.ndarray,
        event: np.ndarray,
        *,
        null: CoxNull,
        Z: np.ndarray | None = None,
        strata: np.ndarray | None = None,
        ties: str = "efron",
        block_size: int = _COLUMN_BLOCK,
    ) -> None:
        self.ties = _check_ties(ties)
        self.time = np.asarray(time, dtype=float).reshape(-1)
        self.event = np.asarray(event).reshape(-1).astype(bool)
        self.n = self.time.shape[0]
        self.strata = _as_strata(strata, self.n)
        self.null = null
        self.block_size = max(int(block_size), 1)
        self.terms: list[_StratumTerms] = []
        for code in np.unique(self.strata):
            terms = _stratum_terms(np.flatnonzero(self.strata == code), self.time, self.event, null.eta, self.ties)
            if terms is not None:
                self.terms.append(terms)
        if not self.terms:
            raise ValueError("The score screen needs at least one event.")
        design = None if Z is None or np.asarray(Z).size == 0 else np.asarray(Z, dtype=float).reshape(self.n, -1)
        # Per stratum, the clinical side of the marker-by-clinical information:
        # I_xz = X' rows + S1x' at_s1 + T1x' at_t1.
        self._z_weights: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._izz_inverse: np.ndarray | None = None
        if design is not None:
            izz = np.zeros((design.shape[1], design.shape[1]), dtype=float)
            for terms in self.terms:
                sorted_z = design[terms.order]
                sorted_z = sorted_z - sorted_z.mean(axis=0, keepdims=True)
                sums = _risk_set_sums(terms, sorted_z)
                side = _weighted_side(terms, sorted_z, sums)
                self._z_weights.append((side.expected, side.b_t1 - side.a_s1, side.b_s1 - side.c_t1))
                izz += _cross_information(sorted_z, sums, side)
            self._izz_inverse = np.linalg.pinv(izz)

    def _score_and_information(self, X: np.ndarray, row_index: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        n_markers = X.shape[1]
        score = np.zeros(n_markers, dtype=float)
        information = np.zeros(n_markers, dtype=float)
        for start in range(0, n_markers, self.block_size):
            stop = min(start + self.block_size, n_markers)
            block = X[:, start:stop]
            u = np.zeros(stop - start, dtype=float)
            ixx = np.zeros(stop - start, dtype=float)
            ixz = None if self._izz_inverse is None else np.zeros((stop - start, self._izz_inverse.shape[0]), dtype=float)
            for index, terms in enumerate(self.terms):
                source_rows = terms.order if row_index is None else row_index[terms.order]
                sorted_block = block[source_rows]
                s1, t1 = _risk_set_sums(terms, sorted_block, tied_sums=terms.tied)
                u += np.einsum("i,ij->j", terms.martingale, sorted_block)
                ixx += np.einsum("i,ij->j", terms.expected, sorted_block * sorted_block)
                ixx -= np.einsum("g,gj->j", terms.a, s1 * s1)
                if terms.tied:
                    ixx += 2.0 * np.einsum("g,gj->j", terms.b, s1 * t1) - np.einsum("g,gj->j", terms.c, t1 * t1)
                if ixz is not None:
                    rows, at_s1, at_t1 = self._z_weights[index]
                    for column in range(rows.shape[1]):
                        ixz[:, column] += np.einsum("ij,i->j", sorted_block, rows[:, column])
                        ixz[:, column] += np.einsum("gj,g->j", s1, at_s1[:, column])
                        if terms.tied:
                            ixz[:, column] += np.einsum("gj,g->j", t1, at_t1[:, column])
            if ixz is not None:
                ixx = ixx - np.einsum("ij,jk,ik->i", ixz, self._izz_inverse, ixz)
            score[start:stop] = u
            information[start:stop] = ixx
        return score, information

    def _prepared(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centred copy of X and the per-column information floor for collinearity.

        Adding the same constant to every row of a column changes neither U (the
        martingale residuals sum to zero in each stratum) nor I, for any row
        permutation; centring keeps the information from cancelling catastrophically
        for large-valued markers.
        """
        block = self._validated(X)
        centred = block - block.mean(axis=0, keepdims=True)
        variance = np.mean(centred * centred, axis=0)
        tolerance = 1e-10 * np.maximum(variance, 1e-300) * max(int(self.event.sum()), 1)
        return centred, tolerance

    @staticmethod
    def _chi2(score: np.ndarray, information: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
        collinear = ~(information > tolerance)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(collinear, np.nan, score * score / information)

    def _finish(self, score: np.ndarray, information: np.ndarray, tolerance: np.ndarray) -> ScoreStats:
        collinear = ~(information > tolerance)
        chi2 = self._chi2(score, information, tolerance)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(collinear, np.nan, score / np.sqrt(np.where(collinear, 1.0, information)))
            beta = np.where(collinear, np.nan, score / information)
        # chdtrc(1, x) is the chi-square(1) survival function without scipy.stats' per-call overhead.
        p_value = np.where(collinear, np.nan, special.chdtrc(1.0, np.where(collinear, 0.0, chi2)))
        return ScoreStats(
            score=score,
            information=information,
            chi2=chi2,
            z=z,
            p_value=p_value,
            beta_one_step=beta,
            collinear=collinear,
        )

    def statistics(self, X: np.ndarray) -> ScoreStats:
        """Score statistics of each column of X (n x p, finite values)."""
        centred, tolerance = self._prepared(X)
        score, information = self._score_and_information(centred, None)
        return self._finish(score, information, tolerance)

    def permuted_chi2(self, X: np.ndarray, permutations: Iterable[np.ndarray]) -> np.ndarray:
        """Chi-square statistics of X with its rows permuted, one row per permutation.

        ``permutations[b]`` maps each subject to the marker row it receives, so row
        b of the result equals ``statistics(X[permutations[b]]).chi2``.
        """
        centred, tolerance = self._prepared(X)
        rows = []
        for permutation in permutations:
            index = np.asarray(permutation, dtype=np.int64).reshape(-1)
            if index.shape[0] != self.n:
                raise ValueError("Each permutation must list every row once.")
            score, information = self._score_and_information(centred, index)
            rows.append(self._chi2(score, information, tolerance))
        return np.vstack(rows) if rows else np.zeros((0, centred.shape[1]), dtype=float)

    def _validated(self, X: np.ndarray) -> np.ndarray:
        block = np.asarray(X, dtype=float)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.shape[0] != self.n:
            raise ValueError("X must have one row per subject.")
        if not np.isfinite(block).all():
            raise ValueError("Marker values must be finite; impute missing values before screening.")
        return block


def stratified_permutation(strata: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
    """A random permutation of the rows that keeps every row inside its stratum."""
    codes = _as_strata(strata, n)
    permutation = np.arange(n)
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        permutation[rows] = rng.permutation(rows)
    return permutation


def _extreme_threshold(values: np.ndarray) -> np.ndarray:
    # Identical arithmetic for observed and permuted statistics, so a tie with the
    # observed value counts as at least as extreme; the tolerance absorbs summation order.
    return values - 1e-9 * np.maximum(1.0, np.abs(values))


class MaxTAccumulator:
    """Streams permuted statistics into Westfall–Young single-step and step-down p-values."""

    def __init__(self, observed: np.ndarray) -> None:
        self.observed = np.asarray(observed, dtype=float).reshape(-1)
        self.valid = np.isfinite(self.observed)
        filled = np.where(self.valid, self.observed, -np.inf)
        self.order = np.argsort(-filled, kind="mergesort")
        self.sorted_threshold = _extreme_threshold(filled[self.order])
        self.threshold = _extreme_threshold(filled)
        self.single_counts = np.zeros(self.observed.shape[0], dtype=np.int64)
        self.step_counts = np.zeros(self.observed.shape[0], dtype=np.int64)
        self.n_permutations = 0

    def update(self, permuted: np.ndarray) -> None:
        block = np.atleast_2d(np.asarray(permuted, dtype=float))
        if block.shape[1] != self.observed.shape[0]:
            raise ValueError("Permuted statistics must have one column per marker.")
        # Markers without a valid observed statistic are not part of the tested family.
        block = np.where(np.isfinite(block) & self.valid[None, :], block, -np.inf)
        maxima = block.max(axis=1)
        self.single_counts += (maxima[:, None] >= self.threshold[None, :]).sum(axis=0)
        # Successive maxima over the markers ranked below each observed rank.
        reordered = block[:, self.order]
        successive = np.maximum.accumulate(reordered[:, ::-1], axis=1)[:, ::-1]
        self.step_counts += (successive >= self.sorted_threshold[None, :]).sum(axis=0)
        self.n_permutations += block.shape[0]

    def result(self) -> MaxT:
        denominator = float(self.n_permutations + 1)
        single = (self.single_counts + 1) / denominator
        sorted_step = (self.step_counts + 1) / denominator
        sorted_step = np.maximum.accumulate(sorted_step)
        step = np.empty_like(sorted_step)
        step[self.order] = sorted_step
        single = np.where(self.valid, np.minimum(single, 1.0), np.nan)
        step = np.where(self.valid, np.minimum(step, 1.0), np.nan)
        return MaxT(p_single_step=single, p_step_down=step, n_permutations=int(self.n_permutations))


class PermutationFdrAccumulator:
    """Streams permuted statistics into permutation-based FDR q-values.

    For a threshold t, the estimated FDR is the mean number of permuted statistics at
    or above t divided by the number of observed statistics at or above t; a marker's
    q-value is the smallest estimate over thresholds at or below its own statistic.
    """

    def __init__(self, observed: np.ndarray) -> None:
        self.observed = np.asarray(observed, dtype=float).reshape(-1)
        self.valid = np.isfinite(self.observed)
        self.sorted_observed = np.sort(self.observed[self.valid])[::-1]
        self.thresholds = _extreme_threshold(self.sorted_observed)
        self.false_counts = np.zeros(self.sorted_observed.shape[0], dtype=float)
        self.n_permutations = 0

    def update(self, permuted: np.ndarray) -> None:
        block = np.atleast_2d(np.asarray(permuted, dtype=float))
        for row in block:
            values = np.sort(row[np.isfinite(row)])
            self.false_counts += values.shape[0] - np.searchsorted(values, self.thresholds, side="left")
            self.n_permutations += 1

    def q_values(self) -> np.ndarray:
        q = np.full(self.observed.shape[0], np.nan, dtype=float)
        if self.n_permutations == 0 or self.sorted_observed.size == 0:
            return q
        ascending = self.sorted_observed[::-1]
        discoveries = ascending.shape[0] - np.searchsorted(ascending, self.thresholds, side="left")
        estimate = np.minimum(1.0, (self.false_counts / self.n_permutations) / np.maximum(discoveries, 1))
        # Smallest estimate among thresholds at or below each statistic (later in the descending order).
        monotone = np.minimum.accumulate(estimate[::-1])[::-1]
        positions = np.searchsorted(-self.sorted_observed, -self.observed[self.valid], side="right") - 1
        q[self.valid] = monotone[positions]
        return q


def bh_vector(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values; non-finite inputs stay NaN (as ``_bh_adjust``)."""
    return np.asarray(_bh_adjust(np.asarray(p_values, dtype=float).tolist()), dtype=float)


def harrell_c_many(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """Harrell's C for each risk column, with the conventions of ``_harrell_c_index``.

    A pair is comparable when the earlier time is an event, or when both times are
    equal and only the first subject had the event (the censored one is known to
    outlive it). Higher risk should mean earlier events; tied risks count one half.
    Columns without any comparable pair give NaN.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(bool)
    scores = np.asarray(risk, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    events = np.flatnonzero(event)
    result = np.full(scores.shape[1], np.nan, dtype=float)
    if events.size == 0:
        return result
    later = time[None, :] > time[events][:, None]
    tied_censored = (time[None, :] == time[events][:, None]) & ~event[None, :]
    comparable = later | tied_censored
    n_comparable = float(comparable.sum())
    if n_comparable <= 0.0:
        return result
    for column in range(scores.shape[1]):
        values = scores[:, column]
        event_values = values[events][:, None]
        concordant = np.sum(comparable & (event_values > values[None, :]))
        ties = np.sum(comparable & (event_values == values[None, :]))
        result[column] = (float(concordant) + 0.5 * float(ties)) / n_comparable
    return result

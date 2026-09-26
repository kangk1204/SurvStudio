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
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

from survival_toolkit.analysis import _bh_adjust, _efron_tie_groups, fit_phreg

TIES_METHODS = ("efron", "breslow")
# Columns processed together; bounds the n x block working arrays for wide marker sets.
_COLUMN_BLOCK = 2048


class CoxNull(NamedTuple):
    """The fitted null model: clinical covariates only (or no covariates at all)."""

    eta: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    loglik: float
    converged: bool


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
    risk: np.ndarray
    tie_starts: np.ndarray
    group_starts: np.ndarray
    dead_pos: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    expected: np.ndarray
    martingale: np.ndarray


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
    group_starts = np.searchsorted(tie_index, np.arange(groups.tie_starts.size), side="left")
    return _StratumTerms(
        order=groups.order,
        risk=groups.risk,
        tie_starts=groups.tie_starts,
        group_starts=group_starts,
        dead_pos=groups.dead_pos,
        a=a,
        b=b,
        c=c,
        expected=expected,
        martingale=died - expected,
    )


def _risk_set_sums(terms: _StratumTerms, sorted_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-set sums S1 and tied-event sums T1 of exp(eta) * x at each event time."""
    weighted = terms.risk[:, None] * sorted_block
    # Rows between consecutive event times, summed from the last one backwards: the
    # risk set of an event time holds every row from its first sorted position on.
    segments = np.add.reduceat(weighted, terms.tie_starts, axis=0)
    s1 = np.cumsum(segments[::-1], axis=0)[::-1]
    t1 = np.add.reduceat(weighted[terms.dead_pos], terms.group_starts, axis=0)
    return s1, t1


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
    model = PHReg(time, design, status=event, strata=strata, ties=ties)
    results, converged = fit_phreg(model)
    beta = np.asarray(results.params, dtype=float)
    return CoxNull(
        eta=design @ beta,
        beta=beta,
        covariance=np.atleast_2d(np.asarray(results.cov_params(), dtype=float)),
        loglik=float(results.llf),
        converged=bool(converged),
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
        self._z_sides: list[_WeightedSide] = []
        self._izz_inverse: np.ndarray | None = None
        if design is not None:
            izz = np.zeros((design.shape[1], design.shape[1]), dtype=float)
            for terms in self.terms:
                sorted_z = design[terms.order]
                sorted_z = sorted_z - sorted_z.mean(axis=0, keepdims=True)
                sums = _risk_set_sums(terms, sorted_z)
                side = _weighted_side(terms, sorted_z, sums)
                self._z_sides.append(side)
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
                sums = _risk_set_sums(terms, sorted_block)
                s1, t1 = sums
                u += terms.martingale @ sorted_block
                ixx += terms.expected @ (sorted_block * sorted_block)
                ixx -= terms.a @ (s1 * s1) - 2.0 * (terms.b @ (s1 * t1)) + terms.c @ (t1 * t1)
                if ixz is not None:
                    ixz += _cross_information(sorted_block, sums, self._z_sides[index])
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

    def _finish(self, score: np.ndarray, information: np.ndarray, tolerance: np.ndarray) -> ScoreStats:
        collinear = ~(information > tolerance)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(collinear, np.nan, score * score / information)
            z = np.where(collinear, np.nan, score / np.sqrt(np.where(collinear, 1.0, information)))
            beta = np.where(collinear, np.nan, score / information)
        p_value = np.where(collinear, np.nan, stats.chi2.sf(np.where(collinear, 0.0, chi2), df=1))
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
            rows.append(self._finish(score, information, tolerance).chi2)
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

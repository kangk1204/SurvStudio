from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.analysis import _efron_schoenfeld_residuals, _harrell_c_index
from survival_toolkit.marker_screen import (
    CoxScoreScreen,
    MaxTAccumulator,
    PermutationFdrAccumulator,
    bh_vector,
    cox_partial_loglik,
    fit_cox,
    fit_cox_null,
    harrell_c_many,
    residualize,
    stratified_permutation,
)

# The 40-row tied-time cohort of the R reference tests in test_analysis.py
# (time, event, x, z, s); markers are built from the row number as in
# tests/r_reference/marker_screen_reference.R.
_ROWS = (
    "2,1,0.034,0,A;2,0,1.36,0,B;1,1,1.225,0,B;6,1,-0.51,1,B;6,0,-0.298,0,B;1,0,-0.527,0,B;"
    "1,1,0.57,0,B;8,1,-0.056,0,B;2,1,0.747,1,A;1,0,-1.847,1,B;1,1,1.567,1,A;4,1,-0.096,0,A;"
    "2,0,0.68,1,B;5,1,-0.137,0,B;2,1,-0.379,0,B;1,1,0.463,1,A;3,1,0.825,1,A;2,0,-0.203,1,A;"
    "1,1,-0.153,1,B;2,1,0.686,0,A;3,0,-0.87,0,A;2,1,-1.514,1,A;5,1,0.395,1,A;2,1,-0.671,0,B;"
    "10,1,-1.92,1,B;1,0,-0.814,0,A;1,0,-0.468,1,A;4,0,-1.193,0,A;7,1,-1.492,0,B;1,0,0.037,0,A;"
    "1,1,0.897,1,A;3,0,-0.233,1,A;4,0,-0.744,1,B;1,1,0.385,0,B;3,0,0.717,0,A;1,0,-0.3,1,B;"
    "2,0,0.545,1,A;2,1,1.043,0,A;8,0,-0.207,0,A;7,1,-0.814,0,A"
)

# R survival 3.8.6: coxph(...)$score with iter.max = 0 (see the generator script).
_R_SCORE_TESTS = {
    ("efron", False): {
        "univariate": [0.270180068981, 0.0971158491936, 1.24284982356],
        "adjusted": [0.438091193311, 0.00808789092398, 0.578738601551],
        "loglik_zero": -63.0625110657,
        "clinical_loglik": -58.3239553559,
    },
    ("efron", True): {
        "univariate": [0.164462296728, 0.26459489935, 1.31906819587],
        "adjusted": [0.227652879778, 0.00326815017844, 0.333445899618],
        "loglik_zero": -48.9969326258,
        "clinical_loglik": -44.5221128905,
    },
    ("breslow", False): {
        "univariate": [0.291986075837, 0.0734361036947, 1.10716797364],
        "adjusted": [0.42912411828, 0.00257741544224, 0.499053365123],
        "loglik_zero": -64.8083570301,
        "clinical_loglik": -60.6622018373,
    },
    ("breslow", True): {
        "univariate": [0.252270348129, 0.21018294458, 1.08791159108],
        "adjusted": [0.319889263043, 9.6538963109e-06, 0.334105707864],
        "loglik_zero": -50.2847273583,
        "clinical_loglik": -46.3290891286,
    },
}


def _reference_cohort() -> pd.DataFrame:
    rows = [item.split(",") for item in _ROWS.split(";")]
    frame = pd.DataFrame(
        {
            "time": [float(row[0]) for row in rows],
            "event": [int(row[1]) for row in rows],
            "x": [float(row[2]) for row in rows],
            "z": [int(row[3]) for row in rows],
            "s": [row[4] for row in rows],
        }
    )
    index = np.arange(1, len(frame) + 1, dtype=float)
    frame["m1"] = np.sin(index)
    frame["m2"] = ((7 * index) % 11) - 5
    frame["m3"] = 0.6 * frame["x"] + np.cos(index)
    return frame


def _arrays(frame: pd.DataFrame, stratified: bool):
    strata = pd.factorize(frame["s"])[0] if stratified else None
    return (
        frame["time"].to_numpy(dtype=float),
        frame["event"].to_numpy(dtype=int),
        frame[["x", "z"]].to_numpy(dtype=float),
        frame[["m1", "m2", "m3"]].to_numpy(dtype=float),
        strata,
    )


@pytest.mark.parametrize(("ties", "stratified"), list(itertools.product(["efron", "breslow"], [False, True])))
def test_score_screen_matches_r_score_tests(ties: str, stratified: bool) -> None:
    time, event, clinical, markers, strata = _arrays(_reference_cohort(), stratified)
    expected = _R_SCORE_TESTS[(ties, stratified)]

    empty_null = fit_cox_null(time, event, None, strata, ties)
    assert empty_null.loglik == pytest.approx(expected["loglik_zero"], rel=1e-10)
    univariate = CoxScoreScreen(time, event, null=empty_null, strata=strata, ties=ties).statistics(markers)
    assert univariate.chi2 == pytest.approx(expected["univariate"], rel=1e-8)

    clinical_null = fit_cox_null(time, event, clinical, strata, ties)
    assert clinical_null.converged
    assert clinical_null.loglik == pytest.approx(expected["clinical_loglik"], rel=1e-8)
    adjusted = CoxScoreScreen(time, event, null=clinical_null, Z=clinical, strata=strata, ties=ties).statistics(markers)
    assert adjusted.chi2 == pytest.approx(expected["adjusted"], rel=1e-6)
    assert np.all(adjusted.p_value > 0.0) and np.all(adjusted.p_value <= 1.0)
    assert np.sign(adjusted.z) == pytest.approx(np.sign(adjusted.score))


@pytest.mark.parametrize(("ties", "stratified"), list(itertools.product(["efron", "breslow"], [False, True])))
def test_fit_cox_matches_statsmodels(ties: str, stratified: bool) -> None:
    from statsmodels.duration.hazard_regression import PHReg

    from survival_toolkit.analysis import fit_phreg

    time, event, clinical, markers, strata = _arrays(_reference_cohort(), stratified)
    design = np.column_stack([clinical, markers[:, 0], 100.0 + 5.0 * markers[:, 1]])

    fit = fit_cox(time, event, design, strata, ties)
    reference, converged = fit_phreg(PHReg(time, design, status=event, strata=strata, ties=ties))

    assert fit.converged and converged
    assert fit.beta == pytest.approx(np.asarray(reference.params), rel=1e-6, abs=1e-8)
    assert fit.covariance == pytest.approx(np.asarray(reference.cov_params()), rel=1e-5, abs=1e-9)
    assert fit.loglik == pytest.approx(float(reference.llf), rel=1e-10)


def test_fit_cox_flags_a_separated_covariate_as_running_to_infinity() -> None:
    time = np.arange(1.0, 21.0)
    event = np.ones(20, dtype=int)
    separated = (time <= 10).astype(float)  # every early event has x = 1: the MLE is infinite
    fit = fit_cox(time, event, separated)
    # The likelihood converges (as in R), but the coefficient is flagged as running to infinity.
    assert fit.converged and fit.beta[0] > 5
    assert fit.separated.tolist() == [True]


def test_fit_cox_flags_only_the_coefficient_that_runs_to_infinity() -> None:
    rng = np.random.default_rng(8)
    n = 120
    time = rng.exponential(size=n)
    event = rng.integers(0, 2, size=n)
    carriers = rng.random(n) < 0.15
    event[carriers] = 0  # no carrier has the event, so the carrier coefficient runs to -infinity
    design = np.column_stack([rng.normal(size=n), carriers.astype(float)])
    for ties in ("efron", "breslow"):
        fit = fit_cox(time, event, design, None, ties)
        assert fit.separated.tolist() == [False, True]
        assert fit.beta[1] < -8

    # Regular fits are never flagged: the reference cohort, a strong effect, and a marker without
    # effect on a large scale (its coefficient is near zero, so only the SD-scaled rule can judge it).
    time_r, event_r, clinical, markers, strata = _arrays(_reference_cohort(), True)
    assert not fit_cox(time_r, event_r, np.column_stack([clinical, markers]), strata).separated.any()
    strong = rng.normal(size=n)
    time_s = rng.exponential(np.exp(-1.5 * strong))
    large = 1e5 * rng.normal(size=n) + 3e6
    fit = fit_cox(time_s, np.ones(n, dtype=int), np.column_stack([strong, large]))
    assert fit.converged and not fit.separated.any()
    assert abs(fit.beta[1]) < 1e-5


def test_fit_cox_ends_as_not_converged_when_the_information_turns_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    # A diverging fit on a heavy-tailed RNA-seq marker left NaN in the information matrix, and
    # numpy's least squares then raised, failing a whole genome-wide screen.
    from survival_toolkit import marker_screen

    time, event = np.arange(1.0, 21.0), np.ones(20, dtype=int)
    real = marker_screen._score_and_information_at
    calls = {"count": 0}

    def breaks_on_second_step(*args, **kwargs):
        calls["count"] += 1
        score, information = real(*args, **kwargs)
        return (score, np.full_like(information, np.nan)) if calls["count"] >= 2 else (score, information)

    monkeypatch.setattr(marker_screen, "_score_and_information_at", breaks_on_second_step)
    fit = fit_cox(time, event, np.linspace(-1.0, 1.0, 20))
    assert not fit.converged
    assert np.all(np.isnan(fit.covariance)) and np.all(np.isfinite(fit.beta))


def test_score_equals_the_schoenfeld_residual_sum_of_the_marker() -> None:
    time, event, clinical, markers, strata = _arrays(_reference_cohort(), True)
    null = fit_cox_null(time, event, clinical, strata, "efron")
    stats = CoxScoreScreen(time, event, null=null, Z=clinical, strata=strata).statistics(markers)
    for column in range(markers.shape[1]):
        exog = np.column_stack([clinical, markers[:, column]])
        residuals = _efron_schoenfeld_residuals(exog, time, event, np.append(null.beta, 0.0), strata)
        assert stats.score[column] == pytest.approx(np.nansum(residuals[:, -1]), abs=1e-10)


def test_partial_loglik_of_the_null_fit_matches_statsmodels() -> None:
    time, event, clinical, _, strata = _arrays(_reference_cohort(), True)
    null = fit_cox_null(time, event, clinical, strata, "efron")
    assert cox_partial_loglik(time, event, null.eta, strata, "efron") == pytest.approx(null.loglik, rel=1e-10)


def test_permuted_statistics_equal_statistics_of_permuted_rows() -> None:
    time, event, clinical, markers, strata = _arrays(_reference_cohort(), True)
    null = fit_cox_null(time, event, clinical, strata, "efron")
    screen = CoxScoreScreen(time, event, null=null, Z=clinical, strata=strata)
    rng = np.random.default_rng(3)
    permutations = [stratified_permutation(strata, len(time), rng) for _ in range(4)]
    permuted = screen.permuted_chi2(markers, permutations)
    for row, permutation in zip(permuted, permutations):
        assert row == pytest.approx(screen.statistics(markers[permutation]).chi2, rel=1e-10)
        # Stratified permutations never move a row to another stratum.
        assert np.array_equal(strata[permutation], strata)


def test_adjusted_statistics_ignore_clinical_combinations_and_stratum_shifts() -> None:
    time, event, clinical, markers, strata = _arrays(_reference_cohort(), True)
    null = fit_cox_null(time, event, clinical, strata, "efron")
    screen = CoxScoreScreen(time, event, null=null, Z=clinical, strata=strata)
    base = screen.statistics(markers).chi2
    shifted = markers + clinical @ np.array([[2.0, -1.0, 0.5], [0.3, 4.0, -2.0]]) + 7.0 * strata[:, None]
    assert screen.statistics(shifted).chi2 == pytest.approx(base, rel=1e-8)
    residuals = residualize(markers, clinical, strata)
    assert screen.statistics(residuals).chi2 == pytest.approx(base, rel=1e-8)
    # Residuals are orthogonal to the clinical design within strata.
    assert np.abs(clinical.T @ residuals).max() < 1e-10


def test_constant_and_clinically_collinear_markers_are_flagged() -> None:
    time, event, clinical, markers, strata = _arrays(_reference_cohort(), False)
    null = fit_cox_null(time, event, clinical, None, "efron")
    screen = CoxScoreScreen(time, event, null=null, Z=clinical)
    block = np.column_stack([markers[:, 0], np.full(len(time), 3.0), 2.0 * clinical[:, 0] - clinical[:, 1]])
    stats = screen.statistics(block)
    assert stats.collinear.tolist() == [False, True, True]
    assert np.isnan(stats.chi2[1:]).all() and np.isnan(stats.p_value[1:]).all()
    with pytest.raises(ValueError, match="finite"):
        screen.statistics(np.where(np.arange(len(time))[:, None] == 0, np.nan, markers))


def _brute_force_max_t(observed: np.ndarray, permuted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    threshold = observed - 1e-9 * np.maximum(1.0, np.abs(observed))
    single = (1 + (permuted.max(axis=1)[:, None] >= threshold[None, :]).sum(axis=0)) / (permuted.shape[0] + 1)
    order = np.argsort(-observed, kind="mergesort")
    step = np.empty_like(observed)
    running = 0.0
    for rank, marker in enumerate(order):
        remaining = order[rank:]
        count = int(np.sum(permuted[:, remaining].max(axis=1) >= threshold[marker]))
        running = max(running, (count + 1) / (permuted.shape[0] + 1))
        step[marker] = running
    return single, step


def test_max_t_accumulator_matches_brute_force_westfall_young() -> None:
    rng = np.random.default_rng(11)
    observed = rng.chisquare(1, size=12) + np.linspace(0, 6, 12)
    permuted = rng.chisquare(1, size=(300, 12))
    accumulator = MaxTAccumulator(observed)
    for chunk in np.array_split(permuted, 7):
        accumulator.update(chunk)
    result = accumulator.result()
    single, step = _brute_force_max_t(observed, permuted)
    assert result.n_permutations == 300
    assert result.p_single_step == pytest.approx(single)
    assert result.p_step_down == pytest.approx(step)
    assert np.all(result.p_step_down <= result.p_single_step + 1e-12)


def test_max_t_leaves_invalid_markers_out() -> None:
    accumulator = MaxTAccumulator(np.array([5.0, np.nan, 1.0]))
    accumulator.update(np.array([[0.5, 9.0, 0.2], [6.0, np.nan, 0.1]]))
    result = accumulator.result()
    assert np.isnan(result.p_single_step[1]) and np.isnan(result.p_step_down[1])
    assert result.p_single_step[0] == pytest.approx(2 / 3)


def test_permutation_fdr_q_values_follow_the_threshold_rule() -> None:
    observed = np.array([10.0, 8.0, 1.0, 0.5])
    accumulator = PermutationFdrAccumulator(observed)
    accumulator.update(np.array([[0.1, 0.2, 9.0, 1.5], [0.3, 2.0, 0.4, 0.2]]))
    q = accumulator.q_values()
    # Mean permuted exceedances / observed exceedances at each threshold:
    # FDR(10) = 0/1, FDR(8) = 0.5/2, FDR(1) = 1.5/3, FDR(0.5) = 1.5/4. A q-value is the
    # smallest FDR over thresholds at or below the statistic.
    assert q == pytest.approx([0.0, 0.25, 0.375, 0.375])


def test_permutation_fdr_leaves_markers_without_an_observed_statistic_out() -> None:
    # A collinear marker has no observed statistic but finite permuted ones (its residuals vary);
    # counting them would inflate every q-value, as MaxT already avoids.
    observed = np.array([10.0, 8.0, np.nan, 1.0])
    permuted = np.array([[0.1, 0.2, 50.0, 1.5], [0.3, 2.0, 40.0, 0.2]])
    accumulator = PermutationFdrAccumulator(observed)
    accumulator.update(permuted)
    reference = PermutationFdrAccumulator(observed[[0, 1, 3]])
    reference.update(permuted[:, [0, 1, 3]])
    q = accumulator.q_values()
    assert np.isnan(q[2])
    assert q[[0, 1, 3]] == pytest.approx(reference.q_values())
    with pytest.raises(ValueError, match="one column per marker"):
        accumulator.update(permuted[:, :3])


def test_score_screen_checks_its_null_model_and_permutations() -> None:
    time, event, clinical, markers, strata = _arrays(_reference_cohort(), False)
    clinical_null = fit_cox_null(time, event, clinical)
    # The efficient information needs the design the null model was fitted on.
    with pytest.raises(ValueError, match="null model has 2 coefficient"):
        CoxScoreScreen(time, event, null=clinical_null)
    with pytest.raises(ValueError, match="null model has 2 coefficient"):
        CoxScoreScreen(time, event, null=clinical_null, Z=clinical[:, :1])
    screen = CoxScoreScreen(time, event, null=clinical_null, Z=clinical)
    repeated = np.arange(len(time))
    repeated[1] = 0
    for bad in (repeated, np.arange(len(time) - 1), np.arange(len(time)) + 1):
        with pytest.raises(ValueError, match="every row once"):
            screen.permuted_chi2(markers, [bad])


def test_bh_vector_matches_the_existing_adjustment() -> None:
    from survival_toolkit.analysis import _bh_adjust

    p_values = [0.01, np.nan, 0.04, 0.03, 0.2, 0.001]
    assert bh_vector(p_values) == pytest.approx(np.asarray(_bh_adjust(p_values)), nan_ok=True)


def test_harrell_c_many_matches_harrell_c_index_with_ties() -> None:
    rng = np.random.default_rng(5)
    time = rng.integers(1, 9, size=90).astype(float)
    event = rng.integers(0, 2, size=90)
    risk = np.round(rng.normal(size=(90, 4)), 1)
    many = harrell_c_many(time, event, risk)
    for column in range(risk.shape[1]):
        assert many[column] == pytest.approx(_harrell_c_index(time, event, risk[:, column]), abs=1e-12)
    assert np.isnan(harrell_c_many(time, np.zeros(90, dtype=int), risk)).all()


def _pairwise_c_reference(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """The events x patients implementation harrell_c_many replaced, kept as the definition."""
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


@pytest.mark.parametrize("pairwise_limit", [0, 10**12])
def test_harrell_c_many_equals_the_pairwise_definition_exactly(monkeypatch: pytest.MonkeyPatch, pairwise_limit: int) -> None:
    from survival_toolkit import marker_screen

    # 0 forces the sorted-prefix counts, 10**12 the direct pair comparison.
    monkeypatch.setattr(marker_screen, "_PAIRWISE_LIMIT", pairwise_limit)
    rng = np.random.default_rng(17)
    for trial in range(400):
        n = int(rng.integers(0, 50)) if trial % 4 else int(rng.integers(50, 700))
        n_columns = int(rng.integers(0, 4))
        # Heavy ties in time (a few distinct values, censored and event ties) and in risk.
        time = rng.integers(0, int(rng.integers(1, 6)), size=n).astype(float) if trial % 3 else rng.exponential(size=n)
        event = rng.integers(0, 2, size=n) if trial % 5 else (rng.random(n) < 0.1).astype(int) * 3
        risk = rng.integers(0, int(rng.integers(1, 5)), size=(n, n_columns)).astype(float)
        if trial % 2:
            risk = np.round(rng.normal(size=(n, n_columns)), 1)
        if trial % 6 == 0 and n:
            risk[rng.random((n, n_columns)) < 0.15] = np.nan
            risk[rng.random((n, n_columns)) < 0.05] = np.inf
            risk[rng.random((n, n_columns)) < 0.05] = -0.0
        if trial % 7 == 0 and n:
            time[rng.random(n) < 0.1] = np.nan
        risk_argument = risk[:, 0] if n_columns == 1 and trial % 2 else risk
        expected = _pairwise_c_reference(time, event, risk_argument)
        assert np.array_equal(harrell_c_many(time, event, risk_argument), expected, equal_nan=True), trial

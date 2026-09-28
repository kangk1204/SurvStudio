"""Regression tests for the second review of the statistics in analysis.py: Cox fitting, the
signature search, Kaplan-Meier tests, level ordering, the C-index, and the cohort table.

Reference values marked "R" come from R 4 with survival 3.8.6.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg

import survival_toolkit.analysis as analysis
from survival_toolkit.analysis import compute_cox_analysis, discover_feature_signature
from survival_toolkit.errors import JobCancelledError

# ---------------------------------------------------------------------------------------
# Cox fitting (R4#1, R4#3, R4#9)

# n = 200, 10% binary prevalence, true hazard ratio 10, times rounded to 0.1 (numpy
# default_rng(1)); statsmodels' plain Newton step from 0 overshoots and diverges on it.
_STRONG_RARE_TIMES = (
    "3,1,7.7,0.8,10.5,25.2,8.6,3.1,12.9,0.5,2,1.1,0.7,6.1,6.5,0.6,1.4,4.3,8,5,2.5,20.2,1.2,0.6,2.3,1.4,1.6,1.2,"
    "2.3,2.6,11.4,8.5,22.4,6.5,3.2,6.3,0.6,7.9,1.9,0.1,1.8,7.7,0.7,14,4.1,1.6,5,1.6,3.6,6.1,5.1,0.5,9.4,18.8,0.1,"
    "0.9,0.3,1.9,0.5,6.2,10.6,1,0.6,1,21.8,1.9,0.9,12.6,14.1,7.4,0.2,1.1,26.5,2.1,0.6,2.7,16.3,2.5,2.5,1.1,2.8,"
    "23.7,1.4,1.1,2.4,0.7,11.4,10.2,5.6,3.1,0.3,1,15,0.6,15.4,1.2,1,0.3,2.9,1.8,12.7,3.8,16,6.8,2.1,3.9,1.5,4.1,"
    "2.6,12.1,9.7,1.3,7.6,1.8,4,5.7,3.1,2.8,2.7,1.4,2,6.7,0.2,3.1,0.9,10.2,2.3,1.1,0.7,4,14.1,13.6,3.9,8.2,3.3,"
    "7.9,1.4,8.4,37.4,0.8,1.1,2.1,1.1,1.1,15.9,11.1,1.6,1.8,10.7,4.9,4.6,2.1,2.6,2.6,1.1,1.7,15.2,13.9,1.2,2.2,"
    "1.2,2.8,6,3.2,27.9,1,8.7,6.5,4.2,0.6,14.2,2.2,2.8,8.6,2.8,0.7,2.6,15.4,8,5.4,9.7,1.7,2.9,0.6,0.8,19.3,3.2,"
    "1.2,1.9,3.4,1.3,10.1,5.1,2.9,0.4,8.3,3.7,1.9,13.4,6.9"
)
_STRONG_RARE_EVENTS = (
    "10001111111001101101111110011110100011011011001101111101111111110110101100111110110001001111110001100010"
    "001011011110111110111011011010111101110011000110011011101100011111111001110100111101000010111100"
)
_STRONG_RARE_X = (
    "00000000010000000000000000000000000010010000000000000001000001000000000000010000000001000000010000000000"
    "000000010000000000000000000000000000100000000000000000000000000000000000100000001000000000100000"
)


def _strong_rare_effect_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [float(value) for value in _STRONG_RARE_TIMES.split(",")],
            "event": [int(value) for value in _STRONG_RARE_EVENTS],
            "x": [float(value) for value in _STRONG_RARE_X],
        }
    )


def _phreg(frame: pd.DataFrame, columns: list[str], ties: str = "efron") -> PHReg:
    return PHReg(
        frame["time"].to_numpy(dtype=float),
        frame[columns].to_numpy(dtype=float),
        status=frame["event"].to_numpy(dtype=int),
        ties=ties,
    )


def test_cox_fit_of_a_strong_rare_binary_effect_matches_r() -> None:
    frame = _strong_rare_effect_frame()
    assert frame.shape == (200, 3) and int(frame["x"].sum()) == 13

    # R: coxph(Surv(time, event) ~ x, ties = "efron") converges in 6 iterations.
    results, converged = analysis.fit_phreg(_phreg(frame, ["x"]))
    assert converged
    assert results.params[0] == pytest.approx(2.1493086627, abs=1e-8)
    assert results.bse[0] == pytest.approx(0.3189362445, abs=1e-8)
    assert results.llf == pytest.approx(-540.9875316286, abs=1e-7)
    breslow, converged = analysis.fit_phreg(_phreg(frame, ["x"], ties="breslow"))
    assert converged
    assert breslow.params[0] == pytest.approx(2.1090618193, abs=1e-8)
    assert breslow.bse[0] == pytest.approx(0.3186731908, abs=1e-8)

    row = compute_cox_analysis(frame, "time", "event", ["x"])["results_table"][0]
    assert row["Hazard ratio"] == pytest.approx(8.5789254158, rel=1e-8)
    assert (row["CI lower"], row["CI upper"]) == pytest.approx((4.5914923875, 16.0292025075), rel=1e-8)
    grouped = frame.assign(grp=np.where(frame["x"] == 1, "pos", "neg"))
    assert compute_cox_analysis(grouped, "time", "event", ["grp"])["results_table"][0]["Beta"] == pytest.approx(
        2.1493086627, abs=1e-8
    )
    metrics = analysis._signature_cox_metrics(frame["time"].to_numpy(), frame["event"].to_numpy(), frame["x"].to_numpy() == 1)
    assert metrics["Hazard ratio (signature+ vs -)"] == pytest.approx(8.5789254158, rel=1e-8)


def test_signature_search_keeps_the_hazard_ratio_of_a_strong_rare_rule() -> None:
    # A 12% subgroup with hazard ratio 12: the plain Newton fit lost its hazard ratio, so the
    # true rule could not pass the confidence-interval rule and a noise rule was reported.
    rng = np.random.default_rng(0)
    n = 300
    flag = np.zeros(n, dtype=int)
    flag[rng.choice(n, 36, replace=False)] = 1
    event_time = rng.exponential(10 / np.where(flag == 1, 12.0, 1.0))
    censor_time = rng.exponential(15, n)
    frame = pd.DataFrame(
        {
            "os_months": np.round(np.minimum(event_time, censor_time), 1) + 0.1,
            "os_event": (event_time <= censor_time).astype(int),
            "mut": np.where(flag == 1, "mutant", "wildtype"),
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
        }
    )
    _, _, payload = discover_feature_signature(
        frame, "os_months", "os_event", ["mut", "noise1", "noise2"], max_combination_size=1,
        bootstrap_iterations=0, random_seed=1,
    )
    best = payload["best_split"]
    assert best["Signature"] == 'mut == "mutant"'
    assert best["Hazard ratio (signature+ vs -)"] > 5.0
    assert best["Statistically significant"] is True
    # The upper-tail p-value keeps its size instead of rounding to 0 (1 - chi2.cdf).
    assert 0.0 < best["P value"] < 1e-20


def test_separated_covariate_is_not_converged_and_worded_as_such() -> None:
    time = np.arange(1.0, 21.0)
    frame = pd.DataFrame({"time": time, "event": 1, "x": (time <= 10).astype(float)})
    results, converged = analysis.fit_phreg(_phreg(frame, ["x"]))
    # The log-likelihood converges (as in R), but the coefficient runs to infinity.
    assert not converged
    assert results.mle_retvals["reason"] == "separated"
    assert results.params[0] > 10
    with pytest.raises(ValueError, match="did not converge cleanly") as exc_info:
        compute_cox_analysis(frame, "time", "event", ["x"])
    assert "non-finite estimates" not in str(exc_info.value)


def test_singular_cox_design_is_reported_as_singular() -> None:
    rng = np.random.default_rng(5)
    n = 150
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {"time": rng.exponential(1, n) + 0.01, "event": (rng.random(n) < 0.7).astype(int), "x": x, "x2": 2 * x + 1.0}
    )
    results, converged = analysis.fit_phreg(_phreg(frame, ["x", "x2"]))
    assert not converged and results.mle_retvals["reason"] == "singular_information"
    with pytest.raises(ValueError, match="design matrix is singular") as exc_info:
        compute_cox_analysis(frame, "time", "event", ["x", "x2"])
    assert "non-finite estimates" not in str(exc_info.value)


@pytest.mark.parametrize("error", [AttributeError("'PHReg' object has no attribute 'surv'"), JobCancelledError("stop")])
def test_cox_fit_lets_programming_errors_and_cancellation_through(monkeypatch, error: Exception) -> None:
    frame = _strong_rare_effect_frame()

    def _broken(model):
        raise error

    monkeypatch.setattr(analysis, "fit_phreg", _broken)
    with pytest.raises(type(error)):
        compute_cox_analysis(frame, "time", "event", ["x"])


# ---------------------------------------------------------------------------------------
# Bootstrap and replication support (R4#2, R5#3, R4#10)


def _boundary_signature_frame(n_positive: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    flag = np.zeros(n, dtype=int)
    flag[rng.choice(n, n_positive, replace=False)] = 1
    event_time = rng.exponential(10 / np.where(flag == 1, 5.0, 1.0))
    censor_time = rng.exponential(15, n)
    return pd.DataFrame(
        {
            "os_months": np.minimum(event_time, censor_time) + 0.01,
            "os_event": (event_time <= censor_time).astype(int),
            "mut": np.where(flag == 1, "mutant", "wildtype"),
        }
    )


def test_bootstrap_support_is_not_capped_for_a_rule_at_the_minimum_group_size() -> None:
    # 30 of 300 rows is exactly the minimum group size: the scaled resample minimum skipped
    # about half of the resamples, which then counted as failures (consistency ~0.5 < 0.6).
    _, _, payload = discover_feature_signature(
        _boundary_signature_frame(30), "os_months", "os_event", ["mut"], max_combination_size=1,
        bootstrap_iterations=30, bootstrap_sample_fraction=0.8, random_seed=1,
    )
    assert payload["search_space"]["min_group_size"] == 30
    best = payload["best_split"]
    assert best["Signature"] == 'mut == "mutant"'
    assert best["Bootstrap skipped resamples"] == 0 and best["Bootstrap valid resamples"] == 30
    assert best["Bootstrap HR direction consistency"] >= 0.9
    assert best["Bootstrap support (p<alpha)"] >= 0.9
    assert best["Statistically significant"] is True


def _two_group_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "time": rng.exponential(10, n) + 0.1,
            "event": np.ones(n, dtype=int),
            "g": np.where(np.arange(n) < n // 2, "b", "a"),
        }
    )


_G_IS_B = [{"column": "g", "kind": "categorical_level", "level": "b", "reference": "a", "label": 'g == "b"'}]


def _scripted_folds(monkeypatch, outcomes: list[tuple[float, float, float]]) -> None:
    """Every fold's log-rank test is significant; its hazard ratio and interval follow ``outcomes``."""
    queue = iter(outcomes)
    monkeypatch.setattr(analysis, "_signature_logrank", lambda times, events, mask: (9.0, 0.002))
    monkeypatch.setattr(analysis, "_signature_hazard_ratio", lambda times, events, mask, alpha: next(queue))


_FOLD_OUTCOMES = [(2.0, 1.2, 3.0), (0.5, 0.3, 0.9), (1.1, 0.7, 1.6), (2.5, 1.5, 4.0)]


@pytest.mark.parametrize(("observed", "expected"), [(1.8, 0.5), (0.6, 0.25), (None, 0.75)])
def test_replication_support_counts_folds_significant_in_the_discovery_direction(
    monkeypatch, observed: float | None, expected: float
) -> None:
    _scripted_folds(monkeypatch, list(_FOLD_OUTCOMES))
    metrics = analysis._validation_signature_metrics(
        _two_group_frame(), "time", "event", _G_IS_B, "AND", min_group_size=5, n_iterations=4,
        validation_fraction=0.5, significance_level=0.05, random_seed=1, observed_hazard_ratio=observed,
    )
    assert metrics["Validation valid folds"] == 4 and metrics["Validation skipped folds"] == 0
    assert metrics["Validation support (p<alpha)"] == pytest.approx(expected)
    assert metrics["Validation median HR"] == pytest.approx(1.55)


def test_replication_and_bootstrap_support_divide_by_the_requested_draws(monkeypatch) -> None:
    calls = {"count": 0}
    original = analysis._resample_is_estimable

    def _second_draw_fails(mask_values, events):
        calls["count"] += 1
        return calls["count"] != 2 and original(mask_values, events)

    monkeypatch.setattr(analysis, "_resample_is_estimable", _second_draw_fails)
    _scripted_folds(monkeypatch, [_FOLD_OUTCOMES[0], _FOLD_OUTCOMES[2], _FOLD_OUTCOMES[3]])
    validation = analysis._validation_signature_metrics(
        _two_group_frame(), "time", "event", _G_IS_B, "AND", min_group_size=5, n_iterations=4,
        validation_fraction=0.5, significance_level=0.05, random_seed=1, observed_hazard_ratio=1.8,
    )
    assert validation["Validation valid folds"] == 3 and validation["Validation skipped folds"] == 1
    # Two supporting folds out of four requested (not of the three scored).
    assert validation["Validation support (p<alpha)"] == pytest.approx(0.5)

    calls["count"] = 0
    _scripted_folds(monkeypatch, [_FOLD_OUTCOMES[0], _FOLD_OUTCOMES[1], _FOLD_OUTCOMES[3]])
    bootstrap = analysis._bootstrap_signature_metrics(
        _two_group_frame(), "time", "event", _G_IS_B, "AND", min_group_size=5, n_iterations=4,
        sample_fraction=1.0, random_seed=1, significance_level=0.05, observed_hazard_ratio=1.8,
    )
    assert bootstrap["Bootstrap valid resamples"] == 3 and bootstrap["Bootstrap skipped resamples"] == 1
    assert bootstrap["Bootstrap support (p<alpha)"] == pytest.approx(0.75)
    assert bootstrap["Bootstrap HR direction consistency"] == pytest.approx(0.5)


def test_resample_floor_needs_two_rows_and_an_event_on_each_side() -> None:
    mask = np.array([True, True, False, False, False])
    assert analysis._resample_is_estimable(mask, np.array([1, 0, 1, 0, 0]))
    assert not analysis._resample_is_estimable(mask, np.array([0, 0, 1, 1, 1]))
    assert not analysis._resample_is_estimable(np.array([True, False, False]), np.array([1, 1, 1]))


def _summary_inputs(**best_overrides):
    best_split = {
        "N signature+": 60,
        "Statistically significant": True,
        "Bootstrap support (p<alpha)": 0.9,
        "Bootstrap HR direction consistency": 0.95,
        "Validation support (p<alpha)": None,
        "Permutation p": 0.03,
        "Bootstrap valid resamples": 30,
        "Permutation valid resamples": 40,
        **best_overrides,
    }
    search_space = {
        "truncated": False,
        "permutation_iterations": 40,
        "validation_iterations": 0,
        "bootstrap_iterations": 30,
        "significance_level": 0.05,
        "min_group_size": 20,
        "tested_combinations": 10,
        "significant_signatures": 1,
        "n_rows_analyzed": 300,
    }
    return best_split, search_space


def test_attainable_permutation_p_follows_the_valid_shuffles() -> None:
    best_split, search_space = _summary_inputs()
    cautions = analysis._signature_scientific_summary(best_split, search_space)["cautions"]
    assert not any("smallest attainable permutation p" in caution for caution in cautions)
    # 40 requested but only 15 shuffles produced a statistic: the floor is 1/16 > 0.05.
    best_split, search_space = _summary_inputs(**{"Permutation valid resamples": 15})
    cautions = analysis._signature_scientific_summary(best_split, search_space)["cautions"]
    assert any("With 15 valid permutations the smallest attainable permutation p is 0.0625" in caution for caution in cautions)


# ---------------------------------------------------------------------------------------
# Kaplan-Meier tests (R5#2, R5#4, R5#9)


def _pairs(result: dict) -> dict[str, dict]:
    return {row["Comparison"]: row for row in result["pairwise_table"]}


def test_zero_variance_pair_is_not_testable_and_the_other_tests_match_r() -> None:
    # A and B each hold one patient dying at time 5: their pair has zero log-rank variance and
    # R's survdiff stops with "system is exactly singular"; the global and other pairs are fine.
    frame = pd.DataFrame(
        {
            "time": [5.0, 5.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0],
            "event": [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            "g": ["A", "B"] + ["C"] * 8,
        }
    )
    km = analysis.compute_km_analysis(frame, "time", "event", group_column="g")
    assert km["test"]["chisq"] == pytest.approx(1.10983800869, rel=1e-10)
    assert km["test"]["p_value"] == pytest.approx(0.574118760438, rel=1e-10)
    pairs = _pairs(km)
    untestable = pairs["A vs B"]
    assert untestable["Chi-square"] is None and untestable["P value"] is None and untestable["BH adjusted p"] is None
    assert untestable["Note"] == "Not testable: zero log-rank variance"
    for name in ("A vs C", "B vs C"):
        assert pairs[name]["Chi-square"] == pytest.approx(0.782107545075, rel=1e-10)
        assert pairs[name]["P value"] == pytest.approx(0.376497351651, rel=1e-10)
        assert pairs[name]["Note"] == ""
        # BH over the two testable pairs only.
        assert pairs[name]["BH adjusted p"] == pytest.approx(0.376497351651, rel=1e-10)
    assert any("1 pairwise comparison(s) could not be tested" in caution for caution in km["scientific_summary"]["cautions"])
    fleming = analysis.compute_km_analysis(frame, "time", "event", group_column="g", logrank_weight="fleming_harrington", fh_p=1.0)
    assert fleming["test"]["chisq"] == pytest.approx(0.733740314628, rel=1e-10)


def test_all_tied_events_keep_the_global_test_and_a_singular_global_test_is_reported() -> None:
    frame = pd.DataFrame({"time": [3.0, 3.0, 3.0, 3.0, 9.0], "event": [1, 1, 1, 1, 0], "g": ["A", "A", "B", "B", "C"]})
    km = analysis.compute_km_analysis(frame, "time", "event", group_column="g")
    # R: chisq 4 on 2 df; A vs C and B vs C chisq 2; A vs B cannot be tested.
    assert km["test"]["chisq"] == pytest.approx(4.0, rel=1e-12)
    assert km["test"]["p_value"] == pytest.approx(0.135335283237, rel=1e-10)
    pairs = _pairs(km)
    assert pairs["A vs B"]["P value"] is None
    assert pairs["A vs C"]["Chi-square"] == pytest.approx(2.0) and pairs["A vs C"]["P value"] == pytest.approx(0.15729920705, rel=1e-10)

    singular = analysis.compute_km_analysis(frame.iloc[:4], "time", "event", group_column="g")
    assert singular["test"] is None and singular["test_p_value"] is None and singular["logrank_p"] is None
    assert [row["P value"] for row in singular["pairwise_table"]] == [None]
    summary = singular["scientific_summary"]
    assert "could not be computed" in summary["headline"]
    assert any("global between-group test could not be computed" in caution for caution in summary["cautions"])
    assert singular["curves"] and singular["summary_table"]


def test_log_rank_p_values_are_upper_tails_that_do_not_round_to_zero() -> None:
    frame = pd.DataFrame(
        {
            "time": np.concatenate([np.arange(1.0, 61.0), np.arange(61.0, 121.0), np.arange(121.0, 151.0)]),
            "event": [1] * 120 + [1, 0] * 15,
            "g": ["A"] * 60 + ["B"] * 60 + ["C"] * 30,
        }
    )
    km = analysis.compute_km_analysis(frame, "time", "event", group_column="g")
    # R: survdiff chisq 253.403569097 on 2 df, pchisq(lower.tail = FALSE) 9.42137137286e-56.
    assert km["test"]["chisq"] == pytest.approx(253.403569097, rel=1e-10)
    assert km["test"]["p_value"] == pytest.approx(9.42137137286e-56, rel=1e-8)
    assert km["logrank_p"] == km["test"]["p_value"]
    pairs = _pairs(km)
    assert pairs["A vs B"]["P value"] == pytest.approx(1.35704429185e-33, rel=1e-8)
    assert pairs["A vs C"]["P value"] == pytest.approx(1.7781747131e-19, rel=1e-8)
    assert pairs["A vs B"]["BH adjusted p"] > 0.0


def test_pairwise_log_rank_loop_stops_when_the_request_is_cancelled(monkeypatch) -> None:
    frame = pd.DataFrame({"time": np.arange(1.0, 31.0), "event": [1, 0, 1] * 10, "g": ["A", "B", "C"] * 10})

    def _cancelled() -> None:
        raise JobCancelledError("stop")

    monkeypatch.setattr(analysis, "raise_if_cancelled", _cancelled)
    with pytest.raises(JobCancelledError):
        analysis.compute_km_analysis(frame, "time", "event", group_column="g")


# ---------------------------------------------------------------------------------------
# Level ordering (R5#1, R5#5, R5#6), candidate rules (R4#4), and covariate typing


def _ordering_frame(seed: int = 3, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    er = rng.choice(["ER+", "ER-"], n)
    return pd.DataFrame(
        {
            "time": rng.exponential(np.where(er == "ER+", 2.0, 1.0)) + 0.01,
            "event": (rng.random(n) < 0.7).astype(int),
            "er": er,
            "age_group": rng.choice(["<65", ">=65"], n),
            "sex": rng.choice(["여성", "남성"], n),
            "smoking": rng.choice(["Never", "Former", "Current"], n),
        }
    )


def _ordering_outputs(frame: pd.DataFrame) -> tuple[list[str], list[float], list[str], list[str], dict[str, list[str]]]:
    from survival_toolkit.encoding import ordered_level_labels

    covariates = ["er", "age_group", "sex", "smoking"]
    cox = compute_cox_analysis(frame, "time", "event", covariates)
    km = analysis.compute_km_analysis(frame, "time", "event", group_column="er")
    table = analysis.compute_cohort_table(frame, ["age_group", "sex"], group_column="smoking")
    return (
        [row["Label"] for row in cox["results_table"]],
        [row["Hazard ratio"] for row in cox["results_table"]],
        [row["Group"] for row in km["summary_table"]],
        table["columns"],
        {column: ordered_level_labels(frame[column], column) for column in covariates},
    )


def test_reference_levels_and_group_order_do_not_depend_on_row_order() -> None:
    frame = _ordering_frame()
    labels, hazard_ratios, km_groups, table_columns, levels = _ordering_outputs(frame)
    assert labels == [
        "er: ER+ vs ER-",
        "age_group: >=65 vs <65",
        "sex: 여성 vs 남성",
        "smoking: Former vs Never",
        "smoking: Current vs Never",
    ]
    assert km_groups == ["ER-", "ER+"]
    assert table_columns == ["Variable", "Statistic", "Overall (grouped subset)", "Never", "Former", "Current"]
    assert levels["er"] == ["ER-", "ER+"] and levels["age_group"] == ["<65", ">=65"]
    for column, ascending in itertools.product(["er", "age_group", "sex", "smoking"], [True, False]):
        reordered = frame.sort_values(column, ascending=ascending, kind="mergesort").reset_index(drop=True)
        other_labels, other_ratios, other_groups, other_columns, other_levels = _ordering_outputs(reordered)
        assert (other_labels, other_groups, other_columns, other_levels) == (labels, km_groups, table_columns, levels)
        assert other_ratios == pytest.approx(hazard_ratios, rel=1e-9)


def test_signs_negations_and_non_latin_labels_order_as_written() -> None:
    order = analysis._ordered_reference_categories
    assert order(["ER+", "ER-"], "er") == ["ER-", "ER+"]
    assert order(["HER2-", "HER2+"][::-1], "her2") == ["HER2-", "HER2+"]
    assert order(["PD-L1(+)", "PD-L1(-)"], "pdl1") == ["PD-L1(-)", "PD-L1(+)"]
    assert order(["Signature+", "Signature-"], "auto_signature_group") == ["Signature-", "Signature+"]
    assert order(["EGFR mutated", "EGFR non-mutated"], "egfr") == ["EGFR non-mutated", "EGFR mutated"]
    assert order(["Non-wildtype", "Wildtype"], "idh") == ["Wildtype", "Non-wildtype"]
    assert order(["MGMT methylated", "MGMT unmethylated"], "mgmt") == ["MGMT unmethylated", "MGMT methylated"]
    # Hangul labels no longer normalise to "" (which tied every level and kept the row order).
    assert analysis._normalize_category_text("여성") == ("여성", "여성")
    assert order(["여성", "남성"], "sex") == order(["남성", "여성"], "sex") == ["남성", "여성"]
    assert order([">=65", "<65"], "age") == order(["<65", ">=65"], "age") == ["<65", ">=65"]


def test_numeric_codes_stay_in_numeric_order_before_text_with_unknown_last() -> None:
    order = analysis._ordered_reference_categories
    assert order(["100", "Unknown", "60", "80"], "kps") == ["60", "80", "100", "Unknown"]
    assert order(["Grade 10", "Grade 2", "Grade 1"], "grade") == ["Grade 1", "Grade 2", "Grade 10"]
    rng = np.random.default_rng(9)
    n = 300
    frame = pd.DataFrame(
        {
            "time": rng.exponential(1, n) + 0.01,
            "event": (rng.random(n) < 0.7).astype(int),
            "kps": rng.choice(["60", "70", "80", "90", "100", "Unknown"], n),
            "gleason": rng.choice(["6", "7", "8", "9", "10", "Unknown"], n),
        }
    )
    cox = compute_cox_analysis(frame, "time", "event", ["kps"])
    assert [row["Label"] for row in cox["results_table"]] == [
        "kps: 70 vs 60",
        "kps: 80 vs 60",
        "kps: 90 vs 60",
        "kps: 100 vs 60",
        "kps: Unknown vs 60",
    ]
    km = analysis.compute_km_analysis(frame, "time", "event", group_column="gleason")
    assert [row["Group"] for row in km["summary_table"]] == ["6", "7", "8", "9", "10", "Unknown"]


def test_clinical_codings_and_derived_groups_follow_their_natural_order() -> None:
    from survival_toolkit.sample_data import make_example_dataset

    order = analysis._ordered_reference_categories
    assert order(["Stage IB", "Stage IA2", "Stage IIA", "Stage IA1", "Stage IA3", "Stage IIIB"], "stage") == [
        "Stage IA1",
        "Stage IA2",
        "Stage IA3",
        "Stage IB",
        "Stage IIA",
        "Stage IIIB",
    ]
    assert order(["Stage 2", "Stage 1B", "Stage 1A"], "stage") == ["Stage 1A", "Stage 1B", "Stage 2"]
    assert order(["Current", "Former", "Never"], "smoking_status") == ["Never", "Former", "Current"]
    assert order(["Ex-smoker", "Current smoker", "Non-smoker"], "smoking") == ["Non-smoker", "Ex-smoker", "Current smoker"]
    assert order(["Current smoker", "Current reformed smoker for > 15 years", "Lifelong Non-smoker"], "tobacco_history") == [
        "Lifelong Non-smoker",
        "Current reformed smoker for > 15 years",
        "Current smoker",
    ]
    assert order(["High", "Intermediate", "Low"], "risk") == ["Low", "Intermediate", "High"]
    assert order(["High risk", "Low risk"], "risk_group") == ["Low risk", "High risk"]
    assert order(["High", "Low", "Normal"], "potassium") == ["Normal", "Low", "High"]

    df = make_example_dataset(seed=11, n_patients=240)
    median, median_column, _ = analysis.derive_group_column(df, "biomarker_score", "median_split")
    assert [row["Label"] for row in compute_cox_analysis(median, "os_months", "os_event", [median_column])["results_table"]] == [
        f"{median_column}: High vs Low"
    ]
    groups = {
        method_cutoff: analysis.compute_km_analysis(
            analysis.derive_group_column(df, "biomarker_score", method_cutoff[0], cutoff=method_cutoff[1])[0],
            "os_months",
            "os_event",
            group_column=f"biomarker_score__{method_cutoff[0]}",
        )["summary_table"]
        for method_cutoff in [("percentile_split", "25,25"), ("percentile_split", "25"), ("quartile_split", None)]
    }
    assert [row["Group"] for row in groups[("percentile_split", "25,25")]] == [
        "At/below 25th percentile threshold",
        "Between percentile thresholds",
        "At/above 75th percentile threshold",
    ]
    assert [row["Group"] for row in groups[("percentile_split", "25")]] == ["Rest", "At/above 75th percentile threshold"]
    assert [row["Group"] for row in groups[("quartile_split", None)]] == ["Q1", "Q2", "Q3", "Q4"]


def test_most_common_level_is_a_candidate_rule_only_beyond_two_levels() -> None:
    rng = np.random.default_rng(0)
    n = 300
    mutation = rng.choice(["wildtype", "missense", "truncating"], size=n, p=[0.6, 0.25, 0.15])
    frame = pd.DataFrame({"mut": mutation, "flag": np.where(np.arange(n) % 2 == 0, "b", "a")})
    indicators = analysis._build_candidate_indicators(frame, ["mut", "flag"], min_group_size=30)
    labels = [indicator["label"] for indicator in indicators]
    # "Wild type vs any mutation" is a rule of its own when there are three levels.
    assert labels == ['mut == "wildtype"', 'mut == "missense"', 'mut == "truncating"', 'flag == "b"']
    assert [indicator["reference"] for indicator in indicators] == ["missense", "wildtype", "wildtype", "a"]
    # Equal counts: the reference follows the level order, not the order of the rows.
    reversed_rows = analysis._build_candidate_indicators(frame.iloc[::-1], ["flag"], min_group_size=30)
    assert [indicator["label"] for indicator in reversed_rows] == ['flag == "b"']


def test_numbers_stored_as_text_are_numeric_cox_covariates() -> None:
    from survival_toolkit.sample_data import make_example_dataset

    df = make_example_dataset(seed=23, n_patients=160)
    df["age"] = (df["age"] // 10) * 10.0
    as_text = df.assign(age=[f" {value:.0f} " for value in df["age"]])
    assert not pd.api.types.is_numeric_dtype(as_text["age"])
    numeric_fit = compute_cox_analysis(df, "os_months", "os_event", ["age", "stage"])
    text_fit = compute_cox_analysis(as_text, "os_months", "os_event", ["age", "stage"])
    assert text_fit["categorical_covariates"] == ["stage"]
    assert [row["Label"] for row in text_fit["results_table"]] == [row["Label"] for row in numeric_fit["results_table"]]
    assert [row["Beta"] for row in text_fit["results_table"]] == pytest.approx(
        [row["Beta"] for row in numeric_fit["results_table"]], rel=1e-10
    )
    # Declared categorical, a pandas categorical dtype, or text that is not all numbers stays categorical.
    preview = analysis.preview_cox_analysis_inputs
    assert preview(as_text, "os_months", "os_event", ["age", "stage"])["categorical_covariates"] == ["stage"]
    declared = preview(as_text, "os_months", "os_event", ["age", "stage"], categorical_covariates=["age"])
    assert declared["categorical_covariates"] == ["age", "stage"]
    coded = df.assign(age=pd.Categorical(df["age"]))
    assert preview(coded, "os_months", "os_event", ["age", "stage"])["categorical_covariates"] == ["age", "stage"]
    mixed = as_text.assign(age=as_text["age"].where(df.index % 7 != 0, "unknown"))
    assert preview(mixed, "os_months", "os_event", ["age", "stage"])["categorical_covariates"] == ["age", "stage"]


# ---------------------------------------------------------------------------------------
# Median survival and median follow-up (R4#5)


@pytest.mark.parametrize(
    ("times", "events", "r_median"),
    [
        # R: summary(survfit(Surv(times, events) ~ 1))$table["median"].
        ([1, 2, 3, 4], [1, 1, 1, 1], 2.5),
        ([1, 2, 3, 4], [1, 1, 0, 0], 2.0),
        ([1, 2, 3, 4, 5], [1, 1, 0, 1, 0], 4.0),
        ([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1], 3.5),
        ([1, 1, 2, 2], [1, 1, 1, 1], 1.5),
        ([1, 2, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1], 2.5),
        ([1, 2, 3, 4, 5, 6], [1, 1, 1, 0, 1, 1], 4.0),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 0, 1, 0, 1, 1, 0, 1, 1, 1], 8.0),
    ],
)
def test_km_median_matches_r_survfit(times: list[int], events: list[int], r_median: float) -> None:
    frame = pd.DataFrame({"t": np.asarray(times, dtype=float), "e": events})
    km = analysis.compute_km_analysis(frame, "t", "e")
    assert km["summary_table"][0]["Median survival"] == pytest.approx(r_median, abs=1e-12)


def test_median_follow_up_uses_the_same_rule_on_the_reverse_curve() -> None:
    # R: survfit(Surv(t, 1 - e) ~ 1) medians 3.5 and 2.
    follow_up = analysis._median_follow_up
    assert follow_up(pd.Series(np.arange(1.0, 7.0)), pd.Series([0, 0, 0, 0, 1, 1])) == pytest.approx(3.5)
    assert follow_up(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([0, 0, 1, 1])) == pytest.approx(2.0)
    km = analysis.compute_km_analysis(pd.DataFrame({"t": np.arange(1.0, 7.0), "e": [0, 0, 0, 0, 1, 1]}), "t", "e")
    assert km["cohort"]["median_follow_up"] == pytest.approx(3.5)

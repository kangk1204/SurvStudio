"""Regression tests for the review of the marker evaluation and its locked-model validation."""

from __future__ import annotations

import copy
import json
import threading

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.marker_evaluation as marker_evaluation
from survival_toolkit.concurrency import cancellation_scope
from survival_toolkit.errors import InternalAnalysisError, JobCancelledError, UserInputError
from survival_toolkit.marker_evaluation import (
    MarkerSettings,
    _legacy_recipe_hash,
    evaluate_markers,
    prepare_marker_cohort,
    recipe_hash,
    validate_locked_recipe,
)
from survival_toolkit.marker_screen import fit_cox

_QUICK = MarkerSettings(n_permutations=49, n_resamples=8, shortlist_size=10, random_seed=3)


def _gene_cohort(seed: int, n: int = 400, offset: float = 0.0) -> pd.DataFrame:
    """Three prognostic genes among six; ``offset`` puts them on a log2-like scale (values near 10)."""
    rng = np.random.default_rng(seed)
    genes = rng.normal(size=(n, 6))
    linear = genes[:, :3] @ np.array([-0.8, -0.7, 0.6])
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(genes + offset, columns=[f"g{index}" for index in range(6)])
    frame["time"] = np.minimum(event_time, censor_time)
    frame["event"] = (event_time <= censor_time).astype(int)
    return frame


def _clinical_cohort(seed: int, n: int = 300, n_noise: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    grade = rng.integers(1, 4, size=n)
    true = rng.normal(size=n)
    linear = 0.7 * age + 0.6 * (grade - 2) + 0.7 * true
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame({"os_time": np.minimum(event_time, censor_time), "os_event": (event_time <= censor_time).astype(int)})
    frame["age"] = age
    frame["grade"] = grade
    frame["true"] = true
    for index in range(n_noise):
        frame[f"noise_{index}"] = rng.normal(size=n)
    return frame


_GENES = [f"g{index}" for index in range(6)]


def _markers(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column == "true" or column.startswith("noise_")]


def _gene_recipe(offset: float) -> dict:
    result = evaluate_markers(_gene_cohort(1, offset=offset), time_column="time", event_column="event", marker_columns=_GENES, settings=_QUICK)
    return result["locked_recipe"]


def _through_javascript(value):
    """JSON.parse then JSON.stringify in a browser: whole numbers lose their decimal point (1e21 and up excepted)."""

    def parse_float(text: str):
        number = float(text)
        return int(number) if number.is_integer() and abs(number) < 1e21 else number

    return json.loads(json.dumps(value), parse_float=parse_float)


# A1: the locked baseline hazard is centred, so absolute risks survive markers far from zero.


def test_absolute_risks_do_not_depend_on_where_the_markers_are_centred() -> None:
    recipe_zero, recipe_log2 = _gene_recipe(0.0), _gene_recipe(10.0)
    assert recipe_log2["recipe_version"] == 2
    baseline = recipe_log2["model"]["baseline"]
    coefficients = dict(zip(recipe_log2["model"]["terms"], recipe_log2["model"]["coefficients"]))
    # The centre is the development mean linear predictor, about -9 for these log2-like markers.
    assert baseline["lp_center"] == pytest.approx(sum(10.0 * coefficients[name] for name in recipe_log2["markers"]), abs=0.5)
    assert np.all(np.isfinite(baseline["log_cumulative_hazard"])) and np.all(np.diff(baseline["log_cumulative_hazard"]) >= 0)

    at_zero = validate_locked_recipe(_gene_cohort(2), recipe_zero, n_bootstrap=0)["metrics"]
    at_log2 = validate_locked_recipe(_gene_cohort(2, offset=10.0), recipe_log2, n_bootstrap=0)["metrics"]
    for key in ("c_index", "expected_risk", "observed_expected_ratio", "brier", "brier_skill"):
        assert at_log2[key] == pytest.approx(at_zero[key], rel=1e-6), key
    assert 0.8 < at_log2["observed_expected_ratio"] < 1.25 and at_log2["brier_skill"] > 0.1


def test_version_1_recipes_keep_their_baseline_and_withhold_risks_it_lost() -> None:
    for offset, precise in ((0.0, True), (10.0, False)):
        recipe = _gene_recipe(offset)
        current = validate_locked_recipe(_gene_cohort(2, offset=offset), recipe, n_bootstrap=0)
        legacy = copy.deepcopy(recipe)
        baseline = legacy["model"]["baseline"]
        # Version 1 stored S0(t) at a linear predictor of 0 and hashed Python's JSON text.
        hazard_at_zero = np.exp(np.asarray(baseline["log_cumulative_hazard"]) - baseline["lp_center"])
        legacy["model"]["baseline"] = {"times": baseline["times"], "survival": np.exp(-hazard_at_zero).tolist()}
        legacy["recipe_version"] = 1
        legacy["recipe_hash"] = _legacy_recipe_hash(legacy)
        report = validate_locked_recipe(_gene_cohort(2, offset=offset), legacy, n_bootstrap=0)
        assert report["recipe_version"] == 1
        assert report["metrics"]["c_index"] == pytest.approx(current["metrics"]["c_index"])
        if precise:
            assert report["metrics"]["expected_risk"] == pytest.approx(current["metrics"]["expected_risk"], rel=1e-9)
        else:
            # exp(-cumulative hazard at lp 0) underflowed to 0: the old "expected risk 1.0" is withheld.
            assert "expected_risk" not in report["metrics"] and "brier" not in report["metrics"]
            assert any("lost its precision" in note for note in report["notes"])
        # A version-1 recipe must match the version-1 hash.
        legacy["recipe_hash"] = recipe_hash(legacy)
        with pytest.raises(UserInputError, match="edited after it was locked"):
            validate_locked_recipe(_gene_cohort(2, offset=offset), legacy, n_bootstrap=0)


# A5: the hash survives the browser's JSON round trip.


def test_a_recipe_sent_back_by_the_browser_still_validates() -> None:
    frame = _clinical_cohort(4)
    frame["os_time"] = (frame["os_time"] * 365).round() + 1  # whole days, as in TCGA
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame),
        clinical_columns=["age", "grade"], categorical_clinical=["grade"], settings=_QUICK,
    )
    recipe = result["locked_recipe"]
    browser = _through_javascript(recipe)
    # Whole numbers such as the horizon lost their ".0", which changed the version-1 hash.
    assert json.dumps(browser) != json.dumps(recipe)
    assert _legacy_recipe_hash(browser) != _legacy_recipe_hash(recipe)
    assert recipe_hash(browser) == recipe["recipe_hash"]
    external = _clinical_cohort(5)
    external["os_time"] = (external["os_time"] * 365).round() + 1
    assert validate_locked_recipe(external, browser, n_bootstrap=0)["recipe_hash"] == recipe["recipe_hash"]
    # Any real edit still breaks the hash.
    browser["model"]["coefficients"][0] += 0.01
    with pytest.raises(UserInputError, match="edited after it was locked"):
        validate_locked_recipe(external, browser, n_bootstrap=0)


# A14: a malformed recipe fails with a clear message, whatever its hash.


def _add_marker_outside_the_model(recipe: dict) -> None:
    recipe["markers"].append("not_a_term")
    recipe["marker_medians"]["not_a_term"] = 0.0
    recipe["marker_development_log_hr"]["not_a_term"] = 0.1


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda recipe: recipe["model"].__setitem__("coefficients", recipe["model"]["coefficients"][:-1]), "one per term"),
        (lambda recipe: recipe["model"]["coefficients"].__setitem__(0, "x"), "finite number"),
        (lambda recipe: _add_marker_outside_the_model(recipe), "not model terms"),
        (lambda recipe: recipe["marker_medians"].pop(recipe["markers"][0]), "marker_medians"),
        (lambda recipe: recipe["model"]["baseline"].__setitem__("lp_center", None), "lp_center"),
        (lambda recipe: recipe["model"].__setitem__("ties", "exact"), "model.ties"),
        (lambda recipe: recipe["clinical"].__setitem__("columns", "age"), "clinical.columns"),
    ],
)
def test_malformed_recipes_are_refused_with_a_clear_message(edit, message) -> None:
    frame = _clinical_cohort(6)
    recipe = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"], settings=_QUICK,
    )["locked_recipe"]
    bad = copy.deepcopy(recipe)
    edit(bad)
    bad["recipe_hash"] = recipe_hash(bad)
    with pytest.raises(UserInputError, match=message):
        validate_locked_recipe(_clinical_cohort(7), bad, n_bootstrap=0)


def test_recipes_without_a_usable_version_or_with_non_finite_numbers_are_refused() -> None:
    frame = _clinical_cohort(6)
    recipe = evaluate_markers(frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), settings=_QUICK)["locked_recipe"]
    for version in (None, "2", 3, True):
        bad = {**copy.deepcopy(recipe), "recipe_version": version}
        with pytest.raises(UserInputError, match="incompatible SurvStudio version"):
            validate_locked_recipe(_clinical_cohort(7), bad, n_bootstrap=0)
    bad = copy.deepcopy(recipe)
    bad["model"]["coefficients"][0] = float("nan")
    with pytest.raises(UserInputError, match="finite number"):
        validate_locked_recipe(_clinical_cohort(7), bad, n_bootstrap=0)
    with pytest.raises(ValueError, match="not finite"):
        recipe_hash(bad)
    with pytest.raises(UserInputError, match="JSON object"):
        validate_locked_recipe(_clinical_cohort(7), ["not", "a", "recipe"], n_bootstrap=0)


# A2: categorical levels the locked model has never seen stop the validation.


def test_numeric_coded_levels_that_do_not_match_stop_validation_loudly() -> None:
    frame = _clinical_cohort(8)
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame),
        clinical_columns=["age", "grade"], categorical_clinical=["grade"], settings=_QUICK,
    )
    recipe = result["locked_recipe"]
    external = _clinical_cohort(9)
    relabelled = external.assign(grade=external["grade"].map({1: "G1", 2: "G2", 3: "G3"}))
    with pytest.raises(UserInputError, match=r"300 of 300 external rows have a grade level the locked model has not seen \('G1', 'G2', 'G3'"):
        validate_locked_recipe(relabelled, recipe, n_bootstrap=0)
    # One missing grade makes pandas read the column as floats (1.0 for 1): either the encoder matches
    # them to the development levels, or validation stops; it never scores them all as the reference.
    as_float = external.assign(grade=external["grade"].astype(float))
    as_float.loc[0, "grade"] = np.nan
    try:
        report = validate_locked_recipe(as_float, recipe, n_bootstrap=0)
    except UserInputError as exc:
        assert "level the locked model has not seen" in str(exc)
    else:
        reference = validate_locked_recipe(external.iloc[1:], recipe, n_bootstrap=0)
        assert report["metrics"]["c_index"] == pytest.approx(reference["metrics"]["c_index"])
    # A few unseen rows are scored as the reference level, with a note.
    few = external.assign(grade=external["grade"].astype(object))
    few.loc[few.index[:5], "grade"] = "4"
    report = validate_locked_recipe(few, recipe, n_bootstrap=0)
    assert any("5 external row(s) have a grade level not seen" in note for note in report["notes"])


# A9 and A10: overlapping roles are refused; redundant clinical columns are left out and reported.


def test_a_column_cannot_be_both_clinical_covariate_and_stratum() -> None:
    frame = _clinical_cohort(10)
    common = dict(time_column="os_time", event_column="os_event", marker_columns=_markers(frame), settings=_QUICK)
    with pytest.raises(UserInputError, match="both a clinical covariate and a stratum: grade"):
        evaluate_markers(frame, clinical_columns=["age", "grade"], strata_columns=["grade"], **common)
    with pytest.raises(UserInputError, match="Clinical covariates cannot also be the outcome"):
        evaluate_markers(frame, clinical_columns=["os_time"], **common)
    with pytest.raises(UserInputError, match="Strata cannot also be the outcome"):
        evaluate_markers(frame, strata_columns=["os_event"], **common)


def test_clinical_columns_a_cox_model_cannot_estimate_are_left_out_and_noted() -> None:
    frame = _clinical_cohort(11)
    frame["centre"] = np.where(np.arange(len(frame)) < 150, "A", "B")
    frame["centre_b"] = (frame["centre"] == "B").astype(float)  # constant within each stratum
    frame["age_copy"] = 2.0 * frame["age"] + 1.0  # aliased with age
    common = dict(time_column="os_time", event_column="os_event", marker_columns=_markers(frame), settings=_QUICK)

    result = evaluate_markers(frame, clinical_columns=["age", "centre_b", "age_copy"], strata_columns=["centre"], **common)
    reference = evaluate_markers(frame, clinical_columns=["age"], strata_columns=["centre"], **common)

    assert result["cohort"]["dropped_clinical_columns"] == [
        {"column": "centre_b", "reason": "constant within every stratum"},
        {"column": "age_copy", "reason": "a linear combination of other clinical columns"},
    ]
    assert result["cohort"]["clinical_design_columns"] == ["age"]
    assert any("centre_b was left out of the clinical model" in note for note in result["cohort"]["notes"])
    # Leaving them out is the same analysis as never including them.
    assert result["marker_table"] == reference["marker_table"]
    assert result["locked_recipe"]["model"]["terms"] == reference["locked_recipe"]["model"]["terms"]


# A20: missing values count at the median they are imputed with.


def test_the_near_constant_filter_counts_missing_values_at_their_imputed_median() -> None:
    frame = _clinical_cohort(12, n=105)
    rare = np.zeros(105)
    rare[:10] = 1.0
    rare[100:] = np.nan  # 90 zeros and 10 ones observed (90% at zero), 5 missing imputed as 0
    frame["rare"] = rare
    cohort = prepare_marker_cohort(frame, time_column="os_time", event_column="os_event", marker_columns=["true", "rare"])
    assert cohort.marker_names == ["true"]
    assert cohort.dropped_markers == [{"marker": "rare", "reason": "near-constant (90% at one value)"}]
    observed_only = frame.dropna(subset=["rare"])
    kept = prepare_marker_cohort(observed_only, time_column="os_time", event_column="os_event", marker_columns=["true", "rare"])
    assert kept.marker_names == ["true", "rare"]


# A11: stability that was never assessed cannot make a marker robust.


def test_markers_are_not_robust_when_no_subsample_was_evaluated() -> None:
    frame = _clinical_cohort(13)
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"],
        settings=_QUICK._replace(n_resamples=0, n_permutations=99),
    )
    assert result["tier_counts"]["robust"] == 0
    assert {row["marker"]: row["tier"] for row in result["marker_table"]}["true"] == "suggestive"
    assert result["resampling"]["stability_assessed"] is False and "not assessed" in result["resampling"]["note"]


def test_only_data_failures_of_a_subsample_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _clinical_cohort(14, n=200)
    settings = _QUICK._replace(n_resamples=4, n_permutations=9)
    original = marker_evaluation.run_procedure
    n = len(frame)

    def failing_with(error):
        def run(cohort, rows, run_settings):
            if rows.size < n:
                error()
            return original(cohort, rows, run_settings)

        return run

    def data_failure():
        # The one data failure of a subsample: its clinical-only model does not converge.
        raise marker_evaluation.ClinicalModelNotConvergedError("The clinical-only Cox model did not converge.")

    monkeypatch.setattr(marker_evaluation, "run_procedure", failing_with(data_failure))
    result = evaluate_markers(frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), settings=settings)
    assert result["resampling"]["n_failed"] == 4 and result["tier_counts"]["robust"] == 0

    def library_bug():
        np.zeros(3) + np.zeros(4)

    def programming_bug():
        {}["missing"]

    for error, expected in ((library_bug, InternalAnalysisError), (programming_bug, KeyError)):
        monkeypatch.setattr(marker_evaluation, "run_procedure", failing_with(error))
        with pytest.raises(expected):
            evaluate_markers(frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), settings=settings)


# A3: coefficients that run to infinity are not estimates.


def _separated_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120
    frame = pd.DataFrame({"os_time": rng.exponential(10, n) + 0.1, "os_event": rng.integers(0, 2, n)})
    mutation = np.zeros(n)
    mutation[np.flatnonzero(frame["os_event"].to_numpy() == 0)[:15]] = 1.0  # no carrier has an event
    frame["mut"] = mutation
    frame["g1"] = rng.normal(size=n)
    frame["g2"] = rng.normal(size=n)
    return frame


def test_a_marker_whose_coefficient_runs_to_infinity_gets_no_hazard_ratio() -> None:
    frame = _separated_frame(4)
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=["mut", "g1", "g2"],
        settings=MarkerSettings(n_permutations=19, n_resamples=4, random_seed=1),
    )
    rows = {row["marker"]: row for row in result["marker_table"]}
    assert rows["mut"]["exact"]["marginal"] is None
    assert rows["g1"]["exact"]["marginal"]["hazard_ratio"] > 0
    # Nothing infinite reaches the JSON response.
    json.dumps(result, allow_nan=False)


def test_left_out_log_hazard_ratios_leave_out_fits_that_run_to_infinity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    original = marker_evaluation._optimism_summary

    def spy(c_in, c_out, clinical_c_out, beta_in, beta_out, *sizes):
        captured.update(beta_out=list(beta_out))
        return original(c_in, c_out, clinical_c_out, beta_in, beta_out, *sizes)

    monkeypatch.setattr(marker_evaluation, "_optimism_summary", spy)
    rng = np.random.default_rng(102)
    n = 70
    mutations = (rng.random((n, 15)) < 0.13).astype(float)
    event_time = rng.exponential(size=n)
    censor_time = rng.exponential(1.0, n)
    frame = pd.DataFrame(mutations, columns=[f"mut{index}" for index in range(15)])
    frame["time"] = np.minimum(event_time, censor_time)
    frame["event"] = (event_time <= censor_time).astype(int)
    result = evaluate_markers(
        frame, time_column="time", event_column="event", marker_columns=[f"mut{index}" for index in range(15)],
        settings=MarkerSettings(n_permutations=19, n_resamples=100, random_seed=2),
    )
    # Before, a fifth of these left-out fits ran to |beta| of about 20 and set the mean to about -4.5.
    assert captured["beta_out"] and max(abs(value) for value in captured["beta_out"]) < 8
    assert abs(result["signature"]["top_marker_log_hr_left_out"]) < 2


def test_single_marker_betas_of_separated_fits_are_missing() -> None:
    frame = _separated_frame(5)
    cohort = prepare_marker_cohort(frame, time_column="os_time", event_column="os_event", marker_columns=["mut", "g1"])
    rows = np.arange(len(frame))
    medians = np.nanmedian(cohort.markers, axis=0)
    assert np.isnan(marker_evaluation._single_marker_beta(cohort, rows, 0, medians, adjusted=False))
    assert np.isfinite(marker_evaluation._single_marker_beta(cohort, rows, 1, medians, adjusted=False))


# A12: an added-value replication that cannot be estimated is not replaced by the marginal test.


def test_an_added_value_replication_that_cannot_be_fitted_is_not_estimable(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _clinical_cohort(15)
    recipe = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"], settings=_QUICK,
    )["locked_recipe"]
    assert "true" in recipe["markers"]
    external = _clinical_cohort(16)
    fitted = validate_locked_recipe(external, recipe, n_bootstrap=0)
    assert {row["tested"] for row in fitted["markers"]} == {"added_value"}

    original = marker_evaluation._external_cox

    def adjusted_fails(time, event, exog, strata, ties="efron"):
        return None if exog.shape[1] > 1 else original(time, event, exog, strata, ties)

    monkeypatch.setattr(marker_evaluation, "_external_cox", adjusted_fails)
    report = validate_locked_recipe(external, recipe, n_bootstrap=0)
    row = next(row for row in report["markers"] if row["marker"] == "true")
    assert row["marginal"] is not None and row["adjusted"] is None
    assert row["tested"] is None and row["replication_p_holm"] is None and not row["replicated"]
    assert any("added-value replication of true is not estimable" in note for note in report["notes"])


# A13 and J9: the bootstrap stops on cancellation and also gives the clinical-only interval.


def test_validation_bootstrap_reports_the_clinical_only_interval_and_stops_when_cancelled() -> None:
    frame = _clinical_cohort(17)
    recipe = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"], settings=_QUICK,
    )["locked_recipe"]
    external = _clinical_cohort(18)
    metrics = validate_locked_recipe(external, recipe, n_bootstrap=60)["metrics"]
    low, high = metrics["clinical_only_c_index_ci"]
    assert low <= metrics["clinical_only_c_index"] <= high and low < high
    stop = threading.Event()
    stop.set()
    with cancellation_scope(stop), pytest.raises(JobCancelledError):
        validate_locked_recipe(external, recipe, n_bootstrap=60)


# A17: the added-value null is the Smith scheme; the old name still works.


def test_the_added_value_null_is_called_smith_and_the_old_name_is_an_alias() -> None:
    assert MarkerSettings().lens2_null == "smith"
    frame = _clinical_cohort(19, n=200)
    common = dict(time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"])
    smith = evaluate_markers(frame, settings=_QUICK._replace(lens2_null="smith"), **common)
    alias = evaluate_markers(frame, settings=_QUICK._replace(lens2_null="freedman_lane"), **common)
    assert smith["null"]["lens2_null"] == alias["null"]["lens2_null"] == alias["settings"]["lens2_null"] == "smith"
    assert alias["marker_table"] == smith["marker_table"]
    with pytest.raises(UserInputError, match='"smith" or "raw"'):
        evaluate_markers(frame, settings=_QUICK._replace(lens2_null="shuffle"), **common)


# A21: the tie method reaches the exact fits and the locked model.


def test_breslow_ties_reach_the_exact_fits_and_the_locked_model() -> None:
    frame = _clinical_cohort(20)
    frame["os_time"] = np.ceil(frame["os_time"] * 4)  # heavy ties, so Efron and Breslow differ
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"],
        settings=_QUICK._replace(ties="breslow"),
    )
    assert result["locked_recipe"]["model"]["ties"] == "breslow"
    row = next(row for row in result["marker_table"] if row["marker"] == "true")
    exog = np.column_stack([frame["age"], frame["true"]])
    breslow = fit_cox(frame["os_time"].to_numpy(), frame["os_event"].to_numpy(), exog, None, "breslow")
    efron = fit_cox(frame["os_time"].to_numpy(), frame["os_event"].to_numpy(), exog, None, "efron")
    assert row["exact"]["adjusted"]["hazard_ratio"] == pytest.approx(float(np.exp(breslow.beta[-1])), rel=1e-10)
    assert abs(breslow.beta[-1] - efron.beta[-1]) > 1e-4


# G6 and G8: a clinical-only model is labelled, and the left-out gain is paired.


def test_a_model_without_selected_markers_is_labelled_clinical_only() -> None:
    rng = np.random.default_rng(21)
    n = 200
    age = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-1.0 * age))
    censor_time = rng.exponential(1.5, n)
    frame = pd.DataFrame({"os_time": np.minimum(event_time, censor_time), "os_event": (event_time <= censor_time).astype(int), "age": age})
    for index in range(4):
        frame[f"noise_{index}"] = rng.normal(size=n)
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=[f"noise_{index}" for index in range(4)],
        clinical_columns=["age"], settings=_QUICK,
    )
    signature = result["signature"]
    assert signature["markers"] == [] and signature["clinical_only"] is True
    assert any("clinical covariates alone" in note for note in signature["notes"])
    # The locked clinical model validates with a gain over the clinical covariates of exactly zero.
    recipe = result["locked_recipe"]
    assert recipe["markers"] == [] and recipe["clinical_only_model"]["terms"] == recipe["model"]["terms"]
    external = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
    metrics = validate_locked_recipe(external, recipe, n_bootstrap=50)["metrics"]
    assert metrics["clinical_only_c_index"] == pytest.approx(metrics["c_index"])
    assert metrics["delta_c_index"] == pytest.approx(0.0)
    selected = evaluate_markers(
        _clinical_cohort(22), time_column="os_time", event_column="os_event", marker_columns=_markers(_clinical_cohort(22)),
        clinical_columns=["age"], settings=_QUICK,
    )
    assert selected["signature"]["clinical_only"] is False and "true" in selected["signature"]["markers"]


def test_the_left_out_clinical_c_is_paired_with_every_signature_replicate(monkeypatch: pytest.MonkeyPatch) -> None:
    lengths: dict = {}
    original = marker_evaluation._optimism_summary

    def spy(c_in, c_out, clinical_c_out, beta_in, beta_out, *sizes):
        lengths.update(signature=len(c_out), clinical=len(clinical_c_out))
        return original(c_in, c_out, clinical_c_out, beta_in, beta_out, *sizes)

    monkeypatch.setattr(marker_evaluation, "_optimism_summary", spy)
    rng = np.random.default_rng(3)
    n = 240
    age = rng.normal(size=n)
    markers = rng.normal(size=(n, 8))
    event_time = rng.exponential(np.exp(-(0.9 * age + 0.25 * markers[:, 0])))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame({"t": np.minimum(event_time, censor_time), "e": (event_time <= censor_time).astype(int), "age": age})
    for index in range(8):
        frame[f"m{index}"] = markers[:, index]
    result = evaluate_markers(
        frame, time_column="t", event_column="e", marker_columns=[f"m{index}" for index in range(8)], clinical_columns=["age"],
        settings=MarkerSettings(n_permutations=19, n_resamples=40, random_seed=1),
    )
    signature = result["signature"]
    # A weak marker is selected in some subsamples only; before, 18 of 40 replicates had a clinical C.
    assert lengths["signature"] == lengths["clinical"] == 40
    assert signature["n_clinical_replicates"] == signature["n_signature_replicates"] == 40
    assert signature["signature_gain_left_out"] == pytest.approx(signature["signature_c_left_out"] - signature["clinical_c_left_out"])


def test_results_are_json_ready() -> None:
    frame = _separated_frame(6)
    result = evaluate_markers(
        frame, time_column="os_time", event_column="os_event", marker_columns=["mut", "g1", "g2"],
        settings=MarkerSettings(n_permutations=19, n_resamples=4, random_seed=1),
    )
    json.dumps(result, allow_nan=False)
    if result["locked_recipe"] is not None:
        report = validate_locked_recipe(_separated_frame(7), result["locked_recipe"], n_bootstrap=30)
        json.dumps(report, allow_nan=False)

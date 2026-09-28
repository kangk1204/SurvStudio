"""Regression tests for the second review of the marker engine (marker_screen, marker_evaluation, duplicates)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import survival_toolkit.marker_evaluation as marker_evaluation
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers, prepare_marker_cohort, validate_locked_recipe
from survival_toolkit.marker_screen import CoxScoreScreen, fit_cox, fit_cox_null

_QUICK = MarkerSettings(n_permutations=19, n_resamples=8, shortlist_size=10, random_seed=1)


# 1: clinical designs that lose the reference level of a categorical covariate.


def _aliased_rows() -> dict[str, np.ndarray]:
    """60 rows built from the row number i, as in the R script below; grade has levels 2 and 3 only, so g2 + g3 = 1.

    R survival 3.8.6:
        i <- 1:60; g2 <- as.numeric((7 * i) %% 5 < 2); g3 <- 1 - g2; age <- sin(i); m <- cos(1.3 * i) + 0.5 * age
        time <- ((37 * i) %% 23) + 1; event <- as.integer(i %% 3 != 0); site_b <- as.numeric(i > 30)
        coxph(Surv(time, event) ~ g2 + g3 + age + m, ties = "efron")
        coxph(Surv(time, event) ~ g2 + g3 + age + m + site_b + strata(site_b), ties = "breslow")
    """
    i = np.arange(1, 61, dtype=float)
    g2 = ((7 * i) % 5 < 2).astype(float)
    age = np.sin(i)
    return {
        "time": (37 * i) % 23 + 1,
        "event": (i % 3 != 0).astype(int),
        "g2": g2,
        "g3": 1.0 - g2,
        "age": age,
        "m": np.cos(1.3 * i) + 0.5 * age,
        "site_b": (i > 30).astype(float),
    }


# R reports NA for the aliased coefficients (g3, and site_b, which is constant within the strata).
_R_ALIASED = {
    "efron": {
        "coef": [0.586463446717, None, -0.090579940498, 0.3829425078],
        "se": [0.342473597882, None, 0.270567346858, 0.256591357699],
        "loglik": -126.972242995,
    },
    "breslow": {
        "coef": [0.762742306515, None, -0.135324264929, 0.45339875549, None],
        "se": [0.399653823374, None, 0.275852444472, 0.267163705449, None],
        "loglik": -99.5720828557,
    },
}


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_fit_cox_leaves_aliased_columns_out_as_r_coxph_does(ties: str) -> None:
    rows = _aliased_rows()
    columns = ["g2", "g3", "age", "m"] + (["site_b"] if ties == "breslow" else [])
    strata = rows["site_b"].astype(int) if ties == "breslow" else None
    fit = fit_cox(rows["time"], rows["event"], np.column_stack([rows[name] for name in columns]), strata, ties)
    expected = _R_ALIASED[ties]
    # The Newton iterations converge; the aliased columns get no coefficient and no variance, never a
    # pseudo-inverse's, and the fit is not reported as failed.
    assert fit.converged
    for index, (coefficient, se) in enumerate(zip(expected["coef"], expected["se"])):
        if coefficient is None:
            assert np.isnan(fit.beta[index])
            assert np.isnan(fit.covariance[index]).all() and np.isnan(fit.covariance[:, index]).all()
        else:
            assert fit.beta[index] == pytest.approx(coefficient, rel=1e-8)
            assert np.sqrt(fit.covariance[index, index]) == pytest.approx(se, rel=1e-6)
    assert fit.loglik == pytest.approx(expected["loglik"], rel=1e-10)
    assert not fit.separated.any()
    # The null model's linear predictor is the one of the fit without the aliased columns.
    kept = [index for index, value in enumerate(expected["coef"]) if value is not None]
    null = fit_cox_null(rows["time"], rows["event"], np.column_stack([rows[name] for name in columns]), strata, ties)
    reduced = fit_cox_null(rows["time"], rows["event"], np.column_stack([rows[columns[index]] for index in kept]), strata, ties)
    assert null.converged and null.eta == pytest.approx(reduced.eta, abs=1e-12)


def _graded_cohort(seed: int, n: int, levels: list[str], probabilities: list[float]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = rng.choice(levels, size=n, p=probabilities)
    age = rng.normal(size=n)
    markers = rng.normal(size=(n, 6))
    linear = 0.6 * age + 0.5 * (grade == "3") + 0.8 * markers[:, 0] - 0.7 * markers[:, 1]
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(markers, columns=[f"m{index}" for index in range(6)])
    frame["time"] = np.minimum(event_time, censor_time)
    frame["event"] = (event_time <= censor_time).astype(int)
    frame["age"] = age
    frame["grade"] = grade
    return frame


_GRADED = dict(time_column="time", event_column="event", marker_columns=[f"m{index}" for index in range(6)],
               clinical_columns=["age", "grade"], categorical_clinical=["grade"])


def test_an_external_cohort_without_the_reference_level_replicates_the_true_markers() -> None:
    recipe = evaluate_markers(_graded_cohort(1, 300, ["1", "2", "3"], [0.3, 0.4, 0.3]), settings=_QUICK, **_GRADED)["locked_recipe"]
    assert {"m0", "m1"} <= set(recipe["markers"])
    assert recipe["model"]["terms"][:3] == ["grade_2", "grade_3", "age"]
    external = _graded_cohort(7, 300, ["2", "3"], [0.5, 0.5])  # grade 1, the reference level, is missing

    report = validate_locked_recipe(external, recipe, n_bootstrap=0)

    assert not [note for note in report["notes"] if "not estimable" in note]
    time, event = external["time"].to_numpy(), external["event"].to_numpy()
    # The identifiable design keeps grade_2 and drops grade_3 (grade_2 + grade_3 = 1 here).
    base = np.column_stack([(external["grade"] == "2").to_numpy(dtype=float), external["age"].to_numpy()])
    for row in report["markers"]:
        assert row["tested"] == "added_value"
        reference = fit_cox(time, event, np.column_stack([base, external[row["marker"]].to_numpy()]))
        beta, se = reference.beta[-1], np.sqrt(reference.covariance[-1, -1])
        assert row["adjusted"]["log_hr"] == pytest.approx(beta, rel=1e-10)
        assert row["adjusted"]["wald_p"] == pytest.approx(2.0 * stats.norm.sf(abs(beta / se)), rel=1e-8, abs=0.0)
    replicated = {row["marker"]: row["replicated"] for row in report["markers"]}
    assert replicated["m0"] and replicated["m1"]


def test_subsamples_that_lose_a_rare_reference_level_are_not_counted_as_failed() -> None:
    rng = np.random.default_rng(4)
    n = 150
    grade = np.array(["2"] * 74 + ["3"] * 75 + ["1"])  # the reference level "1" has one patient
    rng.shuffle(grade)
    age = rng.normal(size=n)
    markers = rng.normal(size=(n, 8))
    linear = 0.6 * age + 0.5 * (grade == "3") + 0.7 * markers[:, 0]
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(markers, columns=[f"m{index}" for index in range(8)])
    frame["time"] = np.minimum(event_time, censor_time)
    frame["event"] = (event_time <= censor_time).astype(int)
    frame["age"] = age
    frame["grade"] = grade

    result = evaluate_markers(
        frame, time_column="time", event_column="event", marker_columns=[f"m{index}" for index in range(8)],
        clinical_columns=["age", "grade"], categorical_clinical=["grade"],
        settings=MarkerSettings(n_permutations=19, n_resamples=40, random_seed=1),
    )

    # About a third of the subsamples leave the one grade-1 patient out; their clinical model lost its
    # reference level and failed as "did not converge", so the stability rested on the others only.
    assert result["resampling"]["n_failed"] == 0 and result["resampling"]["n_valid"] == 40
    signature = result["signature"]
    assert signature["n_signature_replicates"] == 40 and signature["n_top_marker_replicates"] == 40


def test_left_out_and_subsample_fits_drop_the_indicator_of_a_missing_reference_level() -> None:
    frame = _graded_cohort(3, 240, ["1", "2", "3"], [0.2, 0.4, 0.4])
    cohort = prepare_marker_cohort(frame, time_column="time", event_column="event", marker_columns=_GRADED["marker_columns"],
                                   clinical_columns=["age", "grade"], categorical_clinical=["grade"])
    assert cohort.clinical_names == ["grade_2", "grade_3", "age"]
    rows = np.flatnonzero(frame["grade"].to_numpy() != "1")
    medians = np.nanmedian(cohort.markers, axis=0)
    beta = marker_evaluation._single_marker_beta(cohort, rows, 0, medians, adjusted=True)
    design = np.column_stack([cohort.clinical[rows][:, [0, 2]], cohort.markers[rows, 0]])
    assert beta == pytest.approx(fit_cox(cohort.time[rows], cohort.event[rows], design).beta[-1], rel=1e-10)

    full = marker_evaluation.run_procedure(cohort, np.arange(len(frame)), _QUICK)
    signature = marker_evaluation._fit_signature(cohort, rows, full, "added_value", 10)
    assert signature is not None and signature.design_columns.tolist() == [0, 2]
    subsample = marker_evaluation.run_procedure(cohort, rows, _QUICK)
    screen = CoxScoreScreen(
        cohort.time[rows], cohort.event[rows], null=fit_cox_null(cohort.time[rows], cohort.event[rows], cohort.clinical[rows][:, [0, 2]]),
        Z=cohort.clinical[rows][:, [0, 2]],
    )
    expected = screen.statistics(np.where(np.isnan(cohort.markers[rows]), subsample.medians, cohort.markers[rows])).chi2
    assert subsample.lenses["added_value"].chi2 == pytest.approx(expected, rel=1e-10)


def test_a_level_whose_patients_leave_before_the_first_event_is_left_out_of_every_fit() -> None:
    frame = _graded_cohort(5, 200, ["2", "3"], [0.5, 0.5])
    # Grade 1 patients are all censored before the first event: they are in no risk set, so the partial
    # likelihood cannot tell grade 1 from the others although the rows can.
    early = pd.DataFrame({"time": [1e-6, 2e-6, 3e-6], "event": [0, 0, 0], "age": [0.1, -0.2, 0.3], "grade": ["1", "1", "1"]})
    for index in range(6):
        early[f"m{index}"] = [0.5, -0.5, 0.2]
    with_early = pd.concat([frame, early], ignore_index=True)

    result = evaluate_markers(with_early, settings=_QUICK, **_GRADED)
    without = evaluate_markers(frame, settings=_QUICK, **_GRADED)

    # Patients outside every risk set change no score test; the aliased indicator is dropped as in the data without them.
    chi2 = {row["marker"]: row["added_value"]["chi2"] for row in result["marker_table"]}
    reference = {row["marker"]: row["added_value"]["chi2"] for row in without["marker_table"]}
    assert chi2 == pytest.approx(reference, rel=1e-8)
    recipe = result["locked_recipe"]
    assert recipe is not None and "grade_3" not in recipe["model"]["terms"]
    assert all(np.isfinite(recipe["model"]["coefficients"]))


# 2: only a clinical model that does not converge (or a failing matrix routine) counts as a failed subsample.


def test_errors_raised_by_checks_in_a_subsample_stop_the_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    from survival_toolkit import marker_screen
    from survival_toolkit.errors import UserInputError

    rng = np.random.default_rng(1)
    n = 120
    frame = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    frame["time"] = rng.exponential(size=n) + 0.05
    frame["event"] = rng.integers(0, 2, n)
    original = marker_evaluation.run_procedure
    common = dict(time_column="time", event_column="event", marker_columns=list("abcd"),
                  settings=MarkerSettings(n_permutations=9, n_resamples=5, random_seed=1))

    def failing_with(error):
        def run(cohort, rows, settings):
            if rows.size < n:
                error(rows)
            return original(cohort, rows, settings)

        return run

    def singular_matrix(rows):
        np.linalg.inv(np.zeros((2, 2)))

    monkeypatch.setattr(marker_evaluation, "run_procedure", failing_with(singular_matrix))
    assert evaluate_markers(frame, **common)["resampling"]["n_failed"] == 5

    def shape_bug(rows):
        # numpy raises inside a SurvStudio function, so the error looked like SurvStudio's own check.
        marker_screen.residualize(np.zeros((rows.size, 2)), np.zeros((rows.size - 1, 1)))

    def contract_check(rows):
        marker_evaluation._validated_settings(MarkerSettings(alpha=2.0))

    for error in (shape_bug, contract_check):
        monkeypatch.setattr(marker_evaluation, "run_procedure", failing_with(error))
        # Before, every subsample was counted as failed and the analysis finished without its stability.
        with pytest.raises(UserInputError):
            evaluate_markers(frame, **common)


# 4: infinite marker values are never imputed as if they were missing.


def _log_expression_cohort(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-0.9 * signal))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame({"time": np.minimum(event_time, censor_time), "event": (event_time <= censor_time).astype(int)})
    frame["signal"] = signal
    frame["noise"] = rng.normal(size=n)
    expression = np.exp(rng.normal(size=n))
    expression[:15] = 0.0
    with np.errstate(divide="ignore"):
        frame["log_expression"] = np.log(expression)  # log(0) = -inf for 15 patients
    return frame


def test_development_markers_with_infinite_values_are_left_out_with_the_fix() -> None:
    frame = _log_expression_cohort(1)
    common = dict(time_column="time", event_column="event", settings=_QUICK)
    result = evaluate_markers(frame, marker_columns=["signal", "noise", "log_expression"], **common)
    assert {row["marker"] for row in result["marker_table"]} == {"signal", "noise"}
    assert result["cohort"]["dropped_markers"] == [{"marker": "log_expression", "reason": "infinite values", "n_infinite": 15}]
    assert any("1 marker(s) hold infinite values" in note and "log_expression" in note and "log(x + 1)" in note for note in result["cohort"]["notes"])
    with pytest.raises(ValueError, match=r"log_expression \(infinite values\).*log\(x \+ 1\)"):
        evaluate_markers(frame, marker_columns=["log_expression"], **common)
    # A marker with positive infinity written as text is left out the same way.
    frame["as_text"] = frame["signal"].astype(object)
    frame.loc[3, "as_text"] = "inf"
    assert {"marker": "as_text", "reason": "infinite values", "n_infinite": 1} in evaluate_markers(
        frame, marker_columns=["signal", "as_text"], **common
    )["cohort"]["dropped_markers"]


def test_validation_refuses_locked_markers_with_infinite_values() -> None:
    from survival_toolkit.errors import UserInputError

    frame = _log_expression_cohort(2)
    recipe = evaluate_markers(frame, time_column="time", event_column="event", marker_columns=["signal", "noise"], settings=_QUICK)["locked_recipe"]
    assert "signal" in recipe["markers"]
    external = _log_expression_cohort(3)
    external.loc[:4, "signal"] = -np.inf
    with pytest.raises(UserInputError, match=r"infinite values in the external dataset: signal.*log\(x \+ 1\)"):
        validate_locked_recipe(external, recipe, n_bootstrap=0)
    # Missing values are still imputed at the development median, also from a nullable (Parquet-style) column.
    external = _log_expression_cohort(3)
    external["signal"] = external["signal"].astype("Float64")
    external.loc[:4, "signal"] = pd.NA
    report = validate_locked_recipe(external, recipe, n_bootstrap=0)
    assert any("5 missing signal value(s) imputed" in note for note in report["notes"])


# 5: the locked baseline hazard follows the model's tie method.

# R survival 3.8.6:
#   i <- 1:60; time <- ((37 * i) %% 23) + 1; x <- sin(1.7 * i) - time / 10; event <- as.integer(i %% 4 != 0)
#   f <- coxph(Surv(time, event) ~ x, ties = ties); s <- survfit(f, newdata = data.frame(x = mean(x)))
#   s$cumhaz[s$n.event > 0]   (every time 1..23 has a death; most have several)
_R_BASELINE = {
    "efron": {
        "beta": 0.89861411337,
        "cumhaz": [
            0.0234186791053, 0.0496052179763, 0.0783249943705, 0.111022446536, 0.129065821273, 0.168319264516,
            0.234404736289, 0.284179111845, 0.312778662447, 0.373940784826, 0.442529225967, 0.524515459546,
            0.614912296514, 0.72580282408, 0.858817783067, 1.01926190427, 1.32363711309, 1.44995442503,
            1.61113490936, 2.21976363837, 2.81108784645, 4.17437804361, 7.92477506675,
        ],
    },
    "breslow": {
        "beta": 0.870160358567,
        "cumhaz": [
            0.0232564590924, 0.0494236332584, 0.0783890970939, 0.110547457715, 0.128800909522, 0.168068813843,
            0.231461085568, 0.280880285682, 0.309595527764, 0.369371561866, 0.436102971129, 0.516378917604,
            0.605347671812, 0.710859331425, 0.838206010011, 0.994829515336, 1.266789515, 1.39137185446,
            1.54978190049, 2.07275474696, 2.60828623071, 3.7745637469, 6.21394170385,
        ],
    },
}


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_the_locked_baseline_hazard_matches_r_survfit_for_either_tie_method(ties: str) -> None:
    i = np.arange(1, 61, dtype=float)
    time = (37 * i) % 23 + 1
    x = np.sin(1.7 * i) - time / 10
    event = (i % 4 != 0).astype(int)
    expected = _R_BASELINE[ties]
    fit = fit_cox(time, event, x, None, ties)
    assert fit.beta[0] == pytest.approx(expected["beta"], rel=1e-8)
    baseline = marker_evaluation._centred_baseline(time, event, fit.beta[0] * x, ties)
    assert baseline["times"].tolist() == list(range(1, 24))
    # With Efron ties the Breslow increments gave 6.21 instead of R's 7.92 at the last time.
    assert np.exp(baseline["log_cumulative_hazard"]) == pytest.approx(expected["cumhaz"], rel=1e-7)


def test_without_tied_deaths_the_efron_baseline_is_the_breslow_one() -> None:
    rng = np.random.default_rng(6)
    time = rng.exponential(size=80)
    event = rng.integers(0, 2, size=80)
    linear_predictor = rng.normal(size=80)
    efron = marker_evaluation._centred_baseline(time, event, linear_predictor, "efron")
    breslow = marker_evaluation._centred_baseline(time, event, linear_predictor, "breslow")
    assert np.array_equal(efron["log_cumulative_hazard"], breslow["log_cumulative_hazard"])


# 6, 8 and text in numeric covariates: what validation requires of the external dataset.


def _copy_cohort(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    genes = rng.normal(size=(n, 4))
    linear = 0.6 * age + 0.8 * genes[:, 0]
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(genes, columns=[f"g{index}" for index in range(4)])
    frame["time"] = np.minimum(event_time, censor_time)
    frame["event"] = (event_time <= censor_time).astype(int)
    frame["age"] = age
    frame["age_copy"] = 2 * age + 1  # development leaves it out as a linear combination of age
    return frame


def _copy_recipe() -> dict:
    result = evaluate_markers(_copy_cohort(1), time_column="time", event_column="event", marker_columns=[f"g{index}" for index in range(4)],
                              clinical_columns=["age", "age_copy"], settings=_QUICK)
    assert result["cohort"]["dropped_clinical_columns"][0]["column"] == "age_copy"
    return result["locked_recipe"]


def test_validation_needs_only_the_clinical_columns_behind_the_locked_terms() -> None:
    recipe = _copy_recipe()
    assert "age_copy" not in recipe["model"]["terms"]
    reference = validate_locked_recipe(_copy_cohort(2), recipe, n_bootstrap=0)
    # The column the model does not use may be absent, partly missing or even hold text.
    without = validate_locked_recipe(_copy_cohort(2).drop(columns=["age_copy"]), recipe, n_bootstrap=0)
    assert without["metrics"]["c_index"] == reference["metrics"]["c_index"]
    partly = _copy_cohort(2)
    partly.loc[partly.index[:60], "age_copy"] = np.nan
    assert validate_locked_recipe(partly, recipe, n_bootstrap=0)["cohort"]["n"] == 200
    garbled = _copy_cohort(2).astype({"age_copy": object})
    garbled.loc[garbled.index[:3], "age_copy"] = "unknown"
    assert validate_locked_recipe(garbled, recipe, n_bootstrap=0)["metrics"]["c_index"] == reference["metrics"]["c_index"]


def test_validation_refuses_text_in_a_numeric_clinical_covariate() -> None:
    from survival_toolkit.errors import UserInputError

    recipe = _copy_recipe()
    external = _copy_cohort(2).astype({"age": object})
    external.loc[external.index[:4], "age"] = "."
    external.loc[external.index[4], "age"] = "unknown"
    with pytest.raises(UserInputError, match=r'"age" holds 5 value\(s\) .* not numbers, such as "\.", "unknown"\. Blank those cells'):
        validate_locked_recipe(external, recipe, n_bootstrap=0)
    # Blank cells stay missing values: their rows are left out, as in development.
    blank = _copy_cohort(2)
    blank.loc[blank.index[:5], "age"] = np.nan
    assert validate_locked_recipe(blank, recipe, n_bootstrap=0)["cohort"]["n"] == 195


def test_within_cohort_scaling_of_a_clinical_only_model_and_of_a_recipe_without_a_marker_scale() -> None:
    from survival_toolkit.errors import UserInputError

    rng = np.random.default_rng(21)
    n = 200
    age = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-1.0 * age))
    censor_time = rng.exponential(1.5, n)
    frame = pd.DataFrame({"time": np.minimum(event_time, censor_time), "event": (event_time <= censor_time).astype(int), "age": age})
    frame["noise"] = rng.normal(size=n)
    recipe = evaluate_markers(frame, time_column="time", event_column="event", marker_columns=["noise"], clinical_columns=["age"],
                              settings=_QUICK)["locked_recipe"]
    assert recipe["markers"] == [] and recipe["marker_scale"] == {}
    # Nothing to rescale: the clinical-only model validates as measured, without a note about rescaling.
    within = validate_locked_recipe(frame, recipe, n_bootstrap=0, marker_scaling="within_cohort")
    assert within["metrics"]["c_index"] == validate_locked_recipe(frame, recipe, n_bootstrap=0)["metrics"]["c_index"]
    assert not any("rescaled" in note for note in within["notes"])

    marker_recipe = _copy_recipe()
    edited = {**marker_recipe, "marker_scale": {**marker_recipe["marker_scale"], marker_recipe["markers"][0]: None}}
    edited["recipe_hash"] = marker_evaluation.recipe_hash(edited)
    for scaling in ("as_measured", "within_cohort"):
        with pytest.raises(UserInputError, match="marker_scale of .* needs a finite mean and SD"):
            validate_locked_recipe(_copy_cohort(2), edited, n_bootstrap=0, marker_scaling=scaling)


# 7: a marker cannot be named like an encoded clinical indicator.


def test_a_marker_named_like_a_clinical_level_indicator_is_refused() -> None:
    from survival_toolkit.errors import UserInputError

    frame = _graded_cohort(2, 200, ["1", "2", "3"], [0.3, 0.4, 0.3]).rename(columns={"m0": "grade_2"})
    with pytest.raises(UserInputError, match="encoded clinical covariate: grade_2"):
        # Before, the locked model held two terms named grade_2 and could never be validated.
        evaluate_markers(frame, time_column="time", event_column="event", marker_columns=["grade_2", "m1"],
                         clinical_columns=["age", "grade"], categorical_clinical=["grade"], settings=_QUICK)


# 9: a stratified locked model gets its bootstrap intervals too.


def test_stratified_c_many_equals_the_pooled_within_stratum_c_index() -> None:
    rng = np.random.default_rng(9)
    for _ in range(20):
        n = int(rng.integers(20, 120))
        time = rng.integers(1, 12, size=n).astype(float)
        event = rng.integers(0, 2, size=n)
        risks = np.round(rng.normal(size=(n, 2)), 1)
        strata = rng.integers(0, 3, size=n)
        pooled = marker_evaluation._stratified_c_many(time, event, risks, strata)
        for column in range(2):
            assert pooled[column] == pytest.approx(marker_evaluation._pooled_c_index(time, event, risks[:, column], strata), abs=1e-12)


def test_a_stratified_locked_model_gets_bootstrap_intervals() -> None:
    frame = _copy_cohort(1).drop(columns=["age_copy"])
    frame["site"] = np.where(np.arange(len(frame)) % 3 == 0, "A", "B")
    recipe = evaluate_markers(frame, time_column="time", event_column="event", marker_columns=[f"g{index}" for index in range(4)],
                              clinical_columns=["age"], strata_columns=["site"], settings=_QUICK)["locked_recipe"]
    external = _copy_cohort(2).drop(columns=["age_copy"])
    external["site"] = np.where(np.arange(len(external)) % 2 == 0, "A", "B")
    metrics = validate_locked_recipe(external, recipe, n_bootstrap=100)["metrics"]
    # Before, the bootstrap was skipped without a word for stratified models.
    for key in ("c_index", "clinical_only_c_index", "delta_c_index"):
        low, high = metrics[f"{key}_ci"]
        assert low is not None and low <= metrics[key] <= high and low < high


# 10: the permutations, the subsamples and the non-linear lens draw from independent random streams.


def test_the_number_of_permutations_does_not_change_the_subsamples() -> None:
    frame = _copy_cohort(1).drop(columns=["age_copy"])
    common = dict(time_column="time", event_column="event", marker_columns=[f"g{index}" for index in range(4)])
    settings = MarkerSettings(n_permutations=19, n_resamples=6, random_seed=1)
    first = evaluate_markers(frame, settings=settings, **common)
    more_permutations = evaluate_markers(frame, settings=settings._replace(n_permutations=20), **common)
    more_subsamples = evaluate_markers(frame, settings=settings._replace(n_resamples=7), **common)

    def lens(result, key):
        return {row["marker"]: row["marginal"][key] for row in result["marker_table"]}

    for key in ("selection_frequency", "direction_consistency", "median_rank", "rank_interval"):
        assert lens(first, key) == lens(more_permutations, key), key
    assert first["signature"]["signature_optimism"] == more_permutations["signature"]["signature_optimism"]
    assert lens(first, "p_fwer") == lens(more_subsamples, "p_fwer") and lens(first, "q_perm") == lens(more_subsamples, "q_perm")


# 11: a dataset whose index repeats a label is read row by row.


def test_a_dataset_index_with_repeated_labels_is_read_by_position() -> None:
    frame = _copy_cohort(4, n=120).drop(columns=["age_copy"])
    frame.loc[5, "g1"] = np.nan
    common = dict(time_column="time", event_column="event", marker_columns=[f"g{index}" for index in range(4)], clinical_columns=["age"],
                  settings=_QUICK)
    reference = evaluate_markers(frame, **common)
    repeated = frame.set_axis([index // 2 for index in range(len(frame))])  # two samples per patient label
    # Before, reading the markers by label returned both rows of each label and failed with a numpy broadcast error.
    result = evaluate_markers(repeated, **common)
    assert result["marker_table"] == reference["marker_table"] and result["cohort"]["n"] == 120
    recipe = result["locked_recipe"]
    external = _copy_cohort(5, n=120).drop(columns=["age_copy"])
    report = validate_locked_recipe(external.set_axis([0] * len(external)), recipe, n_bootstrap=0)
    assert report["metrics"] == validate_locked_recipe(external, recipe, n_bootstrap=0)["metrics"]


# 12: a replication test that cannot be estimated stays in the Holm family, at p = 1.


def test_a_replication_that_cannot_be_estimated_keeps_its_place_in_the_holm_family(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = evaluate_markers(_graded_cohort(1, 300, ["1", "2", "3"], [0.3, 0.4, 0.3]), settings=_QUICK, **_GRADED)["locked_recipe"]
    assert recipe["markers"] == ["m0", "m1"]
    external = _graded_cohort(7, 300, ["1", "2", "3"], [0.3, 0.4, 0.3])
    original = marker_evaluation._external_cox

    def m1_fails(time, event, exog, strata, ties="efron"):
        if exog.shape[1] > 1 and np.array_equal(exog[:, -1], external["m1"].to_numpy()):
            return None
        return original(time, event, exog, strata, ties)

    monkeypatch.setattr(marker_evaluation, "_external_cox", m1_fails)
    rows = {row["marker"]: row for row in validate_locked_recipe(external, recipe, n_bootstrap=0)["markers"]}
    assert rows["m1"]["tested"] is None and rows["m1"]["replication_p_holm"] is None and not rows["m1"]["replicated"]
    one_sided = rows["m0"]["adjusted"]["wald_p"] / 2.0
    assert rows["m0"]["same_direction"]
    # Before, m0 was adjusted as if the family held it alone (p = one_sided).
    assert rows["m0"]["replication_p_holm"] == pytest.approx(min(1.0, 2.0 * one_sided), rel=1e-12, abs=0.0)


# 13: notes and lenses say what was done.


def test_an_evaluation_whose_clinical_columns_are_all_left_out_is_the_unadjusted_one() -> None:
    frame = _copy_cohort(6).drop(columns=["age_copy"])
    frame["centre"] = np.where(np.arange(len(frame)) < 100, "A", "B")
    frame["centre_b"] = (frame["centre"] == "B").astype(float)  # constant within each stratum
    common = dict(time_column="time", event_column="event", marker_columns=[f"g{index}" for index in range(4)], strata_columns=["centre"],
                  settings=_QUICK)
    result = evaluate_markers(frame, clinical_columns=["centre_b"], **common)
    unadjusted = evaluate_markers(frame, **common)
    # Before, the primary lens stayed "added value" and the null was reported as the Smith scheme, although
    # the markers themselves were permuted and nothing was adjusted for.
    assert result["primary_lens"] == "marginal" and result["null"]["lens2_null"] is None
    assert result["marker_table"] == unadjusted["marker_table"]
    assert any("without clinical adjustment" in note for note in result["cohort"]["notes"])
    assert result["locked_recipe"]["model"]["terms"] == unadjusted["locked_recipe"]["model"]["terms"]


def test_a_clinical_model_that_cannot_be_fitted_is_not_blamed_on_a_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(21)
    n = 150
    frame = pd.DataFrame({"time": rng.exponential(size=n) + 0.1, "event": rng.integers(0, 2, n), "age": rng.normal(size=n)})
    for index in range(3):
        frame[f"noise_{index}"] = rng.normal(size=n)
    monkeypatch.setattr(marker_evaluation, "_fit_signature", lambda *args, **kwargs: None)
    result = evaluate_markers(frame, time_column="time", event_column="event", marker_columns=[f"noise_{index}" for index in range(3)],
                              clinical_columns=["age"], settings=_QUICK._replace(n_resamples=0))
    assert result["signature"]["markers"] == [] and result["locked_recipe"] is None
    assert result["signature"]["notes"] == [
        "No marker was selected, and the model of the clinical covariates alone could not be fitted: it did not converge. "
        "No model was locked."
    ]


# 14: a horizon past the development follow-up is flagged.


def test_a_horizon_past_the_last_development_event_is_noted() -> None:
    recipe = _copy_recipe()
    last = recipe["model"]["baseline"]["times"][-1]
    external = _copy_cohort(2)
    at_default = validate_locked_recipe(external, recipe, n_bootstrap=0)
    assert not any("past the last event time" in note for note in at_default["notes"])
    beyond = validate_locked_recipe(external, recipe, n_bootstrap=0, horizon=2.0 * last)
    # The absolute risks are still reported; before, nothing warned that they rest on a flat baseline.
    assert beyond["metrics"]["expected_risk"] is not None
    assert any(f"past the last event time of the development cohort ({last:g})" in note for note in beyond["notes"])


# 3: the duplicate screen reads the panel in blocks, stops when cancelled, and gives the same results.


def _duplicates_reference(values: np.ndarray, labels: list[str]) -> dict:
    """The whole-panel implementation the blocked screen replaced, kept as the definition (cohorts up to MAX_PATIENTS)."""
    from collections import defaultdict

    from survival_toolkit import duplicates

    report = {"checked": False, "identical_checked": False, "markers_used": 0, "note": None, "pairs": [], "identical": [],
              "n_pairs": 0, "n_identical": 0, "mostly_missing": [], "n_mostly_missing": 0}
    if values.shape[1]:
        sparse = np.isnan(values).mean(axis=1) > duplicates.MAX_MISSING_SHARE
        if sparse.any():
            report.update(mostly_missing=[labels[index] for index in np.flatnonzero(sparse)], n_mostly_missing=int(sparse.sum()))
            values = values[~sparse]
            labels = [label for label, drop in zip(labels, sparse) if not drop]
    n_patients, n_markers = values.shape
    if n_patients < 3:
        report["note"] = "Too few patients to compare."
        return report
    sampled = values[:, :: max(1, n_markers // 200)] if n_markers else values
    distinct = np.array([np.unique(column[~np.isnan(column)]).size for column in sampled.T]) if n_markers else np.zeros(0)
    if n_markers >= duplicates.MIN_IDENTICAL_MARKERS and np.median(distinct) >= duplicates.MIN_DISTINCT_VALUES:
        groups = defaultdict(list)
        for label, row in zip(labels, np.round(np.where(np.isnan(values), np.inf, values), 9) + 0.0):
            groups[row.tobytes()].append(label)
        identical = [members for members in groups.values() if len(members) > 1]
        report.update(identical_checked=True, identical=identical, n_identical=len(identical))
    if n_markers < duplicates.MIN_MARKERS:
        report["note"] = f"Near-identical profiles are checked on panels of at least {duplicates.MIN_MARKERS} markers."
        return report
    with np.errstate(invalid="ignore"):
        spread = np.nanvar(values, axis=0)
    usable = np.flatnonzero(np.isfinite(spread) & (spread > 0))
    chosen = usable[np.argsort(-spread[usable], kind="mergesort")[: duplicates.TOP_MARKERS]]
    block = values[:, chosen]
    block = np.where(np.isnan(block), np.nanmedian(block, axis=0)[None, :], block)
    block = (block - block.mean(axis=0)) / block.std(axis=0)
    block = block - block.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    block = np.divide(block, norms, out=np.zeros_like(block), where=norms > 0)
    correlation = block @ block.T
    np.fill_diagonal(correlation, -np.inf)
    best = correlation.argmax(axis=1)
    top_two = np.sort(correlation, axis=1)[:, -2:]
    pairs = []
    for i, j in enumerate(best):
        if j <= i or best[j] != i:
            continue
        r = float(correlation[i, j])
        gap = r - max(float(top_two[i, 0]), float(top_two[j, 0]))
        if r >= duplicates.MIN_R and gap >= duplicates.MIN_GAP:
            pairs.append({"a": labels[i], "b": labels[j], "r": r, "gap": gap})
    pairs.sort(key=lambda pair: -pair["r"])
    report.update(checked=True, markers_used=int(chosen.size), pairs=pairs, n_pairs=len(pairs))
    return report


def _repeated_panel(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    subtype = rng.integers(0, 5, n)
    values = rng.normal(size=(5, p))[subtype] * 1.5 + rng.normal(size=(n, p))
    values[rng.random(values.shape) < 0.05] = np.nan
    values[rng.integers(0, n)] = np.nan  # mostly missing patients take part in neither check
    values[rng.integers(0, n), : int(0.7 * p)] = np.nan
    for _ in range(2):
        first, second = rng.integers(0, n, size=2)
        values[second] = values[first]
    first, second = rng.integers(0, n, size=2)
    values[second] = values[first] + rng.normal(scale=0.2, size=p)
    return values


@pytest.mark.parametrize("order", ["C", "F"])
def test_the_blocked_duplicate_screen_equals_the_whole_panel_computation(monkeypatch: pytest.MonkeyPatch, order: str) -> None:
    from survival_toolkit import duplicates

    # Blocks of a few rows or columns, so every block boundary is crossed; the marker evaluation passes
    # a Fortran-ordered panel.
    monkeypatch.setattr(duplicates, "_BLOCK_VALUES", 3000)
    rng = np.random.default_rng(12)
    # 90 x 232 leaves a last block of one column, which is merged into the one before it.
    for n, p in ((60, 25), (90, 232), (150, 5300), (40, 201)):
        values = np.asarray(_repeated_panel(rng, n, p), order=order)
        labels = [f"P{index}" for index in range(n)]
        expected = _duplicates_reference(values.copy(order="K"), labels)
        assert duplicates.possible_duplicates(values, labels) == expected, (n, p)
        assert expected["identical"] and (p < duplicates.MIN_MARKERS or expected["pairs"])


def test_the_duplicate_screen_bounds_its_memory_and_stops_when_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import tracemalloc

    from survival_toolkit import duplicates
    from survival_toolkit.concurrency import cancellation_scope
    from survival_toolkit.errors import JobCancelledError

    monkeypatch.setattr(duplicates, "_BLOCK_VALUES", 1 << 16)
    rng = np.random.default_rng(3)
    values = rng.normal(size=(200, 20000))
    values[rng.random(values.shape) < 0.02] = np.nan
    values[150] = values[20]
    labels = [str(index) for index in range(200)]
    tracemalloc.start()
    try:
        report = duplicates.possible_duplicates(values, labels)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert report["identical"] == [["20", "150"]] and report["checked"]
    # Before, rounding the whole panel for the identical check, the variance of every marker and the sorted
    # correlation matrix took about five times the panel.
    assert peak < 0.8 * values.nbytes
    stop = threading.Event()
    stop.set()
    with cancellation_scope(stop), pytest.raises(JobCancelledError):
        duplicates.possible_duplicates(values, labels)
    # Cohorts too large to compare are returned before any pass over the panel.
    monkeypatch.setattr(duplicates, "MAX_PATIENTS", 150)
    large = duplicates.possible_duplicates(values, labels)
    assert not large["checked"] and not large["identical_checked"] and "up to 150 patients" in large["note"]

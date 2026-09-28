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
        assert row["adjusted"]["wald_p"] == pytest.approx(2.0 * stats.norm.sf(abs(beta / se)), rel=1e-8)
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

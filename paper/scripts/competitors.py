"""Shared code of the comparison with the pipelines commonly used to publish prognostic signatures
(scripts 18_competitors_*.py and the R scripts of paper/competitors/).

The pipelines, each developed on TCGA-LUAD and applied to the seven GEO cohorts of case study II:
- P1 Mime (ML.Dev.Prog.Sig, mode "all": 10 algorithms and their combinations, 117 models in the pinned version), its
  univariate-Cox candidate filter at p < 0.05, the candidates capped at the CANDIDATE_CAP smallest p-values;
- P2 univariate Cox (p < 0.05) -> LASSO-Cox (glmnet, 10-fold CV, lambda.min) -> multivariable Cox of the selected
  genes, whose linear predictor is the risk score;
- P3 uncorrected best-cutoff screen: for every gene, the minimum log-rank p over the cut-offs between its quartiles.
  This is a stylised pipeline, not a reproduction of the current KM Plotter service, which documents FDR correction.

Patients, outcomes and clinical covariates are those of case studies I and II (development_data, validation_data);
the genes are case study I's, restricted to those every GEO cohort measures (gene_set), each z-scored within its
cohort. The outcome in R is given in days (Mime's documented unit; months x 365.25 / 12).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from common import BOOTSTRAP_DRAWS, CATEGORICAL, COVARIATES, GEO_COHORTS, LUAD, PAPER, RESULTS, SCRIPTS, XENA_EXPRESSION, random_effects
from survival_toolkit.marker_evaluation import (
    RECIPE_VERSION,
    _centred_baseline,
    _json_ready,
    prepare_marker_cohort,
    recipe_hash,
)
from survival_toolkit.marker_screen import CoxScoreScreen, fit_cox, fit_cox_null, harrell_c_many

# Intermediate files (R inputs, R outputs) live here; the tidy results go to RESULTS itself, stamped.
WORK = RESULTS / "competitors"
R_SCRIPTS = PAPER / "competitors"
DAYS_PER_MONTH = 365.25 / 12
# Every R pipeline runs with this seed: the example seed of Mime's documentation (ML.Dev.Prog.Sig(..., seed = 5201314)).
R_SEED = 5201314
# Mime's candidates: its univariate-Cox filter (p < 0.05) keeps about 2,500 of the genes; StepCox cannot fit a Cox model
# with more genes than deaths (177 in TCGA-LUAD), so the candidates are capped at the smallest univariate p-values.
# CANDIDATE_CAP is the primary analysis; the sensitivity run uses 500 candidates, where the StepCox-first models are
# infeasible (see paper/competitors/README.md).
UNICOX_P = 0.05
CANDIDATE_CAP = 100
SENSITIVITY_CAP = 500
# A gene is measured in a cohort when at most this share of the cohort's patients lack it (script 15's rule); the
# rest take the cohort median.
MAX_MISSING = 0.2
# Match script 03's explicitly configured paper bootstrap, rather than the API's smaller default.
N_BOOTSTRAP = BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = 20260926
SELECTION_COHORTS = 3


# ── Data: patients, outcomes, clinical covariates and genes of case studies I and II ────────────────────────────
def development_data() -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """TCGA-LUAD as case study I reads it: the Xena HiSeqV2 matrix matched to SurvStudio's bundled clinical table
    (484 primary tumours), overall survival in months, age, sex and stage, and every gene case study I evaluated
    (SurvStudio's missing-value and near-constant filters at their defaults). Returns the patients (patient_id,
    os_months, os_event, covariates), their expression (patients x genes, as measured, a missing value at the gene's
    median) and SurvStudio's prepared cohort (its clinical design and encoder)."""
    from survival_toolkit.marker_matrix import matrix_frame, read_marker_matrix
    from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset

    clinical = load_tcga_luad_upload_ready_dataset()
    matrix = read_marker_matrix(XENA_EXPRESSION, XENA_EXPRESSION.name, patient_ids=clinical["patient_id"].tolist())
    frame = matrix_frame(clinical, matrix, id_column="patient_id", columns=["patient_id", "os_months", "os_event", *COVARIATES])
    cohort = prepare_marker_cohort(frame, time_column="os_months", event_column="os_event", marker_columns=list(matrix.marker_names),
                                   clinical_columns=COVARIATES, categorical_clinical=CATEGORICAL, event_positive_value=1)
    patients = frame.loc[list(cohort.source_rows), ["patient_id", "os_months", "os_event", *COVARIATES]].reset_index(drop=True)
    patients["patient_id"] = patients["patient_id"].astype(str)
    if not (np.allclose(patients["os_months"].to_numpy(dtype=float), cohort.time) and np.array_equal(patients["os_event"].to_numpy(dtype=int), cohort.event)):
        raise RuntimeError("The development patients are not in the order of SurvStudio's prepared cohort.")
    expression = pd.DataFrame(_impute_median(np.array(cohort.markers, dtype=float)), columns=list(cohort.marker_names))
    return patients, expression, cohort


def validation_data() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """The seven GEO cohorts as scripts 03 and 15 build them: the QC exclusions applied, overall survival in months
    (years x 12), the patients with expression and every covariate. Returns each cohort's patients and its expression
    (patients x genes, as measured, in the patients' order)."""
    truth = pd.read_csv(LUAD / "harmonized_clinical.csv")
    cohorts = {}
    for cohort in GEO_COHORTS:
        table = truth[(truth["cohort"] == cohort) & truth["exclusion"].isna()]
        patients = pd.DataFrame({
            "patient_id": table["sample_id"].astype(str),
            "os_months": table["os_time"] * 12.0,
            "os_event": table["os_event"],
            "age": table["age"],
            "sex": table["sex"].map({"M": "Male", "F": "Female"}),
            "stage_group": table["stage_group"].map(lambda value: f"Stage {value}" if isinstance(value, str) else np.nan),
        })
        expression = pd.read_csv(LUAD / cohort / "expression_genes.csv.gz").set_index("sample_id")
        expression.index = expression.index.astype(str)
        patients = patients[patients["patient_id"].isin(expression.index)].dropna(subset=["os_months", "os_event", *COVARIATES])
        patients = patients[patients["os_months"] > 0].reset_index(drop=True)
        patients["os_event"] = patients["os_event"].astype(int)
        cohorts[cohort] = (patients, expression.loc[patients["patient_id"]].reset_index(drop=True))
    return cohorts


def _impute_median(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    if np.isnan(values).any():
        medians = np.nanmedian(np.where(np.isnan(values).all(axis=0), 0.0, values), axis=0)
        values = np.where(np.isnan(values), medians, values)
    return values


def measured(expression: pd.DataFrame) -> pd.Series:
    """Script 15's rule: a gene is measured when at most MAX_MISSING of the patients lack it and, missing values at the
    median, it varies."""
    values = expression.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing = np.isnan(values).mean(axis=0)
    spread = np.nanstd(_impute_median(values), axis=0, ddof=1)
    return pd.Series((missing <= MAX_MISSING) & (spread > 0), index=expression.columns)


def gene_set(development: pd.DataFrame, validation: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> list[str]:
    """Case study I's genes that TCGA and every GEO cohort measure, in case study I's order."""
    ok = measured(development)
    genes = [gene for gene in development.columns if ok[gene]]
    for _, expression in validation.values():
        present = [gene for gene in genes if gene in expression.columns]
        ok = measured(expression[present])
        genes = [gene for gene in present if ok[gene]]
    return genes


def zscore(values: np.ndarray) -> np.ndarray:
    """Each column centred and scaled by its SD (n - 1), as R's scale(), after median imputation."""
    values = _impute_median(values)
    spread = values.std(axis=0, ddof=1)
    return (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)


def r_frame(patients: pd.DataFrame, expression: np.ndarray, genes: Sequence[str]) -> pd.DataFrame:
    """A cohort in Mime's input layout: ID, OS.time (days), OS, then the genes."""
    frame = pd.DataFrame(expression, columns=list(genes))
    frame.insert(0, "OS", patients["os_event"].to_numpy(dtype=int))
    frame.insert(0, "OS.time", patients["os_months"].to_numpy(dtype=float) * DAYS_PER_MONTH)
    frame.insert(0, "ID", patients["patient_id"].to_numpy())
    return frame


# ── Statistics ───────────────────────────────────────────────────────────────────────────────────────────────────
def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """Harrell's C of each risk column (higher risk, earlier event), with SurvStudio's conventions."""
    return harrell_c_many(np.asarray(time, dtype=float), np.asarray(event, dtype=int), np.asarray(risk, dtype=float))


def bootstrap_c(time: np.ndarray, event: np.ndarray, risks: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                seed: int = BOOTSTRAP_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Percentile 95% intervals of Harrell's C of each risk column, drawn as SurvStudio's validate_locked_recipe draws
    them (patients resampled with replacement, draws without an event skipped, at least 20 finite draws)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risks = np.asarray(risks, dtype=float)
    risks = risks.reshape(-1, 1) if risks.ndim == 1 else risks
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_bootstrap)):
        rows = rng.integers(0, time.shape[0], size=time.shape[0])
        if not event[rows].any():
            continue
        draws.append(harrell_c_many(time[rows], event[rows], risks[rows]))
    draws = np.asarray(draws)
    lower = np.full(risks.shape[1], np.nan)
    upper = np.full(risks.shape[1], np.nan)
    for column in range(risks.shape[1]):
        finite = draws[:, column][np.isfinite(draws[:, column])] if draws.size else np.zeros(0)
        if finite.size >= 20:
            lower[column], upper[column] = np.quantile(finite, 0.025), np.quantile(finite, 0.975)
    return lower, upper


def pooled(estimates: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> dict[str, float]:
    """Random-effects pooled estimate (common.random_effects), each cohort weighted by its own 95% interval (standard
    error = width / 3.92), as script 03 pools."""
    estimates = np.asarray(estimates, dtype=float)
    errors = (np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)) / 3.92
    return random_effects(estimates, errors)


class RiskSets:
    """A cohort's death times and, for its patients in time order, where each risk set starts, so that two-group
    log-rank tests of many groupings cost a cumulative sum each (``test``)."""

    def __init__(self, time: np.ndarray, event: np.ndarray) -> None:
        time = np.asarray(time, dtype=float)
        event = np.asarray(event).astype(bool)
        self.order = np.argsort(time, kind="mergesort")
        sorted_time, sorted_event = time[self.order], event[self.order]
        self.death_times = np.unique(sorted_time[sorted_event])
        # The risk set of death time t: every patient with time >= t, from this position on.
        self.first = np.searchsorted(sorted_time, self.death_times, side="left")
        self.n_all = (time.size - self.first).astype(float)
        self.death_positions = np.flatnonzero(sorted_event)
        group = np.searchsorted(self.death_times, sorted_time[self.death_positions])
        self.d_all = np.bincount(group, minlength=self.death_times.size).astype(float)
        self.group_starts = np.searchsorted(group, np.arange(self.death_times.size), side="left")

    def test(self, groups: np.ndarray, *, ordered: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log-rank chi-square (1 df), p-value and hazard ratio (group 1 against group 0, from the observed/expected
        ratios) of each row of the 0/1 matrix ``groups`` (columns: patients, in the cohort's order, or already in time
        order with ``ordered``), as R's survdiff computes them (hypergeometric variance with tied deaths)."""
        block = np.atleast_2d(np.asarray(groups, dtype=float))
        if not ordered:
            block = block[:, self.order]
        at_risk_one = np.cumsum(block[:, ::-1], axis=1)[:, ::-1][:, self.first]
        deaths_one = np.add.reduceat(block[:, self.death_positions], self.group_starts, axis=1)
        n_all, d_all = self.n_all, self.d_all
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (d_all * at_risk_one / n_all).sum(axis=1)
            weight = np.where(n_all > 1, d_all * (n_all - d_all) / (n_all**2 * np.where(n_all > 1, n_all - 1.0, 1.0)), 0.0)
            variance = (weight * at_risk_one * (n_all - at_risk_one)).sum(axis=1)
            observed = deaths_one.sum(axis=1)
            chi2 = np.where(variance > 0, (observed - expected) ** 2 / variance, np.nan)
            hazard_ratio = (observed / expected) / ((d_all.sum() - observed) / (d_all.sum() - expected))
        return chi2, stats.chi2.sf(chi2, 1), hazard_ratio


def pooled_c_gain(cohorts: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Random-effects pooled C of the model, of the clinical-only model and of their difference over the cohorts of
    ``cohorts`` (common.validation_row's fields), as script 03 pools them: common.pooled_validation without the
    calibration slope, whose interval results written before it was pooled do not hold."""
    return {"model_c": pooled(cohorts["c"], cohorts["c_lower"], cohorts["c_upper"]),
            "clinical_c": pooled(cohorts["clinical_c"], cohorts["clinical_lower"], cohorts["clinical_upper"]),
            "delta_c": pooled(cohorts["delta_c"], cohorts["delta_lower"], cohorts["delta_upper"])}


def logrank(time: np.ndarray, event: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-group log-rank test of each row of the 0/1 matrix ``groups`` (rows: groupings, columns: patients), as R's
    survdiff computes it. Returns the chi-square (1 df), its p-value and the hazard ratio of group 1 against group 0 as
    survdiff's observed/expected ratios give it (as Mime's rs_sur reports it)."""
    return RiskSets(time, event).test(groups)


def median_split(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> tuple[float, float]:
    """Log-rank p and hazard ratio (high against low) of a risk score cut at the cohort's median, high being above
    the median (Mime's rs_sur and cal_unicox_ml_res)."""
    risk = np.asarray(risk, dtype=float)
    high = (risk > np.quantile(risk, 0.5)).astype(float)
    _, p_value, hazard_ratio = logrank(time, event, high[None, :])
    return float(p_value[0]), float(hazard_ratio[0])


def best_cutoff_scan(time: np.ndarray, event: np.ndarray, expression: np.ndarray, batch: int = 64) -> pd.DataFrame:
    """Uncorrected best-cutoff screen for every column: each value between the lower and upper quartiles
    (numpy's default quantiles, R's type 7) as the cut-off, high expression above it, and the smallest log-rank p.
    Returns per column the minimum p, its cut-off, the hazard ratio (high against low) there, the number of cut-offs
    tried, and the log-rank p of the median cut for comparison."""
    sets = RiskSets(time, event)
    values = np.asarray(expression, dtype=float)[sets.order]
    genes = values.shape[1]
    best_p, best_cut, best_hr, tried, median_p = (np.full(genes, np.nan) for _ in range(5))
    medians = np.quantile(values, 0.5, axis=0)
    _, median_p[:], _ = sets.test((values > medians).T.astype(float), ordered=True)
    for start in range(0, genes, batch):
        rows, owner, cuts = [], [], []
        for column in range(start, min(start + batch, genes)):
            x = values[:, column]
            low, high = np.quantile(x, [0.25, 0.75])
            candidates = np.unique(x[(x >= low) & (x <= high)])
            # A cut-off must leave patients on both sides.
            candidates = candidates[candidates < x.max()]
            if candidates.size == 0:
                continue
            rows.append((x[None, :] > candidates[:, None]).astype(float))
            owner.append(np.full(candidates.size, column))
            cuts.append(candidates)
        if not rows:
            continue
        block, owner, cuts = np.vstack(rows), np.concatenate(owner), np.concatenate(cuts)
        _, p_values, hazard_ratio = sets.test(block, ordered=True)
        # The smallest p of each column (the first such cut-off, from low to high, on ties).
        p_sort = np.where(np.isfinite(p_values), p_values, np.inf)
        order = np.lexsort((p_sort, owner))
        first = order[np.r_[True, owner[order][1:] != owner[order][:-1]]]
        counts = np.bincount(owner - start, minlength=min(batch, genes - start))
        for pick in first:
            column = owner[pick]
            tried[column] = counts[column - start]
            if np.isfinite(p_values[pick]):
                best_p[column], best_cut[column], best_hr[column] = p_values[pick], cuts[pick], hazard_ratio[pick]
    return pd.DataFrame({"best_p": best_p, "cutoff": best_cut, "hazard_ratio": best_hr, "cutoffs_tried": tried, "median_p": median_p})


def adjusted_statistics(patients: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    """Script 15's test in one cohort: each gene standardised within the cohort, SurvStudio's clinically adjusted Cox
    score test (Efron ties) against age, sex and stage, the one-step log hazard ratio per SD with standard error
    1 / sqrt(information). ``expression`` holds the genes measured in this cohort (missing values at the median)."""
    design = pd.get_dummies(patients[COVARIATES], columns=CATEGORICAL, drop_first=True, dtype=float).to_numpy(dtype=float)
    design = design[:, np.ptp(design, axis=0) > 0]
    time = patients["os_months"].to_numpy(dtype=float)
    event = patients["os_event"].to_numpy(dtype=int)
    values = _impute_median(expression.to_numpy(dtype=float))
    spread = values.std(axis=0, ddof=1)
    standardised = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    null = fit_cox_null(time, event, design)
    screen = CoxScoreScreen(time, event, null=null, Z=design).statistics(standardised)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = 1.0 / np.sqrt(screen.information)
    table = pd.DataFrame({"gene": list(expression.columns), "log_hr": screen.beta_one_step, "se": se})
    return table[np.isfinite(table["log_hr"]) & np.isfinite(table["se"]) & (table["se"] > 0)].reset_index(drop=True)


def replicated(tests: pd.DataFrame, direction: pd.Series) -> pd.DataFrame:
    """Script 15's replication rule for each gene of ``direction`` (+1 or -1, the claimed direction): over the cohorts
    measuring it (at least two), the random-effects pooled log hazard ratio has the claimed direction and its 95% CI
    excludes zero. ``tests``: gene, cohort, log_hr, se."""
    rows = []
    for gene, part in tests[tests["gene"].isin(direction.index)].groupby("gene", sort=False):
        record = {"gene": gene, "direction": int(direction[gene]), "cohorts": len(part),
                  "same_direction_share": float((np.sign(part["log_hr"]) == direction[gene]).mean())}
        if len(part) >= 2:
            summary = random_effects(part["log_hr"].to_numpy(), part["se"].to_numpy())
            signed = summary["estimate"] * direction[gene]
            record.update(pooled_log_hr=summary["estimate"], ci_lower=summary["ci_lower"], ci_upper=summary["ci_upper"],
                          replicated=bool(signed > 0 and (summary["ci_lower"] > 0 or summary["ci_upper"] < 0)))
        rows.append(record)
    return pd.DataFrame(rows)


# ── Selection replay (P1) ────────────────────────────────────────────────────────────────────────────────────────
def splits(cohorts: Sequence[str], selection: int = SELECTION_COHORTS) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Every split of the cohorts into ``selection`` selection cohorts and the rest sealed, in lexicographic order."""
    cohorts = list(cohorts)
    return [(chosen, tuple(cohort for cohort in cohorts if cohort not in chosen)) for chosen in itertools.combinations(cohorts, selection)]


def choose_winner(reported: pd.DataFrame, selection: Sequence[str]) -> tuple[str | None, float]:
    """The model with the highest mean reported C over the selection cohorts (``reported``: models x cohorts, in the
    tool's order; a model without a C in one of them cannot win; the first of tied models wins)."""
    means = reported[list(selection)].mean(axis=1, skipna=False)
    if means.notna().sum() == 0:
        return None, float("nan")
    best = means.max()
    winner = means.index[np.flatnonzero((means == best).to_numpy())[0]]
    return str(winner), float(best)


def replay(reported: pd.DataFrame, honest: pd.DataFrame, honest_lower: pd.DataFrame | None = None, honest_upper: pd.DataFrame | None = None,
           training: pd.Series | None = None, selection: int = SELECTION_COHORTS) -> pd.DataFrame:
    """Replay a paper's selection for every split of the cohorts (columns of ``reported``) into ``selection`` selection
    and the rest sealed: the winner, the number it would report (its mean reported C over the selection cohorts), its
    training C, and its mean reported C, mean honest C and (with intervals) random-effects pooled honest C over the
    sealed cohorts."""
    rows = []
    for chosen, sealed in splits(list(reported.columns), selection):
        winner, value = choose_winner(reported, chosen)
        record: dict[str, Any] = {"selection": ";".join(chosen), "sealed": ";".join(sealed), "winner": winner, "reported_c": value}
        if winner is not None:
            record["training_c"] = float(training[winner]) if training is not None else np.nan
            record["sealed_reported_mean"] = float(reported.loc[winner, list(sealed)].mean())
            record["sealed_honest_mean"] = float(honest.loc[winner, list(sealed)].mean())
            if honest_lower is not None and honest_upper is not None and len(sealed) >= 2:
                summary = pooled(honest.loc[winner, list(sealed)], honest_lower.loc[winner, list(sealed)], honest_upper.loc[winner, list(sealed)])
                record.update(sealed_honest_pooled=summary["estimate"], sealed_honest_pooled_lower=summary["ci_lower"],
                              sealed_honest_pooled_upper=summary["ci_upper"])
            record["optimism_vs_sealed_mean"] = record["reported_c"] - record["sealed_honest_mean"]
            record["selection_optimism"] = record["reported_c"] - record["sealed_reported_mean"]
        rows.append(record)
    return pd.DataFrame(rows)


# ── Added value over the clinical covariates (SurvStudio's external gain) ──────────────────────────────────────────
def risk_score_recipe(cohort: Any, risk: np.ndarray, label: str) -> dict[str, Any]:
    """A SurvStudio locked recipe of the clinical covariates plus one risk score, fitted on the development cohort
    (SurvStudio's prepared TCGA cohort: its clinical design and encoder) by Cox regression with Efron ties, with the
    locked clinical-only model beside it, so that validate_locked_recipe computes SurvStudio's external C, clinical C
    and paired gain for the risk score exactly as for SurvStudio's own locked model (script 03)."""
    risk = np.asarray(risk, dtype=float)
    design = np.asarray(cohort.clinical, dtype=float)
    clinical_names = list(cohort.clinical_names)
    full = fit_cox(cohort.time, cohort.event, np.column_stack([design, risk]), ties="efron")
    clinical = fit_cox(cohort.time, cohort.event, design, ties="efron")
    if not (full.converged and clinical.converged and np.isfinite(full.beta).all() and np.isfinite(clinical.beta).all()):
        raise RuntimeError(f"{label}: the Cox model of the clinical covariates and the risk score did not converge on TCGA.")
    linear_predictor = np.column_stack([design, risk]) @ full.beta
    recipe: dict[str, Any] = {
        "recipe_version": RECIPE_VERSION,
        "created_with": f"paper/scripts/competitors.py: clinical covariates plus the risk score of {label}",
        "outcome": {"time_column": "os_months", "event_column": "os_event", "event_positive_value": 1},
        "clinical": {"columns": list(cohort.clinical_columns), "categorical": list(CATEGORICAL), "encoder": cohort.clinical_encoder},
        "strata_columns": [],
        "markers": ["risk_score"],
        "marker_medians": {"risk_score": float(np.median(risk))},
        "marker_scale": {"risk_score": {"mean": float(np.mean(risk)), "sd": float(np.std(risk, ddof=1))}},
        "marker_development_log_hr": {"risk_score": float(full.beta[-1])},
        "primary_lens": "added_value",
        "model": {"terms": [*clinical_names, "risk_score"], "coefficients": full.beta, "ties": "efron",
                  "baseline": _centred_baseline(cohort.time, cohort.event, linear_predictor, "efron"),
                  "default_horizon": float(np.median(cohort.time[cohort.event == 1]))},
        "clinical_only_model": {"terms": clinical_names, "coefficients": clinical.beta},
        "development": {"n": int(cohort.time.shape[0]), "events": int(cohort.event.sum()), "row_mask_hash": cohort.row_mask_hash},
    }
    recipe = _json_ready(recipe)
    recipe["recipe_hash"] = recipe_hash(recipe)
    return recipe


def external_gain(recipe: dict[str, Any], patients: pd.DataFrame, risk: np.ndarray, scaling: str = "as_measured") -> dict[str, Any]:
    """SurvStudio's validate_locked_recipe of a risk-score recipe in one cohort (bootstrap and seed as script 03), as
    common.validation_row's fields."""
    from common import validation_row
    from survival_toolkit.marker_evaluation import validate_locked_recipe

    frame = patients[["patient_id", "os_months", "os_event", *COVARIATES]].copy()
    frame["risk_score"] = np.asarray(risk, dtype=float)
    return validation_row(validate_locked_recipe(frame, recipe, marker_scaling=scaling,
                                                 n_bootstrap=N_BOOTSTRAP, random_seed=BOOTSTRAP_SEED))


def external_gain_point(recipe: dict[str, Any], patients: pd.DataFrame, risk: np.ndarray) -> float:
    """The paired gain of external_gain without its bootstrap (for the null's 3,000 truth patients)."""
    from survival_toolkit.marker_evaluation import validate_locked_recipe

    frame = patients[["patient_id", "os_months", "os_event", *COVARIATES]].copy()
    frame["risk_score"] = np.asarray(risk, dtype=float)
    delta = validate_locked_recipe(frame, recipe, n_bootstrap=0)["metrics"].get("delta_c_index")
    return float("nan") if delta is None else float(delta)


# ── Provenance ──────────────────────────────────────────────────────────────────────────────────────────────────────
def code_files() -> list[Path]:
    """The comparison's own code: this module, the 18_competitors scripts and the R scripts."""
    return sorted([SCRIPTS / "competitors.py", *SCRIPTS.glob("18_competitors_*.py"), *R_SCRIPTS.glob("*.R"), *R_SCRIPTS.glob("*.sh")],
                  key=lambda path: path.name)


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in code_files():
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def r_versions() -> dict[str, str]:
    """The versions setup_r_env.sh recorded (paper/competitors/versions.txt), as name: value."""
    path = R_SCRIPTS / "versions.txt"
    if not path.exists():
        return {}
    pairs = (line.split(":", 1) for line in path.read_text(encoding="utf-8").splitlines() if ":" in line)
    return {name.strip(): value.strip() for name, value in pairs}


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def clean(value: Any) -> Any:
    """A result ready for strict JSON: every non-finite number as null, numpy scalars as Python ones."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return finite_or_none(value)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A data frame as JSON-ready records (NaN as null)."""
    return [{key: finite_or_none(value) for key, value in row.items()} for row in frame.to_dict("records")]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarise(values: Iterable[float]) -> dict[str, float | None]:
    """Mean, median, 2.5th and 97.5th percentiles, min and max of the finite values."""
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "p2_5": None, "p97_5": None, "min": None, "max": None}
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)), "p2_5": float(np.quantile(array, 0.025)),
            "p97_5": float(np.quantile(array, 0.975)), "min": float(array.min()), "max": float(array.max())}

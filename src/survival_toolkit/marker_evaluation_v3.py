# Versioned v3 pipeline derived from d9e8102f marker_evaluation.py.
# Keep the qualified v2 module unchanged. Cox, transformations, permutation and
# prediction calculations are identical; diagnostic calls use marker_bootstrap.
"""One-step evaluation of candidate prognostic markers that checks the selection itself.

A marker list is screened, and the whole screening procedure is then checked the way
a reader should check a published marker claim:

1. Two lenses per marker, both Cox score tests from ``marker_screen``: *marginal*
   association (no covariates) and *added value* over the clinical baseline.
2. Westfall–Young max-statistic permutation p-values over all markers (family-wise
   error) and permutation FDR q-values. The added-value null permutes the marker
   residuals left after projecting on the clinical covariates. This approximates
   a conditional null and requires exchangeable residuals; nonlinear relations
   can invalidate calibration. A misspecified clinical Cox baseline, including
   non-proportional hazards, can also invalidate conditional-null interpretation;
   multiplicity adjustment does not establish model adequacy. Permuting the residualised
   regressor of interest is the Smith scheme (Winkler et al. 2014, NeuroImage
   92:381-397, Table 2), not Freedman–Lane, which permutes residuals of the outcome.
3. The whole procedure rerun on subsamples: how often each marker is selected, its
   rank interval, and whether its direction holds.
4. Tiers from pre-declared rules, and an optimism report for a signature built from
   the selected markers ("winner's curse" of picking the strongest marker), with 95%
   intervals for its left-out C-index and its gain over the clinical covariates that
   allow for the overlap of the subsamples (corrected resampled t).
5. A screen for patients whose marker profiles are identical or near-identical
   (``survival_toolkit.duplicates``): a patient in the data twice can sit on both sides
   of a subsample split and flatter the internal estimates.

Nothing here dichotomizes markers; every statistic uses the continuous values.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, NamedTuple, NoReturn, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from survival_toolkit.analysis import (
    _build_cox_strata_payload,
    _cohort_frame,
    _harrell_c_index_counts,
)
from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.duplicates import possible_duplicates
from survival_toolkit.encoding import fit_feature_encoder, transform_feature_encoder
from survival_toolkit.clinical_basis import (
    CLINICAL_BASES, ClinicalBasisError, check_clinical_encoder,
    fit_clinical_encoder, transform_clinical_encoder,
)
from survival_toolkit.marker_diagnostics import SUPPORTED_METHOD_VERSIONS as LEGACY_METHOD_VERSIONS
from survival_toolkit.marker_bootstrap import METHOD_VERSION, N_BOOTSTRAP, CANDIDATES, diagnose_markers
SUPPORTED_METHOD_VERSIONS = LEGACY_METHOD_VERSIONS | {METHOD_VERSION}
from survival_toolkit.errors import user_input_boundary
from survival_toolkit.marker_screen import (
    TIES_METHODS,
    CoxNull,
    CoxScoreScreen,
    MaxTAccumulator,
    PermutationFdrAccumulator,
    ScoreStats,
    bh_vector,
    fit_cox,
    fit_cox_null,
    harrell_c_many,
    residualize,
    stratified_permutation,
)

LENSES = ("marginal", "added_value")
TIER_ORDER = ("robust", "suggestive", "marginal only", "not supported", "inference withheld")
_PERMUTATION_BATCH = 16
# Null schemes for the added-value lens: "smith" permutes each marker's residual after regression on the
# clinical covariates; "raw" permutes the markers themselves. "freedman_lane", the earlier (wrong) name
# of the first scheme, is still accepted.
LENS2_NULLS = ("smith", "raw")
_LENS2_NULL_ALIASES = {"freedman_lane": "smith"}
# Linear predictors are clipped at this distance from the development mean before exponentiating.
_LP_CLIP = 50.0
# Above this share of external rows at a categorical level the locked model has not seen, validation stops.
MAX_UNSEEN_LEVEL_SHARE = 0.5


class ClinicalModelNotConvergedError(ValueError):
    """The clinical-only Cox model of a set of rows did not converge: a failure of the data, not of the code.

    A subsample whose clinical model fails this way is counted as failed; on the full cohort the message
    reaches the user. It is a ValueError, so callers that catch ValueError still do.
    """


class MarkerSettings(NamedTuple):
    """Pre-declared settings; every threshold used for a tier is listed here."""

    ties: str = "efron"
    alpha: float = 0.05
    fdr_level: float = 0.10
    n_permutations: int = 1000
    n_resamples: int = 200
    resample_fraction: float = 0.632
    max_missing_fraction: float = 0.2
    # Markers with more than this share of patients at one value are left out before the screen. A gene
    # expressed in a few patients has a heavy-tailed score statistic, and the permutation maximum is then
    # made of such genes: in TCGA-LUAD RNA-seq its 95% point was chi-square 239 (about 25 without them)
    # and no marker could pass. The filter never looks at the outcome, but this alone
    # does not establish error control: the permutation assumptions must also hold.
    max_mode_fraction: float = 0.9
    # In the benchmark pilot a 0.8 frequency kept 6% of true markers robust against 21% at
    # 0.5, with family-wise error at most 1% either way; direction 0.9 vs 0.95 made no difference.
    robust_frequency: float = 0.5
    robust_direction: float = 0.9
    nonlinear_frequency: float = 0.8
    max_signature_markers: int = 10
    shortlist_size: int = 50
    lens2_null: str = "smith"
    nonlinear_lens: str = "off"
    nonlinear_replicates: int = 30
    nonlinear_top_markers: int = 100
    random_seed: int = 20260926
    clinical_basis: str = "linear"
    diagnostic_candidate: str | None = None
    diagnostic_bootstraps: int = N_BOOTSTRAP


class MarkerCohort(NamedTuple):
    time: np.ndarray
    event: np.ndarray
    markers: np.ndarray
    marker_names: list[str]
    clinical: np.ndarray | None
    clinical_names: list[str]
    clinical_columns: list[str]
    strata: np.ndarray | None
    strata_columns: list[str]
    source_rows: list[Any]
    row_mask_hash: str
    dropped_markers: list[dict[str, Any]]
    clinical_encoder: dict[str, Any] | None
    # Encoded clinical columns left out because a Cox model cannot estimate them (see prepare_marker_cohort).
    dropped_clinical: tuple[dict[str, str], ...] = ()
    # Notes on what preparing the cohort left out, for the result's cohort notes.
    notes: tuple[str, ...] = ()
    clinical_frame: pd.DataFrame | None = None
    categorical_clinical: tuple[str, ...] = ()
    clinical_basis: str = "linear"
    clinical_error: str | None = None


class LensStats(NamedTuple):
    chi2: np.ndarray
    z: np.ndarray
    p_value: np.ndarray
    q_bh: np.ndarray
    beta_one_step: np.ndarray


class ProcedureFit(NamedTuple):
    lenses: dict[str, LensStats]
    medians: np.ndarray
    selected: dict[str, np.ndarray]
    ranks: dict[str, np.ndarray]


class ResamplingSummary(NamedTuple):
    n_valid: int
    n_failed: int
    selection_frequency: dict[str, np.ndarray]
    direction_consistency: dict[str, np.ndarray]
    median_rank: dict[str, np.ndarray]
    rank_low: dict[str, np.ndarray]
    rank_high: dict[str, np.ndarray]
    optimism: dict[str, Any]
    n_withheld: int = 0


def _validated_settings(settings: MarkerSettings) -> MarkerSettings:
    if settings.diagnostic_candidate is not None and settings.diagnostic_candidate not in CANDIDATES:
        raise ValueError("Unknown joint diagnostic candidate")
    if settings.diagnostic_bootstraps != N_BOOTSTRAP:
        raise ValueError("Product v3 requires exactly 9999 diagnostic draws")
    if settings.clinical_basis not in CLINICAL_BASES:
        raise ValueError(f"clinical_basis must be one of {CLINICAL_BASES}.")
    if settings.ties not in TIES_METHODS:
        raise ValueError(f"ties must be one of {TIES_METHODS}.")
    if not 0.0 < settings.alpha < 0.5 or not 0.0 < settings.fdr_level < 0.5:
        raise ValueError("alpha and fdr_level must be between 0 and 0.5.")
    if settings.n_permutations < 0 or settings.n_resamples < 0:
        raise ValueError("n_permutations and n_resamples cannot be negative.")
    if not 0.3 <= settings.resample_fraction <= 0.9:
        raise ValueError("resample_fraction must be between 0.3 and 0.9.")
    if not 0.0 <= settings.max_missing_fraction < 1.0:
        raise ValueError("max_missing_fraction must be at least 0 and below 1.")
    if not 0.5 <= settings.max_mode_fraction <= 1.0:
        raise ValueError("max_mode_fraction must be between 0.5 and 1.")
    settings = settings._replace(lens2_null=_LENS2_NULL_ALIASES.get(settings.lens2_null, settings.lens2_null))
    if settings.lens2_null not in LENS2_NULLS:
        raise ValueError('lens2_null must be "smith" or "raw".')
    if not all(0.0 <= value <= 1.0 for value in (settings.robust_frequency, settings.robust_direction, settings.nonlinear_frequency)):
        raise ValueError("robust_frequency, robust_direction and nonlinear_frequency must be between 0 and 1.")
    if settings.max_signature_markers < 1 or settings.shortlist_size < 0:
        raise ValueError("max_signature_markers must be at least 1 and shortlist_size at least 0.")
    if settings.nonlinear_lens not in {"off", "gbs", "rsf"}:
        raise ValueError('nonlinear_lens must be "off", "gbs" or "rsf".')
    if settings.nonlinear_replicates < 1 or settings.nonlinear_top_markers < 1:
        raise ValueError("nonlinear_replicates and nonlinear_top_markers must be at least 1.")
    return settings


def prepare_marker_cohort(
    df: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    marker_columns: Sequence[str],
    clinical_columns: Sequence[str] = (),
    categorical_clinical: Sequence[str] = (),
    strata_columns: Sequence[str] = (),
    event_positive_value: Any = None,
    max_missing_fraction: float = 0.2,
    max_mode_fraction: float = 0.9,
    clinical_basis: str = "linear",
) -> MarkerCohort:
    """Align the outcome, clinical design, strata and numeric marker block row by row.

    Rows missing the outcome or a stratum are dropped. Clinical imputation and encoding
    are learned in training rows and frozen for prediction. A missing marker value never drops a row: markers with more missing
    values than ``max_missing_fraction`` are left out, and the rest are imputed inside
    each fit of the procedure. Markers with more than ``max_mode_fraction`` of their
    values equal to one value, counting missing values at the median they are imputed
    with, are left out as near-constant. Markers holding infinite values are left out
    with a note (``notes``). Encoded clinical columns that a Cox model cannot
    estimate (constant within every stratum, or a linear combination of the columns before
    them) are left out and listed in ``dropped_clinical``.
    """
    markers = list(dict.fromkeys(str(column) for column in marker_columns))
    clinical = list(dict.fromkeys(str(column) for column in clinical_columns))
    strata = list(dict.fromkeys(str(column) for column in strata_columns))
    if not markers:
        raise ValueError("Select at least one candidate marker.")
    reserved = {str(time_column), str(event_column), *clinical, *strata}
    overlap = [column for column in markers if column in reserved]
    if overlap:
        raise ValueError(
            "Markers cannot also be the outcome, a clinical covariate or a stratum: " + ", ".join(overlap[:5]) + "."
        )
    outcome = {str(time_column), str(event_column)}
    for names, role in ((clinical, "Clinical covariates"), (strata, "Strata")):
        clash = [column for column in names if column in outcome]
        if clash:
            raise ValueError(f"{role} cannot also be the outcome: " + ", ".join(clash[:5]) + ".")
    both = [column for column in clinical if column in set(strata)]
    if both:
        raise ValueError(
            "A column cannot be both a clinical covariate and a stratum: " + ", ".join(both[:5]) + ". Use it as one or the other."
        )
    missing = [column for column in [*markers, *clinical, *strata] if column not in df.columns]
    if missing:
        raise ValueError("Columns not found in the dataset: " + ", ".join(missing[:5]) + ".")
    df = _unique_rows(df)

    # Only the outcome and clinical columns go through the cohort checks, which inspect every
    # column they are given; a genome-wide panel would make them the slowest step.
    frame = _cohort_frame(
        df[list(dict.fromkeys([time_column, event_column, *clinical, *strata]))],
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=strata,
    )
    source_rows = list(frame.attrs["source_row_index"])
    for column in clinical:
        frame[column] = df.loc[source_rows, column].reset_index(drop=True)
    raw_markers = df.loc[source_rows, markers]
    non_numeric: list[str] = []
    values = np.empty((len(source_rows), len(markers)), dtype=float)
    # Numeric columns convert in one step (a genome-wide panel has tens of thousands);
    # only text-typed columns are checked value by value.
    is_numeric = [pd.api.types.is_numeric_dtype(dtype) for dtype in raw_markers.dtypes]
    numeric_positions = [index for index, numeric in enumerate(is_numeric) if numeric]
    if numeric_positions:
        values[:, numeric_positions] = raw_markers.iloc[:, numeric_positions].to_numpy(dtype=float, na_value=np.nan)
    for index, numeric in enumerate(is_numeric):
        if numeric:
            continue
        series = raw_markers.iloc[:, index]
        coerced = pd.to_numeric(series, errors="coerce")
        if bool((series.notna() & coerced.isna()).any()):
            non_numeric.append(markers[index])
            continue
        values[:, index] = coerced.to_numpy(dtype=float, na_value=np.nan)
    if non_numeric:
        raise ValueError(
            "Markers must be numeric; these columns contain text: "
            + ", ".join(non_numeric[:5])
            + (" ..." if len(non_numeric) > 5 else "")
            + ". Recode text values as blank cells or leave the column out."
        )
    # An infinite value (the log of zero, say) is not a missing value: imputing it at the median would
    # hide the patients with the most extreme values, so such markers are left out instead.
    infinite = np.isinf(values).sum(axis=0)
    values[~np.isfinite(values)] = np.nan
    missing_share = np.mean(np.isnan(values), axis=0)
    kept: list[int] = []
    dropped: list[dict[str, Any]] = []
    notes: list[str] = []
    for index, column in enumerate(markers):
        if infinite[index]:
            # One reason text for all, so reports can count markers by reason; the number of values apart.
            dropped.append({"marker": column, "reason": "infinite values", "n_infinite": int(infinite[index])})
            continue
        observed = values[:, index][~np.isnan(values[:, index])]
        if missing_share[index] > max_missing_fraction:
            dropped.append({"marker": column, "reason": f"{missing_share[index]:.0%} missing"})
            continue
        distinct, counts = np.unique(observed, return_counts=True) if observed.size else (np.zeros(0), np.zeros(0, dtype=int))
        n_missing = values.shape[0] - observed.size
        if n_missing and observed.size:
            # Every fit fills missing values with the median, so they count at that value.
            median = float(np.median(observed))
            position = int(np.searchsorted(distinct, median))
            if position < distinct.size and distinct[position] == median:
                counts = counts.copy()
                counts[position] += n_missing
            else:
                counts = np.append(counts, n_missing)
        if counts.size < 2:
            dropped.append({"marker": column, "reason": "constant"})
        elif counts.max() > max_mode_fraction * values.shape[0]:
            dropped.append({"marker": column, "reason": f"near-constant ({counts.max() / values.shape[0]:.0%} at one value)"})
        else:
            kept.append(index)
    finite_hint = "Transform them so every value is finite, for example log(x + 1) instead of log(x)."
    if not kept:
        reasons = "; ".join(f"{item['marker']} ({item['reason']})" for item in dropped[:5])
        raise ValueError(
            f"No usable markers remain: {reasons}{' ...' if len(dropped) > 5 else ''}. "
            f"Markers may have at most {max_missing_fraction:.0%} missing values and need more than one common value"
            + (f", and no infinite values. {finite_hint}" if infinite.any() else ".")
        )
    if infinite.any():
        names = [markers[index] for index in np.flatnonzero(infinite)]
        notes.append(
            f"{len(names)} marker(s) hold infinite values (such as the log of zero) and were left out: "
            f"{', '.join(names[:5])}{' ...' if len(names) > 5 else ''}. {finite_hint}"
        )

    clinical_design = None
    clinical_names: list[str] = []
    encoder = None
    dropped_clinical: list[dict[str, str]] = []
    clinical_error = None
    strata_codes = _build_cox_strata_payload(frame, strata)["codes"] if strata else None
    if clinical:
        try:
            encoder = fit_clinical_encoder(frame, clinical, categorical_clinical, basis=clinical_basis)
        except ClinicalBasisError as exc:
            clinical_error = str(exc)
            # Preserve the requested basis and raw input; do not fit a replacement model.
            encoder = None
        if encoder is None:
            clinical_design = np.zeros((len(frame), 0))
    if clinical and encoder is not None:
        # A model term names each column; a marker called like a level indicator ("grade_2") would give the
        # locked model two terms of one name, which no validation could tell apart.
        encoded_names = set(encoder["feature_names"])
        clash = [column for column in markers if column in encoded_names]
        if clash:
            raise ValueError(
                "Markers cannot have the name of an encoded clinical covariate: " + ", ".join(clash[:5])
                + ". A categorical covariate is encoded as one column per level, named like grade_2; rename the marker column."
            )
        clinical_design = np.asarray(transform_clinical_encoder(frame, encoder, output="numpy"), dtype=float)
        clinical_names = list(encoder["feature_names"])
        redundant = _redundant_clinical_columns(clinical_design, None if strata_codes is None else np.asarray(strata_codes))
        if redundant:
            dropped_clinical = [{"column": clinical_names[index], "reason": reason} for index, reason in redundant.items()]
            keep = [index for index in range(len(clinical_names)) if index not in redundant]
            clinical_design = clinical_design[:, keep]
            clinical_names = [clinical_names[index] for index in keep]
        if not clinical_names:
            clinical_design = np.zeros((len(frame), 0))
            clinical_error = "No requested clinical covariate could be estimated; added-value inference is withheld."
    return MarkerCohort(
        time=frame[time_column].to_numpy(dtype=float),
        event=frame[event_column].to_numpy(dtype=int),
        markers=values[:, kept],
        marker_names=[markers[index] for index in kept],
        clinical=clinical_design,
        clinical_names=clinical_names,
        clinical_columns=clinical,
        strata=None if strata_codes is None else np.asarray(strata_codes),
        strata_columns=strata,
        source_rows=source_rows,
        row_mask_hash=str(frame.attrs.get("row_mask_hash") or ""),
        dropped_markers=dropped,
        clinical_encoder=encoder,
        dropped_clinical=tuple(dropped_clinical),
        notes=tuple(notes),
        clinical_frame=frame[clinical].copy() if clinical else None,
        categorical_clinical=tuple(categorical_clinical),
        clinical_basis=clinical_basis,
        clinical_error=clinical_error,
    )


def _redundant_clinical_columns(design: np.ndarray, strata: np.ndarray | None) -> dict[int, str]:
    """Encoded clinical columns a (stratified) Cox model cannot estimate, with the reason.

    A column that is constant within every stratum is absorbed by the strata, and one that is a
    linear combination of the columns before it (after removing stratum means) is aliased; either
    makes the clinical model singular, so its fit would not converge.
    """
    if strata is None:
        centred = design - design.mean(axis=0, keepdims=True)
    else:
        _, codes = np.unique(strata, return_inverse=True)
        codes = codes.reshape(-1)
        sums = np.zeros((int(codes.max()) + 1, design.shape[1]))
        np.add.at(sums, codes, design)
        centred = design - (sums / np.bincount(codes)[:, None])[codes]
    redundant: dict[int, str] = {}
    basis = np.zeros((design.shape[0], 0))
    for index in range(design.shape[1]):
        spread = float(np.linalg.norm(design[:, index] - design[:, index].mean()))
        column = centred[:, index]
        norm = float(np.linalg.norm(column))
        if spread == 0.0:
            redundant[index] = "constant"
            continue
        if norm <= 1e-9 * spread:
            redundant[index] = "constant within every stratum"
            continue
        residual = column - basis @ (basis.T @ column)
        remaining = float(np.linalg.norm(residual))
        if remaining <= 1e-8 * norm:
            redundant[index] = "a linear combination of other clinical columns"
            continue
        basis = np.column_stack([basis, residual / remaining])
    return redundant


def _impute(block: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(block), medians[None, :], block)


def _column_medians(block: np.ndarray) -> np.ndarray:
    if not block.size:
        return np.zeros(block.shape[1])
    medians = np.empty(block.shape[1], dtype=float)
    missing = np.isnan(block).any(axis=0)
    # np.nanmedian is many times slower than np.median; only columns with gaps need it.
    # Medians along contiguous rows of the transposed block: partitioning down the columns of
    # a wide row-major block is cache-unfriendly.
    if (~missing).any():
        medians[~missing] = np.median(np.ascontiguousarray(block[:, ~missing].T), axis=1)
    if missing.any():
        with np.errstate(all="ignore"):
            medians[missing] = np.nanmedian(np.ascontiguousarray(block[:, missing].T), axis=1)
    return np.where(np.isfinite(medians), medians, 0.0)


def _estimable_clinical_columns(design: np.ndarray, strata: np.ndarray | None) -> np.ndarray:
    """Positions of the clinical columns a Cox model can estimate in these rows, in their order.

    A subsample, a left-out set or an external cohort can lose a level of a categorical covariate.
    An indicator that no longer varies is left out, and so is one that became a linear combination
    of the columns before it, as when the reference level is missing and the remaining indicators
    sum to one: fitted as they are, the clinical model is singular. The columns kept keep their names.
    Without such a loss (as in the full cohort, whose design ``prepare_marker_cohort`` already reduced)
    every varying column is kept.
    """
    varying = np.flatnonzero(np.ptp(design, axis=0) > 0) if design.shape[1] else np.zeros(0, dtype=np.int64)
    if varying.size == 0:
        return varying
    redundant = _redundant_clinical_columns(design[:, varying], strata)
    return np.asarray([column for position, column in enumerate(varying) if position not in redundant], dtype=np.int64)


def _estimable_clinical_design(design: np.ndarray, strata: np.ndarray | None) -> np.ndarray:
    return design[:, _estimable_clinical_columns(design, strata)]


def _linear_predictor(design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """``design @ beta`` with the coefficients of columns the fit could not estimate (NaN) counted as zero."""
    return design @ np.where(np.isnan(beta), 0.0, beta)


def _clinical_null_model(
    time: np.ndarray,
    event: np.ndarray,
    clinical: np.ndarray,
    strata: np.ndarray | None,
    ties: str,
) -> tuple[np.ndarray, CoxNull | None]:
    """The clinical-only null model on these rows and the design it was fitted on (None without a column to fit).

    Columns the Cox fit still cannot estimate are left out of the design too: aliased within the risk
    sets although not in the rows, as a level whose patients were all censored before the first event.
    """
    design = _estimable_clinical_design(clinical, strata)
    if not design.shape[1]:
        raise ClinicalModelNotConvergedError("The requested clinical model has no estimable terms.")
    fit = fit_cox(time, event, design, strata, ties)
    if not fit.converged or not np.isfinite(fit.beta).all() or not np.isfinite(fit.covariance).all() or np.any(fit.separated):
        raise ClinicalModelNotConvergedError("The clinical-only Cox model has nonconvergence, separation, aliasing or a non-finite fit.")
    null = CoxNull(eta=design @ fit.beta, beta=fit.beta, covariance=fit.covariance,
                   loglik=fit.loglik, converged=True)
    return design, null


def _training_clinical(cohort: MarkerCohort, rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None, list[str]]:
    """Fit only in these training rows; the resulting encoder is frozen for prediction."""
    if cohort.clinical_error:
        raise ClinicalModelNotConvergedError(cohort.clinical_error)
    if cohort.clinical_frame is None:
        design = cohort.clinical[rows]
        return design, cohort.clinical_encoder, cohort.clinical_names
    try:
        encoder = fit_clinical_encoder(cohort.clinical_frame.iloc[rows], cohort.clinical_columns,
                                      cohort.categorical_clinical, basis=cohort.clinical_basis)
        design = transform_clinical_encoder(cohort.clinical_frame.iloc[rows], encoder)
        strata = None if cohort.strata is None else cohort.strata[rows]
        redundant = _redundant_clinical_columns(design, strata)
        if redundant and cohort.clinical_basis == "restricted_cubic_spline":
            raise ClinicalBasisError("The requested spline basis is rank deficient in the training rows.")
        keep = [i for i in range(design.shape[1]) if i not in redundant]
        return design[:, keep], encoder, [encoder["feature_names"][i] for i in keep]
    except ClinicalBasisError as exc:
        raise ClinicalModelNotConvergedError(str(exc)) from exc


def _frozen_clinical(cohort: MarkerCohort, rows: np.ndarray, encoder: dict[str, Any] | None,
                     names: list[str]) -> np.ndarray:
    if cohort.clinical_frame is None or encoder is None:
        return cohort.clinical[rows]
    frame = transform_clinical_encoder(cohort.clinical_frame.iloc[rows], encoder, output="dataframe")
    return frame[names].to_numpy(float)


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, -np.inf)
    return stats.rankdata(-filled, method="min").astype(np.int32)


def _lens_stats(score: ScoreStats) -> LensStats:
    return LensStats(
        chi2=score.chi2,
        z=score.z,
        p_value=score.p_value,
        q_bh=bh_vector(score.p_value),
        beta_one_step=score.beta_one_step,
    )


def _lens_screens(
    cohort: MarkerCohort,
    rows: np.ndarray,
    ties: str,
) -> dict[str, tuple[CoxScoreScreen, np.ndarray | None]]:
    """The score screen of each lens on these rows, with the clinical design it uses."""
    time = cohort.time[rows]
    event = cohort.event[rows]
    strata = None if cohort.strata is None else cohort.strata[rows]
    screens: dict[str, tuple[CoxScoreScreen, np.ndarray | None]] = {}
    marginal_null = fit_cox_null(time, event, None, strata, ties)
    screens["marginal"] = (CoxScoreScreen(time, event, null=marginal_null, strata=strata, ties=ties), None)
    if cohort.clinical is not None:
        try:
            design, _, _ = _training_clinical(cohort, rows)
            design, clinical_null = _clinical_null_model(time, event, design, strata, ties)
        except ClinicalModelNotConvergedError:
            if len(rows) != len(cohort.time):
                raise
            screens["added_value"] = (None, None)
            return screens
        if clinical_null is not None:
            screens["added_value"] = (
                CoxScoreScreen(time, event, null=clinical_null, Z=design, strata=strata, ties=ties),
                design,
            )
        else:
            screens["added_value"] = (None, None)
    return screens


def run_procedure(cohort: MarkerCohort, rows: np.ndarray, settings: MarkerSettings) -> ProcedureFit:
    """The whole selection procedure on a set of rows: impute, screen, adjust, select, rank."""
    raw = cohort.markers[rows]
    medians = _column_medians(raw)
    block = _impute(raw, medians)
    lenses: dict[str, LensStats] = {}
    for lens, (screen, _) in _lens_screens(cohort, rows, settings.ties).items():
        if screen is None:
            empty = np.full(block.shape[1], np.nan)
            lenses[lens] = LensStats(*(empty.copy() for _ in range(5)))
        else:
            lenses[lens] = _lens_stats(screen.statistics(block))
    selected = {lens: np.nan_to_num(values.q_bh, nan=1.0) <= settings.alpha for lens, values in lenses.items()}
    ranks = {lens: _descending_ranks(values.chi2) for lens, values in lenses.items()}
    return ProcedureFit(lenses=lenses, medians=medians, selected=selected, ranks=ranks)


def permutation_null(
    cohort: MarkerCohort,
    full: ProcedureFit,
    settings: MarkerSettings,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Westfall–Young and permutation-FDR adjustment of each lens on the full cohort."""
    rows = np.arange(cohort.time.shape[0])
    block = _impute(cohort.markers, full.medians)
    screens = _lens_screens(cohort, rows, settings.ties)
    sources: dict[str, np.ndarray] = {}
    accumulators: dict[str, tuple[MaxTAccumulator, PermutationFdrAccumulator]] = {}
    lens2_null = _LENS2_NULL_ALIASES.get(settings.lens2_null, settings.lens2_null)
    for lens, (screen, design) in screens.items():
        if screen is None:
            continue
        if lens == "added_value" and design is not None and lens2_null == "smith":
            # Smith scheme: permute each marker's residual after regression on the clinical covariates.
            sources[lens] = residualize(block, design, cohort.strata)
        else:
            sources[lens] = block
        observed = full.lenses[lens].chi2
        accumulators[lens] = (MaxTAccumulator(observed), PermutationFdrAccumulator(observed))
    remaining = int(settings.n_permutations)
    while remaining > 0:
        raise_if_cancelled()
        batch = min(_PERMUTATION_BATCH, remaining)
        permutations = [stratified_permutation(cohort.strata, rows.size, rng) for _ in range(batch)]
        for lens, (screen, _) in screens.items():
            if screen is None:
                continue
            permuted = screen.permuted_chi2(sources[lens], permutations)
            max_t, fdr = accumulators[lens]
            max_t.update(permuted)
            fdr.update(permuted)
        remaining -= batch
    result: dict[str, dict[str, np.ndarray | int]] = {}
    for lens, (max_t, fdr) in accumulators.items():
        if settings.n_permutations > 0:
            adjusted = max_t.result()
            result[lens] = {
                "p_fwer": adjusted.p_step_down,
                "p_fwer_single_step": adjusted.p_single_step,
                "q_perm": fdr.q_values(),
                "n_permutations": adjusted.n_permutations,
            }
        else:
            empty = np.full(len(cohort.marker_names), np.nan)
            result[lens] = {"p_fwer": empty, "p_fwer_single_step": empty, "q_perm": empty, "n_permutations": 0}
    for lens in full.lenses:
        if lens not in result:
            empty = np.full(len(cohort.marker_names), np.nan)
            result[lens] = {"p_fwer": empty, "p_fwer_single_step": empty, "q_perm": empty, "n_permutations": 0}
    return result


def _event_stratified_subsample(event: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    chosen = []
    for value in (1, 0):
        pool = np.flatnonzero(event == value)
        if pool.size:
            size = max(1 if value == 1 else 0, int(round(fraction * pool.size)))
            chosen.append(rng.choice(pool, size=min(size, pool.size), replace=False))
    return np.sort(np.concatenate(chosen))


def _pooled_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray, strata: np.ndarray | None) -> float:
    """Harrell's C, comparing patients only within the same stratum."""
    codes = np.zeros(time.shape[0], dtype=np.int64) if strata is None else strata
    concordant = 0.0
    comparable = 0.0
    for code in np.unique(codes):
        rows = codes == code
        part_concordant, part_comparable = _harrell_c_index_counts(time[rows], event[rows], risk[rows])
        concordant += part_concordant
        comparable += part_comparable
    return float(concordant / comparable) if comparable > 0 else float("nan")


def _comparable_pairs(time: np.ndarray, event: np.ndarray) -> float:
    """Comparable pairs under the conventions of ``harrell_c_many``: an event with a later time or a tied censoring."""
    died = np.asarray(event).astype(bool)
    event_time = time[died]
    later = time.shape[0] - np.searchsorted(np.sort(time), event_time, side="right")
    censored = np.sort(time[~died])
    tied = np.searchsorted(censored, event_time, side="right") - np.searchsorted(censored, event_time, side="left")
    return float(np.sum(later) + np.sum(tied))


def _stratified_c_many(time: np.ndarray, event: np.ndarray, risks: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Harrell's C of each risk column, comparing patients only within the same stratum (as ``_pooled_c_index``)."""
    numerator = np.zeros(risks.shape[1])
    comparable = 0.0
    for code in np.unique(strata):
        rows = strata == code
        pairs = _comparable_pairs(time[rows], event[rows])
        if pairs > 0:
            numerator += harrell_c_many(time[rows], event[rows], risks[rows]) * pairs
            comparable += pairs
    return numerator / comparable if comparable > 0 else np.full(risks.shape[1], np.nan)


class _SignatureFit(NamedTuple):
    columns: np.ndarray
    params: np.ndarray
    medians: np.ndarray
    design_columns: np.ndarray
    # Clinical coefficients that run to infinity (a category without events); marker ones fail the fit.
    runaway_clinical: tuple[int, ...] = ()
    clinical_encoder: dict[str, Any] | None = None
    clinical_names: list[str] | None = None


def _signature_columns(fit: ProcedureFit, primary: str, limit: int) -> np.ndarray:
    lens = fit.lenses[primary]
    candidates = np.flatnonzero(fit.selected[primary])
    if candidates.size == 0:
        return candidates
    order = np.argsort(-np.nan_to_num(lens.chi2[candidates], nan=-np.inf), kind="mergesort")
    return candidates[order][:limit]


def _fit_signature(
    cohort: MarkerCohort,
    rows: np.ndarray,
    fit: ProcedureFit,
    primary: str,
    limit: int,
    ties: str = "efron",
) -> _SignatureFit | None:
    """Cox model of the clinical covariates plus the markers the procedure selected.

    None when the fit does not converge, a selected marker cannot be estimated, or a marker's
    coefficient runs to infinity (monotone likelihood). A clinical coefficient that runs to infinity
    (a category without events) is kept, as the clinical-only null model keeps it, and listed in
    ``runaway_clinical``. Clinical columns these rows cannot estimate are left out of the model.
    """
    columns = _signature_columns(fit, primary, limit)
    parts = []
    design_columns = np.zeros(0, dtype=np.int64)
    strata = None if cohort.strata is None else cohort.strata[rows]
    clinical_encoder, clinical_names = None, []
    if cohort.clinical is not None:
        try:
            training, clinical_encoder, clinical_names = _training_clinical(cohort, rows)
            _clinical_null_model(cohort.time[rows], cohort.event[rows], training, strata, ties)
        except ClinicalModelNotConvergedError:
            return None
        design_columns = _estimable_clinical_columns(training, strata)
        parts.append(training[:, design_columns])
    if columns.size:
        parts.append(_impute(cohort.markers[rows][:, columns], fit.medians[columns]))
    if not parts or sum(part.shape[1] for part in parts) == 0:
        return None
    exog = np.column_stack(parts)
    cox = fit_cox(cohort.time[rows], cohort.event[rows], exog, strata, ties)
    params = cox.beta
    estimable = ~np.isnan(params)
    if not cox.converged or not estimable[design_columns.size :].all() or not np.isfinite(params[estimable]).all():
        return None
    runaway = np.zeros(params.shape[0], dtype=bool) if cox.separated is None else np.asarray(cox.separated, dtype=bool)
    if runaway[design_columns.size :].any():
        return None
    # A clinical column aliased within the risk sets (a level whose patients were all censored before the
    # first event) has no coefficient; the model is the fit without it.
    design_columns = design_columns[estimable[: design_columns.size]]
    params, runaway = params[estimable], runaway[estimable]
    if not params.size:
        return None
    return _SignatureFit(
        columns=columns,
        params=params,
        medians=fit.medians,
        design_columns=design_columns,
        runaway_clinical=tuple(int(design_columns[index]) for index in np.flatnonzero(runaway[: design_columns.size])),
        clinical_encoder=clinical_encoder,
        clinical_names=clinical_names,
    )


def _signature_risk(cohort: MarkerCohort, rows: np.ndarray, signature: _SignatureFit) -> np.ndarray:
    parts = []
    if cohort.clinical is not None:
        design = _frozen_clinical(cohort, rows, signature.clinical_encoder, signature.clinical_names or cohort.clinical_names)
        parts.append(design[:, signature.design_columns])
    if signature.columns.size:
        parts.append(_impute(cohort.markers[rows][:, signature.columns], signature.medians[signature.columns]))
    return np.column_stack(parts) @ signature.params


def _marker_fit_usable(fit: Any) -> bool:
    """A converged fit whose last coefficient (the marker's) is finite and does not run to infinity."""
    runaway = fit.separated is not None and bool(np.asarray(fit.separated)[-1])
    return bool(fit.converged) and bool(np.isfinite(fit.beta[-1])) and not runaway


def _single_marker_beta(
    cohort: MarkerCohort,
    rows: np.ndarray,
    column: int,
    medians: np.ndarray,
    adjusted: bool,
    ties: str = "efron",
) -> float:
    marker = _impute(cohort.markers[rows][:, [column]], medians[[column]])
    exog = marker
    strata = None if cohort.strata is None else cohort.strata[rows]
    if adjusted and cohort.clinical is not None:
        try:
            training, _, _ = _training_clinical(cohort, rows)
        except ClinicalModelNotConvergedError:
            return float("nan")
        exog = np.column_stack([_estimable_clinical_design(training, strata), marker])
    fit = fit_cox(cohort.time[rows], cohort.event[rows], exog, strata, ties)
    # A coefficient that runs to infinity (for example on the few events of a left-out set) is no estimate.
    return float(fit.beta[-1]) if _marker_fit_usable(fit) else float("nan")


def resample_procedure(
    cohort: MarkerCohort,
    full: ProcedureFit,
    settings: MarkerSettings,
    rng: np.random.Generator,
    primary: str,
) -> ResamplingSummary:
    """Rerun the whole procedure on event-stratified subsamples of the cohort.

    Each replicate also fits the selected-marker signature on its subsample and scores
    it on the rows left out, next to the clinical-only model on the same rows (the
    signature itself when no marker was selected), and records the strongest marker's
    log hazard ratio in and out of the subsample (the winner's curse of picking the
    strongest marker). The subsample and left-out sizes of each scored replicate give
    the intervals of the left-out means (see ``_optimism_summary``).
    """
    n = cohort.time.shape[0]
    n_markers = len(cohort.marker_names)
    lenses = list(full.lenses)
    counts = {lens: np.zeros(n_markers, dtype=float) for lens in lenses}
    agree = {lens: np.zeros(n_markers, dtype=float) for lens in lenses}
    finite = {lens: np.zeros(n_markers, dtype=float) for lens in lenses}
    rank_rows: dict[str, list[np.ndarray]] = {lens: [] for lens in lenses}
    full_sign = {lens: np.sign(full.lenses[lens].z) for lens in lenses}
    c_in: list[float] = []
    c_out: list[float] = []
    clinical_c_out: list[float] = []
    # Subsample and left-out sizes of each replicate in c_out, for the intervals of the left-out means.
    sizes: list[tuple[int, int]] = []
    beta_in: list[float] = []
    beta_out: list[float] = []
    n_failed = 0
    n_withheld = 0
    for _ in range(int(settings.n_resamples)):
        raise_if_cancelled()
        rows = _event_stratified_subsample(cohort.event, settings.resample_fraction, rng)
        left_out = np.setdiff1d(np.arange(n), rows)
        try:
            fit = run_procedure(cohort, rows, settings)
        except (ClinicalModelNotConvergedError, np.linalg.LinAlgError):
            # A subsample the procedure cannot fit (its clinical model does not converge, or a matrix
            # routine fails on its data) is counted as failed. Any other error, including a ValueError
            # from a check that a bug would trip, stops the analysis instead of being counted.
            n_failed += 1
            continue
        if cohort.clinical is not None:
            training, encoder, names = _training_clinical(cohort, rows)
            training_cohort = cohort._replace(
                time=cohort.time[rows], event=cohort.event[rows], markers=cohort.markers[rows],
                clinical=training, clinical_names=names, clinical_encoder=encoder,
                clinical_frame=cohort.clinical_frame.iloc[rows] if cohort.clinical_frame is not None else None,
                strata=None if cohort.strata is None else cohort.strata[rows],
                source_rows=[cohort.source_rows[int(i)] for i in rows],
                row_mask_hash=hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
                dropped_clinical=(),
            )
            diagnostic = diagnose_markers(training_cohort, _impute(cohort.markers[rows], fit.medians), ties=settings.ties, candidate=settings.diagnostic_candidate, bootstrap_seed=settings.random_seed)
            if not diagnostic["allowed"]:
                n_withheld += 1
                fit = fit._replace(selected={lens: (np.zeros_like(value) if lens == "added_value" else value)
                                             for lens, value in fit.selected.items()})
        for lens in lenses:
            values = fit.lenses[lens]
            counts[lens] += fit.selected[lens]
            valid = np.isfinite(values.z)
            finite[lens] += valid
            agree[lens] += valid & (np.sign(values.z) == full_sign[lens])
            rank_rows[lens].append(fit.ranks[lens])
        signature = _fit_signature(cohort, rows, fit, primary, settings.max_signature_markers, settings.ties)
        if signature is not None and left_out.size and cohort.event[left_out].any():
            strata_in = None if cohort.strata is None else cohort.strata[rows]
            strata_out = None if cohort.strata is None else cohort.strata[left_out]
            c_in.append(_pooled_c_index(cohort.time[rows], cohort.event[rows], _signature_risk(cohort, rows, signature), strata_in))
            c_out.append(_pooled_c_index(cohort.time[left_out], cohort.event[left_out], _signature_risk(cohort, left_out, signature), strata_out))
            sizes.append((int(rows.size), int(left_out.size)))
            if cohort.clinical is not None:
                # Paired with every signature C: without a selected marker the signature is the clinical-only model.
                if signature.columns.size:
                    clinical_only = _SignatureFit(
                        columns=np.zeros(0, dtype=np.int64),
                        params=_clinical_params(cohort, rows, signature.design_columns, settings.ties),
                        medians=signature.medians,
                        design_columns=signature.design_columns,
                        clinical_encoder=signature.clinical_encoder,
                        clinical_names=signature.clinical_names,
                    )
                    clinical_c_out.append(
                        _pooled_c_index(cohort.time[left_out], cohort.event[left_out], _signature_risk(cohort, left_out, clinical_only), strata_out)
                    )
                else:
                    clinical_c_out.append(c_out[-1])
        top = _top_marker(fit, primary)
        if top is not None and left_out.size and cohort.event[left_out].sum() >= 2:
            adjusted = primary == "added_value"
            inside = _single_marker_beta(cohort, rows, top, fit.medians, adjusted, settings.ties)
            outside = _single_marker_beta(cohort, left_out, top, fit.medians, adjusted, settings.ties)
            if np.isfinite(inside) and np.isfinite(outside):
                beta_in.append(inside)
                beta_out.append(outside)

    n_valid = int(settings.n_resamples) - n_failed
    summary_frequency: dict[str, np.ndarray] = {}
    summary_direction: dict[str, np.ndarray] = {}
    median_rank: dict[str, np.ndarray] = {}
    rank_low: dict[str, np.ndarray] = {}
    rank_high: dict[str, np.ndarray] = {}
    for lens in lenses:
        with np.errstate(invalid="ignore", divide="ignore"):
            summary_frequency[lens] = counts[lens] / n_valid if n_valid else np.full(n_markers, np.nan)
            summary_direction[lens] = np.where(finite[lens] > 0, agree[lens] / np.maximum(finite[lens], 1), np.nan)
        if rank_rows[lens]:
            matrix = np.vstack(rank_rows[lens]).astype(float)
            median_rank[lens] = np.median(matrix, axis=0)
            rank_low[lens] = np.quantile(matrix, 0.025, axis=0)
            rank_high[lens] = np.quantile(matrix, 0.975, axis=0)
        else:
            median_rank[lens] = rank_low[lens] = rank_high[lens] = np.full(n_markers, np.nan)
    optimism = _optimism_summary(c_in, c_out, clinical_c_out, beta_in, beta_out, sizes)
    return ResamplingSummary(
        n_valid=n_valid,
        n_failed=n_failed,
        selection_frequency=summary_frequency,
        direction_consistency=summary_direction,
        median_rank=median_rank,
        rank_low=rank_low,
        rank_high=rank_high,
        optimism=optimism,
        n_withheld=n_withheld,
    )


def _clinical_params(cohort: MarkerCohort, rows: np.ndarray, design_columns: np.ndarray, ties: str = "efron") -> np.ndarray:
    training, _, _ = _training_clinical(cohort, rows)
    design = training[:, design_columns]
    strata = None if cohort.strata is None else cohort.strata[rows]
    beta = fit_cox(cohort.time[rows], cohort.event[rows], design, strata, ties).beta
    # The signature's clinical columns are estimable in these rows; a column that is not adds nothing.
    return np.where(np.isnan(beta), 0.0, beta)


def _top_marker(fit: ProcedureFit, primary: str) -> int | None:
    chi2 = fit.lenses[primary].chi2
    if not np.isfinite(chi2).any():
        return None
    return int(np.nanargmax(chi2))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _corrected_resampled_t_interval(values: Sequence[float], left_out_shares: Sequence[float]) -> list[float] | None:
    """95% interval of a mean over J overlapping subsamples: the corrected resampled t of Nadeau and Bengio.

    Subsamples share most of their patients, so their J values are correlated and s²/J understates the
    variance of their mean. Nadeau and Bengio (Machine Learning 2003;52:239–281) inflate it to (1/J + ρ) s²,
    with s² the sample variance of the values (ddof 1) and ρ the ratio of left-out to subsample size, here the
    mean of ``left_out_shares`` (n_left_out / n_subsample of each value). The interval is
    mean ± t(0.975, J − 1) · sqrt((1/J + ρ) s²); None with fewer than two values, without a share for every
    value, or when s² is not finite.
    """
    if len(values) < 2 or len(left_out_shares) != len(values):
        return None
    variance = float(np.var(values, ddof=1))
    if not math.isfinite(variance):
        return None
    mean = float(np.mean(values))
    rho = float(np.mean(left_out_shares))
    half_width = float(stats.t.ppf(0.975, len(values) - 1)) * math.sqrt((1.0 / len(values) + rho) * variance)
    return [mean - half_width, mean + half_width]


def _optimism_summary(
    c_in: Sequence[float],
    c_out: Sequence[float],
    clinical_c_out: Sequence[float],
    beta_in: Sequence[float],
    beta_out: Sequence[float],
    sizes: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Means over the replicates; ``clinical_c_out`` is empty or holds one value per ``c_out`` (paired).

    ``sizes`` holds the subsample and left-out sizes of each replicate in ``c_out``. With them, the left-out
    C-index and the paired gain over the clinical covariates get 95% corrected resampled t intervals
    (``_corrected_resampled_t_interval``); without them the intervals are None. The gain's SD is over the
    paired replicates.
    """
    pairs = [(inside, outside) for inside, outside in zip(c_in, c_out) if np.isfinite(inside) and np.isfinite(outside)]
    # The clinical-only C over the same replicates as the signature's left-out C.
    paired = [
        (outside, clinical)
        for inside, outside, clinical in zip(c_in, c_out, clinical_c_out)
        if np.isfinite(inside) and np.isfinite(outside) and np.isfinite(clinical)
    ]
    # n_left_out / n_subsample of the replicates in pairs and in paired.
    shares = [left_out / subsample for subsample, left_out in sizes] if len(sizes) == len(c_out) else []
    pair_shares = [share for inside, outside, share in zip(c_in, c_out, shares) if np.isfinite(inside) and np.isfinite(outside)]
    paired_shares = [
        share
        for inside, outside, clinical, share in zip(c_in, c_out, clinical_c_out, shares)
        if np.isfinite(inside) and np.isfinite(outside) and np.isfinite(clinical)
    ]
    gains = [outside - clinical for outside, clinical in paired]
    shrinkage = None
    if beta_in:
        magnitude = float(np.mean(np.abs(beta_in)))
        if magnitude > 0:
            shrinkage = float(np.mean(np.sign(beta_in) * np.asarray(beta_out)) / magnitude)
    return {
        "signature_c_in_subsample": _mean_or_none([inside for inside, _ in pairs]),
        "signature_c_left_out": _mean_or_none([outside for _, outside in pairs]),
        "signature_c_left_out_ci": _corrected_resampled_t_interval([outside for _, outside in pairs], pair_shares),
        "signature_optimism": _mean_or_none([inside - outside for inside, outside in pairs]),
        "clinical_c_left_out": _mean_or_none([clinical for _, clinical in paired]),
        "signature_gain_left_out": _mean_or_none(gains),
        "signature_gain_left_out_sd": float(np.std(gains, ddof=1)) if len(gains) > 1 else None,
        "signature_gain_left_out_ci": _corrected_resampled_t_interval(gains, paired_shares),
        "n_signature_replicates": len(pairs),
        "n_clinical_replicates": len(paired),
        "top_marker_log_hr_in_subsample": _mean_or_none([abs(value) for value in beta_in]),
        "top_marker_log_hr_left_out": _mean_or_none(
            [float(np.sign(inside) * outside) for inside, outside in zip(beta_in, beta_out)]
        ),
        "top_marker_shrinkage": shrinkage,
        "n_top_marker_replicates": len(beta_in),
    }


def nonlinear_lens(cohort: MarkerCohort, settings: MarkerSettings, rng: np.random.Generator) -> dict[str, Any] | None:
    """Descriptive tree-model lens: out-of-sample permutation importance of each marker.

    On each event-stratified subsample a gradient-boosted survival model (or a random
    survival forest) is fitted on the clinical design plus the ``nonlinear_top_markers``
    markers with the largest in-subsample marginal score statistic; each marker's
    importance is the drop in Harrell's C on the left-out rows when it is shuffled.
    It can reveal non-linear or interaction signals the Cox score lenses miss, but it
    does not set tiers.
    """
    if settings.nonlinear_lens == "off":
        return None
    from survival_toolkit.ml_models import (
        SKSURV_AVAILABLE,
        _TREE_N_JOBS,
        _effective_tree_min_samples_leaf,
        _grouped_permutation_importance,
    )

    if not SKSURV_AVAILABLE:
        return {"available": False, "note": "scikit-survival is not installed, so the non-linear lens was skipped."}
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

    n = cohort.time.shape[0]
    n_markers = len(cohort.marker_names)
    total = np.zeros(n_markers, dtype=float)
    positive = np.zeros(n_markers, dtype=float)
    evaluated = np.zeros(n_markers, dtype=float)
    for replicate in range(int(settings.nonlinear_replicates)):
        raise_if_cancelled()
        rows = _event_stratified_subsample(cohort.event, settings.resample_fraction, rng)
        left_out = np.setdiff1d(np.arange(n), rows)
        if int(cohort.event[left_out].sum()) < 2:
            continue
        medians = _column_medians(cohort.markers[rows])
        inside = _impute(cohort.markers[rows], medians)
        outside = _impute(cohort.markers[left_out], medians)
        keep = np.arange(n_markers)
        if n_markers > settings.nonlinear_top_markers:
            strata = None if cohort.strata is None else cohort.strata[rows]
            null = fit_cox_null(cohort.time[rows], cohort.event[rows], None, strata, settings.ties)
            chi2 = CoxScoreScreen(cohort.time[rows], cohort.event[rows], null=null, strata=strata, ties=settings.ties).statistics(inside).chi2
            keep = np.argsort(-np.nan_to_num(chi2, nan=-np.inf), kind="mergesort")[: settings.nonlinear_top_markers]
        names = [cohort.marker_names[int(index)] for index in keep]
        parts_in = [inside[:, keep]]
        parts_out = [outside[:, keep]]
        if cohort.clinical is not None:
            try:
                training, encoder, training_names = _training_clinical(cohort, rows)
            except ClinicalModelNotConvergedError:
                continue
            varying = np.flatnonzero(np.ptp(training, axis=0) > 0)
            outside_clinical = _frozen_clinical(cohort, left_out, encoder, training_names)
            parts_in.insert(0, training[:, varying])
            parts_out.insert(0, outside_clinical[:, varying])
            names = [f"clinical::{training_names[int(index)]}" for index in varying] + names
        y_in = np.empty(rows.size, dtype=[("event", bool), ("time", float)])
        y_in["event"] = cohort.event[rows].astype(bool)
        y_in["time"] = cohort.time[rows]
        y_out = np.empty(left_out.size, dtype=[("event", bool), ("time", float)])
        y_out["event"] = cohort.event[left_out].astype(bool)
        y_out["time"] = cohort.time[left_out]
        seed = int(settings.random_seed) + 7919 * (replicate + 1)
        leaf = _effective_tree_min_samples_leaf(10, int(rows.size))
        if settings.nonlinear_lens == "rsf":
            model = RandomSurvivalForest(n_estimators=100, min_samples_leaf=leaf, random_state=seed, n_jobs=_TREE_N_JOBS)
        else:
            model = GradientBoostingSurvivalAnalysis(
                n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=leaf, random_state=seed
            )
        model.fit(np.column_stack(parts_in), y_in)
        records = _grouped_permutation_importance(
            model,
            pd.DataFrame(np.column_stack(parts_out), columns=names),
            y_out,
            None,
            random_state=seed,
        )
        positions = {name: index for index, name in enumerate(cohort.marker_names)}
        for record in records:
            index = positions.get(str(record["feature"]))
            if index is None or record["importance"] is None:
                continue
            total[index] += float(record["importance"])
            positive[index] += float(record["importance"]) > 0.0
            evaluated[index] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_importance = np.where(evaluated > 0, total / np.maximum(evaluated, 1.0), np.nan)
        positive_fraction = np.where(evaluated > 0, positive / np.maximum(evaluated, 1.0), np.nan)
    # One-sided sign test of "importance > 0 more often than chance", Holm-adjusted over
    # the evaluated markers. Replicates overlap, so read it as a screen, not a test.
    sign_p = np.where(evaluated > 0, stats.binom.sf(positive - 1, evaluated, 0.5), np.nan)
    return {
        "available": True,
        "model": "gradient-boosted survival" if settings.nonlinear_lens == "gbs" else "random survival forest",
        "replicates": int(settings.nonlinear_replicates),
        "mean_importance": mean_importance,
        "positive_fraction": positive_fraction,
        "n_evaluated": evaluated,
        "sign_test_p_holm": np.asarray(_holm(sign_p.tolist()), dtype=float),
    }


def _exact_fits(cohort: MarkerCohort, full: ProcedureFit, columns: np.ndarray, ties: str = "efron") -> dict[int, dict[str, Any]]:
    """Cox fits for shortlisted markers: HRs with Wald CIs, nested LR test, delta C.

    A marker whose coefficient runs to infinity (monotone likelihood) gets no estimate.
    """
    strata = cohort.strata
    fits: dict[int, dict[str, Any]] = {}
    clinical_llf = None
    clinical_c = None
    clinical_design = None if cohort.clinical is None else _estimable_clinical_design(cohort.clinical, strata)
    if clinical_design is not None and not clinical_design.shape[1]:
        clinical_design = None
    if clinical_design is not None:
        clinical_fit = fit_cox(cohort.time, cohort.event, clinical_design, strata, ties)
        if clinical_fit.converged:
            clinical_llf = clinical_fit.loglik
            clinical_c = _pooled_c_index(cohort.time, cohort.event, _linear_predictor(clinical_design, clinical_fit.beta), strata)
    z_value = float(stats.norm.ppf(0.975))
    for column in columns:
        raise_if_cancelled()
        marker = _impute(cohort.markers[:, [column]], full.medians[[column]])
        entry: dict[str, Any] = {}
        for label, exog in (("marginal", marker), ("adjusted", None if clinical_design is None else np.column_stack([clinical_design, marker]))):
            if exog is None:
                continue
            fit = fit_cox(cohort.time, cohort.event, exog, strata, ties)
            beta = float(fit.beta[-1])
            se = float(np.sqrt(fit.covariance[-1, -1])) if fit.covariance[-1, -1] > 0 else float("nan")
            if not _marker_fit_usable(fit) or not np.isfinite(se):
                entry[label] = None
                continue
            with np.errstate(over="ignore"):
                entry[label] = {
                    "hazard_ratio": _finite_or_none(np.exp(beta)),
                    "ci_lower": _finite_or_none(np.exp(beta - z_value * se)),
                    "ci_upper": _finite_or_none(np.exp(beta + z_value * se)),
                    "wald_p": float(2.0 * stats.norm.sf(abs(beta / se))),
                }
            if label == "adjusted" and clinical_llf is not None:
                statistic = max(2.0 * (fit.loglik - clinical_llf), 0.0)
                entry[label]["lr_statistic"] = statistic
                entry[label]["lr_p"] = float(stats.chi2.sf(statistic, df=1))
                if clinical_c is not None:
                    full_c = _pooled_c_index(cohort.time, cohort.event, _linear_predictor(exog, fit.beta), strata)
                    entry[label]["delta_c_apparent"] = float(full_c - clinical_c)
        fits[int(column)] = entry
    return fits


def assign_tiers(
    primary: str,
    adjusted: dict[str, dict[str, np.ndarray | int]],
    resampling: ResamplingSummary,
    lenses: dict[str, LensStats],
    settings: MarkerSettings,
) -> tuple[list[str], list[str]]:
    """Tier and evidence pattern of each marker from the pre-declared rules.

    * robust: step-down Westfall–Young p <= alpha on the primary lens, selected in at
      least ``robust_frequency`` of subsamples, and the same direction in at least
      ``robust_direction`` of them. Robust markers are a subset of the Westfall–Young
      rejections; family-wise error control requires a valid permutation null,
      including exchangeability and the relevant subset-pivotality assumptions.
    * suggestive: primary-lens evidence (Westfall–Young p <= alpha or permutation FDR
      q <= ``fdr_level``) that is not stable enough to be robust; the selection
      frequency and direction consistency columns show why.
    * marginal only: with clinical covariates, no added-value evidence but a marginal
      Westfall–Young p <= alpha (for example a marker that tracks a prognostic
      clinical covariate).

    Without any evaluated subsample (``n_resamples`` = 0, or every subsample failed) stability
    is not assessed, so no marker can be robust; the strongest tier is then suggestive.
    """
    n_markers = lenses[primary].chi2.shape[0]
    p_fwer = np.nan_to_num(np.asarray(adjusted[primary]["p_fwer"], dtype=float), nan=1.0)
    q_perm = np.nan_to_num(np.asarray(adjusted[primary]["q_perm"], dtype=float), nan=1.0)
    frequency = np.nan_to_num(resampling.selection_frequency[primary], nan=0.0)
    direction = np.nan_to_num(resampling.direction_consistency[primary], nan=0.0)
    stability_assessed = resampling.n_valid > 0
    tiers: list[str] = []
    patterns: list[str] = []
    for index in range(n_markers):
        stable = stability_assessed and frequency[index] >= settings.robust_frequency and direction[index] >= settings.robust_direction
        evidence = p_fwer[index] <= settings.alpha or q_perm[index] <= settings.fdr_level
        if p_fwer[index] <= settings.alpha and stable:
            tier = "robust"
        elif evidence:
            tier = "suggestive"
        elif primary == "added_value" and float(np.nan_to_num(adjusted["marginal"]["p_fwer"][index], nan=1.0)) <= settings.alpha:
            tier = "marginal only"
        else:
            tier = "not supported"
        tiers.append(tier)
        pattern = []
        for lens, letter in (("marginal", "M"), ("added_value", "A")):
            if lens not in lenses:
                continue
            significant = float(np.nan_to_num(adjusted[lens]["p_fwer"][index], nan=1.0)) <= settings.alpha
            sign = lenses[lens].z[index]
            pattern.append(letter + (("+" if sign > 0 else "-") if significant and np.isfinite(sign) else "·"))
        patterns.append(" ".join(pattern))
    return tiers, patterns


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


# Version 2 hashes a canonical form (every number as a float) and stores the baseline hazard centred on
# the development mean linear predictor; version-1 recipes are still validated.
RECIPE_VERSION = 3
RECIPE_VERSIONS = (1, 2, 3)
MARKER_SCALINGS = ("as_measured", "within_cohort")
# Below this share of the locked model's marker weight in an external dataset, validation stops.
MIN_MARKER_WEIGHT_AVAILABLE = 0.5


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical_numbers(value: Any) -> Any:
    """The recipe with every number (not booleans) as a float, so 18 and 18.0 hash alike.

    A browser writes whole numbers without a decimal point, so a recipe sent back from the web page
    holds 18 where Python wrote 18.0; -0.0 becomes 0.0 as well. Non-finite numbers are refused.
    """
    if isinstance(value, dict):
        return {str(key): _canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_numbers(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_canonical_numbers(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("The recipe holds a number that is not finite, so it cannot be locked or checked.")
        return number + 0.0
    return value


def recipe_hash(recipe: dict[str, Any]) -> str:
    """SHA-256 of the recipe's canonical JSON (every number as a float), excluding the stored hash itself."""
    payload = {key: value for key, value in recipe.items() if key != "recipe_hash"}
    text = json.dumps(_canonical_numbers(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _legacy_recipe_hash(recipe: dict[str, Any]) -> str:
    """The version-1 hash: Python's JSON text of the recipe as written, so 18.0 and 18 differ."""
    payload = {key: value for key, value in recipe.items() if key != "recipe_hash"}
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _centred_baseline(
    time: np.ndarray,
    event: np.ndarray,
    linear_predictor: np.ndarray,
    ties: str = "efron",
) -> dict[str, Any] | None:
    """Cumulative baseline hazard at the development mean linear predictor, stored on the log scale.

    Stored at a linear predictor of 0 instead, the baseline underflows (or rounds to 1) whenever the
    markers sit far from 0, as log2 expression values do: a mean linear predictor of -9 multiplies the
    cumulative hazard by e^9. Centred, it stays near the cohort's own cumulative hazard.

    The increments follow the model's tie method, as R's ``survfit.coxph`` does: d deaths tied at a time
    add d / S0 with Breslow ties and sum_k 1 / (S0 - (k / d) S0_dead), k = 0..d-1, with Efron ties, where
    S0 sums the risk of everyone at risk and S0_dead that of the d deaths. Without tied deaths they agree.
    """
    centre = float(np.mean(linear_predictor))
    risk = np.exp(np.clip(linear_predictor - centre, -_LP_CLIP, _LP_CLIP))
    died = np.asarray(event) == 1
    event_times = np.unique(time[died])
    if event_times.size == 0:
        return None
    order = np.argsort(time, kind="mergesort")
    at_risk = np.cumsum(risk[order][::-1])[::-1][np.searchsorted(time[order], event_times, side="left")]
    group = np.searchsorted(event_times, time[died])
    deaths = np.bincount(group, minlength=event_times.size).astype(float)
    if ties == "efron":
        dead_risk = np.bincount(group, weights=risk[died], minlength=event_times.size)
        by_time = np.sort(group, kind="mergesort")
        # Rank k = 0..d-1 of each death among those tied with it.
        rank = np.arange(by_time.size) - np.searchsorted(by_time, by_time, side="left")
        shares = 1.0 / (at_risk[by_time] - rank / deaths[by_time] * dead_risk[by_time])
        increments = np.bincount(by_time, weights=shares, minlength=event_times.size)
    else:
        increments = deaths / at_risk
    return {
        "times": event_times,
        "log_cumulative_hazard": np.log(np.cumsum(increments)),
        "lp_center": centre,
    }


def _baseline_survival_at(baseline: dict[str, Any], linear_predictor: np.ndarray, horizon: float) -> np.ndarray | None:
    """Each patient's predicted survival at ``horizon``; None when a version-1 baseline lost its precision there."""
    times = np.asarray(baseline["times"], dtype=float)
    position = int(np.searchsorted(times, horizon, side="right")) - 1
    if "log_cumulative_hazard" in baseline:
        if position < 0:
            return np.ones(linear_predictor.shape[0])
        centred = np.clip(linear_predictor - float(baseline["lp_center"]), -_LP_CLIP, _LP_CLIP)
        return np.exp(-np.exp(float(baseline["log_cumulative_hazard"][position]) + centred))
    # Version 1 stored S0 at a linear predictor of 0: exactly 0 (underflow) or 1 (rounding) past the first
    # event time means the cumulative hazard is lost.
    survival = 1.0 if position < 0 else float(np.asarray(baseline["survival"], dtype=float)[position])
    if position >= 0 and not 0.0 < survival < 1.0 - 1e-12:
        return None
    return np.power(survival, np.exp(np.clip(linear_predictor, -_LP_CLIP, _LP_CLIP)))


def _marker_scale(values: np.ndarray) -> dict[str, float]:
    observed = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(observed)) if observed.size else 0.0,
        "sd": float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0,
    }


def freeze_recipe(
    cohort: MarkerCohort,
    full: ProcedureFit,
    signature: _SignatureFit,
    primary: str,
    *,
    time_column: str,
    event_column: str,
    event_positive_value: Any,
    categorical_clinical: Sequence[str],
    ties: str = "efron",
    inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to apply the selected signature, unchanged, to another cohort."""
    from survival_toolkit import __version__

    all_rows = np.arange(cohort.time.shape[0])
    clinical_terms = [(signature.clinical_names or cohort.clinical_names)[int(index)] for index in signature.design_columns] if cohort.clinical is not None else []
    marker_terms = [cohort.marker_names[int(column)] for column in signature.columns]
    linear_predictor = _signature_risk(cohort, all_rows, signature)
    baseline = None
    if cohort.strata is None:
        baseline = _centred_baseline(cohort.time, cohort.event, linear_predictor, ties)
    clinical_only = None
    # Also locked when no marker was selected: the model is then the clinical model itself, and validation
    # reports its gain over the clinical covariates as exactly zero instead of leaving it out.
    if clinical_terms:
        clinical_only = {
            "terms": clinical_terms,
            "coefficients": _clinical_params(cohort, all_rows, signature.design_columns, ties),
        }
    primary_beta = full.lenses[primary].beta_one_step
    recipe: dict[str, Any] = {
        "recipe_version": RECIPE_VERSION,
        "inference": inference or diagnose_markers(cohort, _impute(cohort.markers, full.medians), ties=ties),
        "created_with": f"SurvStudio {__version__}",
        "outcome": {
            "time_column": time_column,
            "event_column": event_column,
            "event_positive_value": event_positive_value,
        },
        "clinical": {
            "columns": list(cohort.clinical_columns),
            "categorical": [str(column) for column in categorical_clinical],
            "encoder": signature.clinical_encoder or cohort.clinical_encoder,
            "basis": cohort.clinical_basis,
        },
        "strata_columns": list(cohort.strata_columns),
        "markers": marker_terms,
        "marker_medians": {cohort.marker_names[int(column)]: float(signature.medians[int(column)]) for column in signature.columns},
        # Development mean and SD of each locked marker: the scale to map another platform onto, and the
        # weight (|coefficient| x SD) a marker carries in the model.
        "marker_scale": {cohort.marker_names[int(column)]: _marker_scale(cohort.markers[:, int(column)]) for column in signature.columns},
        "marker_development_log_hr": {
            cohort.marker_names[int(column)]: float(primary_beta[int(column)]) for column in signature.columns
        },
        "primary_lens": primary,
        "model": {
            "terms": [*clinical_terms, *marker_terms],
            "coefficients": signature.params,
            "ties": ties,
            "baseline": baseline,
            "default_horizon": float(np.median(cohort.time[cohort.event == 1])),
        },
        "clinical_only_model": clinical_only,
        "development": {
            "n": int(cohort.time.shape[0]),
            "events": int(cohort.event.sum()),
            "row_mask_hash": cohort.row_mask_hash,
        },
    }
    recipe = _json_ready(recipe)
    recipe["recipe_hash"] = recipe_hash(recipe)
    return recipe


def _unique_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The dataset with rows addressed by position when its index repeats a label.

    Markers are read by row label, and a repeated label (two samples of one patient, say) would
    return both rows for each; row numbers name the patients instead.
    """
    return df if df.index.is_unique else df.reset_index(drop=True)


def _patient_labels(df: pd.DataFrame, source_rows: Sequence[Any], id_column: str | None) -> list[str]:
    """How the duplicate screen names patients: by ``id_column`` when given, else by dataset row number.

    ``df`` has a unique index (see ``_unique_rows``).
    """
    if id_column and id_column in df.columns:
        return [str(value) for value in df.loc[list(source_rows), id_column]]
    positions = pd.Index(df.index).get_indexer(list(source_rows))
    return [f"row {position + 1}" if position >= 0 else str(label) for position, label in zip(positions, source_rows)]


@user_input_boundary
def evaluate_markers(
    df: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    marker_columns: Sequence[str],
    clinical_columns: Sequence[str] = (),
    categorical_clinical: Sequence[str] = (),
    strata_columns: Sequence[str] = (),
    event_positive_value: Any = None,
    settings: MarkerSettings | None = None,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Screen candidate markers and check the whole screening procedure (see module docstring).

    ``id_column`` only names patients in the duplicate screen; without it they are named by row number.
    """
    settings = _validated_settings(settings or MarkerSettings())
    df = _unique_rows(df)
    cohort = prepare_marker_cohort(
        df,
        time_column=time_column,
        event_column=event_column,
        marker_columns=marker_columns,
        clinical_columns=clinical_columns,
        categorical_clinical=categorical_clinical,
        strata_columns=strata_columns,
        event_positive_value=event_positive_value,
        max_missing_fraction=settings.max_missing_fraction,
        max_mode_fraction=settings.max_mode_fraction,
        clinical_basis=settings.clinical_basis,
    )
    duplicates = possible_duplicates(cohort.markers, _patient_labels(df, cohort.source_rows, id_column))
    # One independent random stream per stage, so changing the number of permutations does not change which
    # subsamples are drawn (nor the number of subsamples the permutations).
    permutation_seed, resample_seed, nonlinear_seed = np.random.SeedSequence(int(settings.random_seed)).spawn(3)
    all_rows = np.arange(cohort.time.shape[0])
    full = run_procedure(cohort, all_rows, settings)
    diagnostics = diagnose_markers(cohort, _impute(cohort.markers, full.medians), ties=settings.ties, candidate=settings.diagnostic_candidate, bootstrap_seed=settings.random_seed)
    primary = "added_value" if cohort.clinical_columns else "marginal"
    adjusted = permutation_null(cohort, full, settings, np.random.default_rng(permutation_seed))
    resample_settings = settings if np.isfinite(full.lenses[primary].chi2).any() else settings._replace(n_resamples=0)
    resampling = resample_procedure(cohort, full, resample_settings, np.random.default_rng(resample_seed), primary)
    tiers, patterns = assign_tiers(primary, adjusted, resampling, full.lenses, settings)
    nonlinear = nonlinear_lens(cohort, settings, np.random.default_rng(nonlinear_seed)) if not cohort.clinical_error else None
    nonlinear_ready = bool(nonlinear and nonlinear.get("available"))
    if nonlinear_ready:
        for index in range(len(patterns)):
            consistent = (
                float(np.nan_to_num(nonlinear["positive_fraction"][index], nan=0.0)) >= settings.nonlinear_frequency
                and float(np.nan_to_num(nonlinear["sign_test_p_holm"][index], nan=1.0)) <= settings.alpha
                and float(np.nan_to_num(nonlinear["mean_importance"][index], nan=0.0)) > 0.0
            )
            patterns[index] += " N+" if consistent else " N·"

    primary_chi2 = np.nan_to_num(full.lenses[primary].chi2, nan=-np.inf)
    by_strength = np.argsort(-primary_chi2, kind="mergesort")
    shortlist = [int(index) for index in by_strength[: settings.shortlist_size]]
    shortlist += [index for index, tier in enumerate(tiers) if tier != "not supported" and index not in shortlist]
    exact = _exact_fits(cohort, full, np.asarray(shortlist, dtype=np.int64), settings.ties)
    signature = _fit_signature(cohort, all_rows, full, primary, settings.max_signature_markers, settings.ties)
    apparent_c = None
    recipe = None
    signature_notes: list[str] = []
    if signature is not None:
        apparent_c = _pooled_c_index(cohort.time, cohort.event, _signature_risk(cohort, all_rows, signature), cohort.strata)
        recipe = freeze_recipe(
            cohort,
            full,
            signature,
            primary,
            time_column=time_column,
            event_column=event_column,
            event_positive_value=event_positive_value,
            categorical_clinical=categorical_clinical,
            inference=diagnostics,
            ties=settings.ties,
        )
        recipe["inference"] = _json_ready(diagnostics)
        recipe["recipe_hash"] = recipe_hash(recipe)
        if signature.columns.size == 0:
            signature_notes.append("No marker was selected, so the model holds the clinical covariates alone.")
        if signature.runaway_clinical:
            names = ", ".join(cohort.clinical_names[index] for index in signature.runaway_clinical)
            signature_notes.append(
                f"The coefficient of {names} runs to infinity (a group without events, for example), so the model "
                "treats that group as having no risk."
            )
    elif _signature_columns(full, primary, settings.max_signature_markers).size:
        signature_notes.append(
            "The selected-marker model could not be fitted: it did not converge, or a marker's coefficient runs to "
            "infinity (for example a mutation whose carriers never had the event). No model was locked."
        )
    elif cohort.clinical is not None:
        signature_notes.append(
            "No marker was selected, and the model of the clinical covariates alone could not be fitted: it did not converge. "
            "No model was locked."
        )
    cohort_notes = [f"{item['column']} was left out of the clinical model: {item['reason']}." for item in cohort.dropped_clinical]
    cohort_notes += list(cohort.notes)
    stability_assessed = resampling.n_valid > 0

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(cohort.marker_names):
        row: dict[str, Any] = {
            "marker": name,
            "tier": tiers[index],
            "pattern": patterns[index],
            "direction": (
                "higher values, higher hazard" if full.lenses[primary].z[index] > 0 else "higher values, lower hazard"
            )
            if np.isfinite(full.lenses[primary].z[index])
            else None,
        }
        for lens, values in full.lenses.items():
            row[lens] = {
                "chi2": _finite_or_none(values.chi2[index]),
                "p_value": _finite_or_none(values.p_value[index]),
                "q_bh": _finite_or_none(values.q_bh[index]),
                "p_fwer": _finite_or_none(adjusted[lens]["p_fwer"][index]),
                "q_perm": _finite_or_none(adjusted[lens]["q_perm"][index]),
                "log_hr_one_step": _finite_or_none(values.beta_one_step[index]),
                "selection_frequency": _finite_or_none(resampling.selection_frequency[lens][index]),
                "direction_consistency": _finite_or_none(resampling.direction_consistency[lens][index]),
                "median_rank": _finite_or_none(resampling.median_rank[lens][index]),
                "rank_interval": [
                    _finite_or_none(resampling.rank_low[lens][index]),
                    _finite_or_none(resampling.rank_high[lens][index]),
                ],
            }
        if nonlinear_ready:
            row["nonlinear"] = {
                "mean_importance": _finite_or_none(nonlinear["mean_importance"][index]),
                "positive_fraction": _finite_or_none(nonlinear["positive_fraction"][index]),
                "sign_test_p_holm": _finite_or_none(nonlinear["sign_test_p_holm"][index]),
                "n_evaluated": int(nonlinear["n_evaluated"][index]),
            }
        row["exact"] = exact.get(index)
        rows.append(row)
    if not diagnostics["allowed"]:
        for row in rows:
            target = row[primary]
            row["exploratory"] = {primary: dict(target), "tier": row["tier"], "pattern": row["pattern"],
                                  "exact": row.get("exact")}
            row["tier"] = "inference withheld"
            row["pattern"] = "Inference withheld"
            row["inference_status"] = "withheld"
            for key in ("p_value", "q_bh", "p_fwer", "q_perm"):
                target[key] = None
            if row.get("exact") and row["exact"].get("adjusted"):
                row["exact"] = {**row["exact"], "adjusted": {**row["exact"]["adjusted"], "wald_p": None, "lr_p": None}}
    else:
        for row in rows:
            row["inference_status"] = "assumption_dependent"
    tier_rank = {tier: position for position, tier in enumerate(TIER_ORDER)}
    rows.sort(
        key=lambda item: (
            tier_rank[item["tier"]],
            -item[primary]["chi2"] if item[primary]["chi2"] is not None else np.inf,
        )
    )

    result = {
        "method_version": METHOD_VERSION,
        "inference": diagnostics,
        "clinical_basis": settings.clinical_basis,
        "primary_lens": primary,
        "marker_table": rows,
        "tier_counts": {tier: sum(1 for item in rows if item["tier"] == tier) for tier in TIER_ORDER},
        "cohort": {
            "n": int(cohort.time.shape[0]),
            "events": int(cohort.event.sum()),
            "n_markers_evaluated": len(cohort.marker_names),
            "dropped_markers": cohort.dropped_markers,
            "clinical_columns": cohort.clinical_columns,
            "clinical_design_columns": cohort.clinical_names,
            "dropped_clinical_columns": list(cohort.dropped_clinical),
            "strata_columns": cohort.strata_columns,
            "row_mask_hash": cohort.row_mask_hash,
            "notes": cohort_notes,
        },
        "duplicates": duplicates,
        "null": {
            "n_permutations": int(adjusted[primary]["n_permutations"]),
            "lens2_null": settings.lens2_null if primary == "added_value" else None,
            "assumption_note": (
                "Added-value residual permutation approximates a conditional null and assumes exchangeable residuals after the declared clinical-basis adjustment. Nonlinear marker-covariate relations can inflate false positives; family-wise error control also requires subset pivotality. Clinical Cox baseline misspecification, including non-proportional hazards, can invalidate conditional-null interpretation; multiplicity adjustment does not establish model adequacy."
                if primary == "added_value" and settings.lens2_null == "smith"
                else "Permutation inference assumes exchangeability under the chosen null; strong family-wise error control also requires subset pivotality."
                + (" Clinical Cox baseline misspecification can invalidate added-value interpretation; multiplicity adjustment does not establish model adequacy." if primary == "added_value" else "")
            ),
        },
        "resampling": {
            "scheme": "event-stratified subsampling without replacement",
            "fraction": float(settings.resample_fraction),
            "n_valid": resampling.n_valid,
            "n_failed": resampling.n_failed,
            "n_withheld": resampling.n_withheld,
            "inference_policy": "Training diagnostics are refitted; withheld subsamples select no added-value markers and remain in the selection-frequency denominator.",
            "stability_assessed": stability_assessed,
            "note": None
            if stability_assessed
            else "No subsample was evaluated, so the stability of the selection was not assessed and no marker can be robust.",
        },
        "signature": {
            "markers": [] if signature is None else [cohort.marker_names[int(column)] for column in signature.columns],
            # A model of the clinical covariates alone, because the procedure selected no marker.
            "clinical_only": signature is not None and signature.columns.size == 0,
            "apparent_c": apparent_c,
            # A heuristic subsampling gap adjustment: its train/left-out gap also
            # reflects training size. This is not Harrell's bootstrap optimism estimate.
            # Keep the original metric key for saved results and API clients.
            "correction_method": "subsampling_train_vs_left_out_gap",
            "correction_note": "The apparent C-index is adjusted by the mean train-versus-left-out gap of the whole selection procedure. This heuristic includes training-size effects; use locked external validation for final performance claims.",
            "optimism_corrected_c": None
            if apparent_c is None or resampling.optimism["signature_optimism"] is None
            else float(apparent_c - resampling.optimism["signature_optimism"]),
            **resampling.optimism,
            "notes": signature_notes,
        },
        "nonlinear_lens": None
        if nonlinear is None
        else {key: value for key, value in nonlinear.items() if key not in {"mean_importance", "positive_fraction", "n_evaluated", "sign_test_p_holm"}},
        "locked_recipe": recipe,
        "settings": settings._asdict(),
    }
    # JSON has no infinity or NaN (Starlette refuses them): every non-finite number is reported as None.
    return _json_ready(result)


def _km_survival_at(time: np.ndarray, event: np.ndarray, horizon: float) -> float:
    survival = 1.0
    for value in np.unique(time[(event == 1) & (time <= horizon)]):
        at_risk = float(np.sum(time >= value))
        deaths = float(np.sum((time == value) & (event == 1)))
        survival *= 1.0 - deaths / at_risk
    return survival


def _external_cox(
    time: np.ndarray,
    event: np.ndarray,
    exog: np.ndarray,
    strata: np.ndarray | None,
    ties: str = "efron",
) -> dict[str, float | None] | None:
    fit = fit_cox(time, event, exog, strata, ties)
    beta = float(fit.beta[-1])
    se = float(np.sqrt(fit.covariance[-1, -1])) if fit.covariance[-1, -1] > 0 else float("nan")
    if not _marker_fit_usable(fit) or not np.isfinite(se) or se <= 0:
        return None
    z_value = float(stats.norm.ppf(0.975))
    with np.errstate(over="ignore"):
        return {
            "log_hr": beta,
            "hazard_ratio": _finite_or_none(np.exp(beta)),
            "ci_lower": _finite_or_none(np.exp(beta - z_value * se)),
            "ci_upper": _finite_or_none(np.exp(beta + z_value * se)),
            "wald_p": float(2.0 * stats.norm.sf(abs(beta / se))),
        }


def _check_recipe(recipe: Any) -> int:
    """The recipe's version, after checking its structure and types; a malformed recipe fails with a clear message.

    Recipes come back from the browser or from files, so nothing in them is trusted: every name must
    be text, every number finite, the coefficients one per term, and the markers model terms.
    """

    def fail(message: str) -> NoReturn:
        raise ValueError(f"The locked model is malformed: {message} Export it again from a marker evaluation.")

    def is_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value))
        )

    def names(value: Any, what: str) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            fail(f"{what} must be a list of names.")
        if len(set(value)) != len(value):
            fail(f"{what} repeats a name.")
        return value

    def numbers(value: Any, what: str, length: int) -> None:
        if not isinstance(value, list) or len(value) != length or not all(is_number(item) for item in value):
            fail(f"{what} must hold {length} finite number(s).")

    if not isinstance(recipe, dict):
        fail("it must be a JSON object.")
    version = recipe.get("recipe_version")
    if not is_number(version) or float(version) not in RECIPE_VERSIONS:
        raise ValueError("This recipe was written by an incompatible SurvStudio version.")
    version = int(version)
    if not isinstance(recipe.get("recipe_hash"), str):
        fail("recipe_hash is missing.")
    outcome = recipe.get("outcome")
    if not isinstance(outcome, dict) or not all(isinstance(outcome.get(key), str) and outcome.get(key) for key in ("time_column", "event_column")):
        fail("outcome must name the time and event columns.")
    clinical = recipe.get("clinical")
    if not isinstance(clinical, dict):
        fail("clinical must be an object.")
    columns = names(clinical.get("columns"), "clinical.columns")
    if version >= 3 and clinical.get("basis") not in CLINICAL_BASES:
        fail("clinical.basis is invalid.")
    names(clinical.get("categorical") or [], "clinical.categorical")
    feature_names: list[str] = []
    if columns:
        encoder = clinical.get("encoder")
        if not isinstance(encoder, dict):
            fail("clinical.encoder is missing.")
        if list(encoder.get("features") or []) != columns:
            fail("clinical.encoder must encode the clinical columns.")
        feature_names = names(encoder.get("feature_names"), "clinical.encoder.feature_names")
        if version >= 3:
            check_clinical_encoder(encoder)
            if clinical["basis"] != encoder["clinical_basis"]:
                fail("clinical basis and frozen encoder disagree.")
        mappings = encoder.get("categorical_mappings")
        categorical = encoder.get("categorical_features") or []
        if not isinstance(categorical, list) or (categorical and not isinstance(mappings, dict)):
            fail("clinical.encoder lacks its categorical levels.")
        for column in categorical:
            mapping = mappings.get(column)
            if not isinstance(mapping, dict) or not isinstance(mapping.get("all_levels"), list) or not isinstance(mapping.get("retained_levels"), list):
                fail(f"clinical.encoder lacks the levels of {column}.")
        impute = encoder.get("numeric_impute_values") or {}
        if not isinstance(impute, dict) or not all(is_number(value) for value in impute.values()):
            fail("clinical.encoder.numeric_impute_values must be finite numbers.")
    names(recipe.get("strata_columns") or [], "strata_columns")
    markers = names(recipe.get("markers"), "markers")
    for field in ("marker_medians", "marker_development_log_hr"):
        values = recipe.get(field)
        if not isinstance(values, dict) or not all(is_number(values.get(name)) for name in markers):
            fail(f"{field} needs a finite number for every marker.")
    if recipe.get("primary_lens", "marginal") not in LENSES:
        fail("primary_lens must be marginal or added_value.")
    model = recipe.get("model")
    if not isinstance(model, dict):
        fail("model is missing.")
    terms = names(model.get("terms"), "model.terms")
    numbers(model.get("coefficients"), "model.coefficients (one per term)", len(terms))
    missing = [name for name in markers if name not in terms]
    if missing:
        fail("the markers " + ", ".join(missing[:5]) + " are not model terms.")
    unknown = [term for term in terms if term not in markers and term not in feature_names]
    if unknown:
        fail("the model terms " + ", ".join(unknown[:5]) + " are neither markers nor encoded clinical covariates.")
    scale = recipe.get("marker_scale")
    if scale is not None:
        if not isinstance(scale, dict):
            fail("marker_scale must be an object.")
        # Every locked marker has its development scale (a recipe locked before the scale existed has none at all).
        for name in markers:
            entry = scale.get(name)
            if not isinstance(entry, dict) or not is_number(entry.get("mean")) or not is_number(entry.get("sd")) or float(entry["sd"]) < 0:
                fail(f"marker_scale of {name} needs a finite mean and SD.")
    if model.get("ties", "efron") not in TIES_METHODS:
        fail(f"model.ties must be one of {', '.join(TIES_METHODS)}.")
    if not is_number(model.get("default_horizon")):
        fail("model.default_horizon must be a finite number.")
    baseline = model.get("baseline")
    if baseline is not None:
        if not isinstance(baseline, dict):
            fail("model.baseline must be an object.")
        times = baseline.get("times")
        if not isinstance(times, list) or not all(is_number(value) for value in times) or any(b < a for a, b in zip(times, times[1:])):
            fail("model.baseline.times must be increasing finite numbers.")
        if version >= 2:
            numbers(baseline.get("log_cumulative_hazard"), "model.baseline.log_cumulative_hazard", len(times))
            if not is_number(baseline.get("lp_center")):
                fail("model.baseline.lp_center must be a finite number.")
        else:
            survival = baseline.get("survival")
            if not isinstance(survival, list) or len(survival) != len(times) or not all(is_number(value) and 0.0 <= float(value) <= 1.0 for value in survival):
                fail("model.baseline.survival must hold one probability per time.")
    clinical_only = recipe.get("clinical_only_model")
    if clinical_only is not None:
        if not isinstance(clinical_only, dict):
            fail("clinical_only_model must be an object.")
        only_terms = names(clinical_only.get("terms"), "clinical_only_model.terms")
        numbers(clinical_only.get("coefficients"), "clinical_only_model.coefficients (one per term)", len(only_terms))
        unknown = [term for term in only_terms if term not in feature_names]
        if unknown:
            fail("the clinical-only terms " + ", ".join(unknown[:5]) + " are not encoded clinical covariates.")
    if version >= 3:
        inference = recipe.get("inference")
        if not isinstance(inference, dict) or inference.get("status") not in {"withheld", "assumption_dependent"} or inference.get("method_version") not in SUPPORTED_METHOD_VERSIONS:
            fail("v3 must preserve its versioned inference status.")
        if not isinstance(inference.get("allowed"), bool) or inference["allowed"] != (inference["status"] == "assumption_dependent"):
            fail("v3 inference status and permission disagree.")
        if inference.get("clinical_basis") != clinical["basis"]:
            fail("diagnostic and prediction bases disagree.")
        if inference.get("method_version") == METHOD_VERSION:
            bootstrap = inference.get("bootstrap")
            if not isinstance(bootstrap, dict) or bootstrap.get("candidate") not in CANDIDATES or bootstrap.get("draws") != N_BOOTSTRAP or isinstance(bootstrap.get("seed"), bool) or not isinstance(bootstrap.get("seed"), int) or not 0 <= bootstrap["seed"] < 2**32:
                fail("v3 joint diagnostic provenance is missing or invalid.")
        if inference.get("functional_form_encoder") is not None:
            check_clinical_encoder(inference["functional_form_encoder"])
    return version


def _unseen_level_rows(frame: pd.DataFrame, encoder: dict[str, Any], column: str) -> np.ndarray:
    """Rows whose value of a categorical covariate is none of the levels the locked model knows.

    The locked encoder itself decides (with every level given its own indicator), so this agrees
    with how the rows are scored, however the encoder matches values to levels.
    """
    encoder = encoder.get("base_encoder", encoder)
    mapping = encoder["categorical_mappings"][column]
    levels = [str(level) for level in mapping.get("all_levels") or []]
    observed = frame[column].notna().to_numpy()
    if not levels:
        return observed
    indicators = {level: f"level {index}" for index, level in enumerate(levels)}
    probe = {
        **encoder,
        "features": [column],
        "categorical_features": [column],
        "numeric_features": [],
        "categorical_mappings": {
            column: {**mapping, "retained_levels": levels, "level_columns": indicators, "unknown_column": None, "missing_column": None}
        },
        "feature_names": list(indicators.values()),
    }
    seen = np.asarray(transform_feature_encoder(frame, probe, output="numpy"), dtype=float).sum(axis=1) > 0
    return observed & ~seen


def _clinical_columns_behind(encoder: dict[str, Any] | None, columns: Sequence[str], terms: set[str]) -> list[str]:
    """The clinical columns whose encoded features include one of ``terms``, in their order.

    A numeric column is encoded under its own name; a categorical one as one indicator per retained level
    (and one for missing values when development had any).
    """
    if not columns or not encoder:
        return []
    mappings = encoder.get("categorical_mappings") or {}
    categorical = set(encoder.get("categorical_features") or [])
    used = []
    for column in columns:
        encoded = set((encoder.get("spline_specifications", {}).get(column) or {}).get("terms") or [column])
        if column in categorical:
            mapping = mappings.get(column) or {}
            level_columns = mapping.get("level_columns") or {}
            encoded = {level_columns.get(level, f"{column}_{level}") for level in mapping.get("retained_levels") or []}
            encoded |= {mapping[key] for key in ("missing_column", "unknown_column") if mapping.get(key)}
        if encoded & terms:
            used.append(column)
    return used


def _reject_text_in_numeric_covariate(values: pd.Series, name: str) -> None:
    """Refuse a numeric clinical covariate whose non-missing values include text, such as "." or "unknown".

    The locked encoder would score such rows at the development median without a word; blank cells
    are missing values instead, and their rows are left out as in development.
    """
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return
    text = values.dropna().astype(str)
    numbers = pd.to_numeric(text.str.strip(), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    bad = text[~np.isfinite(numbers)]
    if bad.size:
        examples = ", ".join(f'"{value}"' for value in list(dict.fromkeys(bad.tolist()))[:3])
        raise ValueError(
            f'Clinical covariate "{name}" holds {bad.size} value(s) in the external dataset that are not numbers, such as '
            f"{examples}. Blank those cells so their rows are left out as missing, or recode them as numbers."
        )


def _holm(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind="mergesort")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (valid.size - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


@user_input_boundary
def _validate_v3_locked_recipe(
    df: pd.DataFrame,
    recipe: dict[str, Any],
    *,
    column_mapping: dict[str, str] | None = None,
    event_positive_value: Any = None,
    horizon: float | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    random_seed: int = 20260926,
    marker_scaling: str = "as_measured",
) -> dict[str, Any]:
    """Apply a locked marker recipe, unchanged, to an external cohort and score it.

    Reports the locked model's discrimination (Harrell's C with a bootstrap CI), its
    calibration slope, observed/expected risk, Brier score and Brier skill at the
    horizon, the C-index gain over the locked clinical-only model, and each marker's
    external hazard ratio with a Holm-adjusted one-sided replication test in the
    development direction. ``column_mapping`` maps recipe column names to the
    external dataset's names when they differ.

    ``marker_scaling="within_cohort"`` is for a cohort measured on another platform (for
    example microarrays against RNA-seq): each marker is mapped onto its development
    distribution by its z-score within the cohort, so the model's relative weights hold;
    discrimination is then comparable, absolute risks and calibration only roughly. Locked
    markers the external dataset does not have are held at their development median (a
    constant), provided at least half of the model's marker weight (|coefficient| x
    development SD) is measured.

    Recipes of version 1 (hashed on Python's JSON text) are still accepted when their stored hash
    matches that computation; their baseline survival was stored at a linear predictor of 0, so
    absolute risks are withheld at a horizon where it lost its precision.
    """
    from survival_toolkit.ml_models import _brier_scores_from_weights, _ipcw_brier_weights

    version = _check_recipe(recipe)
    expected_hash = _legacy_recipe_hash(recipe) if version == 1 else recipe_hash(recipe)
    if recipe.get("recipe_hash") != expected_hash:
        raise ValueError("The recipe does not match its hash; it was edited after it was locked.")
    df = _unique_rows(df)
    mapping = {str(key): str(value) for key, value in (column_mapping or {}).items()}

    def external(name: str) -> str:
        return mapping.get(name, name)

    outcome = recipe["outcome"]
    time_column = external(outcome["time_column"])
    event_column = external(outcome["event_column"])
    positive = outcome["event_positive_value"] if event_positive_value is None else event_positive_value
    clinical_columns = list(recipe["clinical"]["columns"])
    strata_columns = list(recipe.get("strata_columns") or [])
    markers = list(recipe["markers"])
    encoder = recipe["clinical"]["encoder"]
    # Only the clinical columns behind the locked terms are needed: one development left out of the model
    # (for example a copy of another column) neither has to exist here nor removes rows where it is missing.
    clinical_only_terms = (recipe.get("clinical_only_model") or {}).get("terms") or []
    used_clinical = _clinical_columns_behind(encoder, clinical_columns, {*recipe["model"]["terms"], *clinical_only_terms})
    if marker_scaling not in MARKER_SCALINGS:
        raise ValueError(f"marker_scaling must be one of {MARKER_SCALINGS}.")
    needed = [time_column, event_column, *[external(column) for column in [*used_clinical, *strata_columns]]]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise ValueError("The external dataset lacks columns the recipe needs: " + ", ".join(missing[:6]) + ".")
    scale = recipe.get("marker_scale") or {}
    if marker_scaling == "within_cohort" and markers and not scale:
        raise ValueError("This locked model has no development marker scale; lock it again with this SurvStudio version to rescale markers within the cohort.")
    absent = [name for name in markers if external(name) not in df.columns]
    coefficient_of = dict(zip(recipe["model"]["terms"], recipe["model"]["coefficients"]))
    weight = {name: abs(float(coefficient_of[name])) * float((scale.get(name) or {}).get("sd", 1.0) or 1.0) for name in markers}
    total_weight = sum(weight.values())
    weight_available = 1.0 if total_weight <= 0 else sum(value for name, value in weight.items() if name not in absent) / total_weight
    if absent and (len(absent) == len(markers) or weight_available < MIN_MARKER_WEIGHT_AVAILABLE):
        raise ValueError(
            f"The external dataset lacks locked markers carrying {1.0 - weight_available:.0%} of the model's marker weight "
            f"({', '.join(external(name) for name in absent[:6])}{' ...' if len(absent) > 6 else ''}); at least half must be measured."
        )

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=positive,
        extra_columns=[external(column) for column in (strata_columns if version >= 3 else [*used_clinical, *strata_columns])],
    )
    if version >= 3:
        rows = list(frame.attrs["source_row_index"])
        for column in used_clinical:
            frame[external(column)] = df.loc[rows, external(column)].reset_index(drop=True)
    frame = frame.rename(columns={external(column): column for column in [*used_clinical, *strata_columns]})
    time = frame[time_column].to_numpy(dtype=float)
    event = frame[event_column].to_numpy(dtype=int)
    source_rows = list(frame.attrs["source_row_index"])
    strata = _build_cox_strata_payload(frame, strata_columns)["codes"] if strata_columns else None

    notes: list[str] = []
    columns: dict[str, np.ndarray] = {}
    if clinical_columns:
        numeric_features = set(encoder.get("numeric_features") or [])
        for column in used_clinical:
            if column in numeric_features:
                _reject_text_in_numeric_covariate(frame[column], external(column))
                if np.isinf(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).any():
                    raise ValueError(f'Clinical covariate "{external(column)}" contains infinite external values.')
        # The columns the model does not use are given as missing: their encoded values are never read.
        unused = {column: np.nan for column in clinical_columns if column not in used_clinical}
        design = transform_clinical_encoder(frame.assign(**unused) if unused else frame, encoder, output="dataframe")
        for name in design.columns:
            columns[str(name)] = design[name].to_numpy(dtype=float)
        for column in encoder.get("categorical_features", []):
            if column not in used_clinical:
                continue
            unseen = _unseen_level_rows(frame, encoder, column)
            unseen_count = int(unseen.sum())
            observed_count = int(frame[column].notna().sum())
            if unseen_count and unseen_count > MAX_UNSEEN_LEVEL_SHARE * observed_count:
                # Scoring these rows as the reference level would silently collapse the covariate.
                external_levels = sorted({str(value) for value in frame.loc[unseen, column]})
                known = [str(level) for level in encoder["categorical_mappings"][column].get("all_levels") or []]
                raise ValueError(
                    f"{unseen_count} of {observed_count} external rows have a {column} level the locked model has not seen "
                    f"({', '.join(repr(level) for level in external_levels[:5])}{' ...' if len(external_levels) > 5 else ''}; "
                    f"it knows {', '.join(repr(level) for level in known[:8])}). Recode {column} in the external dataset "
                    "the way it was coded in development."
                )
            if unseen_count:
                notes.append(f"{unseen_count} external row(s) have a {column} level not seen in development; scored as the reference level.")
    infinite: list[str] = []
    for name in markers:
        median = float(recipe["marker_medians"][name])
        if name in absent:
            columns[name] = np.full(time.shape[0], median)
            continue
        raw = df.loc[source_rows, external(name)]
        numeric = pd.to_numeric(raw, errors="coerce")
        if bool((raw.notna() & numeric.isna()).any()):
            raise ValueError(f'Marker "{external(name)}" contains text in the external dataset.')
        values = numeric.to_numpy(dtype=float, copy=True, na_value=np.nan)
        if np.isinf(values).any():
            infinite.append(external(name))
            continue
        imputed = int(np.isnan(values).sum())
        if marker_scaling == "within_cohort":
            observed = values[~np.isnan(values)]
            spread = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
            if spread <= 0:
                notes.append(f"{name} does not vary in the external cohort; it was held at its development median.")
                columns[name] = np.full(time.shape[0], median)
                continue
            values = float(scale[name]["mean"]) + float(scale[name]["sd"]) * (values - float(np.mean(observed))) / spread
        if imputed:
            notes.append(f"{imputed} missing {name} value(s) imputed with the development median.")
        columns[name] = np.where(np.isnan(values), median, values)
    if infinite:
        # Imputing them at the median would score the patients with the most extreme values as typical.
        raise ValueError(
            f"Locked markers hold infinite values in the external dataset: {', '.join(infinite[:6])}"
            f"{' ...' if len(infinite) > 6 else ''}. Transform them as in development so every value is finite, for example "
            "log(x + 1) instead of log(x)."
        )
    if absent:
        notes.append(
            f"{len(absent)} locked marker(s) are not in the external dataset ({', '.join(absent)}); they were held at their "
            f"development median, so the model ran on {weight_available:.0%} of its marker weight."
        )
    if marker_scaling == "within_cohort" and markers:
        notes.append(
            "Markers were rescaled within this cohort to their development mean and SD, as for data from another platform: "
            "the C-index compares like with like, absolute risks and calibration only roughly."
        )

    model = recipe["model"]
    linear_predictor = np.column_stack([columns[term] for term in model["terms"]]) @ np.asarray(model["coefficients"], dtype=float)
    c_index = _pooled_c_index(time, event, linear_predictor, strata)
    rng = np.random.default_rng(int(random_seed))
    clinical_model = recipe.get("clinical_only_model")
    clinical_predictor = None
    if clinical_model:
        clinical_predictor = np.column_stack([columns[term] for term in clinical_model["terms"]]) @ np.asarray(
            clinical_model["coefficients"], dtype=float
        )
    c_draws: list[float] = []
    clinical_draws: list[float] = []
    delta_draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        raise_if_cancelled()
        rows = rng.integers(0, time.shape[0], size=time.shape[0])
        if not event[rows].any():
            continue
        risks = linear_predictor[rows][:, None] if clinical_predictor is None else np.column_stack([linear_predictor[rows], clinical_predictor[rows]])
        # A stratified model compares patients within strata only, as its point estimate does.
        draws = harrell_c_many(time[rows], event[rows], risks) if strata is None else _stratified_c_many(time[rows], event[rows], risks, strata[rows])
        c_draws.append(float(draws[0]))
        if clinical_predictor is not None:
            clinical_draws.append(float(draws[1]))
            delta_draws.append(float(draws[0] - draws[1]))

    def interval(draws: list[float]) -> list[float | None]:
        finite = [value for value in draws if np.isfinite(value)]
        if len(finite) < 20:
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    def log_or_none(value: float | None) -> float | None:
        return None if value is None or value <= 0 else float(np.log(value))

    ties = str(model.get("ties", "efron"))
    slope_fit = _external_cox(time, event, linear_predictor[:, None], strata, ties)
    metrics: dict[str, Any] = {
        "marker_scaling": marker_scaling,
        "marker_weight_available": float(weight_available),
        "absent_markers": absent,
        "c_index": _finite_or_none(c_index),
        "c_index_ci": interval(c_draws),
        "calibration_slope": None if slope_fit is None else slope_fit["log_hr"],
        "calibration_slope_ci": None if slope_fit is None else [log_or_none(slope_fit["ci_lower"]), log_or_none(slope_fit["ci_upper"])],
    }
    if clinical_predictor is not None:
        clinical_c = _pooled_c_index(time, event, clinical_predictor, strata)
        metrics["clinical_only_c_index"] = _finite_or_none(clinical_c)
        # From the same bootstrap draws as the locked model's interval.
        metrics["clinical_only_c_index_ci"] = interval(clinical_draws)
        metrics["delta_c_index"] = _finite_or_none(c_index - clinical_c)
        metrics["delta_c_index_ci"] = interval(delta_draws)
    target = float(model["default_horizon"] if horizon is None else horizon)
    baseline = model.get("baseline")
    predicted = _baseline_survival_at(baseline, linear_predictor, target) if baseline and target > 0 else None
    if baseline and target > 0 and predicted is None:
        notes.append(
            "The locked model's baseline survival was stored far from its patients' linear predictor by an earlier "
            "SurvStudio version and lost its precision at this horizon, so absolute risks, observed/expected risk and the "
            "Brier score are not reported. Lock the model again to get them."
        )
    last_event_time = float(baseline["times"][-1]) if baseline and len(baseline["times"]) else None
    if predicted is not None and last_event_time is not None and target > last_event_time:
        notes.append(
            f"The horizon ({target:g}) is past the last event time of the development cohort ({last_event_time:g}); the locked "
            "baseline hazard stays flat after it, so the predicted risks at this horizon are likely too low."
        )
    if predicted is not None:
        observed_survival = _km_survival_at(time, event, target)
        weights, alive = _ipcw_brier_weights(time, event, np.array([target]), support_times=time, support_events=event)
        brier = float(_brier_scores_from_weights(weights, alive, predicted[:, None])[0])
        null_brier = float(_brier_scores_from_weights(weights, alive, np.full((time.shape[0], 1), observed_survival))[0])
        expected_risk = float(np.mean(1.0 - predicted))
        metrics.update(
            {
                "horizon": target,
                "observed_risk": float(1.0 - observed_survival),
                "expected_risk": expected_risk,
                "observed_expected_ratio": float((1.0 - observed_survival) / expected_risk) if expected_risk > 0 else None,
                "brier": brier,
                "brier_skill": float(1.0 - brier / null_brier) if null_brier > 0 else None,
            }
        )
    elif strata is not None:
        notes.append("The recipe's model is stratified, so absolute risks and calibration at a horizon are not available.")

    primary = recipe.get("primary_lens", "marginal")
    base_design = None
    if clinical_columns:
        # Added value over the clinical terms of the locked model (the covariates it could estimate), less
        # those this cohort cannot estimate: an indicator that does not vary here, or the last one of a
        # covariate whose reference level this cohort lacks.
        clinical_terms = [term for term in model["terms"] if term not in markers]
        base_design = np.column_stack([columns[term] for term in clinical_terms]) if clinical_terms else np.zeros((time.shape[0], 0))
        base_design = _estimable_clinical_design(base_design, strata)
    tested_lens = primary
    if primary == "added_value" and (base_design is None or not base_design.shape[1]):
        tested_lens = "unavailable"
    if primary == "added_value" and tested_lens == "unavailable" and markers:
        notes.append("The requested added-value replication is unavailable: no clinical term is estimable in this external cohort.")
    marker_rows = []
    one_sided: list[float] = []
    for name in markers:
        if name in absent:
            one_sided.append(float("nan"))
            marker_rows.append({"marker": name, "marginal": None, "adjusted": None, "same_direction": False, "absent": True, "tested": None})
            continue
        marginal = _external_cox(time, event, columns[name][:, None], strata, ties)
        adjusted = None if base_design is None else _external_cox(time, event, np.column_stack([base_design, columns[name]]), strata, ties)
        tested = adjusted if tested_lens == "added_value" else marginal if tested_lens == "marginal" else None
        if tested_lens == "added_value" and adjusted is None:
            # Testing the unadjusted association instead would answer another question than the locked claim.
            notes.append(
                f"The added-value replication of {name} is not estimable here: its Cox fit with the clinical covariates did not "
                "converge, its coefficient runs to infinity, or it does not vary apart from the clinical covariates in this cohort."
            )
        development_sign = float(np.sign(recipe["marker_development_log_hr"][name]))
        same_direction = tested is not None and np.sign(tested["log_hr"]) == development_sign
        if tested is None:
            # Still one of the family the locked model declared: whether its fit fails depends on the outcome,
            # so leaving it out would shrink the Holm family in a data-dependent way. It counts as p = 1.
            one_sided.append(1.0)
        else:
            half = tested["wald_p"] / 2.0
            one_sided.append(half if same_direction else 1.0 - half)
        marker_rows.append(
            {
                "marker": name,
                "marginal": marginal,
                "adjusted": adjusted,
                "same_direction": bool(same_direction),
                "tested": tested_lens if tested is not None else None,
            }
        )
    # Markers the external dataset lacks are known to be missing before any outcome is seen, so they are not
    # part of the family; one whose test could not be estimated is, and is reported as not estimable.
    for row, adjusted_p in zip(marker_rows, _holm(one_sided)):
        estimable = row["tested"] is not None
        row["replication_p_holm"] = _finite_or_none(adjusted_p) if estimable else None
        row["replicated"] = bool(estimable and row["same_direction"] and np.isfinite(adjusted_p) and adjusted_p <= alpha)

    inference = recipe.get("inference") if version >= 3 else {
        "status": "not_assessed", "allowed": False, "method_version": None,
        "reasons": ["Legacy v1/v2 recipe: development diagnostics were not assessed."]}
    development_inference = inference
    external_inference = None
    if version >= 3:
        diagnostic_names = list(encoder["feature_names"]) if clinical_columns else []
        diagnostic_frame = frame.assign(**{column: np.nan for column in clinical_columns if column not in frame})
        diagnostic_cohort = MarkerCohort(
            time=time, event=event,
            markers=np.column_stack([columns[name] for name in markers]) if markers else np.zeros((len(time), 0)),
            marker_names=markers,
            clinical=np.column_stack([columns[name] for name in diagnostic_names]) if diagnostic_names else None,
            clinical_names=diagnostic_names, clinical_columns=clinical_columns,
            strata=None if strata is None else np.asarray(strata), strata_columns=strata_columns,
            source_rows=source_rows, row_mask_hash=str(frame.attrs.get("row_mask_hash") or ""),
            dropped_markers=[], clinical_encoder=encoder,
            clinical_frame=diagnostic_frame[clinical_columns] if clinical_columns else None,
            categorical_clinical=tuple(recipe["clinical"].get("categorical") or []),
            clinical_basis=recipe["clinical"].get("basis", "linear"),
        )
        external_inference = diagnose_markers(
            diagnostic_cohort, diagnostic_cohort.markers, ties=ties,
            functional_form_encoder=development_inference.get("functional_form_encoder"), frozen_transform=True,
            candidate=development_inference["bootstrap"]["candidate"], bootstrap_seed=development_inference["bootstrap"]["seed"])
        if not external_inference["allowed"]:
            inference = {**development_inference, "status": "withheld", "allowed": False,
                         "reasons": [*development_inference.get("reasons", []),
                                     *("external: " + reason for reason in external_inference["reasons"])]}
    if not inference["allowed"]:
        for row in marker_rows:
            row["exploratory"] = {"replication_p_holm": row.get("replication_p_holm"), "replicated": row.get("replicated"),
                                  "marginal": row.get("marginal"), "adjusted": row.get("adjusted")}
            row["replication_p_holm"] = None
            row["replicated"] = False
            row["inference_status"] = inference["status"]
            for lens in ("marginal", "adjusted"):
                if row.get(lens):
                    row[lens] = {**row[lens], "wald_p": None}
        notes.append("Development inference status: " + inference["status"] + ". Prediction metrics remain exploratory and do not reinstate the marker claim.")
    return _json_ready(
        {
            "inference": inference,
            "development_inference": development_inference,
            "external_inference": external_inference,
            "recipe_hash": recipe["recipe_hash"],
            "recipe_version": version,
            "cohort": {"n": int(time.shape[0]), "events": int(event.sum()), "row_mask_hash": str(frame.attrs.get("row_mask_hash") or "")},
            "metrics": metrics,
            "markers": marker_rows,
            "notes": notes,
        }
    )


def validate_locked_recipe(df, recipe, **kwargs):
    """Preserve historical recipes through their unchanged v2 validator."""
    inference=recipe.get("inference") if isinstance(recipe,dict) else None
    if not isinstance(inference,dict) or inference.get("method_version") != METHOD_VERSION:
        from survival_toolkit.marker_evaluation import validate_locked_recipe as legacy_validator
        return legacy_validator(df, recipe, **kwargs)
    return _validate_v3_locked_recipe(df, recipe, **kwargs)

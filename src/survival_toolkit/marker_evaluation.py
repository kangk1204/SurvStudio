"""Honest one-step evaluation of candidate prognostic markers.

A marker list is screened, and the whole screening procedure is then checked the way
a reader should check a published marker claim:

1. Two lenses per marker, both Cox score tests from ``marker_screen``: *marginal*
   association (no covariates) and *added value* over the clinical baseline.
2. Westfall–Young max-statistic permutation p-values over all markers (family-wise
   error) and permutation FDR q-values. The added-value null permutes the marker
   residuals left after projecting on the clinical covariates (Freedman–Lane), so it
   keeps each marker's relationship with the clinical covariates.
3. The whole procedure rerun on subsamples: how often each marker is selected, its
   rank interval, and whether its direction holds.
4. Tiers from pre-declared rules, and an optimism report for a signature built from
   the selected markers ("winner's curse" of picking the strongest marker).

Nothing here dichotomizes markers; every statistic uses the continuous values.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

from survival_toolkit.analysis import (
    _build_cox_strata_payload,
    _cohort_frame,
    _harrell_c_index_counts,
    fit_phreg,
)
from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.encoding import fit_feature_encoder, transform_feature_encoder
from survival_toolkit.errors import user_input_boundary
from survival_toolkit.marker_screen import (
    TIES_METHODS,
    CoxScoreScreen,
    MaxTAccumulator,
    PermutationFdrAccumulator,
    ScoreStats,
    bh_vector,
    fit_cox_null,
    residualize,
    stratified_permutation,
)

LENSES = ("marginal", "added_value")
TIER_ORDER = ("robust", "suggestive", "marginal only", "not supported")
_PERMUTATION_BATCH = 16


class MarkerSettings(NamedTuple):
    """Pre-declared settings; every threshold used for a tier is listed here."""

    ties: str = "efron"
    alpha: float = 0.05
    fdr_level: float = 0.10
    n_permutations: int = 1000
    n_resamples: int = 200
    resample_fraction: float = 0.632
    max_missing_fraction: float = 0.2
    robust_frequency: float = 0.8
    robust_direction: float = 0.95
    max_signature_markers: int = 10
    shortlist_size: int = 50
    lens2_null: str = "freedman_lane"
    random_seed: int = 20260926


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


def _validated_settings(settings: MarkerSettings) -> MarkerSettings:
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
    if settings.lens2_null not in {"freedman_lane", "raw"}:
        raise ValueError('lens2_null must be "freedman_lane" or "raw".')
    if settings.max_signature_markers < 1 or settings.shortlist_size < 0:
        raise ValueError("max_signature_markers must be at least 1 and shortlist_size at least 0.")
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
) -> MarkerCohort:
    """Align the outcome, clinical design, strata and numeric marker block row by row.

    Rows missing the outcome, a clinical covariate or a stratum are dropped (as in the
    Cox workflow). A missing marker value never drops a row: markers with more missing
    values than ``max_missing_fraction`` are left out, and the rest are imputed inside
    each fit of the procedure.
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
    missing = [column for column in [*markers, *clinical, *strata] if column not in df.columns]
    if missing:
        raise ValueError("Columns not found in the dataset: " + ", ".join(missing[:5]) + ".")

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=[*clinical, *strata],
    )
    source_rows = list(frame.attrs["source_row_index"])
    raw_markers = df.loc[source_rows, markers]
    non_numeric: list[str] = []
    values = np.empty((len(source_rows), len(markers)), dtype=float)
    for index, column in enumerate(markers):
        series = raw_markers[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if bool((series.notna() & numeric.isna()).any()):
            non_numeric.append(column)
            continue
        column_values = numeric.to_numpy(dtype=float, copy=True)
        column_values[~np.isfinite(column_values)] = np.nan
        values[:, index] = column_values
    if non_numeric:
        raise ValueError(
            "Markers must be numeric; these columns contain text: "
            + ", ".join(non_numeric[:5])
            + (" ..." if len(non_numeric) > 5 else "")
            + ". Recode text values as blank cells or leave the column out."
        )
    missing_share = np.mean(np.isnan(values), axis=0)
    kept: list[int] = []
    dropped: list[dict[str, Any]] = []
    for index, column in enumerate(markers):
        observed = values[:, index][~np.isnan(values[:, index])]
        if missing_share[index] > max_missing_fraction:
            dropped.append({"marker": column, "reason": f"{missing_share[index]:.0%} missing"})
        elif observed.size < 2 or np.unique(observed).size < 2:
            dropped.append({"marker": column, "reason": "constant"})
        else:
            kept.append(index)
    if not kept:
        raise ValueError("No usable markers remain after removing constant or mostly missing columns.")

    clinical_design = None
    clinical_names: list[str] = []
    encoder = None
    if clinical:
        encoder = fit_feature_encoder(frame, clinical, list(categorical_clinical))
        clinical_design = np.asarray(transform_feature_encoder(frame, encoder, output="numpy"), dtype=float)
        clinical_names = list(encoder["feature_names"])
    strata_codes = _build_cox_strata_payload(frame, strata)["codes"] if strata else None
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
    )


def _impute(block: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(block), medians[None, :], block)


def _column_medians(block: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(block, axis=0) if block.size else np.zeros(block.shape[1])
    return np.where(np.isfinite(medians), medians, 0.0)


def _varying_columns(design: np.ndarray) -> np.ndarray:
    """Clinical columns that vary in these rows (a subsample can lose a rare level)."""
    return design[:, np.ptp(design, axis=0) > 0]


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
        design = _varying_columns(cohort.clinical[rows])
        if design.shape[1]:
            clinical_null = fit_cox_null(time, event, design, strata, ties)
            if not clinical_null.converged:
                raise ValueError("The clinical-only Cox model did not converge.")
            screens["added_value"] = (
                CoxScoreScreen(time, event, null=clinical_null, Z=design, strata=strata, ties=ties),
                design,
            )
        else:
            screens["added_value"] = screens["marginal"]
    return screens


def run_procedure(cohort: MarkerCohort, rows: np.ndarray, settings: MarkerSettings) -> ProcedureFit:
    """The whole selection procedure on a set of rows: impute, screen, adjust, select, rank."""
    raw = cohort.markers[rows]
    medians = _column_medians(raw)
    block = _impute(raw, medians)
    lenses: dict[str, LensStats] = {}
    for lens, (screen, _) in _lens_screens(cohort, rows, settings.ties).items():
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
    for lens, (screen, design) in screens.items():
        if lens == "added_value" and design is not None and settings.lens2_null == "freedman_lane":
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


class _SignatureFit(NamedTuple):
    columns: np.ndarray
    params: np.ndarray
    medians: np.ndarray
    design_columns: np.ndarray


def _signature_columns(fit: ProcedureFit, primary: str, limit: int) -> np.ndarray:
    lens = fit.lenses[primary]
    candidates = np.flatnonzero(fit.selected[primary])
    if candidates.size == 0:
        return candidates
    order = np.argsort(-np.nan_to_num(lens.chi2[candidates], nan=-np.inf), kind="mergesort")
    return candidates[order][:limit]


def _fit_signature(cohort: MarkerCohort, rows: np.ndarray, fit: ProcedureFit, primary: str, limit: int) -> _SignatureFit | None:
    """Cox model of the clinical covariates plus the markers the procedure selected."""
    columns = _signature_columns(fit, primary, limit)
    parts = []
    design_columns = np.zeros(0, dtype=np.int64)
    if cohort.clinical is not None:
        varying = np.flatnonzero(np.ptp(cohort.clinical[rows], axis=0) > 0)
        design_columns = varying
        parts.append(cohort.clinical[rows][:, varying])
    if columns.size:
        parts.append(_impute(cohort.markers[rows][:, columns], fit.medians[columns]))
    if not parts or sum(part.shape[1] for part in parts) == 0:
        return None
    exog = np.column_stack(parts)
    strata = None if cohort.strata is None else cohort.strata[rows]
    model = PHReg(cohort.time[rows], exog, status=cohort.event[rows], strata=strata, ties="efron")
    results, converged = fit_phreg(model)
    params = np.asarray(results.params, dtype=float)
    if not converged or not np.isfinite(params).all():
        return None
    return _SignatureFit(columns=columns, params=params, medians=fit.medians, design_columns=design_columns)


def _signature_risk(cohort: MarkerCohort, rows: np.ndarray, signature: _SignatureFit) -> np.ndarray:
    parts = []
    if cohort.clinical is not None:
        parts.append(cohort.clinical[rows][:, signature.design_columns])
    if signature.columns.size:
        parts.append(_impute(cohort.markers[rows][:, signature.columns], signature.medians[signature.columns]))
    return np.column_stack(parts) @ signature.params


def _single_marker_beta(cohort: MarkerCohort, rows: np.ndarray, column: int, medians: np.ndarray, adjusted: bool) -> float:
    marker = _impute(cohort.markers[rows][:, [column]], medians[[column]])
    exog = marker
    if adjusted and cohort.clinical is not None:
        exog = np.column_stack([_varying_columns(cohort.clinical[rows]), marker])
    strata = None if cohort.strata is None else cohort.strata[rows]
    results, converged = fit_phreg(PHReg(cohort.time[rows], exog, status=cohort.event[rows], strata=strata, ties="efron"))
    beta = float(np.asarray(results.params, dtype=float)[-1])
    return beta if converged and np.isfinite(beta) else float("nan")


def resample_procedure(
    cohort: MarkerCohort,
    full: ProcedureFit,
    settings: MarkerSettings,
    rng: np.random.Generator,
    primary: str,
) -> ResamplingSummary:
    """Rerun the whole procedure on event-stratified subsamples of the cohort.

    Each replicate also fits the selected-marker signature on its subsample and scores
    it on the rows left out, and records the strongest marker's log hazard ratio in and
    out of the subsample (the winner's curse of picking the strongest marker).
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
    beta_in: list[float] = []
    beta_out: list[float] = []
    n_failed = 0
    for _ in range(int(settings.n_resamples)):
        raise_if_cancelled()
        rows = _event_stratified_subsample(cohort.event, settings.resample_fraction, rng)
        left_out = np.setdiff1d(np.arange(n), rows)
        try:
            fit = run_procedure(cohort, rows, settings)
        except (ValueError, np.linalg.LinAlgError):
            n_failed += 1
            continue
        for lens in lenses:
            values = fit.lenses[lens]
            counts[lens] += fit.selected[lens]
            valid = np.isfinite(values.z)
            finite[lens] += valid
            agree[lens] += valid & (np.sign(values.z) == full_sign[lens])
            rank_rows[lens].append(fit.ranks[lens])
        signature = _fit_signature(cohort, rows, fit, primary, settings.max_signature_markers)
        if signature is not None and left_out.size and cohort.event[left_out].any():
            strata_in = None if cohort.strata is None else cohort.strata[rows]
            strata_out = None if cohort.strata is None else cohort.strata[left_out]
            c_in.append(_pooled_c_index(cohort.time[rows], cohort.event[rows], _signature_risk(cohort, rows, signature), strata_in))
            c_out.append(_pooled_c_index(cohort.time[left_out], cohort.event[left_out], _signature_risk(cohort, left_out, signature), strata_out))
            if cohort.clinical is not None and signature.columns.size:
                clinical_only = _SignatureFit(
                    columns=np.zeros(0, dtype=np.int64),
                    params=_clinical_params(cohort, rows, signature.design_columns),
                    medians=signature.medians,
                    design_columns=signature.design_columns,
                )
                clinical_c_out.append(
                    _pooled_c_index(cohort.time[left_out], cohort.event[left_out], _signature_risk(cohort, left_out, clinical_only), strata_out)
                )
        top = _top_marker(fit, primary)
        if top is not None and left_out.size and cohort.event[left_out].sum() >= 2:
            adjusted = primary == "added_value"
            inside = _single_marker_beta(cohort, rows, top, fit.medians, adjusted)
            outside = _single_marker_beta(cohort, left_out, top, fit.medians, adjusted)
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
    optimism = _optimism_summary(c_in, c_out, clinical_c_out, beta_in, beta_out)
    return ResamplingSummary(
        n_valid=n_valid,
        n_failed=n_failed,
        selection_frequency=summary_frequency,
        direction_consistency=summary_direction,
        median_rank=median_rank,
        rank_low=rank_low,
        rank_high=rank_high,
        optimism=optimism,
    )


def _clinical_params(cohort: MarkerCohort, rows: np.ndarray, design_columns: np.ndarray) -> np.ndarray:
    design = cohort.clinical[rows][:, design_columns]
    strata = None if cohort.strata is None else cohort.strata[rows]
    results, _ = fit_phreg(PHReg(cohort.time[rows], design, status=cohort.event[rows], strata=strata, ties="efron"))
    return np.asarray(results.params, dtype=float)


def _top_marker(fit: ProcedureFit, primary: str) -> int | None:
    chi2 = fit.lenses[primary].chi2
    if not np.isfinite(chi2).any():
        return None
    return int(np.nanargmax(chi2))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _optimism_summary(
    c_in: Sequence[float],
    c_out: Sequence[float],
    clinical_c_out: Sequence[float],
    beta_in: Sequence[float],
    beta_out: Sequence[float],
) -> dict[str, Any]:
    pairs = [(inside, outside) for inside, outside in zip(c_in, c_out) if np.isfinite(inside) and np.isfinite(outside)]
    shrinkage = None
    if beta_in:
        magnitude = float(np.mean(np.abs(beta_in)))
        if magnitude > 0:
            shrinkage = float(np.mean(np.sign(beta_in) * np.asarray(beta_out)) / magnitude)
    return {
        "signature_c_in_subsample": _mean_or_none([inside for inside, _ in pairs]),
        "signature_c_left_out": _mean_or_none([outside for _, outside in pairs]),
        "signature_optimism": _mean_or_none([inside - outside for inside, outside in pairs]),
        "clinical_c_left_out": _mean_or_none([value for value in clinical_c_out if np.isfinite(value)]),
        "n_signature_replicates": len(pairs),
        "top_marker_log_hr_in_subsample": _mean_or_none([abs(value) for value in beta_in]),
        "top_marker_log_hr_left_out": _mean_or_none(
            [float(np.sign(inside) * outside) for inside, outside in zip(beta_in, beta_out)]
        ),
        "top_marker_shrinkage": shrinkage,
        "n_top_marker_replicates": len(beta_in),
    }


def _exact_fits(cohort: MarkerCohort, full: ProcedureFit, columns: np.ndarray) -> dict[int, dict[str, Any]]:
    """Efron Cox fits for shortlisted markers: HRs with Wald CIs, nested LR test, delta C."""
    strata = cohort.strata
    fits: dict[int, dict[str, Any]] = {}
    clinical_llf = None
    clinical_c = None
    clinical_design = None
    if cohort.clinical is not None and _varying_columns(cohort.clinical).shape[1]:
        clinical_design = _varying_columns(cohort.clinical)
        results, converged = fit_phreg(PHReg(cohort.time, clinical_design, status=cohort.event, strata=strata, ties="efron"))
        if converged:
            clinical_llf = float(results.llf)
            clinical_c = _pooled_c_index(cohort.time, cohort.event, clinical_design @ np.asarray(results.params), strata)
    z_value = float(stats.norm.ppf(0.975))
    for column in columns:
        raise_if_cancelled()
        marker = _impute(cohort.markers[:, [column]], full.medians[[column]])
        entry: dict[str, Any] = {}
        for label, exog in (("marginal", marker), ("adjusted", None if clinical_design is None else np.column_stack([clinical_design, marker]))):
            if exog is None:
                continue
            results, converged = fit_phreg(PHReg(cohort.time, exog, status=cohort.event, strata=strata, ties="efron"))
            beta = float(np.asarray(results.params)[-1])
            se = float(np.asarray(results.bse)[-1])
            if not converged or not np.isfinite(beta) or not np.isfinite(se):
                entry[label] = None
                continue
            entry[label] = {
                "hazard_ratio": float(np.exp(beta)),
                "ci_lower": float(np.exp(beta - z_value * se)),
                "ci_upper": float(np.exp(beta + z_value * se)),
                "wald_p": float(2.0 * stats.norm.sf(abs(beta / se))),
            }
            if label == "adjusted" and clinical_llf is not None:
                statistic = max(2.0 * (float(results.llf) - clinical_llf), 0.0)
                entry[label]["lr_statistic"] = statistic
                entry[label]["lr_p"] = float(stats.chi2.sf(statistic, df=1))
                if clinical_c is not None:
                    full_c = _pooled_c_index(cohort.time, cohort.event, exog @ np.asarray(results.params), strata)
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
      rejections, so they keep its family-wise error control.
    * suggestive: primary-lens evidence (Westfall–Young p <= alpha or permutation FDR
      q <= ``fdr_level``) that is not stable enough to be robust; the selection
      frequency and direction consistency columns show why.
    * marginal only: with clinical covariates, no added-value evidence but a marginal
      Westfall–Young p <= alpha (for example a marker that tracks a prognostic
      clinical covariate).
    """
    n_markers = lenses[primary].chi2.shape[0]
    p_fwer = np.nan_to_num(np.asarray(adjusted[primary]["p_fwer"], dtype=float), nan=1.0)
    q_perm = np.nan_to_num(np.asarray(adjusted[primary]["q_perm"], dtype=float), nan=1.0)
    frequency = np.nan_to_num(resampling.selection_frequency[primary], nan=0.0)
    direction = np.nan_to_num(resampling.direction_consistency[primary], nan=0.0)
    no_resampling = resampling.n_valid == 0
    tiers: list[str] = []
    patterns: list[str] = []
    for index in range(n_markers):
        stable = no_resampling or (frequency[index] >= settings.robust_frequency and direction[index] >= settings.robust_direction)
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
) -> dict[str, Any]:
    """Screen candidate markers and check the whole screening procedure (see module docstring)."""
    settings = _validated_settings(settings or MarkerSettings())
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
    )
    rng = np.random.default_rng(int(settings.random_seed))
    all_rows = np.arange(cohort.time.shape[0])
    full = run_procedure(cohort, all_rows, settings)
    primary = "added_value" if "added_value" in full.lenses and cohort.clinical is not None else "marginal"
    adjusted = permutation_null(cohort, full, settings, rng)
    resampling = resample_procedure(cohort, full, settings, rng, primary)
    tiers, patterns = assign_tiers(primary, adjusted, resampling, full.lenses, settings)

    primary_chi2 = np.nan_to_num(full.lenses[primary].chi2, nan=-np.inf)
    by_strength = np.argsort(-primary_chi2, kind="mergesort")
    shortlist = [int(index) for index in by_strength[: settings.shortlist_size]]
    shortlist += [index for index, tier in enumerate(tiers) if tier != "not supported" and index not in shortlist]
    exact = _exact_fits(cohort, full, np.asarray(shortlist, dtype=np.int64))
    signature = _fit_signature(cohort, all_rows, full, primary, settings.max_signature_markers)
    apparent_c = None
    if signature is not None:
        apparent_c = _pooled_c_index(cohort.time, cohort.event, _signature_risk(cohort, all_rows, signature), cohort.strata)

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
        row["exact"] = exact.get(index)
        rows.append(row)
    tier_rank = {tier: position for position, tier in enumerate(TIER_ORDER)}
    rows.sort(
        key=lambda item: (
            tier_rank[item["tier"]],
            -item[primary]["chi2"] if item[primary]["chi2"] is not None else np.inf,
        )
    )

    return {
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
            "strata_columns": cohort.strata_columns,
            "row_mask_hash": cohort.row_mask_hash,
        },
        "null": {
            "n_permutations": int(adjusted[primary]["n_permutations"]),
            "lens2_null": settings.lens2_null if primary == "added_value" else None,
        },
        "resampling": {
            "scheme": "event-stratified subsampling without replacement",
            "fraction": float(settings.resample_fraction),
            "n_valid": resampling.n_valid,
            "n_failed": resampling.n_failed,
        },
        "signature": {
            "markers": [] if signature is None else [cohort.marker_names[int(column)] for column in signature.columns],
            "apparent_c": apparent_c,
            **resampling.optimism,
        },
        "settings": settings._asdict(),
    }

"""Reporting-guideline checklists for SurvStudio runs.

``remark_checklist`` covers a marker evaluation (REMARK; McShane et al., J Natl Cancer Inst 2005, and the
explanation by Altman et al., PLoS Med 2012). ``tripod_ai_checklist`` covers a comparison of prediction
models (TRIPOD+AI; Collins et al., BMJ 2024). Each item is "reported" when the run supplies its text,
"partly" when the run supplies part of it, and "author" when only the authors can write it (study design,
specimens, interpretation). ``checklist_markdown`` renders a checklist with its methods and results
paragraphs for a manuscript supplement; the authors complete the rest.

The paragraphs describe what the run did, not what its settings asked for: a run without permutations,
without usable subsamples or without an optimism correction says so.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from survival_toolkit import __version__
from survival_toolkit.duplicates import MAX_PATIENTS, MIN_GAP, MIN_IDENTICAL_MARKERS, MIN_MARKERS, MIN_R

STATUS_LABELS = {"reported": "Filled in by SurvStudio", "partly": "Partly filled in", "author": "Authors to complete"}
CHECKLIST_COLUMNS = ("Item", "Section", "Topic", "Status", "Text")


def _item(number: str, section: str, topic: str, status: str, text: str) -> dict[str, str]:
    return {"item": number, "section": section, "topic": topic, "status": status, "text": text}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interval(value: Any) -> tuple[float, float] | None:
    """A [low, high] pair of finite numbers, or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = _finite(value[0]), _finite(value[1])
    return None if low is None or high is None else (low, high)


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "NA"
    return str(value)


def _count(n: Any, singular: str, plural: str | None = None) -> str:
    """A count with its noun in the right number: 1 marker, 2 markers."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _names(values: Sequence[Any], limit: int = 12) -> str:
    names = [str(value) for value in values]
    if not names:
        return "none"
    shown = ", ".join(names[:limit])
    return f"{shown} and {len(names) - limit} more" if len(names) > limit else shown


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}".rstrip("0").rstrip(".") + "%"


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def _dataset_text(dataset: dict[str, Any] | None) -> str:
    if not dataset:
        return "The analysed table is identified in the export notes."
    parts = [f"Data file {dataset.get('filename') or 'uploaded table'}"]
    if dataset.get("n_rows") is not None:
        parts.append(f"{dataset['n_rows']} rows")
    if dataset.get("dataset_hash"):
        parts.append(f"fingerprint {dataset['dataset_hash']}")
    matrix = dataset.get("marker_matrix")
    if matrix:
        parts.append(
            f"markers from {matrix.get('filename')} ({matrix.get('n_markers')} markers, fingerprint {matrix.get('fingerprint')}), "
            f"matched to {matrix.get('n_matched')} patients by {matrix.get('id_column')}"
        )
    return "; ".join(parts) + "."


def _endpoint_clause(request: dict[str, Any]) -> str:
    return (
        f"time to event: {request.get('time_column')}; event: {request.get('event_column')} = "
        f"{request.get('event_positive_value')}, all other values censored"
    )


def _endpoint_text(request: dict[str, Any]) -> str:
    return _capitalized(_endpoint_clause(request)) + "."


# ── REMARK (marker evaluation) ───────────────────────────────────


def signature_is_clinical_only(signature: dict[str, Any] | None) -> bool:
    """Whether a marker evaluation's final model holds the clinical covariates only (no marker was selected)."""
    signature = signature or {}
    if signature.get("clinical_only") is not None:
        return bool(signature["clinical_only"])
    return signature.get("apparent_c") is not None and not signature.get("markers")


def signature_fit_failed(result: dict[str, Any]) -> bool:
    """Whether a marker evaluation's final model could not be fitted in the full cohort: it has no apparent C-index
    although there was a model to fit, the clinical covariates (added value) or markers the screen selected (a
    Benjamini-Hochberg q-value at most alpha on the primary lens)."""
    signature = result.get("signature") or {}
    if signature.get("apparent_c") is not None or signature_is_clinical_only(signature) or signature.get("markers"):
        return False
    if result.get("primary_lens") == "added_value":
        return True
    primary = str(result.get("primary_lens") or "marginal")
    alpha = _finite((result.get("settings") or {}).get("alpha"))
    return alpha is not None and any(
        (q_value := _finite(row[primary].get("q_bh"))) is not None and q_value <= alpha
        for row in result.get("marker_table") or []
        if isinstance(row.get(primary), dict)
    )


def _n_permutations(result: dict[str, Any]) -> int:
    return int((result.get("null") or {}).get("n_permutations") or 0)


def _resample_counts(result: dict[str, Any]) -> tuple[int, int]:
    resampling = result.get("resampling") or {}
    return int(resampling.get("n_valid") or 0), int(resampling.get("n_failed") or 0)


def _stability_assessed(result: dict[str, Any]) -> bool:
    flag = (result.get("resampling") or {}).get("stability_assessed")
    if flag is not None:
        return bool(flag)
    return _resample_counts(result)[0] > 0


def _optimism_missing(result: dict[str, Any]) -> bool:
    signature = result.get("signature") or {}
    return signature.get("apparent_c") is not None and signature.get("optimism_corrected_c") is None


def _exact_fit_count(result: dict[str, Any], lens: str) -> int:
    return sum(1 for row in result.get("marker_table", []) if ((row.get("exact") or {}).get(lens)))


def _permutation_sentence(result: dict[str, Any], added_value: bool) -> str:
    n_permutations = _n_permutations(result)
    if n_permutations <= 0:
        return (
            "No permutations were run, so neither Westfall-Young family-wise p-values nor permutation false discovery rates "
            "were computed."
        )
    scheme = str((result.get("null") or {}).get("lens2_null") or "")
    if added_value and scheme in {"smith", "freedman_lane"}:
        clause = (
            "; for added value, the residuals of each marker after regression on the clinical covariates were permuted "
            "(Smith method; Winkler et al. 2014), approximating a conditional null under exchangeable residuals after linear adjustment"
        )
    elif added_value and scheme == "raw":
        clause = "; for added value the marker values themselves were permuted"
    else:
        clause = ""
    return (
        f"Multiplicity was assessed with Westfall-Young step-down max-T p-values from {_count(n_permutations, 'permutation')}"
        + clause
        + ", and the false discovery rate was estimated from the same permutations. "
        "Permutation inference requires exchangeability under the chosen null; strong family-wise error control also requires subset pivotality. "
        "Nonlinear marker-covariate relations can invalidate the residual-permutation calibration."
    )


def _resampling_sentence(result: dict[str, Any]) -> str:
    n_valid, n_failed = _resample_counts(result)
    fraction = _percent((result.get("resampling") or {}).get("fraction", 0.632))
    alpha = (result.get("settings") or {}).get("alpha")
    if n_valid > 0:
        text = f"The whole screening procedure was repeated on {_count(n_valid, 'event-stratified subsample')} of {fraction} of the patients"
        if n_failed:
            text += f" ({n_failed} more failed and {'was' if n_failed == 1 else 'were'} left out)"
        return text + (
            f"; in each, a marker counted as selected when its Benjamini-Hochberg q-value was at most {alpha}, "
            "and its rank and direction were recorded."
        )
    if n_failed:
        return (
            f"The whole screening procedure was run on {_count(n_failed, 'event-stratified subsample')} of {fraction} of the "
            f"patients, but {'it' if n_failed == 1 else 'every one'} failed, so the stability of the selection was not assessed."
        )
    return "The screening procedure was not repeated on subsamples, so the stability of the selection was not assessed."


def _tier_sentence(result: dict[str, Any], added_value: bool) -> str:
    settings = result.get("settings") or {}
    alpha = settings.get("alpha")
    if _n_permutations(result) <= 0:
        return (
            "Without family-wise p-values or permutation q-values no marker could be called robust"
            + (", suggestive or marginal only." if added_value else " or suggestive.")
        )
    marginal = (
        f", and markers without sufficient evidence of added value but with a marginal family-wise p-value at most {alpha} were called marginal only."
        if added_value
        else "."
    )
    evidence = f"a family-wise p-value at most {alpha} or a permutation q-value at most {settings.get('fdr_level')}"
    if _stability_assessed(result):
        return (
            f"A marker was robust when its family-wise p-value was at most {alpha}, it was selected in at least "
            f"{_percent(settings.get('robust_frequency', 0.5))} of the subsamples and its direction held in at least "
            f"{_percent(settings.get('robust_direction', 0.9))} of them; markers with {evidence} that did not meet the "
            "stability rule were called suggestive" + marginal
        )
    # The current engine never calls a marker robust without subsamples, but 0.2.0 builds before 28 September 2026
    # did (on its family-wise p-value alone), and this text still has to describe their results.
    if int((result.get("tier_counts") or {}).get("robust", 0) or 0) > 0:
        return (
            "Without subsamples the stability rule could not be applied, so a marker was called robust on its family-wise "
            f"p-value alone (at most {alpha}); markers with only a permutation q-value at most {settings.get('fdr_level')} "
            "were called suggestive" + marginal
        )
    return (
        "Without subsamples the stability rule could not be applied, so no marker could be called robust; markers with "
        f"{evidence} were called suggestive" + marginal
    )


def _identical_check_ran(duplicates: dict[str, Any], n_markers: int) -> bool | None:
    """Whether the identical-profile check ran: True, False, or None when the result does not say."""
    if duplicates.get("identical_checked") is not None:
        return bool(duplicates["identical_checked"])
    if int(duplicates.get("n_identical") or 0) > 0:
        return True
    if n_markers < MIN_IDENTICAL_MARKERS:
        return False
    return None


def _near_identical_limit(duplicates: dict[str, Any], n_markers: int) -> str:
    """Why near-identical profiles were not compared: the panel had too few markers, or the cohort too many patients
    (the screen's note says which; a result without it, by the panel size)."""
    note = str(duplicates.get("note") or "")
    capped = f"{MAX_PATIENTS:,} patients" in note if note else n_markers >= MIN_MARKERS
    if capped:
        return f"near-identical profiles are compared only in cohorts of up to {MAX_PATIENTS:,} patients"
    return f"near-identical profiles are checked only on panels of at least {MIN_MARKERS} markers"


def _duplicate_screen_sentence(result: dict[str, Any]) -> str:
    duplicates = result.get("duplicates") or {}
    n_markers = int((result.get("cohort") or {}).get("n_markers_evaluated") or 0)
    near = bool(duplicates.get("checked"))
    identical = _identical_check_ran(duplicates, n_markers)
    conditional = (
        "when the markers took many distinct values (on panels with few values, such as mutation calls, patients share "
        "profiles by chance)"
    )
    if near:
        text = (
            f"Patients were screened for repeated samples: over the {duplicates.get('markers_used')} most variable markers, a pair "
            f"whose profiles were each other's best match, correlated at least {MIN_R:g} and stood {MIN_GAP:g} above either "
            "patient's next-best match was flagged"
        )
        if identical is True:
            return text + ", as were patients with identical values on every marker."
        if identical is None:
            return text + f"; patients with identical values on every marker were also flagged {conditional}."
        return text + "."
    if identical is True:
        return (
            "Patients with identical values on every marker were flagged as possible repeated samples; "
            f"{_near_identical_limit(duplicates, n_markers)}."
        )
    if identical is None:
        return (
            f"Patients with identical values on every marker were flagged as possible repeated samples {conditional}; "
            f"{_near_identical_limit(duplicates, n_markers)}."
        )
    return ""


def _duplicate_screen_ran(result: dict[str, Any]) -> bool:
    duplicates = result.get("duplicates") or {}
    n_markers = int((result.get("cohort") or {}).get("n_markers_evaluated") or 0)
    return bool(duplicates.get("checked")) or _identical_check_ran(duplicates, n_markers) is True


def _signature_methods_sentence(result: dict[str, Any], added_value: bool) -> str:
    settings = result.get("settings") or {}
    signature = result.get("signature") or {}
    if signature.get("apparent_c") is None:
        return ""
    correction = (
        "its apparent C-index was adjusted for the subsampling gap by subtracting the mean difference between the C-index of the whole "
        "procedure{scope} in each subsample and in the patients left out of it; this heuristic includes training-size effects"
    )
    n_valid, n_failed = _resample_counts(result)
    if n_valid > 0:
        missing_reason = "because no subsample gave a model that could be scored in the patients left out"
    elif n_failed > 0:
        missing_reason = "because every subsample failed"
    else:
        missing_reason = "because no subsample was available"
    corrected = signature.get("optimism_corrected_c") is not None
    if signature_is_clinical_only(signature):
        text = "No marker was selected, so the final Cox model held the clinical covariates only"
        if corrected:
            return text + "; " + correction.format(scope=" (which could select markers)") + "."
        return text + f"; its apparent C-index could not be adjusted for the subsampling gap {missing_reason}."
    text = (
        ("A Cox model with the clinical covariates and " if added_value else "A Cox model with ")
        + f"the selected markers (at most the {settings.get('max_signature_markers')} strongest) was fitted"
    )
    if corrected:
        return text + ", and " + correction.format(scope="") + "."
    return text + f"; its apparent C-index could not be adjusted for the subsampling gap {missing_reason}."


def _gain_interval(signature: dict[str, Any]) -> tuple[float, float] | None:
    """The 95% interval of the paired left-out gain over the clinical covariates, when the result has one."""
    if _finite(signature.get("signature_gain_left_out")) is None:
        return None
    return _interval(signature.get("signature_gain_left_out_ci"))


def _gain_interval_sentence(result: dict[str, Any], added_value: bool) -> str:
    if not added_value or _gain_interval(result.get("signature") or {}) is None:
        return ""
    return (
        "In the patients left out of each subsample, the C-index of the model fitted in that subsample was compared with that "
        "of a Cox model of the clinical covariates alone fitted in the same subsample; the 95% confidence interval of the mean "
        "difference used the approximate corrected resampled t statistic (Nadeau and Bengio 2003), which accounts for "
        "overlap between subsamples under its covariance assumptions. Coverage for this survival selection procedure "
        "has not been established."
    )


def marker_methods_paragraph(result: dict[str, Any], request: dict[str, Any] | None = None) -> str:
    settings = result.get("settings") or {}
    cohort = result.get("cohort") or {}
    clinical = cohort.get("clinical_columns") or []
    strata = cohort.get("strata_columns") or []
    added_value = result.get("primary_lens") == "added_value"
    endpoint = f"{request.get('time_column')} / {request.get('event_column')}" if request else "the survival endpoint"
    sentences = [
        f"Candidate markers (n = {cohort.get('n_markers_evaluated')}) were screened for association with {endpoint} in "
        f"{cohort.get('n')} patients ({cohort.get('events')} events) with SurvStudio {__version__}.",
        (
            f"Each marker was tested for added value over the clinical covariates ({_names(clinical)}) with a Cox score test "
            "against the model with the clinical covariates alone"
            if added_value
            else "Each marker was tested with an unadjusted Cox score test"
        )
        + (f", stratified by {_names(strata)}" if strata else "")
        + f", using the {str(settings.get('ties', 'efron')).capitalize()} method for tied event times.",
        _permutation_sentence(result, added_value),
        _resampling_sentence(result),
        _tier_sentence(result, added_value),
        f"Markers with more than {_percent(settings.get('max_missing_fraction', 0.2))} missing values, a constant value or more than "
        f"{_percent(settings.get('max_mode_fraction', 0.9))} of patients at one value were excluded before testing (a filter blind to the outcome); "
        "other missing marker values were replaced by the marker's median among the patients in each fit, and patients with a missing "
        "or invalid outcome, clinical covariate or stratum were excluded.",
        "Markers were analysed as continuous variables without cut-points.",
        _duplicate_screen_sentence(result),
        _signature_methods_sentence(result, added_value),
        _gain_interval_sentence(result, added_value),
    ]
    return " ".join(sentence for sentence in sentences if sentence)


# When the left-out C-indices of the model and of the clinical covariates alone are paired subsample by
# subsample, a result may carry the number of pairs and the mean paired difference; other results carry only
# the two means and the number of model replicates.
_PAIRED_COUNT_KEYS = ("n_clinical_replicates", "n_paired_replicates", "n_left_out_pairs", "n_clinical_pairs")
_PAIRED_DIFFERENCE_KEYS = ("signature_gain_left_out", "delta_c_left_out", "c_left_out_difference", "left_out_c_difference")


def _left_out_comparison(signature: dict[str, Any], added_value: bool) -> str:
    """Selection and refitting against the clinical covariates alone, in the patients left out."""
    model_c, clinical_c = _finite(signature.get("signature_c_left_out")), _finite(signature.get("clinical_c_left_out"))
    if not added_value or model_c is None or clinical_c is None:
        return ""
    difference = next(
        (value for key in _PAIRED_DIFFERENCE_KEYS if (value := _finite(signature.get(key))) is not None),
        model_c - clinical_c,
    )
    replicates = next(
        (int(signature[key]) for key in (*_PAIRED_COUNT_KEYS, "n_signature_replicates") if signature.get(key) is not None),
        None,
    )
    if replicates is None:
        where = "each subsample"
    elif replicates == 1:
        where = "the one subsample that could be scored"
    else:
        where = f"each of {replicates} subsamples"
    interval = _gain_interval(signature)
    spread = f", 95% CI {interval[0]:.3f} to {interval[1]:.3f}" if interval else ""
    return (
        f" In the patients left out of {where}, the whole selection procedure reached a mean C-index of {_number(model_c)} against {_number(clinical_c)} "
        f"for the clinical covariates alone (mean difference {difference:+.3f}{spread})."
        " Markers were selected and models refitted in each subsample; validate the locked final model independently."
    )


def marker_results_paragraph(result: dict[str, Any]) -> str:
    counts = result.get("tier_counts") or {}
    cohort = result.get("cohort") or {}
    signature = result.get("signature") or {}
    added_value = result.get("primary_lens") == "added_value"
    robust = int(counts.get("robust", 0) or 0)
    text = (
        f"Of {_count(cohort.get('n_markers_evaluated'), 'marker')}, {robust} {'was' if robust == 1 else 'were'} robust"
        + (f", {counts.get('suggestive', 0)} suggestive and {counts.get('marginal only', 0)} marginal only" if added_value else f" and {counts.get('suggestive', 0)} suggestive")
        + "."
    )
    if _n_permutations(result) <= 0:
        text += " No permutations were run, so no family-wise p-values or permutation q-values were available."
    elif not _stability_assessed(result):
        # Robust markers without subsamples come only from 0.2.0 builds before 28 September 2026 (see _tier_sentence).
        text += (
            " Stability over subsamples was not assessed, so the robust markers rest on their family-wise p-value alone."
            if robust
            else " Stability over subsamples was not assessed."
        )
    apparent, corrected = signature.get("apparent_c"), signature.get("optimism_corrected_c")
    clinical_only = signature_is_clinical_only(signature)
    if apparent is not None:
        if clinical_only:
            text += f" No marker was selected, so the final model held the clinical covariates only; its apparent C-index was {_number(apparent)}"
            text += f" and its subsample gap-adjusted C-index {_number(corrected)}." if corrected is not None else " (not adjusted for the subsampling gap)."
        else:
            text += f" The selected-marker model ({_names(signature.get('markers') or [])}) had an apparent C-index of {_number(apparent)}"
            text += f" and a subsample gap-adjusted C-index of {_number(corrected)}." if corrected is not None else " (not adjusted for the subsampling gap)."
    elif signature_fit_failed(result):
        text += " The final model could not be fitted in the full cohort, so it has no apparent or subsample gap-adjusted C-index."
    text += _left_out_comparison(signature, added_value)
    duplicates = result.get("duplicates") or {}
    n_pairs, n_identical = int(duplicates.get("n_pairs") or 0), int(duplicates.get("n_identical") or 0)
    if n_pairs or n_identical:
        flagged = []
        if n_pairs:
            flagged.append(f"{_count(n_pairs, 'pair')} of patients with near-identical profiles")
        if n_identical:
            flagged.append(f"{_count(n_identical, 'group')} of patients with identical values")
        text += f" The screen for repeated samples flagged {' and '.join(flagged)}."
    elif _duplicate_screen_ran(result):
        text += " The screen for repeated samples flagged no patients."
    return text


def _internal_validation_text(result: dict[str, Any], added_value: bool) -> str:
    signature = result.get("signature") or {}
    n_valid, n_failed = _resample_counts(result)
    closing = " Check proportional hazards for the reported markers in the Cox model tab."
    if n_valid <= 0:
        if not n_failed:
            reason = "no subsamples were drawn"
        elif n_failed == 1:
            reason = "the only subsample failed"
        else:
            reason = f"all {n_failed} subsamples failed"
        return (
            f"No internal validation was done: {reason}, so neither the stability of the selection nor the optimism of the "
            "model's C-index was assessed." + closing
        )
    clinical_only = signature_is_clinical_only(signature)
    model = "the clinical model's" if clinical_only else "the selected-marker model's"
    text = f"Internal validation: the whole procedure was repeated on {_count(n_valid, 'subsample')}"
    if n_failed:
        text += f" ({n_failed} more failed)"
    apparent, corrected = signature.get("apparent_c"), signature.get("optimism_corrected_c")
    if apparent is not None and corrected is not None:
        text += (
            f"; {model} apparent C-index was {_number(apparent)} and its heuristic subsample gap-adjusted "
            f"C-index was {_number(corrected)}"
        )
    elif apparent is not None:
        text += f"; {model} apparent C-index ({_number(apparent)}) could not be adjusted for the subsampling gap"
    elif signature_fit_failed(result):
        text += "; the final model could not be fitted in the full cohort"
    shrinkage = _finite(signature.get("top_marker_shrinkage"))
    if shrinkage is not None:
        # The engine follows each subsample's top marker by score statistic, whether or not its q-value selected it.
        text += (
            "; in the patients left out, the log hazard ratio of each subsample's strongest marker (the one with the largest "
            f"score statistic, whether or not it was selected) was on average {_percent(shrinkage)} of its value in the subsample"
        )
    text += "." + _left_out_comparison(signature, added_value)
    return text + closing


def _patient_flow_text(cohort: dict[str, Any], dataset: dict[str, Any] | None) -> str:
    """Patients analysed, with the rows that had no marker values kept apart from rows excluded for missing data."""
    events = int(cohort.get("events") or 0)
    text = f"{cohort.get('n')} patients with {events} events were analysed"
    if not dataset or dataset.get("n_rows") is None or cohort.get("n") is None:
        return text + "."
    n_rows, n = int(dataset["n_rows"]), int(cohort["n"])
    reason = "for a missing or invalid outcome, clinical covariate or stratum"
    matrix = dataset.get("marker_matrix") or {}
    if matrix.get("n_matched") is not None:
        n_matched = int(matrix["n_matched"])
        unmatched, excluded = n_rows - n_matched, n_matched - n
        parts = [
            f"{unmatched} of {n_rows} rows had no values in the marker matrix" if unmatched else f"all {n_rows} rows matched the marker matrix"
        ]
        if excluded:
            parts.append(f"{excluded} of the {n_matched} matched patients {'was' if excluded == 1 else 'were'} excluded {reason}")
        else:
            parts.append("no matched patient was excluded")
        return text + "; " + " and ".join(parts) + "."
    excluded = n_rows - n
    if excluded:
        return text + f"; {excluded} of {n_rows} rows {'was' if excluded == 1 else 'were'} excluded {reason}."
    return text + "; no rows were excluded."


def remark_checklist(result: dict[str, Any], *, request: dict[str, Any] | None = None, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    """REMARK checklist for one ``evaluate_markers`` result and the request that produced it."""
    request = request or {}
    cohort = result.get("cohort") or {}
    settings = result.get("settings") or {}
    counts = result.get("tier_counts") or {}
    markers_evaluated = int(cohort.get("n_markers_evaluated") or 0)
    events = int(cohort.get("events") or 0)
    dropped = cohort.get("dropped_markers") or []
    added_value = result.get("primary_lens") == "added_value"
    n_adjusted = _exact_fit_count(result, "adjusted")
    n_unadjusted = _exact_fit_count(result, "marginal")
    # Exact Cox fits are run for the shortlist (the strongest markers and every supported one); a fit whose
    # coefficient runs to infinity gives no estimate.
    n_shortlisted = sum(1 for row in result.get("marker_table", []) if row.get("exact"))

    def exact_markers(count: int) -> str:
        if count == markers_evaluated:
            return "the marker" if count == 1 else f"all {count} markers"
        missing = max(n_shortlisted - count, 0)
        without = f"{_count(missing, 'marker')} had no estimate" if missing else ""
        if n_shortlisted >= markers_evaluated:
            return f"{count} of the {markers_evaluated} markers" + (f" ({without})" if without else "")
        scope = f"the {settings.get('shortlist_size')} strongest and every supported one"
        return f"{_count(count, 'marker')} ({scope}" + (f"; {without})" if without else ")")

    matrix = (dataset or {}).get("marker_matrix")
    marker_text = (
        f"{matrix.get('n_markers')} markers from {matrix.get('filename')}"
        if matrix
        else _names(request.get("marker_columns") or [], limit=20)
    )
    methods = marker_methods_paragraph(result, request)
    # The methods describe what ran; a run without permutations, stable subsamples or an optimism
    # correction leaves part of the analysis for the authors to supply or justify.
    methods_status = (
        "partly"
        if _n_permutations(result) <= 0 or not _stability_assessed(result) or _optimism_missing(result)
        else "reported"
    )
    marginal_only = int(counts.get("marginal only", 0) or 0)
    if marginal_only == 1:
        marginal_text = (
            "1 marker was marginal only: associated with survival on its own but adding nothing beyond the clinical covariates. "
        )
    elif marginal_only:
        marginal_text = (
            f"{marginal_only} markers were marginal only: associated with survival on their own but adding nothing beyond the "
            "clinical covariates. "
        )
    else:
        marginal_text = (
            "No marker was marginal only (associated with survival on its own but adding nothing beyond the clinical covariates). "
        )
    items = [
        _item("1", "Introduction", "Markers, objectives and pre-specified hypotheses", "partly",
              f"Markers evaluated: {marker_text}. State the objectives and the hypotheses fixed before the analysis."),
        _item("2", "Patients", "Patient characteristics, source, inclusion and exclusion criteria", "partly",
              _dataset_text(dataset) + " Describe the source population and the eligibility criteria."),
        _item("3", "Patients", "Treatments received", "author", "Describe the treatments and how they were chosen."),
        _item("4", "Specimens", "Biological material and storage", "author", "Describe the specimens and how they were preserved and stored."),
        _item("5", "Assay methods", "Assay protocol, quality control, blinding", "author",
              "Describe the assay, its quality control and whether it was performed blind to the outcome."),
        _item("6", "Study design", "Case selection, time period, follow-up", "author",
              "State how cases were selected, the time period, the end of follow-up and the median follow-up."),
        _item("7", "Study design", "Clinical endpoints", "reported", _endpoint_text(request)),
        _item("8", "Study design", "Candidate variables", "reported",
              f"{_count(markers_evaluated, 'candidate marker')}; clinical covariates: {_names(cohort.get('clinical_columns') or [])}; strata: {_names(cohort.get('strata_columns') or [])}."
              + (
                  f" {_count(len(dropped), 'marker')} {'was' if len(dropped) == 1 else 'were'} excluded before the analysis ({_dropped_text(dropped)})."
                  if dropped
                  else ""
              )),
        _item("9", "Study design", "Sample size rationale", "partly",
              f"{cohort.get('n')} patients and {events} events for {_count(markers_evaluated, 'candidate marker')} "
              + (f"({events / max(markers_evaluated, 1):.1f} events per marker)" if events >= markers_evaluated else "(fewer events than markers, so the evaluation is a screen)")
              + ". Give the rationale for the sample size."),
        _item("10", "Statistical analysis", "Statistical methods, variable selection, assumptions, missing data", methods_status, methods),
        _item("11", "Statistical analysis", "Handling of marker values and cut-points", "reported",
              "Markers were analysed as continuous variables; the evaluation used no cut-points. Any cut-point used for figures should be fixed before looking at the outcome."),
        _item("12", "Results", "Patient flow, numbers analysed and events", "reported", _patient_flow_text(cohort, dataset)),
        _item("13", "Results", "Distribution of demographics, prognostic variables and the markers", "author",
              "Report age, sex, the standard prognostic variables and the markers, with numbers of missing values; the Table 1 tab builds this table."),
        _item("14", "Results", "Relation of the markers to standard prognostic variables", "partly" if added_value else "author",
              marginal_text + "Show how the reported markers relate to the standard prognostic variables."
              if added_value else "Show how the markers relate to the standard prognostic variables."),
        # With clinical covariates the table's interval columns belong to the adjusted hazard ratio, and the
        # unadjusted one (Unadjusted HR) is a point estimate; without them the table's HR and interval are unadjusted.
        _item("15", "Results", "Univariable analyses", "reported",
              "The marker table gives every marker's unadjusted score-test p-value"
              + (
                  f" (Unadjusted P) and, for {exact_markers(n_unadjusted)}, the hazard ratio from an unadjusted Cox model as a point "
                  "estimate (Unadjusted HR); the table's confidence intervals belong to the adjusted hazard ratios."
                  if added_value
                  else f" and, for {exact_markers(n_unadjusted)}, the hazard ratio with a 95% confidence interval from an unadjusted Cox model."
              )
              + " Kaplan-Meier curves by marker level can be drawn in the Survival curves tab with a cut-point fixed in advance."),
        _item("16", "Results", "Multivariable analyses with confidence intervals", "partly",
              f"The marker table gives, for {exact_markers(n_adjusted)}, hazard ratios with 95% Wald confidence intervals from Cox models "
              "with the clinical covariates. Report the final model with all its variables from the Cox model tab."
              if added_value else "Without clinical covariates only unadjusted hazard ratios are available; report a model with the standard prognostic variables."),
        _item("17", "Results", "Marker effects adjusted for standard prognostic variables regardless of significance", "reported" if added_value else "author",
              f"Adjusted hazard ratios with confidence intervals are given for {exact_markers(n_adjusted)}, whether or not they are significant."
              if added_value else "Report the marker effects adjusted for the standard prognostic variables."),
        _item("18", "Results", "Further investigations: assumptions, sensitivity, internal validation", "partly",
              _internal_validation_text(result, added_value)),
        _item("19", "Discussion", "Interpretation and limitations", "author",
              marker_results_paragraph(result) + " Interpret these results against the pre-specified hypotheses and discuss the limitations."),
        _item("20", "Discussion", "Implications for future research and clinical value", "author",
              "Discuss the implications, including validation in an independent cohort."),
    ]
    return {
        "guideline": "REMARK",
        "reference": "McShane LM et al. J Natl Cancer Inst 2005;97:1180-4; Altman DG et al. PLoS Med 2012;9:e1001216",
        "software": f"SurvStudio {__version__}",
        "methods": methods,
        "results": marker_results_paragraph(result),
        "items": items,
    }


# ── TRIPOD+AI (prediction models) ────────────────────────────────


_FAMILY_LABELS = {"ml": "classical machine-learning models", "dl": "deep-learning models"}
# Models whose whole encoded design is standardised with the training partition's moments.
_STANDARDISED_ML_MODELS = ("Cox PH", "LASSO-Cox")


def _family_label(result: dict[str, Any]) -> str:
    return _FAMILY_LABELS.get(str(result.get("family")), "models")


def _by_family(results: Sequence[dict[str, Any]], fragment: Callable[[dict[str, Any]], str]) -> str:
    """One clause when every comparison gives the same text, else one clause per model family."""
    texts = [fragment(result) for result in results]
    if not texts:
        return ""
    if len(set(texts)) == 1:
        return texts[0]
    return "; ".join(f"for the {_family_label(result)}, {text}" for result, text in zip(results, texts))


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in result.get("comparison_table") or [] if isinstance(row, dict)]


def _layers_text(layers: Any) -> str:
    sizes = [str(size) for size in layers or []]
    if not sizes:
        return "the default hidden layers"
    if len(sizes) == 1:
        return f"one hidden layer of {sizes[0]} units"
    return f"{len(sizes)} hidden layers ({', '.join(sizes[:-1])} and {sizes[-1]} units)"


def _fitted_models(result: dict[str, Any]) -> set[str]:
    """Models fitted at least once; a model whose every fold failed, or that only has an error, never ran."""
    return {
        str(row.get("model"))
        for row in _rows(result)
        if (n_evaluations := _n_evaluations(result, row)) is None or n_evaluations > 0
    }


def _hyperparameter_text(result: dict[str, Any]) -> str:
    request = result.get("request_config") or {}
    if not request:
        return ""
    if result.get("family") == "dl":
        text = (
            f"Neural networks were trained for up to {request.get('epochs')} epochs with learning rate {request.get('learning_rate')}, "
            f"{_layers_text(request.get('hidden_layers'))}, dropout {request.get('dropout')} and batch size {request.get('batch_size')}"
        )
        if request.get("early_stopping_patience"):
            # The deep-model trainers then refit each network on the whole training partition, monitoring rows included
            # (refit_on_training_partition).
            text += (
                f", stopping early after {request.get('early_stopping_patience')} epochs without improvement on a monitoring subset "
                "drawn from each training partition; each network was then refit on the whole training partition for the number of "
                "epochs early stopping selected"
            )
        return text + "."
    # Only the models that ran: without scikit-survival, for example, Cox PH is the only one.
    from survival_toolkit.ml_models import _GBS_DEFAULT_MAX_DEPTH

    fitted = _fitted_models(result)
    trees, depth = request.get("n_estimators"), request.get("max_depth")
    parts = []
    if "Random Survival Forest" in fitted:
        parts.append(f"random survival forests used {trees} trees " + ("without a depth limit" if depth is None else f"of maximum depth {depth}"))
    if "Gradient Boosted Survival" in fitted:
        # An empty depth gives boosting its shallow default, not fully grown trees.
        parts.append(
            f"gradient boosting used {trees} trees of maximum depth {_GBS_DEFAULT_MAX_DEPTH if depth is None else depth} "
            f"with learning rate {request.get('learning_rate')}"
        )
    if "LASSO-Cox" in fitted:
        parts.append("the LASSO-Cox penalty was chosen by inner cross-validation, stratified by event status, within each training partition")
    return _capitalized("; ".join(parts)) + "." if parts else ""


def _incomplete_row(row: dict[str, Any]) -> bool:
    return str(row.get("evaluation_mode") or "") == "repeated_cv_incomplete" or int(row.get("n_failures") or 0) > 0


def _fold_failure_text(result: dict[str, Any], row: dict[str, Any]) -> str:
    """How many cross-validation folds a model lost, for example "failed in 3 of 15 folds"."""
    n_failures = int(row.get("n_failures") or 0)
    fallbacks = int(row.get("n_apparent_fallbacks") or 0)
    total = _total_folds(result, row)
    if total is None:
        total = int(row.get("n_evaluations") or 0) + n_failures
    parts = []
    if n_failures - fallbacks > 0:
        parts.append(f"failed in {n_failures - fallbacks}")
    if fallbacks > 0:
        parts.append(f"fell back to apparent evaluation in {fallbacks}")
    if not parts:
        return f"was scored in {int(row.get('n_evaluations') or 0)} of {total} folds"
    return " and ".join(parts) + f" of {total} folds"


def _unranked_reason(result: dict[str, Any], row: dict[str, Any]) -> str | None:
    """None when the row can be ranked by its C-index, else why it cannot.

    Rows left unranked because they lost cross-validation folds or have no C-index are also marked
    not comparable, so those reasons are checked before the apparent-evaluation one.
    """
    if str(result.get("evaluation_mode", "")).startswith("repeated_cv") and _incomplete_row(row):
        return _fold_failure_text(result, row)
    if _finite(row.get("c_index")) is None:
        return "no C-index"
    if row.get("comparable_for_ranking") is False:
        return "apparent evaluation on the patients used for fitting, not comparable with the others"
    return None


def _holdout_text(result: dict[str, Any]) -> str:
    if result.get("n_evaluation_patients") is not None:
        return (
            f"a holdout set of {result.get('n_evaluation_patients')} patients ({result.get('n_evaluation_events')} events), "
            f"stratified by event status, after fitting on {result.get('n_fit_patients')} patients"
        )
    return "a holdout set stratified by event status"


def _evaluation_text(result: dict[str, Any]) -> str:
    mode = str(result.get("evaluation_mode", ""))
    if mode.startswith("repeated_cv"):
        repeats = result.get("cv_repeats")
        text = (
            f"{result.get('cv_folds')}-fold cross-validation stratified by event status"
            if repeats in (None, 1)
            else f"{repeats} repeats of {result.get('cv_folds')}-fold cross-validation stratified by event status"
        )
        if result.get("n_locked_test_patients"):
            text += (
                f" on a development set of {result.get('n_development_patients')} patients, then once on a locked test set of "
                f"{result.get('n_locked_test_patients')} patients ({result.get('n_locked_test_events')} events"
                + (f"; {_percent(result['locked_test_fraction'])} of the cohort" if result.get("locked_test_fraction") else "")
                + ") that was never used for fitting, preprocessing, tuning or model choice"
            )
        if mode == "repeated_cv_incomplete" or any(_incomplete_row(row) for row in _rows(result)):
            text += "; folds that failed or fell back to apparent evaluation were left out, and a model that lost folds was not ranked"
        return text
    if mode == "holdout":
        return _holdout_text(result)
    if mode == "mixed_holdout_apparent":
        holdout_models = [str(row.get("model")) for row in _rows(result) if str(row.get("evaluation_mode")) == "holdout"]
        apparent_models = [str(row.get("model")) for row in _rows(result) if str(row.get("evaluation_mode")) != "holdout"]
        return (
            f"{_holdout_text(result)} for {_names(holdout_models)}; {_names(apparent_models)} fell back to apparent performance "
            f"on the patients used for fitting, which is optimistic, and {'was' if len(apparent_models) == 1 else 'were'} not ranked"
        )
    return "scoring the models on the patients used for fitting (apparent performance, which is optimistic)"


def _apparent_models(result: dict[str, Any]) -> list[str]:
    """Models whose C-index is apparent performance on the patients used for fitting, as ``_evaluation_text`` reads
    the evaluation mode."""
    mode = str(result.get("evaluation_mode", ""))
    if mode.startswith("repeated_cv") or mode == "holdout":
        return []
    if mode == "mixed_holdout_apparent":
        return [str(row.get("model")) for row in _rows(result) if str(row.get("evaluation_mode")) != "holdout"]
    return [str(row.get("model")) for row in _rows(result)]


def _limitations_text(results: Sequence[dict[str, Any]], locked_text: str) -> str:
    """Where the performance comes from: internal validation, apparent performance, or both."""
    apparent = [name for result in results for name in _apparent_models(result)]
    models = [str(row.get("model")) for result in results for row in _rows(result)]
    if models and len(apparent) == len(models):
        return (
            "Performance is apparent: every model was scored on the patients used for fitting, which is optimistic. Internal "
            "validation (a holdout set or cross-validation) and an external cohort are needed to judge the models."
        )
    text = "Performance comes from internal validation in one data set" + locked_text
    if apparent:
        text += (
            f", except for {_names(apparent)}, which {'was' if len(apparent) == 1 else 'were'} scored on the patients used for "
            "fitting (apparent performance, which is optimistic)"
        )
    return text + "; an external cohort is needed to judge transportability."


def _evaluation_sentence(results: Sequence[dict[str, Any]]) -> str:
    texts = {_evaluation_text(result) for result in results}
    if len(texts) == 1:
        return f"Performance was estimated by {next(iter(texts))}."
    return " ".join(
        f"For the {_family_label(result)}, performance was estimated by {_evaluation_text(result)}." for result in results
    )


def _shared_splits(results: Sequence[dict[str, Any]]) -> bool | None:
    fingerprints = [result.get("evaluation_split_fingerprint") for result in results]
    if not all(fingerprints):
        return None
    return len(set(fingerprints)) == 1


def _preprocessing_text(results: Sequence[dict[str, Any]]) -> str:
    """What was fitted on each training partition, including the standardisation used by the Cox models and networks."""
    models = {str(row.get("model")) for result in results for row in _rows(result)}
    models |= {str(error.get("model")) for result in results for error in result.get("errors") or [] if isinstance(error, dict)}
    text = (
        "numeric predictors were imputed with the training median, categorical predictors were reference-coded (with a "
        "missing-value indicator only when the training data had missing values)"
    )
    scaled = []
    cox_models = [name for name in _STANDARDISED_ML_MODELS if name in models]
    if cox_models:
        scaled.append(
            f"for {' and '.join(cox_models)} every encoded column was standardised with the training mean and standard deviation"
        )
    if any(result.get("family") == "dl" for result in results):
        scaled.append("for neural networks numeric predictors were standardised with the training mean and standard deviation")
    return text + ("; " + "; ".join(scaled) if scaled else "")


def _ranking_exclusion_clause(results: Sequence[dict[str, Any]]) -> str:
    kinds = set()
    for result in results:
        for row in _rows(result):
            reason = _unranked_reason(result, row)
            if reason is None:
                continue
            if reason.startswith("apparent"):
                kinds.add("models that fell back to apparent evaluation")
            elif reason == "no C-index":
                kinds.add("models without a C-index")
            else:
                kinds.add("models that lost cross-validation folds")
    return ("; " + " and ".join(sorted(kinds)) + " were not ranked") if kinds else ""


def prediction_methods_paragraph(results: Sequence[dict[str, Any]]) -> str:
    if not results:
        return ""
    first = results[0]
    models = [str(row.get("model")) for result in results for row in _rows(result)]
    shared = _shared_splits(results)
    cohort = _by_family(results, lambda result: f"{result.get('n_patients')} patients ({result.get('n_events')} events)")
    if len({(result.get("n_patients"), result.get("n_events")) for result in results}) == 1:
        opening = f"Survival models ({_names(models, limit=20)}) were compared in {cohort} with SurvStudio {__version__}."
    else:
        opening = f"Survival models ({_names(models, limit=20)}) were compared with SurvStudio {__version__}: {cohort}."
    sentences = [opening, _evaluation_sentence(results)]
    if shared:
        sentences.append(
            f"All models were trained and scored on the same data partitions (split fingerprint {first.get('evaluation_split_fingerprint')})."
        )
    elif shared is False:
        sentences.append("The model families were scored on different data partitions, so their results are not directly comparable.")
    with_brier = [any(isinstance(row.get("ibs"), (int, float)) for row in _rows(result)) for result in results]
    brier = (
        "the integrated Brier score, weighted by the inverse probability of censoring, and the Brier skill score against a "
        "Kaplan-Meier model"
    )
    if all(with_brier):
        metrics = f"Discrimination was measured with Harrell's C-index and overall accuracy with {brier}"
    elif any(with_brier):
        brier_families = [_family_label(result) for result, has in zip(results, with_brier) if has]
        metrics = f"Discrimination was measured with Harrell's C-index, and overall accuracy of the {' and '.join(brier_families)} with {brier}"
    else:
        metrics = "Discrimination was measured with Harrell's C-index"
    sentences += [
        "Preprocessing was fitted within each training partition and applied unchanged to the evaluation patients: "
        + _preprocessing_text(results) + ".",
        metrics + "; models were ranked by C-index" + _ranking_exclusion_clause(results) + ".",
    ]
    sentences += [text for result in results if (text := _hyperparameter_text(result))]
    return " ".join(sentences)


def _ranked_rows(results: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows that can be ranked (best first), and "model (reason)" for the others."""
    ranked: list[dict[str, Any]] = []
    excluded: list[str] = []
    for result in results:
        for row in _rows(result):
            reason = _unranked_reason(result, row)
            if reason is None:
                ranked.append(row)
            else:
                excluded.append(f"{row.get('model')} ({reason})")
    ranked.sort(key=lambda row: -float(row["c_index"]))
    return ranked, excluded


def _performance_text(results: Sequence[dict[str, Any]]) -> str:
    ranked, excluded = _ranked_rows(results)
    if not ranked:
        text = "No model produced a C-index that could be ranked."
    else:
        best = ranked[0]
        text = f"The highest C-index was {_number(best.get('c_index'))} ({best.get('model')}"
        if isinstance(best.get("brier_skill_score"), (int, float)):
            text += f"; integrated Brier score {_number(best.get('ibs'))}, Brier skill score {_number(best.get('brier_skill_score'))}"
        text += ")"
        if isinstance(best.get("locked_test_c_index"), (int, float)):
            text += f"; on the locked test set its C-index was {_number(best.get('locked_test_c_index'))}"
        text += "."
        if len(ranked) > 1:
            text += f" Across the {len(ranked)} ranked models the C-index ranged from {_number(ranked[-1].get('c_index'))} to {_number(best.get('c_index'))}."
            if _shared_splits(results) is False:
                text += " The model families were scored on different partitions, so the ranking across families is not a fair comparison."
    if excluded:
        text += f" Not ranked: {'; '.join(excluded)}."
    return text


def _total_folds(result: dict[str, Any], row: dict[str, Any]) -> int | None:
    folds = row.get("cv_folds") or result.get("cv_folds")
    repeats = row.get("cv_repeats") or result.get("cv_repeats")
    return int(folds) * int(repeats) if folds and repeats else None


def _n_evaluations(result: dict[str, Any], row: dict[str, Any]) -> int | None:
    """Folds (or holdout fits) in which the model was scored; None for a holdout row, which is scored once."""
    if row.get("n_evaluations") is not None:
        return int(row["n_evaluations"])
    total = _total_folds(result, row)
    if str(result.get("evaluation_mode", "")).startswith("repeated_cv") and total is not None and row.get("n_failures") is not None:
        return max(total - int(row["n_failures"]), 0)
    return None


def _development_text(results: Sequence[dict[str, Any]]) -> str:
    """Models fitted at least once, the folds each lost, and the models that never fitted."""
    fitted: list[str] = []
    notes: list[str] = []
    never: list[str] = []
    for result in results:
        row_models = set()
        for row in _rows(result):
            name = str(row.get("model"))
            row_models.add(name)
            n_evaluations = _n_evaluations(result, row)
            lost_folds = n_evaluations is not None and int(row.get("n_failures") or 0) > 0
            if n_evaluations is not None and n_evaluations <= 0:
                total = _total_folds(result, row)
                if lost_folds and not int(row.get("n_apparent_fallbacks") or 0) and total is not None:
                    never.append(f"{name} (all {total} folds)")
                else:
                    never.append(f"{name} ({_fold_failure_text(result, row)})" if lost_folds else name)
                continue
            fitted.append(name)
            if lost_folds:
                notes.append(f"{name} {_fold_failure_text(result, row)}")
        for error in result.get("errors") or []:
            model = str(error.get("model") or "") if isinstance(error, dict) else ""
            if model and model not in row_models and model not in never:
                never.append(model)
    text = f"{_count(len(fitted), 'model')} {'was' if len(fitted) == 1 else 'were'} fitted"
    if notes:
        text += f" ({'; '.join(notes)})"
    if never:
        text += f"; fitting failed for {_names(never)}"
    return text + "."


def _locked(result: dict[str, Any]) -> bool:
    return bool(result.get("n_locked_test_patients"))


def tripod_ai_checklist(results: Sequence[dict[str, Any]], *, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    """TRIPOD+AI checklist for one or more model comparisons (classical ML and deep learning).

    Each result is a comparison ``analysis`` with its ``request_config`` and a ``family`` of "ml" or "dl".
    Participants, predictors, sample size and the locked test set are described per family when the
    comparisons differ.
    """
    results = [result for result in results if result]

    def features(result: dict[str, Any]) -> list[Any]:
        return list((result.get("request_config") or {}).get("features") or [])

    def resolved_categorical(result: dict[str, Any]) -> bool:
        return isinstance(result.get("categorical_features"), (list, tuple))

    def categorical(result: dict[str, Any]) -> list[Any]:
        # The encoder also reference-codes every predictor stored as text; a result that lists the categorical
        # predictors it resolved is quoted, otherwise the request names only the declared ones.
        if resolved_categorical(result):
            return list(result["categorical_features"])
        return list((result.get("request_config") or {}).get("categorical_features") or [])

    def participants(result: dict[str, Any]) -> str:
        text = f"{result.get('n_patients')} patients and {result.get('n_events')} events"
        if _locked(result):
            text += f"; development set {result.get('n_development_patients')} patients, locked test set {result.get('n_locked_test_patients')} patients"
        return text

    ranked, _ = _ranked_rows(results)
    n_ranked = len(ranked)
    locked_families = [_family_label(result) for result in results if _locked(result)]
    all_locked = bool(results) and len(locked_families) == len(results)
    if all_locked:
        locked_text = " with a locked test set"
    elif locked_families:
        locked_text = f" (with a locked test set for the {' and '.join(locked_families)})"
    else:
        locked_text = ""
    preprocessing = _preprocessing_text(results)
    items = [
        _item("1", "Title", "Development or evaluation, population, outcome", "author",
              "Identify the study as developing or evaluating a prediction model, with the target population and the outcome."),
        _item("2", "Abstract", "Structured summary", "author", "See the TRIPOD+AI for Abstracts checklist."),
        _item("3", "Introduction", "Background", "author", "Explain the healthcare context and the rationale for the model."),
        _item("4", "Introduction", "Objectives", "author", "State whether the study develops, evaluates or compares models."),
        _item("5", "Methods", "Data sources", "partly", _dataset_text(dataset) + " Describe the source, the setting and the dates of data collection."),
        _item("6", "Methods", "Participants", "partly",
              _capitalized(_by_family(results, lambda result: f"{result.get('n_patients')} patients with a usable outcome were analysed"))
              + ". Describe the eligibility criteria and the treatments."),
        _item("7", "Methods", "Data preparation", "reported",
              _capitalized(preprocessing) + "; every step was fitted on the training partition only."),
        _item("8", "Methods", "Outcome", "partly",
              _capitalized(_by_family(results, lambda result: _endpoint_clause(result.get("request_config") or {})))
              + ". State whether the outcome was assessed blind to the predictors."),
        _item("9", "Methods", "Predictors", "partly",
              _capitalized(_by_family(
                  results,
                  lambda result: f"{_count(len(features(result)), 'predictor')}: {_names(features(result), limit=30)}; categorical: {_names(categorical(result))}",
              ))
              + "."
              + ("" if all(resolved_categorical(result) for result in results) else " Any predictor stored as text was also reference-coded as a categorical predictor.")
              + " Describe how and when they were measured."),
        _item("10", "Methods", "Sample size", "partly",
              _capitalized(_by_family(
                  results,
                  lambda result: f"{result.get('n_patients')} patients and {result.get('n_events')} events for {_count(len(features(result)), 'candidate predictor')}",
              ))
              + ". Justify the sample size."),
        _item("11", "Methods", "Missing data", "reported",
              "Within each training partition, missing numeric values were imputed with the training median; missing categorical values "
              "had their own indicator when the training data contained any and were otherwise scored as the reference level. Patients "
              "with a missing or invalid outcome were excluded."),
        _item("12", "Methods", "Analytical methods", "reported", prediction_methods_paragraph(results)),
        _item("13", "Methods", "Class imbalance", "reported", "Not applicable to a time-to-event outcome; data splits were stratified by event status."),
        _item("14", "Methods", "Fairness", "author", "Describe any approach used to assess model fairness across subgroups."),
        _item("15", "Methods", "Model output", "reported",
              "Each model gives a risk score (higher values, higher risk); models that estimate survival curves also give survival probabilities over time."),
        _item("16", "Methods", "Training versus evaluation", "reported", _evaluation_sentence(results)),
        _item("17", "Methods", "Ethical approval", "author", "Name the ethics committee and the approval, or state why none was needed."),
        _item("18", "Open science", "Funding, conflicts, protocol, registration, data and code", "partly",
              f"Analyses were run in SurvStudio {__version__}; every model's settings are recorded in the exported request configuration. "
              "Report funding, conflicts of interest, the protocol, registration and data availability."),
        _item("19", "Patient and public involvement", "Involvement", "author", "Describe any patient and public involvement."),
        _item("20", "Results", "Participants", "partly",
              _capitalized(_by_family(results, participants)) + ". Summarise their characteristics with the Table 1 tab."),
        _item("21", "Results", "Model development", "reported", _development_text(results)),
        _item("22", "Results", "Model specification", "partly",
              "Tree ensembles and neural networks are specified by their settings rather than by an equation; share the settings "
              "(request configuration) so that the models can be refitted."),
        _item("23", "Results", "Model performance", "reported", _performance_text(results)),
        _item("24", "Results", "Model updating", "author", "Report any model updating, or state that none was done."),
        _item("25", "Discussion", "Interpretation", "author", "Give an overall interpretation, including fairness where relevant."),
        _item("26", "Discussion", "Limitations", "partly",
              _limitations_text(results, locked_text)
              + (
                  f" Choosing the best of {n_ranked} models on the same data makes its C-index optimistic (the winner's curse); "
                  + ("report the chosen model's locked-test C-index." if all_locked else "judge it on data that played no part in the choice.")
                  if n_ranked > 1
                  else ""
              )
              + " Discuss the other limitations."),
        _item("27", "Discussion", "Usability of the model in context", "author",
              "Describe how the model would be used, its intended users and the next steps towards implementation."),
    ]
    return {
        "guideline": "TRIPOD+AI",
        "reference": "Collins GS et al. BMJ 2024;385:e078378",
        "software": f"SurvStudio {__version__}",
        "methods": prediction_methods_paragraph(results),
        "results": _performance_text(results),
        "items": items,
    }


# ── Rendering ────────────────────────────────────────────────────


def _dropped_text(dropped: list[dict[str, Any]]) -> str:
    """Excluded markers by reason: every name for a few, counts per reason for a genome-wide panel."""
    if len(dropped) <= 20:
        return _names([entry.get("marker") for entry in dropped])
    reasons: dict[str, int] = {}
    for entry in dropped:
        reason = str(entry.get("reason") or "")
        key = "near-constant" if reason.startswith("near-constant") else "missing values" if reason.endswith("missing") else reason
        reasons[key] = reasons.get(key, 0) + 1
    return "; ".join(f"{count} {reason}" for reason, count in sorted(reasons.items(), key=lambda item: -item[1]))


def checklist_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "Item": entry["item"],
            "Section": entry["section"],
            "Topic": entry["topic"],
            "Status": STATUS_LABELS.get(entry["status"], entry["status"]),
            "Text": entry["text"],
        }
        for entry in report.get("items", [])
    ]


def checklist_intro(report: dict[str, Any]) -> str:
    return (
        f"Guideline: {report['guideline']} ({report['reference']}). Generated with {report['software']}. "
        "Items marked Authors to complete need information SurvStudio does not have; check every filled-in item against the study."
    )


def _markdown_text(value: Any) -> str:
    """Text safe to place in Markdown: column names cannot open raw HTML, and backslashes stay literal."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\\", "\\\\")
    )


def checklist_markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        # GFM removes one backslash before a pipe inside a table cell, so backslashes are doubled first
        # (by _markdown_text) and a pipe then becomes \| : "a\|b" renders as a\|b, "a|b" as a|b.
        return _markdown_text(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    lines = [
        f"# {_markdown_text(report['guideline'])} checklist",
        "",
        _markdown_text(checklist_intro(report)),
        "",
        "## Methods",
        "",
        _markdown_text(report.get("methods", "")),
        "",
        "## Results",
        "",
        _markdown_text(report.get("results", "")),
        "",
        "## Checklist",
        "",
        "| " + " | ".join(CHECKLIST_COLUMNS) + " |",
        "|" + "---|" * len(CHECKLIST_COLUMNS),
    ]
    lines += ["| " + " | ".join(cell(row[column]) for column in CHECKLIST_COLUMNS) + " |" for row in checklist_rows(report)]
    return "\n".join(lines) + "\n"

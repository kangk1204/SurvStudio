"""Reporting-guideline checklists for SurvStudio runs.

``remark_checklist`` covers a marker evaluation (REMARK; McShane et al., J Natl Cancer Inst 2005, and the
explanation by Altman et al., PLoS Med 2012). ``tripod_ai_checklist`` covers a comparison of prediction
models (TRIPOD+AI; Collins et al., BMJ 2024). Each item is "reported" when the run supplies its text,
"partly" when the run supplies part of it, and "author" when only the authors can write it (study design,
specimens, interpretation). ``checklist_markdown`` renders a checklist with its methods and results
paragraphs for a manuscript supplement; the authors complete the rest.
"""

from __future__ import annotations

from typing import Any, Sequence

from survival_toolkit import __version__

STATUS_LABELS = {"reported": "Filled in by SurvStudio", "partly": "Partly filled in", "author": "Authors to complete"}
CHECKLIST_COLUMNS = ("Item", "Section", "Topic", "Status", "Text")


def _item(number: str, section: str, topic: str, status: str, text: str) -> dict[str, str]:
    return {"item": number, "section": section, "topic": topic, "status": status, "text": text}


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _names(values: Sequence[Any], limit: int = 12) -> str:
    names = [str(value) for value in values]
    if not names:
        return "none"
    shown = ", ".join(names[:limit])
    return f"{shown} and {len(names) - limit} more" if len(names) > limit else shown


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}".rstrip("0").rstrip(".") + "%"


def _dataset_text(dataset: dict[str, Any] | None) -> str:
    if not dataset:
        return "The analysed table is identified in the export notes."
    parts = [f"Data file {dataset.get('filename') or 'uploaded table'}"]
    if dataset.get("n_rows") is not None:
        parts.append(f"{dataset['n_rows']} rows")
    if dataset.get("dataset_hash"):
        parts.append(f"fingerprint {dataset['dataset_hash']}")
    return "; ".join(parts) + "."


def _endpoint_text(request: dict[str, Any]) -> str:
    return (
        f"Time to event: {request.get('time_column')}; event: {request.get('event_column')} = "
        f"{request.get('event_positive_value')}, all other values censored."
    )


# ── REMARK (marker evaluation) ───────────────────────────────────


def _exact_fit_count(result: dict[str, Any], lens: str) -> int:
    return sum(1 for row in result.get("marker_table", []) if ((row.get("exact") or {}).get(lens)))


def marker_methods_paragraph(result: dict[str, Any], request: dict[str, Any] | None = None) -> str:
    settings = result.get("settings") or {}
    cohort = result.get("cohort") or {}
    signature = result.get("signature") or {}
    resampling = result.get("resampling") or {}
    null = result.get("null") or {}
    clinical = cohort.get("clinical_columns") or []
    strata = cohort.get("strata_columns") or []
    added_value = result.get("primary_lens") == "added_value"
    alpha = settings.get("alpha")
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
        f"Family-wise error was controlled with Westfall-Young step-down max-T p-values from {null.get('n_permutations')} permutations"
        + (
            "; for added value the marker residuals after regression on the clinical covariates were permuted (Freedman-Lane), "
            "which keeps each marker's relation to the covariates"
            if added_value and null.get("lens2_null") == "freedman_lane"
            else ""
        )
        + ", and the false discovery rate was estimated from the same permutations.",
        f"The whole screening procedure was repeated on {resampling.get('n_valid')} event-stratified subsamples of "
        f"{_percent(resampling.get('fraction', 0.632))} of the patients; in each, a marker counted as selected when its "
        f"Benjamini-Hochberg q-value was at most {alpha}, and its rank and direction were recorded.",
        f"Under rules fixed before the analysis, a marker was robust when its family-wise p-value was at most {alpha}, "
        f"it was selected in at least {_percent(settings.get('robust_frequency', 0.5))} of the subsamples and its direction held in at least "
        f"{_percent(settings.get('robust_direction', 0.9))} of them; markers with a family-wise p-value at most {alpha} or a permutation "
        f"q-value at most {settings.get('fdr_level')} that did not meet the stability rule were called suggestive"
        + (
            f", and markers without added value but with a marginal family-wise p-value at most {alpha} were called marginal only."
            if added_value
            else "."
        ),
        f"Markers with more than {_percent(settings.get('max_missing_fraction', 0.2))} missing values or a constant value were excluded; "
        "other missing marker values were replaced by the marker's median among the patients in each fit, and patients with a missing "
        "or invalid outcome, clinical covariate or stratum were excluded.",
        "Markers were analysed as continuous variables without cut-points.",
    ]
    if signature.get("apparent_c") is not None:
        sentences.append(
            ("A Cox model with the clinical covariates and " if added_value else "A Cox model with ")
            + f"the selected markers (at most the {settings.get('max_signature_markers')} strongest) was fitted, and its apparent C-index "
            "was corrected for optimism by subtracting the mean difference between the C-index of the whole procedure in each subsample "
            "and in the patients left out of it."
        )
    return " ".join(sentences)


def marker_results_paragraph(result: dict[str, Any]) -> str:
    counts = result.get("tier_counts") or {}
    cohort = result.get("cohort") or {}
    signature = result.get("signature") or {}
    added_value = result.get("primary_lens") == "added_value"
    text = (
        f"Of {cohort.get('n_markers_evaluated')} markers, {counts.get('robust', 0)} were robust"
        + (f", {counts.get('suggestive', 0)} suggestive and {counts.get('marginal only', 0)} marginal only" if added_value else f" and {counts.get('suggestive', 0)} suggestive")
        + "."
    )
    if signature.get("apparent_c") is not None:
        text += (
            f" The selected-marker model ({_names(signature.get('markers') or [])}) had an apparent C-index of {_number(signature.get('apparent_c'))} "
            f"and an optimism-corrected C-index of {_number(signature.get('optimism_corrected_c'))}."
        )
    return text


def remark_checklist(result: dict[str, Any], *, request: dict[str, Any] | None = None, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    """REMARK checklist for one ``evaluate_markers`` result and the request that produced it."""
    request = request or {}
    cohort = result.get("cohort") or {}
    settings = result.get("settings") or {}
    resampling = result.get("resampling") or {}
    signature = result.get("signature") or {}
    counts = result.get("tier_counts") or {}
    markers_evaluated = int(cohort.get("n_markers_evaluated") or 0)
    events = int(cohort.get("events") or 0)
    dropped = cohort.get("dropped_markers") or []
    added_value = result.get("primary_lens") == "added_value"
    n_adjusted = _exact_fit_count(result, "adjusted")
    n_unadjusted = _exact_fit_count(result, "marginal")
    def exact_markers(count: int) -> str:
        if count == markers_evaluated:
            return f"all {count} markers"
        return f"{count} markers (the {settings.get('shortlist_size')} strongest and every supported one)"

    excluded_rows = None
    if dataset and dataset.get("n_rows") is not None and cohort.get("n") is not None:
        excluded_rows = int(dataset["n_rows"]) - int(cohort["n"])
    shrinkage = signature.get("top_marker_shrinkage")
    items = [
        _item("1", "Introduction", "Markers, objectives and pre-specified hypotheses", "partly",
              f"Markers evaluated: {_names(request.get('marker_columns') or [], limit=20)}. State the objectives and the hypotheses fixed before the analysis."),
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
              f"{markers_evaluated} candidate markers; clinical covariates: {_names(cohort.get('clinical_columns') or [])}; strata: {_names(cohort.get('strata_columns') or [])}."
              + (f" {len(dropped)} marker(s) were excluded before the analysis ({_names([entry.get('marker') for entry in dropped])})." if dropped else "")),
        _item("9", "Study design", "Sample size rationale", "partly",
              f"{cohort.get('n')} patients and {events} events for {markers_evaluated} candidate markers ({events / max(markers_evaluated, 1):.1f} events per marker). Give the rationale for the sample size."),
        _item("10", "Statistical analysis", "Statistical methods, variable selection, assumptions, missing data", "reported", marker_methods_paragraph(result, request)),
        _item("11", "Statistical analysis", "Handling of marker values and cut-points", "reported",
              "Markers were analysed as continuous variables; the evaluation used no cut-points. Any cut-point used for figures should be fixed before looking at the outcome."),
        _item("12", "Results", "Patient flow, numbers analysed and events", "reported",
              f"{cohort.get('n')} patients with {events} events were analysed"
              + (f"; {excluded_rows} of {dataset['n_rows']} rows were excluded for a missing or invalid outcome, clinical covariate or stratum" if excluded_rows else "")
              + ("; no rows were excluded" if excluded_rows == 0 else "")
              + "."),
        _item("13", "Results", "Distribution of demographics, prognostic variables and the markers", "author",
              "Report age, sex, the standard prognostic variables and the markers, with numbers of missing values; the Table 1 tab builds this table."),
        _item("14", "Results", "Relation of the markers to standard prognostic variables", "partly" if added_value else "author",
              (
                  f"{counts['marginal only']} marker(s) were marginal only: associated with survival on their own but adding nothing beyond "
                  "the clinical covariates. "
                  if counts.get("marginal only")
                  else "No marker was marginal only (associated with survival on its own but adding nothing beyond the clinical covariates). "
              )
              + 
              "Show how the reported markers relate to the standard prognostic variables."
              if added_value else "Show how the markers relate to the standard prognostic variables."),
        _item("15", "Results", "Univariable analyses", "reported",
              f"The marker table gives every marker's unadjusted score-test p-value"
              + (" (Unadjusted P) and, for " if added_value else " and, for ")
              + f"{exact_markers(n_unadjusted)}, the hazard ratio with a 95% confidence interval from an unadjusted Cox model"
              + (" (Unadjusted HR)." if added_value else ".")
              + " Kaplan-Meier curves by marker level can be drawn in the Survival curves tab with a cut-point fixed in advance."),
        _item("16", "Results", "Multivariable analyses with confidence intervals", "partly",
              f"The marker table gives, for {exact_markers(n_adjusted)}, hazard ratios with 95% Wald confidence intervals from Cox models "
              "with the clinical covariates. Report the final model with all its variables from the Cox model tab."
              if added_value else "Without clinical covariates only unadjusted hazard ratios are available; report a model with the standard prognostic variables."),
        _item("17", "Results", "Marker effects adjusted for standard prognostic variables regardless of significance", "reported" if added_value else "author",
              f"Adjusted hazard ratios with confidence intervals are given for {exact_markers(n_adjusted)}, whether or not they are significant."
              if added_value else "Report the marker effects adjusted for the standard prognostic variables."),
        _item("18", "Results", "Further investigations: assumptions, sensitivity, internal validation", "partly",
              f"Internal validation: the whole procedure was repeated on {resampling.get('n_valid')} subsamples"
              + (f", and the selected-marker model's C-index was corrected from {_number(signature.get('apparent_c'))} to {_number(signature.get('optimism_corrected_c'))}" if signature.get("apparent_c") is not None else "")
              + (f"; in the patients left out, the strongest marker's log hazard ratio was {_percent(shrinkage)} of its value in the subsamples that selected it" if isinstance(shrinkage, (int, float)) else "")
              + ". Check proportional hazards for the reported markers in the Cox model tab."),
        _item("19", "Discussion", "Interpretation and limitations", "author",
              marker_results_paragraph(result) + " Interpret these results against the pre-specified hypotheses and discuss the limitations."),
        _item("20", "Discussion", "Implications for future research and clinical value", "author",
              "Discuss the implications, including validation in an independent cohort."),
    ]
    return {
        "guideline": "REMARK",
        "reference": "McShane LM et al. J Natl Cancer Inst 2005;97:1180-4; Altman DG et al. PLoS Med 2012;9:e1001216",
        "software": f"SurvStudio {__version__}",
        "methods": marker_methods_paragraph(result, request),
        "results": marker_results_paragraph(result),
        "items": items,
    }


# ── TRIPOD+AI (prediction models) ────────────────────────────────


_FAMILY_LABELS = {"ml": "classical machine-learning models", "dl": "deep-learning models"}


def _failed_models(result: dict[str, Any]) -> list[str]:
    names = list(result.get("excluded_models") or [])
    names += [str(error.get("model")) for error in result.get("errors") or [] if isinstance(error, dict) and error.get("model")]
    return sorted(set(names))


def _layers_text(layers: Any) -> str:
    sizes = [str(size) for size in layers or []]
    if not sizes:
        return "the default hidden layers"
    if len(sizes) == 1:
        return f"one hidden layer of {sizes[0]} units"
    return f"{len(sizes)} hidden layers ({', '.join(sizes[:-1])} and {sizes[-1]} units)"


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
            text += (
                f", stopping early after {request.get('early_stopping_patience')} epochs without improvement on a monitoring subset "
                "drawn from each training partition"
            )
        return text + "."
    text = f"Random survival forests and gradient boosting used {request.get('n_estimators')} trees"
    if request.get("max_depth") is not None:
        text += f" with maximum depth {request.get('max_depth')}"
    return (
        text
        + f" (learning rate {request.get('learning_rate')} for boosting); the LASSO-Cox penalty was chosen by inner cross-validation, "
        "stratified by event status, within each training partition."
    )


def _evaluation_text(result: dict[str, Any]) -> str:
    mode = str(result.get("evaluation_mode", ""))
    if mode.startswith("repeated_cv"):
        repeats = result.get("cv_repeats")
        text = (
            f"{result.get('cv_folds')}-fold cross-validation stratified by event status"
            if repeats == 1
            else f"{repeats} repeats of {result.get('cv_folds')}-fold cross-validation stratified by event status"
        )
        if result.get("n_locked_test_patients"):
            text += (
                f" on a development set of {result.get('n_development_patients')} patients, then once on a locked test set of "
                f"{result.get('n_locked_test_patients')} patients ({result.get('n_locked_test_events')} events"
                + (f"; {_percent(result['locked_test_fraction'])} of the cohort" if result.get("locked_test_fraction") else "")
                + ") that was never used for fitting, preprocessing, tuning or model choice"
            )
        return text
    if mode == "holdout":
        if result.get("n_evaluation_patients") is not None:
            return (
                f"a holdout set of {result.get('n_evaluation_patients')} patients ({result.get('n_evaluation_events')} events), "
                f"stratified by event status, after fitting on {result.get('n_fit_patients')} patients"
            )
        return "a holdout set stratified by event status"
    return "the patients used for fitting (apparent performance, which is optimistic)"


def _evaluation_sentence(results: Sequence[dict[str, Any]]) -> str:
    texts = {_evaluation_text(result) for result in results}
    if len(texts) == 1:
        return f"Performance was estimated by {next(iter(texts))}."
    return " ".join(
        f"For the {_FAMILY_LABELS.get(str(result.get('family')), 'models')}, performance was estimated by {_evaluation_text(result)}."
        for result in results
    )


def _shared_splits(results: Sequence[dict[str, Any]]) -> bool | None:
    fingerprints = [result.get("evaluation_split_fingerprint") for result in results]
    if not all(fingerprints):
        return None
    return len(set(fingerprints)) == 1


def prediction_methods_paragraph(results: Sequence[dict[str, Any]]) -> str:
    if not results:
        return ""
    first = results[0]
    models = [str(row.get("model")) for result in results for row in result.get("comparison_table", [])]
    shared = _shared_splits(results)
    sentences = [
        f"Survival models ({_names(models, limit=20)}) were compared in {first.get('n_patients')} patients ({first.get('n_events')} events) "
        f"with SurvStudio {__version__}.",
        _evaluation_sentence(results),
    ]
    if shared:
        sentences.append(
            f"All models were trained and scored on the same data partitions (split fingerprint {first.get('evaluation_split_fingerprint')})."
        )
    elif shared is False:
        sentences.append("The model families were scored on different data partitions, so their results are not directly comparable.")
    with_brier = [
        any(isinstance(row.get("ibs"), (int, float)) for row in result.get("comparison_table", [])) for result in results
    ]
    brier = (
        "the integrated Brier score, weighted by the inverse probability of censoring, and the Brier skill score against a "
        "Kaplan-Meier model"
    )
    if all(with_brier):
        metrics = f"Discrimination was measured with Harrell's C-index and overall accuracy with {brier}"
    elif any(with_brier):
        brier_families = [_FAMILY_LABELS.get(str(result.get("family")), "models") for result, has in zip(results, with_brier) if has]
        metrics = f"Discrimination was measured with Harrell's C-index, and overall accuracy of the {' and '.join(brier_families)} with {brier}"
    else:
        metrics = "Discrimination was measured with Harrell's C-index"
    sentences += [
        "Preprocessing was fitted within each training partition and applied unchanged to the evaluation patients: numeric predictors "
        "were imputed with the training median, categorical predictors were reference-coded (with a missing-value indicator only when "
        "the training data had missing values) and, for neural networks, numeric predictors were standardised.",
        metrics + "; models were ranked by C-index.",
    ]
    sentences += [text for result in results if (text := _hyperparameter_text(result))]
    return " ".join(sentences)


def _performance_text(results: Sequence[dict[str, Any]]) -> str:
    rows = [row for result in results for row in result.get("comparison_table", [])]
    ranked = sorted((row for row in rows if isinstance(row.get("c_index"), (int, float))), key=lambda row: -float(row["c_index"]))
    if not ranked:
        return "No model produced a C-index."
    best = ranked[0]
    text = f"The highest C-index was {_number(best.get('c_index'))} ({best.get('model')}"
    if isinstance(best.get("brier_skill_score"), (int, float)):
        text += f"; integrated Brier score {_number(best.get('ibs'))}, Brier skill score {_number(best.get('brier_skill_score'))}"
    text += ")"
    if isinstance(best.get("locked_test_c_index"), (int, float)):
        text += f"; on the locked test set its C-index was {_number(best.get('locked_test_c_index'))}"
    text += f". Across {len(ranked)} models the C-index ranged from {_number(ranked[-1].get('c_index'))} to {_number(best.get('c_index'))}."
    if _shared_splits(results) is False:
        text += " The model families were scored on different partitions, so the ranking across families is not a fair comparison."
    return text


def tripod_ai_checklist(results: Sequence[dict[str, Any]], *, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    """TRIPOD+AI checklist for one or more model comparisons (classical ML and deep learning).

    Each result is a comparison ``analysis`` with its ``request_config`` and a ``family`` of "ml" or "dl".
    """
    results = [result for result in results if result]
    first = results[0] if results else {}
    request = first.get("request_config") or {}
    features = request.get("features") or []
    categorical = request.get("categorical_features") or []
    failed = sorted({name for result in results for name in _failed_models(result)})
    n_models = sum(len(result.get("comparison_table", [])) for result in results)
    events = first.get("n_events")
    locked = bool(first.get("n_locked_test_patients"))
    items = [
        _item("1", "Title", "Development or evaluation, population, outcome", "author",
              "Identify the study as developing or evaluating a prediction model, with the target population and the outcome."),
        _item("2", "Abstract", "Structured summary", "author", "See the TRIPOD+AI for Abstracts checklist."),
        _item("3", "Introduction", "Background", "author", "Explain the healthcare context and the rationale for the model."),
        _item("4", "Introduction", "Objectives", "author", "State whether the study develops, evaluates or compares models."),
        _item("5", "Methods", "Data sources", "partly", _dataset_text(dataset) + " Describe the source, the setting and the dates of data collection."),
        _item("6", "Methods", "Participants", "partly",
              f"{first.get('n_patients')} patients with a usable outcome were analysed. Describe the eligibility criteria and the treatments."),
        _item("7", "Methods", "Data preparation", "reported",
              "Numeric predictors were imputed with the training median, categorical predictors reference-coded (with a missing-value "
              "indicator only when the training data had missing values) and, for neural networks, standardised; every step was fitted "
              "on the training partition only."),
        _item("8", "Methods", "Outcome", "partly", _endpoint_text(request) + " State whether the outcome was assessed blind to the predictors."),
        _item("9", "Methods", "Predictors", "partly",
              f"{len(features)} predictors: {_names(features, limit=30)}; categorical: {_names(categorical)}. Describe how and when they were measured."),
        _item("10", "Methods", "Sample size", "partly",
              f"{first.get('n_patients')} patients and {events} events for {len(features)} candidate predictors. Justify the sample size."),
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
              f"{first.get('n_patients')} patients and {events} events"
              + (f"; development set {first.get('n_development_patients')} patients, locked test set {first.get('n_locked_test_patients')} patients" if locked else "")
              + ". Summarise their characteristics with the Table 1 tab."),
        _item("21", "Results", "Model development", "reported",
              f"{n_models} models were fitted" + (f"; fitting failed for {_names(failed)}" if failed else "") + "."),
        _item("22", "Results", "Model specification", "partly",
              "Tree ensembles and neural networks are specified by their settings rather than by an equation; share the settings "
              "(request configuration) so that the models can be refitted."),
        _item("23", "Results", "Model performance", "reported", _performance_text(results)),
        _item("24", "Results", "Model updating", "author", "Report any model updating, or state that none was done."),
        _item("25", "Discussion", "Interpretation", "author", "Give an overall interpretation, including fairness where relevant."),
        _item("26", "Discussion", "Limitations", "partly",
              "Performance comes from internal validation in one data set" + (" with a locked test set" if locked else "")
              + "; an external cohort is needed to judge transportability."
              + (
                  f" Choosing the best of {n_models} models on the same data makes its C-index optimistic (the winner's curse); "
                  + ("report the chosen model's locked-test C-index." if locked else "judge it on data that played no part in the choice.")
                  if n_models > 1
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


def checklist_markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    lines = [
        f"# {report['guideline']} checklist",
        "",
        checklist_intro(report),
        "",
        "## Methods",
        "",
        report.get("methods", ""),
        "",
        "## Results",
        "",
        report.get("results", ""),
        "",
        "## Checklist",
        "",
        "| " + " | ".join(CHECKLIST_COLUMNS) + " |",
        "|" + "---|" * len(CHECKLIST_COLUMNS),
    ]
    lines += ["| " + " | ".join(cell(row[column]) for column in CHECKLIST_COLUMNS) + " |" for row in checklist_rows(report)]
    return "\n".join(lines) + "\n"

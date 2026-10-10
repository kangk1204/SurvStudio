// SurvStudio front end, part 6/8: Request validation, dataset loading, derived groups, and the KM, signature, Cox, and table runs.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

function currentBaseConfig() {
  if (!state.dataset) throw new Error("Load a dataset first.");
  const timeColumn = refs.timeColumn.value;
  const eventColumn = refs.eventColumn.value;
  if (!timeColumn || !eventColumn) throw new Error("Select both a time column and an event column.");
  const matchingOutcomeWarning = identicalOutcomeColumnMessage();
  if (matchingOutcomeWarning) throw new Error(matchingOutcomeWarning);
  const timeWarning = currentTimeColumnWarning();
  if (timeWarning?.tone === "error") throw new Error(timeWarning.message);
  const eventWarning = currentEventColumnWarning();
  if (eventWarning?.blocking) {
    if (eventWarning.tone === "warning" && !refs.showAllEventColumns?.checked) {
      throw new Error(`${eventWarning.message} If this is intentional, tick All columns next to Event first.`);
    }
    throw new Error(eventWarning.message);
  }
  // The server trims these texts, so send them trimmed: the echoed request_config then matches the controls.
  const eventPositiveValue = String(refs.eventPositiveValue.value ?? "").trim();
  if (!eventPositiveValue) {
    throw new Error(`Choose the Event Value for "${eventColumn}" before running an analysis.`);
  }
  return {
    dataset_id: state.dataset.dataset_id,
    time_column: timeColumn,
    event_column: eventColumn,
    event_positive_value: eventPositiveValue,
    group_column: refs.groupColumn.value || null,
    time_unit_label: String(refs.timeUnitLabel.value ?? "").trim() || DEFAULT_TIME_UNIT_LABEL,
    max_time: refs.maxTime.value ? Number(refs.maxTime.value) : null,
  };
}

// True while the controls still describe the endpoint (time, event, event value) of `config`.
function sameEndpoint(config) {
  let base;
  try {
    base = currentBaseConfig();
  } catch {
    return false;
  }
  return String(base.time_column) === String(config?.time_column ?? "")
    && String(base.event_column) === String(config?.event_column ?? "")
    && String(base.event_positive_value).trim() === String(config?.event_positive_value ?? "").trim();
}

function validateGroupingSelection() {
  const warning = currentGroupColumnWarning();
  if (warning?.tone === "error") throw new Error(warning.message);
}

function validateMinGroupFraction(control, label = "Min group fraction") {
  const value = numericControlValue(control, 0.1);
  // Backend bounds are exclusive: 0.02 < fraction < 0.45.
  if (!(value > 0.02 && value < 0.45)) {
    throw new Error(`${label} must be greater than 0.02 and less than 0.45. Current value: ${formatValue(value)}.`);
  }
  return value;
}

function mlSetting(key, control) {
  return numericControlValue(control, ML_NUMERIC_DEFAULTS[key]);
}

function dlSetting(key, control) {
  return numericControlValue(control, DL_NUMERIC_DEFAULTS[key]);
}

function validateMlControls({ compare = false } = {}) {
  const modelType = compare ? "compare" : String(refs.mlModelType?.value || "rsf");
  if (compare || modelType === "rsf" || modelType === "gbs") {
    const nEstimators = mlSetting("n_estimators", refs.mlNEstimators);
    if (!Number.isInteger(nEstimators) || nEstimators < 10 || nEstimators > 1000) {
      throw new Error(`Trees must be an integer between 10 and 1000. Current value: ${formatValue(nEstimators)}.`);
    }
  }
  if (compare || modelType === "gbs") {
    const learningRate = mlSetting("learning_rate", refs.mlLearningRate);
    if (!Number.isFinite(learningRate) || learningRate <= 0.001 || learningRate > 1) {
      throw new Error(`Learning rate must be greater than 0.001 and at most 1. Current value: ${formatValue(learningRate)}.`);
    }
  }
  validatePredictiveEvaluationControls("ml", { includeLockedTest: compare });
}

function mlModelRequestFields(modelType) {
  // Send only the hyperparameters the chosen model uses so disabled controls never block a run.
  const compare = modelType === "compare";
  const fields = { random_state: sharedPredictiveSeed() };
  if (compare || modelType === "rsf" || modelType === "gbs") fields.n_estimators = mlSetting("n_estimators", refs.mlNEstimators);
  if (compare || modelType === "gbs") fields.learning_rate = mlSetting("learning_rate", refs.mlLearningRate);
  return fields;
}

function dlArchitectureRequestFields(modelType) {
  // Only send the hyperparameters the chosen architecture uses: the backend validates every field it
  // receives, so a hidden or disabled control must not block an unrelated model.
  const compare = modelType === "compare";
  const usesHiddenLayers = compare || modelType !== "transformer";
  const usesDiscreteTime = compare || modelType === "deephit" || modelType === "mtlr";
  const usesTransformer = compare || modelType === "transformer";
  const usesVae = compare || modelType === "vae";
  const repeatedCv = refs.dlEvaluationStrategy.value === "repeated_cv";
  return {
    dropout: dlSetting("dropout", refs.dlDropout),
    learning_rate: dlSetting("learning_rate", refs.dlLearningRate),
    epochs: dlSetting("epochs", refs.dlEpochs),
    // The seed ML uses too, so both families split the patients alike.
    random_seed: sharedPredictiveSeed(),
    early_stopping_patience: dlSetting("early_stopping_patience", refs.dlEarlyStoppingPatience),
    early_stopping_min_delta: dlSetting("early_stopping_min_delta", refs.dlEarlyStoppingMinDelta),
    evaluation_strategy: refs.dlEvaluationStrategy.value,
    ...(usesHiddenLayers ? { hidden_layers: parseHiddenLayersStrict() } : {}),
    ...(usesDiscreteTime ? {
      batch_size: dlSetting("batch_size", refs.dlBatchSize),
      num_time_bins: dlSetting("num_time_bins", refs.dlNumTimeBins),
    } : {}),
    ...(repeatedCv ? {
      cv_folds: dlSetting("cv_folds", refs.dlCvFolds),
      cv_repeats: dlSetting("cv_repeats", refs.dlCvRepeats),
      parallel_jobs: dlSetting("parallel_jobs", refs.dlParallelJobs),
    } : {}),
    ...(usesTransformer ? {
      d_model: dlSetting("d_model", refs.dlDModel),
      n_heads: dlSetting("n_heads", refs.dlHeads),
      n_layers: dlSetting("n_layers", refs.dlLayers),
    } : {}),
    ...(usesVae ? {
      latent_dim: dlSetting("latent_dim", refs.dlLatentDim),
      n_clusters: dlSetting("n_clusters", refs.dlClusters),
    } : {}),
  };
}

function validateDlControls({ compare = false } = {}) {
  const modelType = compare ? "compare" : (refs.dlModelType?.value || "deepsurv");
  const usesHiddenLayers = compare || modelType !== "transformer";
  const usesDiscreteTime = compare || modelType === "deephit" || modelType === "mtlr";
  const usesTransformer = compare || modelType === "transformer";
  const usesVae = compare || modelType === "vae";
  const epochs = dlSetting("epochs", refs.dlEpochs);
  if (!Number.isFinite(epochs) || epochs < 10 || epochs > 1000) {
    throw new Error(`Epochs must be between 10 and 1000. Current value: ${formatValue(epochs)}.`);
  }
  const learningRate = dlSetting("learning_rate", refs.dlLearningRate);
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.1) {
    throw new Error(`Learning rate must be greater than 0 and at most 0.1. Current value: ${formatValue(learningRate)}.`);
  }
  const dropout = dlSetting("dropout", refs.dlDropout);
  if (!Number.isFinite(dropout) || dropout < 0 || dropout > 0.5) {
    throw new Error(`Dropout must be between 0 and 0.5. Current value: ${formatValue(dropout)}.`);
  }
  if (usesHiddenLayers) {
    parseHiddenLayersStrict();
  }
  if (usesDiscreteTime) {
    const batchSize = dlSetting("batch_size", refs.dlBatchSize);
    if (!Number.isFinite(batchSize) || batchSize < 8 || batchSize > 512) {
      throw new Error(`Batch size must be between 8 and 512. Current value: ${formatValue(batchSize)}.`);
    }
  }
  const randomSeed = sharedPredictiveSeed();
  if (!Number.isFinite(randomSeed) || !Number.isInteger(randomSeed)) {
    throw new Error(`Random seed must be an integer. Current value: ${formatValue(randomSeed)}.`);
  }
  const patience = dlSetting("early_stopping_patience", refs.dlEarlyStoppingPatience);
  if (!Number.isFinite(patience) || patience < 1 || patience > 100) {
    throw new Error(`Early stop patience must be between 1 and 100. Current value: ${formatValue(patience)}.`);
  }
  const minDelta = dlSetting("early_stopping_min_delta", refs.dlEarlyStoppingMinDelta);
  if (!Number.isFinite(minDelta) || minDelta < 0 || minDelta > 0.1) {
    throw new Error(`Min delta must be between 0 and 0.1. Current value: ${formatValue(minDelta)}.`);
  }
  const evaluationStrategy = refs.dlEvaluationStrategy?.value || "holdout";
  if (evaluationStrategy === "repeated_cv") {
    const cvFolds = dlSetting("cv_folds", refs.dlCvFolds);
    if (!Number.isFinite(cvFolds) || cvFolds < 2 || cvFolds > 10) {
      throw new Error(`CV folds must be between 2 and 10. Current value: ${formatValue(cvFolds)}.`);
    }
    const cvRepeats = dlSetting("cv_repeats", refs.dlCvRepeats);
    if (!Number.isFinite(cvRepeats) || cvRepeats < 1 || cvRepeats > 20) {
      throw new Error(`CV repeats must be between 1 and 20. Current value: ${formatValue(cvRepeats)}.`);
    }
    const parallelJobs = dlSetting("parallel_jobs", refs.dlParallelJobs);
    if (!Number.isFinite(parallelJobs) || parallelJobs < 1 || parallelJobs > 16) {
      throw new Error(`Parallel jobs must be between 1 and 16. Current value: ${formatValue(parallelJobs)}.`);
    }
  }
  validatePredictiveEvaluationControls("dl", { includeLockedTest: compare });
  if (usesDiscreteTime) {
    const numTimeBins = dlSetting("num_time_bins", refs.dlNumTimeBins);
    if (!Number.isFinite(numTimeBins) || numTimeBins < 10 || numTimeBins > 200) {
      throw new Error(`Time bins must be between 10 and 200. Current value: ${formatValue(numTimeBins)}.`);
    }
  }
  if (usesTransformer) {
    const dModel = dlSetting("d_model", refs.dlDModel);
    const nHeads = dlSetting("n_heads", refs.dlHeads);
    const nLayers = dlSetting("n_layers", refs.dlLayers);
    if (!Number.isFinite(dModel) || dModel < 16 || dModel > 256) {
      throw new Error(`Transformer width must be between 16 and 256. Current value: ${formatValue(dModel)}.`);
    }
    if (!Number.isFinite(nHeads) || nHeads < 1 || nHeads > 16) {
      throw new Error(`Attention heads must be between 1 and 16. Current value: ${formatValue(nHeads)}.`);
    }
    if (!Number.isFinite(nLayers) || nLayers < 1 || nLayers > 8) {
      throw new Error(`Transformer layers must be between 1 and 8. Current value: ${formatValue(nLayers)}.`);
    }
    if (dModel % nHeads !== 0) {
      throw new Error(`Transformer width must be divisible by attention heads. Current values: width=${formatValue(dModel)}, heads=${formatValue(nHeads)}.`);
    }
  }
  if (usesVae) {
    const latentDim = dlSetting("latent_dim", refs.dlLatentDim);
    const nClusters = dlSetting("n_clusters", refs.dlClusters);
    if (!Number.isFinite(latentDim) || latentDim < 2 || latentDim > 32) {
      throw new Error(`Latent dim must be between 2 and 32. Current value: ${formatValue(latentDim)}.`);
    }
    if (!Number.isFinite(nClusters) || nClusters < 2 || nClusters > 10) {
      throw new Error(`Clusters must be between 2 and 10. Current value: ${formatValue(nClusters)}.`);
    }
  }
}

function renderDatasetPreview() {
  // The file's own column order, names and values: object keys would put numeric names such as "7157" first,
  // and header, number or p-value formatting would rewrite the data (a column named "ki67_p" is not a
  // p-value, and a 13-digit patient ID is not "1.70e+12").
  const rows = state.dataset.preview || [];
  const present = new Set(rows.flatMap((row) => Object.keys(row || {})));
  const columns = datasetColumnNames().filter((column) => present.has(column));
  renderTable(refs.datasetPreviewShell, rows, columns.length ? columns : null, { pValueColumns: [], rawHeaders: true, rawValues: true });
}

function duplicateIdentifierColumns(dataset = state.dataset) {
  const entries = dataset?.duplicate_identifier_columns ?? dataset?.profile?.duplicate_identifier_columns;
  return Array.isArray(entries) ? entries.filter((entry) => entry && entry.column != null) : [];
}

function renderDatasetIntegrityWarning() {
  const banner = refs.datasetIntegrityWarning;
  if (!banner) return;
  const duplicates = state.dataset ? duplicateIdentifierColumns() : [];
  if (!duplicates.length) {
    banner.textContent = "";
    banner.classList.add("hidden");
    return;
  }
  const count = (value) => (Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "NA");
  const details = duplicates.map((entry) => {
    const column = String(entry.column);
    return `${column} repeats ${count(entry.n_repeated_ids)} ID${Number(entry.n_repeated_ids) === 1 ? "" : "s"} `
      + `(${count(entry.n_rows)} rows vs ${count(entry.n_unique)} unique; ${count(entry.n_extra_rows)} extra rows)`;
  });
  // textContent keeps column names from the uploaded file inert.
  banner.textContent = `Possible repeated subjects: ${details.join("; ")}. `
    + "If rows belong to the same subject, survival estimates double-count them and train/test splits can leak the same subject; "
    + "keep one row per subject before analysis.";
  banner.classList.remove("hidden");
}

function updateDatasetBadge() {
  renderDatasetIntegrityWarning();
  if (!state.dataset) { refs.datasetBadge.classList.add("hidden"); return; }
  refs.datasetBadge.textContent = `${state.dataset.filename} · ${state.dataset.n_rows.toLocaleString()} rows · ${state.dataset.n_columns} cols`;
  refs.datasetBadge.classList.remove("hidden");
}

function scrollWorkspaceEntryToTop() {
  const resetScroll = () => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  };
  requestAnimationFrame(resetScroll);
  window.setTimeout(resetScroll, 300);
}

function showWorkspace() {
  refs.landing.classList.add("hidden");
  refs.landing.classList.remove("fade-out");
  refs.workspace.classList.remove("hidden");
  refs.workspace.classList.remove("fade-in");
}

function setAdditionalAnalyses(expanded) {
  document.body.classList.toggle("additional-analyses-open", expanded);
  const button = document.getElementById("additionalAnalysesButton");
  button?.setAttribute("aria-expanded", expanded ? "true" : "false");
  if (button) button.textContent = expanded ? "Fewer analyses" : "More analyses";
}

function activateTab(tabName, { historyMode = "replace", focusTabButton = false, syncHistory = true } = {}) {
  let resolvedTabName = tabName;
  if (resolvedTabName === "predictive") {
    resolvedTabName = "benchmark";
  }
  if (tabName === "ml" || tabName === "dl") {
    runtime.predictiveFamily = tabName;
  }
  // ML and DL controls live in the Prediction models tab.
  if (resolvedTabName === "ml" || resolvedTabName === "dl") {
    resolvedTabName = "benchmark";
  }
  // Restored history and completed background runs must reveal their selected tab.
  if (["markers", "tables"].includes(resolvedTabName)) setAdditionalAnalyses(true);
  if (resolvedTabName !== "benchmark" && activeTabName() === "benchmark") {
    runtime.workbenchRevealed = false;
    runtime.predictiveWorkbenchIntent = null;
    refs.benchmarkWorkbench?.classList.add("hidden");
    refs.predictiveModelSelector?.closest(".predictive-model-picker")?.classList.add("hidden");
    refs.runPredictiveSelectedButton?.classList.add("hidden");
    refs.runPredictiveWorkbenchButton?.classList.add("hidden");
  }
  refs.tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === resolvedTabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
    button.setAttribute("tabindex", isActive ? "0" : "-1");
    if (isActive && focusTabButton) {
      try {
        button.focus({ preventScroll: true });
      } catch {
        button.focus();
      }
    }
  });
  refs.tabPanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === resolvedTabName));
  const hint = document.getElementById("analysisStepHint");
  if (hint) hint.textContent = {
    data: "Check your rows and outcome columns before analysing.",
    km: "Choose a group if needed, then draw the curves. Save results when ready.",
    cox: "Choose the variables to include, then run the Cox model and review its checks.",
    benchmark: "Review the shared inputs and evaluation settings, then compare models.",
    markers: "Choose markers and clinical variables. Review diagnostics before interpreting the tests.",
    tables: "Choose the variables to describe, then create your baseline table.",
  }[resolvedTabName] || "";
  renderPredictiveWorkbench();
  updateGroupingDetailsVisibility(resolvedTabName);
  if (state.dataset && syncHistory) syncHistoryState(historyMode);
  renderWorkspaceChrome();
  requestAnimationFrame(() => {
    if (resolvedTabName === "km" && state.km) resizePlotIfDisplayed(refs.kmPlot);
    if (resolvedTabName === "cox" && state.cox) {
      resizePlotIfDisplayed(refs.coxPlot);
      resizePlotIfDisplayed(refs.coxDiagnosticsPlot);
      resizePlotIfDisplayed(refs.coxMartingalePlot);
    }
    if ((resolvedTabName === "ml" || resolvedTabName === "benchmark") && state.ml) {
      resizePlotIfDisplayed(refs.mlImportancePlot);
      resizePlotIfDisplayed(refs.mlShapPlot);
      resizePlotIfDisplayed(refs.mlComparisonPlot);
    }
    if ((resolvedTabName === "dl" || resolvedTabName === "benchmark") && state.dl) {
      resizePlotIfDisplayed(refs.dlImportancePlot);
      resizePlotIfDisplayed(refs.dlLossPlot);
      resizePlotIfDisplayed(refs.dlComparisonPlot);
    }
    if (resolvedTabName === "benchmark") resizePlotIfDisplayed(refs.benchmarkComparisonPlot);
    if (resolvedTabName === "markers" && state.markers) {
      resizePlotIfDisplayed(refs.markersSummaryPlot);
      resizePlotIfDisplayed(refs.markersStabilityPlot);
      resizePlotIfDisplayed(refs.markersRankPlot);
      resizePlotIfDisplayed(refs.markerValidationPlot);
    }
  });
}

// The server's first suggested event column that the dataset has; "" when it suggests none.
function suggestedEventColumn(columnNames, suggestions) {
  return (suggestions?.event_columns || []).find((column) => columnNames.includes(column)) || "";
}

function updateControlsFromDataset({ scrollToTop = false } = {}) {
  const columnNames = state.dataset.columns.map((c) => c.name);
  const suggestions = state.dataset.suggestions;
  if (refs.showAllTimeColumns) refs.showAllTimeColumns.checked = false;
  if (refs.showAllEventColumns) refs.showAllEventColumns.checked = false;
  if (refs.covariateSearchInput) refs.covariateSearchInput.value = "";
  if (refs.categoricalSearchInput) refs.categoricalSearchInput.value = "";
  if (refs.cohortVariableSearchInput) refs.cohortVariableSearchInput.value = "";
  // Only a likely follow-up column is preselected; otherwise Time stays blank for the user to choose. Event
  // likewise starts from a suggested event column only, never from a guessed column position.
  renderTimeColumnOptions({ preferred: "", silent: true });
  renderEventColumnOptions({ preferred: suggestedEventColumn(columnNames, suggestions), silent: true });
  renderSelect(refs.groupColumn, columnNames, { includeBlank: true, blankLabel: "Overall only", selected: null });
  // Display settings from a previous dataset (max time in its units, its time unit) must not carry over, and
  // neither do its variable selections: a new dataset starts from its own defaults.
  if (refs.maxTime) refs.maxTime.value = "";
  applyAutomaticTimeUnitLabel({ force: true });
  refreshVariableSelections({ useDefaults: true });
  updateDatasetBadge();
  renderSharedFeatureSummary();
  renderDatasetPreview();
  applyBundledPresets();
  // Presets set Group by without firing a change event. Refresh the locked controls and their help.
  syncDeriveControlsState();
  refs.downloadSignatureButton.disabled = true;
  showWorkspace();
  if (scrollToTop) scrollWorkspaceEntryToTop();
  renderWorkspaceChrome();
  // The default Cox covariates get their usable-row preview without waiting for a change.
  scheduleCoxPreview({ delay: 0 });
}

function clearSignatureSummary() {
  if (!refs.signatureSummary) return;
  refs.signatureSummary.innerHTML = "";
}

// Outcome-informed grouping outputs (the cut-point search card, an optimal-cutpoint scan and its card)
// were computed for the endpoint of their run, so they go together with the other results.
function clearOutcomeInformedGroupingOutputs() {
  clearSignatureSummary();
  if (refs.cutpointPlot) {
    resetPlotElement(refs.cutpointPlot);
    refs.cutpointPlot.classList.add("hidden");
  }
  const derived = refs.deriveSummary?.dataset.summaryKind === "derived" ? currentDerivedSummaryPayload() : null;
  if (derived && (derived.summary?.outcome_informed || derived.summary?.method === "optimal_cutpoint")) {
    refs.deriveSummary.innerHTML = "";
    refs.deriveSummary.classList.add("hidden");
    refs.deriveSummary.dataset.summaryKind = "";
  }
}

function clearAnalysisOutputs() {
  if (refs.predictorAvailabilityConfirmed) refs.predictorAvailabilityConfirmed.checked = false;
  invalidateRequestTokens(["km", "cox", "tables", "signature", "ml", "dl"]);
  invalidateRequestTokens(["markers", "markerValidation"]);
  // A grouping being created is cancelled too: an optimal cutpoint is optimised for the endpoint of its request.
  invalidateRequestTokens(["derive"]);
  clearMarkerOutputs();
  clearOutcomeInformedGroupingOutputs();
  state.km = null;
  state.cox = null;
  state.cohort = null;
  state.signature = null;
  state.ml = null;
  state.dl = null;
  runtime.compareCache.ml = null;
  runtime.compareCache.dl = null;
  runtime.compareCache.unified = null;
  runtime.workbenchRevealed = false;
  runtime.predictiveWorkbenchIntent = null;
  refs.kmMetaBanner.textContent = "";
  refs.coxMetaBanner.textContent = "Choose variables above, then click Run Cox model.";
  refs.mlMetaBanner.textContent = "Select shared model features in Predictive Models, then run analysis.";
  refs.dlMetaBanner.textContent = "Select model features here, configure hyperparameters, then run analysis.";
  resetCoxPreview({ rerender: false });
  refs.kmSummaryShell.innerHTML = '<div class="empty-state">Survival statistics will appear after you run the analysis.</div>';
  refs.kmRiskShell.innerHTML = '<div class="empty-state">Number of patients at risk over time.</div>';
  refs.kmPairwiseShell.innerHTML = '<div class="empty-state">Group-vs-group comparisons (requires 2+ groups).</div>';
  refs.signatureShell.innerHTML = '<div class="empty-state">Use auto-discovery to find the best feature combinations.</div>';
  refs.coxResultsShell.innerHTML = '<div class="empty-state">Hazard ratios will appear after running Cox analysis.</div>';
  refs.coxDiagnosticsShell.innerHTML = '<div class="empty-state">Grambsch-Therneau proportional-hazards tests (per term and global) will appear here.</div>';
  renderInsightBoard(refs.kmInsightBoard, null, "Run KM to generate an interpretation panel.");
  renderInsightBoard(refs.signatureInsightBoard, null, "Run auto-discovery to assess robustness.");
  renderInsightBoard(refs.coxInsightBoard, null, "Run Cox PH to review diagnostics.");
  renderInsightBoard(refs.mlInsightBoard, null, "ML model results.");
  renderInsightBoard(refs.dlInsightBoard, null, "Deep learning results.");
  clearPlotShell(refs.coxDiagnosticsPlot, '<div class="empty-state plot-empty"><span>Scaled Schoenfeld residual screening appears here after fitting the model.</span></div>', { state: "placeholder" });
  resetCoxMartingaleSelector();
  clearPlotShell(refs.coxMartingalePlot, '<div class="empty-state plot-empty"><span>Martingale residual screening for continuous covariates appears here after fitting the model.</span></div>', { state: "placeholder" });
  refs.cohortTableShell.innerHTML = COHORT_TABLE_EMPTY_STATE_HTML;
  refs.mlComparisonShell.innerHTML = '<div class="empty-state">Click "Compare All" to see Cox vs RSF vs GBS side by side.</div>';
  if (refs.mlComparisonTitle) refs.mlComparisonTitle.textContent = "Model Comparison";
  refs.mlManuscriptShell.innerHTML = '<div class="empty-state">Comparison-ready manuscript rows appear after running a comparison.</div>';
  resetPlotElement(refs.mlComparisonPlot);
  refs.mlComparisonPlot.classList.add("hidden");
  refs.dlComparisonShell.innerHTML = '<div class="empty-state">Click "Compare All" to benchmark DeepSurv, DeepHit, Neural MTLR, Transformer, and VAE.</div>';
  if (refs.dlComparisonTitle) refs.dlComparisonTitle.textContent = "Deep Model Comparison";
  refs.dlManuscriptShell.innerHTML = '<div class="empty-state">Comparison-ready manuscript rows appear after running a deep comparison.</div>';
  resetPlotElement(refs.dlComparisonPlot);
  refs.dlComparisonPlot.classList.add("hidden");
  setPanelResultMode(refs.mlPanel, "idle");
  setPanelResultMode(refs.dlPanel, "idle");
  clearPlotShell(refs.mlImportancePlot, '<div class="empty-state plot-empty"><span>Run Analysis to see feature importance</span></div>', { state: "placeholder" });
  clearPlotShell(refs.mlShapPlot, '<div class="empty-state plot-empty"><span>SHAP values will appear after training</span></div>', { state: "placeholder" });
  clearPlotShell(refs.dlImportancePlot, '<div class="empty-state plot-empty"><span>Run Analysis to see deep learning results</span></div>', { state: "placeholder" });
  clearPlotShell(refs.dlLossPlot, '<div class="empty-state plot-empty"><span>Training and monitor metric curves will appear here</span></div>', { state: "placeholder" });
  purgePlot(refs.kmPlot);
  purgePlot(refs.coxPlot);
  refs.kmPlot.innerHTML = '<div class="empty-state plot-empty"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg><span>Click <strong>Draw curves</strong> to draw the survival curves (Ctrl+Enter).</span></div>';
  refs.coxPlot.innerHTML = '<div class="empty-state plot-empty"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg><span>Choose variables and click <strong>Run Cox model</strong>.</span></div>';
  refs.downloadKmSummaryButton.disabled = true;
  refs.downloadKmPairwiseButton.disabled = true;
  if (refs.downloadKmPngButton) refs.downloadKmPngButton.disabled = true;
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.disabled = true;
  refs.downloadSignatureButton.disabled = true;
  refs.downloadCoxResultsButton.disabled = true;
  refs.downloadCoxDiagnosticsButton.disabled = true;
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.disabled = true;
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.disabled = true;
  refs.downloadCohortTableButton.disabled = true;
  if (refs.downloadCohortTableXlsxButton) refs.downloadCohortTableXlsxButton.disabled = true;
  refs.downloadMlComparisonButton.disabled = true;
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = true;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = true;
  setMlManuscriptDownloadsEnabled(false);
  refs.downloadDlComparisonButton.disabled = true;
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = true;
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = true;
  setDlManuscriptDownloadsEnabled(false);
  setAnalysisConsistencyBanner("");
}

function updateAfterDataset(payload, { scrollToTop = false } = {}) {
  // A different dataset makes any pending derive/signature response obsolete.
  invalidateRequestTokens(["derive", "signature"]);
  state.dataset = payload;
  // A marker matrix belongs to the patients of the dataset it was attached to; derived-column snapshots keep it,
  // and another dataset frees it on the server.
  deleteMarkerMatrixOnServer(state.markerMatrix?.matrix_id);
  state.markerMatrix = null;
  // Results, banners, plots, and export buttons of the previous dataset.
  clearAnalysisOutputs();
  refs.deriveSummary.innerHTML = "";
  refs.deriveSummary.classList.add("hidden");
  refs.deriveSummary.dataset.summaryKind = "";
  runtime.lastDerivedGroup = null;
  runtime.deriveDraftTouched = false;
  runtime.derivedColumnProvenance = normalizeDerivedColumnProvenance(payload.derived_column_provenance);
  runtime.resultPreference.ml = "single";
  runtime.resultPreference.dl = "single";
  refs.deriveStatus.textContent = "";
  setSelectValueIfPresent(refs.deriveMethod, "median_split");
  if (refs.cutpointPlot) { resetPlotElement(refs.cutpointPlot); refs.cutpointPlot.classList.add("hidden"); }
  updateControlsFromDataset({ scrollToTop });
}

function updateAfterDerivedDataset(payload, { deferChrome = false } = {}) {
  const snapshot = captureControlSnapshot();
  const columnNames = payload.columns.map((column) => column.name);
  const suggestions = payload.suggestions || { time_columns: [], event_columns: [] };
  // Without the previous choice only a likely follow-up column is preselected (never the first column).
  const preferredTime = snapshot?.timeColumn && columnNames.includes(snapshot.timeColumn)
    ? snapshot.timeColumn
    : "";
  const preferredEvent = snapshot?.eventColumn && columnNames.includes(snapshot.eventColumn)
    ? snapshot.eventColumn
    : suggestedEventColumn(columnNames, suggestions);
  const preferredGroup = snapshot?.groupColumn && columnNames.includes(snapshot.groupColumn)
    ? snapshot.groupColumn
    : null;

  // Kaplan-Meier can run while a cut-point search (its own scope) creates the snapshot.
  const discardedScopes = ["km", "cox", "markers", "tables", "ml", "dl"].filter((scope) => isScopeBusy(scope));

  state.dataset = payload;
  clearAnalysisOutputs();
  if (discardedScopes.length) {
    showToast(
      `The new derived column created a fresh dataset snapshot, so the in-flight ${discardedScopes.map((scope) => goalLabel(scope)).join(", ")} run was discarded. Rerun it after reviewing the new grouping.`,
      "warning",
      7000,
    );
  }
  runtime.derivedColumnProvenance = normalizeDerivedColumnProvenance(payload.derived_column_provenance);
  runtime.deriveDraftTouched = false;

  if (refs.showAllTimeColumns) refs.showAllTimeColumns.checked = Boolean(snapshot?.showAllTimeColumns);
  if (refs.showAllEventColumns) refs.showAllEventColumns.checked = Boolean(snapshot?.showAllEventColumns);
  renderTimeColumnOptions({ preferred: preferredTime, silent: true });
  renderEventColumnOptions({ preferred: preferredEvent, silent: true });
  renderSelect(refs.groupColumn, columnNames, { includeBlank: true, blankLabel: "Overall only", selected: preferredGroup });
  refreshVariableSelections();
  if (snapshot) applyControlSnapshot(snapshot);
  updateDatasetBadge();
  renderDatasetPreview();
  showWorkspace();

  if (!deferChrome) {
    renderWorkspaceChrome();
    queueHistorySync();
  }
}

function hasCompletedResults() {
  return Boolean(state.km || state.cox || state.cohort || state.signature || state.markers || state.ml || state.dl);
}

function uploadEncodingNote(payload) {
  // UTF-8 needs no comment; a legacy encoding (for example Korean CP949 from Excel) is named
  // so a wrong guess shows up as garbled names instead of passing silently.
  const encoding = String(payload?.text_encoding || "").toLowerCase();
  if (!encoding || encoding === "utf-8" || encoding === "utf-8-sig") return "";
  const label = payload?.text_encoding_label || payload?.text_encoding;
  return ` Text was read as ${label}; if column names look garbled, re-save the file as UTF-8 CSV.`;
}

function uploadFeedbackMessages(payload, { previousDatasetName = "", clearedResults = false } = {}) {
  const datasetName = payload?.filename || "dataset";
  const rowCount = Number(payload?.n_rows || 0).toLocaleString();
  const columnCount = Number(payload?.n_columns || 0).toLocaleString();
  const summary = `${datasetName} loaded (${rowCount} rows, ${columnCount} columns).${uploadEncodingNote(payload)}`;
  if (previousDatasetName) {
    return {
      banner: `${summary} Replaced ${previousDatasetName}${clearedResults ? " and cleared the previous analysis results" : ""}. Confirm the new outcome fields before running again.`,
      toast: `Loaded ${datasetName}.${clearedResults ? " Previous results were cleared." : " Previous dataset was replaced."}`,
    };
  }
  return {
    banner: `${summary} Confirm the suggested outcome fields and continue.`,
    toast: `Loaded ${datasetName}.`,
  };
}

async function fetchLatestDatasetPayload(fetchPayload) {
  // Only the most recently requested dataset load may replace the workspace; starting a
  // new load cancels the previous request.
  const loadToken = beginRequestToken("dataset");
  const payload = await fetchPayload(requestSignal("dataset"));
  if (!requestTokenMatches("dataset", loadToken)) return null;
  return payload;
}

function applyLoadedDataset(payload) {
  updateAfterDataset(payload, { scrollToTop: true });
  runtime.historySyncPaused = true;
  activateTab("km");
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function uploadDataset() {
  if (!refs.datasetFile.files?.length) throw new Error("Choose a dataset file first.");
  const selectedFile = refs.datasetFile.files[0];
  const uploadBanner = setRuntimeBanner(`Uploading ${selectedFile.name} and preparing a fresh analysis workspace.`, "info", { held: true });
  const formData = new FormData();
  formData.append("file", selectedFile);
  let payload;
  try {
    payload = await fetchLatestDatasetPayload((signal) => fetchJSON("/api/upload", { method: "POST", body: formData, signal }));
  } catch (error) {
    // A rejected file ends the upload; its "Uploading" banner goes with it (a newer load keeps its own).
    releaseRuntimeBanner(uploadBanner);
    throw error;
  }
  if (!payload) {
    // A newer load replaced this one.
    releaseRuntimeBanner(uploadBanner);
    return;
  }
  const previousDatasetName = state.dataset?.filename || "";
  const clearedResults = Boolean(state.dataset) && hasCompletedResults();
  applyLoadedDataset(payload);
  const feedback = uploadFeedbackMessages(payload, { previousDatasetName, clearedResults });
  setRuntimeBanner(feedback.banner, "success");
  showToast(feedback.toast, "success", 3400);
}

async function loadBundledDataset(endpoint) {
  const payload = await fetchLatestDatasetPayload((signal) => fetchJSON(endpoint, { method: "POST", signal }));
  if (!payload) return;
  applyLoadedDataset(payload);
}

async function loadExampleDataset() {
  await loadBundledDataset("/api/load-example");
}

async function loadTcgaUploadReadyDataset() {
  await loadBundledDataset("/api/load-tcga-upload-ready");
}

async function loadGbsg2Dataset() {
  await loadBundledDataset("/api/load-gbsg2-example");
}

async function deriveGroup({ autoApplyOverride = null, refreshKmOverride = null, toastMode = "default" } = {}) {
  const sourceColumn = refs.deriveSource.value;
  if (!sourceColumn) throw new Error("Select a numeric source column.");
  const method = refs.deriveMethod.value;
  const isPercentileSplit = method === "percentile_split";
  const isExtremeSplit = method === "extreme_split";
  const usesCutoffInput = isPercentileSplit || isExtremeSplit;
  const isOptimal = method === "optimal_cutpoint";
  const requestedColumnName = validateDerivedColumnName(refs.deriveColumnName.value);
  const cutoffInput = refs.deriveCutoff.value.trim();
  let cutoffValue = null;
  if (usesCutoffInput) {
    if (cutoffInput === "") {
      throw new Error(
        isExtremeSplit
          ? "Enter one percentile value, for example 25."
          : isPercentileSplit
            ? "Enter percentile values, for example 25 or 25,25."
            : "Enter percentile values.",
      );
    }
    cutoffValue = cutoffInput;
  }
  if (!state.dataset) throw new Error("Load a dataset first.");
  const sourceDatasetId = state.dataset.dataset_id;
  let optimalOutcome = null;
  if (isOptimal) {
    // Optimal cutpoints use the outcome, so validate the endpoint like every other outcome-based analysis.
    optimalOutcome = currentBaseConfig();
    validateMinGroupFraction(refs.deriveMinGroupFraction);
    const permutationIterations = numericControlValue(refs.derivePermutationIterations, 500);
    if (!Number.isInteger(permutationIterations) || permutationIterations < 0 || permutationIterations > 500) {
      throw new Error(`Permutation iterations must be an integer between 0 and 500. Current value: ${formatValue(permutationIterations)}.`);
    }
  }
  const deriveToken = beginRequestToken("derive");
  const progressStatus = isOptimal
    ? "Scanning a new grouping column..."
    : "Creating a new grouping column...";
  refs.deriveStatus.textContent = progressStatus;
  // A failed, cancelled or discarded request takes its own progress line with it, never a newer message.
  const clearProgressStatus = () => {
    if (refs.deriveStatus.textContent === progressStatus) refs.deriveStatus.textContent = "";
  };

  const body = {
    dataset_id: sourceDatasetId,
    source_column: sourceColumn,
    method,
    new_column_name: requestedColumnName,
    cutoff: cutoffValue,
  };
  if (isOptimal) {
    body.time_column = optimalOutcome.time_column;
    body.event_column = optimalOutcome.event_column;
    body.event_positive_value = optimalOutcome.event_positive_value;
    body.min_group_fraction = numericControlValue(refs.deriveMinGroupFraction, 0.1);
    body.permutation_iterations = numericControlValue(refs.derivePermutationIterations, 500);
    body.random_seed = numericControlValue(refs.deriveRandomSeed, 20260311);
  }

  const preservedGroup = String(refs.groupColumn?.value || "");
  const shouldAutoApplyDerivedGroup = autoApplyOverride ?? !preservedGroup;
  const shouldRefreshKm = refreshKmOverride ?? (shouldAutoApplyDerivedGroup && activeTabName() === "km");
  let payload;
  try {
    payload = await fetchJSON("/api/derive-group", {
      signal: requestSignal("derive"),
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch (error) {
    clearProgressStatus();
    throw error;
  }
  if (!requestTokenMatches("derive", deriveToken) || state.dataset?.dataset_id !== sourceDatasetId) {
    // The workspace moved on (another dataset or a newer derived snapshot); never swap it back.
    clearProgressStatus();
    return;
  }
  if (isOptimal && !sameEndpoint(optimalOutcome)) {
    // The endpoint changed without the change handlers (a restored history entry, for example): a cut point
    // optimised for the old endpoint must not group patients under the new one.
    clearProgressStatus();
    showToast(
      "The endpoint changed while the optimal cutpoint was being scanned, so the grouping was not created. Create it again for the current endpoint.",
      "warning",
      5200,
    );
    return;
  }
  updateAfterDerivedDataset(payload, { deferChrome: shouldRefreshKm });
  runtime.derivedColumnProvenance[payload.derived_column] = {
    outcomeInformed: Boolean(payload.derive_summary?.outcome_informed),
    recipe: payload.derive_summary?.recipe || {},
    summary: payload.derive_summary || null,
  };
  refreshVariableSelections();
  if (shouldAutoApplyDerivedGroup) {
    refs.groupColumn.value = payload.derived_column;
  } else {
    setSelectValueIfPresent(refs.groupColumn, preservedGroup);
  }
  syncDeriveControlsState();
  runtime.lastDerivedGroup = {
    derivedColumn: payload.derived_column,
    summary: payload.derive_summary,
  };
  runtime.deriveDraftTouched = false;
  const featureUseMessage = isOptimal
    ? "ML/DL features were not changed. This cutpoint used outcome information, so keep it for grouping or visualization rather than predictive training."
    : "ML/DL features were not changed. Add it manually to the shared model feature list only if you want models to use it.";
  refs.deriveStatus.textContent = shouldRefreshKm
    ? "Refreshing Kaplan-Meier with the new grouping..."
    : "";
  updateDatasetBadge();
  renderDerivedGroupSummary(payload.derived_column, payload.derive_summary);
  // The new snapshot already cleared every result, the cohort table included (updateAfterDerivedDataset).
  renderSharedFeatureSummary();
  renderWorkspaceChrome();
  queueHistorySync();
  if (toastMode !== "silent") {
    showToast(
      shouldRefreshKm
        ? `Created ${payload.derived_column} and updated Group by. ${featureUseMessage} Kaplan-Meier is refreshing now.`
        : shouldAutoApplyDerivedGroup
          ? `Created ${payload.derived_column} and updated Group by. ${featureUseMessage}`
          : `Created ${payload.derived_column}. Current Group by remains ${preservedGroup}. ${featureUseMessage} Use Group by or Run again when you want to analyze the new grouping.`,
      "success",
      5200,
    );
  }

  // If optimal cutpoint, also show the scan plot
  if (isOptimal && (payload.cutpoint_figure || payload.derive_summary?.scan_data)) {
    try {
      const scanFigure = payload.cutpoint_figure;
      if (scanFigure && refs.cutpointPlot) {
        refs.cutpointPlot.classList.remove("hidden");
        resetPlotElement(refs.cutpointPlot);
        await Plotly.newPlot(refs.cutpointPlot, scanFigure.data, scanFigure.layout, plotConfig("cutpoint_scan"));
      }
    } catch { /* scan plot is optional */ }
  }

  if (shouldRefreshKm) {
    try {
      await runKaplanMeier();
    } finally {
      refs.deriveStatus.textContent = "";
      renderSharedFeatureSummary();
      renderWorkspaceChrome();
    }
  }
}

function updateMethodVisibility() {
  const isOptimal = refs.deriveMethod.value === "optimal_cutpoint";
  const isPercentileSplit = refs.deriveMethod.value === "percentile_split";
  const isExtremeSplit = refs.deriveMethod.value === "extreme_split";
  const usesCutoffInput = isPercentileSplit || isExtremeSplit;
  refs.cutoffWrap.classList.toggle("hidden", !usesCutoffInput);
  if (refs.deriveCutoffLabel) {
    refs.deriveCutoffLabel.firstChild.textContent = isExtremeSplit ? "Extreme percentile " : "Percentile(s) ";
  }
  if (refs.deriveCutoffHelp) {
    refs.deriveCutoffHelp.dataset.tooltip = isExtremeSplit
      ? "Use one percentile from each tail. Example: 25 = at/below the 25th-percentile threshold vs at/above the 75th-percentile threshold, with the middle range excluded. Ties at the threshold can make the realized groups slightly larger."
      : "Use percentile thresholds from the observed distribution. Example: 25 = at/above the 75th-percentile threshold vs rest. Example: 50 matches Median split. Example: 25,25 = at/below the 25th-percentile threshold / between thresholds / at/above the 75th-percentile threshold. Ties at the threshold can make the realized groups slightly larger.";
  }
  if (refs.deriveCutoff) {
    refs.deriveCutoff.placeholder = isExtremeSplit ? "e.g. 25" : "e.g. 25 or 25,25";
  }
  refs.deriveOptimalControls?.classList.toggle("hidden", !isOptimal);
  if (!isOptimal && refs.cutpointPlot) {
    resetPlotElement(refs.cutpointPlot);
    refs.cutpointPlot.classList.add("hidden");
  }
  syncDeriveControlsState();
}

function updateWeightVisibility() {
  refs.fhPowerWrap.classList.toggle("hidden", refs.logrankWeight.value !== "fleming_harrington");
}

function updateMlModelControlVisibility() {
  const selectedModelType = String(refs.mlModelType?.value || "rsf");
  const treeCountField = refs.mlNEstimators?.closest(".toolbar-field");
  const learningRateField = refs.mlLearningRate?.closest(".toolbar-field");
  const learningRateApplies = selectedModelType === "gbs";
  const treeCountApplies = selectedModelType === "rsf" || selectedModelType === "gbs";
  const shapApplies = mlModelSupportsShap(selectedModelType);
  if (refs.mlNEstimators) {
    refs.mlNEstimators.disabled = !treeCountApplies;
    refs.mlNEstimators.setAttribute("aria-disabled", String(!treeCountApplies));
  }
  if (treeCountField) {
    treeCountField.classList.toggle("is-disabled", !treeCountApplies);
    treeCountField.title = treeCountApplies
      ? ""
      : "Tree count applies to Random Survival Forest and Gradient Boosted Survival only.";
  }
  if (refs.mlLearningRate) {
    refs.mlLearningRate.disabled = !learningRateApplies;
    refs.mlLearningRate.setAttribute("aria-disabled", String(!learningRateApplies));
  }
  if (learningRateField) {
    learningRateField.classList.toggle("is-disabled", !learningRateApplies);
    learningRateField.title = learningRateApplies
      ? ""
      : "Learning rate applies to Gradient Boosted Survival only.";
  }
  if (refs.mlSkipShap) {
    if (!shapApplies) refs.mlSkipShap.checked = true;
    refs.mlSkipShap.disabled = !shapApplies;
    refs.mlSkipShap.title = shapApplies
      ? ""
      : "SHAP is currently available for Random Survival Forest and Gradient Boosted Survival only.";
  }
  if (refs.mlShapSafeMode) {
    const safeModeAvailable = shapApplies && !refs.mlSkipShap?.checked;
    refs.mlShapSafeMode.disabled = !safeModeAvailable;
    refs.mlShapSafeMode.setAttribute("aria-disabled", String(!safeModeAvailable));
    refs.mlShapSafeMode.title = safeModeAvailable
      ? ""
      : (!shapApplies
        ? "SHAP safe mode is only available for Random Survival Forest and Gradient Boosted Survival."
        : "Turn off Fast mode to let SHAP safe mode run when needed.");
  }
}

async function runKaplanMeier() {
  // Inputs are checked before the previous request is cancelled, so an invalid click leaves it running.
  const base = currentBaseConfig();
  validateGroupingSelection();
  const requestToken = beginRequestToken("km");
  const datasetId = base.dataset_id;
  const requestedRiskTicks = numericControlValue(refs.riskTablePoints, KM_NUMERIC_DEFAULTS.risk_table_points);
  const loading = beginShellLoading([refs.kmSummaryShell, refs.kmRiskShell]);
  let payload;
  try {
    payload = await fetchJSON("/api/kaplan-meier", {
      signal: requestSignal("km"),
      method: "POST",
      body: JSON.stringify({
        ...base,
        confidence_level: numericControlValue(refs.confidenceLevel, KM_NUMERIC_DEFAULTS.confidence_level),
        risk_table_points: requestedRiskTicks,
        show_confidence_bands: refs.showConfidenceBands.checked,
        logrank_weight: refs.logrankWeight.value,
        fh_p: numericControlValue(refs.fhPower, KM_NUMERIC_DEFAULTS.fh_p),
      }),
    });
  } catch (error) {
    if (requestTokenMatches("km", requestToken)) loading.restore();
    throw error;
  }
  if (!requestTokenMatches("km", requestToken) || state.dataset?.dataset_id !== datasetId) return;
  loading.finish();
  state.km = payload;
  renderAnalysisConsistencyBanner();
  const kmFigure = payload?.figure || { data: [], layout: {} };
  const kmAnalysis = payload?.analysis || {};
  const kmRiskTable = kmAnalysis.risk_table || {};
  const kmSummary = kmAnalysis.scientific_summary || null;
  const cohort = kmAnalysis.cohort || {};
  const test = kmAnalysis.test || null;
  resetPlotElement(refs.kmPlot);
  await Plotly.newPlot(refs.kmPlot, kmFigure.data || [], responsivePlotLayout(refs.kmPlot, kmFigure.layout || {}), plotConfig("km_curve"));
  markPlotResult(refs.kmPlot, payload);
  stabilizePlotShellHeight(refs.kmPlot);
  renderTable(refs.kmSummaryShell, kmAnalysis.summary_table);
  renderTable(refs.kmRiskShell, kmRiskTable.rows, kmRiskTable.columns);
  flashPresetTargets([refs.kmRiskShell]);
  renderTable(refs.kmPairwiseShell, kmAnalysis.pairwise_table);
  renderInsightBoard(refs.kmInsightBoard, kmSummary, "Run KM to generate an interpretation panel.");
  refs.kmMetaBanner.textContent = `N=${formatValue(cohort.n)}, events=${formatValue(cohort.events)}, censored=${formatValue(cohort.censored)}, median follow-up=${formatValue(cohort.median_follow_up)} ${base.time_unit_label}${test ? `, ${test.test} ${pValuePhrase(test.p_value)}` : ""}`;
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("km", {
    successMessage: `Kaplan-Meier analysis complete. Risk table shows ${formatValue(Math.max((kmRiskTable.columns || []).length - 1, 0))} time points.`,
    backgroundMessage: "Kaplan-Meier finished in the background. Switch back when you are ready to review the updated curve.",
  });
}

// The search's summary card sits with its interpretation and ranking on the Markers tab, where Discover is.
function renderSignatureResult(analysis) {
  renderTable(refs.signatureShell, analysis.results_table);
  renderInsightBoard(refs.signatureInsightBoard, analysis.scientific_summary, "Run auto-discovery to assess robustness.");
  const best = analysis.best_split || {};
  const search = analysis.search_space || {};
  const cell = (label, value) => `<div><strong>${escapeHtml(label)}</strong><br/>${escapeHtml(value)}</div>`;
  if (!refs.signatureSummary) return;
  refs.signatureSummary.innerHTML = `
    <div class="signature-summary-grid">
      ${cell("Best signature", best.Signature || "NA")}
      ${cell("HR (sig+ vs -)", formatValue(best["Hazard ratio (signature+ vs -)"]))}
      ${cell("BH-adjusted p", formatPValue(best["BH adjusted p"]))}
      ${cell("Significant", best["Statistically significant"] ? "Yes" : "No")}
      ${cell("Stability", formatValue(best["Stability score"]))}
      ${cell("Bootstrap support", formatValue(best["Bootstrap support (p<alpha)"]))}
      ${cell("Direction consistency", formatValue(best["Bootstrap HR direction consistency"]))}
      ${cell("Validation support", formatValue(best["Validation support (p<alpha)"]))}
      ${cell("Permutation p", formatPValue(best["Permutation p"]))}
      ${cell("Tested", formatValue(search.tested_combinations))}
      ${cell("Significant combos", formatValue(search.significant_signatures))}
      ${cell("Operator", search.combination_operator || "mixed")}
      ${cell("Seed", formatValue(search.random_seed))}
      ${cell("Alpha", formatValue(search.significance_level))}
    </div>`;
}

function signatureSetting(key, control) {
  return numericControlValue(control, SIGNATURE_NUMERIC_DEFAULTS[key]);
}

async function runSignatureSearch() {
  const base = currentBaseConfig();
  if (markerMatrixAttached()) {
    throw new Error("The cut-point search uses markers from the checklist, not an attached marker file. Remove the file to search the checklist markers.");
  }
  const { markers } = currentMarkerSelections();
  const candidateColumns = signatureCandidateColumns();
  if (!markers.length) throw new Error("Select at least one marker to search for cut-point combinations.");
  const requestedColumnName = validateDerivedColumnName(refs.deriveColumnName.value);
  validateMinGroupFraction(refs.signatureMinFraction);
  const requestToken = beginRequestToken("signature");
  const preservedGroup = String(refs.groupColumn?.value || "");
  const requestConfig = {
    dataset_id: state.dataset.dataset_id,
    time_column: base.time_column,
    event_column: base.event_column,
    event_positive_value: base.event_positive_value,
    candidate_columns: candidateColumns,
    max_combination_size: signatureSetting("max_combination_size", refs.signatureMaxDepth),
    top_k: signatureSetting("top_k", refs.signatureTopK),
    min_group_fraction: signatureSetting("min_group_fraction", refs.signatureMinFraction),
    bootstrap_iterations: signatureSetting("bootstrap_iterations", refs.signatureBootstrapIterations),
    bootstrap_sample_fraction: 0.8,
    permutation_iterations: signatureSetting("permutation_iterations", refs.signaturePermutationIterations),
    validation_iterations: signatureSetting("validation_iterations", refs.signatureValidationIterations),
    validation_fraction: signatureSetting("validation_fraction", refs.signatureValidationFraction),
    significance_level: signatureSetting("significance_level", refs.signatureSignificanceLevel),
    combination_operator: refs.signatureOperator.value,
    random_seed: signatureSetting("random_seed", refs.signatureRandomSeed),
    new_column_name: requestedColumnName,
  };
  const payload = await fetchJSON("/api/discover-signature", {
      signal: requestSignal("signature"),
    method: "POST",
    body: JSON.stringify(requestConfig),
  });
  if (!requestTokenMatches("signature", requestToken) || state.dataset?.dataset_id !== requestConfig.dataset_id) return;
  const derivedGroupMeta = payload.signature_analysis?.derived_group || {};
  const shouldAutoApplyDerivedGroup = Boolean(derivedGroupMeta.auto_apply_recommended) && !preservedGroup;
  updateAfterDerivedDataset(payload);
  runtime.derivedColumnProvenance[payload.derived_column] = {
    outcomeInformed: Boolean(derivedGroupMeta.outcome_informed ?? derivedGroupMeta.outcomeInformed ?? true),
    recipe: derivedGroupMeta.recipe || payload.signature_analysis?.signature_recipe || {},
    summary: derivedGroupMeta.summary || null,
  };
  refreshVariableSelections();
  state.signature = {
    ...payload.signature_analysis,
    request_config: payload.signature_request_config || requestConfig,
    result_dataset_id: payload.dataset_id || state.dataset?.dataset_id || "",
    dataset_hash: payload.dataset_hash || state.dataset?.dataset_hash || "",
  };
  renderAnalysisConsistencyBanner();
  if (shouldAutoApplyDerivedGroup) {
    refs.groupColumn.value = payload.derived_column;
  } else {
    setSelectValueIfPresent(refs.groupColumn, preservedGroup);
  }
  updateDatasetBadge();
  renderSignatureResult(payload.signature_analysis);
  renderSharedFeatureSummary();
  renderWorkspaceChrome();
  queueHistorySync();
  refs.deriveStatus.textContent = shouldAutoApplyDerivedGroup
    ? `Auto-derived ${payload.derived_column}`
    : `Derived exploratory grouping ${payload.derived_column}`;
  syncDownloadButtonAvailability();
  // Discover lives on the Markers tab, so that is where the finished search is shown.
  revealCompletedResultIfCurrent("markers", {
    mode: "signature",
    successMessage: shouldAutoApplyDerivedGroup
      ? `Signature discovery complete. Group by switched to ${payload.derived_column}.`
      : `Signature discovery complete. ${payload.derived_column} was saved but not auto-applied because the top split did not pass the current significance rules.`,
    backgroundMessage: "Signature discovery finished in the background. Open the Markers tab to review the signature ranking.",
  });
}

async function runCox() {
  const base = currentBaseConfig();
  const { covariates, categoricalCovariates, strataColumns } = currentCoxSelections();
  if (!covariates.length) { showToast("Select at least one covariate for the Cox model.", "error"); return; }
  const requestToken = beginRequestToken("cox");
  const datasetId = base.dataset_id;
  const loading = beginShellLoading([refs.coxResultsShell]);
  let payload;
  try {
    payload = await fetchJSON("/api/cox", {
      signal: requestSignal("cox"),
      method: "POST",
      body: JSON.stringify({ ...base, covariates, categorical_covariates: categoricalCovariates, strata_columns: strataColumns }),
    });
  } catch (error) {
    if (requestTokenMatches("cox", requestToken)) loading.restore();
    throw error;
  }
  if (!requestTokenMatches("cox", requestToken) || state.dataset?.dataset_id !== datasetId) return;
  loading.finish();
  state.cox = payload;
  renderAnalysisConsistencyBanner();
  const coxFigure = payload?.figure || { data: [], layout: {} };
  const coxAnalysis = payload?.analysis || {};
  const coxSummary = coxAnalysis.scientific_summary || null;
  const stats = coxAnalysis.model_stats || {};
  resetPlotElement(refs.coxPlot);
  await Plotly.newPlot(refs.coxPlot, coxFigure.data || [], responsivePlotLayout(refs.coxPlot, coxFigure.layout || {}), plotConfig("cox_forest"));
  markPlotResult(refs.coxPlot, payload);
  stabilizePlotShellHeight(refs.coxPlot);
  stabilizeCoxPlotResetAxes(refs.coxPlot);
  if (payload.diagnostics_figure?.data?.length) {
    resetPlotElement(refs.coxDiagnosticsPlot);
    await Plotly.newPlot(
      refs.coxDiagnosticsPlot,
      payload.diagnostics_figure.data,
      plotLayoutConfig(payload.diagnostics_figure.layout, "cox_diagnostics"),
      plotConfig("cox_diagnostics"),
    );
    stabilizePlotShellHeight(refs.coxDiagnosticsPlot);
  } else {
    clearPlotShell(refs.coxDiagnosticsPlot, '<div class="empty-state plot-empty"><span>Scaled Schoenfeld residual screening was unavailable for this fit.</span></div>');
  }
  await renderCoxMartingalePlot(runtime.coxMartingaleTerm);
  renderTable(refs.coxResultsShell, coxAnalysis.results_table);
  renderTable(refs.coxDiagnosticsShell, coxAnalysis.diagnostics_table, exportColumnsFromRows(coxAnalysis.diagnostics_table));
  renderInsightBoard(refs.coxInsightBoard, coxSummary, "Run Cox PH to review diagnostics.");
  const coxMetricLabel = stats.c_index_label || ((stats.evaluation_mode === "apparent") ? "Apparent C-index" : "C-index");
  const hasReportedCoxMetric = stats.c_index != null && stats.evaluation_mode !== "stratified_not_reported";
  const coxMetricCore = hasReportedCoxMetric
    ? `${coxMetricLabel}=${formatValue(stats.c_index)}`
    : "Discrimination metric omitted for stratified Cox";
  const hasCoxMetricCi = hasReportedCoxMetric && stats.c_index_ci_lower != null && stats.c_index_ci_upper != null;
  const coxMetricCi = hasCoxMetricCi
    ? ` (${Math.round((Number(stats.c_index_ci_level) || 0.95) * 100)}% CI ${formatValue(stats.c_index_ci_lower)} to ${formatValue(stats.c_index_ci_upper)})`
    : "";
  const coxStrataMeta = [];
  if (stats.n_strata != null) coxStrataMeta.push(`strata=${formatValue(stats.n_strata)}`);
  if (Number(stats.zero_event_strata_count || 0) > 0) coxStrataMeta.push(`zero-event strata=${formatValue(stats.zero_event_strata_count)}`);
  if (Number(stats.sparse_event_strata_count || 0) > 0) coxStrataMeta.push(`one-event strata=${formatValue(stats.sparse_event_strata_count)}`);
  const coxStrataCore = coxStrataMeta.length ? `, ${coxStrataMeta.join(", ")}` : "";
  refs.coxMetaBanner.textContent = `N=${formatValue(stats.n)}, events=${formatValue(stats.events)}, parameters=${formatValue(stats.parameters)}, EPV=${formatValue(stats.events_per_parameter)}, ${coxMetricCore}${coxMetricCi}${coxStrataCore}, AIC=${formatValue(stats.aic, { scientificLarge: false })}`;
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("cox", {
    successMessage: "Cox PH model fitted.",
    backgroundMessage: "Cox PH finished in the background. Switch back when you are ready to review the updated model.",
  });
}

async function runCohortTable() {
  const datasetId = state.dataset?.dataset_id || "";
  validateGroupingSelection();
  const vars = selectedCheckboxValues(refs.cohortVariableChecklist);
  if (!vars.length) { showToast("Select at least one variable for the cohort table.", "error"); return; }
  const requestToken = beginRequestToken("tables");
  const loading = beginShellLoading([refs.cohortTableShell]);
  let payload;
  try {
    payload = await fetchJSON("/api/cohort-table", {
      signal: requestSignal("tables"),
      method: "POST",
      body: JSON.stringify({
        dataset_id: state.dataset.dataset_id,
        variables: vars,
        group_column: refs.groupColumn.value || null,
        ...cohortTableOutcomeConfig(),
      }),
    });
  } catch (error) {
    if (requestTokenMatches("tables", requestToken)) loading.restore();
    throw error;
  }
  if (!requestTokenMatches("tables", requestToken) || state.dataset?.dataset_id !== datasetId) return;
  loading.finish();
  state.cohort = payload;
  renderAnalysisConsistencyBanner();
  // Group-level columns are data labels ("Test positive", "pT1a"): shown as they are, never as p-values.
  renderTable(refs.cohortTableShell, payload.analysis.rows, payload.analysis.columns, { pValueColumns: [], rawHeaders: true });
  const tableNotes = cohortTableAnalysisNotes(payload);
  if (tableNotes.length && payload.analysis.rows?.length) {
    const noteEl = document.createElement("p");
    noteEl.className = "comparison-table-note";
    noteEl.textContent = tableNotes.join(" ");
    refs.cohortTableShell.prepend(noteEl);
  }
  renderSharedFeatureSummary();
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("tables", {
    successMessage: "Cohort table built.",
    backgroundMessage: "Cohort table finished in the background. Switch back when you are ready to review the updated table.",
  });
}

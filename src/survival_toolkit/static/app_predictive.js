// SurvStudio front end, part 5/8: Variable selections, predictive model UI, benchmark chrome, workspace chrome, and presets.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

function setShimmer(shell) {
  resetPlotElement(shell, '<div class="shimmer"><div class="shimmer-bar"></div><div class="shimmer-bar short"></div><div class="shimmer-bar"></div></div>');
}

function beginShellLoading(shells = []) {
  // Show a loading state while keeping enough of the previous UI to restore it if the run fails.
  const snapshots = shells.filter(Boolean).map((shell) => {
    if (shell.data?.length && shell._fullLayout) {
      shell.classList.add("is-refreshing");
      shell.setAttribute("aria-busy", "true");
      return { shell, plot: true, html: "" };
    }
    const snapshot = { shell, plot: false, html: shell.innerHTML, plotState: shell.dataset.plotState };
    setShimmer(shell);
    return snapshot;
  });
  const settle = () => snapshots.forEach(({ shell }) => {
    shell.classList.remove("is-refreshing");
    shell.removeAttribute("aria-busy");
  });
  return {
    finish: settle,
    restore() {
      settle();
      snapshots.forEach(({ shell, plot, html, plotState }) => {
        if (plot) return;
        shell.innerHTML = html;
        if (plotState !== undefined) shell.dataset.plotState = plotState;
      });
    },
  };
}

function cohortColumnsExcluding(...excluded) {
  const excludedSet = new Set(excluded.filter(Boolean));
  const idPatterns = /^(patient_id|sample_id|subject_id|id|barcode|uuid|cohort)$/i;
  return state.dataset.columns
    .filter((col) => {
      if (excludedSet.has(col.name)) return false;
      if (idPatterns.test(col.name)) return false;
      return true;
    })
    .map((col) => col.name);
}

function isOutcomeDerivedGroupingColumn(columnName) {
  const normalized = String(columnName || "").toLowerCase();
  if (runtime.derivedColumnProvenance?.[columnName]?.outcomeInformed) return true;
  return (
    normalized.endsWith("__optimal_cutpoint")
    || normalized === "auto_signature_group"
    || normalized.startsWith("sig_")
  );
}

function isSurvivalOutcomeLikeColumn(columnName) {
  if (!state.dataset || !columnName) return false;
  if (columnName === refs.timeColumn?.value || columnName === refs.eventColumn?.value) return true;

  const suggestedTimeColumns = new Set(state.dataset?.suggestions?.time_columns || []);
  if (suggestedTimeColumns.has(columnName)) return true;

  const binarySet = new Set(state.dataset?.binary_candidate_columns || []);
  if (binarySet.has(columnName) && isEventLikeColumnName(columnName) && !looksLikeBaselineStatusColumn(columnName)) {
    return true;
  }
  return false;
}

function modelFeatureCandidateColumns() {
  return cohortColumnsExcluding(refs.timeColumn?.value, refs.eventColumn?.value)
    .filter((name) => !isSurvivalOutcomeLikeColumn(name))
    .filter((name) => !isOutcomeDerivedGroupingColumn(name));
}

function sharedModelCategoricalCandidates() {
  if (!state.dataset) return [];
  const availableFeatures = new Set(modelFeatureCandidateColumns());
  return state.dataset.columns
    .filter((column) => availableFeatures.has(column.name))
    .filter((column) => ["categorical", "binary"].includes(column.kind) || (column.n_unique != null && column.n_unique <= AUTO_CATEGORICAL_UNIQUE_THRESHOLD))
    .map((column) => column.name);
}

function refreshVariableSelections() {
  if (!state.dataset) return;
  const availableCovariates = modelFeatureCandidateColumns();
  const previousCovariates = selectedCheckboxValues(refs.covariateChecklist).filter((v) => availableCovariates.includes(v));
  const previousCategoricals = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => availableCovariates.includes(v));
  const previousStrata = selectedCheckboxValues(refs.strataChecklist).filter((v) => availableCovariates.includes(v));
  const previousModelFeatures = selectedCheckboxValues(refs.modelFeatureChecklist).filter((v) => availableCovariates.includes(v));
  const previousModelCategoricals = selectedCheckboxValues(refs.modelCategoricalChecklist).filter((v) => availableCovariates.includes(v));
  const previousDlModelCategoricals = selectedCheckboxValues(refs.dlModelCategoricalChecklist).filter((v) => availableCovariates.includes(v));
  const previousTableVars = selectedCheckboxValues(refs.cohortVariableChecklist).filter((v) => availableCovariates.includes(v));
  const defaultCategoricals = state.dataset.columns
    .filter((c) => ["categorical", "binary"].includes(c.kind) || (c.n_unique != null && c.n_unique <= AUTO_CATEGORICAL_UNIQUE_THRESHOLD))
    .map((c) => c.name)
    .filter((name) => availableCovariates.includes(name));
  const defaultModelFeatures = availableCovariates.slice(0, DEFAULT_MODEL_FEATURE_SELECTION_LIMIT);
  renderChecklist(refs.covariateChecklist, availableCovariates, previousCovariates.length ? previousCovariates : availableCovariates.slice(0, 4));
  renderChecklist(refs.categoricalChecklist, availableCovariates, previousCategoricals.length ? previousCategoricals : defaultCategoricals);
  renderChecklist(refs.strataChecklist, availableCovariates, previousStrata);
  renderChecklist(refs.modelFeatureChecklist, availableCovariates, previousModelFeatures.length ? previousModelFeatures : defaultModelFeatures);
  renderChecklist(refs.modelCategoricalChecklist, availableCovariates, previousModelCategoricals.length ? previousModelCategoricals : defaultCategoricals);
  renderChecklist(refs.dlModelFeatureChecklist, availableCovariates, previousModelFeatures.length ? previousModelFeatures : defaultModelFeatures);
  renderChecklist(refs.dlModelCategoricalChecklist, availableCovariates, previousDlModelCategoricals.length ? previousDlModelCategoricals : defaultCategoricals);
  renderChecklist(refs.cohortVariableChecklist, availableCovariates, previousTableVars.length ? previousTableVars : availableCovariates.slice(0, 6));
  const numericOptions = state.dataset.numeric_columns.filter((c) => !isSurvivalOutcomeLikeColumn(c));
  renderSelect(refs.deriveSource, numericOptions, { selected: numericOptions.includes(refs.deriveSource.value) ? refs.deriveSource.value : numericOptions[0] || null });
  refreshMarkerSelections();
  renderSharedFeatureSummary();
  renderCoxPreviewLine();
}

function setSharedModelFeatureSelection(nextFeatures = [], { clearCategoricals = false } = {}) {
  const availableFeatures = modelFeatureCandidateColumns();
  const normalizedFeatures = nextFeatures.filter((value) => availableFeatures.includes(value));
  const autoCategoricalCandidates = new Set(sharedModelCategoricalCandidates());
  const preservedMlCategoricals = clearCategoricals
    ? []
    : selectedCheckboxValues(refs.modelCategoricalChecklist).filter((value) => normalizedFeatures.includes(value));
  const preservedDlCategoricals = clearCategoricals
    ? []
    : selectedCheckboxValues(refs.dlModelCategoricalChecklist).filter((value) => normalizedFeatures.includes(value));
  normalizedFeatures.forEach((value) => {
    if (autoCategoricalCandidates.has(value) && !preservedMlCategoricals.includes(value)) {
      preservedMlCategoricals.push(value);
    }
    if (autoCategoricalCandidates.has(value) && !preservedDlCategoricals.includes(value)) {
      preservedDlCategoricals.push(value);
    }
  });

  setCheckedValues(refs.modelFeatureChecklist, normalizedFeatures);
  setCheckedValues(refs.dlModelFeatureChecklist, normalizedFeatures);
  setCheckedValues(refs.modelCategoricalChecklist, preservedMlCategoricals);
  setCheckedValues(refs.dlModelCategoricalChecklist, preservedDlCategoricals);
  renderSharedFeatureSummary();
  queueHistorySync();
}

function syncChecklistSelections(sourceContainer, targetContainer) {
  if (!sourceContainer || !targetContainer) return;
  setCheckedValues(targetContainer, selectedCheckboxValues(sourceContainer));
}

function normalizeModelCategoricalSelection(
  categoricalChecklist,
  featureValues,
  { autoSelectCandidates = false } = {},
) {
  if (!categoricalChecklist) return;
  const normalizedCategoricals = selectedCheckboxValues(categoricalChecklist)
    .filter((value) => featureValues.includes(value));
  if (autoSelectCandidates) {
    const autoCategoricalCandidates = new Set(sharedModelCategoricalCandidates());
    featureValues.forEach((value) => {
      if (autoCategoricalCandidates.has(value) && !normalizedCategoricals.includes(value)) {
        normalizedCategoricals.push(value);
      }
    });
  }
  setCheckedValues(categoricalChecklist, normalizedCategoricals);
}

function syncModelFeatureMirrors(sourceContainer = refs.modelFeatureChecklist) {
  const counterpart = sourceContainer === refs.dlModelFeatureChecklist
    ? refs.modelFeatureChecklist
    : refs.dlModelFeatureChecklist;
  syncChecklistSelections(sourceContainer, counterpart);
  const featureValues = selectedCheckboxValues(sourceContainer);
  normalizeModelCategoricalSelection(refs.modelCategoricalChecklist, featureValues, { autoSelectCandidates: true });
  normalizeModelCategoricalSelection(refs.dlModelCategoricalChecklist, featureValues, { autoSelectCandidates: true });
}

function syncModelCategoricalMirrors(sourceContainer = refs.modelCategoricalChecklist) {
  const featureChecklist = sourceContainer === refs.dlModelCategoricalChecklist
    ? refs.dlModelFeatureChecklist
    : refs.modelFeatureChecklist;
  normalizeModelCategoricalSelection(sourceContainer, selectedCheckboxValues(featureChecklist));
}

function setCheckedValues(container, values) {
  const wanted = new Set(values || []);
  container.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.checked = wanted.has(input.value);
  });
}

function summarizeFeatureNames(values, limit = 4) {
  if (!values.length) return "none selected";
  if (values.length <= limit) return values.join(", ");
  return `${values.slice(0, limit).join(", ")} +${values.length - limit} more`;
}

function datasetPresetForCurrentDataset() {
  const presetName = String(state.dataset?.preset_name || "").trim();
  return presetName ? (DATASET_PRESETS[presetName] || null) : null;
}

function renderChipList(container, chips = []) {
  if (!container) return;
  container.innerHTML = "";
  if (!chips.length) {
    container.classList.add("hidden");
    return;
  }
  chips.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "dataset-preset-chip";
    chip.textContent = label;
    container.appendChild(chip);
  });
  container.classList.remove("hidden");
}

function formatOutcomeChip(timeLabel, eventLabel, eventValue) {
  return `Outcome: ${timeLabel} / ${eventLabel}=${eventValue}`;
}

function mlModelLabel(modelType) {
  const labels = {
    rsf: "Random Survival Forest",
    gbs: "Gradient Boosted Survival",
    lasso_cox: "LASSO-Cox",
  };
  return labels[String(modelType || "").toLowerCase()] || String(modelType || "ML model");
}

function dlModelLabel(modelType) {
  const labels = {
    deepsurv: "DeepSurv",
    deephit: "DeepHit",
    mtlr: "Neural MTLR",
    transformer: "Survival Transformer (experimental)",
    vae: "Survival VAE (experimental)",
  };
  return labels[String(modelType || "").toLowerCase()] || String(modelType || "deep model");
}

function predictiveModelMeta(modelKey = currentPredictiveModelKey()) {
  const key = String(modelKey || "").toLowerCase();
  const family = ["rsf", "gbs", "lasso_cox"].includes(key) ? "ml" : "dl";
  return {
    key,
    family,
    label: family === "ml" ? mlModelLabel(key) : dlModelLabel(key),
  };
}

function currentPredictiveModelKey() {
  const family = normalizedPredictiveFamily(runtime.predictiveFamily);
  return family === "ml"
    ? String(refs.mlModelType?.value || "rsf")
    : String(refs.dlModelType?.value || "deepsurv");
}

function selectedPredictiveSingleResult(goal) {
  if (!["ml", "dl"].includes(goal)) return null;
  if (preferredResultMode(goal) !== "single") return null;
  const payload = currentGoalResult(goal);
  if (!payload) return null;
  const requestConfig = payload.request_config || payload.analysis?.request_config || null;
  if (!requestConfig) return null;
  const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
  if (selectedModel.family !== goal) return null;
  const requestModelType = String(requestConfig.model_type || "").toLowerCase();
  return requestModelType === selectedModel.key ? payload : null;
}

function syncPredictiveModelSelector() {
  const currentKey = currentPredictiveModelKey();
  if (refs.predictiveModelSelector && refs.predictiveModelSelector.value !== currentKey) {
    refs.predictiveModelSelector.value = currentKey;
  }
}

function predictiveModelKeyFromComparisonLabel(modelLabel) {
  const normalized = String(modelLabel || "").trim().toLowerCase();
  return {
    "random survival forest": "rsf",
    "gradient boosted survival": "gbs",
    "lasso-cox": "lasso_cox",
    deepsurv: "deepsurv",
    deephit: "deephit",
    "neural mtlr": "mtlr",
    "survival transformer": "transformer",
    "survival vae": "vae",
  }[normalized] || null;
}

function payloadRepresentsCompareRun(payload) {
  const modelType = String(payload?.request_config?.model_type || payload?.analysis?.request_config?.model_type || "").toLowerCase();
  return modelType === "compare";
}

function nextCompareRunGroupId(prefix = "compare") {
  runtime.compareRunSequence = Number(runtime.compareRunSequence || 0) + 1;
  return `${String(prefix || "compare")}-${runtime.compareRunSequence}`;
}

function tagComparePayload(payload, groupId, source = "compare") {
  if (!payload || !payloadRepresentsCompareRun(payload)) return payload;
  const resolvedGroupId = String(groupId || "").trim() || nextCompareRunGroupId("compare");
  const resolvedSource = String(source || "compare");
  payload._client_compare_group_id = resolvedGroupId;
  payload._client_compare_source = resolvedSource;
  if (payload.analysis && typeof payload.analysis === "object") {
    payload.analysis._client_compare_group_id = resolvedGroupId;
    payload.analysis._client_compare_source = resolvedSource;
  }
  return payload;
}

function panelModeForPayload(payload) {
  if (!payload) return "idle";
  return payloadRepresentsCompareRun(payload) ? "compare" : "single";
}

function restorePredictiveFamilyAfterFailedCompare(goal, previousPayload) {
  const panel = goal === "ml" ? refs.mlPanel : refs.dlPanel;
  const previousWasCompare = payloadRepresentsCompareRun(previousPayload);
  const restored = previousWasCompare ? null : (previousPayload || null);
  if (goal === "ml") {
    state.ml = restored;
  } else {
    state.dl = restored;
  }
  // The result mode follows the restored result, so a restored single-model result stays current and shown.
  runtime.resultPreference[goal] = "single";
  setPanelResultMode(panel, previousWasCompare ? "idle" : panelModeForPayload(previousPayload));
}

function benchmarkReviewAction(row) {
  if (row?.excluded) {
    return {
      dataset: {},
      title: row.exclusionReason || "This model was excluded from the current compare run.",
      label: "Excluded",
      disabled: true,
    };
  }
  const modelKey = predictiveModelKeyFromComparisonLabel(row.model);
  if (modelKey) {
    return {
      dataset: {
        benchmarkModel: modelKey,
        benchmarkMode: row.sourceMode || "",
      },
      label: "Train a model",
      disabled: false,
    };
  }
  if (String(row.model || "").trim().toLowerCase() === "cox ph") {
    return {
      dataset: {},
      title: "Cox PH appears here as a screening baseline only. Use the dedicated Cox workspace if you want an inferential Cox run.",
      label: "Screening only",
      disabled: true,
    };
  }
  return {
    dataset: {
      benchmarkTab: row.familyTab,
      benchmarkMode: row.sourceMode || "",
    },
    label: `Open ${row.familyTab.toUpperCase()} controls`,
    disabled: false,
  };
}

function benchmarkParamsPayload(goal, source = "current") {
  if (source === "snapshot") return compareSnapshotPayload(goal) || null;
  return currentCompareGoalPayload(goal) || compareGoalPayload(goal) || null;
}

function benchmarkParamsSummary(goal, modelLabel, source = "current") {
  const payload = benchmarkParamsPayload(goal, source);
  const requestConfig = payload?.request_config || payload?.analysis?.request_config || {};
  if (!requestConfig || !Object.keys(requestConfig).length) {
    return `${modelLabel}: no saved compare-run settings are available for this row yet.`;
  }

  const features = Array.isArray(requestConfig.features) ? requestConfig.features : [];
  const categoricals = Array.isArray(requestConfig.categorical_features) ? requestConfig.categorical_features : [];
  const evaluation = String(requestConfig.evaluation_strategy || "holdout") === "repeated_cv"
    ? `${formatValue(requestConfig.cv_repeats ?? 3)}x${formatValue(requestConfig.cv_folds ?? 5)} repeated CV`
    : "Deterministic Holdout";
  const lockedTestFraction = normalizedLockedTestFraction(requestConfig.locked_test_fraction);

  const parts = [
    `${modelLabel} params`,
    `shared_features=${formatValue(features.length)}`,
    `categoricals=${formatValue(categoricals.length)}`,
    `eval=${evaluation}`,
    ...(lockedTestFraction ? [`locked_test=${formatValue(Math.round(lockedTestFraction * 100))}%`] : []),
  ];

  if (goal === "ml") {
    parts.push(`seed=${formatValue(requestConfig.random_state ?? 42)}`);
    const normalizedModel = String(modelLabel || "").trim().toLowerCase();
    if (normalizedModel === "random survival forest") {
      parts.push(`trees=${formatValue(requestConfig.n_estimators ?? 100)}`);
      parts.push(`max_depth=${formatValue(requestConfig.max_depth || "auto")}`);
    } else if (normalizedModel === "gradient boosted survival") {
      parts.push(`trees=${formatValue(requestConfig.n_estimators ?? 100)}`);
      parts.push(`lr=${formatValue(requestConfig.learning_rate ?? 0.1)}`);
      parts.push(`max_depth=${formatValue(requestConfig.max_depth || "auto")}`);
    } else if (normalizedModel === "lasso-cox") {
      parts.push("alpha=fit on the training split");
    } else if (normalizedModel === "cox ph") {
      parts.push("baseline screening fit");
    }
    return `${parts.join(" | ")}.`;
  }

  parts.push(`seed=${formatValue(requestConfig.random_seed ?? 42)}`);
  parts.push(`hidden=${(requestConfig.hidden_layers || [64, 64]).join("/")}`);
  parts.push(`dropout=${formatValue(requestConfig.dropout ?? 0.1)}`);
  parts.push(`lr=${formatValue(requestConfig.learning_rate ?? 0.001)}`);
  parts.push(`epochs=${formatValue(requestConfig.epochs ?? 100)}`);
  parts.push(`early_stop=${formatValue(requestConfig.early_stopping_patience ?? 10)}/${formatValue(requestConfig.early_stopping_min_delta ?? 0.0001)}`);

  const normalizedModel = String(modelLabel || "").trim().toLowerCase();
  if (normalizedModel === "deephit" || normalizedModel === "neural mtlr") {
    parts.push(`batch=${formatValue(requestConfig.batch_size ?? 64)}`);
    parts.push(`time_bins=${formatValue(requestConfig.num_time_bins ?? 50)}`);
  } else if (normalizedModel === "survival transformer") {
    parts.push(`width=${formatValue(requestConfig.d_model ?? 64)}`);
    parts.push(`heads=${formatValue(requestConfig.n_heads ?? 4)}`);
    parts.push(`layers=${formatValue(requestConfig.n_layers ?? 2)}`);
  } else if (normalizedModel === "survival vae") {
    parts.push(`latent=${formatValue(requestConfig.latent_dim ?? 8)}`);
    parts.push(`clusters=${formatValue(requestConfig.n_clusters ?? 3)}`);
  }
  return `${parts.join(" | ")}.`;
}

function showBenchmarkParams(goal, modelLabel, source = "current") {
  showToast(benchmarkParamsSummary(goal, modelLabel, source), "info", 7600);
}

function mlModelSupportsShap(modelType) {
  return ["rsf", "gbs"].includes(String(modelType || "").toLowerCase());
}

function mlPendingBannerText({ modelType, nEstimators, rowCount, computeShap }) {
  const label = mlModelLabel(modelType);
  const treeSuffix = ["rsf", "gbs"].includes(String(modelType || "").toLowerCase()) && Number.isFinite(nEstimators)
    ? ` with ${nEstimators} trees`
    : "";
  const cohortSuffix = Number.isFinite(rowCount) ? ` on ${rowCount} rows` : "";
  let message = `Training ${label}${treeSuffix}${cohortSuffix}.`;
  if (modelType === "lasso_cox") {
    message += " This penalized Cox path can still take longer on wide feature sets because the training split tunes its penalty internally.";
  } else if (modelType === "rsf" && Number(nEstimators) >= 100 && Number(rowCount) >= 500) {
    message += " This can take longer on a local CPU for real cohorts.";
  } else {
    message += " This usually finishes quickly on small cohorts.";
  }
  message += mlModelSupportsShap(modelType) && computeShap
    ? " SHAP is computed after fitting and can add a short delay."
    : (mlModelSupportsShap(modelType)
      ? " Fast mode is on, so SHAP will be skipped for a faster result."
      : " SHAP is currently available for tree models only.");
  if (mlModelSupportsShap(modelType) && computeShap && refs.mlShapSafeMode?.checked) {
    message += " If the encoded matrix is too wide, SHAP safe mode will refit a reduced companion model for explanation only.";
  }
  return message;
}

function mlComparePendingBannerText({ rowCount, evaluationStrategy, cvFolds, cvRepeats }) {
  const cohortSuffix = Number.isFinite(rowCount) ? ` on ${rowCount} rows` : "";
  const evalSuffix = evaluationStrategy === "repeated_cv"
    ? ` using ${cvRepeats}x${cvFolds} repeated CV`
    : " using deterministic holdout";
  return `Screening Cox PH and, when available, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival${cohortSuffix}${evalSuffix}.`;
}

function dlPendingBannerText({ modelType, rowCount, epochs, evaluationStrategy, cvFolds, cvRepeats }) {
  const label = dlModelLabel(modelType);
  const cohortSuffix = Number.isFinite(rowCount) ? ` on ${rowCount} rows` : "";
  const evalSuffix = evaluationStrategy === "repeated_cv"
    ? ` with ${cvRepeats}x${cvFolds} repeated CV`
    : " with deterministic holdout";
  let message = `Training ${label}${cohortSuffix}${evalSuffix} for up to ${epochs} epochs.`;
  if (Number.isFinite(Number(epochs)) && Number(epochs) >= 200) {
    message += " Early stopping may finish before the requested epoch limit.";
  } else {
    message += " Early stopping can still end training before the epoch limit.";
  }
  if (Number(rowCount) >= 10000 && (modelType === "deepsurv" || modelType === "transformer")) {
    message += " This full-batch objective can run out of memory on larger cohorts, so start smaller if local RAM is limited.";
  }
  return message;
}

function dlComparePendingBannerText({ rowCount, evaluationStrategy, cvFolds, cvRepeats }) {
  const cohortSuffix = Number.isFinite(rowCount) ? ` on ${rowCount} rows` : "";
  const evalSuffix = evaluationStrategy === "repeated_cv"
    ? ` using ${cvRepeats}x${cvFolds} repeated CV`
    : " using deterministic holdout";
  return `Comparing DeepSurv, DeepHit, Neural MTLR, Survival Transformer, and Survival VAE${cohortSuffix}${evalSuffix}.`;
}

function formatGroupChip(groupLabel) {
  return `Grouping only: ${groupLabel}`;
}

function renderContextCards({
  hasDataset,
  timeLabel,
  eventLabel,
  eventValue,
  groupLabel,
  coxFeatures,
  coxCategoricals,
  coxStrata,
  modelFeatures,
  modelCategoricals,
  tableVariables,
}) {
  if (refs.groupingSummaryText) {
    refs.groupingSummaryText.textContent = !hasDataset
      ? "Used mainly for Kaplan-Meier and grouped tables."
      : (refs.groupColumn?.value ? `Curves and Table 1 are split by ${groupLabel}.` : "Choose a column to compare groups, or make groups from a number.");
  }
  if (refs.groupColumnWarning) {
    const warning = hasDataset ? currentGroupColumnWarning() : null;
    if (!warning) {
      refs.groupColumnWarning.textContent = "";
      refs.groupColumnWarning.className = "event-warning hidden";
    } else {
      refs.groupColumnWarning.textContent = warning.message;
      refs.groupColumnWarning.className = `event-warning event-warning-${warning.tone}`;
    }
  }

  if (refs.tableOutputStatusText) {
    const tableState = currentCohortTableOutputState();
    if (!hasDataset || !tableState.hasOutput || tableState.isCurrent) {
      refs.tableOutputStatusText.textContent = "";
      refs.tableOutputStatusText.classList.add("hidden");
    } else {
      refs.tableOutputStatusText.textContent = `Settings changed since this table was built (${tableState.outputVariables.length} variables, group: ${tableState.outputGroupLabel}). Rebuild it to update.`;
      refs.tableOutputStatusText.classList.remove("hidden");
    }
  }
  updateCohortTableButtonLabel();
}

function syncDownloadButtonAvailability() {
  const currentKm = currentGoalResult("km");
  const currentSignature = currentSignatureResult();
  const currentCox = currentGoalResult("cox");
  const currentMl = currentGoalResult("ml");
  const currentDl = currentGoalResult("dl");
  const currentTable = currentCohortTableOutputState();

  refs.downloadKmSummaryButton.disabled = !currentKm;
  refs.downloadKmPairwiseButton.disabled = !currentKm || !(currentKm.analysis?.pairwise_table?.length);
  const kmPlotCurrent = plotShowsResult(refs.kmPlot, currentKm);
  if (refs.downloadKmPngButton) refs.downloadKmPngButton.disabled = !kmPlotCurrent;
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.disabled = !kmPlotCurrent;
  refs.downloadSignatureButton.disabled = !currentSignature || !(currentSignature.results_table?.length);
  refs.downloadCoxResultsButton.disabled = !currentCox;
  refs.downloadCoxDiagnosticsButton.disabled = !currentCox;
  const coxPlotCurrent = plotShowsResult(refs.coxPlot, currentCox);
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.disabled = !coxPlotCurrent;
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.disabled = !coxPlotCurrent;
  refs.downloadCohortTableButton.disabled = !currentTable.hasOutput;
  if (refs.downloadCohortTableXlsxButton) refs.downloadCohortTableXlsxButton.disabled = !currentTable.hasOutput;
  refs.downloadMlComparisonButton.disabled = !currentMl || !(currentMl.analysis?.comparison_table?.length);
  const mlComparisonPlotCurrent = plotShowsResult(refs.mlComparisonPlot, currentMl);
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = !mlComparisonPlotCurrent;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = !mlComparisonPlotCurrent;
  setMlManuscriptDownloadsEnabled(Boolean(currentMl?.analysis?.manuscript_tables?.model_performance_table?.length));
  refs.downloadDlComparisonButton.disabled = !currentDl || !(currentDl.analysis?.comparison_table?.length);
  const dlComparisonPlotCurrent = plotShowsResult(refs.dlComparisonPlot, currentDl);
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = !dlComparisonPlotCurrent;
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = !dlComparisonPlotCurrent;
  setDlManuscriptDownloadsEnabled(Boolean(currentDl?.analysis?.manuscript_tables?.model_performance_table?.length));
  syncMarkerDownloadButtons();
}

function endpointReadinessMessage() {
  if (!state.dataset) return "Load a dataset first.";
  try {
    currentBaseConfig();
    return "";
  } catch (error) {
    return error?.message || "Complete the outcome definition first.";
  }
}

function setActionDisabledState(button, disabled, title = "") {
  if (!button) return;
  button.disabled = Boolean(disabled);
  button.setAttribute("aria-disabled", String(Boolean(disabled)));
  button.title = title;
}

function syncAnalysisRunButtonAvailability() {
  const endpointReady = endpointIsReady();
  const readyMessage = endpointReadinessMessage();
  const coxCovariateCount = goalFeatureCount("cox");
  const tableVariableCount = goalFeatureCount("tables");
  const sharedFeatureCount = selectedCheckboxValues(refs.modelFeatureChecklist).length;
  const hasCoxCovariates = coxCovariateCount > 0;
  const hasSharedFeatures = sharedFeatureCount > 0;
  const hasTableVariables = tableVariableCount > 0;
  // An attached marker matrix supplies the markers for the evaluation, not for the cut-point search, which
  // reads the marker checklist (disabled, but still ticked, while a matrix is attached).
  const matrixAttached = typeof markerMatrixAttached === "function" && markerMatrixAttached();
  const signatureFeatureMessage = matrixAttached
    ? "The cut-point search uses markers from the checklist, not an attached marker file. Remove the file to search the checklist markers."
    : "Select at least one marker to search for cut-point combinations.";
  const markerCount = typeof currentMarkerSelections === "function"
    ? currentMarkerSelections().markers.length
    : selectedCheckboxValues(refs.markerChecklist).length;
  const hasMarkers = markerCount > 0;
  const hasEvaluationMarkers = hasMarkers || matrixAttached;
  const coxFeatureMessage = "Select at least one covariate for the Cox model.";
  const sharedFeatureMessage = "Select at least one shared ML/DL model feature.";
  const tableVariableMessage = "Select at least one variable for the cohort table.";
  const mlRepeatedCv = refs.mlEvaluationStrategy?.value === "repeated_cv";
  const mlSingleMessage = mlRepeatedCv
    ? "Run Analysis uses deterministic holdout only. Use Compare All for repeated CV screening."
    : "";

  setActionDisabledState(
    refs.runKmButton,
    !endpointReady || isScopeBusy("km"),
    endpointReady ? "" : readyMessage,
  );
  setActionDisabledState(
    refs.runSignatureSearchButton,
    !endpointReady || !hasMarkers || isScopeBusy("km"),
    !endpointReady ? readyMessage : (!hasMarkers ? signatureFeatureMessage : ""),
  );
  setActionDisabledState(
    refs.runMarkersButton,
    !endpointReady || !hasEvaluationMarkers || isScopeBusy("markers"),
    !endpointReady ? readyMessage : (!hasEvaluationMarkers ? "Choose at least one marker or attach a marker file." : ""),
  );
  setActionDisabledState(
    refs.runCoxButton,
    !endpointReady || !hasCoxCovariates || isScopeBusy("cox"),
    !endpointReady ? readyMessage : (!hasCoxCovariates ? coxFeatureMessage : ""),
  );
  setActionDisabledState(
    refs.runCohortTableButton,
    !endpointReady || !hasTableVariables || isScopeBusy("tables"),
    !endpointReady ? readyMessage : (!hasTableVariables ? tableVariableMessage : ""),
  );

  const mlSingleDisabled = !endpointReady || !hasSharedFeatures || mlRepeatedCv || isScopeBusy("ml");
  const mlSingleTitle = !endpointReady
    ? readyMessage
    : (!hasSharedFeatures ? sharedFeatureMessage : mlSingleMessage);
  setActionDisabledState(refs.runMlButton, mlSingleDisabled, mlSingleTitle);

  const mlCompareDisabled = !endpointReady || !hasSharedFeatures || isScopeBusy("ml");
  const mlCompareTitle = !endpointReady
    ? readyMessage
    : (!hasSharedFeatures ? sharedFeatureMessage : "");
  setActionDisabledState(refs.runCompareButton, mlCompareDisabled, mlCompareTitle);
  setActionDisabledState(refs.runCompareInlineButton, mlCompareDisabled, mlCompareTitle);

  const dlSingleDisabled = !endpointReady || !hasSharedFeatures || isScopeBusy("dl");
  const dlSingleTitle = !endpointReady
    ? readyMessage
    : (!hasSharedFeatures ? sharedFeatureMessage : "");
  setActionDisabledState(refs.runDlButton, dlSingleDisabled, dlSingleTitle);

  const dlCompareDisabled = !endpointReady || !hasSharedFeatures || isScopeBusy("dl");
  const dlCompareTitle = !endpointReady
    ? readyMessage
    : (!hasSharedFeatures ? sharedFeatureMessage : "");
  setActionDisabledState(refs.runDlCompareButton, dlCompareDisabled, dlCompareTitle);
  setActionDisabledState(refs.runDlCompareInlineButton, dlCompareDisabled, dlCompareTitle);

  const selectedPredictiveModel = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey());
  const predictiveBusy = isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl");
  const predictiveSelectedDisabled = predictiveBusy || (selectedPredictiveModel.family === "ml" ? mlSingleDisabled : dlSingleDisabled);
  const predictiveSelectedTitle = predictiveBusy
    ? "Wait for the current predictive comparison to finish."
    : (selectedPredictiveModel.family === "ml" ? mlSingleTitle : dlSingleTitle);
  setActionDisabledState(refs.runPredictiveSelectedButton, predictiveSelectedDisabled, predictiveSelectedTitle);
  setActionDisabledState(refs.runPredictiveWorkbenchButton, predictiveSelectedDisabled, predictiveSelectedTitle);
  setActionDisabledState(
    refs.predictiveModelSelector,
    predictiveBusy,
    predictiveBusy ? "Wait for the current predictive run to finish." : "",
  );

  const predictiveCompareDisabled = !endpointReady || !hasSharedFeatures || predictiveBusy;
  const predictiveCompareTitle = !endpointReady
    ? readyMessage
    : (!hasSharedFeatures ? sharedFeatureMessage : (predictiveBusy ? "Wait for the current predictive run to finish." : ""));
  setActionDisabledState(refs.runPredictiveCompareAllButton, predictiveCompareDisabled, predictiveCompareTitle);
}

function renderSharedFeatureSummary() {
  const hasDataset = Boolean(state.dataset);
  syncCoxCovariateSelection();
  const { covariates: coxFeatures, categoricalCovariates: coxCategoricals, strataColumns: coxStrata } = hasDataset
    ? currentCoxSelections()
    : { covariates: [], categoricalCovariates: [], strataColumns: [] };
  const features = hasDataset ? selectedCheckboxValues(refs.modelFeatureChecklist) : [];
  const mlCategoricals = hasDataset ? selectedCheckboxValues(refs.modelCategoricalChecklist).filter((value) => features.includes(value)) : [];
  const dlCategoricals = hasDataset ? selectedCheckboxValues(refs.dlModelCategoricalChecklist).filter((value) => features.includes(value)) : [];
  const tableVariables = hasDataset ? selectedCheckboxValues(refs.cohortVariableChecklist) : [];
  const timeLabel = hasDataset ? (refs.timeColumn?.value || "time") : "time";
  const eventLabel = hasDataset ? (refs.eventColumn?.value || "event") : "event";
  const eventValue = hasDataset ? (refs.eventPositiveValue?.value || "choose event value") : "choose event value";
  const groupLabel = hasDataset ? (refs.groupColumn?.value || "overall only") : "overall only";
  const mlSummaryText = !hasDataset
    ? "Load a dataset first. ML uses the shared model feature selections shown here."
    : features.length
      ? `ML and DL share this model feature list: ${summarizeFeatureNames(features)}. ML categorical handling stays local to this tab. Compare All uses the Evaluation section for cross-model screening only. Group by is shown here for context only.`
      : "No model feature set selected yet. Choose ML/DL model features before training.";
  const dlSummaryText = !hasDataset
    ? "Load a dataset first. DL uses the same shared model feature selections shown in this workspace."
    : features.length
      ? `Training inputs come only from the shared ML/DL model feature selections: ${summarizeFeatureNames(features)}. DL categorical handling stays local to this tab. Group by is shown here for context only.`
      : "No model feature set selected yet. Choose ML/DL model features before training.";
  const sharedChips = !hasDataset
    ? []
    : [
        formatOutcomeChip(timeLabel, eventLabel, eventValue),
        formatGroupChip(groupLabel),
        `Model features: ${features.length}`,
        features.length ? `Preview: ${summarizeFeatureNames(features)}` : "Preview: none selected",
      ];
  const mlChips = !hasDataset
    ? []
    : [
        ...sharedChips,
        `Categorical: ${mlCategoricals.length}`,
        mlCategoricals.length ? `Categoricals: ${summarizeFeatureNames(mlCategoricals, 3)}` : "Categoricals: none",
      ];
  const dlChips = !hasDataset
    ? []
    : [
        ...sharedChips,
        `Categorical: ${dlCategoricals.length}`,
        dlCategoricals.length ? `Categoricals: ${summarizeFeatureNames(dlCategoricals, 3)}` : "Categoricals: none",
      ];

  if (refs.mlFeatureSummaryText) refs.mlFeatureSummaryText.textContent = mlSummaryText;
  if (refs.dlFeatureSummaryText) refs.dlFeatureSummaryText.textContent = dlSummaryText;
  renderChipList(refs.mlFeatureSummaryChips, mlChips);
  renderChipList(refs.dlFeatureSummaryChips, dlChips);

  renderContextCards({
    hasDataset,
    timeLabel,
    eventLabel,
    eventValue,
    groupLabel,
    coxFeatures,
    coxCategoricals,
    coxStrata,
    modelFeatures: features,
    modelCategoricals: mlCategoricals,
    tableVariables,
  });
  syncDownloadButtonAvailability();
  syncAnalysisRunButtonAvailability();
  renderBenchmarkBoard();
  renderWorkspaceChrome();
}

// Single-model plots of a result that no longer matches the settings are marked stale, not deleted:
// undoing the edit (or switching the model back) shows them again unchanged.
function syncStaleSingleResultArtifacts() {
  const mlPayload = goalPayload("ml");
  const mlStale = Boolean(mlPayload && panelModeForPayload(mlPayload) === "single" && !currentGoalResult("ml"));
  setPlotStale(refs.mlImportancePlot, mlStale, "Model settings changed since this run. Run Analysis to refresh feature importance.");
  setPlotStale(refs.mlShapPlot, mlStale, "Model settings changed since this run. Run Analysis to refresh SHAP.");

  const dlPayload = goalPayload("dl");
  const dlStale = Boolean(dlPayload && panelModeForPayload(dlPayload) === "single" && !currentGoalResult("dl"));
  setPlotStale(refs.dlImportancePlot, dlStale, "Model settings changed since this run. Train this deep model again to refresh feature salience.");
  setPlotStale(refs.dlLossPlot, dlStale, "Model settings changed since this run. Train this deep model again to refresh learning curves.");

  updateResultVisibility();
  syncPredictiveWorkbenchSingleResultVisibility();
}

function benchmarkGoalMeta(goal) {
  return goal === "ml"
    ? { label: "Classical ML", tab: "ml", panel: refs.mlPanel, reviewLabel: "Open model" }
    : { label: "Deep Learning", tab: "dl", panel: refs.dlPanel, reviewLabel: "Open model" };
}

function syncBenchmarkWorkbenchVisibility() {
  const workbenchOpen = Boolean(runtime.workbenchRevealed);
  refs.benchmarkSummaryGrid?.classList.toggle("hidden", workbenchOpen);
  refs.benchmarkComparisonPlot?.closest(".table-card")?.classList.toggle("hidden", workbenchOpen);
  refs.benchmarkComparisonShell?.closest(".table-card")?.classList.toggle("hidden", workbenchOpen);
  refs.benchmarkWorkbench?.classList.toggle("hidden", !workbenchOpen);
  refs.runPredictiveCompareAllButton?.classList.toggle("hidden", workbenchOpen);
  refs.openPredictiveWorkbenchButton?.classList.toggle("hidden", workbenchOpen);
  refs.mlModelType?.closest(".model-choice-field")?.classList.toggle("hidden", workbenchOpen);
  refs.dlModelType?.closest(".model-choice-field")?.classList.toggle("hidden", workbenchOpen);
  refs.runCompareButton?.classList.toggle("hidden", workbenchOpen);
  refs.runCompareInlineButton?.classList.toggle("hidden", workbenchOpen);
  refs.runDlCompareButton?.classList.toggle("hidden", workbenchOpen);
  refs.runDlCompareInlineButton?.classList.toggle("hidden", workbenchOpen);
  refs.runMlButton?.classList.remove("hidden");
  refs.runDlButton?.classList.remove("hidden");
  refs.predictiveModelSelector?.closest(".predictive-model-picker")?.classList.toggle("hidden", !workbenchOpen);
  refs.runPredictiveSelectedButton?.classList.add("hidden");
  refs.runPredictiveWorkbenchButton?.classList.toggle("hidden", !workbenchOpen);
}

function renderPredictiveWorkbench() {
  const family = normalizedPredictiveFamily(runtime.predictiveFamily);
  const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
  const familyMode = preferredResultMode(family);
  const unifiedWorkspaceActive = activeTabName() === "benchmark";
  runtime.predictiveFamily = family;

  refs.benchmarkMlMount?.classList.toggle("hidden", family !== "ml");
  refs.benchmarkDlMount?.classList.toggle("hidden", family !== "dl");
  refs.benchmarkWorkbench?.setAttribute("data-active-family", family);
  refs.closePredictiveWorkbenchButton?.classList.remove("hidden");
  syncPredictiveModelSelector();
  syncBenchmarkWorkbenchVisibility();

  if (refs.benchmarkWorkbenchCaption) {
    refs.benchmarkWorkbenchCaption.textContent = runtime.workbenchRevealed
      ? `${selectedModel.label} is selected. Train this model directly with the controls below.`
      : (unifiedWorkspaceActive && familyMode === "compare"
        ? `${selectedModel.label} is selected. Cross-family Compare All results stay in the unified chart and leaderboard above. Use Test ${selectedModel.label} when you want model-specific outputs below.`
        : `${selectedModel.label} is selected. The workbench below shows the exact controls for the ${family === "ml" ? "classical ML" : "deep-learning"} family.`);
  }
  if (refs.predictiveActionStatusText) {
    refs.predictiveActionStatusText.textContent = runtime.workbenchRevealed
      ? `Train ${selectedModel.label} directly with the controls below.`
      : "Fits every model on the same splits and ranks them by C-index. Click a result to tune that model.";
  }
  if (refs.runPredictiveSelectedButton) {
    refs.runPredictiveSelectedButton.textContent = `Train ${selectedModel.label}`;
  }
  if (refs.runPredictiveWorkbenchButton) {
    refs.runPredictiveWorkbenchButton.textContent = `Train ${selectedModel.label}`;
  }
  syncPredictiveWorkbenchCompareVisibility();
  syncPredictiveWorkbenchSingleResultVisibility();
}

function setPredictiveWorkbenchFamily(family, { syncHistory = true, historyMode = "replace", scrollIntoView = false } = {}) {
  runtime.predictiveFamily = normalizedPredictiveFamily(family);
  renderPredictiveWorkbench();
  if (scrollIntoView) {
    requestAnimationFrame(() => {
      (runtime.predictiveFamily === "ml" ? refs.benchmarkMlMount : refs.benchmarkDlMount)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }
  scheduleVisiblePlotResize(40);
  if (state.dataset && syncHistory) syncHistoryState(historyMode);
}

function setPredictiveModel(modelKey, { syncHistory = true, historyMode = "replace", scrollIntoView = false } = {}) {
  const meta = predictiveModelMeta(modelKey);
  runtime.predictiveFamily = meta.family;
  if (meta.family === "ml") {
    setSelectValueIfPresent(refs.mlModelType, meta.key);
    updateMlModelControlVisibility();
  } else {
    setSelectValueIfPresent(refs.dlModelType, meta.key);
    updateDlModelControlVisibility();
  }
  renderPredictiveWorkbench();
  if (scrollIntoView) {
    requestAnimationFrame(() => {
      (meta.family === "ml" ? refs.benchmarkMlMount : refs.benchmarkDlMount)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }
  if (syncHistory) {
    queueHistorySync();
    if (state.dataset) syncHistoryState(historyMode);
  }
}

function benchmarkResultTone(goal) {
  const comparePayload = compareGoalPayload(goal);
  if (comparePayload) return currentCompareGoalPayload(goal) ? "current" : "stale";
  const payload = goalPayload(goal);
  if (!payload) return "idle";
  return currentGoalResult(goal) ? "current" : "stale";
}

function benchmarkResultLabel(goal) {
  const tone = benchmarkResultTone(goal);
  if (tone === "current") return "Current";
  if (tone === "stale") return "Stale";
  return "Not run";
}

function benchmarkPanelMode(goal) {
  return benchmarkGoalMeta(goal).panel?.dataset?.resultMode || "idle";
}

function benchmarkEvaluationLabel(mode) {
  const labels = {
    holdout: "Holdout",
    apparent: "Apparent",
    repeated_cv: "Repeated CV",
    repeated_cv_incomplete: "Repeated CV (incomplete)",
    holdout_fallback_apparent: "Holdout fallback apparent",
    mixed_holdout_apparent: "Mixed holdout/apparent",
  };
  return labels[String(mode || "").toLowerCase()] || humanizeHeader(mode || "unknown");
}

function benchmarkMetricNumber(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function predictiveWorkspaceUsesUnifiedBoard() {
  return activeTabName() === "benchmark";
}

function syncPredictiveWorkbenchCompareVisibility() {
  const suppressCompare = predictiveWorkspaceUsesUnifiedBoard();
  const mlComparisonCard = refs.mlComparisonShell?.closest(".table-card");
  const mlManuscriptCard = refs.mlManuscriptShell?.closest(".table-card");
  const dlComparisonCard = refs.dlComparisonShell?.closest(".table-card");
  const dlManuscriptCard = refs.dlManuscriptShell?.closest(".table-card");
  const mlHasPlot = hasRenderedPlot(refs.mlComparisonPlot);
  const dlHasPlot = hasRenderedPlot(refs.dlComparisonPlot);
  const mlCompareActive = preferredResultMode("ml") === "compare";
  const dlCompareActive = preferredResultMode("dl") === "compare";

  refs.mlComparisonPlot?.classList.toggle("hidden", suppressCompare || !mlCompareActive || !mlHasPlot);
  refs.dlComparisonPlot?.classList.toggle("hidden", suppressCompare || !dlCompareActive || !dlHasPlot);
  mlComparisonCard?.classList.toggle("hidden", suppressCompare);
  mlManuscriptCard?.classList.toggle("hidden", suppressCompare);
  dlComparisonCard?.classList.toggle("hidden", suppressCompare);
  dlManuscriptCard?.classList.toggle("hidden", suppressCompare);
}

function syncPredictiveWorkbenchSingleResultVisibility() {
  const workbenchOpen = Boolean(runtime.workbenchRevealed);
  const selectedFamily = normalizedPredictiveFamily(runtime.predictiveFamily);
  const mlHasCurrentSingle = Boolean(selectedPredictiveSingleResult("ml"));
  const dlHasCurrentSingle = Boolean(selectedPredictiveSingleResult("dl"));
  const hideMlSingle = workbenchOpen && (selectedFamily !== "ml" || !mlHasCurrentSingle);
  const hideDlSingle = workbenchOpen && (selectedFamily !== "dl" || !dlHasCurrentSingle);

  refs.mlImportancePlot?.closest(".ml-plots-grid")?.classList.toggle("hidden", hideMlSingle);
  refs.mlMetaBanner?.classList.toggle("hidden", hideMlSingle);
  refs.mlInsightBoard?.classList.toggle("hidden", hideMlSingle);

  refs.dlImportancePlot?.closest(".ml-plots-grid")?.classList.toggle("hidden", hideDlSingle);
  refs.dlMetaBanner?.classList.toggle("hidden", hideDlSingle);
  refs.dlInsightBoard?.classList.toggle("hidden", hideDlSingle);
}

function syncBenchmarkBoardChrome() {
  if (!state.dataset || !refs.benchmarkSummaryGrid || !refs.benchmarkComparisonShell || !refs.benchmarkTableNote) return;
  const board = benchmarkBoardState();
  renderUnifiedBenchmarkSummary(board);
  renderUnifiedBenchmarkTable(board);
}

function renderWorkspaceChrome() {
  syncWorkspaceLayout();
  syncBenchmarkBoardChrome();
  updateResultVisibility();
  renderRunStatus();
}

// Grouping settings only matter for survival curves and Table 1, so other tabs hide them.
function updateGroupingDetailsVisibility(tabName = activeTabName(), { force = false } = {}) {
  if (!refs.groupingDetails) return;
  const grouped = ["km", "tables"].includes(tabName);
  refs.groupingConfigBlock?.classList.toggle("hidden", !grouped);
  if (grouped) refs.groupingDetails.open = true;
}

function focusModelFeatureEditor(tabName = "ml") {
  const focusTargets = () => {
    const featureChecklist = tabName === "dl" ? refs.dlModelFeatureChecklist : refs.modelFeatureChecklist;
    const featureCard = featureChecklist?.closest(".selection-card");
    const featureSummaryCard = featureChecklist?.closest(".workspace-card")?.querySelector(".shared-feature-card");
    (featureCard || featureSummaryCard)?.scrollIntoView({ behavior: "smooth", block: "start" });
    flashPresetTargets([
      featureSummaryCard,
      refs.modelFeatureChecklist,
      refs.modelCategoricalChecklist,
      refs.dlModelFeatureChecklist,
      refs.dlModelCategoricalChecklist,
    ]);
  };

  activateTab(tabName);
  requestAnimationFrame(focusTargets);
}

function validateDerivedColumnName(rawName) {
  const name = String(rawName || "").trim();
  if (!name) return null;
  if (datasetColumnNames().includes(name)) {
    throw new Error(`"${name}" already exists. Choose a new derived-column name instead of overwriting an existing field.`);
  }
  if (name === refs.timeColumn?.value || name === refs.eventColumn?.value) {
    throw new Error(`"${name}" is reserved by the current survival endpoint. Choose a different derived-column name.`);
  }
  return name;
}

function flashPresetTargets(targets) {
  targets.filter(Boolean).forEach((target) => {
    const shell = target.closest(".config-field, .selection-card") || target;
    shell.classList.remove("preset-applied-flash");
    void shell.offsetWidth;
    shell.classList.add("preset-applied-flash");
    window.setTimeout(() => shell.classList.remove("preset-applied-flash"), 1800);
  });
}

function applyDatasetPreset(mode) {
  const preset = datasetPresetForCurrentDataset();
  if (!preset) return false;
  const columnNames = state.dataset.columns.map((c) => c.name);
  if (columnNames.includes(preset.timeColumn)) refs.timeColumn.value = preset.timeColumn;
  if (columnNames.includes(preset.eventColumn)) refs.eventColumn.value = preset.eventColumn;
  if (preset.timeUnitLabel && refs.timeUnitLabel) {
    refs.timeUnitLabel.value = preset.timeUnitLabel;
    runtime.timeUnitAutoLabel = true;
  }
  updateEventPositiveOptions();
  if (refs.eventPositiveValue) refs.eventPositiveValue.value = preset.eventPositiveValue;
  refreshVariableSelections();
  if (columnNames.includes(preset.basicGroup)) refs.groupColumn.value = preset.basicGroup;

  const covariates = mode === "models" ? preset.modelFeatures : preset.coxCovariates;
  const categoricals = mode === "models" ? preset.modelCategoricals : preset.coxCategoricals;
  const tableVariables = preset.tableVariables || covariates;
  if (mode === "models") {
    setCheckedValues(refs.modelFeatureChecklist, covariates);
    setCheckedValues(refs.modelCategoricalChecklist, categoricals);
    setCheckedValues(refs.dlModelFeatureChecklist, covariates);
    setCheckedValues(refs.dlModelCategoricalChecklist, categoricals);
  } else {
    setCheckedValues(refs.covariateChecklist, covariates);
    setCheckedValues(refs.categoricalChecklist, categoricals);
    setCheckedValues(refs.strataChecklist, []);
    syncCoxCovariateSelection();
  }
  setCheckedValues(refs.cohortVariableChecklist, tableVariables);
  updateDatasetBadge();
  return true;
}

// Sample cohorts open with their recommended outcome, grouping and variable selections.
function applyBundledPresets() {
  if (!datasetPresetForCurrentDataset()) return;
  applyDatasetPreset("basic");
  applyDatasetPreset("models");
  renderSharedFeatureSummary();
}

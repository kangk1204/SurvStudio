// SurvStudio front end, part 2/8: Request configs, result currency, workspace layout, history, and plot sizing.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

function currentSharedModelSelections(goal = "ml") {
  const featureChecklist = goal === "dl" ? refs.dlModelFeatureChecklist : refs.modelFeatureChecklist;
  const categoricalChecklist = goal === "dl" ? refs.dlModelCategoricalChecklist : refs.modelCategoricalChecklist;
  const features = selectedCheckboxValues(featureChecklist);
  return {
    features,
    categoricalFeatures: selectedCheckboxValues(categoricalChecklist)
      .filter((value) => features.includes(value)),
  };
}

// The server trims text settings (event value, time unit) before echoing them in request_config, so both
// sides of a currency comparison are trimmed too; otherwise " Dead" or "Months " would make every result
// look out of date.
function trimmedSetting(value) {
  return String(value ?? "").trim();
}

function normalizeBaseRequestConfig(requestConfig) {
  return {
    dataset_id: String(requestConfig?.dataset_id || ""),
    time_column: String(requestConfig?.time_column || ""),
    event_column: String(requestConfig?.event_column || ""),
    event_positive_value: trimmedSetting(requestConfig?.event_positive_value),
  };
}

function normalizedRequestConfig(goal, requestConfig, { expectsCompare = false } = {}) {
  if (!requestConfig) return null;
  const base = normalizeBaseRequestConfig(requestConfig);

  if (goal === "km") {
    return {
      ...base,
      group_column: String(requestConfig.group_column || ""),
      confidence_level: numberOrDefault(requestConfig.confidence_level, KM_NUMERIC_DEFAULTS.confidence_level),
      time_unit_label: trimmedSetting(requestConfig.time_unit_label) || DEFAULT_TIME_UNIT_LABEL,
      // Compare numerically so "2000.0" and 2000 describe the same truncation.
      max_time: numberOrDefault(requestConfig.max_time, null),
      risk_table_points: numberOrDefault(requestConfig.risk_table_points, KM_NUMERIC_DEFAULTS.risk_table_points),
      logrank_weight: String(requestConfig.logrank_weight || "logrank"),
      fh_p: String(requestConfig.logrank_weight || "logrank") === "fleming_harrington"
        ? numberOrDefault(requestConfig.fh_p, KM_NUMERIC_DEFAULTS.fh_p)
        : null,
      show_confidence_bands: Boolean(requestConfig.show_confidence_bands),
    };
  }

  if (goal === "cox") {
    return {
      ...base,
      covariates: sortedStrings(requestConfig.covariates || []),
      categorical_covariates: sortedStrings(requestConfig.categorical_covariates || []),
      strata_columns: sortedStrings(requestConfig.strata_columns || []),
    };
  }

  if (goal === "markers") {
    return {
      ...base,
      marker_columns: sortedStrings(requestConfig.marker_columns || []),
      marker_matrix_id: requestConfig.marker_matrix_id || null,
      marker_matrix_id_column: requestConfig.marker_matrix_id_column || null,
      clinical_columns: sortedStrings(requestConfig.clinical_columns || []),
      categorical_clinical: sortedStrings(requestConfig.categorical_clinical || []),
      n_permutations: numberOrDefault(requestConfig.n_permutations, 1000),
      n_resamples: numberOrDefault(requestConfig.n_resamples, 200),
      random_seed: numberOrDefault(requestConfig.random_seed, 20260926),
      nonlinear_lens: String(requestConfig.nonlinear_lens || "off"),
    };
  }

  if (goal === "ml") {
    const compareRun = String(requestConfig.model_type || "") === "compare";
    if (compareRun !== expectsCompare) return null;
    const effectiveModelType = expectsCompare ? "compare" : String(requestConfig.model_type || "");
    const learningRateApplies = expectsCompare || effectiveModelType === "gbs";
    const treeCountApplies = expectsCompare || effectiveModelType === "rsf" || effectiveModelType === "gbs";
    const evaluationStrategy = expectsCompare ? String(requestConfig.evaluation_strategy || "holdout") : null;
    const repeatedCv = evaluationStrategy === "repeated_cv";
    return {
      ...base,
      model_type: effectiveModelType,
      features: sortedStrings(requestConfig.features || []),
      categorical_features: sortedStrings(requestConfig.categorical_features || []),
      n_estimators: treeCountApplies ? numberOrDefault(requestConfig.n_estimators, ML_NUMERIC_DEFAULTS.n_estimators) : null,
      max_depth: String(requestConfig.max_depth ?? ""),
      learning_rate: learningRateApplies ? numberOrDefault(requestConfig.learning_rate, ML_NUMERIC_DEFAULTS.learning_rate) : null,
      random_state: numberOrDefault(requestConfig.random_state, ML_NUMERIC_DEFAULTS.random_state),
      evaluation_strategy: evaluationStrategy,
      cv_folds: repeatedCv ? numberOrDefault(requestConfig.cv_folds, ML_NUMERIC_DEFAULTS.cv_folds) : null,
      cv_repeats: repeatedCv ? numberOrDefault(requestConfig.cv_repeats, ML_NUMERIC_DEFAULTS.cv_repeats) : null,
      locked_test_fraction: repeatedCv ? normalizedLockedTestFraction(requestConfig.locked_test_fraction) : null,
    };
  }

  if (goal === "dl") {
    const compareRun = String(requestConfig.model_type || "") === "compare";
    if (compareRun !== expectsCompare) return null;
    const effectiveModelType = expectsCompare ? "compare" : String(requestConfig.model_type || "");
    const usesHiddenLayers = effectiveModelType !== "transformer" && effectiveModelType !== "compare";
    const usesDiscreteTime = effectiveModelType === "deephit" || effectiveModelType === "mtlr";
    const usesTransformer = effectiveModelType === "transformer" || effectiveModelType === "compare";
    const usesVae = effectiveModelType === "vae" || effectiveModelType === "compare";
    const evaluationStrategy = String(requestConfig.evaluation_strategy || "holdout");
    const repeatedCv = evaluationStrategy === "repeated_cv";
    const setting = (key) => numberOrDefault(requestConfig[key], DL_NUMERIC_DEFAULTS[key]);
    return {
      ...base,
      model_type: effectiveModelType,
      features: sortedStrings(requestConfig.features || []),
      categorical_features: sortedStrings(requestConfig.categorical_features || []),
      hidden_layers: usesHiddenLayers || expectsCompare ? (requestConfig.hidden_layers || []).map(Number) : null,
      dropout: setting("dropout"),
      learning_rate: setting("learning_rate"),
      epochs: setting("epochs"),
      batch_size: usesDiscreteTime || expectsCompare ? setting("batch_size") : null,
      random_seed: setting("random_seed"),
      evaluation_strategy: evaluationStrategy,
      cv_folds: repeatedCv ? setting("cv_folds") : null,
      cv_repeats: repeatedCv ? setting("cv_repeats") : null,
      early_stopping_patience: setting("early_stopping_patience"),
      early_stopping_min_delta: setting("early_stopping_min_delta"),
      parallel_jobs: repeatedCv ? setting("parallel_jobs") : null,
      num_time_bins: usesDiscreteTime || expectsCompare ? setting("num_time_bins") : null,
      d_model: usesTransformer ? setting("d_model") : null,
      n_heads: usesTransformer ? setting("n_heads") : null,
      n_layers: usesTransformer ? setting("n_layers") : null,
      latent_dim: usesVae ? setting("latent_dim") : null,
      n_clusters: usesVae ? setting("n_clusters") : null,
      locked_test_fraction: expectsCompare && repeatedCv ? normalizedLockedTestFraction(requestConfig.locked_test_fraction) : null,
    };
  }

  if (goal === "tables") {
    const outcomeRestricted = Boolean(requestConfig.time_column && requestConfig.event_column);
    return {
      dataset_id: String(requestConfig?.dataset_id || ""),
      variables: sortedStrings(requestConfig.variables || []),
      group_column: String(requestConfig.group_column || ""),
      time_column: outcomeRestricted ? String(requestConfig.time_column) : "",
      event_column: outcomeRestricted ? String(requestConfig.event_column) : "",
      event_positive_value: outcomeRestricted ? trimmedSetting(requestConfig.event_positive_value) : "",
    };
  }

  return base;
}

function currentGoalRequestConfig(goal, { expectsCompareOverride = null } = {}) {
  if (!state.dataset) return null;
  if (goal === "tables") {
    return normalizedRequestConfig(goal, {
      dataset_id: state.dataset.dataset_id,
      variables: selectedCheckboxValues(refs.cohortVariableChecklist),
      group_column: refs.groupColumn?.value || "",
      ...cohortTableOutcomeConfig(),
    });
  }
  let base;
  try {
    base = currentBaseConfig();
  } catch {
    return null;
  }

  if (goal === "km") {
    return normalizedRequestConfig(goal, {
      ...base,
      group_column: refs.groupColumn?.value || "",
      confidence_level: refs.confidenceLevel?.value,
      time_unit_label: refs.timeUnitLabel?.value,
      max_time: refs.maxTime?.value || "",
      risk_table_points: refs.riskTablePoints?.value,
      logrank_weight: refs.logrankWeight?.value || "logrank",
      fh_p: refs.fhPower?.value,
      show_confidence_bands: Boolean(refs.showConfidenceBands?.checked),
    });
  }

  if (goal === "markers") {
    return normalizedRequestConfig(goal, { ...base, ...markerRequestFields() });
  }

  if (goal === "cox") {
    const { covariates, categoricalCovariates, strataColumns } = currentCoxSelections();
    return normalizedRequestConfig(goal, {
      ...base,
      covariates,
      categorical_covariates: categoricalCovariates,
      strata_columns: strataColumns,
    });
  }

  if (goal === "ml") {
    const { features, categoricalFeatures } = currentSharedModelSelections("ml");
    const expectsCompare = expectsCompareOverride == null
      ? preferredResultMode("ml") === "compare"
      : Boolean(expectsCompareOverride);
    return normalizedRequestConfig(goal, {
      ...base,
      model_type: expectsCompare ? "compare" : String(refs.mlModelType?.value || ""),
      features,
      categorical_features: categoricalFeatures,
      n_estimators: refs.mlNEstimators?.value,
      max_depth: "",
      learning_rate: refs.mlLearningRate?.value,
      random_state: sharedPredictiveSeed(),
      shap_safe_mode: Boolean(refs.mlShapSafeMode?.checked),
      evaluation_strategy: refs.mlEvaluationStrategy?.value || "holdout",
      cv_folds: refs.mlCvFolds?.value,
      cv_repeats: refs.mlCvRepeats?.value,
      locked_test_fraction: currentLockedTestFraction("ml"),
    }, { expectsCompare });
  }

  if (goal === "dl") {
    const { features, categoricalFeatures } = currentSharedModelSelections("dl");
    const expectsCompare = expectsCompareOverride == null
      ? preferredResultMode("dl") === "compare"
      : Boolean(expectsCompareOverride);
    return normalizedRequestConfig(goal, {
      ...base,
      model_type: expectsCompare ? "compare" : String(refs.dlModelType?.value || ""),
      features,
      categorical_features: categoricalFeatures,
      hidden_layers: parseHiddenLayers(),
      dropout: refs.dlDropout?.value,
      learning_rate: refs.dlLearningRate?.value,
      epochs: refs.dlEpochs?.value,
      batch_size: refs.dlBatchSize?.value,
      random_seed: refs.dlRandomSeed?.value,
      evaluation_strategy: refs.dlEvaluationStrategy?.value || "holdout",
      cv_folds: refs.dlCvFolds?.value,
      cv_repeats: refs.dlCvRepeats?.value,
      early_stopping_patience: refs.dlEarlyStoppingPatience?.value,
      early_stopping_min_delta: refs.dlEarlyStoppingMinDelta?.value,
      parallel_jobs: refs.dlParallelJobs?.value,
      num_time_bins: refs.dlNumTimeBins?.value,
      d_model: refs.dlDModel?.value,
      n_heads: refs.dlHeads?.value,
      n_layers: refs.dlLayers?.value,
      latent_dim: refs.dlLatentDim?.value,
      n_clusters: refs.dlClusters?.value,
      locked_test_fraction: currentLockedTestFraction("dl"),
    }, { expectsCompare });
  }

  return normalizedRequestConfig(goal, base);
}

function matchesRequestConfig(goal, requestConfig, { expectsCompareOverride = null } = {}) {
  if (!requestConfig || !state.dataset) return false;
  const expectsCompare = goal === "ml" || goal === "dl"
    ? (expectsCompareOverride == null ? preferredResultMode(goal) === "compare" : Boolean(expectsCompareOverride))
    : false;
  const normalizedStored = normalizedRequestConfig(goal, requestConfig, { expectsCompare });
  const normalizedCurrent = currentGoalRequestConfig(goal, { expectsCompareOverride });
  if (!normalizedStored || !normalizedCurrent) return false;
  return stableStringify(normalizedStored) === stableStringify(normalizedCurrent);
}

function currentGoalResult(goal) {
  if (goal === "predictive") {
    if (typeof benchmarkBoardState === "function") {
      const board = benchmarkBoardState();
      if (
        !board
        || board.predictiveBusy
        || board.showingStaleBoard
        || board.hasMixedEvaluation
        || board.visibleHasMixedRunGroups
        || board.visibleHasSplitMismatch
        || !Array.isArray(board.visibleFamilies)
        || board.visibleFamilies.length !== 2
        || !board.visibleRows.length
      ) {
        return null;
      }
    }
    const currentMl = currentCompareGoalPayload("ml");
    const currentDl = currentCompareGoalPayload("dl");
    return currentMl && currentDl ? { ml: currentMl, dl: currentDl } : null;
  }
  const payload = {
    km: state.km,
    cox: state.cox,
    markers: state.markers,
    ml: state.ml,
    dl: state.dl,
    tables: state.cohort,
  }[goal] || null;
  if (!payload) return null;
  const requestConfig = payload.request_config || payload.analysis?.request_config || null;
  if (!requestConfig) return payload;
  return matchesRequestConfig(goal, requestConfig) ? payload : null;
}

function compareGoalPayload(goal) {
  if (!["ml", "dl"].includes(goal)) return null;
  const latestPayload = goalPayload(goal);
  if (payloadRepresentsCompareRun(latestPayload)) return latestPayload;
  return runtime.compareCache?.[goal] || null;
}

function compareSnapshotPayload(goal) {
  if (!["ml", "dl"].includes(goal)) return null;
  const payload = runtime.compareCache?.unified?.[goal] || null;
  return payloadRepresentsCompareRun(payload) ? payload : null;
}

function currentCompareGoalPayload(goal) {
  const payload = compareGoalPayload(goal);
  if (!payload) return null;
  const requestConfig = payload.request_config || payload.analysis?.request_config || null;
  if (!requestConfig) return payload;
  return matchesRequestConfig(goal, requestConfig, { expectsCompareOverride: true }) ? payload : null;
}

function goalPayload(goal) {
  if (goal === "predictive") {
    return state.ml || state.dl ? { ml: state.ml, dl: state.dl } : null;
  }
  return {
    km: state.km,
    cox: state.cox,
    markers: state.markers,
    ml: state.ml,
    dl: state.dl,
    tables: state.cohort,
  }[goal] || null;
}

function goalHasAnyOutput(goal) {
  if (goal === "predictive") {
    return Boolean(
      state.ml
      || state.dl
      || runtime.compareCache?.ml
      || runtime.compareCache?.dl
      || runtime.compareCache?.unified?.ml
      || runtime.compareCache?.unified?.dl,
    );
  }
  if (goal === "tables") {
    return currentCohortTableOutputState().hasOutput;
  }
  if (goal === "signature") {
    return Boolean(state.signature);
  }
  return Boolean(goalPayload(goal));
}

function goalResultStatusState(goal, { currentLabel = "Ready", noResultLabel = "No result yet" } = {}) {
  if (!goal || !ANALYSIS_GOALS.includes(goal)) return null;
  const scope = runScopeForGoal(goal);
  const predictiveBusy = goal === "predictive" && (isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl"));
  if ((scope && isScopeBusy(scope)) || predictiveBusy) {
    return {
      tone: "running",
      label: "Running",
      title: `${goalLabel(goal)} in progress`,
      text: "Wait for the current run to finish before exporting this result or changing shared inputs.",
    };
  }
  if (goal === "predictive") {
    const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
    const selectedSingleCurrent = Boolean(selectedPredictiveSingleResult(selectedModel.family));
    if (runtime.predictiveWorkbenchIntent === "train" && selectedSingleCurrent) {
      return {
        tone: "ready",
        label: currentLabel,
        title: `${selectedModel.label} result is current`,
        text: "Visible settings match the selected model result shown here.",
      };
    }
    if (predictiveLeaderboardIsCurrent()) {
      const board = typeof benchmarkBoardState === "function" ? benchmarkBoardState() : null;
      if (board?.showingStaleBoard) {
        return {
          tone: "warning",
          label: "Stale reference",
          title: "Compare All result shown as stale reference",
          text: "Visible leaderboard is a stale Compare All snapshot. Rerun Compare All Models to refresh it.",
        };
      }
      return {
        tone: "ready",
        label: currentLabel,
        title: "Compare All result is current",
        text: "Visible settings match the predictive leaderboard shown here.",
      };
    }
  }
  const hasCurrentResult = goal === "tables"
    ? currentCohortTableOutputState().isCurrent
    : Boolean(currentGoalResult(goal));
  if (hasCurrentResult) {
    return {
      tone: "ready",
      label: currentLabel,
      title: `${goalLabel(goal)} result is current`,
      text: "Visible settings match the result shown here.",
    };
  }
  if (goalHasAnyOutput(goal)) {
    if (goal === "tables") {
      return {
        tone: "warning",
        label: "Settings changed",
        title: `${goalLabel(goal)} settings changed`,
        text: "Visible settings changed after this table was built. You can still export the visible table, or rebuild it to refresh the output.",
      };
    }
    return {
      tone: "warning",
      label: "Settings changed",
      title: `${goalLabel(goal)} settings changed`,
      text: "Visible settings no longer match the current result. Run again before exporting or interpreting it.",
    };
  }
  return {
    tone: "idle",
    label: noResultLabel,
    title: `${goalLabel(goal)} not run yet`,
    text: "Run the analysis to populate this result view.",
  };
}

// A small status next to each Run button: running, up to date, or settings changed since the last run.
function renderRunStatus() {
  document.querySelectorAll("[data-run-status]").forEach((node) => {
    const status = state.dataset ? goalResultStatusState(node.dataset.runStatus, { currentLabel: "Up to date" }) : null;
    const visible = Boolean(status && status.tone !== "idle");
    node.className = `run-status run-status-${status?.tone || "idle"}${visible ? "" : " hidden"}`;
    node.textContent = visible ? status.label : "";
    node.title = visible ? status.text : "";
  });
}

function renderCoxPreviewLine() {
  const line = refs.coxPreviewLine;
  if (!line) return;
  const { covariates } = currentCoxSelections();
  const preview = runtime.coxPreview.status === "ready" ? runtime.coxPreview.payload?.preview : null;
  let text = "";
  let warning = "";
  if (!state.dataset || !covariates.length) {
    text = "";
  } else if (runtime.coxPreview.status === "loading") {
    text = "Checking usable rows for the selected covariates...";
  } else if (runtime.coxPreview.status === "blocked" || runtime.coxPreview.status === "error") {
    text = runtime.coxPreview.error || "";
  } else if (preview) {
    const epv = preview.events_per_parameter == null ? "NA" : formatValue(preview.events_per_parameter, { scientificLarge: false });
    text = `${formatValue(preview.analyzable_rows)} of ${formatValue(preview.outcome_rows)} patients usable`
      + (preview.dropped_rows ? ` (${formatValue(preview.dropped_rows)} dropped for missing values)` : "")
      + ` · ${formatValue(preview.events)} events · ${formatValue(preview.estimated_parameters)} parameters · ${epv} events per parameter`;
    warning = (preview.stability_warnings || [])[0] || "";
  }
  line.textContent = warning ? `${text}. ${warning}` : text;
  line.classList.toggle("hidden", !text);
  line.classList.toggle("has-warning", Boolean(warning));
}

function endpointIsReady() {
  if (!state.dataset) return false;
  try {
    currentBaseConfig();
    return true;
  } catch {
    return false;
  }
}

function reparentScrollContainers() {
  return [
    refs.configStrip,
    refs.tabPanelsHome,
    refs.benchmarkMlMount,
    refs.benchmarkDlMount,
    ...refs.tabPanels,
    refs.covariateChecklist,
    refs.categoricalChecklist,
    refs.modelFeatureChecklist,
    refs.modelCategoricalChecklist,
    refs.dlModelFeatureChecklist,
    refs.dlModelCategoricalChecklist,
    refs.cohortVariableChecklist,
  ].filter(Boolean);
}

function captureReparentUiState() {
  const focusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const scrollPositions = new Map();
  [...new Set(reparentScrollContainers())].forEach((element) => {
    if (!element) return;
    scrollPositions.set(element, {
      top: element.scrollTop,
      left: element.scrollLeft,
    });
  });
  return { focusedElement, scrollPositions };
}

function restoreReparentUiState(snapshot) {
  if (!snapshot) return;
  snapshot.scrollPositions?.forEach((position, element) => {
    if (!element?.isConnected) return;
    element.scrollTop = position.top;
    element.scrollLeft = position.left;
  });
  const focusTarget = snapshot.focusedElement;
  if (focusTarget?.isConnected && typeof focusTarget.focus === "function") {
    try {
      focusTarget.focus({ preventScroll: true });
    } catch {
      focusTarget.focus();
    }
  }
}

function predictiveLeaderboardIsCurrent() {
  if (typeof benchmarkBoardState !== "function") return false;
  const board = benchmarkBoardState();
  return Boolean(
    !board?.predictiveBusy
    && Array.isArray(board?.visibleFamilies)
    && board.visibleFamilies.length === 2
    && !board?.hasMixedEvaluation
    && !board?.visibleHasMixedRunGroups
    && !board?.visibleHasSplitMismatch
    && (board?.visibleRows?.length || 0) > 0,
  );
}

function syncWorkspaceLayout() {
  const preservedUiState = captureReparentUiState();
  let didMove = false;
  // The ML and DL workspace cards live in the Prediction models tab once a dataset is open.
  const merged = Boolean(state.dataset);
  [
    [refs.mlWorkspaceCard, refs.benchmarkMlMount, refs.mlPanel],
    [refs.dlWorkspaceCard, refs.benchmarkDlMount, refs.dlPanel],
  ].forEach(([card, mount, panel]) => {
    if (!card || !mount || !panel) return;
    const target = merged ? mount : panel;
    if (card.parentElement !== target) {
      target.appendChild(card);
      didMove = true;
    }
    card.classList.toggle("predictive-workbench-card", merged);
    syncPredictiveWorkbenchCardActions(card, merged);
  });
  syncDeriveToggleButton();
  renderPredictiveWorkbench();
  if (refs.cutpointPlot) {
    refs.cutpointPlot.classList.toggle("hidden", refs.cutpointPlot.innerHTML.trim().length === 0);
  }
  restoreReparentUiState(preservedUiState);
  if (didMove) scheduleVisiblePlotResize(40);
}

function resizeVisiblePlotsNow() {
  const plots = allPlotRefs();
  plots.forEach((plot) => {
    if (!plot?.data?.length || !plotIsDisplayed(plot)) return;
    try {
      Plotly.Plots.resize(plot);
      stabilizePlotShellHeight(plot);
    } catch {
      // Ignore plots that were removed while resizing.
    }
  });
}

function allPlotRefs() {
  return [
    refs.kmPlot,
    refs.markersSummaryPlot,
    refs.markersStabilityPlot,
    refs.markersRankPlot,
    refs.markerValidationPlot,
    refs.coxPlot,
    refs.coxDiagnosticsPlot,
    refs.coxMartingalePlot,
    refs.cutpointPlot,
    refs.mlImportancePlot,
    refs.mlShapPlot,
    refs.mlComparisonPlot,
    refs.dlImportancePlot,
    refs.dlLossPlot,
    refs.dlComparisonPlot,
    refs.benchmarkComparisonPlot,
  ];
}

function queueVisiblePlotResize() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      resizeVisiblePlotsNow();
      window.setTimeout(resizeVisiblePlotsNow, 120);
      window.setTimeout(resizeVisiblePlotsNow, 260);
    });
  });
}

function waitForRenderFrames(count = 1) {
  const frames = Math.max(1, Number(count) || 1);
  return new Promise((resolve) => {
    const step = (remaining) => {
      requestAnimationFrame(() => {
        if (remaining <= 1) {
          resolve();
          return;
        }
        step(remaining - 1);
      });
    };
    step(frames);
  });
}

async function stabilizePlotsBeforeBanner(plots = [], banner, { maxAttempts = 12, tolerance = 1 } = {}) {
  const visiblePlots = plots.filter((plot) => plot && !plot.closest(".hidden"));
  if (!visiblePlots.length || !banner) return;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await waitForRenderFrames(2);
    visiblePlots.forEach((plot) => {
      if (!plot?.data?.length || !plotIsDisplayed(plot)) return;
      try {
        Plotly.Plots.resize(plot);
        stabilizePlotShellHeight(plot);
      } catch {
        // Ignore detached or stale plot nodes during fast UI transitions.
      }
    });
    const bannerRect = banner.getBoundingClientRect();
    const plotsAreOrdered = visiblePlots.every((plot) => plot.getBoundingClientRect().bottom <= bannerRect.top + tolerance);
    if (plotsAreOrdered) return;
    await new Promise((resolve) => window.setTimeout(resolve, 25));
  }
}

function scheduleVisiblePlotResize(delay = 80) {
  if (runtime.plotResizeTimer) {
    window.clearTimeout(runtime.plotResizeTimer);
  }
  runtime.plotResizeTimer = window.setTimeout(() => {
    runtime.plotResizeTimer = null;
    queueVisiblePlotResize();
  }, delay);
}

function initPlotResizeObserver() {
  if (runtime.plotResizeObserver || typeof window.ResizeObserver !== "function") return;
  const widthCache = new WeakMap();
  runtime.plotResizeObserver = new window.ResizeObserver((entries) => {
    const widthChanged = entries.some((entry) => {
      const target = entry?.target;
      if (!target) return false;
      const nextWidth = Number(entry.contentRect?.width || target.clientWidth || 0);
      if (!Number.isFinite(nextWidth) || nextWidth <= 0) return false;
      const previousWidth = widthCache.get(target);
      widthCache.set(target, nextWidth);
      return Number.isFinite(previousWidth) && Math.abs(nextWidth - previousWidth) > 1;
    });
    if (widthChanged) scheduleVisiblePlotResize(30);
  });
  allPlotRefs().forEach((plot) => {
    if (!plot) return;
    widthCache.set(plot, Number(plot.clientWidth || 0));
    runtime.plotResizeObserver.observe(plot);
  });
}

function captureControlSnapshot() {
  if (!state.dataset) return null;
  return {
    timeColumn: refs.timeColumn?.value || "",
    showAllTimeColumns: Boolean(refs.showAllTimeColumns?.checked),
    eventColumn: refs.eventColumn?.value || "",
    eventPositiveValue: refs.eventPositiveValue?.value || "",
    showAllEventColumns: Boolean(refs.showAllEventColumns?.checked),
    groupColumn: refs.groupColumn?.value || "",
    timeUnitLabel: refs.timeUnitLabel?.value || "",
    maxTime: refs.maxTime?.value || "",
    confidenceLevel: refs.confidenceLevel?.value || "",
    derivePanelOpen: !refs.derivePanel?.classList.contains("hidden"),
    deriveSource: refs.deriveSource?.value || "",
    deriveMethod: refs.deriveMethod?.value || "",
    deriveCutoff: refs.deriveCutoff?.value || "",
    deriveMinGroupFraction: refs.deriveMinGroupFraction?.value || "",
    derivePermutationIterations: refs.derivePermutationIterations?.value || "",
    deriveRandomSeed: refs.deriveRandomSeed?.value || "",
    deriveColumnName: refs.deriveColumnName?.value || "",
    deriveDraftTouched: Boolean(runtime.deriveDraftTouched),
    showConfidenceBands: Boolean(refs.showConfidenceBands?.checked),
    riskTablePoints: refs.riskTablePoints?.value || "",
    logrankWeight: refs.logrankWeight?.value || "",
    fhPower: refs.fhPower?.value || "",
    signatureMaxDepth: refs.signatureMaxDepth?.value || "",
    signatureMinFraction: refs.signatureMinFraction?.value || "",
    signatureTopK: refs.signatureTopK?.value || "",
    signatureBootstrapIterations: refs.signatureBootstrapIterations?.value || "",
    signaturePermutationIterations: refs.signaturePermutationIterations?.value || "",
    signatureValidationIterations: refs.signatureValidationIterations?.value || "",
    signatureValidationFraction: refs.signatureValidationFraction?.value || "",
    signatureSignificanceLevel: refs.signatureSignificanceLevel?.value || "",
    signatureOperator: refs.signatureOperator?.value || "",
    signatureRandomSeed: refs.signatureRandomSeed?.value || "",
    covariates: selectedCheckboxValues(refs.covariateChecklist),
    categoricals: selectedCheckboxValues(refs.categoricalChecklist),
    coxStrata: selectedCheckboxValues(refs.strataChecklist),
    coxPreviewKey: runtime.coxPreview?.key || "",
    modelFeatures: selectedCheckboxValues(refs.modelFeatureChecklist),
    modelCategoricals: selectedCheckboxValues(refs.modelCategoricalChecklist),
    dlModelCategoricals: selectedCheckboxValues(refs.dlModelCategoricalChecklist),
    cohortVariables: selectedCheckboxValues(refs.cohortVariableChecklist),
    markers: selectedCheckboxValues(refs.markerChecklist),
    markerClinical: selectedCheckboxValues(refs.markerClinicalChecklist),
    markerPermutations: refs.markerPermutations?.value || "",
    markerResamples: refs.markerResamples?.value || "",
    markerRandomSeed: refs.markerRandomSeed?.value || "",
    markerNonlinearLens: refs.markerNonlinearLens?.value || "",
    mlModelType: refs.mlModelType?.value || "",
    mlNEstimators: refs.mlNEstimators?.value || "",
    mlLearningRate: refs.mlLearningRate?.value || "",
    mlSkipShap: Boolean(refs.mlSkipShap?.checked),
    mlShapSafeMode: Boolean(refs.mlShapSafeMode?.checked),
    mlEvaluationStrategy: refs.mlEvaluationStrategy?.value || "",
    mlCvFolds: refs.mlCvFolds?.value || "",
    mlCvRepeats: refs.mlCvRepeats?.value || "",
    mlRandomSeed: refs.mlRandomSeed?.value || "",
    lockedTestEnabled: Boolean(refs.mlLockedTestToggle?.checked),
    lockedTestPercent: refs.mlLockedTestFraction?.value || "",
    timeUnitAutoLabel: Boolean(runtime.timeUnitAutoLabel),
    mlJournalTemplate: refs.mlJournalTemplate?.value || "",
    dlModelType: refs.dlModelType?.value || "",
    dlEpochs: refs.dlEpochs?.value || "",
    dlLearningRate: refs.dlLearningRate?.value || "",
    dlHiddenLayers: refs.dlHiddenLayers?.value || "",
    dlDropout: refs.dlDropout?.value || "",
    dlBatchSize: refs.dlBatchSize?.value || "",
    dlRandomSeed: refs.dlRandomSeed?.value || "",
    dlEvaluationStrategy: refs.dlEvaluationStrategy?.value || "",
    dlCvFolds: refs.dlCvFolds?.value || "",
    dlCvRepeats: refs.dlCvRepeats?.value || "",
    dlEarlyStoppingPatience: refs.dlEarlyStoppingPatience?.value || "",
    dlEarlyStoppingMinDelta: refs.dlEarlyStoppingMinDelta?.value || "",
    dlParallelJobs: refs.dlParallelJobs?.value || "",
    dlNumTimeBins: refs.dlNumTimeBins?.value || "",
    dlDModel: refs.dlDModel?.value || "",
    dlHeads: refs.dlHeads?.value || "",
    dlLayers: refs.dlLayers?.value || "",
    dlLatentDim: refs.dlLatentDim?.value || "",
    dlClusters: refs.dlClusters?.value || "",
    dlJournalTemplate: refs.dlJournalTemplate?.value || "",
  };
}

function scheduleResultCurrencySync(delay = 60) {
  if (runtime.resultCurrencySyncTimer) window.clearTimeout(runtime.resultCurrencySyncTimer);
  runtime.resultCurrencySyncTimer = window.setTimeout(() => {
    runtime.resultCurrencySyncTimer = null;
    syncDownloadButtonAvailability();
    updateCohortTableButtonLabel();
    renderBenchmarkBoard();
    renderWorkspaceChrome();
  }, delay);
}

function queueHistorySync() {
  syncStaleSingleResultArtifacts();
  if (runtime.historySyncPaused || !state.dataset || !window.history?.replaceState) return;
  if (runtime.historySyncTimer) window.clearTimeout(runtime.historySyncTimer);
  runtime.historySyncTimer = window.setTimeout(() => {
    runtime.historySyncTimer = null;
    syncHistoryState("replace");
  }, 0);
}

function setInputValue(control, value) {
  if (!control || value === undefined || value === null) return;
  control.value = String(value);
}

function setSelectValueIfPresent(control, value) {
  if (!control || value === undefined || value === null) return false;
  const wanted = String(value);
  if ([...control.options].some((option) => option.value === wanted)) {
    control.value = wanted;
    return true;
  }
  return false;
}

function applyControlSnapshot(snapshot) {
  if (!snapshot || !state.dataset) return;
  const columnNames = new Set(state.dataset.columns.map((column) => column.name));
  // The Time menu lists every numeric column only with its override ticked, so restore that first.
  if (refs.showAllTimeColumns) refs.showAllTimeColumns.checked = Boolean(snapshot.showAllTimeColumns);
  renderTimeColumnOptions({
    preferred: snapshot.timeColumn && columnNames.has(snapshot.timeColumn) ? snapshot.timeColumn : null,
    silent: true,
  });
  if (refs.showAllEventColumns) refs.showAllEventColumns.checked = Boolean(snapshot.showAllEventColumns);
  renderEventColumnOptions({
    preferred: snapshot.eventColumn && columnNames.has(snapshot.eventColumn) ? snapshot.eventColumn : null,
    silent: true,
    restoring: true,
  });
  setSelectValueIfPresent(refs.eventPositiveValue, snapshot.eventPositiveValue);
  // The warnings follow the restored endpoint: reading the event value again keeps it and drops a "choose the
  // event value" note it answers, and the time check sees the restored event column.
  updateEventPositiveOptions();
  updateTimeColumnGuidance();
  refreshVariableSelections();
  setSelectValueIfPresent(refs.groupColumn, snapshot.groupColumn ?? "");
  setInputValue(refs.timeUnitLabel, snapshot.timeUnitLabel);
  setInputValue(refs.maxTime, snapshot.maxTime);
  setInputValue(refs.confidenceLevel, snapshot.confidenceLevel);
  setSelectValueIfPresent(refs.deriveSource, snapshot.deriveSource);
  const restoredDeriveMethod = setSelectValueIfPresent(refs.deriveMethod, snapshot.deriveMethod);
  if (restoredDeriveMethod || snapshot.deriveMethod === undefined || snapshot.deriveMethod === null) {
    setInputValue(refs.deriveCutoff, snapshot.deriveCutoff);
  } else if (refs.deriveCutoff) {
    refs.deriveCutoff.value = "";
  }
  setInputValue(refs.deriveColumnName, snapshot.deriveColumnName);
  setInputValue(refs.riskTablePoints, snapshot.riskTablePoints);
  setInputValue(refs.fhPower, snapshot.fhPower);
  setInputValue(refs.signatureMaxDepth, snapshot.signatureMaxDepth);
  setInputValue(refs.signatureMinFraction, snapshot.signatureMinFraction);
  setInputValue(refs.signatureTopK, snapshot.signatureTopK);
  setInputValue(refs.signatureBootstrapIterations, snapshot.signatureBootstrapIterations);
  setInputValue(refs.signaturePermutationIterations, snapshot.signaturePermutationIterations);
  setInputValue(refs.signatureValidationIterations, snapshot.signatureValidationIterations);
  setInputValue(refs.signatureValidationFraction, snapshot.signatureValidationFraction);
  setInputValue(refs.signatureSignificanceLevel, snapshot.signatureSignificanceLevel);
  setInputValue(refs.signatureRandomSeed, snapshot.signatureRandomSeed);
  setInputValue(refs.mlNEstimators, snapshot.mlNEstimators);
  setInputValue(refs.mlLearningRate, snapshot.mlLearningRate);
  setInputValue(refs.mlCvFolds, snapshot.mlCvFolds);
  setInputValue(refs.mlCvRepeats, snapshot.mlCvRepeats);
  setInputValue(refs.dlEpochs, snapshot.dlEpochs);
  setInputValue(refs.dlLearningRate, snapshot.dlLearningRate);
  setInputValue(refs.dlHiddenLayers, snapshot.dlHiddenLayers);
  setInputValue(refs.dlDropout, snapshot.dlDropout);
  setInputValue(refs.dlBatchSize, snapshot.dlBatchSize);
  setInputValue(refs.dlRandomSeed, snapshot.dlRandomSeed);
  setInputValue(refs.dlCvFolds, snapshot.dlCvFolds);
  setInputValue(refs.dlCvRepeats, snapshot.dlCvRepeats);
  setInputValue(refs.dlEarlyStoppingPatience, snapshot.dlEarlyStoppingPatience);
  setInputValue(refs.dlEarlyStoppingMinDelta, snapshot.dlEarlyStoppingMinDelta);
  setInputValue(refs.dlParallelJobs, snapshot.dlParallelJobs);
  setInputValue(refs.dlNumTimeBins, snapshot.dlNumTimeBins);
  setInputValue(refs.dlDModel, snapshot.dlDModel);
  setInputValue(refs.dlHeads, snapshot.dlHeads);
  setInputValue(refs.dlLayers, snapshot.dlLayers);
  setInputValue(refs.dlLatentDim, snapshot.dlLatentDim);
  setInputValue(refs.dlClusters, snapshot.dlClusters);
  setInputValue(refs.mlRandomSeed, snapshot.mlRandomSeed || snapshot.dlRandomSeed);
  setInputValue(refs.mlLockedTestFraction, snapshot.lockedTestPercent);
  setInputValue(refs.dlLockedTestFraction, snapshot.lockedTestPercent);
  if (snapshot.lockedTestEnabled !== undefined) {
    if (refs.mlLockedTestToggle) refs.mlLockedTestToggle.checked = Boolean(snapshot.lockedTestEnabled);
    if (refs.dlLockedTestToggle) refs.dlLockedTestToggle.checked = Boolean(snapshot.lockedTestEnabled);
  }
  runtime.timeUnitAutoLabel = snapshot.timeUnitAutoLabel === undefined
    ? String(refs.timeUnitLabel?.value || "") === automaticTimeUnitLabel()
    : Boolean(snapshot.timeUnitAutoLabel);
  setInputValue(refs.deriveMinGroupFraction, snapshot.deriveMinGroupFraction);
  setInputValue(refs.derivePermutationIterations, snapshot.derivePermutationIterations);
  setInputValue(refs.deriveRandomSeed, snapshot.deriveRandomSeed);
  setSelectValueIfPresent(refs.logrankWeight, snapshot.logrankWeight);
  setSelectValueIfPresent(refs.signatureOperator, snapshot.signatureOperator);
  setSelectValueIfPresent(refs.mlModelType, snapshot.mlModelType);
  setSelectValueIfPresent(refs.mlEvaluationStrategy, snapshot.mlEvaluationStrategy);
  setSelectValueIfPresent(refs.mlJournalTemplate, snapshot.mlJournalTemplate);
  setSelectValueIfPresent(refs.dlModelType, snapshot.dlModelType);
  setSelectValueIfPresent(refs.dlEvaluationStrategy, snapshot.dlEvaluationStrategy);
  setSelectValueIfPresent(refs.dlJournalTemplate, snapshot.dlJournalTemplate);
  if (refs.mlSkipShap) refs.mlSkipShap.checked = snapshot.mlSkipShap !== false;
  if (refs.mlShapSafeMode) refs.mlShapSafeMode.checked = snapshot.mlShapSafeMode !== false;
  if (refs.showConfidenceBands) refs.showConfidenceBands.checked = snapshot.showConfidenceBands !== false;
  updateMethodVisibility();
  updateWeightVisibility();
  updateMlEvaluationControls();
  updateDlEvaluationControls();
  updateMlModelControlVisibility();
  updateDlModelControlVisibility();
  setCheckedValues(refs.covariateChecklist, snapshot.covariates || []);
  setCheckedValues(refs.categoricalChecklist, snapshot.categoricals || []);
  setCheckedValues(refs.strataChecklist, snapshot.coxStrata || []);
  setCheckedValues(refs.modelFeatureChecklist, snapshot.modelFeatures || []);
  syncModelFeatureMirrors(refs.modelFeatureChecklist);
  // The categorical flags come after mirroring the features, which ticks every likely categorical feature: a
  // flag the user unticked stays unticked. (A snapshot without the lists keeps those automatic flags.)
  if (Array.isArray(snapshot.modelCategoricals)) setCheckedValues(refs.modelCategoricalChecklist, snapshot.modelCategoricals);
  const dlCategoricals = Array.isArray(snapshot.dlModelCategoricals) ? snapshot.dlModelCategoricals : snapshot.modelCategoricals;
  if (Array.isArray(dlCategoricals)) setCheckedValues(refs.dlModelCategoricalChecklist, dlCategoricals);
  syncModelCategoricalMirrors(refs.modelCategoricalChecklist);
  syncModelCategoricalMirrors(refs.dlModelCategoricalChecklist);
  setCheckedValues(refs.cohortVariableChecklist, snapshot.cohortVariables || []);
  if (snapshot.markers) setCheckedValues(refs.markerChecklist, snapshot.markers);
  if (snapshot.markerClinical) setCheckedValues(refs.markerClinicalChecklist, snapshot.markerClinical);
  setInputValue(refs.markerPermutations, snapshot.markerPermutations || undefined);
  setInputValue(refs.markerResamples, snapshot.markerResamples || undefined);
  setInputValue(refs.markerRandomSeed, snapshot.markerRandomSeed || undefined);
  setSelectValueIfPresent(refs.markerNonlinearLens, snapshot.markerNonlinearLens || undefined);
  renderMarkerSelectionLine();
  syncCoxCovariateSelection();
  renderSharedFeatureSummary();
  updateDatasetBadge();
  const derivePanelOpen = Boolean(snapshot.derivePanelOpen);
  refs.derivePanel?.classList.toggle("hidden", !derivePanelOpen);
  runtime.deriveDraftTouched = Boolean(snapshot.deriveDraftTouched);
  syncDeriveToggleButton();
  scheduleCoxPreview({ delay: 0 });
}

function currentHistoryState() {
  return shellHelpers.currentHistoryState({
    state,
    runtime,
    activeTabName,
    captureControlSnapshot,
  });
}

function syncHistoryState(mode = "replace") {
  return shellHelpers.syncHistoryState({
    runtime,
    nextState: currentHistoryState(),
    mode,
  });
}

async function restoreHistoryState(historyState) {
  const restoreToken = ++runtime.historyRestoreToken;
  const restoredPredictiveFamily = normalizedPredictiveFamily(historyState?.predictiveFamily);
  if (!historyState || historyState.view !== "workspace" || !historyState.datasetId) {
    runtime.predictiveFamily = restoredPredictiveFamily;
    runtime.predictiveWorkbenchIntent = null;
    goHome({ syncHistory: false });
    return;
  }

  runtime.historySyncPaused = true;
  try {
    runtime.predictiveFamily = restoredPredictiveFamily;
    runtime.workbenchRevealed = Boolean(historyState?.workbenchRevealed);
    runtime.predictiveWorkbenchIntent = normalizedPredictiveWorkbenchIntent(historyState?.predictiveWorkbenchIntent)
      || (runtime.workbenchRevealed ? "train" : null);
    if (!state.dataset || state.dataset.dataset_id !== historyState.datasetId) {
      const datasetToken = beginRequestToken("dataset");
      const payload = await fetchJSON(`/api/dataset/${historyState.datasetId}`, { signal: requestSignal("dataset") });
      if (restoreToken !== runtime.historyRestoreToken || !requestTokenMatches("dataset", datasetToken)) return;
      updateAfterDataset(payload);
    } else {
      showWorkspace();
    }
    if (restoreToken !== runtime.historyRestoreToken) return;
    runtime.workbenchRevealed = Boolean(historyState?.workbenchRevealed);
    runtime.predictiveWorkbenchIntent = normalizedPredictiveWorkbenchIntent(historyState?.predictiveWorkbenchIntent)
      || (runtime.workbenchRevealed ? "train" : null);
    applyControlSnapshot(historyState.controls || null);
    activateTab(historyState.tab || "km");
    renderWorkspaceChrome();
  } catch (error) {
    // A newer navigation or dataset load cancelled this restore; leave the workspace alone. Otherwise the
    // page cannot be shown (for example its cohort left the server's store), so say why before going home.
    if (!isSupersededRequestError(error)) {
      goHome({ syncHistory: false });
      const reason = errorMessageText(error).replace(/([^.!?])$/, "$1.");
      setRuntimeBanner(`That page could not be restored: ${reason} Load the cohort again to continue.`, "warning");
    }
  } finally {
    runtime.historySyncPaused = false;
  }
}

function syncDeriveToggleButton() {
  if (!refs.deriveToggle || !refs.derivePanel) return;
  refs.deriveToggle.classList.remove("hidden");
  refs.deriveButton?.classList.remove("hidden");
  const derivePanelOpen = !refs.derivePanel.classList.contains("hidden");
  refs.deriveToggle.textContent = derivePanelOpen ? "Close" : "Make groups";
  refs.deriveToggle.classList.toggle("primary", !derivePanelOpen);
  refs.deriveToggle.classList.toggle("ghost", derivePanelOpen);
  refs.deriveToggle.setAttribute("aria-expanded", derivePanelOpen ? "true" : "false");
}

function syncDeriveControlsState() {
  const activeGroup = String(refs.groupColumn?.value || "").trim();
  const deriveLocked = Boolean(activeGroup);
  const summaryPayload = refs.deriveSummary?.dataset.summaryKind === "derived"
    ? currentDerivedSummaryPayload()
    : null;
  const displayedDerivedColumn = normalizeDeriveSummaryText(summaryPayload?.derivedColumn);
  const displayedSummary = summaryPayload?.summary || null;
  const controlsMismatch = displayedDerivedColumn
    ? !deriveDraftMatchesStoredRecipe(displayedDerivedColumn, displayedSummary)
    : false;
  const deriveInputs = [
    refs.deriveSource,
    refs.deriveMethod,
    refs.deriveCutoff,
    refs.deriveColumnName,
    refs.deriveMinGroupFraction,
    refs.derivePermutationIterations,
    refs.deriveRandomSeed,
  ].filter(Boolean);
  deriveInputs.forEach((input) => {
    input.disabled = deriveLocked;
    input.setAttribute("aria-disabled", String(deriveLocked));
  });
  if (refs.deriveButton) {
    // Create also stays off while a grouping is being created (the "derive" busy scope).
    const deriveBusy = isScopeBusy("derive");
    refs.deriveButton.disabled = deriveLocked || deriveBusy;
    refs.deriveButton.setAttribute("aria-disabled", String(deriveLocked || deriveBusy));
    refs.deriveButton.title = deriveLocked
      ? `Clear Group by back to Overall only before editing or creating a new grouping. Current Group by is ${activeGroup}.`
      : "";
  }
  refs.derivePanel?.classList.toggle("is-locked", deriveLocked);
  refs.deriveOptimalControls?.querySelectorAll(".toolbar-field").forEach((field) => {
    field.classList.toggle("is-disabled", deriveLocked);
    field.title = deriveLocked
      ? `Clear Group by back to Overall only before editing derived-group settings. Current Group by is ${activeGroup}.`
      : "";
  });
  refs.derivePanel?.querySelectorAll(".config-field").forEach((field) => {
    field.classList.toggle("is-disabled", deriveLocked);
    field.title = deriveLocked
      ? `Clear Group by back to Overall only before editing derived-group settings. Current Group by is ${activeGroup}.`
      : "";
  });
  if (!deriveLocked && !runtime.deriveLockMessageActive) {
    return;
  }
  runtime.deriveLockMessageActive = deriveLocked;
  if (refs.deriveStatus) {
    if (!deriveLocked) {
      refs.deriveStatus.textContent = "";
      return;
    }
    let statusText = `Derived-group settings are locked while Group by uses ${activeGroup}. Set Group by to Overall only if you want Create or these parameters to affect the next grouping analysis. Run again only reuses the current Group by value.`;
    if (displayedDerivedColumn) {
      if (activeGroup === displayedDerivedColumn) {
        statusText += controlsMismatch
          ? ` The card below describes ${displayedDerivedColumn}; the disabled controls above are draft settings only and do not describe that stored result.`
          : ` The card below describes the current derived grouping ${displayedDerivedColumn}.`;
      } else {
        statusText += ` The card below describes ${displayedDerivedColumn}, not the current Group by ${activeGroup}.`;
        if (controlsMismatch) {
          statusText += " The disabled controls above are draft settings only for the next Create action.";
        }
      }
    }
    refs.deriveStatus.textContent = statusText;
  }
}


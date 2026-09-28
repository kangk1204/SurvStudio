// SurvStudio front end, part 7/8: Machine-learning and deep-learning runs.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

// ── ML Models ──────────────────────────────────────────────────

// A run records its result mode (runtime.resultPreference) only once its result is in: a run that fails or
// is cancelled leaves the result of the other mode current and shown (preferredResultMode).
async function runMlModel() {
  if ((refs.mlEvaluationStrategy?.value || "holdout") === "repeated_cv") {
    throw new Error("Run Analysis uses deterministic holdout only. Switch Evaluation Mode back to Deterministic Holdout or use Compare All for repeated CV screening.");
  }
  const base = currentBaseConfig();
  const { features, categoricalFeatures } = currentSharedModelSelections("ml");
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  validateMlControls();
  const requestToken = beginRequestToken("ml");
  const datasetId = base.dataset_id;
  const selectedModelType = refs.mlModelType.value;
  const modelLabel = mlModelLabel(selectedModelType);
  const computeShap = mlModelSupportsShap(selectedModelType) && !refs.mlSkipShap?.checked;
  const shapSafeMode = mlModelSupportsShap(selectedModelType) && !refs.mlSkipShap?.checked && Boolean(refs.mlShapSafeMode?.checked);
  const startedAt = performance.now();
  const previousBannerText = refs.mlMetaBanner.textContent;
  const loading = beginShellLoading([refs.mlImportancePlot]);
  refs.mlMetaBanner.textContent = mlPendingBannerText({
    modelType: selectedModelType,
    nEstimators: mlSetting("n_estimators", refs.mlNEstimators),
    rowCount: Number(state.dataset?.n_rows),
    computeShap,
  });

  let payload;
  try {
    payload = await fetchJSON("/api/ml-model", {
      signal: requestSignal("ml"),
      method: "POST",
      body: JSON.stringify({
        dataset_id: base.dataset_id, time_column: base.time_column,
        event_column: base.event_column, event_positive_value: base.event_positive_value,
        features, categorical_features: categoricalFeatures,
        model_type: selectedModelType,
        ...mlModelRequestFields(selectedModelType),
        compute_shap: computeShap,
        shap_safe_mode: shapSafeMode,
      }),
    });
  } catch (error) {
    if (requestTokenMatches("ml", requestToken)) {
      loading.restore();
      refs.mlMetaBanner.textContent = previousBannerText;
    }
    throw error;
  }
  if (!requestTokenMatches("ml", requestToken) || state.dataset?.dataset_id !== datasetId) return;
  loading.finish();
  const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
  state.ml = payload;
  runtime.resultPreference.ml = "single";
  setPanelResultMode(refs.mlPanel, "single");
  refs.downloadMlComparisonButton.disabled = true;
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = true;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = true;
  setMlManuscriptDownloadsEnabled(false);
  refs.mlComparisonShell.innerHTML = '<div class="empty-state">Run a comparison to populate the cross-model table.</div>';
  if (refs.mlComparisonTitle) refs.mlComparisonTitle.textContent = "Model Comparison";
  refs.mlManuscriptShell.innerHTML = '<div class="empty-state">Run a comparison to populate manuscript-ready rows.</div>';
  resetPlotElement(refs.mlComparisonPlot);
  refs.mlComparisonPlot.classList.add("hidden");

  if (payload.importance_figure) {
    resetPlotElement(refs.mlImportancePlot);
    await Plotly.newPlot(refs.mlImportancePlot, payload.importance_figure.data, plotLayoutConfig(payload.importance_figure.layout, "ml_importance"), plotConfig("ml_importance"));
    stabilizePlotShellHeight(refs.mlImportancePlot);
    setPlotShellState(refs.mlImportancePlot, "plot");
  } else {
    clearPlotShell(refs.mlImportancePlot, '<div class="empty-state plot-empty"><span>No feature importance available</span></div>');
  }
  if (payload.shap_figure) {
    resetPlotElement(refs.mlShapPlot);
    await Plotly.newPlot(refs.mlShapPlot, payload.shap_figure.data, plotLayoutConfig(payload.shap_figure.layout, "shap_importance"), plotConfig("shap_importance"));
    stabilizePlotShellHeight(refs.mlShapPlot);
    setPlotShellState(refs.mlShapPlot, "plot");
  } else {
    clearPlotShell(
      refs.mlShapPlot,
      `<div class="empty-state plot-empty"><span>${
        payload.shap_error
          ? `SHAP failed: ${escapeHtml(payload.shap_error)}`
          : (!mlModelSupportsShap(selectedModelType)
            ? "SHAP is currently available for tree models only"
            : (computeShap ? "SHAP not available for this model" : "SHAP skipped in Fast mode"))
      }</span></div>`,
    );
    if (payload.shap_error) {
      const shapToast = payload.shap_error.includes("high-dimensional inputs")
        ? "SHAP could not be generated because the encoded feature matrix is too wide for the safe fallback path. Reduce the ML feature set to inspect SHAP."
        : `SHAP failed: ${payload.shap_error}`;
      showToast(shapToast, "warning", 5200);
    }
  }
  await stabilizePlotsBeforeBanner([refs.mlImportancePlot, refs.mlShapPlot], refs.mlMetaBanner);
  renderInsightBoard(refs.mlInsightBoard, payload.analysis?.scientific_summary, "ML model results.");
  const stats = payload.analysis?.model_stats || {};
  const mlMetricLabel = stats.metric_name || ((stats.evaluation_mode === "holdout") ? "Holdout C-index" : "Apparent C-index");
  const mlEvaluationMode = stats.evaluation_mode || "unknown";
  const shapStatus = !mlModelSupportsShap(selectedModelType)
    ? "tree-only"
    : (payload.shap_result?.safe_mode
      ? "safe-mode"
      : (payload.shap_result?.method === "kernel"
      ? "approx-screening"
      : (payload.shap_figure ? "computed" : (payload.shap_error ? "failed" : (computeShap ? "unavailable" : "skipped")))));
  const shapApproximationNote = payload.shap_result?.safe_mode
    ? ` (${formatValue(payload.shap_result?.companion_model?.selected_feature_count_raw)} raw / ${formatValue(payload.shap_result?.companion_model?.selected_feature_count_encoded)} encoded companion)`
    : (payload.shap_result?.method === "kernel"
    ? ` (${formatValue(payload.shap_result.n_samples)} eval / ${formatValue(payload.shap_result.background_samples)} bg)`
    : "");
  refs.mlMetaBanner.textContent = `${modelLabel}: ${mlMetricLabel}=${formatValue(stats.c_index)}, eval=${formatValue(mlEvaluationMode)}, N=${formatValue(stats.n_patients)}, features=${formatValue(stats.n_features)}, SHAP=${shapStatus}${shapApproximationNote}, time=${elapsedSeconds}s`;
  if (payload.shap_result?.safe_mode) {
    showToast(
      `SHAP was computed on a reduced companion model (${formatValue(payload.shap_result?.companion_model?.selected_feature_count_raw)} raw features) because the full encoded matrix was too wide.`,
      "info",
      5200,
    );
  }
  renderBenchmarkBoard();
  revealCompletedResultIfCurrent("ml", {
    mode: "single",
    successMessage: `${modelLabel} model trained`,
      backgroundMessage: `${modelLabel} model finished in the background. Open Predictive Models when you are ready to review it.`,
  });
}

async function runCompareModels({ suppressCompletionToast = false, compareGroupId = null, compareSource = "single_family_compare" } = {}) {
  const base = currentBaseConfig();
  const { features, categoricalFeatures } = currentSharedModelSelections("ml");
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  validateMlControls({ compare: true });
  const requestToken = beginRequestToken("ml");
  const datasetId = base.dataset_id;
  const evaluationStrategy = refs.mlEvaluationStrategy.value;
  const repeatedCvRequested = evaluationStrategy === "repeated_cv";
  const cvFolds = mlSetting("cv_folds", refs.mlCvFolds);
  const cvRepeats = mlSetting("cv_repeats", refs.mlCvRepeats);
  const previousBannerText = refs.mlMetaBanner.textContent;
  refs.mlMetaBanner.textContent = mlComparePendingBannerText({
    rowCount: Number(state.dataset?.n_rows),
    evaluationStrategy,
    cvFolds,
    cvRepeats,
  });
  // Inside Compare All the Compare All banner stays up for both phases.
  const runBanner = compareSource === "predictive_compare_all"
    ? 0
    : setRuntimeBanner("Screening Cox PH and, when available, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival on one shared evaluation path. This can take a little while on larger cohorts.", "info", { held: true });
  const loading = beginShellLoading([refs.mlComparisonShell]);

  try {
    let payload;
    try {
      payload = await fetchJSON("/api/ml-model", {
        signal: requestSignal("ml"),
        method: "POST",
        body: JSON.stringify({
          dataset_id: base.dataset_id, time_column: base.time_column,
          event_column: base.event_column, event_positive_value: base.event_positive_value,
          features,
          categorical_features: categoricalFeatures,
          model_type: "compare",
          ...mlModelRequestFields("compare"),
          evaluation_strategy: evaluationStrategy,
          ...(repeatedCvRequested ? { cv_folds: cvFolds, cv_repeats: cvRepeats } : {}),
          locked_test_fraction: repeatedCvRequested ? currentLockedTestFraction("ml") : null,
        }),
      });
    } catch (error) {
      if (requestTokenMatches("ml", requestToken)) {
        loading.restore();
        refs.mlMetaBanner.textContent = previousBannerText;
      }
      throw error;
    }
    if (!requestTokenMatches("ml", requestToken) || state.dataset?.dataset_id !== datasetId) return;
    loading.finish();
    tagComparePayload(payload, compareGroupId || nextCompareRunGroupId("ml-compare"), compareSource);
    state.ml = payload;
    runtime.resultPreference.ml = "compare";
    runtime.compareCache.ml = payload;
    setPanelResultMode(refs.mlPanel, "compare");

    if (payload.analysis?.comparison_table) {
      renderComparisonTable(
        refs.mlComparisonShell,
        payload.analysis,
        ["model", "c_index", "c_index_std", "c_index_interval", "locked_test_c_index", "locked_test_n", "locked_test_error", "evaluation_mode", "n_features", "training_time_ms", "rank"],
      );
    } else {
      refs.mlComparisonShell.innerHTML = '<div class="empty-state">No model returned a comparison row.</div>';
    }
    if (payload.analysis?.manuscript_tables?.model_performance_table) {
      renderTable(refs.mlManuscriptShell, payload.analysis.manuscript_tables.model_performance_table);
    }
    if (payload.figure?.data?.length) {
      refs.mlComparisonPlot.classList.remove("hidden");
      resetPlotElement(refs.mlComparisonPlot);
      await Plotly.newPlot(refs.mlComparisonPlot, payload.figure.data, payload.figure.layout, plotConfig("model_comparison"));
      markPlotResult(refs.mlComparisonPlot, payload);
      stabilizePlotShellHeight(refs.mlComparisonPlot);
    } else {
      resetPlotElement(refs.mlComparisonPlot);
      refs.mlComparisonPlot.classList.add("hidden");
    }
    renderInsightBoard(refs.mlInsightBoard, payload.analysis?.scientific_summary, "Model comparison.");
    const comparisonRows = payload.analysis?.comparison_table || [];
    const bestRow = comparisonRows[0] || {};
    const evaluationMode = payload.analysis?.evaluation_mode || "unknown";
    const repeatedCvLike = evaluationMode === "repeated_cv" || evaluationMode === "repeated_cv_incomplete";
    const evalLabel = compareEvaluationLabel(payload.analysis, evaluationMode);
    const mlMetricLabel = repeatedCvLike ? "Mean C-index" : "C-index";
    refs.mlMetaBanner.textContent = `Screening top model=${formatValue(bestRow.model)}, ${mlMetricLabel}=${formatValue(bestRow.c_index)}, eval=${formatValue(evalLabel)}, models=${formatValue(comparisonRows.length)}${lockedTestBannerSuffix(payload.analysis, bestRow)}`;
    refs.downloadMlComparisonButton.disabled = comparisonRows.length === 0;
    if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = !plotShowsResult(refs.mlComparisonPlot, payload);
    if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = !plotShowsResult(refs.mlComparisonPlot, payload);
    setMlManuscriptDownloadsEnabled(!!(payload.analysis?.manuscript_tables?.model_performance_table?.length));
    renderBenchmarkBoard();
    if (!suppressCompletionToast) {
      revealCompletedResultIfCurrent("ml", {
        mode: "compare",
        successMessage: "Model comparison screening complete",
        backgroundMessage: "ML model comparison finished in the background. Open Predictive Models when you are ready to review it.",
      });
    }
  } finally {
    // Only this run's banner: a newer run (or a dataset load) may own the banner by now.
    releaseRuntimeBanner(runBanner);
  }
}

async function runPredictiveSelectedModel() {
  runtime.workbenchRevealed = true;
  runtime.predictiveWorkbenchIntent = "train";
  const selectedModel = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey());
  setPredictiveModel(selectedModel.key, { syncHistory: false });
  activateTab("benchmark", { historyMode: "replace", syncHistory: false });
  if (selectedModel.family === "ml") {
    await runMlModel();
    return;
  }
  await runDlModel();
}

async function runUnifiedPredictiveComparison() {
  if (isScopeBusy("ml") || isScopeBusy("dl")) {
    showToast("Wait for the current predictive run to finish before starting Compare All Models again.", "warning", 3200);
    return;
  }
  const startFamily = predictiveFamilyGoal();
  // Both families must see the same seed, evaluation mode, CV design, and locked test set.
  alignPredictiveEvaluationControls(startFamily);
  validatePredictiveEvaluationControls(startFamily);
  const startDatasetId = state.dataset?.dataset_id;
  const previousMlPayload = state.ml;
  const previousDlPayload = state.dl;
  const sharedCompareGroupId = nextCompareRunGroupId("predictive-compare-all");
  // A cancelled phase (a new dataset, derived snapshot or endpoint cleared the results) ends the whole
  // comparison: nothing from the old cohort is restored and no further run starts on the new one.
  const superseded = (attempt) => Boolean(attempt?.superseded) || state.dataset?.dataset_id !== startDatasetId;
  const runBanner = setRuntimeBanner("Comparing the full predictive stack across classical ML and deep learning. This can take several minutes on larger cohorts.", "info", { held: true });
  try {
    const mlAttempt = await withLoading(
      refs.runCompareButton,
      () => runCompareModels({
        suppressCompletionToast: true,
        compareGroupId: sharedCompareGroupId,
        compareSource: "predictive_compare_all",
      }),
      "ml",
    );
    if (superseded(mlAttempt)) return;
    const mlFreshCompare = Boolean(mlAttempt?.ok && benchmarkCompareRows("ml").length);
    if (!mlFreshCompare) {
      restorePredictiveFamilyAfterFailedCompare("ml", previousMlPayload);
    }

    const dlAttempt = await withLoading(
      refs.runDlCompareButton,
      () => runDlCompareModels({
        suppressCompletionToast: true,
        compareGroupId: sharedCompareGroupId,
        compareSource: "predictive_compare_all",
      }),
      "dl",
    );
    if (superseded(dlAttempt)) return;
    const dlFreshCompare = Boolean(dlAttempt?.ok && benchmarkCompareRows("dl").length);
    if (!dlFreshCompare) {
      restorePredictiveFamilyAfterFailedCompare("dl", previousDlPayload);
    }

    const familyCount = Number(mlFreshCompare) + Number(dlFreshCompare);
    if (familyCount === 2) {
      runtime.compareCache.unified = {
        ml: compareGoalPayload("ml"),
        dl: compareGoalPayload("dl"),
        group_id: sharedCompareGroupId,
      };
    }
    setPredictiveWorkbenchFamily(startFamily, { syncHistory: false });
    renderBenchmarkBoard();
    if (familyCount === 2) {
      // Shown in place when the Prediction tab is open; otherwise the user stays where they are.
      revealCompletedResultIfCurrent("predictive", {
        successMessage: "Unified predictive comparison complete.",
        backgroundMessage: "Compare All Models finished in the background. Open Prediction models to review the leaderboard.",
      });
    } else if (familyCount === 1) {
      showToast("Predictive comparison finished, but only one model family returned comparison rows. Review the board and any error messages before trusting the result.", "warning", 4200);
    } else {
      showToast("Predictive comparison did not produce any fresh leaderboard rows. Review the error messages before trusting the board.", "error", 4200);
    }
  } finally {
    releaseRuntimeBanner(runBanner);
  }
}

// ── Deep Learning ──────────────────────────────────────────────

async function runDlModel() {
  const base = currentBaseConfig();
  validateDlControls();
  const { features, categoricalFeatures } = currentSharedModelSelections("dl");
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  const requestToken = beginRequestToken("dl");
  const datasetId = base.dataset_id;
  const modelType = refs.dlModelType.value;
  const modelLabel = dlModelLabel(modelType);
  const startedAt = performance.now();

  const previousBannerText = refs.dlMetaBanner.textContent;
  const loading = beginShellLoading([refs.dlImportancePlot, refs.dlLossPlot]);
  refs.dlMetaBanner.textContent = dlPendingBannerText({
    modelType,
    rowCount: Number(state.dataset?.n_rows),
    epochs: dlSetting("epochs", refs.dlEpochs),
    evaluationStrategy: refs.dlEvaluationStrategy.value,
    cvFolds: dlSetting("cv_folds", refs.dlCvFolds),
    cvRepeats: dlSetting("cv_repeats", refs.dlCvRepeats),
  });
  const runBanner = setRuntimeBanner("Training the selected deep-learning model. This can take noticeably longer than a classical fit.", "info", { held: true });

  try {
    let payload;
    try {
      payload = await fetchJSON("/api/deep-model", {
        signal: requestSignal("dl"),
        method: "POST",
        body: JSON.stringify({
          dataset_id: base.dataset_id, time_column: base.time_column,
          event_column: base.event_column, event_positive_value: base.event_positive_value,
          features, categorical_features: categoricalFeatures,
          model_type: modelType,
          ...dlArchitectureRequestFields(modelType),
          locked_test_fraction: null,
        }),
      });
    } catch (error) {
      if (requestTokenMatches("dl", requestToken)) {
        loading.restore();
        refs.dlMetaBanner.textContent = previousBannerText;
      }
      throw error;
    }
    if (!requestTokenMatches("dl", requestToken) || state.dataset?.dataset_id !== datasetId) return;
    loading.finish();
    const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    state.dl = payload;
    runtime.resultPreference.dl = "single";
    setPanelResultMode(refs.dlPanel, "single");
    const stats = payload.analysis || {};

    if (payload.figures?.importance) {
      resetPlotElement(refs.dlImportancePlot);
      await Plotly.newPlot(refs.dlImportancePlot, payload.figures.importance.data, plotLayoutConfig(payload.figures.importance.layout, "dl_importance"), plotConfig("dl_importance"));
      stabilizePlotShellHeight(refs.dlImportancePlot);
      setPlotShellState(refs.dlImportancePlot, "plot");
    } else {
      const importanceEmpty = (stats?.evaluation_mode === "repeated_cv" || stats?.evaluation_mode === "repeated_cv_incomplete")
        ? '<div class="empty-state plot-empty"><span>Repeated-CV aggregate runs do not emit single-fit gradient salience.</span></div>'
        : '<div class="empty-state plot-empty"><span>No feature importance available</span></div>';
      clearPlotShell(refs.dlImportancePlot, importanceEmpty);
    }
    if (payload.figures?.loss) {
      resetPlotElement(refs.dlLossPlot);
      await Plotly.newPlot(refs.dlLossPlot, payload.figures.loss.data, plotLayoutConfig(payload.figures.loss.layout, "dl_loss"), plotConfig("dl_loss"));
      stabilizePlotShellHeight(refs.dlLossPlot);
      setPlotShellState(refs.dlLossPlot, "plot");
    } else {
      const lossEmpty = (stats?.evaluation_mode === "repeated_cv" || stats?.evaluation_mode === "repeated_cv_incomplete")
        ? '<div class="empty-state plot-empty"><span>Repeated-CV aggregate runs do not emit a single training loss curve.</span></div>'
        : '<div class="empty-state plot-empty"><span>No loss curve available</span></div>';
      clearPlotShell(refs.dlLossPlot, lossEmpty);
    }
    const repeatedCvLike = stats.evaluation_mode === "repeated_cv" || stats.evaluation_mode === "repeated_cv_incomplete";
    if (repeatedCvLike && Array.isArray(stats.repeat_results) && stats.repeat_results.length) {
      if (refs.dlComparisonTitle) refs.dlComparisonTitle.textContent = "Repeated-CV Repeat Summary";
      renderTable(refs.dlComparisonShell, stats.repeat_results);
    } else {
      if (refs.dlComparisonTitle) refs.dlComparisonTitle.textContent = "Deep Model Comparison";
      refs.dlComparisonShell.innerHTML = '<div class="empty-state">Run "Compare All" to benchmark all deep models on the same feature set.</div>';
    }
    if (repeatedCvLike && payload.analysis?.manuscript_tables?.model_performance_table) {
      renderTable(refs.dlManuscriptShell, payload.analysis.manuscript_tables.model_performance_table);
    } else {
      refs.dlManuscriptShell.innerHTML = '<div class="empty-state">Run "Compare All" to populate manuscript-ready deep comparison rows.</div>';
    }
    resetPlotElement(refs.dlComparisonPlot);
    refs.dlComparisonPlot.classList.add("hidden");
    refs.downloadDlComparisonButton.disabled = !(Array.isArray(stats.comparison_table) && stats.comparison_table.length);
    if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = true;
    if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = true;
    setDlManuscriptDownloadsEnabled(!!(repeatedCvLike && payload.analysis?.manuscript_tables?.model_performance_table?.length));
    // Backend may emit either `scientific_summary` or `insight_board` depending on model implementation.
    const dlSummary = payload.analysis?.scientific_summary || payload.analysis?.insight_board || null;
    renderInsightBoard(refs.dlInsightBoard, dlSummary, "Deep learning results.");
    const epochsTrained = stats.epochs_trained ?? stats.epochs ?? payload.request_config?.epochs;
    // The early-stopping run's length; epochs_trained is the refit that produced the reported model.
    const earlyStoppingEpochs = stats.early_stopping_epochs ?? epochsTrained;
    const dlMetricLabel = stats.evaluation_mode === "repeated_cv"
      ? "Mean repeated-CV C-index"
      : (stats.evaluation_mode === "repeated_cv_incomplete"
        ? "Mean repeated-CV C-index (incomplete)"
        : (stats.evaluation_mode === "holdout"
          ? "Holdout C-index"
          : (stats.evaluation_mode === "holdout_fallback_apparent" ? "Apparent fallback C-index" : "Apparent C-index")));
    const dlEvalLabel = stats.evaluation_mode === "repeated_cv"
      ? repeatedCvDesignLabel(stats)
      : (stats.evaluation_mode === "repeated_cv_incomplete"
        ? `${repeatedCvDesignLabel(stats)} (incomplete; fallback folds excluded)`
        : (stats.evaluation_mode === "holdout_fallback_apparent"
          ? "holdout requested, reported as apparent fallback"
          : formatValue(stats.evaluation_mode)));
    const dlSeedSuffix = repeatedCvLike
      ? (Array.isArray(stats.training_seeds) && stats.training_seeds.length
        ? `, repeat seeds=${stats.training_seeds.join(", ")}`
        : "")
      : (stats.training_seed != null ? `, seed=${formatValue(stats.training_seed)}` : "");
    const dlTrainingStatus = repeatedCvLike
      ? ""
      : (stats.stopped_early
        ? `, stopped early at epoch ${formatValue(earlyStoppingEpochs)}`
        : (stats.max_epochs_requested != null && Number(earlyStoppingEpochs) >= Number(stats.max_epochs_requested)
          ? `, trained to max epoch (${formatValue(stats.max_epochs_requested)})`
          : ""));
    const dlBestMonitorSuffix = repeatedCvLike
      ? ""
      : (stats.best_monitor_epoch != null ? `, best monitor epoch=${formatValue(stats.best_monitor_epoch)}` : "");
    // Label the banner with the model that was actually trained, not the live dropdown.
    const trainedModelType = String(payload.request_config?.model_type || modelType);
    const trainedModelTag = trainedModelType.toUpperCase();
    refs.dlMetaBanner.textContent = `${trainedModelTag}: ${dlMetricLabel}=${formatValue(stats.c_index)}, eval=${dlEvalLabel}, epochs=${formatValue(epochsTrained)}${dlBestMonitorSuffix}${dlTrainingStatus}${dlSeedSuffix}, time=${elapsedSeconds}s`;
    renderBenchmarkBoard();
    revealCompletedResultIfCurrent("dl", {
      mode: repeatedCvLike ? "compare" : "single",
      successMessage: `${modelLabel} model trained`,
      backgroundMessage: `${modelLabel} model finished in the background. Open Predictive Models when you are ready to review it.`,
    });
  } finally {
    releaseRuntimeBanner(runBanner);
  }
}

async function runDlCompareModels({ suppressCompletionToast = false, compareGroupId = null, compareSource = "single_family_compare" } = {}) {
  const base = currentBaseConfig();
  validateDlControls({ compare: true });
  const { features, categoricalFeatures } = currentSharedModelSelections("dl");
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  const requestToken = beginRequestToken("dl");
  const datasetId = base.dataset_id;
  const evaluationStrategy = refs.dlEvaluationStrategy.value;

  const previousBannerText = refs.dlMetaBanner.textContent;
  refs.dlMetaBanner.textContent = dlComparePendingBannerText({
    rowCount: Number(state.dataset?.n_rows),
    evaluationStrategy,
    cvFolds: dlSetting("cv_folds", refs.dlCvFolds),
    cvRepeats: dlSetting("cv_repeats", refs.dlCvRepeats),
  });
  // Inside Compare All the Compare All banner stays up for both phases.
  const runBanner = compareSource === "predictive_compare_all"
    ? 0
    : setRuntimeBanner("Comparing all deep-learning models. This can take noticeably longer than a single run.", "info", { held: true });
  const loading = beginShellLoading([refs.dlComparisonShell]);

  try {
    let payload;
    try {
      payload = await fetchJSON("/api/deep-model", {
        signal: requestSignal("dl"),
        method: "POST",
        body: JSON.stringify({
          dataset_id: base.dataset_id,
          time_column: base.time_column,
          event_column: base.event_column,
          event_positive_value: base.event_positive_value,
          features,
          categorical_features: categoricalFeatures,
          model_type: "compare",
          ...dlArchitectureRequestFields("compare"),
          locked_test_fraction: evaluationStrategy === "repeated_cv" ? currentLockedTestFraction("dl") : null,
        }),
      });
    } catch (error) {
      if (requestTokenMatches("dl", requestToken)) {
        loading.restore();
        refs.dlMetaBanner.textContent = previousBannerText;
      }
      throw error;
    }
    if (!requestTokenMatches("dl", requestToken) || state.dataset?.dataset_id !== datasetId) return;
    loading.finish();
    tagComparePayload(payload, compareGroupId || nextCompareRunGroupId("dl-compare"), compareSource);
    state.dl = payload;
    runtime.resultPreference.dl = "compare";
    runtime.compareCache.dl = payload;
    setPanelResultMode(refs.dlPanel, "compare");

    if (payload.analysis?.comparison_table?.length) {
      if (refs.dlComparisonTitle) refs.dlComparisonTitle.textContent = "Deep Model Comparison";
      renderComparisonTable(
        refs.dlComparisonShell,
        payload.analysis,
        ["model", "c_index", "c_index_std", "c_index_interval", "locked_test_c_index", "locked_test_n", "locked_test_error", "evaluation_mode", "epochs_trained", "n_features", "training_time_ms", "rank"],
      );
    } else {
      refs.dlComparisonShell.innerHTML = '<div class="empty-state">No deep model returned a comparison row.</div>';
    }
    if (payload.analysis?.manuscript_tables?.model_performance_table) {
      renderTable(refs.dlManuscriptShell, payload.analysis.manuscript_tables.model_performance_table);
    }
    if (payload.figures?.comparison?.data?.length) {
      refs.dlComparisonPlot.classList.remove("hidden");
      resetPlotElement(refs.dlComparisonPlot);
      await Plotly.newPlot(refs.dlComparisonPlot, payload.figures.comparison.data, payload.figures.comparison.layout, plotConfig("dl_model_comparison"));
      markPlotResult(refs.dlComparisonPlot, payload);
      stabilizePlotShellHeight(refs.dlComparisonPlot);
    } else {
      resetPlotElement(refs.dlComparisonPlot);
      refs.dlComparisonPlot.classList.add("hidden");
    }
    clearPlotShell(refs.dlImportancePlot, '<div class="empty-state plot-empty"><span>Single-model feature importance appears when you train one deep model.</span></div>');
    clearPlotShell(refs.dlLossPlot, '<div class="empty-state plot-empty"><span>Single-model training and monitor metric curves appear when you train one deep model.</span></div>');
    const dlSummary = payload.analysis?.scientific_summary || payload.analysis?.insight_board || null;
    renderInsightBoard(refs.dlInsightBoard, dlSummary, "Deep learning comparison results.");
    const bestRow = payload.analysis?.comparison_table?.[0] || {};
    const dlEvalMode = payload.analysis?.evaluation_mode || "unknown";
    const dlEvalLabel = compareEvaluationLabel(payload.analysis, dlEvalMode);
    const dlBestLabel = dlEvalMode === "mixed_holdout_apparent" ? "Screening top holdout-comparable" : "Screening top model";
    const dlMetricLabel = dlEvalMode === "mixed_holdout_apparent"
      ? "Best holdout C-index"
      : (dlEvalMode === "repeated_cv"
        ? "Screening mean C-index"
        : (dlEvalMode === "repeated_cv_incomplete" ? "Screening mean C-index (incomplete)" : "C-index"));
    const rerunSeedSuffix = (bestRow.training_seed != null && dlEvalMode !== "repeated_cv")
      ? `, rerun seed=${formatValue(bestRow.training_seed)}`
      : "";
    const repeatedCvRerunNote = dlEvalMode === "repeated_cv"
      ? ", rerun a single architecture with Run Analysis while keeping repeated CV selected"
      : "";
    refs.dlMetaBanner.textContent = `${dlBestLabel}=${formatValue(bestRow.model)}, ${dlMetricLabel}=${formatValue(bestRow.c_index)}, eval=${formatValue(dlEvalLabel)}, models=${formatValue(payload.analysis?.comparison_table?.length || 0)}${lockedTestBannerSuffix(payload.analysis, bestRow)}${rerunSeedSuffix}${repeatedCvRerunNote}`;
    refs.downloadDlComparisonButton.disabled = !(payload.analysis?.comparison_table?.length);
    if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = !plotShowsResult(refs.dlComparisonPlot, payload);
    if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = !plotShowsResult(refs.dlComparisonPlot, payload);
    setDlManuscriptDownloadsEnabled(!!(payload.analysis?.manuscript_tables?.model_performance_table?.length));
    renderBenchmarkBoard();
    if (!suppressCompletionToast) {
      revealCompletedResultIfCurrent("dl", {
        mode: "compare",
        successMessage: "Deep learning model comparison complete",
        backgroundMessage: "Deep learning model comparison finished in the background. Open Predictive Models when you are ready to review it.",
      });
    }
  } finally {
    releaseRuntimeBanner(runBanner);
  }
}

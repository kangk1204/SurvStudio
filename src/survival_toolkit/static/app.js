// SurvStudio front end, part 8/8: Download wiring, event listeners, and startup.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

if (!window.SurvStudioBenchmark?.createBenchmarkBoardApi) {
  throw new Error("SurvStudio benchmark module failed to load.");
}

const benchmarkBoardApi = window.SurvStudioBenchmark.createBenchmarkBoardApi({
  refs,
  state,
  runtime,
  Plotly,
  currentCompareGoalPayload,
  compareGoalPayload,
  goalPayload,
  panelModeForPayload,
  benchmarkGoalMeta,
  benchmarkResultLabel,
  benchmarkMetricNumber,
  benchmarkEvaluationLabel,
  benchmarkReviewAction,
  mlModelLabel,
  dlModelLabel,
  formatValue,
  escapeHtml,
  isScopeBusy,
  clearPlotShell,
  purgePlot,
  plotLayoutConfig,
  plotConfig,
  stabilizePlotShellHeight,
  renderPredictiveWorkbench,
  syncPredictiveWorkbenchCompareVisibility,
  showError,
});

const benchmarkCompareRows = benchmarkBoardApi.benchmarkCompareRows;
const benchmarkBoardState = benchmarkBoardApi.benchmarkBoardState;
const renderUnifiedBenchmarkPlot = benchmarkBoardApi.renderUnifiedBenchmarkPlot;
const renderUnifiedBenchmarkSummary = benchmarkBoardApi.renderUnifiedBenchmarkSummary;
const renderUnifiedBenchmarkTable = benchmarkBoardApi.renderUnifiedBenchmarkTable;
const renderBenchmarkBoard = benchmarkBoardApi.renderBenchmarkBoard;

// ── Downloads ──────────────────────────────────────────────────

function wireDownloads() {
  refs.downloadKmSummaryButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("km_summary", "csv", { includeGroup: true }),
      buildKmTableExportPayload(payload.analysis.summary_table, "Kaplan-Meier summary", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadKmPairwiseButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("km_pairwise", "csv", { includeGroup: true }),
      buildKmTableExportPayload(payload.analysis.pairwise_table, "Kaplan-Meier pairwise comparisons", payload, { includePairwiseGuardrail: true }),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadSignatureButton.addEventListener("click", () => {
    const payload = currentSignatureResult();
    if (!payload || isScopeBusy("km")) {
      showToast("Visible settings no longer match the current signature result. Run again before exporting.", "warning", 3600);
      return;
    }
    void downloadServerTable(
      buildDownloadFilename("signature_ranking", "csv"),
      buildSignatureTableExportPayload(payload.results_table, "Signature discovery ranking", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadCoxResultsButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("cox_results", "csv"),
      buildCoxTableExportPayload(payload.analysis.results_table, "Cox proportional hazards results", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadCoxDiagnosticsButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("cox_diagnostics", "csv"),
      buildCoxTableExportPayload(payload.analysis.diagnostics_table, "Cox proportional hazards diagnostics", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    if (!requireCurrentPlotForExport(refs.coxPlot, payload)) return;
    downloadPlotImage(refs.coxPlot, buildDownloadFilename("cox_forest", "png").replace(/\.png$/, ""), "png");
  });
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    if (!requireCurrentPlotForExport(refs.coxPlot, payload)) return;
    downloadPlotImage(refs.coxPlot, buildDownloadFilename("cox_forest", "svg").replace(/\.svg$/, ""), "svg");
  });
  refs.downloadCohortTableButton.addEventListener("click", () => {
    const payload = state.cohort;
    if (!requireCurrentResultForExport("tables", { payload })) return;
    const exportPayload = buildCohortTableExportPayload("csv");
    downloadCsv(
      buildDownloadFilename("cohort_summary", "csv", { includeGroup: true, group: cohortTableOutputGroup() }),
      payload?.analysis?.rows,
      payload?.analysis?.columns,
      { caption: exportPayload.caption, notes: exportPayload.notes },
    );
  });
  if (refs.downloadCohortTableXlsxButton) refs.downloadCohortTableXlsxButton.addEventListener("click", () => {
    const payload = state.cohort;
    if (!requireCurrentResultForExport("tables", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("cohort_summary", "xlsx", { includeGroup: true, group: cohortTableOutputGroup() }),
      buildCohortTableExportPayload("xlsx"),
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadMlComparisonButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    const rows = payload?.analysis?.comparison_table;
    if (!requireCurrentResultForExport("ml", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("ml_model_comparison", "csv"),
      buildComparisonTableExportPayload(rows, "ML model comparison", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    if (!requireCurrentPlotForExport(refs.mlComparisonPlot, payload)) return;
    downloadPlotImage(refs.mlComparisonPlot, buildDownloadFilename("ml_model_comparison", "png").replace(/\.png$/, ""), "png");
  });
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    if (!requireCurrentPlotForExport(refs.mlComparisonPlot, payload)) return;
    downloadPlotImage(refs.mlComparisonPlot, buildDownloadFilename("ml_model_comparison", "svg").replace(/\.svg$/, ""), "svg");
  });
  refs.downloadMlManuscriptCsvButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "csv", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "csv", currentMlJournalTemplate(), "Model discrimination summary", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadMlManuscriptMarkdownButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "md", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "markdown", currentMlJournalTemplate(), "Model discrimination summary", payload),
      "text/markdown;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadMlManuscriptLatexButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "tex", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "latex", currentMlJournalTemplate(), "Model discrimination summary", payload),
      "text/x-tex;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadMlManuscriptDocxButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "docx", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "docx", currentMlJournalTemplate(), "Model discrimination summary", payload),
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadDlComparisonButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    const rows = payload?.analysis?.comparison_table;
    if (!requireCurrentResultForExport("dl", { payload })) return;
    void downloadServerTable(
      buildDownloadFilename("dl_model_comparison", "csv"),
      buildComparisonTableExportPayload(rows, "Deep learning model comparison", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    if (!requireCurrentPlotForExport(refs.dlComparisonPlot, payload)) return;
    downloadPlotImage(refs.dlComparisonPlot, buildDownloadFilename("dl_model_comparison", "png").replace(/\.png$/, ""), "png");
  });
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    if (!requireCurrentPlotForExport(refs.dlComparisonPlot, payload)) return;
    downloadPlotImage(refs.dlComparisonPlot, buildDownloadFilename("dl_model_comparison", "svg").replace(/\.svg$/, ""), "svg");
  });
  refs.downloadDlManuscriptCsvButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "csv", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "csv", currentDlJournalTemplate(), "Deep model discrimination summary", payload),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadDlManuscriptMarkdownButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "md", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "markdown", currentDlJournalTemplate(), "Deep model discrimination summary", payload),
      "text/markdown;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadDlManuscriptLatexButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "tex", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "latex", currentDlJournalTemplate(), "Deep model discrimination summary", payload),
      "text/x-tex;charset=utf-8;",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  refs.downloadDlManuscriptDocxButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    const manuscript = payload?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "docx", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "docx", currentDlJournalTemplate(), "Deep model discrimination summary", payload),
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ).catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  if (refs.downloadKmPngButton) refs.downloadKmPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
    if (!requireCurrentPlotForExport(refs.kmPlot, payload)) return;
    downloadPlotImage(refs.kmPlot, buildDownloadFilename("km_curve", "png", { includeGroup: true }).replace(/\.png$/, ""), "png");
  });
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
    if (!requireCurrentPlotForExport(refs.kmPlot, payload)) return;
    downloadPlotImage(refs.kmPlot, buildDownloadFilename("km_curve", "svg", { includeGroup: true }).replace(/\.svg$/, ""), "svg");
  });
}

// ── Utilities ──────────────────────────────────────────────────

async function withLoading(button, action, scopeOverride = null, { swallowErrors = true } = {}) {
  const scope = scopeOverride || (
    button === refs.runMlButton || button === refs.runCompareButton || button === refs.runCompareInlineButton ? "ml"
      : button === refs.runDlButton || button === refs.runDlCompareButton || button === refs.runDlCompareInlineButton ? "dl"
        : button === refs.runKmButton ? "km"
          : button === refs.runCoxButton ? "cox"
            : button === refs.runCohortTableButton ? "tables"
              : null
  );
  if (scope && isScopeBusy(scope)) return;
  if (scope) {
    setScopeBusy(scope, true, button);
  } else {
    setButtonLoading(button, true);
  }
  setRuntimeBanner("");
  try {
    const value = await action();
    return { ok: true, value };
  } catch (error) {
    if (isSupersededRequestError(error)) return { ok: false, error, superseded: true };
    showError(errorMessageText(error));
    if (!swallowErrors) throw error;
    return { ok: false, error };
  } finally {
    if (scope) {
      setScopeBusy(scope, false, button);
    } else {
      setButtonLoading(button, false);
    }
  }
}

async function initializeRuntime() {
  setUiMode(runtime.uiMode, { syncHistory: false });
  syncHistoryState("replace");
  if (!runtime.isFilePreview) { setRuntimeBanner(""); return; }
  try {
    await fetchJSON("/api/health");
    setRuntimeBanner("Direct file preview connected to local API at http://127.0.0.1:8000.", "success");
  } catch {
    setRuntimeBanner("Start `python -m survival_toolkit` and refresh, or open http://127.0.0.1:8000.", "warning");
  }
}

function getActiveRunButton() {
  const tab = document.querySelector(".tab-button.active")?.dataset.tab;
  if (tab === "km") return refs.runKmButton;
  if (tab === "cox") return refs.runCoxButton;
  if (tab === "tables") return refs.runCohortTableButton;
  if (tab === "ml") return refs.runMlButton;
  if (tab === "dl") return refs.runDlButton;
  if (tab === "benchmark") {
    if (runtime.workbenchRevealed && refs.runPredictiveWorkbenchButton && !refs.runPredictiveWorkbenchButton.classList.contains("hidden")) {
      return refs.runPredictiveWorkbenchButton;
    }
    return runtime.workbenchRevealed ? null : refs.runPredictiveCompareAllButton;
  }
  return null;
}

function getActiveRunAction() {
  const tab = document.querySelector(".tab-button.active")?.dataset.tab;
  if (tab === "km") return runKaplanMeier;
  if (tab === "cox") return runCox;
  if (tab === "tables") return runCohortTable;
  if (tab === "ml") return runMlModel;
  if (tab === "dl") return runDlModel;
  if (tab === "benchmark") {
    if (runtime.workbenchRevealed && refs.runPredictiveWorkbenchButton && !refs.runPredictiveWorkbenchButton.classList.contains("hidden")) {
      return runPredictiveSelectedModel;
    }
    return runtime.workbenchRevealed ? null : runUnifiedPredictiveComparison;
  }
  return null;
}

function focusStudyDesignSection() {
  refs.configStrip?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function focusTabWorkspace(tabName, { historyMode = "push" } = {}) {
  activateTab(tabName, { historyMode });
  requestAnimationFrame(() => {
    document.querySelector(`.tab-panel[data-panel="${tabName}"] .workspace-card`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  });
}

function reviewBenchmarkSourceTab(tabName, mode = null) {
  const nextMode = mode || (tabName === "ml" ? benchmarkPanelMode("ml") : benchmarkPanelMode("dl"));
  setPredictiveWorkbenchFamily(tabName, { syncHistory: false });
  activateTab("benchmark", { historyMode: "push", setGuidedGoal: false });
  requestAnimationFrame(() => {
    const sectionTarget = tabName === "ml" ? (refs.benchmarkMlMount || refs.mlWorkspaceCard) : (refs.benchmarkDlMount || refs.dlWorkspaceCard);
    if (sectionTarget) {
      sectionTarget.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    scrollToAnalysisResult(tabName, { mode: nextMode || "single" });
  });
}

function reviewBenchmarkModel(modelKey, mode = null) {
  runtime.workbenchRevealed = true;
  runtime.predictiveWorkbenchIntent = "train";
  const meta = predictiveModelMeta(modelKey);
  setPredictiveModel(meta.key, { syncHistory: false });
  if (runtime.uiMode === "guided" && runtime.guidedGoal === "predictive") {
    setGuidedStep(4, { syncHistory: false, scroll: false, historyMode: "replace" });
  }
  reviewBenchmarkSourceTab(meta.family, mode);
}

function closePredictiveWorkbench() {
  if (!runtime.workbenchRevealed) return;
  runtime.workbenchRevealed = false;
  runtime.predictiveWorkbenchIntent = null;
  const returnToPredictiveLeaderboard = runtime.uiMode === "guided"
    && runtime.guidedGoal === "predictive"
    && guidedPredictiveHasLeaderboardReference();
  if (returnToPredictiveLeaderboard) {
    activateTab("benchmark", { setGuidedGoal: false, historyMode: "replace", syncHistory: false });
    setGuidedStep(5, { syncHistory: false, scroll: false, historyMode: "replace" });
  }
  renderBenchmarkBoard();
  syncPredictiveWorkbenchSingleResultVisibility();
  if (!returnToPredictiveLeaderboard) {
    renderGuidedChrome();
  }
  if (state.dataset) syncHistoryState("push");
  requestAnimationFrame(() => {
    const closeTarget = returnToPredictiveLeaderboard
      ? (refs.benchmarkComparisonShell?.closest(".table-card")
        || refs.benchmarkComparisonPlot?.closest(".table-card")
        || refs.benchmarkSummaryGrid)
      : refs.benchmarkActionCard;
    closeTarget?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function isVisibleResultNode(node) {
  return Boolean(node && !node.classList?.contains("hidden"));
}

function hasRenderedTable(shell) {
  return Boolean(shell && !shell.querySelector(".empty-state") && shell.querySelector("table"));
}

function hasRenderedInsight(board) {
  return Boolean(board && !board.querySelector(".empty-state") && board.textContent.trim());
}

function hasRenderedPlot(plot) {
  if (!plot || plot.classList?.contains("hidden")) return false;
  if (plot.querySelector(".empty-state")) return false;
  if (Array.isArray(plot.data) && plot.data.length) return true;
  return Boolean(plot.querySelector(".js-plotly-plot, .plotly, .main-svg"));
}

function hasPlotMessage(plot) {
  return Boolean(
    plot
    && !plot.classList?.contains("hidden")
    && plot.dataset?.plotState === "message"
    && plot.querySelector(".empty-state"),
  );
}

function setGuidedResultNodeVisible(node, visible) {
  if (!node) return;
  node.classList.toggle("guided-result-hidden", !visible);
}

function updateGuidedResultVisibility() {
  const trackedNodes = [
    refs.kmPlot,
    refs.kmMetaBanner,
    refs.kmInsightBoard,
    refs.kmSummaryShell?.closest(".table-card"),
    refs.kmRiskShell?.closest(".table-card"),
    refs.kmPairwiseShell?.closest(".table-card"),
    refs.signatureInsightBoard?.closest(".table-card"),
    refs.signatureShell?.closest(".table-card"),
    refs.coxPlot,
    refs.coxMetaBanner,
    refs.coxInsightBoard,
    refs.coxResultsShell?.closest(".table-card"),
    refs.coxDiagnosticsPlot,
    refs.coxDiagnosticsShell?.closest(".table-card"),
    refs.coxMartingalePlot,
    refs.coxMartingalePlot?.closest(".table-card"),
    refs.mlImportancePlot,
    refs.mlShapPlot,
    refs.mlImportancePlot?.closest(".ml-plots-grid"),
    refs.mlComparisonPlot,
    refs.mlMetaBanner,
    refs.mlInsightBoard,
    refs.mlComparisonShell?.closest(".table-card"),
    refs.mlManuscriptShell?.closest(".table-card"),
    refs.dlImportancePlot,
    refs.dlLossPlot,
    refs.dlImportancePlot?.closest(".ml-plots-grid"),
    refs.dlComparisonPlot,
    refs.dlMetaBanner,
    refs.dlInsightBoard,
    refs.dlComparisonShell?.closest(".table-card"),
    refs.dlManuscriptShell?.closest(".table-card"),
    refs.cohortTableShell?.closest(".table-card"),
  ];
  trackedNodes.forEach((node) => setGuidedResultNodeVisible(node, true));

  const guidedReview = runtime.uiMode === "guided" && currentGuidedStep() === 5 && Boolean(runtime.guidedGoal);
  if (!guidedReview) return;

  const goal = runtime.guidedGoal === "predictive" ? predictiveFamilyGoal() : runtime.guidedGoal;
  const reveal = (node, visible) => setGuidedResultNodeVisible(node, visible);

  if (goal === "km") {
    const hasPlot = hasRenderedPlot(refs.kmPlot);
    const hasInsight = hasRenderedInsight(refs.kmInsightBoard);
    const hasSummary = hasRenderedTable(refs.kmSummaryShell);
    const hasRisk = hasRenderedTable(refs.kmRiskShell);
    const hasPairwise = hasRenderedTable(refs.kmPairwiseShell);
    const hasSignatureInsight = hasRenderedInsight(refs.signatureInsightBoard);
    const hasSignatureTable = hasRenderedTable(refs.signatureShell);
    const hasAny = hasPlot || hasInsight || hasSummary || hasRisk || hasPairwise || hasSignatureInsight || hasSignatureTable;

    reveal(refs.kmPlot, hasPlot);
    reveal(refs.kmMetaBanner, hasAny);
    reveal(refs.kmInsightBoard, hasInsight);
    reveal(refs.kmSummaryShell?.closest(".table-card"), hasSummary);
    reveal(refs.kmRiskShell?.closest(".table-card"), hasRisk);
    reveal(refs.kmPairwiseShell?.closest(".table-card"), hasPairwise);
    reveal(refs.signatureInsightBoard?.closest(".table-card"), hasSignatureInsight);
    reveal(refs.signatureShell?.closest(".table-card"), hasSignatureTable);
  }

  if (goal === "cox") {
    const hasPlot = hasRenderedPlot(refs.coxPlot);
    const hasDiagnosticsPlot = hasRenderedPlot(refs.coxDiagnosticsPlot);
    const hasMartingalePlot = hasRenderedPlot(refs.coxMartingalePlot);
    const hasInsight = hasRenderedInsight(refs.coxInsightBoard);
    const hasResults = hasRenderedTable(refs.coxResultsShell);
    const hasDiagnostics = hasRenderedTable(refs.coxDiagnosticsShell);
    const hasDiagnosticsCard = hasDiagnosticsPlot || hasDiagnostics;
    const hasAny = hasPlot || hasDiagnosticsCard || hasMartingalePlot || hasInsight || hasResults;

    reveal(refs.coxPlot, hasPlot);
    reveal(refs.coxDiagnosticsPlot, hasDiagnosticsPlot);
    reveal(refs.coxMartingalePlot, hasMartingalePlot);
    reveal(refs.coxMetaBanner, hasAny);
    reveal(refs.coxInsightBoard, hasInsight);
    reveal(refs.coxResultsShell?.closest(".table-card"), hasResults);
    reveal(refs.coxDiagnosticsShell?.closest(".table-card"), hasDiagnosticsCard);
    reveal(refs.coxMartingalePlot?.closest(".table-card"), hasMartingalePlot);
  }

  if (goal === "tables") {
    reveal(refs.cohortTableShell?.closest(".table-card"), hasRenderedTable(refs.cohortTableShell));
  }

  if (goal === "ml") {
    const resultMode = runtime.resultPreference?.ml || "single";
    const hasSingleImportance = resultMode === "single" && (hasRenderedPlot(refs.mlImportancePlot) || hasPlotMessage(refs.mlImportancePlot));
    const hasSingleShap = resultMode === "single" && (hasRenderedPlot(refs.mlShapPlot) || hasPlotMessage(refs.mlShapPlot));
    const hasSingleGrid = hasSingleImportance || hasSingleShap;
    const hasComparePlot = resultMode === "compare" && hasRenderedPlot(refs.mlComparisonPlot);
    const hasCompareTable = resultMode === "compare" && hasRenderedTable(refs.mlComparisonShell);
    const hasManuscript = resultMode === "compare" && hasRenderedTable(refs.mlManuscriptShell);
    const hasInsight = hasRenderedInsight(refs.mlInsightBoard);
    const hasAny = hasSingleGrid || hasComparePlot || hasCompareTable || hasManuscript || hasInsight;

    reveal(refs.mlImportancePlot, hasSingleImportance);
    reveal(refs.mlShapPlot, hasSingleShap);
    reveal(refs.mlImportancePlot?.closest(".ml-plots-grid"), hasSingleGrid);
    reveal(refs.mlComparisonPlot, hasComparePlot);
    reveal(refs.mlComparisonShell?.closest(".table-card"), hasCompareTable);
    reveal(refs.mlManuscriptShell?.closest(".table-card"), hasManuscript);
    reveal(refs.mlInsightBoard, hasInsight);
    reveal(refs.mlMetaBanner, hasAny);
  }

  if (goal === "dl") {
    const resultMode = runtime.resultPreference?.dl || "single";
    const hasSingleImportance = resultMode === "single" && (hasRenderedPlot(refs.dlImportancePlot) || hasPlotMessage(refs.dlImportancePlot));
    const hasSingleLoss = resultMode === "single" && (hasRenderedPlot(refs.dlLossPlot) || hasPlotMessage(refs.dlLossPlot));
    const hasSingleGrid = hasSingleImportance || hasSingleLoss;
    const hasComparePlot = resultMode === "compare" && hasRenderedPlot(refs.dlComparisonPlot);
    const hasCompareTable = resultMode === "compare" && hasRenderedTable(refs.dlComparisonShell);
    const hasManuscript = resultMode === "compare" && hasRenderedTable(refs.dlManuscriptShell);
    const hasInsight = hasRenderedInsight(refs.dlInsightBoard);
    const hasAny = hasSingleGrid || hasComparePlot || hasCompareTable || hasManuscript || hasInsight;

    reveal(refs.dlImportancePlot, hasSingleImportance);
    reveal(refs.dlLossPlot, hasSingleLoss);
    reveal(refs.dlImportancePlot?.closest(".ml-plots-grid"), hasSingleGrid);
    reveal(refs.dlComparisonPlot, hasComparePlot);
    reveal(refs.dlComparisonShell?.closest(".table-card"), hasCompareTable);
    reveal(refs.dlManuscriptShell?.closest(".table-card"), hasManuscript);
    reveal(refs.dlInsightBoard, hasInsight);
    reveal(refs.dlMetaBanner, hasAny);
  }

  scheduleVisiblePlotResize(40);
}

function resultAnchorFor(tabName, { mode = "single" } = {}) {
  const candidates = {
    km: [refs.kmPlot, refs.kmSummaryShell],
    cox: [refs.coxPlot, refs.coxDiagnosticsPlot, refs.coxMartingalePlot, refs.coxResultsShell],
    predictive: [refs.benchmarkSummaryGrid, refs.benchmarkComparisonPlot, refs.benchmarkComparisonShell, refs.benchmarkWorkbench],
    tables: [refs.cohortTableShell],
    ml: mode === "compare"
      ? [refs.mlComparisonPlot, refs.mlComparisonShell, refs.mlMetaBanner]
      : [refs.mlImportancePlot, refs.mlMetaBanner, refs.mlInsightBoard],
    dl: mode === "compare"
      ? [refs.dlComparisonPlot, refs.dlComparisonShell, refs.dlMetaBanner]
      : [refs.dlImportancePlot, refs.dlLossPlot, refs.dlMetaBanner],
  }[tabName] || [];
  return candidates.find(isVisibleResultNode) || null;
}

function scrollToAnalysisResult(tabName, { mode = "single" } = {}) {
  const target = resultAnchorFor(tabName, { mode });
  if (!target) return;
  requestAnimationFrame(() => {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function shouldRevealCompletedResult(goal) {
  if (goal === "predictive") {
    if (activeTabName() !== "benchmark") return false;
    if (runtime.uiMode === "guided") return runtime.guidedGoal === "predictive";
    return true;
  }
  if (runtime.uiMode !== "guided" && activeTabName() === "benchmark" && ["ml", "dl"].includes(goal)) return true;
  if (runtime.uiMode === "guided" && runtime.guidedGoal === "predictive" && ["ml", "dl"].includes(goal)) return true;
  if (activeTabName() !== goal) return false;
  if (runtime.uiMode === "guided") return runtime.guidedGoal === goal;
  return true;
}

function revealCompletedResultIfCurrent(goal, { mode = "single", successMessage = "", backgroundMessage = "" } = {}) {
  const shouldReveal = shouldRevealCompletedResult(goal);
  if (shouldReveal) {
    activateTab(goal);
    updateGuidedResultVisibility();
    scrollToAnalysisResult(goal, { mode });
  }
  showToast(
    shouldReveal
      ? successMessage
      : (backgroundMessage || `${goalLabel(goal)} finished in the background. Switch back when you are ready to review the updated result.`),
    "success",
    shouldReveal ? 3000 : 3600,
  );
  return shouldReveal;
}

function guidedPredictiveCompareReady() {
  return benchmarkCompareRows("ml", { currentOnly: true }).length > 0
    && benchmarkCompareRows("dl", { currentOnly: true }).length > 0;
}

function guidedPredictiveHasLeaderboardReference() {
  if (typeof benchmarkBoardState !== "function") return false;
  const board = benchmarkBoardState();
  return Boolean(
    !board?.predictiveBusy
    && !board?.guidedPredictiveIncomplete
    && Array.isArray(board?.visibleFamilies)
    && board.visibleFamilies.length === 2
    && !board?.hasMixedEvaluation
    && !board?.visibleHasMixedRunGroups
    && !board?.visibleHasSplitMismatch
    && (board?.visibleRows?.length || 0) > 0,
  );
}

function guidedPredictiveSelectedModelReady({ family = predictiveFamilyGoal(), modelKey = currentPredictiveModelKey(), previousPayload = null } = {}) {
  const payload = goalPayload(family);
  if (!payload || payloadRepresentsCompareRun(payload) || (previousPayload && payload === previousPayload)) return false;
  const requestConfig = payload.request_config || payload.analysis?.request_config || null;
  if (!requestConfig) return false;
  const normalizedModelKey = String(modelKey || "").trim().toLowerCase();
  const normalizedRequestModel = String(requestConfig.model_type || "").trim().toLowerCase();
  if (normalizedRequestModel !== normalizedModelKey) return false;
  return matchesRequestConfig(family, requestConfig, { expectsCompareOverride: false });
}

async function runGuidedGoal(tabName, button, action, { resultMode = "single", successCheck = null } = {}) {
  activateTab(tabName, { historyMode: "replace" });
  const runStatus = await withLoading(button, action, tabName);
  if (!runStatus?.ok) return;
  const resolveHasResult = () => (typeof successCheck === "function" ? Boolean(successCheck()) : Boolean(currentGoalResult(tabName)));
  let hasResult = resolveHasResult();
  if (!hasResult) {
    await new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
    hasResult = resolveHasResult();
  }
  if (hasResult && shouldRevealCompletedResult(tabName)) {
    setGuidedStep(5, { scroll: false, historyMode: "push" });
    scrollToAnalysisResult(tabName, { mode: resultMode });
    if (runtime.uiMode === "guided" && runtime.guidedGoal === "predictive" && ["ml", "dl"].includes(tabName)) {
      window.requestAnimationFrame(() => {
        if (!resolveHasResult() || currentGuidedStep() === 5) return;
        setGuidedStep(5, { syncHistory: false, scroll: false, historyMode: "replace" });
        if (state.dataset) syncHistoryState("replace");
      });
    }
  }
}

function handleGuidedPanelAction(target) {
  const action = target.dataset.guidedAction;
  if (!action) return;
  if (action === "go-home") {
    goHome({ historyMode: "push" });
    return;
  }
  if (action === "next-step") {
    setGuidedStep(currentGuidedStep() + 1, { historyMode: "push" });
    return;
  }
  if (action === "previous-step") {
    setGuidedStep(currentGuidedStep() - 1, { historyMode: "push" });
    return;
  }
  if (action === "close-predictive-workbench") {
    closePredictiveWorkbench();
    return;
  }
  if (action === "focus-study-design") {
    focusStudyDesignSection();
    return;
  }
  if (action === "choose-another-analysis") {
    runtime.guidedGoal = null;
    runtime.guidedStep = normalizedGuidedStep(3);
    if (document.body) document.body.dataset.guidedGoal = "";
    activateTab("km", { setGuidedGoal: false, historyMode: "push" });
    return;
  }
  if (action === "choose-goal") {
    setGuidedGoal(target.dataset.goal || null, { historyMode: "push" });
    return;
  }
  if (action === "open-km") { focusTabWorkspace("km", { historyMode: "push" }); return; }
  if (action === "open-cox") { focusTabWorkspace("cox", { historyMode: "push" }); return; }
  if (action === "open-ml") { focusTabWorkspace("ml", { historyMode: "push" }); return; }
  if (action === "open-dl") { focusTabWorkspace("dl", { historyMode: "push" }); return; }
  if (action === "open-tables") { focusTabWorkspace("tables", { historyMode: "push" }); return; }
  if (action === "review-shared-features") {
    const reviewTab = runtime.guidedGoal === "dl"
      ? "dl"
      : (runtime.guidedGoal === "ml" ? "ml" : predictiveFamilyGoal());
    focusModelFeatureEditor(reviewTab);
    return;
  }
  if (action === "run-km") { void runGuidedGoal("km", target, runGuidedKaplanMeier); return; }
  if (action === "run-cox") { void runGuidedGoal("cox", target, runCox); return; }
  if (action === "run-ml") { void runGuidedGoal("ml", target, runMlModel, { resultMode: "single" }); return; }
  if (action === "run-ml-compare") { void runGuidedGoal("ml", target, runCompareModels, { resultMode: "compare" }); return; }
  if (action === "run-dl") { void runGuidedGoal("dl", target, runDlModel, { resultMode: "single" }); return; }
  if (action === "run-dl-compare") { void runGuidedGoal("dl", target, runDlCompareModels, { resultMode: "compare" }); return; }
  if (action === "run-predictive-selected") {
    const family = predictiveFamilyGoal();
    const modelKey = currentPredictiveModelKey();
    const previousPayload = goalPayload(family);
    runtime.workbenchRevealed = true;
    runtime.predictiveWorkbenchIntent = "train";
    void runGuidedGoal(family, target, runPredictiveSelectedModel, {
      resultMode: "single",
      successCheck: () => guidedPredictiveSelectedModelReady({ family, modelKey, previousPayload }),
    });
    return;
  }
  if (action === "run-predictive-compare-all") {
    runtime.workbenchRevealed = false;
    runtime.predictiveWorkbenchIntent = null;
    void runGuidedGoal("predictive", target, runUnifiedPredictiveComparison, {
      resultMode: "compare",
      successCheck: guidedPredictiveCompareReady,
    });
    return;
  }
  if (action === "return-trained-predictive-model") {
    if (!selectedPredictiveSingleResult(predictiveFamilyGoal())) return;
    runtime.workbenchRevealed = true;
    runtime.predictiveWorkbenchIntent = "train";
    activateTab("benchmark", { setGuidedGoal: false, historyMode: "replace", syncHistory: false });
    setGuidedStep(5, { syncHistory: false, scroll: false, historyMode: "replace" });
    renderBenchmarkBoard();
    requestAnimationFrame(() => {
      refs.benchmarkWorkbench?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    if (state.dataset) syncHistoryState("replace");
    return;
  }
  if (action === "run-tables") {
    void runGuidedGoal("tables", target, runCohortTable, {
      successCheck: () => Boolean(state.cohort?.analysis),
    });
  }
}

function updateStepIndicator(step = currentGuidedStep()) {
  if (!refs.stepIndicator) return;
  const activeStep = step;
  const reachableStep = maxReachableGuidedStep();
  const steps = refs.stepIndicator.querySelectorAll(".step");
  const connectors = refs.stepIndicator.querySelectorAll(".step-connector");
  steps.forEach((el) => {
    const s = Number(el.dataset.step);
    const circle = el.querySelector(".step-circle");
    const label = el.querySelector(".step-label")?.textContent?.trim() || `Step ${s}`;
    el.classList.remove("active", "completed");
    if (circle) circle.textContent = String(s);
    if (s < activeStep) {
      el.classList.add("completed");
      if (circle) circle.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (s === activeStep) {
      el.classList.add("active");
    }
    if ("disabled" in el) el.disabled = s > reachableStep;
    el.setAttribute("aria-disabled", String(s > reachableStep));
    el.setAttribute("aria-label", `Step ${s}: ${label}`);
    if (s === activeStep) {
      el.setAttribute("aria-current", "step");
    } else {
      el.removeAttribute("aria-current");
    }
  });
  connectors.forEach((c, i) => c.classList.toggle("completed", i < activeStep - 1));
}

function showSmartBanner(text) {
  if (!refs.smartBanner || !refs.smartBannerText) return;
  refs.smartBannerText.textContent = text;
  refs.smartBanner.classList.remove("hidden");
}

function initTabKeyboard() {
  const strip = document.querySelector(".tab-strip");
  if (!strip) return;
  strip.addEventListener("keydown", (e) => {
    const tabs = refs.tabButtons.filter((button) => button.offsetParent !== null);
    const idx = tabs.indexOf(e.target);
    if (idx < 0) return;
    let next = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % tabs.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    if (next >= 0) { e.preventDefault(); activateTab(tabs[next].dataset.tab, { historyMode: "push", focusTabButton: true }); }
  });
}

function showTooltipAt(dot) {
  const popup = refs.tooltipPopup;
  if (!popup || !dot) return;
  popup.textContent = dot.dataset.tooltip;
  popup.setAttribute("role", "tooltip");
  popup.classList.remove("hidden");
  const rect = dot.getBoundingClientRect();
  popup.style.left = `${Math.min(rect.left, window.innerWidth - 280)}px`;
  popup.style.top = `${rect.bottom + 8}px`;
}

function hideTooltip() {
  if (refs.tooltipPopup) refs.tooltipPopup.classList.add("hidden");
}

function initTooltips() {
  const popup = refs.tooltipPopup;
  if (!popup) return;
  let activeTarget = null;
  const closestFromEvent = (event, selector) => {
    const target = event?.target;
    return target instanceof Element ? target.closest(selector) : null;
  };
  // Mouse
  document.addEventListener("mouseenter", (e) => {
    const dot = closestFromEvent(e, "[data-tooltip]");
    if (!dot) return;
    activeTarget = dot;
    showTooltipAt(dot);
  }, true);
  document.addEventListener("mouseleave", (e) => {
    const dot = closestFromEvent(e, "[data-tooltip]");
    if (dot && dot === activeTarget) { hideTooltip(); activeTarget = null; }
  }, true);
  // Keyboard: focus/blur on help-dot buttons
  document.addEventListener("focusin", (e) => {
    const dot = closestFromEvent(e, "[data-tooltip]");
    if (dot) { activeTarget = dot; showTooltipAt(dot); }
  }, true);
  document.addEventListener("focusout", (e) => {
    const dot = closestFromEvent(e, "[data-tooltip]");
    if (dot) { hideTooltip(); activeTarget = null; }
  }, true);
}

function initDragDrop() {
  const zone = refs.uploadZone;
  if (!zone) return;
  let dragCounter = 0;
  zone.addEventListener("dragenter", (e) => { e.preventDefault(); dragCounter++; zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", (e) => { e.preventDefault(); dragCounter--; if (dragCounter <= 0) { dragCounter = 0; zone.classList.remove("drag-over"); } });
  zone.addEventListener("dragover", (e) => e.preventDefault());
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); dragCounter = 0; zone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) { refs.datasetFile.files = e.dataTransfer.files; withLoading(refs.uploadButton, uploadDataset); }
  });
}

function initKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (!state.dataset) return;
      const btn = getActiveRunButton();
      const action = getActiveRunAction();
      if (btn && action && !btn.disabled) withLoading(btn, action);
    }
  });
}

function goHome({ syncHistory = true, historyMode = "replace" } = {}) {
  // Leaving the workspace makes pending dataset loads/derives obsolete.
  invalidateRequestTokens(["dataset", "derive"]);
  const result = shellHelpers.goHome({
    state,
    runtime,
    refs,
    syncHistory,
    historyMode,
    resetCoxPreview,
    renderSharedFeatureSummary,
    renderGuidedChrome,
    setRuntimeBanner,
    syncHistoryState,
  });
  renderDatasetIntegrityWarning();
  return result;
}

function initListeners() {
  const closestFromEvent = (event, selector) => {
    const target = event?.target;
    return target instanceof Element ? target.closest(selector) : null;
  };
  const brandHome = refs.brandHome;
  if (brandHome) {
    brandHome.addEventListener("click", (e) => { e.preventDefault(); goHome({ historyMode: "push" }); });
  }
  refs.guidedModeButton?.addEventListener("click", () => setUiMode("guided", { historyMode: "push" }));
  refs.expertModeButton?.addEventListener("click", () => setUiMode("expert", { historyMode: "push" }));
  refs.predictiveModelSelector?.addEventListener("change", () => {
    runtime.workbenchRevealed = true;
    setPredictiveModel(refs.predictiveModelSelector.value, { historyMode: "push" });
    renderBenchmarkBoard();
    syncAnalysisRunButtonAvailability();
  });
  refs.runPredictiveCompareAllButton?.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "predictive";
      void runGuidedGoal("predictive", refs.runPredictiveCompareAllButton, runUnifiedPredictiveComparison, {
        resultMode: "compare",
        successCheck: guidedPredictiveCompareReady,
      });
      return;
    }
    withLoading(refs.runPredictiveCompareAllButton, runUnifiedPredictiveComparison, "predictive");
  });
  refs.runPredictiveSelectedButton?.addEventListener("click", () => {
    runtime.workbenchRevealed = true;
    runtime.predictiveWorkbenchIntent = "train";
    const selectedFamily = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey()).family;
    const modelKey = refs.predictiveModelSelector?.value || currentPredictiveModelKey();
    const previousPayload = goalPayload(selectedFamily);
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "predictive";
      void runGuidedGoal(selectedFamily, refs.runPredictiveSelectedButton, runPredictiveSelectedModel, {
        resultMode: "single",
        successCheck: () => guidedPredictiveSelectedModelReady({ family: selectedFamily, modelKey, previousPayload }),
      });
      return;
    }
    withLoading(refs.runPredictiveSelectedButton, runPredictiveSelectedModel, selectedFamily);
  });
  refs.runPredictiveWorkbenchButton?.addEventListener("click", () => {
    runtime.workbenchRevealed = true;
    runtime.predictiveWorkbenchIntent = "train";
    const selectedFamily = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey()).family;
    const modelKey = refs.predictiveModelSelector?.value || currentPredictiveModelKey();
    const previousPayload = goalPayload(selectedFamily);
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "predictive";
      void runGuidedGoal(selectedFamily, refs.runPredictiveWorkbenchButton, runPredictiveSelectedModel, {
        resultMode: "single",
        successCheck: () => guidedPredictiveSelectedModelReady({ family: selectedFamily, modelKey, previousPayload }),
      });
      return;
    }
    withLoading(refs.runPredictiveWorkbenchButton, runPredictiveSelectedModel, selectedFamily);
  });
  refs.closePredictiveWorkbenchButton?.addEventListener("click", () => {
    closePredictiveWorkbench();
  });
  refs.benchmarkSummaryGrid?.addEventListener("click", (event) => {
    const modelButton = closestFromEvent(event, "[data-benchmark-model]");
    if (modelButton) {
      reviewBenchmarkModel(modelButton.dataset.benchmarkModel || currentPredictiveModelKey(), modelButton.dataset.benchmarkMode || null);
      return;
    }
    const button = closestFromEvent(event, "[data-benchmark-tab]");
    if (!button) return;
    reviewBenchmarkSourceTab(button.dataset.benchmarkTab || "ml", button.dataset.benchmarkMode || null);
  });
  refs.benchmarkComparisonShell?.addEventListener("click", (event) => {
    const paramsButton = closestFromEvent(event, "[data-benchmark-params-goal]");
    if (paramsButton) {
      showBenchmarkParams(
        paramsButton.dataset.benchmarkParamsGoal || "ml",
        paramsButton.dataset.benchmarkParamsModel || "Model",
        paramsButton.dataset.benchmarkParamsSource || "current",
      );
      return;
    }
    const modelButton = closestFromEvent(event, "[data-benchmark-model]");
    if (modelButton) {
      reviewBenchmarkModel(modelButton.dataset.benchmarkModel || currentPredictiveModelKey(), modelButton.dataset.benchmarkMode || null);
      return;
    }
    const button = closestFromEvent(event, "[data-benchmark-tab]");
    if (!button) return;
    reviewBenchmarkSourceTab(button.dataset.benchmarkTab || "ml", button.dataset.benchmarkMode || null);
  });
  refs.guidedPanel?.addEventListener("click", (event) => {
    const button = closestFromEvent(event, "[data-guided-action]");
    if (!button) return;
    handleGuidedPanelAction(button);
  });
  refs.guidedRailActions?.addEventListener("click", (event) => {
    const button = closestFromEvent(event, "[data-guided-action]");
    if (!button) return;
    handleGuidedPanelAction(button);
  });
  refs.guidedPanel?.addEventListener("change", (event) => {
    const select = closestFromEvent(event, "[data-guided-predictive-model-selector]");
    if (!select) return;
    setPredictiveModel(select.value, { historyMode: "push" });
    renderBenchmarkBoard();
    syncAnalysisRunButtonAvailability();
  });
  refs.stepIndicator?.addEventListener("click", (event) => {
    const button = closestFromEvent(event, ".step");
    if (!button) return;
    const requestedStep = Number(button.dataset.step || 0);
    if (!requestedStep || !canNavigateToGuidedStep(requestedStep)) return;
    if (requestedStep === 1) {
      goHome({ historyMode: "push" });
      return;
    }
    if (requestedStep === currentGuidedStep()) return;
    setGuidedStep(requestedStep, { historyMode: "push" });
    if (requestedStep >= 4 && runtime.guidedGoal) {
      activateTab(runtime.guidedGoal, { setGuidedGoal: false, historyMode: "replace" });
    }
  });
  window.addEventListener("popstate", (event) => {
    void restoreHistoryState(event.state);
  });
  window.addEventListener("resize", () => {
    scheduleVisiblePlotResize(80);
  });
  refs.datasetFile.addEventListener("click", () => {
    refs.datasetFile.value = "";
  });
  refs.datasetFile.addEventListener("change", () => {
    if (refs.datasetFile.files?.length) withLoading(refs.uploadButton, uploadDataset);
  });
  refs.uploadButton.addEventListener("click", () => refs.datasetFile.click());
  refs.shutdownButton?.addEventListener("click", () => {
    void withLoading(refs.shutdownButton, shutdownServer);
  });
  refs.loadTcgaUploadReadyButton.addEventListener("click", () => withLoading(refs.loadTcgaUploadReadyButton, loadTcgaUploadReadyDataset));
  refs.loadTcgaButton.addEventListener("click", () => withLoading(refs.loadTcgaButton, loadTcgaDataset));
  refs.loadGbsg2Button.addEventListener("click", () => withLoading(refs.loadGbsg2Button, loadGbsg2Dataset));
  refs.loadExampleButton.addEventListener("click", () => withLoading(refs.loadExampleButton, loadExampleDataset));
  refs.applyBasicPresetButton?.addEventListener("click", () => applyDatasetPreset("basic"));
  refs.applyModelPresetButton?.addEventListener("click", () => applyDatasetPreset("models"));
  refs.timeColumn.addEventListener("change", () => {
    clearAnalysisOutputs();
    applyAutomaticTimeUnitLabel();
    updateTimeColumnGuidance();
    refreshVariableSelections();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    updateEventColumnGuidance();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
  });
  refs.eventColumn.addEventListener("change", () => {
    clearAnalysisOutputs();
    updateTimeColumnGuidance();
    updateEventPositiveOptions();
    refreshVariableSelections();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
  });
  refs.eventPositiveValue.addEventListener("change", () => {
    clearAnalysisOutputs();
    updateEventPositiveOptions();
    renderSharedFeatureSummary();
    updateEventColumnGuidance();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
  });
  refs.showAllEventColumns?.addEventListener("change", () => {
    renderEventColumnOptions({ silent: false });
    refreshVariableSelections();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    queueHistorySync();
  });
  refs.groupColumn.addEventListener("change", () => {
    rerenderDerivedGroupSummaryIfVisible();
    syncDeriveControlsState();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    queueHistorySync();
  });
  refs.timeUnitLabel.addEventListener("input", () => { runtime.timeUnitAutoLabel = false; });
  refs.timeUnitLabel.addEventListener("input", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.maxTime.addEventListener("input", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.confidenceLevel.addEventListener("change", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.covariateChecklist?.addEventListener("change", (event) => {
    const input = closestFromEvent(event, 'input[type="checkbox"]');
    syncCoxCovariateSelection({
      preferredValue: input?.checked ? input.value : null,
      preferredScope: "covariate",
      notify: Boolean(input?.checked),
      autoCategoricalValues: input?.checked ? [input.value] : [],
    });
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview();
  });
  refs.covariateSearchInput?.addEventListener("input", () => {
    applyChecklistSearch(refs.covariateChecklist);
  });
  refs.categoricalChecklist?.addEventListener("change", () => {
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview();
  });
  refs.categoricalSearchInput?.addEventListener("input", () => {
    applyChecklistSearch(refs.categoricalChecklist);
  });
  refs.strataChecklist?.addEventListener("change", (event) => {
    const input = closestFromEvent(event, 'input[type="checkbox"]');
    syncCoxCovariateSelection({
      preferredValue: input?.checked ? input.value : null,
      preferredScope: "strata",
      notify: Boolean(input?.checked),
    });
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview();
  });
  refs.strataSearchInput?.addEventListener("input", () => {
    applyChecklistSearch(refs.strataChecklist);
  });
  refs.selectAllCoxCovariatesButton?.addEventListener("click", () => {
    const covariates = allCheckboxValues(refs.covariateChecklist, { visibleOnly: true });
    setCheckedValues(refs.covariateChecklist, covariates);
    syncCoxCovariateSelection({ preferredScope: "covariate", autoCategoricalValues: covariates });
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Selected all Cox covariates.", "success", 2200);
  });
  refs.clearCoxCovariatesButton?.addEventListener("click", () => {
    setCheckedValues(refs.covariateChecklist, []);
    setCheckedValues(refs.categoricalChecklist, []);
    syncCoxCovariateSelection();
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Cleared all Cox covariates and categorical flags.", "success", 2200);
  });
  refs.selectAllCoxCategoricalsButton?.addEventListener("click", () => {
    const covariates = selectedCheckboxValues(refs.covariateChecklist);
    setCheckedValues(
      refs.categoricalChecklist,
      covariates.length ? covariates : allCheckboxValues(refs.categoricalChecklist, { visibleOnly: true }),
    );
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Marked the current Cox covariates as categorical.", "success", 2200);
  });
  refs.clearCoxCategoricalsButton?.addEventListener("click", () => {
    setCheckedValues(refs.categoricalChecklist, []);
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Cleared Cox categorical flags.", "success", 2200);
  });
  refs.selectAllCoxStrataButton?.addEventListener("click", () => {
    const strata = allCheckboxValues(refs.strataChecklist, { visibleOnly: true });
    setCheckedValues(refs.strataChecklist, strata);
    syncCoxCovariateSelection({ preferredScope: "strata" });
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Selected visible Cox strata. Matching covariates were cleared automatically.", "success", 2400);
  });
  refs.clearCoxStrataButton?.addEventListener("click", () => {
    setCheckedValues(refs.strataChecklist, []);
    syncCoxCovariateSelection();
    renderSharedFeatureSummary();
    syncGuidedCoxPanelMounts();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Cleared Cox strata.", "success", 2200);
  });
  refs.modelFeatureChecklist?.addEventListener("change", () => { syncModelFeatureMirrors(refs.modelFeatureChecklist); renderSharedFeatureSummary(); queueHistorySync(); });
  refs.modelCategoricalChecklist?.addEventListener("change", () => { syncModelCategoricalMirrors(refs.modelCategoricalChecklist); renderSharedFeatureSummary(); queueHistorySync(); });
  refs.dlModelFeatureChecklist?.addEventListener("change", () => { syncModelFeatureMirrors(refs.dlModelFeatureChecklist); renderSharedFeatureSummary(); queueHistorySync(); });
  refs.dlModelCategoricalChecklist?.addEventListener("change", () => { syncModelCategoricalMirrors(refs.dlModelCategoricalChecklist); renderSharedFeatureSummary(); queueHistorySync(); });
  refs.cohortVariableChecklist?.addEventListener("change", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.cohortVariableSearchInput?.addEventListener("input", () => {
    applyChecklistSearch(refs.cohortVariableChecklist);
  });
  refs.selectAllCohortVariablesButton?.addEventListener("click", () => {
    const variables = allCheckboxValues(refs.cohortVariableChecklist, { visibleOnly: true });
    setCheckedValues(refs.cohortVariableChecklist, variables);
    renderSharedFeatureSummary();
    queueHistorySync();
    showToast("Selected all visible cohort table variables.", "success", 2200);
  });
  refs.clearCohortVariablesButton?.addEventListener("click", () => {
    setCheckedValues(refs.cohortVariableChecklist, []);
    renderSharedFeatureSummary();
    queueHistorySync();
    showToast("Cleared the cohort table variable list.", "success", 2200);
  });
  refs.reviewMlFeaturesButton?.addEventListener("click", () => focusModelFeatureEditor("ml"));
  refs.reviewDlFeaturesButton?.addEventListener("click", () => focusModelFeatureEditor("dl"));
  refs.selectAllModelFeaturesButton?.addEventListener("click", () => {
    setSharedModelFeatureSelection(modelFeatureCandidateColumns());
    showToast("Selected all eligible ML/DL model features.", "success", 2400);
  });
  refs.clearModelFeaturesButton?.addEventListener("click", () => {
    setSharedModelFeatureSelection([], { clearCategoricals: true });
    showToast("Cleared the shared ML/DL model feature list.", "success", 2400);
  });
  refs.selectAllDlModelFeaturesButton?.addEventListener("click", () => {
    setSharedModelFeatureSelection(modelFeatureCandidateColumns());
    showToast("Selected all eligible ML/DL model features.", "success", 2400);
  });
  refs.clearDlModelFeaturesButton?.addEventListener("click", () => {
    setSharedModelFeatureSelection([], { clearCategoricals: true });
    showToast("Cleared the shared ML/DL model feature list.", "success", 2400);
  });
  const markDeriveDraftTouched = () => {
    runtime.deriveDraftTouched = true;
    rerenderDerivedGroupSummaryIfVisible();
    syncDeriveControlsState();
    queueHistorySync();
  };
  refs.deriveSource?.addEventListener("change", markDeriveDraftTouched);
  refs.deriveMethod.addEventListener("change", () => { updateMethodVisibility(); markDeriveDraftTouched(); });
  refs.deriveCutoff?.addEventListener("input", markDeriveDraftTouched);
  refs.deriveColumnName?.addEventListener("input", markDeriveDraftTouched);
  refs.deriveMinGroupFraction?.addEventListener("input", markDeriveDraftTouched);
  refs.derivePermutationIterations?.addEventListener("input", markDeriveDraftTouched);
  refs.deriveRandomSeed?.addEventListener("input", markDeriveDraftTouched);
  refs.logrankWeight.addEventListener("change", () => { updateWeightVisibility(); scheduleResultCurrencySync(); queueHistorySync(); });
  refs.mlModelType.addEventListener("change", () => {
    updateMlModelControlVisibility();
    renderPredictiveWorkbench();
    queueHistorySync();
  });
  refs.mlSkipShap?.addEventListener("change", () => {
    updateMlModelControlVisibility();
    queueHistorySync();
  });
  refs.mlEvaluationStrategy.addEventListener("change", () => {
    mirrorPredictiveEvaluationControl(refs.mlEvaluationStrategy);
    updateMlEvaluationControls();
    updateDlEvaluationControls();
    scheduleResultCurrencySync();
    queueHistorySync();
  });
  refs.dlModelType.addEventListener("change", () => {
    updateDlModelControlVisibility();
    renderPredictiveWorkbench();
    queueHistorySync();
  });
  refs.coxMartingaleVariableSelect?.addEventListener("change", () => {
    runtime.coxMartingaleTerm = refs.coxMartingaleVariableSelect.value || "";
    void renderCoxMartingalePlot(runtime.coxMartingaleTerm);
  });
  refs.dlEvaluationStrategy.addEventListener("change", () => {
    mirrorPredictiveEvaluationControl(refs.dlEvaluationStrategy);
    updateMlEvaluationControls();
    updateDlEvaluationControls();
    scheduleResultCurrencySync();
    queueHistorySync();
  });
  [
    refs.mlCvFolds,
    refs.dlCvFolds,
    refs.mlCvRepeats,
    refs.dlCvRepeats,
    refs.mlRandomSeed,
    refs.dlRandomSeed,
    refs.mlLockedTestToggle,
    refs.dlLockedTestToggle,
    refs.mlLockedTestFraction,
    refs.dlLockedTestFraction,
  ].filter(Boolean).forEach((control) => {
    const mirror = () => {
      mirrorPredictiveEvaluationControl(control);
      if (control.type === "checkbox") {
        updateMlEvaluationControls();
        updateDlEvaluationControls();
      }
    };
    control.addEventListener("change", mirror);
    if (control.type !== "checkbox") control.addEventListener("input", mirror);
  });
  refs.deriveToggle.addEventListener("click", () => {
    if (refs.groupingDetails) refs.groupingDetails.open = true;
    refs.derivePanel.classList.toggle("hidden");
    syncDeriveToggleButton();
    queueHistorySync();
  });
  refs.deriveButton.addEventListener("click", () => withLoading(refs.deriveButton, deriveGroup));
  refs.runKmButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "km";
      void runGuidedGoal("km", refs.runKmButton, runGuidedKaplanMeier);
      return;
    }
    withLoading(refs.runKmButton, runKaplanMeier);
  });
  refs.runSignatureSearchButton.addEventListener("click", () => withLoading(refs.runSignatureSearchButton, runSignatureSearch, "km"));
  refs.runCoxButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "cox";
      void runGuidedGoal("cox", refs.runCoxButton, runCox);
      return;
    }
    withLoading(refs.runCoxButton, runCox);
  });
  refs.runCohortTableButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "tables";
      void runGuidedGoal("tables", refs.runCohortTableButton, runCohortTable, {
        successCheck: () => Boolean(state.cohort?.analysis),
      });
      return;
    }
    withLoading(refs.runCohortTableButton, runCohortTable);
  });
  refs.runMlButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      if (runtime.guidedGoal === "predictive") {
        void runGuidedGoal("ml", refs.runMlButton, runMlModel, {
          resultMode: "single",
          successCheck: () => Boolean(currentGoalResult("ml")),
        });
        return;
      }
      runtime.guidedGoal = "ml";
      void runGuidedGoal("ml", refs.runMlButton, runMlModel);
      return;
    }
    withLoading(refs.runMlButton, runMlModel);
  });
  refs.runCompareButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      if (runtime.guidedGoal === "predictive") {
        void runGuidedGoal("ml", refs.runCompareButton, runCompareModels, {
          resultMode: "compare",
          successCheck: () => benchmarkCompareRows("ml", { currentOnly: true }).length > 0,
        });
        return;
      }
      runtime.guidedGoal = "ml";
      void runGuidedGoal("ml", refs.runCompareButton, runCompareModels);
      return;
    }
    withLoading(refs.runCompareButton, runCompareModels);
  });
  refs.runCompareInlineButton?.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      if (runtime.guidedGoal === "predictive") {
        void runGuidedGoal("ml", refs.runCompareInlineButton, runCompareModels, {
          resultMode: "compare",
          successCheck: () => benchmarkCompareRows("ml", { currentOnly: true }).length > 0,
        });
        return;
      }
      runtime.guidedGoal = "ml";
      void runGuidedGoal("ml", refs.runCompareInlineButton, runCompareModels);
      return;
    }
    withLoading(refs.runCompareInlineButton, runCompareModels);
  });
  refs.runDlButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      if (runtime.guidedGoal === "predictive") {
        void runGuidedGoal("dl", refs.runDlButton, runDlModel, {
          resultMode: "single",
          successCheck: () => Boolean(currentGoalResult("dl")),
        });
        return;
      }
      runtime.guidedGoal = "dl";
      void runGuidedGoal("dl", refs.runDlButton, runDlModel);
      return;
    }
    withLoading(refs.runDlButton, runDlModel);
  });
  refs.runDlCompareButton.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      if (runtime.guidedGoal === "predictive") {
        void runGuidedGoal("dl", refs.runDlCompareButton, runDlCompareModels, {
          resultMode: "compare",
          successCheck: () => benchmarkCompareRows("dl", { currentOnly: true }).length > 0,
        });
        return;
      }
      runtime.guidedGoal = "dl";
      void runGuidedGoal("dl", refs.runDlCompareButton, runDlCompareModels);
      return;
    }
    withLoading(refs.runDlCompareButton, runDlCompareModels);
  });
  refs.runDlCompareInlineButton?.addEventListener("click", () => {
    if (runtime.uiMode === "guided") {
      if (runtime.guidedGoal === "predictive") {
        void runGuidedGoal("dl", refs.runDlCompareInlineButton, runDlCompareModels, {
          resultMode: "compare",
          successCheck: () => benchmarkCompareRows("dl", { currentOnly: true }).length > 0,
        });
        return;
      }
      runtime.guidedGoal = "dl";
      void runGuidedGoal("dl", refs.runDlCompareInlineButton, runDlCompareModels);
      return;
    }
    withLoading(refs.runDlCompareInlineButton, runDlCompareModels);
  });
  refs.tabButtons.forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab, { historyMode: "replace" })));
  const changeTrackedControls = [
    refs.eventPositiveValue,
    refs.showAllEventColumns,
    refs.timeUnitLabel,
    refs.maxTime,
    refs.confidenceLevel,
    refs.deriveSource,
    refs.deriveCutoff,
    refs.deriveColumnName,
    refs.showConfidenceBands,
    refs.riskTablePoints,
    refs.fhPower,
    refs.signatureMaxDepth,
    refs.signatureMinFraction,
    refs.signatureTopK,
    refs.signatureBootstrapIterations,
    refs.signaturePermutationIterations,
    refs.signatureValidationIterations,
    refs.signatureValidationFraction,
    refs.signatureSignificanceLevel,
    refs.signatureOperator,
    refs.signatureRandomSeed,
    refs.mlModelType,
    refs.mlNEstimators,
    refs.mlLearningRate,
    refs.mlSkipShap,
    refs.mlShapSafeMode,
    refs.mlCvFolds,
    refs.mlCvRepeats,
    refs.mlJournalTemplate,
    refs.dlModelType,
    refs.dlEpochs,
    refs.dlLearningRate,
    refs.dlHiddenLayers,
    refs.dlDropout,
    refs.dlBatchSize,
    refs.dlRandomSeed,
    refs.dlCvFolds,
    refs.dlCvRepeats,
    refs.dlEarlyStoppingPatience,
    refs.dlEarlyStoppingMinDelta,
    refs.dlParallelJobs,
    refs.dlNumTimeBins,
    refs.dlDModel,
    refs.dlHeads,
    refs.dlLayers,
    refs.dlLatentDim,
    refs.dlClusters,
    refs.dlJournalTemplate,
    refs.mlRandomSeed,
    refs.mlLockedTestToggle,
    refs.mlLockedTestFraction,
    refs.dlLockedTestToggle,
    refs.dlLockedTestFraction,
  ];
  // Every control that feeds a request must also refresh result currency (guided status, downloads,
  // leaderboard), not just the history snapshot.
  const onTrackedControlChange = () => {
    scheduleResultCurrencySync();
    queueHistorySync();
  };
  changeTrackedControls.filter(Boolean).forEach((control) => {
    control.addEventListener("change", onTrackedControlChange);
    if (["text", "number"].includes(control.type)) control.addEventListener("input", onTrackedControlChange);
  });
  wireDownloads();
}

initListeners();
initDragDrop();
initKeyboardShortcuts();
initTooltips();
initTabKeyboard();
updateMethodVisibility();
updateWeightVisibility();
updateMlModelControlVisibility();
updateMlEvaluationControls();
updateDlModelControlVisibility();
updateDlEvaluationControls();
syncDeriveToggleButton();
initPlotResizeObserver();
initializeRuntime();

if (refs.smartBannerClose) {
  refs.smartBannerClose.addEventListener("click", () => refs.smartBanner.classList.add("hidden"));
}

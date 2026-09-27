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

function renderBenchmarkBoard() {
  benchmarkBoardApi.renderBenchmarkBoard();
  syncTripodDownloadButtons();
}

// The comparison fields the TRIPOD+AI checklist reads; per-fold and per-repeat detail stays in the browser.
const TRIPOD_ANALYSIS_KEYS = [
  "comparison_table",
  "errors",
  "excluded_models",
  "n_patients",
  "n_events",
  "evaluation_mode",
  "cv_folds",
  "cv_repeats",
  "split_seed",
  "locked_test_fraction",
  "n_development_patients",
  "n_development_events",
  "n_locked_test_patients",
  "n_locked_test_events",
  "n_fit_patients",
  "n_fit_events",
  "n_evaluation_patients",
  "n_evaluation_events",
  "evaluation_split_fingerprint",
];

function tripodComparisons() {
  return ["ml", "dl"].flatMap((family) => {
    const payload = currentCompareGoalPayload(family);
    const table = payload?.analysis?.comparison_table;
    if (!Array.isArray(table) || !table.length) return [];
    const analysis = Object.fromEntries(
      TRIPOD_ANALYSIS_KEYS.filter((key) => key in payload.analysis).map((key) => [key, payload.analysis[key]]),
    );
    analysis.comparison_table = table.map(({ repeat_results, fold_results, ...row }) => row);
    return [{ family, analysis, request_config: payload.request_config || {} }];
  });
}

function syncTripodDownloadButtons() {
  const ready = tripodComparisons().length > 0;
  refs.downloadTripodDocxButton.disabled = !ready;
  refs.downloadTripodMarkdownButton.disabled = !ready;
}

async function downloadTripodChecklist(format) {
  const comparisons = tripodComparisons();
  if (!comparisons.length) {
    showToast("Run Compare All Models first.", "warning", 3600);
    return;
  }
  try {
    const report = await fetchJSON("/api/tripod-ai-checklist", {
      method: "POST",
      body: JSON.stringify({ dataset_id: state.dataset?.dataset_id || null, comparisons }),
    });
    await downloadChecklist(report, format, "tripod_ai_checklist");
  } catch (error) {
    showError(error?.message || "Checklist export failed.");
  }
}

// ── Downloads ──────────────────────────────────────────────────

function wireDownloads() {
  refs.downloadTripodDocxButton.addEventListener("click", () => downloadTripodChecklist("docx"));
  refs.downloadTripodMarkdownButton.addEventListener("click", () => downloadTripodChecklist("markdown"));
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
              : button === refs.runMarkersButton ? "markers"
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
  updateGroupingDetailsVisibility(activeTabName(), { force: true });
  renderWorkspaceChrome();
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
  if (tab === "markers") return refs.runMarkersButton;
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
  if (tab === "markers") return runMarkerEvaluation;
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

function reviewBenchmarkSourceTab(tabName, mode = null) {
  const nextMode = mode || (tabName === "ml" ? benchmarkPanelMode("ml") : benchmarkPanelMode("dl"));
  setPredictiveWorkbenchFamily(tabName, { syncHistory: false });
  activateTab("benchmark", { historyMode: "push" });
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
  reviewBenchmarkSourceTab(meta.family, mode);
}

function closePredictiveWorkbench() {
  if (!runtime.workbenchRevealed) return;
  runtime.workbenchRevealed = false;
  runtime.predictiveWorkbenchIntent = null;
  renderBenchmarkBoard();
  syncPredictiveWorkbenchSingleResultVisibility();
  renderWorkspaceChrome();
  if (state.dataset) syncHistoryState("push");
  requestAnimationFrame(() => {
    refs.benchmarkActionCard?.scrollIntoView({ behavior: "smooth", block: "start" });
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

function setResultNodeVisible(node, visible) {
  if (!node) return;
  node.classList.toggle("result-hidden", !visible);
}

// Result sections stay hidden until they hold a result, so a tab shows its settings, its main plot area
// and nothing else before the first run.
function updateResultVisibility() {
  const reveal = setResultNodeVisible;

  const kmInsight = hasRenderedInsight(refs.kmInsightBoard);
  const kmSummary = hasRenderedTable(refs.kmSummaryShell);
  const kmRisk = hasRenderedTable(refs.kmRiskShell);
  const kmPairwise = hasRenderedTable(refs.kmPairwiseShell);
  const signatureInsight = hasRenderedInsight(refs.signatureInsightBoard);
  const signatureTable = hasRenderedTable(refs.signatureShell);
  reveal(refs.kmMetaBanner, hasRenderedPlot(refs.kmPlot) || kmInsight || kmSummary);
  reveal(refs.kmInsightBoard, kmInsight);
  reveal(refs.kmSummaryShell?.closest(".table-card"), kmSummary);
  reveal(refs.kmRiskShell?.closest(".table-card"), kmRisk);
  reveal(refs.kmPairwiseShell?.closest(".table-card"), kmPairwise);
  reveal(refs.signatureInsightBoard?.closest(".table-card"), signatureInsight);
  reveal(refs.signatureShell?.closest(".table-card"), signatureTable);

  const coxDiagnosticsPlot = hasRenderedPlot(refs.coxDiagnosticsPlot);
  const coxMartingalePlot = hasRenderedPlot(refs.coxMartingalePlot);
  const coxInsight = hasRenderedInsight(refs.coxInsightBoard);
  const coxResults = hasRenderedTable(refs.coxResultsShell);
  const coxDiagnostics = hasRenderedTable(refs.coxDiagnosticsShell);
  reveal(refs.coxMetaBanner, hasRenderedPlot(refs.coxPlot) || coxInsight || coxResults);
  reveal(refs.coxInsightBoard, coxInsight);
  reveal(refs.coxResultsShell?.closest(".table-card"), coxResults);
  reveal(refs.coxDiagnosticsPlot, coxDiagnosticsPlot);
  reveal(refs.coxDiagnosticsShell?.closest(".table-card"), coxDiagnosticsPlot || coxDiagnostics);
  reveal(refs.coxMartingalePlot, coxMartingalePlot);
  reveal(refs.coxMartingalePlot?.closest(".table-card"), coxMartingalePlot);

  // The prediction board appears once a model has run or a comparison is in progress.
  const board = state.dataset && typeof benchmarkBoardState === "function" ? benchmarkBoardState() : null;
  const hasPredictiveOutput = Boolean(state.ml || state.dl || board?.predictiveBusy || board?.visibleRows?.length || board?.staleFamilies?.length);
  reveal(refs.benchmarkSummaryGrid, hasPredictiveOutput);
  reveal(refs.benchmarkComparisonPlot?.closest(".table-card"), hasPredictiveOutput);
  reveal(refs.benchmarkComparisonShell?.closest(".table-card"), hasPredictiveOutput);

  const markerInsight = hasRenderedInsight(refs.markersInsightBoard);
  const markerTable = hasRenderedTable(refs.markersTableShell);
  reveal(refs.markersMetaBanner, markerInsight);
  reveal(refs.markersInsightBoard, markerInsight);
  reveal(refs.markersRankPlot, hasRenderedPlot(refs.markersRankPlot));
  reveal(refs.markersTableShell?.closest(".table-card"), markerTable);

  ["ml", "dl"].forEach((goal) => {
    const isMl = goal === "ml";
    const resultMode = runtime.resultPreference?.[goal] || "single";
    const importancePlot = isMl ? refs.mlImportancePlot : refs.dlImportancePlot;
    const secondPlot = isMl ? refs.mlShapPlot : refs.dlLossPlot;
    const comparisonPlot = isMl ? refs.mlComparisonPlot : refs.dlComparisonPlot;
    const comparisonShell = isMl ? refs.mlComparisonShell : refs.dlComparisonShell;
    const manuscriptShell = isMl ? refs.mlManuscriptShell : refs.dlManuscriptShell;
    const insightBoard = isMl ? refs.mlInsightBoard : refs.dlInsightBoard;
    const metaBanner = isMl ? refs.mlMetaBanner : refs.dlMetaBanner;
    const hasImportance = resultMode === "single" && (hasRenderedPlot(importancePlot) || hasPlotMessage(importancePlot));
    const hasSecond = resultMode === "single" && (hasRenderedPlot(secondPlot) || hasPlotMessage(secondPlot));
    const hasComparePlot = resultMode === "compare" && hasRenderedPlot(comparisonPlot);
    const hasCompareTable = resultMode === "compare" && hasRenderedTable(comparisonShell);
    const hasManuscript = resultMode === "compare" && hasRenderedTable(manuscriptShell);
    const hasInsight = hasRenderedInsight(insightBoard);
    reveal(importancePlot, hasImportance);
    reveal(secondPlot, hasSecond);
    reveal(importancePlot?.closest(".ml-plots-grid"), hasImportance || hasSecond);
    reveal(comparisonPlot, hasComparePlot);
    reveal(comparisonShell?.closest(".table-card"), hasCompareTable);
    reveal(manuscriptShell?.closest(".table-card"), hasManuscript);
    reveal(insightBoard, hasInsight);
    reveal(metaBanner, hasImportance || hasSecond || hasComparePlot || hasCompareTable || hasManuscript || hasInsight);
  });

  scheduleVisiblePlotResize(40);
}

function resultAnchorFor(tabName, { mode = "single" } = {}) {
  const candidates = {
    km: [refs.kmPlot, refs.kmSummaryShell],
    cox: [refs.coxPlot, refs.coxDiagnosticsPlot, refs.coxMartingalePlot, refs.coxResultsShell],
    predictive: [refs.benchmarkSummaryGrid, refs.benchmarkComparisonPlot, refs.benchmarkComparisonShell, refs.benchmarkWorkbench],
    tables: [refs.cohortTableShell],
    markers: [refs.markersStabilityPlot, refs.markersInsightBoard],
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
  if (goal === "predictive") return activeTabName() === "benchmark";
  if (activeTabName() === "benchmark" && ["ml", "dl"].includes(goal)) return true;
  return activeTabName() === goal;
}

function revealCompletedResultIfCurrent(goal, { mode = "single", successMessage = "", backgroundMessage = "" } = {}) {
  const shouldReveal = shouldRevealCompletedResult(goal);
  if (shouldReveal) {
    activateTab(goal);
    updateResultVisibility();
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
    renderWorkspaceChrome,
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
  refs.predictiveModelSelector?.addEventListener("change", () => {
    runtime.workbenchRevealed = true;
    setPredictiveModel(refs.predictiveModelSelector.value, { historyMode: "push" });
    renderBenchmarkBoard();
    syncAnalysisRunButtonAvailability();
  });
  refs.runPredictiveCompareAllButton?.addEventListener("click", () => {
    withLoading(refs.runPredictiveCompareAllButton, runUnifiedPredictiveComparison, "predictive");
  });
  refs.runPredictiveSelectedButton?.addEventListener("click", () => {
    runtime.workbenchRevealed = true;
    runtime.predictiveWorkbenchIntent = "train";
    const selectedFamily = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey()).family;
    withLoading(refs.runPredictiveSelectedButton, runPredictiveSelectedModel, selectedFamily);
  });
  refs.runPredictiveWorkbenchButton?.addEventListener("click", () => {
    runtime.workbenchRevealed = true;
    runtime.predictiveWorkbenchIntent = "train";
    const selectedFamily = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey()).family;
    withLoading(refs.runPredictiveWorkbenchButton, runPredictiveSelectedModel, selectedFamily);
  });
  refs.openPredictiveWorkbenchButton?.addEventListener("click", () => {
    reviewBenchmarkModel(currentPredictiveModelKey(), "single");
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
  refs.loadGbsg2Button.addEventListener("click", () => withLoading(refs.loadGbsg2Button, loadGbsg2Dataset));
  refs.loadExampleButton.addEventListener("click", () => withLoading(refs.loadExampleButton, loadExampleDataset));
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
    queueHistorySync();
    scheduleCoxPreview();
  });
  refs.covariateSearchInput?.addEventListener("input", () => {
    applyChecklistSearch(refs.covariateChecklist);
  });
  refs.categoricalChecklist?.addEventListener("change", () => {
    renderSharedFeatureSummary();
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
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Selected all Cox covariates.", "success", 2200);
  });
  refs.clearCoxCovariatesButton?.addEventListener("click", () => {
    setCheckedValues(refs.covariateChecklist, []);
    setCheckedValues(refs.categoricalChecklist, []);
    syncCoxCovariateSelection();
    renderSharedFeatureSummary();
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
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Marked the current Cox covariates as categorical.", "success", 2200);
  });
  refs.clearCoxCategoricalsButton?.addEventListener("click", () => {
    setCheckedValues(refs.categoricalChecklist, []);
    renderSharedFeatureSummary();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Cleared Cox categorical flags.", "success", 2200);
  });
  refs.selectAllCoxStrataButton?.addEventListener("click", () => {
    const strata = allCheckboxValues(refs.strataChecklist, { visibleOnly: true });
    setCheckedValues(refs.strataChecklist, strata);
    syncCoxCovariateSelection({ preferredScope: "strata" });
    renderSharedFeatureSummary();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
    showToast("Selected visible Cox strata. Matching covariates were cleared automatically.", "success", 2400);
  });
  refs.clearCoxStrataButton?.addEventListener("click", () => {
    setCheckedValues(refs.strataChecklist, []);
    syncCoxCovariateSelection();
    renderSharedFeatureSummary();
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
  refs.runKmButton.addEventListener("click", () => withLoading(refs.runKmButton, runKaplanMeier));
  refs.runSignatureSearchButton.addEventListener("click", () => withLoading(refs.runSignatureSearchButton, runSignatureSearch, "km"));
  refs.runCoxButton.addEventListener("click", () => withLoading(refs.runCoxButton, runCox));
  refs.runCohortTableButton.addEventListener("click", () => withLoading(refs.runCohortTableButton, runCohortTable));
  refs.runMlButton.addEventListener("click", () => withLoading(refs.runMlButton, runMlModel));
  refs.runCompareButton.addEventListener("click", () => withLoading(refs.runCompareButton, runCompareModels));
  refs.runCompareInlineButton?.addEventListener("click", () => withLoading(refs.runCompareInlineButton, runCompareModels));
  refs.runDlButton.addEventListener("click", () => withLoading(refs.runDlButton, runDlModel));
  refs.runDlCompareButton.addEventListener("click", () => withLoading(refs.runDlCompareButton, runDlCompareModels));
  refs.runDlCompareInlineButton?.addEventListener("click", () => withLoading(refs.runDlCompareInlineButton, runDlCompareModels));
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
  // Every control that feeds a request must also refresh result currency (run status, downloads,
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
  wireMarkerControls();
  initExportMenus();
}

// Export menus close after a download is chosen and when the user clicks elsewhere.
function initExportMenus() {
  const menus = () => [...document.querySelectorAll(".export-menu")];
  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    menus().forEach((menu) => {
      if (!menu.open) return;
      if (!target || !menu.contains(target) || target.closest(".export-menu-items button")) menu.open = false;
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") menus().forEach((menu) => { menu.open = false; });
  });
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

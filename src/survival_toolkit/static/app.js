const appState = {
  dataset: null,
  km: null,
  cox: null,
  cohort: null,
  signature: null,
  ml: null,
  dl: null,
  isFilePreview: window.location.protocol === "file:",
  apiBase: window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "",
  historySyncPaused: false,
  historySyncTimer: null,
  lastDerivedGroup: null,
  deriveDraftTouched: false,
  uiMode: "guided",
  guidedGoal: null,
  guidedStep: 1,
  historyRestoreToken: 0,
  derivedColumnProvenance: {},
  busyScopes: {},
  plotResizeTimer: null,
  coxPreview: {
    key: "",
    status: "idle",
    payload: null,
    error: "",
  },
  coxPreviewTimer: null,
  coxPreviewToken: 0,
  resultPreference: {
    ml: "single",
    dl: "single",
  },
  compareCache: {
    ml: null,
    dl: null,
  },
  predictiveFamily: "ml",
  workbenchRevealed: false,
};

// Keep the legacy aliases, but route all mutable client state through one object.
const state = appState;
const runtime = appState;

const shellHelpers = window.SurvStudioShell;
const downloadHelpers = window.SurvStudioDownloads;


const refs = {
  runtimeBanner: document.getElementById("runtimeBanner"),
  landing: document.getElementById("landing"),
  workspace: document.getElementById("workspace"),
  datasetBadge: document.getElementById("datasetBadge"),
  datasetFile: document.getElementById("datasetFile"),
  guidedModeButton: document.getElementById("guidedModeButton"),
  expertModeButton: document.getElementById("expertModeButton"),
  uploadButton: document.getElementById("uploadButton"),
  shutdownButton: document.getElementById("shutdownButton"),
  loadTcgaUploadReadyButton: document.getElementById("loadTcgaUploadReadyButton"),
  loadTcgaButton: document.getElementById("loadTcgaButton"),
  loadGbsg2Button: document.getElementById("loadGbsg2Button"),
  loadExampleButton: document.getElementById("loadExampleButton"),
  datasetPreviewShell: document.getElementById("datasetPreviewShell"),
  guidedShell: document.getElementById("guidedShell"),
  stepIndicator: document.getElementById("stepIndicator"),
  guidedSummaryBar: document.getElementById("guidedSummaryBar"),
  guidedSummaryTitle: document.getElementById("guidedSummaryTitle"),
  guidedSummaryText: document.getElementById("guidedSummaryText"),
  guidedSummaryChips: document.getElementById("guidedSummaryChips"),
  guidedRailStatus: document.getElementById("guidedRailStatus"),
  guidedRailStatusLabel: document.getElementById("guidedRailStatusLabel"),
  guidedRailStatusTitle: document.getElementById("guidedRailStatusTitle"),
  guidedRailStatusText: document.getElementById("guidedRailStatusText"),
  guidedRailActions: document.getElementById("guidedRailActions"),
  guidedRailPanelMount: document.getElementById("guidedRailPanelMount"),
  guidedConfigMount: document.getElementById("guidedConfigMount"),
  guidedPanel: document.getElementById("guidedPanel"),
  guidedActivePanelMount: document.getElementById("guidedActivePanelMount"),
  configStripHome: document.getElementById("configStripHome"),
  tabPanelsHome: document.getElementById("tabPanelsHome"),
  outcomeConfigBlock: document.getElementById("outcomeConfigBlock"),
  groupingConfigBlock: document.getElementById("groupingConfigBlock"),
  smartBanner: document.getElementById("smartBanner"),
  smartBannerText: document.getElementById("smartBannerText"),
  smartBannerClose: document.getElementById("smartBannerClose"),
  datasetPresetBar: document.getElementById("datasetPresetBar"),
  datasetPresetTitle: document.getElementById("datasetPresetTitle"),
  datasetPresetText: document.getElementById("datasetPresetText"),
  datasetPresetStatusTitle: document.getElementById("datasetPresetStatusTitle"),
  datasetPresetStatusText: document.getElementById("datasetPresetStatusText"),
  datasetPresetChips: document.getElementById("datasetPresetChips"),
  groupingDetails: document.getElementById("groupingDetails"),
  groupingSummaryText: document.getElementById("groupingSummaryText"),
  configTitleText: document.getElementById("configTitleText"),
  configHint: document.getElementById("configHint"),
  applyBasicPresetButton: document.getElementById("applyBasicPresetButton"),
  applyModelPresetButton: document.getElementById("applyModelPresetButton"),
  tooltipPopup: document.getElementById("tooltipPopup"),
  configStrip: document.getElementById("configStrip"),
  tabStrip: document.getElementById("tabStrip"),
  timeColumn: document.getElementById("timeColumn"),
  timeColumnHelp: document.getElementById("timeColumnHelp"),
  timeColumnWarning: document.getElementById("timeColumnWarning"),
  eventColumn: document.getElementById("eventColumn"),
  eventPositiveValue: document.getElementById("eventPositiveValue"),
  showAllEventColumns: document.getElementById("showAllEventColumns"),
  eventColumnHelp: document.getElementById("eventColumnHelp"),
  eventColumnWarning: document.getElementById("eventColumnWarning"),
  eventValueWarning: document.getElementById("eventValueWarning"),
  groupColumn: document.getElementById("groupColumn"),
  groupColumnWarning: document.getElementById("groupColumnWarning"),
  timeUnitLabel: document.getElementById("timeUnitLabel"),
  maxTime: document.getElementById("maxTime"),
  confidenceLevel: document.getElementById("confidenceLevel"),
  deriveToggle: document.getElementById("deriveToggle"),
  derivePanel: document.getElementById("derivePanel"),
  deriveSource: document.getElementById("deriveSource"),
  deriveMethod: document.getElementById("deriveMethod"),
  deriveCutoff: document.getElementById("deriveCutoff"),
  deriveMinGroupFraction: document.getElementById("deriveMinGroupFraction"),
  derivePermutationIterations: document.getElementById("derivePermutationIterations"),
  deriveRandomSeed: document.getElementById("deriveRandomSeed"),
  deriveColumnName: document.getElementById("deriveColumnName"),
  cutoffWrap: document.getElementById("cutoffWrap"),
  deriveCutoffLabel: document.getElementById("deriveCutoffLabel"),
  deriveCutoffHelp: document.getElementById("deriveCutoffHelp"),
  deriveOptimalControls: document.getElementById("deriveOptimalControls"),
  deriveButton: document.getElementById("deriveButton"),
  deriveStatus: document.getElementById("deriveStatus"),
  deriveSummary: document.getElementById("deriveSummary"),
  cutpointPlot: document.getElementById("cutpointPlot"),
  showConfidenceBands: document.getElementById("showConfidenceBands"),
  riskTablePoints: document.getElementById("riskTablePoints"),
  logrankWeight: document.getElementById("logrankWeight"),
  fhPowerWrap: document.getElementById("fhPowerWrap"),
  fhPower: document.getElementById("fhPower"),
  runKmButton: document.getElementById("runKmButton"),
  runSignatureSearchButton: document.getElementById("runSignatureSearchButton"),
  downloadSignatureButton: document.getElementById("downloadSignatureButton"),
  signatureMaxDepth: document.getElementById("signatureMaxDepth"),
  signatureMinFraction: document.getElementById("signatureMinFraction"),
  signatureTopK: document.getElementById("signatureTopK"),
  signatureBootstrapIterations: document.getElementById("signatureBootstrapIterations"),
  signaturePermutationIterations: document.getElementById("signaturePermutationIterations"),
  signatureValidationIterations: document.getElementById("signatureValidationIterations"),
  signatureValidationFraction: document.getElementById("signatureValidationFraction"),
  signatureSignificanceLevel: document.getElementById("signatureSignificanceLevel"),
  signatureOperator: document.getElementById("signatureOperator"),
  signatureRandomSeed: document.getElementById("signatureRandomSeed"),
  kmInsightBoard: document.getElementById("kmInsightBoard"),
  kmPlot: document.getElementById("kmPlot"),
  kmMetaBanner: document.getElementById("kmMetaBanner"),
  kmDependencyText: document.getElementById("kmDependencyText"),
  kmDependencyChips: document.getElementById("kmDependencyChips"),
  kmSummaryShell: document.getElementById("kmSummaryShell"),
  kmRiskShell: document.getElementById("kmRiskShell"),
  kmPairwiseShell: document.getElementById("kmPairwiseShell"),
  signatureInsightBoard: document.getElementById("signatureInsightBoard"),
  signatureShell: document.getElementById("signatureShell"),
  downloadKmSummaryButton: document.getElementById("downloadKmSummaryButton"),
  downloadKmPairwiseButton: document.getElementById("downloadKmPairwiseButton"),
  downloadKmPngButton: document.getElementById("downloadKmPngButton"),
  downloadKmSvgButton: document.getElementById("downloadKmSvgButton"),
  covariateChecklist: document.getElementById("covariateChecklist"),
  categoricalChecklist: document.getElementById("categoricalChecklist"),
  covariateSearchInput: document.getElementById("covariateSearchInput"),
  categoricalSearchInput: document.getElementById("categoricalSearchInput"),
  coxCovariateWarning: document.getElementById("coxCovariateWarning"),
  selectAllCoxCovariatesButton: document.getElementById("selectAllCoxCovariatesButton"),
  clearCoxCovariatesButton: document.getElementById("clearCoxCovariatesButton"),
  selectAllCoxCategoricalsButton: document.getElementById("selectAllCoxCategoricalsButton"),
  clearCoxCategoricalsButton: document.getElementById("clearCoxCategoricalsButton"),
  modelFeatureChecklist: document.getElementById("modelFeatureChecklist"),
  modelCategoricalChecklist: document.getElementById("modelCategoricalChecklist"),
  dlModelFeatureChecklist: document.getElementById("dlModelFeatureChecklist"),
  dlModelCategoricalChecklist: document.getElementById("dlModelCategoricalChecklist"),
  selectAllModelFeaturesButton: document.getElementById("selectAllModelFeaturesButton"),
  clearModelFeaturesButton: document.getElementById("clearModelFeaturesButton"),
  selectAllDlModelFeaturesButton: document.getElementById("selectAllDlModelFeaturesButton"),
  clearDlModelFeaturesButton: document.getElementById("clearDlModelFeaturesButton"),
  runCoxButton: document.getElementById("runCoxButton"),
  coxInsightBoard: document.getElementById("coxInsightBoard"),
  coxPlot: document.getElementById("coxPlot"),
  coxMetaBanner: document.getElementById("coxMetaBanner"),
  coxDependencyText: document.getElementById("coxDependencyText"),
  coxDependencyChips: document.getElementById("coxDependencyChips"),
  coxResultsShell: document.getElementById("coxResultsShell"),
  coxDiagnosticsPlot: document.getElementById("coxDiagnosticsPlot"),
  coxDiagnosticsShell: document.getElementById("coxDiagnosticsShell"),
  downloadCoxResultsButton: document.getElementById("downloadCoxResultsButton"),
  downloadCoxDiagnosticsButton: document.getElementById("downloadCoxDiagnosticsButton"),
  downloadCoxPngButton: document.getElementById("downloadCoxPngButton"),
  downloadCoxSvgButton: document.getElementById("downloadCoxSvgButton"),
  cohortVariableChecklist: document.getElementById("cohortVariableChecklist"),
  cohortVariableSearchInput: document.getElementById("cohortVariableSearchInput"),
  selectAllCohortVariablesButton: document.getElementById("selectAllCohortVariablesButton"),
  clearCohortVariablesButton: document.getElementById("clearCohortVariablesButton"),
  runCohortTableButton: document.getElementById("runCohortTableButton"),
  runCohortTableButtonLabel: document.getElementById("runCohortTableButtonLabel"),
  cohortTableShell: document.getElementById("cohortTableShell"),
  tableDependencyText: document.getElementById("tableDependencyText"),
  tableDependencyChips: document.getElementById("tableDependencyChips"),
  tableOutputStatusText: document.getElementById("tableOutputStatusText"),
  downloadCohortTableButton: document.getElementById("downloadCohortTableButton"),
  uploadZone: document.getElementById("uploadZone"),
  brandHome: document.getElementById("brandHome"),
  // ML
  runMlButton: document.getElementById("runMlButton"),
  runCompareButton: document.getElementById("runCompareButton"),
  runCompareInlineButton: document.getElementById("runCompareInlineButton"),
  downloadMlComparisonButton: document.getElementById("downloadMlComparisonButton"),
  downloadMlManuscriptCsvButton: document.getElementById("downloadMlManuscriptCsvButton"),
  downloadMlManuscriptMarkdownButton: document.getElementById("downloadMlManuscriptMarkdownButton"),
  downloadMlManuscriptLatexButton: document.getElementById("downloadMlManuscriptLatexButton"),
  downloadMlManuscriptDocxButton: document.getElementById("downloadMlManuscriptDocxButton"),
  downloadMlComparisonPngButton: document.getElementById("downloadMlComparisonPngButton"),
  downloadMlComparisonSvgButton: document.getElementById("downloadMlComparisonSvgButton"),
  mlModelType: document.getElementById("mlModelType"),
  mlNEstimators: document.getElementById("mlNEstimators"),
  mlLearningRate: document.getElementById("mlLearningRate"),
  mlSkipShap: document.getElementById("mlSkipShap"),
  mlShapSafeMode: document.getElementById("mlShapSafeMode"),
  mlEvaluationStrategy: document.getElementById("mlEvaluationStrategy"),
  mlCvFoldsWrap: document.getElementById("mlCvFoldsWrap"),
  mlCvRepeatsWrap: document.getElementById("mlCvRepeatsWrap"),
  mlCvFolds: document.getElementById("mlCvFolds"),
  mlCvRepeats: document.getElementById("mlCvRepeats"),
  mlJournalTemplate: document.getElementById("mlJournalTemplate"),
  mlFeatureSummaryCard: document.getElementById("mlFeatureSummaryCard"),
  mlFeatureSummaryText: document.getElementById("mlFeatureSummaryText"),
  mlFeatureSummaryChips: document.getElementById("mlFeatureSummaryChips"),
  reviewMlFeaturesButton: document.getElementById("reviewMlFeaturesButton"),
  mlImportancePlot: document.getElementById("mlImportancePlot"),
  mlShapPlot: document.getElementById("mlShapPlot"),
  mlComparisonPlot: document.getElementById("mlComparisonPlot"),
  mlMetaBanner: document.getElementById("mlMetaBanner"),
  mlInsightBoard: document.getElementById("mlInsightBoard"),
  mlComparisonShell: document.getElementById("mlComparisonShell"),
  mlComparisonTitle: document.getElementById("mlComparisonTitle"),
  mlManuscriptShell: document.getElementById("mlManuscriptShell"),
  // DL
  runDlButton: document.getElementById("runDlButton"),
  runDlCompareButton: document.getElementById("runDlCompareButton"),
  runDlCompareInlineButton: document.getElementById("runDlCompareInlineButton"),
  downloadDlComparisonButton: document.getElementById("downloadDlComparisonButton"),
  downloadDlManuscriptCsvButton: document.getElementById("downloadDlManuscriptCsvButton"),
  downloadDlManuscriptMarkdownButton: document.getElementById("downloadDlManuscriptMarkdownButton"),
  downloadDlManuscriptLatexButton: document.getElementById("downloadDlManuscriptLatexButton"),
  downloadDlManuscriptDocxButton: document.getElementById("downloadDlManuscriptDocxButton"),
  downloadDlComparisonPngButton: document.getElementById("downloadDlComparisonPngButton"),
  downloadDlComparisonSvgButton: document.getElementById("downloadDlComparisonSvgButton"),
  dlModelType: document.getElementById("dlModelType"),
  dlEpochs: document.getElementById("dlEpochs"),
  dlLearningRate: document.getElementById("dlLearningRate"),
  dlHiddenLayers: document.getElementById("dlHiddenLayers"),
  dlDropout: document.getElementById("dlDropout"),
  dlBatchSize: document.getElementById("dlBatchSize"),
  dlBatchSizeHint: document.getElementById("dlBatchSizeHint"),
  dlRandomSeed: document.getElementById("dlRandomSeed"),
  dlEvaluationStrategy: document.getElementById("dlEvaluationStrategy"),
  dlCvFoldsWrap: document.getElementById("dlCvFoldsWrap"),
  dlCvRepeatsWrap: document.getElementById("dlCvRepeatsWrap"),
  dlCvFolds: document.getElementById("dlCvFolds"),
  dlCvRepeats: document.getElementById("dlCvRepeats"),
  dlEarlyStoppingPatience: document.getElementById("dlEarlyStoppingPatience"),
  dlEarlyStoppingMinDelta: document.getElementById("dlEarlyStoppingMinDelta"),
  dlParallelJobs: document.getElementById("dlParallelJobs"),
  dlNumTimeBinsWrap: document.getElementById("dlNumTimeBinsWrap"),
  dlNumTimeBins: document.getElementById("dlNumTimeBins"),
  dlDModelWrap: document.getElementById("dlDModelWrap"),
  dlDModel: document.getElementById("dlDModel"),
  dlHeadsWrap: document.getElementById("dlHeadsWrap"),
  dlHeads: document.getElementById("dlHeads"),
  dlLayersWrap: document.getElementById("dlLayersWrap"),
  dlLayers: document.getElementById("dlLayers"),
  dlLatentDimWrap: document.getElementById("dlLatentDimWrap"),
  dlLatentDim: document.getElementById("dlLatentDim"),
  dlClustersWrap: document.getElementById("dlClustersWrap"),
  dlClusters: document.getElementById("dlClusters"),
  dlJournalTemplate: document.getElementById("dlJournalTemplate"),
  dlFeatureSummaryCard: document.getElementById("dlFeatureSummaryCard"),
  dlFeatureSummaryText: document.getElementById("dlFeatureSummaryText"),
  dlFeatureSummaryChips: document.getElementById("dlFeatureSummaryChips"),
  reviewDlFeaturesButton: document.getElementById("reviewDlFeaturesButton"),
  dlImportancePlot: document.getElementById("dlImportancePlot"),
  dlLossPlot: document.getElementById("dlLossPlot"),
  dlComparisonPlot: document.getElementById("dlComparisonPlot"),
  dlComparisonShell: document.getElementById("dlComparisonShell"),
  dlComparisonTitle: document.getElementById("dlComparisonTitle"),
  dlManuscriptShell: document.getElementById("dlManuscriptShell"),
  dlMetaBanner: document.getElementById("dlMetaBanner"),
  dlInsightBoard: document.getElementById("dlInsightBoard"),
  runPredictiveCompareAllButton: document.getElementById("runPredictiveCompareAllButton"),
  predictiveModelSelector: document.getElementById("predictiveModelSelector"),
  runPredictiveSelectedButton: document.getElementById("runPredictiveSelectedButton"),
  predictiveActionStatusText: document.getElementById("predictiveActionStatusText"),
  benchmarkActionCard: document.getElementById("benchmarkActionCard"),
  benchmarkSummaryGrid: document.getElementById("benchmarkSummaryGrid"),
  benchmarkComparisonPlot: document.getElementById("benchmarkComparisonPlot"),
  benchmarkPlotNote: document.getElementById("benchmarkPlotNote"),
  benchmarkComparisonShell: document.getElementById("benchmarkComparisonShell"),
  benchmarkTableNote: document.getElementById("benchmarkTableNote"),
  benchmarkWorkbench: document.getElementById("benchmarkWorkbench"),
  benchmarkWorkbenchCaption: document.getElementById("benchmarkWorkbenchCaption"),
  closePredictiveWorkbenchButton: document.getElementById("closePredictiveWorkbenchButton"),
  benchmarkMlMount: document.getElementById("benchmarkMlMount"),
  benchmarkDlMount: document.getElementById("benchmarkDlMount"),
  mlPanel: document.getElementById("panel-ml"),
  dlPanel: document.getElementById("panel-dl"),
  benchmarkPanel: document.getElementById("panel-benchmark"),
  mlWorkspaceCard: document.querySelector("#panel-ml > .workspace-card"),
  dlWorkspaceCard: document.querySelector("#panel-dl > .workspace-card"),
  tabButtons: [...document.querySelectorAll(".tab-button")],
  tabPanels: [...document.querySelectorAll(".tab-panel")],
};

const REQUIRED_REF_KEYS = Object.freeze(Object.keys(refs));

function assertRequiredRefs() {
  const missing = REQUIRED_REF_KEYS.filter((key) => {
    const value = refs[key];
    return Array.isArray(value) ? value.length === 0 : value == null;
  });
  if (missing.length) {
    throw new Error(`Missing required DOM references: ${missing.join(", ")}`);
  }
}

assertRequiredRefs();

const DEFAULT_MODEL_FEATURE_SELECTION_LIMIT = 20;
const AUTO_CATEGORICAL_UNIQUE_THRESHOLD = 6;
const COX_STAGE_VARIABLE_PREFERENCE = ["stage_group", "pathologic_stage", "stage"];
const DATASET_PRESETS = Object.freeze({
  gbsg2: {
    name: "GBSG2 preset",
    summary: "RFS in days with horTh as the first KM split and a compact Cox or ML feature set.",
    timeColumn: "rfs_days",
    eventColumn: "rfs_event",
    eventPositiveValue: "1",
    timeUnitLabel: "Days",
    basicGroup: "horTh",
    tableVariables: ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
    coxCovariates: ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
    coxCategoricals: ["horTh", "menostat", "tgrade"],
    modelFeatures: ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
    modelCategoricals: ["horTh", "menostat", "tgrade"],
  },
  tcga_luad: {
    name: "TCGA LUAD preset",
    summary: "Overall survival in months with stage_group for Kaplan-Meier and a compact smoking-aware feature set.",
    timeColumn: "os_months",
    eventColumn: "os_event",
    eventPositiveValue: "1",
    timeUnitLabel: "Months",
    basicGroup: "stage_group",
    tableVariables: ["age", "sex", "stage_group", "smoking_status"],
    coxCovariates: ["age", "sex", "stage_group", "smoking_status"],
    coxCategoricals: ["sex", "stage_group", "smoking_status"],
    modelFeatures: ["age", "sex", "stage_group", "smoking_status"],
    modelCategoricals: ["sex", "stage_group", "smoking_status"],
  },
});

const EVENT_TRUE_TOKENS = new Set([
  "1", "true", "t", "yes", "y", "event", "dead", "deceased", "died", "failure",
  "failed", "progressed", "progression", "relapse", "recurred", "recurrence",
]);
const EVENT_FALSE_TOKENS = new Set([
  "0", "false", "f", "no", "n", "censor", "censored", "alive", "living",
  "none", "disease_free", "disease-free", "progression_free", "progression-free",
]);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function apiUrl(url) {
  if (/^https?:\/\//.test(url)) return url;
  if (url.startsWith("/")) return `${runtime.apiBase}${url}`;
  return url;
}

function parseHiddenLayers(control = refs.dlHiddenLayers) {
  const raw = String(control?.value || "").trim();
  if (!raw) return [];
  const tokens = raw.split(",").map((value) => value.trim());
  if (!tokens.length || tokens.some((token) => token.length === 0)) return [];
  const parsed = tokens.map((value) => Number(value));
  if (!parsed.every((value) => Number.isInteger(value) && value > 0)) return [];
  return parsed;
}

function parseHiddenLayersStrict(control = refs.dlHiddenLayers) {
  const hiddenLayerText = String(control?.value || "").trim();
  const hiddenLayers = parseHiddenLayers(control);
  if (!hiddenLayerText || !hiddenLayers.length) {
    throw new Error(`Hidden layers must be a comma-separated list of positive integers. Current value: ${hiddenLayerText || "(empty)"}.`);
  }
  return hiddenLayers;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(apiUrl(url), {
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
    ...options,
  });
  const rawText = await response.text();
  let payload = {};
  if (rawText.trim()) {
    try {
      payload = JSON.parse(rawText);
    } catch (error) {
      const genericMessage = response.ok
        ? "The server returned an invalid JSON response."
        : (rawText.trim() || "Request failed.");
      throw new Error(genericMessage);
    }
  }
  if (!response.ok) {
    const message = extractErrorMessage(payload, rawText);
    if (response.status === 404 && /Unknown dataset id:/i.test(message) && state.dataset) {
      const datasetName = state.dataset?.filename || state.dataset?.dataset_id || "current cohort";
      goHome({ syncHistory: true, historyMode: "replace" });
      setRuntimeBanner(`The previously loaded cohort (${datasetName}) is no longer available on the server. Reload it to continue.`, "warning");
      throw new Error("The loaded dataset is no longer available on the server. Reload a dataset and run the analysis again.");
    }
    throw new Error(message);
  }
  return payload;
}

function extractErrorMessage(payload, fallbackText = "") {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((item) => {
      const path = Array.isArray(item?.loc)
        ? item.loc.filter((part) => part !== "body").join(" > ")
        : "";
      const message = item?.msg || "Invalid input.";
      return path ? `${path}: ${message}` : message;
    }).join(" | ");
  }
  if (typeof fallbackText === "string" && fallbackText.trim()) return fallbackText.trim();
  return "Request failed.";
}

function errorMessageText(error, fallbackText = "Request failed.") {
  if (typeof error === "string" && error.trim()) return error.trim();
  if (typeof error?.message === "string" && error.message.trim()) return error.message.trim();
  return fallbackText;
}

function setRuntimeBanner(text = "", tone = "info") {
  if (!refs.runtimeBanner) return;
  if (!text) {
    refs.runtimeBanner.textContent = "";
    refs.runtimeBanner.className = "runtime-banner hidden";
    return;
  }
  refs.runtimeBanner.textContent = text;
  refs.runtimeBanner.className = `runtime-banner runtime-banner-${tone}`;
}

function renderServerStoppedState(message) {
  const resolvedMessage = message || "SurvStudio is stopping. You can close this tab or restart the server with `python -m survival_toolkit`.";
  const landing = document.createElement("div");
  landing.className = "landing";
  landing.style.display = "grid";
  landing.style.minHeight = "100vh";
  landing.style.placeItems = "center";
  landing.style.padding = "24px";

  const card = document.createElement("div");
  card.className = "landing-card";
  card.style.maxWidth = "720px";
  card.style.width = "min(100%, 720px)";

  const copy = document.createElement("div");
  copy.className = "landing-hero-copy";
  copy.style.padding = "12px 8px";

  const heading = document.createElement("h2");
  heading.textContent = "SurvStudio server stopped";

  const messageParagraph = document.createElement("p");
  messageParagraph.textContent = resolvedMessage;

  const restartParagraph = document.createElement("p");
  restartParagraph.append("Restart with ");
  const commandCode = document.createElement("code");
  commandCode.textContent = "python -m survival_toolkit";
  restartParagraph.append(commandCode);
  restartParagraph.append(", then reopen ");
  const urlCode = document.createElement("code");
  urlCode.textContent = "http://127.0.0.1:8000";
  restartParagraph.append(urlCode);
  restartParagraph.append(".");

  copy.append(heading, messageParagraph, restartParagraph);
  card.append(copy);
  landing.append(card);
  document.body.replaceChildren(landing);
}

function activeTabName() {
  return document.querySelector(".tab-button.active")?.dataset.tab || "km";
}

function normalizedPredictiveFamily(family) {
  return family === "dl" ? "dl" : "ml";
}

function preferredResultMode(goal) {
  if (goal === "ml" || goal === "dl") return runtime.resultPreference?.[goal] || "single";
  return "single";
}

const GUIDED_GOALS = ["km", "cox", "predictive", "tables", "ml", "dl"];

function predictiveFamilyGoal() {
  return normalizedPredictiveFamily(runtime.predictiveFamily);
}

function alternatePredictiveFamilyGoal() {
  return predictiveFamilyGoal() === "ml" ? "dl" : "ml";
}

function goalLabel(goal) {
  return {
    km: "Kaplan-Meier",
    cox: "Cox PH",
    predictive: "ML/DL Models",
    ml: "ML Models",
    dl: "Deep Learning",
    tables: "Cohort Table",
  }[goal] || "Choose analysis";
}

function guidedGoalMeta(goal) {
  return {
    km: {
      badge: "Start here",
      description: "See survival curves first. This is the easiest first check for most datasets.",
      note: "Best first run for beginners",
    },
    cox: {
      badge: "Good next step",
      description: "Estimate which variables are linked to higher or lower risk over time.",
      note: "Use after Kaplan-Meier makes sense",
    },
    tables: {
      badge: "Useful summary",
      description: "Make a cohort summary table for baseline characteristics and reporting.",
      note: "Good for manuscripts and QC",
    },
    predictive: {
      badge: "Advanced",
      description: "Compare classical ML and deep survival models from one predictive workspace.",
      note: "Start with ML for a faster baseline, then switch to DL if needed",
    },
    ml: {
      badge: "Advanced",
      description: "Compare Cox, random survival forest, and gradient boosting models.",
      note: "Use after the classical analysis looks right",
    },
    dl: {
      badge: "Advanced",
      description: "Train deep survival models such as DeepSurv, DeepHit, and Transformer.",
      note: "Slowest option and easiest to misuse",
    },
  }[goal] || {
    badge: "Analysis",
    description: "Choose the analysis you want to run next.",
    note: "",
  };
}

function goalFeatureCount(goal) {
  if (goal === "cox") return selectedCheckboxValues(refs.covariateChecklist).length;
  if (goal === "ml" || goal === "dl" || goal === "predictive") return selectedCheckboxValues(refs.modelFeatureChecklist).length;
  if (goal === "tables") return selectedCheckboxValues(refs.cohortVariableChecklist).length;
  return 0;
}

function evaluationModeLabel(goal) {
  if (goal === "predictive") {
    return evaluationModeLabel(predictiveFamilyGoal());
  }
  if (goal === "ml") {
    return refs.mlEvaluationStrategy?.selectedOptions?.[0]?.textContent || refs.mlEvaluationStrategy?.value || "Holdout";
  }
  if (goal === "dl") {
    return refs.dlEvaluationStrategy?.selectedOptions?.[0]?.textContent || refs.dlEvaluationStrategy?.value || "Holdout";
  }
  return null;
}

function guidedResultModeLabel(goal) {
  if (goal === "predictive") return guidedResultModeLabel(predictiveFamilyGoal());
  if (goal !== "ml" && goal !== "dl") return null;
  return (runtime.resultPreference?.[goal] || "single") === "compare" ? "Compare all" : "Run Analysis";
}

function arrayEquals(left = [], right = []) {
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
}

function sortedStrings(values = []) {
  return [...values].map((value) => String(value)).sort();
}

function currentCoxSelections() {
  const covariates = selectedCheckboxValues(refs.covariateChecklist);
  return {
    covariates,
    categoricalCovariates: selectedCheckboxValues(refs.categoricalChecklist).filter((value) => covariates.includes(value)),
  };
}

function coxPreviewRequestFromCurrentState() {
  const { covariates, categoricalCovariates } = currentCoxSelections();
  if (!covariates.length) return null;
  const base = currentBaseConfig();
  return {
    dataset_id: base.dataset_id,
    time_column: base.time_column,
    event_column: base.event_column,
    event_positive_value: base.event_positive_value,
    covariates,
    categorical_covariates: categoricalCovariates,
  };
}

function coxPreviewRequestKey(requestConfig) {
  if (!requestConfig) return "";
  return JSON.stringify({
    dataset_id: requestConfig.dataset_id,
    time_column: requestConfig.time_column,
    event_column: requestConfig.event_column,
    event_positive_value: requestConfig.event_positive_value,
    covariates: sortedStrings(requestConfig.covariates || []),
    categorical_covariates: sortedStrings(requestConfig.categorical_covariates || []),
  });
}

function resetCoxPreview({ rerender = true } = {}) {
  if (runtime.coxPreviewTimer) {
    window.clearTimeout(runtime.coxPreviewTimer);
    runtime.coxPreviewTimer = null;
  }
  runtime.coxPreviewToken += 1;
  runtime.coxPreview = {
    key: "",
    status: "idle",
    payload: null,
    error: "",
  };
  if (rerender && runtime.uiMode === "guided" && runtime.guidedGoal === "cox") {
    if (!syncGuidedCoxPanelMounts()) renderGuidedChrome();
  }
}

async function refreshCoxPreview({ force = false } = {}) {
  if (!state.dataset) {
    resetCoxPreview({ rerender: false });
    return;
  }
  let requestConfig = null;
  try {
    requestConfig = coxPreviewRequestFromCurrentState();
  } catch (error) {
    runtime.coxPreview = {
      key: "",
      status: "blocked",
      payload: null,
      error: error.message || "Cox preview is unavailable until the endpoint is configured.",
    };
    if (runtime.uiMode === "guided" && runtime.guidedGoal === "cox") {
      if (!syncGuidedCoxPanelMounts()) renderGuidedChrome();
    }
    return;
  }
  if (!requestConfig) {
    resetCoxPreview({ rerender: false });
    if (runtime.uiMode === "guided" && runtime.guidedGoal === "cox") {
      if (!syncGuidedCoxPanelMounts()) renderGuidedChrome();
    }
    return;
  }
  const requestKey = coxPreviewRequestKey(requestConfig);
  if (!force && runtime.coxPreview.key === requestKey && runtime.coxPreview.status === "ready") return;
  const previewToken = ++runtime.coxPreviewToken;
  runtime.coxPreview = {
    key: requestKey,
    status: "loading",
    payload: null,
    error: "",
  };
  if (runtime.uiMode === "guided" && runtime.guidedGoal === "cox") {
    syncGuidedCoxPanelMounts();
  }
  try {
    const payload = await fetchJSON("/api/cox-preview", {
      method: "POST",
      body: JSON.stringify(requestConfig),
    });
    if (previewToken !== runtime.coxPreviewToken) return;
    runtime.coxPreview = {
      key: requestKey,
      status: "ready",
      payload,
      error: "",
    };
  } catch (error) {
    if (previewToken !== runtime.coxPreviewToken) return;
    runtime.coxPreview = {
      key: requestKey,
      status: "error",
      payload: null,
      error: error.message || "Cox preview is unavailable.",
    };
  }
  if (runtime.uiMode === "guided" && runtime.guidedGoal === "cox") {
    if (!syncGuidedCoxPanelMounts()) renderGuidedChrome();
  }
}

function scheduleCoxPreview({ delay = 180, force = false } = {}) {
  if (runtime.coxPreviewTimer) {
    window.clearTimeout(runtime.coxPreviewTimer);
  }
  const effectiveDelay = delay <= 0 ? 60 : delay;
  runtime.coxPreviewTimer = window.setTimeout(() => {
    runtime.coxPreviewTimer = null;
    void refreshCoxPreview({ force });
  }, effectiveDelay);
}

function requestConfigFromPayload(payload) {
  return payload?.request_config || payload?.analysis?.request_config || null;
}

function currentCohortTableOutputState() {
  const requestConfig = requestConfigFromPayload(state.cohort);
  if (!requestConfig || !state.dataset) {
    return {
      hasOutput: false,
      isCurrent: false,
      outputVariables: [],
      outputGroupLabel: "overall only",
    };
  }
  const outputVariables = (requestConfig.variables || []).map(String);
  const outputGroupLabel = String(requestConfig.group_column || "") || "overall only";
  return {
    hasOutput: true,
    isCurrent: matchesRequestConfig("tables", requestConfig),
    outputVariables,
    outputGroupLabel,
  };
}

function currentSignatureResult() {
  const payload = state.signature;
  if (!payload || !state.dataset) return null;
  const requestConfig = requestConfigFromPayload(payload);
  if (!requestConfig) return null;
  const base = currentBaseConfig();
  const currentDerivedName = String(refs.deriveColumnName?.value || "").trim();
  const currentCandidates = sortedStrings(selectedCheckboxValues(refs.covariateChecklist));
  const requestedCandidates = sortedStrings(requestConfig.candidate_columns || []);
  const isCurrent = (
    String(requestConfig.dataset_id || "") === String(state.dataset.dataset_id || "")
    && String(requestConfig.time_column || "") === String(base.time_column || "")
    && String(requestConfig.event_column || "") === String(base.event_column || "")
    && String(requestConfig.event_positive_value ?? "") === String(base.event_positive_value ?? "")
    && String(requestConfig.new_column_name || "") === currentDerivedName
    && String(requestConfig.combination_operator || "mixed") === String(refs.signatureOperator?.value || "mixed")
    && Number(requestConfig.max_combination_size || 3) === Number(refs.signatureMaxDepth?.value || 3)
    && Number(requestConfig.top_k || 15) === Number(refs.signatureTopK?.value || 15)
    && Number(requestConfig.min_group_fraction || 0.1) === Number(refs.signatureMinFraction?.value || 0.1)
    && Number(requestConfig.bootstrap_iterations || 30) === Number(refs.signatureBootstrapIterations?.value || 30)
    && Number(requestConfig.permutation_iterations || 120) === Number(refs.signaturePermutationIterations?.value || 120)
    && Number(requestConfig.validation_iterations || 12) === Number(refs.signatureValidationIterations?.value || 12)
    && Number(requestConfig.validation_fraction || 0.35) === Number(refs.signatureValidationFraction?.value || 0.35)
    && Number(requestConfig.significance_level || 0.05) === Number(refs.signatureSignificanceLevel?.value || 0.05)
    && Number(requestConfig.random_seed || 20260311) === Number(refs.signatureRandomSeed?.value || 20260311)
    && arrayEquals(requestedCandidates, currentCandidates)
  );
  return isCurrent ? payload : null;
}

function updateCohortTableButtonLabel() {
  if (!refs.runCohortTableButtonLabel) return;
  const tableState = currentCohortTableOutputState();
  refs.runCohortTableButtonLabel.textContent = tableState.hasOutput && !tableState.isCurrent
    ? "Rebuild Table"
    : "Build Table";
}

function stableStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableStringify(entry)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function currentSharedModelSelections() {
  const features = selectedCheckboxValues(refs.modelFeatureChecklist);
  return {
    features,
    categoricalFeatures: selectedCheckboxValues(refs.modelCategoricalChecklist)
      .filter((value) => features.includes(value)),
  };
}

function normalizeBaseRequestConfig(requestConfig) {
  return {
    dataset_id: String(requestConfig?.dataset_id || ""),
    time_column: String(requestConfig?.time_column || ""),
    event_column: String(requestConfig?.event_column || ""),
    event_positive_value: String(requestConfig?.event_positive_value ?? ""),
  };
}

function normalizedRequestConfig(goal, requestConfig, { expectsCompare = false } = {}) {
  if (!requestConfig) return null;
  const base = normalizeBaseRequestConfig(requestConfig);

  if (goal === "km") {
    return {
      ...base,
      group_column: String(requestConfig.group_column || ""),
      confidence_level: Number(requestConfig.confidence_level),
      time_unit_label: String(requestConfig.time_unit_label || "Months"),
      max_time: String(requestConfig.max_time ?? ""),
      risk_table_points: Number(requestConfig.risk_table_points),
      logrank_weight: String(requestConfig.logrank_weight || "logrank"),
      fh_p: Number(requestConfig.fh_p ?? 1),
      show_confidence_bands: Boolean(requestConfig.show_confidence_bands),
    };
  }

  if (goal === "cox") {
    return {
      ...base,
      covariates: sortedStrings(requestConfig.covariates || []),
      categorical_covariates: sortedStrings(requestConfig.categorical_covariates || []),
    };
  }

  if (goal === "ml") {
    const compareRun = String(requestConfig.model_type || "") === "compare";
    if (compareRun !== expectsCompare) return null;
    const effectiveModelType = expectsCompare ? "compare" : String(requestConfig.model_type || "");
    const learningRateApplies = expectsCompare || effectiveModelType === "gbs";
    return {
      ...base,
      model_type: effectiveModelType,
      features: sortedStrings(requestConfig.features || []),
      categorical_features: sortedStrings(requestConfig.categorical_features || []),
      n_estimators: Number(requestConfig.n_estimators),
      max_depth: String(requestConfig.max_depth ?? ""),
      learning_rate: learningRateApplies ? Number(requestConfig.learning_rate) : null,
      evaluation_strategy: expectsCompare ? String(requestConfig.evaluation_strategy || "holdout") : null,
      cv_folds: expectsCompare ? Number(requestConfig.cv_folds || 5) : null,
      cv_repeats: expectsCompare ? Number(requestConfig.cv_repeats || 3) : null,
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
    return {
      ...base,
      model_type: effectiveModelType,
      features: sortedStrings(requestConfig.features || []),
      categorical_features: sortedStrings(requestConfig.categorical_features || []),
      hidden_layers: usesHiddenLayers || expectsCompare ? (requestConfig.hidden_layers || []).map(Number) : null,
      dropout: Number(requestConfig.dropout),
      learning_rate: Number(requestConfig.learning_rate),
      epochs: Number(requestConfig.epochs),
      batch_size: Number(requestConfig.batch_size || 64),
      random_seed: Number(requestConfig.random_seed || 42),
      evaluation_strategy: String(requestConfig.evaluation_strategy || "holdout"),
      cv_folds: Number(requestConfig.cv_folds || 5),
      cv_repeats: Number(requestConfig.cv_repeats || 3),
      early_stopping_patience: Number(requestConfig.early_stopping_patience || 10),
      early_stopping_min_delta: Number(requestConfig.early_stopping_min_delta || 0.0001),
      parallel_jobs: Number(requestConfig.parallel_jobs || 1),
      num_time_bins: usesDiscreteTime || expectsCompare ? Number(requestConfig.num_time_bins || 50) : null,
      d_model: usesTransformer ? Number(requestConfig.d_model || 64) : null,
      n_heads: usesTransformer ? Number(requestConfig.n_heads || 4) : null,
      n_layers: usesTransformer ? Number(requestConfig.n_layers || 2) : null,
      latent_dim: usesVae ? Number(requestConfig.latent_dim || 8) : null,
      n_clusters: usesVae ? Number(requestConfig.n_clusters || 3) : null,
    };
  }

  if (goal === "tables") {
    return {
      ...base,
      variables: sortedStrings(requestConfig.variables || []),
      group_column: String(requestConfig.group_column || ""),
    };
  }

  return base;
}

function currentGoalRequestConfig(goal, { expectsCompareOverride = null } = {}) {
  if (!state.dataset) return null;
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
      time_unit_label: refs.timeUnitLabel?.value || "Months",
      max_time: refs.maxTime?.value || "",
      risk_table_points: refs.riskTablePoints?.value,
      logrank_weight: refs.logrankWeight?.value || "logrank",
      fh_p: refs.fhPower?.value || 1,
      show_confidence_bands: Boolean(refs.showConfidenceBands?.checked),
    });
  }

  if (goal === "cox") {
    const { covariates, categoricalCovariates } = currentCoxSelections();
    return normalizedRequestConfig(goal, {
      ...base,
      covariates,
      categorical_covariates: categoricalCovariates,
    });
  }

  if (goal === "ml") {
    const { features, categoricalFeatures } = currentSharedModelSelections();
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
      shap_safe_mode: Boolean(refs.mlShapSafeMode?.checked),
      evaluation_strategy: refs.mlEvaluationStrategy?.value || "holdout",
      cv_folds: refs.mlCvFolds?.value || 5,
      cv_repeats: refs.mlCvRepeats?.value || 3,
    }, { expectsCompare });
  }

  if (goal === "dl") {
    const { features, categoricalFeatures } = currentSharedModelSelections();
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
      batch_size: refs.dlBatchSize?.value || 64,
      random_seed: refs.dlRandomSeed?.value || 42,
      evaluation_strategy: refs.dlEvaluationStrategy?.value || "holdout",
      cv_folds: refs.dlCvFolds?.value || 5,
      cv_repeats: refs.dlCvRepeats?.value || 3,
      early_stopping_patience: refs.dlEarlyStoppingPatience?.value || 10,
      early_stopping_min_delta: refs.dlEarlyStoppingMinDelta?.value || 0.0001,
      parallel_jobs: refs.dlParallelJobs?.value || 1,
      num_time_bins: refs.dlNumTimeBins?.value || 50,
      d_model: refs.dlDModel?.value || 64,
      n_heads: refs.dlHeads?.value || 4,
      n_layers: refs.dlLayers?.value || 2,
      latent_dim: refs.dlLatentDim?.value || 8,
      n_clusters: refs.dlClusters?.value || 3,
    }, { expectsCompare });
  }

  if (goal === "tables") {
    return normalizedRequestConfig(goal, {
      ...base,
      variables: selectedCheckboxValues(refs.cohortVariableChecklist),
      group_column: refs.groupColumn?.value || "",
    });
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
    const currentMl = currentGoalResult("ml");
    const currentDl = currentGoalResult("dl");
    return currentMl || currentDl ? { ml: currentMl, dl: currentDl } : null;
  }
  const payload = {
    km: state.km,
    cox: state.cox,
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
    ml: state.ml,
    dl: state.dl,
    tables: state.cohort,
  }[goal] || null;
}

function goalHasAnyOutput(goal) {
  if (goal === "predictive") {
    return Boolean(state.ml || state.dl);
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
  if (!goal || !GUIDED_GOALS.includes(goal)) return null;
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
    return {
      tone: "warning",
      label: "Needs rerun",
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

function renderGuidedCoxSelectionSummary() {
  const { covariates, categoricalCovariates } = currentCoxSelections();
  if (!covariates.length) {
    return `
      <div class="guided-readiness">
        <strong>No covariates selected</strong>
        <span>Select the variables you want to test before running Cox PH.</span>
      </div>
    `;
  }
  const categoricalSet = new Set(categoricalCovariates);
  const chips = covariates.map((value) => `
    <span class="dataset-preset-chip guided-selection-chip${categoricalSet.has(value) ? " is-categorical" : ""}">
      ${escapeHtml(value)}
      ${categoricalSet.has(value) ? '<span class="guided-selection-chip-tag">cat</span>' : ""}
    </span>
  `).join("");
  return `
    <div class="guided-selection-block">
      <strong>Selected covariates (${covariates.length})</strong>
      <div class="dataset-preset-chips guided-selection-chips">${chips}</div>
    </div>
  `;
}

function renderGuidedCoxPreviewSummary() {
  const { covariates } = currentCoxSelections();
  if (!covariates.length) {
    return `
      <div class="guided-readiness">
        <strong>Cox preview</strong>
        <span>Select at least one covariate to see how many rows remain analyzable.</span>
      </div>
    `;
  }
  if (runtime.coxPreview.status === "loading") {
    return `
      <div class="guided-readiness">
        <strong>Cox preview</strong>
        <span>Checking analyzable rows for the current covariate set.</span>
      </div>
    `;
  }
  if (runtime.coxPreview.status === "blocked" || runtime.coxPreview.status === "error") {
    return `
      <div class="guided-readiness">
        <strong>Cox preview unavailable</strong>
        <span>${escapeHtml(runtime.coxPreview.error || "Preview could not be computed for the current inputs.")}</span>
      </div>
    `;
  }
  const preview = runtime.coxPreview.payload?.preview;
  if (!preview) {
    return `
      <div class="guided-readiness">
        <strong>Cox preview</strong>
        <span>Preview will appear here before you run the model.</span>
      </div>
    `;
  }
  const missingNotes = (preview.missing_by_covariate || [])
    .slice(0, 4)
    .map((item) => `${item.column} (${formatValue(item.missing_rows)})`);
  const stabilityWarnings = (preview.stability_warnings || []).slice(0, 2);
  const epvValue = preview.events_per_parameter == null
    ? "NA"
    : formatValue(preview.events_per_parameter, { scientificLarge: false });
  const noteText = preview.dropped_rows
    ? `Complete-case Cox will drop rows with missing selected covariates. Biggest drops: ${missingNotes.join(", ")}${preview.missing_by_covariate.length > 4 ? " ..." : ""}.`
    : "All rows remain analyzable with the current covariate set.";
  return `
    <div class="guided-selection-block">
      <strong>Cox preview</strong>
      <div class="guided-quick-grid guided-quick-grid-compact">
        <div class="guided-quick-item">
          <strong>Analyzable rows</strong>
          <span>${escapeHtml(`${formatValue(preview.analyzable_rows)} / ${formatValue(preview.outcome_rows)}`)}</span>
        </div>
        <div class="guided-quick-item">
          <strong>Events</strong>
          <span>${escapeHtml(formatValue(preview.events))}</span>
        </div>
        <div class="guided-quick-item">
          <strong>Dropped by missing</strong>
          <span>${escapeHtml(formatValue(preview.dropped_rows))}</span>
        </div>
        <div class="guided-quick-item">
          <strong>Parameters</strong>
          <span>${escapeHtml(formatValue(preview.estimated_parameters))}</span>
        </div>
        <div class="guided-quick-item">
          <strong>EPV</strong>
          <span>${escapeHtml(epvValue)}</span>
        </div>
      </div>
      ${stabilityWarnings.length ? `<div class="event-warning event-warning-warning">${stabilityWarnings.map((warning) => escapeHtml(warning)).join("<br>")}</div>` : ""}
      <span class="guided-inline-note">${escapeHtml(noteText)}</span>
    </div>
  `;
}

function guidedCoxSummaryMount(name) {
  return refs.guidedPanel?.querySelector(`#${name}`) || null;
}

function syncGuidedCoxPanelMounts() {
  if (runtime.uiMode !== "guided" || runtime.guidedGoal !== "cox" || currentGuidedStep() !== 4) {
    return false;
  }
  const selectionMount = guidedCoxSummaryMount("guidedCoxSelectionMount");
  const previewMount = guidedCoxSummaryMount("guidedCoxPreviewMount");
  if (!selectionMount || !previewMount) return false;
  selectionMount.innerHTML = renderGuidedCoxSelectionSummary();
  previewMount.innerHTML = renderGuidedCoxPreviewSummary();
  return true;
}

function guidedRailStatusState() {
  if (!state.dataset) {
    return {
      tone: "idle",
      label: "No result yet",
      title: "Load a cohort to begin.",
      text: "Open a sample cohort or upload a dataset first.",
    };
  }

  const busyGoal = GUIDED_GOALS.find((entry) => isScopeBusy(entry));
  if (busyGoal) {
    return {
      tone: "running",
      label: "Running",
      title: `${goalLabel(busyGoal)} in progress`,
      text: "Wait for the current run to finish before changing shared analysis inputs.",
    };
  }

  if (!endpointIsReady()) {
    return {
      tone: "idle",
      label: "No result yet",
      title: "Complete study design first",
      text: "Choose time, event, and event value to unlock analysis runs.",
    };
  }

  const goal = runtime.guidedGoal;
  if (!goal) {
    return {
      tone: "ready",
      label: "Ready",
      title: "Ready to choose an analysis",
      text: "Outcome is configured. Pick one analysis path when you are ready.",
    };
  }

  return goalResultStatusState(goal, {
    currentLabel: "Ready",
    noResultLabel: "No result yet",
  });
}

function renderGuidedRailStatus() {
  if (!refs.guidedRailStatus || !refs.guidedRailStatusLabel || !refs.guidedRailStatusTitle || !refs.guidedRailStatusText) return;
  const status = guidedRailStatusState();
  const showReviewActions = runtime.uiMode === "guided" && currentGuidedStep() === 5 && Boolean(runtime.guidedGoal);
  const reviewGoal = showReviewActions ? runtime.guidedGoal : null;
  const reviewFamily = reviewGoal === "predictive" ? predictiveFamilyGoal() : reviewGoal;
  const reviewScopeBusy = reviewFamily ? isScopeBusy(reviewFamily) : false;
  const reviewMode = reviewGoal ? guidedResultModeLabel(reviewGoal) : null;
  const mlSingleModelBlocked = reviewFamily === "ml" && refs.mlEvaluationStrategy?.value === "repeated_cv";
  const reviewRunActions = reviewFamily === "ml"
    ? ((runtime.resultPreference?.ml || "single") === "compare"
      ? [
          { label: "Compare all", action: "run-ml-compare", tone: "primary" },
          { label: "Run Analysis", action: "run-ml", tone: "ghost", disabled: mlSingleModelBlocked },
        ]
      : [
          { label: "Run Analysis", action: "run-ml", tone: "primary", disabled: mlSingleModelBlocked },
          { label: "Compare all", action: "run-ml-compare", tone: "ghost" },
        ])
    : reviewFamily === "dl"
      ? ((runtime.resultPreference?.dl || "single") === "compare"
        ? [
            { label: "Compare all", action: "run-dl-compare", tone: "primary" },
            { label: "Run Analysis", action: "run-dl", tone: "ghost" },
          ]
        : [
            { label: "Run Analysis", action: "run-dl", tone: "primary" },
            { label: "Compare all", action: "run-dl-compare", tone: "ghost" },
          ])
      : {
          km: [
            { label: "Run again", action: "run-km", tone: "primary" },
          ],
          cox: [
            { label: "Run again", action: "run-cox", tone: "primary" },
          ],
          tables: [
            { label: "Build again", action: "run-tables", tone: "primary" },
          ],
        }[reviewGoal] || [];
  const predictiveReviewActions = reviewGoal === "predictive"
    ? [
        ...(runtime.workbenchRevealed
          ? [{ label: "Run again", action: "run-predictive-selected", tone: "primary" }]
          : []),
        runtime.workbenchRevealed
          ? { label: "Back to leaderboard", action: "close-predictive-workbench", tone: "ghost" }
          : { label: "Back", action: "previous-step", tone: "ghost" },
        { label: "Change analysis", action: "choose-another-analysis", tone: "ghost" },
      ]
    : null;
  const compactStatus = showReviewActions
    ? {
        ...status,
        title: status.tone === "ready"
          ? `${goalLabel(runtime.guidedGoal)} current${reviewMode ? ` (${reviewMode})` : ""}`
          : status.title,
        text: status.tone === "ready" ? "" : status.text,
      }
    : status;
  refs.guidedRailStatus.className = `guided-rail-status guided-rail-status-${compactStatus.tone}${showReviewActions ? " guided-rail-status-actionable" : ""}`;
  refs.guidedRailStatusLabel.textContent = compactStatus.label;
  refs.guidedRailStatusTitle.textContent = compactStatus.title;
  refs.guidedRailStatusText.textContent = compactStatus.text;
  refs.guidedRailStatusText.classList.toggle("hidden", !compactStatus.text);
  if (refs.guidedRailActions) {
    refs.guidedRailActions.classList.toggle("hidden", !showReviewActions);
    refs.guidedRailActions.innerHTML = showReviewActions
      ? (predictiveReviewActions
        ? `
          ${predictiveReviewActions.map((item) => `
            <button class="button ${item.tone} compact-btn" type="button" data-guided-action="${escapeHtml(item.action)}"${reviewScopeBusy ? " disabled" : ""}>${escapeHtml(item.label)}</button>
          `).join("")}
        `
        : `
          ${reviewRunActions.map((item) => `
            <button class="button ${item.tone} compact-btn" type="button" data-guided-action="${escapeHtml(item.action)}"${(reviewScopeBusy || item.disabled) ? " disabled" : ""}>${escapeHtml(item.label)}</button>
          `).join("")}
          <button class="button ghost compact-btn" type="button" data-guided-action="previous-step">Adjust settings</button>
          <button class="button ghost compact-btn" type="button" data-guided-action="choose-another-analysis">New analysis</button>
        `)
      : "";
  }
}

function guidedGroupingContextActive() {
  const currentTab = activeTabName();
  return (
    currentTab === "km"
    || currentTab === "tables"
    || (runtime.uiMode === "guided" && (runtime.guidedGoal === "km" || runtime.guidedGoal === "tables"))
  );
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

function normalizedGuidedStep(step = runtime.guidedStep) {
  if (!state.dataset) return 1;
  if (!endpointIsReady()) return 2;
  const requested = Number.isFinite(Number(step)) ? Number(step) : 2;
  const bounded = Math.max(2, Math.min(5, requested));
  if (bounded <= 2) return 2;
  if (!runtime.guidedGoal) return 3;
  if (bounded <= 3) return 3;
  if (!currentGoalResult(runtime.guidedGoal) && bounded > 4) return 4;
  return bounded;
}

function currentGuidedStep() {
  return normalizedGuidedStep(runtime.guidedStep);
}

function maxReachableGuidedStep() {
  if (!state.dataset) return 1;
  if (!endpointIsReady()) return 2;
  if (!runtime.guidedGoal) return 3;
  if (!currentGoalResult(runtime.guidedGoal)) return 4;
  return 5;
}

function canNavigateToGuidedStep(step) {
  const requested = Number(step);
  if (!Number.isFinite(requested)) return false;
  if (requested === 1) return Boolean(state.dataset);
  return requested >= 2 && requested <= maxReachableGuidedStep();
}

function setGuidedStep(step, { syncHistory = true, historyMode = "replace", scroll = true } = {}) {
  runtime.guidedStep = normalizedGuidedStep(step);
  if (document.body) {
    document.body.dataset.guidedStep = String(runtime.guidedStep);
    document.body.dataset.guidedGoal = runtime.guidedGoal || "";
  }
  renderGuidedChrome();
  if (runtime.guidedGoal === "cox" && runtime.guidedStep >= 4) scheduleCoxPreview({ delay: 0 });
  if (scroll) refs.guidedShell?.scrollIntoView({ behavior: "smooth", block: "start" });
  if (syncHistory && state.dataset) syncHistoryState(historyMode);
}

function reparentScrollContainers() {
  return [
    refs.configStrip,
    refs.guidedShell,
    refs.guidedPanel,
    refs.guidedConfigMount,
    refs.guidedActivePanelMount,
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

function updateGuidedSurfaceVisibility() {
  const guidedActive = runtime.uiMode === "guided" && Boolean(state.dataset);
  const step = currentGuidedStep();
  const goal = runtime.guidedGoal;
  const guidedPanelName = goal === "predictive" ? "benchmark" : goal;
  const goalNeedsGrouping = goal === "km" || goal === "tables";
  const showOutcomeConfigInRail = guidedActive && step === 2;
  const showOutcomeConfig = guidedActive && step === 2;
  const showGroupingConfig = guidedActive && goalNeedsGrouping && (step === 4 || step === 5);
  const showConfigStrip = !guidedActive || showOutcomeConfig || showGroupingConfig;
  const showGuidedReviewPanel = guidedActive && (step === 4 || step === 5) && GUIDED_GOALS.includes(goal);
  const showGuidedRailPanel = guidedActive && step === 4 && GUIDED_GOALS.includes(goal);
  const preservedUiState = captureReparentUiState();
  let didMove = false;

  if (guidedActive && refs.configStrip) {
    const guidedConfigTarget = showOutcomeConfigInRail ? refs.guidedRailPanelMount : refs.guidedConfigMount;
    if (guidedConfigTarget && refs.configStrip.parentElement !== guidedConfigTarget) {
      guidedConfigTarget.appendChild(refs.configStrip);
      didMove = true;
    }
  } else if (refs.configStripHome && refs.configStrip) {
    if (refs.configStrip.parentElement !== refs.configStripHome.parentElement) {
      refs.configStripHome.after(refs.configStrip);
      didMove = true;
    }
  }

  if (refs.tabPanelsHome) {
    refs.tabPanels.forEach((panel) => {
      const shouldShow = !guidedActive
        ? panel.dataset.panel === activeTabName()
        : showGuidedReviewPanel && panel.dataset.panel === guidedPanelName;
      panel.classList.toggle("guided-visible", shouldShow);
      if (guidedActive && shouldShow && refs.guidedActivePanelMount) {
        if (panel.parentElement !== refs.guidedActivePanelMount) {
          refs.guidedActivePanelMount.appendChild(panel);
          didMove = true;
        }
      } else if (panel.parentElement !== refs.tabPanelsHome) {
        refs.tabPanelsHome.appendChild(panel);
        didMove = true;
      }
    });
  } else {
    refs.tabPanels.forEach((panel) => {
      const shouldShow = !guidedActive
        ? panel.dataset.panel === activeTabName()
        : showGuidedReviewPanel && panel.dataset.panel === guidedPanelName;
      panel.classList.toggle("guided-visible", shouldShow);
    });
  }

  const useMergedPredictiveWorkspace = Boolean(state.dataset) && (
    !guidedActive
    || (guidedPanelName === "benchmark" && showGuidedReviewPanel)
  );
  if (refs.mlWorkspaceCard && refs.benchmarkMlMount && refs.mlPanel) {
    if (useMergedPredictiveWorkspace) {
      if (refs.mlWorkspaceCard.parentElement !== refs.benchmarkMlMount) {
        refs.benchmarkMlMount.appendChild(refs.mlWorkspaceCard);
        didMove = true;
      }
    } else if (refs.mlWorkspaceCard.parentElement !== refs.mlPanel) {
      refs.mlPanel.appendChild(refs.mlWorkspaceCard);
      didMove = true;
    }
  }
  refs.mlWorkspaceCard?.classList.toggle("predictive-workbench-card", useMergedPredictiveWorkspace);
  syncPredictiveWorkbenchCardActions(refs.mlWorkspaceCard, useMergedPredictiveWorkspace);
  if (refs.dlWorkspaceCard && refs.benchmarkDlMount && refs.dlPanel) {
    if (useMergedPredictiveWorkspace) {
      if (refs.dlWorkspaceCard.parentElement !== refs.benchmarkDlMount) {
        refs.benchmarkDlMount.appendChild(refs.dlWorkspaceCard);
        didMove = true;
      }
    } else if (refs.dlWorkspaceCard.parentElement !== refs.dlPanel) {
      refs.dlPanel.appendChild(refs.dlWorkspaceCard);
      didMove = true;
    }
  }
  refs.dlWorkspaceCard?.classList.toggle("predictive-workbench-card", useMergedPredictiveWorkspace);
  syncPredictiveWorkbenchCardActions(refs.dlWorkspaceCard, useMergedPredictiveWorkspace);

  if (refs.guidedPanel) {
    if (showGuidedRailPanel && refs.guidedRailPanelMount) {
      if (refs.guidedPanel.parentElement !== refs.guidedRailPanelMount) {
        refs.guidedRailPanelMount.appendChild(refs.guidedPanel);
        didMove = true;
      }
    } else if (refs.guidedActivePanelMount && refs.guidedPanel.parentElement !== refs.guidedActivePanelMount.parentElement) {
      refs.guidedActivePanelMount.before(refs.guidedPanel);
      didMove = true;
    }
  }

  refs.configStrip?.classList.toggle("hidden", !showConfigStrip);
  refs.outcomeConfigBlock?.classList.toggle("hidden", guidedActive && !showOutcomeConfig);
  refs.groupingConfigBlock?.classList.toggle("hidden", guidedActive && !showGroupingConfig);
  refs.groupingDetails?.classList.toggle("hidden", guidedActive && !showGroupingConfig);
  syncDeriveToggleButton();
  refs.tabStrip?.classList.toggle("hidden", guidedActive);
  refs.datasetPresetBar?.classList.toggle("hidden", guidedActive || !datasetPresetForCurrentDataset());
  renderPredictiveWorkbench();
  if (refs.cutpointPlot) {
    const hasCutpointPlot = refs.cutpointPlot.innerHTML.trim().length > 0;
    const showCutpointPlot = hasCutpointPlot && (!guidedActive || goal === "km");
    refs.cutpointPlot.classList.toggle("hidden", !showCutpointPlot);
  }

  if (showGroupingConfig && refs.groupingDetails) refs.groupingDetails.open = true;
  restoreReparentUiState(preservedUiState);
  if (didMove) scheduleVisiblePlotResize(40);
}

function setUiMode(mode, { syncHistory = true, historyMode = "replace", preserveGuidedState = false } = {}) {
  if (!["guided", "expert"].includes(mode)) return;
  if (mode !== runtime.uiMode && Object.values(runtime.busyScopes || {}).some(Boolean)) {
    showToast("Wait for the current analysis run to finish before switching views.", "warning", 3200);
    return;
  }
  runtime.uiMode = mode;
  document.body.dataset.uiMode = mode;
  const activeTab = activeTabName();
  if (mode === "guided" && GUIDED_GOALS.includes(activeTab) && !preserveGuidedState) {
    runtime.guidedGoal = runtime.guidedGoal || activeTab;
    runtime.guidedStep = normalizedGuidedStep(currentGoalResult(runtime.guidedGoal) ? 5 : 4);
  }
  if (mode === "guided" && state.dataset && !GUIDED_GOALS.includes(activeTab)) {
    runtime.guidedGoal = runtime.guidedGoal || "km";
    activateTab(runtime.guidedGoal, { setGuidedGoal: false, historyMode: "replace", syncHistory: false });
  }
  if (mode === "expert" && ["ml", "dl"].includes(activeTab)) {
    activateTab("benchmark", { setGuidedGoal: false, historyMode: "replace", syncHistory: false });
  }
  refs.guidedModeButton?.classList.toggle("active", mode === "guided");
  refs.guidedModeButton?.setAttribute("aria-selected", mode === "guided" ? "true" : "false");
  refs.expertModeButton?.classList.toggle("active", mode === "expert");
  refs.expertModeButton?.setAttribute("aria-selected", mode === "expert" ? "true" : "false");
  refs.guidedShell?.classList.toggle("hidden", mode !== "guided" || !state.dataset);
  runtime.guidedStep = normalizedGuidedStep(runtime.guidedStep);
  updateGroupingDetailsVisibility(activeTabName(), { force: true });
  renderGuidedChrome();
  if (runtime.uiMode === "guided" && runtime.guidedGoal === "cox") scheduleCoxPreview({ delay: 0 });
  queueVisiblePlotResize();
  if (syncHistory && window.history?.replaceState) syncHistoryState(historyMode);
}

function resizeVisiblePlotsNow() {
  const plots = [
    refs.kmPlot,
    refs.coxPlot,
    refs.coxDiagnosticsPlot,
    refs.cutpointPlot,
    refs.mlImportancePlot,
    refs.mlShapPlot,
    refs.mlComparisonPlot,
    refs.dlImportancePlot,
    refs.dlLossPlot,
    refs.dlComparisonPlot,
    refs.benchmarkComparisonPlot,
  ];
  plots.forEach((plot) => {
    if (!plot?.data?.length) return;
    if (plot.closest(".hidden")) return;
    try {
      Plotly.Plots.resize(plot);
      stabilizePlotShellHeight(plot);
    } catch {
      // Ignore stale nodes during guided view remounts.
    }
  });
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

function scheduleVisiblePlotResize(delay = 80) {
  if (runtime.plotResizeTimer) {
    window.clearTimeout(runtime.plotResizeTimer);
  }
  runtime.plotResizeTimer = window.setTimeout(() => {
    runtime.plotResizeTimer = null;
    queueVisiblePlotResize();
  }, delay);
}

function setGuidedGoal(goal, { activate = true, syncHistory = true, historyMode = "replace" } = {}) {
  runtime.guidedGoal = GUIDED_GOALS.includes(goal) ? goal : null;
  if (activate && runtime.guidedGoal) {
    activateTab(runtime.guidedGoal, { setGuidedGoal: false });
  }
  runtime.guidedStep = normalizedGuidedStep(runtime.guidedGoal ? 4 : 3);
  if (document.body) document.body.dataset.guidedGoal = runtime.guidedGoal || "";
  renderGuidedChrome();
  if (runtime.guidedGoal === "cox") scheduleCoxPreview({ delay: 0 });
  if (syncHistory && state.dataset) syncHistoryState(historyMode);
}

function captureControlSnapshot() {
  if (!state.dataset) return null;
  return {
    timeColumn: refs.timeColumn?.value || "",
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
    coxPreviewKey: runtime.coxPreview?.key || "",
    modelFeatures: selectedCheckboxValues(refs.modelFeatureChecklist),
    modelCategoricals: selectedCheckboxValues(refs.modelCategoricalChecklist),
    cohortVariables: selectedCheckboxValues(refs.cohortVariableChecklist),
    mlModelType: refs.mlModelType?.value || "",
    mlNEstimators: refs.mlNEstimators?.value || "",
    mlLearningRate: refs.mlLearningRate?.value || "",
    mlSkipShap: Boolean(refs.mlSkipShap?.checked),
    mlShapSafeMode: Boolean(refs.mlShapSafeMode?.checked),
    mlEvaluationStrategy: refs.mlEvaluationStrategy?.value || "",
    mlCvFolds: refs.mlCvFolds?.value || "",
    mlCvRepeats: refs.mlCvRepeats?.value || "",
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

function queueHistorySync() {
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
  if (snapshot.timeColumn && columnNames.has(snapshot.timeColumn)) refs.timeColumn.value = snapshot.timeColumn;
  if (refs.showAllEventColumns) refs.showAllEventColumns.checked = Boolean(snapshot.showAllEventColumns);
  renderEventColumnOptions({
    preferred: snapshot.eventColumn && columnNames.has(snapshot.eventColumn) ? snapshot.eventColumn : null,
    silent: true,
  });
  setSelectValueIfPresent(refs.eventPositiveValue, snapshot.eventPositiveValue);
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
  updateDlModelControlVisibility();
  setCheckedValues(refs.covariateChecklist, snapshot.covariates || []);
  setCheckedValues(refs.categoricalChecklist, snapshot.categoricals || []);
  setCheckedValues(refs.modelFeatureChecklist, snapshot.modelFeatures || []);
  setCheckedValues(refs.modelCategoricalChecklist, snapshot.modelCategoricals || []);
  setCheckedValues(refs.dlModelFeatureChecklist, snapshot.modelFeatures || []);
  setCheckedValues(refs.dlModelCategoricalChecklist, snapshot.modelCategoricals || []);
  syncModelFeatureMirrors(refs.modelFeatureChecklist);
  syncModelCategoricalMirrors(refs.modelCategoricalChecklist);
  setCheckedValues(refs.cohortVariableChecklist, snapshot.cohortVariables || []);
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
  const restoredUiMode = historyState?.uiMode || runtime.uiMode;
  const restoredGuidedGoal = GUIDED_GOALS.includes(historyState?.guidedGoal) ? historyState.guidedGoal : null;
  const restoredGuidedStep = normalizedGuidedStep(historyState?.guidedStep || (restoredGuidedGoal ? 4 : 2));
  const restoredPredictiveFamily = normalizedPredictiveFamily(historyState?.predictiveFamily);
  if (!historyState || historyState.view === "home") {
    runtime.predictiveFamily = restoredPredictiveFamily;
    if (restoredUiMode === "guided") {
      runtime.guidedGoal = restoredGuidedGoal;
      runtime.guidedStep = restoredGuidedStep;
    }
    setUiMode(restoredUiMode, { syncHistory: false, preserveGuidedState: restoredUiMode === "guided" });
    goHome({ syncHistory: false });
    return;
  }
  if (historyState.view !== "workspace" || !historyState.datasetId) {
    runtime.predictiveFamily = restoredPredictiveFamily;
    if (restoredUiMode === "guided") {
      runtime.guidedGoal = restoredGuidedGoal;
      runtime.guidedStep = restoredGuidedStep;
    }
    setUiMode(restoredUiMode, { syncHistory: false, preserveGuidedState: restoredUiMode === "guided" });
    goHome({ syncHistory: false });
    return;
  }

  runtime.historySyncPaused = true;
  try {
    runtime.predictiveFamily = restoredPredictiveFamily;
    runtime.workbenchRevealed = Boolean(historyState?.workbenchRevealed);
    if (restoredUiMode === "guided") {
      runtime.guidedGoal = restoredGuidedGoal;
      runtime.guidedStep = restoredGuidedStep;
    }
    setUiMode(restoredUiMode, { syncHistory: false, preserveGuidedState: restoredUiMode === "guided" });
    if (!state.dataset || state.dataset.dataset_id !== historyState.datasetId) {
      const payload = await fetchJSON(`/api/dataset/${historyState.datasetId}`);
      if (restoreToken !== runtime.historyRestoreToken) return;
      updateAfterDataset(payload);
    } else {
      showWorkspace();
    }
    if (restoreToken !== runtime.historyRestoreToken) return;
    runtime.workbenchRevealed = Boolean(historyState?.workbenchRevealed);
    applyControlSnapshot(historyState.controls || null);
    runtime.guidedGoal = restoredGuidedGoal;
    runtime.guidedStep = restoredGuidedStep;
    activateTab(historyState.tab || restoredGuidedGoal || "km", { setGuidedGoal: false });
    renderGuidedChrome();
  } catch {
    goHome({ syncHistory: false });
  } finally {
    runtime.historySyncPaused = false;
  }
}

function syncDeriveToggleButton() {
  if (!refs.deriveToggle || !refs.derivePanel) return;
  const guidedGroupingActive = runtime.uiMode === "guided"
    && runtime.guidedGoal === "km"
    && !refs.groupingConfigBlock?.classList.contains("hidden");
  if (guidedGroupingActive) {
    refs.deriveToggle.classList.add("hidden");
    refs.derivePanel.classList.remove("hidden");
    refs.deriveToggle.setAttribute("aria-expanded", "true");
    refs.deriveButton?.classList.add("hidden");
    return;
  }
  refs.deriveToggle.classList.remove("hidden");
  refs.deriveButton?.classList.remove("hidden");
  const derivePanelOpen = !refs.derivePanel.classList.contains("hidden");
  refs.deriveToggle.textContent = derivePanelOpen ? "Close" : "Derive Group";
  refs.deriveToggle.classList.toggle("primary", !derivePanelOpen);
  refs.deriveToggle.classList.toggle("ghost", derivePanelOpen);
  refs.deriveToggle.setAttribute("aria-expanded", derivePanelOpen ? "true" : "false");
}

function syncDeriveControlsState() {
  const activeGroup = String(refs.groupColumn?.value || "").trim();
  const deriveLocked = Boolean(activeGroup);
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
    refs.deriveButton.disabled = deriveLocked;
    refs.deriveButton.setAttribute("aria-disabled", String(deriveLocked));
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
    refs.deriveStatus.textContent = deriveLocked
      ? `Derived-group settings are locked while Group by uses ${activeGroup}. Set Group by to Overall only if you want Create or these parameters to affect the next grouping analysis. Run again only reuses the current Group by value.`
      : "";
  }
}

function guidedKmHasPendingDerivedGroup() {
  return runtime.uiMode === "guided"
    && runtime.guidedGoal === "km"
    && !String(refs.groupColumn?.value || "")
    && Boolean(runtime.deriveDraftTouched);
}

async function runGuidedKaplanMeier() {
  if (guidedKmHasPendingDerivedGroup()) {
    await deriveGroup({ autoApplyOverride: true, refreshKmOverride: false, toastMode: "silent" });
  }
  return runKaplanMeier();
}

function setButtonLoading(button, isLoading) {
  button.classList.toggle("is-loading", isLoading);
  button.disabled = isLoading;
}

function setPanelResultMode(panel, mode = "idle") {
  if (!panel) return;
  panel.dataset.resultMode = mode;
}

function runScopeForGoal(goal) {
  if (goal === "predictive") return isScopeBusy("predictive") ? "predictive" : predictiveFamilyGoal();
  if (["km", "cox", "ml", "dl", "tables"].includes(goal)) return goal;
  return null;
}

function isScopeBusy(scope) {
  return Boolean(scope && runtime.busyScopes?.[scope]);
}

function buttonsForScope(scope) {
  if (scope === "predictive") return [refs.runPredictiveCompareAllButton];
  if (scope === "ml") return [refs.runMlButton, refs.runCompareButton, refs.runCompareInlineButton];
  if (scope === "dl") return [refs.runDlButton, refs.runDlCompareButton, refs.runDlCompareInlineButton];
  if (scope === "km") return [refs.runKmButton, refs.runSignatureSearchButton];
  if (scope === "cox") {
    return [
      refs.runCoxButton,
      refs.selectAllCoxCovariatesButton,
      refs.clearCoxCovariatesButton,
      refs.selectAllCoxCategoricalsButton,
      refs.clearCoxCategoricalsButton,
    ];
  }
  if (scope === "tables") return [refs.runCohortTableButton];
  return [];
}

function setChecklistDisabled(container, isDisabled) {
  if (!container) return;
  container.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.disabled = Boolean(isDisabled);
  });
}

function syncSharedFeatureControlsBusy() {
  const isBusy = isScopeBusy("ml") || isScopeBusy("dl");
  [
    refs.reviewMlFeaturesButton,
    refs.reviewDlFeaturesButton,
    refs.selectAllModelFeaturesButton,
    refs.clearModelFeaturesButton,
    refs.selectAllDlModelFeaturesButton,
    refs.clearDlModelFeaturesButton,
  ].forEach((button) => {
    if (button) button.disabled = isBusy;
  });
  setChecklistDisabled(refs.modelFeatureChecklist, isBusy);
  setChecklistDisabled(refs.modelCategoricalChecklist, isBusy);
  setChecklistDisabled(refs.dlModelFeatureChecklist, isBusy);
  setChecklistDisabled(refs.dlModelCategoricalChecklist, isBusy);
}

function setScopeBusy(scope, isBusy, activeButton = null) {
  if (!scope) return;
  runtime.busyScopes[scope] = Boolean(isBusy);
  const scopeButtons = buttonsForScope(scope);
  if (activeButton && !scopeButtons.includes(activeButton)) {
    setButtonLoading(activeButton, isBusy);
  }
  scopeButtons.forEach((button) => {
    if (!button) return;
    if (button === activeButton) {
      setButtonLoading(button, isBusy);
      return;
    }
    button.disabled = Boolean(isBusy);
    button.classList.toggle("is-loading", false);
  });
  syncSharedFeatureControlsBusy();
  if (scope === "ml") updateMlEvaluationControls();
  if (scope === "dl") updateDlEvaluationControls();
  syncAnalysisRunButtonAvailability();
  renderGuidedChrome();
  if (scope === "predictive" || scope === "ml" || scope === "dl") {
    renderBenchmarkBoard();
  }
}

function showToast(message, type = "error", duration = 5000) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${escapeHtml(message)}</span><button class="toast-close">&times;</button>`;
  container.appendChild(toast);
  const dismiss = () => {
    toast.classList.add("toast-exit");
    toast.addEventListener("animationend", () => toast.remove());
  };
  toast.querySelector(".toast-close").addEventListener("click", dismiss);
  if (duration > 0) setTimeout(dismiss, duration);
}

function showError(message) { showToast(message, "error"); }

async function shutdownServer() {
  return shellHelpers.shutdownServer({
    runtime,
    refs,
    setButtonLoading,
    setRuntimeBanner,
    fetchJSON,
    renderServerStoppedState,
  });
}

function renderSelect(select, options, { includeBlank = false, blankLabel = "None", selected = null } = {}) {
  select.innerHTML = "";
  if (includeBlank) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = blankLabel;
    select.appendChild(option);
  }
  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (selected !== null && value === selected) option.selected = true;
    select.appendChild(option);
  });
}

function inferDefault(columnNames, suggestions, fallbackIndex = 0) {
  if (suggestions.length > 0 && columnNames.includes(suggestions[0])) return suggestions[0];
  return columnNames[fallbackIndex] || "";
}

function getColumnMeta(columnName) {
  return state.dataset?.columns.find((column) => column.name === columnName) || null;
}

function hasConfidentEventSuggestion() {
  const binaryColumns = binaryCandidateColumns();
  const eventLikeBinary = binaryColumns.filter(
    (column) => isEventLikeColumnName(column) && !looksLikeBaselineStatusColumn(column),
  );
  if (eventLikeBinary.length) return true;
  const suggestionSet = new Set(state.dataset?.suggestions?.event_columns || []);
  const keywordBinary = binaryColumns.filter(
    (column) => suggestionSet.has(column) && !looksLikeBaselineStatusColumn(column),
  );
  return keywordBinary.length > 0;
}

function currentGroupColumnWarning() {
  const groupColumn = refs.groupColumn?.value || "";
  if (!state.dataset || !groupColumn) return null;
  if (groupColumn === refs.timeColumn?.value || groupColumn === refs.eventColumn?.value) {
    return {
      tone: "error",
      message: `"${groupColumn}" is part of the survival endpoint. Use a separate categorical grouping variable for Kaplan-Meier and grouped tables.`,
    };
  }

  const meta = getColumnMeta(groupColumn);
  if (!meta) return null;
  if (meta.kind === "numeric" && Number(meta.n_unique || 0) > 8) {
    return {
      tone: "error",
      message: `"${groupColumn}" is a high-cardinality numeric column. Create a grouped version first, then use that new column for Kaplan-Meier or grouped tables.`,
    };
  }
  if (Number(meta.n_unique || 0) > 20) {
    return {
      tone: "error",
      message: `"${groupColumn}" has too many unique values for meaningful grouped survival curves. Use a lower-cardinality grouping column instead.`,
    };
  }
  return null;
}

function normalizeColumnLabel(columnName) {
  return String(columnName || "").trim().toLowerCase();
}

function isEventLikeColumnName(columnName) {
  const normalized = normalizeColumnLabel(columnName);
  if (!normalized) return false;
  if (normalized === "event" || normalized === "status") return true;
  return [
    /event/,
    /death/,
    /mort/,
    /vital_status/,
    /survival_status/,
    /outcome_status/,
    /relapse/,
    /recur/,
    /progress/,
    /failure/,
    /censor/,
  ].some((pattern) => pattern.test(normalized));
}

function looksLikeBaselineStatusColumn(columnName) {
  const normalized = normalizeColumnLabel(columnName);
  if (!normalized) return false;
  return [
    /egfr/,
    /kras/,
    /braf/,
    /alk/,
    /ros1/,
    /erbb2/,
    /mutation/,
    /mutated/,
    /wildtype/,
    /sex/,
    /gender/,
    /stage/,
    /grade/,
    /treat/,
    /therapy/,
    /drug/,
    /smok/,
    /histolog/,
    /subtype/,
    /cluster/,
    /group/,
    /arm/,
    /cohort/,
    /horth/,
  ].some((pattern) => pattern.test(normalized));
}

function datasetColumnNames() {
  return state.dataset?.columns?.map((column) => column.name) || [];
}

function recommendedTimeColumns() {
  const names = new Set(datasetColumnNames());
  return (state.dataset?.suggestions?.time_columns || []).filter((column) => names.has(column));
}

function allowedTimeColumns() {
  const recommended = recommendedTimeColumns();
  if (recommended.length) return recommended;
  const names = new Set(datasetColumnNames());
  return (state.dataset?.numeric_columns || []).filter((column) => names.has(column));
}

function identicalOutcomeColumnMessage() {
  const timeColumn = refs.timeColumn?.value || "";
  const eventColumn = refs.eventColumn?.value || "";
  if (!state.dataset || !timeColumn || !eventColumn || timeColumn !== eventColumn) return null;
  return "The survival time column and event column must be different.";
}

function currentTimeColumnWarning() {
  const timeColumn = refs.timeColumn?.value || "";
  if (!state.dataset || !timeColumn) return null;
  const matchingOutcomeWarning = identicalOutcomeColumnMessage();
  if (matchingOutcomeWarning) {
    return {
      tone: "error",
      message: matchingOutcomeWarning,
    };
  }

  const numericSet = new Set(state.dataset.numeric_columns || []);
  if (!numericSet.has(timeColumn)) {
    return {
      tone: "error",
      message: `"${timeColumn}" is not numeric. Choose a true follow-up time column such as os_months, pfs_months, days, or follow-up time.`,
    };
  }

  const recommended = recommendedTimeColumns();
  if (!recommended.length || recommended.includes(timeColumn)) return null;
  return {
    tone: "error",
    message: `"${timeColumn}" does not look like a survival follow-up time column. Choose one of the likely time columns instead: ${recommended.slice(0, 3).join(", ")}.`,
  };
}

function updateTimeColumnGuidance() {
  if (!state.dataset) return;

  const recommended = recommendedTimeColumns();
  const numericColumns = allowedTimeColumns();
  const compactGuidedCopy = runtime.uiMode === "guided" && currentGuidedStep() === 2;
  if (refs.timeColumnHelp) {
    if (recommended.length) {
      refs.timeColumnHelp.textContent = compactGuidedCopy
        ? "Showing likely time columns only. Genes and baseline covariates are hidden here."
        : "Showing likely follow-up time columns only. Genes and baseline covariates are hidden here.";
    } else if (numericColumns.length) {
      refs.timeColumnHelp.textContent = compactGuidedCopy
        ? "No clear time-style name was found. Showing numeric columns only."
        : "No clear time-style name was found. Showing numeric columns only, so confirm the follow-up field carefully.";
    } else {
      refs.timeColumnHelp.textContent = "No numeric time candidates were detected in this dataset.";
    }
  }

  const warning = currentTimeColumnWarning();
  if (!refs.timeColumnWarning) return;
  if (!warning) {
    refs.timeColumnWarning.textContent = "";
    refs.timeColumnWarning.className = "event-warning hidden";
    return;
  }
  refs.timeColumnWarning.textContent = warning.message;
  refs.timeColumnWarning.className = `event-warning event-warning-${warning.tone}`;
}

function renderTimeColumnOptions({ preferred = null, silent = true } = {}) {
  if (!state.dataset) return;
  const options = allowedTimeColumns();
  const recommended = recommendedTimeColumns();
  const currentValue = preferred ?? refs.timeColumn?.value ?? "";
  const nextValue = options.includes(currentValue)
    ? currentValue
    : recommended.length
      ? inferDefault(options, recommended, 0)
      : "";
  renderSelect(refs.timeColumn, options, {
    includeBlank: !recommended.length || !nextValue,
    blankLabel: "Select time column",
    selected: nextValue || "",
  });
  updateTimeColumnGuidance();

  if (!silent && currentValue && nextValue && currentValue !== nextValue) {
    showToast(
      `Time column reset to ${nextValue}. Use a true follow-up time field, not a gene or baseline covariate.`,
      "warning",
      3600,
    );
  } else if (!silent && currentValue && !nextValue) {
    showToast(
      "Time column cleared. Choose the follow-up time field again before running an analysis.",
      "warning",
      3600,
    );
  }
}

function normalizeEventToken(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value ? "1" : "0";
  const text = String(value).trim().toLowerCase();
  if (!text) return null;
  if (/^-?\d+(?:\.0+)?$/.test(text)) return String(Number(text));
  return text;
}

function eventTokenCandidates(value) {
  const token = normalizeEventToken(value);
  if (!token) return [];
  const variants = [];
  const add = (candidate) => {
    if (candidate && !variants.includes(candidate)) variants.push(candidate);
  };
  add(token);
  add(token.replace(/[^a-z0-9]+/g, ""));
  token.split(/[^a-z0-9]+/).forEach((part) => add(part));
  return variants;
}

function eventValueFamily(value) {
  const candidates = eventTokenCandidates(value);
  if (!candidates.length) return null;
  if (candidates.some((candidate) => EVENT_FALSE_TOKENS.has(candidate))) return "censor";
  if (candidates.some((candidate) => EVENT_TRUE_TOKENS.has(candidate))) return "event";
  return null;
}

function hasRecognizableEventCoding(values) {
  const normalizedValues = values.filter((value) => value !== null && value !== undefined);
  if (!normalizedValues.length) return false;
  const uniqueTokens = [...new Set(normalizedValues.map((value) => normalizeEventToken(value)).filter(Boolean))];
  const numericTokens = uniqueTokens.filter((token) => /^-?\d+(?:\.\d+)?$/.test(token));
  if (uniqueTokens.length === 2 && numericTokens.length === 2) {
    const numericPair = numericTokens.map((token) => Number(token)).sort((a, b) => a - b);
    if (
      (numericPair[0] === 0 && numericPair[1] === 1)
      || (numericPair[0] === 1 && numericPair[1] === 2)
    ) {
      return true;
    }
  }
  const families = new Set(normalizedValues.map((value) => eventValueFamily(value)).filter(Boolean));
  return families.has("event") && families.has("censor");
}

function inferEventPositiveSelection(eventColumn, values, previousValue = "") {
  const normalized = values.map((value) => ({
    raw: String(value),
    token: normalizeEventToken(value),
    candidates: eventTokenCandidates(value),
  })).filter((entry) => entry.token !== null);
  const available = new Set(normalized.map((entry) => entry.raw));
  if (previousValue && available.has(String(previousValue))) {
    return { value: String(previousValue), warning: null };
  }

  const truthy = normalized.filter((entry) => entry.candidates.some((candidate) => EVENT_TRUE_TOKENS.has(candidate)));
  const falsy = normalized.filter((entry) => entry.candidates.some((candidate) => EVENT_FALSE_TOKENS.has(candidate)));
  const pickTruthy = truthy.find((entry) => entry.token === "1")
    || truthy.find((entry) => entry.token === "event")
    || truthy.find((entry) => entry.token === "dead")
    || truthy[0];

  if (pickTruthy && falsy.length) {
    return { value: pickTruthy.raw, warning: null };
  }
  if (pickTruthy && normalized.length === 1) {
    return { value: pickTruthy.raw, warning: null };
  }

  const uniqueTokens = [...new Set(normalized.map((entry) => entry.token))];
  const numericTokens = uniqueTokens.filter((token) => /^-?\d+(?:\.\d+)?$/.test(token));
  if (uniqueTokens.length === 2 && numericTokens.length === 2) {
    const numericPair = numericTokens.map((token) => Number(token)).sort((a, b) => a - b);
    if (numericPair[0] === 1 && numericPair[1] === 2) {
      return {
        value: "",
        warning: `"${eventColumn}" looks like TCGA-style 1/2 coding (${values.map((value) => String(value)).join(", ")}). Confirm whether 1 means event and 2 means censoring.`,
      };
    }
    return {
      value: "",
      warning: `"${eventColumn}" uses non-standard binary values (${values.map((value) => String(value)).join(", ")}). Choose which value means event.`,
    };
  }

  return {
    value: "",
    warning: `SurvStudio could not safely guess which value in "${eventColumn}" means event. Choose it explicitly.`,
  };
}

function binaryCandidateColumns() {
  const binarySet = new Set(state.dataset?.binary_candidate_columns || []);
  return datasetColumnNames().filter((column) => binarySet.has(column));
}

function recommendedEventColumns() {
  const binaryColumns = binaryCandidateColumns();
  const eventLikeBinary = binaryColumns.filter(
    (column) => isEventLikeColumnName(column) && !looksLikeBaselineStatusColumn(column),
  );
  if (eventLikeBinary.length) return eventLikeBinary;
  const suggestionSet = new Set(state.dataset?.suggestions?.event_columns || []);
  const keywordBinary = binaryColumns.filter(
    (column) => suggestionSet.has(column) && !looksLikeBaselineStatusColumn(column),
  );
  if (keywordBinary.length) return keywordBinary;
  return binaryColumns;
}

function currentEventColumnWarning() {
  const eventColumn = refs.eventColumn?.value || "";
  if (!state.dataset || !eventColumn) return null;
  const matchingOutcomeWarning = identicalOutcomeColumnMessage();
  if (matchingOutcomeWarning) {
    return {
      tone: "error",
      blocking: true,
      message: matchingOutcomeWarning,
    };
  }

  const binarySet = new Set(state.dataset.binary_candidate_columns || []);
  const suggestionSet = new Set(state.dataset?.suggestions?.event_columns || []);
  const looksBaselineLike = looksLikeBaselineStatusColumn(eventColumn);
  const previewValues = getColumnMeta(eventColumn)?.unique_preview?.filter((value) => value !== null) ?? [];
  const looksLikeRecognizableEventCoding = hasRecognizableEventCoding(previewValues);
  if (!binarySet.has(eventColumn)) {
    return {
      tone: "error",
      blocking: true,
      message: `"${eventColumn}" is not a binary event column. Choose a 0/1-style event column or recode it.`,
    };
  }

  if ((isEventLikeColumnName(eventColumn) || suggestionSet.has(eventColumn)) && !looksBaselineLike) return null;

  if (!refs.showAllEventColumns?.checked) {
    return {
      tone: "warning",
      blocking: true,
      message: `"${eventColumn}" is not a standard event column name. Turn on Show all columns only if you intend to use it as the event indicator.`,
    };
  }

  if (looksBaselineLike) {
    return {
      tone: "warning",
      blocking: true,
      message: `"${eventColumn}" looks more like a baseline characteristic than an event indicator. Use it as Group by or as a model feature instead.`,
    };
  }

  if (looksLikeRecognizableEventCoding) {
    return {
      tone: "warning",
      blocking: false,
      message: `"${eventColumn}" uses non-standard event coding. Confirm it is the event indicator before continuing.`,
    };
  }

  return {
    tone: "warning",
    blocking: true,
    message: `"${eventColumn}" does not look like a survival event column. Use the suggested event field or a true event indicator instead.`,
  };
}

function updateEventValueGuidance(message = null) {
  if (!refs.eventValueWarning) return;
  if (!message) {
    refs.eventValueWarning.textContent = "";
    refs.eventValueWarning.className = "event-warning hidden";
    return;
  }
  refs.eventValueWarning.textContent = message;
  refs.eventValueWarning.className = "event-warning event-warning-warning";
}

function updateEventColumnGuidance() {
  if (!state.dataset) return;

  const binaryColumns = binaryCandidateColumns();
  const recommendedColumns = recommendedEventColumns();
  const compactGuidedCopy = runtime.uiMode === "guided" && currentGuidedStep() === 2;
  if (refs.eventColumnHelp) {
    if (refs.showAllEventColumns?.checked) {
      refs.eventColumnHelp.textContent = compactGuidedCopy
        ? "Showing all columns. Use only a true binary event column here."
        : "Showing all columns. Use only a true binary event indicator here.";
    } else if (recommendedColumns.length && recommendedColumns.length < datasetColumnNames().length) {
      refs.eventColumnHelp.textContent = compactGuidedCopy
        ? "Showing likely event columns only. Turn on Show all columns only if yours is missing."
        : "Showing event-like binary columns only. Turn on Show all columns if your event indicator uses a non-standard name.";
    } else if (binaryColumns.length) {
      refs.eventColumnHelp.textContent = compactGuidedCopy
        ? "Showing binary columns that could be the event field."
        : "No clear event-style name was found. Showing binary candidate columns only.";
    } else {
      refs.eventColumnHelp.textContent = compactGuidedCopy
        ? "No clear event column was found. Turn on Show all columns only if you already know it."
        : "No binary event candidates were detected. Turn on Show all columns only if you already know which column is the event indicator.";
    }
  }

  const warning = currentEventColumnWarning();
  if (!refs.eventColumnWarning) return;
  if (!warning) {
    refs.eventColumnWarning.textContent = "";
    refs.eventColumnWarning.className = "event-warning hidden";
    return;
  }
  refs.eventColumnWarning.textContent = warning.message;
  refs.eventColumnWarning.className = `event-warning event-warning-${warning.tone}`;
}

function renderEventColumnOptions({ preferred = null, silent = true } = {}) {
  if (!state.dataset) return;
  const allColumns = datasetColumnNames();
  const options = refs.showAllEventColumns?.checked
    ? allColumns
    : (() => {
        const recommended = recommendedEventColumns();
        if (recommended.length) return recommended;
        const binaryColumns = binaryCandidateColumns();
        return binaryColumns.length ? binaryColumns : allColumns;
      })();

  const currentValue = preferred ?? refs.eventColumn?.value ?? "";
  const confident = hasConfidentEventSuggestion();
  const nextValue = options.includes(currentValue)
    ? currentValue
    : confident
      ? inferDefault(options, recommendedEventColumns(), 0)
      : "";
  renderSelect(refs.eventColumn, options, {
    includeBlank: !confident || !nextValue,
    blankLabel: "Select event column",
    selected: nextValue || "",
  });
  updateEventPositiveOptions();
  updateEventColumnGuidance();

  if (!silent && currentValue && nextValue && currentValue !== nextValue) {
    showToast(
      `Event column reset to ${nextValue}. Turn on Show all columns to select non-standard event fields.`,
      "warning",
      3600,
    );
  } else if (!silent && currentValue && !nextValue) {
    showToast(
      "Event column cleared. Choose the event indicator again before running an analysis.",
      "warning",
      3600,
    );
  }
}

function updateEventPositiveOptions() {
  const eventColumn = refs.eventColumn.value;
  const meta = getColumnMeta(eventColumn);
  const values = meta?.unique_preview?.filter((value) => value !== null) ?? [];
  const preserveSelection = refs.eventColumn?.dataset?.lastColumn === eventColumn;
  const previousValue = preserveSelection ? refs.eventPositiveValue.value : "";
  const eventColumnWarning = currentEventColumnWarning();
  if (refs.eventColumn) refs.eventColumn.dataset.lastColumn = eventColumn;
  refs.eventPositiveValue.innerHTML = "";
  if (values.length === 0) {
    const option = document.createElement("option");
    option.value = "1";
    option.textContent = "1";
    refs.eventPositiveValue.appendChild(option);
    updateEventValueGuidance(null);
    return;
  }
  const inferred = inferEventPositiveSelection(eventColumn, values, previousValue);
  if (!inferred.value) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose event value";
    placeholder.selected = true;
    refs.eventPositiveValue.appendChild(placeholder);
  }
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    if (String(value) === inferred.value) option.selected = true;
    refs.eventPositiveValue.appendChild(option);
  });
  updateEventValueGuidance(eventColumnWarning?.blocking ? null : inferred.warning);
  updateEventColumnGuidance();
}

function renderChecklist(container, values, selected = []) {
  if (!container) return;
  container.innerHTML = "";
  values.forEach((value) => {
    const label = document.createElement("label");
    label.className = "check-item";
    label.dataset.filterValue = String(value).toLowerCase();
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = selected.includes(value);
    const span = document.createElement("span");
    span.textContent = value;
    label.append(input, span);
    container.appendChild(label);
  });
  applyChecklistSearch(container);
}

function searchControlForChecklist(container) {
  if (!container) return null;
  if (container === refs.covariateChecklist) return refs.covariateSearchInput;
  if (container === refs.categoricalChecklist) return refs.categoricalSearchInput;
  if (container === refs.cohortVariableChecklist) return refs.cohortVariableSearchInput;
  return null;
}

function applyChecklistSearch(container) {
  if (!container) return;
  const searchControl = searchControlForChecklist(container);
  if (!searchControl) return;

  const query = String(searchControl.value || "").trim().toLowerCase();
  let visibleCount = 0;
  container.querySelectorAll(".check-item").forEach((item) => {
    const match = !query || String(item.dataset.filterValue || "").includes(query);
    item.classList.toggle("hidden-by-filter", !match);
    if (match) visibleCount += 1;
  });

  let emptyState = container.querySelector(".checklist-filter-empty");
  if (!emptyState) {
    emptyState = document.createElement("div");
    emptyState.className = "checklist-filter-empty hidden";
    container.appendChild(emptyState);
  }
  const showEmpty = Boolean(query) && visibleCount === 0;
  emptyState.textContent = showEmpty ? `No matches for "${searchControl.value}".` : "";
  emptyState.classList.toggle("hidden", !showEmpty);
}

function selectedCheckboxValues(container) {
  if (!container) return [];
  return [...container.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
}

function allCheckboxValues(container, { visibleOnly = false } = {}) {
  if (!container) return [];
  return [...container.querySelectorAll('input[type="checkbox"]')]
    .filter((input) => !visibleOnly || !input.closest(".check-item")?.classList.contains("hidden-by-filter"))
    .map((input) => input.value);
}

function resetChecklistSearch(container) {
  const searchControl = searchControlForChecklist(container);
  if (!searchControl) return;
  searchControl.value = "";
  applyChecklistSearch(container);
}

function normalizeDerivedColumnProvenance(provenance = {}) {
  return Object.fromEntries(
    Object.entries(provenance || {}).map(([columnName, meta]) => {
      const normalizedMeta = meta && typeof meta === "object" ? meta : {};
      return [columnName, {
        ...normalizedMeta,
        outcomeInformed: Boolean(normalizedMeta.outcomeInformed ?? normalizedMeta.outcome_informed),
        recipe: normalizedMeta.recipe || {},
      }];
    }),
  );
}

function availableCoxStageVariables() {
  const available = new Set(allCheckboxValues(refs.covariateChecklist));
  return COX_STAGE_VARIABLE_PREFERENCE.filter((value) => available.has(value));
}

function renderCoxCovariateWarning({ kept = null, removed = [] } = {}) {
  if (!refs.coxCovariateWarning) return;
  const availableStageVars = availableCoxStageVariables();
  if (!kept || availableStageVars.length < 2) {
    refs.coxCovariateWarning.textContent = "";
    refs.coxCovariateWarning.className = "event-warning event-warning-warning hidden";
    return;
  }
  const alternatives = availableStageVars.filter((value) => value !== kept);
  refs.coxCovariateWarning.textContent = removed.length
    ? `Cox can use only one stage variable at a time. Keeping ${kept}; cleared ${removed.join(", ")}.`
    : `Cox can use only one stage variable at a time. Using ${kept}. Leave ${alternatives.join(" or ")} unchecked to avoid redundant stage encoding.`;
  refs.coxCovariateWarning.className = "event-warning event-warning-warning";
}

function shouldAutoCategorizeCoxCovariate(columnName) {
  const meta = getColumnMeta(columnName);
  if (!meta) return false;
  return meta.kind === "categorical" || meta.kind === "binary";
}

function syncCoxCovariateSelection({ preferredValue = null, notify = false, autoCategoricalValues = [] } = {}) {
  if (!refs.covariateChecklist) return { kept: null, removed: [], changed: false };
  const selected = selectedCheckboxValues(refs.covariateChecklist);
  const selectedStageVars = COX_STAGE_VARIABLE_PREFERENCE.filter((value) => selected.includes(value));
  let kept = selectedStageVars[0] || null;
  let removed = [];
  let normalizedCovariates = [...selected];

  if (selectedStageVars.length > 1) {
    kept = selectedStageVars.includes(preferredValue)
      ? preferredValue
      : COX_STAGE_VARIABLE_PREFERENCE.find((value) => selectedStageVars.includes(value)) || selectedStageVars[0];
    removed = selectedStageVars.filter((value) => value !== kept);
    normalizedCovariates = selected.filter((value) => !removed.includes(value));
    setCheckedValues(refs.covariateChecklist, normalizedCovariates);
  }

  if (refs.categoricalChecklist) {
    const normalizedCategoricals = selectedCheckboxValues(refs.categoricalChecklist).filter((value) => normalizedCovariates.includes(value));
    [...new Set(autoCategoricalValues)].forEach((value) => {
      if (
        value
        && normalizedCovariates.includes(value)
        && shouldAutoCategorizeCoxCovariate(value)
        && !normalizedCategoricals.includes(value)
      ) {
        normalizedCategoricals.push(value);
      }
    });
    setCheckedValues(refs.categoricalChecklist, normalizedCategoricals);
  }
  renderCoxCovariateWarning({ kept, removed });
  if (notify && removed.length) {
    showToast(`Cox can use only one stage variable. Keeping ${kept} and clearing ${removed.join(", ")}.`, "warning", 3600);
  }
  return { kept, removed, changed: removed.length > 0 };
}

function formatValue(value, options = {}) {
  if (value === null || value === undefined || value === "") return "NA";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "NA";
    const { scientificLarge = true } = options;
    const absValue = Math.abs(value);
    if (absValue > 0 && absValue < 0.1) return value.toFixed(4).replace(/\.?0+$/, "");
    if ((scientificLarge && absValue >= 1000) || (absValue > 0 && absValue < 0.001)) return value.toExponential(2);
    return value.toFixed(3).replace(/\.?0+$/, "");
  }
  return String(value);
}

function formatPercent(numerator, denominator) {
  if (!denominator) return "0%";
  return `${((100 * numerator) / denominator).toFixed(1).replace(/\.0$/, "")}%`;
}

function statusLabel(status) {
  return { robust: "Robust", review: "Needs review", caution: "Caution" }[status] || "Review";
}

function escapeListItem(value) { return `<li>${escapeHtml(value)}</li>`; }

function renderInsightBoard(container, summary, emptyMessage) {
  if (!container) return;
  if (!summary) {
    container.innerHTML = `<div class="empty-state">${escapeHtml(emptyMessage)}</div>`;
    return;
  }
  const metrics = summary.metrics || [];
  const strengths = summary.strengths || [];
  const cautions = summary.cautions || [];
  const nextSteps = summary.next_steps || [];
  const tone = summary.status || "review";
  const metricsMarkup = metrics.length
    ? `<div class="insight-metrics">${metrics.map((m) => `<div class="metric-pill"><span>${escapeHtml(m.label || "")}</span><strong>${escapeHtml(formatValue(m.value))}</strong></div>`).join("")}</div>`
    : "";
  const sections = [
    strengths.length ? `<div class="insight-section"><h4>What was checked</h4><ul>${strengths.map(escapeListItem).join("")}</ul></div>` : "",
    cautions.length ? `<div class="insight-section"><h4>What to watch</h4><ul>${cautions.map(escapeListItem).join("")}</ul></div>` : "",
    nextSteps.length ? `<div class="insight-section"><h4>Next step</h4><ul>${nextSteps.map(escapeListItem).join("")}</ul></div>` : "",
  ].filter(Boolean).join("");
  container.innerHTML = `
    <article class="insight-card tone-${escapeHtml(tone)}">
      <div class="insight-header"><span class="insight-badge">${escapeHtml(statusLabel(tone))}</span><p>${escapeHtml(summary.headline || "Interpretation unavailable.")}</p></div>
      ${metricsMarkup}
      <div class="insight-sections">${sections}</div>
    </article>`;
}

function summaryHasCaution(summary, phrase) {
  if (!summary || !phrase) return false;
  const needle = String(phrase).trim().toLowerCase();
  if (!needle) return false;
  return (summary.cautions || []).some((item) => String(item || "").toLowerCase().includes(needle));
}

function deriveGroupCountLabel(group) {
  if (group === "High") return "High risk";
  if (group === "Low") return "Low risk";
  return group;
}

function humanizeDeriveMethod(method) {
  const labelMap = {
    median_split: "Median split",
    tertile_split: "Tertile split",
    quartile_split: "Quartile split",
    percentile_split: "Percentile split",
    extreme_split: "Extreme split",
    optimal_cutpoint: "Optimal cutpoint",
  };
  return labelMap[method] || humanizeHeader(method || "NA");
}

function humanizePValueLabel(label) {
  if (label === "selection_adjusted_p_value") return "Selection-adjusted p-value";
  if (label === "raw_p_value") return "Raw p-value";
  return "p-value";
}

function renderDerivedGroupSummary(derivedColumn, summary) {
  const counts = summary?.counts || [];
  const assignmentRule = summary?.assignment_rule || null;
  const pValueLabel = humanizePValueLabel(summary?.p_value_label);
  const percentileSpec = summary?.cutoff_spec || null;
  const thresholds = Array.isArray(summary?.cutoffs)
    ? summary.cutoffs.map((value) => formatValue(value)).join(", ")
    : "";
  const currentGroup = String(refs.groupColumn?.value || "");
  const currentGroupLabel = currentGroup || "overall only";
  const groupingNote = currentGroup === String(derivedColumn || "")
    ? `Current grouping now uses ${derivedColumn}. ML and DL feature selections did not change automatically.`
    : `Derived column ${derivedColumn} is available. Current grouping remains ${currentGroupLabel}. ML and DL feature selections did not change automatically.`;
  const summaryCell = (label, value, extraClass = "") => `
      <div class="${extraClass}">
        <strong>${escapeHtml(label)}</strong>
        <span class="summary-value" title="${escapeHtml(String(value ?? "NA"))}">${escapeHtml(String(value ?? "NA"))}</span>
    </div>`;
  if (refs.cutpointPlot && summary?.method !== "optimal_cutpoint") {
    refs.cutpointPlot.innerHTML = "";
    refs.cutpointPlot.classList.add("hidden");
  }
  refs.deriveSummary.classList.remove("hidden");
  refs.deriveSummary.innerHTML = `
    <div class="count-strip">
      ${counts.map((item) => `<div class="count-pill"><span>${escapeHtml(deriveGroupCountLabel(item.group))}</span><strong>${escapeHtml(formatValue(item.n))}</strong></div>`).join("")}
    </div>
    ${summary?.method === "optimal_cutpoint" ? '<div class="note-box">High/Low indicate risk direction, not whether the source value itself is numerically high or low.</div>' : ""}
    <div class="note-box">${escapeHtml(groupingNote)}</div>
    ${summary?.method === "optimal_cutpoint" ? '<div class="note-box">This optimal cutpoint used outcome information. Use it for grouping or visualization, not as an ML/DL training feature.</div>' : ""}
    ${summary?.method === "extreme_split" ? '<div class="note-box">Middle-range rows are excluded from grouped analyses for extreme split.</div>' : ""}
    <div class="signature-summary-grid">
      ${summaryCell("Derived column", derivedColumn || "NA")}
      ${summaryCell("Method", humanizeDeriveMethod(summary?.method || "NA"))}
      ${percentileSpec ? summaryCell("Percentile(s)", percentileSpec) : ""}
      ${thresholds ? summaryCell(Array.isArray(summary?.cutoffs) && summary.cutoffs.length > 1 ? "Thresholds" : "Threshold", thresholds) : ""}
      ${!percentileSpec && !thresholds ? summaryCell("Cutoff", formatValue(summary?.cutoff)) : ""}
      ${summary?.p_value != null ? summaryCell(pValueLabel, formatValue(summary.p_value), "pvalue-card") : ""}
      ${summaryCell("Groups", formatValue(summary?.n_groups || counts.length || "NA"))}
      ${summary?.method === "optimal_cutpoint" ? summaryCell("Min group fraction", formatValue(summary?.min_group_fraction)) : ""}
      ${summary?.method === "optimal_cutpoint" ? summaryCell("Permutation iterations", formatValue(summary?.permutation_iterations)) : ""}
      ${summary?.method === "optimal_cutpoint" ? summaryCell("Seed", formatValue(summary?.random_seed)) : ""}
      ${assignmentRule ? summaryCell("Assignment rule", assignmentRule, "wide-card") : ""}
    </div>`;
}

function humanizeHeader(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\bc index\b/gi, "C-Index")
    .replace(/\bp value\b/gi, "P-Value")
    .replace(/\bhr\b/gi, "HR")
    .replace(/\bci\b/gi, "CI")
    .replace(/\b(\w)/g, (c) => c.toUpperCase())
    .trim();
}

function syncPredictiveWorkbenchCardActions(card, workbenchActive) {
  if (!card) return;
  const head = card.querySelector(":scope > .card-head");
  const primaryRow = head?.querySelector(":scope > .button-row.compact");
  if (!head || !primaryRow) return;

  let secondaryRow = head.querySelector(":scope > .predictive-workbench-secondary-actions");
  if (workbenchActive) {
    primaryRow.classList.add("predictive-workbench-primary-actions");
    if (!secondaryRow) {
      secondaryRow = document.createElement("div");
      secondaryRow.className = "button-row compact predictive-workbench-secondary-actions";
      primaryRow.insertAdjacentElement("afterend", secondaryRow);
    }
    while (primaryRow.children.length > 1) {
      secondaryRow.appendChild(primaryRow.children[1]);
    }
    return;
  }

  primaryRow.classList.remove("predictive-workbench-primary-actions");
  if (secondaryRow) {
    while (secondaryRow.firstChild) {
      primaryRow.appendChild(secondaryRow.firstChild);
    }
    secondaryRow.remove();
  }
}

const COHORT_TABLE_EMPTY_STATE_HTML = '<div class="empty-state">Check variables on the left, then click <strong>Build Table</strong>.</div>';

function renderTable(shell, rows, columns = null) {
  if (!rows || rows.length === 0) {
    shell.innerHTML = '<div class="empty-state">No rows returned.</div>';
    return;
  }
  const visibleColumns = columns || Object.keys(rows[0]);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const headerRow = document.createElement("tr");
  visibleColumns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = humanizeHeader(column);
    th.title = column;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    visibleColumns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = formatValue(row[column]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  shell.innerHTML = "";
  shell.appendChild(table);
}

function clearCohortTableOutput({ rerenderChrome = true, syncHistory = true } = {}) {
  state.cohort = null;
  if (refs.cohortTableShell) refs.cohortTableShell.innerHTML = COHORT_TABLE_EMPTY_STATE_HTML;
  if (refs.downloadCohortTableButton) refs.downloadCohortTableButton.disabled = true;
  renderSharedFeatureSummary();
  syncDownloadButtonAvailability();
  if (rerenderChrome) renderGuidedChrome();
  if (syncHistory) queueHistorySync();
}

function downloadCsv(filename, rows, columns = null) {
  return downloadHelpers.downloadCsv({ filename, rows, columns, showToast });
}

function downloadText(filename, text, mimeType = "text/plain;charset=utf-8;") {
  return downloadHelpers.downloadText({ filename, text, mimeType });
}

function slugifyDownloadToken(value, fallback = "na") {
  return downloadHelpers.slugifyDownloadToken(value, fallback);
}

function currentDatasetSlug() {
  return downloadHelpers.currentDatasetSlug(state);
}

function currentOutcomeSlug() {
  return downloadHelpers.currentOutcomeSlug(refs);
}

function currentGroupSlug() {
  return downloadHelpers.currentGroupSlug(refs);
}

function buildDownloadFilename(stem, ext, { includeGroup = false, template = null } = {}) {
  return downloadHelpers.buildDownloadFilename({
    state,
    refs,
    stem,
    ext,
    includeGroup,
    template,
  });
}

function triggerBlobDownload(filename, blob, fallbackMimeType = "") {
  return downloadHelpers.triggerBlobDownload(filename, blob, fallbackMimeType);
}

async function downloadServerTable(filename, payload, fallbackMimeType = "text/plain;charset=utf-8;") {
  return downloadHelpers.downloadServerTable({
    filename,
    payload,
    fallbackMimeType,
    apiUrl,
    showToast,
  });
}

function buildMarkdownTable(rows, { caption = "", notes = [] } = {}) {
  return downloadHelpers.buildMarkdownTable(rows, { caption, notes, formatValue });
}

function currentMlJournalTemplate() {
  return refs.mlJournalTemplate?.value || "default";
}

function currentDlJournalTemplate() {
  return refs.dlJournalTemplate?.value || "default";
}

function setMlManuscriptDownloadsEnabled(enabled) {
  refs.downloadMlManuscriptCsvButton.disabled = !enabled;
  refs.downloadMlManuscriptMarkdownButton.disabled = !enabled;
  refs.downloadMlManuscriptLatexButton.disabled = !enabled;
  refs.downloadMlManuscriptDocxButton.disabled = !enabled;
}

function setDlManuscriptDownloadsEnabled(enabled) {
  refs.downloadDlManuscriptCsvButton.disabled = !enabled;
  refs.downloadDlManuscriptMarkdownButton.disabled = !enabled;
  refs.downloadDlManuscriptLatexButton.disabled = !enabled;
  refs.downloadDlManuscriptDocxButton.disabled = !enabled;
}

function manuscriptExportPayload(manuscript, format, template, fallbackCaption, resultPayload = null) {
  const analysis = resultPayload?.analysis || {};
  const requestConfig = resultPayload?.request_config || null;
  return {
    rows: manuscript?.model_performance_table || [],
    format,
    style: "journal",
    template,
    caption: manuscript?.caption || fallbackCaption,
    notes: manuscript?.table_notes || [],
    provenance: {
      request_config: requestConfig,
      analysis: {
        evaluation_mode: analysis?.evaluation_mode,
        cv_folds: analysis?.cv_folds,
        cv_repeats: analysis?.cv_repeats,
        shared_training_seed: analysis?.shared_training_seed,
        shared_split_seed: analysis?.shared_split_seed,
        shared_monitor_seed: analysis?.shared_monitor_seed,
      },
    },
  };
}

function downloadPlotImage(plotEl, filename, format) {
  return downloadHelpers.downloadPlotImage({ plotEl, filename, format });
}

function requireCurrentResultForExport(goal, { payload = null } = {}) {
  const scope = runScopeForGoal(goal);
  if (scope && isScopeBusy(scope)) {
    showToast("Wait for the current run to finish before exporting this result.", "warning", 3200);
    return false;
  }
  if (goal === "tables") {
    const tableState = currentCohortTableOutputState();
    if (!tableState.hasOutput || !tableState.isCurrent || !payload) {
      showToast("Visible settings no longer match the current cohort table. Rebuild the table before exporting.", "warning", 3600);
      return false;
    }
    return true;
  }
  if (!payload || !currentGoalResult(goal)) {
    showToast("Visible settings no longer match the current result. Run again before exporting.", "warning", 3600);
    return false;
  }
  return true;
}

function isReadonlyPlot(filename) {
  return downloadHelpers.isReadonlyPlot(filename);
}

function plotLayoutConfig(layout, filename) {
  return downloadHelpers.plotLayoutConfig(layout, filename);
}

function plotConfig(filename) {
  const isStaticReadonlyPlot = isReadonlyPlot(filename);
  return {
    responsive: true,
    displaylogo: false,
    displayModeBar: true,
    scrollZoom: !isStaticReadonlyPlot,
    doubleClick: isStaticReadonlyPlot ? false : "reset+autosize",
    modeBarButtonsToRemove: isStaticReadonlyPlot
      ? [
          "zoom2d",
          "pan2d",
          "select2d",
          "lasso2d",
          "zoomIn2d",
          "zoomOut2d",
          "autoScale2d",
          "resetScale2d",
          "hoverClosestCartesian",
          "hoverCompareCartesian",
          "toggleSpikelines",
        ]
      : ["select2d", "lasso2d"],
    toImageButtonOptions: {
      format: "svg",
      filename: buildDownloadFilename(filename, "svg").replace(/\.svg$/, ""),
      height: 900,
      width: 1400,
      scale: 1,
    },
  };
}

function stabilizePlotShellHeight(plotEl) {
  if (!plotEl?._fullLayout) return;
  const height = Number(plotEl._fullLayout.height);
  if (!Number.isFinite(height) || height <= 0) return;
  plotEl.style.height = `${Math.ceil(height)}px`;
}

function stabilizeCoxPlotResetAxes(plotEl) {
  if (!plotEl?.on || !plotEl?._fullLayout) return;
  const fullLayout = plotEl._fullLayout;
  const xRange = Array.isArray(fullLayout.xaxis?.range) ? [...fullLayout.xaxis.range] : null;
  const yRange = Array.isArray(fullLayout.yaxis?.range) ? [...fullLayout.yaxis.range] : null;
  const height = Number(fullLayout.height);
  if (!xRange || !yRange || !Number.isFinite(height)) return;

  plotEl.__stableResetAxesState = {
    applying: false,
    height,
    xRange,
    yRange,
  };
  if (typeof plotEl.removeAllListeners === "function") plotEl.removeAllListeners("plotly_relayout");
  plotEl.on("plotly_relayout", (eventData) => {
    const resetRequested = Boolean(eventData?.["xaxis.autorange"] || eventData?.["yaxis.autorange"]);
    const stableState = plotEl.__stableResetAxesState;
    if (!resetRequested || !stableState || stableState.applying) return;

    stableState.applying = true;
    Promise.resolve(
      Plotly.relayout(plotEl, {
        height: stableState.height,
        "xaxis.autorange": false,
        "xaxis.range": stableState.xRange.slice(),
        "yaxis.autorange": false,
        "yaxis.range": stableState.yRange.slice(),
      })
    ).finally(() => {
      if (plotEl.__stableResetAxesState) plotEl.__stableResetAxesState.applying = false;
    });
  });
}

function updateMlEvaluationControls() {
  const isRepeatedCv = refs.mlEvaluationStrategy?.value === "repeated_cv";
  refs.mlCvFoldsWrap?.classList.toggle("hidden", !isRepeatedCv);
  refs.mlCvRepeatsWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (refs.mlCvFolds) refs.mlCvFolds.disabled = !isRepeatedCv;
  if (refs.mlCvRepeats) refs.mlCvRepeats.disabled = !isRepeatedCv;
  syncAnalysisRunButtonAvailability();
  renderGuidedChrome();
}

function updateDlEvaluationControls() {
  const isRepeatedCv = refs.dlEvaluationStrategy?.value === "repeated_cv";
  refs.dlCvFoldsWrap?.classList.toggle("hidden", !isRepeatedCv);
  refs.dlCvRepeatsWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (refs.dlCvFolds) refs.dlCvFolds.disabled = !isRepeatedCv;
  if (refs.dlCvRepeats) refs.dlCvRepeats.disabled = !isRepeatedCv;
  if (refs.dlParallelJobs) refs.dlParallelJobs.disabled = !isRepeatedCv;
  syncAnalysisRunButtonAvailability();
}

function updateDlModelControlVisibility() {
  const modelType = refs.dlModelType?.value || "deepsurv";
  const usesDiscreteTime = modelType === "deephit" || modelType === "mtlr";
  const usesTransformer = modelType === "transformer";
  const usesVae = modelType === "vae";
  const usesHiddenLayers = !usesTransformer;
  const usesMiniBatchTraining = usesDiscreteTime;

  refs.dlHiddenLayers?.closest(".toolbar-field")?.classList.toggle("hidden", !usesHiddenLayers);
  refs.dlNumTimeBinsWrap?.classList.toggle("hidden", !usesDiscreteTime);
  refs.dlDModelWrap?.classList.toggle("hidden", !usesTransformer);
  refs.dlHeadsWrap?.classList.toggle("hidden", !usesTransformer);
  refs.dlLayersWrap?.classList.toggle("hidden", !usesTransformer);
  refs.dlLatentDimWrap?.classList.toggle("hidden", !usesVae);
  refs.dlClustersWrap?.classList.toggle("hidden", !usesVae);
  if (refs.dlBatchSize) {
    refs.dlBatchSize.disabled = !usesMiniBatchTraining;
    refs.dlBatchSize.title = usesMiniBatchTraining
      ? ""
      : "Batch size applies only to DeepHit and Neural MTLR. This architecture uses full-batch optimization.";
    refs.dlBatchSize.closest(".toolbar-field")?.classList.toggle("is-disabled", !usesMiniBatchTraining);
  }
  if (refs.dlBatchSizeHint) {
    refs.dlBatchSizeHint.textContent = usesMiniBatchTraining
      ? "Applies to the current discrete-time trainer."
      : "Ignored for this architecture because training is full-batch.";
  }
}

function purgePlot(el) {
  if (el && el.__stableResetAxesState) delete el.__stableResetAxesState;
  if (el) el.style.height = "";
  if (el && el.data) { try { Plotly.purge(el); } catch { /* ignore */ } }
}

function setPlotShellState(el, state) {
  if (!el) return;
  el.dataset.plotState = state || "";
}

function clearPlotShell(el, emptyHtml, { state = "message" } = {}) {
  purgePlot(el);
  if (el) el.innerHTML = emptyHtml || '';
  setPlotShellState(el, state);
}

function setShimmer(shell) {
  shell.innerHTML = '<div class="shimmer"><div class="shimmer-bar"></div><div class="shimmer-bar short"></div><div class="shimmer-bar"></div></div>';
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
    .filter((column) => ["categorical", "binary"].includes(column.kind) || column.n_unique <= AUTO_CATEGORICAL_UNIQUE_THRESHOLD)
    .map((column) => column.name);
}

function refreshVariableSelections() {
  if (!state.dataset) return;
  const availableCovariates = modelFeatureCandidateColumns();
  const previousCovariates = selectedCheckboxValues(refs.covariateChecklist).filter((v) => availableCovariates.includes(v));
  const previousCategoricals = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => availableCovariates.includes(v));
  const previousModelFeatures = selectedCheckboxValues(refs.modelFeatureChecklist).filter((v) => availableCovariates.includes(v));
  const previousModelCategoricals = selectedCheckboxValues(refs.modelCategoricalChecklist).filter((v) => availableCovariates.includes(v));
  const previousTableVars = selectedCheckboxValues(refs.cohortVariableChecklist).filter((v) => availableCovariates.includes(v));
  const defaultCategoricals = state.dataset.columns
    .filter((c) => ["categorical", "binary"].includes(c.kind) || c.n_unique <= AUTO_CATEGORICAL_UNIQUE_THRESHOLD)
    .map((c) => c.name)
    .filter((name) => availableCovariates.includes(name));
  const defaultModelFeatures = availableCovariates.slice(0, DEFAULT_MODEL_FEATURE_SELECTION_LIMIT);
  renderChecklist(refs.covariateChecklist, availableCovariates, previousCovariates.length ? previousCovariates : availableCovariates.slice(0, 4));
  renderChecklist(refs.categoricalChecklist, availableCovariates, previousCategoricals.length ? previousCategoricals : defaultCategoricals);
  renderChecklist(refs.modelFeatureChecklist, availableCovariates, previousModelFeatures.length ? previousModelFeatures : defaultModelFeatures);
  renderChecklist(refs.modelCategoricalChecklist, availableCovariates, previousModelCategoricals.length ? previousModelCategoricals : defaultCategoricals);
  renderChecklist(refs.dlModelFeatureChecklist, availableCovariates, previousModelFeatures.length ? previousModelFeatures : defaultModelFeatures);
  renderChecklist(refs.dlModelCategoricalChecklist, availableCovariates, previousModelCategoricals.length ? previousModelCategoricals : defaultCategoricals);
  renderChecklist(refs.cohortVariableChecklist, availableCovariates, previousTableVars.length ? previousTableVars : availableCovariates.slice(0, 6));
  const numericOptions = state.dataset.numeric_columns.filter((c) => !isSurvivalOutcomeLikeColumn(c));
  renderSelect(refs.deriveSource, numericOptions, { selected: numericOptions.includes(refs.deriveSource.value) ? refs.deriveSource.value : numericOptions[0] || null });
  renderSharedFeatureSummary();
  syncGuidedCoxPanelMounts();
}

function setSharedModelFeatureSelection(nextFeatures = [], { clearCategoricals = false } = {}) {
  const availableFeatures = modelFeatureCandidateColumns();
  const normalizedFeatures = nextFeatures.filter((value) => availableFeatures.includes(value));
  const autoCategoricalCandidates = new Set(sharedModelCategoricalCandidates());
  const preservedCategoricals = clearCategoricals
    ? []
    : selectedCheckboxValues(refs.modelCategoricalChecklist).filter((value) => normalizedFeatures.includes(value));
  normalizedFeatures.forEach((value) => {
    if (autoCategoricalCandidates.has(value) && !preservedCategoricals.includes(value)) {
      preservedCategoricals.push(value);
    }
  });

  setCheckedValues(refs.modelFeatureChecklist, normalizedFeatures);
  setCheckedValues(refs.dlModelFeatureChecklist, normalizedFeatures);
  setCheckedValues(refs.modelCategoricalChecklist, preservedCategoricals);
  setCheckedValues(refs.dlModelCategoricalChecklist, preservedCategoricals);
  renderSharedFeatureSummary();
  queueHistorySync();
}

function syncChecklistSelections(sourceContainer, targetContainer) {
  if (!sourceContainer || !targetContainer) return;
  setCheckedValues(targetContainer, selectedCheckboxValues(sourceContainer));
}

function syncModelFeatureMirrors(sourceContainer = refs.modelFeatureChecklist) {
  const counterpart = sourceContainer === refs.dlModelFeatureChecklist
    ? refs.modelFeatureChecklist
    : refs.dlModelFeatureChecklist;
  syncChecklistSelections(sourceContainer, counterpart);
  const featureValues = selectedCheckboxValues(sourceContainer);
  const autoCategoricalCandidates = new Set(sharedModelCategoricalCandidates());
  const normalizedCategoricals = selectedCheckboxValues(refs.modelCategoricalChecklist)
    .filter((value) => featureValues.includes(value));
  featureValues.forEach((value) => {
    if (autoCategoricalCandidates.has(value) && !normalizedCategoricals.includes(value)) {
      normalizedCategoricals.push(value);
    }
  });
  setCheckedValues(refs.modelCategoricalChecklist, normalizedCategoricals);
  setCheckedValues(refs.dlModelCategoricalChecklist, normalizedCategoricals);
}

function syncModelCategoricalMirrors(sourceContainer = refs.modelCategoricalChecklist) {
  const counterpart = sourceContainer === refs.dlModelCategoricalChecklist
    ? refs.modelCategoricalChecklist
    : refs.dlModelCategoricalChecklist;
  syncChecklistSelections(sourceContainer, counterpart);
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

function setDatasetPresetButtonState(mode = null) {
  if (!refs.applyBasicPresetButton || !refs.applyModelPresetButton) return;
  refs.applyBasicPresetButton.className = `button ${mode === "basic" ? "primary" : "ghost"} compact-btn`;
  refs.applyModelPresetButton.className = `button ${mode === "models" ? "primary" : "ghost"} compact-btn`;
}

function renderDatasetPresetStatus(title, text, chips = []) {
  if (refs.datasetPresetStatusTitle) refs.datasetPresetStatusTitle.textContent = title;
  if (refs.datasetPresetStatusText) refs.datasetPresetStatusText.textContent = text;
  renderChipList(refs.datasetPresetChips, chips);
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
    transformer: "Survival Transformer",
    vae: "Survival VAE",
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
  if ((runtime.resultPreference?.[goal] || "single") !== "single") return null;
  const payload = currentGoalResult(goal);
  if (!payload) return null;
  const requestConfig = payload.request_config || payload.analysis?.request_config || null;
  if (!requestConfig) return null;
  const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
  if (selectedModel.family !== goal) return null;
  const requestModelType = String(requestConfig.model_type || "").toLowerCase();
  return requestModelType === selectedModel.key ? payload : null;
}

function guidedPredictiveModelPickerMarkup({ disabled = false } = {}) {
  const selectedKey = currentPredictiveModelKey();
  const optionGroups = [
    {
      label: "Classical ML",
      options: [
        ["rsf", "Random Survival Forest"],
        ["gbs", "Gradient Boosted Survival"],
        ["lasso_cox", "LASSO-Cox"],
      ],
    },
    {
      label: "Deep Learning",
      options: [
        ["deepsurv", "DeepSurv"],
        ["deephit", "DeepHit"],
        ["mtlr", "Neural MTLR"],
        ["transformer", "Survival Transformer"],
        ["vae", "Survival VAE"],
      ],
    },
  ];
  return `
    <label class="toolbar-field compact-inline predictive-model-picker guided-predictive-model-picker">
      <span>Selected model</span>
      <select data-guided-predictive-model-selector${disabled ? " disabled" : ""}>
        ${optionGroups.map((group) => `
          <optgroup label="${escapeHtml(group.label)}">
            ${group.options.map(([value, label]) => `<option value="${escapeHtml(value)}"${value === selectedKey ? " selected" : ""}>${escapeHtml(label)}</option>`).join("")}
          </optgroup>
        `).join("")}
      </select>
    </label>
  `;
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

function panelModeForPayload(payload) {
  if (!payload) return "idle";
  return payloadRepresentsCompareRun(payload) ? "compare" : "single";
}

function restorePredictiveFamilyAfterFailedCompare(goal, previousPayload) {
  const panel = goal === "ml" ? refs.mlPanel : refs.dlPanel;
  const previousWasCompare = payloadRepresentsCompareRun(previousPayload);
  if (goal === "ml") {
    state.ml = previousWasCompare ? null : (previousPayload || null);
  } else {
    state.dl = previousWasCompare ? null : (previousPayload || null);
  }
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

function benchmarkParamsPayload(goal) {
  return currentCompareGoalPayload(goal) || compareGoalPayload(goal) || null;
}

function benchmarkParamsSummary(goal, modelLabel) {
  const payload = benchmarkParamsPayload(goal);
  const requestConfig = payload?.request_config || payload?.analysis?.request_config || {};
  if (!requestConfig || !Object.keys(requestConfig).length) {
    return `${modelLabel}: no saved compare-run settings are available for this row yet.`;
  }

  const features = Array.isArray(requestConfig.features) ? requestConfig.features : [];
  const categoricals = Array.isArray(requestConfig.categorical_features) ? requestConfig.categorical_features : [];
  const evaluation = String(requestConfig.evaluation_strategy || "holdout") === "repeated_cv"
    ? `${formatValue(requestConfig.cv_repeats || 3)}x${formatValue(requestConfig.cv_folds || 5)} repeated CV`
    : "Deterministic Holdout";

  const parts = [
    `${modelLabel} params`,
    `shared_features=${formatValue(features.length)}`,
    `categoricals=${formatValue(categoricals.length)}`,
    `eval=${evaluation}`,
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

function showBenchmarkParams(goal, modelLabel) {
  showToast(benchmarkParamsSummary(goal, modelLabel), "info", 7600);
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

function formatMaxTimeChip(maxTimeValue) {
  return maxTimeValue ? `Max time: ${maxTimeValue}` : "Max time: Auto";
}

function renderContextCards({
  hasDataset,
  timeLabel,
  eventLabel,
  eventValue,
  groupLabel,
  coxFeatures,
  coxCategoricals,
  modelFeatures,
  modelCategoricals,
  tableVariables,
}) {
  if (refs.groupingSummaryText) {
    refs.groupingSummaryText.textContent = !hasDataset
      ? "Used mainly for Kaplan-Meier and grouped tables."
      : runtime.uiMode === "guided"
        ? `Current Group by: ${groupLabel}. Open this only if you want subgroup curves or grouped tables.`
        : `Current Group by: ${groupLabel}. These settings mainly affect Kaplan-Meier and grouped tables. Cox, ML, and DL use the outcome definition plus their own feature selections.`;
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

  if (refs.kmDependencyText) {
    refs.kmDependencyText.textContent = !hasDataset
      ? "Kaplan-Meier uses the Study Design outcome definition and the current Group by setting."
      : "Kaplan-Meier uses the Study Design outcome definition, the current Group by field, and the current display settings.";
    renderChipList(refs.kmDependencyChips, hasDataset ? [
      formatOutcomeChip(timeLabel, eventLabel, eventValue),
      `Group: ${groupLabel}`,
      `Time unit: ${refs.timeUnitLabel?.value || "Months"}`,
      formatMaxTimeChip(refs.maxTime?.value || ""),
      `CI: ${refs.confidenceLevel?.selectedOptions?.[0]?.textContent || "95%"}`,
    ] : []);
  }

  if (refs.coxDependencyText) {
    refs.coxDependencyText.textContent = !hasDataset
      ? "Cox uses the Study Design outcome definition and the covariates selected in this tab. The reported C-index is apparent on the analyzable cohort, and PH diagnostics shown here use scaled Schoenfeld residual screening with LOWESS trend lines rather than a full cox.zph test."
      : "Cox uses the Study Design outcome definition and the covariates selected in this tab. Group by does not change the model unless you add that column as a covariate. The reported C-index is apparent on the analyzable cohort, and PH diagnostics shown here use scaled Schoenfeld residual screening with LOWESS trend lines rather than a full cox.zph test.";
    renderChipList(refs.coxDependencyChips, hasDataset ? [
      formatOutcomeChip(timeLabel, eventLabel, eventValue),
      formatGroupChip(groupLabel),
      `Covariates: ${coxFeatures.length}`,
      `Categorical: ${coxCategoricals.length}`,
    ] : []);
  }

  if (refs.tableDependencyText) {
    refs.tableDependencyText.textContent = !hasDataset
      ? "The cohort table uses the selected variables in this tab and applies Group by only when grouping is set."
      : "The cohort table uses the selected variables in this tab and applies Group by only when grouping is set. When Group by is active, Overall summarizes the grouped non-missing subset.";
    renderChipList(refs.tableDependencyChips, hasDataset ? [
      `Variables: ${tableVariables.length}`,
      `Group: ${groupLabel}`,
    ] : []);
  }
  if (refs.tableOutputStatusText) {
    const tableState = currentCohortTableOutputState();
    if (!hasDataset || !tableState.hasOutput || tableState.isCurrent) {
      refs.tableOutputStatusText.textContent = "";
      refs.tableOutputStatusText.classList.add("hidden");
    } else {
      refs.tableOutputStatusText.textContent = `Current output still reflects the last built table: Variables ${tableState.outputVariables.length}, Group ${tableState.outputGroupLabel}. Click Rebuild Table to apply the current settings.`;
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
  if (refs.downloadKmPngButton) refs.downloadKmPngButton.disabled = !currentKm;
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.disabled = !currentKm;
  refs.downloadSignatureButton.disabled = !currentSignature || !(currentSignature.results_table?.length);
  refs.downloadCoxResultsButton.disabled = !currentCox;
  refs.downloadCoxDiagnosticsButton.disabled = !currentCox;
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.disabled = !currentCox;
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.disabled = !currentCox;
  refs.downloadCohortTableButton.disabled = !currentTable.hasOutput || !currentTable.isCurrent;
  refs.downloadMlComparisonButton.disabled = !currentMl || !(currentMl.analysis?.comparison_table?.length);
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = !currentMl || !refs.mlComparisonPlot?.data?.length;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = !currentMl || !refs.mlComparisonPlot?.data?.length;
  setMlManuscriptDownloadsEnabled(Boolean(currentMl?.analysis?.manuscript_tables?.model_performance_table?.length));
  refs.downloadDlComparisonButton.disabled = !currentDl || !(currentDl.analysis?.comparison_table?.length);
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = !currentDl || !refs.dlComparisonPlot?.data?.length;
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = !currentDl || !refs.dlComparisonPlot?.data?.length;
  setDlManuscriptDownloadsEnabled(Boolean(currentDl?.analysis?.manuscript_tables?.model_performance_table?.length));
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
  const signatureFeatureMessage = "Select at least one covariate to search for signatures.";
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
    !endpointReady || !hasCoxCovariates || isScopeBusy("km"),
    !endpointReady ? readyMessage : (!hasCoxCovariates ? signatureFeatureMessage : ""),
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
  const coxFeatures = hasDataset ? selectedCheckboxValues(refs.covariateChecklist) : [];
  const coxCategoricals = hasDataset ? selectedCheckboxValues(refs.categoricalChecklist).filter((value) => coxFeatures.includes(value)) : [];
  const features = hasDataset ? selectedCheckboxValues(refs.modelFeatureChecklist) : [];
  const categoricals = hasDataset ? selectedCheckboxValues(refs.modelCategoricalChecklist).filter((value) => features.includes(value)) : [];
  const tableVariables = hasDataset ? selectedCheckboxValues(refs.cohortVariableChecklist) : [];
  const timeLabel = hasDataset ? (refs.timeColumn?.value || "time") : "time";
  const eventLabel = hasDataset ? (refs.eventColumn?.value || "event") : "event";
  const eventValue = hasDataset ? (refs.eventPositiveValue?.value || "choose event value") : "choose event value";
  const groupLabel = hasDataset ? (refs.groupColumn?.value || "overall only") : "overall only";
  const mlSummaryText = !hasDataset
    ? "Load a dataset first. ML uses the shared model feature selections shown here."
    : features.length
      ? `ML and DL share this model feature list: ${summarizeFeatureNames(features)}. Compare All uses the Evaluation section for cross-model screening only. Group by is shown here for context only.`
      : "No model feature set selected yet. Choose ML/DL model features before training.";
  const dlSummaryText = !hasDataset
    ? "Load a dataset first. DL uses the same shared model feature selections shown in this workspace."
    : features.length
      ? `Training inputs come only from the shared ML/DL model feature selections: ${summarizeFeatureNames(features)}. Group by is shown here for context only.`
      : "No model feature set selected yet. Choose ML/DL model features before training.";
  const finalChips = !hasDataset
    ? []
    : [
        formatOutcomeChip(timeLabel, eventLabel, eventValue),
        formatGroupChip(groupLabel),
        `Model features: ${features.length}`,
        `Categorical: ${categoricals.length}`,
        features.length ? `Preview: ${summarizeFeatureNames(features)}` : "Preview: none selected",
        categoricals.length ? `Categoricals: ${summarizeFeatureNames(categoricals, 3)}` : "Categoricals: none",
      ];

  if (refs.mlFeatureSummaryText) refs.mlFeatureSummaryText.textContent = mlSummaryText;
  if (refs.dlFeatureSummaryText) refs.dlFeatureSummaryText.textContent = dlSummaryText;
  renderChipList(refs.mlFeatureSummaryChips, finalChips);
  renderChipList(refs.dlFeatureSummaryChips, finalChips);

  renderContextCards({
    hasDataset,
    timeLabel,
    eventLabel,
    eventValue,
    groupLabel,
    coxFeatures,
    coxCategoricals,
    modelFeatures: features,
    modelCategoricals: categoricals,
    tableVariables,
  });
  syncDownloadButtonAvailability();
  syncAnalysisRunButtonAvailability();
  renderBenchmarkBoard();
  renderGuidedChrome();
}

function benchmarkGoalMeta(goal) {
  return goal === "ml"
    ? { label: "Classical ML", tab: "ml", panel: refs.mlPanel, reviewLabel: "Open model" }
    : { label: "Deep Learning", tab: "dl", panel: refs.dlPanel, reviewLabel: "Open model" };
}

function syncBenchmarkWorkbenchVisibility() {
  const workbenchOpen = Boolean(runtime.workbenchRevealed);
  const guidedPredictiveWorkbench = workbenchOpen && runtime.uiMode === "guided" && runtime.guidedGoal === "predictive";
  refs.benchmarkSummaryGrid?.classList.toggle("hidden", workbenchOpen);
  refs.benchmarkComparisonPlot?.closest(".table-card")?.classList.toggle("hidden", workbenchOpen);
  refs.benchmarkComparisonShell?.closest(".table-card")?.classList.toggle("hidden", workbenchOpen);
  refs.benchmarkWorkbench?.classList.toggle("hidden", !workbenchOpen);
  refs.runPredictiveCompareAllButton?.classList.toggle("hidden", workbenchOpen);
  refs.mlModelType?.closest(".model-choice-field")?.classList.toggle("hidden", workbenchOpen);
  refs.dlModelType?.closest(".model-choice-field")?.classList.toggle("hidden", workbenchOpen);
  refs.runCompareButton?.classList.toggle("hidden", workbenchOpen);
  refs.runCompareInlineButton?.classList.toggle("hidden", workbenchOpen);
  refs.runDlCompareButton?.classList.toggle("hidden", workbenchOpen);
  refs.runDlCompareInlineButton?.classList.toggle("hidden", workbenchOpen);
  refs.runMlButton?.classList.toggle("hidden", guidedPredictiveWorkbench);
  refs.runDlButton?.classList.toggle("hidden", guidedPredictiveWorkbench);
  refs.predictiveModelSelector?.closest(".predictive-model-picker")?.classList.toggle("hidden", !workbenchOpen);
  refs.runPredictiveSelectedButton?.classList.toggle("hidden", !workbenchOpen);
}

function renderPredictiveWorkbench() {
  const family = normalizedPredictiveFamily(runtime.predictiveFamily);
  const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
  const familyMode = runtime.resultPreference?.[family] || "single";
  const unifiedWorkspaceActive = activeTabName() === "benchmark" || (runtime.uiMode === "guided" && runtime.guidedGoal === "predictive");
  runtime.predictiveFamily = family;

  refs.benchmarkMlMount?.classList.toggle("hidden", family !== "ml");
  refs.benchmarkDlMount?.classList.toggle("hidden", family !== "dl");
  refs.benchmarkWorkbench?.setAttribute("data-active-family", family);
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
      : "Runs all 8 models (RSF, GBS, LASSO-Cox, DeepSurv, DeepHit, MTLR, Transformer, VAE) and ranks them by C-index. Results appear in the Unified Leaderboard below. Click any result to open that model\u2019s controls.";
  }
  if (refs.runPredictiveSelectedButton) {
    refs.runPredictiveSelectedButton.textContent = `Train ${selectedModel.label}`;
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
  return activeTabName() === "benchmark" || (runtime.uiMode === "guided" && runtime.guidedGoal === "predictive");
}

function syncPredictiveWorkbenchCompareVisibility() {
  const suppressCompare = predictiveWorkspaceUsesUnifiedBoard();
  const mlComparisonCard = refs.mlComparisonShell?.closest(".table-card");
  const mlManuscriptCard = refs.mlManuscriptShell?.closest(".table-card");
  const dlComparisonCard = refs.dlComparisonShell?.closest(".table-card");
  const dlManuscriptCard = refs.dlManuscriptShell?.closest(".table-card");
  const mlHasPlot = hasRenderedPlot(refs.mlComparisonPlot);
  const dlHasPlot = hasRenderedPlot(refs.dlComparisonPlot);
  const mlCompareActive = (runtime.resultPreference?.ml || "single") === "compare";
  const dlCompareActive = (runtime.resultPreference?.dl || "single") === "compare";

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

function renderGuidedSummaryChips(items) {
  if (!refs.guidedSummaryChips) return;
  refs.guidedSummaryChips.classList.toggle("hidden", items.length === 0);
  refs.guidedSummaryChips.innerHTML = items.length
    ? items.map((item) => `<span class="guided-summary-chip">${escapeHtml(item)}</span>`).join("")
    : "";
}

function guidedHelpDetails(summaryLabel, lines = []) {
  const cleanLines = lines.filter(Boolean);
  if (!cleanLines.length) return "";
  return `
    <details class="guided-help-toggle">
      <summary class="guided-help-summary">${escapeHtml(summaryLabel)}</summary>
      <div class="guided-help-body">
        <ul class="guided-help-list">
          ${cleanLines.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}
        </ul>
      </div>
    </details>
  `;
}

function renderGuidedChrome() {
  if (!refs.guidedShell || !refs.guidedPanel) return;
  const showGuided = runtime.uiMode === "guided" && Boolean(state.dataset);
  refs.guidedShell.classList.toggle("hidden", !showGuided);
  updateGuidedSurfaceVisibility();
  if (!showGuided) return;

  runtime.guidedStep = normalizedGuidedStep(runtime.guidedStep);
  const step = currentGuidedStep();
  const goal = runtime.guidedGoal;
  if (document.body) {
    document.body.dataset.guidedStep = String(step);
    document.body.dataset.guidedGoal = goal || "";
  }
  const timeColumn = refs.timeColumn?.value || "time";
  const eventColumn = refs.eventColumn?.value || "event";
  const eventValue = refs.eventPositiveValue?.value || "choose event value";
  const evaluation = evaluationModeLabel(goal);
  const resultMode = guidedResultModeLabel(goal);
  const summaryText = {
    2: "Check these three fields. If they look right, continue. Keep grouping and advanced settings for later.",
    3: "The endpoint is ready. Pick one analysis first. After you choose one, SurvStudio keeps one analysis path visible at a time to keep the workflow focused.",
    4: "Use the settings here and run once. Change only one thing at a time if you need to adjust it.",
    5: "Check whether the result looks sensible before moving on or exporting anything.",
  }[step] || "Load a dataset or sample cohort to begin.";

  if (refs.configTitleText) {
    refs.configTitleText.textContent = step === 2 ? "Outcome setup" : "Study Design";
  }
  if (refs.configHint) {
    refs.configHint.textContent = step === 2
      ? "Choose only the time column, event column, and the event value."
      : "Open Group by only if you need subgroup curves or grouped tables.";
  }
  if (refs.guidedSummaryTitle) {
    refs.guidedSummaryTitle.textContent = `${step}. ${["Load data", "Confirm outcome", "Choose analysis", "Configure & run", "Review results"][step - 1]}`;
  }
  if (refs.guidedSummaryText) refs.guidedSummaryText.textContent = summaryText;
  refs.guidedSummaryBar?.classList.toggle("hidden", step >= 3);
  const summaryChips = step === 2
    ? [`Dataset: ${state.dataset.filename}`]
    : step === 3
      ? [
          `Dataset: ${state.dataset.filename}`,
          `Outcome: ${timeColumn} / ${eventColumn} = ${eventValue}`,
          ...(goal ? [`Analysis: ${goalLabel(goal)}`] : []),
        ]
      : [
          `Dataset: ${state.dataset.filename}`,
          ...(goal ? [`Analysis: ${goalLabel(goal)}`] : []),
          ...(evaluation ? [`Evaluation: ${evaluation}`] : []),
          ...(resultMode ? [`Result mode: ${resultMode}`] : []),
        ];
  renderGuidedSummaryChips(summaryChips);

  updateStepIndicator(step);
  refs.guidedPanel.innerHTML = guidedPanelMarkup(step);
  syncGuidedCoxPanelMounts();
  renderGuidedRailStatus();
  updateGuidedResultVisibility();
}

function eventPreviewValues(columnName) {
  return getColumnMeta(columnName)?.unique_preview?.filter((value) => value !== null).map((value) => String(value)) || [];
}

function guidedPanelMarkup(step) {
  const goal = runtime.guidedGoal;
  const eventWarning = currentEventColumnWarning();
  const blockingEventWarning = eventWarning?.blocking ? eventWarning : null;
  const eventValueSelected = Boolean(refs.eventPositiveValue?.value);
  const eventValues = eventPreviewValues(refs.eventColumn?.value || "");
  const suggestedTime = state.dataset?.suggestions?.time_columns?.[0] || refs.timeColumn?.value || "";
  const suggestedEvent = recommendedEventColumns()[0] || refs.eventColumn?.value || "";
  const datasetName = escapeHtml(state.dataset?.filename || "dataset");
  const goalCards = ["km", "cox", "tables", "predictive"].map((entry) => {
    const meta = guidedGoalMeta(entry);
    return `
      <button class="guided-goal-card${goal === entry ? " active" : ""}" type="button" data-guided-action="choose-goal" data-goal="${entry}">
        <div class="guided-goal-top">
          <strong>${escapeHtml(goalLabel(entry))}</strong>
          <span class="guided-goal-badge">${escapeHtml(meta.badge)}</span>
        </div>
        <span>${escapeHtml(meta.description)}</span>
        <small>${escapeHtml(meta.note)}</small>
      </button>
    `;
  }).join("");

  if (step === 2) {
    const canContinue = endpointIsReady() && !blockingEventWarning && eventValueSelected;
    const issueHeading = canContinue ? "Ready to continue" : "What still needs attention";
    const issueMessage = blockingEventWarning?.message
      || (!refs.timeColumn?.value
        ? "Choose the time column first."
        : !refs.eventColumn?.value
          ? "Choose the event column next."
          : !eventValueSelected
            ? `Choose which value means event${eventValues.length ? `: ${eventValues.join(", ")}` : ""}.`
            : `SurvStudio is ready to use ${refs.timeColumn?.value || "time"}, ${refs.eventColumn?.value || "event"}, and ${refs.eventPositiveValue?.value || "event value"}.`);
    const recommendationCards = [
      suggestedTime ? `
        <div class="guided-quick-item">
          <strong>Suggested time</strong>
          <span>${escapeHtml(suggestedTime)}</span>
        </div>
      ` : "",
      suggestedEvent ? `
        <div class="guided-quick-item">
          <strong>Suggested event</strong>
          <span>${escapeHtml(suggestedEvent)}</span>
        </div>
      ` : "",
      refs.eventColumn?.value ? `
        <div class="guided-quick-item">
          <strong>Values seen</strong>
          <span>${escapeHtml(eventValues.length ? eventValues.join(", ") : "Unavailable")}</span>
        </div>
      ` : "",
    ].filter(Boolean).join("");
    const helpBlock = guidedHelpDetails("Show help", [
      "SurvStudio usually guesses these columns for you. Just confirm them before moving on.",
      "Leave Group by for later. This step is only about time, event, and the event-happened value.",
    ]);
    return `
      <div class="guided-panel-grid guided-panel-grid-compact">
        <article class="guided-card guided-card-focus">
          <span class="guided-kicker">Step 2 of 5</span>
          <h3>Tell SurvStudio what counts as survival time and event</h3>
          ${recommendationCards ? `<div class="guided-quick-grid">${recommendationCards}</div>` : ""}
          <div class="guided-readiness${canContinue ? " ready" : " warning"}">
            <strong>${escapeHtml(issueHeading)}</strong>
            <span>${escapeHtml(issueMessage)}</span>
          </div>
          <div class="guided-actions guided-actions-outcome">
            <button class="button ghost compact-btn" type="button" data-guided-action="go-home">Back to data</button>
            <button class="button primary compact-btn" type="button" data-guided-action="next-step"${canContinue ? "" : " disabled"}>Looks good, continue</button>
          </div>
          ${helpBlock}
        </article>
      </div>
    `;
  }

  if (step === 3) {
    const helpBlock = guidedHelpDetails("How do I choose?", [
      "Pick one analysis first. After you choose one, SurvStudio keeps one analysis path visible at a time to keep the workflow focused.",
      "If you are unsure, start with Kaplan-Meier. It is the easiest result to sanity-check.",
    ]);
    return `
      <div class="guided-panel-grid guided-panel-grid-compact">
        <article class="guided-card guided-card-focus">
          <span class="guided-kicker">Step 3 of 5</span>
          <h3>Choose what you want to do next</h3>
          <div class="guided-readiness ready">
            <strong>Current outcome</strong>
            <span>${escapeHtml(`${datasetName}: ${refs.timeColumn?.value || "time"} / ${refs.eventColumn?.value || "event"} = ${refs.eventPositiveValue?.value || "choose event value"}`)}</span>
          </div>
          <div class="guided-goal-grid">${goalCards}</div>
          <div class="guided-actions">
            <button class="button ghost compact-btn" type="button" data-guided-action="previous-step">Back</button>
          </div>
          ${helpBlock}
        </article>
      </div>
    `;
  }

  if (step === 4) {
    const coxSelectionSummary = goal === "cox" ? '<div id="guidedCoxSelectionMount"></div>' : "";
    const coxPreviewSummary = goal === "cox" ? '<div id="guidedCoxPreviewMount"></div>' : "";
    const workbenchSingleModelMode = runtime.workbenchRevealed && ["predictive", "ml", "dl"].includes(goal);
    const configureCopy = {
      km: {
        title: "Run Kaplan-Meier",
        text: "Run the curve once with the current endpoint. Open Group by only if you need subgroup curves or grouped tables.",
        runAction: "run-km",
        runLabel: "Run Kaplan-Meier",
        runScope: "km",
        tip: "Start with Overall only and the default log-rank test.",
      },
      cox: {
        title: "Run Cox PH",
        text: "Review the settings here, then fit the model once.",
        runAction: "run-cox",
        runLabel: "Run Cox",
        runScope: "cox",
        tip: "If the result looks wrong, change the covariates first.",
      },
      ml: {
        title: "Run ML Analysis",
        text: "Review the settings here, then start with one run.",
        runAction: "run-ml",
        runLabel: "Run Analysis",
        secondaryAction: "run-ml-compare",
        secondaryLabel: "Compare all ML models",
        runScope: "ml",
        busyText: "ML model run in progress. Stay on this analysis path if you want the updated result to open here when the run finishes.",
        tip: "Use this after the classical analyses look right.",
      },
      dl: {
        title: "Run DL Analysis",
        text: "Review the settings here, then start with one run.",
        runAction: "run-dl",
        runLabel: "Run Analysis",
        secondaryAction: "run-dl-compare",
        secondaryLabel: "Compare all DL models",
        runScope: "dl",
        busyText: "DL model run in progress. Deep-learning runs can take longer, so stay on this analysis path if you want the updated result to open here when the run finishes.",
        tip: "Start with one model. This is the slowest and most advanced path.",
      },
      predictive: {
        title: runtime.workbenchRevealed ? "Train a model" : "Run ML/DL Models",
        text: runtime.workbenchRevealed
          ? "Train the selected model directly. The leaderboard is already built, so use the selected model controls below."
          : "Compare all models to build the leaderboard, then click any result to open its controls.",
        runAction: runtime.workbenchRevealed ? "run-predictive-selected" : "run-predictive-compare-all",
        runLabel: runtime.workbenchRevealed ? "Run Analysis" : "Compare all models",
        runScope: "predictive",
        busyText: "A predictive model run is already in progress. Wait for it to finish before starting another one.",
        tip: runtime.workbenchRevealed
          ? "Use the selected model controls to train one model at a time."
          : "Run Compare All once to see every model ranked. Then click a result to tune that model.",
      },
      tables: {
        title: "Build Cohort Table",
        text: "Review the settings here, then build the table once.",
        runAction: "run-tables",
        runLabel: "Build cohort table",
        runScope: "tables",
        tip: "Grouping is optional here.",
      },
    }[goal] || {
      title: "Configure analysis",
      text: "Pick one analysis path first.",
      runAction: "previous-step",
      runLabel: "Go back",
      runScope: null,
      tip: "",
    };
    if (workbenchSingleModelMode) {
      configureCopy.secondaryAction = null;
      configureCopy.secondaryLabel = null;
    }
    const scopeBusy = isScopeBusy(configureCopy.runScope);
    const predictivePrimaryBusy = goal === "predictive" && isScopeBusy(predictiveFamilyGoal());
    const predictiveCompareBusy = goal === "predictive" && (isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl"));
  const mlSingleModelBlocked = goal === "ml" && refs.mlEvaluationStrategy?.value === "repeated_cv";
  const mlSingleModelBlockedText = mlSingleModelBlocked
      ? "Run Analysis is disabled while Repeated CV is selected. Use Compare all ML models or switch Evaluation Mode back to Deterministic Holdout."
      : "";
    const primaryDisabled = goal === "predictive"
      ? predictiveCompareBusy
      : (scopeBusy || mlSingleModelBlocked);
    const secondaryDisabled = goal === "predictive"
      ? predictiveCompareBusy
      : scopeBusy;
    const busyStatusText = goal === "predictive"
      ? (
        predictiveCompareBusy
          ? "A unified predictive comparison or model run is already in progress. Wait for it to finish before starting another one."
          : (predictivePrimaryBusy ? "The selected model family is already running. Wait for it to finish before testing this model again." : "")
      )
      : (scopeBusy ? configureCopy.busyText : "");
    return `
      <div class="guided-panel-grid guided-panel-grid-compact">
        <article class="guided-card guided-card-focus">
          <span class="guided-kicker">Step 4 of 5</span>
          <h3>${escapeHtml(configureCopy.title)}</h3>
          <div class="guided-actions guided-actions-priority${configureCopy.secondaryAction ? " guided-actions-dual" : ""}">
            ${configureCopy.secondaryAction
              ? `<button class="button ghost compact-btn guided-run-choice guided-run-choice-secondary" type="button" data-guided-action="${escapeHtml(configureCopy.secondaryAction)}"${secondaryDisabled ? " disabled" : ""}>${escapeHtml(configureCopy.secondaryLabel)}</button>`
              : ""}
            <button class="button primary compact-btn guided-run-choice${primaryDisabled ? " is-loading" : ""}" type="button" data-guided-action="${escapeHtml(configureCopy.runAction)}"${primaryDisabled ? " disabled" : ""}>${escapeHtml(configureCopy.runLabel)}</button>
          </div>
          ${mlSingleModelBlockedText ? `<div class="guided-run-status" role="status">${escapeHtml(mlSingleModelBlockedText)}</div>` : ""}
          <div class="guided-actions guided-actions-secondary">
            <button class="button ghost compact-btn" type="button" data-guided-action="previous-step">Back</button>
          </div>
          ${coxSelectionSummary}
          ${coxPreviewSummary}
        </article>
      </div>
    `;
  }

  const reviewChecks = {
    km: [
      "Do the curves and sample counts look plausible?",
      "If grouped, are the groups the ones you intended?",
      "If it looks wrong, go back and re-check Group by or max time.",
    ],
    cox: [
      "Do the covariates match the variables you meant to test?",
      "Are the hazard ratio directions roughly what you expected?",
      "If not, go back and check the covariate list first.",
    ],
    tables: [
      "Does the table contain the variables you wanted to summarize?",
      "If grouped, is the grouping field correct?",
      "If not, go back and adjust the table variable list.",
    ],
    predictive: [
      "Did you stay on the model family you actually meant to review: ML or DL?",
      "Does the feature count still look reasonable for this cohort size?",
      "If not, go back one step, switch family if needed, and rerun the visible model workspace.",
    ],
    ml: [
      "Did you choose the evaluation mode intentionally?",
      "Does the feature count look reasonable for this cohort size?",
      "If not, go back and simplify the feature set first.",
    ],
    dl: [
      "Did you choose the evaluation mode intentionally?",
      "Are you comfortable with the longer runtime of this path?",
      "If not, go back and start with a smaller or simpler run.",
    ],
  }[goal] || [
    "Check that this is the analysis you meant to run.",
    "If something looks wrong, go back one step and change only one thing.",
  ];
  return `
    <div class="guided-panel-grid guided-panel-grid-compact">
      <article class="guided-card guided-card-focus">
        <span class="guided-kicker">Step 5 of 5</span>
        <h3>Check the result before you trust it</h3>
        ${guidedHelpDetails("Show review checklist", [
          `${goalLabel(goal)} finished. Do one quick sanity check before you move on or export anything.`,
          ...reviewChecks,
          "If something looks off, go back one step, change one thing, and run again.",
        ])}
      </article>
    </div>
  `;
}

function updateGroupingDetailsVisibility(tabName = activeTabName(), { force = false } = {}) {
  if (!refs.groupingDetails) return;
  if (runtime.uiMode === "guided" && !force) return;
  refs.groupingDetails.open = ["km", "tables"].includes(tabName);
}

function focusModelFeatureEditor(tabName = "ml") {
  activateTab(tabName);
  requestAnimationFrame(() => {
    const featureChecklist = tabName === "dl" ? refs.dlModelFeatureChecklist : refs.modelFeatureChecklist;
    featureChecklist?.closest(".selection-card")?.scrollIntoView({ behavior: "smooth", block: "center" });
    flashPresetTargets([refs.modelFeatureChecklist, refs.modelCategoricalChecklist, refs.dlModelFeatureChecklist, refs.dlModelCategoricalChecklist]);
  });
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
  if (!preset) {
    showToast("No dataset-specific preset is available for this file.", "error");
    return;
  }

  const columnNames = state.dataset.columns.map((c) => c.name);
  if (columnNames.includes(preset.timeColumn)) refs.timeColumn.value = preset.timeColumn;
  if (columnNames.includes(preset.eventColumn)) refs.eventColumn.value = preset.eventColumn;
  if (preset.timeUnitLabel && refs.timeUnitLabel) refs.timeUnitLabel.value = preset.timeUnitLabel;
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
    syncCoxCovariateSelection();
  }
  setCheckedValues(refs.cohortVariableChecklist, tableVariables);
  updateDatasetBadge();

  setDatasetPresetButtonState(mode);

  const summaryTitle = `${preset.name} applied`;
  const summaryText = mode === "basic"
    ? "Updated the study columns, group split, Cox covariates, and cohort-table variable selections. No analysis ran yet."
    : "Updated the study columns and the shared feature checklists used by ML and DL. You can review the same selections on either tab. No analysis ran yet.";
  const summaryChips = [
    `Time: ${preset.timeColumn}`,
    `Event: ${preset.eventColumn}=${preset.eventPositiveValue}`,
    `Group: ${preset.basicGroup}`,
    `${mode === "models" ? "Model features" : "Cox covariates"}: ${covariates.length}`,
    `Categorical: ${categoricals.length}`,
    `Table vars: ${tableVariables.length}`,
  ];
  renderDatasetPresetStatus(summaryTitle, summaryText, summaryChips);
  renderSharedFeatureSummary();

  flashPresetTargets([
    refs.timeColumn,
    refs.eventColumn,
    refs.eventPositiveValue,
    refs.groupColumn,
    refs.timeUnitLabel,
    refs.covariateChecklist,
    refs.categoricalChecklist,
    refs.modelFeatureChecklist,
    refs.modelCategoricalChecklist,
    refs.cohortVariableChecklist,
    refs.mlFeatureSummaryCard,
    refs.dlFeatureSummaryCard,
  ]);

  if (mode === "models") {
    const activeTab = document.querySelector(".tab-button.active")?.dataset.tab;
    if (activeTab === "ml") {
      requestAnimationFrame(() => {
        refs.mlFeatureSummaryCard?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    } else if (activeTab === "dl") {
      requestAnimationFrame(() => {
        refs.dlFeatureSummaryCard?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }
  } else {
    requestAnimationFrame(() => {
      refs.configStrip?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  queueHistorySync();
  showToast(`${preset.name} — ${mode === "basic" ? "KM/Cox" : "ML/DL"} preset applied`, "success", 2500);
}

function updateDatasetPresetBar() {
  const preset = datasetPresetForCurrentDataset();
  if (!refs.datasetPresetBar || !refs.datasetPresetTitle || !refs.datasetPresetText) return;
  if (!preset) {
    refs.datasetPresetBar.classList.add("hidden");
    renderDatasetPresetStatus(
      "No preset applied yet.",
      "Applying a preset updates recommended columns and checkbox selections only. It does not run an analysis.",
      [],
    );
    setDatasetPresetButtonState(null);
    return;
  }
  refs.datasetPresetTitle.textContent = preset.name;
  refs.datasetPresetText.textContent = preset.summary;
  renderDatasetPresetStatus(
    "No preset applied yet.",
    "Applying a preset updates recommended columns and checkbox selections only. It does not run an analysis.",
    [],
  );
  setDatasetPresetButtonState(null);
  refs.datasetPresetBar.classList.remove("hidden");
}

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
      throw new Error(`${eventWarning.message} If this is intentional, turn on Show all columns for Event first.`);
    }
    throw new Error(eventWarning.message);
  }
  if (!refs.eventPositiveValue.value) {
    throw new Error(`Choose the Event Value for "${eventColumn}" before running an analysis.`);
  }
  return {
    dataset_id: state.dataset.dataset_id,
    time_column: timeColumn,
    event_column: eventColumn,
    event_positive_value: refs.eventPositiveValue.value,
    group_column: refs.groupColumn.value || null,
    time_unit_label: refs.timeUnitLabel.value || "Months",
    max_time: refs.maxTime.value ? Number(refs.maxTime.value) : null,
  };
}

function validateGroupingSelection() {
  const warning = currentGroupColumnWarning();
  if (warning?.tone === "error") throw new Error(warning.message);
}

function validateDlControls() {
  const modelType = refs.dlModelType?.value || "deepsurv";
  const epochs = Number(refs.dlEpochs?.value);
  if (!Number.isFinite(epochs) || epochs < 10 || epochs > 1000) {
    throw new Error(`Epochs must be between 10 and 1000. Current value: ${formatValue(epochs)}.`);
  }
  const learningRate = Number(refs.dlLearningRate?.value);
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.1) {
    throw new Error(`Learning rate must be greater than 0 and at most 0.1. Current value: ${formatValue(learningRate)}.`);
  }
  const dropout = Number(refs.dlDropout?.value);
  if (!Number.isFinite(dropout) || dropout < 0 || dropout > 0.5) {
    throw new Error(`Dropout must be between 0 and 0.5. Current value: ${formatValue(dropout)}.`);
  }
  if (modelType !== "transformer") {
    parseHiddenLayersStrict();
  }
  const batchSize = Number(refs.dlBatchSize?.value);
  if (!Number.isFinite(batchSize) || batchSize < 8 || batchSize > 512) {
    throw new Error(`Batch size must be between 8 and 512. Current value: ${formatValue(batchSize)}.`);
  }
  const randomSeed = Number(refs.dlRandomSeed?.value);
  if (!Number.isFinite(randomSeed) || !Number.isInteger(randomSeed)) {
    throw new Error(`Random seed must be an integer. Current value: ${formatValue(randomSeed)}.`);
  }
  const patience = Number(refs.dlEarlyStoppingPatience?.value);
  if (!Number.isFinite(patience) || patience < 1 || patience > 100) {
    throw new Error(`Early stop patience must be between 1 and 100. Current value: ${formatValue(patience)}.`);
  }
  const minDelta = Number(refs.dlEarlyStoppingMinDelta?.value);
  if (!Number.isFinite(minDelta) || minDelta < 0 || minDelta > 0.1) {
    throw new Error(`Min delta must be between 0 and 0.1. Current value: ${formatValue(minDelta)}.`);
  }
  const evaluationStrategy = refs.dlEvaluationStrategy?.value || "holdout";
  if (evaluationStrategy === "repeated_cv") {
    const cvFolds = Number(refs.dlCvFolds?.value);
    if (!Number.isFinite(cvFolds) || cvFolds < 2 || cvFolds > 10) {
      throw new Error(`CV folds must be between 2 and 10. Current value: ${formatValue(cvFolds)}.`);
    }
    const cvRepeats = Number(refs.dlCvRepeats?.value);
    if (!Number.isFinite(cvRepeats) || cvRepeats < 1 || cvRepeats > 20) {
      throw new Error(`CV repeats must be between 1 and 20. Current value: ${formatValue(cvRepeats)}.`);
    }
    const parallelJobs = Number(refs.dlParallelJobs?.value);
    if (!Number.isFinite(parallelJobs) || parallelJobs < 1 || parallelJobs > 16) {
      throw new Error(`Parallel jobs must be between 1 and 16. Current value: ${formatValue(parallelJobs)}.`);
    }
  }
  if (modelType === "deephit" || modelType === "mtlr") {
    const numTimeBins = Number(refs.dlNumTimeBins?.value);
    if (!Number.isFinite(numTimeBins) || numTimeBins < 10 || numTimeBins > 200) {
      throw new Error(`Time bins must be between 10 and 200. Current value: ${formatValue(numTimeBins)}.`);
    }
  }
  if (modelType === "transformer") {
    const dModel = Number(refs.dlDModel?.value);
    const nHeads = Number(refs.dlHeads?.value);
    const nLayers = Number(refs.dlLayers?.value);
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
  if (modelType === "vae") {
    const latentDim = Number(refs.dlLatentDim?.value);
    const nClusters = Number(refs.dlClusters?.value);
    if (!Number.isFinite(latentDim) || latentDim < 2 || latentDim > 32) {
      throw new Error(`Latent dim must be between 2 and 32. Current value: ${formatValue(latentDim)}.`);
    }
    if (!Number.isFinite(nClusters) || nClusters < 2 || nClusters > 10) {
      throw new Error(`Clusters must be between 2 and 10. Current value: ${formatValue(nClusters)}.`);
    }
  }
}

function renderDatasetPreview() {
  renderTable(refs.datasetPreviewShell, state.dataset.preview);
}

function updateDatasetBadge() {
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

function activateTab(tabName, { setGuidedGoal = runtime.uiMode === "guided", historyMode = "replace", focusTabButton = false, syncHistory = true } = {}) {
  let resolvedTabName = tabName;
  if (resolvedTabName === "predictive") {
    resolvedTabName = "benchmark";
  }
  if (tabName === "ml" || tabName === "dl") {
    runtime.predictiveFamily = tabName;
  }
  if (runtime.uiMode === "guided" && runtime.guidedGoal === "predictive" && (resolvedTabName === "ml" || resolvedTabName === "dl")) {
    resolvedTabName = "benchmark";
  }
  if (runtime.uiMode === "expert" && (resolvedTabName === "ml" || resolvedTabName === "dl")) {
    resolvedTabName = "benchmark";
  }
  if (resolvedTabName !== "benchmark" && activeTabName() === "benchmark") {
    runtime.workbenchRevealed = false;
    refs.benchmarkWorkbench?.classList.add("hidden");
    refs.predictiveModelSelector?.closest(".predictive-model-picker")?.classList.add("hidden");
    refs.runPredictiveSelectedButton?.classList.add("hidden");
  }
  refs.tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === resolvedTabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
    button.setAttribute("tabindex", isActive ? "0" : "-1");
    if (isActive && runtime.uiMode !== "guided" && focusTabButton) {
      try {
        button.focus({ preventScroll: true });
      } catch {
        button.focus();
      }
    }
  });
  refs.tabPanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === resolvedTabName));
  if (setGuidedGoal && GUIDED_GOALS.includes(resolvedTabName)) runtime.guidedGoal = resolvedTabName;
  renderPredictiveWorkbench();
  updateGroupingDetailsVisibility(resolvedTabName);
  if (state.dataset && syncHistory) syncHistoryState(historyMode);
  renderGuidedChrome();
  requestAnimationFrame(() => {
    if (resolvedTabName === "km" && state.km) Plotly.Plots.resize(refs.kmPlot);
    if (resolvedTabName === "cox" && state.cox) {
      if (refs.coxPlot?.data) Plotly.Plots.resize(refs.coxPlot);
      if (refs.coxDiagnosticsPlot?.data) Plotly.Plots.resize(refs.coxDiagnosticsPlot);
    }
    if ((resolvedTabName === "ml" || resolvedTabName === "benchmark") && state.ml) {
      if (refs.mlImportancePlot?.data) Plotly.Plots.resize(refs.mlImportancePlot);
      if (refs.mlShapPlot?.data) Plotly.Plots.resize(refs.mlShapPlot);
      if (refs.mlComparisonPlot?.data) Plotly.Plots.resize(refs.mlComparisonPlot);
    }
    if ((resolvedTabName === "dl" || resolvedTabName === "benchmark") && state.dl) {
      if (refs.dlImportancePlot?.data) Plotly.Plots.resize(refs.dlImportancePlot);
      if (refs.dlLossPlot?.data) Plotly.Plots.resize(refs.dlLossPlot);
    }
    if (resolvedTabName === "benchmark" && refs.benchmarkComparisonPlot?.data) {
      Plotly.Plots.resize(refs.benchmarkComparisonPlot);
    }
  });
}

function updateControlsFromDataset({ scrollToTop = false } = {}) {
  const columnNames = state.dataset.columns.map((c) => c.name);
  const suggestions = state.dataset.suggestions;
  if (refs.showAllEventColumns) refs.showAllEventColumns.checked = false;
  if (refs.covariateSearchInput) refs.covariateSearchInput.value = "";
  if (refs.categoricalSearchInput) refs.categoricalSearchInput.value = "";
  if (refs.cohortVariableSearchInput) refs.cohortVariableSearchInput.value = "";
  renderTimeColumnOptions({ preferred: inferDefault(columnNames, suggestions.time_columns, 0), silent: true });
  renderEventColumnOptions({
    preferred: inferDefault(columnNames, suggestions.event_columns, 1),
    silent: true,
  });
  renderSelect(refs.groupColumn, columnNames, { includeBlank: true, blankLabel: "Overall only", selected: null });
  refreshVariableSelections();
  updateDatasetBadge();
  renderSharedFeatureSummary();
  renderDatasetPreview();
  updateDatasetPresetBar();
  refs.downloadSignatureButton.disabled = true;
  showWorkspace();
  if (scrollToTop) scrollWorkspaceEntryToTop();
  renderGuidedChrome();
  const timeSugg = suggestions.time_columns?.[0];
  const eventSugg = hasConfidentEventSuggestion() ? (refs.eventColumn?.value || null) : null;
  if (timeSugg && eventSugg) {
    showSmartBanner(`Auto-detected: "${timeSugg}" as time column, "${eventSugg}" as event column. Adjust if needed.`);
  } else if (timeSugg) {
    showSmartBanner(`Auto-detected "${timeSugg}" as the time column. Confirm the event column and event value before running an analysis.`);
  }
}

function updateAfterDataset(payload, { scrollToTop = false } = {}) {
  state.dataset = payload;
  state.km = null;
  state.cox = null;
  state.cohort = null;
  state.signature = null;
  state.ml = null;
  state.dl = null;
  runtime.compareCache.ml = null;
  runtime.compareCache.dl = null;
  runtime.guidedGoal = null;
  runtime.guidedStep = runtime.uiMode === "guided" ? 2 : 1;
  runtime.workbenchRevealed = false;
  refs.kmMetaBanner.textContent = "Configure your study columns above, then click Run Analysis.";
  refs.coxMetaBanner.textContent = "Select covariates above, then click Run Analysis.";
  refs.mlMetaBanner.textContent = "Select shared model features in Predictive Models, then run analysis.";
  refs.dlMetaBanner.textContent = "Select model features here, configure hyperparameters, then run analysis.";
  refs.deriveSummary.innerHTML = "";
  refs.deriveSummary.classList.add("hidden");
  runtime.lastDerivedGroup = null;
  runtime.deriveDraftTouched = false;
  runtime.derivedColumnProvenance = normalizeDerivedColumnProvenance(payload.derived_column_provenance);
  runtime.resultPreference.ml = "single";
  runtime.resultPreference.dl = "single";
  resetCoxPreview({ rerender: false });
  refs.datasetPresetBar?.classList.add("hidden");
  refs.deriveStatus.textContent = "";
  setSelectValueIfPresent(refs.deriveMethod, "median_split");
  if (refs.cutpointPlot) { refs.cutpointPlot.innerHTML = ""; refs.cutpointPlot.classList.add("hidden"); }
  refs.kmSummaryShell.innerHTML = '<div class="empty-state">Survival statistics will appear after you run the analysis.</div>';
  refs.kmRiskShell.innerHTML = '<div class="empty-state">Number of patients at risk over time.</div>';
  refs.kmPairwiseShell.innerHTML = '<div class="empty-state">Group-vs-group comparisons (requires 2+ groups).</div>';
  refs.signatureShell.innerHTML = '<div class="empty-state">Use auto-discovery to find the best feature combinations.</div>';
  refs.coxResultsShell.innerHTML = '<div class="empty-state">Hazard ratios will appear after running Cox analysis.</div>';
  refs.coxDiagnosticsShell.innerHTML = '<div class="empty-state">Scaled Schoenfeld residual screening details will appear here.</div>';
  clearPlotShell(refs.coxDiagnosticsPlot, '<div class="empty-state plot-empty"><span>Scaled Schoenfeld residual screening appears here after fitting the model.</span></div>', { state: "placeholder" });
  refs.cohortTableShell.innerHTML = COHORT_TABLE_EMPTY_STATE_HTML;
  refs.mlComparisonShell.innerHTML = '<div class="empty-state">Click "Compare All" to see Cox vs RSF vs GBS side by side.</div>';
  if (refs.mlComparisonTitle) refs.mlComparisonTitle.textContent = "Model Comparison";
  refs.mlManuscriptShell.innerHTML = '<div class="empty-state">Comparison-ready manuscript rows appear after running a comparison.</div>';
  refs.mlComparisonPlot.innerHTML = "";
  refs.mlComparisonPlot.classList.add("hidden");
  refs.dlComparisonShell.innerHTML = '<div class="empty-state">Click "Compare All" to benchmark DeepSurv, DeepHit, Neural MTLR, Transformer, and VAE.</div>';
  if (refs.dlComparisonTitle) refs.dlComparisonTitle.textContent = "Deep Model Comparison";
  refs.dlManuscriptShell.innerHTML = '<div class="empty-state">Comparison-ready manuscript rows appear after running a deep comparison.</div>';
  refs.dlComparisonPlot.innerHTML = "";
  refs.dlComparisonPlot.classList.add("hidden");
  setPanelResultMode(refs.mlPanel, "idle");
  setPanelResultMode(refs.dlPanel, "idle");
  clearPlotShell(refs.mlImportancePlot, '<div class="empty-state plot-empty"><span>Run Analysis to see feature importance</span></div>', { state: "placeholder" });
  clearPlotShell(refs.mlShapPlot, '<div class="empty-state plot-empty"><span>SHAP values will appear after training</span></div>', { state: "placeholder" });
  clearPlotShell(refs.dlImportancePlot, '<div class="empty-state plot-empty"><span>Run Analysis to see deep learning results</span></div>', { state: "placeholder" });
  clearPlotShell(refs.dlLossPlot, '<div class="empty-state plot-empty"><span>Training and monitor metric curves will appear here</span></div>', { state: "placeholder" });
  purgePlot(refs.kmPlot);
  purgePlot(refs.coxPlot);
  refs.kmPlot.innerHTML = '<div class="empty-state plot-empty"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg><span>Click <strong>Run Analysis</strong> to generate your survival curve</span><small>Tip: Press Ctrl+Enter as a shortcut</small></div>';
  refs.coxPlot.innerHTML = '<div class="empty-state plot-empty"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg><span>Select covariates and click <strong>Run Analysis</strong> to see the forest plot</span></div>';
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
  refs.downloadMlComparisonButton.disabled = true;
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = true;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = true;
  setMlManuscriptDownloadsEnabled(false);
  refs.downloadDlComparisonButton.disabled = true;
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = true;
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = true;
  setDlManuscriptDownloadsEnabled(false);
  updateControlsFromDataset({ scrollToTop });
}

function updateAfterDerivedDataset(payload, { deferChrome = false } = {}) {
  const snapshot = captureControlSnapshot();
  const preservedStates = {
    km: state.km,
    cox: state.cox,
    cohort: state.cohort,
    signature: state.signature,
    ml: state.ml,
    dl: state.dl,
    compareMl: runtime.compareCache.ml,
    compareDl: runtime.compareCache.dl,
  };
  const preservedGuidedGoal = runtime.guidedGoal;
  const preservedGuidedStep = runtime.guidedStep;
  const columnNames = payload.columns.map((column) => column.name);
  const suggestions = payload.suggestions || { time_columns: [], event_columns: [] };
  const preferredTime = snapshot?.timeColumn && columnNames.includes(snapshot.timeColumn)
    ? snapshot.timeColumn
    : inferDefault(columnNames, suggestions.time_columns || [], 0);
  const preferredEvent = snapshot?.eventColumn && columnNames.includes(snapshot.eventColumn)
    ? snapshot.eventColumn
    : inferDefault(columnNames, suggestions.event_columns || [], 1);
  const preferredGroup = snapshot?.groupColumn && columnNames.includes(snapshot.groupColumn)
    ? snapshot.groupColumn
    : null;

  state.dataset = payload;
  runtime.derivedColumnProvenance = normalizeDerivedColumnProvenance(payload.derived_column_provenance);
  runtime.deriveDraftTouched = false;
  runtime.guidedGoal = preservedGuidedGoal;
  runtime.guidedStep = preservedGuidedStep;
  resetCoxPreview({ rerender: false });

  if (refs.showAllEventColumns) refs.showAllEventColumns.checked = Boolean(snapshot?.showAllEventColumns);
  renderTimeColumnOptions({ preferred: preferredTime, silent: true });
  renderEventColumnOptions({ preferred: preferredEvent, silent: true });
  renderSelect(refs.groupColumn, columnNames, { includeBlank: true, blankLabel: "Overall only", selected: preferredGroup });
  refreshVariableSelections();
  if (snapshot) applyControlSnapshot(snapshot);
  updateDatasetBadge();
  renderDatasetPreview();
  updateDatasetPresetBar();
  showWorkspace();

  state.km = preservedStates.km;
  state.cox = preservedStates.cox;
  state.cohort = preservedStates.cohort;
  state.signature = preservedStates.signature;
  state.ml = preservedStates.ml;
  state.dl = preservedStates.dl;
  runtime.compareCache.ml = preservedStates.compareMl;
  runtime.compareCache.dl = preservedStates.compareDl;

  if (!deferChrome) {
    renderGuidedChrome();
    queueHistorySync();
  }
}

function hasCompletedResults() {
  return Boolean(state.km || state.cox || state.cohort || state.signature || state.ml || state.dl);
}

function uploadFeedbackMessages(payload, { previousDatasetName = "", clearedResults = false } = {}) {
  const datasetName = payload?.filename || "dataset";
  const rowCount = Number(payload?.n_rows || 0).toLocaleString();
  const columnCount = Number(payload?.n_columns || 0).toLocaleString();
  const summary = `${datasetName} loaded (${rowCount} rows, ${columnCount} columns).`;
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

async function uploadDataset() {
  if (!refs.datasetFile.files?.length) throw new Error("Choose a dataset file first.");
  const selectedFile = refs.datasetFile.files[0];
  const previousDatasetName = state.dataset?.filename || "";
  const clearedResults = Boolean(state.dataset) && hasCompletedResults();
  setRuntimeBanner(`Uploading ${selectedFile.name} and preparing a fresh analysis workspace.`, "info");
  const formData = new FormData();
  formData.append("file", selectedFile);
  const payload = await fetchJSON("/api/upload", { method: "POST", body: formData });
  updateAfterDataset(payload, { scrollToTop: true });
  runtime.historySyncPaused = true;
  activateTab("km", { setGuidedGoal: false });
  runtime.historySyncPaused = false;
  syncHistoryState("push");
  const feedback = uploadFeedbackMessages(payload, { previousDatasetName, clearedResults });
  setRuntimeBanner(feedback.banner, "success");
  showToast(feedback.toast, "success", 3400);
}

async function loadExampleDataset() {
  const payload = await fetchJSON("/api/load-example", { method: "POST" });
  updateAfterDataset(payload, { scrollToTop: true });
  runtime.historySyncPaused = true;
  activateTab("km", { setGuidedGoal: false });
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadTcgaUploadReadyDataset() {
  const payload = await fetchJSON("/api/load-tcga-upload-ready", { method: "POST" });
  updateAfterDataset(payload, { scrollToTop: true });
  runtime.historySyncPaused = true;
  activateTab("km", { setGuidedGoal: false });
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadTcgaDataset() {
  const payload = await fetchJSON("/api/load-tcga-example", { method: "POST" });
  updateAfterDataset(payload, { scrollToTop: true });
  runtime.historySyncPaused = true;
  activateTab("km", { setGuidedGoal: false });
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadGbsg2Dataset() {
  const payload = await fetchJSON("/api/load-gbsg2-example", { method: "POST" });
  updateAfterDataset(payload, { scrollToTop: true });
  runtime.historySyncPaused = true;
  activateTab("km", { setGuidedGoal: false });
  runtime.historySyncPaused = false;
  syncHistoryState("push");
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
  refs.deriveStatus.textContent = isOptimal
    ? "Scanning a new grouping column..."
    : "Creating a new grouping column...";

  const body = {
    dataset_id: state.dataset.dataset_id,
    source_column: sourceColumn,
    method,
    new_column_name: requestedColumnName,
    cutoff: cutoffValue,
  };
  if (isOptimal) {
    body.time_column = refs.timeColumn.value;
    body.event_column = refs.eventColumn.value;
    body.event_positive_value = refs.eventPositiveValue.value;
    body.min_group_fraction = Number(refs.deriveMinGroupFraction?.value || 0.1);
    body.permutation_iterations = Number(refs.derivePermutationIterations?.value || 500);
    body.random_seed = Number(refs.deriveRandomSeed?.value || 20260311);
  }

  const preservedGroup = String(refs.groupColumn?.value || "");
  const shouldAutoApplyDerivedGroup = autoApplyOverride ?? !preservedGroup;
  const guidedKmRefresh = runtime.uiMode === "guided" && runtime.guidedGoal === "km";
  const shouldRefreshKm = refreshKmOverride ?? (shouldAutoApplyDerivedGroup && (activeTabName() === "km" || guidedKmRefresh));
  const payload = await fetchJSON("/api/derive-group", { method: "POST", body: JSON.stringify(body) });
  updateAfterDerivedDataset(payload, { deferChrome: shouldRefreshKm });
  runtime.derivedColumnProvenance[payload.derived_column] = {
    outcomeInformed: Boolean(payload.derive_summary?.outcome_informed),
    recipe: payload.derive_summary?.recipe || {},
  };
  refreshVariableSelections();
  if (shouldAutoApplyDerivedGroup) {
    refs.groupColumn.value = payload.derived_column;
  } else {
    setSelectValueIfPresent(refs.groupColumn, preservedGroup);
  }
  syncDeriveControlsState();
  const shouldClearTableOutput = shouldAutoApplyDerivedGroup
    && currentCohortTableOutputState().hasOutput
    && (activeTabName() === "tables" || runtime.guidedGoal === "tables");
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
  if (shouldClearTableOutput) {
    clearCohortTableOutput({ rerenderChrome: false, syncHistory: false });
  } else {
    renderSharedFeatureSummary();
  }
  renderGuidedChrome();
  queueHistorySync();
  if (toastMode !== "silent") {
    showToast(
      shouldRefreshKm
        ? `Created ${payload.derived_column} and updated Group by. ${featureUseMessage} Kaplan-Meier is refreshing now.`
        : shouldAutoApplyDerivedGroup
          ? `Created ${payload.derived_column} and updated Group by. ${featureUseMessage}${shouldClearTableOutput ? " Previous cohort table output was cleared; build the table again to match the new grouping." : ""}`
          : `Created ${payload.derived_column}. Current Group by remains ${preservedGroup}. ${featureUseMessage} Use Group by or Run again when you want to analyze the new grouping.`,
      "success",
      toastMode === "guided-inline" ? 4200 : 5200,
    );
  }

  // If optimal cutpoint, also show the scan plot
  if (isOptimal && (payload.cutpoint_figure || payload.derive_summary?.scan_data)) {
    try {
      const scanFigure = payload.cutpoint_figure;
      if (scanFigure && refs.cutpointPlot) {
        refs.cutpointPlot.classList.remove("hidden");
        refs.cutpointPlot.innerHTML = "";
        await Plotly.newPlot(refs.cutpointPlot, scanFigure.data, scanFigure.layout, plotConfig("cutpoint_scan"));
      }
    } catch { /* scan plot is optional */ }
  }

  if (shouldRefreshKm) {
    try {
      await runKaplanMeier();
      if (runtime.uiMode === "guided" && runtime.guidedGoal === "km") {
        setGuidedStep(5, { scroll: false, historyMode: "replace" });
      }
    } finally {
      refs.deriveStatus.textContent = "";
      renderSharedFeatureSummary();
      renderGuidedChrome();
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
    refs.cutpointPlot.innerHTML = "";
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
  const base = currentBaseConfig();
  validateGroupingSelection();
  const requestedRiskTicks = Number(refs.riskTablePoints.value);
  setShimmer(refs.kmSummaryShell);
  setShimmer(refs.kmRiskShell);
  const payload = await fetchJSON("/api/kaplan-meier", {
    method: "POST",
    body: JSON.stringify({
      ...base,
      confidence_level: Number(refs.confidenceLevel.value),
      risk_table_points: requestedRiskTicks,
      show_confidence_bands: refs.showConfidenceBands.checked,
      logrank_weight: refs.logrankWeight.value,
      fh_p: Number(refs.fhPower.value),
    }),
  });
  state.km = payload;
  purgePlot(refs.kmPlot);
  refs.kmPlot.innerHTML = "";
  await Plotly.newPlot(refs.kmPlot, payload.figure.data, payload.figure.layout, plotConfig("km_curve"));
  stabilizePlotShellHeight(refs.kmPlot);
  updateStepIndicator(3);
  renderTable(refs.kmSummaryShell, payload.analysis.summary_table);
  renderTable(refs.kmRiskShell, payload.analysis.risk_table.rows, payload.analysis.risk_table.columns);
  flashPresetTargets([refs.kmRiskShell]);
  renderTable(refs.kmPairwiseShell, payload.analysis.pairwise_table);
  const kmSummary = payload.analysis.scientific_summary;
  renderInsightBoard(refs.kmInsightBoard, kmSummary, "Run KM to generate an interpretation panel.");
  const cohort = payload.analysis.cohort;
  const test = payload.analysis.test;
  const kmCompetingRiskPrefix = summaryHasCaution(kmSummary, "competing risk")
    ? "Competing risks not modeled; 1-KM is not cumulative incidence when competing events can preclude the endpoint. "
    : "";
  refs.kmMetaBanner.textContent = `${kmCompetingRiskPrefix}N=${cohort.n}, events=${cohort.events}, censored=${cohort.censored}, median follow-up=${formatValue(cohort.median_follow_up)} ${base.time_unit_label}${test ? `, ${test.test} p=${formatValue(test.p_value)}` : ""}`;
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("km", {
    successMessage: `Kaplan-Meier analysis complete. Risk table updated to ${requestedRiskTicks} time points.`,
    backgroundMessage: "Kaplan-Meier finished in the background. Switch back when you are ready to review the updated curve.",
  });
}

function renderSignatureResult(analysis) {
  renderTable(refs.signatureShell, analysis.results_table);
  renderInsightBoard(refs.signatureInsightBoard, analysis.scientific_summary, "Run auto-discovery to assess robustness.");
  refs.downloadSignatureButton.disabled = analysis.results_table.length === 0;
  const best = analysis.best_split || {};
  const search = analysis.search_space || {};
  const significantFlag = best["Statistically significant"] ? "Yes" : "No";
  refs.deriveSummary.classList.remove("hidden");
  refs.deriveSummary.innerHTML = `
    <div class="signature-summary-grid">
      <div><strong>Best signature</strong><br/>${escapeHtml(best.Signature || "NA")}</div>
      <div><strong>HR (sig+ vs -)</strong><br/>${formatValue(best["Hazard ratio (signature+ vs -)"])}</div>
      <div><strong>BH-adjusted p</strong><br/>${formatValue(best["BH adjusted p"])}</div>
      <div><strong>Significant</strong><br/>${significantFlag}</div>
      <div><strong>Stability</strong><br/>${formatValue(best["Stability score"])}</div>
      <div><strong>Bootstrap support</strong><br/>${formatValue(best["Bootstrap support (p<alpha)"])}</div>
      <div><strong>Direction consistency</strong><br/>${formatValue(best["Bootstrap HR direction consistency"])}</div>
      <div><strong>Validation support</strong><br/>${formatValue(best["Validation support (p<alpha)"])}</div>
      <div><strong>Permutation p</strong><br/>${formatValue(best["Permutation p"])}</div>
      <div><strong>Tested</strong><br/>${formatValue(search.tested_combinations)}</div>
      <div><strong>Significant combos</strong><br/>${formatValue(search.significant_signatures)}</div>
      <div><strong>Operator</strong><br/>${escapeHtml(search.combination_operator || "mixed")}</div>
      <div><strong>Seed</strong><br/>${formatValue(search.random_seed)}</div>
      <div><strong>Alpha</strong><br/>${formatValue(search.significance_level)}</div>
    </div>`;
}

async function runSignatureSearch() {
  const base = currentBaseConfig();
  const candidateColumns = selectedCheckboxValues(refs.covariateChecklist);
  if (!candidateColumns.length) throw new Error("Select at least one covariate to search for signatures.");
  const requestedColumnName = validateDerivedColumnName(refs.deriveColumnName.value);
  const preservedGroup = String(refs.groupColumn?.value || "");
  const requestConfig = {
    dataset_id: state.dataset.dataset_id,
    time_column: base.time_column,
    event_column: base.event_column,
    event_positive_value: base.event_positive_value,
    candidate_columns: candidateColumns,
    max_combination_size: Number(refs.signatureMaxDepth.value),
    top_k: Number(refs.signatureTopK.value),
    min_group_fraction: Number(refs.signatureMinFraction.value),
    bootstrap_iterations: Number(refs.signatureBootstrapIterations.value),
    bootstrap_sample_fraction: 0.8,
    permutation_iterations: Number(refs.signaturePermutationIterations.value),
    validation_iterations: Number(refs.signatureValidationIterations.value),
    validation_fraction: Number(refs.signatureValidationFraction.value),
    significance_level: Number(refs.signatureSignificanceLevel.value),
    combination_operator: refs.signatureOperator.value,
    random_seed: Number(refs.signatureRandomSeed.value),
    new_column_name: requestedColumnName,
  };
  const payload = await fetchJSON("/api/discover-signature", {
    method: "POST",
    body: JSON.stringify(requestConfig),
  });
  const derivedGroupMeta = payload.signature_analysis?.derived_group || {};
  const shouldAutoApplyDerivedGroup = Boolean(derivedGroupMeta.auto_apply_recommended) && !preservedGroup;
  updateAfterDerivedDataset(payload);
  runtime.derivedColumnProvenance[payload.derived_column] = {
    outcomeInformed: Boolean(derivedGroupMeta.outcome_informed ?? derivedGroupMeta.outcomeInformed ?? true),
    recipe: derivedGroupMeta.recipe || payload.signature_analysis?.signature_recipe || {},
  };
  refreshVariableSelections();
  state.signature = {
    ...payload.signature_analysis,
    request_config: payload.signature_request_config || requestConfig,
  };
  if (shouldAutoApplyDerivedGroup) {
    refs.groupColumn.value = payload.derived_column;
  } else {
    setSelectValueIfPresent(refs.groupColumn, preservedGroup);
  }
  updateDatasetBadge();
  renderSignatureResult(payload.signature_analysis);
  renderSharedFeatureSummary();
  renderGuidedChrome();
  queueHistorySync();
  refs.deriveStatus.textContent = shouldAutoApplyDerivedGroup
    ? `Auto-derived ${payload.derived_column}`
    : `Derived exploratory grouping ${payload.derived_column}`;
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("km", {
    successMessage: shouldAutoApplyDerivedGroup
      ? `Signature discovery complete. Group by switched to ${payload.derived_column}.`
      : `Signature discovery complete. ${payload.derived_column} was saved but not auto-applied because the top split did not pass the current significance rules.`,
    backgroundMessage: "Signature discovery finished in the background. Switch back when you are ready to review the updated signature ranking.",
  });
}

async function runCox() {
  const base = currentBaseConfig();
  const covariates = selectedCheckboxValues(refs.covariateChecklist);
  if (!covariates.length) { showToast("Select at least one covariate for the Cox model.", "error"); return; }
  const categoricalCovariates = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => covariates.includes(v));
  setShimmer(refs.coxResultsShell);
  const payload = await fetchJSON("/api/cox", {
    method: "POST",
    body: JSON.stringify({ ...base, covariates, categorical_covariates: categoricalCovariates }),
  });
  state.cox = payload;
  purgePlot(refs.coxPlot);
  refs.coxPlot.innerHTML = "";
  await Plotly.newPlot(refs.coxPlot, payload.figure.data, payload.figure.layout, plotConfig("cox_forest"));
  stabilizePlotShellHeight(refs.coxPlot);
  stabilizeCoxPlotResetAxes(refs.coxPlot);
  if (payload.diagnostics_figure?.data?.length) {
    purgePlot(refs.coxDiagnosticsPlot);
    refs.coxDiagnosticsPlot.innerHTML = "";
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
  updateStepIndicator(3);
  renderTable(refs.coxResultsShell, payload.analysis.results_table);
  renderTable(refs.coxDiagnosticsShell, payload.analysis.diagnostics_table);
  const coxSummary = payload.analysis.scientific_summary;
  renderInsightBoard(refs.coxInsightBoard, coxSummary, "Run Cox PH to review diagnostics.");
  const stats = payload.analysis.model_stats;
  const coxMetricLabel = stats.c_index_label || ((stats.evaluation_mode === "apparent") ? "Apparent C-index" : "C-index");
  const coxMetricCore = `${coxMetricLabel}=${formatValue(stats.c_index)}`;
  const hasCoxMetricCi = stats.c_index_ci_lower != null && stats.c_index_ci_upper != null;
  const coxMetricCi = hasCoxMetricCi
    ? ` (${Math.round((Number(stats.c_index_ci_level) || 0.95) * 100)}% CI ${formatValue(stats.c_index_ci_lower)} to ${formatValue(stats.c_index_ci_upper)})`
    : "";
  const coxCompetingRiskPrefix = summaryHasCaution(coxSummary, "competing risk")
    ? "Competing risks not modeled; cause-specific questions need dedicated competing-risk methods. "
    : "";
  refs.coxMetaBanner.textContent = `${coxCompetingRiskPrefix}N=${stats.n}, events=${stats.events}, parameters=${stats.parameters}, EPV=${formatValue(stats.events_per_parameter)}, ${coxMetricCore}${coxMetricCi}, AIC=${formatValue(stats.aic, { scientificLarge: false })}`;
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("cox", {
    successMessage: "Cox PH model fitted.",
    backgroundMessage: "Cox PH finished in the background. Switch back when you are ready to review the updated model.",
  });
}

async function runCohortTable() {
  validateGroupingSelection();
  const vars = selectedCheckboxValues(refs.cohortVariableChecklist);
  if (!vars.length) { showToast("Select at least one variable for the cohort table.", "error"); return; }
  setShimmer(refs.cohortTableShell);
  const payload = await fetchJSON("/api/cohort-table", {
    method: "POST",
    body: JSON.stringify({ dataset_id: state.dataset.dataset_id, variables: vars, group_column: refs.groupColumn.value || null }),
  });
  state.cohort = payload;
  renderTable(refs.cohortTableShell, payload.analysis.rows, payload.analysis.columns);
  renderSharedFeatureSummary();
  syncDownloadButtonAvailability();
  updateStepIndicator(3);
  revealCompletedResultIfCurrent("tables", {
    successMessage: "Cohort table built.",
    backgroundMessage: "Cohort table finished in the background. Switch back when you are ready to review the updated table.",
  });
}

// ── ML Models ──────────────────────────────────────────────────

async function runMlModel() {
  runtime.resultPreference.ml = "single";
  if ((refs.mlEvaluationStrategy?.value || "holdout") === "repeated_cv") {
    throw new Error("Run Analysis uses deterministic holdout only. Switch Evaluation Mode back to Deterministic Holdout or use Compare All for repeated CV screening.");
  }
  const base = currentBaseConfig();
  const features = selectedCheckboxValues(refs.modelFeatureChecklist);
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.modelCategoricalChecklist).filter((v) => features.includes(v));
  const selectedModelType = refs.mlModelType.value;
  const modelLabel = mlModelLabel(selectedModelType);
  const computeShap = mlModelSupportsShap(selectedModelType) && !refs.mlSkipShap?.checked;
  const shapSafeMode = mlModelSupportsShap(selectedModelType) && !refs.mlSkipShap?.checked && Boolean(refs.mlShapSafeMode?.checked);
  const startedAt = performance.now();
  setShimmer(refs.mlImportancePlot);
  refs.mlMetaBanner.textContent = mlPendingBannerText({
    modelType: selectedModelType,
    nEstimators: Number(refs.mlNEstimators.value),
    rowCount: Number(state.dataset?.n_rows),
    computeShap,
  });

  const payload = await fetchJSON("/api/ml-model", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: base.dataset_id, time_column: base.time_column,
      event_column: base.event_column, event_positive_value: base.event_positive_value,
      features, categorical_features: categoricalFeatures,
      model_type: selectedModelType,
      n_estimators: Number(refs.mlNEstimators.value),
      learning_rate: Number(refs.mlLearningRate.value),
      compute_shap: computeShap,
      shap_safe_mode: shapSafeMode,
    }),
  });
  const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
  state.ml = payload;
  setPanelResultMode(refs.mlPanel, "single");
  refs.downloadMlComparisonButton.disabled = true;
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = true;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = true;
  setMlManuscriptDownloadsEnabled(false);
  refs.mlComparisonShell.innerHTML = '<div class="empty-state">Run a comparison to populate the cross-model table.</div>';
  if (refs.mlComparisonTitle) refs.mlComparisonTitle.textContent = "Model Comparison";
  refs.mlManuscriptShell.innerHTML = '<div class="empty-state">Run a comparison to populate manuscript-ready rows.</div>';
  refs.mlComparisonPlot.innerHTML = "";
  refs.mlComparisonPlot.classList.add("hidden");

  if (payload.importance_figure) {
    purgePlot(refs.mlImportancePlot);
    refs.mlImportancePlot.innerHTML = "";
    await Plotly.newPlot(refs.mlImportancePlot, payload.importance_figure.data, plotLayoutConfig(payload.importance_figure.layout, "ml_importance"), plotConfig("ml_importance"));
    stabilizePlotShellHeight(refs.mlImportancePlot);
    setPlotShellState(refs.mlImportancePlot, "plot");
  } else {
    clearPlotShell(refs.mlImportancePlot, '<div class="empty-state plot-empty"><span>No feature importance available</span></div>');
  }
  if (payload.shap_figure) {
    purgePlot(refs.mlShapPlot);
    refs.mlShapPlot.innerHTML = "";
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
  updateStepIndicator(3);
  revealCompletedResultIfCurrent("ml", {
    mode: "single",
    successMessage: `${modelLabel} model trained`,
      backgroundMessage: `${modelLabel} model finished in the background. Open Predictive Models when you are ready to review it.`,
  });
}

async function runCompareModels({ suppressCompletionToast = false } = {}) {
  runtime.resultPreference.ml = "compare";
  const base = currentBaseConfig();
  const features = selectedCheckboxValues(refs.modelFeatureChecklist);
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.modelCategoricalChecklist).filter((v) => features.includes(v));
  refs.mlMetaBanner.textContent = mlComparePendingBannerText({
    rowCount: base.row_count,
    evaluationStrategy: refs.mlEvaluationStrategy.value,
    cvFolds: Number(refs.mlCvFolds.value),
    cvRepeats: Number(refs.mlCvRepeats.value),
  });
  setRuntimeBanner("Screening Cox PH and, when available, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival on one shared evaluation path. This can take a little while on larger cohorts.", "info");
  setShimmer(refs.mlComparisonShell);

  try {
    const payload = await fetchJSON("/api/ml-model", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: base.dataset_id, time_column: base.time_column,
        event_column: base.event_column, event_positive_value: base.event_positive_value,
        features,
        categorical_features: categoricalFeatures,
        model_type: "compare",
        n_estimators: Number(refs.mlNEstimators.value),
        learning_rate: Number(refs.mlLearningRate.value),
        evaluation_strategy: refs.mlEvaluationStrategy.value,
        cv_folds: Number(refs.mlCvFolds.value),
        cv_repeats: Number(refs.mlCvRepeats.value),
      }),
    });
    state.ml = payload;
    runtime.compareCache.ml = payload;
    setPanelResultMode(refs.mlPanel, "compare");

    if (payload.analysis?.comparison_table) {
      const mlDisplayCols = ["model", "c_index", "evaluation_mode", "n_features", "training_time_ms", "rank"];
      const mlCols = mlDisplayCols.filter((c) => payload.analysis.comparison_table[0]?.[c] !== undefined);
      renderTable(refs.mlComparisonShell, payload.analysis.comparison_table, mlCols);
    }
    if (payload.analysis?.manuscript_tables?.model_performance_table) {
      renderTable(refs.mlManuscriptShell, payload.analysis.manuscript_tables.model_performance_table);
    }
    if (payload.figure) {
      refs.mlComparisonPlot.classList.remove("hidden");
      refs.mlComparisonPlot.innerHTML = "";
      await Plotly.newPlot(refs.mlComparisonPlot, payload.figure.data, payload.figure.layout, plotConfig("model_comparison"));
      stabilizePlotShellHeight(refs.mlComparisonPlot);
    }
    renderInsightBoard(refs.mlInsightBoard, payload.analysis?.scientific_summary, "Model comparison.");
    const comparisonRows = payload.analysis?.comparison_table || [];
    const bestRow = comparisonRows[0] || {};
    const evaluationMode = payload.analysis?.evaluation_mode || refs.mlEvaluationStrategy.value;
    const repeatedCvLike = evaluationMode === "repeated_cv" || evaluationMode === "repeated_cv_incomplete";
    const evalLabel = evaluationMode === "repeated_cv"
      ? `${payload.analysis?.cv_repeats || refs.mlCvRepeats.value}x${payload.analysis?.cv_folds || refs.mlCvFolds.value} repeated CV`
      : (evaluationMode === "repeated_cv_incomplete"
        ? `${payload.analysis?.cv_repeats || refs.mlCvRepeats.value}x${payload.analysis?.cv_folds || refs.mlCvFolds.value} repeated CV (incomplete)`
        : evaluationMode);
    const mlMetricLabel = repeatedCvLike ? "Mean C-index" : "C-index";
    refs.mlMetaBanner.textContent = `Screening top model=${formatValue(bestRow.model)}, ${mlMetricLabel}=${formatValue(bestRow.c_index)}, eval=${formatValue(evalLabel)}, models=${formatValue(comparisonRows.length)}`;
    refs.downloadMlComparisonButton.disabled = comparisonRows.length === 0;
    if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = !(payload.figure?.data?.length);
    if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = !(payload.figure?.data?.length);
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
    setRuntimeBanner("");
  }
}

async function runPredictiveSelectedModel() {
  runtime.workbenchRevealed = true;
  const selectedModel = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey());
  setPredictiveModel(selectedModel.key, { syncHistory: false });
  activateTab("benchmark", { setGuidedGoal: false, historyMode: "replace", syncHistory: false });
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
  const previousMlPayload = state.ml;
  const previousDlPayload = state.dl;
  setRuntimeBanner("Comparing the full predictive stack across classical ML and deep learning. This can take several minutes on larger cohorts.", "info");
  try {
    const mlAttempt = await withLoading(refs.runCompareButton, () => runCompareModels({ suppressCompletionToast: true }), "ml");
    const mlFreshCompare = Boolean(mlAttempt?.ok && benchmarkCompareRows("ml").length);
    if (!mlFreshCompare) {
      restorePredictiveFamilyAfterFailedCompare("ml", previousMlPayload);
    }

    const dlAttempt = await withLoading(refs.runDlCompareButton, () => runDlCompareModels({ suppressCompletionToast: true }), "dl");
    const dlFreshCompare = Boolean(dlAttempt?.ok && benchmarkCompareRows("dl").length);
    if (!dlFreshCompare) {
      restorePredictiveFamilyAfterFailedCompare("dl", previousDlPayload);
    }

    const familyCount = Number(mlFreshCompare) + Number(dlFreshCompare);
    setPredictiveWorkbenchFamily(startFamily, { syncHistory: false });
    activateTab("benchmark", { setGuidedGoal: false, historyMode: "replace", syncHistory: false });
    renderBenchmarkBoard();
    if (familyCount === 2) {
      showToast("Unified predictive comparison complete.", "success", 3200);
    } else if (familyCount === 1) {
      showToast("Predictive comparison finished, but only one model family returned comparison rows. Review the board and any error messages before trusting the result.", "warning", 4200);
    } else {
      showToast("Predictive comparison did not produce any fresh leaderboard rows. Review the error messages before trusting the board.", "error", 4200);
    }
  } finally {
    setRuntimeBanner("");
  }
}

// ── Deep Learning ──────────────────────────────────────────────

async function runDlModel() {
  runtime.resultPreference.dl = "single";
  const base = currentBaseConfig();
  validateDlControls();
  const features = selectedCheckboxValues(refs.modelFeatureChecklist);
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.modelCategoricalChecklist).filter((v) => features.includes(v));
  const hiddenLayers = parseHiddenLayersStrict();
  const startedAt = performance.now();

  setShimmer(refs.dlImportancePlot);
  setShimmer(refs.dlLossPlot);
  refs.dlMetaBanner.textContent = dlPendingBannerText({
    modelType: refs.dlModelType.value,
    rowCount: Number(state.dataset?.n_rows),
    epochs: Number(refs.dlEpochs.value),
    evaluationStrategy: refs.dlEvaluationStrategy.value,
    cvFolds: Number(refs.dlCvFolds.value),
    cvRepeats: Number(refs.dlCvRepeats.value),
  });
  setRuntimeBanner("Training the selected deep-learning model. This can take noticeably longer than a classical fit.", "info");

  try {
    const payload = await fetchJSON("/api/deep-model", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: base.dataset_id, time_column: base.time_column,
        event_column: base.event_column, event_positive_value: base.event_positive_value,
        features, categorical_features: categoricalFeatures,
        model_type: refs.dlModelType.value,
        hidden_layers: hiddenLayers,
        dropout: Number(refs.dlDropout.value),
        learning_rate: Number(refs.dlLearningRate.value),
        epochs: Number(refs.dlEpochs.value),
        batch_size: Number(refs.dlBatchSize.value),
        random_seed: Number(refs.dlRandomSeed.value),
        early_stopping_patience: Number(refs.dlEarlyStoppingPatience.value),
        early_stopping_min_delta: Number(refs.dlEarlyStoppingMinDelta.value),
        parallel_jobs: Number(refs.dlParallelJobs.value),
        evaluation_strategy: refs.dlEvaluationStrategy.value,
        cv_folds: Number(refs.dlCvFolds.value),
        cv_repeats: Number(refs.dlCvRepeats.value),
        num_time_bins: Number(refs.dlNumTimeBins.value),
        d_model: Number(refs.dlDModel.value),
        n_heads: Number(refs.dlHeads.value),
        n_layers: Number(refs.dlLayers.value),
        latent_dim: Number(refs.dlLatentDim.value),
        n_clusters: Number(refs.dlClusters.value),
      }),
    });
    const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    state.dl = payload;
    setPanelResultMode(refs.dlPanel, "single");
    const stats = payload.analysis || {};

    if (payload.figures?.importance) {
      purgePlot(refs.dlImportancePlot);
      refs.dlImportancePlot.innerHTML = "";
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
      purgePlot(refs.dlLossPlot);
      refs.dlLossPlot.innerHTML = "";
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
    refs.dlComparisonPlot.innerHTML = "";
    refs.dlComparisonPlot.classList.add("hidden");
    refs.downloadDlComparisonButton.disabled = !(Array.isArray(stats.comparison_table) && stats.comparison_table.length);
    if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = true;
    if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = true;
    setDlManuscriptDownloadsEnabled(!!(repeatedCvLike && payload.analysis?.manuscript_tables?.model_performance_table?.length));
    // Backend may emit either `scientific_summary` or `insight_board` depending on model implementation.
    const dlSummary = payload.analysis?.scientific_summary || payload.analysis?.insight_board || null;
    renderInsightBoard(refs.dlInsightBoard, dlSummary, "Deep learning results.");
    const epochsTrained = stats.epochs_trained || stats.epochs || refs.dlEpochs.value;
    const dlMetricLabel = stats.evaluation_mode === "repeated_cv"
      ? "Mean repeated-CV C-index"
      : (stats.evaluation_mode === "repeated_cv_incomplete"
        ? "Mean repeated-CV C-index (incomplete)"
        : (stats.evaluation_mode === "holdout"
          ? "Holdout C-index"
          : (stats.evaluation_mode === "holdout_fallback_apparent" ? "Apparent fallback C-index" : "Apparent C-index")));
    const dlEvalLabel = stats.evaluation_mode === "repeated_cv"
      ? `${formatValue(stats.cv_repeats || refs.dlCvRepeats.value)}x${formatValue(stats.cv_folds || refs.dlCvFolds.value)} repeated CV`
      : (stats.evaluation_mode === "repeated_cv_incomplete"
        ? `${formatValue(stats.cv_repeats || refs.dlCvRepeats.value)}x${formatValue(stats.cv_folds || refs.dlCvFolds.value)} repeated CV (incomplete; fallback folds excluded)`
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
        ? `, stopped early at epoch ${formatValue(epochsTrained)}`
        : (stats.max_epochs_requested != null && Number(epochsTrained) >= Number(stats.max_epochs_requested)
          ? `, trained to max epoch (${formatValue(stats.max_epochs_requested)})`
          : ""));
    const dlBestMonitorSuffix = repeatedCvLike
      ? ""
      : (stats.best_monitor_epoch != null ? `, best monitor epoch=${formatValue(stats.best_monitor_epoch)}` : "");
    refs.dlMetaBanner.textContent = `${refs.dlModelType.value.toUpperCase()}: ${dlMetricLabel}=${formatValue(stats.c_index)}, eval=${dlEvalLabel}, epochs=${formatValue(epochsTrained)}${dlBestMonitorSuffix}${dlTrainingStatus}${dlSeedSuffix}, time=${elapsedSeconds}s`;
    renderBenchmarkBoard();
    updateStepIndicator(3);
    revealCompletedResultIfCurrent("dl", {
      mode: repeatedCvLike ? "compare" : "single",
      successMessage: `${refs.dlModelType.value.toUpperCase()} model trained`,
      backgroundMessage: `${refs.dlModelType.value.toUpperCase()} model finished in the background. Open Predictive Models when you are ready to review it.`,
    });
  } finally {
    setRuntimeBanner("");
  }
}

async function runDlCompareModels({ suppressCompletionToast = false } = {}) {
  runtime.resultPreference.dl = "compare";
  const base = currentBaseConfig();
  validateDlControls();
  const features = selectedCheckboxValues(refs.modelFeatureChecklist);
  if (!features.length) { showToast("Select at least one ML/DL model feature.", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.modelCategoricalChecklist).filter((v) => features.includes(v));
  const hiddenLayers = parseHiddenLayersStrict();

  refs.dlMetaBanner.textContent = dlComparePendingBannerText({
    rowCount: base.row_count,
    evaluationStrategy: refs.dlEvaluationStrategy.value,
    cvFolds: Number(refs.dlCvFolds.value),
    cvRepeats: Number(refs.dlCvRepeats.value),
  });
  setRuntimeBanner("Comparing all deep-learning models. This can take noticeably longer than a single run.", "info");
  setShimmer(refs.dlComparisonShell);

  try {
    const payload = await fetchJSON("/api/deep-model", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: base.dataset_id,
        time_column: base.time_column,
        event_column: base.event_column,
        event_positive_value: base.event_positive_value,
        features,
        categorical_features: categoricalFeatures,
        model_type: "compare",
        hidden_layers: hiddenLayers,
        dropout: Number(refs.dlDropout.value),
        learning_rate: Number(refs.dlLearningRate.value),
        epochs: Number(refs.dlEpochs.value),
        batch_size: Number(refs.dlBatchSize.value),
        random_seed: Number(refs.dlRandomSeed.value),
        early_stopping_patience: Number(refs.dlEarlyStoppingPatience.value),
        early_stopping_min_delta: Number(refs.dlEarlyStoppingMinDelta.value),
        parallel_jobs: Number(refs.dlParallelJobs.value),
        evaluation_strategy: refs.dlEvaluationStrategy.value,
        cv_folds: Number(refs.dlCvFolds.value),
        cv_repeats: Number(refs.dlCvRepeats.value),
        num_time_bins: Number(refs.dlNumTimeBins.value),
        d_model: Number(refs.dlDModel.value),
        n_heads: Number(refs.dlHeads.value),
        n_layers: Number(refs.dlLayers.value),
        latent_dim: Number(refs.dlLatentDim.value),
        n_clusters: Number(refs.dlClusters.value),
      }),
    });
    state.dl = payload;
    runtime.compareCache.dl = payload;
    setPanelResultMode(refs.dlPanel, "compare");

    if (payload.analysis?.comparison_table?.length) {
      if (refs.dlComparisonTitle) refs.dlComparisonTitle.textContent = "Deep Model Comparison";
      const dlDisplayCols = ["model", "c_index", "evaluation_mode", "epochs_trained", "n_features", "training_time_ms", "rank"];
      const dlCols = dlDisplayCols.filter((c) => payload.analysis.comparison_table[0]?.[c] !== undefined);
      renderTable(refs.dlComparisonShell, payload.analysis.comparison_table, dlCols);
    }
    if (payload.analysis?.manuscript_tables?.model_performance_table) {
      renderTable(refs.dlManuscriptShell, payload.analysis.manuscript_tables.model_performance_table);
    }
    if (payload.figures?.comparison) {
      refs.dlComparisonPlot.classList.remove("hidden");
      refs.dlComparisonPlot.innerHTML = "";
      await Plotly.newPlot(refs.dlComparisonPlot, payload.figures.comparison.data, payload.figures.comparison.layout, plotConfig("dl_model_comparison"));
      stabilizePlotShellHeight(refs.dlComparisonPlot);
    }
    refs.dlImportancePlot.innerHTML = '<div class="empty-state plot-empty"><span>Single-model feature importance appears when you train one deep model.</span></div>';
    refs.dlLossPlot.innerHTML = '<div class="empty-state plot-empty"><span>Single-model training and monitor metric curves appear when you train one deep model.</span></div>';
    const dlSummary = payload.analysis?.scientific_summary || payload.analysis?.insight_board || null;
    renderInsightBoard(refs.dlInsightBoard, dlSummary, "Deep learning comparison results.");
    const bestRow = payload.analysis?.comparison_table?.[0] || {};
    const dlEvalMode = payload.analysis?.evaluation_mode || refs.dlEvaluationStrategy.value;
    const dlEvalLabel = dlEvalMode === "repeated_cv"
      ? `${payload.analysis?.cv_repeats || refs.dlCvRepeats.value}x${payload.analysis?.cv_folds || refs.dlCvFolds.value} repeated CV`
      : (dlEvalMode === "repeated_cv_incomplete"
        ? `${payload.analysis?.cv_repeats || refs.dlCvRepeats.value}x${payload.analysis?.cv_folds || refs.dlCvFolds.value} repeated CV (incomplete)`
        : (dlEvalMode === "mixed_holdout_apparent"
          ? "mixed holdout/apparent"
          : formatValue(dlEvalMode)));
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
    refs.dlMetaBanner.textContent = `${dlBestLabel}=${formatValue(bestRow.model)}, ${dlMetricLabel}=${formatValue(bestRow.c_index)}, eval=${formatValue(dlEvalLabel)}, models=${formatValue(payload.analysis?.comparison_table?.length || 0)}${rerunSeedSuffix}${repeatedCvRerunNote}`;
    refs.downloadDlComparisonButton.disabled = !(payload.analysis?.comparison_table?.length);
    if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = !(payload.figures?.comparison?.data?.length);
    if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = !(payload.figures?.comparison?.data?.length);
    setDlManuscriptDownloadsEnabled(!!(payload.analysis?.manuscript_tables?.model_performance_table?.length));
    renderBenchmarkBoard();
    updateStepIndicator(3);
    if (!suppressCompletionToast) {
      revealCompletedResultIfCurrent("dl", {
        mode: "compare",
        successMessage: "Deep learning model comparison complete",
        backgroundMessage: "Deep learning model comparison finished in the background. Open Predictive Models when you are ready to review it.",
      });
    }
  } finally {
    setRuntimeBanner("");
  }
}

// ── Downloads ──────────────────────────────────────────────────

function wireDownloads() {
  refs.downloadKmSummaryButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
    downloadCsv(buildDownloadFilename("km_summary", "csv", { includeGroup: true }), payload.analysis.summary_table);
  });
  refs.downloadKmPairwiseButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
    downloadCsv(buildDownloadFilename("km_pairwise", "csv", { includeGroup: true }), payload.analysis.pairwise_table);
  });
  refs.downloadSignatureButton.addEventListener("click", () => {
    const payload = currentSignatureResult();
    if (!payload || isScopeBusy("km")) {
      showToast("Visible settings no longer match the current signature result. Run again before exporting.", "warning", 3600);
      return;
    }
    downloadCsv(buildDownloadFilename("signature_ranking", "csv"), payload.results_table);
  });
  refs.downloadCoxResultsButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    downloadCsv(buildDownloadFilename("cox_results", "csv"), payload.analysis.results_table);
  });
  refs.downloadCoxDiagnosticsButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    downloadCsv(buildDownloadFilename("cox_diagnostics", "csv"), payload.analysis.diagnostics_table);
  });
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    downloadPlotImage(refs.coxPlot, buildDownloadFilename("cox_forest", "png").replace(/\.png$/, ""), "png");
  });
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("cox");
    if (!requireCurrentResultForExport("cox", { payload })) return;
    downloadPlotImage(refs.coxPlot, buildDownloadFilename("cox_forest", "svg").replace(/\.svg$/, ""), "svg");
  });
  refs.downloadCohortTableButton.addEventListener("click", () => {
    const payload = state.cohort;
    if (!requireCurrentResultForExport("tables", { payload })) return;
    downloadCsv(buildDownloadFilename("cohort_summary", "csv", { includeGroup: true }), payload.analysis.rows, payload.analysis.columns);
  });
  refs.downloadMlComparisonButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    const rows = payload?.analysis?.comparison_table;
    if (!requireCurrentResultForExport("ml", { payload })) return;
    void downloadServerTable(buildDownloadFilename("ml_model_comparison", "csv"), {
      rows,
      format: "csv",
      style: "plain",
    }, "text/csv;charset=utf-8;").catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
    downloadPlotImage(refs.mlComparisonPlot, buildDownloadFilename("ml_model_comparison", "png").replace(/\.png$/, ""), "png");
  });
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("ml");
    if (!requireCurrentResultForExport("ml", { payload })) return;
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
    void downloadServerTable(buildDownloadFilename("dl_model_comparison", "csv"), {
      rows,
      format: "csv",
      style: "plain",
    }, "text/csv;charset=utf-8;").catch((error) => showError(errorMessageText(error, "Download failed.")));
  });
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
    downloadPlotImage(refs.dlComparisonPlot, buildDownloadFilename("dl_model_comparison", "png").replace(/\.png$/, ""), "png");
  });
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("dl");
    if (!requireCurrentResultForExport("dl", { payload })) return;
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
    downloadPlotImage(refs.kmPlot, buildDownloadFilename("km_curve", "png", { includeGroup: true }).replace(/\.png$/, ""), "png");
  });
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.addEventListener("click", () => {
    const payload = currentGoalResult("km");
    if (!requireCurrentResultForExport("km", { payload })) return;
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
  if (tab === "benchmark") return refs.runPredictiveSelectedButton;
  return null;
}

function getActiveRunAction() {
  const tab = document.querySelector(".tab-button.active")?.dataset.tab;
  if (tab === "km") return runKaplanMeier;
  if (tab === "cox") return runCox;
  if (tab === "tables") return runCohortTable;
  if (tab === "ml") return runMlModel;
  if (tab === "dl") return runDlModel;
  if (tab === "benchmark") return runPredictiveSelectedModel;
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
  renderBenchmarkBoard();
  syncPredictiveWorkbenchSingleResultVisibility();
  renderGuidedChrome();
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
    const hasInsight = hasRenderedInsight(refs.coxInsightBoard);
    const hasResults = hasRenderedTable(refs.coxResultsShell);
    const hasDiagnostics = hasRenderedTable(refs.coxDiagnosticsShell);
    const hasDiagnosticsCard = hasDiagnosticsPlot || hasDiagnostics;
    const hasAny = hasPlot || hasDiagnosticsCard || hasInsight || hasResults;

    reveal(refs.coxPlot, hasPlot);
    reveal(refs.coxDiagnosticsPlot, hasDiagnosticsPlot);
    reveal(refs.coxMetaBanner, hasAny);
    reveal(refs.coxInsightBoard, hasInsight);
    reveal(refs.coxResultsShell?.closest(".table-card"), hasResults);
    reveal(refs.coxDiagnosticsShell?.closest(".table-card"), hasDiagnosticsCard);
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
    cox: [refs.coxPlot, refs.coxDiagnosticsPlot, refs.coxResultsShell],
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

async function runGuidedGoal(tabName, button, action, { resultMode = "single", successCheck = null } = {}) {
  activateTab(tabName, { historyMode: "replace" });
  await withLoading(button, action, tabName);
  const hasResult = typeof successCheck === "function" ? Boolean(successCheck()) : Boolean(currentGoalResult(tabName));
  if (hasResult && shouldRevealCompletedResult(tabName)) {
    setGuidedStep(5, { scroll: false, historyMode: "push" });
    scrollToAnalysisResult(tabName, { mode: resultMode });
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
  if (action === "run-km") { void runGuidedGoal("km", target, runGuidedKaplanMeier); return; }
  if (action === "run-cox") { void runGuidedGoal("cox", target, runCox); return; }
  if (action === "run-ml") { void runGuidedGoal("ml", target, runMlModel, { resultMode: "single" }); return; }
  if (action === "run-ml-compare") { void runGuidedGoal("ml", target, runCompareModels, { resultMode: "compare" }); return; }
  if (action === "run-dl") { void runGuidedGoal("dl", target, runDlModel, { resultMode: "single" }); return; }
  if (action === "run-dl-compare") { void runGuidedGoal("dl", target, runDlCompareModels, { resultMode: "compare" }); return; }
  if (action === "run-predictive-selected") { void runGuidedGoal(predictiveFamilyGoal(), target, runPredictiveSelectedModel, { resultMode: "single" }); return; }
  if (action === "run-predictive-compare-all") {
    void runGuidedGoal("predictive", target, runUnifiedPredictiveComparison, {
      resultMode: "compare",
      successCheck: guidedPredictiveCompareReady,
    });
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
  // Mouse
  document.addEventListener("mouseenter", (e) => {
    const dot = e.target.closest("[data-tooltip]");
    if (!dot) return;
    activeTarget = dot;
    showTooltipAt(dot);
  }, true);
  document.addEventListener("mouseleave", (e) => {
    const dot = e.target.closest("[data-tooltip]");
    if (dot && dot === activeTarget) { hideTooltip(); activeTarget = null; }
  }, true);
  // Keyboard: focus/blur on help-dot buttons
  document.addEventListener("focusin", (e) => {
    const dot = e.target.closest("[data-tooltip]");
    if (dot) { activeTarget = dot; showTooltipAt(dot); }
  }, true);
  document.addEventListener("focusout", (e) => {
    const dot = e.target.closest("[data-tooltip]");
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
  return shellHelpers.goHome({
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
}

function initListeners() {
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
    const selectedFamily = predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey()).family;
    if (runtime.uiMode === "guided") {
      runtime.guidedGoal = "predictive";
      void runGuidedGoal(selectedFamily, refs.runPredictiveSelectedButton, runPredictiveSelectedModel, { resultMode: "single" });
      return;
    }
    withLoading(refs.runPredictiveSelectedButton, runPredictiveSelectedModel, selectedFamily);
  });
  refs.closePredictiveWorkbenchButton?.addEventListener("click", () => {
    closePredictiveWorkbench();
  });
  refs.benchmarkSummaryGrid?.addEventListener("click", (event) => {
    const modelButton = event.target.closest("[data-benchmark-model]");
    if (modelButton) {
      reviewBenchmarkModel(modelButton.dataset.benchmarkModel || currentPredictiveModelKey(), modelButton.dataset.benchmarkMode || null);
      return;
    }
    const button = event.target.closest("[data-benchmark-tab]");
    if (!button) return;
    reviewBenchmarkSourceTab(button.dataset.benchmarkTab || "ml", button.dataset.benchmarkMode || null);
  });
  refs.benchmarkComparisonShell?.addEventListener("click", (event) => {
    const paramsButton = event.target.closest("[data-benchmark-params-goal]");
    if (paramsButton) {
      showBenchmarkParams(
        paramsButton.dataset.benchmarkParamsGoal || "ml",
        paramsButton.dataset.benchmarkParamsModel || "Model",
      );
      return;
    }
    const modelButton = event.target.closest("[data-benchmark-model]");
    if (modelButton) {
      reviewBenchmarkModel(modelButton.dataset.benchmarkModel || currentPredictiveModelKey(), modelButton.dataset.benchmarkMode || null);
      return;
    }
    const button = event.target.closest("[data-benchmark-tab]");
    if (!button) return;
    reviewBenchmarkSourceTab(button.dataset.benchmarkTab || "ml", button.dataset.benchmarkMode || null);
  });
  refs.guidedPanel?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-guided-action]");
    if (!button) return;
    handleGuidedPanelAction(button);
  });
  refs.guidedRailActions?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-guided-action]");
    if (!button) return;
    handleGuidedPanelAction(button);
  });
  refs.guidedPanel?.addEventListener("change", (event) => {
    const select = event.target.closest("[data-guided-predictive-model-selector]");
    if (!select) return;
    setPredictiveModel(select.value, { historyMode: "push" });
    renderBenchmarkBoard();
    syncAnalysisRunButtonAvailability();
  });
  refs.stepIndicator?.addEventListener("click", (event) => {
    const button = event.target.closest(".step");
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
    updateTimeColumnGuidance();
    refreshVariableSelections();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    updateEventColumnGuidance();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
  });
  refs.eventColumn.addEventListener("change", () => {
    updateTimeColumnGuidance();
    updateEventPositiveOptions();
    refreshVariableSelections();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    queueHistorySync();
    scheduleCoxPreview({ delay: 0 });
  });
  refs.eventPositiveValue.addEventListener("change", () => {
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
    if (runtime.lastDerivedGroup) {
      renderDerivedGroupSummary(runtime.lastDerivedGroup.derivedColumn, runtime.lastDerivedGroup.summary);
    }
    syncDeriveControlsState();
    updateDatasetBadge();
    renderSharedFeatureSummary();
    queueHistorySync();
  });
  refs.timeUnitLabel.addEventListener("input", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.maxTime.addEventListener("input", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.confidenceLevel.addEventListener("change", () => { renderSharedFeatureSummary(); queueHistorySync(); });
  refs.covariateChecklist?.addEventListener("change", (event) => {
    const input = event.target.closest('input[type="checkbox"]');
    syncCoxCovariateSelection({
      preferredValue: input?.checked ? input.value : null,
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
  refs.selectAllCoxCovariatesButton?.addEventListener("click", () => {
    const covariates = allCheckboxValues(refs.covariateChecklist, { visibleOnly: true });
    setCheckedValues(refs.covariateChecklist, covariates);
    syncCoxCovariateSelection({ autoCategoricalValues: covariates });
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
    queueHistorySync();
  };
  refs.deriveSource?.addEventListener("change", () => { runtime.deriveDraftTouched = true; queueHistorySync(); });
  refs.deriveMethod.addEventListener("change", () => { updateMethodVisibility(); markDeriveDraftTouched(); });
  refs.deriveCutoff?.addEventListener("input", markDeriveDraftTouched);
  refs.deriveColumnName?.addEventListener("input", markDeriveDraftTouched);
  refs.logrankWeight.addEventListener("change", () => { updateWeightVisibility(); queueHistorySync(); });
  refs.mlModelType.addEventListener("change", () => {
    updateMlModelControlVisibility();
    renderPredictiveWorkbench();
    queueHistorySync();
  });
  refs.mlSkipShap?.addEventListener("change", () => {
    updateMlModelControlVisibility();
    queueHistorySync();
  });
  refs.mlEvaluationStrategy.addEventListener("change", () => { updateMlEvaluationControls(); queueHistorySync(); });
  refs.dlModelType.addEventListener("change", () => {
    updateDlModelControlVisibility();
    renderPredictiveWorkbench();
    queueHistorySync();
  });
  refs.dlEvaluationStrategy.addEventListener("change", () => { updateDlEvaluationControls(); queueHistorySync(); });
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
  ];
  changeTrackedControls.filter(Boolean).forEach((control) => {
    control.addEventListener("change", queueHistorySync);
    if (["text", "number"].includes(control.type)) control.addEventListener("input", queueHistorySync);
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
initializeRuntime();

if (refs.smartBannerClose) {
  refs.smartBannerClose.addEventListener("click", () => refs.smartBanner.classList.add("hidden"));
}

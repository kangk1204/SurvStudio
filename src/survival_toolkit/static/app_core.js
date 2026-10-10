// SurvStudio front end, part 1/8: Client state, DOM references, constants, the API client, and request tokens.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

const appState = {
  dataset: null,
  km: null,
  cox: null,
  cohort: null,
  signature: null,
  ml: null,
  dl: null,
  markers: null,
  markerValidation: null,
  // Local installations use relative paths. A hosted page may name its private HTTPS backend.
  apiBase: configuredApiBase(),
  historySyncPaused: false,
  historySyncTimer: null,
  lastDerivedGroup: null,
  deriveDraftTouched: false,
  historyRestoreToken: 0,
  derivedColumnProvenance: {},
  busyScopes: {},
  plotResizeTimer: null,
  plotResizeObserver: null,
  coxPreview: {
    key: "",
    status: "idle",
    payload: null,
    error: "",
  },
  coxPreviewTimer: null,
  requestTokens: {
    km: 0,
    cox: 0,
    coxPreview: 0,
    tables: 0,
    signature: 0,
    ml: 0,
    dl: 0,
    markers: 0,
    markerValidation: 0,
    dataset: 0,
    derive: 0,
  },
  runtimeBannerSerial: 0,
  runtimeBannerHeld: false,
  pendingUploads: 0,
  requestControllers: {},
  coxMartingaleTerm: "",
  resultPreference: {
    ml: "single",
    dl: "single",
  },
  compareCache: {
    ml: null,
    dl: null,
    unified: null,
  },
  compareRunSequence: 0,
  predictiveFamily: "ml",
  workbenchRevealed: false,
  predictiveWorkbenchIntent: null,
  timeUnitAutoLabel: true,
  resultCurrencySyncTimer: null,
};

// Keep the legacy aliases, but route all mutable client state through one object.
const state = appState;
const runtime = appState;

const shellHelpers = window.SurvStudioShell;
const downloadHelpers = window.SurvStudioDownloads;

function configuredApiBase() {
  const value = String(document.body?.dataset?.apiBase || "").trim();
  if (!value) return "";
  if (!/^https:\/\/[a-z0-9.-]+(?::\d+)?\/?$/i.test(value)) {
    throw new Error("The hosted analysis server must have an HTTPS address without a path or credentials.");
  }
  return value.replace(/\/$/, "");
}


const refs = {
  runtimeBanner: document.getElementById("runtimeBanner"),
  analysisConsistencyBanner: document.getElementById("analysisConsistencyBanner"),
  landing: document.getElementById("landing"),
  workspace: document.getElementById("workspace"),
  datasetBadge: document.getElementById("datasetBadge"),
  datasetFile: document.getElementById("datasetFile"),
  uploadButton: document.getElementById("uploadButton"),
  shutdownButton: document.getElementById("shutdownButton"),
  loadTcgaUploadReadyButton: document.getElementById("loadTcgaUploadReadyButton"),
  loadGbsg2Button: document.getElementById("loadGbsg2Button"),
  loadExampleButton: document.getElementById("loadExampleButton"),
  datasetPreviewShell: document.getElementById("datasetPreviewShell"),
  datasetIntegrityWarning: document.getElementById("datasetIntegrityWarning"),
  tabPanelsHome: document.getElementById("tabPanelsHome"),
  outcomeConfigBlock: document.getElementById("outcomeConfigBlock"),
  groupingConfigBlock: document.getElementById("groupingConfigBlock"),
  groupingDetails: document.getElementById("groupingDetails"),
  groupingSummaryText: document.getElementById("groupingSummaryText"),
  tooltipPopup: document.getElementById("tooltipPopup"),
  configStrip: document.getElementById("configStrip"),
  tabStrip: document.getElementById("tabStrip"),
  timeColumn: document.getElementById("timeColumn"),
  showAllTimeColumns: document.getElementById("showAllTimeColumns"),
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
  markerValidationScaling: document.getElementById("markerValidationScaling"),
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
  kmSummaryShell: document.getElementById("kmSummaryShell"),
  kmRiskShell: document.getElementById("kmRiskShell"),
  kmPairwiseShell: document.getElementById("kmPairwiseShell"),
  signatureInsightBoard: document.getElementById("signatureInsightBoard"),
  signatureSummary: document.getElementById("signatureSummary"),
  signatureShell: document.getElementById("signatureShell"),
  downloadKmSummaryButton: document.getElementById("downloadKmSummaryButton"),
  downloadKmPairwiseButton: document.getElementById("downloadKmPairwiseButton"),
  downloadKmPngButton: document.getElementById("downloadKmPngButton"),
  downloadKmSvgButton: document.getElementById("downloadKmSvgButton"),
  covariateChecklist: document.getElementById("covariateChecklist"),
  categoricalChecklist: document.getElementById("categoricalChecklist"),
  strataChecklist: document.getElementById("strataChecklist"),
  covariateSearchInput: document.getElementById("covariateSearchInput"),
  categoricalSearchInput: document.getElementById("categoricalSearchInput"),
  strataSearchInput: document.getElementById("strataSearchInput"),
  coxCovariateWarning: document.getElementById("coxCovariateWarning"),
  selectAllCoxCovariatesButton: document.getElementById("selectAllCoxCovariatesButton"),
  clearCoxCovariatesButton: document.getElementById("clearCoxCovariatesButton"),
  selectAllCoxCategoricalsButton: document.getElementById("selectAllCoxCategoricalsButton"),
  clearCoxCategoricalsButton: document.getElementById("clearCoxCategoricalsButton"),
  selectAllCoxStrataButton: document.getElementById("selectAllCoxStrataButton"),
  clearCoxStrataButton: document.getElementById("clearCoxStrataButton"),
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
  coxPreviewLine: document.getElementById("coxPreviewLine"),
  runMarkersButton: document.getElementById("runMarkersButton"),
  downloadMarkersCsvButton: document.getElementById("downloadMarkersCsvButton"),
  downloadMarkerRecipeButton: document.getElementById("downloadMarkerRecipeButton"),
  downloadMarkerRemarkDocxButton: document.getElementById("downloadMarkerRemarkDocxButton"),
  downloadMarkerRemarkMarkdownButton: document.getElementById("downloadMarkerRemarkMarkdownButton"),
  downloadMarkersSummaryPngButton: document.getElementById("downloadMarkersSummaryPngButton"),
  downloadMarkersStabilityPngButton: document.getElementById("downloadMarkersStabilityPngButton"),
  downloadMarkersRankPngButton: document.getElementById("downloadMarkersRankPngButton"),
  selectAllMarkersButton: document.getElementById("selectAllMarkersButton"),
  clearMarkersButton: document.getElementById("clearMarkersButton"),
  markerSearchInput: document.getElementById("markerSearchInput"),
  markerChecklist: document.getElementById("markerChecklist"),
  markerMatrixDetails: document.getElementById("markerMatrixDetails"),
  markerMatrixIdColumn: document.getElementById("markerMatrixIdColumn"),
  markerMatrixOrientation: document.getElementById("markerMatrixOrientation"),
  markerMatrixFile: document.getElementById("markerMatrixFile"),
  attachMarkerMatrixButton: document.getElementById("attachMarkerMatrixButton"),
  markerMatrixStatus: document.getElementById("markerMatrixStatus"),
  markerMatrixSummary: document.getElementById("markerMatrixSummary"),
  removeMarkerMatrixButton: document.getElementById("removeMarkerMatrixButton"),
  markerClinicalSearchInput: document.getElementById("markerClinicalSearchInput"),
  markerClinicalChecklist: document.getElementById("markerClinicalChecklist"),
  markerSelectionLine: document.getElementById("markerSelectionLine"),
  markerPermutations: document.getElementById("markerPermutations"),
  markerResamples: document.getElementById("markerResamples"),
  markerRandomSeed: document.getElementById("markerRandomSeed"),
  markerNonlinearLens: document.getElementById("markerNonlinearLens"),
  markerClinicalBasis: document.getElementById("markerClinicalBasis"),
  markerDiagnosticPolicy: document.getElementById("markerDiagnosticPolicy"),
  markersInferenceNote: document.getElementById("markersInferenceNote"),
  markersDiagnosticsTable: document.getElementById("markersDiagnosticsTable"),
  markersSummaryPlot: document.getElementById("markersSummaryPlot"),
  markersStabilityPlot: document.getElementById("markersStabilityPlot"),
  markersMetaBanner: document.getElementById("markersMetaBanner"),
  markersInsightBoard: document.getElementById("markersInsightBoard"),
  markersRankPlot: document.getElementById("markersRankPlot"),
  markersTableNote: document.getElementById("markersTableNote"),
  markersTableShell: document.getElementById("markersTableShell"),
  markerValidationSection: document.getElementById("markerValidationSection"),
  markerValidationFile: document.getElementById("markerValidationFile"),
  runMarkerValidationButton: document.getElementById("runMarkerValidationButton"),
  markerValidationSummary: document.getElementById("markerValidationSummary"),
  markerValidationPlot: document.getElementById("markerValidationPlot"),
  markerValidationShell: document.getElementById("markerValidationShell"),
  coxResultsShell: document.getElementById("coxResultsShell"),
  coxDiagnosticsPlot: document.getElementById("coxDiagnosticsPlot"),
  coxDiagnosticsShell: document.getElementById("coxDiagnosticsShell"),
  coxMartingaleVariableField: document.getElementById("coxMartingaleVariableField"),
  coxMartingaleVariableSelect: document.getElementById("coxMartingaleVariableSelect"),
  coxMartingalePlot: document.getElementById("coxMartingalePlot"),
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
  tableOutputStatusText: document.getElementById("tableOutputStatusText"),
  downloadCohortTableButton: document.getElementById("downloadCohortTableButton"),
  downloadCohortTableXlsxButton: document.getElementById("downloadCohortTableXlsxButton"),
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
  mlRandomSeed: document.getElementById("mlRandomSeed"),
  mlLockedTestWrap: document.getElementById("mlLockedTestWrap"),
  mlLockedTestToggle: document.getElementById("mlLockedTestToggle"),
  mlLockedTestFractionWrap: document.getElementById("mlLockedTestFractionWrap"),
  mlLockedTestFraction: document.getElementById("mlLockedTestFraction"),
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
  dlLockedTestWrap: document.getElementById("dlLockedTestWrap"),
  dlLockedTestToggle: document.getElementById("dlLockedTestToggle"),
  dlLockedTestFractionWrap: document.getElementById("dlLockedTestFractionWrap"),
  dlLockedTestFraction: document.getElementById("dlLockedTestFraction"),
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
  openPredictiveWorkbenchButton: document.getElementById("openPredictiveWorkbenchButton"),
  predictiveModelSelector: document.getElementById("predictiveModelSelector"),
  runPredictiveSelectedButton: document.getElementById("runPredictiveSelectedButton"),
  runPredictiveWorkbenchButton: document.getElementById("runPredictiveWorkbenchButton"),
  predictiveActionStatusText: document.getElementById("predictiveActionStatusText"),
  predictiveInputSummary: document.getElementById("predictiveInputSummary"),
  benchmarkActionCard: document.getElementById("benchmarkActionCard"),
  benchmarkSummaryGrid: document.getElementById("benchmarkSummaryGrid"),
  benchmarkComparisonPlot: document.getElementById("benchmarkComparisonPlot"),
  benchmarkPlotNote: document.getElementById("benchmarkPlotNote"),
  benchmarkComparisonShell: document.getElementById("benchmarkComparisonShell"),
  benchmarkTableNote: document.getElementById("benchmarkTableNote"),
  downloadTripodDocxButton: document.getElementById("downloadTripodDocxButton"),
  downloadTripodMarkdownButton: document.getElementById("downloadTripodMarkdownButton"),
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
const DEFAULT_TIME_UNIT_LABEL = "Time";
const DEFAULT_LOCKED_TEST_PERCENT = 30;
// Defaults of the numeric settings: the value each field starts with on the page (index.html). The server's
// own defaults can differ (its cut-point search runs no permutations or validation splits unless asked), so
// the page always sends these. Request builders, input checks and result currency all read a blank field as
// its default, so a run never sends Number("") = 0 while the result currency assumes the default (and ML and
// DL never split the patients with different seeds).
const KM_NUMERIC_DEFAULTS = Object.freeze({ confidence_level: 0.95, risk_table_points: 6, fh_p: 1 });
const ML_NUMERIC_DEFAULTS = Object.freeze({ n_estimators: 100, learning_rate: 0.1, random_state: 42, cv_folds: 5, cv_repeats: 3 });
const DL_NUMERIC_DEFAULTS = Object.freeze({
  dropout: 0.1,
  learning_rate: 0.001,
  epochs: 100,
  batch_size: 64,
  random_seed: 42,
  cv_folds: 5,
  cv_repeats: 3,
  early_stopping_patience: 10,
  early_stopping_min_delta: 0.0001,
  parallel_jobs: 1,
  num_time_bins: 50,
  d_model: 64,
  n_heads: 4,
  n_layers: 2,
  latent_dim: 8,
  n_clusters: 3,
});
const SIGNATURE_NUMERIC_DEFAULTS = Object.freeze({
  max_combination_size: 3,
  top_k: 15,
  min_group_fraction: 0.1,
  bootstrap_iterations: 30,
  permutation_iterations: 120,
  validation_iterations: 12,
  validation_fraction: 0.35,
  significance_level: 0.05,
  random_seed: 20260311,
});
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

class SupersededRequestError extends Error {
  constructor(message = "The request was replaced by a newer one.") {
    super(message);
    this.name = "SupersededRequestError";
    this.superseded = true;
  }
}

function isSupersededRequestError(error) {
  return Boolean(error?.superseded) || error?.name === "AbortError";
}

// An error of an answered request keeps its HTTP status, so a caller can tell a refused request (4xx, which
// would be refused again) from a server failure (5xx); a network failure has no status.
function httpError(message, status) {
  const error = new Error(message);
  error.status = status;
  return error;
}

async function fetchJSON(url, options = {}) {
  // Spread the caller's options first so their own headers cannot drop the JSON content type.
  const { headers: callerHeaders, ...fetchOptions } = options;
  let response;
  try {
    response = await fetch(apiUrl(url), {
      ...fetchOptions,
      headers: {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(callerHeaders || {}),
      },
    });
  } catch (error) {
    if (error?.name === "AbortError") throw new SupersededRequestError();
    if (runtime.apiBase) {
      throw new Error("Cannot reach your private analysis server. Check that Tailscale is connected, then try again.");
    }
    throw error;
  }
  let rawText;
  try {
    rawText = await response.text();
  } catch (error) {
    if (error?.name === "AbortError") throw new SupersededRequestError();
    throw error;
  }
  let payload = {};
  if (rawText.trim()) {
    try {
      payload = JSON.parse(rawText);
    } catch (error) {
      const genericMessage = response.ok
        ? "The server returned an invalid JSON response."
        : (rawText.trim() || "Request failed.");
      throw httpError(genericMessage, response.status);
    }
  }
  if (!response.ok) {
    const message = extractErrorMessage(payload, rawText);
    // Only the loss of the loaded cohort itself ends the workspace; a 404 for another dataset (an older
    // snapshot, an external validation cohort, a history entry) is an ordinary error.
    const missingDatasetId = response.status === 404 ? /Unknown dataset id:\s*([\w-]+)/i.exec(message)?.[1] : null;
    if (missingDatasetId && state.dataset && missingDatasetId === String(state.dataset.dataset_id)) {
      const datasetName = state.dataset?.filename || state.dataset?.dataset_id || "current cohort";
      goHome({ syncHistory: true, historyMode: "replace" });
      setRuntimeBanner(`The previously loaded cohort (${datasetName}) is no longer available on the server. Reload it to continue.`, "warning");
      throw httpError("The loaded dataset is no longer available on the server. Reload a dataset and run the analysis again.", response.status);
    }
    throw httpError(message, response.status);
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

// Returns a serial that identifies this banner, so a run can later clear the banner it set without
// wiping a newer one (releaseRuntimeBanner). A run in flight sets its progress banner `held`; starting
// another action clears only a banner nobody holds, such as the notice of a finished load (clearRuntimeNotice).
function setRuntimeBanner(text = "", tone = "info", { held = false } = {}) {
  runtime.runtimeBannerSerial = Number(runtime.runtimeBannerSerial || 0) + 1;
  runtime.runtimeBannerHeld = Boolean(text) && Boolean(held);
  if (!refs.runtimeBanner) return runtime.runtimeBannerSerial;
  if (!text) {
    refs.runtimeBanner.textContent = "";
    refs.runtimeBanner.className = "runtime-banner hidden";
    return runtime.runtimeBannerSerial;
  }
  refs.runtimeBanner.textContent = text;
  refs.runtimeBanner.className = `runtime-banner runtime-banner-${tone}`;
  return runtime.runtimeBannerSerial;
}

// Clears the banner only while it is still the one that `serial` set.
function releaseRuntimeBanner(serial) {
  if (serial && serial === runtime.runtimeBannerSerial) setRuntimeBanner("");
}

// A newly started action clears a leftover notice, but not the progress banner of a run still in flight.
function clearRuntimeNotice() {
  if (!runtime.runtimeBannerHeld) setRuntimeBanner("");
}

function renderServerStoppedState(message) {
  const resolvedMessage = message || "SurvStudio is stopping. You can close this tab or restart the server with `survstudio`.";
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
  commandCode.textContent = "survstudio";
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

// The mode (one model, or a comparison) of the ML or DL result the page holds, or of the last finished run
// when it holds none. Currency and visibility follow the result actually held, so a run that fails (or is
// still running) never hides the still-valid result of the other mode.
function preferredResultMode(goal) {
  if (goal !== "ml" && goal !== "dl") return "single";
  const payload = goal === "ml" ? state.ml : state.dl;
  if (payload) return payloadRepresentsCompareRun(payload) ? "compare" : "single";
  return runtime.resultPreference?.[goal] || "single";
}

const ANALYSIS_GOALS = ["km", "cox", "markers", "predictive", "tables", "ml", "dl"];

function predictiveFamilyGoal() {
  return normalizedPredictiveFamily(runtime.predictiveFamily);
}

function goalLabel(goal) {
  return {
    km: "Kaplan-Meier",
    cox: "Cox PH",
    markers: "Marker evaluation",
    predictive: "ML/DL Models",
    ml: "ML Models",
    dl: "Deep Learning",
    tables: "Cohort Table",
  }[goal] || "Choose analysis";
}

function goalFeatureCount(goal) {
  if (goal === "cox") return currentCoxSelections().covariates.length;
  if (goal === "ml" || goal === "dl" || goal === "predictive") return selectedCheckboxValues(refs.modelFeatureChecklist).length;
  if (goal === "tables") return selectedCheckboxValues(refs.cohortVariableChecklist).length;
  return 0;
}

function numberOrDefault(value, fallback) {
  // Treat only missing/blank values as "use the default"; 0 is a legitimate setting.
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && !value.trim()) return fallback;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function numericControlValue(control, fallback) {
  return numberOrDefault(control?.value, fallback);
}

function inferTimeUnitLabel(columnName) {
  const tokens = String(columnName || "")
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter(Boolean);
  const has = (...candidates) => tokens.some((token) => candidates.includes(token));
  if (has("day", "days")) return "Days";
  if (has("week", "weeks", "wk", "wks")) return "Weeks";
  if (has("month", "months", "mo", "mos", "mon")) return "Months";
  if (has("year", "years", "yr", "yrs")) return "Years";
  return DEFAULT_TIME_UNIT_LABEL;
}

function automaticTimeUnitLabel() {
  const preset = datasetPresetForCurrentDataset();
  const timeColumn = refs.timeColumn?.value || "";
  if (preset?.timeUnitLabel && preset.timeColumn === timeColumn) return preset.timeUnitLabel;
  return inferTimeUnitLabel(timeColumn);
}

function applyAutomaticTimeUnitLabel({ force = false } = {}) {
  if (!refs.timeUnitLabel) return;
  if (!force && !runtime.timeUnitAutoLabel) return;
  refs.timeUnitLabel.value = automaticTimeUnitLabel();
  runtime.timeUnitAutoLabel = true;
}

function sharedPredictiveSeed() {
  // ML (random_state) and DL (random_seed) share one seed so both families use the same row partitions.
  return numericControlValue(refs.dlRandomSeed, DL_NUMERIC_DEFAULTS.random_seed);
}

function normalizedLockedTestFraction(value) {
  const numeric = numberOrDefault(value, null);
  if (numeric === null || numeric <= 0) return null;
  return Math.round(numeric * 10000) / 10000;
}

function lockedTestControls(goal = "ml") {
  return goal === "dl"
    ? { strategy: refs.dlEvaluationStrategy, toggle: refs.dlLockedTestToggle, input: refs.dlLockedTestFraction }
    : { strategy: refs.mlEvaluationStrategy, toggle: refs.mlLockedTestToggle, input: refs.mlLockedTestFraction };
}

function currentLockedTestFraction(goal = "ml") {
  const { strategy, toggle, input } = lockedTestControls(goal);
  if ((strategy?.value || "holdout") !== "repeated_cv" || !toggle?.checked) return null;
  return normalizedLockedTestFraction(numericControlValue(input, DEFAULT_LOCKED_TEST_PERCENT) / 100);
}

function validatePredictiveEvaluationControls(goal = "ml", { includeLockedTest = true } = {}) {
  // Both families run with sharedPredictiveSeed(), so that is the seed to check.
  const seed = sharedPredictiveSeed();
  if (!Number.isFinite(seed) || !Number.isInteger(seed) || seed < 0) {
    throw new Error(`Random seed must be a non-negative integer. Current value: ${formatValue(seed)}.`);
  }
  const { strategy, toggle, input } = lockedTestControls(goal);
  if ((strategy?.value || "holdout") !== "repeated_cv") return;
  const folds = numericControlValue(goal === "dl" ? refs.dlCvFolds : refs.mlCvFolds, ML_NUMERIC_DEFAULTS.cv_folds);
  if (!Number.isInteger(folds) || folds < 2 || folds > 10) {
    throw new Error(`CV folds must be an integer between 2 and 10. Current value: ${formatValue(folds)}.`);
  }
  const repeats = numericControlValue(goal === "dl" ? refs.dlCvRepeats : refs.mlCvRepeats, ML_NUMERIC_DEFAULTS.cv_repeats);
  if (!Number.isInteger(repeats) || repeats < 1 || repeats > 20) {
    throw new Error(`CV repeats must be an integer between 1 and 20. Current value: ${formatValue(repeats)}.`);
  }
  if (includeLockedTest && toggle?.checked) {
    const percent = numericControlValue(input, DEFAULT_LOCKED_TEST_PERCENT);
    if (!Number.isFinite(percent) || percent < 5 || percent > 50) {
      throw new Error(`Locked test set must be between 5% and 50% of patients. Current value: ${formatValue(percent)}%.`);
    }
  }
}

const PREDICTIVE_EVALUATION_MIRRORS = [
  ["mlEvaluationStrategy", "dlEvaluationStrategy"],
  ["mlCvFolds", "dlCvFolds"],
  ["mlCvRepeats", "dlCvRepeats"],
  ["mlRandomSeed", "dlRandomSeed"],
  ["mlLockedTestToggle", "dlLockedTestToggle"],
  ["mlLockedTestFraction", "dlLockedTestFraction"],
];

function mirrorPredictiveEvaluationControl(source) {
  // Evaluation mode, CV design, seed, and the locked test set are shared by ML and DL so
  // Compare All Models evaluates both families on identical row partitions.
  if (!source) return false;
  let changed = false;
  PREDICTIVE_EVALUATION_MIRRORS.forEach(([mlKey, dlKey]) => {
    const target = source === refs[mlKey] ? refs[dlKey] : (source === refs[dlKey] ? refs[mlKey] : null);
    if (!target) return;
    if (source.type === "checkbox") {
      if (target.checked !== source.checked) {
        target.checked = source.checked;
        changed = true;
      }
    } else if (target.value !== source.value) {
      target.value = source.value;
      changed = true;
    }
  });
  return changed;
}

function alignPredictiveEvaluationControls(sourceFamily = "ml") {
  PREDICTIVE_EVALUATION_MIRRORS.forEach(([mlKey, dlKey]) => {
    mirrorPredictiveEvaluationControl(refs[sourceFamily === "dl" ? dlKey : mlKey]);
  });
  updateMlEvaluationControls();
  updateDlEvaluationControls();
}

function arrayEquals(left = [], right = []) {
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
}

function abortScopeRequest(scope) {
  const controller = runtime.requestControllers?.[scope];
  if (controller) controller.abort();
  if (runtime.requestControllers) delete runtime.requestControllers[scope];
}

function beginRequestToken(scope) {
  if (!scope) return 0;
  // A newer request of the same kind replaces the older one: cancel it so the server
  // stops working on a result nobody will read.
  abortScopeRequest(scope);
  runtime.requestTokens[scope] = Number(runtime.requestTokens?.[scope] || 0) + 1;
  if (typeof AbortController === "function") {
    runtime.requestControllers = runtime.requestControllers || {};
    runtime.requestControllers[scope] = new AbortController();
  }
  return runtime.requestTokens[scope];
}

function requestSignal(scope) {
  return runtime.requestControllers?.[scope]?.signal;
}

function requestTokenMatches(scope, token) {
  if (!scope) return false;
  return Number(runtime.requestTokens?.[scope] || 0) === Number(token || 0);
}

function invalidateRequestTokens(scopes = []) {
  scopes.forEach((scope) => {
    if (!scope) return;
    abortScopeRequest(scope);
    runtime.requestTokens[scope] = Number(runtime.requestTokens?.[scope] || 0) + 1;
  });
}

function sortedStrings(values = []) {
  return [...values].map((value) => String(value)).sort();
}

function currentCoxSelections() {
  const covariates = selectedCheckboxValues(refs.covariateChecklist);
  const strataColumns = selectedCheckboxValues(refs.strataChecklist);
  return {
    covariates,
    categoricalCovariates: selectedCheckboxValues(refs.categoricalChecklist).filter((value) => covariates.includes(value)),
    strataColumns,
  };
}

function coxPreviewRequestFromCurrentState() {
  const { covariates, categoricalCovariates, strataColumns } = currentCoxSelections();
  if (!covariates.length) return null;
  const base = currentBaseConfig();
  return {
    dataset_id: base.dataset_id,
    time_column: base.time_column,
    event_column: base.event_column,
    event_positive_value: base.event_positive_value,
    covariates,
    categorical_covariates: categoricalCovariates,
    strata_columns: strataColumns,
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
    strata_columns: sortedStrings(requestConfig.strata_columns || []),
  });
}

function resetCoxPreview({ rerender = true } = {}) {
  if (runtime.coxPreviewTimer) {
    window.clearTimeout(runtime.coxPreviewTimer);
    runtime.coxPreviewTimer = null;
  }
  // Cancels a preview request still in flight; its answer would describe settings that are gone.
  invalidateRequestTokens(["coxPreview"]);
  runtime.coxPreview = {
    key: "",
    status: "idle",
    payload: null,
    error: "",
  };
  if (rerender) renderCoxPreviewLine();
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
    renderCoxPreviewLine();
    return;
  }
  if (!requestConfig) {
    resetCoxPreview({ rerender: false });
    renderCoxPreviewLine();
    return;
  }
  const requestKey = coxPreviewRequestKey(requestConfig);
  if (!force && runtime.coxPreview.key === requestKey && runtime.coxPreview.status === "ready") return;
  // A newer preview cancels the one still in flight, so the server stops counting rows nobody will read.
  const previewToken = beginRequestToken("coxPreview");
  runtime.coxPreview = {
    key: requestKey,
    status: "loading",
    payload: null,
    error: "",
  };
  renderCoxPreviewLine();
  try {
    const payload = await fetchJSON("/api/cox-preview", {
      signal: requestSignal("coxPreview"),
      method: "POST",
      body: JSON.stringify(requestConfig),
    });
    if (!requestTokenMatches("coxPreview", previewToken)) return;
    runtime.coxPreview = {
      key: requestKey,
      status: "ready",
      payload,
      error: "",
    };
  } catch (error) {
    if (!requestTokenMatches("coxPreview", previewToken) || isSupersededRequestError(error)) return;
    runtime.coxPreview = {
      key: requestKey,
      status: "error",
      payload: null,
      error: error.message || "Cox preview is unavailable.",
    };
  }
  renderCoxPreviewLine();
}

// The key of the preview request the controls describe now; "" when there is none (no covariate, or an
// endpoint that is not ready).
function currentCoxPreviewKey() {
  try {
    return coxPreviewRequestKey(coxPreviewRequestFromCurrentState());
  } catch {
    return "";
  }
}

function scheduleCoxPreview({ delay = 180, force = false } = {}) {
  if (runtime.coxPreviewTimer) {
    window.clearTimeout(runtime.coxPreviewTimer);
  }
  // A preview (or a message) for other settings leaves the line at once instead of after the debounce, and its
  // request is cancelled, so the line never describes settings the controls no longer show.
  if (runtime.coxPreview.status !== "idle" && runtime.coxPreview.key !== currentCoxPreviewKey()) {
    resetCoxPreview();
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

function setAnalysisConsistencyBanner(text = "", tone = "warning") {
  if (!refs.analysisConsistencyBanner) return;
  if (!text) {
    refs.analysisConsistencyBanner.textContent = "";
    refs.analysisConsistencyBanner.className = "runtime-banner hidden";
    return;
  }
  refs.analysisConsistencyBanner.textContent = text;
  refs.analysisConsistencyBanner.className = `runtime-banner runtime-banner-${tone}`;
}

function analysisCohortFingerprint(goal, payload) {
  if (!payload) return null;
  const analysis = payload?.analysis || payload;
  const datasetHash = String(payload?.dataset_hash || state.dataset?.dataset_hash || "");
  if (goal === "km") {
    const cohort = analysis?.cohort || {};
    if (!cohort.row_mask_hash) return null;
    return { goal, label: "KM", n: Number(cohort.n), rowMaskHash: String(cohort.row_mask_hash), datasetHash };
  }
  if (goal === "cox") {
    const stats = analysis?.model_stats || {};
    if (!stats.row_mask_hash) return null;
    return { goal, label: "Cox", n: Number(stats.n), rowMaskHash: String(stats.row_mask_hash), datasetHash };
  }
  if (goal === "signature") {
    const searchSpace = analysis?.search_space || {};
    if (!searchSpace.row_mask_hash) return null;
    return {
      goal,
      label: "Signature",
      n: Number(searchSpace.n_rows_analyzed),
      rowMaskHash: String(searchSpace.row_mask_hash),
      datasetHash,
    };
  }
  if (goal === "tables") {
    if (!analysis?.row_mask_hash) return null;
    return { goal, label: "Table", n: NaN, rowMaskHash: String(analysis.row_mask_hash), datasetHash };
  }
  return null;
}

function currentAnalysisCohortFingerprints() {
  return [
    analysisCohortFingerprint("km", state.km),
    analysisCohortFingerprint("cox", state.cox),
    analysisCohortFingerprint("signature", state.signature),
    analysisCohortFingerprint("tables", state.cohort),
  ].filter(Boolean);
}

function renderAnalysisConsistencyBanner() {
  const fingerprints = currentAnalysisCohortFingerprints();
  if (fingerprints.length < 2) {
    setAnalysisConsistencyBanner("");
    return;
  }
  const datasetHashes = new Set(fingerprints.map((item) => item.datasetHash).filter(Boolean));
  if (datasetHashes.size > 1) {
    setAnalysisConsistencyBanner(
      "Loaded analyses do not share the same dataset fingerprint. Rebuild the analyses from one cohort before comparing them side by side.",
      "warning",
    );
    return;
  }
  const rowMaskHashes = new Set(fingerprints.map((item) => item.rowMaskHash).filter(Boolean));
  if (rowMaskHashes.size <= 1) {
    setAnalysisConsistencyBanner("");
    return;
  }
  const cohortSummary = fingerprints
    .map((item) => (Number.isFinite(item.n) ? `${item.label} N=${formatValue(item.n)}` : item.label))
    .join("; ");
  setAnalysisConsistencyBanner(
    `Loaded analyses currently use different analyzable cohorts on the same dataset (${cohortSummary}), `
      + "usually because of missing values. Do not present them side by side as one cohort.",
    "warning",
  );
}

function cohortTableOutcomeConfig() {
  // With a configured endpoint the table is restricted to the same analyzable outcome rows as KM/Cox.
  if (!endpointIsReady()) return { time_column: null, event_column: null, event_positive_value: null };
  const base = currentBaseConfig();
  return {
    time_column: base.time_column,
    event_column: base.event_column,
    event_positive_value: base.event_positive_value,
  };
}

function cohortTableAnalysisNotes(payload = state.cohort) {
  const notes = payload?.analysis?.notes;
  if (Array.isArray(notes)) return notes.map((note) => String(note ?? "").trim()).filter(Boolean);
  return typeof notes === "string" && notes.trim() ? [notes.trim()] : [];
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
  let base;
  try {
    base = currentBaseConfig();
  } catch {
    return null;
  }
  const currentDerivedName = String(refs.deriveColumnName?.value || "").trim();
  // The same candidates runSignatureSearch sends: the Markers tab's markers and clinical covariates.
  const currentCandidates = sortedStrings(signatureCandidateColumns());
  const requestedCandidates = sortedStrings(requestConfig.candidate_columns || []);
  // Discovery writes its grouping into a new dataset snapshot, so the result belongs to that snapshot.
  const resultDatasetId = String(payload.result_dataset_id || requestConfig.dataset_id || "");
  const setting = (key, control) => numberOrDefault(requestConfig[key], SIGNATURE_NUMERIC_DEFAULTS[key])
    === numericControlValue(control, SIGNATURE_NUMERIC_DEFAULTS[key]);
  const isCurrent = (
    resultDatasetId === String(state.dataset.dataset_id || "")
    && String(requestConfig.time_column || "") === String(base.time_column || "")
    && String(requestConfig.event_column || "") === String(base.event_column || "")
    && String(requestConfig.event_positive_value ?? "").trim() === String(base.event_positive_value ?? "").trim()
    && String(requestConfig.new_column_name || "") === currentDerivedName
    && String(requestConfig.combination_operator || "mixed") === String(refs.signatureOperator?.value || "mixed")
    && setting("max_combination_size", refs.signatureMaxDepth)
    && setting("top_k", refs.signatureTopK)
    && setting("min_group_fraction", refs.signatureMinFraction)
    && setting("bootstrap_iterations", refs.signatureBootstrapIterations)
    && setting("permutation_iterations", refs.signaturePermutationIterations)
    && setting("validation_iterations", refs.signatureValidationIterations)
    && setting("validation_fraction", refs.signatureValidationFraction)
    && setting("significance_level", refs.signatureSignificanceLevel)
    && setting("random_seed", refs.signatureRandomSeed)
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

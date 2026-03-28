const state = {
  dataset: null,
  km: null,
  cox: null,
  cohort: null,
  signature: null,
  ml: null,
  dl: null,
};

const runtime = {
  isFilePreview: window.location.protocol === "file:",
  apiBase: window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "",
  historySyncPaused: false,
};


const refs = {
  runtimeBanner: document.getElementById("runtimeBanner"),
  landing: document.getElementById("landing"),
  workspace: document.getElementById("workspace"),
  datasetBadge: document.getElementById("datasetBadge"),
  datasetFile: document.getElementById("datasetFile"),
  uploadButton: document.getElementById("uploadButton"),
  loadTcgaUploadReadyButton: document.getElementById("loadTcgaUploadReadyButton"),
  loadTcgaButton: document.getElementById("loadTcgaButton"),
  loadGbsg2Button: document.getElementById("loadGbsg2Button"),
  loadExampleButton: document.getElementById("loadExampleButton"),
  datasetPreviewShell: document.getElementById("datasetPreviewShell"),
  stepIndicator: document.getElementById("stepIndicator"),
  smartBanner: document.getElementById("smartBanner"),
  smartBannerText: document.getElementById("smartBannerText"),
  smartBannerClose: document.getElementById("smartBannerClose"),
  datasetPresetBar: document.getElementById("datasetPresetBar"),
  datasetPresetTitle: document.getElementById("datasetPresetTitle"),
  datasetPresetText: document.getElementById("datasetPresetText"),
  datasetPresetStatusTitle: document.getElementById("datasetPresetStatusTitle"),
  datasetPresetStatusText: document.getElementById("datasetPresetStatusText"),
  datasetPresetChips: document.getElementById("datasetPresetChips"),
  applyBasicPresetButton: document.getElementById("applyBasicPresetButton"),
  applyModelPresetButton: document.getElementById("applyModelPresetButton"),
  tooltipPopup: document.getElementById("tooltipPopup"),
  configStrip: document.getElementById("configStrip"),
  timeColumn: document.getElementById("timeColumn"),
  eventColumn: document.getElementById("eventColumn"),
  eventPositiveValue: document.getElementById("eventPositiveValue"),
  groupColumn: document.getElementById("groupColumn"),
  timeUnitLabel: document.getElementById("timeUnitLabel"),
  maxTime: document.getElementById("maxTime"),
  confidenceLevel: document.getElementById("confidenceLevel"),
  deriveToggle: document.getElementById("deriveToggle"),
  derivePanel: document.getElementById("derivePanel"),
  deriveSource: document.getElementById("deriveSource"),
  deriveMethod: document.getElementById("deriveMethod"),
  deriveCutoff: document.getElementById("deriveCutoff"),
  deriveColumnName: document.getElementById("deriveColumnName"),
  cutoffWrap: document.getElementById("cutoffWrap"),
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
  runCoxButton: document.getElementById("runCoxButton"),
  coxInsightBoard: document.getElementById("coxInsightBoard"),
  coxPlot: document.getElementById("coxPlot"),
  coxMetaBanner: document.getElementById("coxMetaBanner"),
  coxResultsShell: document.getElementById("coxResultsShell"),
  coxDiagnosticsShell: document.getElementById("coxDiagnosticsShell"),
  downloadCoxResultsButton: document.getElementById("downloadCoxResultsButton"),
  downloadCoxDiagnosticsButton: document.getElementById("downloadCoxDiagnosticsButton"),
  downloadCoxPngButton: document.getElementById("downloadCoxPngButton"),
  downloadCoxSvgButton: document.getElementById("downloadCoxSvgButton"),
  cohortVariableChecklist: document.getElementById("cohortVariableChecklist"),
  runCohortTableButton: document.getElementById("runCohortTableButton"),
  cohortTableShell: document.getElementById("cohortTableShell"),
  downloadCohortTableButton: document.getElementById("downloadCohortTableButton"),
  // ML
  runMlButton: document.getElementById("runMlButton"),
  runCompareButton: document.getElementById("runCompareButton"),
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
  mlManuscriptShell: document.getElementById("mlManuscriptShell"),
  // DL
  runDlButton: document.getElementById("runDlButton"),
  runDlCompareButton: document.getElementById("runDlCompareButton"),
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
  dlEvaluationStrategy: document.getElementById("dlEvaluationStrategy"),
  dlCvFoldsWrap: document.getElementById("dlCvFoldsWrap"),
  dlCvRepeatsWrap: document.getElementById("dlCvRepeatsWrap"),
  dlCvFolds: document.getElementById("dlCvFolds"),
  dlCvRepeats: document.getElementById("dlCvRepeats"),
  dlEarlyStoppingPatience: document.getElementById("dlEarlyStoppingPatience"),
  dlEarlyStoppingMinDelta: document.getElementById("dlEarlyStoppingMinDelta"),
  dlParallelJobs: document.getElementById("dlParallelJobs"),
  dlJournalTemplate: document.getElementById("dlJournalTemplate"),
  dlFeatureSummaryCard: document.getElementById("dlFeatureSummaryCard"),
  dlFeatureSummaryText: document.getElementById("dlFeatureSummaryText"),
  dlFeatureSummaryChips: document.getElementById("dlFeatureSummaryChips"),
  reviewDlFeaturesButton: document.getElementById("reviewDlFeaturesButton"),
  dlImportancePlot: document.getElementById("dlImportancePlot"),
  dlLossPlot: document.getElementById("dlLossPlot"),
  dlComparisonPlot: document.getElementById("dlComparisonPlot"),
  dlComparisonShell: document.getElementById("dlComparisonShell"),
  dlManuscriptShell: document.getElementById("dlManuscriptShell"),
  dlMetaBanner: document.getElementById("dlMetaBanner"),
  dlInsightBoard: document.getElementById("dlInsightBoard"),
  tabButtons: [...document.querySelectorAll(".tab-button")],
  tabPanels: [...document.querySelectorAll(".tab-panel")],
};

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

async function fetchJSON(url, options = {}) {
  const response = await fetch(apiUrl(url), {
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "Request failed.");
  return payload;
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

function activeTabName() {
  return document.querySelector(".tab-button.active")?.dataset.tab || "km";
}

function currentHistoryState() {
  if (!state.dataset) return { view: "home" };
  return {
    view: "workspace",
    datasetId: state.dataset.dataset_id,
    tab: activeTabName(),
  };
}

function syncHistoryState(mode = "replace") {
  if (runtime.historySyncPaused || !window.history?.replaceState) return;
  const nextState = currentHistoryState();
  if (mode === "push") {
    window.history.pushState(nextState, "", window.location.href);
    return;
  }
  window.history.replaceState(nextState, "", window.location.href);
}

async function restoreHistoryState(historyState) {
  if (!historyState || historyState.view === "home") {
    goHome({ syncHistory: false });
    return;
  }
  if (historyState.view !== "workspace" || !historyState.datasetId) {
    goHome({ syncHistory: false });
    return;
  }

  runtime.historySyncPaused = true;
  try {
    if (!state.dataset || state.dataset.dataset_id !== historyState.datasetId) {
      const payload = await fetchJSON(`/api/dataset/${historyState.datasetId}`);
      updateAfterDataset(payload);
    } else {
      showWorkspace();
    }
    activateTab(historyState.tab || "km");
  } catch {
    goHome({ syncHistory: false });
  } finally {
    runtime.historySyncPaused = false;
  }
}

function setButtonLoading(button, isLoading) {
  button.classList.toggle("is-loading", isLoading);
  button.disabled = isLoading;
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

function updateEventPositiveOptions() {
  const eventColumn = refs.eventColumn.value;
  const meta = getColumnMeta(eventColumn);
  const values = meta?.unique_preview?.filter((value) => value !== null) ?? [];
  refs.eventPositiveValue.innerHTML = "";
  if (values.length === 0) {
    const option = document.createElement("option");
    option.value = "1";
    option.textContent = "1";
    refs.eventPositiveValue.appendChild(option);
    return;
  }
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    if (String(value) === "1" || String(value).toLowerCase() === "event") option.selected = true;
    refs.eventPositiveValue.appendChild(option);
  });
}

function renderChecklist(container, values, selected = []) {
  container.innerHTML = "";
  values.forEach((value) => {
    const label = document.createElement("label");
    label.className = "check-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = selected.includes(value);
    const span = document.createElement("span");
    span.textContent = value;
    label.append(input, span);
    container.appendChild(label);
  });
}

function selectedCheckboxValues(container) {
  return [...container.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "NA";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "NA";
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(2);
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
    strengths.length ? `<div class="insight-section"><h4>What is solid</h4><ul>${strengths.map(escapeListItem).join("")}</ul></div>` : "",
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

function renderDerivedGroupSummary(derivedColumn, summary) {
  const counts = summary?.counts || [];
  refs.deriveSummary.classList.remove("hidden");
  refs.deriveSummary.innerHTML = `
    <div class="count-strip">
      ${counts.map((item) => `<div class="count-pill"><span>${escapeHtml(item.group)}</span><strong>${escapeHtml(formatValue(item.n))}</strong></div>`).join("")}
    </div>
    <div class="signature-summary-grid">
      <div><strong>Derived column</strong><br/>${escapeHtml(derivedColumn || "NA")}</div>
      <div><strong>Method</strong><br/>${escapeHtml(summary?.method || "NA")}</div>
      <div><strong>Cutoff</strong><br/>${escapeHtml(formatValue(summary?.cutoff))}</div>
      ${summary?.p_value != null ? `<div class="pvalue-card"><strong>p-value</strong><br/>${escapeHtml(formatValue(summary.p_value))}</div>` : ""}
      <div><strong>Groups</strong><br/>${escapeHtml(formatValue(summary?.n_groups || counts.length || "NA"))}</div>
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

function downloadCsv(filename, rows, columns = null) {
  if (!rows || rows.length === 0) return;
  const visibleColumns = columns || Object.keys(rows[0]);
  const escapeCell = (value) => {
    const text = value === null || value === undefined ? "" : String(value);
    return `"${text.replaceAll('"', '""')}"`;
  };
  const lines = [visibleColumns.map(escapeCell).join(","), ...rows.map((row) => visibleColumns.map((column) => escapeCell(row[column])).join(","))];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  triggerBlobDownload(filename, blob);
}

function downloadText(filename, text, mimeType = "text/plain;charset=utf-8;") {
  const blob = new Blob([text], { type: mimeType });
  triggerBlobDownload(filename, blob);
}

function slugifyDownloadToken(value, fallback = "na") {
  const text = String(value ?? "").trim().toLowerCase();
  if (!text) return fallback;
  const slug = text
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return slug || fallback;
}

function currentDatasetSlug() {
  return slugifyDownloadToken(state.dataset?.filename || "survstudio_dataset", "survstudio_dataset");
}

function currentOutcomeSlug() {
  return [
    slugifyDownloadToken(refs.timeColumn?.value || "time", "time"),
    slugifyDownloadToken(refs.eventColumn?.value || "event", "event"),
  ].join("_");
}

function currentGroupSlug() {
  return slugifyDownloadToken(refs.groupColumn?.value || "overall", "overall");
}

function buildDownloadFilename(stem, ext, { includeGroup = false, template = null } = {}) {
  const parts = [currentDatasetSlug(), currentOutcomeSlug()];
  if (includeGroup) parts.push(currentGroupSlug());
  parts.push(slugifyDownloadToken(stem, "export"));
  if (template) parts.push(slugifyDownloadToken(template, "default"));
  return `${parts.filter(Boolean).join("_")}.${ext}`;
}

function triggerBlobDownload(filename, blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadServerTable(filename, payload, fallbackMimeType = "text/plain;charset=utf-8;") {
  const response = await fetch(apiUrl("/api/export-table"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Export failed.");
  }
  const blob = await response.blob();
  triggerBlobDownload(filename, blob);
}

function buildMarkdownTable(rows, { caption = "", notes = [] } = {}) {
  if (!rows || rows.length === 0) return "";
  const columns = Object.keys(rows[0]);
  const escapeCell = (value) => String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
  const header = `| ${columns.join(" | ")} |`;
  const divider = `| ${columns.map(() => "---").join(" | ")} |`;
  const body = rows.map((row) => `| ${columns.map((column) => escapeCell(formatValue(row[column]))).join(" | ")} |`);
  const sections = [];
  if (caption) sections.push(`**${caption}**`);
  sections.push([header, divider, ...body].join("\n"));
  if (notes.length) {
    sections.push("Notes:");
    sections.push(notes.map((note) => `- ${note}`).join("\n"));
  }
  return `${sections.join("\n\n")}\n`;
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

function manuscriptExportPayload(manuscript, format, template, fallbackCaption) {
  return {
    rows: manuscript?.model_performance_table || [],
    format,
    style: "journal",
    template,
    caption: manuscript?.caption || fallbackCaption,
    notes: manuscript?.table_notes || [],
  };
}

function downloadPlotImage(plotEl, filename, format) {
  if (!plotEl || !plotEl.data) return;
  Plotly.downloadImage(plotEl, { format, filename, height: 900, width: 1400, scale: format === "png" ? 3 : 1 });
}

function plotConfig(filename) {
  return {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: "svg",
      filename: buildDownloadFilename(filename, "svg").replace(/\.svg$/, ""),
      height: 900,
      width: 1400,
      scale: 1,
    },
  };
}

function updateMlEvaluationControls() {
  const isRepeatedCv = refs.mlEvaluationStrategy?.value === "repeated_cv";
  refs.mlCvFoldsWrap?.classList.toggle("hidden", !isRepeatedCv);
  refs.mlCvRepeatsWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (refs.mlCvFolds) refs.mlCvFolds.disabled = !isRepeatedCv;
  if (refs.mlCvRepeats) refs.mlCvRepeats.disabled = !isRepeatedCv;
}

function updateDlEvaluationControls() {
  const isRepeatedCv = refs.dlEvaluationStrategy?.value === "repeated_cv";
  refs.dlCvFoldsWrap?.classList.toggle("hidden", !isRepeatedCv);
  refs.dlCvRepeatsWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (refs.dlCvFolds) refs.dlCvFolds.disabled = !isRepeatedCv;
  if (refs.dlCvRepeats) refs.dlCvRepeats.disabled = !isRepeatedCv;
  if (refs.dlParallelJobs) refs.dlParallelJobs.disabled = !isRepeatedCv;
}

function purgePlot(el) {
  if (el && el.data) { try { Plotly.purge(el); } catch { /* ignore */ } }
}

function clearPlotShell(el, emptyHtml) {
  purgePlot(el);
  if (el) el.innerHTML = emptyHtml || '';
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

function refreshVariableSelections() {
  if (!state.dataset) return;
  const excluded = [refs.timeColumn.value, refs.eventColumn.value];
  const availableCovariates = cohortColumnsExcluding(...excluded);
  const previousCovariates = selectedCheckboxValues(refs.covariateChecklist).filter((v) => availableCovariates.includes(v));
  const previousCategoricals = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => availableCovariates.includes(v));
  const previousTableVars = selectedCheckboxValues(refs.cohortVariableChecklist).filter((v) => availableCovariates.includes(v));
  const defaultCategoricals = state.dataset.columns
    .filter((c) => ["categorical", "binary"].includes(c.kind) || c.n_unique <= 6)
    .map((c) => c.name)
    .filter((name) => availableCovariates.includes(name));
  renderChecklist(refs.covariateChecklist, availableCovariates, previousCovariates.length ? previousCovariates : availableCovariates.slice(0, 4));
  renderChecklist(refs.categoricalChecklist, availableCovariates, previousCategoricals.length ? previousCategoricals : defaultCategoricals);
  renderChecklist(refs.cohortVariableChecklist, availableCovariates, previousTableVars.length ? previousTableVars : availableCovariates.slice(0, 6));
  const numericOptions = state.dataset.numeric_columns.filter((c) => !excluded.includes(c));
  renderSelect(refs.deriveSource, numericOptions, { selected: numericOptions.includes(refs.deriveSource.value) ? refs.deriveSource.value : numericOptions[0] || null });
  renderSharedFeatureSummary();
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
  const filename = (state.dataset?.filename || "").toLowerCase();
  const columns = new Set((state.dataset?.columns || []).map((column) => column.name));
  const looksLikeGbsg2 = ["rfs_days", "rfs_event", "horTh", "menostat", "tgrade"].every((name) => columns.has(name));
  const looksLikeTcgaLung = ["os_months", "os_event", "stage_group", "smoking_status"].every((name) => columns.has(name));
  if (filename.includes("gbsg2") || looksLikeGbsg2) {
    return {
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
    };
  }
  if (filename.includes("tcga_luad") || looksLikeTcgaLung) {
    return {
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
    };
  }
  return null;
}

function setDatasetPresetButtonState(mode = null) {
  if (!refs.applyBasicPresetButton || !refs.applyModelPresetButton) return;
  refs.applyBasicPresetButton.className = `button ${mode === "basic" ? "primary" : "ghost"} compact-btn`;
  refs.applyModelPresetButton.className = `button ${mode === "models" ? "primary" : "ghost"} compact-btn`;
}

function renderDatasetPresetStatus(title, text, chips = []) {
  if (refs.datasetPresetStatusTitle) refs.datasetPresetStatusTitle.textContent = title;
  if (refs.datasetPresetStatusText) refs.datasetPresetStatusText.textContent = text;
  if (!refs.datasetPresetChips) return;
  refs.datasetPresetChips.innerHTML = "";
  if (!chips.length) {
    refs.datasetPresetChips.classList.add("hidden");
    return;
  }
  chips.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "dataset-preset-chip";
    chip.textContent = label;
    refs.datasetPresetChips.appendChild(chip);
  });
  refs.datasetPresetChips.classList.remove("hidden");
}

function renderSharedFeatureSummary() {
  const hasDataset = Boolean(state.dataset);
  const features = hasDataset ? selectedCheckboxValues(refs.covariateChecklist) : [];
  const categoricals = hasDataset ? selectedCheckboxValues(refs.categoricalChecklist).filter((value) => features.includes(value)) : [];
  const timeLabel = hasDataset ? (refs.timeColumn?.value || "time") : "time";
  const eventLabel = hasDataset ? (refs.eventColumn?.value || "event") : "event";
  const summaryText = !hasDataset
    ? "Load a dataset first. ML and DL use the shared covariate and categorical selections defined on the Cox tab."
    : features.length
      ? `Shared across ML and DL. Current training inputs come from the Cox tab selections: ${summarizeFeatureNames(features)}.`
      : "No shared feature set selected yet. Choose covariates on the Cox tab before training ML or DL models.";
  const finalChips = !hasDataset
    ? []
    : [
        `Time: ${timeLabel}`,
        `Event: ${eventLabel}`,
        `Features: ${features.length}`,
        `Categorical: ${categoricals.length}`,
        features.length ? `Preview: ${summarizeFeatureNames(features)}` : "Preview: none selected",
        categoricals.length ? `Categoricals: ${summarizeFeatureNames(categoricals, 3)}` : "Categoricals: none",
      ];

  [
    [refs.mlFeatureSummaryText, refs.mlFeatureSummaryChips],
    [refs.dlFeatureSummaryText, refs.dlFeatureSummaryChips],
  ].forEach(([textNode, chipContainer]) => {
    if (textNode) textNode.textContent = summaryText;
    if (!chipContainer) return;
    chipContainer.innerHTML = "";
    finalChips.forEach((label) => {
      const chip = document.createElement("span");
      chip.className = "dataset-preset-chip";
      chip.textContent = label;
      chipContainer.appendChild(chip);
    });
    chipContainer.classList.remove("hidden");
  });
}

function focusSharedFeatureEditor() {
  activateTab("cox");
  requestAnimationFrame(() => {
    refs.covariateChecklist?.closest(".selection-card")?.scrollIntoView({ behavior: "smooth", block: "center" });
    flashPresetTargets([refs.covariateChecklist, refs.categoricalChecklist]);
  });
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
  setCheckedValues(refs.covariateChecklist, covariates);
  setCheckedValues(refs.categoricalChecklist, categoricals);
  setCheckedValues(refs.cohortVariableChecklist, tableVariables);
  updateDatasetBadge();

  setDatasetPresetButtonState(mode);

  const summaryTitle = `${preset.name} applied`;
  const summaryText = mode === "basic"
    ? "Updated the study columns, group split, Cox covariates, and cohort-table variables below. No analysis ran yet."
    : "Updated the study columns and the feature checklists used by ML and DL. These shared feature selections live on the Cox tab. No analysis ran yet.";
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

function renderDatasetPreview() {
  renderTable(refs.datasetPreviewShell, state.dataset.preview);
}

function updateDatasetBadge() {
  if (!state.dataset) { refs.datasetBadge.classList.add("hidden"); return; }
  refs.datasetBadge.textContent = `${state.dataset.filename} · ${state.dataset.n_rows.toLocaleString()} rows · ${state.dataset.n_columns} cols`;
  refs.datasetBadge.classList.remove("hidden");
}

function showWorkspace() {
  if (!refs.landing.classList.contains("hidden")) {
    refs.landing.classList.add("fade-out");
    setTimeout(() => { refs.landing.classList.add("hidden"); refs.landing.classList.remove("fade-out"); }, 260);
  }
  refs.workspace.classList.remove("hidden");
  refs.workspace.classList.add("fade-in");
  setTimeout(() => refs.workspace.classList.remove("fade-in"), 360);
}

function activateTab(tabName) {
  refs.tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  refs.tabPanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tabName));
  if (state.dataset) syncHistoryState("replace");
  requestAnimationFrame(() => {
    if (tabName === "km" && state.km) Plotly.Plots.resize(refs.kmPlot);
    if (tabName === "cox" && state.cox) Plotly.Plots.resize(refs.coxPlot);
    if (tabName === "ml" && state.ml) {
      if (refs.mlImportancePlot?.data) Plotly.Plots.resize(refs.mlImportancePlot);
      if (refs.mlShapPlot?.data) Plotly.Plots.resize(refs.mlShapPlot);
      if (refs.mlComparisonPlot?.data) Plotly.Plots.resize(refs.mlComparisonPlot);
    }
    if (tabName === "dl" && state.dl) {
      if (refs.dlImportancePlot?.data) Plotly.Plots.resize(refs.dlImportancePlot);
      if (refs.dlLossPlot?.data) Plotly.Plots.resize(refs.dlLossPlot);
    }
  });
}

function updateControlsFromDataset() {
  const columnNames = state.dataset.columns.map((c) => c.name);
  const suggestions = state.dataset.suggestions;
  renderSelect(refs.timeColumn, columnNames, { selected: inferDefault(columnNames, suggestions.time_columns, 0) });
  renderSelect(refs.eventColumn, columnNames, { selected: inferDefault(columnNames, suggestions.event_columns, 1) });
  renderSelect(refs.groupColumn, columnNames, { includeBlank: true, blankLabel: "Overall only", selected: inferDefault(columnNames, suggestions.group_columns, 2) });
  updateEventPositiveOptions();
  refreshVariableSelections();
  updateDatasetBadge();
  renderSharedFeatureSummary();
  renderDatasetPreview();
  updateDatasetPresetBar();
  refs.downloadSignatureButton.disabled = true;
  showWorkspace();
  updateStepIndicator(2);
  const timeSugg = suggestions.time_columns?.[0];
  const eventSugg = suggestions.event_columns?.[0];
  if (timeSugg && eventSugg) {
    showSmartBanner(`Auto-detected: "${timeSugg}" as time column, "${eventSugg}" as event column. Adjust if needed.`);
  }
}

function updateAfterDataset(payload) {
  state.dataset = payload;
  state.km = null;
  state.cox = null;
  state.cohort = null;
  state.signature = null;
  state.ml = null;
  state.dl = null;
  refs.kmMetaBanner.textContent = "Configure your study columns above, then click Run Analysis.";
  refs.coxMetaBanner.textContent = "Select covariates above, then click Run Analysis.";
  refs.mlMetaBanner.textContent = "Select features from the Cox tab, then train a model.";
  refs.dlMetaBanner.textContent = "Select features from the Cox tab, configure hyperparameters, then train.";
  refs.deriveSummary.innerHTML = "";
  refs.deriveSummary.classList.add("hidden");
  refs.datasetPresetBar?.classList.add("hidden");
  refs.deriveStatus.textContent = "";
  if (refs.cutpointPlot) { refs.cutpointPlot.innerHTML = ""; refs.cutpointPlot.classList.add("hidden"); }
  refs.kmSummaryShell.innerHTML = '<div class="empty-state">Survival statistics will appear after you run the analysis.</div>';
  refs.kmRiskShell.innerHTML = '<div class="empty-state">Number of patients at risk over time.</div>';
  refs.kmPairwiseShell.innerHTML = '<div class="empty-state">Group-vs-group comparisons (requires 2+ groups).</div>';
  refs.signatureShell.innerHTML = '<div class="empty-state">Use auto-discovery to find the best feature combinations.</div>';
  refs.coxResultsShell.innerHTML = '<div class="empty-state">Hazard ratios will appear after running Cox analysis.</div>';
  refs.coxDiagnosticsShell.innerHTML = '<div class="empty-state">Model assumption checks will appear here.</div>';
  refs.cohortTableShell.innerHTML = '<div class="empty-state">Check variables on the left, then click <strong>Build Table</strong>.</div>';
  refs.mlComparisonShell.innerHTML = '<div class="empty-state">Click "Compare All" to see Cox vs RSF vs GBS side by side.</div>';
  refs.mlManuscriptShell.innerHTML = '<div class="empty-state">Comparison-ready manuscript rows appear after running a comparison.</div>';
  refs.mlComparisonPlot.innerHTML = "";
  refs.mlComparisonPlot.classList.add("hidden");
  refs.dlComparisonShell.innerHTML = '<div class="empty-state">Click "Compare All" to benchmark DeepSurv, DeepHit, Neural MTLR, Transformer, and VAE.</div>';
  refs.dlManuscriptShell.innerHTML = '<div class="empty-state">Comparison-ready manuscript rows appear after running a deep comparison.</div>';
  refs.dlComparisonPlot.innerHTML = "";
  refs.dlComparisonPlot.classList.add("hidden");
  clearPlotShell(refs.mlImportancePlot, '<div class="empty-state plot-empty"><span>Train a model to see feature importance</span></div>');
  clearPlotShell(refs.mlShapPlot, '<div class="empty-state plot-empty"><span>SHAP values will appear after training</span></div>');
  clearPlotShell(refs.dlImportancePlot, '<div class="empty-state plot-empty"><span>Train a deep learning model to see results</span></div>');
  clearPlotShell(refs.dlLossPlot, '<div class="empty-state plot-empty"><span>Training loss curve will appear here</span></div>');
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
  updateControlsFromDataset();
}

async function uploadDataset() {
  if (!refs.datasetFile.files?.length) throw new Error("Choose a dataset file first.");
  const formData = new FormData();
  formData.append("file", refs.datasetFile.files[0]);
  const payload = await fetchJSON("/api/upload", { method: "POST", body: formData });
  updateAfterDataset(payload);
  runtime.historySyncPaused = true;
  activateTab("km");
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadExampleDataset() {
  const payload = await fetchJSON("/api/load-example", { method: "POST" });
  updateAfterDataset(payload);
  runtime.historySyncPaused = true;
  activateTab("km");
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadTcgaUploadReadyDataset() {
  const payload = await fetchJSON("/api/load-tcga-upload-ready", { method: "POST" });
  updateAfterDataset(payload);
  runtime.historySyncPaused = true;
  activateTab("km");
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadTcgaDataset() {
  const payload = await fetchJSON("/api/load-tcga-example", { method: "POST" });
  updateAfterDataset(payload);
  runtime.historySyncPaused = true;
  activateTab("km");
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function loadGbsg2Dataset() {
  const payload = await fetchJSON("/api/load-gbsg2-example", { method: "POST" });
  updateAfterDataset(payload);
  runtime.historySyncPaused = true;
  activateTab("km");
  runtime.historySyncPaused = false;
  syncHistoryState("push");
}

async function deriveGroup() {
  const sourceColumn = refs.deriveSource.value;
  if (!sourceColumn) throw new Error("Select a numeric source column.");
  const method = refs.deriveMethod.value;
  const isCustomCutoff = method === "custom_cutoff";
  const isOptimal = method === "optimal_cutpoint";
  const cutoffInput = refs.deriveCutoff.value.trim();
  let cutoffValue = null;
  if (isCustomCutoff) {
    if (cutoffInput === "") throw new Error("Enter a numeric cutoff value.");
    cutoffValue = Number(cutoffInput);
    if (!Number.isFinite(cutoffValue)) throw new Error("Cutoff must be a finite numeric value.");
  }
  refs.deriveStatus.textContent = isOptimal ? "Scanning cutpoints..." : "Deriving...";

  const body = {
    dataset_id: state.dataset.dataset_id,
    source_column: sourceColumn,
    method,
    new_column_name: refs.deriveColumnName.value || null,
    cutoff: cutoffValue,
  };
  if (isOptimal) {
    body.time_column = refs.timeColumn.value;
    body.event_column = refs.eventColumn.value;
    body.event_positive_value = refs.eventPositiveValue.value;
  }

  const payload = await fetchJSON("/api/derive-group", { method: "POST", body: JSON.stringify(body) });
  updateAfterDataset(payload);
  refs.deriveStatus.textContent = `Created ${payload.derived_column}`;
  refs.groupColumn.value = payload.derived_column;
  renderDerivedGroupSummary(payload.derived_column, payload.derive_summary);

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
}

function updateMethodVisibility() {
  refs.cutoffWrap.classList.toggle("hidden", refs.deriveMethod.value !== "custom_cutoff");
}

function updateWeightVisibility() {
  refs.fhPowerWrap.classList.toggle("hidden", refs.logrankWeight.value !== "fleming_harrington");
}

async function runKaplanMeier() {
  const base = currentBaseConfig();
  setShimmer(refs.kmSummaryShell);
  setShimmer(refs.kmRiskShell);
  const payload = await fetchJSON("/api/kaplan-meier", {
    method: "POST",
    body: JSON.stringify({
      ...base,
      confidence_level: Number(refs.confidenceLevel.value),
      risk_table_points: Number(refs.riskTablePoints.value),
      show_confidence_bands: refs.showConfidenceBands.checked,
      logrank_weight: refs.logrankWeight.value,
      fh_p: Number(refs.fhPower.value),
    }),
  });
  state.km = payload;
  purgePlot(refs.kmPlot);
  refs.kmPlot.innerHTML = "";
  await Plotly.newPlot(refs.kmPlot, payload.figure.data, payload.figure.layout, plotConfig("km_curve"));
  updateStepIndicator(3);
  renderTable(refs.kmSummaryShell, payload.analysis.summary_table);
  renderTable(refs.kmRiskShell, payload.analysis.risk_table.rows, payload.analysis.risk_table.columns);
  renderTable(refs.kmPairwiseShell, payload.analysis.pairwise_table);
  renderInsightBoard(refs.kmInsightBoard, payload.analysis.scientific_summary, "Run KM to generate an interpretation panel.");
  const cohort = payload.analysis.cohort;
  const test = payload.analysis.test;
  refs.kmMetaBanner.textContent = `N=${cohort.n}, events=${cohort.events}, censored=${cohort.censored}, median follow-up=${formatValue(cohort.median_follow_up)} ${base.time_unit_label}${test ? `, ${test.test} p=${formatValue(test.p_value)}` : ""}`;
  refs.downloadKmSummaryButton.disabled = false;
  refs.downloadKmPairwiseButton.disabled = payload.analysis.pairwise_table.length === 0;
  if (refs.downloadKmPngButton) refs.downloadKmPngButton.disabled = false;
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.disabled = false;
  activateTab("km");
  requestAnimationFrame(() => refs.kmPlot.scrollIntoView({ behavior: "smooth", block: "center" }));
  showToast("Kaplan-Meier analysis complete", "success", 3000);
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
      <div><strong>Bootstrap support</strong><br/>${formatValue(best["Bootstrap support (p<0.05)"])}</div>
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
  const payload = await fetchJSON("/api/discover-signature", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: state.dataset.dataset_id, time_column: base.time_column, event_column: base.event_column,
      event_positive_value: base.event_positive_value, candidate_columns: candidateColumns,
      max_combination_size: Number(refs.signatureMaxDepth.value), top_k: Number(refs.signatureTopK.value),
      min_group_fraction: Number(refs.signatureMinFraction.value), bootstrap_iterations: Number(refs.signatureBootstrapIterations.value),
      bootstrap_sample_fraction: 0.8, permutation_iterations: Number(refs.signaturePermutationIterations.value),
      validation_iterations: Number(refs.signatureValidationIterations.value), validation_fraction: Number(refs.signatureValidationFraction.value),
      significance_level: Number(refs.signatureSignificanceLevel.value), combination_operator: refs.signatureOperator.value,
      random_seed: Number(refs.signatureRandomSeed.value), new_column_name: refs.deriveColumnName.value || null,
    }),
  });
  updateAfterDataset(payload);
  state.signature = payload.signature_analysis;
  refs.groupColumn.value = payload.derived_column;
  renderSignatureResult(payload.signature_analysis);
  refs.deriveStatus.textContent = `Auto-derived ${payload.derived_column}`;
  activateTab("km");
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
  updateStepIndicator(3);
  renderTable(refs.coxResultsShell, payload.analysis.results_table);
  renderTable(refs.coxDiagnosticsShell, payload.analysis.diagnostics_table);
  renderInsightBoard(refs.coxInsightBoard, payload.analysis.scientific_summary, "Run Cox PH to review diagnostics.");
  const stats = payload.analysis.model_stats;
  const coxMetricLabel = stats.c_index_label || ((stats.evaluation_mode === "apparent") ? "Apparent C-index" : "C-index");
  refs.coxMetaBanner.textContent = `N=${stats.n}, events=${stats.events}, parameters=${stats.parameters}, EPV=${formatValue(stats.events_per_parameter)}, ${coxMetricLabel}=${formatValue(stats.c_index)}, AIC=${formatValue(stats.aic)}`;
  refs.downloadCoxResultsButton.disabled = false;
  refs.downloadCoxDiagnosticsButton.disabled = false;
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.disabled = false;
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.disabled = false;
  activateTab("cox");
  requestAnimationFrame(() => refs.coxPlot.scrollIntoView({ behavior: "smooth", block: "center" }));
  showToast("Cox PH model fitted", "success", 3000);
}

async function runCohortTable() {
  const vars = selectedCheckboxValues(refs.cohortVariableChecklist);
  if (!vars.length) { showToast("Select at least one variable for the cohort table.", "error"); return; }
  setShimmer(refs.cohortTableShell);
  const payload = await fetchJSON("/api/cohort-table", {
    method: "POST",
    body: JSON.stringify({ dataset_id: state.dataset.dataset_id, variables: vars, group_column: refs.groupColumn.value || null }),
  });
  state.cohort = payload;
  renderTable(refs.cohortTableShell, payload.analysis.rows, payload.analysis.columns);
  refs.downloadCohortTableButton.disabled = false;
  updateStepIndicator(3);
  activateTab("tables");
}

// ── ML Models ──────────────────────────────────────────────────

async function runMlModel() {
  const base = currentBaseConfig();
  const features = selectedCheckboxValues(refs.covariateChecklist);
  if (!features.length) { showToast("Select at least one feature (from Cox tab covariates).", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => features.includes(v));
  setShimmer(refs.mlImportancePlot);

  const payload = await fetchJSON("/api/ml-model", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: base.dataset_id, time_column: base.time_column,
      event_column: base.event_column, event_positive_value: base.event_positive_value,
      features, categorical_features: categoricalFeatures,
      model_type: refs.mlModelType.value,
      n_estimators: Number(refs.mlNEstimators.value),
      learning_rate: Number(refs.mlLearningRate.value),
    }),
  });
  state.ml = payload;
  refs.downloadMlComparisonButton.disabled = true;
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = true;
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = true;
  setMlManuscriptDownloadsEnabled(false);
  refs.mlComparisonShell.innerHTML = '<div class="empty-state">Run a comparison to populate the cross-model table.</div>';
  refs.mlManuscriptShell.innerHTML = '<div class="empty-state">Run a comparison to populate manuscript-ready rows.</div>';
  refs.mlComparisonPlot.innerHTML = "";
  refs.mlComparisonPlot.classList.add("hidden");

  if (payload.importance_figure) {
    purgePlot(refs.mlImportancePlot);
    refs.mlImportancePlot.innerHTML = "";
    await Plotly.newPlot(refs.mlImportancePlot, payload.importance_figure.data, payload.importance_figure.layout, plotConfig("ml_importance"));
  } else {
    clearPlotShell(refs.mlImportancePlot, '<div class="empty-state plot-empty"><span>No feature importance available</span></div>');
  }
  if (payload.shap_figure) {
    purgePlot(refs.mlShapPlot);
    refs.mlShapPlot.innerHTML = "";
    await Plotly.newPlot(refs.mlShapPlot, payload.shap_figure.data, payload.shap_figure.layout, plotConfig("shap_importance"));
  } else {
    clearPlotShell(refs.mlShapPlot, '<div class="empty-state plot-empty"><span>SHAP not available for this model</span></div>');
  }
  renderInsightBoard(refs.mlInsightBoard, payload.analysis?.scientific_summary, "ML model results.");
  const stats = payload.analysis?.model_stats || {};
  const mlMetricLabel = stats.metric_name || ((stats.evaluation_mode === "holdout") ? "Holdout C-index" : "Apparent C-index");
  const mlEvaluationMode = stats.evaluation_mode || "unknown";
  refs.mlMetaBanner.textContent = `${refs.mlModelType.value.toUpperCase()}: ${mlMetricLabel}=${formatValue(stats.c_index)}, eval=${formatValue(mlEvaluationMode)}, N=${formatValue(stats.n_patients)}, features=${formatValue(stats.n_features)}`;
  updateStepIndicator(3);
  activateTab("ml");
  showToast(`${refs.mlModelType.value.toUpperCase()} model trained`, "success", 3000);
}

async function runCompareModels() {
  const base = currentBaseConfig();
  const features = selectedCheckboxValues(refs.covariateChecklist);
  if (!features.length) { showToast("Select at least one feature.", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => features.includes(v));
  setShimmer(refs.mlComparisonShell);

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
  }
  renderInsightBoard(refs.mlInsightBoard, payload.analysis?.scientific_summary, "Model comparison.");
  const comparisonRows = payload.analysis?.comparison_table || [];
  const bestRow = comparisonRows[0] || {};
  const evaluationMode = payload.analysis?.evaluation_mode || refs.mlEvaluationStrategy.value;
  const evalLabel = evaluationMode === "repeated_cv"
    ? `${payload.analysis?.cv_repeats || refs.mlCvRepeats.value}x${payload.analysis?.cv_folds || refs.mlCvFolds.value} repeated CV`
    : evaluationMode;
  refs.mlMetaBanner.textContent = `Best=${formatValue(bestRow.model)}, C-index=${formatValue(bestRow.c_index)}, eval=${formatValue(evalLabel)}, models=${formatValue(comparisonRows.length)}`;
  refs.downloadMlComparisonButton.disabled = comparisonRows.length === 0;
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.disabled = !(payload.figure?.data?.length);
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.disabled = !(payload.figure?.data?.length);
  setMlManuscriptDownloadsEnabled(!!(payload.analysis?.manuscript_tables?.model_performance_table?.length));
  activateTab("ml");
  showToast("Model comparison complete", "success", 3000);
}

// ── Deep Learning ──────────────────────────────────────────────

async function runDlModel() {
  const base = currentBaseConfig();
  const features = selectedCheckboxValues(refs.covariateChecklist);
  if (!features.length) { showToast("Select at least one feature (from Cox tab covariates).", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => features.includes(v));
  const hiddenLayers = refs.dlHiddenLayers.value.split(",").map(Number).filter((n) => n > 0);
  if (!hiddenLayers.length) hiddenLayers.push(64, 64);

  setShimmer(refs.dlImportancePlot);
  setShimmer(refs.dlLossPlot);

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
      early_stopping_patience: Number(refs.dlEarlyStoppingPatience.value),
      early_stopping_min_delta: Number(refs.dlEarlyStoppingMinDelta.value),
      parallel_jobs: Number(refs.dlParallelJobs.value),
      batch_size: 64,
      random_seed: 42,
    }),
  });
  state.dl = payload;

  if (payload.figures?.importance) {
    purgePlot(refs.dlImportancePlot);
    refs.dlImportancePlot.innerHTML = "";
    await Plotly.newPlot(refs.dlImportancePlot, payload.figures.importance.data, payload.figures.importance.layout, plotConfig("dl_importance"));
  } else {
    clearPlotShell(refs.dlImportancePlot, '<div class="empty-state plot-empty"><span>No feature importance available</span></div>');
  }
  if (payload.figures?.loss) {
    purgePlot(refs.dlLossPlot);
    refs.dlLossPlot.innerHTML = "";
    await Plotly.newPlot(refs.dlLossPlot, payload.figures.loss.data, payload.figures.loss.layout, plotConfig("dl_loss"));
  } else {
    clearPlotShell(refs.dlLossPlot, '<div class="empty-state plot-empty"><span>No loss curve available</span></div>');
  }
  refs.dlComparisonShell.innerHTML = '<div class="empty-state">Run "Compare All" to benchmark all deep models on the same feature set.</div>';
  refs.dlManuscriptShell.innerHTML = '<div class="empty-state">Run "Compare All" to populate manuscript-ready deep comparison rows.</div>';
  refs.dlComparisonPlot.innerHTML = "";
  refs.dlComparisonPlot.classList.add("hidden");
  refs.downloadDlComparisonButton.disabled = true;
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = true;
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = true;
  setDlManuscriptDownloadsEnabled(false);
  // Backend may emit either `scientific_summary` or `insight_board` depending on model implementation.
  const dlSummary = payload.analysis?.scientific_summary || payload.analysis?.insight_board || null;
  renderInsightBoard(refs.dlInsightBoard, dlSummary, "Deep learning results.");
  const stats = payload.analysis || {};
  const epochsTrained = stats.epochs_trained || stats.epochs || refs.dlEpochs.value;
  const dlMetricLabel = stats.evaluation_mode === "holdout" ? "Holdout C-index" : "Apparent C-index";
  refs.dlMetaBanner.textContent = `${refs.dlModelType.value.toUpperCase()}: ${dlMetricLabel}=${formatValue(stats.c_index)}, eval=${formatValue(stats.evaluation_mode)}, epochs=${formatValue(epochsTrained)}`;
  updateStepIndicator(3);
  activateTab("dl");
  showToast(`${refs.dlModelType.value.toUpperCase()} model trained`, "success", 3000);
}

async function runDlCompareModels() {
  const base = currentBaseConfig();
  const features = selectedCheckboxValues(refs.covariateChecklist);
  if (!features.length) { showToast("Select at least one feature (from Cox tab covariates).", "error"); return; }
  const categoricalFeatures = selectedCheckboxValues(refs.categoricalChecklist).filter((v) => features.includes(v));
  const hiddenLayers = refs.dlHiddenLayers.value.split(",").map(Number).filter((n) => n > 0);
  if (!hiddenLayers.length) hiddenLayers.push(64, 64);

  setShimmer(refs.dlComparisonShell);

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
      early_stopping_patience: Number(refs.dlEarlyStoppingPatience.value),
      early_stopping_min_delta: Number(refs.dlEarlyStoppingMinDelta.value),
      parallel_jobs: Number(refs.dlParallelJobs.value),
      batch_size: 64,
      random_seed: 42,
      evaluation_strategy: refs.dlEvaluationStrategy.value,
      cv_folds: Number(refs.dlCvFolds.value),
      cv_repeats: Number(refs.dlCvRepeats.value),
    }),
  });
  state.dl = payload;

  if (payload.analysis?.comparison_table?.length) {
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
  }
  refs.dlImportancePlot.innerHTML = '<div class="empty-state plot-empty"><span>Single-model feature importance appears when you train one deep model.</span></div>';
  refs.dlLossPlot.innerHTML = '<div class="empty-state plot-empty"><span>Single-model loss curves appear when you train one deep model.</span></div>';
  const dlSummary = payload.analysis?.scientific_summary || payload.analysis?.insight_board || null;
  renderInsightBoard(refs.dlInsightBoard, dlSummary, "Deep learning comparison results.");
  const bestRow = payload.analysis?.comparison_table?.[0] || {};
  const dlEvalMode = payload.analysis?.evaluation_mode || refs.dlEvaluationStrategy.value;
  const dlEvalLabel = dlEvalMode === "repeated_cv"
    ? `${payload.analysis?.cv_repeats || refs.dlCvRepeats.value}x${payload.analysis?.cv_folds || refs.dlCvFolds.value} repeated CV`
    : bestRow.evaluation_mode;
  const dlBestLabel = dlEvalMode === "mixed_holdout_apparent" ? "Best holdout-comparable" : "Best";
  refs.dlMetaBanner.textContent = `${dlBestLabel}=${formatValue(bestRow.model)}, C-index=${formatValue(bestRow.c_index)}, eval=${formatValue(dlEvalLabel)}, models=${formatValue(payload.analysis?.comparison_table?.length || 0)}`;
  refs.downloadDlComparisonButton.disabled = !(payload.analysis?.comparison_table?.length);
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.disabled = !(payload.figures?.comparison?.data?.length);
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.disabled = !(payload.figures?.comparison?.data?.length);
  setDlManuscriptDownloadsEnabled(!!(payload.analysis?.manuscript_tables?.model_performance_table?.length));
  updateStepIndicator(3);
  activateTab("dl");
  showToast("Deep learning model comparison complete", "success", 3000);
}

// ── Downloads ──────────────────────────────────────────────────

function wireDownloads() {
  refs.downloadKmSummaryButton.addEventListener("click", () => { if (state.km) downloadCsv(buildDownloadFilename("km_summary", "csv", { includeGroup: true }), state.km.analysis.summary_table); });
  refs.downloadKmPairwiseButton.addEventListener("click", () => { if (state.km) downloadCsv(buildDownloadFilename("km_pairwise", "csv", { includeGroup: true }), state.km.analysis.pairwise_table); });
  refs.downloadSignatureButton.addEventListener("click", () => { if (state.signature) downloadCsv(buildDownloadFilename("signature_ranking", "csv"), state.signature.results_table); });
  refs.downloadCoxResultsButton.addEventListener("click", () => { if (state.cox) downloadCsv(buildDownloadFilename("cox_results", "csv"), state.cox.analysis.results_table); });
  refs.downloadCoxDiagnosticsButton.addEventListener("click", () => { if (state.cox) downloadCsv(buildDownloadFilename("cox_diagnostics", "csv"), state.cox.analysis.diagnostics_table); });
  if (refs.downloadCoxPngButton) refs.downloadCoxPngButton.addEventListener("click", () => downloadPlotImage(refs.coxPlot, buildDownloadFilename("cox_forest", "png").replace(/\.png$/, ""), "png"));
  if (refs.downloadCoxSvgButton) refs.downloadCoxSvgButton.addEventListener("click", () => downloadPlotImage(refs.coxPlot, buildDownloadFilename("cox_forest", "svg").replace(/\.svg$/, ""), "svg"));
  refs.downloadCohortTableButton.addEventListener("click", () => { if (state.cohort) downloadCsv(buildDownloadFilename("cohort_summary", "csv", { includeGroup: true }), state.cohort.analysis.rows, state.cohort.analysis.columns); });
  refs.downloadMlComparisonButton.addEventListener("click", () => {
    const rows = state.ml?.analysis?.comparison_table;
    if (!rows) return;
    void downloadServerTable(buildDownloadFilename("ml_model_comparison", "csv"), {
      rows,
      format: "csv",
      style: "plain",
    }, "text/csv;charset=utf-8;").catch((error) => showError(error.message));
  });
  if (refs.downloadMlComparisonPngButton) refs.downloadMlComparisonPngButton.addEventListener("click", () => downloadPlotImage(refs.mlComparisonPlot, buildDownloadFilename("ml_model_comparison", "png").replace(/\.png$/, ""), "png"));
  if (refs.downloadMlComparisonSvgButton) refs.downloadMlComparisonSvgButton.addEventListener("click", () => downloadPlotImage(refs.mlComparisonPlot, buildDownloadFilename("ml_model_comparison", "svg").replace(/\.svg$/, ""), "svg"));
  refs.downloadMlManuscriptCsvButton.addEventListener("click", () => {
    const manuscript = state.ml?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "csv", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "csv", currentMlJournalTemplate(), "Model discrimination summary"),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(error.message));
  });
  refs.downloadMlManuscriptMarkdownButton.addEventListener("click", () => {
    const manuscript = state.ml?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "md", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "markdown", currentMlJournalTemplate(), "Model discrimination summary"),
      "text/markdown;charset=utf-8;",
    ).catch((error) => showError(error.message));
  });
  refs.downloadMlManuscriptLatexButton.addEventListener("click", () => {
    const manuscript = state.ml?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "tex", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "latex", currentMlJournalTemplate(), "Model discrimination summary"),
      "text/x-tex;charset=utf-8;",
    ).catch((error) => showError(error.message));
  });
  refs.downloadMlManuscriptDocxButton.addEventListener("click", () => {
    const manuscript = state.ml?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("ml_manuscript_table", "docx", { template: currentMlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "docx", currentMlJournalTemplate(), "Model discrimination summary"),
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ).catch((error) => showError(error.message));
  });
  refs.downloadDlComparisonButton.addEventListener("click", () => {
    const rows = state.dl?.analysis?.comparison_table;
    if (!rows) return;
    void downloadServerTable(buildDownloadFilename("dl_model_comparison", "csv"), {
      rows,
      format: "csv",
      style: "plain",
    }, "text/csv;charset=utf-8;").catch((error) => showError(error.message));
  });
  if (refs.downloadDlComparisonPngButton) refs.downloadDlComparisonPngButton.addEventListener("click", () => downloadPlotImage(refs.dlComparisonPlot, buildDownloadFilename("dl_model_comparison", "png").replace(/\.png$/, ""), "png"));
  if (refs.downloadDlComparisonSvgButton) refs.downloadDlComparisonSvgButton.addEventListener("click", () => downloadPlotImage(refs.dlComparisonPlot, buildDownloadFilename("dl_model_comparison", "svg").replace(/\.svg$/, ""), "svg"));
  refs.downloadDlManuscriptCsvButton.addEventListener("click", () => {
    const manuscript = state.dl?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "csv", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "csv", currentDlJournalTemplate(), "Deep model discrimination summary"),
      "text/csv;charset=utf-8;",
    ).catch((error) => showError(error.message));
  });
  refs.downloadDlManuscriptMarkdownButton.addEventListener("click", () => {
    const manuscript = state.dl?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "md", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "markdown", currentDlJournalTemplate(), "Deep model discrimination summary"),
      "text/markdown;charset=utf-8;",
    ).catch((error) => showError(error.message));
  });
  refs.downloadDlManuscriptLatexButton.addEventListener("click", () => {
    const manuscript = state.dl?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "tex", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "latex", currentDlJournalTemplate(), "Deep model discrimination summary"),
      "text/x-tex;charset=utf-8;",
    ).catch((error) => showError(error.message));
  });
  refs.downloadDlManuscriptDocxButton.addEventListener("click", () => {
    const manuscript = state.dl?.analysis?.manuscript_tables;
    const rows = manuscript?.model_performance_table;
    if (!rows) return;
    void downloadServerTable(
      buildDownloadFilename("dl_manuscript_table", "docx", { template: currentDlJournalTemplate() }),
      manuscriptExportPayload(manuscript, "docx", currentDlJournalTemplate(), "Deep model discrimination summary"),
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ).catch((error) => showError(error.message));
  });
  if (refs.downloadKmPngButton) refs.downloadKmPngButton.addEventListener("click", () => downloadPlotImage(refs.kmPlot, buildDownloadFilename("km_curve", "png", { includeGroup: true }).replace(/\.png$/, ""), "png"));
  if (refs.downloadKmSvgButton) refs.downloadKmSvgButton.addEventListener("click", () => downloadPlotImage(refs.kmPlot, buildDownloadFilename("km_curve", "svg", { includeGroup: true }).replace(/\.svg$/, ""), "svg"));
}

// ── Utilities ──────────────────────────────────────────────────

async function withLoading(button, action) {
  setButtonLoading(button, true);
  setRuntimeBanner("");
  try { await action(); } catch (error) { showError(error.message); } finally { setButtonLoading(button, false); }
}

async function initializeRuntime() {
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
  return null;
}

function getActiveRunAction() {
  const tab = document.querySelector(".tab-button.active")?.dataset.tab;
  if (tab === "km") return runKaplanMeier;
  if (tab === "cox") return runCox;
  if (tab === "tables") return runCohortTable;
  if (tab === "ml") return runMlModel;
  if (tab === "dl") return runDlModel;
  return null;
}

function updateStepIndicator(step) {
  if (!refs.stepIndicator) return;
  const steps = refs.stepIndicator.querySelectorAll(".step");
  const connectors = refs.stepIndicator.querySelectorAll(".step-connector");
  steps.forEach((el) => {
    const s = Number(el.dataset.step);
    el.classList.remove("active", "completed");
    if (s < step) {
      el.classList.add("completed");
      el.querySelector(".step-circle").innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (s === step) {
      el.classList.add("active");
    }
  });
  connectors.forEach((c, i) => c.classList.toggle("completed", i < step - 1));
}

function showSmartBanner(text) {
  if (!refs.smartBanner || !refs.smartBannerText) return;
  refs.smartBannerText.textContent = text;
  refs.smartBanner.classList.remove("hidden");
}

function initTooltips() {
  const popup = refs.tooltipPopup;
  if (!popup) return;
  let activeTarget = null;
  document.addEventListener("mouseenter", (e) => {
    const dot = e.target.closest("[data-tooltip]");
    if (!dot) return;
    activeTarget = dot;
    popup.textContent = dot.dataset.tooltip;
    popup.classList.remove("hidden");
    const rect = dot.getBoundingClientRect();
    popup.style.left = `${Math.min(rect.left, window.innerWidth - 280)}px`;
    popup.style.top = `${rect.bottom + 8}px`;
  }, true);
  document.addEventListener("mouseleave", (e) => {
    const dot = e.target.closest("[data-tooltip]");
    if (dot && dot === activeTarget) { popup.classList.add("hidden"); activeTarget = null; }
  }, true);
}

function initDragDrop() {
  const zone = document.getElementById("uploadZone");
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

function goHome({ syncHistory = true } = {}) {
  state.dataset = null;
  state.km = null;
  state.cox = null;
  state.cohort = null;
  state.signature = null;
  state.ml = null;
  state.dl = null;
  refs.workspace.classList.add("hidden");
  refs.landing.classList.remove("hidden", "fade-out");
  refs.datasetBadge.classList.add("hidden");
  refs.datasetFile.value = "";
  renderSharedFeatureSummary();
  setRuntimeBanner("");
  if (syncHistory) syncHistoryState("replace");
}

function initListeners() {
  const brandHome = document.getElementById("brandHome");
  if (brandHome) {
    brandHome.addEventListener("click", (e) => { e.preventDefault(); goHome(); });
  }
  window.addEventListener("popstate", (event) => {
    void restoreHistoryState(event.state);
  });
  refs.datasetFile.addEventListener("change", () => {
    if (refs.datasetFile.files?.length) withLoading(refs.uploadButton, uploadDataset);
  });
  refs.uploadButton.addEventListener("click", (e) => {
    if (!refs.datasetFile.files?.length) { e.stopPropagation(); return; }
    withLoading(refs.uploadButton, uploadDataset);
  });
  refs.loadTcgaUploadReadyButton.addEventListener("click", () => withLoading(refs.loadTcgaUploadReadyButton, loadTcgaUploadReadyDataset));
  refs.loadTcgaButton.addEventListener("click", () => withLoading(refs.loadTcgaButton, loadTcgaDataset));
  refs.loadGbsg2Button.addEventListener("click", () => withLoading(refs.loadGbsg2Button, loadGbsg2Dataset));
  refs.loadExampleButton.addEventListener("click", () => withLoading(refs.loadExampleButton, loadExampleDataset));
  refs.applyBasicPresetButton?.addEventListener("click", () => applyDatasetPreset("basic"));
  refs.applyModelPresetButton?.addEventListener("click", () => applyDatasetPreset("models"));
  refs.timeColumn.addEventListener("change", () => { refreshVariableSelections(); updateDatasetBadge(); });
  refs.eventColumn.addEventListener("change", () => { updateEventPositiveOptions(); refreshVariableSelections(); updateDatasetBadge(); });
  refs.groupColumn.addEventListener("change", updateDatasetBadge);
  refs.covariateChecklist?.addEventListener("change", renderSharedFeatureSummary);
  refs.categoricalChecklist?.addEventListener("change", renderSharedFeatureSummary);
  refs.reviewMlFeaturesButton?.addEventListener("click", focusSharedFeatureEditor);
  refs.reviewDlFeaturesButton?.addEventListener("click", focusSharedFeatureEditor);
  refs.deriveMethod.addEventListener("change", updateMethodVisibility);
  refs.logrankWeight.addEventListener("change", updateWeightVisibility);
  refs.mlEvaluationStrategy.addEventListener("change", updateMlEvaluationControls);
  refs.dlEvaluationStrategy.addEventListener("change", updateDlEvaluationControls);
  refs.deriveToggle.addEventListener("click", () => {
    refs.derivePanel.classList.toggle("hidden");
    refs.deriveToggle.textContent = refs.derivePanel.classList.contains("hidden") ? "Derive Group" : "Close";
  });
  refs.deriveButton.addEventListener("click", () => withLoading(refs.deriveButton, deriveGroup));
  refs.runKmButton.addEventListener("click", () => withLoading(refs.runKmButton, runKaplanMeier));
  refs.runSignatureSearchButton.addEventListener("click", () => withLoading(refs.runSignatureSearchButton, runSignatureSearch));
  refs.runCoxButton.addEventListener("click", () => withLoading(refs.runCoxButton, runCox));
  refs.runCohortTableButton.addEventListener("click", () => withLoading(refs.runCohortTableButton, runCohortTable));
  refs.runMlButton.addEventListener("click", () => withLoading(refs.runMlButton, runMlModel));
  refs.runCompareButton.addEventListener("click", () => withLoading(refs.runCompareButton, runCompareModels));
  refs.runDlButton.addEventListener("click", () => withLoading(refs.runDlButton, runDlModel));
  refs.runDlCompareButton.addEventListener("click", () => withLoading(refs.runDlCompareButton, runDlCompareModels));
  refs.tabButtons.forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
  wireDownloads();
}

initListeners();
initDragDrop();
initKeyboardShortcuts();
initTooltips();
updateMethodVisibility();
updateWeightVisibility();
updateMlEvaluationControls();
updateDlEvaluationControls();
initializeRuntime();

if (refs.smartBannerClose) {
  refs.smartBannerClose.addEventListener("click", () => refs.smartBanner.classList.add("hidden"));
}

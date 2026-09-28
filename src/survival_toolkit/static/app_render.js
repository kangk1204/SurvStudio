// SurvStudio front end, part 4/8: Value formatting, tables, insight boards, exports, and plot shells.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

function formatValue(value, options = {}) {
  if (value === null || value === undefined || value === "") return "NA";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "NA";
    const { scientificLarge = true } = options;
    const absValue = Math.abs(value);
    // Counts, days, and other whole numbers read best as plain integers (N=1500, not 1.50e+3).
    if (Number.isInteger(value) && absValue < 1e12) return String(value);
    // Scientific notation is reserved for genuinely extreme magnitudes.
    if ((scientificLarge && absValue >= 1e6) || (absValue > 0 && absValue < 0.001)) return value.toExponential(2);
    if (absValue > 0 && absValue < 0.1) return value.toFixed(4).replace(/\.?0+$/, "");
    return value.toFixed(3).replace(/\.?0+$/, "");
  }
  return String(value);
}

function normalizeValueLabel(label) {
  return String(label ?? "")
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Labels of analysis outputs whose values are p-values: "P value", "Logrank p", "Family-wise P",
// "Global PH pvalue", "Permutation p (search-adjusted)". Whole words only, so "Test positive", "raw
// pathology" or "p16" are not p-values; a qualifier in parentheses ("(Holm)", "(p<alpha)") is ignored.
function isPValueLikeLabel(label) {
  const tokens = normalizeValueLabel(String(label ?? "").replace(/\([^)]*\)/g, " ")).split(/[^a-z0-9]+/).filter(Boolean);
  if (!tokens.length) return false;
  const last = tokens[tokens.length - 1];
  if (last === "p" || last === "pvalue") return true;
  return tokens.some((token, index) => token === "pvalue" || (token === "p" && tokens[index + 1] === "value"));
}

function formatPValue(value) {
  if (value === null || value === undefined || value === "") return "NA";
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "NA";
  if (value < 1e-16) return "<1e-16";
  if (value < 0.001) return "<0.001";
  return value.toFixed(3);
}

// "p=0.012" or "p<0.001" (never "p=<0.001").
function pValuePhrase(value) {
  const text = formatPValue(value);
  return text.startsWith("<") ? `p${text}` : `p=${text}`;
}

// Only numbers get p-value formatting; text such as "45 (30.0%)" is shown as it is.
function formatDisplayValue(value, label = "") {
  return typeof value === "number" && isPValueLikeLabel(label) ? formatPValue(value) : formatValue(value);
}

// Own properties only, so a column named "constructor" or "toString" does not pick up Object.prototype.
function ownEntry(record, key) {
  if (record instanceof Map) return record.get(key);
  return record && Object.prototype.hasOwnProperty.call(record, key) ? record[key] : undefined;
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
  // Row-drop counters are counts: a missing value means nothing was dropped, not "NA".
  const metricValue = (m) => ((m.value === null || m.value === undefined) && /^dropped\b/i.test(String(m.label || "")) ? 0 : m.value);
  const metricsMarkup = metrics.length
    ? `<div class="insight-metrics">${metrics.map((m) => `<div class="metric-pill"><span>${escapeHtml(m.label || "")}</span><strong>${escapeHtml(formatDisplayValue(metricValue(m), m.label || ""))}</strong></div>`).join("")}</div>`
    : "";
  // The headline, key numbers and the first two cautions stay in view; everything else folds away.
  const leadCautions = cautions.slice(0, 2);
  const moreCautions = cautions.slice(2);
  const sections = [
    moreCautions.length ? `<div class="insight-section"><h4>What to watch</h4><ul>${moreCautions.map(escapeListItem).join("")}</ul></div>` : "",
    strengths.length ? `<div class="insight-section"><h4>What was checked</h4><ul>${strengths.map(escapeListItem).join("")}</ul></div>` : "",
    nextSteps.length ? `<div class="insight-section"><h4>Next steps</h4><ul>${nextSteps.map(escapeListItem).join("")}</ul></div>` : "",
  ].filter(Boolean).join("");
  container.innerHTML = `
    <article class="insight-card tone-${escapeHtml(tone)}">
      <div class="insight-header"><span class="insight-badge">${escapeHtml(statusLabel(tone))}</span><p>${escapeHtml(summary.headline || "Interpretation unavailable.")}</p></div>
      ${metricsMarkup}
      ${leadCautions.length ? `<ul class="insight-cautions">${leadCautions.map(escapeListItem).join("")}</ul>` : ""}
      ${sections ? `<details class="insight-details"><summary>More detail</summary><div class="insight-sections">${sections}</div></details>` : ""}
    </article>`;
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

function normalizeDeriveSummaryText(value) {
  return String(value ?? "").trim();
}

function normalizeDeriveSummaryNumber(value) {
  const text = normalizeDeriveSummaryText(value);
  if (!text) return "";
  const number = Number(text);
  return Number.isFinite(number) ? String(number) : text;
}

function deriveMethodUsesCutoff(method) {
  return method === "percentile_split" || method === "extreme_split";
}

function currentDeriveDraftConfig() {
  const method = normalizeDeriveSummaryText(refs.deriveMethod?.value);
  return {
    sourceColumn: normalizeDeriveSummaryText(refs.deriveSource?.value),
    method,
    cutoffText: deriveMethodUsesCutoff(method) ? normalizeDeriveSummaryText(refs.deriveCutoff?.value) : "",
    columnName: normalizeDeriveSummaryText(refs.deriveColumnName?.value),
    minGroupFraction: method === "optimal_cutpoint" ? normalizeDeriveSummaryNumber(refs.deriveMinGroupFraction?.value) : "",
    permutationIterations: method === "optimal_cutpoint" ? normalizeDeriveSummaryNumber(refs.derivePermutationIterations?.value) : "",
    randomSeed: method === "optimal_cutpoint" ? normalizeDeriveSummaryNumber(refs.deriveRandomSeed?.value) : "",
  };
}

function storedDeriveRecipeConfig(derivedColumn, summary) {
  const normalizedColumn = normalizeDeriveSummaryText(derivedColumn);
  const recipe = runtime.derivedColumnProvenance?.[normalizedColumn]?.recipe || summary?.recipe || {};
  const method = normalizeDeriveSummaryText(recipe.method ?? summary?.method);
  return {
    sourceColumn: normalizeDeriveSummaryText(recipe.source_column),
    method,
    cutoffText: deriveMethodUsesCutoff(method)
      ? normalizeDeriveSummaryText(recipe.cutoff_spec ?? summary?.cutoff_spec ?? recipe.cutoff ?? summary?.cutoff)
      : "",
    columnName: normalizeDeriveSummaryText(recipe.column_name || normalizedColumn),
    minGroupFraction: method === "optimal_cutpoint" ? normalizeDeriveSummaryNumber(recipe.min_group_fraction ?? summary?.min_group_fraction) : "",
    permutationIterations: method === "optimal_cutpoint" ? normalizeDeriveSummaryNumber(recipe.permutation_iterations ?? summary?.permutation_iterations) : "",
    randomSeed: method === "optimal_cutpoint" ? normalizeDeriveSummaryNumber(recipe.random_seed ?? summary?.random_seed) : "",
  };
}

function deriveDraftMatchesStoredRecipe(derivedColumn, summary) {
  const stored = storedDeriveRecipeConfig(derivedColumn, summary);
  if (!Object.values(stored).some(Boolean)) return true;
  const draft = currentDeriveDraftConfig();
  return ["sourceColumn", "method", "cutoffText", "columnName", "minGroupFraction", "permutationIterations", "randomSeed"]
    .every((key) => draft[key] === stored[key]);
}

function currentDerivedSummaryPayload() {
  const currentGroup = normalizeDeriveSummaryText(refs.groupColumn?.value);
  const currentMeta = currentGroup ? runtime.derivedColumnProvenance?.[currentGroup] : null;
  if (currentMeta?.summary) {
    return {
      derivedColumn: currentGroup,
      summary: currentMeta.summary,
    };
  }
  if (runtime.lastDerivedGroup?.derivedColumn && runtime.lastDerivedGroup?.summary) {
    return runtime.lastDerivedGroup;
  }
  return null;
}

function rerenderDerivedGroupSummaryIfVisible() {
  if (!refs.deriveSummary || refs.deriveSummary.dataset.summaryKind !== "derived") return;
  const payload = currentDerivedSummaryPayload();
  if (!payload) {
    refs.deriveSummary.innerHTML = "";
    refs.deriveSummary.classList.add("hidden");
    refs.deriveSummary.dataset.summaryKind = "";
    return;
  }
  renderDerivedGroupSummary(payload.derivedColumn, payload.summary);
}

function renderDerivedGroupSummary(derivedColumn, summary) {
  const counts = summary?.counts || [];
  const assignmentRule = summary?.assignment_rule || null;
  const pValueLabel = humanizePValueLabel(summary?.p_value_label);
  const percentileSpec = summary?.cutoff_spec || null;
  const thresholds = Array.isArray(summary?.cutoffs)
    ? summary.cutoffs.map((value) => formatValue(value)).join(", ")
    : "";
  const currentGroup = normalizeDeriveSummaryText(refs.groupColumn?.value);
  const currentGroupLabel = currentGroup || "overall only";
  const normalizedDerivedColumn = normalizeDeriveSummaryText(derivedColumn);
  const currentUsesDerived = currentGroup === normalizedDerivedColumn;
  const controlsMismatch = !deriveDraftMatchesStoredRecipe(normalizedDerivedColumn, summary);
  const summaryModeLabel = currentUsesDerived ? "Current derived grouping" : "Stored derived grouping";
  const summaryModeText = currentUsesDerived
    ? "This card describes the grouping currently selected in Group by."
    : "This card describes the last derived column you created, not the current Group by selection.";
  const notes = [];
  if (currentUsesDerived) {
    notes.push(`Current grouping now uses ${normalizedDerivedColumn}. ML and DL feature selections did not change automatically.`);
  } else if (currentGroup) {
    notes.push(`Derived column ${normalizedDerivedColumn} is available. Current grouping remains ${currentGroupLabel}. ML and DL feature selections did not change automatically.`);
    notes.push(`The counts and method details below describe ${normalizedDerivedColumn}, not ${currentGroupLabel}.`);
  } else {
    notes.push(`Derived column ${normalizedDerivedColumn} is available. Current grouping remains overall only. ML and DL feature selections did not change automatically.`);
    notes.push(`The counts and method details below describe ${normalizedDerivedColumn}. Group by is still Overall only until you select it or rerun Kaplan-Meier with auto-apply.`);
  }
  if (controlsMismatch) {
    const controlsLead = currentGroup
      ? "The disabled Source variable / Method / Column name controls above are draft settings for the next Create action after you clear Group by back to Overall only."
      : "The visible Source variable / Method / Column name controls above are draft settings for the next Create action.";
    notes.push(`${controlsLead} They do not describe ${normalizedDerivedColumn}.`);
  }
  const summaryCell = (label, value, extraClass = "") => `
      <div class="${extraClass}">
        <strong>${escapeHtml(label)}</strong>
        <span class="summary-value" title="${escapeHtml(String(value ?? "NA"))}">${escapeHtml(String(value ?? "NA"))}</span>
    </div>`;
  if (refs.cutpointPlot && summary?.method !== "optimal_cutpoint") {
    resetPlotElement(refs.cutpointPlot);
    refs.cutpointPlot.classList.add("hidden");
  }
  refs.deriveSummary.classList.remove("hidden");
  refs.deriveSummary.dataset.summaryKind = "derived";
  refs.deriveSummary.innerHTML = `
    <div class="derive-summary-heading">
      <span class="derive-summary-kicker">${escapeHtml(summaryModeLabel)}</span>
      <strong>${escapeHtml(normalizedDerivedColumn || "NA")}</strong>
      <span>${escapeHtml(summaryModeText)}</span>
    </div>
    ${notes.map((note) => `<div class="note-box">${escapeHtml(note)}</div>`).join("")}
    ${summary?.method === "optimal_cutpoint" ? '<div class="note-box">High/Low indicate risk direction, not whether the source value itself is numerically high or low.</div>' : ""}
    ${summary?.method === "optimal_cutpoint" ? '<div class="note-box">This optimal cutpoint used outcome information. Use it for grouping or visualization, not as an ML/DL training feature.</div>' : ""}
    ${summary?.method === "extreme_split" ? '<div class="note-box">Middle-range rows are excluded from grouped analyses for extreme split.</div>' : ""}
    ${counts.length ? `
      <div class="count-summary-label">${escapeHtml(currentUsesDerived ? "Counts for the current derived grouping" : "Counts for the stored derived column")}</div>
      <div class="count-strip">
        ${counts.map((item) => `<div class="count-pill"><span>${escapeHtml(deriveGroupCountLabel(item.group))}</span><strong>${escapeHtml(formatValue(item.n))}</strong></div>`).join("")}
      </div>
    ` : ""}
    <div class="signature-summary-grid">
      ${summaryCell("Derived column", normalizedDerivedColumn || "NA")}
      ${summaryCell("Method", humanizeDeriveMethod(summary?.method || "NA"))}
      ${percentileSpec ? summaryCell("Percentile(s)", percentileSpec) : ""}
      ${thresholds ? summaryCell(Array.isArray(summary?.cutoffs) && summary.cutoffs.length > 1 ? "Thresholds" : "Threshold", thresholds) : ""}
      ${!percentileSpec && !thresholds ? summaryCell("Cutoff", formatValue(summary?.cutoff)) : ""}
      ${summary?.p_value != null ? summaryCell(pValueLabel, formatPValue(summary.p_value), "pvalue-card") : ""}
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

// `pValueColumns` lists the columns shown as p-values; without it, the labels of analysis outputs decide
// (isPValueLikeLabel). Tables whose column names come from the data (Table 1 group levels, the data
// preview) pass `pValueColumns: []` and `rawHeaders: true`, so a group named "Test positive" or a column
// named "ki67_p" is shown as it is.
function renderTable(shell, rows, columns = null, { labels = {}, pValueColumns = null, rawHeaders = false } = {}) {
  if (!rows || rows.length === 0) {
    shell.innerHTML = '<div class="empty-state">No rows returned.</div>';
    return;
  }
  const visibleColumns = columns || Object.keys(rows[0]);
  const isPValueColumn = Array.isArray(pValueColumns)
    ? (column) => pValueColumns.includes(column)
    : (column) => isPValueLikeLabel(column);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const headerRow = document.createElement("tr");
  visibleColumns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = ownEntry(labels, column) || (rawHeaders ? String(column) : humanizeHeader(String(column)));
    th.title = column;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    visibleColumns.forEach((column) => {
      const td = document.createElement("td");
      const value = ownEntry(row, column);
      td.textContent = typeof value === "number" && isPValueColumn(column) ? formatPValue(value) : formatValue(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  shell.innerHTML = "";
  shell.appendChild(table);
}

function comparisonRowsHaveKey(rows, key) {
  return rows.some((row) => row && Object.prototype.hasOwnProperty.call(row, key));
}

function comparisonRowsHaveValue(rows, key) {
  return rows.some((row) => row?.[key] !== null && row?.[key] !== undefined && row?.[key] !== "");
}

function comparisonRankIsMissing(rank) {
  return rank === null || rank === undefined || rank === "" || !Number.isFinite(Number(rank));
}

function lockedTestSummaryNote(analysis) {
  const rows = Array.isArray(analysis?.comparison_table) ? analysis.comparison_table : [];
  if (!comparisonRowsHaveKey(rows, "locked_test_c_index")) return "";
  const rankOne = rows.find((row) => Number(row?.rank) === 1) || rows[0] || {};
  const parts = [
    `Models are ranked by development-set repeated CV${analysis?.n_development_patients != null ? ` (${formatValue(analysis.n_development_patients)} patients)` : ""}.`,
    `The locked test set${analysis?.n_locked_test_patients != null ? ` (${formatValue(analysis.n_locked_test_patients)} patients, ${formatValue(analysis.n_locked_test_events)} events)` : ""} was not used for ranking; report the locked-test C-index of the rank-1 model (${formatValue(rankOne.model)}: ${formatValue(rankOne.locked_test_c_index)}) as the independent estimate, not the best locked-test value across models.`,
  ];
  if (analysis?.locked_test_note) parts.push(String(analysis.locked_test_note));
  return parts.join(" ");
}

function renderComparisonTable(shell, analysis, preferredColumns = []) {
  const rows = Array.isArray(analysis?.comparison_table) ? analysis.comparison_table : [];
  const hasLockedTest = comparisonRowsHaveKey(rows, "locked_test_c_index");
  const hasInterval = rows.some((row) => row?.c_index_interval_lower != null && row?.c_index_interval_upper != null);
  const displayRows = rows.map((row) => {
    const display = { ...row };
    if (row?.c_index_interval_lower != null && row?.c_index_interval_upper != null) {
      display.c_index_interval = `${formatValue(row.c_index_interval_lower)} to ${formatValue(row.c_index_interval_upper)}`;
    }
    if (hasLockedTest && (row?.locked_test_samples != null || row?.locked_test_events != null)) {
      display.locked_test_n = `${formatValue(row.locked_test_samples)} (${formatValue(row.locked_test_events)} events)`;
    }
    if (comparisonRankIsMissing(row?.rank) && comparisonRowsHaveKey(rows, "rank")) display.rank = "Not ranked";
    return display;
  });
  const columns = [];
  preferredColumns.forEach((column) => {
    if (column === "c_index_std" && !comparisonRowsHaveValue(rows, "c_index_std")) return;
    if (column === "c_index_interval") {
      if (hasInterval) columns.push(column);
      return;
    }
    if (column === "locked_test_n") {
      if (hasLockedTest && comparisonRowsHaveKey(rows, "locked_test_samples")) columns.push(column);
      return;
    }
    if (column === "locked_test_error") {
      if (comparisonRowsHaveValue(rows, "locked_test_error")) columns.push(column);
      return;
    }
    if (rows[0]?.[column] !== undefined || comparisonRowsHaveKey(rows, column)) columns.push(column);
  });
  const repeatedCv = String(analysis?.evaluation_mode || "").startsWith("repeated_cv");
  const labels = {
    c_index_std: "SD (folds)",
    c_index_interval: String(rows.find((row) => row?.c_index_interval_label)?.c_index_interval_label || "Fold-level 2.5th-97.5th percentile range"),
    locked_test_c_index: "Locked-test C-index",
    locked_test_n: "Locked-test N",
    locked_test_error: "Locked-test error",
  };
  if (hasLockedTest) labels.c_index = "Development CV C-index";
  else if (repeatedCv) labels.c_index = "Mean CV C-index";
  renderTable(shell, displayRows, columns, { labels });
  const note = lockedTestSummaryNote(analysis);
  if (note && rows.length) {
    const noteEl = document.createElement("p");
    noteEl.className = "comparison-table-note";
    noteEl.textContent = note;
    shell.prepend(noteEl);
  }
}

function repeatedCvDesignLabel(analysis) {
  const repeats = analysis?.cv_repeats;
  const folds = analysis?.cv_folds;
  return repeats != null && folds != null ? `${formatValue(repeats)}x${formatValue(folds)} repeated CV` : "repeated CV";
}

function compareEvaluationLabel(analysis, evaluationMode) {
  if (evaluationMode === "repeated_cv") return repeatedCvDesignLabel(analysis);
  if (evaluationMode === "repeated_cv_incomplete") return `${repeatedCvDesignLabel(analysis)} (incomplete)`;
  if (evaluationMode === "mixed_holdout_apparent") return "mixed holdout/apparent";
  return evaluationMode;
}

function lockedTestBannerSuffix(analysis, bestRow) {
  if (!bestRow || !Object.prototype.hasOwnProperty.call(bestRow, "locked_test_c_index")) return "";
  const heldOut = analysis?.n_locked_test_patients != null ? ` on ${formatValue(analysis.n_locked_test_patients)} held-out patients` : "";
  return `, locked-test C-index of rank-1 model=${formatValue(bestRow.locked_test_c_index)}${heldOut}`;
}

function clearCohortTableOutput({ rerenderChrome = true, syncHistory = true } = {}) {
  state.cohort = null;
  if (refs.cohortTableShell) refs.cohortTableShell.innerHTML = COHORT_TABLE_EMPTY_STATE_HTML;
  if (refs.downloadCohortTableButton) refs.downloadCohortTableButton.disabled = true;
  if (refs.downloadCohortTableXlsxButton) refs.downloadCohortTableXlsxButton.disabled = true;
  renderSharedFeatureSummary();
  syncDownloadButtonAvailability();
  if (rerenderChrome) renderWorkspaceChrome();
  if (syncHistory) queueHistorySync();
}

function downloadCsv(filename, rows, columns = null, { caption = "", notes = [] } = {}) {
  return downloadHelpers.downloadCsv({ filename, rows, columns, showToast, caption, notes });
}

function downloadText(filename, text, mimeType = "text/plain;charset=utf-8;") {
  return downloadHelpers.downloadText({ filename, text, mimeType });
}

function buildDownloadFilename(stem, ext, { includeGroup = false, template = null, group = null } = {}) {
  return downloadHelpers.buildDownloadFilename({
    state,
    refs,
    stem,
    ext,
    includeGroup,
    template,
    group,
  });
}

function cohortTableOutputGroup() {
  // The table's own grouping, not whatever Group by currently shows.
  return String(requestConfigFromPayload(state.cohort)?.group_column || "");
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

async function downloadChecklist(report, format, stem) {
  // A REMARK or TRIPOD+AI checklist from the server, as a Word or Markdown file.
  const response = await fetch(apiUrl("/api/checklist-export"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...report, format }),
  });
  if (!response.ok) {
    let message = "Checklist export failed.";
    try {
      const detail = (await response.json())?.detail;
      if (typeof detail === "string" && detail.trim()) message = detail.trim();
    } catch {
      // Keep the generic message when the error body is not JSON.
    }
    throw new Error(message);
  }
  triggerBlobDownload(buildDownloadFilename(stem, format === "docx" ? "docx" : "md"), await response.blob());
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

function exportColumnsFromRows(rows) {
  const seen = new Set();
  const ordered = [];
  (rows || []).forEach((row) => {
    Object.keys(row || {}).forEach((column) => {
      if (!column || column.startsWith("_") || seen.has(column)) return;
      seen.add(column);
      ordered.push(column);
    });
  });
  return ordered;
}

function exportNotesFromScientificSummary(summary, extraNotes = []) {
  const notes = [];
  if (summary?.headline) notes.push(summary.headline);
  (summary?.cautions || []).forEach((item) => {
    if (item) notes.push(item);
  });
  extraNotes.forEach((item) => {
    if (item) notes.push(item);
  });
  return [...new Set(notes)];
}

function exportDatasetHash(resultPayload) {
  // Exports record which stored dataset produced them (the server stamps it into the notes).
  return resultPayload?.dataset_hash || state.dataset?.dataset_hash || null;
}

function manuscriptExportPayload(manuscript, format, template, fallbackCaption, resultPayload = null) {
  const resultPayloadForHash = resultPayload;
  const analysis = resultPayload?.analysis || {};
  const requestConfig = resultPayload?.request_config || null;
  return {
    rows: manuscript?.model_performance_table || [],
    columns: exportColumnsFromRows(manuscript?.model_performance_table || []),
    format,
    style: "journal",
    template,
    caption: manuscript?.caption || fallbackCaption,
    notes: manuscript?.table_notes || [],
    provenance: {
      dataset_hash: exportDatasetHash(resultPayloadForHash),
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

function buildCoxTableExportPayload(rows, caption, resultPayload = null) {
  const resultPayloadForHash = resultPayload;
  const analysis = resultPayload?.analysis || {};
  const stats = analysis?.model_stats || {};
  const strataColumns = Array.isArray(analysis?.strata_columns) ? analysis.strata_columns : [];
  const notes = exportNotesFromScientificSummary(
    analysis?.scientific_summary,
    ["The global PH row is the Grambsch-Therneau score test on scaled Schoenfeld residuals versus log time (omnibus across terms)."],
  );
  if (strataColumns.length) {
    notes.push(`Strata variables: ${strataColumns.join(", ")}.`);
  }
  if (stats?.evaluation_mode === "stratified_not_reported") {
    notes.push("Pooled C-index is intentionally omitted for stratified Cox because cross-stratum ranking is not directly interpretable.");
  }
  return {
    rows: rows || [],
    columns: exportColumnsFromRows(rows || []),
    format: "csv",
    style: "plain",
    caption,
    notes,
    provenance: {
      dataset_hash: exportDatasetHash(resultPayloadForHash),
      request_config: resultPayload?.request_config || null,
      analysis: {
        formula: analysis?.formula,
        tie_method: stats?.tie_method,
        n: stats?.n,
        events: stats?.events,
        parameters: stats?.parameters,
        evaluation_mode: stats?.evaluation_mode,
        c_index_label: stats?.c_index_label,
        strata_columns: strataColumns,
        n_strata: stats?.n_strata,
        zero_event_strata_count: stats?.zero_event_strata_count,
        sparse_event_strata_count: stats?.sparse_event_strata_count,
      },
    },
  };
}

function buildKmTableExportPayload(rows, caption, resultPayload = null, { includePairwiseGuardrail = false } = {}) {
  const resultPayloadForHash = resultPayload;
  const analysis = resultPayload?.analysis || {};
  const extraNotes = [];
  if (analysis?.outcome_informed_group) {
    extraNotes.push("This grouping used outcome information. Treat the KM/table output as descriptive rather than confirmatory.");
  }
  if (
    includePairwiseGuardrail
    && analysis?.test?.p_value != null
    && Number(analysis.test.p_value) >= 0.05
  ) {
    extraNotes.push("The omnibus group comparison was not statistically significant. Interpret pairwise rows only if that analysis path was pre-specified.");
  }
  return {
    rows: rows || [],
    columns: exportColumnsFromRows(rows || []),
    format: "csv",
    style: "plain",
    caption,
    notes: exportNotesFromScientificSummary(analysis?.scientific_summary, extraNotes),
    provenance: {
      dataset_hash: exportDatasetHash(resultPayloadForHash),
      request_config: resultPayload?.request_config || null,
      analysis: {
        test: analysis?.test || null,
        outcome_informed_group: analysis?.outcome_informed_group || false,
      },
    },
  };
}

function buildSignatureTableExportPayload(rows, caption, resultPayload = null) {
  const resultPayloadForHash = resultPayload;
  return {
    rows: rows || [],
    columns: exportColumnsFromRows(rows || []),
    format: "csv",
    style: "plain",
    caption,
    notes: exportNotesFromScientificSummary(
      resultPayload?.scientific_summary,
      ["Signature discovery is exploratory and should be treated as hypothesis-generating until externally validated."],
    ),
    provenance: {
      dataset_hash: exportDatasetHash(resultPayloadForHash),
      request_config: resultPayload?.request_config || null,
      analysis: {
        best_split: resultPayload?.best_split || null,
        search_space: resultPayload?.search_space || null,
      },
    },
  };
}

function buildComparisonTableExportPayload(rows, caption, resultPayload = null) {
  const resultPayloadForHash = resultPayload;
  const analysis = resultPayload?.analysis || {};
  const extraNotes = [];
  if (analysis?.evaluation_mode === "mixed_holdout_apparent") {
    extraNotes.push("This comparison mixes holdout-comparable models with apparent-only screening rows. Do not treat the ranking as a single external-validation table.");
  }
  return {
    rows: rows || [],
    columns: exportColumnsFromRows(rows || []),
    format: "csv",
    style: "plain",
    caption,
    notes: exportNotesFromScientificSummary(analysis?.scientific_summary, extraNotes),
    provenance: {
      dataset_hash: exportDatasetHash(resultPayloadForHash),
      request_config: resultPayload?.request_config || null,
      analysis: {
        evaluation_mode: analysis?.evaluation_mode,
        cv_folds: analysis?.cv_folds,
        cv_repeats: analysis?.cv_repeats,
      },
    },
  };
}

function buildCohortTableExportPayload(format = "xlsx") {
  const payload = state.cohort;
  const resultPayloadForHash = payload;
  const tableState = currentCohortTableOutputState();
  const requestConfig = requestConfigFromPayload(payload) || currentGoalRequestConfig("tables");
  const notes = [...cohortTableAnalysisNotes(payload)];
  if (tableState.hasOutput && !tableState.isCurrent) {
    notes.push("Current visible settings no longer match this table. Rebuild Table before sharing if you need the latest selections.");
  }
  return {
    rows: payload?.analysis?.rows || [],
    columns: payload?.analysis?.columns || exportColumnsFromRows(payload?.analysis?.rows || []),
    format,
    style: "plain",
    caption: tableState.outputGroupLabel === "overall only"
      ? "Cohort summary table"
      : `Cohort summary table by ${tableState.outputGroupLabel}`,
    notes,
    provenance: {
      dataset_hash: exportDatasetHash(resultPayloadForHash),
      request_config: requestConfig,
      analysis: {
        output_group_label: tableState.outputGroupLabel,
        output_variables: tableState.outputVariables,
        is_current: tableState.isCurrent,
      },
    },
  };
}

function downloadPlotImage(plotEl, filename, format) {
  return downloadHelpers.downloadPlotImage({ plotEl, filename, format }).catch((error) => {
    showError(`Saving the ${String(format || "image").toUpperCase()} image failed: ${errorMessageText(error, "the chart could not be exported.")}`);
  });
}

function requireCurrentResultForExport(goal, { payload = null } = {}) {
  const scope = runScopeForGoal(goal);
  if (scope && isScopeBusy(scope)) {
    showToast("Wait for the current run to finish before exporting this result.", "warning", 3200);
    return false;
  }
  if (goal === "tables") {
    const tableState = currentCohortTableOutputState();
    if (!tableState.hasOutput || !payload) {
      showToast("Build the cohort table before exporting.", "warning", 3200);
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

function requireCurrentPlotForExport(plotEl, payload) {
  if (plotShowsResult(plotEl, payload)) return true;
  showToast("The visible chart does not belong to the current result. Run again before exporting the image.", "warning", 3600);
  return false;
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

function syncLockedTestControls(goal, isRepeatedCv) {
  const wrap = goal === "dl" ? refs.dlLockedTestWrap : refs.mlLockedTestWrap;
  const fractionWrap = goal === "dl" ? refs.dlLockedTestFractionWrap : refs.mlLockedTestFractionWrap;
  const { toggle, input } = lockedTestControls(goal);
  wrap?.classList.toggle("hidden", !isRepeatedCv);
  fractionWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (toggle) toggle.disabled = !isRepeatedCv;
  if (input) input.disabled = !isRepeatedCv || !toggle?.checked;
}

function updateMlEvaluationControls() {
  const isRepeatedCv = refs.mlEvaluationStrategy?.value === "repeated_cv";
  refs.mlCvFoldsWrap?.classList.toggle("hidden", !isRepeatedCv);
  refs.mlCvRepeatsWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (refs.mlCvFolds) refs.mlCvFolds.disabled = !isRepeatedCv;
  if (refs.mlCvRepeats) refs.mlCvRepeats.disabled = !isRepeatedCv;
  syncLockedTestControls("ml", isRepeatedCv);
  syncAnalysisRunButtonAvailability();
  renderWorkspaceChrome();
}

function updateDlEvaluationControls() {
  const isRepeatedCv = refs.dlEvaluationStrategy?.value === "repeated_cv";
  refs.dlCvFoldsWrap?.classList.toggle("hidden", !isRepeatedCv);
  refs.dlCvRepeatsWrap?.classList.toggle("hidden", !isRepeatedCv);
  if (refs.dlCvFolds) refs.dlCvFolds.disabled = !isRepeatedCv;
  if (refs.dlCvRepeats) refs.dlCvRepeats.disabled = !isRepeatedCv;
  if (refs.dlParallelJobs) refs.dlParallelJobs.disabled = !isRepeatedCv;
  syncLockedTestControls("dl", isRepeatedCv);
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
  if (!el) return;
  if (el.__stableResetAxesState) delete el.__stableResetAxesState;
  if (el.__resultPayload) delete el.__resultPayload;
  el.style.height = "";
  el.classList.remove("is-refreshing");
  el.removeAttribute("aria-busy");
  setPlotStale(el, false);
  if (el.data || el._fullLayout) { try { Plotly.purge(el); } catch { /* ignore */ } }
}

// A plot whose result no longer matches the visible settings stays drawn under a notice instead of being
// purged, so it is back as soon as the settings match its result again (for example after switching the
// model away and back, or undoing an edit).
function setPlotStale(el, stale, message = "") {
  if (!el) return;
  el.classList.toggle("plot-stale", Boolean(stale));
  if (stale) {
    el.dataset.staleMessage = message;
  } else if (el.dataset && "staleMessage" in el.dataset) {
    delete el.dataset.staleMessage;
  }
}

function resetPlotElement(el, html = "") {
  // Always purge Plotly state before replacing markup; clearing innerHTML alone leaves
  // `data`/`_fullLayout` behind, which made PNG/SVG exports reuse a previous chart.
  if (!el) return;
  purgePlot(el);
  el.innerHTML = html;
}

function markPlotResult(el, payload) {
  if (el) el.__resultPayload = payload || null;
}

function plotShowsResult(el, payload) {
  return Boolean(payload && el?.data?.length && el.__resultPayload === payload);
}

function plotIsDisplayed(el) {
  if (!el?.isConnected || el.closest(".hidden")) return false;
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).display !== "none";
}

function resizePlotIfDisplayed(el) {
  if (!el?.data?.length || !el?._fullLayout || !plotIsDisplayed(el)) return;
  try {
    Plotly.Plots.resize(el);
  } catch {
    // Hidden or detached plots cannot be resized; the next visible render resizes them.
  }
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

function coxMartingalePanels() {
  return Array.isArray(state.cox?.analysis?.martingale_plot_data)
    ? state.cox.analysis.martingale_plot_data.filter((panel) => panel && panel.term)
    : [];
}

function resetCoxMartingaleSelector(emptyLabel = "Run Cox first") {
  runtime.coxMartingaleTerm = "";
  if (!refs.coxMartingaleVariableSelect) return;
  refs.coxMartingaleVariableSelect.innerHTML = "";
  const option = document.createElement("option");
  option.value = "";
  option.textContent = emptyLabel;
  refs.coxMartingaleVariableSelect.appendChild(option);
  refs.coxMartingaleVariableSelect.disabled = true;
  refs.coxMartingaleVariableField?.classList.add("is-disabled");
}

function syncCoxMartingaleSelector(panels, preferredTerm = runtime.coxMartingaleTerm) {
  if (!refs.coxMartingaleVariableSelect) return "";
  const terms = panels
    .map((panel) => String(panel?.term || "").trim())
    .filter(Boolean);
  refs.coxMartingaleVariableSelect.innerHTML = "";
  if (!terms.length) {
    resetCoxMartingaleSelector("No continuous covariates");
    return "";
  }
  const normalizedPreferred = terms.includes(preferredTerm) ? preferredTerm : terms[0];
  terms.forEach((term) => {
    const option = document.createElement("option");
    option.value = term;
    option.textContent = term;
    option.selected = term === normalizedPreferred;
    refs.coxMartingaleVariableSelect.appendChild(option);
  });
  refs.coxMartingaleVariableSelect.disabled = terms.length <= 1;
  refs.coxMartingaleVariableField?.classList.toggle("is-disabled", terms.length <= 1);
  runtime.coxMartingaleTerm = normalizedPreferred;
  return normalizedPreferred;
}

function currentCoxMartingaleUnavailableMessage() {
  const note = String(state.cox?.analysis?.model_stats?.martingale_note || "").trim();
  return note || "Martingale residual screening was unavailable for this fit.";
}

function currentCoxMartingaleEmptyLabel() {
  const note = currentCoxMartingaleUnavailableMessage().toLowerCase();
  if (note.includes("no continuous covariates")) return "No continuous covariates";
  return "Diagnostics unavailable";
}

function martingaleResidualAxisRange(residualValues, trendValues) {
  const residuals = residualValues.filter((value) => Number.isFinite(value));
  if (residuals.length < 12) return null;
  const trends = trendValues.filter((value) => Number.isFinite(value));
  const sortedResiduals = [...residuals].sort((a, b) => a - b);
  const fullMin = sortedResiduals[0];
  const fullMax = sortedResiduals[sortedResiduals.length - 1];
  const fullSpan = fullMax - fullMin;
  if (!(fullSpan > 0)) return null;
  const quantile = (q) => {
    const index = (sortedResiduals.length - 1) * q;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sortedResiduals[lower];
    const fraction = index - lower;
    return sortedResiduals[lower] + (sortedResiduals[upper] - sortedResiduals[lower]) * fraction;
  };
  const qLow = quantile(0.05);
  const qHigh = quantile(0.95);
  const trendMin = trends.length ? Math.min(...trends) : 0;
  const trendMax = trends.length ? Math.max(...trends) : 0;
  const coreMin = Math.min(qLow, 0, trendMin);
  const coreMax = Math.max(qHigh, 0, trendMax);
  const coreSpan = coreMax - coreMin;
  if (!(coreSpan > 0) || fullSpan < coreSpan * 3) return null;
  const padding = Math.max(0.06, coreSpan * 0.12);
  const candidate = [coreMin - padding, coreMax + padding];
  const outlierCount = residuals.filter((value) => value < candidate[0] || value > candidate[1]).length;
  return outlierCount ? candidate : null;
}

function buildCoxMartingaleFigure(panel) {
  const term = String(panel?.term || "Covariate");
  const values = Array.isArray(panel?.value) ? panel.value : [];
  const residuals = Array.isArray(panel?.residual) ? panel.residual : [];
  // Number(null) is 0, so map missing values to NaN before filtering instead of plotting them at y=0.
  const finiteOrNaN = (value) => (value === null || value === undefined || value === "" ? NaN : Number(value));
  const pointPairs = values
    .map((value, index) => [finiteOrNaN(value), finiteOrNaN(residuals[index])])
    .filter(([xValue, yValue]) => Number.isFinite(xValue) && Number.isFinite(yValue));
  const x = pointPairs.map(([xValue]) => xValue);
  const y = pointPairs.map(([, yValue]) => yValue);
  const trendValues = Array.isArray(panel?.trend_value) ? panel.trend_value : [];
  const trendResiduals = Array.isArray(panel?.trend_residual) ? panel.trend_residual : [];
  // Keep missing trend estimates as null so Plotly leaves a gap rather than drawing them at 0.
  const trendPairs = trendValues
    .map((value, index) => [finiteOrNaN(value), finiteOrNaN(trendResiduals[index])])
    .filter(([xValue]) => Number.isFinite(xValue))
    .map(([xValue, yValue]) => [xValue, Number.isFinite(yValue) ? yValue : null]);
  const trendX = trendPairs.map(([xValue]) => xValue);
  const trendY = trendPairs.map(([, yValue]) => yValue);
  return {
    data: [
      {
        type: "scatter",
        x,
        y,
        mode: "markers",
        name: term,
        marker: { size: 6, color: "#0891b2", opacity: 0.55 },
        hovertemplate: `${term}<br>Value: %{x:.3f}<br>Martingale residual: %{y:.3f}<extra></extra>`,
        showlegend: false,
      },
      ...(trendX.length && trendY.some((value) => value !== null) ? [{
        type: "scatter",
        x: trendX,
        y: trendY,
        mode: "lines",
        connectgaps: false,
        line: { width: 2.5, color: "rgba(13, 148, 136, 0.95)" },
        hoverinfo: "skip",
        showlegend: false,
      }] : []),
    ],
    layout: {
      template: "simple_white",
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "white",
      font: { family: "Sora, sans-serif", size: 13, color: "#1a2332" },
      margin: { l: 60, r: 30, t: 72, b: 68 },
      height: 360,
      xaxis: {
        title: term,
        linecolor: "rgba(0,0,0,0.15)",
        gridcolor: "rgba(0,0,0,0.04)",
      },
      yaxis: {
        title: "Martingale residual",
        linecolor: "rgba(0,0,0,0.15)",
        gridcolor: "rgba(0,0,0,0.04)",
        range: martingaleResidualAxisRange(y, trendY) || undefined,
      },
      shapes: [
        {
          type: "line",
          xref: "paper",
          yref: "y",
          x0: 0,
          x1: 1,
          y0: 0,
          y1: 0,
          line: { width: 1, dash: "dot", color: "rgba(90, 103, 118, 0.7)" },
        },
      ],
    },
  };
}

async function renderCoxMartingalePlot(selectedTerm = runtime.coxMartingaleTerm) {
  const panels = coxMartingalePanels();
  if (!panels.length) {
    resetCoxMartingaleSelector(currentCoxMartingaleEmptyLabel());
    clearPlotShell(refs.coxMartingalePlot, `<div class="empty-state plot-empty"><span>${escapeHtml(currentCoxMartingaleUnavailableMessage())}</span></div>`);
    return;
  }
  const term = syncCoxMartingaleSelector(panels, selectedTerm);
  const panel = panels.find((item) => String(item.term) === term) || panels[0];
  const figure = buildCoxMartingaleFigure(panel);
  resetPlotElement(refs.coxMartingalePlot);
  await Plotly.newPlot(
    refs.coxMartingalePlot,
    figure.data,
    plotLayoutConfig(figure.layout, `cox_martingale_${term}`),
    plotConfig(`cox_martingale_${term}`),
  );
  stabilizePlotShellHeight(refs.coxMartingalePlot);
}

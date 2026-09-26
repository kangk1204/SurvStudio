// SurvStudio front end, part 8/9: Honest marker evaluation, its locked model, and external validation.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

const MARKER_TABLE_DISPLAY_LIMIT = 300;

function markerCandidateColumns() {
  return modelFeatureCandidateColumns().filter((name) => getColumnMeta(name)?.kind === "numeric");
}

// Markers are numeric columns; clinical covariates start from the Cox selection and are left out of the markers.
function refreshMarkerSelections() {
  if (!state.dataset || !refs.markerChecklist || !refs.markerClinicalChecklist) return;
  const markerCandidates = markerCandidateColumns();
  const clinicalCandidates = modelFeatureCandidateColumns();
  const previousMarkers = selectedCheckboxValues(refs.markerChecklist).filter((value) => markerCandidates.includes(value));
  const previousClinical = selectedCheckboxValues(refs.markerClinicalChecklist).filter((value) => clinicalCandidates.includes(value));
  const clinical = previousMarkers.length || previousClinical.length
    ? previousClinical
    : currentCoxSelections().covariates.filter((value) => clinicalCandidates.includes(value));
  const markers = previousMarkers.length ? previousMarkers : markerCandidates.filter((value) => !clinical.includes(value));
  renderChecklist(refs.markerChecklist, markerCandidates, markers);
  renderChecklist(refs.markerClinicalChecklist, clinicalCandidates, clinical);
  renderMarkerSelectionLine();
}

function currentMarkerSelections() {
  const markers = selectedCheckboxValues(refs.markerChecklist);
  const clinical = selectedCheckboxValues(refs.markerClinicalChecklist).filter((value) => !markers.includes(value));
  const categoricalCandidates = new Set(sharedModelCategoricalCandidates());
  return { markers, clinical, categorical: clinical.filter((value) => categoricalCandidates.has(value)) };
}

function renderMarkerSelectionLine() {
  if (!refs.markerSelectionLine) return;
  const { markers, clinical } = currentMarkerSelections();
  if (!state.dataset) {
    refs.markerSelectionLine.textContent = "";
    return;
  }
  refs.markerSelectionLine.textContent = markers.length
    ? `${formatValue(markers.length)} marker${markers.length === 1 ? "" : "s"}, ${clinical.length ? `judged on added value over ${formatValue(clinical.length)} clinical covariate${clinical.length === 1 ? "" : "s"}` : "judged on marginal association (no clinical covariates)"}.`
    : "Choose at least one numeric marker.";
}

function markerRequestFields() {
  const { markers, clinical, categorical } = currentMarkerSelections();
  return {
    marker_columns: markers,
    clinical_columns: clinical,
    categorical_clinical: categorical,
    n_permutations: numericControlValue(refs.markerPermutations, 1000),
    n_resamples: numericControlValue(refs.markerResamples, 200),
    random_seed: numericControlValue(refs.markerRandomSeed, 20260926),
    nonlinear_lens: refs.markerNonlinearLens?.value || "off",
  };
}

async function runMarkerEvaluation() {
  const base = currentBaseConfig();
  const fields = markerRequestFields();
  if (!fields.marker_columns.length) throw new Error("Choose at least one marker.");
  const requestToken = beginRequestToken("markers");
  const datasetId = base.dataset_id;
  const loading = beginShellLoading([refs.markersStabilityPlot]);
  setRuntimeBanner(
    `Evaluating ${formatValue(fields.marker_columns.length)} marker(s) with ${formatValue(fields.n_permutations)} permutations and ${formatValue(fields.n_resamples)} subsamples. Large panels take a few minutes.`,
    "info",
  );
  let payload;
  try {
    payload = await fetchJSON("/api/marker-evaluation", {
      signal: requestSignal("markers"),
      method: "POST",
      body: JSON.stringify({ ...base, ...fields }),
    });
  } catch (error) {
    if (requestTokenMatches("markers", requestToken)) loading.restore();
    throw error;
  } finally {
    if (requestTokenMatches("markers", requestToken)) setRuntimeBanner("");
  }
  if (!requestTokenMatches("markers", requestToken) || state.dataset?.dataset_id !== datasetId) return;
  loading.finish();
  state.markers = payload;
  state.markerValidation = null;
  clearMarkerValidationOutput();
  await renderMarkerResults(payload);
  syncDownloadButtonAvailability();
  revealCompletedResultIfCurrent("markers", {
    successMessage: "Marker evaluation complete.",
    backgroundMessage: "Marker evaluation finished in the background. Open the Markers tab to review it.",
  });
}

function markerSummary(payload) {
  const analysis = payload?.analysis || {};
  const counts = analysis.tier_counts || {};
  const cohort = analysis.cohort || {};
  const signature = analysis.signature || {};
  const settings = analysis.settings || {};
  const robust = Number(counts.robust || 0);
  const suggestive = Number(counts.suggestive || 0);
  const evaluated = Number(cohort.n_markers_evaluated || 0);
  const addedValue = analysis.primary_lens === "added_value";
  const headline = robust
    ? `${formatValue(robust)} of ${formatValue(evaluated)} markers are robust${addedValue ? " beyond the clinical covariates" : ""}.`
    : suggestive
      ? `No marker is robust; ${formatValue(suggestive)} show evidence that does not hold up across subsamples.`
      : `No marker shows ${addedValue ? "added value beyond the clinical covariates" : "an association"} after family-wise error control.`;
  const cautions = [];
  if (!addedValue) cautions.push("No clinical covariates were given, so markers are judged on marginal association only. Add clinical covariates to test added value.");
  if (signature.signature_optimism != null && Number(signature.signature_optimism) > 0.02) {
    cautions.push(`The selected-marker model's apparent C-index is optimistic by about ${formatValue(signature.signature_optimism)}; report the corrected value.`);
  }
  if ((cohort.dropped_markers || []).length) {
    cautions.push(`${formatValue(cohort.dropped_markers.length)} marker(s) were left out because they were constant or mostly missing.`);
  }
  if (Number(counts["marginal only"] || 0) > 0) {
    cautions.push(`${formatValue(counts["marginal only"])} marker(s) are associated with survival but add nothing beyond the clinical covariates.`);
  }
  return {
    status: robust ? "robust" : "review",
    headline,
    metrics: [
      { label: "Patients", value: cohort.n },
      { label: "Events", value: cohort.events },
      { label: "Markers", value: evaluated },
      { label: "Robust", value: robust },
      { label: "Suggestive", value: suggestive },
      { label: "Model C (corrected)", value: signature.optimism_corrected_c },
    ],
    strengths: [
      `Family-wise p-values (Westfall-Young) from ${formatValue(analysis.null?.n_permutations)} permutations${analysis.null?.lens2_null === "freedman_lane" ? ", keeping each marker's link to the clinical covariates" : ""}.`,
      `The whole screen was repeated on ${formatValue(analysis.resampling?.n_valid)} subsamples of ${Math.round(100 * Number(analysis.resampling?.fraction || 0.632))}% of the patients.`,
      `Robust: family-wise p ≤ ${formatValue(settings.alpha)}, selected in ≥ ${Math.round(100 * Number(settings.robust_frequency || 0.5))}% of subsamples and the same direction in ≥ ${Math.round(100 * Number(settings.robust_direction || 0.9))}%.`,
    ],
    cautions,
    next_steps: [
      robust ? "Validate the locked model in an independent cohort below before claiming the markers." : "Treat suggestive markers as hypotheses for an independent cohort.",
      "Report the optimism-corrected C-index rather than the apparent one.",
    ],
  };
}

function markerMetaBanner(payload) {
  const analysis = payload?.analysis || {};
  const cohort = analysis.cohort || {};
  const signature = analysis.signature || {};
  const lens = analysis.primary_lens === "added_value" ? "added value over clinical covariates" : "marginal association";
  const parts = [
    `N=${formatValue(cohort.n)}`,
    `events=${formatValue(cohort.events)}`,
    `markers=${formatValue(cohort.n_markers_evaluated)}`,
    `tested for ${lens}`,
  ];
  if (signature.apparent_c != null) parts.push(`model C apparent=${formatValue(signature.apparent_c)}, corrected=${formatValue(signature.optimism_corrected_c)}`);
  return parts.join(", ");
}

async function renderMarkerPlot(plot, figure, name, payload) {
  if (!plot) return;
  if (!figure?.data?.length) {
    clearPlotShell(plot, '<div class="empty-state plot-empty"><span>No subsamples were run, so this plot is empty.</span></div>');
    return;
  }
  resetPlotElement(plot);
  await Plotly.newPlot(plot, figure.data, plotLayoutConfig(figure.layout || {}, name), plotConfig(name));
  markPlotResult(plot, payload);
  stabilizePlotShellHeight(plot);
}

async function renderMarkerResults(payload) {
  renderInsightBoard(refs.markersInsightBoard, markerSummary(payload), "Run the evaluation to see which markers hold up.");
  refs.markersMetaBanner.textContent = markerMetaBanner(payload);
  await renderMarkerPlot(refs.markersStabilityPlot, payload.stability_figure, "marker_stability", payload);
  await renderMarkerPlot(refs.markersRankPlot, payload.rank_figure, "marker_ranks", payload);
  const rows = payload.display_table || [];
  renderTable(refs.markersTableShell, rows.slice(0, MARKER_TABLE_DISPLAY_LIMIT));
  if (refs.markersTableNote) {
    refs.markersTableNote.textContent = rows.length > MARKER_TABLE_DISPLAY_LIMIT
      ? `Showing the first ${formatValue(MARKER_TABLE_DISPLAY_LIMIT)} of ${formatValue(rows.length)} markers, strongest first. Export the table for all of them.`
      : "Strongest markers first. HR per unit of the marker, from a Cox model with the clinical covariates.";
  }
  if (refs.markerValidationSection) refs.markerValidationSection.classList.remove("hidden");
  updateResultVisibility();
}

function clearMarkerOutputs() {
  state.markers = null;
  state.markerValidation = null;
  if (refs.markersInsightBoard) refs.markersInsightBoard.innerHTML = '<div class="empty-state">Run the evaluation to see which markers hold up.</div>';
  if (refs.markersMetaBanner) refs.markersMetaBanner.textContent = "";
  clearPlotShell(refs.markersStabilityPlot, '<div class="empty-state plot-empty"><span>Choose markers and click <strong>Run Analysis</strong>.</span></div>', { state: "placeholder" });
  clearPlotShell(refs.markersRankPlot, "", { state: "placeholder" });
  if (refs.markersTableShell) refs.markersTableShell.innerHTML = '<div class="empty-state">Run the evaluation to fill the marker table.</div>';
  refs.markerValidationSection?.classList.add("hidden");
  clearMarkerValidationOutput();
}

function clearMarkerValidationOutput() {
  if (refs.markerValidationSummary) refs.markerValidationSummary.innerHTML = "";
  if (refs.markerValidationShell) refs.markerValidationShell.innerHTML = "";
  clearPlotShell(refs.markerValidationPlot, "", { state: "placeholder" });
  refs.markerValidationPlot?.classList.add("hidden");
}

// The external cohort is uploaded as its own dataset so the workspace keeps the development cohort.
async function runMarkerValidation() {
  const recipe = state.markers?.analysis?.locked_recipe;
  if (!recipe) throw new Error("Run the marker evaluation first; it produces the locked model to validate.");
  if (!currentGoalResult("markers")) throw new Error("Settings changed since the evaluation. Run it again before validating.");
  const file = refs.markerValidationFile?.files?.[0];
  if (!file) throw new Error("Choose the external cohort file first.");
  const requestToken = beginRequestToken("markerValidation");
  const sourceDatasetId = state.dataset?.dataset_id;
  setRuntimeBanner(`Uploading ${file.name} and applying the locked model unchanged.`, "info");
  try {
    const form = new FormData();
    form.append("file", file);
    const external = await fetchJSON("/api/upload", { signal: requestSignal("markerValidation"), method: "POST", body: form });
    if (!requestTokenMatches("markerValidation", requestToken)) return;
    const payload = await fetchJSON("/api/marker-validation", {
      signal: requestSignal("markerValidation"),
      method: "POST",
      body: JSON.stringify({ dataset_id: external.dataset_id, recipe }),
    });
    if (!requestTokenMatches("markerValidation", requestToken) || state.dataset?.dataset_id !== sourceDatasetId) return;
    state.markerValidation = { ...payload, external_filename: file.name };
    await renderMarkerValidation(state.markerValidation);
    syncDownloadButtonAvailability();
    showToast(`Locked model applied to ${file.name}.`, "success", 3200);
  } finally {
    if (requestTokenMatches("markerValidation", requestToken)) setRuntimeBanner("");
  }
}

function markerValidationMetrics(validation) {
  const metrics = validation?.metrics || {};
  const interval = (values) => (Array.isArray(values) && values[0] != null ? ` (${formatValue(values[0])} to ${formatValue(values[1])})` : "");
  const rows = [
    { label: "C-index", value: `${formatValue(metrics.c_index)}${interval(metrics.c_index_ci)}` },
    { label: "Calibration slope", value: `${formatValue(metrics.calibration_slope)}${interval(metrics.calibration_slope_ci)}` },
  ];
  if (metrics.delta_c_index != null) rows.push({ label: "C gain over clinical", value: `${formatValue(metrics.delta_c_index)}${interval(metrics.delta_c_index_ci)}` });
  if (metrics.observed_expected_ratio != null) rows.push({ label: `Observed/expected at ${formatValue(metrics.horizon)}`, value: formatValue(metrics.observed_expected_ratio) });
  if (metrics.brier_skill != null) rows.push({ label: "Brier skill", value: formatValue(metrics.brier_skill) });
  return rows;
}

async function renderMarkerValidation(payload) {
  const validation = payload?.validation || {};
  const cohort = validation.cohort || {};
  const replicated = (validation.markers || []).filter((row) => row.replicated).length;
  const total = (validation.markers || []).length;
  const metrics = markerValidationMetrics(validation);
  if (refs.markerValidationSummary) {
    refs.markerValidationSummary.innerHTML = `
      <p class="marker-validation-headline">${escapeHtml(`${payload.external_filename || "External cohort"}: ${formatValue(cohort.n)} patients, ${formatValue(cohort.events)} events. ${formatValue(replicated)} of ${formatValue(total)} markers replicated.`)}</p>
      <div class="insight-metrics">${metrics.map((metric) => `<div class="metric-pill"><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong></div>`).join("")}</div>
      ${(validation.notes || []).length ? `<ul class="insight-cautions">${validation.notes.map(escapeListItem).join("")}</ul>` : ""}
    `;
  }
  if (refs.markerValidationPlot && payload.figure?.data?.length) {
    refs.markerValidationPlot.classList.remove("hidden");
    resetPlotElement(refs.markerValidationPlot);
    await Plotly.newPlot(refs.markerValidationPlot, payload.figure.data, plotLayoutConfig(payload.figure.layout || {}, "marker_replication"), plotConfig("marker_replication"));
    stabilizePlotShellHeight(refs.markerValidationPlot);
  }
  renderTable(refs.markerValidationShell, markerValidationRows(validation));
}

function markerValidationRows(validation) {
  return (validation?.markers || []).map((row) => {
    const tested = row.adjusted || row.marginal || {};
    return {
      Marker: row.marker,
      "HR per unit": tested.hazard_ratio,
      "CI lower": tested.ci_lower,
      "CI upper": tested.ci_upper,
      "Same direction": row.same_direction ? "yes" : "no",
      "Replication P (Holm)": row.replication_p_holm,
      Replicated: row.replicated ? "yes" : "no",
    };
  });
}

function downloadMarkerRecipe() {
  const payload = currentGoalResult("markers");
  if (!requireCurrentResultForExport("markers", { payload })) return;
  const recipe = payload.analysis?.locked_recipe;
  if (!recipe) {
    showToast("No marker was selected, so there is no locked model to export.", "warning", 3600);
    return;
  }
  downloadText(buildDownloadFilename("locked_marker_model", "json"), `${JSON.stringify(recipe, null, 2)}\n`, "application/json;charset=utf-8;");
}

function downloadMarkerTable() {
  const payload = currentGoalResult("markers");
  if (!requireCurrentResultForExport("markers", { payload })) return;
  downloadCsv(buildDownloadFilename("marker_evaluation", "csv"), payload.display_table || [], null, {
    caption: "Marker evaluation",
    notes: [markerMetaBanner(payload)],
  });
}

function syncMarkerDownloadButtons() {
  const current = currentGoalResult("markers");
  if (refs.downloadMarkersCsvButton) refs.downloadMarkersCsvButton.disabled = !current;
  if (refs.downloadMarkerRecipeButton) refs.downloadMarkerRecipeButton.disabled = !current?.analysis?.locked_recipe;
  const stabilityCurrent = plotShowsResult(refs.markersStabilityPlot, current);
  if (refs.downloadMarkersStabilityPngButton) refs.downloadMarkersStabilityPngButton.disabled = !stabilityCurrent;
  if (refs.downloadMarkersRankPngButton) refs.downloadMarkersRankPngButton.disabled = !plotShowsResult(refs.markersRankPlot, current);
  if (refs.runMarkerValidationButton) {
    refs.runMarkerValidationButton.disabled = !current?.analysis?.locked_recipe || isScopeBusy("markers");
  }
}

function wireMarkerControls() {
  refs.runMarkersButton?.addEventListener("click", () => withLoading(refs.runMarkersButton, runMarkerEvaluation, "markers"));
  refs.runMarkerValidationButton?.addEventListener("click", () => withLoading(refs.runMarkerValidationButton, runMarkerValidation));
  refs.markerChecklist?.addEventListener("change", () => { renderMarkerSelectionLine(); scheduleResultCurrencySync(); queueHistorySync(); });
  refs.markerClinicalChecklist?.addEventListener("change", () => { renderMarkerSelectionLine(); scheduleResultCurrencySync(); queueHistorySync(); });
  refs.markerSearchInput?.addEventListener("input", () => applyChecklistSearch(refs.markerChecklist));
  refs.markerClinicalSearchInput?.addEventListener("input", () => applyChecklistSearch(refs.markerClinicalChecklist));
  refs.selectAllMarkersButton?.addEventListener("click", () => {
    setCheckedValues(refs.markerChecklist, allCheckboxValues(refs.markerChecklist, { visibleOnly: true }));
    renderMarkerSelectionLine();
    scheduleResultCurrencySync();
    queueHistorySync();
  });
  refs.clearMarkersButton?.addEventListener("click", () => {
    setCheckedValues(refs.markerChecklist, []);
    renderMarkerSelectionLine();
    scheduleResultCurrencySync();
    queueHistorySync();
  });
  [refs.markerPermutations, refs.markerResamples, refs.markerRandomSeed, refs.markerNonlinearLens].filter(Boolean).forEach((control) => {
    control.addEventListener("change", () => { scheduleResultCurrencySync(); queueHistorySync(); });
  });
  refs.downloadMarkersCsvButton?.addEventListener("click", downloadMarkerTable);
  refs.downloadMarkerRecipeButton?.addEventListener("click", downloadMarkerRecipe);
  refs.downloadMarkersStabilityPngButton?.addEventListener("click", () => {
    const payload = currentGoalResult("markers");
    if (!requireCurrentPlotForExport(refs.markersStabilityPlot, payload)) return;
    void downloadPlotImage(refs.markersStabilityPlot, buildDownloadFilename("marker_stability", "png").replace(/\.png$/, ""), "png");
  });
  refs.downloadMarkersRankPngButton?.addEventListener("click", () => {
    const payload = currentGoalResult("markers");
    if (!requireCurrentPlotForExport(refs.markersRankPlot, payload)) return;
    void downloadPlotImage(refs.markersRankPlot, buildDownloadFilename("marker_ranks", "png").replace(/\.png$/, ""), "png");
  });
}

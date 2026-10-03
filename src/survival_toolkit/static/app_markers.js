// SurvStudio front end, part 8/9: marker evaluation, its locked model, and external validation.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

const MARKER_TABLE_DISPLAY_LIMIT = 300;

// An attached marker matrix (omics, POST /api/marker-matrix) replaces the marker checklist; the server
// matches its patients to the dataset through the chosen ID column at every run.
// Marker counts run to tens of thousands; thousands separators keep them readable.
function formatCount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("en-US") : "NA";
}

// C-indices are compared at the third decimal, so they keep it (0.650, not 0.65).
function formatCIndexValue(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "NA";
}

function markerMatrixAttached() {
  return Boolean(state.markerMatrix?.matrix_id);
}

function likelyIdColumn(columns) {
  const rows = Number(state.dataset?.n_rows || 0);
  const unique = columns.filter((column) => Number(column.n_unique || 0) === rows);
  const named = unique.find((column) => /(^|[^a-z])(id|patient|sample|subject|case|barcode)([^a-z]|$)/i.test(column.name));
  return (named || unique[0] || columns[0])?.name || "";
}

function refreshMarkerMatrixControls() {
  if (!refs.markerMatrixIdColumn) return;
  const columns = state.dataset?.columns || [];
  const names = columns.map((column) => column.name);
  const previous = refs.markerMatrixIdColumn.value;
  renderSelect(refs.markerMatrixIdColumn, names, { selected: names.includes(previous) ? previous : likelyIdColumn(columns) });
  renderMarkerMatrixState();
}

function renderMarkerMatrixState() {
  const matrix = state.markerMatrix;
  const attached = markerMatrixAttached();
  refs.markerMatrixStatus?.classList.toggle("hidden", !attached);
  if (attached && refs.markerMatrixSummary) {
    refs.markerMatrixSummary.textContent = `${matrix.filename}: ${formatCount(matrix.n_markers)} markers, ${formatCount(matrix.n_matched)} of ${formatCount(matrix.n_patients)} patients matched by ${matrix.id_column}.${matrix.id_note ? ` ${matrix.id_note}` : ""}`;
  }
  if (attached && refs.markerMatrixDetails) refs.markerMatrixDetails.open = true;
  refs.markerChecklist?.classList.toggle("matrix-attached", attached);
  refs.markerChecklist?.querySelectorAll("input").forEach((input) => { input.disabled = attached; });
  [refs.markerSearchInput, refs.markerMatrixIdColumn, refs.markerMatrixOrientation, refs.markerMatrixFile]
    .filter(Boolean)
    .forEach((control) => { control.disabled = attached; });
  // Select all and Clear belong to the "markers" busy scope too: they stay off while a run is in flight.
  const busy = isScopeBusy("markers");
  [refs.selectAllMarkersButton, refs.clearMarkersButton]
    .filter(Boolean)
    .forEach((button) => { button.disabled = attached || busy; });
  // Attach also stays off while its own upload is in flight (withLoading marks it "is-loading").
  if (refs.attachMarkerMatrixButton) {
    refs.attachMarkerMatrixButton.disabled = attached || refs.attachMarkerMatrixButton.classList.contains("is-loading");
  }
}

async function attachMarkerMatrix() {
  if (!state.dataset) throw new Error("Load a dataset first.");
  const file = refs.markerMatrixFile?.files?.[0];
  if (!file) throw new Error("Choose a marker matrix file first.");
  const idColumn = refs.markerMatrixIdColumn?.value;
  if (!idColumn) throw new Error("Choose the dataset's patient ID column.");
  const datasetId = state.dataset.dataset_id;
  const form = new FormData();
  form.append("file", file);
  form.append("dataset_id", datasetId);
  form.append("id_column", idColumn);
  form.append("orientation", refs.markerMatrixOrientation?.value || "auto");
  const payload = await fetchJSON("/api/marker-matrix", { method: "POST", body: form });
  if (state.dataset?.dataset_id !== datasetId) {
    // The matrix was matched to the patients of a dataset the workspace no longer shows: free it again.
    deleteMarkerMatrixOnServer(payload?.matrix_id);
    showToast("The dataset changed while the marker file was being attached, so it was not attached. Attach it again.", "warning", 5200);
    return;
  }
  state.markerMatrix = payload;
  if (refs.markerMatrixFile) refs.markerMatrixFile.value = "";
  renderMarkerMatrixState();
  renderMarkerSelectionLine();
  scheduleResultCurrencySync();
  showToast(`Attached ${formatCount(payload.n_markers)} markers; ${formatCount(payload.n_matched)} of ${formatCount(payload.n_patients)} patients matched.`, "success", 4000);
}

// Frees a marker matrix the workspace no longer uses, so it does not linger in the server's store.
function deleteMarkerMatrixOnServer(matrixId) {
  if (matrixId) fetch(apiUrl(`/api/marker-matrix/${encodeURIComponent(matrixId)}`), { method: "DELETE" }).catch(() => {});
}

function removeMarkerMatrix() {
  const matrixId = state.markerMatrix?.matrix_id;
  state.markerMatrix = null;
  deleteMarkerMatrixOnServer(matrixId);
  renderMarkerMatrixState();
  renderMarkerSelectionLine();
  scheduleResultCurrencySync();
}

function markerCandidateColumns() {
  return modelFeatureCandidateColumns().filter((name) => getColumnMeta(name)?.kind === "numeric");
}

// Why the evaluation would leave a column out (more than 20% missing, or a single value); "" when usable.
function markerExclusionNote(name) {
  const meta = getColumnMeta(name);
  if (!meta) return "";
  const missing = Number(meta.missing || 0);
  const total = missing + Number(meta.non_missing || 0);
  if (total && missing / total > 0.2) return `${Math.round((100 * missing) / total)}% missing`;
  if (Number(meta.n_unique ?? 2) < 2) return "one value";
  return "";
}

// Markers are numeric columns; clinical covariates start from the Cox selection and are left out of the markers.
// Like the other lists (refreshVariableSelections), a newly loaded dataset starts from the defaults and any other
// refresh keeps the user's choices, an emptied list included.
function refreshMarkerSelections({ useDefaults = false } = {}) {
  if (!state.dataset || !refs.markerChecklist || !refs.markerClinicalChecklist) return;
  const markerCandidates = markerCandidateColumns();
  const clinicalCandidates = modelFeatureCandidateColumns();
  const previousMarkers = selectedCheckboxValues(refs.markerChecklist).filter((value) => markerCandidates.includes(value));
  const previousClinical = selectedCheckboxValues(refs.markerClinicalChecklist).filter((value) => clinicalCandidates.includes(value));
  const clinical = useDefaults
    ? currentCoxSelections().covariates.filter((value) => clinicalCandidates.includes(value))
    : previousClinical;
  // A Map, so a column named "constructor" or "toString" has no note unless it earned one.
  const notes = new Map(markerCandidates.map((value) => [value, markerExclusionNote(value)]).filter(([, note]) => note));
  // Columns the evaluation would drop start unchecked, so the default run does not fail on them.
  const markers = useDefaults ? markerCandidates.filter((value) => !clinical.includes(value) && !notes.has(value)) : previousMarkers;
  renderChecklist(refs.markerChecklist, markerCandidates, markers, notes);
  renderChecklist(refs.markerClinicalChecklist, clinicalCandidates, clinical);
  refreshMarkerMatrixControls();
  renderMarkerSelectionLine();
}

function currentMarkerSelections() {
  const markers = markerMatrixAttached() ? [] : selectedCheckboxValues(refs.markerChecklist);
  const clinical = selectedCheckboxValues(refs.markerClinicalChecklist).filter((value) => !markers.includes(value));
  const categoricalCandidates = new Set(sharedModelCategoricalCandidates());
  return { markers, clinical, categorical: clinical.filter((value) => categoricalCandidates.has(value)) };
}

// The columns the cut-point search tries: the ticked markers and the clinical covariates of this tab.
// Discover sends them, and its result stays current (and exportable) only while they are unchanged.
function signatureCandidateColumns() {
  const { markers, clinical } = currentMarkerSelections();
  return [...markers, ...clinical];
}

function renderMarkerSelectionLine() {
  // Every change to the markers (ticks, select all, clear, attaching or removing a file) passes here,
  // so Run is enabled exactly when there is something to evaluate.
  if (typeof syncAnalysisRunButtonAvailability === "function") syncAnalysisRunButtonAvailability();
  if (!refs.markerSelectionLine) return;
  const { markers, clinical } = currentMarkerSelections();
  if (!state.dataset) {
    refs.markerSelectionLine.textContent = "";
    return;
  }
  const matrix = markerMatrixAttached() ? state.markerMatrix : null;
  const count = matrix ? Number(matrix.n_markers || 0) : markers.length;
  const lens = clinical.length
    ? `judged on added value over ${formatValue(clinical.length)} clinical covariate${clinical.length === 1 ? "" : "s"}`
    : "judged on marginal association (no clinical covariates)";
  refs.markerSelectionLine.textContent = count
    ? `${formatCount(count)} marker${count === 1 ? "" : "s"}${matrix ? ` from ${matrix.filename}` : ""}, ${lens}.`
    : "Choose at least one numeric marker, or attach a marker file such as gene expression.";
}

// The defaults of a blank Permutations, Subsamples or Seed field: the ones result currency uses
// (MARKER_NUMERIC_DEFAULTS of the workspace part), read when called so the load order cannot matter.
function markerNumericDefaults() {
  return typeof MARKER_NUMERIC_DEFAULTS !== "undefined"
    ? MARKER_NUMERIC_DEFAULTS
    : { n_permutations: 1000, n_resamples: 200, random_seed: 20260926 };
}

function markerRequestFields() {
  const { markers, clinical, categorical } = currentMarkerSelections();
  const matrix = markerMatrixAttached() ? state.markerMatrix : null;
  const defaults = markerNumericDefaults();
  return {
    marker_columns: markers,
    marker_matrix_id: matrix ? matrix.matrix_id : null,
    marker_matrix_id_column: matrix ? matrix.id_column : null,
    clinical_columns: clinical,
    categorical_clinical: categorical,
    n_permutations: numericControlValue(refs.markerPermutations, defaults.n_permutations),
    n_resamples: numericControlValue(refs.markerResamples, defaults.n_resamples),
    random_seed: numericControlValue(refs.markerRandomSeed, defaults.random_seed),
    clinical_basis: refs.markerClinicalBasis?.value || "linear",
    nonlinear_lens: refs.markerNonlinearLens?.value || "off",
  };
}

async function runMarkerEvaluation() {
  const base = currentBaseConfig();
  const fields = markerRequestFields();
  if (!fields.marker_columns.length && !fields.marker_matrix_id) throw new Error("Choose at least one marker.");
  const markerCount = fields.marker_matrix_id ? Number(state.markerMatrix?.n_markers || 0) : fields.marker_columns.length;
  const requestToken = beginRequestToken("markers");
  // A validation of the previous locked model must not land under this evaluation.
  invalidateRequestTokens(["markerValidation"]);
  const datasetId = base.dataset_id;
  const loading = beginShellLoading([refs.markersStabilityPlot]);
  const runBanner = setRuntimeBanner(
    `Evaluating ${formatValue(markerCount)} marker(s) with ${formatValue(fields.n_permutations)} permutations and ${formatValue(fields.n_resamples)} subsamples. ${markerCount > 5000 ? "A genome-wide panel takes 10 minutes or more." : "Large panels take a few minutes."}`,
    "info",
    { held: true },
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
    releaseRuntimeBanner(runBanner);
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
  // With Permutations at 0 there are no family-wise p-values, so no marker was tested; with Subsamples at 0
  // (or every subsample failed) the stability of the selection was not assessed, so no marker can be robust.
  const nPermutations = analysis.null?.n_permutations;
  const permutationsRun = nPermutations == null || Number(nPermutations) > 0;
  const resampling = analysis.resampling || {};
  const stabilityAssessed = resampling.stability_assessed ?? (resampling.n_valid == null || Number(resampling.n_valid) > 0);
  const lensText = addedValue ? "added value beyond the clinical covariates" : "an association";
  const repeated = markerDuplicateCaution(analysis.duplicates);
  if (analysis.inference?.status === "withheld") {
    const dataCautions = repeated ? [repeated] : [];
    if ((cohort.dropped_markers || []).length) dataCautions.push(markerDroppedCaution(cohort.dropped_markers));
    (Array.isArray(cohort.notes) ? cohort.notes : []).forEach((note) => {
      const text = String(note ?? "").trim();
      if (text) dataCautions.push(text);
    });
    return {
      status: "review", headline: "Added-value inference is withheld; marker claims and robust tiers are unavailable.",
      metrics: [{ label: "Patients", value: cohort.n }, { label: "Markers", value: evaluated },
                { label: "Inference", value: "Withheld" }],
      strengths: [], cautions: [...dataCautions, ...(analysis.inference.reasons || []), analysis.inference.interpretation],
      next_steps: ["Review the model and residual diagnostics. Exported estimates and the locked model remain exploratory."]
    };
  }
  let headline;
  if (!permutationsRun) {
    headline = `No marker was tested for ${lensText}: with Permutations at 0 there are no family-wise p-values.`;
  } else if (robust) {
    headline = `${formatCount(robust)} of ${formatCount(evaluated)} markers are robust${addedValue ? " beyond the clinical covariates" : ""}.`;
  } else if (suggestive) {
    headline = stabilityAssessed
      ? `No marker is robust; ${formatCount(suggestive)} show evidence that does not hold up across subsamples.`
      : `${formatCount(suggestive)} marker(s) show evidence, but their stability was not assessed (Subsamples at 0), so none can be robust.`;
  } else {
    headline = `No marker shows ${lensText} after family-wise error control.`;
  }
  const cautions = [];
  if (repeated) cautions.push(repeated);
  if (!permutationsRun) {
    cautions.push("No permutations were run, so there are no family-wise p-values (Westfall-Young) and every marker is left untested.");
  }
  if (!stabilityAssessed) {
    cautions.push(String(resampling.note || "No subsample was evaluated, so the stability of the selection was not assessed and no marker can be robust."));
  }
  if (!addedValue) cautions.push("No clinical covariates were given, so markers are judged on marginal association only. Add clinical covariates to test added value.");
  if (permutationsRun && addedValue && markerResidualNull(analysis)) {
    cautions.push("Residual permutation assumes exchangeable residuals after the declared clinical-basis adjustment. Nonlinear marker-covariate relations can inflate false positives; a robust tier is internal evidence and needs independent validation.");
  }
  if (permutationsRun && addedValue) {
    cautions.push("Added-value testing uses a fitted clinical Cox model. Check its functional form and proportional-hazards assumptions; multiple-testing correction does not resolve an unsuitable clinical baseline.");
  }
  const clinicalOnly = markerModelIsClinicalOnly(signature);
  if (clinicalOnly) cautions.push("No marker was selected, so the final model is the clinical-only model (the clinical covariates alone).");
  // Keep uncertainty about added discrimination ahead of the separate training-to-left-out gap.
  const leftOut = markerLeftOutComparison(signature, addedValue);
  if (leftOut && leftOut.verdict !== "adds") cautions.push(leftOut.text);
  // The gap comes from repeated selection and refitting, even when a full-cohort model was fitted.
  if (signature.apparent_c != null && signature.signature_optimism != null && Number(signature.signature_optimism) > 0.02) {
    cautions.push(`The selection procedure's mean subsample-to-left-out C-index gap is ${Number(signature.signature_optimism).toFixed(3)}; the gap adjustment is a heuristic that includes training-size effects.`);
  }
  if ((cohort.dropped_markers || []).length) {
    cautions.push(markerDroppedCaution(cohort.dropped_markers));
  }
  // The engine's notes on the cohort: clinical columns it left out, markers with infinite values, a clinical
  // model that could not be estimated.
  (Array.isArray(cohort.notes) ? cohort.notes : []).forEach((note) => {
    const text = String(note ?? "").trim();
    if (text) cautions.push(text);
  });
  if (Number(counts["marginal only"] || 0) > 0) {
    cautions.push(`${formatCount(counts["marginal only"])} marker(s) show marginal association, but insufficient evidence of added value at these thresholds.`);
  }
  const duplicateScreen = analysis.duplicates || {};
  return {
    status: robust && !repeated ? "robust" : "review",
    headline,
    metrics: [
      { label: "Patients", value: cohort.n },
      { label: "Events", value: cohort.events },
      { label: "Markers", value: formatCount(evaluated) },
      { label: "Robust", value: robust },
      { label: "Suggestive", value: suggestive },
      { label: clinicalOnly ? "Clinical-only C (gap-adjusted)" : "Model C (gap-adjusted)", value: formatCIndexValue(signature.optimism_corrected_c) },
      ...(leftOut ? [{ label: "Clinical-only C (left out)", value: formatCIndexValue(signature.clinical_c_left_out) }] : []),
    ],
    strengths: [
      ...(leftOut?.verdict === "adds" ? [leftOut.text] : []),
      ...(permutationsRun
        ? [`Family-wise p-values (Westfall-Young) from ${formatValue(nPermutations)} permutations${addedValue && markerResidualNull(analysis) ? ", permuting each marker's residuals after regression on the clinical covariates (Smith method), approximating a conditional null" : ""}.`]
        : []),
      ...(stabilityAssessed
        ? [`The whole screen was repeated on ${formatValue(resampling.n_valid)} subsamples of ${Math.round(100 * Number(resampling.fraction || 0.632))}% of the patients.`]
        : []),
      `Robust: family-wise p ≤ ${formatValue(settings.alpha)}, selected in ≥ ${Math.round(100 * Number(settings.robust_frequency || 0.5))}% of subsamples and the same direction in ≥ ${Math.round(100 * Number(settings.robust_direction || 0.9))}%.`,
      ...(duplicateScreen.checked && !repeated
        ? [`No repeated patients: no two patients have near-identical profiles over the ${formatCount(duplicateScreen.markers_used)} most variable markers.`]
        : []),
    ],
    cautions,
    next_steps: [
      ...(permutationsRun ? [] : ["Run again with Permutations above 0 to test the markers."]),
      ...(stabilityAssessed ? [] : ["Run again with Subsamples above 0 to see which markers hold up across subsamples."]),
      ...(permutationsRun
        ? [robust ? "Validate the locked model in an independent cohort below before claiming the markers." : "Treat suggestive markers as hypotheses for an independent cohort."]
        : []),
      ...(signature.optimism_corrected_c != null ? ["Report the subsample gap-adjusted C-index as a heuristic internal summary, and use locked external validation for final performance claims."] : []),
    ],
  };
}

// The markers left out before testing, counted by the engine's reason; a reason this page does not know yet
// counts as "another reason", never under a wrong one.
const MARKER_DROP_GROUPS = [
  { test: (reason) => reason.startsWith("near-constant"), label: "near-constant (most patients at one value, as for genes expressed in few patients)" },
  { test: (reason) => reason === "constant" || /^\d+(?:\.\d+)?% missing$/.test(reason), label: "constant or mostly missing" },
  { test: (reason) => reason === "infinite values", label: "with infinite values (for example the log of 0; use log(x + 1) instead)" },
];

function markerDroppedCaution(dropped) {
  const counts = MARKER_DROP_GROUPS.map(() => 0);
  let other = 0;
  dropped.forEach((item) => {
    const reason = String(item?.reason || "").trim();
    const group = MARKER_DROP_GROUPS.findIndex((candidate) => candidate.test(reason));
    if (group >= 0) counts[group] += 1;
    else other += 1;
  });
  const parts = MARKER_DROP_GROUPS
    .map((group, index) => (counts[index] ? `${formatCount(counts[index])} ${group.label}` : ""))
    .filter(Boolean);
  if (other) parts.push(`${formatCount(other)} for another reason`);
  const listed = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0];
  return `${formatCount(dropped.length)} marker(s) were left out before testing: ${listed}.`;
}

// Patients who look like the same tumour twice: identical values, or each other's clear best match.
function markerDuplicateCaution(duplicates) {
  const pairs = duplicates?.pairs || [];
  const identical = duplicates?.identical || [];
  const total = Number(duplicates?.n_pairs || pairs.length) + Number(duplicates?.n_identical || identical.length);
  if (!total) return "";
  const examples = [
    ...identical.map((group) => group.join(" = ")),
    ...pairs.map((pair) => `${pair.a} and ${pair.b} (r = ${Number(pair.r).toFixed(2)})`),
  ].slice(0, 3);
  return `Possible repeated patients (${formatCount(total)}): ${examples.join("; ")}${total > examples.length ? "; ..." : ""}. `
    + "Their marker profiles are identical or near-identical, as for one tumour entered twice. A patient in the data twice can sit on both sides "
    + "of a subsample split and flatter the internal C-index estimates; keep one sample per patient and run again.";
}

// The added-value lens permutes each marker's residuals after regression on the clinical covariates ("smith";
// "freedman_lane" is its former name). Older results name the scheme only in their settings.
function markerResidualNull(analysis) {
  const scheme = String(analysis?.null?.lens2_null ?? analysis?.settings?.lens2_null ?? "");
  return scheme === "smith" || scheme === "freedman_lane";
}

// True when the procedure selected no marker, so its final model holds the clinical covariates only.
function markerModelIsClinicalOnly(signature) {
  if (signature?.clinical_only != null) return Boolean(signature.clinical_only);
  return signature?.apparent_c != null && Array.isArray(signature?.markers) && !signature.markers.length;
}

// The 95% interval of the left-out gain (corrected resampled t over the subsamples), or null for a result without
// one: saved before the interval existed, or with fewer than two paired subsamples.
function markerGainInterval(signature) {
  const interval = signature?.signature_gain_left_out_ci;
  if (!Array.isArray(interval) || interval.length !== 2 || interval.some((value) => value == null)) return null;
  const [low, high] = interval.map(Number);
  return Number.isFinite(low) && Number.isFinite(high) ? [low, high] : null;
}

const MARKER_GAIN_VERDICTS = {
  little: "The procedure's internal gain is below the display threshold of 0.02 across this interval.",
  adds: "Internal resampling suggests that the selection procedure adds discrimination beyond the clinical covariates.",
  uncertain: "The gain is uncertain: its interval includes both no gain and a gain of 0.02 or more.",
};

// The model of the whole procedure against the clinical covariates alone, both in the patients left out of each
// subsample; with paired subsamples the mean paired gain and their number. The verdict follows the gain's interval:
// "little" when all of it lies below 0.02, "adds" when all of it lies above 0, "uncertain" otherwise. A result without
// the interval has no verdict on added discrimination. Every left-out estimate assesses selection and refitting,
// regardless of which markers were selected when fitting the final model in the full cohort.
function markerLeftOutComparison(signature, addedValue) {
  const model = signature?.signature_c_left_out;
  const clinical = signature?.clinical_c_left_out;
  if (!addedValue || model == null || clinical == null) return null;
  const pairedGain = signature?.signature_gain_left_out;
  const gain = pairedGain != null && Number.isFinite(Number(pairedGain)) ? Number(pairedGain) : Number(model) - Number(clinical);
  const replicates = Number(signature?.n_clinical_replicates);
  const where = Number.isInteger(replicates) && replicates > 0
    ? (replicates === 1 ? "the one subsample that could be scored" : `each of ${formatCount(replicates)} subsamples`)
    : "each subsample";
  let subject = "the whole selection procedure";
  if (markerModelIsClinicalOnly(signature)) {
    subject = "the whole selection procedure (it selected no marker in the full cohort, so the final model is the clinical-only model)";
  }
  const comparison = `In the patients left out of ${where}, ${subject} reached C ${formatCIndexValue(model)} against ${formatCIndexValue(clinical)} for the clinical covariates alone`;
  const signedGain = `${gain >= 0 ? "+" : ""}${gain.toFixed(3)}`;
  const interval = markerGainInterval(signature);
  const scope = "Markers were selected and models refitted in each subsample; validate the locked final model independently.";
  if (!interval) {
    return {
      gain,
      verdict: "uncertain",
      text: `${comparison} (${signedGain}). No interval was available to assess the gain's uncertainty. ${scope}`,
    };
  }
  const [low, high] = interval;
  let verdict = "uncertain";
  if (high < 0.02) verdict = "little";
  else if (low > 0) verdict = "adds";
  return {
    gain,
    verdict,
    text: `${comparison}, a gain of ${signedGain} (95% CI ${low.toFixed(3)} to ${high.toFixed(3)}). ${MARKER_GAIN_VERDICTS[verdict]} ${scope}`,
  };
}

function markerMetaBanner(payload) {
  const analysis = payload?.analysis || {};
  const cohort = analysis.cohort || {};
  const signature = analysis.signature || {};
  const lens = analysis.primary_lens === "added_value" ? "added value over clinical covariates" : "marginal association";
  const parts = [
    ...(payload?.marker_matrix ? [`markers from ${payload.marker_matrix.filename}`] : []),
    `N=${formatValue(cohort.n)}`,
    `events=${formatValue(cohort.events)}`,
    `markers=${formatCount(cohort.n_markers_evaluated)}`,
    `tested for ${lens}`,
    `inference=${analysis.inference?.status || "not assessed"}`,
    `clinical model=${analysis.clinical_basis || "linear"}`,
  ];
  if (signature.apparent_c != null) {
    const model = markerModelIsClinicalOnly(signature) ? "clinical-only model (no marker selected)" : "model";
    parts.push(`${model} C apparent=${formatCIndexValue(signature.apparent_c)}, subsample gap-adjusted=${formatCIndexValue(signature.optimism_corrected_c)}`);
  }
  if (analysis.primary_lens === "added_value" && signature.clinical_c_left_out != null) parts.push(`clinical-only C (left out)=${formatCIndexValue(signature.clinical_c_left_out)}`);
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
  const inference = payload?.analysis?.inference;
  if (refs.markersInferenceNote) refs.markersInferenceNote.textContent = inference
    ? `${inference.status}. ${inference.interpretation} ${(inference.reasons || []).join("; ")} ${inference.engineering_qualification?.claim_boundary || ""} ${inference.engineering_qualification?.extension_limitations || ""}`
    : "Diagnostics were not assessed in this saved result.";
  if (refs.markersDiagnosticsTable) renderTable(refs.markersDiagnosticsTable,
    [...(inference?.clinical_tests || []), ...(inference?.residual_tests || [])].map((test) => ({
      Marker: test.marker || "Clinical model", Diagnostic: test.name || test.diagnostic,
      Method: test.method || "", Status: test.status || (test.p_value == null ? "failed" : "calculated"),
      "Raw P": test.p_value, "Holm P": test.p_holm, "Withhold threshold": inference.threshold,
      Reason: test.reason || ""
    })));
  // Results saved before the summary figure existed simply leave its shell hidden.
  if (payload.summary_figure?.data?.length) await renderMarkerPlot(refs.markersSummaryPlot, payload.summary_figure, "marker_summary", payload);
  else clearPlotShell(refs.markersSummaryPlot, "", { state: "placeholder" });
  await renderMarkerPlot(refs.markersStabilityPlot, payload.stability_figure, "marker_stability", payload);
  await renderMarkerPlot(refs.markersRankPlot, payload.rank_figure, "marker_ranks", payload);
  const rows = markerDisplayRows(payload);
  renderTable(refs.markersTableShell, rows.slice(0, MARKER_TABLE_DISPLAY_LIMIT));
  if (refs.markersTableNote) {
    const addedValue = payload?.analysis?.primary_lens === "added_value";
    const order = rows.length > MARKER_TABLE_DISPLAY_LIMIT
      ? `Showing the first ${formatValue(MARKER_TABLE_DISPLAY_LIMIT)} of ${formatValue(rows.length)} markers, strongest first. Export the table for all of them.`
      : "Strongest markers first.";
    // The HR column comes from the model of the primary lens: with the clinical covariates, or the marker alone.
    const hazardRatio = addedValue
      ? "HR per unit of the marker, from a Cox model with the clinical covariates."
      : "HR per unit of the marker, from a Cox model with the marker alone (no clinical covariates were given).";
    const evidence = addedValue
      ? "Evidence: M marginal association and A added value over the clinical covariates, each + or − when family-wise significant (higher or lower hazard) and · when not; N+ when the non-linear check agrees."
      : "Evidence: M marginal association, + or − when family-wise significant (higher or lower hazard) and · when not; N+ when the non-linear check agrees.";
    refs.markersTableNote.textContent = `${order} ${hazardRatio} ${evidence}`;
  }
  if (refs.markerValidationSection) refs.markerValidationSection.classList.remove("hidden");
  updateResultVisibility();
}

// The marker table as shown and exported. A rank interval reads "1 to 3", because a spreadsheet opens
// "1-3" as a date.
function markerDisplayRows(payload) {
  return (payload?.display_table || []).map((row) => {
    const interval = row?.["Rank 95% interval"];
    const match = typeof interval === "string" ? /^\s*(\d+)\s*-\s*(\d+)\s*$/.exec(interval) : null;
    return match ? { ...row, "Rank 95% interval": `${match[1]} to ${match[2]}` } : row;
  });
}

function clearMarkerOutputs() {
  state.markers = null;
  state.markerValidation = null;
  if (refs.markersInsightBoard) refs.markersInsightBoard.innerHTML = '<div class="empty-state">Run the evaluation to see which markers hold up.</div>';
  if (refs.markersMetaBanner) refs.markersMetaBanner.textContent = "";
  if (refs.markersInferenceNote) refs.markersInferenceNote.textContent = "";
  if (refs.markersDiagnosticsTable) refs.markersDiagnosticsTable.innerHTML = "";
  clearPlotShell(refs.markersStabilityPlot, '<div class="empty-state plot-empty"><span>Choose markers and click <strong>Run Analysis</strong>.</span></div>', { state: "placeholder" });
  clearPlotShell(refs.markersSummaryPlot, "", { state: "placeholder" });
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
  const recipeHash = recipe.recipe_hash;
  const runBanner = setRuntimeBanner(`Uploading ${file.name} and applying the locked model unchanged.`, "info", { held: true });
  let externalDatasetId = null;
  try {
    const form = new FormData();
    form.append("file", file);
    // The upload is not cancelled with the validation: the server stores the cohort either way, and only its
    // answer carries the id the cleanup below needs.
    const external = await fetchJSON("/api/upload", { method: "POST", body: form });
    externalDatasetId = external?.dataset_id || null;
    if (!requestTokenMatches("markerValidation", requestToken)) return;
    const payload = await fetchJSON("/api/marker-validation", {
      signal: requestSignal("markerValidation"),
      method: "POST",
      body: JSON.stringify({ dataset_id: external.dataset_id, recipe, marker_scaling: refs.markerValidationScaling?.value || "as_measured" }),
    });
    if (!requestTokenMatches("markerValidation", requestToken) || state.dataset?.dataset_id !== sourceDatasetId) return;
    // Shown only under the evaluation whose locked model it validated.
    if (state.markers?.analysis?.locked_recipe?.recipe_hash !== recipeHash) return;
    state.markerValidation = { ...payload, external_filename: file.name };
    await renderMarkerValidation(state.markerValidation);
    syncDownloadButtonAvailability();
    showToast(`Locked model applied to ${file.name}.`, "success", 3200);
  } finally {
    releaseRuntimeBanner(runBanner);
    // The external cohort was stored only for this validation; free it so it cannot crowd out the
    // development dataset in the server's store.
    if (externalDatasetId) {
      fetch(apiUrl(`/api/dataset/${encodeURIComponent(externalDatasetId)}`), { method: "DELETE" }).catch(() => {});
    }
  }
}

// The unit of the horizon: the time unit set for the endpoint, or the development time column's name when
// that unit is still the generic "Time".
function markerHorizonUnit(timeColumn = "") {
  const unit = String(refs.timeUnitLabel?.value || "").trim();
  if (unit && unit !== DEFAULT_TIME_UNIT_LABEL) return unit.toLowerCase();
  return timeColumn ? `(${timeColumn})` : "";
}

function markerValidationMetrics(validation, { timeColumn = "" } = {}) {
  const metrics = validation?.metrics || {};
  const interval = (values) => (Array.isArray(values) && values[0] != null ? ` (${formatValue(values[0])} to ${formatValue(values[1])})` : "");
  const rows = [
    { label: "C-index", value: `${formatValue(metrics.c_index)}${interval(metrics.c_index_ci)}` },
    { label: "Calibration slope", value: `${formatValue(metrics.calibration_slope)}${interval(metrics.calibration_slope_ci)}` },
  ];
  if (metrics.clinical_only_c_index != null) {
    rows.push({ label: "Clinical-only C-index", value: `${formatValue(metrics.clinical_only_c_index)}${interval(metrics.clinical_only_c_index_ci)}` });
  }
  if (metrics.delta_c_index != null) rows.push({ label: "C gain over clinical", value: `${formatValue(metrics.delta_c_index)}${interval(metrics.delta_c_index_ci)}` });
  if (metrics.observed_expected_ratio != null) {
    const unit = markerHorizonUnit(timeColumn);
    rows.push({ label: `Observed/expected at ${formatValue(metrics.horizon)}${unit ? ` ${unit}` : ""}`, value: formatValue(metrics.observed_expected_ratio) });
  }
  if (metrics.brier_skill != null) rows.push({ label: "Brier skill", value: formatValue(metrics.brier_skill) });
  if (metrics.marker_weight_available != null && Number(metrics.marker_weight_available) < 1) {
    rows.push({ label: "Marker weight measured", value: `${Math.round(100 * Number(metrics.marker_weight_available))}%` });
  }
  if (metrics.marker_scaling === "within_cohort") rows.push({ label: "Marker scale", value: "rescaled within cohort" });
  return rows;
}

async function renderMarkerValidation(payload) {
  const validation = payload?.validation || {};
  const cohort = validation.cohort || {};
  const replicated = (validation.markers || []).filter((row) => row.replicated).length;
  const total = (validation.markers || []).filter((row) => !row.absent).length;
  const metrics = markerValidationMetrics(validation, {
    timeColumn: state.markers?.analysis?.locked_recipe?.outcome?.time_column || "",
  });
  if (refs.markerValidationSummary) {
    refs.markerValidationSummary.innerHTML = `
      <p class="marker-validation-headline">${escapeHtml(`${payload.external_filename || "External cohort"}: ${formatValue(cohort.n)} patients, ${formatValue(cohort.events)} events. ${validation.inference && !validation.inference.allowed ? `Inference ${validation.inference.status}; prediction estimates are exploratory.` : `${formatValue(replicated)} of ${formatValue(total)} markers replicated.`}`)}</p>
      <div class="insight-metrics">${metrics.map((metric) => `<div class="metric-pill"><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong></div>`).join("")}</div>
      ${(validation.notes || []).length ? `<ul class="insight-cautions">${validation.notes.map(escapeListItem).join("")}</ul>` : ""}
    `;
  }
  if (refs.markerValidationPlot && payload.figure?.data?.length) {
    refs.markerValidationPlot.classList.remove("hidden");
    resetPlotElement(refs.markerValidationPlot);
    await Plotly.newPlot(refs.markerValidationPlot, payload.figure.data, plotLayoutConfig(payload.figure.layout || {}, "marker_replication"), plotConfig("marker_replication"));
    stabilizePlotShellHeight(refs.markerValidationPlot);
  } else if (refs.markerValidationPlot) {
    // A validation without a replication figure must not show the plot of an earlier one.
    clearPlotShell(refs.markerValidationPlot, "", { state: "placeholder" });
    refs.markerValidationPlot.classList.add("hidden");
  }
  renderTable(refs.markerValidationShell, markerValidationRows(validation), null, { pValueColumns: ["Replication P (Holm)"] });
}

// One row per locked marker. `tested` names the fit its replication used ("added_value" with the clinical
// covariates, "marginal" without; null when that fit was not estimable); results without it used the
// adjusted fit when there was one.
function markerValidationRows(validation) {
  const markers = (validation?.markers || []).filter((row) => row && typeof row === "object");
  const showTested = markers.some((row) => "tested" in row);
  return markers.map((row) => {
    const lens = "tested" in row ? row.tested : (row.adjusted ? "added_value" : (row.marginal ? "marginal" : null));
    const fit = lens === "added_value" ? row.adjusted : (lens === "marginal" ? row.marginal : null);
    const estimable = !row.absent && row.replication_p_holm != null;
    const status = (text) => (row.absent ? "not measured" : (row.inference_status ? `inference ${row.inference_status}` : (estimable ? text : "not estimable")));
    return {
      Marker: row.marker,
      "Inference status": row.inference_status || validation.inference?.status || "not assessed",
      ...(showTested ? { "Tested as": row.absent ? "not measured" : ({ added_value: "added value", marginal: "marginal" }[lens] || "not estimable") } : {}),
      "HR per unit": fit?.hazard_ratio ?? null,
      "CI lower": fit?.ci_lower ?? null,
      "CI upper": fit?.ci_upper ?? null,
      "Same direction": status(row.same_direction ? "yes" : "no"),
      "Replication P (Holm)": row.inference_status ? null : (estimable ? row.replication_p_holm : status("")),
      Replicated: status(row.replicated ? "yes" : "no"),
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
  downloadCsv(buildDownloadFilename("marker_evaluation", "csv"), markerDisplayRows(payload), null, {
    caption: "Marker evaluation",
    notes: [markerMetaBanner(payload)],
  });
}

function downloadMarkerChecklist(format) {
  const payload = currentGoalResult("markers");
  if (!requireCurrentResultForExport("markers", { payload })) return;
  if (!payload.report) {
    showToast("Run the evaluation again to build the REMARK checklist.", "warning", 3600);
    return;
  }
  downloadChecklist(payload.report, format, "remark_checklist").catch((error) => showError(error?.message || "Checklist export failed."));
}

function syncMarkerDownloadButtons() {
  const current = currentGoalResult("markers");
  if (refs.downloadMarkersCsvButton) refs.downloadMarkersCsvButton.disabled = !current;
  if (refs.downloadMarkerRecipeButton) refs.downloadMarkerRecipeButton.disabled = !current?.analysis?.locked_recipe;
  if (refs.downloadMarkerRemarkDocxButton) refs.downloadMarkerRemarkDocxButton.disabled = !current?.report;
  if (refs.downloadMarkerRemarkMarkdownButton) refs.downloadMarkerRemarkMarkdownButton.disabled = !current?.report;
  const stabilityCurrent = plotShowsResult(refs.markersStabilityPlot, current);
  if (refs.downloadMarkersSummaryPngButton) refs.downloadMarkersSummaryPngButton.disabled = !plotShowsResult(refs.markersSummaryPlot, current);
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
    addVisibleCheckboxesToSelection(refs.markerChecklist);
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
  [refs.markerPermutations, refs.markerResamples, refs.markerRandomSeed, refs.markerNonlinearLens, refs.markerClinicalBasis].filter(Boolean).forEach((control) => {
    control.addEventListener("change", () => { scheduleResultCurrencySync(); queueHistorySync(); });
  });
  refs.attachMarkerMatrixButton?.addEventListener("click", async () => {
    await withLoading(refs.attachMarkerMatrixButton, attachMarkerMatrix);
    // withLoading re-enables its button; it stays off while a matrix is attached.
    renderMarkerMatrixState();
  });
  refs.removeMarkerMatrixButton?.addEventListener("click", removeMarkerMatrix);
  refs.downloadMarkersCsvButton?.addEventListener("click", downloadMarkerTable);
  refs.downloadMarkerRecipeButton?.addEventListener("click", downloadMarkerRecipe);
  refs.downloadMarkerRemarkDocxButton?.addEventListener("click", () => downloadMarkerChecklist("docx"));
  refs.downloadMarkerRemarkMarkdownButton?.addEventListener("click", () => downloadMarkerChecklist("markdown"));
  refs.downloadMarkersSummaryPngButton?.addEventListener("click", () => {
    const payload = currentGoalResult("markers");
    if (!requireCurrentPlotForExport(refs.markersSummaryPlot, payload)) return;
    void downloadPlotImage(refs.markersSummaryPlot, buildDownloadFilename("marker_summary", "png").replace(/\.png$/, ""), "png");
  });
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

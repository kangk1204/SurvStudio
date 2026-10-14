// Explicit clinical-versus-added comparison; never reuses a leaderboard fit or its selection.
const patternRefs = Object.fromEntries([
  "Clinical", "Added", "Categorical", "Model", "Target", "Group", "Seed", "Trees", "Depth", "Epochs", "Bootstrap",
  "Demo", "Run", "Json", "Csv", "RiskCsv", "RiskNote", "RiskPlot", "RiskTable", "Status", "Results", "Interpretation", "ComparisonPlot", "CurvePlot", "CurveNote",
].map((name) => [name, document.getElementById(`pattern${name}`)]));
const predictorPatternState = { payload: null, signature: null, controller: null, sequence: 0 };

function patternConfig() {
  const base = currentBaseConfig();
  return {
    dataset_id: base.dataset_id, time_column: base.time_column, event_column: base.event_column,
    event_positive_value: base.event_positive_value, time_unit_label: base.time_unit_label,
    clinical_features: selectedCheckboxValues(patternRefs.Clinical),
    added_features: selectedCheckboxValues(patternRefs.Added),
    categorical_features: selectedCheckboxValues(patternRefs.Categorical),
    target_feature: patternRefs.Target.value, group_feature: patternRefs.Group.value,
    model_type: patternRefs.Model.value, random_seed: Number(patternRefs.Seed.value),
    n_estimators: Number(patternRefs.Trees.value), max_depth: Number(patternRefs.Depth.value),
    epochs: Number(patternRefs.Epochs.value), n_bootstrap: Number(patternRefs.Bootstrap.value),
  };
}
function currentPatternSignature() {
  try { return JSON.stringify(patternConfig()); } catch { return null; }
}
function clearPredictorPatterns() {
  predictorPatternState.sequence += 1;
  predictorPatternState.controller?.abort();
  predictorPatternState.controller = null;
  predictorPatternState.payload = null;
  predictorPatternState.signature = null;
  patternRefs.Results.classList.add("hidden");
  patternRefs.Status.textContent = "Load data and choose two separate predictor sets.";
  patternRefs.Status.dataset.stale = "false";
  patternRefs.Run.disabled = false;
  patternRefs.Json.disabled = patternRefs.Csv.disabled = patternRefs.RiskCsv.disabled = true;
  patternRefs.RiskTable.replaceChildren();
  [patternRefs.ComparisonPlot, patternRefs.CurvePlot, patternRefs.RiskPlot].forEach((el) => { Plotly.purge(el); el.replaceChildren(); });
}
function syncPatternCurrency() {
  const current = currentPatternSignature();
  if (refs.outcomeProxyReviewPanel?.dataset.inputScope === "patterns"
      && refs.outcomeProxyReviewPanel.dataset.inputContext !== current) resetOutcomeProxyReview();
  if (predictorPatternState.controller && current !== predictorPatternState.signature) clearPredictorPatterns();
  const ready = Boolean(predictorPatternState.payload) && current === predictorPatternState.signature;
  patternRefs.Json.disabled = !ready;
  patternRefs.Csv.disabled = !ready || !predictorPatternState.payload?.analysis.pattern;
  patternRefs.RiskCsv.disabled = !ready || !predictorPatternState.payload?.analysis.risk_display;
  if (predictorPatternState.payload && !ready) {
    clearPredictorPatterns();
    patternRefs.Status.textContent = "Inputs changed. Run again to show results for these settings.";
    patternRefs.Status.dataset.stale = "true";
  }
}
function refreshPatternMenus() {
  const clinical = selectedCheckboxValues(patternRefs.Clinical);
  const added = selectedCheckboxValues(patternRefs.Added);
  const selectedCategories = selectedCheckboxValues(patternRefs.Categorical);
  const selected = [...new Set([...clinical, ...added])];
  const categorical = selectedCategories.filter((c) => selected.includes(c));
  renderChecklist(patternRefs.Categorical, selected, categorical);
  renderSelect(patternRefs.Target, added, { selected: patternRefs.Target.value, includeBlank: true, blankLabel: "Choose added predictor" });
  renderSelect(patternRefs.Group, clinical.filter((c) => categorical.includes(c)), { selected: patternRefs.Group.value, includeBlank: true, blankLabel: "Choose categorical group" });
}
function refreshPatternControls() {
  const columns = (state.dataset?.columns || []).map((c) => c.name)
    .filter((c) => c !== refs.timeColumn.value && c !== refs.eventColumn.value && !/(^|_)(id|barcode)$/i.test(c));
  // Named demo presets are transparent defaults, not selected from an outcome or model score.
  const molecular = columns.includes("gene_A") && columns.includes("gene_B") && columns.includes("stage");
  const tcga = state.dataset?.filename?.includes("tcga_luad");
  const clinical = molecular ? ["age", "stage"] : tcga ? ["age", "sex", "stage_group"] : [];
  const added = molecular ? ["gene_A", "gene_B"] : tcga ? ["kras_status", "egfr_status"] : [];
  const categorical = molecular ? ["stage"] : tcga ? ["sex", "stage_group", "kras_status", "egfr_status"] : [];
  renderChecklist(patternRefs.Clinical, columns, clinical);
  renderChecklist(patternRefs.Added, columns, added);
  renderChecklist(patternRefs.Categorical, columns, categorical);
  refreshPatternMenus();
  patternRefs.Target.value = molecular ? "gene_A" : tcga ? "egfr_status" : "";
  patternRefs.Group.value = molecular ? "stage" : tcga ? "stage_group" : "";
}
async function runPredictorPatterns() {
  const config = patternConfig();
  if (!config.clinical_features.length || !config.added_features.length) throw new Error("Choose clinical and added predictors.");
  if (config.clinical_features.some((c) => config.added_features.includes(c))) throw new Error("Each predictor must belong to just one set.");
  if (!config.target_feature || !config.group_feature) throw new Error("Choose an added predictor and a categorical clinical group.");
  for (const el of [patternRefs.Seed, patternRefs.Trees, patternRefs.Depth, patternRefs.Epochs, patternRefs.Bootstrap]) {
    if (!el.value || !el.checkValidity()) throw new Error("Check the numeric run settings.");
  }
  clearPredictorPatterns();
  const sequence = predictorPatternState.sequence;
  const signature = JSON.stringify(config);
  const controller = new AbortController();
  predictorPatternState.controller = controller;
  predictorPatternState.signature = signature;
  patternRefs.Run.disabled = true;
  patternRefs.Status.textContent = "Fitting two models on a shared training split, then comparing the same test patients…";
  try {
    const payload = await fetchJSON("/api/predictor-patterns", { method: "POST", body: JSON.stringify({ ...config, ...outcomeProxyReviewFields("patterns") }), signal: controller.signal });
    if (sequence !== predictorPatternState.sequence || signature !== currentPatternSignature()) return;
    predictorPatternState.payload = payload;
    const a = payload.analysis;
    patternRefs.Results.classList.remove("hidden");
    patternRefs.Status.textContent = `${a.training_patients} training patients; ${a.test_patients} shared test patients (${a.test_events} events). ${a.excluded_missing_outcome_rows} rows excluded for missing outcomes. ${a.comparison.n_bootstrap_valid}/${a.comparison.n_bootstrap_requested} paired bootstrap draws defined.`;
    const audit = a.input_audit?.outcome_proxy;
    if (audit) {
      const label = audit.status === "reviewed_by_user" ? "Predictor timing and source reviewed; findings retained."
        : audit.status === "no_pattern_detected" ? "No near-exact outcome pattern detected in training data."
        : "Outcome-pattern screening has limited coverage.";
      patternRefs.Status.textContent += ` ${label} This does not certify that every predictor is safe.`;
    }
    const interval = a.comparison.delta_ci;
    const range = interval[0] === null ? "not estimable" : `${interval[0].toFixed(3)} to ${interval[1].toFixed(3)}`;
    patternRefs.Interpretation.textContent = `Change in C-index: ${a.comparison.delta_c.toFixed(3)} (95% paired interval: ${range}). ${a.interpretation} The interval describes test-patient uncertainty for these fitted models.`;
    await Plotly.newPlot(patternRefs.ComparisonPlot, payload.comparison_figure.data, plotLayoutConfig(payload.comparison_figure.layout, "predictor_comparison"), plotConfig("predictor_comparison"));
    const display = a.risk_display;
    if (payload.risk_group_figure) {
      const layout = { ...payload.risk_group_figure.layout, title: { ...payload.risk_group_figure.layout.title, text: "Observed survival in predicted risk groups" } };
      await Plotly.newPlot(patternRefs.RiskPlot, payload.risk_group_figure.data, plotLayoutConfig(layout, "predicted_risk_groups"), plotConfig("predicted_risk_groups"));
      patternRefs.RiskNote.textContent = "Groups use the training risk median. Curves show observed survival in held-out patients; C-index assesses prediction. The cutoff is not a clinical decision threshold.";
    } else {
      patternRefs.RiskPlot.replaceChildren();
      patternRefs.RiskNote.textContent = display?.curve_error || "Risk-group curves are unavailable in this saved result.";
    }
    patternRefs.RiskTable.replaceChildren();
    if (display) {
      const table = document.createElement("table");
      const head = document.createElement("thead"); const header = document.createElement("tr");
      ["Source row (0-based)", "Relative risk score", "Rank in test patients", "Predicted group"].forEach((label) => { const cell = document.createElement("th"); cell.textContent = label; header.appendChild(cell); });
      head.appendChild(header); table.appendChild(head); const body = document.createElement("tbody");
      display.rows.slice(0, 10).forEach((row) => {
        const tr = document.createElement("tr");
        [row.source_row, row.risk_score.toFixed(3), row.test_risk_rank, row.predicted_risk_group].forEach((value) => { const td = document.createElement("td"); td.textContent = value; tr.appendChild(td); });
        body.appendChild(tr);
      });
      table.appendChild(body); patternRefs.RiskTable.appendChild(table);
      const note = document.createElement("p"); note.textContent = `Showing the ten highest scores. Save patient risk scores for all ${display.rows.length} held-out patients; source rows trace to the loaded table.`; patternRefs.RiskTable.appendChild(note);
    }
    if (payload.pattern_figure) {
      await Plotly.newPlot(patternRefs.CurvePlot, payload.pattern_figure.data, plotLayoutConfig(payload.pattern_figure.layout, "predictor_pattern"), plotConfig("predictor_pattern"));
      patternRefs.CurveNote.textContent = "The fitted model is fixed. Curves use the same training background; gaps mark values without enough observed training support in that group. They do not confirm a biological interaction.";
    } else {
      patternRefs.CurvePlot.replaceChildren();
      patternRefs.CurveNote.textContent = `Curve unavailable: ${a.pattern_error}. The predictor-set comparison above was completed.`;
    }
    if (sequence !== predictorPatternState.sequence || signature !== currentPatternSignature()) return;
    syncPatternCurrency();
  } catch (error) {
    if (sequence === predictorPatternState.sequence && !isSupersededRequestError(error)) {
      patternRefs.Status.textContent = `Comparison not completed: ${error.message}`;
      throw error;
    }
  } finally {
    if (sequence === predictorPatternState.sequence) {
      predictorPatternState.controller = null;
      patternRefs.Run.disabled = false;
    }
  }
}
function initPredictorPatterns() {
  [patternRefs.Clinical, patternRefs.Added, patternRefs.Categorical].forEach((el) => el.addEventListener("change", () => { refreshPatternMenus(); syncPatternCurrency(); }));
  [patternRefs.Model, patternRefs.Target, patternRefs.Group, patternRefs.Seed, patternRefs.Trees, patternRefs.Depth, patternRefs.Epochs, patternRefs.Bootstrap, refs.timeColumn, refs.eventColumn, refs.eventPositiveValue, refs.timeUnitLabel].forEach((el) => {
    el.addEventListener("change", syncPatternCurrency);
    if (el.tagName === "INPUT") el.addEventListener("input", syncPatternCurrency);
  });
  patternRefs.Run.addEventListener("click", () => runPredictorPatterns().catch(showError));
  patternRefs.Demo.addEventListener("click", async () => {
    try {
      const payload = await fetchLatestDatasetPayload((signal) => fetchJSON("/api/load-molecular-demo", { method: "POST", signal }));
      if (!payload) return;
      applyLoadedDataset(payload);
      activateTab("benchmark");
      document.getElementById("predictorPatternCard").open = true;
      refreshPatternControls();
    } catch (error) { if (!isSupersededRequestError(error)) showError(error); }
  });
  patternRefs.Json.addEventListener("click", () => {
    syncPatternCurrency();
    if (!patternRefs.Json.disabled) downloadHelpers.downloadText({ filename: "predictor_comparison.json", text: JSON.stringify(predictorPatternState.payload, null, 2), mimeType: "application/json;charset=utf-8;" });
  });
  patternRefs.RiskCsv.addEventListener("click", () => {
    syncPatternCurrency(); if (patternRefs.RiskCsv.disabled) return;
    const p = predictorPatternState.payload;
    downloadHelpers.downloadCsv({ filename: "heldout_patient_risk_scores.csv", rows: p.analysis.risk_display.rows,
      showToast, caption: "Fixed-model held-out risk scores", notes: [p.analysis.risk_display.risk_score_interpretation, p.analysis.risk_display.cutoff_source, p.analysis.risk_display.group_interpretation, ...(p.analysis.input_audit?.notes || []), `Outcome-pattern status: ${p.analysis.input_audit?.outcome_proxy?.status || "not_assessed"}`, JSON.stringify(p.request_config)] });
  });
  patternRefs.Csv.addEventListener("click", () => {
    syncPatternCurrency();
    if (patternRefs.Csv.disabled) return;
    const p = predictorPatternState.payload;
    const rows = p.analysis.pattern.rows.map((row) => ({ ...row, model: p.analysis.model_type, target: p.analysis.pattern.target_feature, group_variable: p.analysis.pattern.group_feature, split_fingerprint: p.analysis.evaluation_split_fingerprint }));
    downloadHelpers.downloadCsv({ filename: "predictor_patterns.csv", rows, showToast, caption: "Exploratory fixed-model predictions", notes: [p.analysis.pattern.interpretation, p.analysis.pattern.scope, ...(p.analysis.input_audit?.notes || []), `Outcome-pattern status: ${p.analysis.input_audit?.outcome_proxy?.status || "not_assessed"}`, JSON.stringify(p.request_config)] });
  });
}

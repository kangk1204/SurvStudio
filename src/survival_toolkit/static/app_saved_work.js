// Server-side workspaces contain exact completed results, never browser-local patient data.
const savedWorkState = {
  enabled: false, modeKnown: false, pending: false, error: false, timer: null, generation: 0,
  restoring: false, deleting: false, chain: Promise.resolve(), initialHistory: window.history?.state,
};
const savedWorkRefs = Object.fromEntries(["Button", "Panel", "List"].map((key) => [key, document.getElementById(`savedWork${key}`)]));

function setAutosaveStatus(text, tone = "") {
  const target = document.getElementById("autosaveStatus");
  if (!target) return;
  target.textContent = text;
  target.dataset.tone = tone;
}

function captureSavedWorkspace() {
  const results = Object.fromEntries(["km", "cox", "cohort", "signature", "ml", "dl", "markers", "markerValidation"].map((key) => [key, state[key]]));
  return {
    schema_version: 1,
    workspace: {
      dataset_id: state.dataset.dataset_id, dataset_hash: state.dataset.dataset_hash,
      controls: captureControlSnapshot(), results,
      compareCache: runtime.compareCache, resultPreference: runtime.resultPreference,
      benchmarkIntervals: runtime.benchmarkIntervals?.status === "ready" ? runtime.benchmarkIntervals : null,
      predictiveFamily: runtime.predictiveFamily, workbenchRevealed: runtime.workbenchRevealed,
      predictiveWorkbenchIntent: runtime.predictiveWorkbenchIntent, tab: activeTabName(),
      markerMatrix: state.markerMatrix || null, derivedColumnProvenance: runtime.derivedColumnProvenance,
      lastDerivedGroup: runtime.lastDerivedGroup, savedValidationFile: runtime.savedValidationFile || null,
      pattern: { payload: predictorPatternState.payload, signature: predictorPatternState.signature,
        config: (() => { try { return patternConfig(); } catch { return null; } })() },
    },
  };
}

function queueSavedWorkspace() {
  if (!savedWorkState.enabled || savedWorkState.restoring || !state.dataset) return;
  savedWorkState.pending = true;
  savedWorkState.generation += 1;
  setAutosaveStatus("Saving…");
  if (savedWorkState.timer) clearTimeout(savedWorkState.timer);
  savedWorkState.timer = setTimeout(() => { savedWorkState.timer = null; void saveCurrentWorkspace(); }, 350);
}

async function saveCurrentWorkspace() {
  if (!savedWorkState.enabled || savedWorkState.restoring || !state.dataset) return false;
  if (savedWorkState.timer) clearTimeout(savedWorkState.timer);
  savedWorkState.timer = null;
  const datasetId = state.dataset.dataset_id;
  const generation = savedWorkState.generation;
  // Freeze before waiting for an earlier write; a dataset switch must not change this request.
  let body;
  try { body = JSON.stringify(captureSavedWorkspace()); }
  catch (error) {
    savedWorkState.pending = false; savedWorkState.error = true;
    setAutosaveStatus("Screen not saved — open Saved work to retry", "error");
    setRuntimeBanner(`This screen could not be saved: ${errorMessageText(error)} Download important results before leaving.`, "warning");
    return false;
  }
  savedWorkState.pending = true;
  setAutosaveStatus("Saving…");
  const operation = savedWorkState.chain.catch(() => {}).then(() => fetchJSON(`/api/saved-workspaces/${encodeURIComponent(datasetId)}/snapshot`, { method: "POST", body }));
  savedWorkState.chain = operation;
  try {
    await operation;
    if (state.dataset?.dataset_id === datasetId && generation === savedWorkState.generation) {
      savedWorkState.pending = false;
      savedWorkState.error = false;
      setAutosaveStatus("Saved automatically", "saved");
    }
    return true;
  } catch (error) {
    if (state.dataset?.dataset_id === datasetId && generation === savedWorkState.generation) {
      savedWorkState.error = true;
      savedWorkState.pending = false;
      setAutosaveStatus("Screen not saved — open Saved work to retry", "error");
      setRuntimeBanner(`This screen could not be saved: ${errorMessageText(error)} Completed analysis responses already saved by the server remain available in Saved work.`, "warning");
    }
    return false;
  }
}

async function refreshSavedWork() {
  const payload = await fetchJSON("/api/saved-workspaces");
  savedWorkState.modeKnown = typeof payload.autosave === "boolean";
  savedWorkState.enabled = payload.autosave === true;
  savedWorkRefs.List.replaceChildren();
  if (!savedWorkState.enabled) {
    setAutosaveStatus("Automatic saving is off", "warning");
    savedWorkRefs.List.textContent = "Automatic saving is off on this server. Download results before closing the page.";
    return;
  }
  if (!savedWorkState.pending && !savedWorkState.error) setAutosaveStatus("Automatic saving is on", "saved");
  if (!payload.datasets.length) savedWorkRefs.List.textContent = "No saved work yet. Upload a dataset or load a sample to start.";
  for (const row of payload.datasets) {
    const item = document.createElement("div"); item.className = "saved-work-item";
    const copy = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = row.filename;
    const summary = document.createElement("span"); summary.textContent = `${row.rows} rows · ${row.result_count} completed results${row.snapshot_saved_at ? ` · screen saved ${new Date(row.snapshot_saved_at).toLocaleString()}` : " · no saved screen yet"}`;
    copy.append(title, summary);
    const open = document.createElement("button"); open.type = "button"; open.className = "button ghost compact-btn"; open.textContent = "Reopen saved screen";
    open.addEventListener("click", () => reopenSavedWorkspace(row.dataset_id).catch(showError));
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "button ghost compact-btn"; remove.textContent = "Delete…";
    remove.addEventListener("click", async () => {
      if (savedWorkState.restoring || savedWorkState.deleting || Object.values(runtime.busyScopes || {}).some(Boolean)) { showError("Wait for the current analysis or workspace operation to finish before deleting saved work."); return; }
      if (!window.confirm(`Permanently delete ${row.filename}, its saved results and any validation results that use it? Download anything you need first.`)) return;
      savedWorkState.deleting = true;
      try {
        await fetchJSON(`/api/dataset/${encodeURIComponent(row.dataset_id)}`, { method: "DELETE" });
        if (state.dataset?.dataset_id === row.dataset_id) {
          savedWorkState.restoring = true;
          try { goHome(); } finally { savedWorkState.restoring = false; }
          savedWorkState.pending = false; savedWorkState.error = false;
        } else if (state.markerValidation?.request_config?.dataset_id === row.dataset_id) {
          state.markerValidation = null; runtime.savedValidationFile = null; clearMarkerValidationOutput(); queueSavedWorkspace();
        }
        await refreshSavedWork();
      } catch (error) { showError(error); }
      finally { savedWorkState.deleting = false; }
    });
    item.append(copy, open, remove); savedWorkRefs.List.append(item);
    const history = document.createElement("details");
    const heading = document.createElement("summary"); heading.textContent = "All completed results (newest first)";
    history.append(heading);
    history.addEventListener("toggle", async () => {
      if (!history.open || history.dataset.loaded) return;
      try {
        const saved = await fetchJSON(`/api/saved-workspaces/${encodeURIComponent(row.dataset_id)}`);
        for (const record of saved.results || []) {
          const button = document.createElement("button"); button.type = "button"; button.className = "button ghost compact-btn";
          button.textContent = `${record.route.replace("/api/", "")} · ${new Date(record.created_at).toLocaleString()}`;
          const canReopen = ["/api/kaplan-meier", "/api/cox", "/api/cohort-table", "/api/ml-model", "/api/deep-model", "/api/marker-evaluation", "/api/marker-validation", "/api/discover-signature", "/api/predictor-patterns"].includes(record.route);
          if (canReopen) button.addEventListener("click", () => reopenSavedWorkspace(row.dataset_id, { resultId: record.result_id }).catch(showError));
          else {
            button.textContent += " · Download JSON";
            button.addEventListener("click", async () => {
              try { const savedResult = await fetchJSON(`/api/saved-results/${encodeURIComponent(record.result_id)}`); downloadText(`${record.route.replace("/api/", "")}_${record.result_id}.json`, JSON.stringify(savedResult, null, 2), "application/json;charset=utf-8;"); }
              catch (error) { showError(error); }
            });
          }
          history.append(button);
        }
        if (!saved.results.length) history.append(document.createTextNode("No completed results yet."));
        history.dataset.loaded = "true";
      } catch (error) { showError(error); }
    });
    savedWorkRefs.List.append(history);
  }
}

function restorePatternControls(config) {
  if (!config) return;
  setCheckedValues(patternRefs.Clinical, config.clinical_features || []);
  setCheckedValues(patternRefs.Added, config.added_features || []);
  refreshPatternMenus();
  setCheckedValues(patternRefs.Categorical, config.categorical_features || []);
  refreshPatternMenus();
  for (const [key, value] of Object.entries({ Model: config.model_type, Target: config.target_feature, Group: config.group_feature,
    Seed: config.random_seed, Trees: config.n_estimators, Depth: config.max_depth, Epochs: config.epochs, Bootstrap: config.n_bootstrap })) setInputValue(patternRefs[key], value);
}

function snapshotControlsFromRequest(config, route = "") {
  const controls = { timeColumn: config.time_column, eventColumn: config.event_column,
    eventPositiveValue: config.event_positive_value, groupColumn: config.group_column,
    timeUnitLabel: config.time_unit_label, maxTime: config.max_time ?? "", covariates: config.covariates,
    categoricals: config.categorical_covariates, coxStrata: config.strata_columns,
    modelFeatures: config.features, modelCategoricals: config.categorical_features,
    dlModelCategoricals: config.categorical_features, cohortVariables: config.variables,
    markers: config.marker_columns, markerClinical: config.clinical_columns,
    markerPermutations: config.n_permutations, markerResamples: config.n_resamples,
    markerRandomSeed: config.random_seed, markerClinicalBasis: config.clinical_basis,
    markerDiagnosticPolicy: config.diagnostic_policy, markerNonlinearLens: config.nonlinear_lens,
    markerMaxSignatureMarkers: config.max_signature_markers,
    confidenceLevel: config.confidence_level, riskTablePoints: config.risk_table_points,
    logrankWeight: config.logrank_weight, fhPower: config.fh_p, showConfidenceBands: config.show_confidence_bands,
    signatureMaxDepth: config.max_combination_size, signatureMinFraction: config.min_group_fraction,
    signatureTopK: config.top_k, signatureBootstrapIterations: config.bootstrap_iterations,
    signaturePermutationIterations: config.permutation_iterations, signatureValidationIterations: config.validation_iterations,
    signatureValidationFraction: config.validation_fraction, signatureSignificanceLevel: config.significance_level,
    signatureOperator: config.combination_operator, signatureRandomSeed: config.random_seed,
  };
  if (route === "/api/ml-model") Object.assign(controls, {
    mlModelType: config.model_type === "compare" ? undefined : config.model_type,
    mlNEstimators: config.n_estimators, mlLearningRate: config.learning_rate,
    mlRandomSeed: config.random_state, dlRandomSeed: config.random_state,
    mlEvaluationStrategy: config.evaluation_strategy || "holdout", dlEvaluationStrategy: config.evaluation_strategy || "holdout",
    mlCvFolds: config.cv_folds, mlCvRepeats: config.cv_repeats,
    dlCvFolds: config.cv_folds, dlCvRepeats: config.cv_repeats,
    mlSkipShap: !config.compute_shap, mlShapSafeMode: config.shap_safe_mode,
  });
  if (route === "/api/deep-model") Object.assign(controls, {
    dlModelType: config.model_type === "compare" ? undefined : config.model_type,
    dlEpochs: config.epochs, dlLearningRate: config.learning_rate,
    dlHiddenLayers: Array.isArray(config.hidden_layers) ? config.hidden_layers.join(",") : undefined,
    dlDropout: config.dropout, dlBatchSize: config.batch_size,
    mlRandomSeed: config.random_seed, dlRandomSeed: config.random_seed,
    dlEvaluationStrategy: config.evaluation_strategy || "holdout", mlEvaluationStrategy: config.evaluation_strategy || "holdout",
    dlCvFolds: config.cv_folds, dlCvRepeats: config.cv_repeats,
    mlCvFolds: config.cv_folds, mlCvRepeats: config.cv_repeats,
    dlEarlyStoppingPatience: config.early_stopping_patience, dlEarlyStoppingMinDelta: config.early_stopping_min_delta,
    dlParallelJobs: config.parallel_jobs, dlNumTimeBins: config.num_time_bins, dlDModel: config.d_model,
    dlHeads: config.n_heads, dlLayers: config.n_layers, dlLatentDim: config.latent_dim, dlClusters: config.n_clusters,
  });
  if (config.locked_test_fraction != null) {
    controls.lockedTestEnabled = true; controls.lockedTestPercent = 100 * Number(config.locked_test_fraction);
  } else controls.lockedTestEnabled = false;
  return controls;
}

async function renderSavedResults(results, isCurrent = () => true) {
  for (const [key, value] of Object.entries(results)) state[key] = value;
  if (results.km) await renderKaplanMeierResult(results.km);
  if (!isCurrent()) return false;
  if (results.cox) await renderCoxResult(results.cox);
  if (!isCurrent()) return false;
  if (results.cohort) await renderCohortResult(results.cohort);
  if (!isCurrent()) return false;
  if (results.signature) renderSignatureResult(results.signature.analysis || results.signature);
  if (results.ml) await (results.ml.request_config?.model_type === "compare" ? renderMlComparisonResult(results.ml) : renderMlModelResult(results.ml));
  if (!isCurrent()) return false;
  if (results.dl) await (results.dl.request_config?.model_type === "compare" ? renderDlComparisonResult(results.dl) : renderDlModelResult(results.dl));
  if (!isCurrent()) return false;
  if (results.markers) await renderMarkerResults(results.markers);
  if (!isCurrent()) return false;
  if (results.markerValidation) await renderMarkerValidation(results.markerValidation);
  if (!isCurrent()) return false;
}

async function reopenSavedWorkspace(datasetId, { resultId = null } = {}) {
  if (savedWorkState.restoring || savedWorkState.deleting) throw new Error("Wait for the saved workspace operation to finish.");
  if (Object.values(runtime.busyScopes || {}).some(Boolean)) throw new Error("Wait for the current analysis to finish before reopening saved work.");
  if ((savedWorkState.pending || savedWorkState.error) && state.dataset && !await saveCurrentWorkspace()) throw new Error("This screen could not be saved. Keep it open and download important results before reopening another workspace.");
  savedWorkState.restoring = true;
  runtime.restoringSavedWork = true;
  savedWorkState.generation += 1;
  const token = ++runtime.historyRestoreToken;
  const datasetToken = beginRequestToken("dataset");
  const isCurrent = () => token === runtime.historyRestoreToken && requestTokenMatches("dataset", datasetToken);
  try {
    setAutosaveStatus("Opening saved work…");
    const payload = await fetchJSON(`/api/saved-workspaces/${encodeURIComponent(datasetId)}`);
    if (!isCurrent()) return;
    const snapshot = payload.snapshot;
    if (snapshot && snapshot.schema_version !== 1) throw new Error("This saved screen uses an unsupported format. Its source data and completed results were retained.");
    const saved = snapshot?.workspace || {};
    if (saved.dataset_hash && saved.dataset_hash !== payload.dataset.dataset_hash) throw new Error("This saved screen belongs to an earlier dataset revision. Saved results were retained.");
    let missingMatrix = false;
    const attachedMatrix = payload.dataset.attached_marker_matrix || saved.markerMatrix;
    if (attachedMatrix?.matrix_id) {
      try { await fetchJSON(`/api/marker-matrix/${encodeURIComponent(attachedMatrix.matrix_id)}`); }
      catch (error) {
        if (Number(error?.status) !== 404) throw error;
        missingMatrix = true; saved.markerMatrix = null; delete payload.dataset.attached_marker_matrix;
      }
      if (!isCurrent()) return;
    }
    updateAfterDataset(payload.dataset);
    runtime.savedValidationFile = saved.savedValidationFile || null;
    state.markerMatrix = payload.dataset.attached_marker_matrix || saved.markerMatrix || null;
    renderMarkerMatrixState();
    const results = resultId ? {} : (saved.results || {});
    let newest = null;
    // A browser crash may occur after the server saves a response but before the screen snapshot.
    // Recover missing results directly; never refit them or borrow another result's settings.
    const routeKeys = { "/api/kaplan-meier": "km", "/api/cox": "cox", "/api/cohort-table": "cohort", "/api/ml-model": "ml", "/api/deep-model": "dl", "/api/marker-evaluation": "markers", "/api/marker-validation": "markerValidation", "/api/discover-signature": "signature", "/api/predictor-patterns": "pattern" };
    for (const metadata of payload.results || []) {
      const key = routeKeys[metadata.route];
      if (!key || results[key] || (resultId && metadata.result_id !== resultId)) continue;
      const record = await fetchJSON(`/api/saved-results/${encodeURIComponent(metadata.result_id)}`);
      if (!isCurrent()) return;
      const result = { ...record.payload, saved_result: { ...record, payload: undefined } };
      if (key === "signature") results.signature = { ...result.signature_analysis, request_config: result.signature_request_config, result_dataset_id: result.dataset_id, dataset_hash: result.dataset_hash, saved_result: result.saved_result };
      else if (key === "pattern") saved.pattern = { payload: result, config: result.request_config, signature: JSON.stringify(result.request_config) };
      else results[key] = result;
      newest ||= record;
    }
    if (!isCurrent()) return;
    applyControlSnapshot((resultId ? null : saved.controls) || snapshotControlsFromRequest(newest?.payload?.request_config || newest?.payload?.signature_request_config || {}, newest?.route));
    restorePatternControls(saved.pattern?.config);
    runtime.compareCache = resultId ? runtime.compareCache : (saved.compareCache || runtime.compareCache);
    runtime.benchmarkIntervals = resultId ? null : (saved.benchmarkIntervals || null);
    await renderSavedResults(results, isCurrent);
    if (!isCurrent()) return;
    runtime.compareCache = resultId ? runtime.compareCache : (saved.compareCache || runtime.compareCache);
    runtime.resultPreference = resultId ? runtime.resultPreference : (saved.resultPreference || runtime.resultPreference);
    runtime.predictiveFamily = resultId && newest?.route === "/api/deep-model" ? "dl" : (saved.predictiveFamily || "ml");
    const singleModelHistory = resultId && ["/api/ml-model", "/api/deep-model"].includes(newest?.route) && newest?.payload?.request_config?.model_type !== "compare";
    runtime.workbenchRevealed = singleModelHistory || Boolean(saved.workbenchRevealed);
    runtime.predictiveWorkbenchIntent = singleModelHistory ? "train" : (saved.predictiveWorkbenchIntent || null);
    if (saved.pattern?.payload && (!resultId || newest?.route === "/api/predictor-patterns")) {
      predictorPatternState.signature = saved.pattern.signature;
      if (newest?.route === "/api/predictor-patterns") { const card = document.getElementById("predictorPatternCard"); if (card) card.open = true; }
      await renderPredictorPatternResult(saved.pattern.payload, { isCurrent });
      if (!isCurrent()) return;
    }
    renderBenchmarkBoard();
    activateTab((resultId ? null : saved.tab) || (results.markers || results.signature ? "markers" : results.ml || results.dl || newest?.route === "/api/predictor-patterns" ? "benchmark" : results.cox ? "cox" : results.cohort ? "tables" : "km"));
    syncDownloadButtonAvailability();
    scheduleResultCurrencySync(0);
    savedWorkRefs.Panel.classList.add("hidden");
    savedWorkRefs.Button.setAttribute("aria-expanded", "false");
    savedWorkState.pending = false; savedWorkState.error = false;
    setAutosaveStatus("Saved work restored", "saved");
    let source = newest?.source_version || Object.values(results).find((p) => p?.saved_result)?.saved_result?.source_version;
    if (source) {
      try { const info = JSON.parse(source); source = `${info.source || info.version || "saved"}${info.source_sha256 ? ` (${info.source_sha256.slice(0, 12)})` : ""}`; } catch { /* A legacy source label is already readable text. */ }
    }
    setRuntimeBanner(`${resultId ? "Completed result" : "Saved screen"} restored without rerunning the analysis.${payload.snapshot_saved_at && !resultId ? ` Screen saved ${new Date(payload.snapshot_saved_at).toLocaleString()}.` : ""}${source ? ` Saved source: ${source}.` : ""} ${missingMatrix ? "The saved marker file is unavailable; reattach it before rerunning marker analysis. " : ""}All completed results, including newer runs, are available in Saved work history. Changed settings still require a new analysis.`, missingMatrix ? "warning" : "info");
    syncHistoryState("replace");
  } finally { savedWorkState.restoring = false; runtime.restoringSavedWork = false; }
}

function initSavedWorkspaces() {
  savedWorkRefs.Button.addEventListener("click", async () => {
    savedWorkRefs.Panel.classList.toggle("hidden");
    savedWorkRefs.Button.setAttribute("aria-expanded", String(!savedWorkRefs.Panel.classList.contains("hidden")));
    if (!savedWorkRefs.Panel.classList.contains("hidden")) {
      if (savedWorkState.error) await saveCurrentWorkspace();
      try { await refreshSavedWork(); } catch (error) { showError(error); }
    }
  });
  document.getElementById("closeSavedWorkButton").addEventListener("click", () => {
    savedWorkRefs.Panel.classList.add("hidden"); savedWorkRefs.Button.setAttribute("aria-expanded", "false");
  });
  document.addEventListener("change", (event) => {
    if (event.target === refs.markerValidationFile) runtime.savedValidationFile = null;
    queueSavedWorkspace();
  });
  refreshSavedWork().then(async () => {
    if (savedWorkState.enabled && savedWorkState.initialHistory?.datasetId) await reopenSavedWorkspace(savedWorkState.initialHistory.datasetId);
  }).catch((error) => {
    savedWorkState.error = true;
    setAutosaveStatus("Saved work unavailable", "error");
    // Do not replace an upload or analysis message while this background check finishes.
    document.getElementById("autosaveStatus").title = errorMessageText(error);
  });
}

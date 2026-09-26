// SurvStudio front end, part 2/8: Request configs, result currency, guided navigation, history, and plot sizing.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

function currentSharedModelSelections(goal = "ml") {
  const featureChecklist = goal === "dl" ? refs.dlModelFeatureChecklist : refs.modelFeatureChecklist;
  const categoricalChecklist = goal === "dl" ? refs.dlModelCategoricalChecklist : refs.modelCategoricalChecklist;
  const features = selectedCheckboxValues(featureChecklist);
  return {
    features,
    categoricalFeatures: selectedCheckboxValues(categoricalChecklist)
      .filter((value) => features.includes(value)),
  };
}

function estimateEncodedFeatureWidth(features = [], categoricalFeatures = []) {
  if (!state.dataset) return 0;
  const categoricalSet = new Set(categoricalFeatures);
  const datasetColumns = new Map((state.dataset.columns || []).map((column) => [column.name, column]));
  return features.reduce((width, feature) => {
    const column = datasetColumns.get(feature);
    if (!column) return width;
    const inferredCategorical = categoricalSet.has(feature) || ["categorical", "datetime"].includes(String(column.kind || ""));
    if (!inferredCategorical) return width + 1;
    const uniqueCount = Number(column.n_unique);
    const resolvedUniqueCount = Number.isFinite(uniqueCount) ? Math.max(uniqueCount, 0) : 0;
    return width + Math.max(resolvedUniqueCount - 1, 0) + 2;
  }, 0);
}

function guidedPredictiveFeatureSummaryState() {
  if (!state.dataset) return null;
  const { features, categoricalFeatures: mlCategoricalFeatures } = currentSharedModelSelections("ml");
  const { categoricalFeatures: dlCategoricalFeatures } = currentSharedModelSelections("dl");
  const eligibleCount = modelFeatureCandidateColumns().length;
  const featureCount = features.length;
  const mlEncodedWidth = estimateEncodedFeatureWidth(features, mlCategoricalFeatures);
  const dlEncodedWidth = estimateEncodedFeatureWidth(features, dlCategoricalFeatures);
  const widestEncodedWidth = Math.max(mlEncodedWidth, dlEncodedWidth);

  let readiness = null;
  if (!featureCount) {
    readiness = {
      tone: "warning",
      title: "No shared features selected",
      text: "Select at least one shared ML/DL feature before running Compare All or a single predictive model.",
    };
  } else if (featureCount === DEFAULT_MODEL_FEATURE_SELECTION_LIMIT && eligibleCount > featureCount) {
    readiness = {
      tone: "ready",
      title: "Compact starter set selected",
      text: `Fresh cohorts start with up to 20 shared features for a faster first run. This cohort has ${formatValue(eligibleCount)} eligible features, so review the list if you expected a wider benchmark.`,
    };
  } else if (featureCount >= GUIDED_FEATURESET_HIGH_COUNT || widestEncodedWidth >= GUIDED_FEATURESET_HIGH_WIDTH) {
    readiness = {
      tone: "warning",
      title: "Large feature set selected",
      text: "Compare All can slow down substantially with a wide shared feature list. Review the selection before benchmarking; runtime and instability usually become the limiting factors before raw memory on moderate cohorts.",
    };
  } else if (featureCount >= GUIDED_FEATURESET_WARNING_COUNT || widestEncodedWidth >= GUIDED_FEATURESET_WARNING_WIDTH) {
    readiness = {
      tone: "warning",
      title: "Expanded feature set selected",
      text: "Expect slower Compare All runs with this many shared inputs. Review the list if you only need a first-pass benchmark.",
    };
  }

  return {
    featureCount,
    eligibleCount,
    mlCategoricalCount: mlCategoricalFeatures.length,
    dlCategoricalCount: dlCategoricalFeatures.length,
    mlEncodedWidth,
    dlEncodedWidth,
    featurePreview: summarizeFeatureNames(features, 5),
    readiness,
  };
}

function renderGuidedPredictiveFeatureSummary(goal = runtime.guidedGoal) {
  if (!["ml", "dl", "predictive"].includes(goal) || !state.dataset) return "";
  const summary = guidedPredictiveFeatureSummaryState();
  if (!summary) return "";
  return `
    <div class="guided-selection-block">
      <strong>Shared model inputs</strong>
      <div class="guided-quick-grid guided-quick-grid-compact">
        <div class="guided-quick-item">
          <strong>Selected raw features</strong>
          <span>${escapeHtml(`${formatValue(summary.featureCount)} / ${formatValue(summary.eligibleCount)} eligible`)}</span>
        </div>
        <div class="guided-quick-item">
          <strong>ML encoded width</strong>
          <span>${escapeHtml(`${formatValue(summary.mlEncodedWidth)} cols`)}</span>
        </div>
        <div class="guided-quick-item">
          <strong>DL encoded width</strong>
          <span>${escapeHtml(`${formatValue(summary.dlEncodedWidth)} cols`)}</span>
        </div>
        <div class="guided-quick-item">
          <strong>Categorical flags</strong>
          <span>${escapeHtml(`ML ${formatValue(summary.mlCategoricalCount)} · DL ${formatValue(summary.dlCategoricalCount)}`)}</span>
        </div>
      </div>
      <span class="guided-inline-note">Compare All uses the shared raw feature list. ML and DL keep their own categorical flags on top of the same raw inputs.</span>
      <span class="guided-inline-note">Preview: ${escapeHtml(summary.featurePreview)}</span>
    </div>
    ${summary.readiness
      ? `
        <div class="guided-readiness${summary.readiness.tone === "ready" ? " ready" : ""}">
          <strong>${escapeHtml(summary.readiness.title)}</strong>
          <span>${escapeHtml(summary.readiness.text)}</span>
        </div>
      `
      : ""}
  `;
}

function syncGuidedPredictiveFeatureSummaryMount() {
  if (!refs.benchmarkGuidedFeatureSummary) return false;
  const showSummary = (
    runtime.uiMode === "guided"
    && currentGuidedStep() === 4
    && runtime.guidedGoal === "predictive"
    && Boolean(state.dataset)
  );
  refs.benchmarkGuidedFeatureSummary.classList.toggle("hidden", !showSummary);
  refs.benchmarkGuidedFeatureSummary.innerHTML = showSummary
    ? renderGuidedPredictiveFeatureSummary("predictive")
    : "";
  return showSummary;
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
      time_unit_label: String(requestConfig.time_unit_label || DEFAULT_TIME_UNIT_LABEL),
      // Compare numerically so "2000.0" and 2000 describe the same truncation.
      max_time: numberOrDefault(requestConfig.max_time, null),
      risk_table_points: Number(requestConfig.risk_table_points),
      logrank_weight: String(requestConfig.logrank_weight || "logrank"),
      fh_p: String(requestConfig.logrank_weight || "logrank") === "fleming_harrington" ? numberOrDefault(requestConfig.fh_p, 1) : null,
      show_confidence_bands: Boolean(requestConfig.show_confidence_bands),
    };
  }

  if (goal === "cox") {
    return {
      ...base,
      covariates: sortedStrings(requestConfig.covariates || []),
      categorical_covariates: sortedStrings(requestConfig.categorical_covariates || []),
      strata_columns: sortedStrings(requestConfig.strata_columns || []),
    };
  }

  if (goal === "ml") {
    const compareRun = String(requestConfig.model_type || "") === "compare";
    if (compareRun !== expectsCompare) return null;
    const effectiveModelType = expectsCompare ? "compare" : String(requestConfig.model_type || "");
    const learningRateApplies = expectsCompare || effectiveModelType === "gbs";
    const treeCountApplies = expectsCompare || effectiveModelType === "rsf" || effectiveModelType === "gbs";
    const evaluationStrategy = expectsCompare ? String(requestConfig.evaluation_strategy || "holdout") : null;
    const repeatedCv = evaluationStrategy === "repeated_cv";
    return {
      ...base,
      model_type: effectiveModelType,
      features: sortedStrings(requestConfig.features || []),
      categorical_features: sortedStrings(requestConfig.categorical_features || []),
      n_estimators: treeCountApplies ? Number(requestConfig.n_estimators) : null,
      max_depth: String(requestConfig.max_depth ?? ""),
      learning_rate: learningRateApplies ? Number(requestConfig.learning_rate) : null,
      random_state: numberOrDefault(requestConfig.random_state, 42),
      evaluation_strategy: evaluationStrategy,
      cv_folds: repeatedCv ? numberOrDefault(requestConfig.cv_folds, 5) : null,
      cv_repeats: repeatedCv ? numberOrDefault(requestConfig.cv_repeats, 3) : null,
      locked_test_fraction: repeatedCv ? normalizedLockedTestFraction(requestConfig.locked_test_fraction) : null,
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
    const evaluationStrategy = String(requestConfig.evaluation_strategy || "holdout");
    const repeatedCv = evaluationStrategy === "repeated_cv";
    return {
      ...base,
      model_type: effectiveModelType,
      features: sortedStrings(requestConfig.features || []),
      categorical_features: sortedStrings(requestConfig.categorical_features || []),
      hidden_layers: usesHiddenLayers || expectsCompare ? (requestConfig.hidden_layers || []).map(Number) : null,
      dropout: Number(requestConfig.dropout),
      learning_rate: Number(requestConfig.learning_rate),
      epochs: Number(requestConfig.epochs),
      batch_size: usesDiscreteTime || expectsCompare ? numberOrDefault(requestConfig.batch_size, 64) : null,
      random_seed: numberOrDefault(requestConfig.random_seed, 42),
      evaluation_strategy: evaluationStrategy,
      cv_folds: repeatedCv ? numberOrDefault(requestConfig.cv_folds, 5) : null,
      cv_repeats: repeatedCv ? numberOrDefault(requestConfig.cv_repeats, 3) : null,
      early_stopping_patience: numberOrDefault(requestConfig.early_stopping_patience, 10),
      early_stopping_min_delta: numberOrDefault(requestConfig.early_stopping_min_delta, 0.0001),
      parallel_jobs: repeatedCv ? numberOrDefault(requestConfig.parallel_jobs, 1) : null,
      num_time_bins: usesDiscreteTime || expectsCompare ? numberOrDefault(requestConfig.num_time_bins, 50) : null,
      d_model: usesTransformer ? numberOrDefault(requestConfig.d_model, 64) : null,
      n_heads: usesTransformer ? numberOrDefault(requestConfig.n_heads, 4) : null,
      n_layers: usesTransformer ? numberOrDefault(requestConfig.n_layers, 2) : null,
      latent_dim: usesVae ? numberOrDefault(requestConfig.latent_dim, 8) : null,
      n_clusters: usesVae ? numberOrDefault(requestConfig.n_clusters, 3) : null,
      locked_test_fraction: expectsCompare && repeatedCv ? normalizedLockedTestFraction(requestConfig.locked_test_fraction) : null,
    };
  }

  if (goal === "tables") {
    const outcomeRestricted = Boolean(requestConfig.time_column && requestConfig.event_column);
    return {
      dataset_id: String(requestConfig?.dataset_id || ""),
      variables: sortedStrings(requestConfig.variables || []),
      group_column: String(requestConfig.group_column || ""),
      time_column: outcomeRestricted ? String(requestConfig.time_column) : "",
      event_column: outcomeRestricted ? String(requestConfig.event_column) : "",
      event_positive_value: outcomeRestricted ? String(requestConfig.event_positive_value ?? "") : "",
    };
  }

  return base;
}

function currentGoalRequestConfig(goal, { expectsCompareOverride = null } = {}) {
  if (!state.dataset) return null;
  if (goal === "tables") {
    return normalizedRequestConfig(goal, {
      dataset_id: state.dataset.dataset_id,
      variables: selectedCheckboxValues(refs.cohortVariableChecklist),
      group_column: refs.groupColumn?.value || "",
      ...cohortTableOutcomeConfig(),
    });
  }
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
      time_unit_label: refs.timeUnitLabel?.value || DEFAULT_TIME_UNIT_LABEL,
      max_time: refs.maxTime?.value || "",
      risk_table_points: refs.riskTablePoints?.value,
      logrank_weight: refs.logrankWeight?.value || "logrank",
      fh_p: refs.fhPower?.value || 1,
      show_confidence_bands: Boolean(refs.showConfidenceBands?.checked),
    });
  }

  if (goal === "cox") {
    const { covariates, categoricalCovariates, strataColumns } = currentCoxSelections();
    return normalizedRequestConfig(goal, {
      ...base,
      covariates,
      categorical_covariates: categoricalCovariates,
      strata_columns: strataColumns,
    });
  }

  if (goal === "ml") {
    const { features, categoricalFeatures } = currentSharedModelSelections("ml");
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
      random_state: sharedPredictiveSeed(),
      shap_safe_mode: Boolean(refs.mlShapSafeMode?.checked),
      evaluation_strategy: refs.mlEvaluationStrategy?.value || "holdout",
      cv_folds: refs.mlCvFolds?.value,
      cv_repeats: refs.mlCvRepeats?.value,
      locked_test_fraction: currentLockedTestFraction("ml"),
    }, { expectsCompare });
  }

  if (goal === "dl") {
    const { features, categoricalFeatures } = currentSharedModelSelections("dl");
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
      batch_size: refs.dlBatchSize?.value,
      random_seed: refs.dlRandomSeed?.value,
      evaluation_strategy: refs.dlEvaluationStrategy?.value || "holdout",
      cv_folds: refs.dlCvFolds?.value,
      cv_repeats: refs.dlCvRepeats?.value,
      early_stopping_patience: refs.dlEarlyStoppingPatience?.value,
      early_stopping_min_delta: refs.dlEarlyStoppingMinDelta?.value,
      parallel_jobs: refs.dlParallelJobs?.value,
      num_time_bins: refs.dlNumTimeBins?.value,
      d_model: refs.dlDModel?.value,
      n_heads: refs.dlHeads?.value,
      n_layers: refs.dlLayers?.value,
      latent_dim: refs.dlLatentDim?.value,
      n_clusters: refs.dlClusters?.value,
      locked_test_fraction: currentLockedTestFraction("dl"),
    }, { expectsCompare });
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
    if (typeof benchmarkBoardState === "function") {
      const board = benchmarkBoardState();
      if (
        !board
        || board.predictiveBusy
        || board.guidedPredictiveIncomplete
        || board.showingStaleBoard
        || board.hasMixedEvaluation
        || board.visibleHasMixedRunGroups
        || board.visibleHasSplitMismatch
        || !Array.isArray(board.visibleFamilies)
        || board.visibleFamilies.length !== 2
        || !board.visibleRows.length
      ) {
        return null;
      }
    }
    const currentMl = currentCompareGoalPayload("ml");
    const currentDl = currentCompareGoalPayload("dl");
    return currentMl && currentDl ? { ml: currentMl, dl: currentDl } : null;
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

function compareSnapshotPayload(goal) {
  if (!["ml", "dl"].includes(goal)) return null;
  const payload = runtime.compareCache?.unified?.[goal] || null;
  return payloadRepresentsCompareRun(payload) ? payload : null;
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
    return Boolean(
      state.ml
      || state.dl
      || runtime.compareCache?.ml
      || runtime.compareCache?.dl
      || runtime.compareCache?.unified?.ml
      || runtime.compareCache?.unified?.dl,
    );
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
  if (goal === "predictive") {
    const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
    const selectedSingleCurrent = Boolean(selectedPredictiveSingleResult(selectedModel.family));
    if (runtime.predictiveWorkbenchIntent === "train" && selectedSingleCurrent) {
      return {
        tone: "ready",
        label: currentLabel,
        title: `${selectedModel.label} result is current`,
        text: "Visible settings match the selected model result shown here.",
      };
    }
    if (guidedPredictiveHasLeaderboardReference()) {
      const board = typeof benchmarkBoardState === "function" ? benchmarkBoardState() : null;
      if (board?.showingStaleBoard) {
        return {
          tone: "warning",
          label: "Stale reference",
          title: "Compare All result shown as stale reference",
          text: "Visible leaderboard is a stale Compare All snapshot. Rerun Compare All Models to refresh it.",
        };
      }
      return {
        tone: "ready",
        label: currentLabel,
        title: "Compare All result is current",
        text: "Visible settings match the predictive leaderboard shown here.",
      };
    }
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
    if (goal === "tables") {
      return {
        tone: "warning",
        label: "Needs rerun",
        title: `${goalLabel(goal)} settings changed`,
        text: "Visible settings changed after this table was built. You can still export the visible table, or rebuild it to refresh the output.",
      };
    }
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

function guidedGoalCanReachReviewStep(goal = runtime.guidedGoal) {
  if (!goal) return false;
  if (goal === "predictive") {
    return Boolean(selectedPredictiveSingleResult(predictiveFamilyGoal()) || guidedPredictiveHasLeaderboardReference());
  }
  return Boolean(currentGoalResult(goal));
}

function renderGuidedCoxSelectionSummary() {
  const { covariates, categoricalCovariates, strataColumns } = currentCoxSelections();
  if (!covariates.length && !strataColumns.length) {
    return `
      <div class="guided-readiness">
        <strong>No covariates selected</strong>
        <span>Select the variables you want to test before running Cox PH.</span>
      </div>
    `;
  }
  const categoricalSet = new Set(categoricalCovariates);
  const covariateChips = covariates.map((value) => `
    <span class="dataset-preset-chip guided-selection-chip${categoricalSet.has(value) ? " is-categorical" : ""}">
      ${escapeHtml(value)}
      ${categoricalSet.has(value) ? '<span class="guided-selection-chip-tag">cat</span>' : ""}
    </span>
  `).join("");
  const strataChips = strataColumns.map((value) => `
    <span class="dataset-preset-chip guided-selection-chip">${escapeHtml(value)}<span class="guided-selection-chip-tag">strata</span></span>
  `).join("");
  const covariateBlock = covariates.length
    ? `
      <div class="guided-selection-block">
        <strong>Selected covariates (${covariates.length})</strong>
        <div class="dataset-preset-chips guided-selection-chips">${covariateChips}</div>
      </div>
    `
    : `
      <div class="guided-readiness">
        <strong>No covariates selected</strong>
        <span>Select at least one non-stratified covariate before running Cox PH.</span>
      </div>
    `;
  const strataBlock = strataColumns.length
    ? `
      <div class="guided-selection-block">
        <strong>Selected strata (${strataColumns.length})</strong>
        <div class="dataset-preset-chips guided-selection-chips">${strataChips}</div>
        <span class="guided-inline-note">Stratified variables use stratum-specific baseline hazards and are not reported with hazard ratios.</span>
      </div>
    `
    : "";
  return `
    ${covariateBlock}
    ${strataBlock}
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
    ? `Complete-case Cox will drop rows with missing selected Cox inputs. Biggest drops: ${missingNotes.join(", ")}${preview.missing_by_covariate.length > 4 ? " ..." : ""}.`
    : "All rows remain analyzable with the current Cox input set.";
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
  const predictiveSingleReview = reviewGoal === "predictive"
    && runtime.predictiveWorkbenchIntent === "train"
    && Boolean(selectedPredictiveSingleResult(reviewFamily || predictiveFamilyGoal()));
  const reviewScopeBusy = reviewGoal === "predictive"
    ? (isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl"))
    : (reviewFamily ? isScopeBusy(reviewFamily) : false);
  const reviewMode = reviewGoal ? (predictiveSingleReview ? "Run Analysis" : guidedResultModeLabel(reviewGoal)) : null;
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
        ...(Boolean(selectedPredictiveSingleResult(reviewFamily || predictiveFamilyGoal())) && !runtime.workbenchRevealed
          ? [{ label: `Return to ${predictiveModelMeta(currentPredictiveModelKey()).label}`, action: "return-trained-predictive-model", tone: "ghost" }]
          : []),
        { label: "Compare all models", action: "run-predictive-compare-all", tone: "primary" },
        { label: "Review shared features", action: "review-shared-features", tone: "ghost" },
        { label: "Back", action: "previous-step", tone: "ghost" },
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
  if (!guidedGoalCanReachReviewStep(runtime.guidedGoal) && bounded > 4) return 4;
  return bounded;
}

function currentGuidedStep() {
  return normalizedGuidedStep(runtime.guidedStep);
}

function maxReachableGuidedStep() {
  if (!state.dataset) return 1;
  if (!endpointIsReady()) return 2;
  if (!runtime.guidedGoal) return 3;
  if (!guidedGoalCanReachReviewStep(runtime.guidedGoal)) return 4;
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
    runtime.guidedStep = normalizedGuidedStep(guidedGoalCanReachReviewStep(runtime.guidedGoal) ? 5 : 4);
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
  const plots = allPlotRefs();
  plots.forEach((plot) => {
    if (!plot?.data?.length || !plotIsDisplayed(plot)) return;
    try {
      Plotly.Plots.resize(plot);
      stabilizePlotShellHeight(plot);
    } catch {
      // Ignore stale nodes during guided view remounts.
    }
  });
}

function allPlotRefs() {
  return [
    refs.kmPlot,
    refs.coxPlot,
    refs.coxDiagnosticsPlot,
    refs.coxMartingalePlot,
    refs.cutpointPlot,
    refs.mlImportancePlot,
    refs.mlShapPlot,
    refs.mlComparisonPlot,
    refs.dlImportancePlot,
    refs.dlLossPlot,
    refs.dlComparisonPlot,
    refs.benchmarkComparisonPlot,
  ];
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

function waitForRenderFrames(count = 1) {
  const frames = Math.max(1, Number(count) || 1);
  return new Promise((resolve) => {
    const step = (remaining) => {
      requestAnimationFrame(() => {
        if (remaining <= 1) {
          resolve();
          return;
        }
        step(remaining - 1);
      });
    };
    step(frames);
  });
}

async function stabilizePlotsBeforeBanner(plots = [], banner, { maxAttempts = 12, tolerance = 1 } = {}) {
  const visiblePlots = plots.filter((plot) => plot && !plot.closest(".hidden"));
  if (!visiblePlots.length || !banner) return;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await waitForRenderFrames(2);
    visiblePlots.forEach((plot) => {
      if (!plot?.data?.length || !plotIsDisplayed(plot)) return;
      try {
        Plotly.Plots.resize(plot);
        stabilizePlotShellHeight(plot);
      } catch {
        // Ignore detached or stale plot nodes during fast UI transitions.
      }
    });
    const bannerRect = banner.getBoundingClientRect();
    const plotsAreOrdered = visiblePlots.every((plot) => plot.getBoundingClientRect().bottom <= bannerRect.top + tolerance);
    if (plotsAreOrdered) return;
    await new Promise((resolve) => window.setTimeout(resolve, 25));
  }
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

function initPlotResizeObserver() {
  if (runtime.plotResizeObserver || typeof window.ResizeObserver !== "function") return;
  const widthCache = new WeakMap();
  runtime.plotResizeObserver = new window.ResizeObserver((entries) => {
    const widthChanged = entries.some((entry) => {
      const target = entry?.target;
      if (!target) return false;
      const nextWidth = Number(entry.contentRect?.width || target.clientWidth || 0);
      if (!Number.isFinite(nextWidth) || nextWidth <= 0) return false;
      const previousWidth = widthCache.get(target);
      widthCache.set(target, nextWidth);
      return Number.isFinite(previousWidth) && Math.abs(nextWidth - previousWidth) > 1;
    });
    if (widthChanged) scheduleVisiblePlotResize(30);
  });
  allPlotRefs().forEach((plot) => {
    if (!plot) return;
    widthCache.set(plot, Number(plot.clientWidth || 0));
    runtime.plotResizeObserver.observe(plot);
  });
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
    coxStrata: selectedCheckboxValues(refs.strataChecklist),
    coxPreviewKey: runtime.coxPreview?.key || "",
    modelFeatures: selectedCheckboxValues(refs.modelFeatureChecklist),
    modelCategoricals: selectedCheckboxValues(refs.modelCategoricalChecklist),
    dlModelCategoricals: selectedCheckboxValues(refs.dlModelCategoricalChecklist),
    cohortVariables: selectedCheckboxValues(refs.cohortVariableChecklist),
    mlModelType: refs.mlModelType?.value || "",
    mlNEstimators: refs.mlNEstimators?.value || "",
    mlLearningRate: refs.mlLearningRate?.value || "",
    mlSkipShap: Boolean(refs.mlSkipShap?.checked),
    mlShapSafeMode: Boolean(refs.mlShapSafeMode?.checked),
    mlEvaluationStrategy: refs.mlEvaluationStrategy?.value || "",
    mlCvFolds: refs.mlCvFolds?.value || "",
    mlCvRepeats: refs.mlCvRepeats?.value || "",
    mlRandomSeed: refs.mlRandomSeed?.value || "",
    lockedTestEnabled: Boolean(refs.mlLockedTestToggle?.checked),
    lockedTestPercent: refs.mlLockedTestFraction?.value || "",
    timeUnitAutoLabel: Boolean(runtime.timeUnitAutoLabel),
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

function scheduleResultCurrencySync(delay = 60) {
  if (runtime.resultCurrencySyncTimer) window.clearTimeout(runtime.resultCurrencySyncTimer);
  runtime.resultCurrencySyncTimer = window.setTimeout(() => {
    runtime.resultCurrencySyncTimer = null;
    syncDownloadButtonAvailability();
    updateCohortTableButtonLabel();
    renderBenchmarkBoard();
    renderGuidedChrome();
  }, delay);
}

function queueHistorySync() {
  syncStaleSingleResultArtifacts();
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
  setInputValue(refs.mlRandomSeed, snapshot.mlRandomSeed || snapshot.dlRandomSeed);
  setInputValue(refs.mlLockedTestFraction, snapshot.lockedTestPercent);
  setInputValue(refs.dlLockedTestFraction, snapshot.lockedTestPercent);
  if (snapshot.lockedTestEnabled !== undefined) {
    if (refs.mlLockedTestToggle) refs.mlLockedTestToggle.checked = Boolean(snapshot.lockedTestEnabled);
    if (refs.dlLockedTestToggle) refs.dlLockedTestToggle.checked = Boolean(snapshot.lockedTestEnabled);
  }
  runtime.timeUnitAutoLabel = snapshot.timeUnitAutoLabel === undefined
    ? String(refs.timeUnitLabel?.value || "") === automaticTimeUnitLabel()
    : Boolean(snapshot.timeUnitAutoLabel);
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
  setCheckedValues(refs.strataChecklist, snapshot.coxStrata || []);
  setCheckedValues(refs.modelFeatureChecklist, snapshot.modelFeatures || []);
  setCheckedValues(refs.modelCategoricalChecklist, snapshot.modelCategoricals || []);
  setCheckedValues(refs.dlModelFeatureChecklist, snapshot.modelFeatures || []);
  setCheckedValues(refs.dlModelCategoricalChecklist, snapshot.dlModelCategoricals || snapshot.modelCategoricals || []);
  syncModelFeatureMirrors(refs.modelFeatureChecklist);
  syncModelCategoricalMirrors(refs.modelCategoricalChecklist);
  syncModelCategoricalMirrors(refs.dlModelCategoricalChecklist);
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
    runtime.predictiveWorkbenchIntent = null;
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
    runtime.predictiveWorkbenchIntent = null;
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
    runtime.predictiveWorkbenchIntent = normalizedPredictiveWorkbenchIntent(historyState?.predictiveWorkbenchIntent)
      || (runtime.workbenchRevealed ? "train" : null);
    if (restoredUiMode === "guided") {
      runtime.guidedGoal = restoredGuidedGoal;
      runtime.guidedStep = restoredGuidedStep;
    }
    setUiMode(restoredUiMode, { syncHistory: false, preserveGuidedState: restoredUiMode === "guided" });
    if (!state.dataset || state.dataset.dataset_id !== historyState.datasetId) {
      const datasetToken = beginRequestToken("dataset");
      const payload = await fetchJSON(`/api/dataset/${historyState.datasetId}`, { signal: requestSignal("dataset") });
      if (restoreToken !== runtime.historyRestoreToken || !requestTokenMatches("dataset", datasetToken)) return;
      updateAfterDataset(payload);
    } else {
      showWorkspace();
    }
    if (restoreToken !== runtime.historyRestoreToken) return;
    runtime.workbenchRevealed = Boolean(historyState?.workbenchRevealed);
    runtime.predictiveWorkbenchIntent = normalizedPredictiveWorkbenchIntent(historyState?.predictiveWorkbenchIntent)
      || (runtime.workbenchRevealed ? "train" : null);
    applyControlSnapshot(historyState.controls || null);
    runtime.guidedGoal = restoredGuidedGoal;
    runtime.guidedStep = restoredGuidedStep;
    activateTab(historyState.tab || restoredGuidedGoal || "km", { setGuidedGoal: false });
    renderGuidedChrome();
  } catch (error) {
    // A newer navigation or dataset load cancelled this restore; leave the workspace alone.
    if (!isSupersededRequestError(error)) goHome({ syncHistory: false });
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
  const summaryPayload = refs.deriveSummary?.dataset.summaryKind === "derived"
    ? currentDerivedSummaryPayload()
    : null;
  const displayedDerivedColumn = normalizeDeriveSummaryText(summaryPayload?.derivedColumn);
  const displayedSummary = summaryPayload?.summary || null;
  const controlsMismatch = displayedDerivedColumn
    ? !deriveDraftMatchesStoredRecipe(displayedDerivedColumn, displayedSummary)
    : false;
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
    if (!deriveLocked) {
      refs.deriveStatus.textContent = "";
      return;
    }
    let statusText = `Derived-group settings are locked while Group by uses ${activeGroup}. Set Group by to Overall only if you want Create or these parameters to affect the next grouping analysis. Run again only reuses the current Group by value.`;
    if (displayedDerivedColumn) {
      if (activeGroup === displayedDerivedColumn) {
        statusText += controlsMismatch
          ? ` The card below describes ${displayedDerivedColumn}; the disabled controls above are draft settings only and do not describe that stored result.`
          : ` The card below describes the current derived grouping ${displayedDerivedColumn}.`;
      } else {
        statusText += ` The card below describes ${displayedDerivedColumn}, not the current Group by ${activeGroup}.`;
        if (controlsMismatch) {
          statusText += " The disabled controls above are draft settings only for the next Create action.";
        }
      }
    }
    refs.deriveStatus.textContent = statusText;
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

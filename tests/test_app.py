from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tomllib
import zipfile

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

import survival_toolkit.app as app_module
from survival_toolkit.app import DeepModelRequest, app, fail_bad_request, store
from survival_toolkit.sample_data import make_example_dataset


client = TestClient(app)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_index_uses_relative_static_assets() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert '../static/styles.css?v=' in response.text
    assert '<script src="../static/vendor/plotly-3.4.0.min.js?v=' in response.text
    assert '<script src="../static/app.js?v=' in response.text
    assert "cdn.plot.ly" not in response.text
    assert 'id="mlEvaluationStrategy"' in response.text
    assert 'id="downloadMlManuscriptMarkdownButton"' in response.text
    assert 'id="downloadMlManuscriptLatexButton"' in response.text
    assert 'id="downloadMlManuscriptDocxButton"' in response.text
    assert 'id="downloadMlComparisonPngButton"' in response.text
    assert 'id="downloadMlComparisonSvgButton"' in response.text
    assert 'id="mlJournalTemplate"' in response.text
    assert 'id="downloadCoxPngButton"' in response.text
    assert 'id="downloadCoxSvgButton"' in response.text
    assert 'id="runDlCompareButton"' in response.text
    assert 'id="dlEvaluationStrategy"' in response.text
    assert 'id="downloadDlManuscriptMarkdownButton"' in response.text
    assert 'id="downloadDlManuscriptLatexButton"' in response.text
    assert 'id="downloadDlManuscriptDocxButton"' in response.text
    assert 'id="downloadDlComparisonPngButton"' in response.text
    assert 'id="downloadDlComparisonSvgButton"' in response.text
    assert 'id="dlEarlyStoppingPatience"' in response.text
    assert 'id="dlParallelJobs"' in response.text
    assert 'id="dlBatchSize"' in response.text
    assert 'id="dlRandomSeed"' in response.text
    assert 'id="dlNumTimeBins"' in response.text
    assert 'id="dlDModel"' in response.text
    assert 'id="dlHeads"' in response.text
    assert 'id="dlLayers"' in response.text
    assert 'id="dlLatentDim"' in response.text
    assert 'id="dlClusters"' in response.text
    assert 'id="shutdownButton"' in response.text
    assert 'id="selectAllCoxCovariatesButton"' in response.text
    assert 'id="clearCoxCovariatesButton"' in response.text
    assert 'id="selectAllCoxCategoricalsButton"' in response.text
    assert 'id="clearCoxCategoricalsButton"' in response.text


def test_index_mentions_fleming_harrington_p_only_label() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Fleming-Harrington (fh_p only)" in response.text
    assert 'id="dlJournalTemplate"' in response.text
    assert 'id="runCompareInlineButton"' in response.text
    assert 'id="runDlCompareInlineButton"' in response.text
    assert "publication-ready" not in response.text
    assert "exploratory Kaplan-Meier curves" in response.text
    assert 'class="button ghost" id="loadTcgaUploadReadyButton"' in response.text
    assert 'class="button ghost" id="loadTcgaButton"' in response.text
    assert 'class="button ghost" id="loadGbsg2Button"' in response.text
    assert 'class="button ghost" id="loadExampleButton"' in response.text
    assert "Compact upload-style test cohort with fewer columns" in response.text
    assert "Broader bundled clinical cohort with extra fields" in response.text
    assert "Classic breast cancer recurrence-survival cohort" in response.text
    assert "Fastest demo path for checking the app and workflow" in response.text
    assert 'id="uploadButton" type="button"' in response.text
    assert 'id="guidedModeButton"' in response.text
    assert 'id="expertModeButton"' in response.text
    assert 'id="guidedShell"' in response.text
    assert 'id="guidedSummaryBar"' in response.text
    assert 'id="guidedPanel"' in response.text
    assert 'id="guidedRailStatus"' in response.text
    assert 'id="guidedRailStatusLabel"' in response.text
    assert 'id="guidedRailStatusTitle"' in response.text
    assert 'id="guidedRailStatusText"' in response.text
    assert 'class="guided-rail"' in response.text
    assert "Confirm outcome" in response.text
    assert "Choose analysis" in response.text
    assert "Configure &amp; run" in response.text
    assert "Review results" in response.text
    assert "Upload or open a sample cohort" in response.text
    assert "Configure &amp; run" in response.text
    assert '<span>Evaluation Mode</span>' in response.text
    assert "Evaluation Mode applies to both <strong>Train a model</strong> and <strong>Compare All</strong>" in response.text
    assert "Risk table ticks" in response.text
    assert "It does not change the Kaplan-Meier curve itself." in response.text
    assert "Used everywhere" in response.text
    assert "Used mainly for grouping and display" in response.text
    assert 'id="showAllEventColumns"' in response.text
    assert 'id="eventColumnHelp"' in response.text
    assert 'id="eventColumnWarning"' in response.text
    assert 'id="eventValueWarning"' in response.text
    assert 'id="groupColumnWarning"' in response.text
    assert "Show all columns for Event" in response.text
    assert "Showing event-like binary columns only." in response.text
    assert 'id="groupingDetails"' in response.text
    assert 'id="groupingSummaryText"' in response.text
    assert 'id="kmDependencyText"' in response.text
    assert 'id="coxDependencyText"' in response.text
    assert 'id="tableDependencyText"' in response.text
    assert 'id="tableOutputStatusText"' in response.text
    assert 'id="runCohortTableButtonLabel"' in response.text
    assert "The reported C-index is apparent on the analyzable cohort, and PH diagnostics below are approximate rank-based checks rather than a full cox.zph test." in response.text
    assert "What this tab uses" in response.text
    assert "KM / grouped summary settings" in response.text
    assert 'id="deriveApplyButton"' in response.text
    assert "Create and apply" in response.text
    assert "Approximate rank-based Spearman screening of Schoenfeld residuals against log time appears here." in response.text


def test_index_exposes_dataset_preset_feedback_ui() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="datasetPresetStatusTitle"' in response.text
    assert 'id="datasetPresetStatusText"' in response.text
    assert 'id="datasetPresetChips"' in response.text
    assert 'id="mlFeatureSummaryText"' in response.text
    assert 'id="mlFeatureSummaryChips"' in response.text
    assert 'id="dlFeatureSummaryText"' in response.text
    assert 'id="dlFeatureSummaryChips"' in response.text
    assert 'id="modelFeatureChecklist"' in response.text
    assert 'id="modelCategoricalChecklist"' in response.text
    assert 'id="dlModelFeatureChecklist"' in response.text
    assert 'id="dlModelCategoricalChecklist"' in response.text
    assert 'id="reviewMlFeaturesButton"' in response.text
    assert 'id="reviewDlFeaturesButton"' in response.text
    assert "Review shared features" in response.text
    assert 'id="mlSkipShap"' in response.text
    assert "Fast mode (skip SHAP)" in response.text
    assert "ML uses the Study Design outcome definition and the shared ML/DL model features selected below." in response.text
    assert "DL uses the Study Design outcome definition and the model features selected below. The same feature list stays synced with the ML tab." in response.text
    assert "No preset applied yet." in response.text
    assert "Applying a preset updates recommended columns and checkbox selections only." in response.text
    assert 'class="button ghost compact-btn" id="applyBasicPresetButton"' in response.text
    assert 'class="button ghost compact-btn" id="applyModelPresetButton"' in response.text
    assert 'class="button-row dataset-preset-actions"' in response.text
    assert "Run setup" in response.text
    assert "Validation and runtime" in response.text
    assert 'class="table-card-head"' in response.text
    assert 'option value="lasso_cox"' in response.text
    assert "screening comparison across Cox PH, LASSO-Cox, RSF, and GBS" in response.text
    assert "Partial dependence and counterfactual analysis are available for tree-model runs only" in response.text
    assert "Fresh datasets preselect up to 20 eligible model features for a faster first run." in response.text
    assert 'id="dlBatchSizeHint"' in response.text


def test_frontend_tracks_workspace_controls_in_history_state() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "captureControlSnapshot()" in text
    assert "controls: captureControlSnapshot()" in text
    assert "applyControlSnapshot(historyState.controls || null);" in text
    assert "queueHistorySync()" in text
    assert "showAllEventColumns: Boolean(refs.showAllEventColumns?.checked)" in text
    assert 'deriveApplyButton: document.getElementById("deriveApplyButton")' in text
    assert "updateGroupingDetailsVisibility(tabName);" in text
    assert "function scrollWorkspaceEntryToTop()" in text
    assert "syncModelFeatureMirrors(refs.modelFeatureChecklist);" in text
    assert "syncModelCategoricalMirrors(refs.modelCategoricalChecklist);" in text


def test_guided_step_indicator_exposes_navigation_a11y_labels() -> None:
    index_html = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "templates" / "index.html"
    html = index_html.read_text()
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'id="stepIndicator" role="navigation" aria-label="Guided workflow steps"' in html
    assert 'aria-label="Step 1: Load data"' in html
    assert 'aria-label="Step 5: Review results"' in html
    assert 'el.setAttribute("aria-disabled", String(s > reachableStep));' in text
    assert "updateAfterDataset(payload, { scrollToTop: true });" in text


def test_frontend_disables_expert_run_buttons_until_endpoint_is_ready() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function syncAnalysisRunButtonAvailability()" in text
    assert 'const readyMessage = endpointReadinessMessage();' in text
    assert "refs.runKmButton," in text
    assert "refs.runCoxButton," in text
    assert "refs.runCohortTableButton," in text
    assert 'setActionDisabledState(refs.runMlButton, mlSingleDisabled, mlSingleTitle);' in text
    assert 'setActionDisabledState(refs.runDlButton, dlSingleDisabled, dlSingleTitle);' in text


def test_frontend_limits_fresh_model_feature_defaults_and_marks_dl_batch_size_scope() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "const DEFAULT_MODEL_FEATURE_SELECTION_LIMIT = 20;" in text
    assert "const defaultModelFeatures = availableCovariates.slice(0, DEFAULT_MODEL_FEATURE_SELECTION_LIMIT);" in text
    assert 'refs.dlBatchSize.disabled = !usesMiniBatchTraining;' in text
    assert "Batch size applies only to DeepHit and Neural MTLR." in text
    assert "Ignored for this architecture because training is full-batch." in text


def test_readme_states_current_scope_and_validation_limitations() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    assert "single-event survival analysis" in readme
    assert "right-censored data" in readme
    assert "no left-truncated entry-time handling" in readme
    assert "no competing-risks analysis" in readme
    assert 'no built-in "apply the locked model directly to an external cohort" workflow yet' in readme
    assert "Apparent C-index" in readme
    assert "rank-based Spearman correlations between Schoenfeld residuals and log time" in readme
    assert "LASSO-Cox (penalized Cox)" in readme
    assert "0.50` is chance-level ranking" in readme


def test_app_uses_threadpool_for_builtin_loaders_and_derive_group() -> None:
    app_py = (Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "app.py").read_text()

    assert "async def _load_builtin_dataset_response(" in app_py
    assert "dataframe = await run_in_threadpool(loader)" in app_py
    assert "return await run_in_threadpool(_run)" in app_py


def test_frontend_exposes_guided_mode_shell_and_history_state() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'uiMode: "guided"' in text
    assert "guidedGoal: null" in text
    assert "guidedStep: 1" in text
    assert "function setUiMode(mode" in text
    assert "function setGuidedGoal(goal" in text
    assert "function setGuidedStep(step" in text
    assert "function maxReachableGuidedStep()" in text
    assert "function canNavigateToGuidedStep(step)" in text
    assert "function updateGuidedSurfaceVisibility()" in text
    assert "function renderGuidedChrome()" in text
    assert "view: \"home\", uiMode: runtime.uiMode" in text
    assert "guidedGoal: runtime.guidedGoal" in text
    assert "guidedStep: runtime.guidedStep" in text
    assert "setUiMode(historyState?.uiMode || runtime.uiMode, { syncHistory: false });" in text
    assert "function queueVisiblePlotResize()" in text
    assert "function resizeVisiblePlotsNow()" in text
    assert "queueVisiblePlotResize();" in text
    assert "runtime.guidedGoal = historyState.guidedGoal || null;" in text
    assert "runtime.guidedStep = historyState.guidedStep" in text
    assert 'const compareRun = String(requestConfig.model_type || "") === "compare";' in text
    assert 'function preferredResultMode(goal)' in text
    assert "handleGuidedPanelAction(button);" in text
    assert 'setGuidedStep(currentGuidedStep() + 1, { historyMode: "push" });' in text
    assert 'setGuidedStep(currentGuidedStep() - 1, { historyMode: "push" });' in text
    assert 'setGuidedGoal(target.dataset.goal || null, { historyMode: "push" });' in text
    assert 'setGuidedStep(5, { scroll: false, historyMode: "push" });' in text
    assert 'refs.stepIndicator?.addEventListener("click", (event) => {' in text
    assert 'if (requestedStep === 1) {' in text
    assert 'if (requestedStep >= 4 && runtime.guidedGoal) {' in text
    assert "historyRestoreToken: 0" in text
    assert "const restoreToken = ++runtime.historyRestoreToken;" in text
    assert 'goHome({ historyMode: "push" });' in text
    assert 'setUiMode("guided", { historyMode: "push" })' in text
    assert 'setUiMode("expert", { historyMode: "push" })' in text


def test_frontend_limits_event_columns_by_default_and_warns_on_nonstandard_selection() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function isEventLikeColumnName(columnName)" in text
    assert "function looksLikeBaselineStatusColumn(columnName)" in text
    assert "function recommendedEventColumns()" in text
    assert "function renderEventColumnOptions(" in text
    assert "function inferEventPositiveSelection(" in text
    assert "function updateEventValueGuidance(" in text
    assert "function currentEventColumnWarning()" in text
    assert "Turn on Show all columns to select non-standard event fields." in text
    assert "looks like TCGA-style 1/2 coding" in text
    assert "looks more like a baseline characteristic than a time-to-event indicator" in text
    assert "does not look like a binary event indicator" in text
    assert 'If this is intentional, turn on Show all columns for Event first.' in text
    assert "Choose event value" in text
    assert 'Choose the Event Value that means the event happened' in text
    assert "function currentGroupColumnWarning()" in text
    assert "high-cardinality numeric column" in text


def test_frontend_disables_ml_learning_rate_for_rsf() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function updateMlModelControlVisibility()" in text
    assert 'const treeCountApplies = selectedModelType === "rsf" || selectedModelType === "gbs";' in text
    assert 'const learningRateApplies = selectedModelType === "gbs";' in text
    assert "Learning rate applies to Gradient Boosted Survival only." in text
    assert "Tree count applies to Random Survival Forest and Gradient Boosted Survival only." in text
    assert "SHAP is currently available for Random Survival Forest and Gradient Boosted Survival only." in text
    assert 'refs.mlModelType.addEventListener("change", () => { runtime.resultPreference.ml = "single"; updateMlModelControlVisibility(); queueHistorySync(); });' in text


def test_frontend_formats_validation_errors_and_guards_dl_epoch_range() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function extractErrorMessage(payload)" in text
    assert 'item.loc.filter((part) => part !== "body").join(" > ")' in text
    assert "Epochs must be between 10 and 1000." in text
    assert "Hidden layers must be a comma-separated list of positive integers." in text
    assert "Batch size must be between 8 and 512." in text
    assert "Random seed must be an integer." in text
    assert "CV folds must be between 2 and 10." in text
    assert "Parallel jobs must be between 1 and 16." in text
    assert "Transformer width must be divisible by attention heads." in text
    assert "repeated CV (incomplete; fallback folds excluded)" in text
    assert "holdout requested, reported as apparent fallback" in text
    assert "Latent dim must be between 2 and 32." in text
    assert "validateDlControls();" in text
    assert "evaluation_strategy: refs.dlEvaluationStrategy.value" in text
    assert "cv_folds: Number(refs.dlCvFolds.value)" in text
    assert "cv_repeats: Number(refs.dlCvRepeats.value)" in text
    assert "batch_size: Number(refs.dlBatchSize.value)" in text
    assert "random_seed: Number(refs.dlRandomSeed.value)" in text
    assert "num_time_bins: Number(refs.dlNumTimeBins.value)" in text
    assert "d_model: Number(refs.dlDModel.value)" in text
    assert "n_heads: Number(refs.dlHeads.value)" in text
    assert "n_layers: Number(refs.dlLayers.value)" in text
    assert "latent_dim: Number(refs.dlLatentDim.value)" in text
    assert "n_clusters: Number(refs.dlClusters.value)" in text
    assert "rerun seed=" in text
    assert "rerun a single architecture with Train a model while keeping repeated CV selected" in text
    assert "seed=" in text


def test_frontend_explains_long_ml_runtime_before_fetch() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function mlPendingBannerText(" in text
    assert "This can take longer on a local CPU for real cohorts." in text
    assert "SHAP is computed after fitting and can add a short delay." in text
    assert "Fast mode is on, so SHAP will be skipped for a faster result." in text
    assert "SHAP is currently available for tree models only." in text
    assert "refs.mlMetaBanner.textContent = mlPendingBannerText(" in text
    assert "compute_shap: computeShap" in text
    assert "SHAP skipped in Fast mode" in text
    assert "SHAP failed:" in text
    assert "SHAP is currently available for tree models only" in text
    assert 'const shapStatus = !mlModelSupportsShap(selectedModelType)' in text
    assert '? "approx-screening"' in text
    assert "SHAP=${shapStatus}${shapApproximationNote}, time=${elapsedSeconds}s" in text
    assert "Screening Cox PH, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival" in text
    assert "Screening Cox PH, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival on one shared evaluation path." in text
    assert "Screening top model=" in text
    assert "Model comparison screening complete" in text


def test_frontend_labels_incomplete_repeated_cv_compare_as_mean_c_index() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'const repeatedCvLike = evaluationMode === "repeated_cv" || evaluationMode === "repeated_cv_incomplete";' in text
    assert 'const mlMetricLabel = repeatedCvLike ? "Mean C-index" : "C-index";' in text
    assert 'repeated CV (incomplete)' in text


def test_frontend_warns_that_large_full_batch_dl_runs_can_hit_memory_limits() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function dlPendingBannerText(" in text
    assert 'This full-batch objective can run out of memory on larger cohorts, so start smaller if local RAM is limited.' in text


def test_frontend_recovers_from_missing_dataset_and_blocks_ml_single_model_repeated_cv() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'if (response.status === 404 && /Unknown dataset id:/i.test(message) && state.dataset) {' in text
    assert 'goHome({ syncHistory: true, historyMode: "replace" });' in text
    assert 'The loaded dataset is no longer available on the server. Reload a dataset and run the analysis again.' in text
    assert 'const mlSingleDisabled = !endpointReady || !hasSharedFeatures || mlRepeatedCv || isScopeBusy("ml");' in text
    assert 'setActionDisabledState(refs.runMlButton, mlSingleDisabled, mlSingleTitle);' in text
    assert 'Train a model uses deterministic holdout only. Use Compare All for repeated CV screening.' in text
    assert 'if ((refs.mlEvaluationStrategy?.value || "holdout") === "repeated_cv") {' in text
    assert 'Train a model uses deterministic holdout only. Switch Evaluation Mode back to Deterministic Holdout or use Compare All for repeated CV screening.' in text


def test_plot_config_removes_box_and_lasso_select_tools() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function plotConfig(filename)" in text
    assert 'function isReadonlyPlot(filename) {' in text
    assert 'return ["dl_loss", "ml_importance", "shap_importance", "dl_importance"].includes(filename);' in text
    assert "function plotLayoutConfig(layout, filename) {" in text
    assert "nextLayout.dragmode = false;" in text
    assert 'const isStaticReadonlyPlot = isReadonlyPlot(filename);' in text
    assert "scrollZoom: !isStaticReadonlyPlot," in text
    assert 'doubleClick: isStaticReadonlyPlot ? false : "reset+autosize",' in text
    assert 'modeBarButtonsToRemove: isStaticReadonlyPlot' in text
    assert ': ["select2d", "lasso2d"],' in text
    assert '"resetScale2d"' in text
    assert '"autoScale2d"' in text


def test_frontend_guards_identical_outcome_columns_and_preserves_boundary_precision() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'if (timeColumn === eventColumn) {' in text
    assert 'throw new Error("The survival time column and event column must be different.");' in text
    assert 'if (absValue > 0 && absValue < 0.1) return value.toFixed(4).replace(/\\.?0+$/, "");' in text
    assert 'AIC=${formatValue(stats.aic, { scientificLarge: false })}' in text


def test_cox_plot_reset_axes_restores_initial_layout() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function stabilizePlotShellHeight(plotEl) {" in text
    assert 'plotEl.style.height = `${Math.ceil(height)}px`;' in text
    assert "stabilizePlotShellHeight(plot);" in text
    assert "stabilizePlotShellHeight(refs.kmPlot);" in text
    assert "stabilizePlotShellHeight(refs.mlImportancePlot);" in text
    assert "stabilizePlotShellHeight(refs.dlComparisonPlot);" in text
    assert "function stabilizeCoxPlotResetAxes(plotEl) {" in text
    assert 'plotEl.__stableResetAxesState = {' in text
    assert 'const resetRequested = Boolean(eventData?.["xaxis.autorange"] || eventData?.["yaxis.autorange"]);' in text
    assert '"xaxis.range": stableState.xRange.slice(),' in text
    assert '"yaxis.range": stableState.yRange.slice(),' in text
    assert 'height: stableState.height,' in text
    assert 'stabilizeCoxPlotResetAxes(refs.coxPlot);' in text


def test_ml_current_result_ignores_compare_only_and_explanation_only_controls() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'const expectsCompare = preferredResultMode("ml") === "compare";' in text
    assert 'const selectedModelType = String(refs.mlModelType?.value || "");' in text
    assert '(expectsCompare ? compareRun : String(requestConfig.model_type || "") === selectedModelType)' in text
    assert 'const learningRateApplies = compareRun || selectedModelType === "gbs";' in text
    assert '&& (!expectsCompare || String(requestConfig.evaluation_strategy || "holdout") === String(refs.mlEvaluationStrategy?.value || "holdout"))' in text
    assert '&& (!expectsCompare || Number(requestConfig.cv_folds || 5) === Number(refs.mlCvFolds?.value || 5))' in text
    assert '&& (!expectsCompare || Number(requestConfig.cv_repeats || 3) === Number(refs.mlCvRepeats?.value || 3))' in text
    assert '&& (compareRun || Boolean(requestConfig.compute_shap) === !Boolean(refs.mlSkipShap?.checked))' not in text


def test_guided_ml_inline_compare_uses_clicked_button_as_loading_target() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'void runGuidedGoal("ml", refs.runCompareInlineButton, runCompareModels);' in text


def test_guided_dl_inline_compare_uses_clicked_button_as_loading_target() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'void runGuidedGoal("dl", refs.runDlCompareInlineButton, runDlCompareModels);' in text


def test_review_shared_features_buttons_keep_the_user_on_matching_model_tab() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'function focusModelFeatureEditor(tabName = "ml") {' in text
    assert 'const featureChecklist = tabName === "dl" ? refs.dlModelFeatureChecklist : refs.modelFeatureChecklist;' in text
    assert 'refs.reviewMlFeaturesButton?.addEventListener("click", () => focusModelFeatureEditor("ml"));' in text
    assert 'refs.reviewDlFeaturesButton?.addEventListener("click", () => focusModelFeatureEditor("dl"));' in text


def test_shared_feature_controls_lock_while_ml_or_dl_scope_is_busy() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function setChecklistDisabled(container, isDisabled) {" in text
    assert 'const isBusy = isScopeBusy("ml") || isScopeBusy("dl");' in text
    assert "refs.reviewMlFeaturesButton," in text
    assert "refs.reviewDlFeaturesButton," in text
    assert "refs.selectAllModelFeaturesButton," in text
    assert "refs.clearModelFeaturesButton," in text
    assert "setChecklistDisabled(refs.modelFeatureChecklist, isBusy);" in text
    assert "setChecklistDisabled(refs.dlModelFeatureChecklist, isBusy);" in text
    assert "syncSharedFeatureControlsBusy();" in text


def test_guided_rail_status_tracks_running_ready_and_stale_states() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function guidedRailStatusState() {" in text
    assert 'label: "Running"' in text
    assert 'label: "Ready"' in text
    assert 'label: "Needs rerun"' in text
    assert 'label: "No result yet"' in text
    assert "function renderGuidedRailStatus() {" in text
    assert 'refs.guidedRailStatus.className = `guided-rail-status guided-rail-status-${compactStatus.tone}${showReviewActions ? " guided-rail-status-actionable" : ""}`;' in text
    assert 'guided-rail-status-actionable' in text
    assert "renderGuidedRailStatus();" in text


def test_refresh_cox_preview_does_not_rerender_guided_chrome_during_loading_state() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()
    refresh_body = text.split("async function refreshCoxPreview({ force = false } = {}) {", 1)[1].split("function scheduleCoxPreview(", 1)[0]

    assert 'status: "loading"' in refresh_body
    assert 'syncGuidedCoxPanelMounts();' in refresh_body
    assert 'if (!syncGuidedCoxPanelMounts()) renderGuidedChrome();' in refresh_body


def test_cox_checklist_search_select_all_is_limited_to_visible_rows() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function allCheckboxValues(container, { visibleOnly = false } = {}) {" in text
    assert '!input.closest(".check-item")?.classList.contains("hidden-by-filter")' in text
    assert 'allCheckboxValues(refs.covariateChecklist, { visibleOnly: true })' in text
    assert 'allCheckboxValues(refs.categoricalChecklist, { visibleOnly: true })' in text


def test_dataset_refresh_clears_cox_search_filters() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    update_body = text.split("function updateControlsFromDataset({ scrollToTop = false } = {}) {", 1)[1].split("function updateAfterDataset(", 1)[0]
    assert 'refs.covariateSearchInput.value = "";' in update_body
    assert 'refs.categoricalSearchInput.value = "";' in update_body


def test_legacy_derive_restore_only_reuses_cutoff_when_method_is_still_available() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "const restoredDeriveMethod = setSelectValueIfPresent(refs.deriveMethod, snapshot.deriveMethod);" in text
    assert 'refs.deriveCutoff.value = "";' in text


def test_dl_guided_review_hides_compare_tables_when_single_mode_is_active() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'const hasCompareTable = resultMode === "compare" && hasRenderedTable(refs.dlComparisonShell);' in text
    assert 'const hasManuscript = resultMode === "compare" && hasRenderedTable(refs.dlManuscriptShell);' in text


def test_guided_cox_preview_summary_surfaces_parameter_count_and_epv() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function renderGuidedCoxPreviewSummary() {" in text
    assert "<strong>Parameters</strong>" in text
    assert "<strong>EPV</strong>" in text


def test_expert_surface_status_tracks_running_current_and_stale_states() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function goalResultStatusState(goal" in text
    assert "function expertSurfaceStatusState() {" in text
    assert 'currentLabel: "Current"' in text
    assert "function renderExpertSurfaceStatus() {" in text
    assert 'refs.expertSurfaceLabel.className = `expert-surface-label expert-surface-label-${status.tone}`;' in text
    assert "Visible settings no longer match the current result. Run again before exporting or interpreting it." in text


def test_mode_switch_busy_guard_and_tab_focus_scroll_protection() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'Object.values(runtime.busyScopes || {}).some(Boolean)' in text
    assert "Wait for the current analysis run to finish before switching between Guided and Expert mode." in text
    assert 'function activateTab(tabName, { setGuidedGoal = runtime.uiMode === "guided", historyMode = "replace", focusTabButton = false } = {}) {' in text
    assert 'if (isActive && runtime.uiMode !== "guided" && focusTabButton)' in text
    assert 'activateTab(tabs[next].dataset.tab, { historyMode: "push", focusTabButton: true });' in text


def test_reparenting_preserves_focus_scroll_and_schedules_extra_plot_resize() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function captureReparentUiState() {" in text
    assert "function restoreReparentUiState(snapshot) {" in text
    assert "const preservedUiState = captureReparentUiState();" in text
    assert "restoreReparentUiState(preservedUiState);" in text
    assert "if (didMove) scheduleVisiblePlotResize(40);" in text
    assert "window.setTimeout(resizeVisiblePlotsNow, 260);" in text


def test_ml_dl_result_reveal_is_conditional_on_current_view() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function shouldRevealCompletedResult(goal) {" in text
    assert "function revealCompletedResultIfCurrent(goal" in text
    assert "if (hasResult && shouldRevealCompletedResult(tabName)) {" in text
    assert 'revealCompletedResultIfCurrent("ml", {' in text
    assert 'revealCompletedResultIfCurrent("dl", {' in text


def test_frontend_exposes_shutdown_button_and_stop_flow_copy() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'shutdownButton: document.getElementById("shutdownButton")' in text
    assert "async function shutdownServer() {" in text
    assert "Stopping the local SurvStudio server. This will cancel active runs and release memory." in text
    assert "SurvStudio server stopped" in text
    assert 'await fetchJSON("/api/shutdown", { method: "POST" });' in text
    assert 'refs.shutdownButton?.addEventListener("click", () => {' in text


def test_shutdown_endpoint_schedules_local_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}

    def _fake_schedule(delay_seconds: float = 0.35) -> None:
        called["count"] += 1

    monkeypatch.setattr(app_module, "_schedule_process_shutdown", _fake_schedule)

    response = client.post("/api/shutdown")

    assert response.status_code == 200
    assert response.json()["status"] == "shutting_down"
    assert "restart the server" in response.json()["detail"]
    assert called["count"] == 1


def test_guided_grouping_context_only_uses_guided_goal_inside_guided_mode() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert '|| (runtime.uiMode === "guided" && (runtime.guidedGoal === "km" || runtime.guidedGoal === "tables"))' in text
    assert 'const guidedKmRefresh = runtime.uiMode === "guided" && runtime.guidedGoal === "km";' in text


def test_change_analysis_clears_guided_goal_before_pushing_history() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert 'runtime.guidedGoal = null;' in text
    assert 'activateTab("data", { setGuidedGoal: false, historyMode: "push" });' in text
    assert 'activateTab("data", { setGuidedGoal: false, historyMode: "replace" });' not in text


def test_ml_model_fast_mode_skips_shap_computation(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    stored = store.create(
        make_example_dataset(seed=77, n_patients=64),
        filename="ml_fast_mode.csv",
        copy_dataframe=False,
    )

    monkeypatch.setattr(
        ml_models,
        "train_random_survival_forest",
        lambda *args, **kwargs: {
            "feature_importance": [{"feature": "age", "importance": 1.0}],
            "feature_names": ["age"],
            "model_stats": {
                "c_index": 0.71,
                "evaluation_mode": "holdout",
                "n_patients": 64,
                "n_features": 1,
            },
            "_model": object(),
            "_X_encoded": pd.DataFrame({"age": [51.0, 63.0, 72.0]}),
        },
    )

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("compute_shap_values should not run when Fast mode is enabled.")

    monkeypatch.setattr(ml_models, "compute_shap_values", _raise_if_called)

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "model_type": "rsf",
            "n_estimators": 20,
            "compute_shap": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_config"]["compute_shap"] is False
    assert body["shap_result"] is None
    assert body["shap_figure"] is None


def test_ml_model_surfaces_shap_failures_without_failing_entire_request(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    stored = store.create(
        make_example_dataset(seed=177, n_patients=64),
        filename="ml_shap_failure.csv",
        copy_dataframe=False,
    )

    monkeypatch.setattr(
        ml_models,
        "train_random_survival_forest",
        lambda *args, **kwargs: {
            "feature_importance": [{"feature": "age", "importance": 1.0}],
            "feature_names": ["age"],
            "model_stats": {
                "c_index": 0.71,
                "evaluation_mode": "holdout",
                "n_patients": 64,
                "n_features": 1,
            },
            "_model": object(),
            "_X_encoded": pd.DataFrame({"age": [51.0, 63.0, 72.0]}),
        },
    )
    monkeypatch.setattr(
        ml_models,
        "compute_shap_values",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kernel failed")),
    )

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "model_type": "rsf",
            "n_estimators": 20,
            "compute_shap": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["shap_result"] is None
    assert body["shap_figure"] is None
    assert body["shap_error"] == "RuntimeError: kernel failed"


def test_ml_model_supports_lasso_cox_without_tree_shap(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    stored = store.create(
        make_example_dataset(seed=188, n_patients=64),
        filename="ml_lasso.csv",
        copy_dataframe=False,
    )

    monkeypatch.setattr(
        ml_models,
        "train_lasso_cox",
        lambda *args, **kwargs: {
            "feature_importance": [{"feature": "age", "importance": 0.8, "coefficient": 0.8}],
            "feature_names": ["age"],
            "model_stats": {
                "c_index": 0.69,
                "evaluation_mode": "holdout",
                "n_patients": 64,
                "n_features": 1,
                "n_active_features": 1,
                "alpha": 0.01,
            },
        },
    )

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "model_type": "lasso_cox",
            "compute_shap": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["model_stats"]["alpha"] == pytest.approx(0.01)
    assert body["shap_result"] is None
    assert body["shap_figure"] is None
    assert body["shap_error"] == "SHAP is currently available for tree models only (RSF or GBS)."


def test_deep_model_request_accepts_1000_epochs() -> None:
    request_model = DeepModelRequest(
        dataset_id="demo",
        time_column="os_months",
        event_column="os_event",
        features=["age", "stage"],
        categorical_features=["stage"],
        model_type="deepsurv",
        hidden_layers=[64, 64],
        epochs=1000,
    )

    assert request_model.epochs == 1000


def test_health_allows_null_origin_for_file_preview() -> None:
    response = client.get("/api/health", headers={"Origin": "null"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "null"


def test_upload_dataset_preserves_too_large_status(monkeypatch) -> None:
    import survival_toolkit.app as app_module

    monkeypatch.setattr(app_module, "_MAX_UPLOAD_BYTES", 1)
    response = client.post(
        "/api/upload",
        files={"file": ("big.csv", b"12", "text/csv")},
    )

    assert response.status_code == 413
    assert "200 MB limit" in response.json()["detail"]


def test_upload_dataset_rejects_too_many_rows(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_MAX_UPLOAD_ROWS", 1)

    response = client.post(
        "/api/upload",
        files={"file": ("rows.csv", b"os_months,os_event,age\n12,1,60\n18,0,55\n", "text/csv")},
    )

    assert response.status_code == 400
    assert "supports at most 1 rows" in response.json()["detail"]


def test_upload_dataset_rejects_too_many_cells(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_MAX_UPLOAD_CELLS", 4)

    response = client.post(
        "/api/upload",
        files={"file": ("cells.csv", b"os_months,os_event,age\n12,1,60\n18,0,55\n", "text/csv")},
    )

    assert response.status_code == 400
    assert "supports at most 4 parsed cells" in response.json()["detail"]


def test_upload_dataset_accepts_tsv_files() -> None:
    payload = b"os_months\tos_event\tage\n12\t1\t60\n18\t0\t55\n24\t1\t70\n"
    response = client.post(
        "/api/upload",
        files={"file": ("cohort.tsv", payload, "text/tab-separated-values")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "cohort.tsv"
    assert body["n_rows"] == 3
    assert {column["name"] for column in body["columns"]} >= {"os_months", "os_event", "age"}


def test_upload_dataset_accepts_exactly_1000_model_feature_candidates() -> None:
    feature_names = [f"gene_{idx}" for idx in range(1000)]
    header = ",".join(["os_months", "os_event", *feature_names])
    rows = [
        ",".join(["12", "1", *(str(idx) for idx in range(1000))]),
        ",".join(["18", "0", *(str(idx + 1) for idx in range(1000))]),
    ]
    payload = (header + "\n" + "\n".join(rows) + "\n").encode("utf-8")

    response = client.post(
        "/api/upload",
        files={"file": ("wide_1000.csv", payload, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_feature_candidate_count"] == 1000


def test_upload_dataset_rejects_more_than_1000_model_feature_candidates() -> None:
    feature_names = [f"gene_{idx}" for idx in range(1001)]
    header = ",".join(["os_months", "os_event", *feature_names])
    rows = [
        ",".join(["12", "1", *(str(idx) for idx in range(1001))]),
        ",".join(["18", "0", *(str(idx + 1) for idx in range(1001))]),
    ]
    payload = (header + "\n" + "\n".join(rows) + "\n").encode("utf-8")

    response = client.post(
        "/api/upload",
        files={"file": ("wide_1001.csv", payload, "text/csv")},
    )

    assert response.status_code == 400
    assert "supports at most 1000 model features" in response.json()["detail"]


def test_get_ml_artifact_returns_isolated_copy() -> None:
    import pandas as pd

    request_config = {
        "model_type": "rsf",
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "features": ["age"],
        "categorical_features": [],
        "n_estimators": 20,
        "max_depth": None,
        "random_state": 42,
    }
    app_module._remember_ml_artifact(
        "dataset-cache",
        request_config,
        {
            "_model": {"state": ["original"]},
            "_X_encoded": pd.DataFrame({"age": [10.0, 20.0]}),
            "_feature_encoder": {"features": ["age"]},
            "_analysis_frame": pd.DataFrame({"age": [10.0, 20.0]}),
        },
    )

    first = app_module._get_ml_artifact("dataset-cache", request_config)
    second = app_module._get_ml_artifact("dataset-cache", request_config)

    assert first is not None
    assert second is not None
    first["_model"]["state"].append("mutated")
    first["_X_encoded"].iloc[0, 0] = -999.0
    assert second["_model"]["state"] == ["original"]
    assert second["_X_encoded"].iloc[0, 0] == pytest.approx(10.0)


def test_readme_highlights_synthetic_columns_cli_inspect_and_dl_runtime_note() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text()

    assert "Synthetic Example Workflow" in text
    assert "This synthetic dataset does **not** use `stage_group` or `treatment_group`." in text
    assert "survival-toolkit inspect path/to/data.csv" in text
    assert "This is the fastest way to catch file-format problems before uploading a cohort in the browser." in text
    assert "## DL Runtime Note" in text
    assert "Batch Size` currently affects DeepHit and Neural MTLR only." in text
    assert "weight_decay=1e-4" in text
    assert "DeepHit uses a stabilized ranking-loss scale (`sigma=1.0`)" in text
    assert "MTLR-inspired discrete-time neural variant" in text
    assert "SurvStudio does not claim validated generative simulation or uncertainty estimation from this path." in text


def test_kaplan_meier_response_exposes_groups_and_logrank_p_summary_fields() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "stage",
            "confidence_level": 0.95,
            "max_time": None,
            "risk_table_points": 6,
            "show_confidence_bands": True,
            "logrank_weight": "logrank",
            "fh_p": 1.0,
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["groups"] == 4
    assert analysis["logrank_p"] is not None


def test_kaplan_meier_response_includes_request_config() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "stage",
            "time_unit_label": "Months",
            "confidence_level": 0.95,
            "risk_table_points": 6,
            "show_confidence_bands": True,
            "logrank_weight": "logrank",
            "fh_p": 1.0,
        },
    )

    assert response.status_code == 200
    request_config = response.json()["request_config"]
    assert request_config["dataset_id"] == dataset["dataset_id"]
    assert request_config["group_column"] == "stage"
    assert request_config["risk_table_points"] == 6
    assert request_config["show_confidence_bands"] is True


def test_ml_compare_response_includes_request_config() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "n_estimators": 20,
            "learning_rate": 0.05,
            "evaluation_strategy": "holdout",
        },
    )

    assert response.status_code == 200
    request_config = response.json()["request_config"]
    assert request_config["dataset_id"] == dataset["dataset_id"]
    assert request_config["time_column"] == "os_months"
    assert request_config["features"] == ["age", "sex", "stage", "biomarker_score"]
    assert request_config["evaluation_strategy"] == "holdout"


def test_cox_response_includes_request_config() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "sex", "stage"],
            "categorical_covariates": ["sex", "stage"],
        },
    )

    assert response.status_code == 200
    request_config = response.json()["request_config"]
    assert request_config["covariates"] == ["age", "sex", "stage"]
    assert request_config["categorical_covariates"] == ["sex", "stage"]


def test_cohort_table_response_includes_request_config() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/cohort-table",
        json={
            "dataset_id": dataset["dataset_id"],
            "variables": ["age", "sex", "stage"],
            "group_column": "stage",
        },
    )

    assert response.status_code == 200
    request_config = response.json()["request_config"]
    assert request_config["variables"] == ["age", "sex", "stage"]
    assert request_config["group_column"] == "stage"


def test_ml_compare_response_appends_replay_notes_to_manuscript_tables() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
            "n_estimators": 10,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )

    assert response.status_code == 200
    notes = response.json()["analysis"]["manuscript_tables"]["table_notes"]
    assert any(note.startswith("Replay dataset: ") for note in notes)
    assert any(note.startswith("Replay settings: ") for note in notes)
    assert any(note.startswith("Replay features: ") for note in notes)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_compare_response_appends_replay_notes_to_manuscript_tables() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 31,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
            "evaluation_strategy": "holdout",
        },
    )

    assert response.status_code == 200
    notes = response.json()["analysis"]["manuscript_tables"]["table_notes"]
    assert any(note.startswith("Replay dataset: ") for note in notes)
    assert any(note.startswith("Replay settings: ") for note in notes)
    assert any(note.startswith("Replay features: ") for note in notes)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_model_response_includes_request_config_and_seed_metadata() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "mtlr",
            "hidden_layers": [16],
            "num_time_bins": 10,
            "epochs": 10,
            "batch_size": 16,
            "random_seed": 77,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_config"]["random_seed"] == 77
    assert body["request_config"]["model_type"] == "mtlr"
    assert body["analysis"]["training_seed"] == 77
    assert body["analysis"]["split_seed"] == 77
    assert body["analysis"]["monitor_seed"] == 77


def test_endpoints_accept_boundary_configuration_values() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": None,
            "confidence_level": 0.998,
            "max_time": 36,
            "risk_table_points": 12,
            "show_confidence_bands": True,
            "logrank_weight": "fleming_harrington",
            "fh_p": 5.0,
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["summary_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "n_estimators": 10,
            "learning_rate": 0.002,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["comparison_table"]

    deep_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "hidden_layers": [16],
            "dropout": 0.0,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 11,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "parallel_jobs": 1,
            "num_time_bins": 10,
            "n_heads": 1,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 2,
            "n_clusters": 2,
        },
    )
    assert deep_response.status_code == 200
    assert deep_response.json()["analysis"]["comparison_table"]


@pytest.mark.parametrize(
    ("path", "payload_overrides"),
    [
        ("/api/kaplan-meier", {"confidence_level": 0.5}),
        ("/api/ml-model", {"model_type": "rsf", "features": ["age"], "n_estimators": 9}),
        ("/api/deep-model", {"model_type": "deepsurv", "features": ["age"], "batch_size": 7}),
        ("/api/deep-model", {"model_type": "compare", "features": ["age"], "parallel_jobs": 17}),
    ],
)
def test_endpoints_reject_out_of_bounds_values(path: str, payload_overrides: dict[str, object]) -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    base_payload: dict[str, object] = {
        "dataset_id": dataset["dataset_id"],
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "features": ["age", "sex", "stage", "biomarker_score"],
        "categorical_features": ["sex", "stage"],
        "covariates": ["age", "sex", "stage", "biomarker_score"],
        "categorical_covariates": ["sex", "stage"],
        "hidden_layers": [16],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 8,
        "random_seed": 11,
        "evaluation_strategy": "holdout",
        "cv_folds": 2,
        "cv_repeats": 1,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,
        "parallel_jobs": 1,
        "num_time_bins": 10,
        "n_heads": 1,
        "d_model": 16,
        "n_layers": 1,
        "latent_dim": 2,
        "n_clusters": 2,
        "confidence_level": 0.95,
        "risk_table_points": 6,
        "show_confidence_bands": True,
        "logrank_weight": "logrank",
        "fh_p": 1.0,
        "model_type": "compare",
        "n_estimators": 10,
    }
    base_payload.update(payload_overrides)

    response = client.post(path, json=base_payload)

    assert response.status_code == 422


def test_missing_dataset_returns_404() -> None:
    response = client.get("/api/dataset/does-not-exist")

    assert response.status_code == 404
    assert "Unknown dataset id" in response.json()["detail"]


def test_fail_bad_request_reraises_server_errors() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        fail_bad_request(RuntimeError("boom"))


def test_fail_bad_request_maps_dependency_errors_to_503() -> None:
    with pytest.raises(HTTPException) as excinfo:
        fail_bad_request(ImportError("scikit-survival is required"))

    assert excinfo.value.status_code == 503
    assert "scikit-survival" in excinfo.value.detail


def test_ml_model_endpoint_reports_missing_dependency(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    def _raise_missing_dependency(*args, **kwargs):
        raise ImportError("scikit-survival is required for Random Survival Forest.")

    monkeypatch.setattr(ml_models, "train_random_survival_forest", _raise_missing_dependency)

    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "rsf",
        },
    )

    assert response.status_code == 503
    assert "scikit-survival" in response.json()["detail"]


def test_deep_model_endpoint_reports_missing_torch_dependency(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    def _raise_missing_dependency(*args, **kwargs):
        raise ImportError("PyTorch is required for deep learning models.")

    monkeypatch.setattr(deep_models, "evaluate_single_deep_survival_model", _raise_missing_dependency)

    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "deepsurv",
            "hidden_layers": [8],
        },
    )

    assert response.status_code == 503
    assert "PyTorch" in response.json()["detail"]


def test_discover_signature_endpoint_persists_derived_group() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "stage", "treatment", "biomarker_score", "immune_index"],
            "max_combination_size": 3,
            "top_k": 10,
            "min_group_fraction": 0.1,
            "bootstrap_iterations": 8,
            "bootstrap_sample_fraction": 0.8,
            "permutation_iterations": 16,
            "validation_iterations": 5,
            "validation_fraction": 0.35,
            "significance_level": 0.05,
            "combination_operator": "mixed",
            "random_seed": 777,
            "new_column_name": "auto_sig",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["derived_column"] == "auto_sig"
    assert payload["signature_analysis"]["results_table"]
    assert payload["signature_analysis"]["search_space"]["bootstrap_iterations"] == 8
    assert payload["signature_analysis"]["search_space"]["permutation_iterations"] == 16
    assert payload["signature_analysis"]["search_space"]["validation_iterations"] == 5
    assert payload["signature_analysis"]["search_space"]["validation_fraction"] == 0.35
    assert payload["signature_analysis"]["search_space"]["significance_level"] == 0.05
    assert payload["signature_analysis"]["search_space"]["combination_operator"] == "mixed"
    assert payload["signature_analysis"]["search_space"]["random_seed"] == 777
    assert payload["signature_analysis"]["scientific_summary"]["headline"]
    assert payload["signature_analysis"]["scientific_summary"]["status"] in {"robust", "review", "caution"}
    assert any(column["name"] == "auto_sig" for column in payload["columns"])


def test_discover_signature_returns_new_dataset_snapshot_and_preserves_original_dataset() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()
    original_dataset_id = dataset["dataset_id"]

    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": original_dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "sex", "stage"],
            "max_combination_size": 2,
            "top_k": 5,
            "bootstrap_iterations": 1,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "new_column_name": "sig_age_stage",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] != original_dataset_id
    assert any(column["name"] == "sig_age_stage" for column in payload["columns"])

    original = client.get(f"/api/dataset/{original_dataset_id}")
    assert original.status_code == 200
    assert all(column["name"] != "sig_age_stage" for column in original.json()["columns"])


@pytest.mark.parametrize("operator", ["and", "or", "mixed"])
def test_discover_signature_endpoint_supports_all_combination_operators(operator: str) -> None:
    stored = store.create(
        make_example_dataset(seed=73, n_patients=84),
        filename=f"sig_{operator}.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "stage", "treatment", "biomarker_score"],
            "max_combination_size": 2,
            "top_k": 6,
            "min_group_fraction": 0.1,
            "bootstrap_iterations": 0,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "combination_operator": operator,
            "new_column_name": f"sig_{operator}",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["derived_column"] == f"sig_{operator}"
    assert payload["signature_analysis"]["search_space"]["combination_operator"] == operator
    assert payload["signature_analysis"]["results_table"]


def test_optimal_cutpoint_endpoint_reports_adjusted_p_value() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/optimal-cutpoint",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "variable": "biomarker_score",
            "min_group_fraction": 0.1,
            "permutation_iterations": 20,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert result["selection_adjusted_p_value"] is not None
    assert result["raw_p_value"] is not None
    assert result["p_value"] == result["selection_adjusted_p_value"]
    assert result["p_value_label"] == "selection_adjusted_p_value"
    annotations = payload["figure"]["layout"].get("annotations", [])
    assert any("Adj. p" in (a.get("text", "") or "") for a in annotations)


def test_optimal_cutpoint_derive_group_returns_inline_scan_figure() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/derive-group",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_column": "biomarker_score",
            "method": "optimal_cutpoint",
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cutpoint_figure"]["data"][0]["type"] == "scatter"
    assert payload["derive_summary"]["scan_data"]
    assert payload["derive_summary"]["assignment_rule"]
    assert payload["derive_summary"]["p_value_label"] in {"selection_adjusted_p_value", "raw_p_value"}


def test_derive_group_returns_new_dataset_snapshot_and_preserves_original_dataset() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()
    original_dataset_id = dataset["dataset_id"]

    response = client.post(
        "/api/derive-group",
        json={
            "dataset_id": original_dataset_id,
            "source_column": "age",
            "method": "median_split",
            "new_column_name": "age_bin",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] != original_dataset_id
    assert any(column["name"] == "age_bin" for column in payload["columns"])

    original = client.get(f"/api/dataset/{original_dataset_id}")
    assert original.status_code == 200
    assert all(column["name"] != "age_bin" for column in original.json()["columns"])


def test_time_dependent_importance_endpoint_uses_time_major_matrix() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/time-dependent-importance",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "eval_times": [12.0, 24.0, 36.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    analysis = payload["analysis"]
    assert analysis["importance_matrix_orientation"] == "time_major"
    assert len(analysis["importance_matrix"]) == len(analysis["eval_times"])
    assert len(analysis["importance_matrix"][0]) == len(analysis["features"])
    assert len(analysis["importance_matrix_feature_major"]) == len(analysis["features"])
    assert payload["figure"]["data"][0]["type"] == "heatmap"


def test_modeling_and_xai_endpoints_reject_outcome_informed_columns() -> None:
    stored = store.create(
        make_example_dataset(seed=171, n_patients=72),
        filename="outcome_informed_guard.csv",
        copy_dataframe=False,
    )
    store.update_metadata(
        stored.dataset_id,
        {"derived_column_provenance": {"age__optimal_cutpoint": {"outcome_informed": True}}},
    )

    common = {
        "dataset_id": stored.dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "features": ["age__optimal_cutpoint"],
        "categorical_features": [],
    }

    ml_response = client.post("/api/ml-model", json={**common, "model_type": "rsf"})
    deep_response = client.post("/api/deep-model", json={**common, "model_type": "deepsurv"})
    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age__optimal_cutpoint"],
            "categorical_covariates": [],
        },
    )
    pdp_response = client.post("/api/pdp", json={**common, "target_feature": "age__optimal_cutpoint"})
    counterfactual_response = client.post(
        "/api/counterfactual",
        json={**common, "target_feature": "age__optimal_cutpoint", "counterfactual_value": "High"},
    )

    for response in [ml_response, deep_response, cox_response, pdp_response, counterfactual_response]:
        assert response.status_code == 400
        assert "Outcome-informed derived columns" in response.json()["detail"]


def test_modeling_and_xai_endpoints_reject_survival_outcome_columns_as_features() -> None:
    stored = store.create(
        make_example_dataset(seed=271, n_patients=96),
        filename="outcome_feature_guard.csv",
        copy_dataframe=False,
    )

    common = {
        "dataset_id": stored.dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "features": ["os_months", "age"],
        "categorical_features": [],
    }

    ml_response = client.post("/api/ml-model", json={**common, "model_type": "rsf"})
    deep_response = client.post("/api/deep-model", json={**common, "model_type": "deepsurv"})
    time_dep_response = client.post("/api/time-dependent-importance", json={**common})
    pdp_response = client.post("/api/pdp", json={**common, "target_feature": "age"})
    counterfactual_response = client.post(
        "/api/counterfactual",
        json={**common, "target_feature": "age", "counterfactual_value": 70},
    )

    for response in [ml_response, deep_response, time_dep_response, pdp_response, counterfactual_response]:
        assert response.status_code == 400
        assert "Survival outcome columns cannot be used" in response.json()["detail"]


def test_xai_endpoints_support_explicit_event_positive_value_and_categorical_counterfactual() -> None:
    df = make_example_dataset(seed=67, n_patients=96)
    df["custom_event"] = df["os_event"].map({1: "R", 0: "N"})
    stored = store.create(df, filename="xai_custom_event.csv", copy_dataframe=False)

    time_dep_response = client.post(
        "/api/time-dependent-importance",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "eval_times": [12.0, 24.0],
        },
    )

    assert time_dep_response.status_code == 200
    time_dep_analysis = time_dep_response.json()["analysis"]
    assert time_dep_analysis["importance_matrix_orientation"] == "time_major"

    pdp_response = client.post(
        "/api/pdp",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "target_feature": "age",
        },
    )

    assert pdp_response.status_code == 200
    assert pdp_response.json()["analysis"]["feature"] == "age"

    counterfactual_response = client.post(
        "/api/counterfactual",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "target_feature": "stage",
            "original_value": "I",
            "counterfactual_value": "III",
            "model_type": "gbs",
            "n_estimators": 20,
            "learning_rate": 0.05,
        },
    )

    assert counterfactual_response.status_code == 200
    counterfactual_analysis = counterfactual_response.json()["analysis"]
    assert counterfactual_analysis["target_feature"] == "stage"
    assert counterfactual_analysis["original_value"] == "I"
    assert counterfactual_analysis["counterfactual_value"] == "III"


def test_pdp_endpoint_supports_categorical_target_feature() -> None:
    stored = store.create(
        make_example_dataset(seed=68, n_patients=72),
        filename="pdp_categorical.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/pdp",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "target_feature": "stage",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["feature"] == "stage"
    assert payload["analysis"]["feature_type"] == "categorical"
    assert len(payload["analysis"]["values"]) >= 2
    assert payload["figure"]["data"][0]["type"] == "bar"


def test_pdp_endpoint_accepts_gbs_model_type(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    class _LinearModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
    stored = store.create(
        make_example_dataset(seed=170, n_patients=48),
        filename="pdp_gbs.csv",
        copy_dataframe=False,
    )

    monkeypatch.setattr(
        ml_models,
        "train_gradient_boosted_survival",
        lambda *args, **kwargs: {
            "_model": _LinearModel(),
            "_X_encoded": pd.DataFrame({"age": analysis_frame["age"]}),
            "_analysis_frame": analysis_frame.copy(),
            "_feature_encoder": None,
        },
    )

    response = client.post(
        "/api/pdp",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "target_feature": "age",
            "model_type": "gbs",
            "n_estimators": 20,
            "learning_rate": 0.05,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["model_type"] == "gbs"
    assert body["request_config"]["model_type"] == "gbs"


def test_pdp_endpoint_reuses_matching_cached_ml_artifact(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    calls = {"train": 0}

    class _LinearModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    def _fake_train(*args, **kwargs):
        calls["train"] += 1
        return {
            "feature_importance": [{"feature": "age", "importance": 1.0}],
            "feature_names": ["age"],
            "model_stats": {"c_index": 0.70, "evaluation_mode": "holdout", "n_patients": 30, "n_features": 1},
            "_model": _LinearModel(),
            "_X_encoded": pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            "_analysis_frame": pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            "_feature_encoder": None,
        }

    monkeypatch.setattr(ml_models, "train_random_survival_forest", _fake_train)

    dataset = client.post("/api/load-example").json()
    train_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "model_type": "rsf",
        },
    )
    assert train_response.status_code == 200

    pdp_response = client.post(
        "/api/pdp",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "target_feature": "age",
            "model_type": "rsf",
        },
    )

    assert pdp_response.status_code == 200
    assert calls["train"] == 1
    assert pdp_response.json()["analysis"]["artifact_reused"] is True
    assert "RSF/GBS model" in pdp_response.json()["analysis"]["contract_note"]
    assert "Compare All ranking table" in pdp_response.json()["analysis"]["contract_note"]


def test_counterfactual_endpoint_uses_original_value_for_baseline(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    class _LinearAgeModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "train_random_survival_forest",
        lambda *args, **kwargs: {
            "_model": _LinearAgeModel(),
            "_X_encoded": pd.DataFrame({"age": analysis_frame["age"]}),
            "_analysis_frame": analysis_frame.copy(),
        },
    )

    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/counterfactual",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age"],
            "categorical_features": [],
            "target_feature": "age",
            "original_value": 5.0,
            "counterfactual_value": 15.0,
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["original_median_risk"] == pytest.approx(5.0)
    assert analysis["counterfactual_median_risk"] == pytest.approx(15.0)
    assert analysis["risk_change_pct"] == pytest.approx(200.0)


def test_counterfactual_endpoint_reuses_matching_cached_ml_artifact(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    calls = {"train": 0}

    class _LinearAgeModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    def _fake_train(*args, **kwargs):
        calls["train"] += 1
        return {
            "feature_importance": [{"feature": "age", "importance": 1.0}],
            "feature_names": ["age"],
            "model_stats": {"c_index": 0.70, "evaluation_mode": "holdout", "n_patients": 30, "n_features": 1},
            "_model": _LinearAgeModel(),
            "_X_encoded": pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            "_analysis_frame": pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            "_feature_encoder": None,
        }

    monkeypatch.setattr(ml_models, "train_random_survival_forest", _fake_train)

    dataset = client.post("/api/load-example").json()
    train_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "model_type": "rsf",
        },
    )
    assert train_response.status_code == 200

    response = client.post(
        "/api/counterfactual",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "target_feature": "age",
            "original_value": 5.0,
            "counterfactual_value": 15.0,
            "model_type": "rsf",
        },
    )

    assert response.status_code == 200
    assert calls["train"] == 1
    assert response.json()["analysis"]["artifact_reused"] is True
    assert "RSF/GBS model" in response.json()["analysis"]["contract_note"]
    assert "Compare All ranking table" in response.json()["analysis"]["contract_note"]


def test_counterfactual_endpoint_accepts_gbs_model_type(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    class _LinearAgeModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
    called: dict[str, bool] = {"gbs": False}

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "train_gradient_boosted_survival",
        lambda *args, **kwargs: called.__setitem__("gbs", True) or {
            "_model": _LinearAgeModel(),
            "_X_encoded": pd.DataFrame({"age": analysis_frame["age"]}),
            "_analysis_frame": analysis_frame.copy(),
        },
    )

    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/counterfactual",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age"],
            "categorical_features": [],
            "target_feature": "age",
            "original_value": 5.0,
            "counterfactual_value": 15.0,
            "model_type": "gbs",
            "n_estimators": 20,
            "learning_rate": 0.05,
        },
    )

    assert response.status_code == 200
    assert called["gbs"] is True


def test_classical_endpoint_option_matrix_runs_without_errors() -> None:
    stored = store.create(
        make_example_dataset(seed=69, n_patients=96),
        filename="classical_matrix.csv",
        copy_dataframe=False,
    )

    derive_payloads = [
        {
            "source_column": "age",
            "method": "median_split",
            "new_column_name": "age_median_group",
        },
        {
            "source_column": "age",
            "method": "tertile_split",
            "new_column_name": "age_tertile_group",
        },
        {
            "source_column": "biomarker_score",
            "method": "quartile_split",
            "new_column_name": "biomarker_quartile_group",
        },
        {
            "source_column": "immune_index",
            "method": "percentile_split",
            "new_column_name": "immune_percentile_group",
            "cutoff": "25,25",
        },
        {
            "source_column": "age",
            "method": "extreme_split",
            "new_column_name": "age_extreme_group",
            "cutoff": "20",
        },
    ]

    for payload in derive_payloads:
        response = client.post(
            "/api/derive-group",
            json={"dataset_id": stored.dataset_id, **payload},
        )
        assert response.status_code == 200
        assert response.json()["derived_column"] == payload["new_column_name"]

    for weight in ["logrank", "gehan_breslow", "tarone_ware", "fleming_harrington"]:
        km_response = client.post(
            "/api/kaplan-meier",
            json={
                "dataset_id": stored.dataset_id,
                "time_column": "os_months",
                "event_column": "os_event",
                "event_positive_value": 1,
                "group_column": "stage",
                "time_unit_label": "Months",
                "confidence_level": 0.9,
                "risk_table_points": 5,
                "show_confidence_bands": False,
                "logrank_weight": weight,
                "fh_p": 1.5,
            },
        )
        assert km_response.status_code == 200
        km_payload = km_response.json()
        assert km_payload["analysis"]["curves"]
        assert km_payload["figure"]["data"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_covariates": ["stage", "treatment"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    cohort_response = client.post(
        "/api/cohort-table",
        json={
            "dataset_id": stored.dataset_id,
            "variables": ["age", "stage", "treatment", "biomarker_score"],
            "group_column": "stage",
        },
    )
    assert cohort_response.status_code == 200
    assert cohort_response.json()["analysis"]["rows"]


def test_cox_preview_reports_complete_case_counts() -> None:
    stored = store.create(
        make_example_dataset(seed=71, n_patients=120),
        filename="cox_preview.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/cox-preview",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "sex", "stage"],
            "categorical_covariates": ["sex", "stage"],
        },
    )

    assert response.status_code == 200
    payload = response.json()["preview"]
    assert payload["outcome_rows"] >= payload["analyzable_rows"] >= 1
    assert payload["events"] >= 1
    assert payload["estimated_parameters"] >= 1
    assert payload["events_per_parameter"] is None or payload["events_per_parameter"] > 0
    assert payload["covariates"] == ["age", "sex", "stage"]
    assert payload["categorical_covariates"] == ["sex", "stage"]


def test_cox_preview_flags_one_sided_categorical_levels() -> None:
    dataset = client.post("/api/load-tcga-example").json()

    response = client.post(
        "/api/cox-preview",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "sex", "pathologic_stage", "histology", "kras_status", "egfr_status"],
            "categorical_covariates": ["sex", "pathologic_stage", "histology", "kras_status", "egfr_status"],
        },
    )

    assert response.status_code == 200
    payload = response.json()["preview"]
    assert payload["stability_warnings"]
    assert any("histology" in warning for warning in payload["stability_warnings"])
    assert any(item["column"] == "histology" for item in payload["risky_levels"])


def test_cox_preview_flags_small_reference_levels() -> None:
    dataset = client.post("/api/load-tcga-example").json()

    response = client.post(
        "/api/cox-preview",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": [
                "age",
                "sex",
                "pathologic_stage",
                "kras_status",
                "egfr_status",
                "expression_subtype",
                "tumor_longest_dimension_cm",
            ],
            "categorical_covariates": [
                "sex",
                "pathologic_stage",
                "kras_status",
                "egfr_status",
                "expression_subtype",
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()["preview"]
    assert any("reference level" in warning for warning in payload["stability_warnings"])
    assert any(item["column"] == "pathologic_stage" and item["issue"] == "small_reference_level" for item in payload["risky_levels"])


def test_derive_group_rejects_existing_column_name_collision() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/derive-group",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_column": "age",
            "method": "median_split",
            "new_column_name": "age",
        },
    )

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_derive_group_response_tracks_outcome_informed_provenance() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/derive-group",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_column": "age",
            "method": "optimal_cutpoint",
            "new_column_name": "age_bin",
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
        },
    )

    assert response.status_code == 200
    provenance = response.json()["derived_column_provenance"]
    assert provenance["age_bin"]["outcome_informed"] is True
    assert provenance["age_bin"]["recipe"]["method"] == "optimal_cutpoint"
    assert provenance["age_bin"]["recipe"]["source_column"] == "age"


def test_signature_search_rejects_existing_column_name_collision() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "sex", "stage"],
            "max_combination_size": 2,
            "top_k": 5,
            "bootstrap_iterations": 1,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "new_column_name": "stage",
        },
    )

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_signature_search_response_tracks_outcome_informed_provenance() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "sex", "stage"],
            "max_combination_size": 2,
            "top_k": 5,
            "bootstrap_iterations": 1,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "new_column_name": "sig_age_stage",
        },
    )

    assert response.status_code == 200
    provenance = response.json()["derived_column_provenance"]
    assert provenance["sig_age_stage"]["outcome_informed"] is True


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_core_endpoints_support_explicit_string_event_positive_value() -> None:
    df = make_example_dataset(seed=72, n_patients=84)
    df["custom_event"] = df["os_event"].map({1: "R", 0: "N"})
    stored = store.create(df, filename="custom_event_core.csv", copy_dataframe=False)

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "group_column": "stage",
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["cohort"]["events"] > 0

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "covariates": ["age", "stage", "treatment"],
            "categorical_covariates": ["stage", "treatment"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "rsf",
            "n_estimators": 20,
            "max_depth": 3,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["model_stats"]["n_events"] > 0

    deep_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "deepsurv",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
        },
    )
    assert deep_response.status_code == 200
    assert deep_response.json()["analysis"]["n_samples"] > 0


@pytest.mark.parametrize("model_type", ["rsf", "gbs"])
def test_single_ml_model_endpoints_handle_mixed_features(model_type: str) -> None:
    stored = store.create(
        make_example_dataset(seed=70, n_patients=96),
        filename=f"single_{model_type}.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": model_type,
            "n_estimators": 24,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["model_stats"]["evaluation_mode"] in {"holdout", "apparent"}
    assert payload["analysis"]["feature_importance"]
    assert payload["importance_figure"]["data"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
@pytest.mark.parametrize("model_type", ["deepsurv", "deephit", "mtlr", "transformer", "vae"])
def test_single_deep_model_endpoints_handle_mixed_features(model_type: str) -> None:
    stored = store.create(
        make_example_dataset(seed=71, n_patients=64),
        filename=f"single_{model_type}.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": model_type,
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 23,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["evaluation_mode"] in {"holdout", "apparent", "holdout_fallback_apparent"}
    assert payload["analysis"]["scientific_summary"]["headline"]
    assert payload["figures"]["loss"]["data"]


def test_deep_model_endpoint_rejects_invalid_transformer_width() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score"],
            "categorical_features": [],
            "model_type": "transformer",
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 23,
            "d_model": 30,
            "n_heads": 4,
            "n_layers": 1,
        },
    )

    assert response.status_code == 422
    assert "Transformer width must be divisible by attention heads." in response.text


def test_ml_model_compare_endpoint_supports_repeated_cv_export() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "compare",
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 2,
            "random_state": 17,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    analysis = payload["analysis"]
    assert analysis["evaluation_mode"] == "repeated_cv"
    assert analysis["cv_folds"] == 2
    assert analysis["cv_repeats"] == 2
    assert analysis["fold_results"]
    assert analysis["manuscript_tables"]["model_performance_table"]
    assert payload["figure"]["data"][0]["type"] == "bar"


def test_frontend_ml_compare_forwards_visible_hyperparameters() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    run_compare_start = app_js.index("async function runCompareModels()")
    run_compare_end = app_js.index("state.ml = payload;", run_compare_start)
    run_compare_body = app_js[run_compare_start:run_compare_end]

    assert 'n_estimators: Number(refs.mlNEstimators.value)' in run_compare_body
    assert 'learning_rate: Number(refs.mlLearningRate.value)' in run_compare_body


def test_frontend_exposes_real_dataset_loader_buttons() -> None:
    index_html = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "templates"
        / "index.html"
    ).read_text(encoding="utf-8")
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'id="loadTcgaUploadReadyButton"' in index_html
    assert 'Upload-Ready TCGA' in index_html
    assert 'id="loadGbsg2Button"' in index_html
    assert 'GBSG2 (Real)' in index_html
    assert 'id="datasetPresetBar"' in index_html
    assert 'id="applyBasicPresetButton"' in index_html
    assert 'id="applyModelPresetButton"' in index_html
    assert 'fetchJSON("/api/load-tcga-upload-ready"' in app_js
    assert 'fetchJSON("/api/load-gbsg2-example"' in app_js
    assert "function datasetPresetForCurrentDataset()" in app_js
    assert 'applyDatasetPreset("basic")' in app_js
    assert 'applyDatasetPreset("models")' in app_js


def test_frontend_uses_dataset_aware_download_filenames() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function buildDownloadFilename(stem, ext" in app_js
    assert "function currentDatasetSlug()" in app_js
    assert "function currentOutcomeSlug()" in app_js
    assert 'buildDownloadFilename("km_summary", "csv", { includeGroup: true })' in app_js
    assert 'buildDownloadFilename("cox_results", "csv")' in app_js
    assert 'buildDownloadFilename("cox_forest", "png")' in app_js
    assert 'buildDownloadFilename("cox_forest", "svg")' in app_js
    assert 'showToast("No rows available for export.", "warning")' in app_js
    assert 'buildDownloadFilename("ml_manuscript_table", "docx", { template: currentMlJournalTemplate() })' in app_js
    assert 'buildDownloadFilename("ml_model_comparison", "png")' in app_js
    assert 'buildDownloadFilename("ml_model_comparison", "svg")' in app_js
    assert 'buildDownloadFilename("dl_manuscript_table", "docx", { template: currentDlJournalTemplate() })' in app_js
    assert 'buildDownloadFilename("dl_model_comparison", "png")' in app_js
    assert 'buildDownloadFilename("dl_model_comparison", "svg")' in app_js
    assert 'buildDownloadFilename("km_curve", "png", { includeGroup: true })' in app_js


def test_frontend_csv_download_sanitizes_formula_like_cells() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function downloadCsv(filename, rows, columns = null)" in app_js
    assert "const sanitizeCsvCell = (value) => {" in app_js
    assert 'if (/^[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?$/.test(trimmed)) return text;' in app_js
    assert 'trimmed.startsWith("=")' in app_js
    assert 'trimmed.startsWith("+")' in app_js
    assert 'trimmed.startsWith("-")' in app_js
    assert 'trimmed.startsWith("@")' in app_js
    assert "return `'${text}`;" in app_js


def test_frontend_covariate_picker_keeps_all_unique_continuous_columns() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "col.n_unique === state.dataset.n_rows" not in app_js


def test_frontend_uses_inline_cutpoint_figure_without_refetch() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    derive_start = app_js.index("async function deriveGroup({ applyToGroup = false } = {})")
    derive_end = app_js.index("function updateMethodVisibility()", derive_start)
    derive_body = app_js[derive_start:derive_end]
    assert "payload.cutpoint_figure" in derive_body
    assert 'fetchJSON("/api/optimal-cutpoint"' not in derive_body


def test_frontend_derive_group_explains_that_dl_features_do_not_change() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "async function deriveGroup({ applyToGroup = false } = {})" in app_js
    assert "Current grouping still uses" in app_js
    assert "Current grouping is still Overall only." in app_js
    assert "Not applied yet." in app_js
    assert "Set as Group by. ML/DL features were not changed." in app_js
    assert "Use it for grouping or visualization, not as an ML/DL training feature." in app_js
    assert "Grouping only:" in app_js
    assert "ML and DL share this model feature list" in app_js
    assert "Cox, ML, and DL use the outcome definition plus their own feature selections." in app_js


def test_frontend_derive_group_preserves_existing_group_unless_user_applies_new_column() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    derive_start = app_js.index("async function deriveGroup({ applyToGroup = false } = {})")
    derive_end = app_js.index("function updateMethodVisibility()", derive_start)
    derive_body = app_js[derive_start:derive_end]
    assert "const preservedGroup = refs.groupColumn.value || \"\";" in derive_body
    assert "setSelectValueIfPresent(refs.groupColumn, preservedGroup);" in derive_body
    assert "if (applyToGroup) {" in derive_body


def test_frontend_refreshes_km_after_creating_and_applying_a_new_group() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    derive_start = app_js.index("async function deriveGroup({ applyToGroup = false } = {})")
    derive_end = app_js.index("function updateMethodVisibility()", derive_start)
    derive_body = app_js[derive_start:derive_end]
    assert 'const guidedKmRefresh = runtime.uiMode === "guided" && runtime.guidedGoal === "km";' in derive_body
    assert 'const shouldRefreshKm = applyToGroup && (activeTabName() === "km" || guidedKmRefresh);' in derive_body
    assert "Refreshing Kaplan-Meier with the new grouping..." in derive_body
    assert "Kaplan-Meier is refreshing now." in derive_body
    assert "await runKaplanMeier();" in derive_body


def test_frontend_derive_group_uses_lightweight_dataset_refresh() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function updateAfterDerivedDataset(payload, { deferChrome = false } = {})" in app_js
    assert "state.km = preservedStates.km;" in app_js
    assert "state.ml = preservedStates.ml;" in app_js
    assert "if (!deferChrome) {" in app_js
    derive_start = app_js.index("async function deriveGroup({ applyToGroup = false } = {})")
    derive_end = app_js.index("function updateMethodVisibility()", derive_start)
    derive_body = app_js[derive_start:derive_end]
    assert "updateAfterDerivedDataset(payload, { deferChrome: shouldRefreshKm });" in derive_body
    assert "updateAfterDataset(payload);" not in derive_body


def test_guided_confirm_outcome_shows_ready_status_when_event_value_is_set() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'const issueHeading = canContinue ? "Ready to continue" : "What still needs attention";' in app_js
    assert '`SurvStudio is ready to use ${refs.timeColumn?.value || "time"}, ${refs.eventColumn?.value || "event"}, and ${refs.eventPositiveValue?.value || "event value"}.`' in app_js


def test_guided_mode_exposes_compare_all_actions_for_ml_and_dl() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    styles = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "styles.css"
    ).read_text(encoding="utf-8")

    assert 'secondaryAction: "run-ml-compare"' in app_js
    assert 'secondaryAction: "run-dl-compare"' in app_js
    assert 'secondaryLabel: "Compare all ML models"' in app_js
    assert 'secondaryLabel: "Compare all DL models"' in app_js
    assert 'guided-actions guided-actions-priority' in app_js
    assert 'guided-actions guided-actions-secondary' in app_js
    assert 'guided-run-choice' in app_js
    assert 'runCompareInlineButton' in app_js
    assert 'runDlCompareInlineButton' in app_js
    assert 'if (action === "run-ml-compare")' in app_js
    assert 'if (action === "run-dl-compare")' in app_js
    assert 'body[data-ui-mode="guided"][data-guided-step="5"] #panel-ml #runCompareButton' in styles
    assert 'body[data-ui-mode="guided"][data-guided-step="5"] #panel-dl #runDlCompareButton' in styles
    assert ".guided-actions-priority" in styles
    assert ".guided-run-choice" in styles
    assert ".run-setup-quick-actions" in styles


def test_frontend_locks_ml_and_dl_run_buttons_by_scope() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "busyScopes: {}" in app_js
    assert "function buttonsForScope(scope)" in app_js
    assert 'if (scope === "ml") return [refs.runMlButton, refs.runCompareButton, refs.runCompareInlineButton];' in app_js
    assert 'if (scope === "dl") return [refs.runDlButton, refs.runDlCompareButton, refs.runDlCompareInlineButton];' in app_js
    assert "function setScopeBusy(scope, isBusy, activeButton = null)" in app_js
    assert 'button === refs.runMlButton || button === refs.runCompareButton || button === refs.runCompareInlineButton ? "ml"' in app_js
    assert 'button === refs.runDlButton || button === refs.runDlCompareButton || button === refs.runDlCompareInlineButton ? "dl"' in app_js
    assert 'setScopeBusy(scope, true, button);' in app_js
    assert 'setScopeBusy(scope, false, button);' in app_js


def test_frontend_scrolls_to_results_after_runs_finish() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function resultAnchorFor(tabName, { mode = \"single\" } = {})" in app_js
    assert "function scrollToAnalysisResult(tabName, { mode = \"single\" } = {})" in app_js
    assert 'setGuidedStep(5, { scroll: false, historyMode: "push" });' in app_js
    assert 'scrollToAnalysisResult(tabName, { mode: resultMode });' in app_js
    assert 'revealCompletedResultIfCurrent("km", {' in app_js
    assert 'revealCompletedResultIfCurrent("cox", {' in app_js
    assert 'revealCompletedResultIfCurrent("tables", {' in app_js
    assert 'revealCompletedResultIfCurrent("ml",' in app_js
    assert 'revealCompletedResultIfCurrent("dl",' in app_js


def test_compare_results_hide_single_model_plot_sections() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    styles = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "styles.css"
    ).read_text(encoding="utf-8")

    assert 'mlPanel: document.getElementById("panel-ml")' in app_js
    assert 'dlPanel: document.getElementById("panel-dl")' in app_js
    assert 'function setPanelResultMode(panel, mode = "idle")' in app_js
    assert 'setPanelResultMode(refs.mlPanel, "single");' in app_js
    assert 'setPanelResultMode(refs.mlPanel, "compare");' in app_js
    assert 'setPanelResultMode(refs.dlPanel, "single");' in app_js
    assert 'setPanelResultMode(refs.dlPanel, "compare");' in app_js
    assert '#panel-ml[data-result-mode="compare"] .ml-plots-grid' in styles
    assert '#panel-dl[data-result-mode="compare"] .ml-plots-grid' in styles


def test_shared_model_feature_bulk_actions_are_exposed() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="selectAllModelFeaturesButton"' in response.text
    assert 'id="clearModelFeaturesButton"' in response.text
    assert 'id="selectAllDlModelFeaturesButton"' in response.text
    assert 'id="clearDlModelFeaturesButton"' in response.text


def test_index_exposes_optimal_cutpoint_controls_and_non_ai_empty_states() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="deriveOptimalControls"' in response.text
    assert 'id="deriveMinGroupFraction"' in response.text
    assert 'id="derivePermutationIterations"' in response.text
    assert 'id="deriveRandomSeed"' in response.text
    assert "Interpretation notes will appear here after analysis." in response.text
    assert "Model interpretation and diagnostic notes will appear here after analysis." in response.text
    assert "When Group by is active, Overall summarizes the grouped non-missing subset." in response.text


def test_frontend_exports_require_current_results_and_signature_scope_guard() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function currentSignatureResult()" in app_js
    assert 'if (!requireCurrentResultForExport("km", { payload })) return;' in app_js
    assert 'if (!requireCurrentResultForExport("cox", { payload })) return;' in app_js
    assert 'if (!requireCurrentResultForExport("tables", { payload })) return;' in app_js
    assert 'if (!requireCurrentResultForExport("ml", { payload })) return;' in app_js
    assert 'if (!requireCurrentResultForExport("dl", { payload })) return;' in app_js
    assert 'if (!payload || isScopeBusy("km")) {' in app_js
    assert 'refs.runSignatureSearchButton, runSignatureSearch, "km"' in app_js
    assert "function syncDownloadButtonAvailability()" in app_js


def test_frontend_syncs_bulk_model_feature_actions_across_ml_and_dl() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function setSharedModelFeatureSelection(nextFeatures = [], { clearCategoricals = false } = {})" in app_js
    assert 'setCheckedValues(refs.modelFeatureChecklist, normalizedFeatures);' in app_js
    assert 'setCheckedValues(refs.dlModelFeatureChecklist, normalizedFeatures);' in app_js
    assert 'refs.selectAllModelFeaturesButton?.addEventListener("click"' in app_js
    assert 'refs.clearModelFeaturesButton?.addEventListener("click"' in app_js
    assert 'refs.selectAllDlModelFeaturesButton?.addEventListener("click"' in app_js
    assert 'refs.clearDlModelFeaturesButton?.addEventListener("click"' in app_js
    assert 'setSharedModelFeatureSelection(modelFeatureCandidateColumns());' in app_js
    assert 'setSharedModelFeatureSelection([], { clearCategoricals: true });' in app_js


def test_guided_summary_bar_is_not_sticky() -> None:
    styles = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "styles.css"
    ).read_text(encoding="utf-8")

    start = styles.index(".guided-summary-bar {")
    end = styles.index(".guided-summary-copy", start)
    guided_summary_css = styles[start:end]
    assert "position: static;" in guided_summary_css
    assert "position: sticky;" not in guided_summary_css


def test_frontend_caps_importance_plot_container_height() -> None:
    styles = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "styles.css"
    ).read_text(encoding="utf-8")

    assert "#mlImportancePlot," in styles
    assert "#dlImportancePlot" in styles
    assert "max-height: 720px;" in styles
    assert "overflow: auto;" in styles


def test_export_table_endpoint_returns_journal_markdown() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH", "Mean C-index": 0.7123, "95% CI": "0.650 to 0.774"},
                {"Rank": 2, "Model": "RSF", "Mean C-index": 0.7011, "95% CI": "0.640 to 0.762"},
            ],
            "format": "markdown",
            "style": "journal",
            "template": "nejm",
            "caption": "Table 1. Model discrimination summary.",
            "notes": ["C-index values are cross-validated."],
        },
    )

    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "*Table 1. Model discrimination summary.*" in response.text
    assert "| Rank | Model | Mean C-index | 95% CI |" in response.text
    assert "Notes:" in response.text


def test_export_table_endpoint_appends_provenance_notes() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [{"Rank": 1, "Model": "RSF", "C-index": 0.712}],
            "format": "markdown",
            "style": "journal",
            "template": "default",
            "caption": "Table 1. Replayable summary.",
            "notes": ["Primary note."],
            "provenance": {
                "request_config": {"model_type": "compare", "random_state": 42},
                "analysis": {"evaluation_mode": "holdout", "shared_training_seed": 42},
            },
        },
    )

    assert response.status_code == 200
    assert "Primary note." in response.text
    assert "Replay request_config:" in response.text
    assert "Replay analysis metadata:" in response.text


def test_export_table_endpoint_preserves_union_of_row_keys() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH"},
                {"Rank": 2, "Model": "RSF", "Mean C-index": 0.7011},
            ],
            "format": "csv",
            "style": "plain",
        },
    )

    assert response.status_code == 200
    assert "Mean C-index" in response.text.splitlines()[0]
    assert "0.7011" in response.text


def test_export_table_endpoint_csv_ignores_caption_and_notes_preamble() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH", "C-index": 0.712},
            ],
            "format": "csv",
            "style": "plain",
            "caption": "Table 1. Should not appear in CSV preamble.",
            "notes": ["This note should stay out of the CSV body."],
        },
    )

    assert response.status_code == 200
    lines = response.text.splitlines()
    assert lines[0] == "Rank,Model,C-index"
    assert not response.text.startswith("# ")


def test_export_table_endpoint_sanitizes_formula_like_csv_cells() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Sample": "=cmd|'/C calc'!A0", "Model": "+SUM(1,1)", "Note": "@risk"},
                {"Sample": "-10", "Model": "RSF", "Note": "safe"},
            ],
            "format": "csv",
            "style": "plain",
        },
    )

    assert response.status_code == 200
    lines = response.text.splitlines()
    first_row = next(csv.reader([lines[1]]))
    second_row = next(csv.reader([lines[2]]))
    assert first_row[0] == "'=cmd|'/C calc'!A0"
    assert first_row[1] == "'+SUM(1,1)"
    assert first_row[2] == "'@risk"
    assert second_row[0] == "-10"


def test_export_table_endpoint_returns_journal_latex() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "DeepSurv", "Mean C-index": 0.7342, "95% CI": "0.681 to 0.787"},
            ],
            "format": "latex",
            "style": "journal",
            "template": "jco",
            "caption": "Table 2. Deep-model comparison.",
            "notes": ["Mean C-index is averaged over repeated CV folds."],
        },
    )

    assert response.status_code == 200
    assert "text/x-tex" in response.headers["content-type"]
    assert "\\caption{Table 2. Deep-model comparison.}" in response.text
    assert "\\textit{Footnotes:}" in response.text
    assert "DeepSurv" in response.text


def test_export_table_endpoint_escapes_backslashes_without_double_escaping_braces() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Path": "C:\\tmp\\results", "Value": "{alpha}_1"},
            ],
            "format": "latex",
            "style": "journal",
            "template": "default",
            "caption": "Table 1. Paths.",
            "notes": [],
        },
    )

    assert response.status_code == 200
    assert r"\textbackslash{}tmp\textbackslash{}results" in response.text
    assert r"\textbackslash\{\}" not in response.text


def test_export_table_endpoint_returns_docx_archive() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH", "C-index": 0.712},
                {"Rank": 2, "Model": "RSF", "C-index": 0.701},
            ],
            "format": "docx",
            "style": "journal",
            "template": "lancet",
            "caption": "Table 1. Model performance summary.",
            "notes": ["Lancet-style comments block."],
        },
    )

    assert response.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in response.headers["content-type"]
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert "[Content_Types].xml" in names
    assert "_rels/.rels" in names
    assert "word/document.xml" in names
    assert "word/_rels/document.xml.rels" in names
    document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Table 1. Model performance summary." in document_xml
    assert "Comments:" in document_xml
    assert "Lancet-style comments block." in document_xml


def test_export_table_endpoint_rejects_empty_rows() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [],
            "format": "markdown",
            "style": "journal",
            "template": "default",
        },
    )

    assert response.status_code == 400
    assert "No rows available for export." in response.json()["detail"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_model_compare_endpoint_returns_comparison_figure() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
            "num_time_bins": 10,
            "n_heads": 4,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["comparison_table"]
    assert payload["analysis"]["scientific_summary"]["headline"]
    assert payload["figures"]["comparison"]["data"][0]["type"] == "bar"


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_model_compare_endpoint_supports_repeated_cv() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
            "num_time_bins": 10,
            "n_heads": 4,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "parallel_jobs": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["evaluation_mode"] == "repeated_cv"
    assert payload["analysis"]["cv_folds"] == 2
    assert payload["analysis"]["fold_results"]
    assert payload["analysis"]["manuscript_tables"]["model_performance_table"]


def test_deep_single_model_endpoint_supports_repeated_cv(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    dataset = client.post("/api/load-example").json()

    def _fake_evaluate_single(*args, **kwargs):
        assert kwargs["evaluation_strategy"] == "repeated_cv"
        assert kwargs["cv_folds"] == 2
        assert kwargs["cv_repeats"] == 2
        return {
            "model": "Neural MTLR",
            "c_index": 0.701,
            "evaluation_mode": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 2,
            "epochs_trained": 9,
            "training_seeds": [42, 43],
            "repeat_results": [{"repeat": 1, "c_index": 0.69}, {"repeat": 2, "c_index": 0.712}],
            "comparison_table": [{"model": "Neural MTLR", "c_index": 0.701}],
            "manuscript_tables": {"model_performance_table": [{"Model": "Neural MTLR"}]},
            "scientific_summary": {"headline": "stub"},
            "insight_board": {"headline": "stub"},
        }

    monkeypatch.setattr(deep_models, "evaluate_single_deep_survival_model", _fake_evaluate_single)

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "mtlr",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
            "num_time_bins": 10,
            "n_heads": 4,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "parallel_jobs": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["evaluation_mode"] == "repeated_cv"
    assert payload["analysis"]["training_seeds"] == [42, 43]
    assert payload["analysis"]["comparison_table"][0]["model"] == "Neural MTLR"


def test_deep_single_model_endpoint_builds_monitor_loss_curve_when_history_is_available(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    dataset = client.post("/api/load-example").json()

    def _fake_evaluate_single(*args, **kwargs):
        return {
            "model": "Survival Transformer",
            "c_index": 0.701,
            "evaluation_mode": "holdout",
            "epochs_trained": 3,
            "max_epochs_requested": 10,
            "best_monitor_epoch": 2,
            "stopped_early": True,
            "loss_history": [1.2, 0.9, 0.7],
            "monitor_loss_history": [1.3, 1.0, 0.8],
            "feature_importance": [{"feature": "age", "importance": 0.4}],
            "scientific_summary": {"headline": "stub"},
            "insight_board": {"headline": "stub"},
        }

    monkeypatch.setattr(deep_models, "evaluate_single_deep_survival_model", _fake_evaluate_single)

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "transformer",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
            "num_time_bins": 10,
            "n_heads": 4,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["figures"]["loss"]["layout"]["title"]["text"] == "TRANSFORMER Training and Monitor Loss"
    assert len(payload["figures"]["loss"]["data"]) == 2
    assert payload["figures"]["loss"]["data"][0]["name"] == "Training loss"
    assert payload["figures"]["loss"]["data"][1]["name"] == "Monitor loss"
    annotation_text = " ".join(
        str(annotation.get("text", ""))
        for annotation in payload["figures"]["loss"]["layout"].get("annotations", [])
    )
    assert "Best monitor epoch: 2" in annotation_text
    assert "Stopped early at epoch 3" in annotation_text


def test_ml_compare_workflow_with_mixed_features_exports_manuscript_table() -> None:
    stored = store.create(
        make_example_dataset(seed=41, n_patients=96),
        filename="workflow_mixed_ml.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "compare",
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
            "n_estimators": 16,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 31,
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    manuscript = analysis["manuscript_tables"]
    assert analysis["comparison_table"]
    assert manuscript["model_performance_table"]

    export_response = client.post(
        "/api/export-table",
        json={
            "rows": manuscript["model_performance_table"],
            "format": "markdown",
            "style": "journal",
            "template": "nejm",
            "caption": manuscript["caption"],
            "notes": manuscript["table_notes"],
        },
    )

    assert export_response.status_code == 200
    assert "text/markdown" in export_response.headers["content-type"]
    assert manuscript["caption"] in export_response.text


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_compare_workflow_with_mixed_features_exports_manuscript_table() -> None:
    stored = store.create(
        make_example_dataset(seed=42, n_patients=84),
        filename="workflow_mixed_dl.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 33,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
            "evaluation_strategy": "holdout",
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    manuscript = analysis["manuscript_tables"]
    assert analysis["comparison_table"]
    assert manuscript["model_performance_table"]

    export_response = client.post(
        "/api/export-table",
        json={
            "rows": manuscript["model_performance_table"],
            "format": "latex",
            "style": "journal",
            "template": "jco",
            "caption": manuscript["caption"],
            "notes": manuscript["table_notes"],
        },
    )

    assert export_response.status_code == 200
    assert "text/x-tex" in export_response.headers["content-type"]
    assert "\\caption{" in export_response.text


def test_load_tcga_example_endpoint_returns_real_public_cohort() -> None:
    response = client.post("/api/load-tcga-example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "tcga_luad_xena_example"
    assert payload["n_rows"] >= 500
    column_names = {column["name"] for column in payload["columns"]}
    assert {"os_months", "os_event", "age", "pathologic_stage", "stage_group", "smoking_status"}.issubset(column_names)


@pytest.mark.parametrize(
    "path",
    [
        "/api/load-example",
        "/api/load-tcga-example",
        "/api/load-tcga-upload-ready",
        "/api/load-gbsg2-example",
    ],
)
def test_builtin_dataset_loaders_mark_preset_eligible(path: str) -> None:
    response = client.post(path)

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_source"] == "builtin_demo"
    assert payload["preset_eligible"] is True


def test_load_tcga_upload_ready_endpoint_returns_compact_real_cohort() -> None:
    response = client.post("/api/load-tcga-upload-ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "tcga_luad_upload_ready"
    assert payload["n_rows"] == 609
    column_names = {column["name"] for column in payload["columns"]}
    assert {"os_months", "os_event", "age", "sex", "stage_group", "smoking_status"}.issubset(column_names)


def test_load_gbsg2_example_endpoint_returns_real_public_cohort() -> None:
    response = client.post("/api/load-gbsg2-example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "gbsg2_upload_ready"
    assert payload["n_rows"] == 686
    column_names = {column["name"] for column in payload["columns"]}
    assert {"rfs_days", "rfs_event", "age", "horTh", "menostat", "pnodes", "tgrade", "tsize"}.issubset(column_names)


@pytest.mark.parametrize(
    (
        "load_path",
        "time_column",
        "event_column",
        "group_column",
        "covariates",
        "categorical_covariates",
    ),
    [
        (
            "/api/load-example",
            "os_months",
            "os_event",
            "stage",
            ["age", "stage", "treatment", "biomarker_score"],
            ["stage", "treatment"],
        ),
        (
            "/api/load-tcga-example",
            "os_months",
            "os_event",
            "stage_group",
            ["age", "sex", "stage_group", "smoking_status"],
            ["sex", "stage_group", "smoking_status"],
        ),
        (
            "/api/load-tcga-upload-ready",
            "os_months",
            "os_event",
            "stage_group",
            ["age", "sex", "stage_group", "smoking_status"],
            ["sex", "stage_group", "smoking_status"],
        ),
        (
            "/api/load-gbsg2-example",
            "rfs_days",
            "rfs_event",
            "horTh",
            ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            ["horTh", "menostat", "tgrade"],
        ),
    ],
)
def test_builtin_examples_run_classical_and_ml_smoke(
    load_path: str,
    time_column: str,
    event_column: str,
    group_column: str,
    covariates: list[str],
    categorical_covariates: list[str],
) -> None:
    load_response = client.post(load_path)

    assert load_response.status_code == 200
    dataset = load_response.json()

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": time_column,
            "event_column": event_column,
            "event_positive_value": 1,
            "group_column": group_column,
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["curves"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": time_column,
            "event_column": event_column,
            "event_positive_value": 1,
            "covariates": covariates,
            "categorical_covariates": categorical_covariates,
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": time_column,
            "event_column": event_column,
            "event_positive_value": 1,
            "features": covariates,
            "categorical_features": categorical_covariates,
            "model_type": "rsf",
            "n_estimators": 12,
            "max_depth": 3,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    ml_analysis = ml_response.json()["analysis"]
    assert ml_analysis["model_stats"]["n_evaluation_patients"] > 0
    assert ml_analysis["feature_importance"]


def test_uploaded_dataset_marks_preset_ineligible() -> None:
    payload = b"os_months,os_event,age\n12,1,60\n18,0,55\n24,1,70\n"
    response = client.post(
        "/api/upload",
        files={"file": ("custom_upload.csv", payload, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_source"] == "upload"
    assert body["preset_eligible"] is False


def test_upload_ready_real_tcga_file_runs_classical_and_ml_smoke() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "tcga_luad_nature2014_upload_ready.csv"
    payload = example_path.read_bytes()

    upload_response = client.post(
        "/api/upload",
        files={"file": (example_path.name, payload, "text/csv")},
    )

    assert upload_response.status_code == 200
    dataset = upload_response.json()
    assert dataset["n_rows"] == 609

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "stage_group",
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["curves"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "sex", "stage_group", "smoking_status"],
            "categorical_covariates": ["sex", "stage_group", "smoking_status"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage_group", "smoking_status"],
            "categorical_features": ["sex", "stage_group", "smoking_status"],
            "model_type": "rsf",
            "n_estimators": 20,
            "max_depth": 3,
            "random_state": 13,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["model_stats"]["n_evaluation_patients"] > 0


def test_upload_ready_real_gbsg2_file_runs_classical_and_ml_smoke() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "gbsg2_jco1994_upload_ready.csv"
    payload = example_path.read_bytes()

    upload_response = client.post(
        "/api/upload",
        files={"file": (example_path.name, payload, "text/csv")},
    )

    assert upload_response.status_code == 200
    dataset = upload_response.json()
    assert dataset["n_rows"] == 686

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "group_column": "horTh",
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["curves"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "covariates": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_covariates": ["horTh", "menostat", "tgrade"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "features": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_features": ["horTh", "menostat", "tgrade"],
            "model_type": "rsf",
            "n_estimators": 20,
            "max_depth": 3,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["model_stats"]["n_evaluation_patients"] > 0


def test_switching_datasets_keeps_derived_groups_isolated_across_reanalysis() -> None:
    first_dataset = client.post("/api/load-example").json()

    derive_first = client.post(
        "/api/derive-group",
        json={
            "dataset_id": first_dataset["dataset_id"],
            "source_column": "age",
            "method": "median_split",
            "new_column_name": "risk_split",
        },
    )
    assert derive_first.status_code == 200
    first_derived_dataset = derive_first.json()

    first_km = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": first_derived_dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "risk_split",
        },
    )
    assert first_km.status_code == 200
    assert len(first_km.json()["analysis"]["curves"]) == 2

    second_dataset = client.post("/api/load-gbsg2-example").json()

    missing_group_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": second_dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "group_column": "risk_split",
        },
    )
    assert missing_group_response.status_code == 404
    assert "Column not found" in missing_group_response.json()["detail"]

    derive_second = client.post(
        "/api/derive-group",
        json={
            "dataset_id": second_dataset["dataset_id"],
            "source_column": "age",
            "method": "median_split",
            "new_column_name": "risk_split",
        },
    )
    assert derive_second.status_code == 200
    second_derived_dataset = derive_second.json()

    second_km = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": second_derived_dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "group_column": "risk_split",
        },
    )
    assert second_km.status_code == 200
    assert len(second_km.json()["analysis"]["curves"]) == 2

    first_km_rerun = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": first_derived_dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "risk_split",
        },
    )
    assert first_km_rerun.status_code == 200
    assert len(first_km_rerun.json()["analysis"]["curves"]) == 2


def test_export_endpoints_can_be_saved_to_nonempty_files(tmp_path: Path) -> None:
    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "covariates": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_covariates": ["horTh", "menostat", "tgrade"],
        },
    )
    assert cox_response.status_code == 200
    cox_rows = cox_response.json()["analysis"]["results_table"]
    cox_path = tmp_path / "cox_results.csv"
    cox_path.write_text(
        "\n".join(
            [
                ",".join(cox_rows[0].keys()),
                *[",".join(str(row.get(col, "")) for col in cox_rows[0].keys()) for row in cox_rows[:3]],
            ]
        ),
        encoding="utf-8",
    )
    assert cox_path.stat().st_size > 0

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "features": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_features": ["horTh", "menostat", "tgrade"],
            "model_type": "compare",
            "evaluation_strategy": "holdout",
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    manuscript = ml_response.json()["analysis"]["manuscript_tables"]

    for fmt, ext in [("markdown", "md"), ("latex", "tex"), ("docx", "docx")]:
        export_response = client.post(
            "/api/export-table",
            json={
                "rows": manuscript["model_performance_table"],
                "format": fmt,
                "style": "journal",
                "template": "jco",
                "caption": manuscript["caption"],
                "notes": manuscript["table_notes"],
            },
        )
        assert export_response.status_code == 200
        out_path = tmp_path / f"ml_manuscript.{ext}"
        if fmt == "docx":
            out_path.write_bytes(export_response.content)
        else:
            out_path.write_text(export_response.text, encoding="utf-8")
        assert out_path.stat().st_size > 0


def test_km_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "group_column": "horTh",
        },
    )
    assert km_response.status_code == 200
    figure = go.Figure(km_response.json()["figure"])

    png_path = tmp_path / "km_plot.png"
    svg_path = tmp_path / "km_plot.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


def test_cox_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "covariates": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_covariates": ["horTh", "menostat", "tgrade"],
        },
    )
    assert cox_response.status_code == 200
    figure = go.Figure(cox_response.json()["figure"])

    png_path = tmp_path / "cox_plot.png"
    svg_path = tmp_path / "cox_plot.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


def test_ml_comparison_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "features": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_features": ["horTh", "menostat", "tgrade"],
            "model_type": "compare",
            "evaluation_strategy": "holdout",
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    figure = go.Figure(ml_response.json()["figure"])

    png_path = tmp_path / "ml_comparison.png"
    svg_path = tmp_path / "ml_comparison.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_dl_comparison_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    stored = store.create(
        make_example_dataset(seed=84, n_patients=72),
        filename="dl_compare_render.csv",
        copy_dataframe=False,
    )

    dl_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 21,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )
    assert dl_response.status_code == 200
    figure = go.Figure(dl_response.json()["figures"]["comparison"])

    png_path = tmp_path / "dl_comparison.png"
    svg_path = tmp_path / "dl_comparison.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_upload_ready_real_tcga_file_runs_deep_smoke() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "tcga_luad_nature2014_upload_ready.csv"
    payload = example_path.read_bytes()

    upload_response = client.post(
        "/api/upload",
        files={"file": (example_path.name, payload, "text/csv")},
    )

    assert upload_response.status_code == 200
    dataset = upload_response.json()

    deep_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage_group", "smoking_status"],
            "categorical_features": ["sex", "stage_group", "smoking_status"],
            "model_type": "deepsurv",
            "hidden_layers": [16],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
            "random_seed": 13,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )
    assert deep_response.status_code == 200
    assert deep_response.json()["analysis"]["n_samples"] > 0


def test_dev_extra_includes_ml_and_dl_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dev_dependencies = set(pyproject["project"]["optional-dependencies"]["dev"])

    assert {"scikit-survival>=0.23.0", "shap>=0.45.0", "torch>=2.0.0"}.issubset(dev_dependencies)
    assert {"httpx>=0.28.1", "pytest>=8.3.5"}.issubset(dev_dependencies)


def test_guided_tables_hide_cutpoint_scan_when_goal_is_not_km() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "const showCutpointPlot = hasCutpointPlot && (!guidedActive || goal === \"km\");" in app_js


def test_guided_tables_use_single_column_builder_layout() -> None:
    styles = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "styles.css"
    ).read_text(encoding="utf-8")

    assert 'body[data-ui-mode="guided"] #panel-tables.guided-visible .table-builder-grid {' in styles
    assert "grid-template-columns: 1fr;" in styles
    assert 'body[data-ui-mode="guided"] #panel-tables.guided-visible .table-card {' in styles
    assert "order: 1;" in styles
    assert 'body[data-ui-mode="guided"] #panel-tables.guided-visible .selection-card {' in styles
    assert "order: 2;" in styles
    assert 'body[data-ui-mode="guided"][data-guided-step="4"] #panel-tables .table-card {' not in styles


def test_guided_tables_run_uses_clicked_button_and_state_based_success_check() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'async function runGuidedGoal(tabName, button, action, { resultMode = "single", successCheck = null } = {})' in app_js
    assert 'const hasResult = typeof successCheck === "function" ? Boolean(successCheck()) : Boolean(currentGoalResult(tabName));' in app_js
    assert 'if (action === "run-tables") {' in app_js
    assert 'void runGuidedGoal("tables", target, runCohortTable, {' in app_js
    assert 'successCheck: () => Boolean(state.cohort?.analysis),' in app_js
    assert 'void runGuidedGoal("tables", refs.runCohortTableButton, runCohortTable, {' in app_js


def test_guided_runs_use_scope_override_for_loading_locks() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "async function withLoading(button, action, scopeOverride = null) {" in app_js
    assert "const scope = scopeOverride || (" in app_js
    assert "await withLoading(button, action, tabName);" in app_js
    assert "const scopeButtons = buttonsForScope(scope);" in app_js
    assert "if (activeButton && !scopeButtons.includes(activeButton)) {" in app_js
    assert "setButtonLoading(activeButton, isBusy);" in app_js


def test_compare_all_actions_surface_pending_feedback() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function mlComparePendingBannerText({ rowCount, evaluationStrategy, cvFolds, cvRepeats }) {" in app_js
    assert "function dlComparePendingBannerText({ rowCount, evaluationStrategy, cvFolds, cvRepeats }) {" in app_js
    assert 'setRuntimeBanner("Screening Cox PH, LASSO-Cox, Random Survival Forest, and Gradient Boosted Survival on one shared evaluation path. This can take a little while on larger cohorts.", "info");' in app_js
    assert 'setRuntimeBanner("Comparing all deep-learning models. This can take noticeably longer than a single run.", "info");' in app_js
    assert 'busyText: "DL model run in progress. Deep-learning runs can take longer, so stay on this analysis path if you want the updated result to open here when the run finishes."' in app_js
    assert 'class="guided-run-status" role="status"' in app_js
    assert "refs.mlMetaBanner.textContent = mlComparePendingBannerText({" in app_js
    assert "refs.dlMetaBanner.textContent = dlComparePendingBannerText({" in app_js


def test_guided_run_tips_use_polished_analysis_specific_copy() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'text: "Review the settings below, then run the curve once."' in app_js
    assert 'text: "Review the settings below, then fit the model once."' in app_js
    assert app_js.count('text: "Review the settings below, then start with one run."') >= 2
    assert 'text: "Review the settings below, then build the table once."' in app_js
    assert 'tip: "Use this after the classical analyses look right."' in app_js
    assert 'tip: "Start with one model. This is the slowest and most advanced path."' in app_js
    assert 'If the result looks odd, go back and change one setting before running again.' in app_js


def test_guided_tables_configure_panel_uses_stacked_layout() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    styles = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "styles.css"
    ).read_text(encoding="utf-8")

    assert '<div class="guided-panel-grid guided-panel-grid-compact">' in app_js
    assert ".guided-panel-grid.guided-panel-grid-stacked {" in styles
    assert "grid-template-columns: 1fr;" in styles
    assert ".guided-actions-priority:not(.guided-actions-dual) .guided-run-choice {" in styles
    assert "min-width: 220px;" in styles
    assert 'body[data-ui-mode="guided"] #panel-tables.guided-visible .table-builder-grid {' in styles
    assert 'body[data-ui-mode="guided"] #panel-tables.guided-visible .table-card {' in styles
    assert 'body[data-ui-mode="guided"] #panel-tables.guided-visible .selection-card {' in styles
    assert "justify-self: stretch;" in styles
    assert "body[data-ui-mode=\"guided\"] #guidedConfigMount," in styles
    assert "body[data-ui-mode=\"guided\"] #guidedActivePanelMount {" in styles
    assert "display: contents;" in styles
    assert "body[data-ui-mode=\"guided\"] .guided-main," in styles
    assert "body[data-ui-mode=\"guided\"] .smart-banner {" in styles
    assert "body[data-ui-mode=\"guided\"] .guided-main,\n  body[data-ui-mode=\"guided\"] .config-strip,\n  body[data-ui-mode=\"guided\"] .smart-banner,\n  body[data-ui-mode=\"guided\"] .tab-panel.guided-visible {" not in styles


def test_cohort_table_dependency_copy_marks_stale_output_and_rebuild_label() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function currentCohortTableOutputState() {" in app_js
    assert "if (!hasDataset || !tableState.hasOutput || tableState.isCurrent) {" in app_js
    assert "Output below still reflects the last built table:" in app_js
    assert "function updateCohortTableButtonLabel() {" in app_js
    assert '? "Rebuild Table"' in app_js
    assert "renderSharedFeatureSummary();" in app_js


def test_window_resize_schedules_plot_resizing() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "plotResizeTimer: null," in app_js
    assert "function scheduleVisiblePlotResize(delay = 80) {" in app_js
    assert "window.clearTimeout(runtime.plotResizeTimer);" in app_js
    assert "window.addEventListener(\"resize\", () => {" in app_js
    assert "scheduleVisiblePlotResize(80);" in app_js


def test_ml_single_model_endpoint_rejects_repeated_cv_strategy() -> None:
    dataset = client.post("/api/load-example").json()

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "model_type": "rsf",
            "features": ["age", "biomarker_score"],
            "categorical_features": [],
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 3,
            "cv_repeats": 2,
        },
    )

    assert response.status_code == 400
    assert "deterministic holdout only" in response.json()["detail"]


def test_deep_model_endpoint_rejects_empty_hidden_layers() -> None:
    dataset = client.post("/api/load-example").json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "model_type": "deepsurv",
            "features": ["age"],
            "categorical_features": [],
            "hidden_layers": [],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 42,
            "evaluation_strategy": "holdout",
            "cv_folds": 2,
            "cv_repeats": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "parallel_jobs": 1,
            "num_time_bins": 10,
            "n_heads": 1,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 2,
            "n_clusters": 2,
        },
    )

    assert response.status_code == 422
    assert "Hidden layers must contain at least one positive integer." in json.dumps(response.json())


def test_cohort_table_rejects_outcome_informed_grouping_column() -> None:
    dataset = client.post("/api/load-example").json()
    derived = client.post(
        "/api/derive-group",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_column": "age",
            "method": "optimal_cutpoint",
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "new_column_name": "age_opt",
        },
    ).json()

    response = client.post(
        "/api/cohort-table",
        json={
            "dataset_id": derived["dataset_id"],
            "variables": ["age", "sex"],
            "group_column": "age_opt",
        },
    )

    assert response.status_code == 400
    assert "grouped cohort tables" in response.json()["detail"]


def test_signature_search_response_includes_request_config_for_staleness_checks() -> None:
    dataset = client.post("/api/load-example").json()

    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "stage", "treatment"],
            "max_combination_size": 2,
            "top_k": 5,
            "bootstrap_iterations": 0,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "new_column_name": "sig_age_stage",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["signature_request_config"]["dataset_id"] == dataset["dataset_id"]
    assert payload["signature_request_config"]["candidate_columns"] == ["age", "stage", "treatment"]

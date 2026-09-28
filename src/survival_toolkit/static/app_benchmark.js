(function attachSurvStudioBenchmark(global) {
  function createBenchmarkBoardApi(deps) {
    const {
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
      fetchJSON,
      requestBoardRender,
    } = deps;

    function createBenchmarkActionButton(action) {
      const button = document.createElement("button");
      button.className = "button ghost compact-btn";
      button.type = "button";
      button.textContent = String(action?.label || "Review");
      if (action?.title) {
        button.title = String(action.title);
      }
      if (action?.disabled) {
        button.disabled = true;
      }
      if (action?.dataset && typeof action.dataset === "object") {
        Object.entries(action.dataset).forEach(([key, value]) => {
          if (value === null || value === undefined || value === "") return;
          button.dataset[key] = String(value);
        });
      }
      return button;
    }

    function benchmarkRowFamilyMeta(row) {
      const explicitTab = String(row?.familyTab || "").trim().toLowerCase();
      if (explicitTab === "ml") {
        return { familyTab: "ml", familyLabel: "Classical ML", familyShortLabel: "ML" };
      }
      if (explicitTab === "dl") {
        return { familyTab: "dl", familyLabel: "Deep Learning", familyShortLabel: "DL" };
      }
      const familyLabel = String(row?.family || "").trim();
      const normalizedFamily = familyLabel.toLowerCase();
      if (normalizedFamily.includes("deep")) {
        return { familyTab: "dl", familyLabel: familyLabel || "Deep Learning", familyShortLabel: "DL" };
      }
      if (normalizedFamily.includes("ml")) {
        return { familyTab: "ml", familyLabel: familyLabel || "Classical ML", familyShortLabel: "ML" };
      }
      return { familyTab: "unknown", familyLabel: familyLabel || "Unknown", familyShortLabel: familyLabel || "Unknown" };
    }

    function createBenchmarkActionGroup(row) {
      const familyMeta = benchmarkRowFamilyMeta(row);
      const actionGroup = document.createElement("div");
      actionGroup.className = "button-row compact benchmark-row-actions";
      actionGroup.appendChild(createBenchmarkActionButton(benchmarkReviewAction(row)));
      actionGroup.appendChild(createBenchmarkActionButton({
        dataset: {
          benchmarkParamsGoal: familyMeta.familyTab === "unknown" ? "" : familyMeta.familyTab,
          benchmarkParamsModel: row.model,
          benchmarkParamsSource: row.paramsSource || "current",
        },
        label: "Params",
        disabled: familyMeta.familyTab === "unknown",
        title: "Show the exact compare-run settings used for this leaderboard row.",
      }));
      return actionGroup;
    }

    function benchmarkComparePayload(goal, { currentOnly = false } = {}) {
      const payload = currentOnly ? currentCompareGoalPayload(goal) : compareGoalPayload(goal);
      return panelModeForPayload(payload) === "compare" ? payload : null;
    }

    function benchmarkSnapshotComparePayload(goal) {
      const payload = runtime.compareCache?.unified?.[goal] || null;
      return panelModeForPayload(payload) === "compare" ? payload : null;
    }

    function comparisonRowsFromPayload(payload) {
      return Array.isArray(payload?.analysis?.comparison_table) ? payload.analysis.comparison_table : [];
    }

    function comparePayloadSplitFingerprint(payload) {
      return String(payload?.analysis?.evaluation_split_fingerprint ?? payload?.evaluation_split_fingerprint ?? "").trim();
    }

    function hasBackendRank(rank) {
      return rank !== null && rank !== undefined && rank !== "" && Number.isFinite(Number(rank));
    }

    function familySplitMismatch(families, payloads) {
      // Cross-family ranking is only valid when both families were scored on the same row partitions.
      if (!Array.isArray(families) || families.length < 2) return null;
      const fingerprints = families.map((goal) => comparePayloadSplitFingerprint(payloads?.[goal]));
      if (fingerprints.some((fingerprint) => !fingerprint)) return "missing";
      return new Set(fingerprints).size > 1 ? "differ" : null;
    }

    const SPLIT_NOTES = {
      differ: "ML and DL rows were evaluated on different row partitions (split fingerprints differ); rerun both with the same seed and evaluation settings to rank them together.",
      missing: "ML and DL rows cannot be confirmed to use the same row partitions (a split fingerprint is missing); rerun both with the same seed and evaluation settings to rank them together.",
    };

    function splitMismatchNote(board) {
      return SPLIT_NOTES[board?.splitMismatchReason] || SPLIT_NOTES.differ;
    }
    const LOCKED_TEST_RANKING_NOTE = "Ranking uses development-set cross-validation; the locked-test C-index of the rank-1 model is the independent estimate to report.";

    function comparePayloadGroupId(payload) {
      return String(payload?._client_compare_group_id || payload?.analysis?._client_compare_group_id || "").trim();
    }

    function benchmarkCompareRows(goal, { currentOnly = false } = {}) {
      const payload = benchmarkComparePayload(goal, { currentOnly });
      return comparisonRowsFromPayload(payload);
    }

    function excludedModelsFromPayload(payload) {
      const explicit = Array.isArray(payload?.analysis?.excluded_models)
        ? payload.analysis.excluded_models.map((value) => String(value || "").trim()).filter(Boolean)
        : [];
      const errors = Array.isArray(payload?.analysis?.errors) ? payload.analysis.errors : [];
      const erroredModels = errors.map((entry) => String(entry?.model || "").trim()).filter(Boolean);
      return [...new Set([...explicit, ...erroredModels])];
    }

    function benchmarkExcludedModels(goal, { currentOnly = false } = {}) {
      const payload = benchmarkComparePayload(goal, { currentOnly });
      return excludedModelsFromPayload(payload);
    }

    function excludedModelsCopy(goal, models, { sourceLabel = "current compare run" } = {}) {
      if (!Array.isArray(models) || !models.length) return "";
      return `Excluded from ${sourceLabel} ${benchmarkGoalMeta(goal).label} compare run: ${models.join(", ")}.`;
    }

    function benchmarkStarterActionMarkup() {
      return `
        <div class="button-row compact benchmark-row-actions">
          <button
            class="button ghost compact-btn"
            type="button"
            data-benchmark-model="rsf"
            data-benchmark-mode="single"
          >
            Open model controls
          </button>
        </div>
      `;
    }

    const EXPERIMENTAL_MODELS = new Set(["Survival Transformer", "Survival VAE"]);

    function showBenchmarkStarterAction() {
      return true;
    }

    function benchmarkRowsFromPayload(goal, payload, { statusOverride = null, paramsSource = "current" } = {}) {
      if (!payload) return [];
      const meta = benchmarkGoalMeta(goal);
      const status = statusOverride || benchmarkResultLabel(goal);
      const runGroupId = comparePayloadGroupId(payload);
      return comparisonRowsFromPayload(payload).map((row, index) => {
        // A null rank means the backend deliberately left this row unranked; never coerce it to 0.
        const rankProvided = Object.prototype.hasOwnProperty.call(row || {}, "rank");
        const ranked = rankProvided ? hasBackendRank(row.rank) : true;
        return {
          family: meta.label,
          familyTab: meta.tab,
          model: row.model,
          c_index: row.c_index,
          numericCIndex: benchmarkMetricNumber(row.c_index),
          hasLockedTest: Object.prototype.hasOwnProperty.call(row || {}, "locked_test_c_index"),
          locked_test_c_index: row.locked_test_c_index,
          evaluation_mode: row.evaluation_mode || payload?.analysis?.evaluation_mode || "",
          sourceRank: rankProvided ? (ranked ? Number(row.rank) : null) : index + 1,
          comparableForRanking: ranked && row.comparable_for_ranking !== false && benchmarkMetricNumber(row.c_index) !== null,
          status,
          sourceMode: "compare",
          runGroupId,
          paramsSource,
        };
      });
    }

    function benchmarkExcludedRowsForPayload(goal, payload, { statusOverride = null, paramsSource = "current" } = {}) {
      if (!payload) return [];
      const meta = benchmarkGoalMeta(goal);
      const status = statusOverride || benchmarkResultLabel(goal);
      const runGroupId = comparePayloadGroupId(payload);
      const comparisonRows = comparisonRowsFromPayload(payload);
      const seenModels = new Set(comparisonRows.map((row) => String(row?.model || "").trim().toLowerCase()).filter(Boolean));
      const errorRows = Array.isArray(payload?.analysis?.errors) ? payload.analysis.errors : [];
      const rows = [];

      errorRows.forEach((entry, index) => {
        const modelLabel = String(entry?.model || "").trim();
        if (!modelLabel) return;
        const normalized = modelLabel.toLowerCase();
        if (seenModels.has(normalized)) return;
        seenModels.add(normalized);
        rows.push({
          family: meta.label,
          familyTab: meta.tab,
          model: modelLabel,
          c_index: null,
          numericCIndex: null,
          evaluation_mode: payload?.analysis?.evaluation_mode || "",
          sourceRank: comparisonRows.length + index + 1,
          comparableForRanking: false,
          status: `${status} (Excluded)`,
          excluded: true,
          exclusionReason: String(entry?.error || "").trim() || "This model did not return a compare row for the current run.",
          sourceMode: "compare",
          runGroupId,
          paramsSource,
        });
      });

      const explicit = Array.isArray(payload?.analysis?.excluded_models) ? payload.analysis.excluded_models : [];
      explicit.forEach((value, index) => {
        const modelLabel = String(value || "").trim();
        if (!modelLabel) return;
        const normalized = modelLabel.toLowerCase();
        if (seenModels.has(normalized)) return;
        seenModels.add(normalized);
        rows.push({
          family: meta.label,
          familyTab: meta.tab,
          model: modelLabel,
          c_index: null,
          numericCIndex: null,
          evaluation_mode: payload?.analysis?.evaluation_mode || "",
          sourceRank: comparisonRows.length + errorRows.length + index + 1,
          comparableForRanking: false,
          status: `${status} (Excluded)`,
          excluded: true,
          exclusionReason: "This model did not produce a leaderboard row for the current compare run.",
          sourceMode: "compare",
          runGroupId,
          paramsSource,
        });
      });

      return rows;
    }

    function benchmarkExcludedRows(goal, { currentOnly = false } = {}) {
      const payload = benchmarkComparePayload(goal, { currentOnly });
      return benchmarkExcludedRowsForPayload(goal, payload, { paramsSource: currentOnly ? "current" : "latest" });
    }

    function benchmarkSingleRunSummary(goal, payload) {
      const requestConfig = payload?.request_config || payload?.analysis?.request_config || {};
      if (goal === "ml") {
        const stats = payload?.analysis?.model_stats || {};
        const label = mlModelLabel(requestConfig.model_type || "ML model");
        return {
          title: "Latest single run",
          text: `${label} ${stats.metric_name || "C-index"}=${formatValue(stats.c_index)} on ${benchmarkEvaluationLabel(stats.evaluation_mode)} evaluation. Use Compare All if you want this family to appear in the unified leaderboard.`,
          chips: [
            `Eval: ${benchmarkEvaluationLabel(stats.evaluation_mode)}`,
            `Features: ${formatValue(stats.n_features)}`,
            `N: ${formatValue(stats.n_patients)}`,
          ],
        };
      }
      const stats = payload?.analysis || {};
      const label = dlModelLabel(requestConfig.model_type || "deep model");
      return {
        title: "Latest single run",
        text: `${label} C-index=${formatValue(stats.c_index)} on ${benchmarkEvaluationLabel(stats.evaluation_mode)} evaluation. Use Compare All if you want this family to appear in the unified leaderboard.`,
        chips: [
          `Eval: ${benchmarkEvaluationLabel(stats.evaluation_mode)}`,
          `Epochs: ${formatValue(stats.epochs_trained || stats.epochs)}`,
          `Features: ${formatValue(stats.n_features)}`,
        ],
      };
    }

    function unifiedBenchmarkRows({ currentOnly = true } = {}) {
      const families = ["ml", "dl"];
      const rows = families.flatMap((goal) => benchmarkRowsFromPayload(
        goal,
        benchmarkComparePayload(goal, { currentOnly }),
        { paramsSource: currentOnly ? "current" : "latest" },
      ));
      return rows.sort((left, right) => {
        const comparableDelta = Number(Boolean(right.comparableForRanking)) - Number(Boolean(left.comparableForRanking));
        if (comparableDelta !== 0) return comparableDelta;
        const safeLeft = left.numericCIndex ?? -Infinity;
        const safeRight = right.numericCIndex ?? -Infinity;
        if (safeRight !== safeLeft) return safeRight - safeLeft;
        return left.family.localeCompare(right.family) || left.model.localeCompare(right.model);
      });
    }

    function familyGroupedRows(rows) {
      return [...rows].sort((left, right) => {
        const familyDelta = left.family.localeCompare(right.family);
        if (familyDelta !== 0) return familyDelta;
        const leftRanked = left.sourceRank !== null && left.sourceRank !== undefined && !left.excluded;
        const rightRanked = right.sourceRank !== null && right.sourceRank !== undefined && !right.excluded;
        if (leftRanked !== rightRanked) return leftRanked ? -1 : 1;
        if (leftRanked && left.sourceRank !== right.sourceRank) return left.sourceRank - right.sourceRank;
        return String(left.model).localeCompare(String(right.model));
      });
    }

    function pendingFamilyText(board) {
      const pending = (board?.pendingFamilies ?? []).map((goal) => benchmarkGoalMeta(goal).label);
      return pending.length ? pending.join(" and ") : "the remaining compare runs";
    }

    function benchmarkMethodologyNotes(board) {
      const visibleFamilies = Array.isArray(board?.visibleFamilies) ? board.visibleFamilies : [];
      if (!(visibleFamilies.includes("ml") && visibleFamilies.includes("dl"))) return [];
      return [
        "Cross-family partial-likelihood models do not share one tie method: Cox PH and LASSO-Cox use Efron, whereas DeepSurv, Survival Transformer, and Survival VAE use Breslow. Treat small cross-family C-index gaps cautiously when many event times are tied.",
        "ML comparison rows may include IBS / Brier Skill Score, but DL comparison rows currently report C-index only, so calibration/error comparisons are not symmetric across families.",
      ];
    }

    function hasUnifiedCoverage(families) {
      return Array.isArray(families) && families.length === 2;
    }

    function benchmarkBoardState() {
      const rawCurrentRows = unifiedBenchmarkRows({ currentOnly: true });
      const currentFamilies = ["ml", "dl"].filter((goal) => benchmarkCompareRows(goal, { currentOnly: true }).length > 0);
      const snapshotPayloads = {
        ml: benchmarkSnapshotComparePayload("ml"),
        dl: benchmarkSnapshotComparePayload("dl"),
      };
      const snapshotRowsRaw = ["ml", "dl"].flatMap((goal) => benchmarkRowsFromPayload(
        goal,
        snapshotPayloads[goal],
        { statusOverride: "Stale reference", paramsSource: "snapshot" },
      ));
      const snapshotFamilies = ["ml", "dl"].filter((goal) => comparisonRowsFromPayload(snapshotPayloads[goal]).length > 0);
      const staleFamilies = ["ml", "dl"].filter((goal) => benchmarkCompareRows(goal).length > 0 && benchmarkCompareRows(goal, { currentOnly: true }).length === 0);
      const excludedByFamily = Object.fromEntries(
        ["ml", "dl"].map((goal) => [goal, benchmarkExcludedModels(goal, { currentOnly: true })]),
      );
      const snapshotExcludedByFamily = Object.fromEntries(
        ["ml", "dl"].map((goal) => [goal, excludedModelsFromPayload(snapshotPayloads[goal])]),
      );
      const currentExcludedRows = ["ml", "dl"].flatMap((goal) => benchmarkExcludedRows(goal, { currentOnly: true }));
      const snapshotExcludedRows = ["ml", "dl"].flatMap((goal) => benchmarkExcludedRowsForPayload(
        goal,
        snapshotPayloads[goal],
        { statusOverride: "Stale reference", paramsSource: "snapshot" },
      ));
      const evaluationModes = [
        ...new Set(rawCurrentRows.map((row) => String(row.evaluation_mode || "").trim().toLowerCase()).filter(Boolean)),
      ];
      const currentGroupIds = [
        ...new Set(rawCurrentRows.map((row) => String(row.runGroupId || "").trim()).filter(Boolean)),
      ];
      const snapshotEvaluationModes = [
        ...new Set(snapshotRowsRaw.map((row) => String(row.evaluation_mode || "").trim().toLowerCase()).filter(Boolean)),
      ];
      const snapshotGroupIds = [
        ...new Set(snapshotRowsRaw.map((row) => String(row.runGroupId || "").trim()).filter(Boolean)),
      ];
      const hasMixedEvaluation = evaluationModes.length > 1;
      const hasMixedRunGroups = currentFamilies.length > 1 && currentGroupIds.length > 1;
      const snapshotHasMixedEvaluation = snapshotEvaluationModes.length > 1;
      const snapshotHasMixedRunGroups = snapshotFamilies.length > 1 && snapshotGroupIds.length > 1;
      const currentPayloads = {
        ml: benchmarkComparePayload("ml", { currentOnly: true }),
        dl: benchmarkComparePayload("dl", { currentOnly: true }),
      };
      const splitMismatchReason = familySplitMismatch(currentFamilies, currentPayloads);
      const snapshotSplitMismatchReason = familySplitMismatch(snapshotFamilies, snapshotPayloads);
      const hasSplitMismatch = Boolean(splitMismatchReason);
      const snapshotHasSplitMismatch = Boolean(snapshotSplitMismatchReason);
      const currentRows = (hasMixedEvaluation || hasMixedRunGroups || hasSplitMismatch) ? familyGroupedRows(rawCurrentRows) : rawCurrentRows;
      const snapshotRows = (snapshotHasMixedEvaluation || snapshotHasMixedRunGroups || snapshotHasSplitMismatch) ? familyGroupedRows(snapshotRowsRaw) : snapshotRowsRaw;
      const showingStaleBoard = snapshotRows.length > 0 && hasUnifiedCoverage(snapshotFamilies) && (!hasUnifiedCoverage(currentFamilies) || hasMixedRunGroups);
      const hiddenStaleFamilies = showingStaleBoard ? [] : staleFamilies;
      const visibleRows = showingStaleBoard ? snapshotRows : currentRows;
      const visibleExcludedRows = showingStaleBoard ? snapshotExcludedRows : currentExcludedRows;
      const visibleFamilies = showingStaleBoard ? snapshotFamilies : currentFamilies;
      const visibleEvaluationModes = showingStaleBoard ? snapshotEvaluationModes : evaluationModes;
      const visibleHasMixedEvaluation = showingStaleBoard ? snapshotHasMixedEvaluation : hasMixedEvaluation;
      const visibleHasMixedRunGroups = showingStaleBoard ? snapshotHasMixedRunGroups : hasMixedRunGroups;
      const visibleHasSplitMismatch = showingStaleBoard ? snapshotHasSplitMismatch : hasSplitMismatch;
      const withholdCrossFamilyRanking = visibleHasMixedEvaluation || visibleHasMixedRunGroups || visibleHasSplitMismatch;
      const rankingRows = withholdCrossFamilyRanking ? [] : visibleRows.filter((row) => row.comparableForRanking);
      const plottableRows = withholdCrossFamilyRanking ? [] : visibleRows.filter((row) => row.numericCIndex !== null);
      const tableRows = withholdCrossFamilyRanking
        ? familyGroupedRows([...visibleRows, ...visibleExcludedRows])
        : [...visibleRows, ...visibleExcludedRows];
      const hasLockedTest = visibleRows.some((row) => row.hasLockedTest);
      const rawVisibleRows = showingStaleBoard ? snapshotRowsRaw : rawCurrentRows;
      const missingMetricCount = rawVisibleRows.filter((row) => row.numericCIndex === null).length;
      const nonComparableCount = rawVisibleRows.filter((row) => !row.comparableForRanking).length;
      const predictiveBusy = isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl");
      const pendingFamilies = ["ml", "dl"].filter((goal) => !currentFamilies.includes(goal));
      return {
        currentRows,
        visibleRows,
        visibleExcludedRows,
        tableRows,
        currentFamilies,
        visibleFamilies,
        staleFamilies,
        hiddenStaleFamilies,
        rankingRows,
        plottableRows,
        evaluationModes: visibleEvaluationModes,
        hasMixedEvaluation: visibleHasMixedEvaluation,
        hasMixedRunGroups,
        visibleHasMixedRunGroups,
        hasSplitMismatch,
        visibleHasSplitMismatch,
        splitMismatchReason: showingStaleBoard ? snapshotSplitMismatchReason : splitMismatchReason,
        withholdCrossFamilyRanking,
        hasLockedTest,
        missingMetricCount,
        nonComparableCount,
        predictiveBusy,
        pendingFamilies,
        excludedByFamily: showingStaleBoard ? snapshotExcludedByFamily : excludedByFamily,
        showingStaleBoard,
        currentPayloads,
        snapshotPayloads,
        snapshotRowCounts: {
          ml: comparisonRowsFromPayload(snapshotPayloads.ml).length,
          dl: comparisonRowsFromPayload(snapshotPayloads.dl).length,
        },
      };
    }

    async function renderUnifiedBenchmarkPlot(board) {
      if (!refs.benchmarkComparisonPlot || !refs.benchmarkPlotNote) return;
      if (board.predictiveBusy) {
        refs.benchmarkPlotNote.textContent = `Waiting on ${pendingFamilyText(board)} before charting the shared C-index board.`;
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>The chart will publish after both model families finish.</span></div>');
        return;
      }
      if (!board.visibleRows.length) {
        refs.benchmarkPlotNote.textContent = board.staleFamilies.length
          ? "Current settings no longer match the last compare run. Rerun Compare All Models to rebuild the cross-family board."
          : "Run Compare All Models to chart ML and DL together on one board.";
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Run Compare All Models to compare ML and DL C-index values on one chart.</span></div>');
        return;
      }
      if (board.hasMixedEvaluation) {
        refs.benchmarkPlotNote.textContent = `${board.showingStaleBoard ? "Showing the last Compare All board as a stale reference. " : ""}Unified chart hidden because visible ML and DL compare rows use mixed evaluation paths (${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}). Rerun both families with the same evaluation mode to publish one shared C-index axis.`;
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Unified chart is hidden until ML and DL compare rows use the same evaluation mode.</span></div>');
        return;
      }
      if (board.visibleHasMixedRunGroups) {
        refs.benchmarkPlotNote.textContent = "Unified chart hidden because the visible ML and DL rows come from different compare runs. Rerun Compare All Models to publish one atomic cross-family board.";
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Unified chart is hidden until one Compare All run produces both ML and DL families together.</span></div>');
        return;
      }
      if (board.visibleHasSplitMismatch) {
        refs.benchmarkPlotNote.textContent = `${board.showingStaleBoard ? "Showing the last Compare All board as a stale reference. " : ""}Unified chart hidden because ${splitMismatchNote(board)}`;
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Unified chart is hidden until ML and DL are evaluated on the same row partitions.</span></div>');
        return;
      }
      if (!board.plottableRows.length) {
        refs.benchmarkPlotNote.textContent = `${board.showingStaleBoard ? "Showing the last Compare All board as a stale reference. " : ""}Visible comparison rows exist, but none reported a numeric C-index that can be charted. Review the table below.`;
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>No numeric C-index values are available to chart for the current board.</span></div>');
        return;
      }

      const noteParts = [
        board.showingStaleBoard
          ? "Showing the last Compare All board as a stale reference."
          : `Showing ${board.plottableRows.length} current screening rows on one C-index axis${board.evaluationModes[0] ? ` using ${benchmarkEvaluationLabel(board.evaluationModes[0])} evaluation.` : "."}`,
      ];
      if (board.showingStaleBoard) {
        noteParts.push("Current settings no longer match these rows. Rerun Compare All Models to refresh the board.");
      }
      if (board.hiddenStaleFamilies.length) {
        noteParts.push(`Stale compare rows from ${board.hiddenStaleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden until rerun.`);
      }
      if (board.missingMetricCount) {
        noteParts.push(`Omitted ${board.missingMetricCount} row(s) without a numeric C-index.`);
      }
      // The ranking, interval and methodology notes sit once, under the leaderboard.
      refs.benchmarkPlotNote.textContent = noteParts.join(" ");

      const intervals = board.intervals?.status === "ready" ? board.intervals.result : null;
      const intervalByModel = new Map((intervals?.rows || []).map((row) => [String(row.model), row]));
      const valueOf = (row) => {
        const interval = intervalByModel.get(String(row.model));
        if (interval?.c_index != null) return Number(interval.c_index);
        return board.hasLockedTest ? benchmarkMetricNumber(row.locked_test_c_index) : row.numericCIndex;
      };
      // Best model on top; each dot carries its bootstrap interval when the board has one.
      const ordered = board.plottableRows.filter((row) => Number.isFinite(valueOf(row))).sort((left, right) => valueOf(left) - valueOf(right));
      const label = (row) => `${row.model} (${benchmarkRowFamilyMeta(row).familyShortLabel})`;
      const traces = ["ml", "dl"].map((family) => {
        const members = ordered.filter((row) => benchmarkRowFamilyMeta(row).familyTab === family);
        const intervalsOf = members.map((row) => intervalByModel.get(String(row.model)) || null);
        return {
          type: "scatter",
          mode: "markers",
          name: family === "ml" ? "Classical ML" : "Deep Learning",
          x: members.map(valueOf),
          y: members.map(label),
          marker: {
            size: 11,
            color: family === "ml" ? "rgba(47, 101, 217, 0.95)" : "rgba(219, 126, 21, 0.95)",
            line: { color: "#1a2332", width: 1 },
          },
          error_x: intervals
            ? {
              type: "data",
              symmetric: false,
              array: members.map((row, index) => (intervalsOf[index]?.c_index_ci?.[1] ?? valueOf(row)) - valueOf(row)),
              arrayminus: members.map((row, index) => valueOf(row) - (intervalsOf[index]?.c_index_ci?.[0] ?? valueOf(row))),
              thickness: 1.6,
              width: 0,
              color: family === "ml" ? "rgba(34, 72, 156, 0.9)" : "rgba(156, 86, 15, 0.9)",
            }
            : undefined,
          customdata: members.map((row, index) => [
            intervalRangeText(intervalsOf[index]?.c_index_ci),
            deltaText(intervalsOf[index]),
            benchmarkEvaluationLabel(row.evaluation_mode),
          ]),
          hovertemplate: [
            "<b>%{y}</b>",
            `${board.hasLockedTest ? "Locked-test C-index" : "C-index"}: %{x:.3f} %{customdata[0]}`,
            "%{customdata[1]}",
            "Evaluation: %{customdata[2]}",
            "<extra></extra>",
          ].join("<br>"),
        };
      }).filter((trace) => trace.x.length);
      const lows = ordered.map((row) => intervalByModel.get(String(row.model))?.c_index_ci?.[0] ?? valueOf(row));
      const highs = ordered.map((row) => intervalByModel.get(String(row.model))?.c_index_ci?.[1] ?? valueOf(row));
      // The chance line (0.5) joins the axis only when an interval comes near it.
      const showChance = Math.min(...lows) < 0.58;
      const low = showChance ? Math.min(...lows, 0.5) : Math.min(...lows);
      const high = Math.max(...highs);
      const reference = intervalByModel.get("Cox PH");
      const shapes = showChance
        ? [{ type: "line", yref: "paper", y0: 0, y1: 1, xref: "x", x0: 0.5, x1: 0.5, line: { color: "rgba(90, 103, 118, 0.7)", width: 1.2, dash: "dot" } }]
        : [];
      const annotations = showChance
        ? [{ xref: "x", x: 0.5, yref: "paper", y: 1, yanchor: "bottom", text: "0.5", showarrow: false, font: { size: 11, color: "rgba(90, 103, 118, 0.95)" } }]
        : [];
      if (reference?.c_index != null) {
        shapes.push({ type: "line", yref: "paper", y0: 0, y1: 1, xref: "x", x0: reference.c_index, x1: reference.c_index, line: { color: "rgba(34, 72, 156, 0.6)", width: 1.2, dash: "dash" } });
        annotations.push({ xref: "x", x: reference.c_index, yref: "paper", y: 1, yanchor: "bottom", text: "Cox PH", showarrow: false, font: { size: 11, color: "rgba(34, 72, 156, 0.95)" } });
      }
      const layout = {
        title: {
          text: board.hasLockedTest ? "Locked-test C-index with 95% intervals" : "C-index on the same test patients, with 95% intervals",
          x: 0.02,
          xanchor: "left",
          font: { family: "Source Serif 4, serif", size: 20, color: "#1a2332" },
        },
        font: { family: "Sora, sans-serif", size: 13, color: "#1a2332" },
        height: Math.max(300, 120 + 34 * ordered.length),
        margin: { l: 24, r: 24, t: 70, b: 56 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        xaxis: {
          title: { text: board.hasLockedTest ? "Locked-test C-index" : "C-index" },
          range: [low - 0.02, Math.min(1, high + 0.02)],
          gridcolor: "rgba(27, 39, 51, 0.08)",
          zeroline: false,
        },
        // Categories in C-index order across both families (best on top), not grouped by family.
        yaxis: { automargin: true, tickfont: { size: 12 }, categoryorder: "array", categoryarray: ordered.map(label) },
        shapes,
        annotations,
        showlegend: true,
        legend: { orientation: "h", x: 1, xanchor: "right", y: 1.02, yanchor: "bottom" },
      };

      refs.benchmarkComparisonPlot.classList.remove("hidden");
      purgePlot(refs.benchmarkComparisonPlot);
      refs.benchmarkComparisonPlot.innerHTML = "";
      await Plotly.newPlot(
        refs.benchmarkComparisonPlot,
        traces,
        plotLayoutConfig(layout, "benchmark_comparison"),
        plotConfig("benchmark_comparison"),
      );
      stabilizePlotShellHeight(refs.benchmarkComparisonPlot);
    }

    function intervalRangeText(interval) {
      return Array.isArray(interval) && interval[0] != null
        ? `(95% ${Number(interval[0]).toFixed(3)} to ${Number(interval[1]).toFixed(3)})`
        : "";
    }

    function deltaText(row) {
      if (!row || row.delta_vs_reference == null) return row && String(row.model) === "Cox PH" ? "Reference model" : "";
      const sign = Number(row.delta_vs_reference) >= 0 ? "+" : "";
      return `ΔC vs Cox PH: ${sign}${Number(row.delta_vs_reference).toFixed(3)} ${intervalRangeText(row.delta_ci)}`;
    }

    // The test-set predictions behind the visible board, when its families scored one shared test set.
    function boardPredictionBlocks(board) {
      if (board.withholdCrossFamilyRanking || board.predictiveBusy || !board.visibleFamilies.length) return null;
      const payloads = board.showingStaleBoard ? board.snapshotPayloads : board.currentPayloads;
      const blocks = board.visibleFamilies.map((goal) => {
        const analysis = payloads?.[goal]?.analysis || {};
        return board.hasLockedTest ? analysis.locked_test_predictions : analysis.test_predictions;
      });
      return blocks.every((block) => Array.isArray(block?.row_ids) && block.row_ids.length) ? blocks : null;
    }

    function boardIntervalKey(board) {
      const payloads = board.showingStaleBoard ? board.snapshotPayloads : board.currentPayloads;
      return [
        board.showingStaleBoard ? "snapshot" : "current",
        board.hasLockedTest ? "locked" : "holdout",
        ...board.visibleFamilies.map((goal) => {
          const payload = payloads?.[goal];
          const rows = comparisonRowsFromPayload(payload).map((row) => `${row.model}=${row.c_index}`).join(",");
          return `${goal}:${comparePayloadSplitFingerprint(payload)}:${comparePayloadGroupId(payload)}:${rows}`;
        }),
      ].join("|");
    }

    // Bootstrap intervals for the visible board, fetched once per board and kept in runtime.
    function boardIntervals(board) {
      const blocks = boardPredictionBlocks(board);
      if (!blocks) return null;
      const key = boardIntervalKey(board);
      if (runtime.benchmarkIntervals?.key === key) return runtime.benchmarkIntervals;
      runtime.benchmarkIntervals = { key, status: "loading" };
      fetchJSON("/api/model-comparison-intervals", { method: "POST", body: JSON.stringify({ predictions: blocks }) })
        .then((result) => {
          if (runtime.benchmarkIntervals?.key !== key) return;
          runtime.benchmarkIntervals = { key, status: "ready", result };
          requestBoardRender();
        })
        .catch((error) => {
          if (runtime.benchmarkIntervals?.key !== key) return;
          runtime.benchmarkIntervals = { key, status: "error", error: error?.message || String(error) };
          requestBoardRender();
        });
      return runtime.benchmarkIntervals;
    }

    function intervalNote(board) {
      const intervals = board.intervals;
      if (intervals?.status === "ready") {
        const result = intervals.result || {};
        return `Intervals are 95% bootstrap intervals over the ${formatValue(result.n)} ${board.hasLockedTest ? "locked-test" : "test"} patients all models share (${formatValue(result.events)} events).`;
      }
      if (intervals?.status === "loading") return "Computing bootstrap intervals for the C-index of each model.";
      if (intervals?.status === "error") return `Bootstrap intervals are unavailable: ${intervals.error}`;
      return "Leaderboard order is a point-estimate screening view; repeated cross-validation reports the spread across folds instead of intervals.";
    }

    function intervalDetail(board) {
      if (board.intervals?.status !== "ready") return "";
      return "ΔC vs Cox PH is paired: every draw scores all models on the same resampled patients, so a model whose ΔC interval contains 0 is not distinguishable from Cox PH on this split.";
    }

    // One line stays in view; everything a reader needs only when writing up folds into "Method notes".
    function tableNoteMarkup(lead, details) {
      const leadText = lead.filter(Boolean).join(" ");
      const items = details.filter(Boolean);
      if (!items.length) return escapeHtml(leadText);
      return `${escapeHtml(leadText)}<details class="benchmark-note-details"><summary>Method notes (${items.length})</summary><ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>`;
    }

    function buildBenchmarkSummaryContent(board, hasAnyResult, currentMlRows, currentDlRows) {
      const completedFamiliesLabel = board.currentFamilies.length
        ? board.currentFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")
        : "none yet";
      const pendingFamiliesLabel = board.pendingFamilies.length
        ? board.pendingFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")
        : "none";
      const coverageText = board.visibleFamilies.length === 2
        ? "Both model families are currently represented."
        : (board.visibleFamilies.length === 1
          ? `${benchmarkGoalMeta(board.visibleFamilies[0]).label} is currently represented.`
          : "No current compare rows are available.");
      const cautionParts = [];
      if (board.hiddenStaleFamilies.length) {
        cautionParts.push(`Stale compare rows from ${board.hiddenStaleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden until rerun.`);
      }
      if (board.hasMixedEvaluation) {
        cautionParts.push(`Current compare rows use mixed evaluation paths (${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}).`);
      }
      if (board.missingMetricCount) {
        cautionParts.push(`${board.missingMetricCount} row(s) have no numeric C-index.`);
      }
      if (board.hasLockedTest && !board.withholdCrossFamilyRanking) cautionParts.push(LOCKED_TEST_RANKING_NOTE);
      cautionParts.push(...benchmarkMethodologyNotes(board));
      ["ml", "dl"].forEach((goal) => {
        const copy = excludedModelsCopy(goal, board.excludedByFamily?.[goal], {
          sourceLabel: board.showingStaleBoard ? "the last complete snapshot" : "the current",
        });
        if (copy) cautionParts.push(copy);
      });
      const cautionSuffix = cautionParts.length ? ` ${cautionParts.join(" ")}` : "";

      if (board.predictiveBusy) {
        return {
          chips: [
            `Completed families: ${completedFamiliesLabel}`,
            `Pending families: ${pendingFamiliesLabel}`,
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
          ],
          status: "Running",
          title: "Unified predictive comparison in progress",
          text: `Compare All Models is still running. Completed families: ${completedFamiliesLabel}. Waiting on ${pendingFamiliesLabel} before publishing the final unified board.${cautionSuffix}`,
          tone: "running",
        };
      }

      if (!hasAnyResult) {
        return {
          chips: [
            "Board not built yet",
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
          ],
          status: "Not run",
          title: "Predictive workspace not run yet",
          text: "Use Compare All Models once to benchmark the full predictive stack, or test one selected model directly.",
          tone: "idle",
        };
      }

      if (board.showingStaleBoard) {
        return {
          chips: [
            "Board freshness: stale reference",
            `Last board rows: ${board.visibleRows.length}`,
            `ML rows in last snapshot: ${board.snapshotRowCounts?.ml || 0}`,
            `DL rows in last snapshot: ${board.snapshotRowCounts?.dl || 0}`,
          ],
          status: "Stale reference",
          title: "Previous predictive screening board",
          text: `Current settings no longer match at least one family from the last Compare All snapshot, so this screen is showing that last complete cross-family board as reference only. Rerun Compare All Models to refresh it.${cautionSuffix}`,
          tone: "warning",
        };
      }


      if (!board.visibleRows.length) {
        return {
          chips: [
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
          ],
          status: "Needs rerun",
          title: "Predictive board needs rerun",
          text: "Stored predictive results exist, but they no longer match the current outcome, feature, or evaluation settings. Rerun Compare All Models to rebuild the board.",
          tone: "warning",
        };
      }

      if (board.hasMixedEvaluation) {
        return {
          chips: [
            `Families represented: ${board.visibleFamilies.length}`,
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
            `Evaluation paths: ${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(" / ")}`,
          ],
          status: "Needs alignment",
          title: "Predictive results need alignment",
          text: `Current compare rows are grouped by family only. Unified ranking and charting are hidden until ML and DL are rerun with the same evaluation mode. ${coverageText}${cautionSuffix}`,
          tone: "warning",
        };
      }

      if (board.visibleHasSplitMismatch) {
        return {
          chips: [
            `Families represented: ${board.visibleFamilies.length}`,
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
          ],
          status: "Needs alignment",
          title: board.splitMismatchReason === "missing" ? "ML and DL row partitions could not be verified" : "ML and DL used different row partitions",
          text: `Current compare rows are grouped by family only. ${splitMismatchNote(board)} ${coverageText}${cautionSuffix}`,
          tone: "warning",
        };
      }

      if (board.visibleHasMixedRunGroups) {
        return {
          chips: [
            `Families represented: ${board.visibleFamilies.length}`,
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
          ],
          status: "Needs alignment",
          title: "Predictive results come from different compare runs",
          text: `Visible ML and DL screening rows were produced by different compare runs, so SurvStudio is withholding one shared ranking board. Rerun Compare All Models to rebuild one atomic cross-family benchmark.${cautionSuffix}`,
          tone: "warning",
        };
      }

      const needsReview = board.visibleFamilies.length < 2 || board.missingMetricCount || board.nonComparableCount || board.visibleExcludedRows.length;
      return {
        chips: [
          `Families represented: ${board.visibleFamilies.length}`,
          `Current board rows: ${board.visibleRows.length}`,
          `ML rows ready: ${currentMlRows}`,
          `DL rows ready: ${currentDlRows}`,
        ],
        status: needsReview ? "Needs review" : "Board ready",
        title: "Predictive screening board",
        text: `Showing ${board.visibleRows.length} successful current screening row(s) across the predictive workspace. ${coverageText} The ordering is a convenience screen, not a strict head-to-head benchmark, because ML and DL still run through family-specific evaluation pipelines.${cautionSuffix}`,
        tone: needsReview ? "warning" : "current",
      };
    }

    function renderUnifiedBenchmarkSummary(board) {
      if (!refs.benchmarkSummaryGrid) return;
      const hasAnyResult = Boolean(goalPayload("ml") || goalPayload("dl") || runtime.compareCache?.unified?.ml || runtime.compareCache?.unified?.dl);
      const currentMlRows = benchmarkCompareRows("ml", { currentOnly: true }).length;
      const currentDlRows = benchmarkCompareRows("dl", { currentOnly: true }).length;
      const summary = buildBenchmarkSummaryContent(board, hasAnyResult, currentMlRows, currentDlRows);
      refs.benchmarkSummaryGrid.innerHTML = `
        <article class="benchmark-family-card tone-${escapeHtml(summary.tone)} benchmark-family-card-wide">
          <div class="benchmark-family-head">
            <div>
              <span class="benchmark-family-badge">${escapeHtml(summary.status)}</span>
              <h3>Predictive Overview</h3>
            </div>
          </div>
          <strong class="benchmark-family-title">${escapeHtml(summary.title)}</strong>
          <p class="benchmark-family-copy">${escapeHtml(summary.text)}</p>
          <div class="dataset-preset-chips">${summary.chips.map((label) => `<span class="dataset-preset-chip">${escapeHtml(label)}</span>`).join("")}</div>
          ${!hasAnyResult && showBenchmarkStarterAction() ? benchmarkStarterActionMarkup() : ""}
        </article>
      `;
    }

    function renderUnifiedBenchmarkTable(board) {
      if (!refs.benchmarkComparisonShell || !refs.benchmarkTableNote) return;
      if (board.predictiveBusy) {
        refs.benchmarkTableNote.textContent = `Waiting on ${pendingFamilyText(board)} before publishing the leaderboard.`;
        refs.benchmarkComparisonShell.innerHTML = '<div class="empty-state">Partial leaderboard rows stay hidden until both model families finish.</div>';
        return;
      }
      if (!board.visibleRows.length) {
        refs.benchmarkTableNote.textContent = board.staleFamilies.length
          ? "Stored compare rows are stale and hidden. Rerun Compare All Models to rebuild the shared board with the current settings."
          : "Run Compare All Models to build a shared leaderboard across classical ML and deep learning.";
        refs.benchmarkComparisonShell.innerHTML = showBenchmarkStarterAction()
          ? `
            <div class="empty-state">
              <span>Run Compare All Models to build a shared leaderboard across classical ML and deep learning.</span>
              ${benchmarkStarterActionMarkup()}
            </div>
          `
          : '<div class="empty-state">Run Compare All Models to build a shared leaderboard across classical ML and deep learning.</div>';
        return;
      }

      const presentFamilies = [...new Set(board.visibleRows.map((row) => benchmarkRowFamilyMeta(row).familyLabel))];
      const leadParts = [
        board.hasMixedEvaluation
          ? "Visible compare rows are grouped by family because evaluation modes differ. No cross-family ranking is published."
          : board.visibleHasMixedRunGroups
            ? "Visible compare rows are grouped by family because ML and DL come from different compare runs. No cross-family ranking is published."
          : board.visibleHasSplitMismatch
            ? `Visible compare rows are grouped by family: ${splitMismatchNote(board)} No cross-family ranking is published.`
          : (presentFamilies.length === 2
            ? (board.showingStaleBoard
              ? `Showing ${board.visibleRows.length} stale screening rows from the last complete Compare All snapshot.`
              : `Showing ${board.visibleRows.length} current screening rows from the latest ML and DL comparison outputs.`)
            : `Showing ${board.visibleRows.length} ${board.showingStaleBoard ? "stale" : "current"} screening row(s) from ${presentFamilies[0] ?? "one family"} only.`),
      ];
      if (board.showingStaleBoard) {
        leadParts.push("Current settings no longer match these rows. Rerun Compare All Models to refresh the leaderboard.");
      }
      if (board.hasLockedTest) leadParts.push(LOCKED_TEST_RANKING_NOTE);
      leadParts.push(intervalNote(board));
      const detailParts = [intervalDetail(board)];
      if (board.visibleExcludedRows.length) {
        detailParts.push(`${board.visibleExcludedRows.length} excluded model row(s) are listed below without rank or C-index.`);
      }
      if (board.hiddenStaleFamilies.length) {
        detailParts.push(`Stale compare rows from ${board.hiddenStaleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden.`);
      }
      if (board.hasMixedEvaluation) {
        detailParts.push(`Current evaluation modes: ${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}.`);
      }
      if (board.visibleHasMixedRunGroups) {
        detailParts.push("Visible ML and DL rows come from different compare runs, so no cross-family rank or shared chart is published.");
      }
      detailParts.push(...benchmarkMethodologyNotes(board));
      ["ml", "dl"].forEach((goal) => {
        const copy = excludedModelsCopy(goal, board.excludedByFamily?.[goal], {
          sourceLabel: board.showingStaleBoard ? "the last complete snapshot" : "the current",
        });
        if (copy) detailParts.push(copy);
      });
      refs.benchmarkTableNote.innerHTML = tableNoteMarkup(leadParts, detailParts);

      const intervals = board.intervals?.status === "ready" ? board.intervals.result : null;
      const intervalByModel = new Map((intervals?.rows || []).map((row) => [String(row.model), row]));
      const rankLabel = board.hasMixedEvaluation ? "Family rank" : "Screen rank";
      const displayedRankLabel = board.withholdCrossFamilyRanking ? "Family rank" : rankLabel;
      let screenRank = 0;
      const rankCells = board.tableRows.map((row) => {
        if (row.excluded) return "—";
        if (board.withholdCrossFamilyRanking) {
          return row.comparableForRanking && row.sourceRank !== null && row.sourceRank !== undefined ? String(row.sourceRank) : "Not ranked";
        }
        return row.comparableForRanking ? String(++screenRank) : "Not ranked";
      });
      refs.benchmarkComparisonShell.innerHTML = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>${escapeHtml(displayedRankLabel)}</th>
              <th>Family</th>
              <th>Model</th>
              <th>${board.hasLockedTest ? "CV C-index (development)" : "C-index"}</th>
              ${board.hasLockedTest ? "<th>Locked-test C-index</th>" : ""}
              ${intervals ? `<th>${board.hasLockedTest ? "Locked-test 95% CI" : "95% CI"}</th><th>ΔC vs Cox PH (95% CI)</th>` : ""}
              <th>Evaluation</th>
              <th>Status</th>
              <th class="benchmark-review-column">Review</th>
              <th class="benchmark-notes-column">Notes</th>
            </tr>
          </thead>
          <tbody>
            ${board.tableRows.map((row, index) => {
              const familyMeta = benchmarkRowFamilyMeta(row);
              return `
              <tr>
                <td>${escapeHtml(rankCells[index])}</td>
                <td><span class="benchmark-family-pill family-${escapeHtml(familyMeta.familyTab)}">${escapeHtml(familyMeta.familyLabel)}</span></td>
                <td>${escapeHtml(formatValue(row.model))}</td>
                <td>${escapeHtml(formatValue(row.c_index))}</td>
                ${board.hasLockedTest ? `<td>${row.hasLockedTest ? escapeHtml(formatValue(row.locked_test_c_index)) : "—"}</td>` : ""}
                ${intervals ? intervalCells(intervalByModel.get(String(row.model))) : ""}
                <td>${escapeHtml(benchmarkEvaluationLabel(row.evaluation_mode))}</td>
                <td>${escapeHtml(row.status)}</td>
                <td class="benchmark-review-column"><span class="benchmark-action-slot" data-benchmark-action-slot="${index}"></span></td>
                <td class="benchmark-notes-column">${row.excluded && row.exclusionReason ? `<div class="benchmark-row-note">${escapeHtml(row.exclusionReason)}</div>` : ""}${EXPERIMENTAL_MODELS.has(String(row.model)) ? '<div class="benchmark-row-note">Experimental architecture</div>' : ""}</td>
              </tr>
            `;
            }).join("")}
          </tbody>
        </table>
      `;
      refs.benchmarkComparisonShell.querySelectorAll("[data-benchmark-action-slot]").forEach((slot) => {
        const index = Number(slot.getAttribute("data-benchmark-action-slot"));
        const row = board.tableRows[index];
        if (!row) return;
        slot.replaceWith(createBenchmarkActionGroup(row));
      });
    }

    function intervalCells(row) {
      if (!row) return "<td>—</td><td>—</td>";
      const delta = row.delta_vs_reference == null
        ? (String(row.model) === "Cox PH" ? "reference" : "—")
        : `${Number(row.delta_vs_reference) >= 0 ? "+" : ""}${Number(row.delta_vs_reference).toFixed(3)} ${intervalRangeText(row.delta_ci).replace("95% ", "")}`;
      return `<td>${escapeHtml(intervalRangeText(row.c_index_ci).replace("(95% ", "").replace(")", ""))}</td><td>${escapeHtml(delta)}</td>`;
    }

    function renderBenchmarkBoard() {
      if (!refs.benchmarkSummaryGrid || !refs.benchmarkComparisonShell) return;
      renderPredictiveWorkbench();
      const hasDataset = Boolean(state.dataset);
      if (!hasDataset) {
        refs.benchmarkSummaryGrid.innerHTML = '<div class="empty-state">Load a dataset first, then compare all predictive models or test one selected model here.</div>';
        const board = benchmarkBoardState();
        renderUnifiedBenchmarkPlot(board).catch((error) => showError(error?.message || "Failed to render unified benchmark plot."));
        renderUnifiedBenchmarkTable(board);
        return;
      }
      const board = benchmarkBoardState();
      board.intervals = boardIntervals(board);
      renderUnifiedBenchmarkSummary(board);
      renderUnifiedBenchmarkPlot(board).catch((error) => showError(error?.message || "Failed to render unified benchmark plot."));
      renderUnifiedBenchmarkTable(board);
      syncPredictiveWorkbenchCompareVisibility();
    }

    return {
      benchmarkCompareRows,
      benchmarkBoardState,
      renderUnifiedBenchmarkPlot,
      renderUnifiedBenchmarkSummary,
      renderUnifiedBenchmarkTable,
      renderBenchmarkBoard,
      benchmarkSingleRunSummary,
    };
  }

  global.SurvStudioBenchmark = {
    createBenchmarkBoardApi,
  };
})(window);

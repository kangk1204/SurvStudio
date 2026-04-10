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

    function showBenchmarkStarterAction() {
      return !(runtime.uiMode === "guided" && runtime.guidedGoal === "predictive");
    }

    function benchmarkRowsFromPayload(goal, payload, { statusOverride = null, paramsSource = "current" } = {}) {
      if (!payload) return [];
      const meta = benchmarkGoalMeta(goal);
      const status = statusOverride || benchmarkResultLabel(goal);
      const runGroupId = comparePayloadGroupId(payload);
      return comparisonRowsFromPayload(payload).map((row, index) => ({
        family: meta.label,
        familyTab: meta.tab,
        model: row.model,
        c_index: row.c_index,
        numericCIndex: benchmarkMetricNumber(row.c_index),
        evaluation_mode: row.evaluation_mode || payload?.analysis?.evaluation_mode || "",
        sourceRank: Number.isFinite(Number(row.rank)) ? Number(row.rank) : index + 1,
        comparableForRanking: row.comparable_for_ranking !== false && benchmarkMetricNumber(row.c_index) !== null,
        status,
        sourceMode: "compare",
        runGroupId,
        paramsSource,
      }));
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
        if (left.sourceRank !== right.sourceRank) return left.sourceRank - right.sourceRank;
        return left.model.localeCompare(right.model);
      });
    }

    function pendingFamilyText(board) {
      const pending = (board?.pendingFamilies ?? []).map((goal) => benchmarkGoalMeta(goal).label);
      return pending.length ? pending.join(" and ") : "the remaining compare runs";
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
      const currentRows = (hasMixedEvaluation || hasMixedRunGroups) ? familyGroupedRows(rawCurrentRows) : rawCurrentRows;
      const snapshotRows = (snapshotHasMixedEvaluation || snapshotHasMixedRunGroups) ? familyGroupedRows(snapshotRowsRaw) : snapshotRowsRaw;
      const showingStaleBoard = snapshotRows.length > 0 && hasUnifiedCoverage(snapshotFamilies) && (!hasUnifiedCoverage(currentFamilies) || hasMixedRunGroups);
      const hiddenStaleFamilies = showingStaleBoard ? [] : staleFamilies;
      const visibleRows = showingStaleBoard ? snapshotRows : currentRows;
      const visibleExcludedRows = showingStaleBoard ? snapshotExcludedRows : currentExcludedRows;
      const visibleFamilies = showingStaleBoard ? snapshotFamilies : currentFamilies;
      const visibleEvaluationModes = showingStaleBoard ? snapshotEvaluationModes : evaluationModes;
      const visibleHasMixedEvaluation = showingStaleBoard ? snapshotHasMixedEvaluation : hasMixedEvaluation;
      const visibleHasMixedRunGroups = showingStaleBoard ? snapshotHasMixedRunGroups : hasMixedRunGroups;
      const rankingRows = (visibleHasMixedEvaluation || visibleHasMixedRunGroups) ? [] : visibleRows.filter((row) => row.comparableForRanking);
      const plottableRows = (visibleHasMixedEvaluation || visibleHasMixedRunGroups) ? [] : visibleRows.filter((row) => row.numericCIndex !== null);
      const tableRows = (visibleHasMixedEvaluation || visibleHasMixedRunGroups)
        ? familyGroupedRows([...visibleRows, ...visibleExcludedRows])
        : [...visibleRows, ...visibleExcludedRows];
      const rawVisibleRows = showingStaleBoard ? snapshotRowsRaw : rawCurrentRows;
      const missingMetricCount = rawVisibleRows.filter((row) => row.numericCIndex === null).length;
      const nonComparableCount = rawVisibleRows.filter((row) => !row.comparableForRanking).length;
      const predictiveBusy = isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl");
      const guidedPredictiveIncomplete = runtime.uiMode === "guided"
        && runtime.guidedGoal === "predictive"
        && !predictiveBusy
        && currentFamilies.length > 0
        && (!hasUnifiedCoverage(currentFamilies) || hasMixedRunGroups)
        && !showingStaleBoard;
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
        missingMetricCount,
        nonComparableCount,
        predictiveBusy,
        guidedPredictiveIncomplete,
        pendingFamilies,
        excludedByFamily: showingStaleBoard ? snapshotExcludedByFamily : excludedByFamily,
        showingStaleBoard,
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
      if (board.guidedPredictiveIncomplete) {
        refs.benchmarkPlotNote.textContent = "The unified chart publishes only after both ML and DL comparison rows are current.";
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Compare All Models must finish with both model families before SurvStudio publishes the unified chart.</span></div>');
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
      refs.benchmarkPlotNote.textContent = noteParts.join(" ");

      const x = board.plottableRows.map((row) => {
        const familyMeta = benchmarkRowFamilyMeta(row);
        return `${row.model}<br>${familyMeta.familyShortLabel}`;
      });
      const y = board.plottableRows.map((row) => row.numericCIndex);
      const colors = board.plottableRows.map((row) => {
        const familyMeta = benchmarkRowFamilyMeta(row);
        return familyMeta.familyTab === "ml" ? "rgba(47, 101, 217, 0.92)" : "rgba(219, 126, 21, 0.92)";
      });
      const borderColors = board.plottableRows.map((row) => {
        const familyMeta = benchmarkRowFamilyMeta(row);
        return familyMeta.familyTab === "ml" ? "rgba(34, 72, 156, 1)" : "rgba(156, 86, 15, 1)";
      });
      const customdata = board.plottableRows.map((row, index) => {
        const familyMeta = benchmarkRowFamilyMeta(row);
        return ([
        index + 1,
        familyMeta.familyLabel,
        benchmarkEvaluationLabel(row.evaluation_mode),
        row.status,
        ]);
      });
      const referenceMax = Math.max(0.72, ...y.filter((value) => Number.isFinite(value)).map((value) => value + 0.04));
      const layout = {
        title: { text: "Unified C-index Screening Board", x: 0.02, xanchor: "left" },
        height: 420,
        margin: { l: 72, r: 24, t: 54, b: 92 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        xaxis: {
          title: { text: "Model" },
          tickfont: { size: 12 },
          automargin: true,
        },
        yaxis: {
          title: { text: "Concordance index" },
          range: [0, referenceMax],
          gridcolor: "rgba(27, 39, 51, 0.08)",
          zeroline: false,
        },
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            yref: "y",
            y0: 0.5,
            y1: 0.5,
            line: { color: "rgba(90, 103, 118, 0.75)", width: 1.5, dash: "dot" },
          },
        ],
        annotations: [
          {
            xref: "paper",
            x: 0.02,
            yref: "y",
            y: 0.5,
            yanchor: "bottom",
            text: "Reference 0.5",
            showarrow: false,
            font: { size: 11, color: "rgba(90, 103, 118, 0.95)" },
            bgcolor: "rgba(255,255,255,0.85)",
          },
        ],
        showlegend: false,
      };
      const trace = {
        type: "bar",
        x,
        y,
        text: board.plottableRows.map((row) => formatValue(row.c_index)),
        textposition: "outside",
        cliponaxis: false,
        marker: {
          color: colors,
          line: {
            color: borderColors,
            width: 1.2,
          },
        },
        customdata,
        hovertemplate: [
          "<b>%{x}</b>",
          "Rank: %{customdata[0]}",
          "Family: %{customdata[1]}",
          "C-index: %{y:.3f}",
          "Evaluation: %{customdata[2]}",
          "Status: %{customdata[3]}",
          "<extra></extra>",
        ].join("<br>"),
      };

      refs.benchmarkComparisonPlot.classList.remove("hidden");
      purgePlot(refs.benchmarkComparisonPlot);
      refs.benchmarkComparisonPlot.innerHTML = "";
      await Plotly.newPlot(
        refs.benchmarkComparisonPlot,
        [trace],
        plotLayoutConfig(layout, "benchmark_comparison"),
        plotConfig("benchmark_comparison"),
      );
      stabilizePlotShellHeight(refs.benchmarkComparisonPlot);
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

      if (board.guidedPredictiveIncomplete) {
        return {
          chips: [
            `Families represented: ${board.visibleFamilies.length}`,
            `ML rows ready: ${currentMlRows}`,
            `DL rows ready: ${currentDlRows}`,
          ],
          status: "Incomplete compare",
          title: "Unified predictive board is incomplete",
          text: `Compare All Models has not produced current rows for both model families yet. ${coverageText} Keep waiting if a family is still running, or rerun Compare All Models before interpreting the predictive board.${cautionSuffix}`,
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
      if (board.guidedPredictiveIncomplete) {
        refs.benchmarkTableNote.textContent = "The unified leaderboard publishes only after both ML and DL comparison rows are current.";
        refs.benchmarkComparisonShell.innerHTML = '<div class="empty-state">Compare All Models must finish with both model families before SurvStudio publishes the unified leaderboard.</div>';
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
      const noteParts = [
        board.hasMixedEvaluation
          ? "Visible compare rows are grouped by family because evaluation modes differ. No cross-family ranking is published."
          : board.visibleHasMixedRunGroups
            ? "Visible compare rows are grouped by family because ML and DL come from different compare runs. No cross-family ranking is published."
          : (presentFamilies.length === 2
            ? (board.showingStaleBoard
              ? `Showing ${board.visibleRows.length} stale screening rows from the last complete Compare All snapshot.`
              : `Showing ${board.visibleRows.length} current screening rows from the latest ML and DL comparison outputs.`)
            : `Showing ${board.visibleRows.length} ${board.showingStaleBoard ? "stale" : "current"} screening row(s) from ${presentFamilies[0] ?? "one family"} only.`),
      ];
      if (board.visibleExcludedRows.length) {
        noteParts.push(`${board.visibleExcludedRows.length} excluded model row(s) are listed below without rank or C-index.`);
      }
      if (board.showingStaleBoard) {
        noteParts.push("Current settings no longer match these rows. Rerun Compare All Models to refresh the leaderboard.");
      }
      if (board.hiddenStaleFamilies.length) {
        noteParts.push(`Stale compare rows from ${board.hiddenStaleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden.`);
      }
      if (board.hasMixedEvaluation) {
        noteParts.push(`Current evaluation modes: ${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}.`);
      }
      if (board.visibleHasMixedRunGroups) {
        noteParts.push("Visible ML and DL rows come from different compare runs, so no cross-family rank or shared chart is published.");
      }
      ["ml", "dl"].forEach((goal) => {
        const copy = excludedModelsCopy(goal, board.excludedByFamily?.[goal], {
          sourceLabel: board.showingStaleBoard ? "the last complete snapshot" : "the current",
        });
        if (copy) noteParts.push(copy);
      });
      refs.benchmarkTableNote.textContent = noteParts.join(" ");

      const rankLabel = board.hasMixedEvaluation ? "Family rank" : "Screen rank";
      const displayedRankLabel = board.visibleHasMixedRunGroups ? "Family rank" : rankLabel;
      refs.benchmarkComparisonShell.innerHTML = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>${escapeHtml(displayedRankLabel)}</th>
              <th>Family</th>
              <th>Model</th>
              <th>C-index</th>
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
                <td>${row.excluded ? "—" : ((board.hasMixedEvaluation || board.visibleHasMixedRunGroups) ? row.sourceRank : index + 1)}</td>
                <td><span class="benchmark-family-pill family-${escapeHtml(familyMeta.familyTab)}">${escapeHtml(familyMeta.familyLabel)}</span></td>
                <td>${escapeHtml(formatValue(row.model))}</td>
                <td>${escapeHtml(formatValue(row.c_index))}</td>
                <td>${escapeHtml(benchmarkEvaluationLabel(row.evaluation_mode))}</td>
                <td>${escapeHtml(row.status)}</td>
                <td class="benchmark-review-column"><span class="benchmark-action-slot" data-benchmark-action-slot="${index}"></span></td>
                <td class="benchmark-notes-column">${row.excluded && row.exclusionReason ? `<div class="benchmark-row-note">${escapeHtml(row.exclusionReason)}</div>` : ""}</td>
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

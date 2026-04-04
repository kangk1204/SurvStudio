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
      currentPredictiveModelKey,
      predictiveModelMeta,
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

    function benchmarkCompareRows(goal, { currentOnly = false } = {}) {
      const payload = currentOnly ? currentCompareGoalPayload(goal) : compareGoalPayload(goal);
      if (panelModeForPayload(payload) !== "compare") return [];
      return Array.isArray(payload?.analysis?.comparison_table) ? payload.analysis.comparison_table : [];
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
      const rows = families.flatMap((goal) => {
        const meta = benchmarkGoalMeta(goal);
        const status = benchmarkResultLabel(goal);
        return benchmarkCompareRows(goal, { currentOnly }).map((row, index) => ({
          family: meta.label,
          familyTab: meta.tab,
          model: row.model,
          c_index: row.c_index,
          numericCIndex: benchmarkMetricNumber(row.c_index),
          evaluation_mode: row.evaluation_mode || goalPayload(goal)?.analysis?.evaluation_mode || "",
          sourceRank: Number.isFinite(Number(row.rank)) ? Number(row.rank) : index + 1,
          comparableForRanking: row.comparable_for_ranking !== false && benchmarkMetricNumber(row.c_index) !== null,
          status,
        }));
      });
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

    function benchmarkBoardState() {
      const rawCurrentRows = unifiedBenchmarkRows({ currentOnly: true });
      const latestRows = unifiedBenchmarkRows({ currentOnly: false });
      const currentFamilies = ["ml", "dl"].filter((goal) => benchmarkCompareRows(goal, { currentOnly: true }).length > 0);
      const staleFamilies = ["ml", "dl"].filter((goal) => benchmarkCompareRows(goal).length > 0 && benchmarkCompareRows(goal, { currentOnly: true }).length === 0);
      const evaluationModes = [
        ...new Set(rawCurrentRows.map((row) => String(row.evaluation_mode || "").trim().toLowerCase()).filter(Boolean)),
      ];
      const hasMixedEvaluation = evaluationModes.length > 1;
      const currentRows = hasMixedEvaluation ? familyGroupedRows(rawCurrentRows) : rawCurrentRows;
      const rankingRows = hasMixedEvaluation ? [] : currentRows.filter((row) => row.comparableForRanking);
      const plottableRows = hasMixedEvaluation ? [] : currentRows.filter((row) => row.numericCIndex !== null);
      const missingMetricCount = rawCurrentRows.filter((row) => row.numericCIndex === null).length;
      const nonComparableCount = rawCurrentRows.filter((row) => !row.comparableForRanking).length;
      const predictiveBusy = isScopeBusy("predictive") || isScopeBusy("ml") || isScopeBusy("dl");
      const pendingFamilies = ["ml", "dl"].filter((goal) => !currentFamilies.includes(goal));
      return {
        currentRows,
        latestRows,
        currentFamilies,
        staleFamilies,
        rankingRows,
        plottableRows,
        evaluationModes,
        hasMixedEvaluation,
        missingMetricCount,
        nonComparableCount,
        predictiveBusy,
        pendingFamilies,
      };
    }

    async function renderUnifiedBenchmarkPlot(board) {
      if (!refs.benchmarkComparisonPlot || !refs.benchmarkPlotNote) return;
      if (board.predictiveBusy) {
        refs.benchmarkPlotNote.textContent = `Unified comparison is still running. Completed families: ${board.currentFamilies.length ? board.currentFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ") : "none yet"}. Waiting on ${board.pendingFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} before charting the final board.`;
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Compare All Models is still running. Partial rows are withheld until both model families finish.</span></div>');
        return;
      }
      if (!board.currentRows.length) {
        refs.benchmarkPlotNote.textContent = board.staleFamilies.length
          ? "Current settings no longer match the last compare run. Rerun Compare All Models to rebuild the cross-family board."
          : "Run Compare All Models to chart ML and DL together on one board.";
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Run Compare All Models to compare ML and DL C-index values on one chart.</span></div>');
        return;
      }
      if (board.hasMixedEvaluation) {
        refs.benchmarkPlotNote.textContent = `Unified chart hidden because current ML and DL compare rows use mixed evaluation paths (${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}). Rerun both families with the same evaluation mode to publish one shared C-index axis.`;
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>Unified chart is hidden until ML and DL compare rows use the same evaluation mode.</span></div>');
        return;
      }
      if (!board.plottableRows.length) {
        refs.benchmarkPlotNote.textContent = "Current comparison rows exist, but none reported a numeric C-index that can be charted. Review the table below.";
        refs.benchmarkComparisonPlot.classList.add("hidden");
        clearPlotShell(refs.benchmarkComparisonPlot, '<div class="empty-state plot-empty"><span>No numeric C-index values are available to chart for the current board.</span></div>');
        return;
      }

      const noteParts = [
        `Showing ${board.plottableRows.length} current screening rows on one C-index axis${board.evaluationModes[0] ? ` using ${benchmarkEvaluationLabel(board.evaluationModes[0])} evaluation.` : "."}`,
      ];
      if (board.staleFamilies.length) {
        noteParts.push(`Stale compare rows from ${board.staleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden until rerun.`);
      }
      if (board.missingMetricCount) {
        noteParts.push(`Omitted ${board.missingMetricCount} row(s) without a numeric C-index.`);
      }
      refs.benchmarkPlotNote.textContent = noteParts.join(" ");

      const x = board.plottableRows.map((row) => `${row.model}<br>${row.family === "Classical ML" ? "ML" : "DL"}`);
      const y = board.plottableRows.map((row) => row.numericCIndex);
      const colors = board.plottableRows.map((row) => (row.familyTab === "ml" ? "rgba(47, 101, 217, 0.92)" : "rgba(219, 126, 21, 0.92)"));
      const borderColors = board.plottableRows.map((row) => (row.familyTab === "ml" ? "rgba(34, 72, 156, 1)" : "rgba(156, 86, 15, 1)"));
      const customdata = board.plottableRows.map((row, index) => ([
        index + 1,
        row.family,
        benchmarkEvaluationLabel(row.evaluation_mode),
        row.status,
      ]));
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

    function renderUnifiedBenchmarkSummary(board) {
      if (!refs.benchmarkSummaryGrid) return;
      const selectedModel = predictiveModelMeta(currentPredictiveModelKey());
      const hasAnyResult = Boolean(goalPayload("ml") || goalPayload("dl"));
      const chips = [
        `Selected model: ${selectedModel.label}`,
        `ML compare rows: ${benchmarkCompareRows("ml", { currentOnly: true }).length}`,
        `DL compare rows: ${benchmarkCompareRows("dl", { currentOnly: true }).length}`,
      ];
      const status = board.predictiveBusy
        ? "Running"
        : !hasAnyResult
        ? "Not run"
        : !board.currentRows.length
        ? "Needs rerun"
        : board.hasMixedEvaluation
        ? "Needs alignment"
        : (board.staleFamilies.length || board.currentFamilies.length < 2 || board.missingMetricCount || board.nonComparableCount
          ? "Needs review"
          : "Board ready");
      const title = board.predictiveBusy
        ? "Unified predictive comparison in progress"
        : !hasAnyResult
        ? "Predictive workspace not run yet"
        : board.hasMixedEvaluation
        ? "Predictive results need alignment"
        : "Predictive screening board";
      const coverageText = board.currentFamilies.length === 2
        ? "Both model families are currently represented."
        : (board.currentFamilies.length === 1
          ? `${benchmarkGoalMeta(board.currentFamilies[0]).label} is currently represented.`
          : "No current compare rows are available.");
      const cautionParts = [];
      if (board.staleFamilies.length) {
        cautionParts.push(`Stale compare rows from ${board.staleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden until rerun.`);
      }
      if (board.hasMixedEvaluation) {
        cautionParts.push(`Current compare rows use mixed evaluation paths (${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}).`);
      }
      if (board.missingMetricCount) {
        cautionParts.push(`${board.missingMetricCount} row(s) have no numeric C-index.`);
      }
      const text = board.predictiveBusy
        ? `Compare All Models is still running. Completed families: ${board.currentFamilies.length ? board.currentFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ") : "none yet"}. Waiting on ${board.pendingFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} before publishing the final unified board.`
        : !hasAnyResult
        ? "Use Compare All Models once to benchmark the full predictive stack, or test one selected model directly."
        : !board.currentRows.length
        ? "Stored predictive results exist, but they no longer match the current outcome, feature, or evaluation settings. Rerun Compare All Models to rebuild the board."
        : board.hasMixedEvaluation
        ? `Current compare rows are grouped by family only. Unified ranking and charting are hidden until ML and DL are rerun with the same evaluation mode. ${coverageText}${cautionParts.length ? ` ${cautionParts.join(" ")}` : ""}`
        : `Showing ${board.currentRows.length} current screening row(s) across the predictive workspace. ${coverageText} The ordering is a convenience screen, not a strict head-to-head benchmark, because ML and DL still run through family-specific evaluation pipelines.${cautionParts.length ? ` ${cautionParts.join(" ")}` : ""}`;
      const tone = board.predictiveBusy ? "running" : (!hasAnyResult ? "idle" : (status === "Board ready" ? "current" : "warning"));
      refs.benchmarkSummaryGrid.innerHTML = `
        <article class="benchmark-family-card tone-${escapeHtml(tone)} benchmark-family-card-wide">
          <div class="benchmark-family-head">
            <div>
              <span class="benchmark-family-badge">${escapeHtml(status)}</span>
              <h3>Predictive Overview</h3>
            </div>
            <button class="button ghost compact-btn" type="button" data-benchmark-model="${escapeHtml(selectedModel.key)}">Show selected controls</button>
          </div>
          <strong class="benchmark-family-title">${escapeHtml(title)}</strong>
          <p class="benchmark-family-copy">${escapeHtml(text)}</p>
          <div class="dataset-preset-chips">${chips.map((label) => `<span class="dataset-preset-chip">${escapeHtml(label)}</span>`).join("")}</div>
        </article>
      `;
    }

    function renderUnifiedBenchmarkTable(board) {
      if (!refs.benchmarkComparisonShell || !refs.benchmarkTableNote) return;
      if (board.predictiveBusy) {
        refs.benchmarkTableNote.textContent = `Unified comparison is still running. Partial rows are withheld until ${board.pendingFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} finish.`;
        refs.benchmarkComparisonShell.innerHTML = '<div class="empty-state">Compare All Models is still running. Partial leaderboard rows are hidden until both model families finish.</div>';
        return;
      }
      if (!board.currentRows.length) {
        refs.benchmarkTableNote.textContent = board.staleFamilies.length
          ? "Stored compare rows are stale and hidden. Rerun Compare All Models to rebuild the shared board with the current settings."
          : "Run Compare All Models to build a shared leaderboard across classical ML and deep learning.";
        refs.benchmarkComparisonShell.innerHTML = '<div class="empty-state">Run Compare All Models to build a shared leaderboard across classical ML and deep learning.</div>';
        return;
      }

      const presentFamilies = [...new Set(board.currentRows.map((row) => row.family))];
      const noteParts = [
        board.hasMixedEvaluation
          ? "Visible compare rows are grouped by family because evaluation modes differ. No cross-family ranking is published."
          : (presentFamilies.length === 2
            ? `Showing ${board.currentRows.length} current screening rows from the latest ML and DL comparison outputs.`
            : `Showing ${board.currentRows.length} current screening row(s) from ${presentFamilies[0]} only.`),
      ];
      if (board.staleFamilies.length) {
        noteParts.push(`Stale compare rows from ${board.staleFamilies.map((goal) => benchmarkGoalMeta(goal).label).join(" and ")} are hidden.`);
      }
      if (board.hasMixedEvaluation) {
        noteParts.push(`Current evaluation modes: ${board.evaluationModes.map((mode) => benchmarkEvaluationLabel(mode)).join(", ")}.`);
      }
      refs.benchmarkTableNote.textContent = noteParts.join(" ");

      const rankLabel = board.hasMixedEvaluation ? "Family rank" : "Rank";
      refs.benchmarkComparisonShell.innerHTML = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>${escapeHtml(rankLabel)}</th>
              <th>Family</th>
              <th>Model</th>
              <th>C-index</th>
              <th>Evaluation</th>
              <th>Status</th>
              <th>Review</th>
            </tr>
          </thead>
          <tbody>
            ${board.currentRows.map((row, index) => `
              <tr>
                <td>${board.hasMixedEvaluation ? row.sourceRank : index + 1}</td>
                <td><span class="benchmark-family-pill family-${escapeHtml(row.familyTab)}">${escapeHtml(row.family)}</span></td>
                <td>${escapeHtml(formatValue(row.model))}</td>
                <td>${escapeHtml(formatValue(row.c_index))}</td>
                <td>${escapeHtml(benchmarkEvaluationLabel(row.evaluation_mode))}</td>
                <td>${escapeHtml(row.status)}</td>
                <td><span class="benchmark-action-slot" data-benchmark-action-slot="${index}"></span></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      refs.benchmarkComparisonShell.querySelectorAll("[data-benchmark-action-slot]").forEach((slot) => {
        const index = Number(slot.getAttribute("data-benchmark-action-slot"));
        const row = board.currentRows[index];
        if (!row) return;
        slot.replaceWith(createBenchmarkActionButton(benchmarkReviewAction(row)));
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

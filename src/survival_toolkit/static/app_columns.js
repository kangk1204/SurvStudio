// SurvStudio front end, part 3/8: Busy state, notifications, column pickers, and outcome/event guidance.
// Classic scripts loaded in order by index.html; top-level declarations are shared by all
// app_*.js parts, and only app.js (loaded last) runs startup code.

function setButtonLoading(button, isLoading) {
  if (!button) return;
  const loading = Boolean(isLoading);
  button.classList.toggle("is-loading", loading);
  button.disabled = loading;
  button.setAttribute("aria-busy", loading ? "true" : "false");
}

function setPanelResultMode(panel, mode = "idle") {
  if (!panel) return;
  panel.dataset.resultMode = mode;
}

function runScopeForGoal(goal) {
  if (goal === "predictive") return isScopeBusy("predictive") ? "predictive" : predictiveFamilyGoal();
  if (["km", "cox", "markers", "ml", "dl", "tables"].includes(goal)) return goal;
  return null;
}

function normalizedPredictiveWorkbenchIntent(value) {
  return value === "train" || value === "features" ? value : null;
}

function isScopeBusy(scope) {
  return Boolean(scope && runtime.busyScopes?.[scope]);
}

function buttonsForScope(scope) {
  if (scope === "predictive") return [refs.runPredictiveCompareAllButton];
  if (scope === "ml") {
    return [
      refs.runMlButton,
      refs.runCompareButton,
      refs.runCompareInlineButton,
      ...(runtime.workbenchRevealed && predictiveFamilyGoal() === "ml" ? [refs.runPredictiveWorkbenchButton] : []),
    ];
  }
  if (scope === "dl") {
    return [
      refs.runDlButton,
      refs.runDlCompareButton,
      refs.runDlCompareInlineButton,
      ...(runtime.workbenchRevealed && predictiveFamilyGoal() === "dl" ? [refs.runPredictiveWorkbenchButton] : []),
    ];
  }
  if (scope === "km") return [refs.runKmButton, refs.runSignatureSearchButton];
  if (scope === "cox") {
    return [
      refs.runCoxButton,
      refs.selectAllCoxCovariatesButton,
      refs.clearCoxCovariatesButton,
      refs.selectAllCoxCategoricalsButton,
      refs.clearCoxCategoricalsButton,
    ];
  }
  if (scope === "tables") return [refs.runCohortTableButton];
  if (scope === "markers") return [refs.runMarkersButton, refs.selectAllMarkersButton, refs.clearMarkersButton, refs.runMarkerValidationButton];
  if (scope === "derive") return [refs.deriveButton];
  return [];
}

// The busy scope a Run button belongs to, for clicks and for Ctrl+Enter alike.
function runScopeForButton(button) {
  if (!button) return null;
  if (button === refs.runPredictiveCompareAllButton) return "predictive";
  if (button === refs.runPredictiveWorkbenchButton || button === refs.runPredictiveSelectedButton) {
    return predictiveModelMeta(refs.predictiveModelSelector?.value || currentPredictiveModelKey()).family;
  }
  if ([refs.runMlButton, refs.runCompareButton, refs.runCompareInlineButton].includes(button)) return "ml";
  if ([refs.runDlButton, refs.runDlCompareButton, refs.runDlCompareInlineButton].includes(button)) return "dl";
  if (button === refs.runKmButton || button === refs.runSignatureSearchButton) return "km";
  if (button === refs.runCoxButton) return "cox";
  if (button === refs.runCohortTableButton) return "tables";
  if (button === refs.runMarkersButton || button === refs.runMarkerValidationButton) return "markers";
  if (button === refs.deriveButton) return "derive";
  return null;
}

function setChecklistDisabled(container, isDisabled) {
  if (!container) return;
  container.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.disabled = Boolean(isDisabled);
  });
}

function syncSharedFeatureControlsBusy() {
  const isBusy = isScopeBusy("ml") || isScopeBusy("dl");
  [
    refs.reviewMlFeaturesButton,
    refs.reviewDlFeaturesButton,
    refs.selectAllModelFeaturesButton,
    refs.clearModelFeaturesButton,
    refs.selectAllDlModelFeaturesButton,
    refs.clearDlModelFeaturesButton,
  ].forEach((button) => {
    if (button) button.disabled = isBusy;
  });
  setChecklistDisabled(refs.modelFeatureChecklist, isBusy);
  setChecklistDisabled(refs.modelCategoricalChecklist, isBusy);
  setChecklistDisabled(refs.dlModelFeatureChecklist, isBusy);
  setChecklistDisabled(refs.dlModelCategoricalChecklist, isBusy);
}

function setScopeBusy(scope, isBusy, activeButton = null) {
  if (!scope) return;
  runtime.busyScopes[scope] = Boolean(isBusy);
  const scopeButtons = buttonsForScope(scope);
  if (activeButton && !scopeButtons.includes(activeButton)) {
    setButtonLoading(activeButton, isBusy);
  }
  scopeButtons.forEach((button) => {
    if (!button) return;
    if (button === activeButton) {
      setButtonLoading(button, isBusy);
      return;
    }
    button.disabled = Boolean(isBusy);
    button.classList.toggle("is-loading", false);
    button.setAttribute("aria-busy", "false");
  });
  syncSharedFeatureControlsBusy();
  if (scope === "ml") updateMlEvaluationControls();
  if (scope === "dl") updateDlEvaluationControls();
  syncAnalysisRunButtonAvailability();
  // The loop above re-enabled every button of the scope; the ones with their own conditions (Validate
  // needs a locked model, the marker list is locked while a file is attached, Create is locked while
  // Group by is set) take their state from those conditions again.
  if (scope === "markers") {
    renderMarkerMatrixState();
    syncMarkerDownloadButtons();
  }
  if (scope === "derive") syncDeriveControlsState();
  renderWorkspaceChrome();
  if (scope === "predictive" || scope === "ml" || scope === "dl") {
    renderBenchmarkBoard();
  }
}

function showToast(message, type = "error", duration = 5000) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${escapeHtml(message)}</span><button class="toast-close">&times;</button>`;
  container.appendChild(toast);
  const dismiss = () => {
    toast.classList.add("toast-exit");
    toast.addEventListener("animationend", () => toast.remove());
  };
  toast.querySelector(".toast-close").addEventListener("click", dismiss);
  if (duration > 0) setTimeout(dismiss, duration);
}

function showError(message) { showToast(message, "error"); }

async function shutdownServer() {
  return shellHelpers.shutdownServer({
    runtime,
    refs,
    setButtonLoading,
    setRuntimeBanner,
    fetchJSON,
    renderServerStoppedState,
  });
}

function renderSelect(select, options, { includeBlank = false, blankLabel = "None", selected = null } = {}) {
  select.innerHTML = "";
  if (includeBlank) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = blankLabel;
    select.appendChild(option);
  }
  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (selected !== null && value === selected) option.selected = true;
    select.appendChild(option);
  });
}

function inferDefault(columnNames, suggestions, fallbackIndex = 0) {
  if (suggestions.length > 0 && columnNames.includes(suggestions[0])) return suggestions[0];
  return columnNames[fallbackIndex] || "";
}

function getColumnMeta(columnName) {
  return state.dataset?.columns.find((column) => column.name === columnName) || null;
}

function hasConfidentEventSuggestion() {
  const binaryColumns = binaryCandidateColumns();
  const eventLikeBinary = binaryColumns.filter(
    (column) => isEventLikeColumnName(column) && !looksLikeBaselineStatusColumn(column),
  );
  if (eventLikeBinary.length) return true;
  const suggestionSet = new Set(state.dataset?.suggestions?.event_columns || []);
  const keywordBinary = binaryColumns.filter(
    (column) => suggestionSet.has(column) && !looksLikeBaselineStatusColumn(column),
  );
  return keywordBinary.length > 0;
}

function currentGroupColumnWarning() {
  const groupColumn = refs.groupColumn?.value || "";
  if (!state.dataset || !groupColumn) return null;
  if (groupColumn === refs.timeColumn?.value || groupColumn === refs.eventColumn?.value) {
    return {
      tone: "error",
      message: `"${groupColumn}" is part of the survival endpoint. Use a separate categorical grouping variable for Kaplan-Meier and grouped tables.`,
    };
  }

  const meta = getColumnMeta(groupColumn);
  if (!meta) return null;
  if (meta.kind === "numeric" && Number(meta.n_unique || 0) > 8) {
    return {
      tone: "error",
      message: `"${groupColumn}" is a high-cardinality numeric column. Create a grouped version first, then use that new column for Kaplan-Meier or grouped tables.`,
    };
  }
  if (Number(meta.n_unique || 0) > 20) {
    return {
      tone: "error",
      message: `"${groupColumn}" has too many unique values for meaningful grouped survival curves. Use a lower-cardinality grouping column instead.`,
    };
  }
  return null;
}

function normalizeColumnLabel(columnName) {
  return String(columnName || "").trim().toLowerCase();
}

function isEventLikeColumnName(columnName) {
  const normalized = normalizeColumnLabel(columnName);
  if (!normalized) return false;
  if (normalized === "event" || normalized === "status") return true;
  return [
    /event/,
    /death/,
    /mort/,
    /vital_status/,
    /survival_status/,
    /outcome_status/,
    /relapse/,
    /recur/,
    /progress/,
    /failure/,
    /censor/,
  ].some((pattern) => pattern.test(normalized));
}

function looksLikeBaselineStatusColumn(columnName) {
  const normalized = normalizeColumnLabel(columnName);
  if (!normalized) return false;
  return [
    /egfr/,
    /kras/,
    /braf/,
    /alk/,
    /ros1/,
    /erbb2/,
    /mutation/,
    /mutated/,
    /wildtype/,
    /sex/,
    /gender/,
    /stage/,
    /grade/,
    /treat/,
    /therapy/,
    /drug/,
    /smok/,
    /histolog/,
    /subtype/,
    /cluster/,
    /group/,
    /arm/,
    /cohort/,
    /horth/,
  ].some((pattern) => pattern.test(normalized));
}

function datasetColumnNames() {
  return state.dataset?.columns?.map((column) => column.name) || [];
}

function recommendedTimeColumns() {
  const names = new Set(datasetColumnNames());
  return (state.dataset?.suggestions?.time_columns || []).filter((column) => names.has(column));
}

function numericTimeCandidateColumns() {
  const numeric = new Set(state.dataset?.numeric_columns || []);
  return datasetColumnNames().filter((column) => numeric.has(column));
}

// The Time menu lists the likely follow-up columns; "All numeric" (or a dataset without a likely one)
// lists every numeric column, so a follow-up column with an unusual name can still be chosen.
function allowedTimeColumns() {
  const recommended = recommendedTimeColumns();
  if (recommended.length && !refs.showAllTimeColumns?.checked) return recommended;
  return numericTimeCandidateColumns();
}

function identicalOutcomeColumnMessage() {
  const timeColumn = refs.timeColumn?.value || "";
  const eventColumn = refs.eventColumn?.value || "";
  if (!state.dataset || !timeColumn || !eventColumn || timeColumn !== eventColumn) return null;
  return "The survival time column and event column must be different.";
}

function currentTimeColumnWarning() {
  const timeColumn = refs.timeColumn?.value || "";
  if (!state.dataset || !timeColumn) return null;
  const matchingOutcomeWarning = identicalOutcomeColumnMessage();
  if (matchingOutcomeWarning) {
    return {
      tone: "error",
      message: matchingOutcomeWarning,
    };
  }

  const numericSet = new Set(state.dataset.numeric_columns || []);
  if (!numericSet.has(timeColumn)) {
    return {
      tone: "error",
      message: `"${timeColumn}" is not numeric. Choose a true follow-up time column such as os_months, pfs_months, days, or follow-up time.`,
    };
  }

  const recommended = recommendedTimeColumns();
  if (!recommended.length || recommended.includes(timeColumn)) return null;
  if (refs.showAllTimeColumns?.checked) {
    // Chosen on purpose through "All numeric": a reminder, not a block.
    return {
      tone: "warning",
      message: `"${timeColumn}" is not one of the likely follow-up time columns (${recommended.slice(0, 3).join(", ")}). Use it only if it holds each patient's follow-up time.`,
    };
  }
  return {
    tone: "error",
    message: `"${timeColumn}" does not look like a survival follow-up time column. Choose one of the likely time columns instead: ${recommended.slice(0, 3).join(", ")}, or tick All numeric if it is the follow-up time.`,
  };
}

function updateTimeColumnGuidance() {
  if (!state.dataset) return;

  const recommended = recommendedTimeColumns();
  const numericColumns = allowedTimeColumns();
  if (refs.timeColumnHelp) {
    if (recommended.length && refs.showAllTimeColumns?.checked) {
      refs.timeColumnHelp.textContent = "Showing all numeric columns. Use only a true follow-up time.";
    } else if (recommended.length) {
      refs.timeColumnHelp.textContent = "Showing likely time columns only.";
    } else if (numericColumns.length) {
      refs.timeColumnHelp.textContent = "No clear time column name was found; showing numeric columns. Check the follow-up field.";
    } else {
      refs.timeColumnHelp.textContent = "No numeric time candidates were detected in this dataset.";
    }
  }

  const warning = currentTimeColumnWarning();
  if (!refs.timeColumnWarning) return;
  if (!warning) {
    refs.timeColumnWarning.textContent = "";
    refs.timeColumnWarning.className = "event-warning hidden";
    return;
  }
  refs.timeColumnWarning.textContent = warning.message;
  refs.timeColumnWarning.className = `event-warning event-warning-${warning.tone}`;
}

function renderTimeColumnOptions({ preferred = null, silent = true } = {}) {
  if (!state.dataset) return;
  const options = allowedTimeColumns();
  const recommended = recommendedTimeColumns();
  const currentValue = preferred ?? refs.timeColumn?.value ?? "";
  // Without a likely follow-up column nothing is preselected: the first numeric column is usually an ID.
  const nextValue = options.includes(currentValue)
    ? currentValue
    : recommended.find((column) => options.includes(column)) || "";
  renderSelect(refs.timeColumn, options, {
    includeBlank: !recommended.length || !nextValue,
    blankLabel: "Select time column",
    selected: nextValue || "",
  });
  updateTimeColumnGuidance();

  if (!silent && currentValue && nextValue && currentValue !== nextValue) {
    showToast(
      `Time column reset to ${nextValue}. Use a true follow-up time field, not a gene or baseline covariate.`,
      "warning",
      3600,
    );
  } else if (!silent && currentValue && !nextValue) {
    showToast(
      "Time column cleared. Choose the follow-up time field again before running an analysis.",
      "warning",
      3600,
    );
  }
}

function normalizeEventToken(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value ? "1" : "0";
  const text = String(value).trim().toLowerCase();
  if (!text) return null;
  if (/^-?\d+(?:\.0+)?$/.test(text)) return String(Number(text));
  return text;
}

function eventTokenCandidates(value) {
  const token = normalizeEventToken(value);
  if (!token) return [];
  const variants = [];
  const add = (candidate) => {
    if (candidate && !variants.includes(candidate)) variants.push(candidate);
  };
  add(token);
  add(token.replace(/[^a-z0-9]+/g, ""));
  token.split(/[^a-z0-9]+/).forEach((part) => add(part));
  return variants;
}

// A label that negates an event ("No recurrence", "Not progressed", "Non-relapse", "Never", "Without
// death", "Recurrence-free") names the censored side even though it contains an event word.
const EVENT_NEGATION_TOKENS = new Set(["no", "not", "non", "never", "without"]);

function isNegatedEventLabel(value) {
  if (typeof value !== "string") return false;
  const tokens = value.trim().toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  return tokens.some((token) => EVENT_NEGATION_TOKENS.has(token) || token.endsWith("free") || /^non[a-z]{3,}$/.test(token));
}

function eventValueFamily(value) {
  const candidates = eventTokenCandidates(value);
  if (!candidates.length) return null;
  if (isNegatedEventLabel(value)) return "censor";
  if (candidates.some((candidate) => EVENT_FALSE_TOKENS.has(candidate))) return "censor";
  if (candidates.some((candidate) => EVENT_TRUE_TOKENS.has(candidate))) return "event";
  return null;
}

function hasRecognizableEventCoding(values) {
  const normalizedValues = values.filter((value) => value !== null && value !== undefined);
  if (!normalizedValues.length) return false;
  const uniqueTokens = [...new Set(normalizedValues.map((value) => normalizeEventToken(value)).filter(Boolean))];
  const numericTokens = uniqueTokens.filter((token) => /^-?\d+(?:\.\d+)?$/.test(token));
  if (uniqueTokens.length === 2 && numericTokens.length === 2) {
    const numericPair = numericTokens.map((token) => Number(token)).sort((a, b) => a - b);
    if (
      (numericPair[0] === 0 && numericPair[1] === 1)
      || (numericPair[0] === 1 && numericPair[1] === 2)
    ) {
      return true;
    }
  }
  const families = new Set(normalizedValues.map((value) => eventValueFamily(value)).filter(Boolean));
  return families.has("event") && families.has("censor");
}

function inferEventPositiveSelection(eventColumn, values, previousValue = "") {
  const normalized = values.map((value) => ({
    value,
    raw: String(value),
    token: normalizeEventToken(value),
    candidates: eventTokenCandidates(value),
  })).filter((entry) => entry.token !== null);
  const available = new Set(normalized.map((entry) => entry.raw));
  if (previousValue && available.has(String(previousValue))) {
    return { value: String(previousValue), warning: null };
  }

  // A negated label ("No recurrence") is never the event value, however many event words it holds.
  const negated = normalized.filter((entry) => isNegatedEventLabel(entry.value));
  const truthy = normalized.filter((entry) => !negated.includes(entry)
    && entry.candidates.some((candidate) => EVENT_TRUE_TOKENS.has(candidate)));
  const falsy = normalized.filter((entry) => negated.includes(entry)
    || entry.candidates.some((candidate) => EVENT_FALSE_TOKENS.has(candidate)));
  const pickTruthy = truthy.find((entry) => entry.token === "1")
    || truthy.find((entry) => entry.token === "event")
    || truthy.find((entry) => entry.token === "dead")
    || truthy[0];

  if (pickTruthy && falsy.length) {
    return { value: pickTruthy.raw, warning: null };
  }
  if (pickTruthy && normalized.length === 1) {
    return { value: pickTruthy.raw, warning: null };
  }
  if (negated.length && !pickTruthy) {
    return {
      value: "",
      warning: `"${eventColumn}" has values that negate an event (${negated.map((entry) => entry.raw).join(", ")}), so SurvStudio did not guess which value means event. Choose it explicitly.`,
    };
  }

  const uniqueTokens = [...new Set(normalized.map((entry) => entry.token))];
  const numericTokens = uniqueTokens.filter((token) => /^-?\d+(?:\.\d+)?$/.test(token));
  if (uniqueTokens.length === 2 && numericTokens.length === 2) {
    const numericPair = numericTokens.map((token) => Number(token)).sort((a, b) => a - b);
    if (numericPair[0] === 1 && numericPair[1] === 2) {
      return {
        value: "",
        warning: `"${eventColumn}" looks like TCGA-style 1/2 coding (${values.map((value) => String(value)).join(", ")}). Confirm whether 1 means event and 2 means censoring.`,
      };
    }
    return {
      value: "",
      warning: `"${eventColumn}" uses non-standard binary values (${values.map((value) => String(value)).join(", ")}). Choose which value means event.`,
    };
  }

  return {
    value: "",
    warning: `SurvStudio could not safely guess which value in "${eventColumn}" means event. Choose it explicitly.`,
  };
}

function binaryCandidateColumns() {
  const binarySet = new Set(state.dataset?.binary_candidate_columns || []);
  return datasetColumnNames().filter((column) => binarySet.has(column));
}

function recommendedEventColumns() {
  const binaryColumns = binaryCandidateColumns();
  const eventLikeBinary = binaryColumns.filter(
    (column) => isEventLikeColumnName(column) && !looksLikeBaselineStatusColumn(column),
  );
  if (eventLikeBinary.length) return eventLikeBinary;
  const suggestionSet = new Set(state.dataset?.suggestions?.event_columns || []);
  const keywordBinary = binaryColumns.filter(
    (column) => suggestionSet.has(column) && !looksLikeBaselineStatusColumn(column),
  );
  if (keywordBinary.length) return keywordBinary;
  return binaryColumns;
}

function currentEventColumnWarning() {
  const eventColumn = refs.eventColumn?.value || "";
  if (!state.dataset || !eventColumn) return null;
  const matchingOutcomeWarning = identicalOutcomeColumnMessage();
  if (matchingOutcomeWarning) {
    return {
      tone: "error",
      blocking: true,
      message: matchingOutcomeWarning,
    };
  }

  const binarySet = new Set(state.dataset.binary_candidate_columns || []);
  const suggestionSet = new Set(state.dataset?.suggestions?.event_columns || []);
  const looksBaselineLike = looksLikeBaselineStatusColumn(eventColumn);
  const previewValues = getColumnMeta(eventColumn)?.unique_preview?.filter((value) => value !== null) ?? [];
  const looksLikeRecognizableEventCoding = hasRecognizableEventCoding(previewValues);
  if (!binarySet.has(eventColumn)) {
    return {
      tone: "error",
      blocking: true,
      message: `"${eventColumn}" is not a binary event column. Choose a 0/1-style event column or recode it.`,
    };
  }

  if ((isEventLikeColumnName(eventColumn) || suggestionSet.has(eventColumn)) && !looksBaselineLike) return null;

  if (!refs.showAllEventColumns?.checked) {
    return {
      tone: "warning",
      blocking: true,
      message: `"${eventColumn}" is not a standard event column name. Tick All columns only if you intend to use it as the event indicator.`,
    };
  }

  if (looksBaselineLike) {
    return {
      tone: "warning",
      blocking: true,
      message: `"${eventColumn}" looks more like a baseline characteristic than an event indicator. Use it as Group by or as a model feature instead.`,
    };
  }

  if (looksLikeRecognizableEventCoding) {
    return {
      tone: "warning",
      blocking: false,
      message: `"${eventColumn}" uses non-standard event coding. Confirm it is the event indicator before continuing.`,
    };
  }

  return {
    tone: "warning",
    blocking: true,
    message: `"${eventColumn}" does not look like a survival event column. Use the suggested event field or a true event indicator instead.`,
  };
}

function updateEventValueGuidance(message = null) {
  if (!refs.eventValueWarning) return;
  if (!message) {
    refs.eventValueWarning.textContent = "";
    refs.eventValueWarning.className = "event-warning hidden";
    return;
  }
  refs.eventValueWarning.textContent = message;
  refs.eventValueWarning.className = "event-warning event-warning-warning";
}

function updateEventColumnGuidance() {
  if (!state.dataset) return;

  const binaryColumns = binaryCandidateColumns();
  const recommendedColumns = recommendedEventColumns();
  if (refs.eventColumnHelp) {
    if (refs.showAllEventColumns?.checked) {
      refs.eventColumnHelp.textContent = "Showing all columns. Use only a true binary event indicator.";
    } else if (recommendedColumns.length && recommendedColumns.length < datasetColumnNames().length) {
      refs.eventColumnHelp.textContent = "Showing likely event columns only.";
    } else if (binaryColumns.length) {
      refs.eventColumnHelp.textContent = "No clear event column name was found; showing binary columns.";
    } else {
      refs.eventColumnHelp.textContent = "No binary event column was found. Tick All columns if you know which one it is.";
    }
  }

  const warning = currentEventColumnWarning();
  if (!refs.eventColumnWarning) return;
  if (!warning) {
    refs.eventColumnWarning.textContent = "";
    refs.eventColumnWarning.className = "event-warning hidden";
    return;
  }
  refs.eventColumnWarning.textContent = warning.message;
  refs.eventColumnWarning.className = `event-warning event-warning-${warning.tone}`;
}

function renderEventColumnOptions({ preferred = null, silent = true } = {}) {
  if (!state.dataset) return;
  const allColumns = datasetColumnNames();
  const options = refs.showAllEventColumns?.checked
    ? allColumns
    : (() => {
        const recommended = recommendedEventColumns();
        if (recommended.length) return recommended;
        const binaryColumns = binaryCandidateColumns();
        return binaryColumns.length ? binaryColumns : allColumns;
      })();

  const currentValue = preferred ?? refs.eventColumn?.value ?? "";
  const confident = hasConfidentEventSuggestion();
  const nextValue = options.includes(currentValue)
    ? currentValue
    : confident
      ? inferDefault(options, recommendedEventColumns(), 0)
      : "";
  renderSelect(refs.eventColumn, options, {
    includeBlank: !confident || !nextValue,
    blankLabel: "Select event column",
    selected: nextValue || "",
  });
  updateEventPositiveOptions();
  updateEventColumnGuidance();

  if (!silent && currentValue && nextValue && currentValue !== nextValue) {
    showToast(
      `Event column reset to ${nextValue}. Tick All columns to select a non-standard event field.`,
      "warning",
      3600,
    );
  } else if (!silent && currentValue && !nextValue) {
    showToast(
      "Event column cleared. Choose the event indicator again before running an analysis.",
      "warning",
      3600,
    );
  }
}

function updateEventPositiveOptions() {
  const eventColumn = refs.eventColumn.value;
  const meta = getColumnMeta(eventColumn);
  const values = meta?.unique_preview?.filter((value) => value !== null) ?? [];
  const preserveSelection = refs.eventColumn?.dataset?.lastColumn === eventColumn;
  const previousValue = preserveSelection ? refs.eventPositiveValue.value : "";
  const eventColumnWarning = currentEventColumnWarning();
  if (refs.eventColumn) refs.eventColumn.dataset.lastColumn = eventColumn;
  refs.eventPositiveValue.innerHTML = "";
  if (values.length === 0) {
    // Without observed values there is nothing safe to default to; require an explicit choice.
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = eventColumn ? "No event values found" : "Choose event column first";
    placeholder.selected = true;
    refs.eventPositiveValue.appendChild(placeholder);
    updateEventValueGuidance(
      eventColumn
        ? `No non-missing values were found in "${eventColumn}", so no event value can be selected. Choose a different event column.`
        : null,
    );
    return;
  }
  const inferred = inferEventPositiveSelection(eventColumn, values, previousValue);
  if (!inferred.value) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose event value";
    placeholder.selected = true;
    refs.eventPositiveValue.appendChild(placeholder);
  }
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    if (String(value) === inferred.value) option.selected = true;
    refs.eventPositiveValue.appendChild(option);
  });
  updateEventValueGuidance(eventColumnWarning?.blocking ? null : inferred.warning);
  updateEventColumnGuidance();
}

// `notes` maps a value to a short note (a Map or a plain object; only its own entries count, so a column
// named "constructor" gets no note from Object.prototype).
function renderChecklist(container, values, selected = [], notes = new Map()) {
  if (!container) return;
  container.innerHTML = "";
  values.forEach((value) => {
    const label = document.createElement("label");
    label.className = "check-item";
    label.dataset.filterValue = String(value).toLowerCase();
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = selected.includes(value);
    const span = document.createElement("span");
    span.textContent = value;
    label.append(input, span);
    const noteText = ownEntry(notes, value);
    if (noteText) {
      const note = document.createElement("small");
      note.className = "check-item-note";
      note.textContent = noteText;
      label.appendChild(note);
    }
    container.appendChild(label);
  });
  applyChecklistSearch(container);
}

function searchControlForChecklist(container) {
  if (!container) return null;
  if (container === refs.covariateChecklist) return refs.covariateSearchInput;
  if (container === refs.categoricalChecklist) return refs.categoricalSearchInput;
  if (container === refs.strataChecklist) return refs.strataSearchInput;
  if (container === refs.cohortVariableChecklist) return refs.cohortVariableSearchInput;
  if (container === refs.markerChecklist) return refs.markerSearchInput;
  if (container === refs.markerClinicalChecklist) return refs.markerClinicalSearchInput;
  return null;
}

function applyChecklistSearch(container) {
  if (!container) return;
  const searchControl = searchControlForChecklist(container);
  if (!searchControl) return;

  const query = String(searchControl.value || "").trim().toLowerCase();
  let visibleCount = 0;
  container.querySelectorAll(".check-item").forEach((item) => {
    const match = !query || String(item.dataset.filterValue || "").includes(query);
    item.classList.toggle("hidden-by-filter", !match);
    if (match) visibleCount += 1;
  });

  let emptyState = container.querySelector(".checklist-filter-empty");
  if (!emptyState) {
    emptyState = document.createElement("div");
    emptyState.className = "checklist-filter-empty hidden";
    container.appendChild(emptyState);
  }
  const showEmpty = Boolean(query) && visibleCount === 0;
  emptyState.textContent = showEmpty ? `No matches for "${searchControl.value}".` : "";
  emptyState.classList.toggle("hidden", !showEmpty);
}

function selectedCheckboxValues(container) {
  if (!container) return [];
  return [...container.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
}

function allCheckboxValues(container, { visibleOnly = false } = {}) {
  if (!container) return [];
  return [...container.querySelectorAll('input[type="checkbox"]')]
    .filter((input) => !visibleOnly || !input.closest(".check-item")?.classList.contains("hidden-by-filter"))
    .map((input) => input.value);
}

function normalizeDerivedColumnProvenance(provenance = {}) {
  return Object.fromEntries(
    Object.entries(provenance || {}).map(([columnName, meta]) => {
      const normalizedMeta = meta && typeof meta === "object" ? meta : {};
      return [columnName, {
        ...normalizedMeta,
        outcomeInformed: Boolean(normalizedMeta.outcomeInformed ?? normalizedMeta.outcome_informed),
        recipe: normalizedMeta.recipe || {},
        summary: normalizedMeta.summary || null,
      }];
    }),
  );
}

function availableCoxStageVariables() {
  const available = new Set(allCheckboxValues(refs.covariateChecklist));
  return COX_STAGE_VARIABLE_PREFERENCE.filter((value) => available.has(value));
}

function renderCoxCovariateWarning({ kept = null, removed = [] } = {}) {
  if (!refs.coxCovariateWarning) return;
  const availableStageVars = availableCoxStageVariables();
  if (!kept || availableStageVars.length < 2) {
    refs.coxCovariateWarning.textContent = "";
    refs.coxCovariateWarning.className = "event-warning event-warning-warning hidden";
    return;
  }
  const alternatives = availableStageVars.filter((value) => value !== kept);
  refs.coxCovariateWarning.textContent = removed.length
    ? `Cox can use only one stage variable at a time. Keeping ${kept}; cleared ${removed.join(", ")}.`
    : `Cox can use only one stage variable at a time. Using ${kept}. Leave ${alternatives.join(" or ")} unchecked to avoid redundant stage encoding.`;
  refs.coxCovariateWarning.className = "event-warning event-warning-warning";
}

function shouldAutoCategorizeCoxCovariate(columnName) {
  const meta = getColumnMeta(columnName);
  if (!meta) return false;
  return meta.kind === "categorical" || meta.kind === "binary";
}

function syncCoxCovariateSelection({
  preferredValue = null,
  preferredScope = null,
  notify = false,
  autoCategoricalValues = [],
} = {}) {
  if (!refs.covariateChecklist) return { kept: null, removed: [], changed: false };
  const selected = selectedCheckboxValues(refs.covariateChecklist);
  let normalizedStrata = refs.strataChecklist
    ? selectedCheckboxValues(refs.strataChecklist)
    : [];
  let overlapRemovedFromCovariates = [];
  let overlapRemovedFromStrata = [];
  const overlappingSelections = normalizedStrata.filter((value) => selected.includes(value));
  let normalizedCovariates = [...selected];

  if (overlappingSelections.length) {
    if (preferredScope === "covariate") {
      overlapRemovedFromStrata = overlappingSelections;
      normalizedStrata = normalizedStrata.filter((value) => !overlappingSelections.includes(value));
    } else {
      overlapRemovedFromCovariates = overlappingSelections;
      normalizedCovariates = normalizedCovariates.filter((value) => !overlappingSelections.includes(value));
    }
  }

  const selectedStageVars = COX_STAGE_VARIABLE_PREFERENCE.filter((value) => normalizedCovariates.includes(value));
  let kept = selectedStageVars[0] || null;
  let removed = [];

  if (selectedStageVars.length > 1) {
    kept = selectedStageVars.includes(preferredValue)
      ? preferredValue
      : COX_STAGE_VARIABLE_PREFERENCE.find((value) => selectedStageVars.includes(value)) || selectedStageVars[0];
    removed = selectedStageVars.filter((value) => value !== kept);
    normalizedCovariates = normalizedCovariates.filter((value) => !removed.includes(value));
  }

  setCheckedValues(refs.covariateChecklist, normalizedCovariates);
  if (refs.strataChecklist) setCheckedValues(refs.strataChecklist, normalizedStrata);

  if (refs.categoricalChecklist) {
    const normalizedCategoricals = selectedCheckboxValues(refs.categoricalChecklist).filter((value) => normalizedCovariates.includes(value));
    [...new Set(autoCategoricalValues)].forEach((value) => {
      if (
        value
        && normalizedCovariates.includes(value)
        && shouldAutoCategorizeCoxCovariate(value)
        && !normalizedCategoricals.includes(value)
      ) {
        normalizedCategoricals.push(value);
      }
    });
    setCheckedValues(refs.categoricalChecklist, normalizedCategoricals);
  }
  renderCoxCovariateWarning({ kept, removed });
  if (notify) {
    if (overlapRemovedFromCovariates.length) {
      showToast(
        `Moved ${overlapRemovedFromCovariates.join(", ")} from Cox covariates to Strata. Stratified variables are not estimated as hazard ratios.`,
        "info",
        4200,
      );
    } else if (overlapRemovedFromStrata.length) {
      showToast(
        `Removed ${overlapRemovedFromStrata.join(", ")} from Strata and kept them as Cox covariates.`,
        "info",
        3600,
      );
    } else if (removed.length) {
      showToast(`Cox can use only one stage variable. Keeping ${kept} and clearing ${removed.join(", ")}.`, "warning", 3600);
    }
  }
  return {
    kept,
    removed,
    changed: removed.length > 0 || overlapRemovedFromCovariates.length > 0 || overlapRemovedFromStrata.length > 0,
  };
}

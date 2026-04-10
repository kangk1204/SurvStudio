(function registerSurvStudioShell() {
  function currentHistoryState({ state, runtime, activeTabName, captureControlSnapshot }) {
    if (!state.dataset) return { view: "home", uiMode: runtime.uiMode };
    return {
      view: "workspace",
      datasetId: state.dataset.dataset_id,
      tab: activeTabName(),
      uiMode: runtime.uiMode,
      guidedGoal: runtime.guidedGoal,
      guidedStep: runtime.guidedStep,
      predictiveFamily: runtime.predictiveFamily,
      workbenchRevealed: runtime.workbenchRevealed,
      predictiveWorkbenchIntent: runtime.predictiveWorkbenchIntent,
      controls: captureControlSnapshot(),
    };
  }

  function syncHistoryState({ runtime, nextState, mode = "replace" }) {
    if (runtime.historySyncPaused || !window.history?.replaceState) return;
    if (mode === "push") {
      window.history.pushState(nextState, "", window.location.href);
      return;
    }
    window.history.replaceState(nextState, "", window.location.href);
  }

  async function shutdownServer({
    runtime,
    refs,
    setButtonLoading,
    setRuntimeBanner,
    fetchJSON,
    renderServerStoppedState,
  }) {
    const activeScopes = Object.entries(runtime.busyScopes || {})
      .filter(([, isBusy]) => Boolean(isBusy))
      .map(([scope]) => scope.toUpperCase());
    const warning = activeScopes.length
      ? `A run is still in progress (${activeScopes.join(", ")}). Stopping the server will cancel it and release memory. Continue?`
      : "Stop the local SurvStudio server and release its memory?";
    if (!window.confirm(warning)) return;

    refs.shutdownButton.disabled = true;
    setButtonLoading(refs.shutdownButton, true);
    setRuntimeBanner("Stopping the local SurvStudio server. This will cancel active runs and release memory.", "warning");
    try {
      const payload = await fetchJSON("/api/shutdown", { method: "POST" });
      renderServerStoppedState(payload?.detail);
    } catch (error) {
      refs.shutdownButton.disabled = false;
      setButtonLoading(refs.shutdownButton, false);
      throw error;
    }
  }

  function goHome({
    state,
    runtime,
    refs,
    syncHistory = true,
    historyMode = "replace",
    resetCoxPreview,
    renderSharedFeatureSummary,
    renderGuidedChrome,
    setRuntimeBanner,
    syncHistoryState: syncHistoryStateFn,
  }) {
    state.dataset = null;
    state.km = null;
    state.cox = null;
    state.cohort = null;
    state.signature = null;
    state.ml = null;
    state.dl = null;
    refs.workspace.classList.add("hidden");
    refs.landing.classList.remove("hidden", "fade-out");
    refs.datasetBadge.classList.add("hidden");
    refs.datasetFile.value = "";
    runtime.guidedGoal = null;
    runtime.guidedStep = 1;
    runtime.deriveDraftTouched = false;
    runtime.predictiveFamily = "ml";
    runtime.workbenchRevealed = false;
    runtime.predictiveWorkbenchIntent = null;
    runtime.compareCache.ml = null;
    runtime.compareCache.dl = null;
    runtime.compareCache.unified = null;
    runtime.resultPreference.ml = "single";
    runtime.resultPreference.dl = "single";
    resetCoxPreview({ rerender: false });
    renderSharedFeatureSummary();
    renderGuidedChrome();
    setRuntimeBanner("");
    if (syncHistory) syncHistoryStateFn(historyMode);
  }

  window.SurvStudioShell = {
    currentHistoryState,
    goHome,
    shutdownServer,
    syncHistoryState,
  };
}());

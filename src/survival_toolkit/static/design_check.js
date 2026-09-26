// Design check page: describes a multi-algorithm signature study and shows the audit from /api/design-audit.
(function designCheckPage() {
  const form = document.getElementById("designForm");
  if (!form) return;
  const field = (id) => document.getElementById(id);
  const SEVERITY_LABELS = { high: "High risk", moderate: "Moderate", info: "Note" };

  function cohortRow(containerId, { name = "", n = "", events = "" } = {}) {
    const row = document.createElement("div");
    row.className = "cohort-row";
    row.innerHTML = `
      <label class="config-field"><span>Name</span><input type="text" data-cohort-field="name" maxlength="120" placeholder="e.g. GSE72094" /></label>
      <label class="config-field narrow"><span>Patients</span><input type="number" data-cohort-field="n" min="2" step="1" required /></label>
      <label class="config-field narrow"><span>Events</span><input type="number" data-cohort-field="events" min="1" step="1" placeholder="optional" /></label>
      <button class="button ghost compact-btn" type="button" data-remove-cohort aria-label="Remove cohort">Remove</button>
    `;
    row.querySelector('[data-cohort-field="name"]').value = name;
    row.querySelector('[data-cohort-field="n"]').value = n;
    row.querySelector('[data-cohort-field="events"]').value = events;
    field(containerId).appendChild(row);
  }

  function cohorts(containerId) {
    return [...field(containerId).querySelectorAll(".cohort-row")]
      .map((row) => {
        const value = (key) => row.querySelector(`[data-cohort-field="${key}"]`).value.trim();
        return { name: value("name"), n: value("n") === "" ? null : Number(value("n")), events: value("events") === "" ? null : Number(value("events")) };
      })
      .filter((cohort) => cohort.n !== null || cohort.name);
  }

  function formatNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "NA";
  }

  function formatRange(range) {
    return Array.isArray(range) && range[0] !== null ? `Range over the simulated scenarios: ${formatNumber(range[0])} to ${formatNumber(range[1])}.` : "";
  }

  function showError(message) {
    const error = field("designError");
    error.textContent = message;
    error.classList.toggle("hidden", !message);
  }

  function errorText(detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => `${(item.loc || []).filter((part) => part !== "body").join(" > ")}: ${item.msg}`).join(" | ");
    }
    return "The design could not be checked.";
  }

  function renderResult(result) {
    const optimism = result.expected_optimism || {};
    const regret = result.expected_regret || {};
    field("optimismValue").textContent = `+${formatNumber(optimism.value)}`;
    field("optimismNote").textContent = [optimism.note, formatRange(optimism.range)].filter(Boolean).join(" ");
    field("regretValue").textContent = formatNumber(regret.value);
    field("regretNote").textContent = [
      regret.note,
      formatRange(regret.range),
      regret.if_chosen_on_selection_cohorts_only != null && regret.if_chosen_on_selection_cohorts_only !== regret.value
        ? `Choosing on the selection cohorts alone: ${formatNumber(regret.if_chosen_on_selection_cohorts_only)}.`
        : "",
    ].filter(Boolean).join(" ");
    const list = field("designFlags");
    list.innerHTML = "";
    (result.flags || []).forEach((flag) => {
      const item = document.createElement("li");
      item.className = `design-flag design-flag-${flag.severity}`;
      const badge = document.createElement("span");
      badge.className = "design-flag-badge";
      badge.textContent = SEVERITY_LABELS[flag.severity] || flag.severity;
      const message = document.createElement("p");
      message.textContent = flag.message;
      const remedy = document.createElement("p");
      remedy.className = "design-flag-remedy";
      remedy.textContent = flag.remedy;
      item.append(badge, message, remedy);
      list.appendChild(item);
    });
    if (!(result.flags || []).length) {
      const item = document.createElement("li");
      item.className = "design-flag design-flag-ok";
      item.textContent = "No risky practices flagged.";
      list.appendChild(item);
    }
    const placement = result.placement || {};
    field("designMapNote").textContent = `Estimates from a simulation map (${result.map?.version || "benchmark pilot"}), placed at ${placement.selection_cohorts} selection cohort(s) with a median of ${placement.median_cohort_size} ${placement.size_unit}, ${placement.candidate_models} candidate models, ${placement.features}.`;
    field("designResult").classList.remove("hidden");
    field("designResult").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function checkDesign(event) {
    event.preventDefault();
    showError("");
    const body = {
      candidate_models: Number(field("candidateModels").value),
      gene_only: field("geneOnly").value === "true",
      selection_cohorts: cohorts("selectionCohorts"),
      sealed_cohorts: cohorts("sealedCohorts"),
      headline: field("headline").value,
      training_in_selection: field("trainingInSelection").checked,
      prefilter_used_validation_outcomes: field("prefilterLeak").checked,
      refit_in_validation: field("refitInValidation").checked,
      cutoff_per_cohort: field("cutoffPerCohort").checked,
      compared_with_clinical: field("comparedWithClinical").checked,
    };
    if (!body.selection_cohorts.length) {
      showError("Add at least one cohort that was used to choose the model.");
      return;
    }
    const button = field("checkDesignButton");
    button.disabled = true;
    try {
      const response = await fetch("/api/design-audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        showError(errorText(payload.detail));
        return;
      }
      renderResult(payload);
    } catch {
      showError("The local SurvStudio server did not answer.");
    } finally {
      button.disabled = false;
    }
  }

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const add = target?.closest("[data-add-cohort]");
    if (add) cohortRow(add.dataset.addCohort);
    const remove = target?.closest("[data-remove-cohort]");
    if (remove) remove.closest(".cohort-row")?.remove();
  });
  form.addEventListener("submit", checkDesign);
  cohortRow("selectionCohorts", { name: "Validation 1", n: 200, events: 80 });
}());

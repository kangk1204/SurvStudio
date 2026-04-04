(function registerSurvStudioDownloads() {
  function slugifyDownloadToken(value, fallback = "na") {
    const text = String(value ?? "").trim().toLowerCase();
    if (!text) return fallback;
    const slug = text
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 48);
    return slug || fallback;
  }

  function currentDatasetSlug(state) {
    return slugifyDownloadToken(state?.dataset?.filename || "survstudio_dataset", "survstudio_dataset");
  }

  function currentOutcomeSlug(refs) {
    return [
      slugifyDownloadToken(refs?.timeColumn?.value || "time", "time"),
      slugifyDownloadToken(refs?.eventColumn?.value || "event", "event"),
    ].join("_");
  }

  function currentGroupSlug(refs) {
    return slugifyDownloadToken(refs?.groupColumn?.value || "overall", "overall");
  }

  function buildDownloadFilename({ state, refs, stem, ext, includeGroup = false, template = null }) {
    const parts = [currentDatasetSlug(state), currentOutcomeSlug(refs)];
    if (includeGroup) parts.push(currentGroupSlug(refs));
    parts.push(slugifyDownloadToken(stem, "export"));
    if (template) parts.push(slugifyDownloadToken(template, "default"));
    return `${parts.filter(Boolean).join("_")}.${ext}`;
  }

  function triggerBlobDownload(filename, blob, fallbackMimeType = "") {
    const safeBlob = fallbackMimeType && !blob.type
      ? new Blob([blob], { type: fallbackMimeType })
      : blob;
    const url = URL.createObjectURL(safeBlob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    try {
      anchor.click();
    } finally {
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }

  function downloadCsv({ filename, rows, columns = null, showToast }) {
    if (!rows || rows.length === 0) {
      showToast?.("No rows available for export.", "warning");
      return;
    }
    const visibleColumns = columns || Object.keys(rows[0]);
    const sanitizeCsvCell = (value) => {
      const text = value === null || value === undefined ? "" : String(value);
      const trimmed = text.trimStart();
      if (trimmed.startsWith("'") && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed.slice(1))) {
        return `${text.slice(0, text.length - trimmed.length)}${trimmed.slice(1)}`;
      }
      if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) return text;
      if (trimmed.startsWith("=") || trimmed.startsWith("+") || trimmed.startsWith("-") || trimmed.startsWith("@")) {
        return `'${text}`;
      }
      return text;
    };
    const escapeCell = (value) => {
      const text = sanitizeCsvCell(value);
      return `"${text.replaceAll('"', '""')}"`;
    };
    const lines = [
      visibleColumns.map(escapeCell).join(","),
      ...rows.map((row) => visibleColumns.map((column) => escapeCell(row[column])).join(",")),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    triggerBlobDownload(filename, blob);
  }

  function downloadText({ filename, text, mimeType = "text/plain;charset=utf-8;" }) {
    const blob = new Blob([text], { type: mimeType });
    triggerBlobDownload(filename, blob);
  }

  async function downloadServerTable({ filename, payload, apiUrl, showToast, fallbackMimeType = "text/plain;charset=utf-8;" }) {
    if (!payload?.rows || payload.rows.length === 0) {
      showToast?.("No rows available for export.", "warning");
      return;
    }
    const response = await fetch(apiUrl("/api/export-table"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Export failed.");
    }
    const blob = await response.blob();
    triggerBlobDownload(filename, blob, fallbackMimeType);
  }

  function buildMarkdownTable(rows, { caption = "", notes = [], formatValue = (value) => value } = {}) {
    if (!rows || rows.length === 0) return "";
    const columns = Object.keys(rows[0]);
    const escapeCell = (value) => String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
    const header = `| ${columns.join(" | ")} |`;
    const divider = `| ${columns.map(() => "---").join(" | ")} |`;
    const body = rows.map((row) => `| ${columns.map((column) => escapeCell(formatValue(row[column]))).join(" | ")} |`);
    const sections = [];
    if (caption) sections.push(`**${caption}**`);
    sections.push([header, divider, ...body].join("\n"));
    if (notes.length) {
      sections.push("Notes:");
      sections.push(notes.map((note) => `- ${note}`).join("\n"));
    }
    return `${sections.join("\n\n")}\n`;
  }

  function downloadPlotImage({ plotEl, filename, format }) {
    if (!plotEl || !plotEl.data) return;
    window.Plotly.downloadImage(plotEl, {
      format,
      filename,
      height: 900,
      width: 1400,
      scale: format === "png" ? 3 : 1,
    });
  }

  function isReadonlyPlot(filename) {
    return ["dl_loss", "ml_importance", "shap_importance", "dl_importance"].includes(filename);
  }

  function plotLayoutConfig(layout, filename) {
    const nextLayout = { ...(layout || {}) };
    if (isReadonlyPlot(filename)) {
      nextLayout.dragmode = false;
    }
    return nextLayout;
  }

  window.SurvStudioDownloads = {
    buildDownloadFilename,
    buildMarkdownTable,
    currentDatasetSlug,
    currentGroupSlug,
    currentOutcomeSlug,
    downloadCsv,
    downloadPlotImage,
    downloadServerTable,
    downloadText,
    isReadonlyPlot,
    plotLayoutConfig,
    slugifyDownloadToken,
    triggerBlobDownload,
  };
}());

(function registerSurvStudioDownloads() {
  function parseExportErrorResponse(payload, fallbackText = "") {
    const detail = payload?.detail;
    if (typeof detail === "string" && detail.trim()) return detail.trim();
    if (Array.isArray(detail) && detail.length) {
      return detail.map((item) => {
        const path = Array.isArray(item?.loc)
          ? item.loc.filter((part) => part !== "body").join(" > ")
          : "";
        const message = item?.msg || "Invalid input.";
        return path ? `${path}: ${message}` : message;
      }).join(" | ");
    }
    if (typeof fallbackText === "string" && fallbackText.trim()) return fallbackText.trim();
    return "Request failed.";
  }

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

  function buildDownloadFilename({ state, refs, stem, ext, includeGroup = false, template = null, group = null }) {
    const parts = [currentDatasetSlug(state), currentOutcomeSlug(refs)];
    // `group` names the grouping the exported result actually used; fall back to the live control.
    if (includeGroup) parts.push(group === null ? currentGroupSlug(refs) : slugifyDownloadToken(group || "overall", "overall"));
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

  const UTF8_BOM = "\uFEFF";

  // Signed display values such as "-0.50 ± 1.20", "-12%", "-1,234" or "-1.2 (−3.4 to 0.5)" are data,
  // not formulas: they contain no letters (other than "to" and an exponent), so they cannot call functions.
  function isNumericLikeText(value) {
    const compact = String(value ?? "").replace(/\bto\b/gi, " ");
    return /^[+-]?(?:\d|\.\d)/.test(compact) && /^[\d\s.,%±()[\]/:;+\-–—−eE]*$/.test(compact);
  }

  function downloadCsv({ filename, rows, columns = null, showToast, caption = "", notes = [] }) {
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
      if (/^[\t\r]/.test(text) || trimmed.startsWith("=") || trimmed.startsWith("@")) {
        return `'${text}`;
      }
      if ((trimmed.startsWith("+") || trimmed.startsWith("-")) && !isNumericLikeText(trimmed)) {
        return `'${text}`;
      }
      return text;
    };
    const escapeCell = (value) => {
      const text = sanitizeCsvCell(value);
      return `"${text.replaceAll('"', '""')}"`;
    };
    const commentLine = (text) => `# ${sanitizeCsvCell(String(text ?? "").replace(/[\r\n]+/g, " "))}`;
    const cleanNotes = (notes || []).map((note) => String(note ?? "").trim()).filter(Boolean);
    const lines = [
      ...(caption ? [commentLine(caption)] : []),
      ...(cleanNotes.length ? ["# Notes:", ...cleanNotes.map((note) => commentLine(`- ${note}`))] : []),
      visibleColumns.map(escapeCell).join(","),
      ...rows.map((row) => visibleColumns.map((column) => escapeCell(row[column])).join(",")),
    ];
    // The BOM makes Excel open the file as UTF-8 so symbols such as "±" survive.
    const blob = new Blob([UTF8_BOM, lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    triggerBlobDownload(filename, blob);
  }

  async function withCsvByteOrderMark(blob, filename, mimeType) {
    const isCsv = /\.csv$/i.test(String(filename || "")) || /text\/csv/i.test(String(blob?.type || mimeType || ""));
    if (!isCsv || typeof blob?.slice !== "function") return blob;
    try {
      const head = new Uint8Array(await blob.slice(0, 3).arrayBuffer());
      if (head[0] === 0xef && head[1] === 0xbb && head[2] === 0xbf) return blob;
    } catch {
      return blob;
    }
    return new Blob([UTF8_BOM, blob], { type: blob.type || mimeType || "text/csv;charset=utf-8;" });
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
      const rawText = await response.text();
      let errorPayload = {};
      if (rawText.trim()) {
        try {
          errorPayload = JSON.parse(rawText);
        } catch {
          // Export errors can be plain text from the backend or an upstream proxy.
          errorPayload = {};
        }
      }
      throw new Error(parseExportErrorResponse(errorPayload, rawText || "Export failed."));
    }
    const blob = await withCsvByteOrderMark(await response.blob(), filename, fallbackMimeType);
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
    isNumericLikeText,
    isReadonlyPlot,
    plotLayoutConfig,
    slugifyDownloadToken,
    triggerBlobDownload,
  };
}());

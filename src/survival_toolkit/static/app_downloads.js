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

  // The same rules as the server's CSV export (_is_number_like_cell and _sanitize_csv_cell in app.py), so a
  // table exported here and one exported by the server treat every cell alike.
  const SIGNED_NUMERIC_LITERAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
  const NUMBER_LIKE_CELL_CHARS = /^[0-9.\s±%()[\],;:/|–−+-]*$/;
  const NUMBER_LIKE_EXPONENT = /(?<=[0-9.])[eE][+-]?(?=[0-9])/g;
  const NUMBER_LIKE_TO_WORD = /(?<=[\s0-9])to(?=[\s+\-−0-9])/g;
  const FORMULA_TRIGGER_CHARS = ["=", "@", "\t", "\r"];

  // Signed display values such as "-0.50 ± 1.20", "-12%", "-1,234" or "-1.2 (−3.4 to 0.5)" are data, not
  // formulas: after an optionally signed leading number they hold only digits and numeric punctuation (plus
  // "to" between numbers and an exponent after a digit), so they cannot spell a function call or a cell
  // reference such as the "E1" in "-1+E1".
  function isNumericLikeText(value) {
    const stripped = String(value ?? "").trim();
    const body = ["+", "-", "−"].includes(stripped.slice(0, 1)) ? stripped.slice(1) : stripped;
    if (!body) return false;
    if (!(/^\d/.test(body) || /^\.\d/.test(body))) return false;
    const reduced = stripped.replace(NUMBER_LIKE_EXPONENT, "").replace(NUMBER_LIKE_TO_WORD, " ");
    return NUMBER_LIKE_CELL_CHARS.test(reduced);
  }

  // Neutralize spreadsheet formula injection in one CSV cell. Control characters become spaces first, as on
  // the server, so one cannot hide a formula trigger behind it.
  function sanitizeCsvCell(value) {
    const text = value === null || value === undefined
      ? ""
      : String(value).replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]/g, " ");
    const stripped = text.replace(/^ +/, "");
    if (stripped.startsWith("'") && SIGNED_NUMERIC_LITERAL.test(stripped.slice(1))) {
      return `${text.slice(0, text.length - stripped.length)}${stripped.slice(1)}`;
    }
    if (FORMULA_TRIGGER_CHARS.some((char) => text.startsWith(char) || stripped.startsWith(char))) return `'${text}`;
    // A bare "-" placeholder cannot form a formula.
    if ((stripped.startsWith("+") || stripped.startsWith("-")) && stripped.replace(/[+\- ]/g, "") && !isNumericLikeText(stripped)) {
      return `'${text}`;
    }
    return text;
  }

  function escapeCsvCell(value) {
    return `"${sanitizeCsvCell(value).replaceAll('"', '""')}"`;
  }

  function buildCsvText({ rows, columns = null, caption = "", notes = [] }) {
    const visibleColumns = columns || Object.keys(rows[0]);
    // Each preamble line is one quoted cell, so a comma in a caption or note (for example a file name)
    // cannot start a new cell that a spreadsheet would read as a formula.
    const commentLine = (text) => escapeCsvCell(`# ${String(text ?? "").replace(/[\r\n]+/g, " ")}`);
    const cleanNotes = (notes || []).map((note) => String(note ?? "").trim()).filter(Boolean);
    const lines = [
      ...(caption ? [commentLine(caption)] : []),
      ...(cleanNotes.length ? [commentLine("Notes:"), ...cleanNotes.map((note) => commentLine(`- ${note}`))] : []),
      visibleColumns.map(escapeCsvCell).join(","),
      ...rows.map((row) => visibleColumns.map((column) => escapeCsvCell(row[column])).join(",")),
    ];
    return lines.join("\n");
  }

  function downloadCsv({ filename, rows, columns = null, showToast, caption = "", notes = [] }) {
    if (!rows || rows.length === 0) {
      showToast?.("No rows available for export.", "warning");
      return;
    }
    // The BOM makes Excel open the file as UTF-8 so symbols such as "±" survive.
    const blob = new Blob([UTF8_BOM, buildCsvText({ rows, columns, caption, notes })], { type: "text/csv;charset=utf-8;" });
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

  // Resolves when the image was handed to the browser; rejects (for the caller to report) when Plotly fails.
  function downloadPlotImage({ plotEl, filename, format }) {
    if (!plotEl || !plotEl.data) return Promise.resolve();
    return Promise.resolve(window.Plotly.downloadImage(plotEl, {
      format,
      filename,
      height: 900,
      width: 1400,
      scale: format === "png" ? 3 : 1,
    }));
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
    buildCsvText,
    buildDownloadFilename,
    downloadCsv,
    downloadPlotImage,
    downloadServerTable,
    downloadText,
    isNumericLikeText,
    isReadonlyPlot,
    plotLayoutConfig,
    sanitizeCsvCell,
    triggerBlobDownload,
  };
}());

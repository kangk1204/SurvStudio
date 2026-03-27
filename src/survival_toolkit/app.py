from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
import zipfile
from typing import Any, Literal, NoReturn
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from survival_toolkit.analysis import (
    compute_cohort_table,
    compute_cox_analysis,
    compute_km_analysis,
    discover_feature_signature,
    derive_group_column,
    load_dataframe_from_path,
    profile_dataframe,
)
from survival_toolkit.plots import build_cox_forest_figure, build_km_figure
from survival_toolkit.sample_data import (
    load_gbsg2_upload_ready_dataset,
    load_tcga_luad_example_dataset,
    load_tcga_luad_upload_ready_dataset,
    make_example_dataset,
)
from survival_toolkit.store import DatasetStore

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Survival Analysis Toolkit",
    description="Local survival analysis dashboard for exploratory and validation-oriented cohort work.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
store = DatasetStore()


# ── Request models ──────────────────────────────────────────────


class DeriveGroupRequest(BaseModel):
    dataset_id: str
    source_column: str
    method: Literal["median_split", "tertile_split", "quartile_split", "custom_cutoff", "optimal_cutpoint"]
    new_column_name: str | None = None
    cutoff: float | None = None
    lower_label: str = "Low"
    upper_label: str = "High"
    time_column: str | None = None
    event_column: str | None = None
    event_positive_value: Any = None


class KaplanMeierRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    group_column: str | None = None
    event_positive_value: Any = 1
    time_unit_label: str = "Months"
    confidence_level: float = Field(default=0.95, gt=0.5, lt=0.999)
    max_time: float | None = Field(default=None, gt=0)
    risk_table_points: int = Field(default=6, ge=4, le=12)
    show_confidence_bands: bool = True
    logrank_weight: Literal["logrank", "gehan_breslow", "tarone_ware", "fleming_harrington"] = "logrank"
    fh_p: float = Field(default=1.0, ge=0.0, le=5.0)


class CoxRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    covariates: list[str]
    categorical_covariates: list[str] = Field(default_factory=list)


class CohortTableRequest(BaseModel):
    dataset_id: str
    variables: list[str]
    group_column: str | None = None


class SignatureSearchRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    candidate_columns: list[str]
    max_combination_size: int = Field(default=3, ge=1, le=4)
    top_k: int = Field(default=15, ge=3, le=50)
    min_group_fraction: float = Field(default=0.1, gt=0.02, lt=0.45)
    bootstrap_iterations: int = Field(default=30, ge=0, le=120)
    bootstrap_sample_fraction: float = Field(default=0.8, ge=0.4, le=1.0)
    permutation_iterations: int = Field(default=0, ge=0, le=400)
    validation_iterations: int = Field(default=0, ge=0, le=80)
    validation_fraction: float = Field(default=0.35, ge=0.2, le=0.6)
    significance_level: float = Field(default=0.05, gt=0.0, le=0.2)
    combination_operator: Literal["and", "or", "mixed"] = "mixed"
    random_seed: int = Field(default=20260311, ge=0)
    new_column_name: str | None = None


class MLModelRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str]
    categorical_features: list[str] = Field(default_factory=list)
    model_type: Literal["rsf", "gbs", "compare"]
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = None
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = 42
    evaluation_strategy: Literal["holdout", "repeated_cv"] = "holdout"
    cv_folds: int = Field(default=5, ge=2, le=10)
    cv_repeats: int = Field(default=3, ge=1, le=20)


class DeepModelRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str]
    categorical_features: list[str] = Field(default_factory=list)
    model_type: Literal["deepsurv", "deephit", "mtlr", "transformer", "vae", "compare"]
    hidden_layers: list[int] = Field(default=[64, 64])
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)
    epochs: int = Field(default=100, ge=10, le=500)
    batch_size: int = Field(default=64, ge=8, le=512)
    random_seed: int = 42
    evaluation_strategy: Literal["holdout", "repeated_cv"] = "holdout"
    cv_folds: int = Field(default=5, ge=2, le=10)
    cv_repeats: int = Field(default=3, ge=1, le=20)
    early_stopping_patience: int | None = Field(default=10, ge=1, le=100)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0.0, le=0.1)
    parallel_jobs: int = Field(default=1, ge=1, le=16)
    # DeepHit / MTLR specific
    num_time_bins: int = Field(default=50, ge=10, le=200)
    # Transformer specific
    n_heads: int = Field(default=4, ge=1, le=16)
    d_model: int = Field(default=64, ge=16, le=256)
    n_layers: int = Field(default=2, ge=1, le=8)
    # VAE specific
    latent_dim: int = Field(default=8, ge=2, le=32)
    n_clusters: int = Field(default=3, ge=2, le=10)


class OptimalCutpointRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    variable: str
    min_group_fraction: float = Field(default=0.1, gt=0.02, lt=0.45)
    permutation_iterations: int = Field(default=100, ge=0, le=500)


class TableExportRequest(BaseModel):
    rows: list[dict[str, Any]]
    format: Literal["csv", "markdown", "latex", "docx"]
    style: Literal["plain", "journal"] = "journal"
    template: Literal["default", "nejm", "lancet", "jco"] = "default"
    caption: str | None = None
    notes: list[str] = Field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────


def dataset_response(dataset_id: str) -> dict[str, Any]:
    stored = store.get(dataset_id)
    return profile_dataframe(stored.dataframe, dataset_id=stored.dataset_id, filename=stored.filename)


def fail_bad_request(exc: Exception) -> NoReturn:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ImportError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (ValueError, TypeError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


def _format_export_value(value: Any, style: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if value != value or value in {float("inf"), float("-inf")}:
                return ""
            if style == "journal":
                return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)
    return str(value)


EXPORT_TEMPLATE_PROFILES: dict[str, dict[str, str]] = {
    "default": {
        "markdown_open": "**",
        "markdown_close": "**",
        "notes_heading": "Notes",
        "latex_position": "htbp",
        "latex_size": "\\small",
    },
    "nejm": {
        "markdown_open": "*",
        "markdown_close": "*",
        "notes_heading": "Notes",
        "latex_position": "t",
        "latex_size": "\\footnotesize",
    },
    "lancet": {
        "markdown_open": "",
        "markdown_close": "",
        "notes_heading": "Comments",
        "latex_position": "t",
        "latex_size": "\\small",
    },
    "jco": {
        "markdown_open": "**",
        "markdown_close": "**",
        "notes_heading": "Footnotes",
        "latex_position": "htbp",
        "latex_size": "\\small",
    },
}


def _export_template_profile(template: str) -> dict[str, str]:
    return EXPORT_TEMPLATE_PROFILES.get(template, EXPORT_TEMPLATE_PROFILES["default"])


def _normalize_export_text(value: Any, style: str) -> str:
    return _format_export_value(value, style).replace("\n", " ").strip()


def _default_export_caption(template: str) -> str:
    defaults = {
        "default": "Table 1. Model performance summary.",
        "nejm": "Table 1. Model discrimination summary.",
        "lancet": "Table 1. Model discrimination summary",
        "jco": "Table 1. Model performance summary.",
    }
    return defaults.get(template, defaults["default"])


def _resolve_export_caption(caption: str | None, template: str) -> str:
    clean_caption = (caption or "").strip()
    return clean_caption or _default_export_caption(template)


def _export_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("No rows available for export.")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row.keys():
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def _export_rows_to_csv(rows: list[dict[str, Any]], style: str) -> str:
    if not rows:
        raise ValueError("No rows available for export.")
    columns = _export_columns(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_format_export_value(row.get(column), style) for column in columns])
    return buffer.getvalue()


def _export_rows_to_markdown(
    rows: list[dict[str, Any]],
    *,
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> str:
    if not rows:
        raise ValueError("No rows available for export.")
    columns = _export_columns(rows)
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    header = f"| {' | '.join(columns)} |"
    divider = f"| {' | '.join(['---'] * len(columns))} |"
    body = [
        "| "
        + " | ".join(_normalize_export_text(row.get(column), style).replace("|", "\\|") for column in columns)
        + " |"
        for row in rows
    ]
    sections = []
    if resolved_caption:
        open_marker = template_profile["markdown_open"]
        close_marker = template_profile["markdown_close"]
        sections.append(f"{open_marker}{resolved_caption}{close_marker}" if open_marker or close_marker else resolved_caption)
    sections.append("\n".join([header, divider, *body]))
    if notes:
        sections.append(f"{template_profile['notes_heading']}:")
        sections.append("\n".join(f"- {note}" for note in notes))
    return "\n\n".join(sections) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for source, replacement in replacements.items():
        escaped = escaped.replace(source, replacement)
    return escaped


def _export_rows_to_latex(
    rows: list[dict[str, Any]],
    *,
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> str:
    if not rows:
        raise ValueError("No rows available for export.")
    columns = _export_columns(rows)
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    column_spec = "l" * len(columns)
    body = [
        " & ".join(_latex_escape(_normalize_export_text(row.get(column), style)) for column in columns) + r" \\"
        for row in rows
    ]
    lines = [
        f"\\begin{{table}}[{template_profile['latex_position']}]",
        "\\centering",
        template_profile["latex_size"],
        f"\\caption{{{_latex_escape(resolved_caption)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in columns) + r" \\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if notes:
        notes_heading = _latex_escape(template_profile["notes_heading"])
        notes_text = " ".join(_latex_escape(note.strip()) for note in notes if note.strip())
        if notes_text:
            lines.extend(
                [
                    "\\vspace{0.35em}",
                    "\\begin{minipage}{0.96\\linewidth}",
                    f"\\footnotesize\\textit{{{notes_heading}:}} {notes_text}",
                    "\\end{minipage}",
                ]
            )
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _docx_run(text: str, *, bold: bool = False, italic: bool = False) -> str:
    properties: list[str] = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    props_xml = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    safe_text = xml_escape(text.replace("\n", " "))
    return f'<w:r>{props_xml}<w:t xml:space="preserve">{safe_text}</w:t></w:r>'


def _docx_paragraph(text: str, *, bold: bool = False, italic: bool = False) -> str:
    if not text:
        return "<w:p/>"
    return f"<w:p>{_docx_run(text, bold=bold, italic=italic)}</w:p>"


def _docx_cell(text: str, *, width: int, bold: bool = False) -> str:
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
        f"{_docx_paragraph(text, bold=bold)}</w:tc>"
    )


def _docx_table(rows: list[dict[str, Any]], *, style: str) -> str:
    columns = _export_columns(rows)
    cell_width = max(1200, int(9000 / max(len(columns), 1)))
    grid = "".join(f'<w:gridCol w:w="{cell_width}"/>' for _ in columns)
    borders = (
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )
    header_row = "<w:tr>" + "".join(_docx_cell(column, width=cell_width, bold=True) for column in columns) + "</w:tr>"
    body_rows = [
        "<w:tr>"
        + "".join(_docx_cell(_normalize_export_text(row.get(column), style), width=cell_width) for column in columns)
        + "</w:tr>"
        for row in rows
    ]
    return (
        "<w:tbl>"
        f"<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>{borders}</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{header_row}{''.join(body_rows)}"
        "</w:tbl>"
    )


def _export_rows_to_docx(
    rows: list[dict[str, Any]],
    *,
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> bytes:
    if not rows:
        raise ValueError("No rows available for export.")
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    caption_bold = template in {"default", "jco"}
    caption_italic = template == "nejm"
    body_parts = [
        _docx_paragraph(resolved_caption, bold=caption_bold, italic=caption_italic),
        _docx_table(rows, style=style),
    ]
    clean_notes = [note.strip() for note in notes if note.strip()]
    if clean_notes:
        body_parts.append(_docx_paragraph(f"{template_profile['notes_heading']}:", italic=True))
        for note in clean_notes:
            body_parts.append(_docx_paragraph(note, italic=True))
    body_parts.append(
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body_parts)}</w:body>"
        "</w:document>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


# ── Core endpoints ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    temp_path: Path | None = None
    filename = file.filename or "uploaded_dataset.csv"
    try:
        suffix = Path(filename).suffix or ".csv"
        total_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds the 200 MB limit.")
                temp_file.write(chunk)
        dataframe = await run_in_threadpool(load_dataframe_from_path, temp_path)
        stored = store.create(dataframe, filename=filename, copy_dataframe=False)
        return await run_in_threadpool(dataset_response, stored.dataset_id)
    except HTTPException:
        raise
    except Exception as exc:
        fail_bad_request(exc)
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/api/load-example")
async def load_example() -> dict[str, Any]:
    dataframe = make_example_dataset()
    stored = store.create(dataframe, filename="example_survival_cohort", copy_dataframe=False)
    return dataset_response(stored.dataset_id)


@app.post("/api/load-tcga-example")
async def load_tcga_example() -> dict[str, Any]:
    dataframe = load_tcga_luad_example_dataset()
    stored = store.create(dataframe, filename="tcga_luad_xena_example", copy_dataframe=False)
    return dataset_response(stored.dataset_id)


@app.post("/api/load-tcga-upload-ready")
async def load_tcga_upload_ready() -> dict[str, Any]:
    dataframe = load_tcga_luad_upload_ready_dataset()
    stored = store.create(dataframe, filename="tcga_luad_upload_ready", copy_dataframe=False)
    return dataset_response(stored.dataset_id)


@app.post("/api/load-gbsg2-example")
async def load_gbsg2_example() -> dict[str, Any]:
    dataframe = load_gbsg2_upload_ready_dataset()
    stored = store.create(dataframe, filename="gbsg2_upload_ready", copy_dataframe=False)
    return dataset_response(stored.dataset_id)


@app.get("/api/dataset/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        return dataset_response(dataset_id)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/export-table")
async def export_table(request_model: TableExportRequest) -> Response:
    try:
        if request_model.format == "csv":
            content = _export_rows_to_csv(request_model.rows, request_model.style)
            return Response(content=content, media_type="text/csv; charset=utf-8")
        if request_model.format == "markdown":
            content = _export_rows_to_markdown(
                request_model.rows,
                caption=request_model.caption,
                notes=request_model.notes,
                style=request_model.style,
                template=request_model.template,
            )
            return Response(content=content, media_type="text/markdown; charset=utf-8")
        if request_model.format == "latex":
            content = _export_rows_to_latex(
                request_model.rows,
                caption=request_model.caption,
                notes=request_model.notes,
                style=request_model.style,
                template=request_model.template,
            )
            return Response(content=content, media_type="text/x-tex; charset=utf-8")
        content = _export_rows_to_docx(
            request_model.rows,
            caption=request_model.caption,
            notes=request_model.notes,
            style=request_model.style,
            template=request_model.template,
        )
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/derive-group")
async def derive_group(request_model: DeriveGroupRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.plots import build_cutpoint_scan_figure

        stored = store.get(request_model.dataset_id)
        updated, column_name, summary = derive_group_column(
            stored.dataframe,
            source_column=request_model.source_column,
            method=request_model.method,
            new_column_name=request_model.new_column_name,
            cutoff=request_model.cutoff,
            lower_label=request_model.lower_label,
            upper_label=request_model.upper_label,
            time_column=request_model.time_column,
            event_column=request_model.event_column,
            event_positive_value=request_model.event_positive_value,
        )
        store.update_dataframe(request_model.dataset_id, updated, copy_dataframe=False)
        payload = dataset_response(request_model.dataset_id)
        payload["derived_column"] = column_name
        payload["derive_summary"] = summary
        if request_model.method == "optimal_cutpoint" and summary.get("scan_data"):
            payload["cutpoint_figure"] = build_cutpoint_scan_figure(summary, variable_name=request_model.source_column)
        return payload
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/kaplan-meier")
async def kaplan_meier(request_model: KaplanMeierRequest) -> dict[str, Any]:
    try:
        stored = store.get(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            analysis = compute_km_analysis(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                group_column=request_model.group_column,
                event_positive_value=request_model.event_positive_value,
                confidence_level=request_model.confidence_level,
                max_time=request_model.max_time,
                risk_table_points=request_model.risk_table_points,
                logrank_weight=request_model.logrank_weight,
                fh_p=request_model.fh_p,
            )
            figure = build_km_figure(
                analysis,
                time_unit_label=request_model.time_unit_label,
                show_confidence_bands=request_model.show_confidence_bands,
            )
            return {"analysis": analysis, "figure": figure}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cox")
async def cox(request_model: CoxRequest) -> dict[str, Any]:
    try:
        stored = store.get(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            analysis = compute_cox_analysis(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                covariates=request_model.covariates,
                categorical_covariates=request_model.categorical_covariates,
            )
            figure = build_cox_forest_figure(analysis)
            return {"analysis": analysis, "figure": figure}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cohort-table")
async def cohort_table(request_model: CohortTableRequest) -> dict[str, Any]:
    try:
        stored = store.get(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            return {"analysis": compute_cohort_table(
                stored.dataframe,
                variables=request_model.variables,
                group_column=request_model.group_column,
            )}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/discover-signature")
async def discover_signature(request_model: SignatureSearchRequest) -> dict[str, Any]:
    try:
        stored = store.get(request_model.dataset_id)

        def _run() -> tuple:
            return discover_feature_signature(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                candidate_columns=request_model.candidate_columns,
                max_combination_size=request_model.max_combination_size,
                top_k=request_model.top_k,
                min_group_fraction=request_model.min_group_fraction,
                bootstrap_iterations=request_model.bootstrap_iterations,
                bootstrap_sample_fraction=request_model.bootstrap_sample_fraction,
                permutation_iterations=request_model.permutation_iterations,
                validation_iterations=request_model.validation_iterations,
                validation_fraction=request_model.validation_fraction,
                significance_level=request_model.significance_level,
                combination_operator=request_model.combination_operator,
                random_seed=request_model.random_seed,
                new_column_name=request_model.new_column_name,
            )

        updated, column_name, analysis = await run_in_threadpool(_run)
        store.update_dataframe(request_model.dataset_id, updated, copy_dataframe=False)
        payload = dataset_response(request_model.dataset_id)
        payload["derived_column"] = column_name
        payload["signature_analysis"] = analysis
        return payload
    except Exception as exc:
        fail_bad_request(exc)


# ── ML model endpoints ──────────────────────────────────────────


@app.post("/api/optimal-cutpoint")
async def optimal_cutpoint(request_model: OptimalCutpointRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import find_optimal_cutpoint
        from survival_toolkit.plots import build_cutpoint_scan_figure

        stored = store.get(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            result = find_optimal_cutpoint(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                variable=request_model.variable,
                event_positive_value=request_model.event_positive_value,
                min_group_fraction=request_model.min_group_fraction,
                permutation_iterations=request_model.permutation_iterations,
            )
            figure = build_cutpoint_scan_figure(result, variable_name=request_model.variable)
            return {"result": result, "figure": figure}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/ml-model")
async def ml_model(request_model: MLModelRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import (
            train_random_survival_forest,
            train_gradient_boosted_survival,
            compare_survival_models,
            cross_validate_survival_models,
            compute_shap_values,
        )
        from survival_toolkit.plots import build_feature_importance_figure, build_model_comparison_figure

        stored = store.get(request_model.dataset_id)
        df = stored.dataframe

        def _run() -> dict[str, Any]:
            if request_model.model_type == "compare":
                if request_model.evaluation_strategy == "repeated_cv":
                    comparison = cross_validate_survival_models(
                        df,
                        time_column=request_model.time_column,
                        event_column=request_model.event_column,
                        features=request_model.features,
                        categorical_features=request_model.categorical_features,
                        event_positive_value=request_model.event_positive_value,
                        n_estimators=request_model.n_estimators,
                        max_depth=request_model.max_depth,
                        learning_rate=request_model.learning_rate,
                        cv_folds=request_model.cv_folds,
                        cv_repeats=request_model.cv_repeats,
                        random_state=request_model.random_state,
                    )
                else:
                    comparison = compare_survival_models(
                        df,
                        time_column=request_model.time_column,
                        event_column=request_model.event_column,
                        features=request_model.features,
                        categorical_features=request_model.categorical_features,
                        event_positive_value=request_model.event_positive_value,
                        n_estimators=request_model.n_estimators,
                        max_depth=request_model.max_depth,
                        learning_rate=request_model.learning_rate,
                        random_state=request_model.random_state,
                    )
                figure = build_model_comparison_figure(comparison)
                return {"analysis": comparison, "figure": figure}

            common_ml = dict(
                df=df,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
                event_positive_value=request_model.event_positive_value,
                n_estimators=request_model.n_estimators,
                max_depth=request_model.max_depth,
                random_state=request_model.random_state,
            )
            if request_model.model_type == "gbs":
                common_ml["learning_rate"] = request_model.learning_rate
                result = train_gradient_boosted_survival(**common_ml)
            else:
                result = train_random_survival_forest(**common_ml)
            importance_figure = build_feature_importance_figure(result["feature_importance"], model_name=request_model.model_type.upper())

            shap_result = None
            shap_figure = None
            model_obj = result.get("_model")
            x_encoded = result.get("_X_encoded")
            if model_obj is not None and x_encoded is not None:
                try:
                    shap_result = compute_shap_values(model_obj, x_encoded, result["feature_names"])
                    from survival_toolkit.plots import build_shap_figure
                    shap_figure = build_shap_figure(shap_result)
                except (MemoryError, KeyboardInterrupt):
                    raise
                except Exception:
                    pass  # SHAP is optional — log if logging is added

            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            return {
                "analysis": clean_result,
                "importance_figure": importance_figure,
                "shap_result": shap_result,
                "shap_figure": shap_figure,
            }

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/deep-model")
async def deep_model(request_model: DeepModelRequest) -> dict[str, Any]:
    try:
        from survival_toolkit import deep_models
        from survival_toolkit.plots import (
            build_feature_importance_figure,
            build_loss_curve_figure,
            build_model_comparison_figure,
        )

        stored = store.get(request_model.dataset_id)
        df = stored.dataframe

        def _run() -> dict[str, Any]:
            base = dict(
                df=df,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
                learning_rate=request_model.learning_rate,
                epochs=request_model.epochs,
                batch_size=request_model.batch_size,
                random_seed=request_model.random_seed,
                early_stopping_patience=request_model.early_stopping_patience,
                early_stopping_min_delta=request_model.early_stopping_min_delta,
            )

            if request_model.model_type == "compare":
                result = deep_models.compare_deep_survival_models(
                    **base,
                    hidden_layers=request_model.hidden_layers,
                    dropout=request_model.dropout,
                    num_time_bins=request_model.num_time_bins,
                    d_model=request_model.d_model,
                    n_heads=request_model.n_heads,
                    n_layers=request_model.n_layers,
                    latent_dim=request_model.latent_dim,
                    n_clusters=request_model.n_clusters,
                    evaluation_strategy=request_model.evaluation_strategy,
                    cv_folds=request_model.cv_folds,
                    cv_repeats=request_model.cv_repeats,
                    parallel_jobs=request_model.parallel_jobs,
                )
                clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
                return {
                    "analysis": clean_result,
                    "figures": {"comparison": build_model_comparison_figure(result)},
                }

            if request_model.model_type == "deepsurv":
                result = deep_models.train_deepsurv(**base, hidden_layers=request_model.hidden_layers, dropout=request_model.dropout)
            elif request_model.model_type == "deephit":
                result = deep_models.train_deephit(**base, hidden_layers=request_model.hidden_layers, dropout=request_model.dropout, num_time_bins=request_model.num_time_bins)
            elif request_model.model_type == "mtlr":
                result = deep_models.train_neural_mtlr(**base, hidden_layers=request_model.hidden_layers, num_time_bins=request_model.num_time_bins)
            elif request_model.model_type == "transformer":
                result = deep_models.train_survival_transformer(**base, d_model=request_model.d_model, n_heads=request_model.n_heads, n_layers=request_model.n_layers, dropout=request_model.dropout)
            elif request_model.model_type == "vae":
                result = deep_models.train_survival_vae(**base, latent_dim=request_model.latent_dim, hidden_dim=request_model.hidden_layers[0] if request_model.hidden_layers else 64, n_clusters=request_model.n_clusters, dropout=request_model.dropout)
            else:
                raise ValueError(f"Unknown model type: {request_model.model_type}")

            figures = {}
            if result.get("feature_importance"):
                figures["importance"] = build_feature_importance_figure(
                    result["feature_importance"],
                    model_name=request_model.model_type.upper(),
                )
            if result.get("loss_history"):
                figures["loss"] = build_loss_curve_figure(result["loss_history"], model_name=request_model.model_type.upper())

            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            if "scientific_summary" not in clean_result and "insight_board" in clean_result:
                clean_result["scientific_summary"] = clean_result["insight_board"]
            if "epochs_trained" not in clean_result:
                clean_result["epochs_trained"] = request_model.epochs
            return {"analysis": clean_result, "figures": figures}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


# ── XAI endpoints ──────────────────────────────────────────────


class TimeDependentImportanceRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str]
    categorical_features: list[str] = Field(default_factory=list)
    eval_times: list[float] | None = None


class CounterfactualRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str]
    categorical_features: list[str] = Field(default_factory=list)
    target_feature: str
    original_value: Any = None
    counterfactual_value: Any


class PDPRequest(BaseModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str]
    categorical_features: list[str] = Field(default_factory=list)
    target_feature: str


@app.post("/api/time-dependent-importance")
async def time_dependent_importance(request_model: TimeDependentImportanceRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import compute_time_dependent_importance
        from survival_toolkit.plots import build_time_dependent_importance_figure

        stored = store.get(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            result = compute_time_dependent_importance(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
                eval_times=request_model.eval_times,
            )
            figure = build_time_dependent_importance_figure(result)
            return {"analysis": result, "figure": figure}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/counterfactual")
async def counterfactual(request_model: CounterfactualRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import counterfactual_survival

        stored = store.get(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            return {"analysis": counterfactual_survival(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
                target_feature=request_model.target_feature,
                original_value=request_model.original_value,
                counterfactual_value=request_model.counterfactual_value,
            )}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/pdp")
async def pdp(request_model: PDPRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import (
            compute_partial_dependence,
            train_random_survival_forest,
        )
        from survival_toolkit.plots import build_pdp_figure

        stored = store.get(request_model.dataset_id)
        df = stored.dataframe

        def _run() -> dict[str, Any]:
            rsf_result = train_random_survival_forest(
                df,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
            )
            model = rsf_result["_model"]
            X_encoded = rsf_result["_X_encoded"]

            result = compute_partial_dependence(
                model,
                X_encoded,
                feature_name=request_model.target_feature,
                categorical_features=request_model.categorical_features,
            )
            figure = build_pdp_figure(result)
            return {"analysis": result, "figure": figure}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)

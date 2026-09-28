from __future__ import annotations

import asyncio
import contextvars
import copy
import csv
import functools
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
import platform
import re
import socket
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import threading
import time
import zipfile
from typing import Any, Callable, Literal, NoReturn, Sequence, TypeVar
from urllib.parse import urlsplit
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from survival_toolkit.analysis import (
    _cohort_frame,
    _model_feature_candidate_columns_from_metadata,
    _profile_dataframe_column,
    _survival_outcome_like_columns,
    compute_cohort_table,
    compute_cox_analysis,
    compute_km_analysis,
    discover_feature_signature,
    derive_group_column,
    ensure_model_feature_candidate_limit,
    find_event_equivalent_columns,
    load_dataframe_from_path,
    preview_rows,
    preview_cox_analysis_inputs,
    profile_dataframe,
    suggest_columns,
)
from survival_toolkit import __version__ as SURVSTUDIO_VERSION
from survival_toolkit.concurrency import cancellation_scope, raise_if_cancelled
from survival_toolkit.design_audit import audit_design, design_from_dict
from survival_toolkit.errors import (
    DependencyError,
    InternalAnalysisError,
    JobCancelledError,
    NotFoundError,
    UserInputError,
    _raised_by_survstudio,
    is_programming_error,
    must_propagate,
    user_input_boundary,
)
from survival_toolkit.evaluation import c_index_intervals, merge_prediction_blocks
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers, validate_locked_recipe
from survival_toolkit.marker_screen import _PAIRWISE_LIMIT as _C_INDEX_PAIRWISE_LIMIT
from survival_toolkit.marker_matrix import (
    MATRIX_SUFFIXES,
    MAX_DECOMPRESSED_BYTES,
    MAX_MATRIX_MARKERS,
    MAX_MATRIX_SAMPLES,
    ORIENTATIONS,
    MarkerMatrixStore,
    match_summary,
    matrix_format,
    matrix_frame,
    read_marker_matrix,
)
from survival_toolkit.reporting import (
    CHECKLIST_COLUMNS,
    checklist_intro,
    checklist_markdown,
    checklist_rows,
    remark_checklist,
    tripod_ai_checklist,
)
from survival_toolkit.sample_data import (
    load_gbsg2_upload_ready_dataset,
    load_tcga_luad_example_dataset,
    load_tcga_luad_upload_ready_dataset,
    make_example_dataset,
)
from survival_toolkit.store import DatasetStore

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def _package_version_or_unknown(distribution_name: str) -> str:
    try:
        return package_version(distribution_name)
    except PackageNotFoundError:
        return "not_installed"


def _static_asset_version() -> str:
    asset_roots = (
        BASE_DIR / "templates",
        BASE_DIR / "static",
    )
    digest = hashlib.sha256()
    saw_asset = False
    for asset_root in asset_roots:
        if not asset_root.exists():
            continue
        for asset_path in sorted(asset_root.rglob("*")):
            if not asset_path.is_file():
                continue
            try:
                stat = asset_path.stat()
            except OSError:
                continue
            saw_asset = True
            digest.update(asset_path.relative_to(BASE_DIR).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()[:12] if saw_asset else "0"


# ── Local request guard (CSRF / DNS rebinding) ──────────────────

BIND_HOST_ENV_VAR = "SURVSTUDIO_BIND_HOST"
ALLOWED_HOSTS_ENV_VAR = "SURVSTUDIO_ALLOWED_HOSTS"
_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::"})
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _normalize_hostname(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.split("%", 1)[0].rstrip(".")
    return text or None


def _split_host_and_port(value: str) -> tuple[str, int | None] | None:
    """Parse a Host-header style ``host[:port]`` value; ``None`` when malformed."""

    text = str(value or "").strip()
    if not text or any(character in text for character in "/?#@\\ \t"):
        return None
    try:
        parsed = urlsplit(f"//{text}")
        hostname = _normalize_hostname(parsed.hostname)
        port = parsed.port
    except ValueError:
        return None
    if hostname is None:
        return None
    return hostname, port


def _is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _primary_outbound_address() -> str | None:
    """Best-effort LAN address of this machine (a UDP connect sends no packets)."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return str(probe.getsockname()[0])
    except OSError:  # pragma: no cover - platform dependent
        return None


def _local_interface_hostnames() -> set[str]:
    names: set[str] = set()
    outbound = _normalize_hostname(_primary_outbound_address())
    if outbound:
        names.add(outbound)
    try:
        candidates = {socket.gethostname(), socket.getfqdn()}
    except OSError:  # pragma: no cover - platform dependent
        return names
    for candidate in candidates:
        normalized = _normalize_hostname(candidate)
        if normalized:
            names.add(normalized)
        try:
            for info in socket.getaddrinfo(candidate, None):
                address = _normalize_hostname(info[4][0])
                if address:
                    names.add(address)
        except OSError:
            continue
    return names


@functools.lru_cache(maxsize=16)
def _configured_request_hosts(bind_host: str, allowed_hosts: str) -> tuple[frozenset[str], bool]:
    """Return (extra allowed hostnames, allow_any) from the serve configuration."""

    hosts: set[str] = set()
    allow_any = False
    for raw_item in str(allowed_hosts or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item == "*":
            allow_any = True
            continue
        parsed = _split_host_and_port(item)
        if parsed is not None:
            hosts.add(parsed[0])
    bind = _normalize_hostname(bind_host)
    if bind in _WILDCARD_BIND_HOSTS:
        hosts.update(_local_interface_hostnames())
        # A page opened as http://0.0.0.0:<port> (or [::]) reaches this server and sends that literal as its Host.
        hosts.update(_WILDCARD_BIND_HOSTS)
    elif bind:
        hosts.add(bind)
    return frozenset(hosts), allow_any


def _request_hostname_is_allowed(hostname: str) -> bool:
    if _is_loopback_hostname(hostname):
        return True
    configured_hosts, allow_any = _configured_request_hosts(
        os.environ.get(BIND_HOST_ENV_VAR, ""),
        os.environ.get(ALLOWED_HOSTS_ENV_VAR, ""),
    )
    return allow_any or hostname in configured_hosts


def _default_port_for_scheme(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _request_origin_is_allowed(origin: str, host_header: str, *, source_is_referer: bool = False) -> bool:
    """Allow loopback origins (any port) and same-origin requests to an allowed Host."""

    text = str(origin or "").strip()
    if not text or text.lower() == "null":
        return False
    try:
        parsed = urlsplit(text)
        origin_hostname = _normalize_hostname(parsed.hostname)
        origin_port = parsed.port
    except ValueError:
        return False
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"} or origin_hostname is None:
        return False
    if not source_is_referer and (parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        return False
    if _is_loopback_hostname(origin_hostname):
        return True
    host = _split_host_and_port(host_header)
    if host is None:
        return False
    host_hostname, host_port = host
    return (
        origin_hostname == host_hostname
        and (origin_port or _default_port_for_scheme(scheme)) == (host_port or _default_port_for_scheme(scheme))
    )


def _local_request_rejection(method: str, headers: Headers) -> tuple[int, str] | None:
    host_header = headers.get("host", "")
    host = _split_host_and_port(host_header)
    if host is None or not _request_hostname_is_allowed(host[0]):
        return (
            400,
            "Invalid Host header. SurvStudio only answers requests addressed to localhost or to the host it was "
            f"started with. Set {ALLOWED_HOSTS_ENV_VAR} to allow additional host names.",
        )
    if method.upper() not in _STATE_CHANGING_METHODS:
        return None
    origin = headers.get("origin")
    if origin is not None:
        allowed = _request_origin_is_allowed(origin, host_header)
    else:
        referer = headers.get("referer")
        allowed = referer is None or _request_origin_is_allowed(referer, host_header, source_is_referer=True)
    if not allowed:
        return (
            403,
            "Cross-site request rejected. State-changing SurvStudio requests must come from the local SurvStudio page.",
        )
    return None


class LocalRequestGuardMiddleware:
    """Reject foreign Host headers (DNS rebinding) and cross-site state-changing requests (CSRF)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            rejection = _local_request_rejection(str(scope.get("method", "GET")), Headers(scope=scope))
            if rejection is not None:
                status_code, detail = rejection
                response = JSONResponse({"detail": detail}, status_code=status_code)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Every endpoint that receives a file: the table upload and the marker-matrix attachment.
_UPLOAD_PATHS = frozenset({"/api/upload", "/api/marker-matrix"})
_UPLOAD_TOO_LARGE_DETAIL = "Upload exceeds the 200 MB limit."
# Allowance for multipart boundaries and part headers on top of the file-size limit.
_UPLOAD_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
# Largest body of any other request (all JSON). The largest legitimate one, the test-set predictions
# of four comparisons of 10 models on 100,000 patients sent to the interval endpoint, is about 100 MB.
_MAX_JSON_BODY_BYTES = 128 * 1024 * 1024
_JSON_BODY_TOO_LARGE_DETAIL = "Request body exceeds the 128 MB limit."


def _max_upload_request_bytes() -> int:
    return int(_MAX_UPLOAD_BYTES) + _UPLOAD_MULTIPART_OVERHEAD_BYTES


def _request_body_limit(scope: Scope) -> tuple[int, str] | None:
    """Byte limit and message for a request's body; None for methods whose bodies no route reads."""

    if str(scope.get("method", "GET")).upper() not in _STATE_CHANGING_METHODS:
        return None
    if scope.get("path") in _UPLOAD_PATHS:
        return _max_upload_request_bytes(), _UPLOAD_TOO_LARGE_DETAIL
    return int(_MAX_JSON_BODY_BYTES), _JSON_BODY_TOO_LARGE_DETAIL


class UploadSizeLimitMiddleware:
    """Reject oversized request bodies from Content-Length before they are read.

    Uploads get the upload limit and every other state-changing request (JSON) the smaller body
    limit. Chunked requests without Content-Length are cut off as soon as the streamed body
    passes the limit, instead of being spooled to disk (multipart) or memory (JSON) in full first.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        body_limit = _request_body_limit(scope) if scope["type"] == "http" else None
        if body_limit is None:
            await self.app(scope, receive, send)
            return
        limit, too_large_detail = body_limit
        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = -1
            if declared_bytes < 0:
                response = JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)
                await response(scope, receive, send)
                return
            if declared_bytes > limit:
                response = JSONResponse({"detail": too_large_detail}, status_code=413)
                await response(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    # FastAPI re-raises HTTPExceptions raised while reading the body.
                    raise HTTPException(status_code=413, detail=too_large_detail)
            return message

        await self.app(scope, limited_receive, send)


app = FastAPI(
    title="SurvStudio",
    description="Local survival analysis dashboard for exploratory and validation-oriented cohort work.",
)
app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Added last so it wraps CORS as the outermost user middleware.
app.add_middleware(LocalRequestGuardMiddleware)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
_DATASET_PROFILE_CACHE_KEY = "_dataset_profile_cache"
# The cached column profile is large for wide tables and only ever read (then copied) by this
# module, so the store shares it instead of deep-copying it on every dataset lookup.
store = DatasetStore(shared_metadata_keys=(_DATASET_PROFILE_CACHE_KEY,))
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
_MAX_UPLOAD_ROWS = 100_000
_MAX_UPLOAD_COLUMNS = 5_000
_MAX_UPLOAD_CELLS = 5_000_000
# Decompression guards: an .xlsx is a zip of XML parts and Parquet pages are compressed, so a
# small upload can expand to gigabytes. Check declared sizes before handing the file to a parser.
# openpyxl keeps the shared-strings table of a workbook in memory, so the workbook cap is lower
# than the Parquet one; larger sheets should be exported as CSV.
_MAX_XLSX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_PARQUET_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
# A workbook shared string or a Parquet dictionary value is stored once but decodes once per
# cell that uses it, so a small file can still decode to far more text than its declared size.
# A parsed table may hold at most the text a 200 MB CSV upload could carry.
_MAX_UPLOAD_TEXT_CHARS = 200 * 1024 * 1024
# Upper bound on rows x model columns (after categorical coding) of one model design matrix.
_MAX_DESIGN_CELLS = 25_000_000
# Lower bound on the indicator combinations a signature search iterates (one indicator per
# candidate); searches past it would run for hours.
_MAX_SIGNATURE_COMBINATIONS = 1_000_000
# Most columns an exported table may have (the same cap as the explicit ``columns`` list).
_MAX_EXPORT_COLUMNS = 500
# Most groups of a grouped cohort table (the Kaplan-Meier group limit).
_MAX_COHORT_TABLE_GROUPS = 50
# Most levels a categorical variable may list in a cohort table.
_MAX_COHORT_TABLE_LEVELS = 200
# Widest hidden layer a deep-learning request may ask for.
_MAX_HIDDEN_LAYER_WIDTH = 1024
_SHAP_SAFE_MODE_MAX_ENCODED_FEATURES = 80
_SHAP_SAFE_MODE_MAX_RAW_FEATURES = 30
_SIGNED_NUMERIC_CSV_LITERAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
# Characters that may follow a leading number in a "number-like" cell such as "-0.42 ± 1.00",
# "-1.2 (-2.0 to -0.4)", "-12%", "-0.3–0.5" or a cohort summary "-0.42 ± 1.00 | -0.42 [-1.09, 0.28]".
# Only digits and punctuation are allowed, so such a cell cannot spell a function call or DDE target.
_NUMBER_LIKE_CELL_CHARS = re.compile(r"[0-9.\s±%()\[\],;:/|\u2013\u2212+\-]*")
_NUMBER_LIKE_EXPONENT = re.compile(r"(?<=[0-9.])[eE][+\-]?(?=[0-9])")
_NUMBER_LIKE_TO_WORD = re.compile(r"(?<=[\s0-9])to(?=[\s+\-\u22120-9])")
_CSV_FORMULA_TRIGGER_CHARS = ("=", "@", "\t", "\r")
_CSV_SIGN_CHARS = ("+", "-")
# Characters that are illegal in XML 1.0 (DOCX/XLSX) plus DEL; replaced with a space in exports.
_EXPORT_ILLEGAL_CHAR_PATTERN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")
_CSV_UTF8_BOM = "\ufeff"
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_NOTE_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A run of control characters (for example an Excel Alt+Enter line break) with the spaces around it.
_HEADER_CONTROL_RUN_PATTERN = re.compile(r"\s*[\x00-\x1f\x7f]+\s*")
_LATEX_ESCAPE_TABLE = str.maketrans(
    {
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
        # Under the default OT1 font encoding these print as other glyphs ("<" becomes "¡").
        "<": r"\textless{}",
        ">": r"\textgreater{}",
        "|": r"\textbar{}",
        '"': "''",
        # Common statistics symbols mapped to macros that compile without extra packages.
        "±": r"\ensuremath{\pm}",
        "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}",
        "≠": r"\ensuremath{\neq}",
        "≈": r"\ensuremath{\approx}",
        "×": r"\ensuremath{\times}",
        "−": r"\ensuremath{-}",
        "–": "--",
        "—": "---",
        "…": r"\ldots{}",
        "·": r"\ensuremath{\cdot}",
        "°": r"\ensuremath{^{\circ}}",
        "²": r"\ensuremath{^{2}}",
        "³": r"\ensuremath{^{3}}",
        "χ": r"\ensuremath{\chi}",
        "α": r"\ensuremath{\alpha}",
        "β": r"\ensuremath{\beta}",
        "μ": r"\ensuremath{\mu}",
        "µ": r"\ensuremath{\mu}",
        "‘": "`",
        "’": "'",
        "“": "``",
        "”": "''",
    }
)


# A lone UTF-16 surrogate (a valid JSON escape such as "\ud800") cannot be written as UTF-8.
_SURROGATE_PATTERN = re.compile("[\ud800-\udfff]")
# Largest magnitude up to which every integer has an exact float value (and so can match a data value).
_MAX_EXACT_FLOAT_INT = 2**53


def _json_safe_text(text: str) -> str:
    """``text`` with unpaired UTF-16 surrogates written as ``\\udxxx`` escapes, which a UTF-8 response can carry."""

    return text.encode("utf-8", "backslashreplace").decode("utf-8") if _SURROGATE_PATTERN.search(text) else text


def _normalize_optional_text_field(
    value: Any,
    *,
    field_name: str,
    allow_empty_as_none: bool = False,
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        if allow_empty_as_none:
            return None
        raise ValueError(f"{field_name} must not be empty.")
    if _CONTROL_CHAR_PATTERN.search(text):
        raise ValueError(f"{field_name} must not contain control characters.")
    if _SURROGATE_PATTERN.search(text):
        raise ValueError(f"{field_name} must be valid Unicode text (it holds an unpaired surrogate).")
    return text


def _normalize_column_name(value: Any, *, field_name: str, optional: bool = False) -> str | None:
    """A column name from a request: text, trimmed, without control characters or unpaired surrogates.

    An optional field reads null or blank text as "no column"; a required one refuses them.
    """

    if value is None or (optional and isinstance(value, str) and not value.strip()):
        if optional:
            return None
        raise ValueError(f"{field_name} must name a column.")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a column name (text).")
    return _normalize_optional_text_field(value, field_name=field_name)


def _normalize_event_positive_value(value: Any) -> Any:
    if value is None or value == "":
        return value
    if isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            text = _normalize_optional_text_field(value, field_name="event_positive_value")
            return text
        if not isinstance(value, bool) and abs(value) > _MAX_EXACT_FLOAT_INT:
            # A numeric column holds floats, so a larger code could never match its values exactly.
            raise ValueError("event_positive_value must be a number of at most 2^53 in magnitude.")
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("event_positive_value must be a finite scalar value.")
        return value
    raise ValueError("event_positive_value must be a scalar JSON value (string, number, boolean, or null).")


def _normalize_unique_name_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of column names.")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in value:
        if not isinstance(raw_value, str):
            raise ValueError(f"{field_name} must list column names as text.")
        text = _normalize_column_name(raw_value, field_name=field_name)
        if text is None or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _validate_subset_names(
    subset: Sequence[str],
    superset: Sequence[str],
    *,
    subset_name: str,
    superset_name: str,
) -> None:
    allowed = {str(value) for value in superset}
    extras = [str(value) for value in subset if str(value) not in allowed]
    if extras:
        raise ValueError(
            f"{subset_name} must be a subset of {superset_name}; unexpected value(s): {', '.join(extras)}."
        )


# Request fields that name one dataset column; every model normalises them the same way.
_COLUMN_NAME_FIELDS = (
    "time_column",
    "event_column",
    "group_column",
    "source_column",
    "variable",
    "target_feature",
    "marker_matrix_id_column",
)


class _DatasetRequestModel(BaseModel):
    # NaN/Infinity (accepted by Python's JSON parser) are never valid analysis settings.
    model_config = ConfigDict(allow_inf_nan=False)

    @field_validator("dataset_id", mode="before", check_fields=False)
    @classmethod
    def validate_dataset_id(cls, value: Any) -> str:
        text = _normalize_optional_text_field(value, field_name="dataset_id")
        if text is None:
            raise ValueError("dataset_id must not be empty or null.")
        if len(text) > 64:
            raise ValueError("dataset_id must be 64 characters or fewer.")
        return text

    @field_validator(*_COLUMN_NAME_FIELDS, mode="before", check_fields=False)
    @classmethod
    def validate_column_name(cls, value: Any, info: Any) -> str | None:
        # A field with a default (a grouping column, an optional outcome) reads blank as no column.
        field = cls.model_fields.get(info.field_name)
        optional = field is not None and not field.is_required()
        return _normalize_column_name(value, field_name=info.field_name, optional=optional)


class _EventPositiveValueRequestModel(_DatasetRequestModel):
    @field_validator("event_positive_value", mode="before", check_fields=False)
    @classmethod
    def validate_event_positive_value(cls, value: Any) -> Any:
        return _normalize_event_positive_value(value)


class _FeatureSelectionRequestModel(_EventPositiveValueRequestModel):
    @field_validator("features", "categorical_features", mode="before", check_fields=False)
    @classmethod
    def validate_feature_name_lists(cls, value: Any, info: Any) -> list[str]:
        return _normalize_unique_name_list(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_categorical_feature_subset(self) -> "_FeatureSelectionRequestModel":
        features = getattr(self, "features", None)
        categorical_features = getattr(self, "categorical_features", None)
        if features is not None and categorical_features is not None:
            _validate_subset_names(
                categorical_features,
                features,
                subset_name="categorical_features",
                superset_name="features",
            )
        return self


def _estimate_model_bytes(model: Any) -> int:
    """Approximate memory held by a fitted tree ensemble (RSF/GBS), 0 for other objects.

    Random survival forest trees store a survival curve per node, so a large forest on a
    large cohort can hold gigabytes; the node arrays dominate the footprint.
    """
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return 0
    total = 0
    for estimator in np.ravel(np.asarray(estimators, dtype=object)):
        tree = getattr(estimator, "tree_", None)
        if tree is None:
            continue
        for attribute in ("value", "threshold", "feature", "children_left", "children_right", "impurity", "n_node_samples"):
            array = getattr(tree, attribute, None)
            if isinstance(array, np.ndarray):
                total += int(array.nbytes)
    return total


def _estimate_artifact_bytes(result: dict[str, Any]) -> int:
    total = _estimate_model_bytes(result.get("_model"))
    for key in ("_X_encoded", "_analysis_frame"):
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            total += int(frame.memory_usage(deep=True).sum())
    return total


class _MlArtifactCache:
    """Fitted single-model artifacts reused by counterfactual and partial-dependence requests.

    Bounded by entry count and by estimated memory; an artifact larger than the whole
    budget is not cached (those requests then refit the model).
    """

    def __init__(self, *, max_items: int, max_bytes: int = 1024 * 1024 * 1024) -> None:
        self._max_items = int(max_items)
        self._max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._sizes: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _copy_frame(frame: Any) -> Any:
        if isinstance(frame, pd.DataFrame):
            return frame.copy(deep=True)
        return frame

    @staticmethod
    def _copy_model(model: Any) -> Any:
        if isinstance(model, (dict, list, tuple, set)):
            return copy.deepcopy(model)
        return model

    @classmethod
    def _copy_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "_model": cls._copy_model(result.get("_model")),
            "_X_encoded": cls._copy_frame(result.get("_X_encoded")),
            "_feature_encoder": copy.deepcopy(result.get("_feature_encoder")),
            "_analysis_frame": cls._copy_frame(result.get("_analysis_frame")),
        }

    def remember(
        self,
        *,
        dataset_id: str,
        model_type: str,
        signature: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        artifact_bytes = _estimate_artifact_bytes(result)
        cache_key = (dataset_id, model_type)
        if artifact_bytes > self._max_bytes:
            with self._lock:
                # Never serve an older artifact for a key whose latest fit was not cached.
                self._items.pop(cache_key, None)
                self._sizes.pop(cache_key, None)
            return
        artifact = {
            "signature": copy.deepcopy(signature),
            "result": self._copy_result(result),
        }
        with self._lock:
            self._items[cache_key] = artifact
            self._sizes[cache_key] = artifact_bytes
            self._items.move_to_end(cache_key)
            while len(self._items) > self._max_items or sum(self._sizes.values()) > self._max_bytes:
                evicted_key, _ = self._items.popitem(last=False)
                self._sizes.pop(evicted_key, None)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return int(sum(self._sizes.values()))

    def get(
        self,
        *,
        dataset_id: str,
        model_type: str,
        signature: dict[str, Any],
    ) -> dict[str, Any] | None:
        with self._lock:
            cache_key = (dataset_id, model_type)
            cached = self._items.get(cache_key)
            if cached is not None:
                self._items.move_to_end(cache_key)
            if not cached or cached.get("signature") != signature:
                return None
            result = cached.get("result")
            if not isinstance(result, dict):
                return None
            return self._copy_result(result)

    def purge_dataset(self, dataset_id: str) -> None:
        """Drop every cached model/frames for a dataset that left the store."""

        with self._lock:
            for cache_key in [key for key in self._items if key[0] == dataset_id]:
                del self._items[cache_key]
                self._sizes.pop(cache_key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


_ML_ARTIFACT_CACHE_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB
_ml_artifact_cache = _MlArtifactCache(max_items=8, max_bytes=_ML_ARTIFACT_CACHE_MAX_BYTES)
# Fitted models hold deep-copied training frames; release them together with their dataset.
store.add_eviction_listener(_ml_artifact_cache.purge_dataset)
# Marker matrices are matched to a dataset by patient ID at each use, so they outlive derived-column snapshots.
marker_matrices = MarkerMatrixStore()

_T = TypeVar("_T")


MAX_HEAVY_JOBS_ENV_VAR = "SURVSTUDIO_MAX_HEAVY_JOBS"


def _max_heavy_jobs() -> int:
    try:
        configured = int(os.environ.get(MAX_HEAVY_JOBS_ENV_VAR, "") or 0)
    except ValueError:
        configured = 0
    return configured if configured > 0 else 2


# Model training, signature search, cutpoint permutation, marker and XAI jobs each use several
# cores (and deep-learning comparisons can start worker processes), so only a few run at once.
# They run on their own small executor: a heavy request that waits for its turn sits in this
# executor's queue and holds no worker of the shared threadpool, which uploads, dataset views,
# exports and light analyses keep using.
_HEAVY_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=_max_heavy_jobs(), thread_name_prefix="survstudio-heavy-job")
_DISCONNECT_POLL_SECONDS = 0.5


async def _watch_for_disconnect(request: Request, cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            return
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


def _json_ready_key(key: Any) -> Any:
    return key.item() if isinstance(key, np.generic) else key


def _json_ready(value: Any) -> Any:
    """``value`` as plain JSON data: NumPy scalars and arrays become Python numbers and lists,
    and NaN, +/-infinity and pandas missing markers become null.

    The response encoder refuses non-finite floats and NumPy integer or boolean scalars, so a
    result that carried one would otherwise fail as a bare server error after the analysis ran.
    """

    if isinstance(value, dict):
        return {_json_ready_key(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):  # no exact Python type (for example long double)
            item = float(item) if isinstance(item, np.floating) else str(item)
        return _json_ready(item)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


async def _run_job(
    job: Callable[[], _T],
    *,
    request: Request | None = None,
    heavy: bool = False,
) -> _T:
    """Run a blocking job off the event loop and return its result as plain JSON data.

    Light jobs use the shared threadpool. Heavy jobs queue on the heavy-job executor, so at
    most a few run at once and waiting ones hold no shared worker. When ``request`` is given
    and its client disconnects (for example the page cancelled the request), the job stops
    at its next cancellation checkpoint or, if it is still queued, never starts.
    """

    cancel_event = threading.Event()

    def _guarded() -> _T:
        with cancellation_scope(cancel_event):
            raise_if_cancelled()
            return _json_ready(job())

    watcher = asyncio.create_task(_watch_for_disconnect(request, cancel_event)) if request is not None else None
    try:
        if not heavy:
            return await run_in_threadpool(_guarded)
        future = _HEAVY_JOB_EXECUTOR.submit(contextvars.copy_context().run, _guarded)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # The request task itself was cancelled (for example at shutdown): stop the job too.
            cancel_event.set()
            raise
    finally:
        if watcher is not None:
            watcher.cancel()


async def _run_dataset_job(
    dataset_id: str,
    job: Callable[[], _T],
    *,
    request: Request | None = None,
    heavy: bool = False,
) -> _T:
    """Run an analysis job (see :func:`_run_job`) while leasing its dataset.

    The lease keeps the dataset from expiring (idle TTL) or being LRU-evicted while the job
    waits or runs, and marks it as just used when the job finishes.
    """

    with store.lease(dataset_id):
        return await _run_job(job, request=request, heavy=heavy)


# ── Request models ──────────────────────────────────────────────


class DeriveGroupRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    source_column: str
    method: Literal[
        "median_split",
        "tertile_split",
        "quartile_split",
        "percentile_split",
        "extreme_split",
        "optimal_cutpoint",
    ]
    new_column_name: str | None = Field(default=None, max_length=200)
    cutoff: str | float | None = None
    lower_label: str = Field(default="Low", max_length=100)
    upper_label: str = Field(default="High", max_length=100)
    time_column: str | None = None
    event_column: str | None = None
    # Same default as the KM/Cox/ML endpoints, so an outcome-informed cutpoint uses the same
    # event coding as the analyses it feeds (the UI always sends the selected value explicitly).
    event_positive_value: Any = 1
    min_group_fraction: float = Field(default=0.1, gt=0.02, lt=0.45)
    permutation_iterations: int = Field(default=500, ge=0, le=500)
    random_seed: int = Field(default=20260311, ge=0, le=2**32 - 1)

    @field_validator("new_column_name", mode="before")
    @classmethod
    def validate_new_column_name(cls, value: Any) -> str | None:
        return _normalize_optional_text_field(
            value,
            field_name="Derived column names",
            allow_empty_as_none=True,
        )

    @field_validator("cutoff", mode="before")
    @classmethod
    def validate_cutoff(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if len(text) > 200:
                raise ValueError("cutoff must be 200 characters or fewer.")
            return text or None
        return value

    @model_validator(mode="after")
    def validate_cutoff_method(self) -> "DeriveGroupRequest":
        # Only percentile and extreme splits read a cutoff; any other method would silently ignore it.
        if self.cutoff is not None and self.method not in {"percentile_split", "extreme_split"}:
            raise ValueError(
                f"cutoff applies only to percentile_split and extreme_split; leave it empty for {self.method}."
            )
        return self

    @field_validator("lower_label", "upper_label", mode="before")
    @classmethod
    def validate_group_label(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Group labels must be text.")
        text = _normalize_optional_text_field(value, field_name="Group labels")
        if text is None:  # pragma: no cover - text values are never None
            raise ValueError("Group labels must not be empty.")
        return text


class KaplanMeierRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    group_column: str | None = None
    event_positive_value: Any = 1
    time_unit_label: str = Field(default="Months", max_length=40)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=0.999)
    max_time: float | None = Field(default=None, gt=0)
    risk_table_points: int = Field(default=6, ge=4, le=12)
    # UI display flag; confidence intervals are always computed server-side.
    show_confidence_bands: bool = True
    logrank_weight: Literal["logrank", "gehan_breslow", "tarone_ware", "fleming_harrington"] = "logrank"
    fh_p: float = Field(default=1.0, ge=0.0, le=5.0)

    @field_validator("time_unit_label", mode="before")
    @classmethod
    def validate_time_unit_label(cls, value: Any) -> str:
        text = _normalize_optional_text_field(value, field_name="time_unit_label")
        if text is None:
            raise ValueError("time_unit_label must not be empty or null.")
        return text


class CoxRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    covariates: list[str] = Field(max_length=200)
    categorical_covariates: list[str] = Field(default_factory=list, max_length=200)
    strata_columns: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("covariates", "categorical_covariates", "strata_columns", mode="before")
    @classmethod
    def validate_name_lists(cls, value: Any, info: Any) -> list[str]:
        return _normalize_unique_name_list(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_categorical_subset(self) -> "CoxRequest":
        _validate_subset_names(
            self.categorical_covariates,
            self.covariates,
            subset_name="categorical_covariates",
            superset_name="covariates",
        )
        return self


class CohortTableRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    variables: list[str] = Field(max_length=200)
    group_column: str | None = None
    # Optional survival outcome: when both columns are given, the table summarizes the analysis
    # cohort (rows with valid time/event and non-negative time) instead of the whole upload.
    time_column: str | None = None
    event_column: str | None = None
    event_positive_value: Any = 1

    @field_validator("variables", mode="before")
    @classmethod
    def validate_variables(cls, value: Any) -> list[str]:
        return _normalize_unique_name_list(value, field_name="variables")

    @model_validator(mode="after")
    def validate_outcome_pair(self) -> "CohortTableRequest":
        if (self.time_column is None) != (self.event_column is None):
            raise ValueError("time_column and event_column must be provided together to restrict the cohort table.")
        return self


class SignatureSearchRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    candidate_columns: list[str] = Field(max_length=_MAX_UPLOAD_COLUMNS)
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
    random_seed: int = Field(default=20260311, ge=0, le=2**32 - 1)
    new_column_name: str | None = Field(default=None, max_length=200)

    @field_validator("candidate_columns", mode="before")
    @classmethod
    def validate_candidate_columns(cls, value: Any) -> list[str]:
        return _normalize_unique_name_list(value, field_name="candidate_columns")

    @field_validator("new_column_name", mode="before")
    @classmethod
    def validate_new_column_name(cls, value: Any) -> str | None:
        return _normalize_optional_text_field(
            value,
            field_name="Signature-derived column names",
            allow_empty_as_none=True,
        )

    def minimum_combinations(self) -> int:
        """Indicator combinations the search iterates at the least (one indicator per candidate)."""
        n_candidates = len(dict.fromkeys(self.candidate_columns))
        return sum(math.comb(n_candidates, size) for size in range(1, int(self.max_combination_size) + 1))


class MLModelRequest(_FeatureSelectionRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    model_type: Literal["rsf", "gbs", "lasso_cox", "compare"]
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    compute_shap: bool = False
    shap_safe_mode: bool = True
    evaluation_strategy: Literal["holdout", "repeated_cv"] = "holdout"
    cv_folds: int = Field(default=5, ge=2, le=10)
    cv_repeats: int = Field(default=3, ge=1, le=20)
    # Fraction of the cohort reserved as an untouched test set for repeated-CV
    # comparisons (None disables it).
    locked_test_fraction: float | None = Field(default=None, ge=0.05, le=0.5)


class DeepModelRequest(_FeatureSelectionRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    model_type: Literal["deepsurv", "deephit", "mtlr", "transformer", "vae", "compare"]
    hidden_layers: list[int] = Field(default=[64, 64], max_length=20)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)
    epochs: int = Field(default=100, ge=10, le=1000)
    batch_size: int = Field(default=64, ge=8, le=512)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    evaluation_strategy: Literal["holdout", "repeated_cv"] = "holdout"
    cv_folds: int = Field(default=5, ge=2, le=10)
    cv_repeats: int = Field(default=3, ge=1, le=20)
    locked_test_fraction: float | None = Field(default=None, ge=0.05, le=0.5)
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

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("Hidden layers must contain at least one positive integer.")
        if any((not isinstance(layer, int)) or layer <= 0 for layer in value):
            raise ValueError("Hidden layers must contain positive integers only.")
        if any(layer > _MAX_HIDDEN_LAYER_WIDTH for layer in value):
            raise ValueError(f"Each hidden layer can have at most {_MAX_HIDDEN_LAYER_WIDTH} units.")
        return value

    @model_validator(mode="after")
    def validate_transformer_width(self) -> "DeepModelRequest":
        if self.model_type in {"transformer", "compare"} and self.d_model % self.n_heads != 0:
            raise ValueError("Transformer width must be divisible by attention heads.")
        return self


class OptimalCutpointRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    variable: str
    min_group_fraction: float = Field(default=0.1, gt=0.02, lt=0.45)
    permutation_iterations: int = Field(default=500, ge=0, le=500)


class MarkerEvaluationRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    marker_columns: list[str] = Field(default_factory=list, max_length=_MAX_UPLOAD_COLUMNS)
    # An attached marker matrix (POST /api/marker-matrix) replaces marker_columns; its patients are
    # matched to this dataset through marker_matrix_id_column.
    marker_matrix_id: str | None = Field(default=None, max_length=64)
    marker_matrix_id_column: str | None = Field(default=None, max_length=512)
    clinical_columns: list[str] = Field(default_factory=list, max_length=200)
    categorical_clinical: list[str] = Field(default_factory=list, max_length=200)
    strata_columns: list[str] = Field(default_factory=list, max_length=20)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    fdr_level: float = Field(default=0.10, gt=0.0, lt=0.5)
    n_permutations: int = Field(default=1000, ge=0, le=10_000)
    n_resamples: int = Field(default=200, ge=0, le=1_000)
    resample_fraction: float = Field(default=0.632, ge=0.3, le=0.9)
    max_missing_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    max_mode_fraction: float = Field(default=0.9, ge=0.5, le=1.0)
    max_signature_markers: int = Field(default=10, ge=1, le=50)
    nonlinear_lens: Literal["off", "gbs", "rsf"] = "off"
    random_seed: int = Field(default=20260926, ge=0, le=2**32 - 1)

    @field_validator("marker_columns", "clinical_columns", "categorical_clinical", "strata_columns", mode="before")
    @classmethod
    def validate_name_lists(cls, value: Any, info: Any) -> list[str]:
        return _normalize_unique_name_list(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_categorical_subset(self) -> "MarkerEvaluationRequest":
        _validate_subset_names(
            self.categorical_clinical,
            self.clinical_columns,
            subset_name="categorical_clinical",
            superset_name="clinical_columns",
        )
        if self.marker_matrix_id:
            if self.marker_columns:
                raise ValueError("Give marker_columns or a marker matrix, not both.")
            if not self.marker_matrix_id_column:
                raise ValueError("A marker matrix needs marker_matrix_id_column, the dataset's patient ID column.")
        elif not self.marker_columns:
            raise ValueError("Choose at least one marker.")
        return self

    def marker_settings(self) -> MarkerSettings:
        return MarkerSettings(
            alpha=self.alpha,
            fdr_level=self.fdr_level,
            n_permutations=self.n_permutations,
            n_resamples=self.n_resamples,
            resample_fraction=self.resample_fraction,
            max_missing_fraction=self.max_missing_fraction,
            max_mode_fraction=self.max_mode_fraction,
            max_signature_markers=self.max_signature_markers,
            nonlinear_lens=self.nonlinear_lens,
            random_seed=self.random_seed,
        )


class MarkerValidationRequest(_EventPositiveValueRequestModel):
    # The external cohort; the recipe comes from a marker evaluation of another dataset.
    dataset_id: str
    recipe: dict[str, Any]
    column_mapping: dict[str, str] = Field(default_factory=dict, max_length=_MAX_UPLOAD_COLUMNS)
    # None applies the recipe's own event coding.
    event_positive_value: Any = None
    horizon: float | None = Field(default=None, gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    n_bootstrap: int = Field(default=200, ge=0, le=2_000)
    random_seed: int = Field(default=20260926, ge=0, le=2**32 - 1)
    # "within_cohort" rescales each marker to its development distribution (a cohort from another platform).
    marker_scaling: Literal["as_measured", "within_cohort"] = "as_measured"

    @field_validator("column_mapping", mode="before")
    @classmethod
    def validate_column_mapping(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("column_mapping must map recipe column names to dataset column names.")
        return {
            _normalize_column_name(key, field_name="column_mapping"): _normalize_column_name(target, field_name="column_mapping")
            for key, target in value.items()
        }


_MAX_INTERVAL_MODELS_PER_BLOCK = 50


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    """A JSON number with a finite float value; an integer too large for a float is not one."""

    if not _is_number(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_prediction_block(block: dict[str, Any], index: int) -> None:
    """Shape and type checks of one ``test_predictions`` block, so a malformed block is a 422."""

    where = f"Prediction block {index}"
    row_ids = block.get("row_ids")
    if not isinstance(row_ids, list) or not row_ids:
        raise ValueError(f"{where} needs a non-empty row_ids list.")
    n_rows = len(row_ids)
    if n_rows > _MAX_UPLOAD_ROWS:
        raise ValueError(f"{where} has {n_rows:,} patients; at most {_MAX_UPLOAD_ROWS:,} are supported.")
    if not all(isinstance(row, (str, int)) and not isinstance(row, bool) for row in row_ids):
        raise ValueError(f"{where}: row_ids must be text or integer patient row labels.")
    if len({str(row) for row in row_ids}) != n_rows:
        raise ValueError(f"{where}: row_ids repeat a patient.")
    time_values = block.get("time")
    if not isinstance(time_values, list) or len(time_values) != n_rows:
        raise ValueError(f"{where}: time must be a list with one value per patient.")
    if not all(_is_finite_number(value) and value >= 0 for value in time_values):
        raise ValueError(f"{where}: every time must be a finite, non-negative number.")
    event_values = block.get("event")
    if not isinstance(event_values, list) or len(event_values) != n_rows:
        raise ValueError(f"{where}: event must be a list with one value per patient.")
    if not all(isinstance(value, (int, float)) and value in (0, 1) for value in event_values):
        raise ValueError(f"{where}: every event must be 0 (censored) or 1 (event).")
    risk = block.get("risk")
    if not isinstance(risk, dict) or not risk:
        raise ValueError(f"{where}: risk must map each model name to its risk scores.")
    if len(risk) > _MAX_INTERVAL_MODELS_PER_BLOCK:
        raise ValueError(f"{where} has {len(risk)} models; at most {_MAX_INTERVAL_MODELS_PER_BLOCK} are supported.")
    for name, values in risk.items():
        if not isinstance(values, list) or len(values) != n_rows:
            raise ValueError(f"{where}: the risk scores of '{name}' must be a list with one value per patient.")
        if not all(_is_finite_number(value) for value in values):
            raise ValueError(f"{where}: every risk score of '{name}' must be a finite number.")


class ModelComparisonIntervalsRequest(BaseModel):
    # The ``test_predictions`` (or ``locked_test_predictions``) blocks of ML and DL comparisons run on one split.
    model_config = ConfigDict(extra="forbid")

    predictions: list[dict[str, Any]] = Field(min_length=1, max_length=4)
    reference: str | None = Field(default="Cox PH", max_length=200)
    n_bootstrap: int = Field(default=1000, ge=100, le=5000)
    random_seed: int = Field(default=20260926, ge=0, le=2**32 - 1)

    @field_validator("predictions")
    @classmethod
    def validate_prediction_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, block in enumerate(blocks, start=1):
            _check_prediction_block(block, index)
        return blocks


# Work units (see _c_index_draw_work; about 3 ns each on one laptop core) of all bootstrap draws of
# one request, so it stays within about half a minute: 2,500 test patients scored by 10 models get
# 1,000 draws. It is a hard cap: a larger test set gets fewer draws (and a note), and one too large
# for the minimum number of draws is refused.
_INTERVAL_WORK_BUDGET = 12_000_000_000
# Fewest draws that still give a usable 95% percentile interval (the request minimum).
_INTERVAL_MIN_DRAWS = 100


def _c_index_draw_work(n_patients: int, n_events: int, n_models: int) -> int:
    """Work units of one bootstrap draw of Harrell's C for ``n_models`` risk columns.

    Mirrors ``marker_screen.harrell_c_many``: when events x patients is at most its pairwise
    limit it compares every event with every patient, and otherwise it counts pairs from sorted
    blocks, about (patients + 2 x events) x log2(patients)^2 operations per column. One column
    more stands for the per-draw work that does not depend on the number of models.
    """

    n = max(int(n_patients), 1)
    events = min(max(int(n_events), 1), n)
    columns = max(int(n_models), 1) + 1
    if events * n <= _C_INDEX_PAIRWISE_LIMIT:
        return columns * events * n
    log_n = math.log2(n)
    return int(columns * (n + 2 * events) * log_n * log_n)


class DesignAuditCohort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=120)
    n: int = Field(ge=2, le=10_000_000)
    events: int | None = Field(default=None, ge=1)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: Any) -> str:
        return _normalize_optional_text_field(value, field_name="cohort name", allow_empty_as_none=True) or ""


class DesignAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_models: int = Field(ge=2, le=1_000_000)
    gene_only: bool
    selection_cohorts: list[DesignAuditCohort] = Field(min_length=1, max_length=200)
    training_in_selection: bool = False
    sealed_cohorts: list[DesignAuditCohort] = Field(default_factory=list, max_length=200)
    headline: Literal["validation", "average including training", "training", "sealed"] = "validation"
    prefilter_used_validation_outcomes: bool = False
    refit_in_validation: bool = False
    cutoff_per_cohort: bool = False
    compared_with_clinical: bool = False


class ChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(max_length=8)
    section: str = Field(max_length=80)
    topic: str = Field(max_length=200)
    status: Literal["reported", "partly", "author"]
    text: str = Field(max_length=8000)


class ChecklistExportRequest(BaseModel):
    """A REMARK or TRIPOD+AI checklist, as returned by the marker evaluation or the TRIPOD+AI endpoint."""

    model_config = ConfigDict(extra="ignore")

    format: Literal["markdown", "docx"]
    guideline: Literal["REMARK", "TRIPOD+AI"]
    reference: str = Field(max_length=400)
    software: str = Field(max_length=80)
    methods: str = Field(default="", max_length=12000)
    results: str = Field(default="", max_length=4000)
    items: list[ChecklistItem] = Field(min_length=1, max_length=40)


class ComparisonForReport(BaseModel):
    family: Literal["ml", "dl"]
    analysis: dict[str, Any]
    request_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_report_fields(self) -> "ComparisonForReport":
        # The checklist reads these fields of a comparison result; anything else is ignored.
        table = self.analysis.get("comparison_table")
        if table is not None:
            if not isinstance(table, list) or not all(isinstance(row, dict) for row in table):
                raise ValueError("analysis.comparison_table must be a list of model rows (objects).")
            if len(table) > 500:
                raise ValueError("analysis.comparison_table can have at most 500 model rows.")
        for key in ("excluded_models", "errors"):
            value = self.analysis.get(key)
            if value is not None and not isinstance(value, list):
                raise ValueError(f"analysis.{key} must be a list.")
        fingerprint = self.analysis.get("evaluation_split_fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ValueError("analysis.evaluation_split_fingerprint must be text.")
        for key in ("features", "categorical_features", "hidden_layers"):
            value = self.request_config.get(key)
            if value is not None and not isinstance(value, list):
                raise ValueError(f"request_config.{key} must be a list.")
        return self


class TripodChecklistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The dataset the comparisons ran on, for the data-source item; optional because it may have expired.
    dataset_id: str | None = None
    comparisons: list[ComparisonForReport] = Field(min_length=1, max_length=2)


class TableExportRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=2000)
    columns: list[str] = Field(default_factory=list, max_length=500)
    format: Literal["csv", "markdown", "latex", "docx", "xlsx"]
    style: Literal["plain", "journal"] = "journal"
    template: Literal["default", "nejm", "lancet", "jco"] = "default"
    caption: str | None = Field(default=None, max_length=4000)
    notes: list[str] = Field(default_factory=list, max_length=50)
    provenance: dict[str, Any] | None = None

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for note in value:
            text = str(note)
            if len(text) > 4000:
                raise ValueError("Each table note must be 4000 characters or fewer.")
            if _NOTE_CONTROL_CHAR_PATTERN.search(text):
                raise ValueError("Table notes must not contain control characters.")
            cleaned.append(text)
        return cleaned

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: list[str]) -> list[str]:
        # Names keep their spelling, so they still match the row keys; the writers turn control
        # characters (a line break in a group value that became a column) into spaces.
        cleaned: list[str] = []
        for column in value:
            text = str(column)
            if len(text) > 4000:
                raise ValueError("Each export column name must be 4000 characters or fewer.")
            cleaned.append(text)
        return cleaned

    @field_validator("caption", mode="before")
    @classmethod
    def validate_caption(cls, value: Any) -> str | None:
        return _normalize_optional_text_field(
            value,
            field_name="caption",
            allow_empty_as_none=True,
        )

    @model_validator(mode="after")
    def validate_export_width(self) -> "TableExportRequest":
        # Every row is written for every column, and the columns are the union of the given ones
        # and all row keys, so the row keys count toward the same cap as ``columns``.
        names: set[str] = set()
        for group in (self.columns, *self.rows):
            for name in group:
                text = str(name)
                if not text or text.startswith("_"):
                    continue
                names.add(text)
                if len(names) > _MAX_EXPORT_COLUMNS:
                    raise ValueError(
                        f"An exported table can have at most {_MAX_EXPORT_COLUMNS} columns (row keys included)."
                    )
        return self


# ── Helpers ─────────────────────────────────────────────────────


def _profile_template_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(
        {
            key: value
            for key, value in payload.items()
            if key not in {"dataset_id", "filename"}
        }
    )


def _profile_template_column_names(template: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(template, dict):
        return None
    columns = template.get("columns")
    if not isinstance(columns, list):
        return None
    names: list[str] = []
    for column in columns:
        if not isinstance(column, dict) or "name" not in column:
            return None
        names.append(str(column["name"]))
    return names


def _profile_template_matches_dataframe(template: dict[str, Any] | None, dataframe: pd.DataFrame) -> bool:
    column_names = _profile_template_column_names(template)
    if column_names is None:
        return False
    if int(template.get("n_rows", -1)) != int(dataframe.shape[0]):
        return False
    if int(template.get("n_columns", -1)) != int(dataframe.shape[1]):
        return False
    return column_names == [str(column) for column in dataframe.columns]


def _payload_from_profile_template(
    template: dict[str, Any],
    *,
    dataset_id: str,
    filename: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(template)
    payload["dataset_id"] = dataset_id
    payload["filename"] = filename
    return payload


def _extend_profile_template_for_appended_columns(
    template: dict[str, Any] | None,
    dataframe: pd.DataFrame,
) -> dict[str, Any] | None:
    previous_column_names = _profile_template_column_names(template)
    if previous_column_names is None:
        return None
    current_column_names = [str(column) for column in dataframe.columns]
    if int(template.get("n_rows", -1)) != int(dataframe.shape[0]):
        return None
    if len(current_column_names) <= len(previous_column_names):
        return None
    if current_column_names[: len(previous_column_names)] != previous_column_names:
        return None

    next_template = copy.deepcopy(template)
    columns = list(next_template.get("columns", []))
    numeric_columns = [str(column) for column in next_template.get("numeric_columns", [])]
    categorical_columns = [str(column) for column in next_template.get("categorical_columns", [])]
    binary_candidate_columns = [str(column) for column in next_template.get("binary_candidate_columns", [])]

    for column in current_column_names[len(previous_column_names) :]:
        profile, is_numeric, is_binary = _profile_dataframe_column(column, dataframe[column])
        columns.append(profile)
        if is_numeric:
            if column not in numeric_columns:
                numeric_columns.append(column)
        elif column not in categorical_columns:
            categorical_columns.append(column)
        if is_binary and column not in binary_candidate_columns:
            binary_candidate_columns.append(column)
        if profile["kind"] == "binary" and column not in categorical_columns and column not in numeric_columns:
            categorical_columns.append(column)

    suggestions = suggest_columns(dataframe)
    model_feature_candidates = _model_feature_candidate_columns_from_metadata(
        current_column_names,
        suggested_time_columns=suggestions.get("time_columns", []),
        binary_candidate_columns=binary_candidate_columns,
    )
    next_template.update(
        {
            "n_rows": int(dataframe.shape[0]),
            "n_columns": int(dataframe.shape[1]),
            "columns": columns,
            "preview": preview_rows(dataframe),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "binary_candidate_columns": binary_candidate_columns,
            "model_feature_candidate_count": len(model_feature_candidates),
            "suggestions": suggestions,
        }
    )
    return next_template


def _resolved_dataset_profile_payload(stored: Any) -> dict[str, Any]:
    cached_template = stored.metadata.get(_DATASET_PROFILE_CACHE_KEY)
    if _profile_template_matches_dataframe(cached_template, stored.dataframe):
        return _payload_from_profile_template(
            cached_template,
            dataset_id=stored.dataset_id,
            filename=stored.filename,
        )

    payload = profile_dataframe(stored.dataframe, dataset_id=stored.dataset_id, filename=stored.filename)
    cached_metadata = {
        **stored.metadata,
        _DATASET_PROFILE_CACHE_KEY: _profile_template_from_payload(payload),
    }
    store.update_metadata(stored.dataset_id, cached_metadata)
    return payload


def dataset_response(dataset_id: str) -> dict[str, Any]:
    stored = store.get(dataset_id, copy_dataframe=False)
    payload = _resolved_dataset_profile_payload(stored)
    payload["dataset_source"] = stored.source
    payload["preset_eligible"] = stored.source == "builtin_demo"
    payload["preset_name"] = str(stored.metadata.get("preset_name")) if stored.metadata.get("preset_name") else None
    payload["derived_column_provenance"] = dict(stored.metadata.get("derived_column_provenance", {}))
    payload["dataset_hash"] = str(stored.metadata.get("dataset_hash") or "")
    return _json_ready(payload)


def _attach_dataset_hash(payload: dict[str, Any], stored: Any) -> dict[str, Any]:
    payload["dataset_hash"] = str(stored.metadata.get("dataset_hash") or "")
    return payload


def _create_dataset_snapshot(stored, dataframe, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot_metadata = dict(metadata if metadata is not None else stored.metadata)
    snapshot_metadata.pop(_DATASET_PROFILE_CACHE_KEY, None)
    profile_template = _extend_profile_template_for_appended_columns(
        stored.metadata.get(_DATASET_PROFILE_CACHE_KEY),
        dataframe,
    )
    if profile_template is not None:
        snapshot_metadata[_DATASET_PROFILE_CACHE_KEY] = profile_template
    snapshot = store.create(
        dataframe,
        filename=stored.filename,
        source=stored.source,
        metadata=snapshot_metadata,
        copy_dataframe=False,
    )
    return dataset_response(snapshot.dataset_id)


def _outcome_informed_columns(stored: Any) -> set[str]:
    provenance = stored.metadata.get("derived_column_provenance", {})
    if not isinstance(provenance, dict):
        return set()
    return {
        str(column)
        for column, meta in provenance.items()
        if isinstance(meta, dict) and bool(meta.get("outcome_informed"))
    }


def _get_stored_dataset(dataset_id: str):
    """Return the shared dataset object for read-only request handling.

    The underlying DataFrame is not copied here. Callers must snapshot before
    any mutation and treat the returned frame as immutable.
    """
    return store.get(dataset_id, copy_dataframe=False)


def _reject_outcome_informed_columns(
    stored: Any,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    forbidden = _outcome_informed_columns(stored)
    offenders = sorted({str(column) for column in columns if str(column) in forbidden})
    if offenders:
        raise UserInputError(
            f"Outcome-informed derived columns cannot be used for {context}: "
            + ", ".join(offenders)
            + ". Use them only for exploratory grouping/visualization."
        )


def _reject_survival_outcome_feature_columns(
    stored: Any,
    columns: Sequence[str],
    *,
    time_column: str,
    event_column: str,
    event_positive_value: Any = None,
    context: str,
) -> None:
    forbidden = {str(time_column), str(event_column)}
    forbidden.update(str(column) for column in _survival_outcome_like_columns(stored.dataframe))
    forbidden.update(
        find_event_equivalent_columns(
            stored.dataframe,
            event_column=str(event_column),
            event_positive_value=event_positive_value,
        )
    )
    offenders = sorted({str(column) for column in columns if str(column) in forbidden})
    if offenders:
        raise UserInputError(
            f"Survival outcome columns cannot be used for {context}: "
            + ", ".join(offenders)
            + ". Use baseline covariates or biomarker features instead."
        )


# Values of the matrix block examined at once by the 0/1 scan below (float32, so about 16 MB).
_MATRIX_BINARY_SCAN_CELLS = 4_000_000


def _matrix_outcome_markers(
    frame: pd.DataFrame,
    markers: Sequence[str],
    *,
    event_column: str,
    event_positive_value: Any,
) -> list[str]:
    """Markers of an attached matrix that are survival outcome columns.

    The rules are those applied to dataset columns by `_reject_survival_outcome_feature_columns`:
    outcome-like names, and columns that code the same events as the event column. Matrix
    values are numbers, so only a marker whose values are all 0 or 1 can code the events; the
    event comparison runs on those markers alone instead of on every marker of a wide matrix.
    """

    marker_names = [str(marker) for marker in markers]
    marker_set = set(marker_names)
    flagged = {str(column) for column in _survival_outcome_like_columns(frame)} & marker_set
    zero_one: list[str] = []
    step = max(1, _MATRIX_BINARY_SCAN_CELLS // max(int(frame.shape[0]), 1))
    for start in range(0, len(marker_names), step):
        raise_if_cancelled()
        names = marker_names[start : start + step]
        values = frame[names].to_numpy(dtype=np.float32, na_value=np.nan)
        missing = np.isnan(values)
        is_zero_one = np.all(missing | (values == 0.0) | (values == 1.0), axis=0) & ~np.all(missing, axis=0)
        zero_one.extend(name for name, keep in zip(names, is_zero_one) if keep)
    if zero_one:
        flagged |= {
            str(column)
            for column in find_event_equivalent_columns(
                frame[[event_column, *zero_one]],
                event_column=event_column,
                event_positive_value=event_positive_value,
            )
        } & marker_set
    return sorted(flagged)


def _encoded_width(frame: pd.DataFrame, features: Sequence[str], categorical_features: Sequence[str]) -> int:
    """Model columns the features become: one per numeric feature, about one per level of a categorical one."""

    categorical = {str(column) for column in categorical_features}
    width = 0
    for column in dict.fromkeys(str(feature) for feature in features):
        if column not in frame.columns:
            continue  # reported by the analysis itself
        series = frame[column]
        if column in categorical or not pd.api.types.is_numeric_dtype(series):
            width += max(int(series.nunique(dropna=True)), 1)
        else:
            width += 1
    return width


def _reject_oversized_design(
    stored: Any,
    features: Sequence[str],
    categorical_features: Sequence[str] = (),
    *,
    context: str,
) -> None:
    """Refuse a model whose design matrix (rows x columns after categorical coding) is too large to fit.

    A categorical feature with thousands of levels (an ID or a free-text column) becomes
    thousands of indicator columns, which no survival model can fit and which can exhaust memory.
    """

    n_rows = int(stored.dataframe.shape[0])
    width = _encoded_width(stored.dataframe, features, categorical_features)
    if n_rows * width > _MAX_DESIGN_CELLS:
        raise UserInputError(
            f"The {context} expand to about {width:,} model columns after categorical coding, "
            f"{n_rows * width:,} values for {n_rows:,} rows; SurvStudio fits at most {_MAX_DESIGN_CELLS:,}. "
            "Leave out high-cardinality categorical columns (for example IDs or free text) or recode them into a few groups."
        )


def _ml_artifact_signature(request_config: dict[str, Any]) -> dict[str, Any]:
    model_type = str(request_config.get("model_type", "rsf"))
    signature = {
        "model_type": model_type,
        "time_column": request_config.get("time_column"),
        "event_column": request_config.get("event_column"),
        "event_positive_value": request_config.get("event_positive_value"),
        "features": [str(value) for value in request_config.get("features") or []],
        "categorical_features": [str(value) for value in request_config.get("categorical_features") or []],
        "n_estimators": request_config.get("n_estimators"),
        "max_depth": request_config.get("max_depth"),
        "random_state": request_config.get("random_state"),
    }
    if model_type == "gbs":
        signature["learning_rate"] = request_config.get("learning_rate")
    return signature


def _remember_ml_artifact(dataset_id: str, request_config: dict[str, Any], result: dict[str, Any]) -> None:
    model_type = str(request_config.get("model_type", ""))
    if model_type not in {"rsf", "gbs"}:
        return
    model = result.get("_model")
    x_encoded = result.get("_X_encoded")
    if model is None or x_encoded is None:
        return
    artifact_result = {
        "_model": model,
        "_X_encoded": x_encoded,
        "_feature_encoder": result.get("_feature_encoder"),
        "_analysis_frame": result.get("_analysis_frame"),
    }
    _ml_artifact_cache.remember(
        dataset_id=dataset_id,
        model_type=model_type,
        signature=_ml_artifact_signature(request_config),
        result=artifact_result,
    )


def _get_ml_artifact(dataset_id: str, request_config: dict[str, Any]) -> dict[str, Any] | None:
    model_type = str(request_config.get("model_type", ""))
    if model_type not in {"rsf", "gbs"}:
        return None
    return _ml_artifact_cache.get(
        dataset_id=dataset_id,
        model_type=model_type,
        signature=_ml_artifact_signature(request_config),
    )


def _encoded_to_raw_feature_map(feature_encoder: dict[str, Any], requested_features: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    requested = {str(feature) for feature in requested_features}
    for column in feature_encoder.get("numeric_features", []):
        column_name = str(column)
        if column_name in requested:
            mapping[column_name] = column_name
    categorical_mappings = feature_encoder.get("categorical_mappings", {})
    for column in feature_encoder.get("categorical_features", []):
        column_name = str(column)
        if column_name not in requested:
            continue
        meta = categorical_mappings.get(column_name, {})
        level_columns = meta.get("level_columns") or {}
        for level in meta.get("retained_levels", []):
            mapping[str(level_columns.get(level, f"{column_name}_{level}"))] = column_name
        unknown_column = meta.get("unknown_column")
        missing_column = meta.get("missing_column")
        if unknown_column:
            mapping[str(unknown_column)] = column_name
        if missing_column:
            mapping[str(missing_column)] = column_name
    return mapping


def _raw_feature_encoded_widths(feature_encoder: dict[str, Any], requested_features: Sequence[str]) -> dict[str, int]:
    widths: dict[str, int] = {}
    requested = {str(feature) for feature in requested_features}
    for column in feature_encoder.get("numeric_features", []):
        column_name = str(column)
        if column_name in requested:
            widths[column_name] = 1
    categorical_mappings = feature_encoder.get("categorical_mappings", {})
    for column in feature_encoder.get("categorical_features", []):
        column_name = str(column)
        if column_name not in requested:
            continue
        meta = categorical_mappings.get(column_name, {})
        widths[column_name] = (
            len(meta.get("retained_levels", []))
            + int(bool(meta.get("unknown_column")))
            + int(bool(meta.get("missing_column")))
        )
    return widths


def _must_propagate_through_boundary(exc: BaseException) -> bool:
    """``must_propagate``, also for a coding error that ``user_input_boundary`` wrapped.

    The public analysis functions turn a TypeError raised by SurvStudio code into an
    InternalAnalysisError; that is still a coding error, not a failure to report as a result.
    """

    if must_propagate(exc):
        return True
    cause = exc.__cause__
    return isinstance(exc, InternalAnalysisError) and cause is not None and must_propagate(cause)


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(score) else score


def _select_shap_safe_mode_subset(
    result: dict[str, Any],
    requested_features: Sequence[str],
) -> dict[str, Any] | None:
    feature_encoder = result.get("_feature_encoder")
    if not isinstance(feature_encoder, dict):
        return None
    requested = [str(feature) for feature in requested_features]
    if len(requested) <= 1:
        return None

    encoded_to_raw = _encoded_to_raw_feature_map(feature_encoder, requested)
    feature_widths = _raw_feature_encoded_widths(feature_encoder, requested)
    raw_scores = {feature: 0.0 for feature in requested}
    for row in result.get("feature_importance", []) or []:
        if not isinstance(row, dict):
            continue
        encoded_feature = str(row.get("feature") or "")
        raw_feature = encoded_to_raw.get(encoded_feature, encoded_feature if encoded_feature in raw_scores else None)
        if raw_feature is None:
            continue
        score = _coerce_score(row.get("importance"))
        if score is None:
            continue
        raw_scores[raw_feature] += score

    ranked_features = sorted(
        requested,
        key=lambda feature: (-raw_scores.get(feature, 0.0), requested.index(feature)),
    )
    selected_ranked: list[str] = []
    encoded_total = 0
    for feature in ranked_features:
        encoded_width = max(1, int(feature_widths.get(feature, 1)))
        if encoded_width > _SHAP_SAFE_MODE_MAX_ENCODED_FEATURES:
            continue
        if selected_ranked and encoded_total + encoded_width > _SHAP_SAFE_MODE_MAX_ENCODED_FEATURES:
            continue
        selected_ranked.append(feature)
        encoded_total += encoded_width
        if len(selected_ranked) >= min(_SHAP_SAFE_MODE_MAX_RAW_FEATURES, len(requested)):
            break

    if not selected_ranked or len(selected_ranked) >= len(requested):
        return None

    selected_set = set(selected_ranked)
    ordered_selected = [feature for feature in requested if feature in selected_set]
    omitted_features = [feature for feature in requested if feature not in selected_set]
    return {
        "selected_features": ordered_selected,
        "omitted_features": omitted_features,
        "selected_feature_count_raw": len(ordered_selected),
        "selected_feature_count_encoded": encoded_total,
        "requested_feature_count_raw": len(requested),
    }


def _enforce_upload_shape_limits(dataframe: Any) -> None:
    if not hasattr(dataframe, "shape"):
        return
    n_rows = int(dataframe.shape[0])
    n_columns = int(dataframe.shape[1])
    n_cells = int(n_rows * n_columns)
    if n_rows < 1:
        raise UserInputError("Dataset has no rows. Upload a table with at least one patient row.")
    if n_rows > _MAX_UPLOAD_ROWS:
        raise UserInputError(
            f"Upload has {n_rows:,} rows. SurvStudio currently supports at most {_MAX_UPLOAD_ROWS:,} rows per uploaded cohort."
        )
    if n_columns > _MAX_UPLOAD_COLUMNS:
        raise UserInputError(
            f"Upload has {n_columns:,} columns. SurvStudio currently supports at most {_MAX_UPLOAD_COLUMNS:,} columns per uploaded cohort."
        )
    if n_cells > _MAX_UPLOAD_CELLS:
        raise UserInputError(
            f"Upload expands to {n_cells:,} cells after parsing. SurvStudio currently supports at most {_MAX_UPLOAD_CELLS:,} parsed cells per uploaded cohort."
        )


def _format_megabytes(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):,.0f} MB"


def _text_too_large(text_size: int) -> HTTPException:
    logger.info("Rejected an upload holding at least %d characters of decoded text", text_size)
    return HTTPException(
        status_code=413,
        detail=(
            f"The uploaded table holds more than {_format_megabytes(_MAX_UPLOAD_TEXT_CHARS)} of text once decoded. "
            f"SurvStudio accepts at most {_format_megabytes(_MAX_UPLOAD_TEXT_CHARS)} of text per table, the text of a "
            "200 MB CSV file; remove long free-text columns before uploading."
        ),
    )


def _parquet_text_bytes(path: Path, limit: int) -> int:
    """Decoded size of the text and binary columns of a Parquet file, counted until it passes ``limit``.

    Every column that does not hold numbers, dates, times or booleans decodes to a text or bytes
    value per cell, whatever its Arrow type: plain, large or view strings and binaries, fixed-size
    binaries, or dictionaries of them. Text columns are read with their dictionaries kept, so a
    value stored once and used by many rows is measured, not materialised, once per row; a
    fixed-size binary column is measured from its width, without reading it. Other text-holding
    types (extension types such as JSON) cannot be read that way and are refused.
    """

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    def _is_extension(data_type: Any) -> bool:
        while pa.types.is_dictionary(data_type):
            data_type = data_type.value_type
        return isinstance(data_type, pa.BaseExtensionType)

    def _value_type(data_type: Any) -> Any:
        # A dictionary column holds its value type; an extension type (UUID, ...) its storage type.
        while True:
            if pa.types.is_dictionary(data_type):
                data_type = data_type.value_type
            elif isinstance(data_type, pa.BaseExtensionType):
                data_type = data_type.storage_type
            else:
                return data_type

    def _holds_text(data_type: Any) -> bool:
        value_type = _value_type(data_type)
        return not any(
            check(value_type)
            for check in (
                pa.types.is_integer,
                pa.types.is_floating,
                pa.types.is_decimal,
                pa.types.is_boolean,
                pa.types.is_temporal,
                pa.types.is_null,
            )
        )

    def _read_as_dictionary(data_type: Any) -> bool:
        # Byte-array columns of these types come back as dictionaries when read with read_dictionary;
        # an extension type comes back as itself, with every value materialised.
        return not _is_extension(data_type) and any(
            check(_value_type(data_type))
            for check in (
                pa.types.is_string,
                pa.types.is_large_string,
                pa.types.is_binary,
                pa.types.is_large_binary,
                pa.types.is_string_view,
                pa.types.is_binary_view,
            )
        )

    def _value_lengths(array: Any) -> Any:
        if pa.types.is_dictionary(array.type):
            return pc.take(_value_lengths(array.dictionary), array.indices)
        if pa.types.is_string_view(array.type) or pa.types.is_binary_view(array.type):
            # The length kernel does not read the view layouts; plain binary holds the same bytes.
            array = array.cast(pa.large_binary())
        return pc.binary_length(array)

    parquet_file = pq.ParquetFile(path)
    try:
        schema = parquet_file.schema_arrow
        n_rows = int(parquet_file.metadata.num_rows)
    finally:
        parquet_file.close()
    nested = [field.name for field in schema if pa.types.is_nested(field.type)]
    if nested:
        raise UserInputError(
            "Parquet columns with nested values (lists, structs or maps) are not supported: "
            + ", ".join(str(name) for name in nested[:5])
            + ". Flatten them or export the table as CSV."
        )
    text_fields = [field for field in schema if _holds_text(field.type)]
    fixed_width = [field for field in text_fields if pa.types.is_fixed_size_binary(_value_type(field.type))]
    text_columns = [field.name for field in text_fields if _read_as_dictionary(field.type)]
    unsupported = [field for field in text_fields if field not in fixed_width and field.name not in text_columns]
    if unsupported:
        raise UserInputError(
            "Parquet columns of these types are not supported: "
            + ", ".join(f"{field.name} ({field.type})" for field in unsupported[:5])
            + ". Store them as plain text columns or export the table as CSV."
        )
    # Each cell of a fixed-size binary column decodes to its full width.
    total = sum(n_rows * int(_value_type(field.type).byte_width) for field in fixed_width)
    if total > limit or not text_columns:
        return total
    reader = pq.ParquetFile(path, read_dictionary=text_columns)
    try:
        for row_group in range(reader.metadata.num_row_groups):
            table = reader.read_row_group(row_group, columns=text_columns)
            for column in table.columns:
                for chunk in column.chunks:
                    total += int(pc.sum(_value_lengths(chunk)).as_py() or 0)
                    if total > limit:
                        return total
    finally:
        reader.close()
    return total


def _dataframe_text_chars(dataframe: pd.DataFrame, limit: int) -> int:
    """Characters of text held by a parsed table, counted until the count passes ``limit``.

    A cell that shares its string with other cells (an Excel shared string, a categorical
    level) counts once per cell, since later steps may copy it per cell.
    """

    total = 0
    for position, dtype in enumerate(dataframe.dtypes):
        if isinstance(dtype, pd.CategoricalDtype):
            series = dataframe.iloc[:, position]
            categories = series.cat.categories
            lengths = np.fromiter(
                (len(value) if isinstance(value, (str, bytes)) else 0 for value in categories),
                dtype=np.int64,
                count=len(categories),
            )
            codes = series.cat.codes.to_numpy()
            total += int(lengths[codes[codes >= 0]].sum())
        elif isinstance(dtype, pd.StringDtype):
            total += int(dataframe.iloc[:, position].str.len().sum())
        elif dtype == object:
            values = dataframe.iloc[:, position].array
            total += sum(len(value) for value in values if isinstance(value, (str, bytes)))
        else:
            continue  # numbers, booleans and dates are bounded by the cell limit
        if total > limit:
            break
    return total


# Most members an .xlsx container may hold: a workbook has a few dozen parts, one with many sheets
# or images a few thousand.
_MAX_XLSX_MEMBERS = 10_000
_XML_READ_CHUNK_BYTES = 64 * 1024
# How an XML part can start in the encodings XML parsers detect: a UTF-8 or UTF-16 byte order
# mark, UTF-16 text without one, or (after optional whitespace) "<".
_XML_PART_SIGNATURES = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff", b"\x00<")


class _XmlRootReached(Exception):
    """Ends the prolog scan of an XML part at its root element."""


def _is_xml_part(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    """True when a workbook member starts like an XML document, whatever its name.

    openpyxl finds the shared strings through the content types and the sheets through the
    workbook relationships, which may give parts any extension, so parts are recognised by their
    content. Images and other binary members never start like XML.
    """

    with archive.open(info) as handle:
        head = handle.read(1024)
    if not head:
        return False
    stripped = head.lstrip(b" \t\r\n")
    return not stripped or stripped.startswith(b"<") or head.startswith(_XML_PART_SIGNATURES)


def _xml_part_declares_document_type(handle: Any) -> bool:
    """True when an XML part declares a document type or an entity before its root element.

    The part is parsed up to its root element in whatever encoding it declares, so a declaration
    after a long comment or in UTF-16 text is found too; a part that cannot be parsed that far
    raises the parser's error.
    """

    from xml.parsers import expat

    parser = expat.ParserCreate()
    declared = False

    def _declaration(*_args: Any) -> None:
        nonlocal declared
        declared = True
        raise _XmlRootReached

    def _root(*_args: Any) -> None:
        raise _XmlRootReached

    parser.StartDoctypeDeclHandler = _declaration
    parser.EntityDeclHandler = _declaration
    parser.StartElementHandler = _root
    try:
        while chunk := handle.read(_XML_READ_CHUNK_BYTES):
            parser.Parse(chunk, False)
        parser.Parse(b"", True)
    except _XmlRootReached:
        pass
    return declared


def _iter_xml_elements(handle: Any, wanted: frozenset[str]) -> Any:
    """Yield each completed element whose local name is in ``wanted``, keeping memory flat.

    Elements outside a wanted one are dropped as soon as they end, so a part with millions
    of cells is streamed rather than built into a tree.
    """

    import xml.etree.ElementTree as ElementTree

    stack: list[Any] = []
    open_wanted = 0
    for event, element in ElementTree.iterparse(handle, events=("start", "end")):
        is_wanted = element.tag.rpartition("}")[2] in wanted
        if event == "start":
            stack.append(element)
            open_wanted += is_wanted
            continue
        stack.pop()
        if is_wanted:
            open_wanted -= 1
            yield element
        if stack and not open_wanted:
            stack[-1].remove(element)


def _xlsx_text_chars(path: Path, limit: int) -> int:
    """Characters that the shared-string cells of a workbook decode to, counted until the count passes ``limit``.

    A shared string is stored once but every cell that uses it becomes its own text value
    when the sheet is read, so the declared part sizes do not bound the parsed text. Parts are
    recognised by their content, not their names: every XML part is checked for document type
    declarations and scanned for shared-string cells. The declared sizes of all parts are
    capped before this runs, and the zip reader never inflates a part past its declared size.
    """

    import xml.etree.ElementTree as ElementTree
    from array import array

    shared_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > _MAX_XLSX_MEMBERS:
            raise UserInputError(
                f"Failed to read Excel file: the workbook holds {len(members):,} parts; SurvStudio reads workbooks "
                f"with at most {_MAX_XLSX_MEMBERS:,}. Export the sheet as CSV instead."
            )
        names = [info.filename for info in members if _is_xml_part(archive, info)]
        for name in names:
            with archive.open(name) as handle:
                if _xml_part_declares_document_type(handle):
                    raise UserInputError("Failed to read Excel file: the workbook contains XML declarations that are not allowed.")
        shared_parts: list[str] = []
        if "[Content_Types].xml" in names:
            # openpyxl locates the shared strings through the content types.
            with archive.open("[Content_Types].xml") as handle:
                for element in _iter_xml_elements(handle, frozenset({"Override"})):
                    if element.get("ContentType") == shared_type:
                        shared_parts.append(str(element.get("PartName") or "").lstrip("/"))
        shared_parts += [name for name in names if name.lower().endswith("sharedstrings.xml")]
        shared_name = next((name for name in shared_parts if name in names), None)
        if shared_name is None:
            return 0
        lengths = array("q")
        with archive.open(shared_name) as handle:
            for element in _iter_xml_elements(handle, frozenset({"si"})):
                lengths.append(sum(len(node.text or "") for node in element.iter() if node.tag.rpartition("}")[2] == "t"))
        if not lengths:
            return 0
        total = 0
        # Every part is scanned, not only the first sheet, so a relocated sheet cannot skip the count.
        for name in names:
            if name in (shared_name, "[Content_Types].xml"):
                continue
            with archive.open(name) as handle:
                try:
                    for cell in _iter_xml_elements(handle, frozenset({"c"})):
                        if cell.get("t") != "s":
                            continue
                        value = next((child.text for child in cell if child.tag.rpartition("}")[2] == "v"), None)
                        try:
                            index = int(str(value).strip())
                        except ValueError:
                            continue
                        if 0 <= index < len(lengths):
                            total += lengths[index]
                            if total > limit:
                                return total
                except ElementTree.ParseError:
                    # A part that is not well-formed (legacy VML drawings can be) is read no further by
                    # the workbook reader either, so the cells before the error are all it can decode.
                    continue
    return total


def _xls_text_chars(path: Path, limit: int) -> int:
    """Characters of text in the first sheet of a legacy .xls workbook (the sheet the loader reads)."""

    try:
        import xlrd
    except ImportError:  # pragma: no cover - the loader reports the missing engine itself
        return 0
    book = xlrd.open_workbook(str(path), on_demand=True)
    try:
        if book.nsheets < 1:
            return 0
        sheet = book.sheet_by_index(0)
        total = 0
        for row in range(min(sheet.nrows, _MAX_UPLOAD_ROWS + 1)):
            total += sum(len(value) for value in sheet.row_values(row) if isinstance(value, str))
            if total > limit:
                break
        return total
    finally:
        book.release_resources()


def _clean_column_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace line breaks and other control characters in column names with a space.

    Excel headers often carry a line break (Alt+Enter). Request fields refuse control
    characters in column names, so such a column would load but could not be analysed.
    Names that need no cleaning keep their spelling (the loader has made them unique); a
    cleaned name that would repeat one of them, or an earlier cleaned name, gets a suffix,
    so a column never takes over the name of another.
    """

    names = [str(column) for column in dataframe.columns]
    if not any(_CONTROL_CHAR_PATTERN.search(name) for name in names):
        return dataframe
    taken = {name for name in names if not _CONTROL_CHAR_PATTERN.search(name)}
    cleaned_names: list[str] = []
    for name in names:
        if not _CONTROL_CHAR_PATTERN.search(name):
            cleaned_names.append(name)
            continue
        base = _HEADER_CONTROL_RUN_PATTERN.sub(" ", name).strip() or "unnamed"
        candidate, counter = base, 1
        while candidate in taken:
            counter += 1
            candidate = f"{base}_{counter}"
        taken.add(candidate)
        cleaned_names.append(candidate)
    dataframe.columns = cleaned_names
    return dataframe


# Leading bytes of the legacy Excel formats (the signatures pandas uses to pick a reader).
_XLS_SIGNATURES = (
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # compound file (BIFF5/BIFF8)
    b"\x09\x00\x04\x00\x07\x00\x10\x00",  # BIFF2
    b"\x09\x02\x06\x00\x00\x00\x10\x00",  # BIFF3
    b"\x09\x04\x06\x00\x00\x00\x10\x00",  # BIFF4
)


def _excel_container(path: Path) -> str | None:
    """"zip" or "xls" from the file's leading bytes; the reader picks its engine by content, not by extension."""

    with path.open("rb") as handle:
        head = handle.read(8)
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head in _XLS_SIGNATURES:
        return "xls"
    return None


def _guard_compressed_upload(path: Path, filename: str) -> None:
    """Refuse workbooks/Parquet files whose decompressed size, decoded text or shape exceeds the upload limits."""

    suffix = Path(filename).suffix.lower()
    container = _excel_container(path) if suffix in {".xlsx", ".xls"} else None
    # The reader picks its engine from the content, so the checks follow the content as well: a
    # legacy workbook named .xlsx is read as .xls, and a zip workbook named .xls as .xlsx.
    if container == "xls":
        try:
            text_chars = _xls_text_chars(path, _MAX_UPLOAD_TEXT_CHARS)
        except MemoryError:
            raise
        except Exception as exc:
            logger.info("Rejected an unreadable Excel upload: %s", exc)
            raise UserInputError("Failed to read Excel file: the .xls file is not a valid workbook.") from exc
        if text_chars > _MAX_UPLOAD_TEXT_CHARS:
            raise _text_too_large(text_chars)
        return
    if suffix == ".xlsx" or container == "zip":
        try:
            with zipfile.ZipFile(path) as archive:
                uncompressed_bytes = sum(max(0, int(info.file_size)) for info in archive.infolist())
        except (zipfile.BadZipFile, OSError) as exc:
            raise UserInputError("Failed to read Excel file: the .xlsx container is not a valid workbook.") from exc
        if uncompressed_bytes > _MAX_XLSX_UNCOMPRESSED_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The Excel workbook expands to {_format_megabytes(uncompressed_bytes)} when decompressed. "
                    f"SurvStudio accepts workbooks up to {_format_megabytes(_MAX_XLSX_UNCOMPRESSED_BYTES)} uncompressed; "
                    "export the sheet as CSV instead."
                ),
            )
        try:
            text_chars = _xlsx_text_chars(path, _MAX_UPLOAD_TEXT_CHARS)
        except (UserInputError, MemoryError):
            raise
        except Exception as exc:
            logger.info("Rejected an unreadable Excel upload: %s", exc)
            raise UserInputError("Failed to read Excel file: the .xlsx container is not a valid workbook.") from exc
        if text_chars > _MAX_UPLOAD_TEXT_CHARS:
            raise _text_too_large(text_chars)
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:  # pragma: no cover - pandas reports the missing engine itself
            return
        try:
            # Open the file here so its handle is closed even when pyarrow fails: a handle held by the
            # exception's traceback would stop the temporary upload from being deleted on Windows.
            with open(path, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                try:
                    metadata = parquet_file.metadata
                finally:
                    parquet_file.close()
        except Exception as exc:
            # The parser's own message can name server-side paths; log it and answer generically.
            logger.info("Rejected an unreadable Parquet upload: %s", exc)
            raise UserInputError("Failed to read Parquet file: the file is not a valid Parquet file.") from exc
        _enforce_upload_shape_limits(SimpleNamespace(shape=(int(metadata.num_rows), int(metadata.num_columns))))
        uncompressed_bytes = sum(
            max(0, int(metadata.row_group(index).total_byte_size)) for index in range(metadata.num_row_groups)
        )
        if uncompressed_bytes > _MAX_PARQUET_UNCOMPRESSED_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The Parquet file expands to {_format_megabytes(uncompressed_bytes)} when decompressed. "
                    f"SurvStudio accepts Parquet data up to {_format_megabytes(_MAX_PARQUET_UNCOMPRESSED_BYTES)} uncompressed."
                ),
            )
        try:
            text_bytes = _parquet_text_bytes(path, _MAX_UPLOAD_TEXT_CHARS)
        except (UserInputError, MemoryError):
            raise
        except Exception as exc:
            logger.info("Rejected an unreadable Parquet upload: %s", exc)
            raise UserInputError("Failed to read Parquet file: the file is not a valid Parquet file.") from exc
        if text_bytes > _MAX_UPLOAD_TEXT_CHARS:
            raise _text_too_large(text_bytes)


def _store_loaded_dataframe(
    dataframe: Any,
    *,
    filename: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    copy_dataframe: bool = True,
) -> dict[str, Any]:
    """Validate, hash, store and profile a freshly loaded table (blocking; run in a worker thread)."""

    _enforce_upload_shape_limits(dataframe)
    # Checked before hashing and profiling, which touch every text cell.
    text_chars = _dataframe_text_chars(dataframe, _MAX_UPLOAD_TEXT_CHARS)
    if text_chars > _MAX_UPLOAD_TEXT_CHARS:
        raise _text_too_large(text_chars)
    ensure_model_feature_candidate_limit(dataframe)
    stored = store.create(
        dataframe,
        filename=filename,
        source=source,
        metadata=metadata,
        copy_dataframe=copy_dataframe,
    )
    return dataset_response(stored.dataset_id)


def _ingest_uploaded_file(path: Path, filename: str) -> dict[str, Any]:
    _guard_compressed_upload(path, filename)
    # Text and Excel inputs are checked against the shape limits from their header and a
    # bounded read, so an oversized table is refused before it is fully parsed.
    dataframe = load_dataframe_from_path(
        path,
        max_rows=_MAX_UPLOAD_ROWS,
        max_columns=_MAX_UPLOAD_COLUMNS,
        max_cells=_MAX_UPLOAD_CELLS,
    )
    _clean_column_labels(dataframe)
    # The parsed frame is private to this request, so the store can keep it without a deep copy.
    return _store_loaded_dataframe(dataframe, filename=filename, source="upload", copy_dataframe=False)


# Longest table note an export accepts (TableExportRequest.notes). The replay and provenance notes
# the server writes itself stay within it, so an export never refuses the server's own notes.
_MAX_EXPORT_NOTE_CHARS = 4000
# Room kept after the listed names for the count of those left out.
_NAME_LIST_COUNT_ROOM = 64
# Items of each list a JSON replay note keeps, tried in turn until the note fits the limit.
_REPLAY_JSON_LIST_ITEMS = (50, 20, 5, 0)


def _capped_note(text: str) -> str:
    """``text`` cut to the export note limit and marked where it was cut."""

    if len(text) <= _MAX_EXPORT_NOTE_CHARS:
        return text
    marker = " ... (truncated)"
    return text[: _MAX_EXPORT_NOTE_CHARS - len(marker)] + marker


def _name_list_note(label: str, names: Sequence[str]) -> str:
    """``"<label>: a, b, c."``; when that passes the note limit, the first names that fit and a count of the rest."""

    full = f"{label}: " + ", ".join(names) + "."
    if len(full) <= _MAX_EXPORT_NOTE_CHARS:
        return full
    budget = _MAX_EXPORT_NOTE_CHARS - len(label) - _NAME_LIST_COUNT_ROOM
    shown: list[str] = []
    used = 0
    for name in names:
        used += len(name) + 2
        if used > budget:
            break
        shown.append(name)
    listed = ", ".join(shown) + ", and " if shown else ""
    return _capped_note(f"{label}: {listed}{len(names) - len(shown):,} more ({len(names):,} in total).")


def _shortened_lists(value: Any, keep: int) -> Any:
    """``value`` with every list longer than ``keep`` items cut to its first items and a count of the rest."""

    if isinstance(value, dict):
        return {key: _shortened_lists(item, keep) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_shortened_lists(item, keep) for item in value[:keep]]
        if len(value) > keep:
            items.append(f"... and {len(value) - keep:,} more ({len(value):,} in total)")
        return items
    return value


def _replay_json_note(label: str, payload: dict[str, Any]) -> str:
    """``"<label>: <payload as JSON>"``; long lists (such as wide feature sets) are summarised to fit the note limit."""

    def _note(value: Any) -> str:
        return f"{label}: " + json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)

    text = _note(payload)
    for keep in _REPLAY_JSON_LIST_ITEMS:
        if len(text) <= _MAX_EXPORT_NOTE_CHARS:
            break
        text = _note(_shortened_lists(payload, keep))
    return _capped_note(text)


def _replay_dataset_note(request_config: dict[str, Any], *, dataset_filename: str) -> str:
    return (
        "Replay dataset: "
        f"{dataset_filename}. Outcome: time={request_config.get('time_column')}, "
        f"event={request_config.get('event_column')}={request_config.get('event_positive_value')}."
    )


def _replay_feature_notes(request_config: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    features = [str(value) for value in request_config.get("features") or []]
    categorical = [str(value) for value in request_config.get("categorical_features") or []]
    if features:
        notes.append(_name_list_note("Replay features", features))
    if categorical:
        notes.append(_name_list_note("Replay categorical features", categorical))
    return notes


def _ml_replay_notes(request_config: dict[str, Any], *, dataset_filename: str) -> list[str]:
    evaluation_strategy = str(request_config.get("evaluation_strategy", "holdout"))
    settings = [
        f"evaluation={evaluation_strategy}",
        f"random_state={request_config.get('random_state')}",
        f"n_estimators={request_config.get('n_estimators')}",
        (
            f"max_depth={request_config.get('max_depth')}"
            if request_config.get("max_depth") is not None
            else "max_depth=auto (RSF: unlimited; GBS: 3)"
        ),
    ]
    if request_config.get("learning_rate") is not None:
        settings.append(f"learning_rate={request_config.get('learning_rate')}")
    if evaluation_strategy == "repeated_cv":
        settings.append(
            f"cv={request_config.get('cv_repeats', 1)}x{request_config.get('cv_folds', 1)}"
        )
        if request_config.get("locked_test_fraction"):
            settings.append(f"locked_test_fraction={request_config.get('locked_test_fraction')}")

    notes = [
        _replay_dataset_note(request_config, dataset_filename=dataset_filename),
        "Replay settings: " + "; ".join(settings) + ".",
        *_replay_feature_notes(request_config),
    ]
    return [_capped_note(note) for note in notes]


def _dl_replay_notes(
    request_config: dict[str, Any],
    *,
    dataset_filename: str,
    resolved_analysis: dict[str, Any] | None = None,
) -> list[str]:
    evaluation_strategy = str(request_config.get("evaluation_strategy", "holdout"))
    settings = [
        f"evaluation={evaluation_strategy}",
        f"random_seed={request_config.get('random_seed')}",
        f"epochs={request_config.get('epochs')}",
        f"batch_size={request_config.get('batch_size')}",
        f"learning_rate={request_config.get('learning_rate')}",
        f"hidden_layers={request_config.get('hidden_layers')}",
        f"dropout={request_config.get('dropout')}",
        f"early_stopping_patience={request_config.get('early_stopping_patience')}",
        f"early_stopping_min_delta={request_config.get('early_stopping_min_delta')}",
    ]
    if request_config.get("parallel_jobs") is not None:
        settings.append(f"parallel_jobs={request_config.get('parallel_jobs')}")
    if evaluation_strategy == "repeated_cv":
        settings.append(
            f"cv={request_config.get('cv_repeats', 1)}x{request_config.get('cv_folds', 1)}"
        )
        if request_config.get("locked_test_fraction"):
            settings.append(f"locked_test_fraction={request_config.get('locked_test_fraction')}")
    if request_config.get("num_time_bins") is not None:
        settings.append(f"num_time_bins={request_config.get('num_time_bins')}")
    if request_config.get("d_model") is not None:
        settings.append(f"d_model={request_config.get('d_model')}")
    if request_config.get("n_heads") is not None:
        settings.append(f"n_heads={request_config.get('n_heads')}")
    if request_config.get("n_layers") is not None:
        settings.append(f"n_layers={request_config.get('n_layers')}")
    if request_config.get("latent_dim") is not None:
        settings.append(f"latent_dim={request_config.get('latent_dim')}")
    if request_config.get("n_clusters") is not None:
        settings.append(f"n_clusters={request_config.get('n_clusters')}")

    notes = [
        _replay_dataset_note(request_config, dataset_filename=dataset_filename),
        "Replay settings: " + "; ".join(settings) + ".",
    ]
    if resolved_analysis:
        actual_eval = resolved_analysis.get("evaluation_mode")
        if actual_eval is not None:
            notes.append(f"Reported evaluation outcome: {actual_eval}.")
        actual_note = (resolved_analysis.get("evaluation_note") or "").strip()
        if actual_note:
            notes.append("Reported evaluation note: " + actual_note)
    notes.extend(_replay_feature_notes(request_config))
    return [_capped_note(note) for note in notes]


def _attach_manuscript_notes(
    analysis: dict[str, Any],
    extra_notes: Sequence[str],
) -> None:
    manuscript = analysis.get("manuscript_tables")
    if not isinstance(manuscript, dict):
        return
    existing = list(manuscript.get("table_notes", []))
    for note in extra_notes:
        if note not in existing:
            existing.append(note)
    manuscript["table_notes"] = existing


def _export_provenance_notes(provenance: dict[str, Any] | None) -> list[str]:
    """Replay notes for an exported table, stamped with the SurvStudio version and dataset hash.

    Results changed between releases (for example the 0.2.0 statistical fixes), so every
    analysis export records which version produced it and, when known, which stored
    dataset. Plain table exports without provenance stay data-only. Each note stays within
    the export note limit: long lists in the replayed settings (a wide feature set) are
    summarised by their first items and a count.
    """
    if not provenance:
        return []
    notes: list[str] = [f"Generated with SurvStudio {SURVSTUDIO_VERSION}."]
    dataset_hash = str(provenance.get("dataset_hash") or "").strip()
    if dataset_hash:
        notes.append(_capped_note(f"Dataset fingerprint: {dataset_hash}."))
    request_config = provenance.get("request_config")
    if isinstance(request_config, dict) and request_config:
        notes.append(_replay_json_note("Replay request_config", request_config))

    analysis_meta = provenance.get("analysis")
    if isinstance(analysis_meta, dict) and analysis_meta:
        notes.append(_replay_json_note("Replay analysis metadata", analysis_meta))

    return notes


# Memory exhaustion reported by PyTorch ("CUDA out of memory", "DefaultCPUAllocator: can't allocate memory").
_OUT_OF_MEMORY_MESSAGE = re.compile(r"\bout of memory\b|\bcan(?:'|no)t allocate memory\b", re.IGNORECASE)
# Non-finite values named in a library message, as whole words ("infer" or "information" do not count).
_NON_FINITE_MESSAGE = re.compile(r"\b(?:nan|inf|infinity|non-finite|overflow|underflow)\b|too large for dtype", re.IGNORECASE)
_UNPROCESSABLE_REQUEST_DETAIL = "The request could not be processed with the selected dataset and settings."


def _classify_runtime_request_error(exc: Exception) -> tuple[int, str] | None:
    """The 4xx status and guidance for a numerical failure that the request's data or settings cause.

    Only specific exception types count, and messages only by whole words: a message that merely
    contains "shape", "alloc" or "inf" (as in "Could not infer dtype") is a server fault and is
    reported as one.
    """

    if isinstance(exc, np.linalg.LinAlgError):
        return (
            400,
            "The analysis hit a linear-algebra stability problem (for example, a singular or redundant design matrix). "
            "Reduce overlapping variables, sparse categories, or feature count and try again.",
        )
    if type(exc).__name__ == "ConvergenceError":
        return (
            400,
            "The model did not converge. Reduce overlapping variables, sparse categories, or feature count and try again.",
        )
    if isinstance(exc, InternalAnalysisError):
        if _NON_FINITE_MESSAGE.search(str(exc.__cause__ or "")):
            return (
                400,
                "The analysis encountered missing, infinite, or out-of-range values in the selected columns. "
                "Check those columns for invalid values and try again.",
            )
        return None
    if isinstance(exc, ArithmeticError):
        return (
            400,
            "The analysis hit a numerical stability problem. Reduce sparse levels, simplify the model, or narrow the feature set and try again.",
        )
    if isinstance(exc, RuntimeError):
        message = str(exc)
        if type(exc).__name__ == "OutOfMemoryError" or _OUT_OF_MEMORY_MESSAGE.search(message):
            return (
                400,
                "The run ran out of memory. Reduce batch size, model width, or feature count and try again.",
            )
        if _NON_FINITE_MESSAGE.search(message):
            return (
                400,
                "The run became numerically unstable. Check for invalid values, simplify the model, or reduce the learning rate and try again.",
            )
    return None


def _dependency_error_detail(exc: ImportError) -> str:
    """What a 503 says about a package that could not be imported.

    SurvStudio's own install hints (a DependencyError, or an ImportError that SurvStudio code
    raised with a message) are shown as written. A library's import failure can name server
    paths, so it is summarised by the module that is missing.
    """

    if isinstance(exc, DependencyError) or (exc.name is None and exc.path is None and _raised_by_survstudio(exc)):
        message = str(exc).strip()
        if message:
            return message
    module = str(exc.name or "").split(".", 1)[0]
    subject = f'The package "{module}"' if module else "A package"
    return (
        f"{subject} that this analysis needs is not installed or could not be loaded. Install SurvStudio's optional "
        'dependencies (for example pip install -e ".[all]") and restart it; the server log has the details.'
    )


def fail_bad_request(exc: Exception) -> NoReturn:
    """Raise the HTTP error for an exception from a request handler.

    User input errors keep their message (404 for a missing dataset or column, else 400).
    Numerical failures caused by the data map to a 400 with guidance, and an untyped
    ValueError to a generic 400. TypeErrors, the coding errors `is_programming_error` names and
    every other failure are 500s with a generic message. Every path but a user input error is
    logged with its traceback.
    """

    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, JobCancelledError):
        # 499 (client closed request): the page abandoned this request, nobody reads the body.
        logger.info("Request cancelled by its client: %s", exc)
        raise HTTPException(status_code=499, detail=_json_safe_text(str(exc))) from exc
    if isinstance(exc, UserInputError):
        # Messages can quote request text (a column name), which may hold an unpaired surrogate.
        status_code = 404 if isinstance(exc, NotFoundError) else 400
        raise HTTPException(status_code=status_code, detail=_json_safe_text(str(exc))) from exc
    exc_info = (type(exc), exc, exc.__traceback__)
    if isinstance(exc, MemoryError):
        logger.error("Request ran out of memory", exc_info=exc_info)
        raise HTTPException(
            status_code=500,
            detail="The analysis ran out of memory. Reduce the cohort size, feature count, or model complexity and try again.",
        ) from exc
    if isinstance(exc, csv.Error):
        # Raised by the csv module for an over-long field or broken quoting in an uploaded file.
        logger.warning("Rejected an unreadable delimited upload", exc_info=exc_info)
        raise HTTPException(
            status_code=400,
            detail="The file could not be read as delimited text: a field is too long or its quoting is malformed.",
        ) from exc
    if isinstance(exc, UnicodeDecodeError):
        logger.warning("Rejected an upload that is not UTF-8 text", exc_info=exc_info)
        raise HTTPException(
            status_code=400,
            detail="The file is not valid UTF-8 text. Save it as UTF-8 (for example \"CSV UTF-8\" in Excel) and try again.",
        ) from exc
    # Coding errors come first, so an error that errors.py counts as one is never reported as a data problem.
    if isinstance(exc, TypeError) or is_programming_error(exc):
        logger.error("SurvStudio coding error while handling a request", exc_info=exc_info)
        raise HTTPException(status_code=500, detail=InternalAnalysisError.default_message) from exc
    runtime_classification = _classify_runtime_request_error(exc)
    if runtime_classification is not None:
        logger.warning("Request failed on its data or settings (%s)", type(exc).__name__, exc_info=exc_info)
        status_code, detail = runtime_classification
        raise HTTPException(status_code=status_code, detail=detail) from exc
    if isinstance(exc, ImportError):
        logger.error("A package the request needs could not be imported", exc_info=exc_info)
        raise HTTPException(status_code=503, detail=_json_safe_text(_dependency_error_detail(exc))) from exc
    if isinstance(exc, InternalAnalysisError):
        logger.error("Unexpected internal analysis error", exc_info=exc_info)
        raise HTTPException(status_code=500, detail=_json_safe_text(str(exc)) or InternalAnalysisError.default_message) from exc
    if isinstance(exc, ValueError):
        logger.warning("Request failed with an untyped ValueError", exc_info=exc_info)
        raise HTTPException(status_code=400, detail=_UNPROCESSABLE_REQUEST_DETAIL) from exc
    logger.error("Unhandled SurvStudio request error", exc_info=exc_info)
    raise HTTPException(status_code=500, detail=InternalAnalysisError.default_message) from exc


_P_VALUE_LABEL_TOKENS = frozenset({"p", "pvalue", "pvalues", "pval", "qvalue", "qvalues"})
_JOURNAL_P_VALUE_FLOOR = 0.001
_JOURNAL_SIGNIFICANCE_THRESHOLD = 0.05


def _is_p_value_column(column: Any) -> bool:
    """Mirror the frontend's p-value label detection (P value, p-value, BH adjusted p, logrank_p, ...)."""

    tokens = re.sub(r"[\s_\-.]+", " ", str(column or "").lower()).split()
    if any(token in _P_VALUE_LABEL_TOKENS for token in tokens):
        return True
    return any(first == "q" and second.startswith("value") for first, second in zip(tokens, tokens[1:]))


def _format_journal_p_value(value: float) -> str:
    if value < _JOURNAL_P_VALUE_FLOOR:
        return f"<{_JOURNAL_P_VALUE_FLOOR:.3f}"
    below_threshold = value < _JOURNAL_SIGNIFICANCE_THRESHOLD
    for decimals in range(3, 9):
        text = f"{value:.{decimals}f}"
        # Never let rounding move a p-value across the conventional 0.05 threshold (0.0496 -> "0.050").
        if (float(text) < _JOURNAL_SIGNIFICANCE_THRESHOLD) == below_threshold:
            return text
    # Only a p-value within 5e-9 below the threshold gets here; the shortest exact text keeps it below.
    return repr(float(value)) if below_threshold else f"{value:.3f}"


# Words of a column label that mark whole-number counts ("Events, n", "Rank", "Number at risk"),
# which journal style keeps as integers.
_COUNT_LABEL_TOKENS = frozenset(
    {
        "n",
        "number",
        "count",
        "counts",
        "total",
        "rank",
        "events",
        "patients",
        "subjects",
        "samples",
        "cases",
        "rows",
        "folds",
        "repeats",
        "evaluations",
        "failures",
        "seed",
        "seeds",
        "epochs",
        "iterations",
    }
)


def _is_count_column(column: Any) -> bool:
    tokens = re.sub(r"[\s_\-.,;:()/%]+", " ", str(column or "").lower()).split()
    return any(token in _COUNT_LABEL_TOKENS for token in tokens)


def _journal_number_value(value: Any, column: Any) -> Any:
    """A whole number in a measurement column as a float, so journal style formats it like its neighbours.

    Browsers serialise 1.0 as 1, so a C-index or a P value of exactly 1 (or 0) arrives as an
    integer. Counts and ranks keep their integers.
    """

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and column is not None
        and abs(value) <= _MAX_EXACT_FLOAT_INT
        and not _is_count_column(column)
    ):
        return float(value)
    return value


def _format_journal_number(value: float) -> str:
    if value == 0:
        return "0.000"
    if abs(value) < 0.001:
        # Tiny non-zero values would otherwise collapse to "0.000"/"-0.000".
        return f"{value:.2e}"
    return f"{value:.3f}"


def _format_export_value(value: Any, style: str, column: Any = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if style == "journal":
        value = _journal_number_value(value, column)
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if value == 0:
                value = 0.0  # normalize -0.0
            if style == "journal":
                if column is not None and _is_p_value_column(column) and 0.0 <= value <= 1.0:
                    return _format_journal_p_value(value)
                return _format_journal_number(value)
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


def _normalize_export_text(value: Any, style: str, column: Any = None) -> str:
    text = _format_export_value(value, style, column).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return _clean_export_characters(text).strip()


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


def _export_columns(rows: list[dict[str, Any]], columns: Sequence[str] | None = None) -> list[str]:
    if not rows:
        raise UserInputError("No rows available for export.")
    resolved_columns: list[str] = []
    seen: set[str] = set()

    def _maybe_add(column: Any) -> None:
        name = str(column)
        if not name or name.startswith("_") or name in seen:
            return
        seen.add(name)
        resolved_columns.append(name)

    for column in columns or []:
        _maybe_add(column)
    for row in rows:
        for column in row.keys():
            _maybe_add(column)

    if not resolved_columns:
        raise UserInputError("No exportable columns available for export.")
    return resolved_columns


def _clean_export_characters(text: str) -> str:
    """Replace control characters (and lone surrogates) that break XML-based or UTF-8 exports."""

    return _EXPORT_ILLEGAL_CHAR_PATTERN.sub(" ", text)


def _export_header_text(column: Any) -> str:
    """A column name as the exports write it: a run of control characters (a line break in a group
    value that became a column) becomes one space."""

    return _clean_export_characters(_HEADER_CONTROL_RUN_PATTERN.sub(" ", str(column))).strip()


def _is_number_like_cell(text: str) -> bool:
    """True for signed numeric summaries like "-0.42 ± 1.00" or "-1.2 (-2.0 to -0.4)".

    Linear-time check: the cell must start with an optionally signed number and contain only
    digits and numeric punctuation afterwards, so it cannot spell a spreadsheet function call.
    """

    stripped = text.strip()
    body = stripped[1:] if stripped[:1] in {"+", "-", "\u2212"} else stripped
    if not body:
        return False
    if not (body[0].isdigit() or (body[0] == "." and body[1:2].isdigit())):
        return False
    reduced = _NUMBER_LIKE_TO_WORD.sub(" ", _NUMBER_LIKE_EXPONENT.sub("", stripped))
    return _NUMBER_LIKE_CELL_CHARS.fullmatch(reduced) is not None


def _sanitize_csv_cell(value: Any) -> str:
    """Neutralize spreadsheet formula injection in CSV cells (CSV output only)."""

    text = "" if value is None else _clean_export_characters(str(value))
    stripped = text.lstrip(" ")
    if stripped.startswith("'") and _SIGNED_NUMERIC_CSV_LITERAL.fullmatch(stripped[1:]):
        prefix_len = len(text) - len(stripped)
        return text[:prefix_len] + stripped[1:]
    if text.startswith(_CSV_FORMULA_TRIGGER_CHARS) or stripped.startswith(_CSV_FORMULA_TRIGGER_CHARS):
        return f"'{text}"
    if (
        stripped.startswith(_CSV_SIGN_CHARS)
        and stripped.strip("+- ")  # a bare "-" placeholder cannot form a formula
        and not _is_number_like_cell(stripped)
    ):
        return f"'{text}"
    return text


def _markdown_literal(text: str) -> str:
    """Text that Markdown shows as written, escaped as reporting.checklist_markdown does: characters
    that open raw HTML become entities and backslashes are doubled."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\\", "\\\\")


def _sanitize_markdown_cell(value: Any, style: str, column: Any = None) -> str:
    # GFM removes one backslash before a pipe inside a table cell, so backslashes are doubled first and
    # a pipe then becomes \| : "a\|b" stays one cell reading a\|b, and "a|b" one cell reading a|b.
    return _markdown_literal(_normalize_export_text(value, style, column)).replace("|", "\\|")


def _clean_export_note(note: Any) -> str:
    return _normalize_export_text(note, "plain")


def _export_rows_to_csv(
    rows: list[dict[str, Any]],
    style: str,
    *,
    columns: Sequence[str] | None = None,
    caption: str | None = None,
    notes: Sequence[str] | None = None,
    template: str = "default",
) -> str:
    if not rows:
        raise UserInputError("No rows available for export.")
    resolved_columns = _export_columns(rows, columns)
    buffer = io.StringIO()
    # A UTF-8 BOM lets Excel detect the encoding (otherwise "±" renders as mojibake).
    buffer.write(_CSV_UTF8_BOM)
    writer = csv.writer(buffer)
    clean_caption = _clean_export_note(caption) if caption else ""
    clean_notes = [clean_note for note in (notes or []) if (clean_note := _clean_export_note(note))]
    # Preamble lines are written through the CSV writer as one quoted cell each, so commas in
    # captions/notes (for example filename-derived text) cannot open a new formula cell.
    if clean_caption:
        writer.writerow([_sanitize_csv_cell(f"# {clean_caption}")])
    if clean_notes:
        writer.writerow([_sanitize_csv_cell(f"# {_export_template_profile(template)['notes_heading']}:")])
        for note in clean_notes:
            writer.writerow([_sanitize_csv_cell(f"# - {note}")])
    writer.writerow([_sanitize_csv_cell(_export_header_text(column)) for column in resolved_columns])
    for row in rows:
        writer.writerow(
            [
                _sanitize_csv_cell(_format_export_value(row.get(column), style, column))
                for column in resolved_columns
            ]
        )
    return buffer.getvalue()


def _xlsx_text_cell(value: Any) -> tuple[str | None, None]:
    text = _clean_export_characters("" if value is None else str(value))
    return (text if text else None), None


def _xlsx_cell(value: Any, style: str, column: Any) -> tuple[Any, str | None]:
    """Return an XLSX cell value plus number format; numbers stay numeric cells."""

    if value is None or isinstance(value, bool):
        return _xlsx_text_cell(_format_export_value(value, style, column))
    if style == "journal":
        value = _journal_number_value(value, column)
    if isinstance(value, int):
        return value, None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, None
        numeric_value = 0.0 if value == 0 else value
        if style != "journal":
            return numeric_value, None
        display = _format_export_value(numeric_value, style, column)
        try:
            float(display)
        except ValueError:
            return _xlsx_text_cell(display)  # e.g. "<0.001"
        if "e" in display.lower():
            return numeric_value, "0.00E+00"
        decimals = len(display.split(".", 1)[1]) if "." in display else 0
        return numeric_value, ("0." + "0" * decimals) if decimals else "0"
    return _xlsx_text_cell(_format_export_value(value, style, column))


def _export_rows_to_xlsx(
    rows: list[dict[str, Any]],
    style: str,
    *,
    columns: Sequence[str] | None = None,
    caption: str | None = None,
    notes: Sequence[str] | None = None,
    template: str = "default",
) -> bytes:
    if not rows:
        raise UserInputError("No rows available for export.")
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise UserInputError(
            "XLSX export requires openpyxl. Install the formats extra with `pip install -e \".[formats]\"`."
        ) from exc

    resolved_columns = _export_columns(rows, columns)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Table"
    current_row = 1

    def _write_row(values: Sequence[tuple[Any, str | None]]) -> None:
        nonlocal current_row
        for column_index, (value, number_format) in enumerate(values, start=1):
            if value is None:
                continue
            cell = worksheet.cell(row=current_row, column=column_index)
            cell.value = value
            if isinstance(value, str) and value.startswith("="):
                # openpyxl would store "=..." as a live formula; keep it literal text instead.
                cell.data_type = "s"
                cell.quotePrefix = True
            if number_format:
                cell.number_format = number_format
        current_row += 1

    resolved_caption = (caption or "").strip()
    if resolved_caption:
        _write_row([_xlsx_text_cell(resolved_caption)])
        current_row += 1

    _write_row([_xlsx_text_cell(_export_header_text(column)) for column in resolved_columns])
    for row in rows:
        _write_row([_xlsx_cell(row.get(column), style, column) for column in resolved_columns])

    clean_notes = [clean_note for note in (notes or []) if (clean_note := _clean_export_note(note))]
    if clean_notes:
        current_row += 1
        _write_row([_xlsx_text_cell(_export_template_profile(template)["notes_heading"])])
        for note in clean_notes:
            _write_row([_xlsx_text_cell(note)])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _export_rows_to_markdown(
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> str:
    if not rows:
        raise UserInputError("No rows available for export.")
    resolved_columns = _export_columns(rows, columns)
    template_profile = _export_template_profile(template)
    resolved_caption = _markdown_literal(_clean_export_note(_resolve_export_caption(caption, template)))
    clean_notes = [_markdown_literal(clean_note) for note in notes if (clean_note := _clean_export_note(note))]
    header = f"| {' | '.join(_sanitize_markdown_cell(_export_header_text(column), 'plain') for column in resolved_columns)} |"
    divider = f"| {' | '.join(['---'] * len(resolved_columns))} |"
    body = [
        "| "
        + " | ".join(_sanitize_markdown_cell(row.get(column), style, column) for column in resolved_columns)
        + " |"
        for row in rows
    ]
    sections = []
    if resolved_caption:
        open_marker = template_profile["markdown_open"]
        close_marker = template_profile["markdown_close"]
        sections.append(f"{open_marker}{resolved_caption}{close_marker}" if open_marker or close_marker else resolved_caption)
    sections.append("\n".join([header, divider, *body]))
    if clean_notes:
        sections.append(f"{template_profile['notes_heading']}:")
        sections.append("\n".join(f"- {note}" for note in clean_notes))
    return "\n\n".join(sections) + "\n"


def _latex_escape(value: str) -> str:
    return value.translate(_LATEX_ESCAPE_TABLE)


def _export_rows_to_latex(
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> str:
    if not rows:
        raise UserInputError("No rows available for export.")
    resolved_columns = _export_columns(rows, columns)
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    column_spec = "l" * len(resolved_columns)
    # Every row opens with an empty group: "\\" and "\midrule" read a "[" (or "*") that follows them,
    # even on the next line, as their own argument, which a cell such as "[0.61, 0.74]" would supply.
    body = [
        "{}"
        + " & ".join(_latex_escape(_normalize_export_text(row.get(column), style, column)) for column in resolved_columns)
        + r" \\"
        for row in rows
    ]
    lines = [
        f"\\begin{{table}}[{template_profile['latex_position']}]",
        "\\centering",
        template_profile["latex_size"],
        f"\\caption{{{_latex_escape(resolved_caption)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        "{}" + " & ".join(_latex_escape(_export_header_text(column)) for column in resolved_columns) + r" \\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if notes:
        notes_heading = _latex_escape(template_profile["notes_heading"])
        notes_text = " ".join(
            _latex_escape(normalized_note)
            for note in notes
            if (normalized_note := _normalize_export_text(note, "plain"))
        )
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
    preamble_hints = ["% Requires \\usepackage{booktabs} in the document preamble."]
    if any(ord(character) > 127 for line in lines for character in line):
        preamble_hints.append(
            "% Contains UTF-8 characters: needs \\usepackage[utf8]{inputenc} on LaTeX releases before 2018 "
            "(and \\usepackage[T1]{fontenc} for accented glyphs)."
        )
    return "\n".join([*preamble_hints, *lines]) + "\n"


def _docx_run(text: str, *, bold: bool = False, italic: bool = False) -> str:
    properties: list[str] = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    props_xml = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    safe_text = xml_escape(_clean_export_characters(text.replace("\r", " ").replace("\n", " ")))
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


def _docx_table(
    rows: list[dict[str, Any]],
    *,
    style: str,
    columns: Sequence[str] | None = None,
    widths: Sequence[int] | None = None,
) -> str:
    resolved_columns = _export_columns(rows, columns)
    cell_width = max(1200, int(9000 / max(len(resolved_columns), 1)))
    # Column widths in twentieths of a point; equal unless the caller gives one per column.
    column_widths = list(widths) if widths is not None and len(widths) == len(resolved_columns) else [cell_width] * len(resolved_columns)
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in column_widths)
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
    header_row = (
        "<w:tr>"
        + "".join(
            _docx_cell(_export_header_text(column), width=width, bold=True)
            for column, width in zip(resolved_columns, column_widths)
        )
        + "</w:tr>"
    )
    body_rows = [
        "<w:tr>"
        + "".join(
            _docx_cell(_normalize_export_text(row.get(column), style, column), width=width)
            for column, width in zip(resolved_columns, column_widths)
        )
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
    columns: Sequence[str] | None = None,
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> bytes:
    if not rows:
        raise UserInputError("No rows available for export.")
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    caption_bold = template in {"default", "jco"}
    caption_italic = template == "nejm"
    body_parts = [
        _docx_paragraph(resolved_caption, bold=caption_bold, italic=caption_italic),
        _docx_table(rows, style=style, columns=columns),
    ]
    clean_notes = [clean_note for note in notes if (clean_note := _clean_export_note(note))]
    if clean_notes:
        body_parts.append(_docx_paragraph(f"{template_profile['notes_heading']}:", italic=True))
        for note in clean_notes:
            body_parts.append(_docx_paragraph(note, italic=True))
    return _docx_package(body_parts)


def _docx_package(body_parts: list[str]) -> bytes:
    """A minimal Word document (Letter page, 1-inch margins) holding these body paragraphs and tables."""
    body_parts = [*body_parts]
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
    document_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels_xml)
    return buffer.getvalue()


# ── Core endpoints ──────────────────────────────────────────────


def _json_safe_error_payload(value: Any) -> Any:
    """Replace non-finite floats (e.g. an echoed NaN input) and escape unpaired surrogates, so the 422 body can be serialized."""

    if isinstance(value, float) and not math.isfinite(value):
        if value != value:
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, str):
        return _json_safe_text(value)
    if isinstance(value, dict):
        return {
            (_json_safe_text(key) if isinstance(key, str) else key): _json_safe_error_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_error_payload(item) for item in value]
    return value


_ERROR_INPUT_ECHO_LIMIT = 200


def _is_small_error_input(value: Any) -> bool:
    """True when the echoed input has at most a few hundred values and no long text."""

    pending = [value]
    seen = 0
    while pending:
        item = pending.pop()
        seen += 1
        if seen > _ERROR_INPUT_ECHO_LIMIT:
            return False
        if isinstance(item, (dict, list, tuple)):
            if len(item) > _ERROR_INPUT_ECHO_LIMIT:
                return False
            pending.extend(item.values() if isinstance(item, dict) else item)
        elif isinstance(item, str) and len(item) > 1000:
            return False
    return True


def _trimmed_validation_errors(errors: Sequence[Any]) -> list[Any]:
    """Validation errors without large echoed inputs (a whole prediction block or table), which the page never shows."""

    trimmed: list[Any] = []
    for error in errors:
        if isinstance(error, dict) and "input" in error and not _is_small_error_input(error["input"]):
            error = {**error, "input": "(omitted: too large to echo)"}
        trimmed.append(error)
    return trimmed


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe_error_payload(jsonable_encoder(_trimmed_validation_errors(exc.errors())))},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    response = templates.TemplateResponse(request, "index.html", {"static_version": _static_asset_version()})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.get("/design-check", response_class=HTMLResponse)
async def design_check_page(request: Request) -> HTMLResponse:
    response = templates.TemplateResponse(request, "design_check.html", {"static_version": _static_asset_version()})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "python_version": platform.python_version(),
        "app_version": SURVSTUDIO_VERSION,
        "dependency_versions": {
            "fastapi": _package_version_or_unknown("fastapi"),
            "numpy": _package_version_or_unknown("numpy"),
            "pandas": _package_version_or_unknown("pandas"),
            "plotly": _package_version_or_unknown("plotly"),
            "scipy": _package_version_or_unknown("scipy"),
            "statsmodels": _package_version_or_unknown("statsmodels"),
            "torch": _package_version_or_unknown("torch"),
        },
    }


def _schedule_process_shutdown(delay_seconds: float = 0.35) -> None:
    pid = os.getpid()

    def _shutdown() -> None:
        time.sleep(delay_seconds)
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            try:
                signal.raise_signal(signal.SIGTERM)
            except Exception:
                return

    threading.Thread(target=_shutdown, daemon=True).start()


_SHUTDOWN_CUSTOM_HEADERS = ("x-requested-with", "x-survstudio-request")


def _is_non_simple_shutdown_request(request: Request) -> bool:
    """True when the request could not have been sent by a plain cross-site HTML form."""

    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        return True
    return any(request.headers.get(header) for header in _SHUTDOWN_CUSTOM_HEADERS)


@app.post("/api/shutdown")
async def shutdown_server(request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else ""
    # Behind a reverse proxy on the same machine every client looks like 127.0.0.1, so the
    # page must also have been opened through a loopback address.
    host = _split_host_and_port(request.headers.get("host", ""))
    host_is_loopback = host is not None and _is_loopback_hostname(host[0])
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"} or not host_is_loopback:
        raise HTTPException(status_code=403, detail="Shutdown is allowed only from a local session.")
    if not _is_non_simple_shutdown_request(request):
        raise HTTPException(
            status_code=415,
            detail="Shutdown requests must be sent as JSON (Content-Type: application/json) from the SurvStudio page.",
        )
    _schedule_process_shutdown()
    return {
        "status": "shutting_down",
        "detail": "SurvStudio is stopping. You can close this tab or restart the server with `python -m survival_toolkit`.",
    }


async def _load_builtin_dataset_response(
    loader: Callable[[], Any],
    *,
    filename: str,
    source: str = "builtin_demo",
    preset_name: str | None = None,
) -> dict[str, Any]:
    try:
        metadata = {"preset_name": preset_name} if preset_name else None
        return await run_in_threadpool(
            lambda: _store_loaded_dataframe(loader(), filename=filename, source=source, metadata=metadata)
        )
    except Exception as exc:
        fail_bad_request(exc)


_ALLOWED_UPLOAD_SUFFIXES = frozenset({".csv", ".txt", ".tsv", ".xlsx", ".xls", ".parquet"})


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    temp_path: Path | None = None
    filename = file.filename or "uploaded_dataset.csv"
    try:
        # Check the type before anything is written: the temp file's suffix comes from the
        # client's filename and must never carry path or stream syntax (for example "x.c:sv").
        suffix = (Path(filename).suffix or ".csv").lower()
        if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported input file extension '{suffix}' for '{filename}'. "
                    "Supported formats are CSV, TSV, TXT, XLSX, XLS, and Parquet."
                ),
            )
        total_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_DETAIL)
                temp_file.write(chunk)
        return await run_in_threadpool(_ingest_uploaded_file, temp_path, filename)
    except HTTPException:
        raise
    except Exception as exc:
        fail_bad_request(exc)
    finally:
        await file.close()
        if temp_path is not None:
            _remove_temporary_upload(temp_path)


def _remove_temporary_upload(path: Path) -> None:
    """Delete an uploaded file's temporary copy without hiding the request's own outcome.

    On Windows a file still open elsewhere cannot be deleted; a parser that failed may hold it until its
    traceback is collected, so collect once and retry, and otherwise leave the file to the system.
    """
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        import gc

        gc.collect()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete the temporary upload %s; it is left for the system to clear.", path)


def _reject_overlong_matrix_text(path: Path, *, compressed: bool) -> None:
    """Refuse a text matrix with more lines than any accepted layout can have, before it is parsed.

    Either layout puts a header and then one line per marker or per patient, so no accepted
    matrix has more lines than the larger of the two limits (plus the header). Counting runs
    over the unpacked stream in fixed-size chunks and stops at the limit. UTF-16 text (as the
    readers detect it) is decoded first, because its code units can hold a newline byte inside
    another character.
    """

    import codecs
    import gzip
    import zlib

    from survival_toolkit.analysis import _text_encoding_candidates

    limit = max(MAX_MATRIX_MARKERS, MAX_MATRIX_SAMPLES) + 1
    lines = 0
    unpacked = 0
    chunk_bytes = 1 << 20
    try:
        with (gzip.open(path, "rb") if compressed else path.open("rb")) as handle:
            chunk = handle.read(chunk_bytes)
            encoding = _text_encoding_candidates(chunk, complete=len(chunk) < chunk_bytes)[0] if chunk else "utf-8"
            decoder = codecs.getincrementaldecoder(encoding)(errors="replace") if encoding.startswith("utf-16") else None
            while chunk:
                lines += decoder.decode(chunk).count("\n") if decoder is not None else chunk.count(b"\n")
                unpacked += len(chunk)
                if lines > limit:
                    raise UserInputError(
                        f"The matrix has more than {limit:,} lines; SurvStudio reads at most {MAX_MATRIX_MARKERS:,} markers "
                        f"and {MAX_MATRIX_SAMPLES:,} patients. Keep fewer markers, for example the most variable ones."
                    )
                if unpacked > MAX_DECOMPRESSED_BYTES:
                    return  # the reader reports the size limit itself
                chunk = handle.read(chunk_bytes)
    except (OSError, EOFError, zlib.error):
        return  # a damaged file is reported by the reader with its own message


@app.post("/api/marker-matrix")
async def upload_marker_matrix(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(..., max_length=128),
    id_column: str = Form(..., max_length=512),
    orientation: str = Form("auto"),
) -> dict[str, Any]:
    temp_path: Path | None = None
    filename = file.filename or "marker_matrix.csv"
    try:
        suffix, compressed = matrix_format(filename)
        if not compressed and suffix not in MATRIX_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported matrix file type '{suffix}' for '{filename}'. Use CSV, TSV, TXT or Parquet, optionally gzip-compressed (.gz).",
            )
        suffix = ".gz" if compressed else suffix
        if orientation not in ORIENTATIONS:
            raise HTTPException(status_code=422, detail=f"Unknown matrix layout '{orientation}'.")
        total_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_DETAIL)
                temp_file.write(chunk)

        def _ingest() -> dict[str, Any]:
            stored = _get_stored_dataset(dataset_id)
            if id_column not in stored.dataframe.columns:
                raise UserInputError(f"The ID column '{id_column}' is not in the dataset.")
            if suffix != ".parquet":
                _reject_overlong_matrix_text(temp_path, compressed=compressed)
            patient_ids = stored.dataframe[id_column].tolist()
            matrix = read_marker_matrix(temp_path, filename, patient_ids=patient_ids, orientation=orientation)
            summary = match_summary(matrix, patient_ids)
            if summary["n_matched"] < 10:
                raise UserInputError(
                    f"Only {summary['n_matched']} patients of the dataset are in the matrix; at least 10 are needed. "
                    f"The IDs in '{id_column}' must be written exactly as in the matrix."
                )
            return {
                "matrix_id": marker_matrices.add(matrix),
                "filename": filename,
                "orientation": matrix.orientation,
                "n_markers": len(matrix.marker_names),
                "marker_preview": list(matrix.marker_names[:8]),
                "id_column": id_column,
                "fingerprint": matrix.fingerprint,
                "id_note": matrix.id_note,
                **summary,
            }

        # Parsing a matrix of up to 200 MB is heavy work: it queues with the other heavy jobs, holds the
        # dataset, and never starts (or stops at the reader's checkpoints) once the page has gone away.
        return await _run_dataset_job(dataset_id, _ingest, request=request, heavy=True)
    except HTTPException:
        raise
    except Exception as exc:
        fail_bad_request(exc)
    finally:
        await file.close()
        if temp_path is not None:
            _remove_temporary_upload(temp_path)


@app.delete("/api/marker-matrix/{matrix_id}")
async def remove_marker_matrix(matrix_id: str) -> dict[str, Any]:
    marker_matrices.remove(matrix_id)
    return {"removed": True}


@app.post("/api/load-example")
async def load_example() -> dict[str, Any]:
    return await _load_builtin_dataset_response(make_example_dataset, filename="example_survival_cohort")


@app.post("/api/load-tcga-example")
async def load_tcga_example() -> dict[str, Any]:
    return await _load_builtin_dataset_response(
        load_tcga_luad_example_dataset,
        filename="tcga_luad_xena_example",
        preset_name="tcga_luad",
    )


@app.post("/api/load-tcga-upload-ready")
async def load_tcga_upload_ready() -> dict[str, Any]:
    return await _load_builtin_dataset_response(
        load_tcga_luad_upload_ready_dataset,
        filename="tcga_luad_upload_ready",
        preset_name="tcga_luad",
    )


@app.post("/api/load-gbsg2-example")
async def load_gbsg2_example() -> dict[str, Any]:
    return await _load_builtin_dataset_response(
        load_gbsg2_upload_ready_dataset,
        filename="gbsg2_upload_ready",
        preset_name="gbsg2",
    )


@app.get("/api/dataset/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        # Profiling a large table is CPU-bound; keep it off the event loop.
        return await run_in_threadpool(dataset_response, dataset_id)
    except Exception as exc:
        fail_bad_request(exc)


@app.delete("/api/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict[str, str]:
    """Free a stored dataset (and its cached models) before its idle expiry."""
    try:
        await run_in_threadpool(store.delete, dataset_id)
        return {"status": "deleted", "dataset_id": dataset_id}
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/export-table")
async def export_table(request_model: TableExportRequest) -> Response:
    try:
        return await run_in_threadpool(_render_table_export, request_model)
    except Exception as exc:
        fail_bad_request(exc)


def _render_table_export(request_model: TableExportRequest) -> Response:
    export_notes = list(request_model.notes or [])
    for provenance_note in _export_provenance_notes(request_model.provenance):
        if provenance_note not in export_notes:
            export_notes.append(provenance_note)
    if request_model.format == "csv":
        content = _export_rows_to_csv(
            request_model.rows,
            request_model.style,
            columns=request_model.columns,
            caption=request_model.caption,
            notes=export_notes,
            template=request_model.template,
        )
        return Response(content=content, media_type="text/csv; charset=utf-8")
    if request_model.format == "xlsx":
        content = _export_rows_to_xlsx(
            request_model.rows,
            request_model.style,
            columns=request_model.columns,
            caption=request_model.caption,
            notes=export_notes,
            template=request_model.template,
        )
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if request_model.format == "markdown":
        content = _export_rows_to_markdown(
            request_model.rows,
            columns=request_model.columns,
            caption=request_model.caption,
            notes=export_notes,
            style=request_model.style,
            template=request_model.template,
        )
        return Response(content=content, media_type="text/markdown; charset=utf-8")
    if request_model.format == "latex":
        content = _export_rows_to_latex(
            request_model.rows,
            columns=request_model.columns,
            caption=request_model.caption,
            notes=export_notes,
            style=request_model.style,
            template=request_model.template,
        )
        return Response(content=content, media_type="text/x-tex; charset=utf-8")
    content = _export_rows_to_docx(
        request_model.rows,
        columns=request_model.columns,
        caption=request_model.caption,
        notes=export_notes,
        style=request_model.style,
        template=request_model.template,
    )
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.post("/api/derive-group")
async def derive_group(request_model: DeriveGroupRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.plots import build_cutpoint_scan_figure

        stored = _get_stored_dataset(request_model.dataset_id)

        def _run() -> dict[str, Any]:
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
                min_group_fraction=request_model.min_group_fraction,
                permutation_iterations=request_model.permutation_iterations,
                random_seed=request_model.random_seed,
            )
            # Groups cut from an outcome-informed column (an optimal cutpoint or a signature) carry the
            # outcome information on, whatever the method of the new cut.
            if not summary.get("outcome_informed") and str(request_model.source_column) in _outcome_informed_columns(stored):
                summary["outcome_informed"] = True
                summary["outcome_informed_source"] = str(request_model.source_column)
                if isinstance(summary.get("recipe"), dict):
                    summary["recipe"]["outcome_informed"] = True
            provenance = dict(stored.metadata.get("derived_column_provenance", {}))
            provenance[column_name] = {
                "outcome_informed": bool(summary.get("outcome_informed")),
                "recipe": copy.deepcopy(summary.get("recipe", {})),
            }
            new_metadata = {
                **stored.metadata,
                "derived_column_provenance": provenance,
            }
            payload = _create_dataset_snapshot(stored, updated, metadata=new_metadata)
            payload["derived_column"] = column_name
            payload["derive_summary"] = summary
            if request_model.method == "optimal_cutpoint" and summary.get("scan_data"):
                payload["cutpoint_figure"] = build_cutpoint_scan_figure(summary, variable_name=request_model.source_column)
            return payload

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=request_model.method == "optimal_cutpoint")
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/kaplan-meier")
async def kaplan_meier(request_model: KaplanMeierRequest, request: Request) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        outcome_informed_group = (
            request_model.group_column is not None
            and str(request_model.group_column) in _outcome_informed_columns(stored)
        )

        def _run() -> dict[str, Any]:
            from survival_toolkit.plots import build_km_figure

            group_column = request_model.group_column
            if group_column and group_column not in (request_model.time_column, request_model.event_column):
                # Curves split by the outcome itself (a copy of the event, another endpoint's event) show
                # nothing but the split. The time and event columns get their own message from the analysis.
                # Input checks scan every column, so they run in the worker thread too.
                _reject_survival_outcome_feature_columns(
                    stored,
                    [group_column],
                    time_column=request_model.time_column,
                    event_column=request_model.event_column,
                    event_positive_value=request_model.event_positive_value,
                    context="Kaplan-Meier grouping",
                )
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
                suppress_group_inference=outcome_informed_group,
                outcome_informed_group=outcome_informed_group,
            )
            figure = build_km_figure(
                analysis,
                time_unit_label=request_model.time_unit_label,
                show_confidence_bands=request_model.show_confidence_bands,
            )
            return _attach_dataset_hash(
                {"analysis": analysis, "figure": figure, "request_config": request_config},
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cox")
async def cox(request_model: CoxRequest, request: Request) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        selected_cox_inputs = [
            *request_model.covariates,
            *request_model.categorical_covariates,
            *request_model.strata_columns,
        ]

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                selected_cox_inputs,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="Cox covariates",
            )
            _reject_outcome_informed_columns(
                stored,
                selected_cox_inputs,
                context="Cox covariates",
            )
            _reject_oversized_design(
                stored,
                request_model.covariates,
                request_model.categorical_covariates,
                context="Cox covariates",
            )
            from survival_toolkit.plots import (
                build_cox_diagnostics_figure,
                build_cox_forest_figure,
                build_cox_martingale_figure,
            )

            analysis = compute_cox_analysis(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                covariates=request_model.covariates,
                categorical_covariates=request_model.categorical_covariates,
                strata_columns=request_model.strata_columns,
            )
            figure = build_cox_forest_figure(analysis)
            diagnostics_figure = build_cox_diagnostics_figure(analysis)
            martingale_figure = build_cox_martingale_figure(analysis)
            return _attach_dataset_hash(
                {
                    "analysis": analysis,
                    "figure": figure,
                    "diagnostics_figure": diagnostics_figure,
                    "martingale_figure": martingale_figure,
                    "request_config": request_config,
                },
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cox-preview")
async def cox_preview(request_model: CoxRequest, request: Request) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        selected_cox_inputs = [
            *request_model.covariates,
            *request_model.categorical_covariates,
            *request_model.strata_columns,
        ]

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                selected_cox_inputs,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="Cox covariates",
            )
            _reject_outcome_informed_columns(
                stored,
                selected_cox_inputs,
                context="Cox covariates",
            )
            _reject_oversized_design(
                stored,
                request_model.covariates,
                request_model.categorical_covariates,
                context="Cox covariates",
            )
            preview = preview_cox_analysis_inputs(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                covariates=request_model.covariates,
                categorical_covariates=request_model.categorical_covariates,
                strata_columns=request_model.strata_columns,
            )
            return _attach_dataset_hash({"preview": preview, "request_config": request_config}, stored)

        return await _run_dataset_job(request_model.dataset_id, _run, request=request)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cohort-table")
async def cohort_table(request_model: CohortTableRequest, request: Request) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            if request_model.group_column:
                _reject_outcome_informed_columns(
                    stored,
                    [request_model.group_column],
                    context="grouped cohort tables",
                )
                group_column = str(request_model.group_column)
                if group_column in stored.dataframe.columns:
                    # Each group adds a column and a pass over the cohort; the same cap as Kaplan-Meier groups.
                    n_groups = int(stored.dataframe[group_column].nunique(dropna=True))
                    if n_groups > _MAX_COHORT_TABLE_GROUPS:
                        raise UserInputError(
                            f'"{group_column}" has {n_groups:,} distinct values; a grouped cohort table compares at most '
                            f"{_MAX_COHORT_TABLE_GROUPS} groups. Group the column into fewer categories first."
                        )
            # A text column is listed level by level (each level scans the cohort again), so an ID or
            # free-text column would produce one row per patient.
            for variable in dict.fromkeys(str(name) for name in request_model.variables):
                if variable not in stored.dataframe.columns:
                    continue  # reported by the analysis itself
                series = stored.dataframe[variable]
                if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                    continue
                n_levels = int(series.nunique(dropna=True))
                if n_levels > _MAX_COHORT_TABLE_LEVELS:
                    raise UserInputError(
                        f'"{variable}" has {n_levels:,} distinct values; a cohort table lists at most '
                        f"{_MAX_COHORT_TABLE_LEVELS} levels per categorical variable. Leave out ID or free-text columns "
                        "or group the values into fewer categories."
                    )
            dataframe = stored.dataframe
            cohort_note: str | None = None
            analysis_cohort: dict[str, Any] | None = None
            if request_model.time_column and request_model.event_column:
                # Same row rule as KM/Cox: valid time and event values and non-negative follow-up time. The
                # boundary keeps the rule's own messages ("No events were found ...") for the user.
                survival_frame = user_input_boundary(_cohort_frame)(
                    dataframe,
                    time_column=request_model.time_column,
                    event_column=request_model.event_column,
                    event_positive_value=request_model.event_positive_value,
                )
                keep_mask = dataframe.index.isin(pd.Index(survival_frame.attrs["source_row_index"]))
                n_total = int(dataframe.shape[0])
                dataframe = dataframe.loc[keep_mask]
                n_kept = int(dataframe.shape[0])
                analysis_cohort = {
                    "time_column": request_model.time_column,
                    "event_column": request_model.event_column,
                    "event_positive_value": request_model.event_positive_value,
                    "n_rows_total": n_total,
                    "n_rows_analyzed": n_kept,
                    "n_rows_excluded": n_total - n_kept,
                    "dropped_missing_rows": int(survival_frame.attrs.get("dropped_missing_rows", 0)),
                    "dropped_nonpositive_time_rows": int(survival_frame.attrs.get("dropped_nonpositive_time_rows", 0)),
                }
                cohort_note = (
                    f"Restricted to the survival analysis cohort: {n_kept} of {n_total} rows with a valid "
                    f"{request_model.time_column} time, a valid {request_model.event_column} status, and non-negative "
                    "follow-up time (the same rows used by Kaplan-Meier and Cox analyses)."
                )
            analysis = compute_cohort_table(
                dataframe,
                variables=request_model.variables,
                group_column=request_model.group_column,
            )
            if cohort_note is not None:
                analysis["analysis_cohort"] = analysis_cohort
                analysis["notes"] = [cohort_note]
                analysis["overall_scope"] = f"{analysis.get('overall_scope', '')} {cohort_note}".strip()
            return _attach_dataset_hash(
                {
                    "analysis": analysis,
                    "request_config": request_config,
                },
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/discover-signature")
async def discover_signature(request_model: SignatureSearchRequest, request: Request) -> dict[str, Any]:
    try:
        minimum_combinations = request_model.minimum_combinations()
        if minimum_combinations > _MAX_SIGNATURE_COMBINATIONS:
            raise UserInputError(
                f"This search would test at least {minimum_combinations:,} feature combinations; SurvStudio searches at most "
                f"{_MAX_SIGNATURE_COMBINATIONS:,}. Choose fewer candidate columns or a smaller maximum combination size."
            )
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                request_model.candidate_columns,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="signature discovery candidates",
            )
            _reject_outcome_informed_columns(
                stored,
                request_model.candidate_columns,
                context="signature discovery candidates",
            )
            updated, column_name, analysis = discover_feature_signature(
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
            provenance = dict(stored.metadata.get("derived_column_provenance", {}))
            signature_recipe = copy.deepcopy(
                analysis.get("derived_group", {}).get("recipe")
                or analysis.get("signature_recipe", {})
            )
            provenance[column_name] = {
                "outcome_informed": True,
                "recipe": signature_recipe,
                "statistically_significant": bool(analysis.get("best_split", {}).get("Statistically significant")),
            }
            # Snapshot hashing/profiling is CPU-bound, so it stays in the worker thread too.
            payload = _create_dataset_snapshot(stored, updated, metadata={
                **stored.metadata,
                "derived_column_provenance": provenance,
            })
            payload["derived_column"] = column_name
            payload["signature_analysis"] = analysis
            payload["signature_request_config"] = request_config
            # dataset_response() already carries the new snapshot's dataset_hash; do not overwrite it
            # with the parent dataset's hash.
            return payload

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


# ── ML model endpoints ──────────────────────────────────────────


@app.post("/api/optimal-cutpoint")
async def optimal_cutpoint(request_model: OptimalCutpointRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import find_optimal_cutpoint
        from survival_toolkit.plots import build_cutpoint_scan_figure

        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                [request_model.variable],
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="optimal cutpoint searches",
            )
            _reject_outcome_informed_columns(stored, [request_model.variable], context="optimal cutpoint searches")
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
            return _attach_dataset_hash({"result": result, "figure": figure, "request_config": request_config}, stored)

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/ml-model")
async def ml_model(request_model: MLModelRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import (
            train_random_survival_forest,
            train_gradient_boosted_survival,
            train_lasso_cox,
            compare_survival_models,
            cross_validate_survival_models,
            compute_shap_values,
        )
        from survival_toolkit.plots import (
            build_feature_importance_figure,
            build_model_comparison_figure,
            build_shap_figure,
        )

        stored = _get_stored_dataset(request_model.dataset_id)
        df = stored.dataframe
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                [*request_model.features, *request_model.categorical_features],
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="machine-learning model inputs",
            )
            _reject_outcome_informed_columns(
                stored,
                [*request_model.features, *request_model.categorical_features],
                context="machine-learning model inputs",
            )
            _reject_oversized_design(
                stored,
                request_model.features,
                request_model.categorical_features,
                context="machine-learning model inputs",
            )
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
                        locked_test_fraction=request_model.locked_test_fraction,
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
                _attach_manuscript_notes(
                    comparison,
                    _ml_replay_notes(request_config, dataset_filename=stored.filename),
                )
                figure = build_model_comparison_figure(comparison)
                return _attach_dataset_hash(
                    {"analysis": comparison, "figure": figure, "request_config": request_config},
                    stored,
                )

            if request_model.evaluation_strategy != "holdout":
                raise UserInputError(
                    "Train Model currently supports deterministic holdout only. "
                    "Use Compare All for repeated cross-validation screening."
                )

            def _train_single_ml_model(
                *,
                features: Sequence[str],
                categorical_features: Sequence[str],
            ) -> dict[str, Any]:
                common_ml = dict(
                    df=df,
                    time_column=request_model.time_column,
                    event_column=request_model.event_column,
                    features=list(features),
                    categorical_features=list(categorical_features),
                    event_positive_value=request_model.event_positive_value,
                    random_state=request_model.random_state,
                )
                if request_model.model_type == "gbs":
                    common_ml["n_estimators"] = request_model.n_estimators
                    common_ml["max_depth"] = request_model.max_depth
                    common_ml["learning_rate"] = request_model.learning_rate
                    return train_gradient_boosted_survival(**common_ml)
                if request_model.model_type == "lasso_cox":
                    return train_lasso_cox(**common_ml)
                common_ml["n_estimators"] = request_model.n_estimators
                common_ml["max_depth"] = request_model.max_depth
                return train_random_survival_forest(**common_ml)

            result = _train_single_ml_model(
                features=request_model.features,
                categorical_features=request_model.categorical_features,
            )
            _remember_ml_artifact(request_model.dataset_id, request_config, result)
            model_label = {
                "rsf": "Random Survival Forest",
                "gbs": "Gradient Boosted Survival",
                "lasso_cox": "LASSO-Cox",
            }.get(request_model.model_type, request_model.model_type.upper())
            importance_figure = build_feature_importance_figure(
                result["feature_importance"],
                model_name=model_label,
            )

            shap_result = None
            shap_figure = None
            shap_error = None
            shap_companion = None
            model_obj = result.get("_model")
            x_encoded = result.get("_X_eval_encoded")
            if x_encoded is None:
                x_encoded = result.get("_X_encoded")
            shap_supported = request_model.model_type in {"rsf", "gbs"}
            if request_model.compute_shap and not shap_supported:
                shap_error = "SHAP is currently available for tree models only (RSF or GBS)."
            elif request_model.compute_shap and model_obj is not None and x_encoded is not None:
                try:
                    shap_result = compute_shap_values(model_obj, x_encoded, result["feature_names"])
                    shap_figure = build_shap_figure(shap_result)
                except (MemoryError, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    # A cancelled request or a coding error is not a SHAP failure to report next to the model.
                    if _must_propagate_through_boundary(exc):
                        raise
                    logger.warning("SHAP explanation failed; the model is reported without it", exc_info=exc)
                    shap_error = f"{type(exc).__name__}: {exc}"
                    if request_model.shap_safe_mode and "high-dimensional inputs" in str(exc).lower():
                        subset = _select_shap_safe_mode_subset(result, request_model.features)
                        if subset is not None:
                            try:
                                companion_features = list(subset["selected_features"])
                                companion_categoricals = [
                                    feature for feature in request_model.categorical_features
                                    if feature in companion_features
                                ]
                                companion_result = _train_single_ml_model(
                                    features=companion_features,
                                    categorical_features=companion_categoricals,
                                )
                                companion_model = companion_result.get("_model")
                                companion_x = companion_result.get("_X_eval_encoded")
                                if companion_x is None:
                                    companion_x = companion_result.get("_X_encoded")
                                if companion_model is None or companion_x is None:
                                    raise ValueError("SHAP safe mode companion model did not return an encoded evaluation matrix.")
                                shap_result = compute_shap_values(
                                    companion_model,
                                    companion_x,
                                    companion_result["feature_names"],
                                )
                                companion_note = (
                                    f"SHAP safe mode refit a reduced companion {model_label} model on "
                                    f"{subset['selected_feature_count_raw']} raw features "
                                    f"({subset['selected_feature_count_encoded']} encoded) selected from the full fit's "
                                    "built-in importance ranking."
                                )
                                usage_note = str(shap_result.get("usage_note") or "").strip()
                                shap_result["usage_note"] = f"{usage_note} {companion_note}".strip()
                                shap_result["safe_mode"] = True
                                shap_result["safe_mode_reason"] = "high_dimensional_encoded_matrix"
                                shap_result["companion_model"] = {
                                    "selection_basis": "full_model_importance",
                                    "requested_feature_count_raw": int(subset["requested_feature_count_raw"]),
                                    "selected_feature_count_raw": int(subset["selected_feature_count_raw"]),
                                    "selected_feature_count_encoded": int(subset["selected_feature_count_encoded"]),
                                    "selected_features": companion_features,
                                    "omitted_features": list(subset["omitted_features"]),
                                    "encoded_feature_limit": int(_SHAP_SAFE_MODE_MAX_ENCODED_FEATURES),
                                }
                                shap_companion = shap_result["companion_model"]
                                shap_figure = build_shap_figure(shap_result)
                                shap_error = None
                            except (MemoryError, KeyboardInterrupt):
                                raise
                            except Exception as safe_mode_exc:
                                if _must_propagate_through_boundary(safe_mode_exc):
                                    raise
                                logger.warning("SHAP safe mode failed as well", exc_info=safe_mode_exc)
                                shap_error = (
                                    f"{type(exc).__name__}: {exc} "
                                    f"SHAP safe mode also failed: {type(safe_mode_exc).__name__}: {safe_mode_exc}"
                                )

            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            return _attach_dataset_hash(
                {
                    "analysis": clean_result,
                    "importance_figure": importance_figure,
                    "shap_result": shap_result,
                    "shap_figure": shap_figure,
                    "shap_error": shap_error,
                    "shap_companion": shap_companion,
                    "request_config": request_config,
                },
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/deep-model")
async def deep_model(request_model: DeepModelRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit import deep_models
        from survival_toolkit.plots import (
            build_feature_importance_figure,
            build_loss_curve_figure,
            build_model_comparison_figure,
        )

        stored = _get_stored_dataset(request_model.dataset_id)
        df = stored.dataframe
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                [*request_model.features, *request_model.categorical_features],
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="deep-learning model inputs",
            )
            _reject_outcome_informed_columns(
                stored,
                [*request_model.features, *request_model.categorical_features],
                context="deep-learning model inputs",
            )
            _reject_oversized_design(
                stored,
                request_model.features,
                request_model.categorical_features,
                context="deep-learning model inputs",
            )
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
                    locked_test_fraction=(
                        request_model.locked_test_fraction
                        if request_model.evaluation_strategy == "repeated_cv"
                        else None
                    ),
                )
                _attach_manuscript_notes(
                    result,
                    _dl_replay_notes(request_config, dataset_filename=stored.filename, resolved_analysis=result),
                )
                clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
                return _attach_dataset_hash(
                    {
                        "analysis": clean_result,
                        "figures": {"comparison": build_model_comparison_figure(result)},
                        "request_config": request_config,
                    },
                    stored,
                )

            result = deep_models.evaluate_single_deep_survival_model(
                request_model.model_type,
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
                locked_test_fraction=(
                    request_model.locked_test_fraction
                    if request_model.evaluation_strategy == "repeated_cv"
                    else None
                ),
            )

            figures = {}
            if result.get("feature_importance"):
                figures["importance"] = build_feature_importance_figure(
                    result["feature_importance"],
                    model_name=request_model.model_type.upper(),
                    title_label="Gradient-Based Feature Salience",
                )
            if result.get("loss_history"):
                figures["loss"] = build_loss_curve_figure(
                    result["loss_history"],
                    model_name=request_model.model_type.upper(),
                    monitor_loss_history=result.get("monitor_history", result.get("monitor_loss_history")),
                    best_monitor_epoch=result.get("best_monitor_epoch"),
                    epochs_trained=result.get("early_stopping_epochs", result.get("epochs_trained")),
                    max_epochs_requested=result.get("max_epochs_requested"),
                    stopped_early=result.get("stopped_early"),
                    monitor_label=str(result.get("monitor_metric_label", "Monitor loss")),
                    monitor_goal=str(result.get("monitor_metric_goal", "min")),
                )

            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            if "scientific_summary" not in clean_result and "insight_board" in clean_result:
                clean_result["scientific_summary"] = clean_result["insight_board"]
            if "epochs_trained" not in clean_result:
                clean_result["epochs_trained"] = request_model.epochs
            return _attach_dataset_hash(
                {"analysis": clean_result, "figures": figures, "request_config": request_config},
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


# ── XAI endpoints ──────────────────────────────────────────────


class TimeDependentImportanceRequest(_FeatureSelectionRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    eval_times: list[float] | None = Field(default=None, max_length=100)
    # The survival model whose reliance on each feature is measured over time.
    model_type: Literal["rsf", "gbs"] = "rsf"
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)


class CounterfactualRequest(_FeatureSelectionRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    target_feature: str
    original_value: Any = None
    counterfactual_value: Any
    model_type: Literal["rsf", "gbs"] = "rsf"
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("original_value", "counterfactual_value", mode="before")
    @classmethod
    def validate_feature_value(cls, value: Any, info: Any) -> Any:
        # One value of the target feature: a number, a category label, a boolean, or null.
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{info.field_name} must be a finite number.")
            return value
        if isinstance(value, str):
            if len(value) > 1000:
                raise ValueError(f"{info.field_name} must be 1000 characters or fewer.")
            return value
        raise ValueError(f"{info.field_name} must be a single number, text label, boolean, or null.")

    @model_validator(mode="after")
    def validate_target_feature_membership(self) -> "CounterfactualRequest":
        _validate_subset_names(
            [self.target_feature],
            self.features,
            subset_name="target_feature",
            superset_name="features",
        )
        return self


class PDPRequest(_FeatureSelectionRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    target_feature: str
    model_type: Literal["rsf", "gbs"] = "rsf"
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_target_feature_membership(self) -> "PDPRequest":
        _validate_subset_names(
            [self.target_feature],
            self.features,
            subset_name="target_feature",
            superset_name="features",
        )
        return self


@app.post("/api/time-dependent-importance")
async def time_dependent_importance(request_model: TimeDependentImportanceRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import compute_time_dependent_importance
        from survival_toolkit.plots import build_time_dependent_importance_figure

        stored = _get_stored_dataset(request_model.dataset_id)

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                [*request_model.features, *request_model.categorical_features],
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="time-dependent importance inputs",
            )
            _reject_outcome_informed_columns(
                stored,
                [*request_model.features, *request_model.categorical_features],
                context="time-dependent importance inputs",
            )
            _reject_oversized_design(
                stored,
                request_model.features,
                request_model.categorical_features,
                context="time-dependent importance inputs",
            )
            result = compute_time_dependent_importance(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
                eval_times=request_model.eval_times,
                model_type=request_model.model_type,
                n_estimators=request_model.n_estimators,
                max_depth=request_model.max_depth,
                learning_rate=request_model.learning_rate,
                random_state=request_model.random_state,
            )
            figure = build_time_dependent_importance_figure(result)
            return _attach_dataset_hash({"analysis": result, "figure": figure}, stored)

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/counterfactual")
async def counterfactual(request_model: CounterfactualRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import counterfactual_survival

        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                [*request_model.features, *request_model.categorical_features, request_model.target_feature],
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="counterfactual analysis",
            )
            _reject_outcome_informed_columns(
                stored,
                [*request_model.features, *request_model.categorical_features, request_model.target_feature],
                context="counterfactual analysis",
            )
            _reject_oversized_design(
                stored,
                request_model.features,
                request_model.categorical_features,
                context="counterfactual model inputs",
            )
            artifact = _get_ml_artifact(request_model.dataset_id, request_config)
            analysis = counterfactual_survival(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                features=request_model.features,
                categorical_features=request_model.categorical_features,
                target_feature=request_model.target_feature,
                original_value=request_model.original_value,
                counterfactual_value=request_model.counterfactual_value,
                model_type=request_model.model_type,
                n_estimators=request_model.n_estimators,
                max_depth=request_model.max_depth,
                learning_rate=request_model.learning_rate,
                random_state=request_model.random_state,
                trained_result=artifact,
            )
            analysis["artifact_reused"] = artifact is not None
            analysis["explanation_scope"] = (
                "trained_tree_model" if artifact is not None else "refit_tree_model"
            )
            analysis["contract_note"] = (
                "Counterfactual analysis reused the exact fitted RSF/GBS model from the latest matching single-model run. It does not explain a Compare All ranking table."
                if artifact is not None
                else "Counterfactual analysis refit an RSF/GBS model from the requested configuration because no matching single-model artifact was cached. It does not explain a previously displayed Compare All winner."
            )
            return _attach_dataset_hash({"analysis": analysis, "request_config": request_config}, stored)

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/pdp")
async def pdp(request_model: PDPRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import (
            compute_partial_dependence,
            train_gradient_boosted_survival,
            train_random_survival_forest,
        )
        from survival_toolkit.plots import build_pdp_figure

        stored = _get_stored_dataset(request_model.dataset_id)
        df = stored.dataframe
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                [*request_model.features, *request_model.categorical_features, request_model.target_feature],
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="partial dependence analysis",
            )
            _reject_outcome_informed_columns(
                stored,
                [*request_model.features, *request_model.categorical_features, request_model.target_feature],
                context="partial dependence analysis",
            )
            _reject_oversized_design(
                stored,
                request_model.features,
                request_model.categorical_features,
                context="partial dependence model inputs",
            )
            trained = _get_ml_artifact(request_model.dataset_id, request_config)
            artifact_reused = trained is not None
            if trained is None:
                common_kwargs = dict(
                    df=df,
                    time_column=request_model.time_column,
                    event_column=request_model.event_column,
                    event_positive_value=request_model.event_positive_value,
                    features=request_model.features,
                    categorical_features=request_model.categorical_features,
                    n_estimators=request_model.n_estimators,
                    max_depth=request_model.max_depth,
                    random_state=request_model.random_state,
                    # Partial dependence needs only the fitted model; this refit is not cached for reuse.
                    compute_importance=False,
                    compute_brier=False,
                )
                if request_model.model_type == "gbs":
                    trained = train_gradient_boosted_survival(
                        **common_kwargs,
                        learning_rate=request_model.learning_rate,
                    )
                else:
                    trained = train_random_survival_forest(**common_kwargs)
            model = trained["_model"]
            X_encoded = trained["_X_encoded"]
            feature_encoder = trained.get("_feature_encoder")
            analysis_frame = trained.get("_analysis_frame")

            result = compute_partial_dependence(
                model,
                X_encoded,
                feature_name=request_model.target_feature,
                categorical_features=request_model.categorical_features,
                feature_encoder=feature_encoder,
                analysis_frame=analysis_frame,
            )
            result["model_type"] = request_model.model_type
            result["artifact_reused"] = artifact_reused
            result["explanation_scope"] = (
                "trained_tree_model" if artifact_reused else "refit_tree_model"
            )
            result["contract_note"] = (
                "Partial dependence reused the exact fitted RSF/GBS model from the latest matching single-model run. It does not explain a Compare All ranking table."
                if artifact_reused
                else "Partial dependence refit an RSF/GBS model from the requested configuration because no matching single-model artifact was cached. It does not explain a previously displayed Compare All winner."
            )
            figure = build_pdp_figure(result)
            return _attach_dataset_hash(
                {"analysis": result, "figure": figure, "request_config": request_config},
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


# ── Marker evaluation and design audit endpoints ────────────────


_ID_COLUMN_NAME = re.compile(r"(^|[^a-z])(id|patient|sample|subject|case|barcode)([^a-z]|$)", re.IGNORECASE)


def _likely_id_column(table: pd.DataFrame) -> str | None:
    """A column that names patients (unique values under a name such as patient_id), for messages about them."""
    for column in table.columns:
        if _ID_COLUMN_NAME.search(str(column)) and table[column].notna().all() and table[column].is_unique:
            return str(column)
    return None


def _marker_display_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The marker table as flat rows on the primary lens, for display and export."""
    primary = result["primary_lens"]
    exact_lens = "adjusted" if primary == "added_value" else "marginal"
    rows: list[dict[str, Any]] = []
    for row in result["marker_table"]:
        stats = row[primary]
        exact = (row.get("exact") or {}).get(exact_lens) or {}
        low, high = stats["rank_interval"]
        display = {
            "Marker": row["marker"],
            "Tier": row["tier"],
            "Evidence": row["pattern"],
            "Direction": row["direction"],
            "HR per unit": exact.get("hazard_ratio"),
            "CI lower": exact.get("ci_lower"),
            "CI upper": exact.get("ci_upper"),
            "P value": stats["p_value"],
            "Family-wise P": stats["p_fwer"],
            "Permutation q value": stats["q_perm"],
            "Selection frequency": stats["selection_frequency"],
            "Direction consistency": stats["direction_consistency"],
            "Median rank": stats["median_rank"],
            # "1 to 3", not "1-3", which a spreadsheet opens as a date.
            "Rank 95% interval": None if low is None or high is None else f"{low:.0f} to {high:.0f}",
        }
        if primary == "added_value":
            display["LR test P"] = exact.get("lr_p")
            display["Apparent C gain"] = exact.get("delta_c_apparent")
            display["Unadjusted HR"] = ((row.get("exact") or {}).get("marginal") or {}).get("hazard_ratio")
            display["Unadjusted P"] = row["marginal"]["p_value"]
        rows.append(display)
    return rows


def _reject_repeated_column_roles(roles: Sequence[tuple[str, Sequence[str]]], *, analysis: str) -> None:
    """Each column may have one role in ``analysis``: time, event, marker, clinical covariate, stratum or patient ID."""

    first_role: dict[str, str] = {}
    clashes: list[str] = []
    for role, columns in roles:
        for column in columns:
            name = str(column)
            earlier = first_role.setdefault(name, role)
            if earlier != role:
                clashes.append(f"'{name}' ({earlier} and {role})")
    if clashes:
        raise UserInputError(
            f"Each column can have only one role in {analysis}; these were given more than one: "
            + ", ".join(clashes[:5])
            + (" ..." if len(clashes) > 5 else "")
            + "."
        )


def _reject_repeated_marker_roles(request_model: MarkerEvaluationRequest) -> None:
    """Each column may have one role: time, event, marker, clinical covariate, stratum or patient ID."""

    roles: list[tuple[str, Sequence[str]]] = [
        ("outcome time", [request_model.time_column]),
        ("outcome event", [request_model.event_column]),
        ("marker", request_model.marker_columns),
        ("clinical covariate", request_model.clinical_columns),
        ("stratum", request_model.strata_columns),
    ]
    if request_model.marker_matrix_id and request_model.marker_matrix_id_column:
        roles.append(("patient ID", [request_model.marker_matrix_id_column]))
    _reject_repeated_column_roles(roles, analysis="a marker evaluation")


def _reject_leaky_validation_columns(
    stored: Any,
    recipe: dict[str, Any],
    column_mapping: dict[str, str],
    event_positive_value: Any,
) -> None:
    """Apply the marker-evaluation role and outcome rules to the external columns a locked recipe will read.

    ``column_mapping`` can point a locked marker or covariate at any external column, including
    the outcome, a copy of it, or an outcome-informed derived column; those would validate the
    model on the outcome itself. A recipe without the fields read here is left to the validation,
    which reports what is missing.
    """

    outcome = recipe.get("outcome")
    clinical = recipe.get("clinical")
    markers = recipe.get("markers")
    strata = recipe.get("strata_columns") or []
    if not (isinstance(outcome, dict) and isinstance(clinical, dict) and isinstance(markers, list) and isinstance(strata, list)):
        return
    clinical_columns = clinical.get("columns") or []
    names = [outcome.get("time_column"), outcome.get("event_column"), *markers, *clinical_columns, *strata]
    if not isinstance(clinical_columns, list) or not all(isinstance(name, str) for name in names):
        return

    def external(name: str) -> str:
        return str(column_mapping.get(name, name))

    time_column = external(outcome["time_column"])
    event_column = external(outcome["event_column"])
    # One role per recipe column, so two locked columns mapped onto one external column clash too.
    inputs: list[tuple[str, list[str]]] = [
        *[(f"marker {name}", [external(name)]) for name in markers],
        *[(f"clinical covariate {name}", [external(name)]) for name in clinical_columns],
        *[(f"stratum {name}", [external(name)]) for name in strata],
    ]
    _reject_repeated_column_roles(
        [("outcome time", [time_column]), ("outcome event", [event_column]), *inputs],
        analysis="a marker validation",
    )
    external_inputs = [column for _role, columns in inputs for column in columns]
    _reject_survival_outcome_feature_columns(
        stored,
        external_inputs,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=outcome.get("event_positive_value") if event_positive_value is None else event_positive_value,
        context="marker validation",
    )
    _reject_outcome_informed_columns(stored, external_inputs, context="marker validation")


def _reject_missing_marker_columns(stored: Any, columns: Sequence[str]) -> None:
    present = set(stored.dataframe.columns)
    missing = [column for column in dict.fromkeys(str(column) for column in columns) if column not in present]
    if missing:
        raise UserInputError(
            "Columns not found in the dataset: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "") + "."
        )


@app.post("/api/marker-evaluation")
async def marker_evaluation(request_model: MarkerEvaluationRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.plots import build_marker_rank_figure, build_marker_stability_figure, build_marker_summary_figure

        _reject_repeated_marker_roles(request_model)
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        inputs = [*request_model.marker_columns, *request_model.clinical_columns, *request_model.strata_columns]
        matrix = marker_matrices.get(request_model.marker_matrix_id) if request_model.marker_matrix_id else None

        def _run() -> dict[str, Any]:
            _reject_missing_marker_columns(stored, [request_model.time_column, request_model.event_column, *inputs])
            # Input checks scan every column, so they run in the worker thread too.
            _reject_survival_outcome_feature_columns(
                stored,
                inputs,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                context="marker evaluation",
            )
            _reject_outcome_informed_columns(stored, inputs, context="marker evaluation")
            # Every marker model carries all clinical covariates.
            _reject_oversized_design(
                stored,
                request_model.clinical_columns,
                request_model.categorical_clinical,
                context="clinical covariates",
            )
            frame = stored.dataframe
            markers = list(request_model.marker_columns)
            matrix_info = None
            # Names patients in the duplicate screen only; it is never a marker or covariate.
            id_column = _likely_id_column(stored.dataframe)
            if matrix is not None:
                id_column = str(request_model.marker_matrix_id_column)
                frame = matrix_frame(
                    stored.dataframe,
                    matrix,
                    id_column=id_column,
                    columns=[id_column, request_model.time_column, request_model.event_column, *request_model.clinical_columns, *request_model.strata_columns],
                )
                markers = list(matrix.marker_names)
                # The same outcome-leakage rules as for dataset columns, applied to the matrix markers.
                leaked = _matrix_outcome_markers(
                    frame,
                    markers,
                    event_column=request_model.event_column,
                    event_positive_value=request_model.event_positive_value,
                )
                if leaked:
                    raise UserInputError(
                        "The marker matrix holds survival outcome columns, which cannot be used as markers: "
                        + ", ".join(leaked[:10])
                        + (" ..." if len(leaked) > 10 else "")
                        + ". Remove them from the matrix file and attach it again."
                    )
                matrix_info = {
                    "filename": matrix.filename,
                    "n_markers": len(markers),
                    "n_matched": int(frame.shape[0]),
                    "id_column": request_model.marker_matrix_id_column,
                    "fingerprint": matrix.fingerprint,
                }
            result = evaluate_markers(
                frame,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                marker_columns=markers,
                clinical_columns=request_model.clinical_columns,
                categorical_clinical=request_model.categorical_clinical,
                strata_columns=request_model.strata_columns,
                event_positive_value=request_model.event_positive_value,
                settings=request_model.marker_settings(),
                id_column=id_column,
            )
            report_dataset = {**_report_dataset(stored), "marker_matrix": matrix_info} if matrix_info else _report_dataset(stored)
            payload = {
                "analysis": _trim_marker_table(result),
                "display_table": _marker_display_rows(result),
                # Figures read the full marker table; the payload carries a trimmed one.
                "summary_figure": build_marker_summary_figure(result),
                "stability_figure": build_marker_stability_figure(result),
                "rank_figure": build_marker_rank_figure(result),
                "report": remark_checklist(result, request={**request_config, "marker_columns": markers}, dataset=report_dataset),
                "request_config": request_config,
            }
            if matrix_info:
                payload["marker_matrix"] = matrix_info
            return _attach_dataset_hash(payload, stored)

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/marker-validation")
async def marker_validation(request_model: MarkerValidationRequest, request: Request) -> dict[str, Any]:
    try:
        from survival_toolkit.plots import build_marker_replication_figure

        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump(exclude={"recipe"})

        def _run() -> dict[str, Any]:
            recipe = request_model.recipe
            # Input checks scan every column, so they run in the worker thread too.
            _reject_leaky_validation_columns(stored, recipe, request_model.column_mapping, request_model.event_positive_value)
            # A stratified recipe is scored without bootstrap draws; the others share the interval work budget.
            if recipe.get("strata_columns"):
                n_bootstrap, bootstrap_note = int(request_model.n_bootstrap), None
            else:
                n_bootstrap, bootstrap_note = _validation_bootstrap_draws(
                    int(stored.dataframe.shape[0]),
                    2 if recipe.get("clinical_only_model") else 1,
                    request_model.n_bootstrap,
                )
            try:
                validation = validate_locked_recipe(
                    stored.dataframe,
                    recipe,
                    column_mapping=request_model.column_mapping,
                    event_positive_value=request_model.event_positive_value,
                    horizon=request_model.horizon,
                    alpha=request_model.alpha,
                    n_bootstrap=n_bootstrap,
                    random_seed=request_model.random_seed,
                    marker_scaling=request_model.marker_scaling,
                )
            except (KeyError, AttributeError, IndexError) as exc:
                # The recipe is client JSON guarded only by a content hash, which a client can recompute, so a
                # missing or mistyped field surfaces as a lookup error in SurvStudio's own code. The same errors
                # raised inside a library are coding errors and reach the error handler (a logged 500).
                if not _raised_by_survstudio(exc):
                    raise
                logger.warning("Rejected an incomplete or malformed marker recipe", exc_info=exc)
                if isinstance(exc, KeyError):
                    raise UserInputError("The recipe is incomplete; export it again from a marker evaluation.") from exc
                raise UserInputError("The recipe is malformed; export it again from a marker evaluation.") from exc
            if bootstrap_note and isinstance(validation.get("notes"), list):
                validation["notes"].append(bootstrap_note)
            return _attach_dataset_hash(
                {
                    "validation": validation,
                    "figure": build_marker_replication_figure(validation),
                    "request_config": request_config,
                },
                stored,
            )

        return await _run_dataset_job(request_model.dataset_id, _run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


_MARKER_TABLE_RESPONSE_ROWS = 2_000


def _trim_marker_table(result: dict[str, Any]) -> dict[str, Any]:
    """Send only the strongest rows of a very long per-marker table in the JSON response.

    The flat display table still lists every marker; the nested per-lens rows add about a
    kilobyte per marker, which a genome-wide screen would turn into tens of megabytes.
    """
    rows = result.get("marker_table") or []
    if len(rows) <= _MARKER_TABLE_RESPONSE_ROWS:
        return result
    return {**result, "marker_table": rows[:_MARKER_TABLE_RESPONSE_ROWS], "marker_table_rows_total": len(rows)}


def _report_dataset(stored: Any) -> dict[str, Any]:
    return {
        "filename": stored.filename,
        "n_rows": int(stored.dataframe.shape[0]),
        "dataset_hash": str(stored.metadata.get("dataset_hash") or ""),
    }


@app.post("/api/tripod-ai-checklist")
async def tripod_ai_report(request_model: TripodChecklistRequest) -> dict[str, Any]:
    try:
        dataset = None
        if request_model.dataset_id:
            try:
                dataset = _report_dataset(_get_stored_dataset(request_model.dataset_id))
            except NotFoundError:
                dataset = None
        comparisons = [
            {**item.analysis, "family": item.family, "request_config": item.request_config}
            for item in request_model.comparisons
        ]
        return await _run_job(lambda: tripod_ai_checklist(comparisons, dataset=dataset))
    except Exception as exc:
        fail_bad_request(exc)


def _checklist_docx(report: dict[str, Any]) -> bytes:
    rows = checklist_rows(report)
    return _docx_package(
        [
            _docx_paragraph(f"{report['guideline']} checklist", bold=True),
            _docx_paragraph(checklist_intro(report), italic=True),
            _docx_paragraph("Methods", bold=True),
            _docx_paragraph(report.get("methods", "")),
            _docx_paragraph("Results", bold=True),
            _docx_paragraph(report.get("results", "")),
            _docx_paragraph("Checklist", bold=True),
            # Item, Section, Topic, Status, Text across a 6.5-inch text width.
            _docx_table(rows, style="plain", columns=list(CHECKLIST_COLUMNS), widths=[600, 1300, 1900, 1300, 4260]),
        ]
    )


@app.post("/api/checklist-export")
async def checklist_export(request_model: ChecklistExportRequest) -> Response:
    try:
        report = request_model.model_dump(exclude={"format"})
        if request_model.format == "markdown":
            return Response(content=checklist_markdown(report), media_type="text/markdown; charset=utf-8")
        content = await run_in_threadpool(_checklist_docx, report)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as exc:
        fail_bad_request(exc)


def _interval_draws(n_patients: int, n_events: int, n_models: int, requested: int) -> tuple[int, str | None]:
    """Bootstrap draws that fit the work budget, and a note when that is fewer than requested."""

    affordable = _INTERVAL_WORK_BUDGET // _c_index_draw_work(n_patients, n_events, n_models)
    if affordable < _INTERVAL_MIN_DRAWS:
        raise UserInputError(
            f"The shared test set ({n_patients:,} patients, {n_events:,} events, {n_models} models) is too large for "
            f"bootstrap intervals: only {affordable} draws fit the work budget and at least {_INTERVAL_MIN_DRAWS} are needed. "
            "Compare fewer models at once or report the point estimates."
        )
    draws = min(int(requested), int(affordable))
    if draws >= requested:
        return draws, None
    return draws, (
        f"Bootstrap draws were limited to {draws} of the {requested} requested, so the intervals for "
        f"{n_patients:,} test patients ({n_events:,} events, {n_models} models) stay within the work budget."
    )


def _validation_bootstrap_draws(n_rows: int, n_models: int, requested: int) -> tuple[int, str | None]:
    """Bootstrap draws of a locked-model validation that fit the interval work budget, and a note when fewer run.

    The analysable rows and events of the external cohort are known only once the recipe is
    applied, so every row counts as a patient with an event (an upper bound on the work). When not
    even the minimum number of draws fits, the validation runs without intervals.
    """

    requested = int(requested)
    if requested <= 0:
        return 0, None
    affordable = int(_INTERVAL_WORK_BUDGET // _c_index_draw_work(n_rows, n_rows, n_models))
    if affordable >= requested:
        return requested, None
    if affordable < _INTERVAL_MIN_DRAWS:
        return 0, (
            f"The bootstrap intervals were not computed: {n_rows:,} patients need more work than the budget allows for "
            f"{_INTERVAL_MIN_DRAWS} draws. The point estimates are reported."
        )
    return affordable, (
        f"Bootstrap draws were limited to {affordable} of the {requested} requested, so the intervals for "
        f"{n_rows:,} patients stay within the work budget."
    )


@app.post("/api/model-comparison-intervals")
async def model_comparison_intervals(request_model: ModelComparisonIntervalsRequest, request: Request) -> dict[str, Any]:
    try:

        def _run() -> dict[str, Any]:
            # Deliberate messages ("the comparisons share no test patients") reach the user as 400s.
            time, event, risks, rows = user_input_boundary(merge_prediction_blocks)(request_model.predictions)
            draws, note = _interval_draws(int(time.shape[0]), int(event.sum()), len(risks), int(request_model.n_bootstrap))
            result = user_input_boundary(c_index_intervals)(
                time,
                event,
                risks,
                reference=request_model.reference,
                n_bootstrap=draws,
                random_seed=request_model.random_seed,
            )
            payload = {**result, "n_shared": len(rows), "n_bootstrap_requested": int(request_model.n_bootstrap)}
            if note:
                payload["bootstrap_note"] = note
            if request_model.reference is not None and result.get("reference") is None:
                compared = [str(row.get("model")) for row in result.get("rows", [])]
                payload["reference_note"] = (
                    f'No paired differences were computed: the reference model "{request_model.reference}" is not among '
                    f"the compared models ({', '.join(compared[:10])}{', ...' if len(compared) > 10 else ''})."
                )
            return payload

        return await _run_job(_run, request=request, heavy=True)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/design-audit")
async def design_audit(request_model: DesignAuditRequest) -> dict[str, Any]:
    try:
        request_config = request_model.model_dump()
        design = design_from_dict(request_config)
        result = await _run_job(lambda: audit_design(design))
        return {**result, "request_config": request_config}
    except Exception as exc:
        fail_bad_request(exc)

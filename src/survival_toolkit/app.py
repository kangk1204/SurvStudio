from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import platform
import re
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import signal
import tempfile
import threading
import time
import zipfile
from typing import Any, Callable, Literal, NoReturn, Sequence
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from survival_toolkit.analysis import (
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
from survival_toolkit.errors import NotFoundError, UserInputError
from survival_toolkit.sample_data import (
    load_gbsg2_upload_ready_dataset,
    load_tcga_luad_example_dataset,
    load_tcga_luad_upload_ready_dataset,
    make_example_dataset,
)
from survival_toolkit.store import DatasetStore

BASE_DIR = Path(__file__).resolve().parent


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

app = FastAPI(
    title="SurvStudio",
    description="Local survival analysis dashboard for exploratory and validation-oriented cohort work.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
store = DatasetStore()
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
_MAX_UPLOAD_ROWS = 100_000
_MAX_UPLOAD_COLUMNS = 5_000
_MAX_UPLOAD_CELLS = 5_000_000
_SHAP_SAFE_MODE_MAX_ENCODED_FEATURES = 80
_SHAP_SAFE_MODE_MAX_RAW_FEATURES = 30
_SIGNED_NUMERIC_CSV_LITERAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_NOTE_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DATASET_PROFILE_CACHE_KEY = "_dataset_profile_cache"
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
    }
)


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
    return text


def _normalize_event_positive_value(value: Any) -> Any:
    if value is None or value == "":
        return value
    if isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            text = _normalize_optional_text_field(value, field_name="event_positive_value")
            return text
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("event_positive_value must be a finite scalar value.")
        return value
    raise ValueError("event_positive_value must be a scalar JSON value (string, number, boolean, or null).")


class _DatasetRequestModel(BaseModel):
    @field_validator("dataset_id", mode="before", check_fields=False)
    @classmethod
    def validate_dataset_id(cls, value: Any) -> str:
        text = _normalize_optional_text_field(value, field_name="dataset_id")
        if text is None:
            raise ValueError("dataset_id must not be empty or null.")
        if len(text) > 64:
            raise ValueError("dataset_id must be 64 characters or fewer.")
        return text


class _EventPositiveValueRequestModel(_DatasetRequestModel):
    @field_validator("event_positive_value", mode="before", check_fields=False)
    @classmethod
    def validate_event_positive_value(cls, value: Any) -> Any:
        return _normalize_event_positive_value(value)


class _MlArtifactCache:
    def __init__(self, *, max_items: int) -> None:
        self._max_items = int(max_items)
        self._items: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
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
        artifact = {
            "signature": copy.deepcopy(signature),
            "result": self._copy_result(result),
        }
        with self._lock:
            cache_key = (dataset_id, model_type)
            self._items[cache_key] = artifact
            self._items.move_to_end(cache_key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

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


_ml_artifact_cache = _MlArtifactCache(max_items=8)


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
    event_positive_value: Any = None
    min_group_fraction: float = Field(default=0.1, gt=0.02, lt=0.45)
    permutation_iterations: int = Field(default=500, ge=0, le=500)
    random_seed: int = 20260311

    @field_validator("new_column_name", mode="before")
    @classmethod
    def validate_new_column_name(cls, value: Any) -> str | None:
        return _normalize_optional_text_field(
            value,
            field_name="Derived column names",
            allow_empty_as_none=True,
        )

    @field_validator("lower_label", "upper_label", mode="before")
    @classmethod
    def validate_group_label(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Group labels must not be empty.")
        if _CONTROL_CHAR_PATTERN.search(text):
            raise ValueError("Group labels must not contain control characters.")
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


class CohortTableRequest(_DatasetRequestModel):
    dataset_id: str
    variables: list[str] = Field(max_length=200)
    group_column: str | None = None


class SignatureSearchRequest(_EventPositiveValueRequestModel):
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
    new_column_name: str | None = Field(default=None, max_length=200)

    @field_validator("new_column_name", mode="before")
    @classmethod
    def validate_new_column_name(cls, value: Any) -> str | None:
        return _normalize_optional_text_field(
            value,
            field_name="Signature-derived column names",
            allow_empty_as_none=True,
        )


class MLModelRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    model_type: Literal["rsf", "gbs", "lasso_cox", "compare"]
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = None
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = 42
    compute_shap: bool = False
    shap_safe_mode: bool = True
    evaluation_strategy: Literal["holdout", "repeated_cv"] = "holdout"
    cv_folds: int = Field(default=5, ge=2, le=10)
    cv_repeats: int = Field(default=3, ge=1, le=20)


class DeepModelRequest(_EventPositiveValueRequestModel):
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

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("Hidden layers must contain at least one positive integer.")
        if any((not isinstance(layer, int)) or layer <= 0 for layer in value):
            raise ValueError("Hidden layers must contain positive integers only.")
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


class TableExportRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=2000)
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

    @field_validator("caption", mode="before")
    @classmethod
    def validate_caption(cls, value: Any) -> str | None:
        return _normalize_optional_text_field(
            value,
            field_name="caption",
            allow_empty_as_none=True,
        )


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
        for level in meta.get("retained_levels", []):
            mapping[f"{column_name}_{level}"] = column_name
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


def _ml_replay_notes(request_config: dict[str, Any], *, dataset_filename: str) -> list[str]:
    features = [str(value) for value in request_config.get("features") or []]
    categorical = [str(value) for value in request_config.get("categorical_features") or []]
    evaluation_strategy = str(request_config.get("evaluation_strategy", "holdout"))
    settings = [
        f"evaluation={evaluation_strategy}",
        f"random_state={request_config.get('random_state')}",
        f"n_estimators={request_config.get('n_estimators')}",
        (
            f"max_depth={request_config.get('max_depth')}"
            if request_config.get("max_depth") is not None
            else "max_depth=auto"
        ),
    ]
    if request_config.get("learning_rate") is not None:
        settings.append(f"learning_rate={request_config.get('learning_rate')}")
    if evaluation_strategy == "repeated_cv":
        settings.append(
            f"cv={request_config.get('cv_repeats', 1)}x{request_config.get('cv_folds', 1)}"
        )

    notes = [
        (
            "Replay dataset: "
            f"{dataset_filename}. Outcome: time={request_config.get('time_column')}, "
            f"event={request_config.get('event_column')}={request_config.get('event_positive_value')}."
        ),
        "Replay settings: " + "; ".join(settings) + ".",
    ]
    if features:
        notes.append("Replay features: " + ", ".join(features) + ".")
    if categorical:
        notes.append("Replay categorical features: " + ", ".join(categorical) + ".")
    return notes


def _dl_replay_notes(
    request_config: dict[str, Any],
    *,
    dataset_filename: str,
    resolved_analysis: dict[str, Any] | None = None,
) -> list[str]:
    features = [str(value) for value in request_config.get("features") or []]
    categorical = [str(value) for value in request_config.get("categorical_features") or []]
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
        (
            "Replay dataset: "
            f"{dataset_filename}. Outcome: time={request_config.get('time_column')}, "
            f"event={request_config.get('event_column')}={request_config.get('event_positive_value')}."
        ),
        "Replay settings: " + "; ".join(settings) + ".",
    ]
    if resolved_analysis:
        actual_eval = resolved_analysis.get("evaluation_mode")
        if actual_eval is not None:
            notes.append(f"Reported evaluation outcome: {actual_eval}.")
        actual_note = (resolved_analysis.get("evaluation_note") or "").strip()
        if actual_note:
            notes.append("Reported evaluation note: " + actual_note)
    if features:
        notes.append("Replay features: " + ", ".join(features) + ".")
    if categorical:
        notes.append("Replay categorical features: " + ", ".join(categorical) + ".")
    return notes


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
    if not provenance:
        return []

    notes: list[str] = []
    request_config = provenance.get("request_config")
    if isinstance(request_config, dict) and request_config:
        notes.append(
            "Replay request_config: "
            + json.dumps(request_config, sort_keys=True, default=str, ensure_ascii=False)
        )

    analysis_meta = provenance.get("analysis")
    if isinstance(analysis_meta, dict) and analysis_meta:
        notes.append(
            "Replay analysis metadata: "
            + json.dumps(analysis_meta, sort_keys=True, default=str, ensure_ascii=False)
        )

    return notes


def fail_bad_request(exc: Exception) -> NoReturn:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, MemoryError):
        raise exc
    if isinstance(exc, ImportError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, NotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, UserInputError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, (ValueError, TypeError)):
        raise HTTPException(
            status_code=400,
            detail="The request could not be processed with the selected dataset and settings.",
        ) from exc
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
    return _format_export_value(value, style).replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()


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
        raise UserInputError("No rows available for export.")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row.keys():
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def _sanitize_csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    stripped = text.lstrip()
    if stripped.startswith("'") and _SIGNED_NUMERIC_CSV_LITERAL.fullmatch(stripped[1:]):
        prefix_len = len(text) - len(stripped)
        return text[:prefix_len] + stripped[1:]
    if _SIGNED_NUMERIC_CSV_LITERAL.fullmatch(stripped):
        return text
    if stripped.startswith(("=", "+", "-", "@")):
        return f"'{text}"
    return text


def _sanitize_markdown_cell(value: Any, style: str) -> str:
    return _sanitize_csv_cell(_normalize_export_text(value, style)).replace("|", "\\|")


def _sanitize_export_note(note: Any) -> str:
    return _sanitize_csv_cell(_normalize_export_text(note, "plain"))


def _export_rows_to_csv(
    rows: list[dict[str, Any]],
    style: str,
    *,
    caption: str | None = None,
    notes: Sequence[str] | None = None,
    template: str = "default",
) -> str:
    if not rows:
        raise UserInputError("No rows available for export.")
    columns = _export_columns(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([_sanitize_csv_cell(column) for column in columns])
    for row in rows:
        writer.writerow(
            [
                _sanitize_csv_cell(_format_export_value(row.get(column), style))
                for column in columns
            ]
        )
    return buffer.getvalue()


def _export_rows_to_xlsx(
    rows: list[dict[str, Any]],
    style: str,
    *,
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

    columns = _export_columns(rows)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Table"

    resolved_caption = (caption or "").strip()
    if resolved_caption:
        worksheet.append([_sanitize_csv_cell(resolved_caption)])
        worksheet.append([])

    worksheet.append([_sanitize_csv_cell(column) for column in columns])
    for row in rows:
        worksheet.append(
            [
                _sanitize_csv_cell(_format_export_value(row.get(column), style))
                for column in columns
            ]
        )

    clean_notes: list[str] = []
    for note in notes or []:
        sanitized_note = _sanitize_export_note(note)
        if sanitized_note:
            clean_notes.append(sanitized_note)
    if clean_notes:
        worksheet.append([])
        worksheet.append([_export_template_profile(template)["notes_heading"]])
        for note in clean_notes:
            worksheet.append([note])

    buffer = io.BytesIO()
    workbook.save(buffer)
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
        raise UserInputError("No rows available for export.")
    columns = _export_columns(rows)
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    clean_notes: list[str] = []
    for note in notes:
        sanitized_note = _sanitize_export_note(note)
        if sanitized_note:
            clean_notes.append(sanitized_note)
    header = f"| {' | '.join(_sanitize_markdown_cell(column, 'plain') for column in columns)} |"
    divider = f"| {' | '.join(['---'] * len(columns))} |"
    body = [
        "| "
        + " | ".join(_sanitize_markdown_cell(row.get(column), style) for column in columns)
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
    caption: str | None,
    notes: list[str],
    style: str,
    template: str,
) -> str:
    if not rows:
        raise UserInputError("No rows available for export.")
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
        raise UserInputError("No rows available for export.")
    template_profile = _export_template_profile(template)
    resolved_caption = _resolve_export_caption(caption, template)
    caption_bold = template in {"default", "jco"}
    caption_italic = template == "nejm"
    body_parts = [
        _docx_paragraph(resolved_caption, bold=caption_bold, italic=caption_italic),
        _docx_table(rows, style=style),
    ]
    clean_notes: list[str] = []
    for note in notes:
        sanitized_note = _sanitize_export_note(note)
        if sanitized_note:
            clean_notes.append(sanitized_note)
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    response = templates.TemplateResponse(request, "index.html", {"static_version": _static_asset_version()})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "python_version": platform.python_version(),
        "app_version": _package_version_or_unknown("survival-toolkit"),
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


@app.post("/api/shutdown")
async def shutdown_server(request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Shutdown is allowed only from a local session.")
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
        dataframe = await run_in_threadpool(loader)
        _enforce_upload_shape_limits(dataframe)
        ensure_model_feature_candidate_limit(dataframe)
        metadata = {"preset_name": preset_name} if preset_name else None
        stored = store.create(dataframe, filename=filename, source=source, metadata=metadata)
        return await run_in_threadpool(dataset_response, stored.dataset_id)
    except Exception as exc:
        fail_bad_request(exc)


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
        _enforce_upload_shape_limits(dataframe)
        ensure_model_feature_candidate_limit(dataframe)
        stored = store.create(dataframe, filename=filename, source="upload")
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
        return dataset_response(dataset_id)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/export-table")
async def export_table(request_model: TableExportRequest) -> Response:
    try:
        export_notes = list(request_model.notes or [])
        for provenance_note in _export_provenance_notes(request_model.provenance):
            if provenance_note not in export_notes:
                export_notes.append(provenance_note)
        if request_model.format == "csv":
            content = _export_rows_to_csv(
                request_model.rows,
                request_model.style,
                caption=request_model.caption,
                notes=export_notes,
                template=request_model.template,
            )
            return Response(content=content, media_type="text/csv; charset=utf-8")
        if request_model.format == "xlsx":
            content = _export_rows_to_xlsx(
                request_model.rows,
                request_model.style,
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
                caption=request_model.caption,
                notes=export_notes,
                style=request_model.style,
                template=request_model.template,
            )
            return Response(content=content, media_type="text/markdown; charset=utf-8")
        if request_model.format == "latex":
            content = _export_rows_to_latex(
                request_model.rows,
                caption=request_model.caption,
                notes=export_notes,
                style=request_model.style,
                template=request_model.template,
            )
            return Response(content=content, media_type="text/x-tex; charset=utf-8")
        content = _export_rows_to_docx(
            request_model.rows,
            caption=request_model.caption,
            notes=export_notes,
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

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/kaplan-meier")
async def kaplan_meier(request_model: KaplanMeierRequest) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        outcome_informed_group = (
            request_model.group_column is not None
            and str(request_model.group_column) in _outcome_informed_columns(stored)
        )

        def _run() -> dict[str, Any]:
            from survival_toolkit.plots import build_km_figure

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
            return {"analysis": analysis, "figure": figure, "request_config": request_config}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cox")
async def cox(request_model: CoxRequest) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        _reject_survival_outcome_feature_columns(
            stored,
            [*request_model.covariates, *request_model.categorical_covariates],
            time_column=request_model.time_column,
            event_column=request_model.event_column,
            event_positive_value=request_model.event_positive_value,
            context="Cox covariates",
        )
        _reject_outcome_informed_columns(
            stored,
            [*request_model.covariates, *request_model.categorical_covariates],
            context="Cox covariates",
        )

        def _run() -> dict[str, Any]:
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
            )
            figure = build_cox_forest_figure(analysis)
            diagnostics_figure = build_cox_diagnostics_figure(analysis)
            martingale_figure = build_cox_martingale_figure(analysis)
            return {
                "analysis": analysis,
                "figure": figure,
                "diagnostics_figure": diagnostics_figure,
                "martingale_figure": martingale_figure,
                "request_config": request_config,
            }

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cox-preview")
async def cox_preview(request_model: CoxRequest) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        _reject_survival_outcome_feature_columns(
            stored,
            [*request_model.covariates, *request_model.categorical_covariates],
            time_column=request_model.time_column,
            event_column=request_model.event_column,
            event_positive_value=request_model.event_positive_value,
            context="Cox covariates",
        )
        _reject_outcome_informed_columns(
            stored,
            [*request_model.covariates, *request_model.categorical_covariates],
            context="Cox covariates",
        )

        def _run() -> dict[str, Any]:
            preview = preview_cox_analysis_inputs(
                stored.dataframe,
                time_column=request_model.time_column,
                event_column=request_model.event_column,
                event_positive_value=request_model.event_positive_value,
                covariates=request_model.covariates,
                categorical_covariates=request_model.categorical_covariates,
            )
            return {"preview": preview, "request_config": request_config}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/cohort-table")
async def cohort_table(request_model: CohortTableRequest) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
        if request_model.group_column:
            _reject_outcome_informed_columns(
                stored,
                [request_model.group_column],
                context="grouped cohort tables",
            )

        def _run() -> dict[str, Any]:
            return {"analysis": compute_cohort_table(
                stored.dataframe,
                variables=request_model.variables,
                group_column=request_model.group_column,
            ), "request_config": request_config}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/discover-signature")
async def discover_signature(request_model: SignatureSearchRequest) -> dict[str, Any]:
    try:
        stored = _get_stored_dataset(request_model.dataset_id)
        request_config = request_model.model_dump()
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
        payload = _create_dataset_snapshot(stored, updated, metadata={
            **stored.metadata,
            "derived_column_provenance": provenance,
        })
        payload["derived_column"] = column_name
        payload["signature_analysis"] = analysis
        payload["signature_request_config"] = request_config
        return payload
    except Exception as exc:
        fail_bad_request(exc)


# ── ML model endpoints ──────────────────────────────────────────


@app.post("/api/optimal-cutpoint")
async def optimal_cutpoint(request_model: OptimalCutpointRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import find_optimal_cutpoint
        from survival_toolkit.plots import build_cutpoint_scan_figure

        stored = _get_stored_dataset(request_model.dataset_id)

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
                _attach_manuscript_notes(
                    comparison,
                    _ml_replay_notes(request_config, dataset_filename=stored.filename),
                )
                figure = build_model_comparison_figure(comparison)
                return {"analysis": comparison, "figure": figure, "request_config": request_config}

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
                                shap_error = (
                                    f"{type(exc).__name__}: {exc} "
                                    f"SHAP safe mode also failed: {type(safe_mode_exc).__name__}: {safe_mode_exc}"
                                )

            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            return {
                "analysis": clean_result,
                "importance_figure": importance_figure,
                "shap_result": shap_result,
                "shap_figure": shap_figure,
                "shap_error": shap_error,
                "shap_companion": shap_companion,
                "request_config": request_config,
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

        stored = _get_stored_dataset(request_model.dataset_id)
        df = stored.dataframe
        request_config = request_model.model_dump()
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
                _attach_manuscript_notes(
                    result,
                    _dl_replay_notes(request_config, dataset_filename=stored.filename, resolved_analysis=result),
                )
                clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
                return {
                    "analysis": clean_result,
                    "figures": {"comparison": build_model_comparison_figure(result)},
                    "request_config": request_config,
                }

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
                    epochs_trained=result.get("epochs_trained"),
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
            return {"analysis": clean_result, "figures": figures, "request_config": request_config}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


# ── XAI endpoints ──────────────────────────────────────────────


class TimeDependentImportanceRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    eval_times: list[float] | None = Field(default=None, max_length=100)


class CounterfactualRequest(_EventPositiveValueRequestModel):
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
    max_depth: int | None = None
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = 42


class PDPRequest(_EventPositiveValueRequestModel):
    dataset_id: str
    time_column: str
    event_column: str
    event_positive_value: Any = 1
    features: list[str] = Field(max_length=1000)
    categorical_features: list[str] = Field(default_factory=list, max_length=1000)
    target_feature: str
    model_type: Literal["rsf", "gbs"] = "rsf"
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = None
    learning_rate: float = Field(default=0.1, gt=0.001, le=1.0)
    random_state: int = 42


@app.post("/api/time-dependent-importance")
async def time_dependent_importance(request_model: TimeDependentImportanceRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import compute_time_dependent_importance
        from survival_toolkit.plots import build_time_dependent_importance_figure

        stored = _get_stored_dataset(request_model.dataset_id)
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

        stored = _get_stored_dataset(request_model.dataset_id)
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
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
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
            return {"analysis": analysis, "request_config": request_config}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)


@app.post("/api/pdp")
async def pdp(request_model: PDPRequest) -> dict[str, Any]:
    try:
        from survival_toolkit.ml_models import (
            compute_partial_dependence,
            train_gradient_boosted_survival,
            train_random_survival_forest,
        )
        from survival_toolkit.plots import build_pdp_figure

        stored = _get_stored_dataset(request_model.dataset_id)
        df = stored.dataframe
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
        request_config = request_model.model_dump()

        def _run() -> dict[str, Any]:
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
            return {"analysis": result, "figure": figure, "request_config": request_config}

        return await run_in_threadpool(_run)
    except Exception as exc:
        fail_bad_request(exc)

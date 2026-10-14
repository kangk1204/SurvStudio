"""Trusted local checkpoints for an operating workspace.

These files contain sensitive tables and fitted objects. They are never an upload
format or a network endpoint. A caller must pause admission and drain every job
before calling either operation. Existing browser-only responses and model objects
that were never cached cannot be reconstructed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import stat
import time
from uuid import uuid4
from collections import OrderedDict
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, MutableMapping

import numpy as np
import pandas as pd

from survival_toolkit.store import StoredDataset

SCHEMA_VERSION = 1
_MAX_STATE_BYTES = 8 * 1024 * 1024 * 1024
_VERSION_PACKAGES = (
    "numpy", "pandas", "scipy", "joblib", "scikit-learn", "scikit-survival", "statsmodels", "torch",
)


class WorkspaceStateError(ValueError):
    """A checkpoint cannot safely be captured or restored."""


def runtime_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in _VERSION_PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _owner_digest(owner_scope: str) -> str:
    if not isinstance(owner_scope, str) or not owner_scope.strip():
        raise WorkspaceStateError("A trusted owner scope is required.")
    return hashlib.sha256(owner_scope.encode("utf-8")).hexdigest()


def _check_quiescence(check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if not callable(check):
        raise WorkspaceStateError("A global admission-and-drain check is required.")
    result = check()
    if not isinstance(result, dict) or result.get("admission_paused") is not True:
        raise WorkspaceStateError("Request admission must be paused before checkpointing.")
    for key in ("active_requests", "active_jobs", "queued_jobs", "active_uploads"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise WorkspaceStateError("Requests, uploads and queued/running jobs must be drained.")
    return {key: result[key] for key in ("admission_paused", "active_requests",
        "active_jobs", "queued_jobs", "active_uploads")}


def _frame_digest(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise WorkspaceStateError("A stored table is not a DataFrame.")
    digest = hashlib.sha256()
    # Pandas Index/Series pickle bytes depend on internal caches and object
    # identity after decoding. Fingerprint their declared semantics instead.
    schema = {"shape": list(frame.shape), "columns": [repr(c) for c in frame.columns],
        "column_names": [repr(c) for c in frame.columns.names],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_type": type(frame.index).__qualname__,
        "index_names": [repr(c) for c in frame.index.names],
        "index_dtype": str(getattr(frame.index, "dtype", "MultiIndex")),
        "categories": [{"column": i, "ordered": dtype.ordered,
                        "levels": [repr(c) for c in dtype.categories]}
                       for i, dtype in enumerate(frame.dtypes) if isinstance(dtype, pd.CategoricalDtype)]}
    digest.update(json.dumps(schema, sort_keys=True).encode())
    try:
        rows = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype=np.uint64)
    except TypeError:
        rows = pd.util.hash_pandas_object(frame.astype("string"), index=True, categorize=True).to_numpy(dtype=np.uint64)
    digest.update(np.ascontiguousarray(rows).tobytes())
    return digest.hexdigest()


def _private_directory(path: Path) -> None:
    if os.name != "posix":
        raise WorkspaceStateError("Trusted local checkpoints require POSIX ownership checks.")
    if path.is_symlink():
        raise WorkspaceStateError("Checkpoint directories must not be symlinks.")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise WorkspaceStateError("Checkpoint directories must be private to the current account.")


def _open_private(path: Path, *, maximum: int) -> bytes:
    if path.is_symlink():
        raise WorkspaceStateError("Checkpoint files must not be symlinks.")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise WorkspaceStateError("Checkpoint files must be private regular files.")
        if info.st_size <= 0 or info.st_size > maximum:
            raise WorkspaceStateError("Checkpoint file size is outside the supported range.")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read(maximum + 1)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def write_workspace_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    """Publish a private operator restore receipt once, without overwriting.

    Only aggregate restore fields are accepted. An exclusive hard-link commit
    makes the final filename visible only after its complete contents are durable
    and also protects against a concurrently created destination.
    """
    destination = Path(path).absolute()
    _private_directory(destination.parent)
    allowed = {"restored", "schema_version", "captured_source_revision",
        "runtime_versions", "payload_sha256", "dataset_count", "artifact_count",
        "marker_count", "completed_result_count", "idle_policy",
        "dataset_ids_and_metadata_preserved", "uncached_results_reconstructed"}
    if not isinstance(receipt, dict) or set(receipt) - allowed:
        raise WorkspaceStateError("Restore receipts must contain aggregate fields only.")
    if receipt.get("restored") is not True:
        raise WorkspaceStateError("An operator receipt must describe a successful restore.")
    digest = receipt.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise WorkspaceStateError("An operator receipt requires its checkpoint SHA256.")
    for key in ("dataset_count", "artifact_count", "marker_count", "completed_result_count"):
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkspaceStateError("Restore receipt counts must be nonnegative integers.")
    for key in ("dataset_ids_and_metadata_preserved", "uncached_results_reconstructed"):
        if key in receipt and not isinstance(receipt[key], bool):
            raise WorkspaceStateError("Restore receipt flags must be boolean.")
    if "runtime_versions" in receipt:
        versions = receipt["runtime_versions"]
        if not isinstance(versions, dict) or set(versions) - {"python", *_VERSION_PACKAGES} or any(
            not isinstance(value, str) or len(value) > 128 for value in versions.values()
        ):
            raise WorkspaceStateError("Restore receipt versions must be package version strings.")
    if destination.exists() or destination.is_symlink():
        raise WorkspaceStateError("Use a new operator receipt path; earlier receipts are immutable.")
    staged = destination.with_name(".workspace-receipt-" + uuid4().hex)
    try:
        _atomic_write(staged, json.dumps(receipt, indent=2, allow_nan=False).encode())
        try:
            os.link(staged, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise WorkspaceStateError("The operator receipt path was already created; it was not overwritten.") from exc
        fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        staged.unlink(missing_ok=True)
        staged.with_name(staged.name + ".partial").unlink(missing_ok=True)


def _locks(stack: ExitStack, dataset_store: Any, artifact_cache: Any, marker_store: Any) -> None:
    for component in (dataset_store, artifact_cache, marker_store):
        if component is not None:
            stack.enter_context(component._lock)


def _assert_idle_store(store: Any) -> None:
    if any(store._leases.values()):
        raise WorkspaceStateError("Dataset leases remain active; do not interrupt their jobs.")


def _config(payload: dict[str, Any], key: str, fields: tuple[str, ...]) -> dict[str, Any]:
    result = payload.get(key)
    if not isinstance(result, dict) or any(
        isinstance(result.get(name), bool) or not isinstance(result.get(name), (int, float))
        or not math.isfinite(result[name]) or result[name] <= 0 for name in fields
    ):
        raise WorkspaceStateError("Checkpoint cache configuration is invalid.")
    return result


def save_workspace_state(
    directory: str | Path,
    *,
    dataset_store: Any,
    artifact_cache: Any = None,
    marker_store: Any = None,
    completed_results: MutableMapping[str, Any] | None = None,
    owner_scope: str,
    source_revision: str,
    quiescence_check: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Capture current objects without expiring, touching or recreating datasets.

    The directory must be a new path under a trusted private parent. Earlier
    checkpoints are never overwritten. Serialization is performed with the cache
    locks held and admission paused; callable/thread/lock objects are excluded.
    """
    owner = _owner_digest(owner_scope)
    if not isinstance(source_revision, str) or not source_revision:
        raise WorkspaceStateError("The captured source revision is required.")
    quiescence = _check_quiescence(quiescence_check)
    destination = Path(directory).absolute()
    _private_directory(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise WorkspaceStateError("Use a new checkpoint directory; preserved states are immutable.")
    versions = runtime_versions()
    with ExitStack() as stack:
        _locks(stack, dataset_store, artifact_cache, marker_store)
        _assert_idle_store(dataset_store)
        now = dataset_store._clock()
        datasets = []
        for key, stored in dataset_store._datasets.items():
            if key != stored.dataset_id:
                raise WorkspaceStateError("Dataset identity is inconsistent.")
            datasets.append({
                "dataset_id": key, "filename": stored.filename, "source": stored.source,
                "dataframe": stored.dataframe, "created_at": stored.created_at,
                "last_accessed": stored.last_accessed, "metadata": stored.metadata,
                "idle_age": max(0.0, now - dataset_store._last_used.get(key, now)),
                "frame_digest": _frame_digest(stored.dataframe),
            })
        artifacts = list(artifact_cache._items.items()) if artifact_cache is not None else None
        markers = [] if marker_store is not None else None
        if marker_store is not None:
            marker_now = time.monotonic()
            for key, (matrix, used) in marker_store._items.items():
                markers.append({"matrix_id": key, "matrix": matrix,
                    "idle_age": max(0.0, marker_now - used),
                    "values_writeable": matrix.values.flags.writeable})
        payload = {
            "schema_version": SCHEMA_VERSION, "owner_digest": owner,
            "source_revision": source_revision, "versions": versions,
            "datasets": datasets,
            "store_config": {"max_datasets": dataset_store._max_datasets,
                "ttl_seconds": dataset_store._ttl_seconds,
                "shared_metadata_keys": dataset_store._shared_metadata_keys},
            "artifacts": artifacts,
            "artifact_sizes": dict(artifact_cache._sizes) if artifact_cache is not None else None,
            "artifact_config": {"max_items": artifact_cache._max_items,
                "max_bytes": artifact_cache._max_bytes} if artifact_cache is not None else None,
            "markers": markers,
            "marker_config": {"max_items": marker_store._max_items,
                "max_bytes": marker_store._max_bytes,
                "ttl_seconds": marker_store._ttl_seconds} if marker_store is not None else None,
            "completed_results": dict(completed_results) if completed_results is not None else None,
        }
        raw = pickle.dumps(payload, protocol=5)
        _check_quiescence(quiescence_check)
    if len(raw) > _MAX_STATE_BYTES:
        raise WorkspaceStateError("The checkpoint is too large; no existing state was changed.")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision, "versions": versions,
        "owner_digest": owner, "payload_file": "workspace.pkl",
        "payload_bytes": len(raw), "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "datasets": [{"identity_digest": hashlib.sha256(row["dataset_id"].encode()).hexdigest(),
            "rows": len(row["dataframe"]), "columns": len(row["dataframe"].columns),
            "frame_digest": row["frame_digest"]} for row in datasets],
        "dataset_count": len(datasets), "artifact_count": len(artifacts or []),
        "marker_count": len(markers or []),
        "completed_result_count": len(completed_results) if completed_results is not None else 0,
        "completed_results_available": completed_results is not None,
        "quiescence": quiescence, "live_objects_preserved": True,
        "limitations": ["Only objects present in the process can be captured.",
            "Browser-only responses and uncached neural model weights cannot be recovered."],
    }
    destination.mkdir(mode=0o700)
    _atomic_write(destination / "workspace.pkl", raw)
    _atomic_write(destination / "receipt.json", json.dumps(receipt, indent=2).encode())
    fd = os.open(destination, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def restore_workspace_state(
    directory: str | Path,
    *,
    dataset_store: Any,
    artifact_cache: Any = None,
    marker_store: Any = None,
    completed_results: MutableMapping[str, Any] | None = None,
    owner_scope: str,
    quiescence_check: Callable[[], dict[str, Any]],
    expected_versions: dict[str, str] | None = None,
    idle_policy: str = "preserve",
) -> dict[str, Any]:
    """Restore a trusted local checkpoint into empty caches after full validation.

    ``preserve`` rebases monotonic ages while excluding maintenance downtime.
    ``refresh`` explicitly starts a new idle-TTL window without changing captured
    timestamps or scientific results. Owner/version/hash checks precede decoding;
    local permissions are the trust boundary, not the unkeyed integrity digest.
    """
    owner = _owner_digest(owner_scope)
    if idle_policy not in {"preserve", "refresh"}:
        raise WorkspaceStateError("Choose preserve or refresh for the idle policy.")
    _check_quiescence(quiescence_check)
    source = Path(directory).absolute()
    _private_directory(source)
    receipt = json.loads(_open_private(source / "receipt.json", maximum=4 * 1024 * 1024))
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("owner_digest") != owner:
        raise WorkspaceStateError("Checkpoint owner or schema does not match.")
    versions = runtime_versions() if expected_versions is None else dict(expected_versions)
    if receipt.get("versions") != versions:
        raise WorkspaceStateError("Checkpoint runtime versions differ; use the captured environment.")
    if receipt.get("payload_file") != "workspace.pkl":
        raise WorkspaceStateError("Checkpoint payload path is invalid.")
    raw = _open_private(source / "workspace.pkl", maximum=_MAX_STATE_BYTES)
    if len(raw) != receipt.get("payload_bytes") or hashlib.sha256(raw).hexdigest() != receipt.get("payload_sha256"):
        raise WorkspaceStateError("Checkpoint bytes do not match their receipt.")
    payload = pickle.loads(raw)  # trusted current-account local state, never uploaded data
    if not isinstance(payload, dict) or any(payload.get(key) != receipt.get(key)
            for key in ("schema_version", "owner_digest", "source_revision", "versions")):
        raise WorkspaceStateError("Checkpoint payload and receipt disagree.")
    rows = payload.get("datasets")
    if not isinstance(rows, list) or len(rows) != receipt.get("dataset_count"):
        raise WorkspaceStateError("Checkpoint dataset count is invalid.")
    datasets: OrderedDict[str, StoredDataset] = OrderedDict()
    ages: dict[str, float] = {}
    for row, info in zip(rows, receipt.get("datasets", []), strict=True):
        key = row["dataset_id"]
        age = row["idle_age"]
        if not isinstance(key, str) or not key or key in datasets or not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0:
            raise WorkspaceStateError("Checkpoint dataset identity or idle age is invalid.")
        digest = _frame_digest(row["dataframe"])
        if digest != row["frame_digest"] or digest != info.get("frame_digest") or hashlib.sha256(key.encode()).hexdigest() != info.get("identity_digest"):
            raise WorkspaceStateError("A checkpoint dataset failed its integrity check.")
        if not isinstance(row["metadata"], dict) or not all(isinstance(row[k], datetime) for k in ("created_at", "last_accessed")):
            raise WorkspaceStateError("Checkpoint dataset metadata is invalid.")
        datasets[key] = StoredDataset(**{k: row[k] for k in (
            "dataset_id", "filename", "source", "dataframe", "created_at", "last_accessed", "metadata")})
        ages[key] = 0.0 if idle_policy == "refresh" else float(age)
    artifacts = OrderedDict(payload.get("artifacts") or [])
    sizes = payload.get("artifact_sizes") or {}
    if len(artifacts) != receipt.get("artifact_count") or set(artifacts) != set(sizes):
        raise WorkspaceStateError("Checkpoint artifact inventory is invalid.")
    for key, value in artifacts.items():
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in datasets or not isinstance(value, dict) or not isinstance(value.get("result"), dict):
            raise WorkspaceStateError("Checkpoint artifact has an invalid dataset reference.")
        if not isinstance(sizes[key], int) or sizes[key] < 0:
            raise WorkspaceStateError("Checkpoint artifact byte estimate is invalid.")
    markers = OrderedDict()
    for row in payload.get("markers") or []:
        key = row["matrix_id"]; matrix = row["matrix"]; age = row["idle_age"]
        if not isinstance(key, str) or not key or key in markers or not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0:
            raise WorkspaceStateError("Checkpoint marker identity or idle age is invalid.")
        if not isinstance(matrix.values, np.ndarray) or matrix.values.shape != (len(matrix.sample_keys), len(matrix.marker_names)):
            raise WorkspaceStateError("Checkpoint marker dimensions are invalid.")
        matrix.values.setflags(write=bool(row["values_writeable"]))
        markers[key] = (matrix, 0.0 if idle_policy == "refresh" else float(age))
    if len(markers) != receipt.get("marker_count"):
        raise WorkspaceStateError("Checkpoint marker inventory is invalid.")
    saved_results = payload.get("completed_results")
    if saved_results is not None and (not isinstance(saved_results, dict) or len(saved_results) != receipt.get("completed_result_count")):
        raise WorkspaceStateError("Checkpoint completed-result inventory is invalid.")
    if (payload.get("artifacts") is not None and artifact_cache is None) or (payload.get("markers") is not None and marker_store is None) or (saved_results is not None and completed_results is None):
        raise WorkspaceStateError("A target cache is missing; no checkpoint component may be dropped.")
    store_config = _config(payload, "store_config", ("max_datasets", "ttl_seconds"))
    shared_keys = store_config.get("shared_metadata_keys")
    if not isinstance(shared_keys, (set, frozenset, list, tuple)) or not all(isinstance(k, str) for k in shared_keys):
        raise WorkspaceStateError("Checkpoint shared metadata keys are invalid.")
    artifact_config = _config(payload, "artifact_config", ("max_items", "max_bytes")) if payload["artifact_config"] is not None else None
    marker_config = _config(payload, "marker_config", ("max_items", "max_bytes", "ttl_seconds")) if payload["marker_config"] is not None else None
    if completed_results is not None and type(completed_results) not in {dict, OrderedDict}:
        raise WorkspaceStateError("Completed results require a plain local dictionary.")
    shared_keys = frozenset(shared_keys)
    with ExitStack() as stack:
        _locks(stack, dataset_store, artifact_cache, marker_store)
        _check_quiescence(quiescence_check)
        _assert_idle_store(dataset_store)
        if dataset_store._datasets or (artifact_cache is not None and artifact_cache._items) or (marker_store is not None and marker_store._items) or completed_results:
            raise WorkspaceStateError("Restore requires empty target caches; existing state is protected.")
        now = dataset_store._clock()
        last_used = {key: now - age for key, age in ages.items()}
        marker_now = time.monotonic()
        marker_items = OrderedDict((key, (matrix, marker_now-age)) for key, (matrix, age) in markers.items())
        dataset_store._max_datasets = store_config["max_datasets"]
        dataset_store._ttl_seconds = store_config["ttl_seconds"]
        dataset_store._shared_metadata_keys = shared_keys
        dataset_store._datasets = datasets
        dataset_store._last_used = last_used
        if artifact_cache is not None and artifact_config is not None:
            artifact_cache._max_items = artifact_config["max_items"]; artifact_cache._max_bytes = artifact_config["max_bytes"]
            artifact_cache._items = artifacts; artifact_cache._sizes = sizes
        if marker_store is not None and marker_config is not None:
            marker_store._max_items = marker_config["max_items"]; marker_store._max_bytes = marker_config["max_bytes"]
            marker_store._ttl_seconds = marker_config["ttl_seconds"]
            marker_store._items = marker_items
        if completed_results is not None and saved_results is not None:
            completed_results.update(saved_results)
    return {"restored": True, "schema_version": SCHEMA_VERSION,
        "captured_source_revision": receipt["source_revision"], "runtime_versions": versions,
        "payload_sha256": receipt["payload_sha256"], "dataset_count": len(datasets),
        "artifact_count": len(artifacts), "marker_count": len(markers),
        "completed_result_count": len(saved_results or {}), "idle_policy": idle_policy,
        "dataset_ids_and_metadata_preserved": True,
        "uncached_results_reconstructed": False}

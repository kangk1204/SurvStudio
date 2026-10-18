"""Private, authenticated local storage for uploaded tables and completed analyses.

This database is application state, never an upload format. Pickled tables and fitted
objects are decoded only after owner, runtime, checksum and HMAC verification. API
results use JSON so missing estimates remain null. No record expires automatically.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import pickle
import secrets
import sqlite3
import stat
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_SCHEMA = 1
_MAX_BLOB = 8 * 1024**3


class DurableWorkspaceError(RuntimeError):
    """Local saving or restoration failed; the previous saved state is retained."""


def _runtime() -> dict[str, str]:
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "statsmodels", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _json(value: Any) -> str:
    # JSON may legally carry an unpaired UTF-16 surrogate. Match the public
    # response encoder by escaping text rather than failing after a completed fit.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _private(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or (directory and not stat.S_ISDIR(info.st_mode)) or (not directory and not stat.S_ISREG(info.st_mode)):
        raise DurableWorkspaceError("The saved workspace path is not a private regular file or directory.")
    if not directory and info.st_nlink != 1:
        raise DurableWorkspaceError("Saved workspace files must not have additional hard links.")
    if os.name == "posix" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise DurableWorkspaceError("The saved workspace must belong to this account and allow access only to its owner.")


def _fsync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class DurableWorkspace:
    """One writer, with thread-safe SQLite transactions and authenticated records."""

    def __init__(self, root: Path, owner_scope: str) -> None:
        if not owner_scope or not owner_scope.strip():
            raise DurableWorkspaceError("A trusted workspace owner is required.")
        self.root = Path(root).absolute()
        self._owner = hashlib.sha256(owner_scope.encode()).hexdigest()
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        self._writer = None
        self._closed = False
        try:
            if not self.root.exists():
                self.root.mkdir(mode=0o700, parents=True)
            _private(self.root, directory=True)
            self._acquire_writer()
            key_path = self.root / "workspace.key"
            database = self.root / "workspace.sqlite3"
            if database.exists() and not key_path.exists():
                raise DurableWorkspaceError("The saved workspace authentication key is missing; no data was changed.")
            if not key_path.exists():
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(secrets.token_bytes(32))
                    handle.flush()
                    os.fsync(handle.fileno())
            _private(key_path)
            self._key = key_path.read_bytes()
            if len(self._key) != 32:
                raise DurableWorkspaceError("The saved workspace authentication key is invalid.")
            if database.exists():
                _private(database)
            else:
                fd = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                os.close(fd)
            for suffix in ("-journal", "-wal", "-shm"):
                sidecar = Path(str(database) + suffix)
                if sidecar.exists():
                    _private(sidecar)
            self._db = sqlite3.connect(database, timeout=30, isolation_level=None, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, key TEXT NOT NULL, dataset_id TEXT NOT NULL, metadata TEXT NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(kind,key))")
            expected = {"schema": str(_SCHEMA), "owner": self._owner, "runtime": _json(_runtime())}
            present = dict(self._db.execute("SELECT key,value FROM settings"))
            if present:
                if present != expected:
                    raise DurableWorkspaceError("Saved workspace owner, schema or runtime differs. Preserve it and use the original environment to export it.")
            else:
                if self._db.execute("SELECT count(*) FROM records").fetchone()[0]:
                    raise DurableWorkspaceError("Saved workspace identity is missing; no records were decoded.")
                self._db.execute("BEGIN IMMEDIATE")
                self._db.executemany("INSERT INTO settings VALUES (?,?)", expected.items())
                self._commit()
            _fsync_directory(self.root)
        except Exception as exc:
            self.close()
            if isinstance(exc, DurableWorkspaceError):
                raise
            raise DurableWorkspaceError("The saved workspace could not be opened safely.") from exc

    def _acquire_writer(self) -> None:
        path = self.root / "writer.lock"
        if path.exists():
            _private(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self._writer = os.fdopen(fd, "r+b", buffering=0)
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - exercised by Windows CI
                import msvcrt
                if not path.stat().st_size:
                    self._writer.write(b"0")
                self._writer.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise DurableWorkspaceError("This saved workspace is already open by another server process.") from exc

    def _connection(self) -> sqlite3.Connection:
        if self._closed or self._db is None:
            raise DurableWorkspaceError("The saved workspace is closed.")
        return self._db

    def _commit(self) -> None:
        self._connection().commit()

    def _tag(self, kind: str, key: str, dataset_id: str, metadata: str, digest: str) -> str:
        message = _json([self._owner, kind, key, dataset_id, metadata, digest]).encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    @staticmethod
    def _identifier(value: Any) -> str:
        key = value if isinstance(value, str) else _json(value)
        if not key or len(key) > 4096:
            raise DurableWorkspaceError("The saved record identifier is invalid.")
        return key

    def _metadata_verified(self, kind: str, key: str, dataset_id: str, metadata: str, digest: str, tag: str) -> dict:
        if not hmac.compare_digest(self._tag(kind, key, dataset_id, metadata, digest), tag):
            raise DurableWorkspaceError("A saved record failed its integrity check. It was not decoded or replaced.")
        return json.loads(metadata)

    def _verified(self, row: tuple) -> tuple[dict, bytes]:
        kind, key, dataset_id, metadata, payload, digest, tag = row
        parsed = self._metadata_verified(kind, key, dataset_id, metadata, digest, tag)
        if len(payload) > _MAX_BLOB or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), digest):
            raise DurableWorkspaceError("A saved record failed its integrity check. It was not decoded or replaced.")
        return parsed, payload

    def _write(self, kind: str, key: Any, payload: bytes, metadata: dict, *, dataset_id: str = "", invalidate: bool = False, expected_dataset_hash: str | None = None) -> None:
        key = self._identifier(key)
        if len(payload) > _MAX_BLOB:
            raise DurableWorkspaceError("The saved record exceeds the workspace size limit.")
        text = _json(metadata)
        digest = hashlib.sha256(payload).hexdigest()
        tag = self._tag(kind, key, dataset_id, text, digest)
        with self._lock:
            db = self._connection()
            try:
                db.execute("BEGIN IMMEDIATE")
                if dataset_id and kind in ("result", "snapshot", "artifact"):
                    parent = db.execute("SELECT kind,key,dataset_id,metadata,sha256,tag FROM records WHERE kind='dataset' AND key=?", (dataset_id,)).fetchone()
                    if parent is None or self._metadata_verified(*parent).get("dataset_hash") != expected_dataset_hash:
                        raise DurableWorkspaceError("The source dataset changed or was removed before saving. This result was not attached to a different input.")
                for related_id in metadata.get("related_dataset_ids", []):
                    related = db.execute("SELECT kind,key,dataset_id,metadata,sha256,tag FROM records WHERE kind='dataset' AND key=?", (related_id,)).fetchone()
                    if related is None:
                        raise DurableWorkspaceError("An input used by this saved result was explicitly removed. The result was not saved again.")
                    self._metadata_verified(*related)
                previous = db.execute("SELECT kind,key,dataset_id,metadata,payload,sha256,tag FROM records WHERE kind=? AND key=?", (kind, key)).fetchone()
                if previous is not None:
                    previous_metadata = self._verified(previous)[0]
                    if kind == "dataset" and (previous_metadata.get("dataset_hash") != metadata.get("dataset_hash")
                                              or previous_metadata.get("table_sha256") != metadata.get("table_sha256")):
                        invalidate = True
                if invalidate:
                    db.execute("DELETE FROM records WHERE dataset_id=? AND kind IN ('result','snapshot','artifact')", (dataset_id,))
                db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?)", (kind, key, dataset_id, text, payload, digest, tag))
                self._commit()
            except Exception as exc:
                db.rollback()
                if isinstance(exc, DurableWorkspaceError):
                    raise
                raise DurableWorkspaceError("Automatic saving failed. The previous saved state was retained; free disk space and retry.") from exc

    def _read(self, kind: str, key: Any) -> tuple[dict, bytes] | None:
        with self._lock:
            try:
                row = self._connection().execute("SELECT kind,key,dataset_id,metadata,payload,sha256,tag FROM records WHERE kind=? AND key=?", (kind, self._identifier(key))).fetchone()
                return self._verified(row) if row is not None else None
            except (sqlite3.Error, OSError, ValueError) as exc:
                raise DurableWorkspaceError("The saved record could not be restored safely.") from exc

    def _read_metadata(self, kind: str, key: Any) -> dict | None:
        with self._lock:
            try:
                row = self._connection().execute("SELECT kind,key,dataset_id,metadata,sha256,tag FROM records WHERE kind=? AND key=?", (kind, self._identifier(key))).fetchone()
                return self._metadata_verified(*row) if row is not None else None
            except (sqlite3.Error, OSError, ValueError) as exc:
                raise DurableWorkspaceError("The saved record metadata could not be verified.") from exc

    def has_dataset(self, dataset_id: str) -> bool:
        return self._read_metadata("dataset", dataset_id) is not None

    def _list(self, kind: str, dataset_id: str | None = None) -> list[dict]:
        with self._lock:
            db = self._connection()
            # Listings authenticate metadata without reading large patient or model blobs.
            query = "SELECT kind,key,dataset_id,metadata,sha256,tag FROM records WHERE kind=?"
            args: tuple = (kind,)
            if dataset_id is not None:
                query += " AND dataset_id=?"
                args += (dataset_id,)
            try:
                return [self._metadata_verified(*row) for row in db.execute(query, args)]
            except (sqlite3.Error, OSError, ValueError) as exc:
                raise DurableWorkspaceError("The saved workspace listing could not be verified.") from exc

    def _delete(self, kind: str, key: Any, *, cascade: bool = False) -> None:
        with self._lock:
            db = self._connection()
            key = self._identifier(key)
            try:
                db.execute("BEGIN IMMEDIATE")
                if cascade:
                    # Validation is shown in the development workspace, but its
                    # saved patient outputs also belong to the external input.
                    # Verify signed dependency metadata without reading blobs.
                    linked = []
                    for row in db.execute("SELECT kind,key,dataset_id,metadata,sha256,tag FROM records WHERE kind IN ('result','snapshot')"):
                        if key in self._metadata_verified(*row).get("related_dataset_ids", []):
                            linked.append((row[0], row[1]))
                    db.executemany("DELETE FROM records WHERE kind=? AND key=?", linked)
                    db.execute("DELETE FROM records WHERE dataset_id=?", (key,))
                else:
                    db.execute("DELETE FROM records WHERE kind=? AND key=?", (kind, key))
                self._commit()
            except Exception as exc:
                db.rollback()
                raise DurableWorkspaceError("The saved item could not be removed. Its previous state was retained.") from exc

    def save_dataset(self, dataset: Any, *, import_existing: bool = False) -> None:
        with self._lock:
            self._save_dataset(dataset, import_existing=import_existing)

    def _save_dataset(self, dataset: Any, *, import_existing: bool = False) -> None:
        from survival_toolkit.store import StoredDataset
        from survival_toolkit.workspace_state import _frame_digest
        if not isinstance(dataset, StoredDataset):
            raise DurableWorkspaceError("Only an internal dataset can be saved.")
        metadata = {"dataset_id": dataset.dataset_id, "filename": dataset.filename, "source": dataset.source, "created_at": dataset.created_at.isoformat(), "last_accessed": dataset.last_accessed.isoformat(), "dataset_hash": dataset.metadata.get("dataset_hash"), "rows": len(dataset.dataframe), "columns": len(dataset.dataframe.columns), "table_sha256": _frame_digest(dataset.dataframe)}
        attachment = dataset.metadata.get("attached_marker_matrix")
        metadata["attached_marker_matrix_id"] = attachment.get("matrix_id") if isinstance(attachment, dict) else None
        previous = self._read_metadata("dataset", dataset.dataset_id)
        if import_existing and previous is not None:
            existing = self.load_dataset(dataset.dataset_id)
            same_frame = (existing.dataframe.equals(dataset.dataframe)
                          and existing.dataframe.dtypes.equals(dataset.dataframe.dtypes)
                          and existing.dataframe.index.equals(dataset.dataframe.index)
                          and existing.dataframe.columns.equals(dataset.dataframe.columns))
            same_identity = (existing.filename == dataset.filename and existing.source == dataset.source
                             and existing.created_at == dataset.created_at)
            same_metadata = pickle.dumps(existing.metadata, protocol=5) == pickle.dumps(dataset.metadata, protocol=5)
            same_attrs = pickle.dumps(existing.dataframe.attrs, protocol=5) == pickle.dumps(dataset.dataframe.attrs, protocol=5)
            if not (same_frame and same_identity and same_metadata and same_attrs):
                raise DurableWorkspaceError("An older startup checkpoint conflicts with saved data. Both copies were preserved; automatic replacement was refused.")
            return
        invalidate = previous is not None and previous.get("dataset_hash") != metadata["dataset_hash"]
        self._write("dataset", dataset.dataset_id, pickle.dumps(dataset, protocol=5), metadata, dataset_id=dataset.dataset_id, invalidate=invalidate)

    def load_dataset(self, dataset_id: str) -> Any | None:
        from survival_toolkit.store import StoredDataset
        value = self._read("dataset", dataset_id)
        if value is None:
            return None
        try:
            dataset = pickle.loads(value[1])
        except Exception as exc:
            raise DurableWorkspaceError("The authenticated dataset is incompatible with this runtime.") from exc
        if not isinstance(dataset, StoredDataset) or dataset.dataset_id != dataset_id:
            raise DurableWorkspaceError("The saved dataset identity is invalid.")
        return dataset

    def list_datasets(self) -> list[dict]:
        return sorted(self._list("dataset"), key=lambda row: row["created_at"], reverse=True)

    def delete_dataset(self, dataset_id: str) -> None:
        self._delete("dataset", dataset_id, cascade=True)

    def save_matrix(self, matrix_id: str, matrix: Any, *, import_existing: bool = False) -> None:
        from survival_toolkit.marker_matrix import MarkerMatrix
        if not isinstance(matrix, MarkerMatrix):
            raise DurableWorkspaceError("Only an internal marker matrix can be saved.")
        if import_existing:
            existing = self.load_matrix(matrix_id)
            if existing is not None:
                import numpy as np
                if (existing[:4] != matrix[:4] or existing.fingerprint != matrix.fingerprint
                        or existing.id_note != matrix.id_note or existing.values.dtype != matrix.values.dtype
                        or not np.array_equal(existing.values, matrix.values, equal_nan=True)):
                    raise DurableWorkspaceError("An older startup matrix conflicts with its saved copy. Both were preserved; replacement was refused.")
                return
        self._write("matrix", matrix_id, pickle.dumps(matrix, protocol=5), {"matrix_id": matrix_id, "filename": matrix.filename, "markers": len(matrix.marker_names), "samples": len(matrix.sample_keys), "fingerprint": matrix.fingerprint})

    def load_matrix(self, matrix_id: str) -> Any | None:
        from survival_toolkit.marker_matrix import MarkerMatrix
        value = self._read("matrix", matrix_id)
        if value is None:
            return None
        try:
            matrix = pickle.loads(value[1])
        except Exception as exc:
            raise DurableWorkspaceError("The authenticated matrix is incompatible with this runtime.") from exc
        if not isinstance(matrix, MarkerMatrix):
            raise DurableWorkspaceError("The saved matrix identity is invalid.")
        matrix.values.flags.writeable = False
        return matrix

    def list_matrices(self) -> list[dict]:
        return self._list("matrix")

    def delete_matrix(self, matrix_id: str) -> None:
        self._delete("matrix", matrix_id)

    def save_result(self, dataset_id: str, route: str, payload: Any, *, source_version: str = "", controls: Any = None, expected_dataset_hash: str | None = None) -> dict:
        parent = self._read_metadata("dataset", dataset_id)
        if parent is None:
            raise DurableWorkspaceError("Save the source dataset before saving its analysis result.")
        source_hash = parent.get("dataset_hash") if expected_dataset_hash is None else expected_dataset_hash
        if source_hash != parent.get("dataset_hash"):
            raise DurableWorkspaceError("The source dataset changed before saving. The previous analysis was not attached to the new input.")
        raw = _json(payload).encode()
        result_id = uuid4().hex
        metadata = {"result_id": result_id, "dataset_id": dataset_id, "route": route, "source_version": source_version, "controls": controls, "created_at": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "dataset_hash": source_hash}
        if route == "/api/marker-validation" and isinstance(payload, dict):
            config = payload.get("request_config") or {}
            external_id = config.get("dataset_id") if isinstance(config, dict) else None
            metadata["related_dataset_ids"] = [external_id] if isinstance(external_id, str) and external_id != dataset_id else []
        self._write("result", result_id, raw, metadata, dataset_id=dataset_id, expected_dataset_hash=source_hash)
        return metadata

    def list_results(self, dataset_id: str | None = None) -> list[dict]:
        return sorted(self._list("result", dataset_id), key=lambda row: row["created_at"], reverse=True)

    def load_result(self, result_id: str) -> dict | None:
        value = self._read("result", result_id)
        return {**value[0], "payload": json.loads(value[1])} if value else None

    def save_snapshot(self, dataset_id: str, payload: Any) -> None:
        parent = self._read_metadata("dataset", dataset_id)
        if parent is None:
            raise DurableWorkspaceError("Save the source dataset before its workspace state.")
        related = []
        if isinstance(payload, dict):
            screen = payload.get("workspace") or {}
            validation = (screen.get("results") or {}).get("markerValidation") if isinstance(screen, dict) else None
            config = validation.get("request_config") if isinstance(validation, dict) else None
            external_id = config.get("dataset_id") if isinstance(config, dict) else None
            if isinstance(external_id, str) and external_id != dataset_id:
                related.append(external_id)
        self._write("snapshot", dataset_id, _json(payload).encode(), {"dataset_id": dataset_id, "dataset_hash": parent.get("dataset_hash"), "related_dataset_ids": related, "created_at": datetime.now(timezone.utc).isoformat()}, dataset_id=dataset_id, expected_dataset_hash=parent.get("dataset_hash"))

    def list_snapshots(self) -> list[dict]:
        return self._list("snapshot")

    def load_snapshot(self, dataset_id: str) -> Any | None:
        value = self._read("snapshot", dataset_id)
        return json.loads(value[1]) if value else None

    def save_artifact(self, key: Any, payload: Any, *, import_existing: bool = False) -> None:
        dataset_id = str(key[0]) if isinstance(key, (tuple, list)) and key else ""
        parent = self._read_metadata("dataset", dataset_id) if dataset_id else None
        if dataset_id and parent is None:
            raise DurableWorkspaceError("Save the source dataset before its fitted model.")
        raw = pickle.dumps(payload, protocol=5)
        if import_existing:
            existing = self._read("artifact", key)
            if existing is not None:
                if existing[1] != raw:
                    raise DurableWorkspaceError("An older startup model conflicts with its saved copy. Both were preserved; replacement was refused.")
                return
        self._write("artifact", key, raw, {"key": key, "dataset_id": dataset_id, "dataset_hash": parent.get("dataset_hash") if parent else None}, dataset_id=dataset_id, expected_dataset_hash=parent.get("dataset_hash") if parent else None)

    def load_artifact(self, key: Any) -> Any | None:
        value = self._read("artifact", key)
        if value is None:
            return None
        try:
            return pickle.loads(value[1])
        except Exception as exc:
            raise DurableWorkspaceError("The authenticated fitted model is incompatible with this runtime.") from exc

    def delete_artifacts(self, dataset_id: str) -> None:
        with self._lock:
            rows = self._connection().execute("SELECT key FROM records WHERE kind='artifact' AND dataset_id=?", (dataset_id,)).fetchall()
            for (key,) in rows:
                self._delete("artifact", key)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._closed = True

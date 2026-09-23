from __future__ import annotations

import copy
import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from survival_toolkit.errors import DatasetNotFoundError

_MAX_DATASETS = 10
_TTL_SECONDS = 3600  # 1 hour

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StoredDataset:
    dataset_id: str
    filename: str
    source: str
    dataframe: pd.DataFrame
    created_at: datetime
    last_accessed: datetime
    metadata: dict[str, Any]


class DatasetStore:
    """Thread-safe dataset cache.

    Expiration is enforced on every mutating read/write path, so a `get()` for one
    dataset may evict unrelated expired entries before it returns. Datasets held by an
    active :meth:`lease` (a running analysis) are never expired or LRU-evicted, and
    eviction listeners are notified after a dataset leaves the store so per-dataset
    caches can be purged.
    """

    def __init__(self, max_datasets: int = _MAX_DATASETS, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._datasets: OrderedDict[str, StoredDataset] = OrderedDict()
        self._max_datasets = max_datasets
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._leases: dict[str, int] = {}
        self._eviction_listeners: list[Callable[[str], None]] = []

    def add_eviction_listener(self, listener: Callable[[str], None]) -> None:
        """Call ``listener(dataset_id)`` when a dataset is expired, evicted, deleted or its table replaced."""

        with self._lock:
            if listener not in self._eviction_listeners:
                self._eviction_listeners.append(listener)

    def _notify_evicted(self, dataset_ids: list[str]) -> None:
        if not dataset_ids:
            return
        with self._lock:
            listeners = list(self._eviction_listeners)
        for dataset_id in dataset_ids:
            for listener in listeners:
                try:
                    listener(dataset_id)
                except Exception:  # pragma: no cover - listeners must not break the store
                    logger.exception("Dataset eviction listener failed for %s", dataset_id)

    @contextmanager
    def lease(self, dataset_id: str) -> Iterator[None]:
        """Keep ``dataset_id`` from expiring or being LRU-evicted while a job uses it.

        Releasing the lease refreshes the idle TTL, so a long run does not leave the
        dataset about to expire. Unknown ids are tolerated (the lease is then a no-op).
        """

        with self._lock:
            self._leases[dataset_id] = self._leases.get(dataset_id, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                remaining = self._leases.get(dataset_id, 0) - 1
                if remaining > 0:
                    self._leases[dataset_id] = remaining
                else:
                    self._leases.pop(dataset_id, None)
                stored = self._datasets.get(dataset_id)
                if stored is not None:
                    stored.last_accessed = datetime.now(timezone.utc)

    def contains(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._datasets

    @staticmethod
    def _copy_dataframe(dataframe: pd.DataFrame, *, copy_dataframe: bool) -> pd.DataFrame:
        if not copy_dataframe:
            return dataframe
        return dataframe.copy(deep=True)

    @staticmethod
    def _dataframe_hash(dataframe: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([int(dataframe.shape[0]), int(dataframe.shape[1])], dtype=np.int64).tobytes())
        digest.update("|".join(str(column) for column in dataframe.columns).encode("utf-8"))
        digest.update("|".join(str(dtype) for dtype in dataframe.dtypes.astype(str)).encode("utf-8"))
        try:
            hashed = pd.util.hash_pandas_object(dataframe, index=True, categorize=True).to_numpy(
                dtype=np.uint64,
                copy=False,
            )
        except TypeError:
            hashed = pd.util.hash_pandas_object(
                dataframe.astype("string"),
                index=True,
                categorize=True,
            ).to_numpy(dtype=np.uint64, copy=False)
        digest.update(np.ascontiguousarray(hashed).tobytes())
        return digest.hexdigest()[:16]

    def _evict_expired(self) -> list[str]:
        now = datetime.now(timezone.utc)
        expired = [
            key
            for key, stored in self._datasets.items()
            if not self._leases.get(key)
            and (now - stored.last_accessed).total_seconds() > self._ttl_seconds
        ]
        for key in expired:
            del self._datasets[key]
        return expired

    def _evict_lru(self) -> list[str]:
        evicted: list[str] = []
        while len(self._datasets) >= self._max_datasets:
            victim = next((key for key in self._datasets if not self._leases.get(key)), None)
            if victim is None:
                # Every stored dataset is in use by a running job; allow a temporary overflow.
                break
            del self._datasets[victim]
            evicted.append(victim)
        return evicted

    def create(
        self,
        dataframe: pd.DataFrame,
        filename: str,
        *,
        source: str = "upload",
        metadata: dict[str, Any] | None = None,
        copy_dataframe: bool = True,
    ) -> StoredDataset:
        # Hashing and copying are CPU-bound; do them before taking the store-wide lock.
        dataset_hash = self._dataframe_hash(dataframe)
        stored_dataframe = self._copy_dataframe(dataframe, copy_dataframe=copy_dataframe)
        stored_metadata = copy.deepcopy(metadata or {})
        stored_metadata["dataset_hash"] = dataset_hash
        with self._lock:
            evicted = self._evict_expired()
            evicted.extend(self._evict_lru())
            dataset_id = uuid4().hex
            created_at = datetime.now(timezone.utc)
            stored = StoredDataset(
                dataset_id=dataset_id,
                filename=filename,
                source=source,
                dataframe=stored_dataframe,
                created_at=created_at,
                last_accessed=created_at,
                metadata=stored_metadata,
            )
            self._datasets[dataset_id] = stored
            result = self._clone_stored(stored, copy_dataframe=copy_dataframe)
        self._notify_evicted(evicted)
        return result

    def _clone_stored(self, stored: StoredDataset, *, copy_dataframe: bool = True) -> StoredDataset:
        return StoredDataset(
            dataset_id=stored.dataset_id,
            filename=stored.filename,
            source=stored.source,
            dataframe=self._copy_dataframe(stored.dataframe, copy_dataframe=copy_dataframe),
            created_at=stored.created_at,
            last_accessed=stored.last_accessed,
            metadata=copy.deepcopy(stored.metadata),
        )

    def get(self, dataset_id: str, *, copy_dataframe: bool = True) -> StoredDataset:
        """Return a stored dataset.

        `copy_dataframe=False` exposes the shared in-store DataFrame for read-only
        use. Callers must treat that frame as immutable and snapshot before any
        mutation.
        """
        with self._lock:
            evicted = self._evict_expired()
            stored = self._datasets.get(dataset_id)
            if stored is not None:
                stored.last_accessed = datetime.now(timezone.utc)
                self._datasets.move_to_end(dataset_id)
                result = self._clone_stored(stored, copy_dataframe=copy_dataframe)
        self._notify_evicted(evicted)
        if stored is None:
            raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}")
        return result

    def delete(self, dataset_id: str) -> None:
        with self._lock:
            try:
                del self._datasets[dataset_id]
            except KeyError as exc:
                raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}") from exc
        self._notify_evicted([dataset_id])

    def update_dataframe(self, dataset_id: str, dataframe: pd.DataFrame, *, copy_dataframe: bool = True) -> StoredDataset:
        with self._lock:
            evicted = self._evict_expired()
            stored = self._datasets.get(dataset_id)
            if stored is not None:
                self._datasets.move_to_end(dataset_id)
                stored.dataframe = self._copy_dataframe(dataframe, copy_dataframe=copy_dataframe)
                stored.last_accessed = datetime.now(timezone.utc)
                stored.metadata = {
                    **stored.metadata,
                    "dataset_hash": self._dataframe_hash(stored.dataframe),
                }
                result = self._clone_stored(stored, copy_dataframe=copy_dataframe)
        self._notify_evicted(evicted)
        if stored is None:
            raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}")
        # The table changed, so anything cached for the old contents is stale.
        self._notify_evicted([dataset_id])
        return result

    def update_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> StoredDataset:
        with self._lock:
            evicted = self._evict_expired()
            stored = self._datasets.get(dataset_id)
            if stored is not None:
                self._datasets.move_to_end(dataset_id)
                stored.last_accessed = datetime.now(timezone.utc)
                stored.metadata = {
                    **copy.deepcopy(metadata),
                    "dataset_hash": stored.metadata.get("dataset_hash") or self._dataframe_hash(stored.dataframe),
                }
                result = self._clone_stored(stored, copy_dataframe=False)
        self._notify_evicted(evicted)
        if stored is None:
            raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}")
        return result

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._datasets)

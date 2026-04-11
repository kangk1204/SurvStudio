from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from survival_toolkit.errors import DatasetNotFoundError

_MAX_DATASETS = 10
_TTL_SECONDS = 3600  # 1 hour


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
    dataset may evict unrelated expired entries before it returns.
    """

    def __init__(self, max_datasets: int = _MAX_DATASETS, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._datasets: OrderedDict[str, StoredDataset] = OrderedDict()
        self._max_datasets = max_datasets
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

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

    def _evict_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            key
            for key, stored in self._datasets.items()
            if (now - stored.last_accessed).total_seconds() > self._ttl_seconds
        ]
        for key in expired:
            del self._datasets[key]

    def _evict_lru(self) -> None:
        while len(self._datasets) >= self._max_datasets:
            self._datasets.popitem(last=False)

    def create(
        self,
        dataframe: pd.DataFrame,
        filename: str,
        *,
        source: str = "upload",
        metadata: dict[str, Any] | None = None,
        copy_dataframe: bool = True,
    ) -> StoredDataset:
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            dataset_id = uuid4().hex
            dataset_hash = self._dataframe_hash(dataframe)
            created_at = datetime.now(timezone.utc)
            stored_metadata = copy.deepcopy(metadata or {})
            stored_metadata["dataset_hash"] = dataset_hash
            stored = StoredDataset(
                dataset_id=dataset_id,
                filename=filename,
                source=source,
                dataframe=self._copy_dataframe(dataframe, copy_dataframe=copy_dataframe),
                created_at=created_at,
                last_accessed=created_at,
                metadata=stored_metadata,
            )
            self._datasets[dataset_id] = stored
            return self._clone_stored(stored, copy_dataframe=copy_dataframe)

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
            self._evict_expired()
            try:
                stored = self._datasets[dataset_id]
            except KeyError as exc:
                raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}") from exc
            stored.last_accessed = datetime.now(timezone.utc)
            self._datasets.move_to_end(dataset_id)
            return self._clone_stored(stored, copy_dataframe=copy_dataframe)

    def delete(self, dataset_id: str) -> None:
        with self._lock:
            try:
                del self._datasets[dataset_id]
            except KeyError as exc:
                raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}") from exc

    def update_dataframe(self, dataset_id: str, dataframe: pd.DataFrame, *, copy_dataframe: bool = True) -> StoredDataset:
        with self._lock:
            self._evict_expired()
            try:
                stored = self._datasets[dataset_id]
            except KeyError as exc:
                raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}") from exc
            self._datasets.move_to_end(dataset_id)
            stored.dataframe = self._copy_dataframe(dataframe, copy_dataframe=copy_dataframe)
            stored.last_accessed = datetime.now(timezone.utc)
            stored.metadata = {
                **stored.metadata,
                "dataset_hash": self._dataframe_hash(stored.dataframe),
            }
            return self._clone_stored(stored, copy_dataframe=copy_dataframe)

    def update_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> StoredDataset:
        with self._lock:
            self._evict_expired()
            try:
                stored = self._datasets[dataset_id]
            except KeyError as exc:
                raise DatasetNotFoundError(f"Unknown dataset id: {dataset_id}") from exc
            self._datasets.move_to_end(dataset_id)
            stored.last_accessed = datetime.now(timezone.utc)
            stored.metadata = {
                **copy.deepcopy(metadata),
                "dataset_hash": stored.metadata.get("dataset_hash") or self._dataframe_hash(stored.dataframe),
            }
            return self._clone_stored(stored, copy_dataframe=False)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._datasets)

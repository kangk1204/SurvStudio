from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

_MAX_DATASETS = 10
_TTL_SECONDS = 3600  # 1 hour


@dataclass(slots=True)
class StoredDataset:
    dataset_id: str
    filename: str
    source: str
    dataframe: pd.DataFrame
    created_at: datetime
    metadata: dict[str, Any]


class DatasetStore:
    def __init__(self, max_datasets: int = _MAX_DATASETS, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._datasets: OrderedDict[str, StoredDataset] = OrderedDict()
        self._max_datasets = max_datasets
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _evict_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            key
            for key, stored in self._datasets.items()
            if (now - stored.created_at).total_seconds() > self._ttl_seconds
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
            stored = StoredDataset(
                dataset_id=dataset_id,
                filename=filename,
                source=source,
                dataframe=dataframe.copy(deep=True) if copy_dataframe else dataframe,
                created_at=datetime.now(timezone.utc),
                metadata=copy.deepcopy(metadata or {}),
            )
            self._datasets[dataset_id] = stored
            return self._clone_stored(stored, copy_dataframe=copy_dataframe)

    def _clone_stored(self, stored: StoredDataset, *, copy_dataframe: bool = True) -> StoredDataset:
        return StoredDataset(
            dataset_id=stored.dataset_id,
            filename=stored.filename,
            source=stored.source,
            dataframe=stored.dataframe.copy(deep=True) if copy_dataframe else stored.dataframe,
            created_at=stored.created_at,
            metadata=copy.deepcopy(stored.metadata),
        )

    def get(self, dataset_id: str, *, copy_dataframe: bool = True) -> StoredDataset:
        with self._lock:
            self._evict_expired()
            try:
                stored = self._datasets[dataset_id]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset id: {dataset_id}") from exc
            self._datasets.move_to_end(dataset_id)
            return self._clone_stored(stored, copy_dataframe=copy_dataframe)

    def delete(self, dataset_id: str) -> None:
        with self._lock:
            try:
                del self._datasets[dataset_id]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset id: {dataset_id}") from exc

    def update_dataframe(self, dataset_id: str, dataframe: pd.DataFrame, *, copy_dataframe: bool = True) -> StoredDataset:
        with self._lock:
            self._evict_expired()
            try:
                stored = self._datasets[dataset_id]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset id: {dataset_id}") from exc
            self._datasets.move_to_end(dataset_id)
            stored.dataframe = dataframe.copy(deep=True) if copy_dataframe else dataframe
            return self._clone_stored(stored, copy_dataframe=copy_dataframe)

    def update_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> StoredDataset:
        with self._lock:
            self._evict_expired()
            try:
                stored = self._datasets[dataset_id]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset id: {dataset_id}") from exc
            self._datasets.move_to_end(dataset_id)
            stored.metadata = copy.deepcopy(metadata)
            return self._clone_stored(stored, copy_dataframe=False)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._datasets)

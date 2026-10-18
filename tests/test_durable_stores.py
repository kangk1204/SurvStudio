from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.durable_workspace import DurableWorkspace, DurableWorkspaceError
from survival_toolkit.errors import DatasetNotFoundError, NotFoundError
from survival_toolkit.marker_matrix import MarkerMatrix, MarkerMatrixStore
from survival_toolkit.store import DatasetStore


def test_datasets_survive_ttl_and_lru_only_resident_cache_is_bounded(tmp_path):
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    store = DatasetStore(max_datasets=2, ttl_seconds=1)
    store.bind_persistence(backend)
    now = [100.0]
    store._clock = lambda: now[0]
    datasets = [store.create(pd.DataFrame({"time": [i, i + 1]}), f"cohort{i}.csv") for i in range(12)]
    assert store.count == 2
    assert len(store.list_saved_datasets()) == 12
    now[0] += 3600
    assert store.contains(datasets[0].dataset_id)
    pd.testing.assert_frame_equal(store.get(datasets[0].dataset_id).dataframe, datasets[0].dataframe)
    assert store.count <= 2
    backend.close()
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    new_store = DatasetStore(max_datasets=2)
    new_store.bind_persistence(backend)
    assert len(new_store.list_saved_datasets()) == 12
    assert new_store.get(datasets[-1].dataset_id).filename == "cohort11.csv"
    new_store.delete(datasets[0].dataset_id)
    assert not new_store.contains(datasets[0].dataset_id)
    with pytest.raises(DatasetNotFoundError):
        new_store.get(datasets[0].dataset_id)
    backend.close()


def test_binding_imports_existing_residents_without_replacing_ids(tmp_path):
    store = DatasetStore()
    before = store.create(pd.DataFrame({"value": [1.5]}), "live.csv")
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    store.bind_persistence(backend)
    assert backend.load_dataset(before.dataset_id).dataset_id == before.dataset_id
    assert store.count == 1
    store.unbind_persistence()
    backend.close()
    assert store.get(before.dataset_id).filename == "live.csv"


def test_failed_create_update_delete_and_binding_retain_residents(tmp_path, monkeypatch):
    store = DatasetStore(max_datasets=1)
    previous = store.create(pd.DataFrame({"x": [1]}), "before.csv")
    broken = Mock()
    broken.save_dataset.side_effect = DurableWorkspaceError("disk full")
    with pytest.raises(DurableWorkspaceError):
        store.bind_persistence(broken)
    assert store._persistence is None
    assert store.get(previous.dataset_id).filename == "before.csv"
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    store.bind_persistence(backend)
    monkeypatch.setattr(backend, "_commit", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(DurableWorkspaceError):
        store.create(pd.DataFrame({"x": [2]}), "new.csv")
    assert store.get(previous.dataset_id).filename == "before.csv"
    with pytest.raises(DurableWorkspaceError):
        store.update_metadata(previous.dataset_id, {"wrong": "updated"})
    assert "wrong" not in store.get(previous.dataset_id).metadata
    with pytest.raises(DurableWorkspaceError):
        store.delete(previous.dataset_id)
    assert store.contains(previous.dataset_id)
    backend.close()


def _matrix(index):
    return MarkerMatrix(f"matrix{index}.tsv", "samples_in_rows", ("gene",), ("patient",), np.array([[index]], dtype=np.float32), f"hash{index}")


def test_matrix_survives_ttl_lru_and_restart_with_same_id(tmp_path, monkeypatch):
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    matrices = MarkerMatrixStore(max_items=1, ttl_seconds=1)
    matrices.bind_persistence(backend)
    now = [100.0]
    monkeypatch.setattr("survival_toolkit.marker_matrix.time.monotonic", lambda: now[0])
    first = matrices.add(_matrix(1))
    second = matrices.add(_matrix(2))
    assert len(matrices._items) == 1
    now[0] += 3600
    assert matrices.get(first).fingerprint == "hash1"
    assert len(matrices.list_saved_matrices()) == 2
    backend.close()
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    new_store = MarkerMatrixStore(max_items=1)
    new_store.bind_persistence(backend)
    assert new_store.get(second).fingerprint == "hash2"
    new_store.remove(first)
    with pytest.raises(NotFoundError):
        new_store.get(first)
    backend.close()


def test_volatile_stores_still_expire_when_not_bound():
    store = DatasetStore(ttl_seconds=1)
    now = [100.0]
    store._clock = lambda: now[0]
    dataset = store.create(pd.DataFrame({"x": [1]}), "test.csv")
    now[0] += 2
    with pytest.raises(DatasetNotFoundError):
        store.get(dataset.dataset_id)


def test_restored_matrix_larger_than_ram_budget_does_not_expand_cache(tmp_path):
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    matrices = MarkerMatrixStore(max_items=1, max_bytes=1)
    matrices.bind_persistence(backend)
    matrix = _matrix(7)
    matrix_id = "previously-saved-matrix"
    backend.save_matrix(matrix_id, matrix)
    assert not matrices._items
    assert matrices.get(matrix_id).fingerprint == matrix.fingerprint
    assert not matrices._items
    assert backend.load_matrix(matrix_id).fingerprint == matrix.fingerprint
    backend.close()


def test_old_startup_checkpoint_never_overwrites_new_saved_table(tmp_path):
    root = tmp_path / "saved"
    backend = DurableWorkspace(root, "owner")
    live = DatasetStore()
    original = live.create(pd.DataFrame({"x": [1]}), "cohort.csv")
    live.bind_persistence(backend)
    live.update_dataframe(original.dataset_id, pd.DataFrame({"x": [2]}))
    backend.close()
    recovered = DurableWorkspace(root, "owner")
    stale = DatasetStore()
    stale._datasets[original.dataset_id] = original
    with pytest.raises(DurableWorkspaceError, match="older startup checkpoint"):
        stale.bind_persistence(recovered)
    assert stale._persistence is None
    assert stale._datasets[original.dataset_id].dataframe.x.tolist() == [1]
    assert recovered.load_dataset(original.dataset_id).dataframe.x.tolist() == [2]
    recovered.close()


def test_same_checkpoint_import_is_idempotent_and_preserves_results(tmp_path):
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    store = DatasetStore()
    dataset = store.create(pd.DataFrame({"x": [1]}), "cohort.csv", metadata={"status": "withheld"})
    store.bind_persistence(backend)
    result = backend.save_result(dataset.dataset_id, "/api/km", {"p": None})
    store.bind_persistence(backend)
    assert backend.load_result(result["result_id"])["payload"] == {"p": None}
    assert len(backend.list_datasets()) == 1
    backend.close()


def test_stale_matrix_and_model_imports_refused_without_overwrite(tmp_path):
    backend = DurableWorkspace(tmp_path / "saved", "owner")
    store = DatasetStore()
    dataset = store.create(pd.DataFrame({"x": [1]}), "cohort.csv")
    store.bind_persistence(backend)
    backend.save_matrix("fixed-id", _matrix(2))
    with pytest.raises(DurableWorkspaceError, match="older startup matrix"):
        backend.save_matrix("fixed-id", _matrix(1), import_existing=True)
    assert backend.load_matrix("fixed-id").fingerprint == "hash2"
    backend.save_artifact((dataset.dataset_id, "cox"), {"beta": 2})
    with pytest.raises(DurableWorkspaceError, match="older startup model"):
        backend.save_artifact((dataset.dataset_id, "cox"), {"beta": 1}, import_existing=True)
    assert backend.load_artifact((dataset.dataset_id, "cox")) == {"beta": 2}
    backend.close()

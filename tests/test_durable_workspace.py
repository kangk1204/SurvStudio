from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.durable_workspace import DurableWorkspace, DurableWorkspaceError
from survival_toolkit.marker_matrix import MarkerMatrix
from survival_toolkit.store import DatasetStore


@pytest.fixture
def saved(tmp_path):
    backend = DurableWorkspace(tmp_path / "saved", "local-owner")
    store = DatasetStore()
    store.bind_persistence(backend)
    dataset = store.create(pd.DataFrame({"time": [1.25, 2.5], "group": pd.Categorical(["low", "high"], categories=["high", "low"])}), "cohort.csv", metadata={"inference_status": "withheld", "reason": ["ph"]})
    yield backend, store, dataset
    backend.close()


def test_restart_preserves_types_metadata_nulls_and_no_refit(saved):
    backend, _, dataset = saved
    result = backend.save_result(dataset.dataset_id, "/api/marker-evaluation", {"p": None, "q": None, "value": 0, "status": "withheld"}, source_version="fixed-source", controls={"seed": 42})
    backend.save_snapshot(dataset.dataset_id, {"selected": ["gene"], "result_id": result["result_id"]})
    backend.save_artifact((dataset.dataset_id, "test-model"), {"coefficient": np.array([1.5]), "fit_count": 1})
    root = backend.root
    backend.close()
    restored = DurableWorkspace(root, "local-owner")
    try:
        item = restored.load_dataset(dataset.dataset_id)
        pd.testing.assert_frame_equal(item.dataframe, dataset.dataframe)
        assert item.metadata == dataset.metadata
        assert item.created_at == dataset.created_at
        assert restored.load_result(result["result_id"])["payload"] == {"p": None, "q": None, "value": 0, "status": "withheld"}
        assert restored.load_snapshot(dataset.dataset_id)["result_id"] == result["result_id"]
        assert restored.load_artifact((dataset.dataset_id, "test-model"))["fit_count"] == 1
        assert restored.list_datasets()[0]["rows"] == 2
        assert "dataframe" not in restored.list_datasets()[0]
    finally:
        restored.close()


def test_wrong_owner_and_second_writer_refused(saved):
    backend, _, _ = saved
    with pytest.raises(DurableWorkspaceError, match="already open"):
        DurableWorkspace(backend.root, "local-owner")
    root = backend.root
    backend.close()
    with pytest.raises(DurableWorkspaceError, match="owner, schema or runtime"):
        DurableWorkspace(root, "another-owner")
    reopened = DurableWorkspace(root, "local-owner")
    assert len(reopened.list_datasets()) == 1
    reopened.close()


def test_runtime_change_is_not_automatically_decoded(saved, monkeypatch):
    backend, _, _ = saved
    root = backend.root
    backend.close()
    monkeypatch.setattr("survival_toolkit.durable_workspace._runtime", lambda: {"python": "different"})
    with pytest.raises(DurableWorkspaceError, match="runtime"):
        DurableWorkspace(root, "local-owner")


def test_record_tampering_is_checked_before_pickle(saved, monkeypatch):
    backend, _, dataset = saved
    backend._connection().execute("UPDATE records SET payload=? WHERE kind='dataset'", (b"untrusted pickle",))
    monkeypatch.setattr(pickle, "loads", lambda _: pytest.fail("unverified pickle was decoded"))
    with pytest.raises(DurableWorkspaceError, match="integrity"):
        backend.load_dataset(dataset.dataset_id)
    with pytest.raises(DurableWorkspaceError, match="integrity"):
        backend.save_dataset(dataset)
    # The metadata list does not read large payloads; loading still refuses corruption.
    assert len(backend.list_datasets()) == 1


def test_metadata_tampering_rejected(saved):
    backend, _, dataset = saved
    backend._connection().execute("UPDATE records SET metadata=? WHERE kind='dataset'", (json.dumps({"filename": "changed"}),))
    with pytest.raises(DurableWorkspaceError, match="integrity"):
        backend.load_dataset(dataset.dataset_id)


def test_saved_json_preserves_surrogate_text_without_loosening_finite_values(saved):
    backend, _, dataset = saved
    text = "gene\ud800 — 유전자"
    receipt = backend.save_result(dataset.dataset_id, "/api/km", {"label": text, "p": None, "value": 0})
    assert backend.load_result(receipt["result_id"])["payload"] == {"label": text, "p": None, "value": 0}
    backend.save_snapshot(dataset.dataset_id, {"label": text})
    assert backend.load_snapshot(dataset.dataset_id) == {"label": text}
    with pytest.raises(ValueError):
        backend.save_result(dataset.dataset_id, "/api/km", {"estimate": float("nan")})


def test_metadata_updates_do_not_invalidate_outputs_due_to_pandas_caches(saved):
    backend, store, dataset = saved
    receipt = backend.save_result(dataset.dataset_id, "/api/km", {"estimate": 0.8})
    frame = store.get(dataset.dataset_id, copy_dataframe=False).dataframe
    frame["group"]
    frame.index.is_unique
    frame.columns.is_unique
    store.update_metadata(dataset.dataset_id, {"profile_cache": {"ready": True}})
    assert backend.load_result(receipt["result_id"])["payload"] == {"estimate": 0.8}


def test_explicit_external_delete_removes_linked_validation_and_screen_only(saved):
    backend, store, development = saved
    external = store.create(pd.DataFrame({"time": [3, 4]}), "external.csv")
    validation = {"request_config": {"dataset_id": external.dataset_id, "development_dataset_id": development.dataset_id}, "p": None}
    receipt = backend.save_result(development.dataset_id, "/api/marker-validation", validation)
    ordinary = backend.save_result(development.dataset_id, "/api/km", {"estimate": 0.8})
    screen = {"workspace": {"results": {"markerValidation": validation}}}
    backend.save_snapshot(development.dataset_id, screen)
    store.delete(external.dataset_id)
    assert backend.load_result(receipt["result_id"]) is None
    assert backend.load_snapshot(development.dataset_id) is None
    assert backend.load_result(ordinary["result_id"])["payload"] == {"estimate": 0.8}
    assert backend.load_dataset(development.dataset_id) is not None
    with pytest.raises(DurableWorkspaceError, match="explicitly removed"):
        backend.save_result(development.dataset_id, "/api/marker-validation", validation)
    with pytest.raises(DurableWorkspaceError, match="explicitly removed"):
        backend.save_snapshot(development.dataset_id, screen)


def test_failed_commit_rolls_back_table_and_associated_outputs(saved, monkeypatch):
    backend, store, dataset = saved
    result = backend.save_result(dataset.dataset_id, "/api/km", {"estimate": 0.8})
    backend.save_artifact((dataset.dataset_id, "cox"), {"beta": 1.0})
    backend.save_snapshot(dataset.dataset_id, {"mode": "km"})
    original = backend._commit
    monkeypatch.setattr(backend, "_commit", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(DurableWorkspaceError, match="previous saved state"):
        store.update_dataframe(dataset.dataset_id, pd.DataFrame({"time": [8, 9]}))
    monkeypatch.setattr(backend, "_commit", original)
    pd.testing.assert_frame_equal(store.get(dataset.dataset_id).dataframe, dataset.dataframe)
    pd.testing.assert_frame_equal(backend.load_dataset(dataset.dataset_id).dataframe, dataset.dataframe)
    assert backend.load_result(result["result_id"]) is not None
    assert backend.load_artifact((dataset.dataset_id, "cox")) == {"beta": 1.0}
    assert backend.load_snapshot(dataset.dataset_id) == {"mode": "km"}


def test_replacing_table_invalidates_prior_outputs_but_metadata_does_not(saved):
    backend, store, dataset = saved
    result = backend.save_result(dataset.dataset_id, "/api/km", {"estimate": 0.8})
    backend.save_artifact((dataset.dataset_id, "cox"), {"beta": 1.0})
    backend.save_snapshot(dataset.dataset_id, {"mode": "km"})
    store.update_metadata(dataset.dataset_id, {"note": "reviewed"})
    assert backend.load_result(result["result_id"]) is not None
    store.update_dataframe(dataset.dataset_id, pd.DataFrame({"time": [8, 9]}))
    assert backend.load_result(result["result_id"]) is None
    assert backend.load_artifact((dataset.dataset_id, "cox")) is None
    assert backend.load_snapshot(dataset.dataset_id) is None


def test_explicit_delete_cascades_but_not_other_dataset(saved):
    backend, store, dataset = saved
    other = store.create(pd.DataFrame({"time": [3, 4]}), "other.csv")
    result = backend.save_result(dataset.dataset_id, "/api/km", {"estimate": None})
    store.delete(dataset.dataset_id)
    assert backend.load_dataset(dataset.dataset_id) is None
    assert backend.load_result(result["result_id"]) is None
    assert backend.load_dataset(other.dataset_id) is not None


def test_threaded_results_keep_every_committed_result(saved):
    backend, _, dataset = saved
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda index: backend.save_result(dataset.dataset_id, "/api/km", {"index": index, "p": None}), range(40)))
    assert len({record["result_id"] for record in records}) == 40
    assert len(backend.list_results(dataset.dataset_id)) == 40
    assert sorted(backend.load_result(row["result_id"])["payload"]["index"] for row in records) == list(range(40))


def test_matrix_roundtrip_is_immutable(saved):
    backend, _, _ = saved
    matrix = MarkerMatrix("matrix.tsv", "samples_in_rows", ("gene",), ("1", "2"), np.array([[1], [2]], dtype=np.float32), "fingerprint")
    backend.save_matrix("matrix-id", matrix)
    restored = backend.load_matrix("matrix-id")
    assert restored.fingerprint == "fingerprint"
    assert not restored.values.flags.writeable
    backend.delete_matrix("matrix-id")
    assert backend.load_matrix("matrix-id") is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner permission policy")
@pytest.mark.parametrize("target", ["directory", "database", "key"])
def test_public_permissions_refused(tmp_path, target):
    root = tmp_path / "saved"
    backend = DurableWorkspace(root, "owner")
    backend.close()
    path = {"directory": root, "database": root / "workspace.sqlite3", "key": root / "workspace.key"}[target]
    path.chmod(0o755 if target == "directory" else 0o644)
    with pytest.raises(DurableWorkspaceError, match="only to its owner"):
        DurableWorkspace(root, "owner")


@pytest.mark.skipif(os.name != "posix", reason="symlinks can require privileges on Windows")
@pytest.mark.parametrize("target", ["directory", "database", "key", "writer"])
def test_symlinks_refused(tmp_path, target):
    root = tmp_path / "saved"
    if target == "directory":
        actual = tmp_path / "actual"
        actual.mkdir(mode=0o700)
        root.symlink_to(actual, target_is_directory=True)
    else:
        backend = DurableWorkspace(root, "owner")
        backend.close()
        name = {"database": "workspace.sqlite3", "key": "workspace.key", "writer": "writer.lock"}[target]
        original = root / name
        moved = root / (name + ".original")
        original.rename(moved)
        original.symlink_to(moved)
    with pytest.raises(DurableWorkspaceError):
        DurableWorkspace(root, "owner")


def test_missing_key_does_not_create_replacement(saved):
    backend, _, _ = saved
    root = backend.root
    backend.close()
    (root / "workspace.key").unlink()
    with pytest.raises(DurableWorkspaceError, match="key is missing"):
        DurableWorkspace(root, "local-owner")
    assert not (root / "workspace.key").exists()


@pytest.mark.skipif(os.name != "posix", reason="hard link check on POSIX")
@pytest.mark.parametrize("name", ["workspace.sqlite3", "workspace.key", "writer.lock"])
def test_hardlinked_internal_files_are_refused(tmp_path, name):
    root = tmp_path / "saved"
    backend = DurableWorkspace(root, "owner")
    backend.close()
    os.link(root / name, tmp_path / "additional-link")
    with pytest.raises(DurableWorkspaceError, match="hard links"):
        DurableWorkspace(root, "owner")


@pytest.mark.parametrize("kind", ["result", "snapshot", "artifact"])
@pytest.mark.parametrize("change", ["remove", "replace"])
def test_source_relation_is_checked_in_the_same_transaction(saved, monkeypatch, kind, change):
    backend, store, dataset = saved
    original_write = backend._write

    def raced_write(record_kind, *args, **kwargs):
        if record_kind == kind:
            monkeypatch.setattr(backend, "_write", original_write)
            if change == "remove":
                store.delete(dataset.dataset_id)
            else:
                store.update_dataframe(dataset.dataset_id, pd.DataFrame({"time": [99, 100]}))
        return original_write(record_kind, *args, **kwargs)

    monkeypatch.setattr(backend, "_write", raced_write)
    with pytest.raises(DurableWorkspaceError, match="source dataset changed or was removed"):
        if kind == "result":
            backend.save_result(dataset.dataset_id, "/api/km", {"estimate": 0.8})
        elif kind == "snapshot":
            backend.save_snapshot(dataset.dataset_id, {"mode": "km"})
        else:
            backend.save_artifact((dataset.dataset_id, "cox"), {"beta": 1})
    assert backend.list_results(dataset.dataset_id) == []
    assert backend.load_snapshot(dataset.dataset_id) is None
    assert backend.load_artifact((dataset.dataset_id, "cox")) is None


def test_process_exit_before_commit_keeps_last_completed_save(saved):
    import subprocess
    import sys

    backend, _, dataset = saved
    before = backend.save_result(dataset.dataset_id, "/api/km", {"estimate": 0.75})
    root = backend.root
    backend.close()
    script = """
import os, sys
from pathlib import Path
from survival_toolkit.durable_workspace import DurableWorkspace
workspace = DurableWorkspace(Path(sys.argv[1]), 'local-owner')
workspace._commit = lambda: os._exit(23)
workspace.save_result(sys.argv[2], '/api/km', {'estimate': 0.99})
"""
    child = subprocess.run([sys.executable, "-c", script, str(root), dataset.dataset_id], check=False, capture_output=True, text=True)
    assert child.returncode == 23, child.stderr
    restored = DurableWorkspace(root, "local-owner")
    try:
        assert len(restored.list_results(dataset.dataset_id)) == 1
        assert restored.load_result(before["result_id"])["payload"] == {"estimate": 0.75}
        assert restored.load_dataset(dataset.dataset_id) is not None
    finally:
        restored.close()


def test_listing_and_contains_never_decode_patient_tables(saved, monkeypatch):
    backend, store, dataset = saved
    store._datasets.clear()
    monkeypatch.setattr(pickle, "loads", lambda _: pytest.fail("metadata-only operation decoded data"))
    assert backend.has_dataset(dataset.dataset_id)
    assert store.contains(dataset.dataset_id)
    assert backend.list_datasets()[0]["dataset_id"] == dataset.dataset_id


def test_result_computed_from_older_table_is_not_attached_to_new_input(saved):
    backend, store, dataset = saved
    original_hash = dataset.metadata["dataset_hash"]
    store.update_dataframe(dataset.dataset_id, pd.DataFrame({"time": [8, 9]}))
    with pytest.raises(DurableWorkspaceError, match="source dataset changed"):
        backend.save_result(dataset.dataset_id, "/api/km", {"estimate": 0.8}, expected_dataset_hash=original_hash)
    assert backend.list_results(dataset.dataset_id) == []


def test_category_reference_change_invalidates_previous_outputs(saved):
    backend, store, dataset = saved
    result = backend.save_result(dataset.dataset_id, "/api/cox", {"estimate": 0.8})
    changed = dataset.dataframe.copy()
    changed["group"] = changed["group"].cat.reorder_categories(["low", "high"])
    store.update_dataframe(dataset.dataset_id, changed)
    assert backend.load_result(result["result_id"]) is None

"""Synthetic operating-workspace round trips; no real uploaded tables."""
from collections import OrderedDict
import hashlib
import json
import os
import pickle
import threading
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.store import DatasetStore
from survival_toolkit.marker_matrix import MarkerMatrix, MarkerMatrixStore
from survival_toolkit.workspace_state import (
    WorkspaceStateError, save_workspace_state, restore_workspace_state,
    write_workspace_receipt,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Private operating snapshots require POSIX ownership.")


def drained():
    return {"admission_paused": True, "active_requests": 0, "active_jobs": 0,
            "queued_jobs": 0, "active_uploads": 0}


class ArtifactCache:
    def __init__(self):
        self._items = OrderedDict(); self._sizes = {}; self._lock = threading.RLock()
        self._max_items = 8; self._max_bytes = 1024**3


def frame():
    result = pd.DataFrame({"time": [0.0, 2.0, 3.0, 4.0],
        "event": [1, 0, 1, 0],
        "category": pd.Categorical(["a", "b", "a", "b"], categories=["b", "a"], ordered=True),
        "nullable": pd.array([1, None, 3, 4], dtype="Int64")},
        index=pd.Index([4, 7, 9, 11], name="synthetic_row"))
    return result


def save(path, store, **kwargs):
    return save_workspace_state(path, dataset_store=store, owner_scope="synthetic-owner",
        source_revision="synthetic-source", quiescence_check=drained, **kwargs)


def restore(path, store, **kwargs):
    return restore_workspace_state(path, dataset_store=store, owner_scope="synthetic-owner",
        quiescence_check=drained, **kwargs)


def rewrite_payload(path, edit):
    payload = pickle.loads((path / "workspace.pkl").read_bytes())
    edit(payload)
    raw = pickle.dumps(payload, protocol=5)
    (path / "workspace.pkl").write_bytes(raw)
    receipt = json.loads((path / "receipt.json").read_text())
    receipt.update(payload_bytes=len(raw), payload_sha256=hashlib.sha256(raw).hexdigest())
    (path / "receipt.json").write_text(json.dumps(receipt))


def test_actual_frames_ids_owner_metadata_results_and_markers_survive(tmp_path):
    store = DatasetStore(ttl_seconds=50, shared_metadata_keys=("profile",))
    clock = [100.0]; store._clock = lambda: clock[0]
    metadata = {"owner": {"scope": "synthetic-owner"}, "profile": {"rows": 4},
                "derived_column_provenance": {"group": {"outcome_informed": True}},
                "review_reason": "synthetic recorded source review"}
    first = store.create(frame(), "synthetic.csv", metadata=metadata)
    second = store.create(frame(), "synthetic2.csv")
    clock[0] = 109; store.get(first.dataset_id)
    ordered = list(store._datasets)
    cache = ArtifactCache(); model = {"coef": np.array([1.0, -2.0])}
    key = (first.dataset_id, "gbs")
    cache._items[key] = {"signature": {"features": ["nullable"]},
        "result": {"_model": model, "_X_encoded": frame()[["time"]],
                   "_analysis_frame": frame(),
                   "_feature_encoder": {"median": {"nullable": 3.0}, "scale": np.array([2.0])}}}
    cache._sizes[key] = 100
    matrices = MarkerMatrixStore(); values = np.arange(8, dtype=np.float32).reshape(4, 2)
    values.setflags(write=False)
    matrix = MarkerMatrix("synthetic_matrix.csv", "samples_by_markers", ("A", "B"),
        ("s1", "s2", "s3", "s4"), values, "synthetic-fingerprint")
    matrix_id = matrices.add(matrix)
    completed = {"run": {"p": None, "q": 0.0, "inference_status": "withheld"}}
    path = tmp_path / "checkpoint"
    original_times = {k: (v.created_at, v.last_accessed) for k, v in store._datasets.items()}
    receipt = save(path, store, artifact_cache=cache, marker_store=matrices, completed_results=completed)
    assert list(store._datasets) == ordered
    assert {k: (v.created_at, v.last_accessed) for k, v in store._datasets.items()} == original_times
    assert receipt["dataset_count"] == 2 and receipt["marker_count"] == 1
    assert receipt["completed_results_available"]
    assert "synthetic.csv" not in (path / "receipt.json").read_text()
    fresh = DatasetStore(); fresh._clock = lambda: 1000.0
    listener = lambda dataset_id: None
    fresh.add_eviction_listener(listener)
    new_cache = ArtifactCache(); new_matrices = MarkerMatrixStore(); new_completed = {}
    restored = restore(path, fresh, artifact_cache=new_cache, marker_store=new_matrices,
        completed_results=new_completed)
    assert restored["restored"] and list(fresh._datasets) == ordered
    assert listener in fresh._eviction_listeners
    for dataset_id in ordered:
        pd.testing.assert_frame_equal(fresh._datasets[dataset_id].dataframe, store._datasets[dataset_id].dataframe)
        assert fresh._datasets[dataset_id].metadata == store._datasets[dataset_id].metadata
        assert (fresh._datasets[dataset_id].created_at, fresh._datasets[dataset_id].last_accessed) == original_times[dataset_id]
    assert fresh._last_used[first.dataset_id] == 1000
    assert fresh._last_used[second.dataset_id] == 991
    assert new_completed == completed
    assert list(new_matrices._items) == [matrix_id]
    np.testing.assert_array_equal(new_matrices._items[matrix_id][0].values, values)
    assert not new_matrices._items[matrix_id][0].values.flags.writeable
    pd.testing.assert_frame_equal(new_cache._items[key]["result"]["_analysis_frame"], frame())
    np.testing.assert_array_equal(new_cache._items[key]["result"]["_feature_encoder"]["scale"], [2])
    assert (path.stat().st_mode & 0o777) == 0o700
    assert all(p.stat().st_mode & 0o077 == 0 for p in path.iterdir())


@pytest.mark.parametrize("kind", ["gbs", "rsf"])
def test_fitted_survival_model_predictions_survive_restart(tmp_path, kind):
    ensemble = pytest.importorskip("sksurv.ensemble")
    rng = np.random.default_rng(70)
    x = pd.DataFrame(rng.normal(size=(60, 2)), columns=["baseline_a", "baseline_b"])
    y = np.empty(60, dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = np.arange(60) % 3 != 0; y["time"] = np.exp(rng.normal(size=60))
    cls = ensemble.GradientBoostingSurvivalAnalysis if kind == "gbs" else ensemble.RandomSurvivalForest
    model = cls(n_estimators=5, max_depth=2, random_state=8).fit(x, y)
    original = model.predict(x)
    store = DatasetStore(); dataset = store.create(x, "synthetic.csv")
    cache = ArtifactCache(); key = (dataset.dataset_id, kind)
    cache._items[key] = {"signature": {"features": list(x.columns)},
        "result": {"_model": model, "_X_encoded": x, "_analysis_frame": x,
                   "_feature_encoder": {"columns": list(x.columns)}}}
    cache._sizes[key] = 100
    path = tmp_path / kind; save(path, store, artifact_cache=cache)
    fresh = DatasetStore(); new_cache = ArtifactCache()
    restore(path, fresh, artifact_cache=new_cache)
    result = new_cache._items[key]["result"]
    np.testing.assert_array_equal(result["_model"].predict(result["_X_encoded"]), original)


def test_native_artifact_signature_and_eviction_listener_survive_restore(tmp_path):
    app_module = pytest.importorskip("survival_toolkit.app")
    source = DatasetStore(); dataset = source.create(frame(), "synthetic.csv")
    cache = app_module._MlArtifactCache(max_items=8)
    signature = {"model_type": "gbs", "features": ["time"], "random_state": 6}
    cache.remember(dataset_id=dataset.dataset_id, model_type="gbs", signature=signature,
        result={"_model": {"coefficients": [1.0]}, "_X_encoded": frame()[["time"]],
                "_analysis_frame": frame(), "_feature_encoder": {"numeric_features": ["time"]}})
    path = tmp_path / "state"; save(path, source, artifact_cache=cache)
    target = DatasetStore(); restored_cache = app_module._MlArtifactCache(max_items=8)
    target.add_eviction_listener(restored_cache.purge_dataset)
    restore(path, target, artifact_cache=restored_cache)
    assert restored_cache.get(dataset_id=dataset.dataset_id, model_type="gbs", signature=signature) is not None
    assert restored_cache.get(dataset_id=dataset.dataset_id, model_type="gbs", signature={"different": True}) is None
    target.delete(dataset.dataset_id)
    assert len(restored_cache) == 0


@pytest.mark.parametrize("key,value", [("admission_paused", False), ("queued_jobs", 1),
    ("active_jobs", 1), ("active_requests", 1), ("active_uploads", 1)])
def test_active_requests_jobs_or_uploads_refuse_capture(tmp_path, key, value):
    state = drained(); state[key] = value
    with pytest.raises(WorkspaceStateError):
        save_workspace_state(tmp_path / "state", dataset_store=DatasetStore(),
            owner_scope="synthetic", source_revision="synthetic",
            quiescence_check=lambda: state)
    assert not (tmp_path / "state").exists()


def test_active_dataset_lease_refuses_capture(tmp_path):
    store = DatasetStore(); dataset = store.create(frame(), "synthetic.csv")
    with store.lease(dataset.dataset_id), pytest.raises(WorkspaceStateError, match="leases"):
        save(tmp_path / "state", store)


def test_owner_or_version_mismatch_is_rejected_before_pickle_decode(tmp_path):
    path = tmp_path / "state"; save(path, DatasetStore())
    with patch("survival_toolkit.workspace_state.pickle.loads", side_effect=AssertionError("must not decode")):
        with pytest.raises(WorkspaceStateError, match="owner"):
            restore_workspace_state(path, dataset_store=DatasetStore(), owner_scope="different",
                quiescence_check=drained)
        with pytest.raises(WorkspaceStateError, match="versions"):
            restore(path, DatasetStore(), expected_versions={"python": "different"})


def test_hash_tampering_and_public_permissions_are_rejected(tmp_path):
    path = tmp_path / "state"; save(path, DatasetStore())
    original = (path / "workspace.pkl").read_bytes()
    (path / "workspace.pkl").write_bytes(original + b"changed")
    with pytest.raises(WorkspaceStateError, match="bytes"):
        restore(path, DatasetStore())
    (path / "workspace.pkl").write_bytes(original)
    (path / "workspace.pkl").chmod(0o644)
    with pytest.raises(WorkspaceStateError, match="private"):
        restore(path, DatasetStore())


def test_existing_targets_and_checkpoint_are_not_overwritten(tmp_path):
    path = tmp_path / "state"; source = DatasetStore(); source.create(frame(), "synthetic.csv")
    save(path, source)
    with pytest.raises(WorkspaceStateError, match="new checkpoint"):
        save(path, source)
    target = DatasetStore(); original = target.create(frame(), "new-synthetic.csv")
    with pytest.raises(WorkspaceStateError, match="empty"):
        restore(path, target)
    assert list(target._datasets) == [original.dataset_id]


def test_bad_cache_config_does_not_partially_restore(tmp_path):
    path = tmp_path / "state"; source = DatasetStore(); source.create(frame(), "synthetic.csv")
    save(path, source)
    rewrite_payload(path, lambda p: p["store_config"].update(ttl_seconds=-1))
    target = DatasetStore()
    with pytest.raises(WorkspaceStateError, match="configuration"):
        restore(path, target)
    assert target.count == 0


def test_frame_corruption_is_rejected_even_if_outer_receipt_hash_is_updated(tmp_path):
    path = tmp_path / "state"; source = DatasetStore(); source.create(frame(), "synthetic.csv")
    save(path, source)
    def alter(payload):
        data = payload["datasets"][0]["dataframe"]
        data.loc[:, "time"] = 10.0
    rewrite_payload(path, alter)
    target = DatasetStore()
    with pytest.raises(WorkspaceStateError, match="integrity"):
        restore(path, target)
    assert target.count == 0


def test_payload_symlink_is_rejected_before_decoding(tmp_path):
    path = tmp_path / "state"; save(path, DatasetStore())
    actual = path / "original.pkl"; (path / "workspace.pkl").rename(actual)
    (path / "workspace.pkl").symlink_to(actual)
    with pytest.raises(WorkspaceStateError, match="symlinks"):
        restore(path, DatasetStore())


def test_no_checkpoint_component_may_be_silently_dropped(tmp_path):
    path = tmp_path / "state"; source = DatasetStore(); source.create(frame(), "synthetic.csv")
    save(path, source, completed_results={"run": {"p": None}})
    target = DatasetStore()
    with pytest.raises(WorkspaceStateError, match="target cache"):
        restore(path, target)
    assert target.count == 0


def test_expired_but_present_tables_are_captured_and_explicit_refresh_is_recorded(tmp_path):
    source = DatasetStore(ttl_seconds=10); source._clock = lambda: 100
    dataset = source.create(frame(), "synthetic.csv"); source._clock = lambda: 200
    path = tmp_path / "state"; save(path, source)
    assert dataset.dataset_id in source._datasets
    target = DatasetStore(); target._clock = lambda: 1000
    result = restore(path, target, idle_policy="refresh")
    assert result["idle_policy"] == "refresh" and target._last_used[dataset.dataset_id] == 1000
    assert target.get(dataset.dataset_id).created_at == dataset.created_at


def operator_receipt():
    return {"restored": True, "payload_sha256": "a"*64, "dataset_count": 2,
            "artifact_count": 1, "marker_count": 0, "completed_result_count": 0}


def test_operator_receipt_is_private_complete_and_never_overwritten(tmp_path):
    path = tmp_path / "restore-receipt.json"
    write_workspace_receipt(path, operator_receipt())
    assert json.loads(path.read_text()) == operator_receipt()
    assert path.stat().st_mode & 0o077 == 0
    original = path.read_bytes()
    with pytest.raises(WorkspaceStateError, match="new operator"):
        write_workspace_receipt(path, operator_receipt())
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_operator_receipt_rejects_public_parent_and_symlink_destination(tmp_path):
    public = tmp_path / "public"; public.mkdir(mode=0o755)
    with pytest.raises(WorkspaceStateError, match="private"):
        write_workspace_receipt(public / "receipt.json", operator_receipt())
    path = tmp_path / "receipt.json"; target = tmp_path / "kept.json"
    target.write_text("keep"); path.symlink_to(target)
    with pytest.raises(WorkspaceStateError, match="new operator"):
        write_workspace_receipt(path, operator_receipt())
    assert target.read_text() == "keep"


def test_operator_receipt_rejects_patient_rows_and_raw_dataset_ids(tmp_path):
    for key in ["rows", "dataset_ids", "dataset_id"]:
        value = operator_receipt(); value[key] = ["synthetic"]
        with pytest.raises(WorkspaceStateError, match="aggregate"):
            write_workspace_receipt(tmp_path / "receipt.json", value)
    assert not list(tmp_path.iterdir())
    value = operator_receipt(); value["dataset_ids_and_metadata_preserved"] = ["synthetic-id"]
    with pytest.raises(WorkspaceStateError, match="boolean"):
        write_workspace_receipt(tmp_path / "receipt.json", value)
    value = operator_receipt(); value["runtime_versions"] = {"patient_ids": ["synthetic-id"]}
    with pytest.raises(WorkspaceStateError, match="versions"):
        write_workspace_receipt(tmp_path / "receipt.json", value)


def test_operator_receipt_cannot_overwrite_a_concurrent_creator(tmp_path):
    path = tmp_path / "receipt.json"
    original_link = os.link
    def race(source, destination, **kwargs):
        path.write_text("concurrent operator receipt")
        return original_link(source, destination, **kwargs)
    with patch("survival_toolkit.workspace_state.os.link", side_effect=race):
        with pytest.raises(WorkspaceStateError, match="not overwritten"):
            write_workspace_receipt(path, operator_receipt())
    assert path.read_text() == "concurrent operator receipt"
    assert list(tmp_path.iterdir()) == [path]

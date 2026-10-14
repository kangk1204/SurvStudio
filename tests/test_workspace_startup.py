"""Startup restores optional operator state before serving requests."""
from importlib import import_module
import os
import pytest

from fastapi.testclient import TestClient
import pandas as pd


def test_default_startup_does_not_restore_or_expose_a_checkpoint(monkeypatch):
    module = import_module("survival_toolkit.app")
    state = import_module("survival_toolkit.workspace_state")
    monkeypatch.delenv("SURVSTUDIO_WORKSPACE_RESTORE", raising=False)
    monkeypatch.setattr(state, "restore_workspace_state", lambda *a, **k: pytest.fail("Not configured"))
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/workspace-state").status_code == 404


def test_operator_restore_happens_before_health_and_preserves_owner_binding(monkeypatch):
    module = import_module("survival_toolkit.app")
    state = import_module("survival_toolkit.workspace_state")
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_RESTORE", "/private/synthetic-checkpoint")
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_OWNER", "synthetic-owner")
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_IDLE_POLICY", "refresh")
    monkeypatch.delenv("SURVSTUDIO_WORKSPACE_RESTORE_RECEIPT", raising=False)
    calls = []
    def restore(path, **kwargs):
        assert path == "/private/synthetic-checkpoint"
        assert kwargs["dataset_store"] is module.store
        assert kwargs["artifact_cache"] is module._ml_artifact_cache
        assert kwargs["marker_store"] is module.marker_matrices
        assert kwargs["owner_scope"] == "synthetic-owner"
        assert kwargs["idle_policy"] == "refresh"
        assert all(kwargs["quiescence_check"]()[key] == 0 for key in ("active_jobs", "active_requests", "queued_jobs", "active_uploads"))
        calls.append(path)
        return {"restored": True, "dataset_count": 2}
    monkeypatch.setattr(state, "restore_workspace_state", restore)
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        assert calls and client.get("/api/health").status_code == 200
        assert module.app.state.workspace_restore_receipt["dataset_count"] == 2


def test_invalid_checkpoint_prevents_startup_instead_of_serving_empty_state(monkeypatch):
    module = import_module("survival_toolkit.app")
    state = import_module("survival_toolkit.workspace_state")
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_RESTORE", "/private/synthetic-checkpoint")
    def fail(*args, **kwargs):
        raise state.WorkspaceStateError("Checkpoint bytes do not match their receipt.")
    monkeypatch.setattr(state, "restore_workspace_state", fail)
    with pytest.raises(state.WorkspaceStateError, match="receipt"):
        with TestClient(module.app, base_url="http://127.0.0.1"):
            pytest.fail("The server must not admit requests with missing workspace state")


@pytest.mark.skipif(os.name != "posix", reason="Trusted local checkpoint ownership requires POSIX")
def test_real_startup_preserves_uploaded_id_rows_and_metadata_and_writes_private_receipt(tmp_path, monkeypatch):
    from survival_toolkit.store import DatasetStore
    from survival_toolkit.marker_matrix import MarkerMatrixStore
    from survival_toolkit.workspace_state import save_workspace_state
    module = import_module("survival_toolkit.app")
    tmp_path.chmod(0o700)
    original = DatasetStore()
    frame = pd.DataFrame({"follow_up": [1., 2., 3.], "event": [1, 0, 1]})
    stored = original.create(frame, filename="synthetic-upload.csv", source="upload",
                             metadata={"inference_status": "withheld", "p_value": None, "other": 0})
    checkpoint = tmp_path / "checkpoint"
    save_workspace_state(checkpoint, dataset_store=original,
        artifact_cache=module._MlArtifactCache(max_items=8), marker_store=MarkerMatrixStore(),
        owner_scope="synthetic-owner", source_revision="synthetic-source",
        quiescence_check=lambda: dict(admission_paused=True, active_requests=0, active_jobs=0, queued_jobs=0, active_uploads=0))
    target = DatasetStore()
    target_cache = module._MlArtifactCache(max_items=8)
    target.add_eviction_listener(target_cache.purge_dataset)
    monkeypatch.setattr(module, "store", target)
    monkeypatch.setattr(module, "_ml_artifact_cache", target_cache)
    monkeypatch.setattr(module, "marker_matrices", MarkerMatrixStore())
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_RESTORE", str(checkpoint))
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_OWNER", "synthetic-owner")
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_IDLE_POLICY", "refresh")
    receipt = tmp_path / "restored.json"
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_RESTORE_RECEIPT", str(receipt))
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/health").status_code == 200
        restored = target.get(stored.dataset_id)
        pd.testing.assert_frame_equal(restored.dataframe, frame)
        assert restored.filename == "synthetic-upload.csv"
        assert restored.metadata["p_value"] is None and restored.metadata["other"] == 0
        assert restored.metadata["inference_status"] == "withheld"
    assert receipt.exists() and receipt.stat().st_mode & 0o777 == 0o600

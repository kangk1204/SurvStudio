from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from survival_toolkit.durable_workspace import DurableWorkspaceError
from survival_toolkit.marker_matrix import MarkerMatrixStore
from survival_toolkit.store import DatasetStore


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("survival_toolkit.app")
    def reset():
        monkeypatch.setattr(module, "store", DatasetStore(max_datasets=2, ttl_seconds=1))
        monkeypatch.setattr(module, "marker_matrices", MarkerMatrixStore())
        monkeypatch.setattr(module, "_ml_artifact_cache", module._MlArtifactCache(max_items=2))
    reset()
    return module, reset


def _upload(client):
    text = "time,event,group,age\n" + "\n".join(f"{i + 1},{int(i % 3 != 0)},{'A' if i % 2 else 'B'},{40 + i}" for i in range(24))
    response = client.post("/api/upload", files={"file": ("cohort.csv", text, "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def _km(client, dataset):
    response = client.post("/api/kaplan-meier", json={"dataset_id": dataset["dataset_id"], "time_column": "time", "event_column": "event", "event_positive_value": 1, "group_column": "group"})
    assert response.status_code == 200, response.text
    return response.json()


def test_completed_result_survives_memory_eviction_and_fresh_lifespan(api):
    module, reset = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        result = _km(client, dataset)
        result_id = result.pop("saved_result")["result_id"]
        module.store._last_used[dataset["dataset_id"]] -= 10
        _upload(client)
        record = client.get(f"/api/saved-results/{result_id}").json()
        assert record["payload"] == result
        assert client.get(f"/api/dataset/{dataset['dataset_id']}").status_code == 200
    reset()
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rows = client.get("/api/saved-workspaces").json()
        assert rows["autosave"] is True
        assert any(row["dataset_id"] == dataset["dataset_id"] and row["result_count"] == 1 for row in rows["datasets"])
        restored = client.get(f"/api/saved-workspaces/{dataset['dataset_id']}").json()
        assert restored["dataset"]["dataset_hash"] == dataset["dataset_hash"]
        assert client.get(f"/api/saved-results/{result_id}").json()["payload"] == result
        assert "source_sha256" in restored["results"][0]["source_version"]


def test_screen_snapshot_preserves_null_zero_settings_and_delete_cascades(api):
    module, reset = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        result = _km(client, dataset)
        snapshot = {"schema_version": 1, "workspace": {"dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"], "controls": {"markerDiagnosticPolicy": "v2_holm", "markerRandomSeed": 42}, "results": {"markers": {"analysis": {"inference": {"status": "withheld", "allowed": False}, "p_value": None, "exploratory_p": 0}}}, "compareCache": {"ml": None, "dl": None, "unified": None}}}
        saved = client.post(f"/api/saved-workspaces/{dataset['dataset_id']}/snapshot", json=snapshot)
        assert saved.status_code == 200, saved.text
    reset()
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        restored = client.get(f"/api/saved-workspaces/{dataset['dataset_id']}").json()
        assert restored["snapshot"] == snapshot
        assert client.delete(f"/api/dataset/{dataset['dataset_id']}").status_code == 200
        assert client.get(f"/api/saved-results/{result['saved_result']['result_id']}").status_code == 404
        assert client.get(f"/api/dataset/{dataset['dataset_id']}").status_code == 404


def test_failed_result_write_is_not_acknowledged_as_saved(api, monkeypatch):
    module, _ = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        _km(client, dataset)
        persistence = module.app.state.durable_workspace
        def disk_full(*args, **kwargs):
            raise DurableWorkspaceError("disk full")
        monkeypatch.setattr(persistence, "save_result", disk_full)
        response = client.post("/api/kaplan-meier", json={"dataset_id": dataset["dataset_id"], "time_column": "time", "event_column": "event", "event_positive_value": 1})
        assert response.status_code == 503
        assert "not confirmed as saved" in response.json()["detail"]
        assert len(persistence.list_results(dataset["dataset_id"])) == 1


@pytest.mark.parametrize("patch", [{"dataset_id": "different"}, {"dataset_hash": "old"}, {"__proto__": {}}, {"results": {"dataset": {}}}])
def test_screen_snapshot_rejects_identity_or_unknown_sections(api, patch):
    module, _ = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        workspace = {"dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"], "controls": {}, "results": {}, **patch}
        response = client.post(f"/api/saved-workspaces/{dataset['dataset_id']}/snapshot", json={"schema_version": 1, "workspace": workspace})
        assert response.status_code in (400, 409)


def test_volatile_optout_is_explicit(api, monkeypatch):
    module, _ = api
    monkeypatch.setenv("SURVSTUDIO_AUTOSAVE", "off")
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/saved-workspaces").json() == {"autosave": False, "datasets": []}
        assert client.get("/api/saved-results/unknown").status_code == 409


def test_fitted_artifact_restores_without_refitting(api):
    module, reset = api
    signature = {"features": ["age"], "random_state": 42}
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        module._ml_artifact_cache.remember(dataset_id=dataset["dataset_id"], model_type="rsf", signature=signature,
                                           result={"_model": {"fixed": [0.0, 0.5]}, "_feature_encoder": {"levels": ["A", "B"]}})
    reset()
    with TestClient(module.app, base_url="http://127.0.0.1"):
        restored = module._ml_artifact_cache.get(dataset_id=dataset["dataset_id"], model_type="rsf", signature=signature)
        assert restored["_model"] == {"fixed": [0.0, 0.5]}
        assert module._ml_artifact_cache.get(dataset_id=dataset["dataset_id"], model_type="rsf", signature={"different": True}) is None


def test_partial_startup_failure_releases_writer_and_store_bindings(api, monkeypatch, tmp_path):
    from survival_toolkit.durable_workspace import DurableWorkspace
    module, _ = api
    def fail_binding(_):
        raise DurableWorkspaceError("injected marker restore failure")
    monkeypatch.setattr(module.marker_matrices, "bind_persistence", fail_binding)
    with pytest.raises(DurableWorkspaceError, match="injected marker"):
        with TestClient(module.app, base_url="http://127.0.0.1"):
            pass
    assert module.store._persistence is None
    assert module._ml_artifact_cache._persistence is None
    backend = DurableWorkspace(tmp_path / "saved-work", "synthetic-test-owner")
    try:
        assert backend.list_datasets() == []
    finally:
        backend.close()


def test_matrix_attachment_restores_before_any_browser_snapshot(api):
    module, reset = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        matrix = "sample,GeneA,GeneB\n" + "\n".join(f"{i + 1},{0.1 * i},{1 + 0.2 * i}" for i in range(24))
        response = client.post("/api/marker-matrix", data={"dataset_id": dataset["dataset_id"], "id_column": "time", "orientation": "samples_in_rows"}, files={"file": ("matrix.csv", matrix, "text/csv")})
        assert response.status_code == 200, response.text
        attached = response.json()
        assert client.get(f"/api/dataset/{dataset['dataset_id']}").json()["attached_marker_matrix"] == attached
    reset()
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        restored = client.get(f"/api/saved-workspaces/{dataset['dataset_id']}").json()
        assert restored["snapshot"] is None
        assert restored["dataset"]["attached_marker_matrix"] == attached
        assert client.delete(f"/api/marker-matrix/{attached['matrix_id']}?dataset_id={dataset['dataset_id']}").status_code == 200
        assert "attached_marker_matrix" not in client.get(f"/api/dataset/{dataset['dataset_id']}").json()


@pytest.mark.parametrize("patch", [{"pattern": []}, {"pattern": {"unknown": 1}}, {"benchmarkIntervals": {"status": "loading"}}, {"resultPreference": {"ml": {}}}, {"predictiveFamily": {}}, {"tab": {}}, {"workbenchRevealed": "false"}, {"results": {"ml": {"request_config": {"dataset_id": "other"}}}}])
def test_screen_snapshot_rejects_malformed_nested_state(api, patch):
    module, _ = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        workspace = {"dataset_id": dataset["dataset_id"], "dataset_hash": dataset["dataset_hash"], "controls": {}, **patch}
        response = client.post(f"/api/saved-workspaces/{dataset['dataset_id']}/snapshot", json={"schema_version": 1, "workspace": workspace})
        assert response.status_code == 400, response.text


def test_failed_matrix_detach_preserves_original_matrix(api, monkeypatch):
    module, _ = api
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        dataset = _upload(client)
        matrix = "sample,GeneA\n" + "\n".join(f"{i + 1},{0.1 * i}" for i in range(24))
        response = client.post("/api/marker-matrix", data={"dataset_id": dataset["dataset_id"], "id_column": "time", "orientation": "samples_in_rows"}, files={"file": ("matrix.csv", matrix, "text/csv")})
        assert response.status_code == 200
        matrix_id = response.json()["matrix_id"]
        def failed_update(*args, **kwargs):
            raise DurableWorkspaceError("injected metadata write failure")
        monkeypatch.setattr(module.store, "update_metadata", failed_update)
        response = client.delete(f"/api/marker-matrix/{matrix_id}?dataset_id={dataset['dataset_id']}")
        assert response.status_code == 503
        assert client.get(f"/api/marker-matrix/{matrix_id}").status_code == 200

"""Never let an application test open the user's saved workspace."""

import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_workspace_directory(tmp_path, monkeypatch):
    # Browser subprocesses inherit this directory too. A test may deliberately
    # override it when checking a restart or an unsafe-directory failure.
    monkeypatch.setenv("SURVSTUDIO_STATE_DIR", str(tmp_path / "saved-work"))
    monkeypatch.setenv("SURVSTUDIO_WORKSPACE_OWNER", "synthetic-test-owner")
    monkeypatch.delenv("SURVSTUDIO_WORKSPACE_RESTORE", raising=False)
    monkeypatch.delenv("SURVSTUDIO_WORKSPACE_RESTORE_RECEIPT", raising=False)
    # Legacy tests intentionally cache a model under a synthetic, nonexistent
    # dataset ID. Do not import that test-only orphan during a later lifespan.
    # Preserve real datasets: module-scoped browser fixtures reuse their IDs.
    module = sys.modules.get("survival_toolkit.app")
    if module is not None:
        store, cache = module.store, module._ml_artifact_cache
        assert getattr(cache, "_persistence", None) is None, "Previous test left an active workspace binding"
        with store._lock, cache._lock:
            for key in list(cache._items):
                if key[0] not in store._datasets:
                    cache._items.pop(key)
                    cache._sizes.pop(key, None)

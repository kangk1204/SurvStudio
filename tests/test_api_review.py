"""Regression tests for the API review fixes: request bounds, leakage checks on attached
marker matrices, heavy-job scheduling, decoded upload sizes, and 4xx error mapping."""

from __future__ import annotations

import asyncio
import io
import math
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.concurrency import run_in_threadpool

import survival_toolkit.app as app_module
from survival_toolkit.app import app
from survival_toolkit.errors import JobCancelledError
from survival_toolkit.sample_data import make_example_dataset
from survival_toolkit.store import DatasetStore

client = TestClient(app, base_url="http://127.0.0.1")

_FAST_MARKERS = {"n_permutations": 9, "n_resamples": 2, "random_seed": 7}


def _example_id() -> str:
    return client.post("/api/load-example").json()["dataset_id"]


def _detail(response) -> str:
    detail = response.json().get("detail")
    if isinstance(detail, list):
        return " | ".join(str(item.get("msg")) for item in detail)
    return str(detail)


# ── F1: outcome leakage checks on marker-matrix markers ─────────


def _attach_matrix(tmp_path: Path, dataset_id: str, extra: dict[str, np.ndarray] | None = None) -> str:
    frame = make_example_dataset()
    rng = np.random.default_rng(3)
    table = pd.DataFrame(rng.normal(size=(len(frame), 12)), columns=[f"GENE{index:02d}" for index in range(12)])
    table.insert(0, "patient_id", frame["patient_id"].to_numpy())
    for name, values in (extra or {}).items():
        table[name] = values
    path = tmp_path / "matrix.tsv"
    table.to_csv(path, sep="\t", index=False)
    response = client.post(
        "/api/marker-matrix",
        data={"dataset_id": dataset_id, "id_column": "patient_id", "orientation": "samples_in_rows"},
        files={"file": ("matrix.tsv", path.read_bytes(), "text/tab-separated-values")},
    )
    assert response.status_code == 200, response.text
    return response.json()["matrix_id"]


def _matrix_request(dataset_id: str, matrix_id: str, **changes) -> dict:
    return {
        "dataset_id": dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "marker_columns": [],
        "marker_matrix_id": matrix_id,
        "marker_matrix_id_column": "patient_id",
        **_FAST_MARKERS,
        **changes,
    }


def test_marker_matrix_outcome_columns_are_rejected_like_dataset_columns(tmp_path: Path) -> None:
    frame = make_example_dataset()
    dataset_id = _example_id()

    named = _attach_matrix(tmp_path, dataset_id, {"OS_time": frame["os_months"].to_numpy(), "OS": frame["os_event"].to_numpy()})
    response = client.post("/api/marker-evaluation", json=_matrix_request(dataset_id, named))
    assert response.status_code == 400, response.text
    assert "outcome" in _detail(response) and "OS_time" in _detail(response) and "OS" in _detail(response)

    # A copy of the event indicator under a gene-like name codes the same events.
    hidden = _attach_matrix(tmp_path, dataset_id, {"GENE_X": frame["os_event"].to_numpy()})
    response = client.post("/api/marker-evaluation", json=_matrix_request(dataset_id, hidden))
    assert response.status_code == 400, response.text
    assert "GENE_X" in _detail(response)

    clean = _attach_matrix(tmp_path, dataset_id)
    response = client.post("/api/marker-evaluation", json=_matrix_request(dataset_id, clean))
    assert response.status_code == 200, response.text
    assert response.json()["marker_matrix"]["n_markers"] == 12


def test_marker_evaluation_rejects_a_column_in_two_roles(tmp_path: Path) -> None:
    dataset_id = _example_id()
    base = {"dataset_id": dataset_id, "time_column": "os_months", "event_column": "os_event", **_FAST_MARKERS}

    for changes in (
        {"marker_columns": ["immune_index"], "clinical_columns": ["age"], "strata_columns": ["age"]},
        {"marker_columns": ["immune_index", "age"], "clinical_columns": ["age"]},
        {"marker_columns": ["immune_index"], "event_column": "os_months"},
    ):
        response = client.post("/api/marker-evaluation", json={**base, **changes})
        assert response.status_code == 400, (changes, response.text)
        assert "only one role" in _detail(response)

    matrix_id = _attach_matrix(tmp_path, dataset_id)
    response = client.post("/api/marker-evaluation", json=_matrix_request(dataset_id, matrix_id, clinical_columns=["patient_id"]))
    assert response.status_code == 400
    assert "patient ID" in _detail(response)


# ── F2: heavy jobs wait without holding shared threadpool workers ──


def test_queued_heavy_jobs_hold_no_shared_threadpool_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    import anyio.to_thread

    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(app_module, "_HEAVY_JOB_EXECUTOR", executor)
    dataset_id = _example_id()
    release = threading.Event()

    def _blocking_job() -> str:
        release.wait(10)
        return "done"

    async def _scenario() -> tuple[int, str, list[str]]:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 3
        heavy = [asyncio.create_task(app_module._run_dataset_job(dataset_id, _blocking_job, heavy=True)) for _ in range(6)]
        await asyncio.sleep(0.2)
        borrowed = limiter.borrowed_tokens
        # A light job (upload, dataset view, export) still gets a worker at once.
        light = await asyncio.wait_for(run_in_threadpool(lambda: "light"), timeout=5)
        release.set()
        return borrowed, light, await asyncio.gather(*heavy)

    try:
        borrowed, light, results = asyncio.run(_scenario())
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert borrowed == 0
    assert light == "light"
    assert results == ["done"] * 6


def test_queued_heavy_job_whose_client_left_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(app_module, "_HEAVY_JOB_EXECUTOR", executor)
    monkeypatch.setattr(app_module, "_DISCONNECT_POLL_SECONDS", 0.02)
    release = threading.Event()
    ran: list[str] = []

    class _GoneRequest:
        async def is_disconnected(self) -> bool:
            return True

    def _blocking_job() -> str:
        release.wait(10)
        return "done"

    def _queued_job() -> str:
        ran.append("queued")
        return "queued"

    async def _scenario() -> list[object]:
        first = asyncio.create_task(app_module._run_job(_blocking_job, heavy=True))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(app_module._run_job(_queued_job, request=_GoneRequest(), heavy=True))
        await asyncio.sleep(0.2)
        release.set()
        return await asyncio.gather(first, second, return_exceptions=True)

    try:
        results = asyncio.run(_scenario())
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert results[0] == "done"
    assert isinstance(results[1], JobCancelledError)
    assert ran == []


# ── F3: client mistakes are 4xx with a clear message ────────────


@pytest.mark.parametrize(
    ("path", "changes"),
    [
        ("/api/ml-model", {"model_type": "rsf", "max_depth": 0}),
        ("/api/ml-model", {"model_type": "gbs", "max_depth": 65}),
        ("/api/pdp", {"target_feature": "age", "max_depth": 0}),
        ("/api/counterfactual", {"target_feature": "age", "counterfactual_value": 50, "max_depth": -1}),
        ("/api/counterfactual", {"target_feature": "age", "counterfactual_value": [1, 2]}),
    ],
)
def test_model_settings_out_of_range_are_422(path: str, changes: dict) -> None:
    body = {
        "dataset_id": _example_id(),
        "time_column": "os_months",
        "event_column": "os_event",
        "features": ["age", "stage"],
        "categorical_features": ["stage"],
        "n_estimators": 10,
        **changes,
    }
    response = client.post(path, json=body)
    assert response.status_code == 422, response.text


def test_deep_model_hidden_layer_width_is_capped() -> None:
    body = {
        "dataset_id": _example_id(),
        "time_column": "os_months",
        "event_column": "os_event",
        "features": ["age"],
        "model_type": "deepsurv",
        "epochs": 10,
        "hidden_layers": [app_module._MAX_HIDDEN_LAYER_WIDTH + 1, 8],
    }
    response = client.post("/api/deep-model", json=body)
    assert response.status_code == 422
    assert "at most 1024 units" in _detail(response)
    request = app_module.DeepModelRequest(**{**body, "hidden_layers": [1024] * 20})
    assert request.hidden_layers == [1024] * 20


def test_derive_group_rejects_negative_seeds_and_cutoffs_the_method_ignores() -> None:
    dataset_id = _example_id()
    negative_seed = client.post(
        "/api/derive-group",
        json={
            "dataset_id": dataset_id,
            "source_column": "age",
            "method": "optimal_cutpoint",
            "time_column": "os_months",
            "event_column": "os_event",
            "permutation_iterations": 0,
            "random_seed": -5,
        },
    )
    assert negative_seed.status_code == 422
    ignored_cutoff = client.post(
        "/api/derive-group",
        json={"dataset_id": dataset_id, "source_column": "age", "method": "median_split", "cutoff": "150"},
    )
    assert ignored_cutoff.status_code == 422
    assert "cutoff applies only to" in _detail(ignored_cutoff)
    empty_cutoff = client.post(
        "/api/derive-group",
        json={"dataset_id": dataset_id, "source_column": "age", "method": "median_split", "cutoff": ""},
    )
    assert empty_cutoff.status_code == 200, empty_cutoff.text


def test_marker_evaluation_with_a_missing_outcome_column_is_a_400() -> None:
    dataset_id = _example_id()
    for changes in ({"event_column": "no_such_column"}, {"time_column": "no_such_column"}):
        body = {
            "dataset_id": dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "marker_columns": ["immune_index"],
            **_FAST_MARKERS,
            **changes,
        }
        response = client.post("/api/marker-evaluation", json=body)
        assert response.status_code == 400, response.text
        assert _detail(response) == "Columns not found in the dataset: no_such_column."


def test_tripod_checklist_rejects_malformed_comparison_fields() -> None:
    good = {"comparison_table": [{"model": "LASSO-Cox", "c_index": 0.7}], "n_patients": 360, "n_events": 259, "evaluation_mode": "holdout"}
    for analysis in (
        {**good, "comparison_table": "x"},
        {**good, "comparison_table": [1, 2]},
        {**good, "excluded_models": 3},
    ):
        response = client.post("/api/tripod-ai-checklist", json={"comparisons": [{"family": "ml", "analysis": analysis}]})
        assert response.status_code == 422, response.text
    features_not_a_list = {"family": "ml", "analysis": good, "request_config": {"features": 5}}
    assert client.post("/api/tripod-ai-checklist", json={"comparisons": [features_not_a_list]}).status_code == 422
    assert client.post("/api/tripod-ai-checklist", json={"comparisons": [{"family": "ml", "analysis": good}]}).status_code == 200


def test_marker_matrix_with_an_overlong_text_field_is_a_400() -> None:
    dataset_id = _example_id()
    payload = ("gene," + "P" * 200_000 + "\nG1,1\n").encode()
    response = client.post(
        "/api/marker-matrix",
        data={"dataset_id": dataset_id, "id_column": "patient_id"},
        files={"file": ("matrix.csv", payload, "text/csv")},
    )
    assert response.status_code == 400
    assert "delimited text" in _detail(response)


def test_unreadable_parquet_upload_gets_a_generic_message() -> None:
    response = client.post("/api/upload", files={"file": ("broken.parquet", b"PAR1 not a parquet file" * 10, "application/octet-stream")})
    assert response.status_code == 400
    assert _detail(response) == "Failed to read Parquet file: the file is not a valid Parquet file."


# ── Model-comparison intervals ──────────────────────────────────


def _prediction_block(n: int = 80, *, seed: int = 4, model: str = "Cox PH") -> dict:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-signal))
    censor = rng.exponential(1.5, size=n)
    return {
        "row_ids": [f"r{index}" for index in range(n)],
        "time": np.minimum(event_time, censor).round(4).tolist(),
        "event": (event_time <= censor).astype(int).tolist(),
        "risk": {model: signal.round(4).tolist()},
    }


def test_interval_blocks_are_validated_before_use() -> None:
    block = _prediction_block()
    for bad in (
        {key: value for key, value in block.items() if key != "time"},
        {**block, "risk": [0.1, 0.2]},
        {**block, "risk": {"Cox PH": block["risk"]["Cox PH"][:-1]}},
        {**block, "row_ids": 5},
        {**block, "event": [2] * len(block["row_ids"])},
        {**block, "row_ids": ["same"] * len(block["row_ids"])},
    ):
        response = client.post("/api/model-comparison-intervals", json={"predictions": [bad]})
        assert response.status_code == 422, response.text
    nan_risk = (
        b'{"predictions":[{"row_ids":["1","2","3"],"time":[1,2,3],"event":[1,0,1],'
        b'"risk":{"Cox PH":[0.1,NaN,0.3]}}]}'
    )
    response = client.post("/api/model-comparison-intervals", content=nan_risk, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "finite number" in _detail(response)


def test_interval_merge_problems_keep_their_message() -> None:
    first = _prediction_block()
    other = {**_prediction_block(model="RSF"), "row_ids": [f"x{index}" for index in range(80)]}
    response = client.post("/api/model-comparison-intervals", json={"predictions": [first, other]})
    assert response.status_code == 400
    assert "share no test patients" in _detail(response)


def test_interval_bootstrap_budget_is_a_hard_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _prediction_block()
    n_patients, n_events = len(block["row_ids"]), int(sum(block["event"]))
    work_per_draw = n_patients * n_events * 1

    exact = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 100})
    assert exact.status_code == 200
    assert exact.json()["n_bootstrap"] == 100 and "bootstrap_note" not in exact.json()

    monkeypatch.setattr(app_module, "_INTERVAL_WORK_BUDGET", work_per_draw * 150)
    capped = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 1000})
    assert capped.status_code == 200, capped.text
    payload = capped.json()
    assert payload["n_bootstrap"] == 150 and payload["n_bootstrap_requested"] == 1000
    assert "limited to 150 of the 1000" in payload["bootstrap_note"]

    monkeypatch.setattr(app_module, "_INTERVAL_WORK_BUDGET", work_per_draw * 50)
    refused = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 1000})
    assert refused.status_code == 400
    assert "too large for bootstrap intervals" in _detail(refused)


def test_interval_endpoint_runs_as_a_cancellable_heavy_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    original = app_module._run_job

    async def _recording(job, **kwargs):
        calls.append(kwargs)
        return await original(job, **kwargs)

    monkeypatch.setattr(app_module, "_run_job", _recording)
    response = client.post("/api/model-comparison-intervals", json={"predictions": [_prediction_block()], "n_bootstrap": 100})
    assert response.status_code == 200
    assert calls and calls[0]["heavy"] is True and calls[0]["request"] is not None


def test_large_validation_inputs_are_not_echoed_in_422_bodies() -> None:
    block = _prediction_block(n=400)
    del block["risk"]
    response = client.post("/api/model-comparison-intervals", json={"predictions": [block]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["input"] == "(omitted: too large to echo)"


# ── Responses stay serialisable ────────────────────────────────


def test_non_finite_and_numpy_values_in_results_become_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_checklist(comparisons, *, dataset=None):
        return {
            "nan": float("nan"),
            "values": [np.float32("inf"), np.int64(3), -math.inf],
            "flag": np.bool_(True),
            "array": np.arange(3),
            "nested": {"missing": pd.NA, "tuple": (np.float64(1.5), np.nan)},
        }

    monkeypatch.setattr(app_module, "tripod_ai_checklist", _fake_checklist)
    comparison = {"family": "ml", "analysis": {"comparison_table": []}}
    response = client.post("/api/tripod-ai-checklist", json={"comparisons": [comparison]})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "nan": None,
        "values": [None, 3, None],
        "flag": True,
        "array": [0, 1, 2],
        "nested": {"missing": None, "tuple": [1.5, None]},
    }


# ── Store: LRU order after a job and metadata copies ────────────


def test_dataset_used_by_a_finished_job_is_not_the_next_one_evicted() -> None:
    local_store = DatasetStore(max_datasets=3, ttl_seconds=3600)
    frame = pd.DataFrame({"a": [1, 2, 3]})
    first = local_store.create(frame, "A").dataset_id
    second = local_store.create(frame, "B").dataset_id
    with local_store.lease(first):  # a long analysis runs on A ...
        third = local_store.create(frame, "C").dataset_id
        local_store.get(second)
        local_store.get(third)
    local_store.create(frame, "D")
    assert local_store.contains(first)
    assert not local_store.contains(second)


def test_store_shares_cached_metadata_instead_of_deep_copying_it() -> None:
    local_store = DatasetStore(shared_metadata_keys=("cache",))
    cache = {"columns": [{"name": "a"}]}
    stored = local_store.create(pd.DataFrame({"a": [1]}), "A", metadata={"cache": cache, "provenance": {"x": {"y": 1}}})
    fetched = local_store.get(stored.dataset_id, copy_dataframe=False)
    assert fetched.metadata["cache"] is cache
    assert local_store.get(stored.dataset_id).metadata["cache"] is cache
    fetched.metadata["provenance"]["x"]["y"] = 2
    assert local_store.get(stored.dataset_id).metadata["provenance"]["x"]["y"] == 1
    assert app_module.store._shared_metadata_keys == {app_module._DATASET_PROFILE_CACHE_KEY}


# ── Upload sizes: streamed bodies and decoded text ─────────────


async def _stream_upload(path: str, *, fields: bytes, file_chunks: int, chunk: bytes) -> tuple[int | None, int]:
    head = fields + (
        b"--bnd\r\n"
        b'Content-Disposition: form-data; name="file"; filename="big.csv"\r\n'
        b"Content-Type: text/csv\r\n\r\n"
    )
    parts = [head, *([chunk] * file_chunks), b"\r\n--bnd--\r\n"]
    state = {"index": 0, "consumed": 0, "status": None}

    async def receive() -> dict:
        index = state["index"]
        if index >= len(parts):
            await asyncio.sleep(3600)
        state["index"] = index + 1
        state["consumed"] += len(parts[index])
        return {"type": "http.request", "body": parts[index], "more_body": index + 1 < len(parts)}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            state["status"] = message["status"]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8000"), (b"content-type", b"multipart/form-data; boundary=bnd")],
        "client": ("127.0.0.1", 5555),
        "server": ("127.0.0.1", 8000),
    }
    try:
        await asyncio.wait_for(app(scope, receive, send), timeout=60)
    except Exception:  # the server may still raise after the response was sent
        pass
    return state["status"], state["consumed"]


@pytest.mark.parametrize("path", ["/api/upload", "/api/marker-matrix"])
def test_upload_endpoints_stop_reading_a_streamed_body_past_the_limit(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(app_module, "_MAX_UPLOAD_BYTES", 64 * 1024)
    monkeypatch.setattr(app_module, "_UPLOAD_MULTIPART_OVERHEAD_BYTES", 16 * 1024)
    chunk = b"a,b\n" * 4096  # 16 KB
    fields = (
        b'--bnd\r\nContent-Disposition: form-data; name="dataset_id"\r\n\r\nunknown\r\n'
        b'--bnd\r\nContent-Disposition: form-data; name="id_column"\r\n\r\npatient_id\r\n'
        if path == "/api/marker-matrix"
        else b""
    )
    status, consumed = asyncio.run(_stream_upload(path, fields=fields, file_chunks=200, chunk=chunk))
    assert status == 413
    # The body is 3.2 MB; reading stops within a chunk of the 80 KB request limit.
    assert consumed <= 64 * 1024 + 16 * 1024 + 2 * len(chunk) + len(fields) + 512


def test_marker_matrix_rejects_an_oversized_content_length_before_reading() -> None:
    declared = app_module._max_upload_request_bytes() + 1
    response = client.post(
        "/api/marker-matrix",
        content=b"--x--\r\n",
        headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(declared)},
    )
    assert response.status_code == 413


def test_parquet_dictionary_text_is_measured_after_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    n_rows = 2000
    table = pa.table(
        {
            "os_months": np.linspace(1, 50, n_rows),
            "os_event": np.arange(n_rows) % 2,
            "note": ["x" * 2000] * n_rows,  # one dictionary value used by every row: 4 MB decoded
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    assert len(buffer.getvalue()) < 100_000

    def _must_not_load(*args, **kwargs):
        raise AssertionError("the decoded size must be checked before the table is loaded")

    monkeypatch.setattr(app_module, "_MAX_UPLOAD_TEXT_CHARS", 1_000_000)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    response = client.post("/api/upload", files={"file": ("notes.parquet", buffer.getvalue(), "application/octet-stream")})
    assert response.status_code == 413
    assert "of text" in _detail(response)


def test_parquet_nested_columns_are_refused_with_a_clear_message() -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"os_months": [1.0, 2.0, 3.0], "os_event": [1, 0, 1], "tags": [[1, 2], [3], []]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    response = client.post("/api/upload", files={"file": ("nested.parquet", buffer.getvalue(), "application/octet-stream")})
    assert response.status_code == 400
    assert "nested values" in _detail(response)


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _shared_string_workbook(n_rows: int = 300, note_length: int = 5000, *, prolog: str = "") -> bytes:
    """A minimal workbook whose note cells all use one shared string (openpyxl writes inline strings instead)."""
    pytest.importorskip("openpyxl")
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    package_rels = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{_PACKAGE_REL_NS}">'
        f'<Relationship Id="rId1" Type="{_REL_NS}/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    )
    workbook = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{_MAIN_NS}" xmlns:r="{_REL_NS}">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{_PACKAGE_REL_NS}">'
        f'<Relationship Id="rId1" Type="{_REL_NS}/worksheet" Target="worksheets/sheet1.xml"/>'
        f'<Relationship Id="rId2" Type="{_REL_NS}/sharedStrings" Target="sharedStrings.xml"/></Relationships>'
    )
    shared = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>{prolog}<sst xmlns="{_MAIN_NS}" count="4" uniqueCount="4">'
        f"<si><t>{'x' * note_length}</t></si><si><t>os_months</t></si><si><t>os_event</t></si><si><t>note</t></si></sst>"
    )
    rows = ['<row r="1"><c r="A1" t="s"><v>1</v></c><c r="B1" t="s"><v>2</v></c><c r="C1" t="s"><v>3</v></c></row>']
    for index in range(2, n_rows + 2):
        rows.append(
            f'<row r="{index}"><c r="A{index}"><v>{index % 97 + 1}</v></c><c r="B{index}"><v>{index % 2}</v></c>'
            f'<c r="C{index}" t="s"><v>0</v></c></row>'
        )
    sheet = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="{_MAIN_NS}"><sheetData>{"".join(rows)}</sheetData></worksheet>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def test_xlsx_shared_strings_are_measured_before_the_sheet_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _shared_string_workbook()
    original_loader = app_module.load_dataframe_from_path

    def _must_not_load(*args, **kwargs):
        raise AssertionError("the decoded size must be checked before the sheet is parsed")

    monkeypatch.setattr(app_module, "_MAX_UPLOAD_TEXT_CHARS", 1_000_000)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    for name in ("notes.xlsx", "notes.xls"):  # a workbook named .xls is still read as a zip workbook
        response = client.post("/api/upload", files={"file": (name, payload, "application/octet-stream")})
        assert response.status_code == 413, (name, response.text)
        assert "of text" in _detail(response)

    monkeypatch.setattr(app_module, "load_dataframe_from_path", original_loader)
    monkeypatch.setattr(app_module, "_MAX_UPLOAD_TEXT_CHARS", 2_000_000)
    accepted = client.post("/api/upload", files={"file": ("notes.xlsx", payload, "application/octet-stream")})
    assert accepted.status_code == 200, accepted.text


def test_parsed_table_text_is_counted_per_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    note = "x" * 1000
    frame = pd.DataFrame({"os_months": np.arange(1.0, 301.0), "os_event": np.arange(300) % 2, "note": [note] * 300})
    frame["category"] = pd.Categorical([note] * 300)
    assert app_module._dataframe_text_chars(frame, 10**9) == 600_000
    frame["text"] = pd.array([note] * 300, dtype="string")
    assert app_module._dataframe_text_chars(frame, 10**9) == 900_000
    monkeypatch.setattr(app_module, "_MAX_UPLOAD_TEXT_CHARS", 500_000)
    with pytest.raises(HTTPException) as excinfo:
        app_module._store_loaded_dataframe(frame, filename="notes.csv", source="upload")
    assert excinfo.value.status_code == 413


def test_zip_workbook_named_xls_gets_the_decompression_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_MAX_XLSX_UNCOMPRESSED_BYTES", 2048)
    response = client.post("/api/upload", files={"file": ("renamed.xls", _shared_string_workbook(), "application/octet-stream")})
    assert response.status_code == 413
    assert "decompressed" in _detail(response)


def test_workbook_xml_with_a_doctype_is_refused() -> None:
    payload = _shared_string_workbook(n_rows=20, note_length=10, prolog='<!DOCTYPE sst [<!ENTITY a "aaaa">]>')
    response = client.post("/api/upload", files={"file": ("doctype.xlsx", payload, "application/octet-stream")})
    assert response.status_code == 400
    assert "not allowed" in _detail(response)


def test_excel_headers_with_line_breaks_become_usable_column_names() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    rng = np.random.default_rng(1)
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["Overall survival\n(months)", "Death", "Age at \n diagnosis", "Tumour\nstage"])
    for index in range(120):
        worksheet.append([float(rng.exponential(30)) + 0.5, int(rng.random() < 0.6), float(rng.normal(60, 10)), ["I", "II", "III"][index % 3]])
    buffer = io.BytesIO()
    workbook.save(buffer)

    upload = client.post("/api/upload", files={"file": ("headers.xlsx", buffer.getvalue(), "application/octet-stream")})
    assert upload.status_code == 200, upload.text
    names = [column["name"] for column in upload.json()["columns"]]
    assert names == ["Overall survival (months)", "Death", "Age at diagnosis", "Tumour stage"]
    cox = client.post(
        "/api/cox",
        json={
            "dataset_id": upload.json()["dataset_id"],
            "time_column": "Overall survival (months)",
            "event_column": "Death",
            "covariates": ["Age at diagnosis"],
        },
    )
    assert cox.status_code == 200, cox.text


def test_cleaned_header_names_stay_unique() -> None:
    frame = pd.DataFrame(columns=["a\nb", "a b", "c"])
    assert list(app_module._clean_column_labels(frame).columns) == ["a b", "a b_2", "c"]


# ── Work bounds on other endpoints ─────────────────────────────


def test_export_row_keys_count_toward_the_column_cap() -> None:
    too_wide = [{f"k{index}": 1 for index in range(app_module._MAX_EXPORT_COLUMNS + 1)}]
    response = client.post("/api/export-table", json={"rows": too_wide, "format": "csv", "style": "plain"})
    assert response.status_code == 422
    assert "at most 500 columns" in _detail(response)
    fits = [{f"k{index}": 1 for index in range(app_module._MAX_EXPORT_COLUMNS)}]
    assert client.post("/api/export-table", json={"rows": fits, "format": "csv", "style": "plain"}).status_code == 200


def test_signature_search_refuses_combinatorial_explosions(monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_run(*args, **kwargs):
        raise AssertionError("an oversized search must be refused before it starts")

    monkeypatch.setattr(app_module, "discover_feature_signature", _must_not_run)
    body = {
        "dataset_id": _example_id(),
        "time_column": "os_months",
        "event_column": "os_event",
        "candidate_columns": [f"column_{index}" for index in range(200)],
        "max_combination_size": 3,
        "bootstrap_iterations": 0,
    }
    response = client.post("/api/discover-signature", json=body)
    assert response.status_code == 400
    assert "at most 1,000,000" in _detail(response)


def test_high_cardinality_categorical_models_are_refused_before_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_MAX_DESIGN_CELLS", 10_000)
    dataset_id = _example_id()
    cox = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "covariates": ["age", "patient_id"],
            "categorical_covariates": ["patient_id"],
        },
    )
    assert cox.status_code == 400
    assert "high-cardinality" in _detail(cox)
    ml = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age", "patient_id"],
            "categorical_features": ["patient_id"],
            "model_type": "lasso_cox",
        },
    )
    assert ml.status_code == 400
    assert "high-cardinality" in _detail(ml)


def test_grouped_cohort_table_caps_the_number_of_groups() -> None:
    dataset_id = _example_id()
    too_many = client.post("/api/cohort-table", json={"dataset_id": dataset_id, "variables": ["stage"], "group_column": "age"})
    assert too_many.status_code == 400
    assert "at most 50 groups" in _detail(too_many)
    grouped = client.post("/api/cohort-table", json={"dataset_id": dataset_id, "variables": ["age"], "group_column": "stage"})
    assert grouped.status_code == 200, grouped.text


def test_cohort_table_refuses_id_like_text_variables() -> None:
    dataset_id = _example_id()  # 360 distinct patient IDs
    response = client.post("/api/cohort-table", json={"dataset_id": dataset_id, "variables": ["age", "patient_id"]})
    assert response.status_code == 400
    assert "at most 200 levels" in _detail(response)


def test_marker_matrix_with_too_many_lines_is_refused_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    import gzip

    def _must_not_parse(*args, **kwargs):
        raise AssertionError("an over-long matrix must be refused before it is parsed")

    monkeypatch.setattr(app_module, "MAX_MATRIX_MARKERS", 50)
    monkeypatch.setattr(app_module, "MAX_MATRIX_SAMPLES", 100)
    monkeypatch.setattr(app_module, "read_marker_matrix", _must_not_parse)
    dataset_id = _example_id()
    lines = b"gene,P1\n" + b"a,1\n" * 500
    for name, payload in (("matrix.csv", lines), ("matrix.csv.gz", gzip.compress(lines))):
        response = client.post(
            "/api/marker-matrix",
            data={"dataset_id": dataset_id, "id_column": "patient_id"},
            files={"file": (name, payload, "application/octet-stream")},
        )
        assert response.status_code == 400, (name, response.text)
        assert "more than 101 lines" in _detail(response)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (JobCancelledError("The analysis was stopped because its request was cancelled."), 499),
        (KeyError("feature_names"), 500),
        (ValueError("SHAP could not explain this model"), 200),
    ],
)
def test_shap_step_reraises_cancellation_and_coding_errors(monkeypatch: pytest.MonkeyPatch, error: Exception, expected_status: int) -> None:
    pytest.importorskip("sksurv")
    import survival_toolkit.ml_models as ml_models

    def _failing_shap(*args, **kwargs):
        raise error

    monkeypatch.setattr(ml_models, "compute_shap_values", _failing_shap)
    lenient = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    body = {
        "dataset_id": _example_id(),
        "time_column": "os_months",
        "event_column": "os_event",
        "features": ["age", "biomarker_score"],
        "model_type": "rsf",
        "n_estimators": 10,
        "compute_shap": True,
    }
    response = lenient.post("/api/ml-model", json=body)
    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        # A data-driven SHAP failure is still reported next to the fitted model.
        assert "SHAP could not explain this model" in response.json()["shap_error"]


def test_json_ready_handles_numpy_and_missing_values() -> None:
    value = {np.int64(1): [np.float32(0.5), np.nan, np.array([[1, 2]])], "t": (pd.NaT, np.longdouble(2.5))}
    assert app_module._json_ready(value) == {1: [0.5, None, [[1, 2]]], "t": [None, 2.5]}
    started = time.perf_counter()
    app_module._json_ready({"rows": [{"a": 1.0, "b": "x"}] * 20_000})
    assert time.perf_counter() - started < 5

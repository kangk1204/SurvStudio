"""Regression tests for the second API review: upload guards, export notes and headers, error
mapping, outcome-leakage checks, request limits, and interval and validation budgets."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

import survival_toolkit.app as app_module
from survival_toolkit.app import app

client = TestClient(app, base_url="http://127.0.0.1")


def _detail(response) -> str:
    detail = response.json().get("detail")
    if isinstance(detail, list):
        return " | ".join(str(item.get("msg")) for item in detail)
    return str(detail)


def _must_not_load(*args, **kwargs):
    raise AssertionError("the upload must be refused before the table is parsed")


# ── Upload guards: workbook parts, Parquet text types, legacy Excel content ──

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_SHARED_STRINGS_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"


def _workbook(
    *,
    shared_name: str = "xl/sharedStrings.xml",
    sheet_name: str = "xl/worksheets/sheet1.xml",
    n_rows: int = 300,
    note_length: int = 5000,
    shared_prolog: str = "",
    shared_encoding: str = "utf-8",
    extra_members: int = 0,
) -> bytes:
    """A workbook whose note cells all use one shared string, with the parts stored under the given names."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f'<Override PartName="/{sheet_name}" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        f'<Override PartName="/{shared_name}" ContentType="{_SHARED_STRINGS_TYPE}"/>'
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
        f'<Relationship Id="rId1" Type="{_REL_NS}/worksheet" Target="/{sheet_name}"/>'
        f'<Relationship Id="rId2" Type="{_REL_NS}/sharedStrings" Target="/{shared_name}"/></Relationships>'
    )
    declared = "UTF-16" if shared_encoding == "utf-16" else "UTF-8"
    shared = (
        f'<?xml version="1.0" encoding="{declared}" standalone="yes"?>{shared_prolog}<sst xmlns="{_MAIN_NS}" count="4" uniqueCount="4">'
        f"<si><t>{'x' * note_length}</t></si><si><t>os_months</t></si><si><t>os_event</t></si><si><t>note</t></si></sst>"
    ).encode(shared_encoding)
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
        archive.writestr(shared_name, shared)
        archive.writestr(sheet_name, sheet)
        for index in range(extra_members):
            archive.writestr(f"xl/media/image{index}.png", b"\x89PNG\r\n\x1a\n" + bytes(16))
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("shared_name", "sheet_name"),
    [("xl/strings.bin", "xl/worksheets/sheet1.xml"), ("xl/sharedStrings.xml", "xl/worksheets/sheet1.bin"), ("xl/s.dat", "xl/w.dat")],
    ids=["shared-strings", "sheet", "both"],
)
def test_workbook_parts_under_any_name_are_measured(monkeypatch: pytest.MonkeyPatch, shared_name: str, sheet_name: str) -> None:
    pytest.importorskip("openpyxl")
    payload = _workbook(shared_name=shared_name, sheet_name=sheet_name)
    # openpyxl finds the parts through the content types and relationships, whatever their names.
    accepted = client.post("/api/upload", files={"file": ("renamed.xlsx", payload, "application/octet-stream")})
    assert accepted.status_code == 200, accepted.text
    assert [column["name"] for column in accepted.json()["columns"]] == ["os_months", "os_event", "note"]

    monkeypatch.setattr(app_module, "_MAX_UPLOAD_TEXT_CHARS", 1_000_000)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    refused = client.post("/api/upload", files={"file": ("renamed.xlsx", payload, "application/octet-stream")})
    assert refused.status_code == 413, refused.text
    assert "of text" in _detail(refused)


@pytest.mark.parametrize(
    ("shared_name", "prolog", "encoding"),
    [
        ("xl/strings.bin", '<!DOCTYPE sst [<!ENTITY a "aaaa">]>', "utf-8"),
        ("xl/sharedStrings.xml", "<!--" + " padding " * 1000 + '--><!DOCTYPE sst [<!ENTITY a "aaaa">]>', "utf-8"),
        ("xl/sharedStrings.xml", '<!DOCTYPE sst [<!ENTITY a "aaaa">]>', "utf-16"),
    ],
    ids=["part-not-named-xml", "after-a-long-comment", "utf-16"],
)
def test_workbook_document_types_are_refused_in_every_xml_part(
    monkeypatch: pytest.MonkeyPatch, shared_name: str, prolog: str, encoding: str
) -> None:
    payload = _workbook(shared_name=shared_name, n_rows=20, note_length=10, shared_prolog=prolog, shared_encoding=encoding)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    response = client.post("/api/upload", files={"file": ("doctype.xlsx", payload, "application/octet-stream")})
    assert response.status_code == 400, response.text
    assert "not allowed" in _detail(response)


def test_workbooks_with_too_many_parts_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_MAX_XLSX_MEMBERS", 8)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    payload = _workbook(n_rows=5, note_length=5, extra_members=10)
    response = client.post("/api/upload", files={"file": ("many.xlsx", payload, "application/octet-stream")})
    assert response.status_code == 400, response.text
    assert "parts" in _detail(response)


def _parquet_text_table(kind: str, n_rows: int = 2000, width: int = 2000):
    pa = pytest.importorskip("pyarrow")
    if kind == "string_view":
        values = pa.array(["x" * width] * n_rows, type=pa.string_view())
    elif kind == "binary_view":
        values = pa.array([b"y" * width] * n_rows, type=pa.binary_view())
    elif kind == "fixed_size_binary":
        values = pa.array([b"z" * width] * n_rows, type=pa.binary(width))
    elif kind == "json":
        if not hasattr(pa, "json_"):
            pytest.skip("this pyarrow has no JSON extension type")
        values = pa.array(['"' + "j" * width + '"'] * n_rows, type=pa.json_())
    else:  # pragma: no cover - test parameter typo
        raise AssertionError(kind)
    return pa.table({"os_months": np.linspace(1.0, 50.0, n_rows), "os_event": np.arange(n_rows) % 2, "note": values})


@pytest.mark.parametrize(
    ("kind", "status", "message"),
    [
        ("string_view", 413, "of text"),
        ("binary_view", 413, "of text"),
        ("fixed_size_binary", 413, "of text"),
        # Extension types cannot be read with their dictionaries kept, so they are refused by name.
        ("json", 400, "not supported"),
    ],
)
def test_parquet_text_of_every_non_numeric_type_is_measured(
    monkeypatch: pytest.MonkeyPatch, kind: str, status: int, message: str
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    buffer = io.BytesIO()
    pq.write_table(_parquet_text_table(kind), buffer, compression="zstd")
    assert len(buffer.getvalue()) < 200_000

    monkeypatch.setattr(app_module, "_MAX_UPLOAD_TEXT_CHARS", 1_000_000)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    response = client.post("/api/upload", files={"file": (f"{kind}.parquet", buffer.getvalue(), "application/octet-stream")})
    assert response.status_code == status, response.text
    assert message in _detail(response)


def test_parquet_text_measurement_counts_each_type_once_per_cell(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    for kind in ("string_view", "binary_view", "fixed_size_binary"):
        path = tmp_path / f"{kind}.parquet"
        pq.write_table(_parquet_text_table(kind, n_rows=50, width=100), path)
        assert app_module._parquet_text_bytes(path, 10**9) == 50 * 100, kind


_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def test_legacy_excel_content_is_routed_by_signature_whatever_its_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    payload = _OLE_SIGNATURE + bytes(504)
    checked: list[str] = []

    def _huge_text(path, limit):
        checked.append(str(path))
        return limit + 1

    monkeypatch.setattr(app_module, "_xls_text_chars", _huge_text)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", _must_not_load)
    refused = client.post("/api/upload", files={"file": ("legacy.xlsx", payload, "application/octet-stream")})
    assert refused.status_code == 413, refused.text
    assert checked, "the legacy workbook check must run for a legacy file named .xlsx"

    loaded = pd.DataFrame({"os_months": [1.0, 2.0, 3.0], "os_event": [1, 0, 1]})
    monkeypatch.setattr(app_module, "_xls_text_chars", lambda path, limit: 0)
    monkeypatch.setattr(app_module, "load_dataframe_from_path", lambda *args, **kwargs: loaded.copy())
    accepted = client.post("/api/upload", files={"file": ("legacy.xlsx", payload, "application/octet-stream")})
    assert accepted.status_code == 200, accepted.text


# ── Replay and provenance notes stay within the export note limit ──


def _wide_request_config(n_features: int = 250, *, name_length: int = 15) -> dict:
    return {
        "model_type": "compare",
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "features": [f"ENSG{index:0{name_length - 4}d}" for index in range(n_features)],
        "categorical_features": [f"CAT{index:0{name_length - 3}d}" for index in range(n_features)],
        "evaluation_strategy": "holdout",
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": None,
        "learning_rate": 0.1,
        "random_seed": 7,
        "epochs": 10,
        "batch_size": 32,
        "hidden_layers": [16],
        "dropout": 0.1,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
    }


def test_replay_notes_of_wide_comparisons_fit_the_export_note_limit() -> None:
    config = _wide_request_config()
    for notes in (
        app_module._ml_replay_notes(config, dataset_filename="tcga_expression.csv"),
        app_module._dl_replay_notes(config, dataset_filename="tcga_expression.csv", resolved_analysis={"evaluation_note": "x" * 5000}),
    ):
        assert notes and all(len(note) <= 4000 for note in notes), [len(note) for note in notes]
        features_note = next(note for note in notes if note.startswith("Replay features:"))
        assert "ENSG00000000000" in features_note and "250 in total" in features_note
        response = client.post(
            "/api/export-table",
            json={"rows": [{"Model": "RSF", "C-index": 0.71}], "format": "csv", "style": "journal", "notes": notes},
        )
        assert response.status_code == 200, response.text


def test_short_replay_feature_lists_are_written_in_full() -> None:
    notes = app_module._ml_replay_notes(
        {"features": ["age", "stage"], "categorical_features": ["stage"]}, dataset_filename="cohort.csv"
    )
    assert "Replay features: age, stage." in notes
    assert "Replay categorical features: stage." in notes


@pytest.mark.parametrize("fmt", ["csv", "xlsx", "markdown", "latex", "docx"])
def test_provenance_notes_of_wide_configs_are_summarised(fmt: str) -> None:
    config = _wide_request_config(1000, name_length=30)
    notes = app_module._export_provenance_notes(
        {"dataset_hash": "abc123", "request_config": config, "analysis": {"evaluation_mode": "holdout", "folds": list(range(500))}}
    )
    assert all(len(note) <= 4000 for note in notes), [len(note) for note in notes]
    replay = next(note for note in notes if note.startswith("Replay request_config:"))
    assert "1,000 in total" in replay and '"random_state": 42' in replay
    response = client.post(
        "/api/export-table",
        json={
            "rows": [{"Model": "RSF", "C-index": 0.71}],
            "format": fmt,
            "style": "journal",
            "provenance": {"dataset_hash": "abc123", "request_config": config},
        },
    )
    assert response.status_code == 200, response.text
    if fmt == "xlsx":
        from openpyxl import load_workbook

        worksheet = load_workbook(io.BytesIO(response.content)).active
        assert all(len(str(cell.value or "")) <= 4000 for row in worksheet.iter_rows() for cell in row)


def test_small_provenance_configs_are_recorded_verbatim() -> None:
    notes = app_module._export_provenance_notes({"request_config": {"model_type": "compare", "features": ["age"]}})
    assert 'Replay request_config: {"features": ["age"], "model_type": "compare"}' in notes


# ── Export writers: headers, LaTeX rows, Markdown cells, journal numbers ──


def _cohort_with_multiline_groups() -> str:
    rng = np.random.default_rng(0)
    n = 120
    histology = np.where(rng.random(n) < 0.5, "Adenocarcinoma\n(NOS)", "Squamous")
    lines = ["os_months,os_event,age,histology"]
    for index in range(n):
        lines.append(f'{rng.exponential(30) + 0.5:.2f},{int(rng.random() < 0.6)},{rng.normal(60, 10):.1f},"{histology[index]}"')
    upload = client.post("/api/upload", files={"file": ("cohort.csv", "\n".join(lines).encode(), "text/csv")})
    assert upload.status_code == 200, upload.text
    return upload.json()["dataset_id"]


@pytest.mark.parametrize("fmt", ["csv", "xlsx", "markdown", "latex", "docx"])
def test_server_produced_column_names_with_line_breaks_export_with_spaces(fmt: str) -> None:
    dataset_id = _cohort_with_multiline_groups()
    table = client.post("/api/cohort-table", json={"dataset_id": dataset_id, "variables": ["age"], "group_column": "histology"})
    assert table.status_code == 200, table.text
    analysis = table.json()["analysis"]
    assert any("\n" in column for column in analysis["columns"])

    response = client.post(
        "/api/export-table",
        json={"rows": analysis["rows"], "columns": analysis["columns"], "format": fmt, "style": "plain", "caption": "Cohort summary"},
    )
    assert response.status_code == 200, response.text
    if fmt == "xlsx":
        from openpyxl import load_workbook

        worksheet = load_workbook(io.BytesIO(response.content)).active
        text = "\n".join(str(cell.value) for row in worksheet.iter_rows() for cell in row if cell.value is not None)
    elif fmt == "docx":
        text = zipfile.ZipFile(io.BytesIO(response.content)).read("word/document.xml").decode("utf-8")
    else:
        text = response.text
    assert "Adenocarcinoma (NOS)" in text
    assert "Adenocarcinoma\n(NOS)" not in text


def test_latex_rows_cannot_be_read_as_optional_arguments() -> None:
    hazard = "Hazard\nratio\t(HR)"
    response = client.post(
        "/api/export-table",
        json={
            "rows": [{"CI": "[0.61, 0.74]", hazard: 1.2}, {"CI": "*0.50 to 0.90", hazard: 1.1}],
            "columns": ["CI", hazard],
            "format": "latex",
            "style": "plain",
        },
    )
    assert response.status_code == 200, response.text
    lines = response.text.splitlines()
    start = lines.index("\\toprule")
    end = lines.index("\\bottomrule")
    table_rows = [line for line in lines[start + 1 : end] if line != "\\midrule"]
    # "\\" and "\midrule" read a following "[" (or "*") as their own argument; an empty group ends that.
    assert table_rows and all(line.startswith("{}") for line in table_rows), table_rows
    assert table_rows[0] == "{}CI & Hazard ratio (HR) \\\\"


def _gfm_cells(row: str) -> list[str]:
    """Split a Markdown table row the way a renderer that honours every backslash escape does."""
    cells, current, escaped = [], "", False
    for character in row.strip().strip("|"):
        if escaped:
            current += character
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append(current.strip())
            current = ""
        else:
            current += character
    cells.append(current.strip())
    return cells


def test_markdown_cells_keep_pipes_backslashes_and_markup_literal() -> None:
    rows = [{"Term": "a\\|b", "Level": "x|y", "Note": "<b>bold</b> & p<0.05", "C:\\path": 1}]
    response = client.post("/api/export-table", json={"rows": rows, "format": "markdown", "style": "plain"})
    assert response.status_code == 200, response.text
    table = [line for line in response.text.splitlines() if line.startswith("|")]
    header, body = _gfm_cells(table[0]), _gfm_cells(table[2])
    assert header == ["Term", "Level", "Note", "C:\\path"]
    assert body == ["a\\|b", "x|y", "&lt;b&gt;bold&lt;/b&gt; &amp; p&lt;0.05", "1"]


def test_journal_style_formats_whole_numbers_of_measurement_columns() -> None:
    # The browser sends 1.0 as 1 and 0.0 as 0; counts and ranks stay whole numbers.
    rows = [
        {"Rank": 1, "Model": "A", "C-index": 0.7123, "P value": 0.0312, "Events, n": 45, "Training Time, ms": 1234.5},
        {"Rank": 2, "Model": "B", "C-index": 1, "P value": 1, "Events, n": 40, "Training Time, ms": 1234},
        {"Rank": 3, "Model": "C", "C-index": 0.5, "P value": 0, "Events, n": 12, "Training Time, ms": 99.0},
    ]
    csv_text = client.post("/api/export-table", json={"rows": rows, "format": "csv", "style": "journal"}).text
    import csv as csv_module

    parsed = list(csv_module.reader(io.StringIO(csv_text.lstrip("\ufeff"))))
    header_index = next(index for index, row in enumerate(parsed) if row and row[0] == "Rank")
    body = parsed[header_index + 1 :]
    assert body[1] == ["2", "B", "1.000", "1.000", "40", "1234.000"]
    assert body[2] == ["3", "C", "0.500", "<0.001", "12", "99.000"]

    from openpyxl import load_workbook

    xlsx = client.post("/api/export-table", json={"rows": rows, "format": "xlsx", "style": "journal"})
    worksheet = load_workbook(io.BytesIO(xlsx.content)).active
    second = [cell for cell in worksheet[3]]
    assert second[0].value == 2 and second[0].number_format == "General"
    assert second[2].value == 1.0 and second[2].number_format == "0.000"
    assert second[3].value == 1.0 and second[3].number_format == "0.000"
    assert second[4].value == 40 and second[4].number_format == "General"
    third = [cell for cell in worksheet[4]]
    assert third[3].value == "<0.001"


def test_journal_p_values_just_below_the_threshold_never_print_as_it() -> None:
    for value in (0.0499999951, 0.04999999999999999):
        text = app_module._format_journal_p_value(value)
        assert float(text) < 0.05, text


# ── Header clean-up keeps the names that needed none ────────────


def test_cleaned_headers_never_take_the_name_of_an_untouched_column() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["Overall survival\n(months)", "Overall survival (months)", "Death", "Age\nat\ndiagnosis"])
    for index in range(60):
        worksheet.append([1000.0 + index, float(index + 1), index % 2, 50.0 + index])
    buffer = io.BytesIO()
    workbook.save(buffer)

    upload = client.post("/api/upload", files={"file": ("headers.xlsx", buffer.getvalue(), "application/octet-stream")})
    assert upload.status_code == 200, upload.text
    names = [column["name"] for column in upload.json()["columns"]]
    assert names == ["Overall survival (months)_2", "Overall survival (months)", "Death", "Age at diagnosis"]
    frame = app_module.store.get(upload.json()["dataset_id"], copy_dataframe=False).dataframe
    assert frame["Overall survival (months)"].tolist()[:3] == [1.0, 2.0, 3.0]
    assert frame["Overall survival (months)_2"].tolist()[:3] == [1000.0, 1001.0, 1002.0]


# ── Error mapping: server faults are logged 500s, data problems 4xx ──


def _raised(exc: BaseException, *, module: str = "tests._probe") -> BaseException:
    """``exc`` raised (so it carries a traceback) from code whose module is ``module``."""
    namespace: dict = {"__name__": module}
    exec("def _raise(exc):\n    raise exc\n", namespace)
    try:
        namespace["_raise"](exc)
    except BaseException as caught:  # noqa: BLE001 - the probe re-raises whatever it is given
        return caught
    raise AssertionError("the probe did not raise")


@pytest.mark.parametrize(
    ("exc", "status", "detail"),
    [
        (RuntimeError("Could not infer dtype of numpy.object_"), 500, "unexpected internal error"),
        (RuntimeError("mat1 and mat2 shapes cannot be multiplied (64x10 and 12x64)"), 500, "unexpected internal error"),
        (RuntimeError("DataLoader worker (pid 123) is killed by signal (allocation)"), 500, "unexpected internal error"),
        (RuntimeError("CUDA error: device-side assert triggered"), 500, "unexpected internal error"),
        (RuntimeError("Expected all tensors to be on the same device, dimension 1"), 500, "unexpected internal error"),
        (TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'"), 500, "unexpected internal error"),
        (AttributeError("'NoneType' object has no attribute 'fit'"), 500, "unexpected internal error"),
        (RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"), 400, "ran out of memory"),
        (RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 8 bytes"), 400, "ran out of memory"),
        (RuntimeError("Function 'MulBackward0' returned nan values in its 0th output."), 400, "numerically unstable"),
        (ValueError("parser internals"), 400, "could not be processed"),
    ],
)
def test_request_errors_are_classified_by_type_and_whole_words(
    caplog: pytest.LogCaptureFixture, exc: Exception, status: int, detail: str
) -> None:
    from fastapi import HTTPException

    raised = _raised(exc)
    with caplog.at_level("WARNING", logger="survival_toolkit.app"):
        with pytest.raises(HTTPException) as excinfo:
            app_module.fail_bad_request(raised)
    assert excinfo.value.status_code == status
    assert detail in excinfo.value.detail
    # Every path but a user input error is logged with its traceback.
    assert any(record.exc_info and record.exc_info[1] is raised for record in caplog.records)


def test_user_input_errors_keep_their_message_and_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    from fastapi import HTTPException

    from survival_toolkit.errors import UserInputError

    with caplog.at_level("WARNING", logger="survival_toolkit.app"):
        with pytest.raises(HTTPException) as excinfo:
            app_module.fail_bad_request(_raised(UserInputError("Choose at least one marker.")))
    assert (excinfo.value.status_code, excinfo.value.detail) == (400, "Choose at least one marker.")
    assert not caplog.records


def test_import_errors_show_only_survstudio_install_hints() -> None:
    from fastapi import HTTPException

    from survival_toolkit.errors import DependencyError

    library_failure = ImportError(
        "cannot import name '_C' from 'torch' (/home/user/.venv/lib/python3.11/site-packages/torch/__init__.py)",
        name="torch",
        path="/home/user/.venv/lib/python3.11/site-packages/torch/__init__.py",
    )
    hint = "scikit-survival is required for Random Survival Forest."
    cases = [
        (_raised(library_failure), False),
        (_raised(ImportError("pip install something from /srv/private/path")), False),
        (_raised(ImportError(hint), module="survival_toolkit._import_probe"), True),
        (_raised(DependencyError(hint)), True),
    ]
    for exc, shows_message in cases:
        with pytest.raises(HTTPException) as excinfo:
            app_module.fail_bad_request(exc)
        assert excinfo.value.status_code == 503
        detail = excinfo.value.detail
        assert "/home/user" not in detail and "/srv/private" not in detail
        assert (detail == hint) is shows_message, detail
    with pytest.raises(HTTPException) as excinfo:
        app_module.fail_bad_request(_raised(library_failure))
    assert '"torch"' in excinfo.value.detail


def _shap_request() -> dict:
    return {
        "dataset_id": client.post("/api/load-example").json()["dataset_id"],
        "time_column": "os_months",
        "event_column": "os_event",
        "features": ["age", "biomarker_score"],
        "model_type": "rsf",
        "n_estimators": 10,
        "compute_shap": True,
    }


def test_shap_coding_errors_behind_the_input_boundary_are_server_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("sksurv")
    import survival_toolkit.ml_models as ml_models
    from survival_toolkit.errors import user_input_boundary

    # A TypeError raised by SurvStudio code reaches the endpoint as an InternalAnalysisError.
    namespace: dict = {"__name__": "survival_toolkit._shap_probe"}
    exec("def broken(*args, **kwargs):\n    return None + 1\n", namespace)
    monkeypatch.setattr(ml_models, "compute_shap_values", user_input_boundary(namespace["broken"]))
    lenient = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    with caplog.at_level("ERROR", logger="survival_toolkit.app"):
        response = lenient.post("/api/ml-model", json=_shap_request())
    assert response.status_code == 500, response.text
    assert "unexpected internal error" in _detail(response)
    assert any(record.exc_info for record in caplog.records)


def test_shap_data_failures_are_reported_next_to_the_model_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("sksurv")
    import survival_toolkit.ml_models as ml_models
    from survival_toolkit.errors import user_input_boundary

    # A ValueError raised inside a library is wrapped the same way but is a genuine SHAP failure.
    namespace: dict = {"__name__": "shap._probe"}
    exec("def failing(*args, **kwargs):\n    raise ValueError('The explainer could not handle this model')\n", namespace)
    monkeypatch.setattr(ml_models, "compute_shap_values", user_input_boundary(namespace["failing"]))
    with caplog.at_level("WARNING", logger="survival_toolkit.app"):
        response = client.post("/api/ml-model", json=_shap_request())
    assert response.status_code == 200, response.text
    assert response.json()["shap_error"]
    assert any("SHAP" in record.getMessage() and record.exc_info for record in caplog.records)


def _locked_recipe() -> tuple[str, dict]:
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    body = {
        "dataset_id": dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "marker_columns": ["biomarker_score", "immune_index"],
        "clinical_columns": ["age", "stage"],
        "categorical_clinical": ["stage"],
        "n_permutations": 9,
        "n_resamples": 2,
        "random_seed": 7,
    }
    response = client.post("/api/marker-evaluation", json=body)
    assert response.status_code == 200, response.text
    return dataset_id, response.json()["analysis"]["locked_recipe"]


def test_marker_rank_intervals_read_as_ranges_not_dates() -> None:
    import re

    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    body = {
        "dataset_id": dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "marker_columns": ["biomarker_score", "immune_index", "age"],
        "n_permutations": 9,
        "n_resamples": 4,
        "random_seed": 7,
    }
    response = client.post("/api/marker-evaluation", json=body)
    assert response.status_code == 200, response.text
    intervals = [row["Rank 95% interval"] for row in response.json()["display_table"]]
    # A spreadsheet opens "1-3" as a date.
    assert intervals and all(re.fullmatch(r"\d+ to \d+", value) for value in intervals), intervals


def test_incomplete_recipes_are_user_errors_but_library_lookup_errors_are_not(monkeypatch: pytest.MonkeyPatch) -> None:
    import copy

    from survival_toolkit.marker_evaluation import recipe_hash

    dataset_id, recipe = _locked_recipe()
    incomplete = copy.deepcopy(recipe)
    del incomplete["outcome"]["event_positive_value"]
    incomplete["recipe_hash"] = recipe_hash(incomplete)
    response = client.post("/api/marker-validation", json={"dataset_id": dataset_id, "recipe": incomplete, "n_bootstrap": 0})
    assert response.status_code == 400, response.text
    # The recipe check may name the missing field itself; otherwise the lookup error is mapped here.
    assert "recipe is incomplete" in _detail(response) or "locked model is malformed" in _detail(response)

    def _library_key_error(*args, **kwargs):
        return {}["missing"]  # raised outside SurvStudio code: a coding error, not a recipe problem

    monkeypatch.setattr(app_module, "validate_locked_recipe", _library_key_error)
    lenient = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    response = lenient.post("/api/marker-validation", json={"dataset_id": dataset_id, "recipe": recipe, "n_bootstrap": 0})
    assert response.status_code == 500, response.text


# ── Outcome information and outcome columns ─────────────────────


def test_a_group_cut_from_an_outcome_informed_column_is_outcome_informed() -> None:
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    # Numeric labels let a median split re-cut the outcome-informed groups; that split needs the
    # larger group coded 0, which depends on where the cutpoint falls, so both codings are tried.
    for lower_label, upper_label in (("0", "1"), ("1", "0")):
        optimal = client.post(
            "/api/derive-group",
            json={
                "dataset_id": dataset_id,
                "source_column": "biomarker_score",
                "method": "optimal_cutpoint",
                "time_column": "os_months",
                "event_column": "os_event",
                "lower_label": lower_label,
                "upper_label": upper_label,
                "permutation_iterations": 0,
            },
        )
        assert optimal.status_code == 200, optimal.text
        informed = optimal.json()["derived_column"]
        laundered = client.post(
            "/api/derive-group",
            json={"dataset_id": optimal.json()["dataset_id"], "source_column": informed, "method": "median_split", "new_column_name": "laundered"},
        )
        if laundered.status_code == 200:
            break
    assert laundered.status_code == 200, laundered.text
    payload = laundered.json()
    assert payload["derived_column_provenance"]["laundered"]["outcome_informed"] is True
    assert payload["derive_summary"]["outcome_informed"] is True
    assert payload["derive_summary"]["recipe"]["outcome_informed"] is True

    cox = client.post(
        "/api/cox",
        json={
            "dataset_id": payload["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "covariates": ["laundered"],
            "categorical_covariates": ["laundered"],
        },
    )
    assert cox.status_code == 400, cox.text
    assert "Outcome-informed" in _detail(cox)
    km = client.post(
        "/api/kaplan-meier",
        json={"dataset_id": payload["dataset_id"], "time_column": "os_months", "event_column": "os_event", "group_column": "laundered"},
    )
    assert km.status_code == 200, km.text
    assert km.json()["analysis"]["outcome_informed_group"] is True


def _example_with_event_copy() -> str:
    from survival_toolkit.sample_data import make_example_dataset

    frame = make_example_dataset()
    frame["vital"] = frame["os_event"]
    frame["bm"] = frame["biomarker_score"]
    upload = client.post("/api/upload", files={"file": ("with_copy.csv", frame.to_csv(index=False).encode(), "text/csv")})
    assert upload.status_code == 200, upload.text
    return upload.json()["dataset_id"]


@pytest.mark.parametrize("group", ["vital", "pfs_event"])
def test_kaplan_meier_refuses_outcome_columns_as_groups(group: str) -> None:
    dataset_id = _example_with_event_copy()
    response = client.post(
        "/api/kaplan-meier",
        json={"dataset_id": dataset_id, "time_column": "os_months", "event_column": "os_event", "group_column": group},
    )
    assert response.status_code == 400, response.text
    assert "Survival outcome columns cannot be used" in _detail(response) and group in _detail(response)
    grouped = client.post(
        "/api/kaplan-meier",
        json={"dataset_id": dataset_id, "time_column": "os_months", "event_column": "os_event", "group_column": "stage"},
    )
    assert grouped.status_code == 200, grouped.text


def test_optimal_cutpoint_refuses_outcome_and_outcome_informed_variables() -> None:
    dataset_id = _example_with_event_copy()
    base = {"dataset_id": dataset_id, "time_column": "os_months", "event_column": "os_event", "permutation_iterations": 0}
    copy_of_event = client.post("/api/optimal-cutpoint", json={**base, "variable": "vital"})
    assert copy_of_event.status_code == 400, copy_of_event.text
    assert "Survival outcome columns cannot be used" in _detail(copy_of_event)

    derived = client.post("/api/derive-group", json={**base, "source_column": "age", "method": "optimal_cutpoint", "lower_label": "0", "upper_label": "1"})
    assert derived.status_code == 200, derived.text
    informed = client.post(
        "/api/optimal-cutpoint",
        json={**base, "dataset_id": derived.json()["dataset_id"], "variable": derived.json()["derived_column"]},
    )
    assert informed.status_code == 400, informed.text
    assert "Outcome-informed" in _detail(informed)

    response = client.post("/api/optimal-cutpoint", json={**base, "variable": "biomarker_score"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["request_config"]["variable"] == "biomarker_score"
    assert payload["dataset_hash"] == client.get(f"/api/dataset/{dataset_id}").json()["dataset_hash"]


def test_marker_validation_checks_the_mapped_columns_for_roles_and_outcomes() -> None:
    _, recipe = _locked_recipe()
    marker = recipe["markers"][0]
    external = _example_with_event_copy()

    def validate(mapping: dict):
        return client.post(
            "/api/marker-validation", json={"dataset_id": external, "recipe": recipe, "column_mapping": mapping, "n_bootstrap": 0}
        )

    for mapping in ({marker: "os_event"}, {marker: "os_months"}, {"age": "os_months"}):
        response = validate(mapping)
        assert response.status_code == 400, (mapping, response.text)
        assert "only one role" in _detail(response), _detail(response)
    for mapping in ({marker: "vital"}, {marker: "pfs_event"}):
        response = validate(mapping)
        assert response.status_code == 400, (mapping, response.text)
        assert "Survival outcome columns cannot be used" in _detail(response)

    renamed = validate({"biomarker_score": "bm"})
    assert renamed.status_code == 200, renamed.text


@pytest.mark.parametrize(
    ("time_column", "event_column", "event_value"),
    [("os_months", "os_months", 1), ("os_months", "stage", "IX"), ("os_months", "os_event", 7)],
)
def test_cohort_table_outcome_restriction_keeps_the_validation_message(time_column: str, event_column: str, event_value) -> None:
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    outcome = {"time_column": time_column, "event_column": event_column, "event_positive_value": event_value}
    km = client.post("/api/kaplan-meier", json={"dataset_id": dataset_id, **outcome})
    table = client.post("/api/cohort-table", json={"dataset_id": dataset_id, "variables": ["age"], **outcome})
    assert km.status_code == table.status_code == 400, (km.text, table.text)
    assert _detail(table) == _detail(km)
    assert "could not be processed" not in _detail(table)


# ── Request validation: surrogates, strict coercion, number ranges ──


def _post_raw(path: str, body: str):
    lenient = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    return lenient.post(path, content=body.encode(), headers={"Content-Type": "application/json"})


def test_unpaired_surrogates_are_refused_with_a_readable_422() -> None:
    import json

    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    base = f'"dataset_id":"{dataset_id}","time_column":"os_months","event_column":"os_event"'
    cases = [
        ("/api/cox", "{" + base + r',"covariates":["age\ud800"]}'),
        ("/api/kaplan-meier", "{" + base + r',"event_positive_value":"x\ud800"}'),
        ("/api/kaplan-meier", "{" + base + r',"time_unit_label":"Mo\ud800"}'),
        ("/api/kaplan-meier", '{"dataset_id":"%s","time_column":"os_\\ud800","event_column":"os_event"}' % dataset_id),
        ("/api/export-table", r'{"rows":[{"a":1}],"format":"latex","caption":"Table \ud800 1"}'),
        ("/api/export-table", r'{"rows":[{"a":1}],"format":"x\udc00"}'),
    ]
    for path, body in cases:
        response = _post_raw(path, body)
        assert response.status_code == 422, (path, body, response.status_code, response.text[:200])
        payload = json.loads(response.content.decode("utf-8"))
        assert payload["detail"]


def test_error_details_escape_unpaired_surrogates() -> None:
    from fastapi import HTTPException

    from survival_toolkit.errors import UserInputError

    with pytest.raises(HTTPException) as excinfo:
        app_module.fail_bad_request(UserInputError('Column not found in dataset: "age\ud800".'))
    assert excinfo.value.detail == 'Column not found in dataset: "age\\ud800".'
    excinfo.value.detail.encode("utf-8")


def test_request_fields_are_coerced_strictly_with_one_column_name_rule() -> None:
    from pydantic import ValidationError

    base = {"dataset_id": "demo", "time_column": "os_months", "event_column": "os_event"}
    for build in (
        lambda: app_module.CoxRequest(**base, covariates={"age": 1, "stage": 2}),
        lambda: app_module.CoxRequest(**base, covariates=["age", None]),
        lambda: app_module.CoxRequest(**base, covariates=["age", 3]),
        lambda: app_module.DeriveGroupRequest(dataset_id="demo", source_column="age", method="median_split", lower_label=None),
        lambda: app_module.DeriveGroupRequest(dataset_id="demo", source_column="age", method="median_split", upper_label=5),
        lambda: app_module.KaplanMeierRequest(**{**base, "time_column": "os\nmonths"}),
        lambda: app_module.KaplanMeierRequest(**{**base, "event_column": None}),
        lambda: app_module.OptimalCutpointRequest(**base, variable=["age"]),
        lambda: app_module.SignatureSearchRequest(**base, candidate_columns=["age", None]),
        lambda: app_module.CohortTableRequest(dataset_id="demo", variables="age"),
        lambda: app_module.MarkerValidationRequest(dataset_id="demo", recipe={}, column_mapping={"a": 1}),
    ):
        with pytest.raises(ValidationError):
            build()

    km = app_module.KaplanMeierRequest(**{**base, "time_column": " os_months ", "group_column": "  "})
    assert (km.time_column, km.group_column) == ("os_months", None)
    table = app_module.CohortTableRequest(dataset_id="demo", variables=[" age ", "age", "stage"], time_column="")
    assert table.variables == ["age", "stage"] and table.time_column is None
    search = app_module.SignatureSearchRequest(**base, candidate_columns=[" age", "age "])
    assert search.candidate_columns == ["age"]
    validation = app_module.MarkerValidationRequest(dataset_id="demo", recipe={}, column_mapping={" GENE1 ": " gene_1 "})
    assert validation.column_mapping == {"GENE1": "gene_1"}
    derive = app_module.DeriveGroupRequest(dataset_id="demo", source_column=" age ", method="median_split", time_column="")
    assert (derive.source_column, derive.time_column) == ("age", None)


def test_event_codes_must_be_numbers_a_column_can_hold_exactly() -> None:
    from pydantic import ValidationError

    base = {"dataset_id": "demo", "time_column": "os_months", "event_column": "os_event"}
    assert app_module.KaplanMeierRequest(**base, event_positive_value=2**53).event_positive_value == 2**53
    for value in (2**53 + 1, 10**20, -(10**400)):
        with pytest.raises(ValidationError):
            app_module.KaplanMeierRequest(**base, event_positive_value=value)
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    response = _post_raw(
        "/api/kaplan-meier",
        '{"dataset_id":"%s","time_column":"os_months","event_column":"os_event","event_positive_value":1%s}' % (dataset_id, "0" * 400),
    )
    assert response.status_code == 422, response.text[:200]


@pytest.mark.parametrize("field", ["time", "risk"])
def test_prediction_blocks_with_integers_beyond_float_range_are_422(field: str) -> None:
    huge = "1" + "0" * 400
    time_values = f"[1,2,{huge}]" if field == "time" else "[1,2,3]"
    risk_values = f"[0.1,0.2,{huge}]" if field == "risk" else "[0.1,0.2,0.3]"
    body = (
        '{"predictions":[{"row_ids":["a","b","c"],"time":%s,"event":[1,0,1],"risk":{"Cox PH":%s}}]}'
        % (time_values, risk_values)
    )
    response = _post_raw("/api/model-comparison-intervals", body)
    assert response.status_code == 422, response.text[:200]
    assert "finite" in _detail(response)


@pytest.mark.parametrize("bind", ["0.0.0.0", "::"])
def test_wildcard_binds_answer_requests_addressed_to_the_wildcard_literal(monkeypatch: pytest.MonkeyPatch, bind: str) -> None:
    monkeypatch.setattr(app_module, "_local_interface_hostnames", lambda: {"192.168.1.20"})
    monkeypatch.setenv(app_module.BIND_HOST_ENV_VAR, bind)
    monkeypatch.setenv(app_module.ALLOWED_HOSTS_ENV_VAR, "")
    app_module._configured_request_hosts.cache_clear()
    try:
        for host in ("0.0.0.0:8000", "[::]:8000"):
            assert client.get("/api/health", headers={"Host": host}).status_code == 200, host
        # State-changing requests still need an Origin that matches the Host.
        same_origin = client.post("/api/load-example", headers={"Host": "0.0.0.0:8000", "Origin": "http://0.0.0.0:8000"})
        assert same_origin.status_code == 200, same_origin.text
        cross_site = client.post("/api/load-example", headers={"Host": "0.0.0.0:8000", "Origin": "http://evil.example"})
        assert cross_site.status_code == 403
    finally:
        app_module._configured_request_hosts.cache_clear()

    monkeypatch.setenv(app_module.BIND_HOST_ENV_VAR, "127.0.0.1")
    app_module._configured_request_hosts.cache_clear()
    try:
        assert client.get("/api/health", headers={"Host": "0.0.0.0:8000"}).status_code == 400
    finally:
        app_module._configured_request_hosts.cache_clear()


# ── Stronger checks of caches, profiles, figures and validation ──


def test_ml_artifact_cache_isolates_frames_and_shares_the_fitted_model() -> None:
    import pandas as pd

    class _FittedForest:
        """Stands in for a fitted scikit-survival model, which is an object, not a container."""

        estimators_: list = []

    model = _FittedForest()
    frame = pd.DataFrame({"age": [10.0, 20.0]})
    encoder = {"numeric_features": ["age"], "categorical_mappings": {}}
    cache = app_module._MlArtifactCache(max_items=2)
    signature = {"model_type": "rsf", "features": ["age"]}
    cache.remember(
        dataset_id="dataset-1",
        model_type="rsf",
        signature=signature,
        result={"_model": model, "_X_encoded": frame, "_feature_encoder": encoder, "_analysis_frame": frame},
    )
    # The caller keeps working with its own frame and encoder after the fit is cached.
    frame.iloc[0, 0] = -1.0
    encoder["numeric_features"].append("changed")

    first = cache.get(dataset_id="dataset-1", model_type="rsf", signature=signature)
    second = cache.get(dataset_id="dataset-1", model_type="rsf", signature=signature)
    assert first["_X_encoded"].iloc[0, 0] == 10.0 and first["_feature_encoder"]["numeric_features"] == ["age"]
    first["_analysis_frame"].iloc[1, 0] = 99.0
    first["_feature_encoder"]["numeric_features"].append("mutated")
    assert second["_analysis_frame"].iloc[1, 0] == 20.0 and second["_feature_encoder"]["numeric_features"] == ["age"]
    # Every consumer only predicts with the fitted model, so it is shared instead of copied.
    assert first["_model"] is model and second["_model"] is model
    assert cache.get(dataset_id="dataset-1", model_type="rsf", signature={**signature, "features": ["x"]}) is None


def test_reused_profiles_of_derived_snapshots_match_a_fresh_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from survival_toolkit.analysis import profile_dataframe

    text = (
        "patient_id,os_months,os_event,score,site\n"
        "A,1.5,1,0.2,Montréal\nA,2.5,0,0.9,Québec\nB,3.5,1,0.4,Montréal\nC,4.5,0,0.7,Québec\n"
    )
    upload = client.post("/api/upload", files={"file": ("cohort.csv", text.encode("cp1252"), "text/csv")})
    assert upload.status_code == 200, upload.text
    assert upload.json()["text_encoding"] not in (None, "utf-8")
    assert [entry["column"] for entry in upload.json()["duplicate_identifier_columns"]] == ["patient_id"]

    def _unexpected_profile(*args, **kwargs):
        raise AssertionError("a derived snapshot must reuse the cached profile")

    monkeypatch.setattr(app_module, "profile_dataframe", _unexpected_profile)
    # Two groups on four rows: an identifier-like name with repeated values, flagged like any column.
    derived = client.post(
        "/api/derive-group",
        json={"dataset_id": upload.json()["dataset_id"], "source_column": "score", "method": "median_split", "new_column_name": "patient_id_group"},
    )
    assert derived.status_code == 200, derived.text
    payload = derived.json()
    stored = app_module.store.get(payload["dataset_id"], copy_dataframe=False)
    fresh = app_module._json_ready(profile_dataframe(stored.dataframe, dataset_id=stored.dataset_id, filename=stored.filename))
    for key, value in fresh.items():
        assert payload[key] == value, key
    assert [entry["column"] for entry in payload["duplicate_identifier_columns"]] == ["patient_id", "patient_id_group"]


def test_deep_model_loss_figure_uses_the_early_stopping_run_of_a_real_fit() -> None:
    pytest.importorskip("torch")
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age", "biomarker_score", "immune_index"],
            "model_type": "deepsurv",
            "hidden_layers": [8],
            "epochs": 30,
            "early_stopping_patience": 2,
            "random_seed": 3,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    analysis, loss = payload["analysis"], payload["figures"]["loss"]
    run_length = analysis["early_stopping_epochs"]
    assert len(loss["data"][0]["y"]) == run_length == len(analysis["loss_history"])
    assert [trace["name"] for trace in loss["data"]] == ["Training loss", analysis["monitor_metric_label"]]
    annotations = " ".join(str(note.get("text", "")) for note in loss["layout"].get("annotations", []))
    if analysis["stopped_early"]:
        assert f"Stopped early at epoch {run_length}" in annotations
    assert f"Best monitor epoch: {analysis['best_monitor_epoch']}" in annotations


def test_marker_validation_on_an_independent_cohort() -> None:
    from survival_toolkit.sample_data import make_example_dataset

    _, recipe = _locked_recipe()
    external_frame = make_example_dataset(seed=11, n_patients=240)
    upload = client.post("/api/upload", files={"file": ("external.csv", external_frame.to_csv(index=False).encode(), "text/csv")})
    assert upload.status_code == 200, upload.text
    response = client.post("/api/marker-validation", json={"dataset_id": upload.json()["dataset_id"], "recipe": recipe, "n_bootstrap": 60})
    assert response.status_code == 200, response.text
    validation = response.json()["validation"]
    assert validation["cohort"]["n"] == 240
    assert validation["cohort"]["events"] == int(external_frame["os_event"].sum())
    metrics = validation["metrics"]
    assert 0.5 < metrics["c_index"] < 1.0
    low, high = metrics["c_index_ci"]
    assert low < metrics["c_index"] < high
    assert {row["marker"] for row in validation["markers"]} == set(recipe["markers"])


# ── Request body limits on JSON routes ──────────────────────────


def test_json_routes_refuse_an_oversized_content_length_before_reading() -> None:
    declared = app_module._MAX_JSON_BODY_BYTES + 1
    for path in ("/api/export-table", "/api/model-comparison-intervals", "/api/marker-validation"):
        response = client.post(path, content=b"{}", headers={"Content-Type": "application/json", "Content-Length": str(declared)})
        assert response.status_code == 413, (path, response.text)
        assert "limit" in _detail(response)


def test_json_routes_stop_reading_a_streamed_body_past_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setattr(app_module, "_MAX_JSON_BODY_BYTES", 64 * 1024)
    chunk = b" " * (16 * 1024)
    parts = [b'{"rows": [', *([chunk] * 200), b"]}"]
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
        "path": "/api/export-table",
        "raw_path": b"/api/export-table",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8000"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 5555),
        "server": ("127.0.0.1", 8000),
    }

    async def _run() -> None:
        try:
            await asyncio.wait_for(app(scope, receive, send), timeout=60)
        except Exception:  # the server may still raise after the response was sent
            pass

    asyncio.run(_run())
    assert state["status"] == 413
    # The body is 3.2 MB; reading stops within a chunk of the 64 KB limit.
    assert state["consumed"] <= 64 * 1024 + 2 * len(chunk) + 64


def test_uploads_keep_their_own_larger_limit() -> None:
    assert app_module._max_upload_request_bytes() > app_module._MAX_JSON_BODY_BYTES
    response = client.post(
        "/api/upload",
        content=b"--x--\r\n",
        headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(app_module._MAX_JSON_BODY_BYTES + 1)},
    )
    assert response.status_code != 413, response.text


def test_utf16_matrix_lines_are_counted_on_decoded_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import gzip

    from survival_toolkit.errors import UserInputError

    def _reached_the_reader(*args, **kwargs):
        raise UserInputError("reached the matrix reader")

    monkeypatch.setattr(app_module, "MAX_MATRIX_MARKERS", 50)
    monkeypatch.setattr(app_module, "MAX_MATRIX_SAMPLES", 100)
    monkeypatch.setattr(app_module, "read_marker_matrix", _reached_the_reader)
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    # "Ċ" is stored as the bytes 0A 01 in UTF-16-LE: a newline byte inside another character.
    short = ("gene,P1\n" + "".join(f"G{index}ĊĊĊĊ,1\n" for index in range(40))).encode("utf-16")
    long = ("gene,P1\n" + "".join(f"G{index},1\n" for index in range(500))).encode("utf-16")
    for name, payload, expected in (
        ("short.csv", short, "reached the matrix reader"),
        ("short.csv.gz", gzip.compress(short), "reached the matrix reader"),
        ("long.csv", long, "more than 101 lines"),
    ):
        response = client.post(
            "/api/marker-matrix",
            data={"dataset_id": dataset_id, "id_column": "patient_id"},
            files={"file": (name, payload, "application/octet-stream")},
        )
        assert response.status_code == 400, (name, response.text)
        assert expected in _detail(response), (name, _detail(response))


# ── Bootstrap budgets and long jobs ─────────────────────────────


def _interval_block(n: int, n_models: int, *, seed: int = 1, reference: str = "Cox PH") -> dict:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-signal))
    censor = rng.exponential(1.5, size=n)
    names = [reference, *[f"Model {index}" for index in range(1, n_models)]]
    return {
        "row_ids": [f"r{index}" for index in range(n)],
        "time": np.minimum(event_time, censor).round(4).tolist(),
        "event": (event_time <= censor).astype(int).tolist(),
        "risk": {name: (signal + rng.normal(scale=index + 0.5, size=n)).round(4).tolist() for index, name in enumerate(names)},
    }


def test_realistic_test_sets_get_bootstrap_intervals() -> None:
    block = _interval_block(2500, 10)
    response = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 100})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["n"] == 2500 and payload["n_bootstrap"] == 100 and "bootstrap_note" not in payload
    assert all(row["c_index_ci"][0] is not None for row in payload["rows"])


def test_missing_reference_models_are_reported() -> None:
    block = _interval_block(200, 2, reference="CoxPH")
    response = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 100})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["reference"] is None
    assert '"Cox PH"' in payload["reference_note"] and "CoxPH" in payload["reference_note"]
    named = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 100, "reference": "CoxPH"})
    assert named.json()["reference"] == "CoxPH" and "reference_note" not in named.json()
    unpaired = client.post("/api/model-comparison-intervals", json={"predictions": [block], "n_bootstrap": 100, "reference": None})
    assert "reference_note" not in unpaired.json()


def test_validation_bootstrap_is_budgeted(monkeypatch: pytest.MonkeyPatch) -> None:
    assert app_module._validation_bootstrap_draws(1000, 2, 200) == (200, None)
    draws, note = app_module._validation_bootstrap_draws(100_000, 2, 2000)
    assert draws == 0 and "not computed" in note
    draws, note = app_module._validation_bootstrap_draws(20_000, 2, 2000)
    assert app_module._INTERVAL_MIN_DRAWS <= draws < 2000 and f"limited to {draws} of the 2000" in note

    dataset_id, recipe = _locked_recipe()
    seen: list[int] = []
    original = app_module.validate_locked_recipe

    def _recording(*args, **kwargs):
        seen.append(kwargs["n_bootstrap"])
        return original(*args, **kwargs)

    n_rows = client.get(f"/api/dataset/{dataset_id}").json()["n_rows"]
    n_models = 2 if recipe.get("clinical_only_model") else 1
    per_draw = app_module._c_index_draw_work(n_rows, n_rows, n_models)
    monkeypatch.setattr(app_module, "_INTERVAL_WORK_BUDGET", per_draw * 150)
    monkeypatch.setattr(app_module, "validate_locked_recipe", _recording)
    response = client.post("/api/marker-validation", json={"dataset_id": dataset_id, "recipe": recipe, "n_bootstrap": 400})
    assert response.status_code == 200, response.text
    assert seen == [150]
    assert any("limited to 150 of the 400" in note for note in response.json()["validation"]["notes"])


def test_marker_matrix_ingest_is_a_cancellable_heavy_job_on_a_leased_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import pandas as pd

    from survival_toolkit.sample_data import make_example_dataset

    calls: list[dict] = []
    leased: list[str] = []
    original_job = app_module._run_job
    original_lease = app_module.store.lease

    async def _recording(job, **kwargs):
        calls.append(kwargs)
        return await original_job(job, **kwargs)

    def _recording_lease(dataset_id: str):
        leased.append(dataset_id)
        return original_lease(dataset_id)

    monkeypatch.setattr(app_module, "_run_job", _recording)
    monkeypatch.setattr(app_module.store, "lease", _recording_lease)
    dataset_id = client.post("/api/load-example").json()["dataset_id"]
    frame = make_example_dataset()
    table = pd.DataFrame(np.random.default_rng(3).normal(size=(len(frame), 4)), columns=[f"G{index}" for index in range(4)])
    table.insert(0, "patient_id", frame["patient_id"].to_numpy())
    path = tmp_path / "matrix.csv"
    table.to_csv(path, index=False)
    response = client.post(
        "/api/marker-matrix",
        data={"dataset_id": dataset_id, "id_column": "patient_id", "orientation": "samples_in_rows"},
        files={"file": ("matrix.csv", path.read_bytes(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert calls and calls[-1]["heavy"] is True and calls[-1]["request"] is not None
    assert dataset_id in leased

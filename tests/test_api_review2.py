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

"""Analysis table schemas survive canonical JSON storage and file exports."""
import csv
import io
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import pytest

from test_frontend_review import _run_page, client

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")

TABLES = {
    "km_summary": ["Group", "N", "Events", "Censored", "Median survival", "Median CI lower", "Median CI upper", "RMST", "RMST SE", "RMST CI lower", "RMST CI upper"],
    "km_pairwise": ["Comparison", "Chi-square", "P value", "BH adjusted p", "Note"],
    "cox_results": ["Variable", "Label", "Reference", "Beta", "Hazard ratio", "CI lower", "CI upper", "SE", "Z", "P value"],
    "cox_diagnostics": ["Term", "Schoenfeld rho", "Chi-square", "P value"],
}


@pytest.mark.parametrize("table_type", TABLES)
def test_analysis_table_order_is_stable_and_retains_new_public_columns(tmp_path: Path, table_type: str) -> None:
    result = _run_page(tmp_path, r"""
      const { analysisTableColumns, buildKmTableExportPayload, buildCoxTableExportPayload, renderTable } = page.context;
      const refs = page.run('refs');
      const first = Object.fromEntries(fixtures.columns.map((column, index) => [column, index]));
      const rows = [{ ...first, 'Future Z': 8 }, { ...first, 'Future A': 9, _kind: 'internal' }];
      const restored = rows.map(row => Object.fromEntries(Object.entries(row).sort(([a], [b]) => a.localeCompare(b))));
      const build = fixtures.tableType.startsWith('km_') ? buildKmTableExportPayload : buildCoxTableExportPayload;
      renderTable(refs.kmSummaryShell, restored, analysisTableColumns(restored, fixtures.tableType));
      const rendered = refs.kmSummaryShell.querySelectorAll('th').map(node => node.title);
      const preview = [{ 'z user': 1, Group: 'A', 'a user': 2 }];
      renderTable(refs.kmPairwiseShell, preview, null, { rawHeaders: true, rawValues: true, pValueColumns: [] });
      return { before: analysisTableColumns(rows, fixtures.tableType),
        after: analysisTableColumns(restored, fixtures.tableType), rendered,
        freshExport: build(rows, 'Synthetic table'), savedExport: build(restored, 'Synthetic table'),
        preview: refs.kmPairwiseShell.querySelectorAll('th').map(node => node.title) };
    """, tableType=table_type, columns=TABLES[table_type])
    expected = [*TABLES[table_type], "Future A", "Future Z"]
    assert result["before"] == result["after"] == result["rendered"] == expected
    assert result["freshExport"]["columns"] == result["savedExport"]["columns"] == expected
    assert result["preview"] == ["z user", "Group", "a user"]
    # Both server export formats consume the same explicit schema, including fields
    # that occur only in later rows. No numerical output is recomputed.
    for format_name in ("csv", "docx"):
        response = client.post("/api/export-table", json={**result["savedExport"], "format": format_name})
        assert response.status_code == 200, response.text
        if format_name == "csv":
            header = next(row for row in csv.reader(io.StringIO(response.content.decode("utf-8-sig"))) if row and not row[0].startswith("#"))
        else:
            with zipfile.ZipFile(io.BytesIO(response.content)) as document:
                root = ET.fromstring(document.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            cells = root.findall(".//w:tbl/w:tr", ns)[0].findall("w:tc", ns)
            header = ["".join(cell.itertext()) for cell in cells]
        assert header == expected

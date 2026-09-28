"""Regression tests for the second review of the analysis loader, event coding and outcome rules
(file loading, number formats, encodings, event and time column checks, outcome detection)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.analysis as analysis
from survival_toolkit.analysis import _cohort_frame, compute_km_analysis, load_dataframe


# ---------------------------------------------------------------------------------------
# Row index and header/data field counts (R3 #3, #15)


def test_r_write_table_row_names_become_a_named_column() -> None:
    # R's write.table writes no header field above the row names, so each data row has one
    # field more than the header; pandas used to hide the row names in the index.
    text = "time\tstatus\tage\n1\t12\t1\t60\n2\t5\t0\t70\n3\t8\t1\t65\n4\t20\t0\t50\n5\t3\t1\t55\n6\t9\t0\t58\n"
    frame = load_dataframe(text.encode(), "r_export.tsv")
    assert frame.columns.tolist() == ["row_names", "time", "status", "age"]
    assert isinstance(frame.index, pd.RangeIndex) and frame.index.tolist() == list(range(6))
    assert frame["row_names"].tolist() == [1, 2, 3, 4, 5, 6]
    assert frame["time"].tolist() == [12, 5, 8, 20, 3, 9]
    cohort = _cohort_frame(frame, "time", "status")
    assert cohort.attrs["source_row_index"] == list(range(6))


def test_row_names_column_does_not_take_an_existing_column_name() -> None:
    text = "row_names,time,status\nA,x,12,1\nB,y,5,0\nC,z,8,1\n"
    frame = load_dataframe(text.encode(), "r_export.csv")
    assert frame.columns.tolist() == ["row_names_2", "row_names", "time", "status"]
    assert frame["row_names"].tolist() == ["x", "y", "z"]


def test_trailing_separator_on_data_rows_does_not_shift_column_labels() -> None:
    lines = ["id,time,event,age,gene_a"]
    lines += [f"P{i},{10 + i},{i % 2},{50 + i},{round(1.5 * i, 2)}," for i in range(8)]
    frame = load_dataframe("\n".join(lines).encode(), "trailing.csv")
    assert frame.columns.tolist() == ["id", "time", "event", "age", "gene_a"]
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["id"].tolist() == [f"P{i}" for i in range(8)]
    assert frame["time"].tolist() == list(range(10, 18))
    assert frame["gene_a"].tolist() == [round(1.5 * i, 2) for i in range(8)]


def test_rows_wider_than_the_header_in_different_ways_are_refused() -> None:
    text = "id,time,event\nP0,10,1,5\nP1,11,0\nP2,12,1,7,9\n"
    with pytest.raises(ValueError, match="more fields than the header row"):
        load_dataframe(text.encode(), "ragged.csv")


def test_parquet_row_numbers_are_reset_and_named_indexes_become_columns() -> None:
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(3)

    def _part() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "os_months": rng.exponential(20, 30).round(1) + 0.1,
                "os_event": rng.integers(0, 2, 30),
                "grp": rng.choice(["x", "y"], 30),
            }
        )

    combined = pd.concat([_part(), _part()])  # without ignore_index=True: row numbers repeat
    combined.iloc[3, 0] = np.nan
    buffer = io.BytesIO()
    combined.to_parquet(buffer)
    frame = load_dataframe(buffer.getvalue(), "cohort.parquet")
    assert isinstance(frame.index, pd.RangeIndex) and frame.index.is_unique
    assert frame.columns.tolist() == ["os_months", "os_event", "grp"]
    cohort = _cohort_frame(frame, "os_months", "os_event", extra_columns=["grp"])
    # The app restricts the cohort table with this membership test.
    assert int(frame.index.isin(pd.Index(cohort.attrs["source_row_index"])).sum()) == len(cohort) == 59

    by_patient = combined.head(5).copy()
    by_patient.index = pd.Index([f"P{i}" for i in range(5)], name="patient_id")
    buffer = io.BytesIO()
    by_patient.to_parquet(buffer)
    frame = load_dataframe(buffer.getvalue(), "cohort.parquet")
    assert frame.columns.tolist() == ["patient_id", "os_months", "os_event", "grp"]
    assert frame["patient_id"].tolist() == [f"P{i}" for i in range(5)]
    assert frame.index.tolist() == list(range(5))

    # An index named like a column keeps the column's name for the column.
    clash = pd.DataFrame({"age": [50, 60]}, index=pd.Index(["P0", "P1"], name="age"))
    reset = analysis._with_default_row_index(clash)
    assert reset.columns.tolist() == ["age_2", "age"]
    assert reset["age_2"].tolist() == ["P0", "P1"] and reset["age"].tolist() == [50, 60]


def test_excel_reader_errors_are_user_errors_but_code_errors_are_not_masked(monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    with pytest.raises(ValueError, match="Failed to read Excel file"):
        load_dataframe(b"PK\x03\x04 not really a workbook", "broken.xlsx", max_rows=10)
    buffer = io.BytesIO()
    pd.DataFrame({"time": [1.0, 2.0], "event": [1, 0]}).to_excel(buffer, index=False)

    def _broken_limit(*args, **kwargs):
        raise KeyError("bug in SurvStudio code")

    monkeypatch.setattr(analysis, "_row_read_limit", _broken_limit)
    with pytest.raises(KeyError, match="bug in SurvStudio code"):
        load_dataframe(buffer.getvalue(), "cohort.xlsx", max_rows=10)

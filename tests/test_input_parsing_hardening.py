"""Input interpretation must preserve categories and refuse unresolved number formats."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.analysis import (
    _coerce_survival_time_values,
    _resolve_cox_categorical_covariates,
    coerce_event,
    compute_km_analysis,
    load_dataframe,
    profile_dataframe,
)


def test_csv_preserves_all_none_outcomes_before_event_mapping() -> None:
    source = "time,status,treatment\n" + "\n".join(
        f"{index + 1},{'Dead' if index < 20 else 'None'},None" for index in range(60)
    )
    frame = load_dataframe(source.encode(), "none_outcome.csv")
    assert len(frame) == 60
    assert frame["status"].value_counts().to_dict() == {"None": 40, "Dead": 20}
    assert frame["status"].notna().sum() == 60
    assert frame["treatment"].eq("None").all()
    # A literal None category is retained; only an explicit observed event label
    # confirms its use as the opposite (censored) state in this two-label column.
    with pytest.raises(ValueError, match="literal 'None'.*not assumed"):
        coerce_event(frame["status"])
    result = compute_km_analysis(frame, "time", "status", event_positive_value="Dead")
    assert result["cohort"]["n"] == 60
    assert result["cohort"]["events"] == 20
    assert result["cohort"]["censored"] == 40
    assert result["cohort"]["event_coding"]["event_positive_value"] == "Dead"
    assert result["cohort"]["event_coding"]["note"] in result["scientific_summary"]["cautions"]
    assert frame["status"].eq("None").sum() == 40


def test_csv_and_excel_preserve_literal_none_categories() -> None:
    expected = pd.DataFrame({"time": [1, 2, 3], "event": [1, 0, 1], "therapy": ["None", "Drug", "None"]})
    frame = load_dataframe(expected.to_csv(index=False).encode(), "category.csv")
    assert frame["therapy"].tolist() == expected["therapy"].tolist()
    pytest.importorskip("openpyxl")
    stream = io.BytesIO()
    expected.to_excel(stream, index=False)
    workbook = load_dataframe(stream.getvalue(), "category.xlsx")
    assert workbook["therapy"].tolist() == expected["therapy"].tolist()


@pytest.mark.parametrize("value,expected", [("0,113", 0.113), ("-0,113", -0.113), ("0,001", 0.001)])
def test_quoted_leading_zero_comma_is_decimal(value: str, expected: float) -> None:
    frame = load_dataframe(f'time,event,marker\n1,1,"{value}"\n2,0,"{value}"\n'.encode(), "decimal.csv")
    assert frame["marker"].tolist() == pytest.approx([expected, expected])
    np.testing.assert_allclose(_coerce_survival_time_values(pd.Series([value]), "time"), [expected])


@pytest.mark.parametrize("suffix,content", [
    ("csv", 'time,event\n"1,500",1\n"2,500",0\n'),
    ("tsv", "time\tevent\n1,500\t1\n2,500\t0\n"),
])
def test_unresolved_comma_grouping_is_refused(suffix: str, content: str) -> None:
    with pytest.raises(ValueError, match="read either as"):
        load_dataframe(content.encode(), f"ambiguous.{suffix}")
    with pytest.raises(ValueError, match="read either as"):
        _coerce_survival_time_values(pd.Series(["1,500", "2,500"]), "time")


def test_us_grouping_requires_explicit_column_or_file_evidence() -> None:
    own = load_dataframe(b'time,event\n"1,500.0",1\n"2,500",0\n', "us.csv")
    file = load_dataframe(b'time,event,age\n"1,500",1,50.5\n"2,500",0,61.25\n', "us.csv")
    assert own["time"].tolist() == file["time"].tolist() == [1500, 2500]
    np.testing.assert_allclose(_coerce_survival_time_values(pd.Series(["1,500.0", "2,500"]), "time"), [1500, 2500])


def test_eu_semicolon_keeps_declared_file_convention() -> None:
    frame = load_dataframe(b"time;event;marker\n1.500;1;0,113\n2.500;0;1,125\n", "eu.csv")
    assert frame["time"].tolist() == [1500, 2500]
    assert frame["marker"].tolist() == pytest.approx([0.113, 1.125])


def test_mixed_decimal_marks_in_one_column_are_refused() -> None:
    with pytest.raises(ValueError, match="mixes decimal marks"):
        load_dataframe(b'time,event,marker\n1,1,"0,113"\n2,0,0.25\n', "mixed.csv")


@pytest.mark.parametrize("placeholder", [".", "unknown", "-", "None"])
def test_numeric_text_is_preserved_and_reviewed_before_fitting(placeholder: str) -> None:
    rows = [f"{index + 1},{index % 2},{placeholder if index < 3 else 30 + index * 0.5},Unknown" for index in range(80)]
    frame = load_dataframe(("time,event,age,group\n" + "\n".join(rows)).encode(), "numeric.csv")
    assert frame["age"].eq(placeholder).sum() == 3
    assert frame["age"].isna().sum() == 0
    assert frame["group"].eq("Unknown").all()
    with pytest.raises(ValueError, match='Feature "age" looks numeric but contains 3 non-numeric'):
        _resolve_cox_categorical_covariates(frame, ["age", "group"], [])
    profile = profile_dataframe(frame, "test", "numeric.csv")
    assert "age" not in profile["numeric_columns"]
    assert profile["numeric_review_columns"] == ["age"]
    assert profile["input_review"] == [{
        "column": "age", "count": 3, "tokens": {placeholder: 3}, "action": "numeric_text_review",
    }]
    assert profile["input_normalization"] == []


def test_low_cardinality_codes_with_unknown_remain_categories() -> None:
    frame = load_dataframe(b"time,event,grade\n1,1,1\n2,0,2\n3,1,Unknown\n4,0,None\n", "codes.csv")
    assert frame["grade"].tolist() == ["1", "2", "Unknown", "None"]
    assert _resolve_cox_categorical_covariates(frame, ["grade"], []) == ["grade"]
    assert frame.attrs.get("input_normalization", []) == []


def test_continuous_looking_category_codes_keep_every_original_level() -> None:
    labels = [str(value) for value in range(101, 113)] + ["Unknown"]
    rows = [f"{index + 1},{index % 2},{label}" for index, label in enumerate(labels)]
    frame = load_dataframe(("time,event,batch\n" + "\n".join(rows)).encode(), "batch.csv")
    assert frame["batch"].tolist() == labels
    assert frame["batch"].nunique() == 13
    assert frame["batch"].isna().sum() == 0
    assert _resolve_cox_categorical_covariates(frame, ["batch"], ["batch"]) == ["batch"]
    assert frame["batch"].tolist() == labels
    profile = profile_dataframe(frame, "test", "batch.csv")
    assert profile["numeric_review_columns"] == ["batch"]
    assert profile["input_review"] == [{
        "column": "batch", "count": 1, "tokens": {"Unknown": 1}, "action": "numeric_text_review",
    }]


def test_formatted_time_preserves_duplicate_indices_and_all_patients() -> None:
    frame = pd.DataFrame(
        {"time": ["1,500.0", "2,500.0", "3,500.0", "4,500.0"], "event": [1, 0, 1, 0]},
        index=[0, 0, 1, 1],
    )
    parsed = _coerce_survival_time_values(frame["time"], "time")
    assert parsed.index.tolist() == [0, 0, 1, 1]
    assert parsed.tolist() == [1500, 2500, 3500, 4500]
    result = compute_km_analysis(frame, "time", "event", event_positive_value=1)
    assert result["cohort"]["n"] == 4
    assert result["cohort"]["events"] == 2
    assert result["cohort"]["time_max"] == 4500


def test_unexpected_numeric_contamination_is_still_refused() -> None:
    rows = [f"{index + 1},{index % 2},{'typo' if index == 0 else 30 + index * 0.5}" for index in range(80)]
    frame = load_dataframe(("time,event,age\n" + "\n".join(rows)).encode(), "typo.csv")
    with pytest.raises(ValueError, match='looks numeric but contains 1 non-numeric'):
        _resolve_cox_categorical_covariates(frame, ["age"], [])


def test_genuine_all_event_data_are_not_rejected() -> None:
    frame = load_dataframe(b"time,event\n1,1\n2,1\n3,1\n4,1\n", "all_events.csv")
    result = compute_km_analysis(frame, "time", "event", event_positive_value=1)
    assert result["cohort"]["n"] == result["cohort"]["events"] == 4

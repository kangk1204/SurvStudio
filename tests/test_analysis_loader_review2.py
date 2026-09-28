"""Regression tests for the second review of the analysis loader, event coding and outcome rules
(file loading, number formats, encodings, event and time column checks, outcome detection)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.analysis as analysis
from survival_toolkit.analysis import (
    _cohort_frame,
    _survival_outcome_like_columns,
    coerce_event,
    compute_km_analysis,
    find_event_equivalent_columns,
    load_dataframe,
    model_feature_candidate_columns,
    suggest_columns,
)


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


def test_r_write_table_european_export_keeps_row_names_and_reads_dot_groups() -> None:
    # write.table(sep=";", dec=",") with row names; one blank day count.
    days = ["1.234", "856", "", "310", "1.502", "95", "3.020", "640"]
    text = "\n".join(["days;status;bmi"] + [f"P{i};{d};{i % 2};{20 + i},5" for i, d in enumerate(days)])
    frame = load_dataframe(text.encode(), "r_eu.csv")
    assert frame.columns.tolist() == ["row_names", "days", "status", "bmi"]
    assert frame["days"].tolist()[:2] == [1234.0, 856.0] and np.isnan(frame["days"].iloc[2])
    assert frame["days"].tolist()[3:] == [310.0, 1502.0, 95.0, 3020.0, 640.0]
    assert frame["bmi"].tolist()[:2] == [20.5, 21.5]


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


# ---------------------------------------------------------------------------------------
# Text encodings (R3 #12)


def test_western_bytes_after_the_sniffed_prefix_fall_back_to_windows_1252() -> None:
    late = ("time,event,site\n" + "1,0,A\n" * 200_000 + "2,1,Ärzteklinik\n").encode("cp1252")
    assert len(late) > analysis._TEXT_SNIFF_BYTES
    frame = load_dataframe(late, "late.csv")
    assert frame.attrs["source_encoding"] == "cp1252"
    assert frame["site"].iloc[-1] == "Ärzteklinik"


def test_korean_bytes_after_the_sniffed_prefix_are_read_as_cp949() -> None:
    late = ("time,event,site\n" + "1,0,A\n" * 200_000 + "2,1,서울대학교병원\n").encode("cp949")
    frame = load_dataframe(late, "late.csv")
    assert frame.attrs["source_encoding"] == "cp949"
    assert frame["site"].iloc[-1] == "서울대학교병원"


def test_western_text_that_also_decodes_as_cp949_is_read_as_windows_1252() -> None:
    # "Är" and "Ån" are valid (rare) CP949 syllables, which once made this file "Korean".
    western = "name,time,event\nÄrzte,12.5,1\nÅnge,7,0\nÄrger,3,1\n".encode("cp1252")
    assert western.decode("cp949")  # the whole sample decodes as CP949
    frame = load_dataframe(western, "western.csv")
    assert frame.attrs["source_encoding"] == "cp1252"
    assert frame["name"].tolist() == ["Ärzte", "Ånge", "Ärger"]
    korean = "환자,생존기간,사망\n김,12.5,1\n이,7,0\n".encode("cp949")
    assert load_dataframe(korean, "korean.csv").attrs["source_encoding"] == "cp949"


# ---------------------------------------------------------------------------------------
# Decimal marks and thousands separators (R3 #1, #8) and numbers that look like dates (#5)

_EU_DAYS = ["1.234", "856", "2.045", "310", "1.502", "95", "3.020", "640", "1.100", "75"]
_EU_DAY_VALUES = [1234, 856, 2045, 310, 1502, 95, 3020, 640, 1100, 75]


def _semicolon_file(days: list[str], *, bmi: bool) -> bytes:
    header = "id;os_days;os_event" + (";bmi" if bmi else ";age")
    rows = [f"p{i};{day};{i % 2};" + (f"{20 + i},5" if bmi else str(60 + i)) for i, day in enumerate(days)]
    return "\n".join([header, *rows]).encode()


def test_semicolon_export_reads_dot_groups_as_thousands_without_a_decimal_comma_column() -> None:
    frame = load_dataframe(_semicolon_file(_EU_DAYS, bmi=False), "eu_days.csv")
    assert frame["os_days"].tolist() == _EU_DAY_VALUES
    assert _cohort_frame(frame, "os_days", "os_event")["os_days"].tolist() == [float(day) for day in _EU_DAY_VALUES]


@pytest.mark.parametrize("marker", ["-", "[Not Available]", "unknown", "NA", ""])
@pytest.mark.parametrize("bmi", [False, True])
def test_missing_value_markers_do_not_change_how_numbers_read(marker: str, bmi: bool) -> None:
    days = list(_EU_DAYS)
    days[3] = marker
    frame = load_dataframe(_semicolon_file(days, bmi=bmi), "eu_days.csv")
    expected = [float(day) for day in _EU_DAY_VALUES]
    loaded = frame["os_days"].tolist()
    assert np.isnan(loaded[3])
    assert loaded[:3] + loaded[4:] == expected[:3] + expected[4:]
    cohort = _cohort_frame(frame, "os_days", "os_event")
    assert cohort["os_days"].tolist() == expected[:3] + expected[4:]


def test_semicolon_export_reads_comma_values_as_decimals_in_mostly_whole_columns() -> None:
    for values, expected in (
        (["12", "24", "36", "1,250", "18", "30", "6", "10,125", "15", "40"], [12, 24, 36, 1.25, 18, 30, 6, 10.125, 15, 40]),
        (["12", "24", "36", "1,5", "18", "30", "6", "10,5", "15", "40"], [12, 24, 36, 1.5, 18, 30, 6, 10.5, 15, 40]),
    ):
        text = "\n".join(["id;time;event"] + [f"{i};{value};{i % 2}" for i, value in enumerate(values)])
        frame = load_dataframe(text.encode(), "eu.csv")
        assert frame["time"].tolist() == pytest.approx(expected)
        # Analysis-time parsing cannot disagree with the loader: the column is numeric already.
        assert _cohort_frame(frame, "time", "event")["time"].tolist() == pytest.approx(expected)


def test_semicolon_file_with_dot_decimals_reads_like_its_other_columns() -> None:
    # A ";" file written with "." decimals (pandas to_csv(sep=";")): the data override the separator.
    text = (
        "id;ratio;score;count\n"
        "p0;0.5;1.500;1,234\np1;12.25;2.250;2,500\np2;3.75;3.125;12,000\np3;0.125;4.000;999\n"
    )
    frame = load_dataframe(text.encode(), "dots.csv")
    assert frame["score"].tolist() == [1.5, 2.25, 3.125, 4.0]
    assert frame["count"].tolist() == [1234, 2500, 12000, 999]


def test_tab_file_with_only_dot_groups_keeps_them_as_decimals() -> None:
    text = "id\tscore\tevent\np0\t1.500\t1\np1\t2.250\t0\np2\t3.125\t1\n"
    assert load_dataframe(text.encode(), "scores.tsv")["score"].tolist() == [1.5, 2.25, 3.125]


@pytest.mark.parametrize("marker", [None, "[Not Available]"])
def test_comma_groups_that_nothing_in_a_tab_file_explains_are_refused(marker: str | None) -> None:
    values = ["1,234", "2,500", "3,750", "1,020", "4,410", "2,205", "1,860", "3,125", "2,990", "1,475"]
    if marker is not None:
        values[-1] = marker
    rows = ["time\tevent\tage"] + [f"{value}\t{i % 2}\t{50 + i}" for i, value in enumerate(values)]
    with pytest.raises(ValueError, match='"1,234", which read either as 1.234 or as 1234'):
        load_dataframe("\n".join(rows).encode(), "data.tsv")


def test_ambiguous_columns_in_a_file_with_both_decimal_marks_are_refused() -> None:
    text = "id;days;event;age;ratio\np0;1.234;1;60,5;0.25\np1;856;0;55,0;1.5\np2;2.045;1;70,25;2.75\n"
    with pytest.raises(ValueError, match=r'Column "days" .*both with "," \(column "age"\) and with "." \(column "ratio"\)'):
        load_dataframe(text.encode(), "mixed.csv")


def test_quoted_comma_thousands_with_a_missing_marker_load_as_numbers() -> None:
    values = ["1,234", "2,500", "3,750", "1,020", "4,410", "2,205", "1,860", "3,125", "2,990", "[Not Available]"]
    rows = ["time,event,age"] + [f'"{value}",{i % 2},{50 + i}' for i, value in enumerate(values)]
    frame = load_dataframe("\n".join(rows).encode(), "data.csv")
    assert frame["time"].tolist()[:3] == [1234.0, 2500.0, 3750.0] and np.isnan(frame["time"].iloc[-1])
    km = compute_km_analysis(frame, "time", "event")
    assert km["cohort"]["n"] == 9
    assert km["cohort"]["time_max"] == pytest.approx(4410.0)


def test_numbers_with_separators_are_never_taken_for_calendar_dates() -> None:
    comma_decimals = pd.Series(["12,5", "7,25", "30,1", "4,75", "-", "18,2"], dtype=object)
    analysis._reject_calendar_date_time_column(comma_decimals, "os_months")  # no "calendar dates" error
    frame = pd.DataFrame({"os_months": comma_decimals, "os_event": [1, 0, 1, 0, 1, 1]})
    with pytest.raises(ValueError, match='not numbers, such as "12,5"'):
        compute_km_analysis(frame, "os_months", "os_event")
    grouped = pd.Series(["1,234", "2,500", "3,750", "1,020", "[Not Available]", "2,205"], dtype=object)
    frame = pd.DataFrame({"os_days": grouped, "os_event": [1, 0, 1, 0, 1, 1]})
    km = compute_km_analysis(frame, "os_days", "os_event")
    assert km["cohort"]["n"] == 5 and km["cohort"]["time_max"] == pytest.approx(3750.0)
    dated = pd.DataFrame({"os_date": ["2020-01-03", "2021-05-06", "-", "2022-01-01"], "os_event": [1, 0, 1, 0]})
    with pytest.raises(ValueError, match="calendar dates"):
        compute_km_analysis(dated, "os_date", "os_event")


# ---------------------------------------------------------------------------------------
# Missing-value markers and negated labels in the event column (R3 #6)


def test_missing_value_markers_in_the_event_column_are_missing_not_censored() -> None:
    status = pd.Series(
        ["Dead", "Alive", "Not Reported", "Dead", "Not available", "No data", "Alive", "Not applicable", "Never assessed"],
        name="vital_status",
    )
    expected = [1.0, 0.0, None, 1.0, None, None, 0.0, None, None]
    for coded in (coerce_event(status), coerce_event(status, "Dead")):
        assert [None if np.isnan(value) else value for value in coded.tolist()] == expected
    for label in ("Not reported", "Not available", "Not applicable", "No data", "Never assessed"):
        assert analysis._outcome_status_family_match(label) == (None, False), label
    for label in ("No recurrence", "0:Not Recurred", "Never relapsed", "without progression", "No evidence of disease"):
        assert analysis._outcome_status_family_match(label)[0] == "censor", label


def test_tcga_markers_in_the_event_column_leave_the_other_patients_usable() -> None:
    frame = pd.DataFrame(
        {
            "os_days": [10, 20, 30, 40, 50, 60, 70],
            "vital_status": ["Dead", "Alive", "Alive", "[Not Available]", "Dead", "Alive", "Dead"],
        }
    )
    for positive in (None, "Dead"):
        km = compute_km_analysis(frame, "os_days", "vital_status", event_positive_value=positive)
        assert km["cohort"]["n"] == 6 and km["cohort"]["events"] == 3
    blanks = pd.DataFrame({"time": ["10", " ", "30", "40", "50", "60"], "status": ["1", "0", " ", "1", "0", "1"]})
    assert len(_cohort_frame(blanks, "time", "status")) == 4


def test_none_in_an_event_column_is_not_dropped_as_missing() -> None:
    # "None" can mean no event ("Recurrence: None"); dropping it would keep only the events.
    with pytest.raises(ValueError, match="Could not infer event coding"):
        coerce_event(pd.Series(["None", "Relapse", "None", "Relapse"], name="recurrence"))


# ---------------------------------------------------------------------------------------
# Columns that record the absence of the event (R3 #2)


def _absence_frame(name: str, time_name: str = "pfs_months") -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(8)
    event = (rng.random(60) < 0.6).astype(int)
    frame = pd.DataFrame({time_name: rng.exponential(20, 60).round(1) + 0.1, "event": event, name: 1 - event})
    return frame, int(event.sum())


@pytest.mark.parametrize(
    ("name", "time_name", "state"),
    [
        ("progression_free", "pfs_months", "being free of the event"),
        ("event_free", "efs_months", "being free of the event"),
        ("disease_free", "dfs_months", "being free of the event"),
        ("relapse_free_status", "rfs_months", "being free of the event"),
        ("ProgressionFree", "pfs_months", "being free of the event"),
        ("alive", "os_months", "being alive"),
        ("is_alive", "os_months", "being alive"),
        ("living", "os_months", "being alive"),
        ("survived", "os_months", "being alive"),
    ],
)
def test_columns_recording_the_absence_of_the_event_are_refused_with_generic_codes(name: str, time_name: str, state: str) -> None:
    frame, _ = _absence_frame(name, time_name)
    for codes in ((0, 1), ("No", "Yes"), (False, True)):
        coded = frame.assign(**{name: np.where(frame[name] == 1, codes[1], codes[0])})
        with pytest.raises(ValueError, match=f"records {state}"):
            compute_km_analysis(coded, time_name, name)
        with pytest.raises(ValueError, match=f"records {state}"):
            compute_km_analysis(coded, time_name, name, event_positive_value=codes[1])
        assert name not in suggest_columns(coded)["event_columns"]
    assert suggest_columns(frame)["event_columns"] == ["event"]


def test_absence_named_columns_with_outcome_labels_and_endpoint_names_stay_usable() -> None:
    frame, n_events = _absence_frame("alive", "os_months")
    frame["alive"] = np.where(frame["event"] == 1, "Dead", "Alive")
    km = compute_km_analysis(frame, "os_months", "alive", event_positive_value="Dead")
    assert km["cohort"]["events"] == n_events
    # "Progression-free survival" names the endpoint: its status column holds the events.
    frame, n_events = _absence_frame("progression_free_survival_status")
    frame["progression_free_survival_status"] = frame["event"]
    km = compute_km_analysis(frame, "pfs_months", "progression_free_survival_status")
    assert km["cohort"]["events"] == n_events
    assert "progression_free_survival_status" in suggest_columns(frame)["event_columns"]


# ---------------------------------------------------------------------------------------
# Outcome-like names, complements, and follow-up time names (R3 #4, #7, #11)


def test_death_alive_and_censoring_flags_are_outcome_columns() -> None:
    rng = np.random.default_rng(1)
    n = 40
    dead = rng.integers(0, 2, n)
    dfs_event = np.maximum(dead, rng.integers(0, 2, n))
    frame = pd.DataFrame(
        {
            "os_months": rng.uniform(1, 60, n).round(1),
            "dfs_months": rng.uniform(1, 60, n).round(1),
            "dfs_event": dfs_event,
            "dead": dead,
            "deceased": np.where(dead == 1, "Yes", "No"),
            "died": dead,
            "alive": 1 - dead,
            "vital": np.where(dead == 1, "Dead", "Alive"),
            "censored": 1 - dfs_event,
            "studied": rng.integers(0, 2, n),  # holds "died" only as a substring
            "age": rng.integers(40, 80, n),
        }
    )
    flagged = {"dead", "deceased", "died", "alive", "vital", "censored"}
    assert flagged <= _survival_outcome_like_columns(frame)
    assert "studied" not in _survival_outcome_like_columns(frame)
    assert not flagged & set(model_feature_candidate_columns(frame))
    assert {"studied", "age"} <= set(model_feature_candidate_columns(frame))
    # The censoring flag is the complement of the event it censors.
    assert "censored" in find_event_equivalent_columns(frame, "dfs_event")
    assert {"alive", "vital", "deceased", "died"} <= find_event_equivalent_columns(frame, "dead")


def test_complements_of_the_event_are_found_on_the_overlapping_rows() -> None:
    n = 12
    frame = pd.DataFrame({"time": np.arange(1, n + 1, dtype=float), "event": [1, 0] * (n // 2)})
    frame["cens_flag"] = 1 - frame["event"]
    frame["free_label"] = np.where(frame["event"] == 1, "No", "Yes")
    frame.loc[[0, 5], "cens_flag"] = np.nan
    frame["age"] = np.arange(50, 50 + n)
    frame["unrelated"] = [1, 1, 0, 0] * (n // 4)
    assert find_event_equivalent_columns(frame, "event") == {"cens_flag", "free_label"}


def test_follow_up_names_are_time_columns_and_baseline_intervals_are_not() -> None:
    for name in ("days_to_last_known_alive", "last_contact_days", "time_to_next_treatment_days", "days_to_last_contact"):
        assert analysis._looks_like_survival_time_column_name(name), name
    for name in ("time_to_surgery_days", "time_since_diagnosis_years", "os_dead", "time_to_treatment_start_months"):
        assert not analysis._looks_like_survival_time_column_name(name), name
    for name in ("time_to_death_days", "time_to_progression_months", "time_in_months", "time_to_last_follow_up_days"):
        assert analysis._looks_like_survival_time_column_name(name), name


def test_baseline_status_columns_are_not_outcomes_unless_their_values_name_one() -> None:
    rng = np.random.default_rng(2)
    n = 50
    frame = pd.DataFrame(
        {
            "os_months": rng.exponential(30, n).round(1) + 0.1,
            "os_event": rng.integers(0, 2, n),
            "diabetes_status": rng.choice(["Yes", "No"], n),
            "marital_status": rng.choice(["Married", "Never married", "Divorced"], n),
            "hiv_status": rng.choice(["Positive", "Negative", "Not tested"], n),
            "insurance_status": rng.choice(["Insured", "Uninsured"], n),
            "patient_status": rng.choice(["Alive", "Dead"], n),
            "status": rng.integers(0, 2, n),
        }
    )
    assert _survival_outcome_like_columns(frame) == {"os_months", "os_event", "patient_status", "status"}
    assert suggest_columns(frame)["event_columns"] == ["os_event", "patient_status", "status"]
    assert "diabetes_status" in model_feature_candidate_columns(frame)
    # An explicit choice is still possible: the user said what the column means.
    km = compute_km_analysis(frame, "os_months", "diabetes_status", event_positive_value="Yes")
    assert km["cohort"]["events"] == int((frame["diabetes_status"] == "Yes").sum())


def _missing_time_frame(missing_censored: int, missing_events: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = rng.exponential(30, 200).round(1) + 0.1
    events = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    times[:missing_censored] = np.nan
    times[100 : 100 + missing_events] = np.nan
    return pd.DataFrame({"os_months": times, "os_event": events})


def test_time_missing_for_most_censored_rows_is_refused_despite_a_few_missing_event_times() -> None:
    # 60 of 100 censored rows but only 4 of 100 event rows lack a time.
    with pytest.raises(ValueError, match="missing for 60 of 100 censored rows but for only 4 of 100 rows with an event"):
        _cohort_frame(_missing_time_frame(60, 4), "os_months", "os_event")
    with pytest.raises(ValueError, match="missing for 60 of 100 rows with an event but for only 4 of 100 censored"):
        _cohort_frame(_missing_time_frame(4, 60), "os_months", "os_event")


def test_clearly_outcome_dependent_missing_time_brings_a_caution() -> None:
    frame = _missing_time_frame(30, 5)
    cohort = _cohort_frame(frame, "os_months", "os_event")
    note = cohort.attrs["time_column_note"]
    assert "missing for 30 of 100 censored rows and for 5 of 100 rows with an event" in note
    assert "underestimated" in note
    km = compute_km_analysis(frame, "os_months", "os_event")
    assert any("30 of 100 censored rows" in caution for caution in km["scientific_summary"]["cautions"])
    assert km["scientific_summary"]["status"] != "robust"
    # Missing times unrelated to the outcome bring no caution.
    assert "time_column_note" not in _cohort_frame(_missing_time_frame(10, 10), "os_months", "os_event").attrs


def test_unusual_time_column_brings_a_caution_even_without_alternatives() -> None:
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"tumour_size": rng.uniform(1, 9, 40).round(2), "os_event": rng.integers(0, 2, 40)})
    cohort = _cohort_frame(frame, "tumour_size", "os_event")
    assert cohort.attrs["time_column_note"].startswith('"tumour_size" does not look like a survival follow-up time column.')
    km = compute_km_analysis(frame, "tumour_size", "os_event")
    assert any("tumour_size" in caution for caution in km["scientific_summary"]["cautions"])
    renamed = frame.rename(columns={"tumour_size": "os_months"})
    assert "time_column_note" not in _cohort_frame(renamed, "os_months", "os_event").attrs


def test_measurements_named_after_the_patient_or_sample_are_not_duplicate_identifiers() -> None:
    rng = np.random.default_rng(5)
    n = 120
    frame = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "os_months": rng.exponential(30, n).round(1) + 0.1,
            "os_event": rng.integers(0, 2, n),
            "sample_purity": rng.uniform(0.2, 0.95, n).round(2),
            "patient_age": rng.integers(35, 90, n),
        }
    )
    assert analysis.detect_duplicate_identifier_columns(frame) == []
    km = compute_km_analysis(frame, "os_months", "os_event")
    assert not any("Identifier column" in caution for caution in km["scientific_summary"]["cautions"])

    repeated = pd.DataFrame(
        {
            "PatientID": ["P1", "P1", "P2", "P3", "P4", "P5"],
            "sample_barcode": ["S1", "S2", "S2", "S3", "S4", "S5"],
            "case_number": [1.0, 2.0, 2.0, 3.0, np.nan, 4.0],  # whole numbers read as floats because of a blank
        }
    )
    flagged = [finding["column"] for finding in analysis.detect_duplicate_identifier_columns(repeated)]
    assert flagged == ["PatientID", "sample_barcode", "case_number"]


def test_generic_time_to_event_names_pair_with_any_endpoint() -> None:
    analysis._validate_endpoint_family_pair("tte", "os_event")
    analysis._validate_endpoint_family_pair("tts", "pfs_event")
    analysis._validate_endpoint_family_pair("time_to_event", "dfs_status")
    with pytest.raises(ValueError, match="different survival endpoints"):
        analysis._validate_endpoint_family_pair("os_months", "pfs_event")

"""Regression tests for the analysis-module review findings (loader, cohort rules, KM/Cox
summaries, and the signature search)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.analysis as analysis
from survival_toolkit.analysis import (
    _ordered_reference_categories,
    _survival_outcome_like_columns,
    compute_cohort_table,
    compute_cox_analysis,
    compute_km_analysis,
    derive_group_column,
    discover_feature_signature,
    find_event_equivalent_columns,
    load_dataframe,
    looks_binary,
    model_feature_candidate_columns,
    preview_cox_analysis_inputs,
    suggest_columns,
)
from survival_toolkit.sample_data import make_example_dataset


def _survival_frame(seed: int, n: int, *, event_rate: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "os_months": np.round(rng.exponential(20.0, n), 2) + 0.1,
            "os_event": (rng.random(n) < event_rate).astype(int),
        }
    )


# ---------------------------------------------------------------------------------------
# Loader: decimal commas, thousands separators, delimiter sniffing, UTF-16 (B1, B4, B8, B9)


def test_decimal_comma_conversion_leaves_dot_decimal_columns_alone() -> None:
    text = "id;time;event;ratio\np0;12.5;1;0,5\np1;3.25;0;1,5\np2;40.75;1;2,5\np3;0.5;1;0,25\np4;7.125;0;3,5\n"
    frame = load_dataframe(text.encode(), "cohort.csv")
    assert frame["time"].tolist() == [12.5, 3.25, 40.75, 0.5, 7.125]
    assert frame["ratio"].tolist() == [0.5, 1.5, 2.5, 0.25, 3.5]


def test_decimal_comma_conversion_keeps_dotted_dates_as_text() -> None:
    text = (
        "id;time;event;visit\np0;12,5;1;01.02.2020\np1;3,25;0;15.03.2019\np2;40,75;1;20.11.2018\n"
        "p3;0,5;1;05.05.2021\np4;1.234,5;0;30.06.2017\n"
    )
    frame = load_dataframe(text.encode(), "cohort.csv")
    assert frame["time"].tolist() == [12.5, 3.25, 40.75, 0.5, 1234.5]
    assert frame["visit"].tolist() == ["01.02.2020", "15.03.2019", "20.11.2018", "05.05.2021", "30.06.2017"]


def test_european_export_keeps_reading_dot_groups_as_thousands() -> None:
    # A German-locale Excel export writes 1234 days as "1.234" next to decimal commas.
    text = "id;days;event;age\np0;1.234;1;60,5\np1;567;0;55,0\np2;12.345;1;70,25\np3;89;1;48,0\n"
    frame = load_dataframe(text.encode(), "cohort.csv")
    assert frame["days"].tolist() == [1234, 567, 12345, 89]
    assert frame["age"].tolist() == [60.5, 55.0, 70.25, 48.0]
    # A genuine dot-decimal value ("12.5") shows "." is a decimal mark, so nothing is regrouped.
    mixed = "id;days;event;age\np0;1.234;1;60,5\np1;12.5;0;55,0\np2;3.25;1;70,25\n"
    frame = load_dataframe(mixed.encode(), "cohort.csv")
    assert frame["days"].tolist() == [1.234, 12.5, 3.25]


def test_tab_file_with_dot_decimals_reads_comma_groups_as_thousands() -> None:
    rows = ["patient\tos_months\tos_event\tread_count"]
    rows += [
        "p1\t12.5\t1\t1,234",
        "p2\t3.25\t0\t12,345",
        "p3\t40.75\t1\t2,001",
        "p4\t0.5\t1\t9,999",
        "p5\t7.125\t0\t3,210",
    ]
    frame = load_dataframe(("\n".join(rows) + "\n").encode(), "cohort.tsv")
    assert frame["os_months"].tolist() == [12.5, 3.25, 40.75, 0.5, 7.125]
    assert frame["read_count"].tolist() == [1234, 12345, 2001, 9999, 3210]


def test_comma_csv_parses_quoted_thousands_separators_and_keeps_every_patient() -> None:
    rng = np.random.default_rng(41)
    n = 120
    days = rng.exponential(900, n).round().astype(int) + 1
    events = rng.integers(0, 2, n)
    cells = [f'"{value:,}"' if value >= 1000 else str(value) for value in days]
    text = "days_to_last_follow_up_or_death,vital_status\n" + "\n".join(
        f"{cell},{'Dead' if event else 'Alive'}" for cell, event in zip(cells, events, strict=True)
    )
    frame = load_dataframe(text.encode(), "export.csv")
    assert frame.iloc[:, 0].tolist() == days.tolist()
    km = compute_km_analysis(frame, "days_to_last_follow_up_or_death", "vital_status", event_positive_value="Dead")
    assert km["cohort"]["n"] == n
    assert km["cohort"]["time_max"] == pytest.approx(float(days.max()))


def test_text_time_values_that_are_not_numbers_are_refused_not_dropped() -> None:
    frame = _survival_frame(1, 30)
    frame["os_months"] = frame["os_months"].astype(object)
    frame.loc[0, "os_months"] = "12 months"
    frame.loc[1, "os_months"] = "[Not Available]"
    frame.loc[2, "os_months"] = "1,234.5"
    with pytest.raises(ValueError, match='value\\(s\\) that are not numbers, such as "12 months"'):
        compute_km_analysis(frame, "os_months", "os_event")
    frame.loc[0, "os_months"] = "NA"
    km = compute_km_analysis(frame, "os_months", "os_event")
    # "NA" and "[Not Available]" are missing; the thousands-grouped value is parsed.
    assert km["cohort"]["n"] == 28
    assert km["cohort"]["time_max"] == pytest.approx(1234.5)


def test_delimiter_sniffing_is_linear_on_crafted_input_and_still_detects_delimiters() -> None:
    crafted = "\n".join([',"a' * 700] * 50)
    start = time.perf_counter()
    analysis._sniff_delimiter(crafted * 2, ",")
    assert time.perf_counter() - start < 1.0
    assert analysis._sniff_delimiter("a;b;c\n1,5;2;3\n4;5,5;6\n", ",") == ";"
    assert analysis._sniff_delimiter('name,age\n"Smith, John",40\n"Doe, Jane",41\n', ";") == ","
    assert analysis._sniff_delimiter("a\tb\n1\tx|y\n2\tz|w\n", ",") == "\t"
    assert analysis._sniff_delimiter("time\n1\n2\n", "\t") == "\t"


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_utf16_text_files_larger_than_the_sniffed_prefix_load(encoding: str) -> None:
    rng = np.random.default_rng(0)
    lines = ["os_months\tos_event\tage\tgroup"]
    lines += [f"{rng.exponential(20):.3f}\t{rng.integers(0, 2)}\t{rng.integers(30, 90)}\tarm_{i % 3}" for i in range(30000)]
    payload = ("\r\n".join(lines) + "\r\n").encode(encoding)
    assert len(payload) > analysis._TEXT_SNIFF_BYTES
    frame = load_dataframe(payload, "export.txt")
    assert frame.shape == (30000, 4)
    assert frame["group"].iloc[-1] == "arm_2"


# ---------------------------------------------------------------------------------------
# Event and time column rules (B2, B3, B5, B6, B10, B11, B20)


@pytest.mark.parametrize(
    "codes",
    [("No", "Yes"), ("N", "Y"), ("false", "true"), (False, True), (0, 1)],
)
def test_censoring_named_columns_with_generic_codes_are_refused(codes: tuple[Any, Any]) -> None:
    rng = np.random.default_rng(31)
    died = rng.random(60) < 0.7
    frame = pd.DataFrame({"os_months": rng.exponential(24, 60) + 0.1, "censored": np.where(died, codes[0], codes[1])})
    if isinstance(codes[0], bool):
        frame["censored"] = ~died
    with pytest.raises(ValueError, match="censoring indicator"):
        compute_km_analysis(frame, "os_months", "censored")
    with pytest.raises(ValueError, match="censoring indicator"):
        compute_km_analysis(frame, "os_months", "censored", event_positive_value=codes[1])


def test_censoring_named_column_with_outcome_labels_is_read_by_its_labels() -> None:
    frame = _survival_frame(2, 40)
    frame["censored"] = np.where(frame["os_event"] == 1, "Dead", "Alive")
    km = compute_km_analysis(frame, "os_months", "censored", event_positive_value="Dead")
    assert km["cohort"]["events"] == int(frame["os_event"].sum())


def _gdc_frame(seed: int = 3, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dead = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "vital_status": np.where(dead == 1, "Dead", "Alive"),
            "days_to_death": np.where(dead == 1, rng.exponential(700, n).round() + 1, np.nan),
            "days_to_last_follow_up": np.where(dead == 1, np.nan, rng.exponential(900, n).round() + 1),
        }
    )


def test_time_missing_for_every_censored_row_is_refused() -> None:
    frame = _gdc_frame()
    with pytest.raises(ValueError, match="missing for .* censored rows.*last follow-up"):
        compute_km_analysis(frame, "days_to_death", "vital_status", event_positive_value="Dead")
    with pytest.raises(ValueError, match="rows with an event"):
        compute_km_analysis(frame, "days_to_last_follow_up", "vital_status", event_positive_value="Dead")
    combined = frame.assign(os_days=frame["days_to_death"].fillna(frame["days_to_last_follow_up"]))
    km = compute_km_analysis(combined, "os_days", "vital_status", event_positive_value="Dead")
    assert km["cohort"]["n"] == len(frame)


def test_a_few_missing_times_are_still_dropped_as_missing() -> None:
    frame = _survival_frame(4, 200)
    censored_rows = frame.index[frame["os_event"] == 0][:6]
    frame.loc[censored_rows, "os_months"] = np.nan
    km = compute_km_analysis(frame, "os_months", "os_event")
    assert km["cohort"]["n"] == 194
    assert km["cohort"]["dropped_missing_rows"] == 6


def test_all_missing_time_reports_missing_values_not_positivity() -> None:
    frame = _survival_frame(5, 30)
    frame["os_months"] = np.nan
    with pytest.raises(ValueError, match="has no usable values"):
        compute_km_analysis(frame, "os_months", "os_event")


def test_binary_biomarker_status_columns_are_not_survival_outcomes() -> None:
    rng = np.random.default_rng(5)
    n = 120
    frame = pd.DataFrame(
        {
            "os_months": rng.exponential(30, n) + 0.1,
            "os_event": rng.integers(0, 2, n),
            "her2_status": rng.choice(["Positive", "Negative"], n),
            "er_status": rng.integers(0, 2, n),
            "msi_status": rng.choice(["MSI-H", "MSS"], n),
            "idh_status": rng.choice(["Mutant", "Wildtype"], n),
            "mgmt_status": rng.choice(["Methylated", "Unmethylated"], n),
            "nodal_status": rng.integers(0, 2, n),
            "menopausal_status": rng.choice(["Pre", "Post"], n),
            "performance_status": rng.integers(0, 2, n),
            "hpv_status": rng.choice(["HPV+", "HPV-"], n),
            "vital_status": rng.choice(["Alive", "Dead"], n),
        }
    )
    outcome_like = _survival_outcome_like_columns(frame)
    biomarkers = [column for column in frame.columns if column not in {"os_months", "os_event", "vital_status"}]
    assert outcome_like == {"os_months", "os_event", "vital_status"}
    assert set(biomarkers) <= set(model_feature_candidate_columns(frame))
    assert not set(biomarkers) & set(suggest_columns(frame)["event_columns"])


def test_tcga_cdr_and_cbioportal_endpoints_are_usable_time_columns() -> None:
    rng = np.random.default_rng(9)
    n = 100
    cdr = pd.DataFrame(
        {
            "OS": rng.integers(0, 2, n),
            "OS.time": rng.exponential(900, n).round() + 1,
            "PFI": rng.integers(0, 2, n),
            "PFI.time": rng.exponential(600, n).round() + 1,
            "DFI": rng.integers(0, 2, n),
            "DFI.time": rng.exponential(700, n).round() + 1,
        }
    )
    suggestions = suggest_columns(cdr)
    assert suggestions["time_columns"] == ["OS.time", "PFI.time", "DFI.time"]
    assert suggestions["event_columns"] == ["OS", "PFI", "DFI"]
    for time_column, event_column in (("PFI.time", "PFI"), ("DFI.time", "DFI"), ("OS.time", "OS")):
        km = compute_km_analysis(cdr, time_column, event_column)
        assert km["cohort"]["n"] == n
        assert not any("does not look like" in caution for caution in km["scientific_summary"]["cautions"])
    with pytest.raises(ValueError, match="different survival endpoints"):
        compute_km_analysis(cdr, "OS.time", "PFI")
    with pytest.raises(ValueError, match="only the values 0 and 1"):
        compute_km_analysis(cdr.assign(dead=cdr["OS"]), "PFI", "dead")

    cbio = pd.DataFrame(
        {
            "Overall Survival (Months)": rng.exponential(30, n) + 0.1,
            "Overall Survival Status": rng.choice(["0:LIVING", "1:DECEASED"], n),
            "Disease Free (Months)": rng.exponential(20, n) + 0.1,
            "Disease Free Status": rng.choice(["0:DiseaseFree", "1:Recurred/Progressed"], n),
        }
    )
    km = compute_km_analysis(cbio, "Disease Free (Months)", "Disease Free Status", event_positive_value="1:Recurred/Progressed")
    assert km["cohort"]["n"] == n
    for name in ("overall_time", "TTP", "Recurrence Free Months", "obs_time"):
        assert analysis._looks_like_survival_time_column_name(name), name
    for name in ("overall_response", "observed_value", "Disease Free Status", "year", "treatment free interval"):
        assert not analysis._looks_like_survival_time_column_name(name), name
    # An unrecognized name is a caution with the results, not a refusal.
    unusual = cdr.rename(columns={"PFI.time": "months"})
    km = compute_km_analysis(unusual, "months", "PFI")
    assert any('"months" does not look like a survival follow-up time column' in c for c in km["scientific_summary"]["cautions"])


@pytest.mark.parametrize(
    "event_column",
    ["treatment_failure", "progression_on_therapy", "death_on_treatment", "relapse_after_therapy"],
)
def test_treatment_related_endpoint_names_are_accepted_as_events(event_column: str) -> None:
    frame = _survival_frame(21, 80)
    frame = frame.rename(columns={"os_months": "ttf_months", "os_event": event_column})
    km = compute_km_analysis(frame, "ttf_months", event_column)
    assert km["cohort"]["n"] == 80


def test_baseline_status_columns_are_still_refused_as_events() -> None:
    frame = _survival_frame(22, 80)
    frame["treatment_status"] = np.random.default_rng(1).integers(0, 2, 80)
    with pytest.raises(ValueError, match="looks more like a baseline characteristic"):
        compute_km_analysis(frame, "os_months", "treatment_status")


def test_time_and_event_columns_cannot_be_reused_as_inputs() -> None:
    frame = make_example_dataset(seed=1, n_patients=80)
    with pytest.raises(ValueError, match='"os_event" is the event column'):
        compute_km_analysis(frame, "os_months", "os_event", group_column="os_event")
    with pytest.raises(ValueError, match='"os_months" is the survival time column'):
        compute_km_analysis(frame, "os_months", "os_event", group_column="os_months")
    with pytest.raises(ValueError, match='"os_months" is the survival time column'):
        compute_cox_analysis(frame, "os_months", "os_event", ["age", "os_months"])
    with pytest.raises(ValueError, match='"os_event" is the event column'):
        compute_cox_analysis(frame, "os_months", "os_event", ["age"], strata_columns=["os_event"])
    with pytest.raises(ValueError, match='"os_event" is the event column'):
        discover_feature_signature(frame, "os_months", "os_event", ["os_event", "age"], bootstrap_iterations=0)


def test_duplicate_and_unselected_column_choices_are_handled() -> None:
    frame = make_example_dataset(seed=2, n_patients=80)
    table = compute_cohort_table(frame, ["age", "age", "sex", "sex"])
    assert [row["Variable"] for row in table["rows"]].count("age") == 2  # summary + missing rows
    with pytest.raises(ValueError, match="Categorical covariates must also be selected as covariates: stage"):
        compute_cox_analysis(frame, "os_months", "os_event", ["age"], categorical_covariates=["stage"])
    duplicated = compute_cox_analysis(frame, "os_months", "os_event", ["age", "age"])
    assert [row["Variable"] for row in duplicated["results_table"]] == ["age"]


# ---------------------------------------------------------------------------------------
# Profiling cost (B16)


def test_event_decoding_runs_once_per_distinct_value(monkeypatch) -> None:
    calls = {"count": 0}
    original = analysis._outcome_status_value_family

    def _counting(value: Any) -> str | None:
        calls["count"] += 1
        return original(value)

    monkeypatch.setattr(analysis, "_outcome_status_value_family", _counting)
    status = pd.Series(np.where(np.arange(20000) % 3 == 0, "Dead", "Alive"))
    assert looks_binary(status)
    assert calls["count"] < 20
    calls["count"] = 0
    numbers = pd.Series(np.random.default_rng(0).normal(size=20000))
    assert not looks_binary(numbers)
    assert not analysis._has_recognizable_event_coding(numbers)
    assert calls["count"] == 0

    wide = pd.DataFrame({f"gene_{j}": np.random.default_rng(j).normal(size=200) for j in range(300)})
    wide["os_event"] = np.random.default_rng(1).integers(0, 2, 200)
    wide["os_copy"] = wide["os_event"]
    assert find_event_equivalent_columns(wide, "os_event") == {"os_copy"}
    assert calls["count"] < 50


def test_decimal_strings_are_not_split_into_event_tokens() -> None:
    # "1.5" once decoded as the event token "1" and "0.5" as censoring.
    with pytest.raises(ValueError, match="Could not infer event coding"):
        analysis.coerce_event(pd.Series(["1.5", "0.5", "1.5", "0.5"], dtype=object))


# ---------------------------------------------------------------------------------------
# Kaplan-Meier (B13, B18, B21, C8, C22, many groups)


def test_outcome_informed_groups_never_get_a_fresh_test_or_rmst_interval() -> None:
    rng = np.random.default_rng(8)
    frame = _survival_frame(8, 200)
    marker = rng.normal(size=200)
    frame["grp"] = np.where(marker > np.median(marker), "high", "low")
    km = compute_km_analysis(frame, "os_months", "os_event", group_column="grp", outcome_informed_group=True)
    assert km["test"] is None and km["logrank_p"] is None and km["pairwise_table"] == []
    contrast = km["rmst_contrast"]
    assert contrast["estimate"] is not None
    assert contrast["ci_lower"] is None and contrast["ci_upper"] is None and contrast["se"] is None
    assert contrast["p_value"] is None
    assert not any("confidence interval" in strength and "RMST contrast" in strength for strength in km["scientific_summary"]["strengths"])
    assert any("interval, because the split was chosen using the outcome" in caution for caution in km["scientific_summary"]["cautions"])


def test_single_level_group_does_not_claim_a_global_test() -> None:
    frame = _survival_frame(11, 120).assign(arm="A")
    km = compute_km_analysis(frame, "os_months", "os_event", group_column="arm")
    summary = km["scientific_summary"]
    assert km["test"] is None
    assert not any("comparison was run" in strength for strength in summary["strengths"])
    assert any("Only one group of arm" in caution for caution in summary["cautions"])
    assert "Only one group of arm" in summary["headline"]


def test_exhausted_risk_set_does_not_raise_the_retired_rmst_caution() -> None:
    frame = pd.DataFrame({"time": [1.0, 1.0, 2.0, 2.0], "event": [1, 1, 1, 1], "group": ["A", "A", "B", "B"]})
    km = compute_km_analysis(frame, "time", "event", group_column="group", max_time=2.0)
    assert not any("risk set was exhausted" in caution for caution in km["scientific_summary"]["cautions"])


@pytest.mark.parametrize("max_time", [float("nan"), float("inf"), -1.0])
def test_km_rejects_non_finite_or_non_positive_max_time(max_time: float) -> None:
    frame = make_example_dataset(seed=3, n_patients=60)
    with pytest.raises(ValueError, match="max_time must be positive and finite"):
        compute_km_analysis(frame, "os_months", "os_event", max_time=max_time)


def test_many_small_groups_match_r_and_pairs_without_events_do_not_crash() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"t": rng.exponential(10, 60), "e": rng.integers(0, 2, 60), "g": [f"g{i:02d}" for i in range(30)] * 2}
    )
    km = compute_km_analysis(frame, "t", "e", group_column="g")
    # R survival 3.8.6: survdiff(Surv(t, e) ~ g) and rho = 1 on the same data.
    assert km["test"]["chisq"] == pytest.approx(64.236675708589, rel=1e-10)
    fleming = compute_km_analysis(frame, "t", "e", group_column="g", logrank_weight="fleming_harrington", fh_p=1.0)
    assert fleming["test"]["chisq"] == pytest.approx(53.590857163811, rel=1e-10)
    no_event_pair = next(row for row in km["pairwise_table"] if row["Comparison"] == "g05 vs g09")
    assert no_event_pair["Chi-square"] == 0.0 and no_event_pair["P value"] == 1.0


def test_groups_never_at_risk_at_an_event_time_are_left_out_like_r() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.5, 0.6, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 4.5, 6.0],
            "event": [0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
            "g": ["early", "early", "a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    km = compute_km_analysis(frame, "time", "event", group_column="g")
    reference = compute_km_analysis(frame[frame["g"] != "early"], "time", "event", group_column="g")
    assert km["test"]["chisq"] == pytest.approx(reference["test"]["chisq"], rel=1e-12)


# ---------------------------------------------------------------------------------------
# Cox (B7, B12, B14, B15, B19, C7, C16, C23)


def test_identifier_like_categorical_covariate_is_refused_before_the_fit() -> None:
    frame = _survival_frame(3, 300)
    frame["age"] = np.random.default_rng(3).normal(60, 8, 300)
    frame["case_submitter_id"] = [f"case-{i:06d}" for i in range(300)]
    start = time.perf_counter()
    with pytest.raises(ValueError, match="300 observed levels, which looks like an identifier"):
        compute_cox_analysis(frame, "os_months", "os_event", ["age", "case_submitter_id"])
    assert time.perf_counter() - start < 10.0
    preview = preview_cox_analysis_inputs(frame, "os_months", "os_event", ["age", "case_submitter_id"])
    assert any("looks like an identifier" in warning for warning in preview["stability_warnings"])


def test_cox_with_more_coefficients_than_events_is_refused() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "event": [1, 0, 1, 0, 0, 0, 0, 0],
            "x1": [0.1, 0.4, 0.3, 0.8, 0.2, 0.9, 0.5, 0.7],
            "x2": [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0],
        }
    )
    with pytest.raises(ValueError, match="2 coefficient\\(s\\) from only 2 event"):
        compute_cox_analysis(frame, "time", "event", ["x1", "x2"])


def test_clean_cox_fit_reaches_robust_status() -> None:
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-0.7 * x)) * 10
    censor_time = rng.exponential(12, n)
    frame = pd.DataFrame(
        {"os_time": np.minimum(event_time, censor_time) + 0.01, "os_event": (event_time <= censor_time).astype(int), "x": x}
    )
    summary = compute_cox_analysis(frame, "os_time", "os_event", ["x"])["scientific_summary"]
    assert summary["status"] == "robust"
    assert any("complete-case rows" in caution for caution in summary["cautions"])


def test_cox_counts_covariate_infinities_and_words_time_infinities_separately() -> None:
    frame = make_example_dataset(seed=3, n_patients=120)
    frame.loc[:2, "age"] = np.inf
    frame.loc[3:4, "os_months"] = np.inf
    result = compute_cox_analysis(frame, "os_months", "os_event", ["age", "biomarker_score"])
    stats = result["model_stats"]
    assert stats["rows_with_infinite_values"] == 3
    assert stats["rows_with_infinite_outcome_values"] == 2
    assert stats["outcome_rows"] == 118 and stats["dropped_rows"] == 3
    cautions = result["scientific_summary"]["cautions"]
    assert any(caution.startswith("3 outcome-valid row(s) contained +/-Inf") for caution in cautions)
    assert any(caution.startswith("2 row(s) with a +/-Inf survival time were excluded before") for caution in cautions)


def test_numeric_coded_categorical_covariates_use_the_smallest_value_as_reference() -> None:
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame(
        {
            "os_time": rng.exponential(700, n).round() + 1,
            "os_event": rng.integers(0, 2, n),
            "kps": rng.choice([60, 70, 80, 90, 100], n),
        }
    )
    result = compute_cox_analysis(frame, "os_time", "os_event", ["kps"], categorical_covariates=["kps"])
    assert [row["Label"] for row in result["results_table"]] == [
        "kps: 70 vs 60",
        "kps: 80 vs 60",
        "kps: 90 vs 60",
        "kps: 100 vs 60",
    ]


def test_cox_preview_names_the_reference_level_the_fit_uses() -> None:
    rng = np.random.default_rng(12)
    n = 120
    frame = pd.DataFrame(
        {
            "os_months": rng.exponential(20, n) + 0.1,
            "os_event": rng.integers(0, 2, n),
            "stage": rng.choice(["Stage I", "Stage II", "Stage III"], n),
            "age": rng.normal(60, 8, n),
        }
    )
    frame.loc[frame["stage"] == "Stage I", "age"] = np.nan
    preview = preview_cox_analysis_inputs(frame, "os_months", "os_event", ["stage", "age"], categorical_covariates=["stage"])
    assert all(level["rows"] > 0 for level in preview["risky_levels"])
    assert not any('"Stage I"' in warning or "0 rows" in warning for warning in preview["stability_warnings"])
    fit = compute_cox_analysis(frame, "os_months", "os_event", ["stage", "age"], categorical_covariates=["stage"])
    assert [row["Label"] for row in fit["results_table"]][0] == "stage: Stage III vs Stage II"


def test_nullable_boolean_covariate_fits_like_float_indicator() -> None:
    frame = make_example_dataset(seed=3, n_patients=150)
    flag = frame["age"] > 55
    nullable = frame.assign(flag=pd.array(flag, dtype="boolean"))
    nullable.loc[:3, "flag"] = pd.NA
    plain = frame.assign(flag=flag.astype(float))
    plain.loc[:3, "flag"] = np.nan
    hr_nullable = compute_cox_analysis(nullable, "os_months", "os_event", ["flag", "age"])["results_table"]
    hr_plain = compute_cox_analysis(plain, "os_months", "os_event", ["flag", "age"])["results_table"]
    assert [row["Hazard ratio"] for row in hr_nullable] == pytest.approx([row["Hazard ratio"] for row in hr_plain], rel=1e-12)


def test_binary_covariates_are_not_screened_as_continuous_terms() -> None:
    frame = make_example_dataset(seed=3, n_patients=150)
    frame["flag"] = frame["age"] > 60
    frame["flag01"] = (frame["biomarker_score"] > 0).astype(int)
    stats = compute_cox_analysis(frame, "os_months", "os_event", ["flag", "flag01", "age"])["model_stats"]
    assert stats["martingale_terms"] == ["age"]


@pytest.mark.parametrize("name", ["tumor\nsize", 'size"x', "size\\", "grade\r"])
def test_cox_fits_columns_whose_names_break_formulas(name: str) -> None:
    rng = np.random.default_rng(4)
    frame = _survival_frame(4, 80)
    frame[name] = rng.normal(size=80)
    result = compute_cox_analysis(frame, "os_months", "os_event", [name])
    assert [row["Variable"] for row in result["results_table"]] == [name]
    assert [row["Label"] for row in result["results_table"]] == [name]
    assert [row["Term"] for row in result["diagnostics_table"] if row.get("_kind") != "global"] == [name]

    quoted = frame.drop(columns=[name]).assign(**{'stage "path"': rng.choice(["I", "II"], 80)})
    categorical = compute_cox_analysis(quoted, "os_months", "os_event", ['stage "path"'])
    assert [(row["Variable"], row["Label"]) for row in categorical["results_table"]] == [
        ('stage "path"', 'stage "path": II vs I')
    ]


def test_safe_column_names_fit_exactly_as_before() -> None:
    frame = make_example_dataset(seed=23, n_patients=160)
    result = compute_cox_analysis(frame, "os_months", "os_event", ["age", "stage"])
    assert result["formula"] == 'Q("os_months") ~ Q("age") + C(Q("stage"))'
    aliased = frame.rename(columns={"age": 'age "years"'})
    renamed = compute_cox_analysis(aliased, "os_months", "os_event", ['age "years"', "stage"])
    assert [row["Beta"] for row in renamed["results_table"]] == [row["Beta"] for row in result["results_table"]]


# ---------------------------------------------------------------------------------------
# Cohort table (C2)


def test_grouped_cohort_table_counts_non_integer_levels_in_every_group() -> None:
    rng = np.random.default_rng(7)
    n = 80
    frame = pd.DataFrame({"dose": rng.choice([0.5, 1.0], n), "arm": rng.choice(["A", "B"], n)})
    frame.loc[frame["arm"] == "A", "dose"] = 1.0
    table = compute_cohort_table(frame, ["dose"], group_column="arm")
    rows = {row["Statistic"]: row for row in table["rows"] if row["Variable"] == "dose"}
    n_a = int((frame["arm"] == "A").sum())
    assert rows["1.0"]["A"] == f"{n_a} (100.0%)"
    assert rows["0.5"]["A"] == "0 (0.0%)"
    n_b_half = int(((frame["arm"] == "B") & (frame["dose"] == 0.5)).sum())
    assert rows["0.5"]["B"].startswith(f"{n_b_half} (")


# ---------------------------------------------------------------------------------------
# Derived groups (C7, C14)


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
@pytest.mark.parametrize("method", ["median_split", "tertile_split", "extreme_split"])
def test_derived_groups_accept_nullable_numeric_sources(dtype: str, method: str) -> None:
    frame = make_example_dataset(seed=3, n_patients=120)
    values = (frame["biomarker_score"] * 100).round()
    frame["marker"] = pd.array(values.astype(int) if dtype == "Int64" else values, dtype=dtype)
    frame.loc[:4, "marker"] = pd.NA
    updated, column, summary = derive_group_column(frame, "marker", method, cutoff="25")
    assert updated.loc[:4, column].isna().all()
    assert summary["missing_count"] == 5


def test_derived_group_counts_separate_missing_values_from_excluded_middle_rows() -> None:
    frame = make_example_dataset(seed=23, n_patients=160)
    frame.loc[:9, "age"] = np.nan
    _, _, summary = derive_group_column(frame, "age", "extreme_split", cutoff="25")
    counts = {row["group"]: row["n"] for row in summary["counts"]}
    assert counts["Missing"] == 10
    assert counts["Excluded (middle range)"] == summary["excluded_count"]
    assert sum(counts.values()) == 160
    with pytest.raises(ValueError, match='"Missing" is reserved'):
        derive_group_column(frame, "age", "median_split", lower_label="Missing", upper_label="High")
    with pytest.raises(ValueError, match="must differ"):
        derive_group_column(frame, "age", "median_split", lower_label="High", upper_label="High")


# ---------------------------------------------------------------------------------------
# Reference levels (C20)


def test_negated_labels_are_the_reference_level() -> None:
    assert _ordered_reference_categories(["Mutated", "Non-mutated"], "EGFR_status") == ["Non-mutated", "Mutated"]
    assert _ordered_reference_categories(["Mutant", "Unmutated"], "IGHV") == ["Unmutated", "Mutant"]
    assert _ordered_reference_categories(["Positive", "Not detected"], "HPV") == ["Not detected", "Positive"]
    assert _ordered_reference_categories(["Methylated", "Unmethylated"], "MGMT") == ["Unmethylated", "Methylated"]
    assert _ordered_reference_categories(["Mutated", "Wildtype"], "kras_status") == ["Wildtype", "Mutated"]
    assert _ordered_reference_categories(["Yes", "No"], "flag") == ["No", "Yes"]
    assert _ordered_reference_categories(["Positive", "Negative"], "er") == ["Negative", "Positive"]


# ---------------------------------------------------------------------------------------
# Signature search (C1, C3, C4, C5, C6, C13, C17, C18, C19, C21)


def _signature_frame(seed: int, n: int = 240, n_features: int = 6, event_rate: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = _survival_frame(seed, n, event_rate=event_rate)
    for index in range(n_features):
        frame[f"g{index}"] = rng.normal(size=n)
    return frame


def test_permutation_family_is_every_size_feasible_combination(monkeypatch) -> None:
    # Few events: many combinations meet the group-size rule but not the event rule.
    frame = _signature_frame(5, n=300, event_rate=0.08)
    captured: dict[str, Any] = {}
    original = analysis._search_adjusted_permutation_p_values

    def _capture(times, events, masks, observed_stats, **kwargs):
        captured["masks"] = np.asarray(masks, dtype=bool)
        captured["observed"] = np.asarray(observed_stats, dtype=float)
        return original(times, events, masks, observed_stats, **kwargs)

    monkeypatch.setattr(analysis, "_search_adjusted_permutation_p_values", _capture)
    _, _, payload = discover_feature_signature(
        frame, "os_months", "os_event", [f"g{index}" for index in range(6)],
        max_combination_size=2, bootstrap_iterations=0, permutation_iterations=19, random_seed=4,
    )
    search = payload["search_space"]
    family = captured["masks"]
    assert family.shape[1] == search["size_feasible_combinations"] == search["permutation_family_size"]
    assert family.shape[1] > search["tested_combinations"] == captured["observed"].size
    # Every family member meets the size rule; some fail the event rule on the observed outcome.
    min_group = search["min_group_size"]
    sizes = family.sum(axis=0)
    assert np.all((sizes >= min_group) & (family.shape[0] - sizes >= min_group))
    cohort = analysis._cohort_frame(frame, "os_months", "os_event", extra_columns=[f"g{index}" for index in range(6)])
    events = cohort["os_event"].to_numpy(dtype=float)
    group_events = events @ family
    event_rule = (group_events >= search["min_events_per_group"]) & (events.sum() - group_events >= search["min_events_per_group"])
    assert (~event_rule).any()


def test_larger_permutation_family_never_lowers_permutation_p_values() -> None:
    rng = np.random.default_rng(2)
    n = 150
    times = rng.exponential(1.0, n)
    events = (rng.random(n) < 0.3).astype(float)
    family = rng.random((n, 40)) < 0.4
    observed = analysis._vectorized_logrank_chisq(times, events, family[:, :10])
    subset_p, _ = analysis._search_adjusted_permutation_p_values(
        times, events, family[:, :10], observed, min_events_per_group=3, n_iterations=60, random_seed=9
    )
    full_p, _ = analysis._search_adjusted_permutation_p_values(
        times, events, family, observed, min_events_per_group=3, n_iterations=60, random_seed=9
    )
    assert np.all(full_p >= subset_p - 1e-12)
    assert np.any(full_p > subset_p)


def _reference_permutation_p_values(times, events, masks, observed, *, min_events_per_group, n_iterations, random_seed):
    """The previous permutation loop, kept to check the blocked rewrite gives identical p-values."""
    mask_arr = np.asarray(masks, dtype=bool)
    events_arr = np.asarray(events, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    total_events = float(events_arr.sum())
    rng = np.random.default_rng(random_seed)
    max_stats = []
    for _ in range(int(n_iterations)):
        permutation = rng.permutation(times_arr.size)
        perm_times = times_arr[permutation]
        perm_events = events_arr[permutation]
        group_events = perm_events @ mask_arr
        eligible = (group_events >= min_events_per_group) & ((total_events - group_events) >= min_events_per_group)
        if not np.any(eligible):
            continue
        order = np.argsort(perm_times, kind="mergesort")
        outcome = analysis._logrank_outcome(perm_times, perm_events)
        statistics = analysis._logrank_chisq_sorted(outcome, mask_arr[:, eligible][order])
        finite = statistics[np.isfinite(statistics)]
        if finite.size:
            max_stats.append(float(np.max(finite)))
    max_array = np.asarray(max_stats, dtype=float)
    exceed = (max_array[np.newaxis, :] >= np.asarray(observed)[:, np.newaxis] - 1e-9).sum(axis=1)
    return (exceed + 1.0) / (max_array.size + 1.0), int(max_array.size)


def test_blocked_permutation_scoring_matches_the_plain_loop(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    n = 90
    times = np.round(rng.exponential(1.0, n), 1)
    events = (rng.random(n) < 0.5).astype(float)
    masks = rng.random((n, 25)) < 0.5
    observed = analysis._vectorized_logrank_chisq(times, events, masks)
    expected, expected_valid = _reference_permutation_p_values(
        times, events, masks, observed, min_events_per_group=3, n_iterations=40, random_seed=5
    )
    monkeypatch.setattr(analysis, "_LOGRANK_BLOCK_CELLS", n * 4)
    blocked, valid = analysis._search_adjusted_permutation_p_values(
        times, events, masks, observed, min_events_per_group=3, n_iterations=40, random_seed=5
    )
    assert valid == expected_valid
    assert np.array_equal(blocked, expected)
    # Column blocks give the same statistics as one pass over every column.
    assert np.allclose(analysis._vectorized_logrank_chisq(times, events, masks), observed, rtol=0, atol=1e-12)


def test_permutation_p_values_are_roughly_calibrated_under_the_null() -> None:
    # Small null simulation: the smallest search-adjusted p-value falls at or below 0.1 in
    # roughly 10% of datasets; the lenient bound only catches gross anti-conservatism.
    hits = 0
    n_datasets = 50
    for seed in range(n_datasets):
        frame = _signature_frame(100 + seed, n=150, n_features=4, event_rate=0.15)
        _, _, payload = discover_feature_signature(
            frame, "os_months", "os_event", [f"g{index}" for index in range(4)],
            max_combination_size=1, bootstrap_iterations=0, permutation_iterations=39, random_seed=seed,
        )
        smallest = min(row["Permutation p"] for row in payload["results_table"])
        hits += int(smallest <= 0.1)
    assert hits / n_datasets <= 0.25


def test_bootstrap_minimums_scale_with_the_resample_and_skips_count_as_failures() -> None:
    rng = np.random.default_rng(5)
    n = 400
    flag = np.zeros(n, dtype=int)
    flag[rng.choice(n, 52, replace=False)] = 1
    hazard = np.where(flag == 1, 4.0, 1.0)
    event_time = rng.exponential(10 / hazard)
    censor_time = rng.exponential(15, n)
    frame = pd.DataFrame(
        {
            "os_months": np.minimum(event_time, censor_time) + 0.01,
            "os_event": (event_time <= censor_time).astype(int),
            "mut": np.where(flag == 1, "mutant", "wildtype"),
            "noise": rng.normal(size=n),
        }
    )
    _, _, payload = discover_feature_signature(
        frame, "os_months", "os_event", ["mut", "noise"], max_combination_size=1,
        bootstrap_iterations=40, bootstrap_sample_fraction=0.4, random_seed=1,
    )
    best = payload["best_split"]
    assert best["Signature"] == 'mut == "mutant"'
    assert best["Bootstrap valid resamples"] > 30
    support = best["Bootstrap support (p<alpha)"]
    assert support <= best["Bootstrap valid resamples"] / 40 + 1e-12

    metrics = analysis._bootstrap_signature_metrics(
        frame=pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "event": [1, 1, 0, 1], "flag": ["a", "b", "a", "b"]}),
        time_column="time",
        event_column="event",
        combo=[{"column": "flag", "kind": "categorical_level", "level": "b", "reference": "a", "label": "flag"}],
        combo_operator="AND",
        min_group_size=1,
        n_iterations=20,
        sample_fraction=1.0,
        random_seed=3,
        significance_level=0.05,
    )
    valid = metrics["Bootstrap valid resamples"]
    assert valid + metrics["Bootstrap skipped resamples"] == 20
    if metrics["Bootstrap HR direction consistency"] is not None:
        assert metrics["Bootstrap HR direction consistency"] <= valid / 20 + 1e-12


def test_robustness_checks_stop_once_enough_significant_signatures_are_known(monkeypatch) -> None:
    frame = _signature_frame(17, n=260, n_features=10)
    calls = {"count": 0}

    def _fake_cox(times, events, mask, alpha=0.05):
        calls["count"] += 1
        return {"Hazard ratio (signature+ vs -)": 1.5, "HR CI lower": 1.1, "HR CI upper": 2.0}

    monkeypatch.setattr(analysis, "survdiff", lambda times, events, groups: (20.0, 1e-5))
    monkeypatch.setattr(analysis, "_signature_cox_metrics", _fake_cox)
    _, _, payload = discover_feature_signature(
        frame, "os_months", "os_event", [f"g{index}" for index in range(10)],
        max_combination_size=2, top_k=5, bootstrap_iterations=0, random_seed=3,
    )
    search = payload["search_space"]
    # Every row passes BH, but the first 80 already hold 5 significant signatures.
    assert search["tested_combinations"] > 200
    assert calls["count"] == 80 == search["robustness_evaluated_signatures"]
    assert search["robustness_unevaluated_signatures"] == search["tested_combinations"] - 80
    assert any("not screened for robustness" in caution for caution in payload["scientific_summary"]["cautions"])


def test_signature_search_counts_every_evaluated_combination_toward_the_cap(monkeypatch) -> None:
    frame = _signature_frame(19, n=200, n_features=8)
    evaluate_calls = {"count": 0}
    original = analysis._evaluate_indicator

    def _counting(df, indicator):
        evaluate_calls["count"] += 1
        return original(df, indicator)

    monkeypatch.setattr(analysis, "_evaluate_indicator", _counting)
    monkeypatch.setattr(analysis, "_SIGNATURE_MAX_EVALUATED_COMBINATIONS", 50)
    _, _, payload = discover_feature_signature(
        frame, "os_months", "os_event", [f"g{index}" for index in range(8)],
        max_combination_size=3, combination_operator="and", bootstrap_iterations=0, random_seed=1,
    )
    search = payload["search_space"]
    assert search["truncated"] is True
    assert search["evaluated_combinations"] == 50
    # Screening evaluates each indicator once, not once per combination.
    assert evaluate_calls["count"] <= search["generated_indicators"] + 5


def test_signature_thresholds_follow_the_significance_level() -> None:
    rng = np.random.default_rng(6)
    times = rng.exponential(1.0, 200)
    events = (rng.random(200) < 0.6).astype(int)
    mask = rng.random(200) < 0.4
    wide = analysis._signature_cox_metrics(times, events, mask, alpha=0.05)
    narrow = analysis._signature_cox_metrics(times, events, mask, alpha=0.2)
    assert narrow["HR CI lower"] > wide["HR CI lower"] and narrow["HR CI upper"] < wide["HR CI upper"]
    row = {
        "BH adjusted p": 0.01,
        "Hazard ratio (signature+ vs -)": 2.0,
        "Bootstrap support (p<alpha)": 0.5,
        "Bootstrap HR direction consistency": 0.8,
        "Validation support (p<alpha)": None,
        "Permutation p": 0.08,
        "Rule count": 1,
    }
    assert analysis._stability_score(row, 0.1) > analysis._stability_score(row, 0.05)


def test_programming_errors_in_the_screen_propagate(monkeypatch) -> None:
    frame = _signature_frame(23, n=120, n_features=3)

    def _broken(times, events, groups):
        raise KeyError("bug")

    monkeypatch.setattr(analysis, "survdiff", _broken)
    with pytest.raises(KeyError):
        discover_feature_signature(frame, "os_months", "os_event", ["g0", "g1", "g2"], bootstrap_iterations=0)


def test_signature_recipe_keeps_exact_rules_and_reports_skipped_candidates() -> None:
    frame = _signature_frame(29, n=200, n_features=3)
    frame["center"] = [f"site{i % 12}" for i in range(200)]
    _, _, payload = discover_feature_signature(
        frame, "os_months", "os_event", ["g0", "g1", "g2", "center"], max_combination_size=1, bootstrap_iterations=0
    )
    rules = payload["signature_recipe"]["rules"]
    assert rules and all("kind" in rule and "column" in rule for rule in rules)
    numeric_rules = [rule for rule in rules if rule["kind"] == "numeric_gt"]
    for rule in numeric_rules:
        expected = float(pd.to_numeric(frame[rule["column"]]).quantile(rule["quantile"]))
        assert rule["threshold"] == expected
    skipped = payload["search_space"]["skipped_candidates"]
    assert skipped == [{"column": "center", "reason": "12 levels; categorical candidates may have at most 8"}]
    assert any("center (12 levels" in caution for caution in payload["scientific_summary"]["cautions"])


def test_signature_summary_can_be_robust() -> None:
    summary = analysis._signature_scientific_summary(
        best_split={
            "N signature+": 60,
            "Statistically significant": True,
            "Bootstrap support (p<alpha)": 0.9,
            "Bootstrap HR direction consistency": 0.95,
            "Validation support (p<alpha)": 0.8,
            "Permutation p": 0.01,
            "Bootstrap valid resamples": 30,
            "Permutation valid resamples": 200,
            "Validation valid folds": 20,
        },
        search_space={
            "truncated": False,
            "permutation_iterations": 200,
            "validation_iterations": 20,
            "bootstrap_iterations": 30,
            "significance_level": 0.05,
            "min_group_size": 20,
            "tested_combinations": 40,
            "significant_signatures": 2,
            "n_rows_analyzed": 300,
        },
    )
    assert summary["status"] == "robust"
    assert any("heuristic" in caution for caution in summary["cautions"])


# ---------------------------------------------------------------------------------------
# ThresholdLogrankScan (C12)


def test_threshold_logrank_scan_matches_survdiff_with_small_blocks(monkeypatch) -> None:
    from statsmodels.duration.survfunc import survdiff

    rng = np.random.default_rng(0)
    checked = 0
    for block in (None, 3, 1):
        for _ in range(25):
            n = int(rng.integers(10, 60))
            times = rng.integers(0, 15, n).astype(float)
            events = (rng.random(n) < 0.6).astype(float)
            if events.sum() == 0:
                continue
            marker = rng.integers(0, 10, n).astype(float)
            cuts = np.unique(np.concatenate([marker, marker + 0.5]))
            scan = analysis.ThresholdLogrankScan(times, events)
            cells = 2_000_000 if block is None else block * max(len(np.unique(times[events > 0])), 1)
            monkeypatch.setattr(analysis.ThresholdLogrankScan, "_MAX_BLOCK_CELLS", cells)
            result = scan.scan(marker, cuts)
            for index, cut in enumerate(cuts):
                high = marker > cut
                assert result["n_high"][index] == high.sum()
                assert result["events_high"][index] == events[high].sum()
                if high.all() or (~high).all():
                    continue
                try:
                    reference, _ = survdiff(times, events, np.where(high, "a", "b"))
                except np.linalg.LinAlgError:
                    # Zero variance (for example every event before the other group enters):
                    # survdiff cannot invert it, and the scan reports a statistic of 0.
                    reference = 0.0
                assert result["statistic"][index] == pytest.approx(reference, rel=1e-9, abs=1e-9)
                checked += 1
    assert checked > 100

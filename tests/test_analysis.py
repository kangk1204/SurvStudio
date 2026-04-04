from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from survival_toolkit.__main__ import main as cli_main
from survival_toolkit.analysis import (
    MAX_MODEL_FEATURE_CANDIDATES,
    _bh_adjust,
    _cohort_frame,
    _harrell_c_index,
    _harrell_c_index_bootstrap_ci,
    _has_ambiguous_competing_event_tokens,
    _ordered_reference_categories,
    _permutation_p_value,
    _prepare_cox_frame,
    _reference_levels,
    _pointwise_km_ci,
    _stability_score,
    _survival_outcome_like_columns,
    _signature_scientific_summary,
    _safe_float,
    compute_cohort_table,
    compute_cox_analysis,
    compute_km_analysis,
    coerce_event,
    discover_feature_signature,
    derive_group_column,
    ensure_model_feature_candidate_limit,
    find_event_equivalent_columns,
    load_dataframe_from_path,
    make_unique_columns,
    looks_binary,
    preview_cox_analysis_inputs,
    suggest_columns,
)
from survival_toolkit.sample_data import load_tcga_luad_example_dataset
from survival_toolkit.sample_data import load_gbsg2_upload_ready_dataset
from survival_toolkit.sample_data import make_example_dataset


def test_derive_group_column_creates_named_split() -> None:
    df = make_example_dataset(seed=7, n_patients=80)
    updated, column_name, summary = derive_group_column(
        df,
        source_column="biomarker_score",
        method="median_split",
        new_column_name="biomarker_group",
    )
    assert column_name == "biomarker_group"
    assert "biomarker_group" in updated.columns
    assert summary["counts"]
    assert summary["recipe"]["source_column"] == "biomarker_score"
    assert summary["recipe"]["column_name"] == "biomarker_group"
    assert summary["recipe"]["method"] == "median_split"


def test_percentile_split_top_vs_rest_creates_two_groups() -> None:
    df = make_example_dataset(seed=17, n_patients=120)
    updated, column_name, summary = derive_group_column(
        df,
        source_column="age",
        method="percentile_split",
        new_column_name="age_percentile_group",
        cutoff="25",
    )

    observed = set(updated[column_name].dropna().astype(str).unique().tolist())
    assert observed == {"Rest", "At/above 75th percentile threshold"}
    assert summary["cutoff_spec"] == "25"
    assert summary["n_groups"] == 2
    assert len(summary["cutoffs"]) == 1
    assert "Realized non-missing shares:" in summary["assignment_rule"]
    assert summary["realized_group_shares"]


def test_derive_group_rejects_survival_endpoint_source_column() -> None:
    df = make_example_dataset(seed=21, n_patients=100)

    with pytest.raises(ValueError, match="looks like a survival endpoint column"):
        derive_group_column(
            df,
            source_column="os_months",
            method="median_split",
            new_column_name="os_months_split",
        )


def test_percentile_split_50_matches_median_split_with_ties() -> None:
    df = pd.DataFrame({"age": [1, 2, 2, 2, 3, 4]})
    median_updated, median_column, _ = derive_group_column(
        df,
        source_column="age",
        method="median_split",
        new_column_name="age_median_group",
    )
    percentile_updated, percentile_column, _ = derive_group_column(
        df,
        source_column="age",
        method="percentile_split",
        new_column_name="age_percentile_group",
        cutoff="50",
    )

    median_is_high = median_updated[median_column].astype(str) == "High"
    percentile_is_top = percentile_updated[percentile_column].astype(str) == "Above 50th percentile threshold"
    assert median_is_high.tolist() == percentile_is_top.tolist()


def test_bh_adjust_matches_standard_monotone_fdr() -> None:
    from scipy.stats import false_discovery_control

    p_values = [0.04, 0.002, 0.03, 0.01]
    adjusted = _bh_adjust(p_values)
    expected = false_discovery_control(p_values, method="bh").tolist()

    assert adjusted == pytest.approx(expected)


def test_permutation_p_value_compares_logrank_statistics(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    results = iter([(6.0, 0.20), (4.0, 1e-6), (3.0, 1e-9)])

    def _fake_survdiff(times, events, groups):
        return next(results)

    monkeypatch.setattr(analysis, "survdiff", _fake_survdiff)

    empirical_p, valid = analysis._permutation_p_value(
        times=np.asarray([1.0, 2.0, 3.0, 4.0]),
        events=np.asarray([1, 1, 1, 1]),
        mask=np.asarray([True, True, False, False]),
        observed_stat=5.0,
        n_iterations=3,
        random_seed=42,
    )

    assert valid == 3
    assert empirical_p == pytest.approx(0.5)


def test_cox_preview_warns_when_events_per_parameter_is_extremely_low() -> None:
    df = pd.DataFrame(
        {
            "os_months": [1, 2, 3, 4, 5, 6],
            "os_event": [1, 0, 1, 0, 0, 0],
            "age": [50, 51, 52, 53, 54, 55],
            "marker": [10, 11, 12, 13, 14, 15],
            "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "stage": ["I", "I", "II", "II", "III", "III"],
        }
    )

    preview = preview_cox_analysis_inputs(
        df,
        time_column="os_months",
        event_column="os_event",
        covariates=["age", "marker", "sex", "stage"],
        categorical_covariates=["sex", "stage"],
        event_positive_value=1,
    )

    assert preview["estimated_parameters"] == 5
    assert preview["events_per_parameter"] == pytest.approx(0.4)
    assert any("Events per parameter is 0.40" in warning for warning in preview["stability_warnings"])


def test_percentile_split_two_tails_creates_three_groups() -> None:
    df = make_example_dataset(seed=19, n_patients=150)
    updated, column_name, summary = derive_group_column(
        df,
        source_column="age",
        method="percentile_split",
        new_column_name="age_three_band_group",
        cutoff="25,25",
    )

    observed = set(updated[column_name].dropna().astype(str).unique().tolist())
    assert observed == {
        "At/below 25th percentile threshold",
        "Between percentile thresholds",
        "At/above 75th percentile threshold",
    }
    assert summary["cutoff_spec"] == "25,25"
    assert summary["n_groups"] == 3
    assert len(summary["cutoffs"]) == 2


def test_extreme_split_excludes_middle_rows() -> None:
    df = make_example_dataset(seed=23, n_patients=160)
    updated, column_name, summary = derive_group_column(
        df,
        source_column="age",
        method="extreme_split",
        new_column_name="age_extreme_group",
        cutoff="25",
    )

    observed = set(updated[column_name].dropna().astype(str).unique().tolist())
    assert observed == {"At/below 25th percentile threshold", "At/above 75th percentile threshold"}
    assert int(updated[column_name].isna().sum()) > 0
    assert summary["cutoff_spec"] == "25"
    assert summary["excluded_count"] > 0
    assert summary["n_groups"] == 2


def test_make_example_dataset_small_n_is_supported() -> None:
    df = make_example_dataset(seed=3, n_patients=10)
    assert df.shape[0] == 10


def test_make_unique_columns_handles_preexisting_suffixes() -> None:
    assert make_unique_columns(["A", "A_2", "A"]) == ["A", "A_2", "A_3"]


def test_load_dataframe_from_path_rejects_unknown_suffix(tmp_path) -> None:
    path = tmp_path / "cohort.weird"
    path.write_text("age,os_months,os_event\n60,12,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported input file extension"):
        load_dataframe_from_path(path)


def test_model_feature_candidate_limit_accepts_1000_and_rejects_1001() -> None:
    base = {
        "os_months": [12, 18, 24],
        "os_event": [1, 0, 1],
    }
    allowed = pd.DataFrame(base | {f"gene_{idx}": [idx, idx + 1, idx + 2] for idx in range(MAX_MODEL_FEATURE_CANDIDATES)})
    assert ensure_model_feature_candidate_limit(allowed) == MAX_MODEL_FEATURE_CANDIDATES

    too_wide = pd.DataFrame(base | {f"gene_{idx}": [idx, idx + 1, idx + 2] for idx in range(MAX_MODEL_FEATURE_CANDIDATES + 1)})
    with pytest.raises(ValueError, match="supports at most 1000 model features"):
        ensure_model_feature_candidate_limit(too_wide)


def test_rnaseq_top100_upload_example_matches_bundled_tcga_clinical_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    bundled = pd.read_csv(root / "src" / "survival_toolkit" / "data" / "tcga_luad_xena_example.csv")
    upload = pd.read_csv(root / "examples" / "tcga_luad_rnaseq_top100_upload.csv")

    key_columns = ["os_months", "os_event", "age", "sex", "stage_group"]
    gene_columns = [column for column in upload.columns if column not in {
        "patient_id",
        "os_months",
        "os_event",
        "age",
        "sex",
        "stage_group",
        "smoking_status",
        "pack_years_smoked",
        "tumor_longest_dimension_cm",
        "kras_status",
        "egfr_status",
        "expression_subtype",
    }]

    assert upload.shape == (609, 112)
    assert len(gene_columns) == 100
    assert int(upload[gene_columns[0]].isna().sum()) == 4
    assert Counter(map(tuple, upload[key_columns].itertuples(index=False, name=None))) == Counter(
        map(tuple, bundled[key_columns].itertuples(index=False, name=None))
    )


def test_rnaseq_top500_upload_example_matches_bundled_tcga_clinical_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    bundled = pd.read_csv(root / "src" / "survival_toolkit" / "data" / "tcga_luad_xena_example.csv")
    upload = pd.read_csv(root / "examples" / "tcga_luad_rnaseq_top500_upload.csv")

    key_columns = ["os_months", "os_event", "age", "sex", "stage_group"]
    gene_columns = [column for column in upload.columns if column not in {
        "patient_id",
        "os_months",
        "os_event",
        "age",
        "sex",
        "stage_group",
        "smoking_status",
        "pack_years_smoked",
        "tumor_longest_dimension_cm",
        "kras_status",
        "egfr_status",
        "expression_subtype",
    }]

    assert upload.shape == (609, 512)
    assert len(gene_columns) == 500
    assert int(upload[gene_columns[0]].isna().sum()) == 4
    assert Counter(map(tuple, upload[key_columns].itertuples(index=False, name=None))) == Counter(
        map(tuple, bundled[key_columns].itertuples(index=False, name=None))
    )


def test_cli_inspect_reports_profile(tmp_path, capsys) -> None:
    path = tmp_path / "cohort.csv"
    path.write_text("age,os_months,os_event\n60,12,1\n", encoding="utf-8")

    exit_code = cli_main(["inspect", str(path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"filename": "cohort.csv"' in output
    assert '"n_rows": 1' in output


def test_cli_inspect_reports_controlled_error_for_missing_file(capsys) -> None:
    exit_code = cli_main(["inspect", "does_not_exist.csv"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Input file not found" in captured.err


def test_coerce_event_handles_mixed_coding() -> None:
    series = pd.Series(["0", "1", "event", "censored", None])
    coerced = coerce_event(series, event_positive_value=1)

    assert coerced.tolist()[:4] == [0.0, 1.0, 1.0, 0.0]


def test_coerce_event_rejects_unknown_tokens_when_explicit_positive_value_is_standard() -> None:
    series = pd.Series([1, 0, "dead", "alive", "maybe"])
    with pytest.raises(ValueError, match="unrecognized tokens|normalize|infer event coding"):
        coerce_event(series, event_positive_value=1)


def test_coerce_event_rejects_explicit_censor_token_as_event_positive_value() -> None:
    series = pd.Series(["alive", "dead", "alive", "dead"])

    with pytest.raises(ValueError, match="maps to censoring"):
        coerce_event(series, event_positive_value="alive")


def test_coerce_event_rejects_numeric_zero_as_event_positive_value() -> None:
    series = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError, match="maps to censoring"):
        coerce_event(series, event_positive_value=0)


def test_coerce_event_rejects_multistate_numeric_status_columns() -> None:
    series = pd.Series([0, 1, 2, 1, 0])
    with pytest.raises(ValueError, match="more than two distinct states|pre-binarized"):
        coerce_event(series, event_positive_value=1)


def test_coerce_event_rejects_mixed_recognized_multistate_tokens() -> None:
    series = pd.Series(["alive", "death", "relapse", "alive"], dtype="string")

    with pytest.raises(ValueError, match="more than one recognized event state|binary event indicator"):
        coerce_event(series, event_positive_value="death")


def test_coerce_event_rejects_competing_risk_style_compound_death_labels() -> None:
    series = pd.Series(["alive", "cancer_death", "other_death", "alive"], dtype="string")

    with pytest.raises(ValueError, match="more than one recognized event state|binary event indicator"):
        coerce_event(series, event_positive_value="death")


def test_competing_event_token_detection_flags_exact_multifamily_tokens() -> None:
    assert _has_ambiguous_competing_event_tokens(["death", "progression", "censored"]) is True
    assert _has_ambiguous_competing_event_tokens(["death", "deceased", "alive"]) is False


def test_coerce_event_handles_living_deceased_tokens_without_explicit_mapping() -> None:
    series = pd.Series(["LIVING", "DECEASED", "living", "deceased"], dtype="string")

    coerced = coerce_event(series)

    assert coerced.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_coerce_event_handles_tcga_style_composite_status_tokens() -> None:
    series = pd.Series(["0:LIVING", "1:DECEASED", "0:Living", "1:Deceased"], dtype="string")

    coerced = coerce_event(series)

    assert coerced.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_coerce_event_accepts_explicit_deceased_for_tcga_style_composite_status_tokens() -> None:
    series = pd.Series(["0:LIVING", "1:DECEASED", "0:Living", "1:Deceased"], dtype="string")

    coerced = coerce_event(series, event_positive_value="deceased")

    assert coerced.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_coerce_event_accepts_explicit_numeric_one_for_tcga_style_composite_status_tokens() -> None:
    series = pd.Series(["0:LIVING", "1:DECEASED", "0:Living", "1:Deceased"], dtype="string")

    coerced = coerce_event(series, event_positive_value=1)

    assert coerced.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_survival_outcome_like_columns_do_not_flag_generic_duration_covariates() -> None:
    df = pd.DataFrame(
        {
            "os_months": [12, 18, 24, 30],
            "os_event": [1, 0, 1, 0],
            "treatment_duration_months": [6, 8, 10, 12],
        }
    )

    detected = _survival_outcome_like_columns(df)

    assert "os_months" in detected
    assert "os_event" in detected
    assert "treatment_duration_months" not in detected


def test_find_event_equivalent_columns_detects_binary_duplicate_of_selected_event() -> None:
    df = pd.DataFrame(
        {
            "custom_event": ["0:LIVING", "1:DECEASED", "0:LIVING", "1:DECEASED"],
            "delta": [0, 1, 0, 1],
            "sex_binary": [0, 1, 1, 0],
        }
    )

    equivalents = find_event_equivalent_columns(df, "custom_event", event_positive_value="deceased")

    assert "delta" in equivalents
    assert "sex_binary" not in equivalents


def test_cohort_frame_rejects_identical_time_and_event_columns() -> None:
    df = pd.DataFrame({"status": [1, 0, 1], "age": [60, 55, 70]})

    with pytest.raises(ValueError, match="must be different"):
        _cohort_frame(
            df,
            time_column="status",
            event_column="status",
            event_positive_value=1,
            extra_columns=["age"],
        )


def test_cohort_frame_rejects_non_time_numeric_column_when_likely_time_exists() -> None:
    df = pd.DataFrame(
        {
            "os_months": [12, 18, 24],
            "os_event": [1, 0, 1],
            "SFTPC": [6.1, 8.4, 5.9],
            "age": [60, 55, 70],
        }
    )

    with pytest.raises(ValueError, match="does not look like a survival follow-up time column"):
        _cohort_frame(
            df,
            time_column="SFTPC",
            event_column="os_event",
            event_positive_value=1,
            extra_columns=["age"],
        )


def test_ordered_level_strings_deduplicates_string_levels() -> None:
    from survival_toolkit.analysis import _ordered_level_strings

    series = pd.Series(["no", "yes", "no", None], dtype="string")

    assert _ordered_level_strings(series, "horTh") == ["no", "yes"]


def test_looks_binary_accepts_nonstandard_two_value_numeric_status() -> None:
    series = pd.Series([1, 2, 1, 2, None])

    assert looks_binary(series) is True


def test_looks_binary_rejects_mixed_recognized_event_families() -> None:
    series = pd.Series(["alive", "death", "relapse"], dtype="string")

    assert looks_binary(series) is False


def test_suggest_columns_does_not_treat_menostat_as_time_or_event() -> None:
    df = pd.DataFrame(
        {
            "rfs_days": [100, 200, 300],
            "rfs_event": [1, 0, 1],
            "menostat": ["Pre", "Post", "Post"],
            "age": [45, 52, 61],
        }
    )

    suggestions = suggest_columns(df)

    assert "rfs_days" in suggestions["time_columns"]
    assert "rfs_event" in suggestions["event_columns"]
    assert "menostat" not in suggestions["time_columns"]
    assert "menostat" not in suggestions["event_columns"]


def test_suggest_columns_detects_concatenated_survival_naming_patterns() -> None:
    df = pd.DataFrame(
        {
            "OverallSurvivalMonths": [12, 24, 36],
            "deathstatus": [1, 0, 1],
            "riskgroup": ["A", "B", "A"],
            "treatmentarm": ["x", "y", "x"],
        }
    )

    suggestions = suggest_columns(df)

    assert "OverallSurvivalMonths" in suggestions["time_columns"]
    assert "deathstatus" in suggestions["event_columns"]
    assert "riskgroup" in suggestions["group_columns"]
    assert "treatmentarm" in suggestions["group_columns"]


def test_suggest_columns_detects_common_survival_time_aliases() -> None:
    df = pd.DataFrame(
        {
            "fu_time": [12, 24, 36],
            "surv_time": [10, 20, 30],
            "time_in_months": [8, 18, 28],
            "mfs_days": [120, 180, 365],
            "dmfs_months": [6, 12, 18],
            "os_event": [1, 0, 1],
        }
    )

    suggestions = suggest_columns(df)

    assert "fu_time" in suggestions["time_columns"]
    assert "surv_time" in suggestions["time_columns"]
    assert "time_in_months" in suggestions["time_columns"]
    assert "mfs_days" in suggestions["time_columns"]
    assert "dmfs_months" in suggestions["time_columns"]


def test_suggest_columns_and_outcome_guard_do_not_treat_age_years_as_survival_endpoint() -> None:
    df = pd.DataFrame(
        {
            "os_months": [12, 24, 36],
            "os_event": [1, 0, 1],
            "age_years": [51, 62, 73],
            "smoking_years": [0, 20, 35],
        }
    )

    suggestions = suggest_columns(df)
    outcome_like = _survival_outcome_like_columns(df)

    assert "os_months" in suggestions["time_columns"]
    assert "age_years" not in suggestions["time_columns"]
    assert "smoking_years" not in suggestions["time_columns"]
    assert "age_years" not in outcome_like
    assert "smoking_years" not in outcome_like


def test_harrell_c_index_matches_naive_pairwise_result() -> None:
    time_values = np.array([5.0, 8.0, 10.0, 12.0, 14.0, 20.0], dtype=float)
    event_values = np.array([1, 0, 1, 1, 0, 1], dtype=int)
    risk_score = np.array([0.9, 0.2, 0.7, 0.6, 0.3, 0.1], dtype=float)

    naive_concordant = 0.0
    naive_comparable = 0.0
    for i, (time_i, event_i, risk_i) in enumerate(zip(time_values, event_values, risk_score, strict=False)):
        if event_i != 1:
            continue
        for j, (time_j, risk_j) in enumerate(zip(time_values, risk_score, strict=False)):
            if i == j or time_j <= time_i:
                continue
            naive_comparable += 1.0
            if risk_i > risk_j:
                naive_concordant += 1.0
            elif risk_i == risk_j:
                naive_concordant += 0.5

    assert _harrell_c_index(time_values, event_values, risk_score) == pytest.approx(
        naive_concordant / naive_comparable
    )


def test_cohort_frame_rejects_non_event_like_binary_column_when_likely_event_exists() -> None:
    df = pd.DataFrame(
        {
            "rfs_days": [120, 180, 240, 300],
            "rfs_event": [1, 0, 1, 0],
            "menostat": ["Post", "Pre", "Post", "Pre"],
        }
    )

    with pytest.raises(ValueError, match="does not look like a survival event column"):
        _cohort_frame(
            df,
            time_column="rfs_days",
            event_column="menostat",
            event_positive_value="Post",
        )


def test_cohort_frame_allows_nonstandard_binary_event_column_when_coding_is_explicit() -> None:
    df = pd.DataFrame(
        {
            "fu_time": [12, 18, 24, 30],
            "delta": [1, 0, 1, 1],
            "age": [53, 61, 49, 72],
        }
    )

    frame = _cohort_frame(
        df,
        time_column="fu_time",
        event_column="delta",
        event_positive_value=1,
        extra_columns=["age"],
    )

    assert frame.shape[0] == 4
    assert frame["delta"].astype(int).tolist() == [1, 0, 1, 1]


def test_survival_outcome_like_columns_flag_multistate_status_surrogates() -> None:
    df = pd.DataFrame(
        {
            "os_months": [8, 12, 16, 20],
            "os_event": [1, 0, 1, 0],
            "vital_status": ["alive", "dead", "unknown", "dead"],
            "status": ["NED", "AWD", "DOD", "AWD"],
            "egfr_status": ["mut", "wt", "mut", "wt"],
        }
    )

    outcome_like = _survival_outcome_like_columns(df)

    assert "vital_status" in outcome_like
    assert "status" in outcome_like
    assert "egfr_status" not in outcome_like


def test_survival_outcome_like_columns_flag_css_and_dss_endpoints() -> None:
    df = pd.DataFrame(
        {
            "css_time": [10, 12, 14, 16],
            "css_status": [1, 0, 1, 0],
            "dss_months": [9, 11, 13, 15],
            "dss_status": ["alive", "dead", "alive", "dead"],
            "age": [51, 63, 58, 70],
        }
    )

    outcome_like = _survival_outcome_like_columns(df)

    assert "css_time" in outcome_like
    assert "css_status" in outcome_like
    assert "dss_months" in outcome_like
    assert "dss_status" in outcome_like
    assert "age" not in outcome_like


def test_cohort_frame_rejects_mismatched_endpoint_family_pair() -> None:
    df = pd.DataFrame(
        {
            "os_months": [10, 20, 30, 40],
            "pfs_event": [1, 0, 1, 0],
            "age": [50, 60, 70, 80],
        }
    )

    with pytest.raises(ValueError, match="different survival endpoints"):
        _cohort_frame(
            df,
            time_column="os_months",
            event_column="pfs_event",
            event_positive_value=1,
            extra_columns=["age"],
        )


def test_km_analysis_returns_grouped_results() -> None:
    df = make_example_dataset(seed=11, n_patients=180)
    updated, column_name, _ = derive_group_column(
        df,
        source_column="biomarker_score",
        method="median_split",
        new_column_name="biomarker_group",
    )
    result = compute_km_analysis(
        updated,
        time_column="os_months",
        event_column="os_event",
        group_column=column_name,
        event_positive_value=1,
    )
    assert len(result["curves"]) == 2
    assert result["test"] is not None
    assert result["test"]["p_value"] < 0.05
    assert result["scientific_summary"]["headline"]
    assert result["scientific_summary"]["status"] in {"robust", "review", "caution"}
    assert result["scientific_summary"]["metrics"]
    assert any("independent" in caution.lower() and "censoring" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("competing risks" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("left truncation" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_km_analysis_marks_outcome_informed_group_results_as_descriptive() -> None:
    df = make_example_dataset(seed=18, n_patients=180)
    updated, column_name, _ = derive_group_column(
        df,
        source_column="biomarker_score",
        method="median_split",
        new_column_name="biomarker_group",
    )

    result = compute_km_analysis(
        updated,
        time_column="os_months",
        event_column="os_event",
        group_column=column_name,
        event_positive_value=1,
        suppress_group_inference=True,
        outcome_informed_group=True,
    )

    assert result["test"] is None
    assert result["pairwise_table"] == []
    assert result["logrank_p"] is None
    assert result["outcome_informed_group"] is True
    assert "descriptive" in result["scientific_summary"]["headline"].lower()
    assert any("exploratory rather than confirmatory" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_km_analysis_extends_curve_to_followup_horizon_when_last_observation_is_censored() -> None:
    df = pd.DataFrame(
        {
            "time": [5.0, 8.0, 12.0],
            "event": [1, 0, 0],
        }
    )

    result = compute_km_analysis(
        df,
        time_column="time",
        event_column="event",
        event_positive_value=1,
    )

    curve = result["curves"][0]
    assert curve["timeline"][-1] == pytest.approx(12.0)
    assert curve["survival"][-1] == pytest.approx(curve["survival"][-2])
    assert curve["ci_lower"][-1] == pytest.approx(curve["ci_lower"][-2])
    assert curve["ci_upper"][-1] == pytest.approx(curve["ci_upper"][-2])
    assert curve["censor_times"] == [8.0, 12.0]


def test_derive_group_column_tracks_optimal_cutpoint_reproducibility_settings() -> None:
    df = make_example_dataset(seed=16, n_patients=200)
    updated, column_name, summary = derive_group_column(
        df,
        source_column="biomarker_score",
        method="optimal_cutpoint",
        new_column_name="optimal_group",
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        min_group_fraction=0.15,
        permutation_iterations=20,
        random_seed=777,
    )

    assert column_name == "optimal_group"
    assert "optimal_group" in updated.columns
    assert summary["min_group_fraction"] == pytest.approx(0.15)
    assert summary["permutation_iterations"] == 20
    assert summary["random_seed"] == 777


def test_weighted_km_does_not_report_weighted_test_as_logrank_p() -> None:
    df = make_example_dataset(seed=12, n_patients=180)
    result = compute_km_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        group_column="stage",
        event_positive_value=1,
        logrank_weight="gehan_breslow",
    )

    assert result["test"] is not None
    assert result["test"]["test"] == "gehan_breslow"
    assert result["logrank_p"] is None
    assert result["test_p_value"] == pytest.approx(result["test"]["p_value"])
    assert result["test_p_value_label"] == "Gehan-Breslow"


def test_fleming_harrington_label_includes_fh_parameter() -> None:
    df = make_example_dataset(seed=13, n_patients=180)
    result = compute_km_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        group_column="stage",
        event_positive_value=1,
        logrank_weight="fleming_harrington",
        fh_p=1.5,
    )

    assert result["test_p_value_label"] == "Fleming-Harrington (fh_p=1.5)"
    assert "fh_p=1.5" in result["scientific_summary"]["headline"]


def test_pointwise_km_ci_handles_survival_near_one_without_blowing_up() -> None:
    lower, upper = _pointwise_km_ci(
        np.asarray([1.0 - 1e-16], dtype=float),
        np.asarray([0.2], dtype=float),
        alpha=0.05,
    )

    assert lower[0] == pytest.approx(1.0 - 1e-16)
    assert upper[0] == pytest.approx(1.0 - 1e-16)


def test_pointwise_km_ci_keeps_exact_one_bounded_at_one() -> None:
    lower, upper = _pointwise_km_ci(
        np.asarray([1.0], dtype=float),
        np.asarray([0.2], dtype=float),
        alpha=0.05,
    )

    assert lower[0] == pytest.approx(1.0)
    assert upper[0] == pytest.approx(1.0)


def test_km_analysis_log_log_ci_matches_r_survfit_reference() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "event": [1, 1, 0, 1, 1],
        }
    )

    result = compute_km_analysis(
        df,
        time_column="time",
        event_column="event",
        event_positive_value=1,
    )

    curve = result["curves"][0]
    timeline = curve["timeline"]
    ci_lookup = {
        float(time): (float(lower), float(upper))
        for time, lower, upper in zip(timeline, curve["ci_lower"], curve["ci_upper"], strict=True)
    }

    assert ci_lookup[1.0] == pytest.approx((0.20380926, 0.96917979), abs=1e-6)
    assert ci_lookup[2.0] == pytest.approx((0.12573018, 0.88175641), abs=1e-6)
    assert ci_lookup[4.0] == pytest.approx((0.01230153, 0.71921802), abs=1e-6)


def test_safe_float_returns_none_on_overflowing_float_conversion() -> None:
    class _OverflowingValue:
        def __float__(self) -> float:
            raise OverflowError("too large")

    assert _safe_float(_OverflowingValue()) is None


def test_cox_analysis_recovers_expected_directions() -> None:
    df = make_example_dataset(seed=21, n_patients=260)
    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age", "stage", "treatment", "biomarker_score"],
        categorical_covariates=["stage", "treatment"],
    )
    rows = {row["Label"]: row for row in result["results_table"]}
    assert rows["age"]["Hazard ratio"] > 1.0
    assert rows["biomarker_score"]["Hazard ratio"] > 1.0
    treatment_row = next(row for row in result["results_table"] if row["Variable"] == "treatment")
    assert treatment_row["Reference"] == "Combination"
    assert treatment_row["Label"] == "treatment: Standard vs Combination"
    assert treatment_row["Hazard ratio"] > 1.0
    assert result["model_stats"]["evaluation_mode"] == "apparent"
    assert result["model_stats"]["c_index_label"] == "Apparent C-index (training cohort)"
    assert result["model_stats"]["apparent_c_index"] == result["model_stats"]["c_index"]
    assert result["model_stats"]["c_index_ci_method"] == "bootstrap_percentile"
    assert result["model_stats"]["c_index_ci_level"] == pytest.approx(0.95)
    assert result["model_stats"]["c_index_ci_lower"] is not None
    assert result["model_stats"]["c_index_ci_upper"] is not None
    assert result["model_stats"]["c_index_ci_lower"] <= result["model_stats"]["c_index"] <= result["model_stats"]["c_index_ci_upper"]
    assert result["model_stats"]["lr_statistic"] is not None
    assert result["model_stats"]["lr_pvalue"] is not None
    assert result["scientific_summary"]["headline"]
    assert result["scientific_summary"]["status"] in {"robust", "review", "caution"}
    assert result["scientific_summary"]["metrics"]
    assert any(metric["label"] == "Apparent C-index (training cohort)" for metric in result["scientific_summary"]["metrics"])
    assert any(metric["label"] == "Apparent C-index 95% CI" for metric in result["scientific_summary"]["metrics"])
    assert any(metric["label"] == "LR chi-square" for metric in result["scientific_summary"]["metrics"])
    assert any("apparent" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("independent" in caution.lower() and "censoring" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("competing risks" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("left truncation" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("analyzable cohort" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("spearman" in strength.lower() and "schoenfeld" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("likelihood-ratio test" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("comparable patient pairs" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("external-cohort apply workflow" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("changing the covariate set" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert result["diagnostics_plot_data"]
    first_trace = result["diagnostics_plot_data"][0]
    assert {"term", "log_time", "residual", "trend_log_time", "trend_residual"} <= set(first_trace)
    assert len(first_trace["log_time"]) == len(first_trace["residual"])
    assert len(first_trace["trend_log_time"]) == len(first_trace["trend_residual"])


def test_cox_analysis_scales_schoenfeld_diagnostics_and_reports_ci(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=31, n_patients=4)
    frame = pd.DataFrame(
        {
            "os_months": [1.0, 2.0, 3.0, 4.0],
            "os_event": [1, 1, 0, 0],
            "age": [50.0, 55.0, 60.0, 65.0],
        }
    )

    class _FakeResults:
        params = np.asarray([0.2], dtype=float)
        bse = np.asarray([0.1], dtype=float)
        tvalues = np.asarray([2.0], dtype=float)
        pvalues = np.asarray([0.04], dtype=float)
        schoenfeld_residuals = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float)
        llf = -4.0
        llnull = -7.0
        model = type("_FakeModelMeta", (), {"exog_names": ["Q(\"age\")"], "exog": np.ones((4, 1), dtype=float)})()

        def conf_int(self):
            return np.asarray([[0.1, 0.3]], dtype=float)

        def cov_params(self):
            return np.asarray([[2.0]], dtype=float)

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            class _FakeFit:
                @staticmethod
                def fit(disp=False):
                    return _FakeResults()

            return _FakeFit()

    monkeypatch.setattr(analysis, "_prepare_cox_frame", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(analysis, "PHReg", _FakePHReg)
    monkeypatch.setattr(analysis, "_reference_levels", lambda *args, **kwargs: {})
    monkeypatch.setattr(analysis, "_harrell_c_index", lambda *args, **kwargs: 0.62)
    monkeypatch.setattr(
        analysis,
        "_harrell_c_index_bootstrap_ci",
        lambda *args, **kwargs: {"c_index_std": 0.03, "c_index_ci_lower": 0.57, "c_index_ci_upper": 0.67},
    )

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age"],
    )

    first_trace = result["diagnostics_plot_data"][0]
    assert first_trace["residual"] == pytest.approx([4.0, 8.0, 12.0, 16.0])
    assert len(first_trace["trend_log_time"]) == len(first_trace["trend_residual"])
    assert result["model_stats"]["c_index_ci_lower"] == pytest.approx(0.57)
    assert result["model_stats"]["c_index_ci_upper"] == pytest.approx(0.67)
    assert result["model_stats"]["lr_statistic"] == pytest.approx(6.0)
    assert result["model_stats"]["lr_pvalue"] is not None


def test_cox_analysis_reports_missing_covariate_exclusions() -> None:
    df = make_example_dataset(seed=211, n_patients=180)
    df.loc[df.index[:24], "biomarker_score"] = np.nan

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age", "biomarker_score"],
        categorical_covariates=[],
    )

    assert result["model_stats"]["dropped_rows"] == 24
    assert result["model_stats"]["outcome_rows"] == result["model_stats"]["n"] + result["model_stats"]["dropped_rows"]
    assert any(metric["label"] == "Dropped for missing covariates" and metric["value"] == 24 for metric in result["scientific_summary"]["metrics"])
    assert any("were excluded" in caution.lower() and "missing" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_cox_analysis_gbsg2_pnodes_hr_matches_r_coxph_reference() -> None:
    df = load_gbsg2_upload_ready_dataset()

    result = compute_cox_analysis(
        df,
        time_column="rfs_days",
        event_column="rfs_event",
        event_positive_value=1,
        covariates=["pnodes"],
    )

    pnodes_row = result["results_table"][0]
    assert pnodes_row["Label"] == "pnodes"
    assert pnodes_row["Hazard ratio"] == pytest.approx(1.060354, rel=1e-6)
    assert pnodes_row["Beta"] == pytest.approx(0.05860287, rel=1e-6)
    assert pnodes_row["P value"] == pytest.approx(3.488721e-18, rel=1e-6)


def test_compute_cohort_table_discloses_grouped_subset_overall_scope() -> None:
    df = make_example_dataset(seed=27, n_patients=120)
    result = compute_cohort_table(df, variables=["age", "sex"], group_column="stage")

    assert result["columns"][2] == "Overall (grouped subset)"
    assert "grouped subset" in result["overall_scope"].lower()


def test_cox_analysis_keeps_low_cardinality_numeric_covariates_continuous_by_default() -> None:
    df = make_example_dataset(seed=23, n_patients=180)
    df["dose_level"] = (df["age"] // 10).astype(int)

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["dose_level"],
    )

    assert len(result["results_table"]) == 1
    assert result["results_table"][0]["Label"] == "dose_level"
    assert result["results_table"][0]["Reference"] is None


def test_cox_analysis_rejects_overlapping_stage_representations() -> None:
    df = pd.DataFrame(
        {
            "os_months": [10, 12, 14, 16, 18, 20],
            "os_event": [1, 0, 1, 0, 1, 1],
            "pathologic_stage": ["Stage I", "Stage II", "Stage III", "Stage I", "Stage II", "Stage III"],
            "stage_group": ["Stage I", "Stage II", "Stage III", "Stage I", "Stage II", "Stage III"],
        }
    )

    with pytest.raises(ValueError, match="overlapping stage representations"):
        compute_cox_analysis(
            df,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            covariates=["pathologic_stage", "stage_group"],
            categorical_covariates=["pathologic_stage", "stage_group"],
        )


def test_cox_reference_ordering_prefers_clinical_baselines_for_common_categories() -> None:
    df = pd.DataFrame(
        {
            "os_months": np.linspace(6, 65, 12),
            "os_event": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "smoking_status": [
                "Current smoker",
                "Former smoker <=15y",
                "Lifelong Non-smoker",
                "Current smoker",
                "Former smoker <=15y",
                "Lifelong Non-smoker",
                "Current smoker",
                "Former smoker <=15y",
                "Lifelong Non-smoker",
                "Current smoker",
                "Former smoker <=15y",
                "Lifelong Non-smoker",
            ],
            "kras_status": [
                "Mutated",
                "Wildtype",
                "Mutated",
                "Wildtype",
                "Mutated",
                "Wildtype",
                "Mutated",
                "Wildtype",
                "Mutated",
                "Wildtype",
                "Mutated",
                "Wildtype",
            ],
            "stage_group": [
                "Stage IV",
                "Stage II",
                "Stage I",
                "Stage III",
                "Stage II",
                "Stage I",
                "Stage IV",
                "Stage III",
                "Stage I",
                "Stage II",
                "Stage IV",
                "Stage III",
            ],
        }
    )

    frame = _prepare_cox_frame(
        df,
        time_column="os_months",
        event_column="os_event",
        covariates=["smoking_status", "kras_status", "stage_group"],
        categorical_covariates=["smoking_status", "kras_status", "stage_group"],
        event_positive_value=1,
    )
    refs = _reference_levels(frame, ["smoking_status", "kras_status", "stage_group"])

    assert list(frame["smoking_status"].cat.categories) == [
        "Lifelong Non-smoker",
        "Former smoker <=15y",
        "Current smoker",
    ]
    assert list(frame["kras_status"].cat.categories) == ["Wildtype", "Mutated"]
    assert list(frame["stage_group"].cat.categories) == ["Stage I", "Stage II", "Stage III", "Stage IV"]
    assert refs == {
        "smoking_status": "Lifelong Non-smoker",
        "kras_status": "Wildtype",
        "stage_group": "Stage I",
    }


def test_ordered_reference_categories_keeps_unknown_levels_last() -> None:
    ordered = _ordered_reference_categories(
        ["unknown", "Stage III", "Stage I", "Stage II"],
        "stage_group",
    )

    assert ordered == ["Stage I", "Stage II", "Stage III", "unknown"]


def test_km_analysis_orders_common_group_labels_clinically() -> None:
    stage_group = (["Stage IV"] * 10) + (["Stage II"] * 10) + (["Stage I"] * 10) + (["Stage III"] * 10) + (["unknown"] * 5)
    df = pd.DataFrame(
        {
            "os_months": np.linspace(6, 120, len(stage_group)),
            "os_event": [1 if idx % 3 != 0 else 0 for idx in range(len(stage_group))],
            "stage_group": stage_group,
        }
    )

    result = compute_km_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        group_column="stage_group",
        event_positive_value=1,
    )

    assert [row["Group"] for row in result["summary_table"]] == ["Stage I", "Stage II", "Stage III", "Stage IV", "unknown"]


def test_cohort_table_orders_group_columns_clinically() -> None:
    df = pd.DataFrame(
        {
            "age": np.linspace(45, 75, 8),
            "sex": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
            "stage_group": [
                "Stage III",
                "Stage I",
                "unknown",
                "Stage II",
                "Stage IV",
                "Stage I",
                "Stage II",
                "Stage III",
            ],
        }
    )

    table = compute_cohort_table(df, variables=["age", "sex"], group_column="stage_group")

    assert table["columns"] == ["Variable", "Statistic", "Overall (grouped subset)", "Stage I", "Stage II", "Stage III", "Stage IV", "unknown"]


def test_cohort_table_overall_matches_grouped_subset_when_group_values_missing() -> None:
    df = pd.DataFrame(
        {
            "age": [50, 55, 60, 65],
            "sex": ["Female", "Male", "Female", "Male"],
            "group_flag": ["A", "B", None, "A"],
        }
    )

    table = compute_cohort_table(df, variables=["age", "sex"], group_column="group_flag")
    cohort_size_row = next(row for row in table["rows"] if row["Variable"] == "Cohort size")

    assert cohort_size_row["Overall (grouped subset)"] == 3
    assert cohort_size_row["A"] == 2
    assert cohort_size_row["B"] == 1


def test_cox_analysis_marks_extreme_hazard_ratios_as_non_estimable(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=23, n_patients=40)
    frame = pd.DataFrame(
        {
            "os_months": np.linspace(1.0, 40.0, 40),
            "os_event": [1] * 20 + [0] * 20,
            "age": np.linspace(50.0, 80.0, 40),
        }
    )

    class _FakeResults:
        params = np.asarray([10_000.0], dtype=float)
        bse = np.asarray([1.0], dtype=float)
        tvalues = np.asarray([2.0], dtype=float)
        pvalues = np.asarray([0.01], dtype=float)
        schoenfeld_residuals = np.ones((40, 1), dtype=float)
        llf = -10.0
        model = type("_FakeModelMeta", (), {"exog_names": ["Q(\"age\")"], "exog": np.ones((40, 1), dtype=float)})()

        def conf_int(self):
            return np.asarray([[9_000.0, 11_000.0]], dtype=float)

        def cov_params(self):
            return np.asarray([[1.0]], dtype=float)

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            class _FakeFit:
                @staticmethod
                def fit(disp=False):
                    return _FakeResults()

            return _FakeFit()

    monkeypatch.setattr(analysis, "_prepare_cox_frame", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(analysis, "PHReg", _FakePHReg)
    monkeypatch.setattr(analysis, "_reference_levels", lambda *args, **kwargs: {})
    monkeypatch.setattr(analysis, "_harrell_c_index", lambda *args, **kwargs: 0.61)

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age"],
    )

    row = result["results_table"][0]
    assert row["Hazard ratio"] is None
    assert row["CI lower"] is None
    assert row["CI upper"] is None
    assert any(
        "non-estimable hazard ratios or confidence intervals" in caution
        for caution in result["scientific_summary"]["cautions"]
    )


def test_cox_analysis_rejects_non_finite_model_fit(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=23, n_patients=40)
    frame = pd.DataFrame(
        {
            "os_months": [1, 2, 3, 4, 5, 6],
            "os_event": [1, 1, 0, 0, 0, 0],
            "age": [50.0, 54.0, 58.0, 62.0, 66.0, 70.0],
            "pathologic_stage": pd.Categorical(
                ["Stage I", "Stage I", "Stage II", "Stage II", "Stage III", "Stage III"],
                categories=["Stage I", "Stage II", "Stage III"],
                ordered=True,
            ),
            "histology": pd.Categorical(
                ["RareType", "CommonType", "CommonType", "CommonType", "CommonType", "CommonType"],
                categories=["RareType", "CommonType"],
                ordered=True,
            ),
        }
    )

    class _FakeResults:
        params = np.asarray([np.nan], dtype=float)
        bse = np.asarray([np.nan], dtype=float)
        tvalues = np.asarray([np.nan], dtype=float)
        pvalues = np.asarray([np.nan], dtype=float)
        schoenfeld_residuals = np.ones((6, 1), dtype=float)
        llf = np.nan
        model = type("_FakeModelMeta", (), {"exog_names": ["Q(\"age\")"], "exog": np.ones((6, 1), dtype=float)})()

        def conf_int(self):
            return np.asarray([[np.nan, np.nan]], dtype=float)

        def cov_params(self):
            return np.asarray([[1.0]], dtype=float)

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            class _FakeFit:
                @staticmethod
                def fit(disp=False):
                    return _FakeResults()

            return _FakeFit()

    monkeypatch.setattr(analysis, "_prepare_cox_frame", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(analysis, "PHReg", _FakePHReg)

    with pytest.raises(ValueError, match="non-finite estimates") as exc_info:
        compute_cox_analysis(
            df,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            covariates=["age", "pathologic_stage", "histology"],
            categorical_covariates=["pathologic_stage", "histology"],
        )

    message = str(exc_info.value)
    assert "EPV=0.50" in message
    assert 'pathologic_stage="Stage I" (n=2)' in message
    assert 'histology="RareType" (n=1)' in message


def test_cox_analysis_translates_singular_matrix_into_user_facing_message(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = pd.DataFrame(
        {
            "rfs_days": [100, 120, 140, 160, 180, 200],
            "rfs_event": [1, 0, 1, 0, 1, 0],
            "estrec": [10.0, 12.0, 11.0, 9.0, 13.0, 8.0],
            "estrec_median_split": pd.Categorical(
                ["High", "High", "High", "Low", "High", "Low"],
                categories=["Low", "High"],
                ordered=True,
            ),
        }
    )

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            class _FakeFit:
                @staticmethod
                def fit(disp=False):
                    raise np.linalg.LinAlgError("Singular matrix")

            return _FakeFit()

    monkeypatch.setattr(analysis, "PHReg", _FakePHReg)

    with pytest.raises(ValueError, match="design matrix is singular") as exc_info:
        compute_cox_analysis(
            df,
            time_column="rfs_days",
            event_column="rfs_event",
            event_positive_value=1,
            covariates=["estrec", "estrec_median_split"],
            categorical_covariates=["estrec_median_split"],
        )

    message = str(exc_info.value)
    assert "overlapping encodings of the same signal" in message
    assert "Remove one of the overlapping variables" in message


def test_cox_analysis_lists_all_current_ph_alert_terms_in_summary_caution() -> None:
    df = load_tcga_luad_example_dataset()
    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age", "sex", "stage_group", "kras_status", "egfr_status"],
        categorical_covariates=["sex", "stage_group", "kras_status", "egfr_status"],
    )

    significant_terms = [
        str(row["Term"])
        for row in result["diagnostics_table"]
        if row["P value"] is not None and float(row["P value"]) < 0.05
    ]
    cautions = result["scientific_summary"]["cautions"]
    ph_caution = next(
        caution
        for caution in cautions
        if "Possible proportional-hazards violations detected for:" in caution
    )

    assert significant_terms
    for term in significant_terms:
        assert term in ph_caution


def test_cox_analysis_warns_when_reference_levels_are_too_small() -> None:
    df = load_tcga_luad_example_dataset()
    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=[
            "age",
            "sex",
            "pathologic_stage",
            "kras_status",
            "egfr_status",
            "expression_subtype",
            "tumor_longest_dimension_cm",
        ],
        categorical_covariates=["sex", "pathologic_stage", "kras_status", "egfr_status", "expression_subtype"],
    )

    cautions = result["scientific_summary"]["cautions"]
    assert any('pathologic_stage reference "Stage I" (n=2)' in caution for caution in cautions)
    assert "appear unstable" in result["scientific_summary"]["headline"]


def test_cox_analysis_uses_ph_review_headline_when_fit_is_not_structurally_unstable() -> None:
    df = load_gbsg2_upload_ready_dataset()
    result = compute_cox_analysis(
        df,
        time_column="rfs_days",
        event_column="rfs_event",
        event_positive_value=1,
        covariates=["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
        categorical_covariates=["horTh", "menostat", "tgrade"],
    )

    headline = result["scientific_summary"]["headline"]
    assert "closer proportional-hazards review" in headline
    assert "appear unstable" not in headline


def test_cohort_table_includes_overall_column() -> None:
    df = make_example_dataset(seed=5, n_patients=120)
    table = compute_cohort_table(df, variables=["age", "sex", "stage"], group_column="treatment")
    assert "Overall (grouped subset)" in table["columns"]
    assert any(row["Variable"] == "age" for row in table["rows"])


def test_cohort_table_treats_binary_numeric_variables_as_counts() -> None:
    df = make_example_dataset(seed=29, n_patients=100)
    df["binary_flag"] = (df["age"] >= df["age"].median()).astype(int)

    table = compute_cohort_table(df, variables=["binary_flag"], group_column=None)
    stats = {(row["Variable"], row["Statistic"]): row["Overall"] for row in table["rows"]}

    assert ("binary_flag", "Mean ± SD | Median [IQR]") not in stats
    assert ("binary_flag", "0") in stats
    assert ("binary_flag", "1") in stats
    assert "%" in str(stats[("binary_flag", "1")])


def test_cohort_table_deduplicates_string_levels() -> None:
    df = pd.DataFrame(
        {
            "age": [50, 55, 60, 65],
            "horTh": pd.Series(["no", "yes", "no", None], dtype="string"),
            "group_flag": ["A", "A", "B", "B"],
        }
    )

    table = compute_cohort_table(df, variables=["horTh"], group_column="group_flag")
    hormone_rows = [row for row in table["rows"] if row["Variable"] == "horTh"]

    assert [row["Statistic"] for row in hormone_rows] == ["no", "yes", "Missing"]


def test_tertile_split_handles_tied_quantile_edges() -> None:
    df = make_example_dataset(seed=31, n_patients=90)
    pattern = [0, 0, 1, 1, 2]
    df["biomarker_score"] = [pattern[idx % len(pattern)] for idx in range(len(df))]

    updated, column_name, summary = derive_group_column(
        df,
        source_column="biomarker_score",
        method="tertile_split",
        new_column_name="biomarker_tertile",
    )

    values = set(updated[column_name].dropna().astype(str).unique().tolist())
    assert values.issubset({"T1", "T2", "T3"})
    assert summary["method"] == "tertile_split"
    assert 2 <= summary["n_groups"] <= 3


def test_derive_group_column_generates_unique_default_name_on_repeat() -> None:
    df = make_example_dataset(seed=31, n_patients=90)

    updated_first, first_name, _ = derive_group_column(
        df,
        source_column="biomarker_score",
        method="median_split",
    )
    updated_second, second_name, _ = derive_group_column(
        updated_first,
        source_column="biomarker_score",
        method="median_split",
    )

    assert first_name == "biomarker_score__median_split"
    assert second_name == "biomarker_score__median_split_2"
    assert second_name in updated_second.columns


def test_quartile_split_rejects_constant_series() -> None:
    df = make_example_dataset(seed=37, n_patients=60)
    df["biomarker_score"] = 3.14

    with pytest.raises(ValueError, match="enough unique values"):
        derive_group_column(
            df,
            source_column="biomarker_score",
            method="quartile_split",
            new_column_name="biomarker_quartile",
        )


def test_discover_feature_signature_ranks_and_persists_best_group() -> None:
    df = make_example_dataset(seed=41, n_patients=320)
    updated, column_name, payload = discover_feature_signature(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        candidate_columns=["age", "stage", "treatment", "biomarker_score", "immune_index"],
        max_combination_size=3,
        top_k=12,
        min_group_fraction=0.1,
        bootstrap_iterations=10,
        bootstrap_sample_fraction=0.8,
        permutation_iterations=20,
        validation_iterations=6,
        validation_fraction=0.35,
        significance_level=0.05,
        combination_operator="and",
        random_seed=1234,
        new_column_name="signature_group",
    )

    assert column_name == "signature_group"
    assert "signature_group" in updated.columns
    assert payload["results_table"]
    assert payload["best_split"]["Signature"] == payload["results_table"][0]["Signature"]
    assert payload["best_split"]["Stability score"] is not None
    assert payload["search_space"]["min_events_per_group"] >= 3
    assert payload["search_space"]["bootstrap_iterations"] == 10
    assert payload["search_space"]["bootstrap_scored_signatures"] >= 1
    assert payload["search_space"]["permutation_iterations"] == 20
    assert payload["search_space"]["permutation_scored_signatures"] >= 1
    assert payload["search_space"]["validation_iterations"] == 6
    assert payload["search_space"]["validation_fraction"] == 0.35
    assert payload["search_space"]["validation_scored_signatures"] >= 1
    assert payload["search_space"]["significance_level"] == 0.05
    assert payload["search_space"]["combination_operator"] == "and"
    assert payload["search_space"]["random_seed"] == 1234
    assert payload["search_space"]["significant_signatures"] >= 0
    support = payload["best_split"]["Bootstrap support (p<alpha)"]
    assert support is None or 0.0 <= support <= 1.0
    permutation_p = payload["best_split"]["Permutation p"]
    assert permutation_p is None or 0.0 <= permutation_p <= 1.0
    direction_consistency = payload["best_split"]["Bootstrap HR direction consistency"]
    assert direction_consistency is None or 0.0 <= direction_consistency <= 1.0
    assert isinstance(payload["best_split"]["Statistically significant"], bool)
    assert payload["best_split"]["Combination operator"] == "AND"
    assert payload["scientific_summary"]["headline"]
    assert payload["scientific_summary"]["status"] in {"robust", "review", "caution"}
    assert payload["scientific_summary"]["metrics"]
    if payload["search_space"]["significant_signatures"] > 0:
        assert payload["best_split"]["Statistically significant"] is True
    observed_groups = set(updated[column_name].dropna().astype(str).unique().tolist())
    assert observed_groups.issubset({"Signature+", "Signature-"})


def test_stability_score_caps_extreme_significance_term() -> None:
    base_row = {
        "Hazard ratio (signature+ vs -)": 2.0,
        "Bootstrap support (p<alpha)": 0.65,
        "Bootstrap HR direction consistency": 0.8,
        "Validation support (p<alpha)": 0.55,
        "Permutation p": 0.02,
        "Rule count": 2,
    }

    moderately_extreme = _stability_score({**base_row, "BH adjusted p": 1e-10})
    absurdly_extreme = _stability_score({**base_row, "BH adjusted p": 1e-50})

    assert absurdly_extreme == pytest.approx(moderately_extreme)


def test_stability_score_tolerates_missing_hazard_ratio() -> None:
    row = {
        "Hazard ratio (signature+ vs -)": None,
        "Bootstrap support (p<alpha)": 0.65,
        "Bootstrap HR direction consistency": 0.8,
        "Validation support (p<alpha)": 0.55,
        "Permutation p": 0.02,
        "Rule count": 2,
        "BH adjusted p": 0.01,
    }

    score = _stability_score(row)

    assert np.isfinite(score)


def test_signature_rules_do_not_repeat_same_feature_within_combo() -> None:
    df = make_example_dataset(seed=55, n_patients=280)
    _, _, payload = discover_feature_signature(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        candidate_columns=["age", "biomarker_score", "immune_index"],
        max_combination_size=3,
        top_k=20,
        min_group_fraction=0.1,
        bootstrap_iterations=0,
        permutation_iterations=0,
    )

    for row in payload["results_table"]:
        features = row["Features"]
        assert len(features) == len(set(features))


def test_discover_feature_signature_is_reproducible_with_fixed_seed() -> None:
    df = make_example_dataset(seed=71, n_patients=260)
    params = dict(
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        candidate_columns=["age", "stage", "treatment", "biomarker_score", "immune_index"],
        max_combination_size=3,
        top_k=8,
        min_group_fraction=0.1,
        bootstrap_iterations=8,
        permutation_iterations=12,
        validation_iterations=5,
        validation_fraction=0.35,
        significance_level=0.05,
        combination_operator="mixed",
        random_seed=2026,
    )
    _, _, payload_first = discover_feature_signature(df, **params)
    _, _, payload_second = discover_feature_signature(df, **params)

    assert payload_first["best_split"]["Signature"] == payload_second["best_split"]["Signature"]
    assert payload_first["best_split"]["Combination operator"] == payload_second["best_split"]["Combination operator"]
    assert payload_first["best_split"]["Stability score"] == payload_second["best_split"]["Stability score"]


def test_discover_feature_signature_generates_unique_default_name_on_repeat() -> None:
    df = make_example_dataset(seed=71, n_patients=260)
    params = dict(
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        candidate_columns=["age", "stage", "treatment", "biomarker_score", "immune_index"],
        max_combination_size=2,
        top_k=6,
        min_group_fraction=0.1,
        bootstrap_iterations=0,
        permutation_iterations=0,
        validation_iterations=0,
        significance_level=0.05,
        combination_operator="mixed",
        random_seed=2026,
    )

    updated_first, first_name, _ = discover_feature_signature(df, **params)
    updated_second, second_name, _ = discover_feature_signature(updated_first, **params)

    assert first_name == "auto_signature_group"
    assert second_name == "auto_signature_group_2"
    assert second_name in updated_second.columns


def test_discover_feature_signature_limits_cox_estimation_to_ranked_candidates(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    rng = np.random.default_rng(17)
    df = make_example_dataset(seed=17, n_patients=220)
    for idx in range(10):
        df[f"signal_{idx}"] = rng.normal(size=len(df))

    calls = {"count": 0}

    def _flat_survdiff(times, events, groups):
        return 1.0, 0.9

    def _fake_cox(times, events, mask):
        calls["count"] += 1
        return {
            "Hazard ratio (signature+ vs -)": 1.2,
            "HR CI lower": 1.01,
            "HR CI upper": 1.5,
        }

    monkeypatch.setattr(analysis, "survdiff", _flat_survdiff)
    monkeypatch.setattr(analysis, "_signature_cox_metrics", _fake_cox)

    _, _, payload = analysis.discover_feature_signature(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        candidate_columns=[f"signal_{idx}" for idx in range(10)],
        max_combination_size=2,
        top_k=5,
        bootstrap_iterations=0,
        permutation_iterations=0,
        validation_iterations=0,
        combination_operator="mixed",
        random_seed=99,
    )

    assert payload["search_space"]["tested_combinations"] > 80
    assert calls["count"] == 80


def test_discover_feature_signature_warns_when_cox_estimation_fails(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=23, n_patients=80)

    monkeypatch.setattr(analysis, "survdiff", lambda times, events, groups: (4.5, 0.03))
    monkeypatch.setattr(
        analysis,
        "_signature_cox_metrics",
        lambda times, events, mask: (_ for _ in ()).throw(ValueError("unstable fit")),
    )

    with pytest.warns(RuntimeWarning, match="Skipping Cox robustness metrics"):
        _, _, payload = analysis.discover_feature_signature(
            df,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            candidate_columns=["age", "biomarker_score", "immune_index"],
            max_combination_size=1,
            top_k=3,
            bootstrap_iterations=0,
            permutation_iterations=0,
            validation_iterations=0,
            combination_operator="mixed",
            random_seed=99,
        )

    assert payload["results_table"]


def test_discover_feature_signature_reraises_memory_error_from_cox_metrics(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=41, n_patients=80)

    monkeypatch.setattr(analysis, "survdiff", lambda times, events, groups: (4.5, 0.03))
    monkeypatch.setattr(
        analysis,
        "_signature_cox_metrics",
        lambda times, events, mask: (_ for _ in ()).throw(MemoryError("out of memory")),
    )

    with pytest.raises(MemoryError, match="out of memory"):
        analysis.discover_feature_signature(
            df,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            candidate_columns=["age", "biomarker_score", "immune_index"],
            max_combination_size=1,
            top_k=3,
            bootstrap_iterations=0,
            permutation_iterations=0,
            validation_iterations=0,
            combination_operator="mixed",
            random_seed=99,
        )


def test_permutation_p_value_reraises_memory_error(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    times = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    events = np.asarray([1, 1, 0, 0], dtype=int)
    mask = np.asarray([True, False, True, False], dtype=bool)

    monkeypatch.setattr(
        analysis,
        "survdiff",
        lambda times, events, groups: (_ for _ in ()).throw(MemoryError("oom in permutation")),
    )

    with pytest.raises(MemoryError, match="oom in permutation"):
        _permutation_p_value(times, events, mask, observed_stat=3.5, n_iterations=3, random_seed=7)


def test_harrell_c_index_bootstrap_ci_warns_when_internal_bootstrap_is_skipped() -> None:
    times = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    events = np.asarray([1, 0, 1, 0], dtype=int)
    risk = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=float)

    with pytest.warns(RuntimeWarning, match="bootstrap CI skipped"):
        result = _harrell_c_index_bootstrap_ci(times, events, risk, n_bootstrap=20)

    assert result["c_index_ci_lower"] is None
    assert result["c_index_ci_upper"] is None


def test_signature_summary_stays_exploratory_without_permutation_or_holdout_confirmation() -> None:
    summary = _signature_scientific_summary(
        best_split={
            "N signature+": 42,
            "Statistically significant": True,
            "Bootstrap support (p<alpha)": None,
            "Bootstrap HR direction consistency": None,
            "Validation support (p<alpha)": None,
            "Permutation p": None,
        },
        search_space={
            "truncated": False,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "significance_level": 0.05,
            "min_group_size": 10,
            "tested_combinations": 12,
            "significant_signatures": 1,
        },
    )

    assert "remains exploratory" in summary["headline"].lower()
    assert any("optimistic" in caution.lower() for caution in summary["cautions"])
    assert any("changing the candidate set can change the search cohort" in caution.lower() for caution in summary["cautions"])


def test_signature_summary_keeps_internal_confirmation_language_conservative() -> None:
    summary = _signature_scientific_summary(
        best_split={
            "N signature+": 42,
            "Statistically significant": True,
            "Bootstrap support (p<alpha)": 0.82,
            "Bootstrap HR direction consistency": 0.91,
            "Validation support (p<alpha)": 0.67,
            "Permutation p": 0.021,
        },
        search_space={
            "truncated": False,
            "permutation_iterations": 50,
            "validation_iterations": 5,
            "significance_level": 0.05,
            "min_group_size": 10,
            "tested_combinations": 24,
            "significant_signatures": 3,
        },
    )

    assert "passes the current internal significance" not in summary["headline"].lower()
    assert "external validation" in summary["headline"].lower()


def test_signature_summary_warns_that_truncation_depends_on_candidate_order() -> None:
    summary = _signature_scientific_summary(
        best_split={
            "N signature+": 42,
            "Statistically significant": False,
            "Bootstrap support (p<alpha)": None,
            "Bootstrap HR direction consistency": None,
            "Validation support (p<alpha)": None,
            "Permutation p": None,
        },
        search_space={
            "truncated": True,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "significance_level": 0.05,
            "min_group_size": 10,
            "tested_combinations": 5000,
            "significant_signatures": 0,
        },
    )

    assert any("candidate order" in caution.lower() for caution in summary["cautions"])

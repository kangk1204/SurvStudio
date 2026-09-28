from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pandas as pd

from survival_toolkit.__main__ import main as cli_main
from survival_toolkit.analysis import (
    MAX_MODEL_FEATURE_CANDIDATES,
    _bh_adjust,
    _cohort_frame,
    _cox_martingale_plot_data,
    _cox_scientific_summary,
    _harrell_c_index,
    _harrell_c_index_bootstrap_ci,
    _has_ambiguous_competing_event_tokens,
    _ordered_reference_categories,
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
from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset
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


def test_bh_adjust_preserves_finite_values_when_some_p_values_are_nan() -> None:
    adjusted = _bh_adjust([0.01, 0.05, np.nan, 0.20])

    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.075)
    assert np.isnan(adjusted[2])
    assert adjusted[3] == pytest.approx(0.20)


def test_vectorized_logrank_matches_statsmodels_survdiff() -> None:
    from statsmodels.duration.survfunc import survdiff

    from survival_toolkit.analysis import _vectorized_logrank_chisq

    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(20, 120))
        times = rng.integers(1, 30, n).astype(float)
        events = rng.integers(0, 2, n)
        if events.sum() == 0:
            continue
        masks = rng.random((n, 4)) < 0.4
        statistics = _vectorized_logrank_chisq(times, events, masks)
        for column in range(4):
            if masks[:, column].all() or (~masks[:, column]).all():
                continue
            reference, _ = survdiff(times, events, np.where(masks[:, column], "a", "b"))
            assert statistics[column] == pytest.approx(reference, abs=1e-9)


def test_search_adjusted_permutation_uses_best_statistic_of_every_shuffle() -> None:
    from survival_toolkit.analysis import _search_adjusted_permutation_p_values

    rng = np.random.default_rng(1)
    n = 120
    times = rng.exponential(1.0, n)
    events = (rng.random(n) < 0.7).astype(int)
    masks = rng.random((n, 30)) < 0.5
    observed = np.array([0.0] * 29 + [1e6])
    p_values, valid = _search_adjusted_permutation_p_values(
        times, events, masks, observed, min_events_per_group=3, n_iterations=49, random_seed=3,
    )
    assert valid == 49
    # A statistic of 0 is always matched by the per-shuffle maximum; a huge one never is.
    assert p_values[0] == pytest.approx(1.0)
    assert p_values[-1] == pytest.approx(1.0 / 50.0)


def test_cohort_frame_tracks_missing_and_infinite_rows() -> None:
    df = pd.DataFrame(
        {
            "os_months": [12.0, np.inf, 24.0, 36.0],
            "os_event": [1, 0, 1, 0],
            "age": [60.0, 61.0, np.nan, 63.0],
        }
    )

    frame = _cohort_frame(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        extra_columns=["age"],
        drop_missing_extra_columns=True,
    )

    assert frame.shape[0] == 2
    assert frame.attrs["rows_with_infinite_values"] == 1
    assert frame.attrs["dropped_missing_rows"] == 2


def test_bootstrap_signature_metrics_reports_skipped_resamples(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 1, 0, 0],
            "marker": [0.0, 0.0, 1.0, 1.0],
        }
    )

    monkeypatch.setattr(
        analysis,
        "_signature_mask",
        lambda sampled, combo, operator="AND": np.ones(len(sampled), dtype=bool),
    )

    metrics = analysis._bootstrap_signature_metrics(
        frame=df,
        time_column="time",
        event_column="event",
        combo=[{"column": "marker", "label": "marker"}],
        combo_operator="AND",
        min_group_size=1,
        n_iterations=4,
        sample_fraction=0.8,
        random_seed=7,
        significance_level=0.05,
    )

    assert metrics["Bootstrap valid resamples"] == 0
    assert metrics["Bootstrap skipped resamples"] == 4


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

    assert upload.shape == (489, 112)
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

    assert upload.shape == (489, 512)
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


def test_cohort_frame_notes_non_time_numeric_column_when_likely_time_exists() -> None:
    df = pd.DataFrame(
        {
            "os_months": [12, 18, 24],
            "os_event": [1, 0, 1],
            "SFTPC": [6.1, 8.4, 5.9],
            "age": [60, 55, 70],
        }
    )

    # An unrecognized time-column name is a caution carried with the results, not a hard error:
    # standard endpoints are named in too many ways for a name check to refuse them.
    frame = _cohort_frame(
        df,
        time_column="SFTPC",
        event_column="os_event",
        event_positive_value=1,
        extra_columns=["age"],
    )
    assert "does not look like a survival follow-up time column" in frame.attrs["time_column_note"]
    assert "os_months" in frame.attrs["time_column_note"]
    km = compute_km_analysis(df, "SFTPC", "os_event")
    assert any("SFTPC" in caution for caution in km["scientific_summary"]["cautions"])
    assert km["scientific_summary"]["status"] != "robust"


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


def test_harrell_c_index_returns_none_when_no_comparable_pairs_exist() -> None:
    time_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    event_values = np.array([0, 0, 0, 0], dtype=int)
    risk_score = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)

    assert _harrell_c_index(time_values, event_values, risk_score) is None


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
    assert result["rmst_contrast"] is not None
    assert any(metric["label"] == "RMST difference" for metric in result["scientific_summary"]["metrics"])
    assert any(metric["label"] == "RMST difference p" for metric in result["scientific_summary"]["metrics"])
    assert result["rmst_contrast"]["p_value"] is not None
    assert result["rmst_horizon"] <= result["display_horizon"]
    for row in result["summary_table"]:
        assert row["RMST CI lower"] is not None
        assert row["RMST CI upper"] is not None
        assert row["RMST SE"] is not None
        assert row["RMST CI lower"] <= row["RMST"] <= row["RMST CI upper"]
    assert result["rmst_contrast"]["ci_lower"] <= result["rmst_contrast"]["estimate"] <= result["rmst_contrast"]["ci_upper"]


def test_km_analysis_keeps_rmst_ci_when_risk_set_is_exhausted() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 1.0, 2.0, 2.0],
            "event": [1, 1, 1, 1],
            "group": ["A", "A", "B", "B"],
        }
    )

    result = compute_km_analysis(
        df,
        time_column="time",
        event_column="event",
        group_column="group",
        event_positive_value=1,
        max_time=2.0,
    )

    # survRM2 convention: an exhausted risk set (n == d) contributes zero
    # variance because the KM tail area beyond that event is zero.
    assert result["rmst_contrast"] is not None
    assert result["rmst_contrast"]["estimate"] == pytest.approx(0.0)
    for row in result["summary_table"]:
        assert row["RMST"] == pytest.approx(1.0)
        assert row["RMST SE"] == pytest.approx(0.0)
        assert row["RMST CI lower"] == pytest.approx(1.0)
        assert row["RMST CI upper"] == pytest.approx(1.0)


def test_km_rmst_standard_errors_match_survrm2_convention_on_tcga() -> None:
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "data" / "tcga_luad_upload_ready.csv")
    result = compute_km_analysis(df, "os_months", "os_event", "stage_group")
    # Reference values from R 4 with survival 3.8.6: survfit(Surv(os_months, os_event) ~ 1) per stage
    # group, restricted at the shortest group follow-up (88.07 months), with the variance formula of
    # survRM2::rmst1 (Greenwood terms d / (n (n - d)) weighted by the squared area after each event
    # time, no n / (n - 1) correction). survRM2 was not installed, so its rmst1 formula was applied to
    # the survfit output directly.
    reference = {
        "Stage I": (61.3004096782, 2.5986477879),
        "Stage II": (43.6288937667, 3.8330724375),
        "Stage III": (35.8646347781, 4.2409324037),
        "Stage IV": (32.0596143644, 6.0489523150),
    }
    assert result["rmst_horizon"] == pytest.approx(88.07)
    z_value = 1.959963984540054  # two-sided 95% normal quantile
    assert sorted(row["Group"] for row in result["summary_table"]) == sorted(reference)
    for row in result["summary_table"]:
        rmst, se = reference[row["Group"]]
        assert row["RMST"] == pytest.approx(rmst, rel=1e-9)
        assert row["RMST SE"] == pytest.approx(se, rel=1e-9)
        assert row["RMST CI lower"] == pytest.approx(rmst - z_value * se, rel=1e-9)
        assert row["RMST CI upper"] == pytest.approx(rmst + z_value * se, rel=1e-9)


def test_km_group_curves_stop_at_each_group_last_follow_up() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0],
            "event": [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    result = compute_km_analysis(df, "time", "event", "group")
    ends = {curve["group"]: curve["timeline"][-1] for curve in result["curves"]}
    assert ends["A"] == pytest.approx(5.0)
    assert ends["B"] == pytest.approx(20.0)


def test_km_median_ci_uses_requested_confidence_level() -> None:
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "data" / "gbsg2_upload_ready.csv")
    wide = compute_km_analysis(df, "rfs_days", "rfs_event", confidence_level=0.95)["summary_table"][0]
    narrow = compute_km_analysis(df, "rfs_days", "rfs_event", confidence_level=0.80)["summary_table"][0]
    assert narrow["Median CI lower"] >= wide["Median CI lower"]
    assert narrow["Median CI upper"] <= wide["Median CI upper"]
    assert (narrow["Median CI lower"], narrow["Median CI upper"]) != (wide["Median CI lower"], wide["Median CI upper"])


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
    assert result["rmst_contrast"] is not None
    assert result["rmst_contrast"]["estimate"] is not None
    assert result["rmst_contrast"]["p_value"] is None
    assert result["rmst_contrast"]["p_value_method"] is None
    assert "descriptive" in result["scientific_summary"]["headline"].lower()
    assert any("exploratory rather than confirmatory" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("wald p-value is intentionally withheld" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert not any(metric["label"] == "RMST difference p" for metric in result["scientific_summary"]["metrics"])


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


def test_km_analysis_uses_common_group_follow_up_for_rmst_horizon() -> None:
    df = pd.DataFrame(
        {
            "time": [5.0, 8.0, 11.0, 18.0, 20.0, 24.0],
            "event": [1, 0, 1, 1, 0, 0],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    result = compute_km_analysis(
        df,
        time_column="time",
        event_column="event",
        group_column="group",
        event_positive_value=1,
    )

    assert result["display_horizon"] == pytest.approx(24.0)
    assert result["rmst_horizon"] == pytest.approx(11.0)
    assert result["rmst_contrast"] is not None
    assert result["rmst_contrast"]["horizon"] == pytest.approx(11.0)
    assert any(
        "shared observed follow-up support" in strength.lower()
        for strength in result["scientific_summary"]["strengths"]
    )


def test_km_analysis_all_event_group_keeps_final_ci_finite() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 1, 1, 1],
        }
    )

    result = compute_km_analysis(
        df,
        time_column="time",
        event_column="event",
        event_positive_value=1,
    )

    curve = result["curves"][0]
    assert np.isfinite(curve["ci_lower"][-1])
    assert np.isfinite(curve["ci_upper"][-1])
    assert curve["ci_lower"][-1] == pytest.approx(0.0)
    assert curve["ci_upper"][-1] == pytest.approx(0.0)
    assert result["summary_table"][0]["Censored"] == 0


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
    assert result["model_stats"]["c_index_ci_method"] == "bootstrap_percentile_fixed_score"
    assert result["model_stats"]["c_index_ci_level"] == pytest.approx(0.95)
    assert result["model_stats"]["c_index_ci_lower"] is not None
    assert result["model_stats"]["c_index_ci_upper"] is not None
    assert result["model_stats"]["c_index_ci_lower"] <= result["model_stats"]["c_index"] <= result["model_stats"]["c_index_ci_upper"]
    assert result["model_stats"]["lr_statistic"] is not None
    assert result["model_stats"]["lr_pvalue"] is not None
    assert result["model_stats"]["global_ph_pvalue"] is not None
    assert any(
        row["Term"] == "Global PH test (Grambsch-Therneau)"
        for row in result["diagnostics_table"]
    )
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
    assert any("grambsch-therneau" in strength.lower() and "schoenfeld" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("likelihood-ratio test" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("evaluable patient pairs" in strength.lower() for strength in result["scientific_summary"]["strengths"])
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
    monkeypatch.setattr(
        analysis,
        "_efron_schoenfeld_residuals",
        lambda *args, **kwargs: np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float),
    )
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

    # Scaled Schoenfeld residuals are d * r * V (2 events, variance 2).
    first_trace = result["diagnostics_plot_data"][0]
    assert first_trace["residual"] == pytest.approx([4.0, 8.0, 12.0, 16.0])
    assert len(first_trace["trend_log_time"]) == len(first_trace["trend_residual"])
    martingale_trace = result["martingale_plot_data"][0]
    assert martingale_trace["term"] == "age"
    assert len(martingale_trace["value"]) == len(martingale_trace["residual"])
    assert result["model_stats"]["c_index_ci_lower"] == pytest.approx(0.57)
    assert result["model_stats"]["c_index_ci_upper"] == pytest.approx(0.67)
    assert result["model_stats"]["lr_statistic"] == pytest.approx(6.0)
    assert result["model_stats"]["lr_pvalue"] is not None
    # A single-term model still has a (1-df) global Grambsch-Therneau test.
    assert result["model_stats"]["global_ph_pvalue"] is not None
    assert result["model_stats"]["global_ph_df"] == pytest.approx(1.0)


def test_cox_martingale_plot_data_skips_mismatched_residual_lengths() -> None:
    frame = pd.DataFrame({"age": [50.0, 55.0, 60.0, 65.0]})

    with pytest.warns(RuntimeWarning, match="residual vector length"):
        result = _cox_martingale_plot_data(
            frame,
            np.asarray([0.1, 0.2, 0.3], dtype=float),
            covariates=["age"],
            categorical_covariates=[],
        )

    assert result == []


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
    assert any(metric["label"] == "Dropped for missing Cox inputs" and metric["value"] == 24 for metric in result["scientific_summary"]["metrics"])
    assert any("were excluded" in caution.lower() and "missing" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_km_analysis_reports_nonpositive_time_exclusions() -> None:
    df = make_example_dataset(seed=212, n_patients=60)
    df.loc[df.index[:3], "os_months"] = [0.0, -1.0, 0.0]

    result = compute_km_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
    )

    # Time 0 is kept (as in R survival / lifelines); only the negative time is dropped.
    assert result["cohort"]["dropped_nonpositive_time_rows"] == 1
    assert result["cohort"]["n"] == 59
    assert any(
        metric["label"] == "Dropped for negative time" and metric["value"] == 1
        for metric in result["scientific_summary"]["metrics"]
    )
    assert any("negative survival time" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_cox_analysis_reports_nonpositive_time_exclusions() -> None:
    df = make_example_dataset(seed=213, n_patients=120)
    df.loc[df.index[:4], "os_months"] = [0.0, -2.0, 0.0, -0.5]

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age", "biomarker_score"],
        categorical_covariates=[],
    )

    assert result["model_stats"]["dropped_nonpositive_time_rows"] == 2
    assert any(
        metric["label"] == "Dropped for negative time" and metric["value"] == 2
        for metric in result["scientific_summary"]["metrics"]
    )
    assert any("negative survival time" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_cox_scientific_summary_reports_when_lr_test_is_not_reportable() -> None:
    summary = _cox_scientific_summary(
        model_rows=[],
        diagnostic_rows=[],
        model_stats={
            "n": 40,
            "outcome_rows": 40,
            "dropped_rows": 0,
            "events": 12,
            "parameters": 2,
            "events_per_parameter": 6.0,
            "c_index": 0.64,
            "lr_statistic": None,
            "lr_pvalue": None,
            "lr_note": "Overall likelihood-ratio test was not reportable because the null-model comparison returned an invalid negative chi-square candidate.",
        },
    )

    assert any(
        "likelihood-ratio test was not reportable" in caution.lower()
        for caution in summary["cautions"]
    )


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


def test_km_analysis_rejects_nonpositive_max_time_and_invalid_confidence_level() -> None:
    df = make_example_dataset(seed=41, n_patients=40)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        compute_km_analysis(
            df,
            time_column="os_months",
            event_column="os_event",
            confidence_level=1.0,
        )

    with pytest.raises(ValueError, match="max_time must be positive"):
        compute_km_analysis(
            df,
            time_column="os_months",
            event_column="os_event",
            max_time=0,
        )


def test_km_analysis_rejects_identifier_like_group_columns_with_too_many_levels() -> None:
    n_patients = 51
    df = pd.DataFrame(
        {
            "time": np.linspace(1.0, float(n_patients), n_patients),
            "event": [1 if idx % 2 == 0 else 0 for idx in range(n_patients)],
            "patient_id": [f"P{idx:03d}" for idx in range(n_patients)],
        }
    )

    with pytest.raises(ValueError, match="Kaplan-Meier can plot at most 50 groups"):
        compute_km_analysis(
            df,
            time_column="time",
            event_column="event",
            group_column="patient_id",
        )


def test_cox_grambsch_therneau_test_matches_lifelines_on_gbsg2() -> None:
    lifelines = pytest.importorskip("lifelines")
    from lifelines.statistics import proportional_hazard_test

    df = pd.read_csv(Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "data" / "gbsg2_upload_ready.csv")
    covariates = ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"]
    categorical = ["horTh", "menostat", "tgrade"]
    result = compute_cox_analysis(df, "rfs_days", "rfs_event", covariates, categorical)
    by_term = {row["Term"]: row for row in result["diagnostics_table"]}

    dummies = pd.get_dummies(df[["rfs_days", "rfs_event", *covariates]], columns=categorical, drop_first=True, dtype=float)
    fitter = lifelines.CoxPHFitter().fit(dummies, "rfs_days", "rfs_event")
    reference = proportional_hazard_test(fitter, dummies, time_transform="log").summary
    mapping = {
        "age": "age",
        "pnodes": "pnodes",
        "tsize": "tsize",
        "horTh: yes vs no": "horTh_yes",
        "menostat: Pre vs Post": "menostat_Pre",
        "tgrade: II vs I": "tgrade_II",
        "tgrade: III vs I": "tgrade_III",
    }
    for term, reference_name in mapping.items():
        assert by_term[term]["Chi-square"] == pytest.approx(float(reference.loc[reference_name, "test_statistic"]), rel=0.02, abs=0.01)
    assert result["model_stats"]["global_ph_method"] == "grambsch_therneau_log_time"
    assert result["model_stats"]["global_ph_df"] == pytest.approx(7.0)


def test_cox_ph_test_keeps_nominal_type_one_error_for_categorical_terms() -> None:
    rng = np.random.default_rng(11)
    rejections = 0
    n_sim = 60
    for _ in range(n_sim):
        n = 250
        x1 = rng.normal(size=n)
        group = rng.choice(list("ABCD"), size=n)
        linear_predictor = 0.5 * x1 + np.select([group == "B", group == "C", group == "D"], [0.3, 0.6, 0.9], 0.0)
        event_time = rng.exponential(1.0 / np.exp(linear_predictor))
        censor_time = rng.exponential(2.0, size=n)
        frame = pd.DataFrame(
            {
                "time": np.minimum(event_time, censor_time),
                "event": (event_time <= censor_time).astype(int),
                "x1": x1,
                "group": group,
            }
        )
        result = compute_cox_analysis(frame, "time", "event", ["x1", "group"], ["group"])
        rejections += int(result["model_stats"]["global_ph_pvalue"] < 0.05)
    # The retired Spearman screen rejected ~50% of these PH-true datasets.
    assert rejections / n_sim < 0.15


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
    # Five events for four coefficients: fewer coefficients than events, so the design is not
    # refused before fitting, but the events per parameter are still very low.
    frame = pd.DataFrame(
        {
            "os_months": [1, 2, 3, 4, 5, 6],
            "os_event": [1, 1, 1, 1, 1, 0],
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
    assert "EPV=1.25" in message
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


def test_cox_analysis_rejects_constant_numeric_covariates_before_fit() -> None:
    df = pd.DataFrame(
        {
            "os_months": [10, 12, 14, 16, 18, 20],
            "os_event": [1, 0, 1, 0, 1, 0],
            "age": [60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
            "biomarker": [0.3, 0.4, 0.8, 0.9, 1.1, 1.2],
        }
    )

    with pytest.raises(ValueError, match="constant value") as exc_info:
        compute_cox_analysis(
            df,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            covariates=["age", "biomarker"],
        )

    assert "age (constant value)" in str(exc_info.value)


def test_preview_cox_analysis_inputs_warns_on_single_level_categorical_covariates() -> None:
    df = pd.DataFrame(
        {
            "os_months": [10, 12, 14, 16, 18, 20],
            "os_event": [1, 0, 1, 0, 1, 0],
            "stage_group": ["Stage I", "Stage I", "Stage I", "Stage I", "Stage I", "Stage I"],
            "biomarker": [0.3, 0.4, 0.8, 0.9, 1.1, 1.2],
        }
    )

    preview = preview_cox_analysis_inputs(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["stage_group", "biomarker"],
        categorical_covariates=["stage_group"],
    )

    assert any(
        "stage_group has only one observed level" in warning
        for warning in preview["stability_warnings"]
    )


def test_preview_cox_analysis_inputs_counts_missing_strata_rows() -> None:
    df = pd.DataFrame(
        {
            "os_months": [10, 12, 14, 16, 18, 20],
            "os_event": [1, 0, 1, 0, 1, 0],
            "age": [50, 52, 54, 56, 58, 60],
            "stage_group": ["Stage I", None, "Stage II", None, "Stage I", "Stage II"],
        }
    )

    preview = preview_cox_analysis_inputs(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age"],
        categorical_covariates=[],
        strata_columns=["stage_group"],
    )

    assert preview["dropped_rows"] == 2
    assert preview["strata_columns"] == ["stage_group"]
    assert {"column": "stage_group", "missing_rows": 2} in preview["missing_by_covariate"]


def test_cox_analysis_passes_strata_to_phreg_and_reports_stratification(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=31, n_patients=4)
    frame = pd.DataFrame(
        {
            "os_months": [1.0, 2.0, 3.0, 4.0],
            "os_event": [1, 1, 0, 0],
            "age": [50.0, 55.0, 60.0, 65.0],
            "sex": ["Female", "Female", "Male", "Male"],
            "stage_group": ["Stage I", "Stage II", "Stage I", "Stage II"],
        }
    )
    captured: dict[str, Any] = {}

    class _FakeResults:
        params = np.asarray([0.2], dtype=float)
        bse = np.asarray([0.1], dtype=float)
        tvalues = np.asarray([2.0], dtype=float)
        pvalues = np.asarray([0.04], dtype=float)
        schoenfeld_residuals = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float)
        martingale_residuals = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=float)
        llf = -4.0
        llnull = -7.0
        model = type("_FakeModelMeta", (), {"exog_names": ['Q("age")'], "exog": np.ones((4, 1), dtype=float)})()

        def conf_int(self):
            return np.asarray([[0.1, 0.3]], dtype=float)

        def cov_params(self):
            return np.asarray([[2.0]], dtype=float)

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            captured["formula"] = args[0]
            captured["strata"] = list(kwargs["strata"]) if kwargs.get("strata") is not None else []

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
        strata_columns=["sex", "stage_group"],
    )

    assert captured["formula"] == 'Q("os_months") ~ Q("age")'
    assert len(set(captured["strata"])) == 4
    assert result["strata_columns"] == ["sex", "stage_group"]
    assert result["model_stats"]["n_strata"] == 4
    assert result["model_stats"]["evaluation_mode"] == "stratified_not_reported"
    assert result["model_stats"]["c_index"] is None
    assert any(
        "baseline hazards were stratified by" in strength.lower()
        and "sex" in strength.lower()
        and "stage_group" in strength.lower()
        for strength in result["scientific_summary"]["strengths"]
    )
    assert any("do not receive hazard-ratio estimates" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("discrimination was not reported" in strength.lower() for strength in result["scientific_summary"]["strengths"])


def test_preview_cox_analysis_warns_on_high_cardinality_numeric_strata() -> None:
    df = pd.DataFrame(
        {
            "os_months": np.arange(1, 31, dtype=float),
            "os_event": ([1, 0] * 15),
            "age": np.linspace(50, 79, 30),
            "tsize": np.linspace(1.0, 30.0, 30),
        }
    )

    preview = preview_cox_analysis_inputs(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age"],
        categorical_covariates=[],
        strata_columns=["tsize"],
    )

    assert preview["n_strata"] == 30
    assert any("high-cardinality numeric column" in warning.lower() for warning in preview["stability_warnings"])


def test_cox_strata_encoding_is_collision_free(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    df = make_example_dataset(seed=33, n_patients=4)
    frame = pd.DataFrame(
        {
            "os_months": [1.0, 2.0, 3.0, 4.0],
            "os_event": [1, 1, 0, 0],
            "age": [50.0, 55.0, 60.0, 65.0],
            "left": ["A", "A | right=B", "A", "A | right=B"],
            "right": ["B | tail=C", "C", "B | tail=C", "C"],
        }
    )
    captured: dict[str, Any] = {}

    class _FakeResults:
        params = np.asarray([0.2], dtype=float)
        bse = np.asarray([0.1], dtype=float)
        tvalues = np.asarray([2.0], dtype=float)
        pvalues = np.asarray([0.04], dtype=float)
        schoenfeld_residuals = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float)
        martingale_residuals = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=float)
        llf = -4.0
        llnull = -7.0
        model = type("_FakeModelMeta", (), {"exog_names": ['Q("age")'], "exog": np.ones((4, 1), dtype=float)})()

        def conf_int(self):
            return np.asarray([[0.1, 0.3]], dtype=float)

        def cov_params(self):
            return np.asarray([[2.0]], dtype=float)

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            captured["strata"] = list(kwargs["strata"]) if kwargs.get("strata") is not None else []

            class _FakeFit:
                @staticmethod
                def fit(disp=False):
                    return _FakeResults()

            return _FakeFit()

    monkeypatch.setattr(analysis, "_prepare_cox_frame", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(analysis, "PHReg", _FakePHReg)
    monkeypatch.setattr(analysis, "_reference_levels", lambda *args, **kwargs: {})

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age"],
        strata_columns=["left", "right"],
    )

    assert len(set(captured["strata"])) == 2
    assert result["model_stats"]["n_strata"] == 2


def test_cox_analysis_raises_when_the_optimizer_does_not_converge(monkeypatch) -> None:
    import survival_toolkit.analysis as analysis

    frame = pd.DataFrame(
        {
            "os_months": [1.0, 2.0, 3.0, 4.0],
            "os_event": [1, 1, 0, 0],
            "age": [50.0, 55.0, 60.0, 65.0],
        }
    )

    class _FakePHReg:
        @staticmethod
        def from_formula(*args, **kwargs):
            return object()

    monkeypatch.setattr(analysis, "_prepare_cox_frame", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(analysis, "PHReg", _FakePHReg)
    # Convergence comes from the optimizer's own flag, not from process-wide warning capture.
    monkeypatch.setattr(analysis, "fit_phreg", lambda model: (object(), False))

    with pytest.raises(ValueError, match="did not converge cleanly"):
        compute_cox_analysis(
            frame,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            covariates=["age"],
            categorical_covariates=[],
        )


def test_cox_analysis_lists_all_current_ph_alert_terms_in_summary_caution() -> None:
    rng = np.random.default_rng(5)
    n = 600
    arm = rng.choice(["control", "treated"], size=n)
    age = rng.normal(60.0, 8.0, size=n)
    # Crossing hazards: the treated arm is harmful early and protective late.
    early = rng.exponential(1.0 / np.where(arm == "treated", 2.5, 0.6))
    late = 1.0 + rng.exponential(1.0 / np.where(arm == "treated", 0.15, 0.9))
    event_time = np.where(early < 1.0, early, late)
    censor_time = rng.uniform(0.5, 6.0, size=n)
    df = pd.DataFrame(
        {
            "time": np.minimum(event_time, censor_time),
            "event": (event_time <= censor_time).astype(int),
            "arm": arm,
            "age": age,
        }
    )
    result = compute_cox_analysis(
        df,
        time_column="time",
        event_column="event",
        event_positive_value=1,
        covariates=["arm", "age"],
        categorical_covariates=["arm"],
    )

    significant_terms = [
        str(row["Term"])
        for row in result["diagnostics_table"]
        if row.get("_kind") != "global"
        if row["P value"] is not None and float(row["P value"]) < 0.05
    ]
    cautions = result["scientific_summary"]["cautions"]
    ph_caution = next(
        caution
        for caution in cautions
        if "Possible proportional-hazards violations detected for:" in caution
    )

    assert significant_terms
    assert any("grambsch-therneau" in caution.lower() for caution in cautions)
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
    headline = result["scientific_summary"]["headline"]
    # The headline names the estimates that look unstable, not the significant terms.
    unstable = headline.split("appear unstable:")[1]
    assert 'pathologic_stage reference "Stage I" (n=2)' in unstable


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


def test_cross_analysis_row_mask_hashes_expose_cohort_mismatch() -> None:
    df = make_example_dataset(seed=123, n_patients=96)
    df.loc[df.index[:8], "age"] = np.nan
    df.loc[df.index[8:18], "biomarker_score"] = np.nan

    km = compute_km_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        group_column="stage",
        event_positive_value=1,
    )
    cox = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age", "stage"],
        categorical_covariates=["stage"],
    )
    _, _, signature = discover_feature_signature(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        candidate_columns=["biomarker_score"],
        max_combination_size=1,
        top_k=5,
        bootstrap_iterations=0,
        permutation_iterations=0,
        validation_iterations=0,
        new_column_name="sig_biomarker",
    )

    km_hash = str(km["cohort"]["row_mask_hash"])
    cox_hash = str(cox["model_stats"]["row_mask_hash"])
    signature_hash = str(signature["search_space"]["row_mask_hash"])

    assert len(km_hash) == 16
    assert len(cox_hash) == 16
    assert len(signature_hash) == 16
    assert km_hash != cox_hash
    assert km_hash != signature_hash


def test_cox_analysis_reports_large_design_condition_number_warning() -> None:
    rng = np.random.default_rng(20260411)
    n_rows = 120
    df = pd.DataFrame(
        {
            "os_months": rng.uniform(5.0, 200.0, size=n_rows),
            "os_event": rng.binomial(1, 0.55, size=n_rows),
            "x_small": rng.normal(size=n_rows),
            "x_large": 1e12 * rng.normal(size=n_rows),
        }
    )
    if int(df["os_event"].sum()) == 0:
        df.loc[0, "os_event"] = 1

    result = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        covariates=["x_small", "x_large"],
        categorical_covariates=[],
    )

    condition_number = float(result["model_stats"]["design_condition_number"])
    cautions = result["scientific_summary"]["cautions"]

    assert condition_number > 1e10
    assert any("condition number" in caution.lower() for caution in cautions)


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

    # Tied edges collapse the tertiles to two bins; renaming the survivors
    # "T1"/"T2" would mislabel the top third, so the split is refused.
    with pytest.raises(ValueError, match="enough unique values"):
        derive_group_column(
            df,
            source_column="biomarker_score",
            method="tertile_split",
            new_column_name="biomarker_tertile",
        )


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

    def _fake_cox(times, events, mask, alpha=0.05):
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
        lambda times, events, mask, alpha=0.05: (_ for _ in ()).throw(ValueError("unstable fit")),
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
        lambda times, events, mask, alpha=0.05: (_ for _ in ()).throw(MemoryError("out of memory")),
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
    assert "within-cohort screening rules" in summary["headline"].lower()
    assert "internally supported" not in summary["headline"].lower()
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


def test_bundled_tcga_files_have_one_row_per_patient() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "src/survival_toolkit/data/tcga_luad_xena_example.csv",
        "src/survival_toolkit/data/tcga_luad_upload_ready.csv",
        "examples/tcga_luad_nature2014_upload_ready.csv",
        "examples/tcga_luad_rnaseq_top100_upload.csv",
        "examples/tcga_luad_rnaseq_top500_upload.csv",
    ]:
        frame = pd.read_csv(root / relative)
        assert frame["patient_id"].is_unique, relative
        assert len(frame) == 489, relative


def test_duplicate_identifier_columns_are_flagged_in_profile_and_km_cautions() -> None:
    from survival_toolkit.analysis import detect_duplicate_identifier_columns, profile_dataframe

    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P3", "P4", "P5", "P6", "P7"],
            "sample_type": ["tumor"] * 8,
            "time": [5.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0],
            "event": [1, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    findings = detect_duplicate_identifier_columns(df)
    assert [item["column"] for item in findings] == ["patient_id"]
    assert findings[0]["n_repeated_ids"] == 1
    assert findings[0]["n_extra_rows"] == 1
    profile = profile_dataframe(df, dataset_id="x", filename="x.csv")
    assert profile["duplicate_identifier_columns"][0]["column"] == "patient_id"
    km = compute_km_analysis(df, "time", "event")
    assert any("patient_id" in caution for caution in km["scientific_summary"]["cautions"])
    unique = df.drop_duplicates("patient_id")
    assert detect_duplicate_identifier_columns(unique) == []


def test_tertile_split_refuses_to_merge_tied_bins_silently() -> None:
    df = make_example_dataset(seed=31, n_patients=100)
    df["biomarker_score"] = [0.0] * 60 + list(np.linspace(1.0, 2.0, 40))
    with pytest.raises(ValueError, match="enough unique values"):
        derive_group_column(df, source_column="biomarker_score", method="tertile_split")


def test_quantile_split_reports_exact_cutoffs() -> None:
    df = make_example_dataset(seed=32, n_patients=999)
    _, column, summary = derive_group_column(df, source_column="biomarker_score", method="tertile_split")
    expected = np.quantile(pd.to_numeric(df["biomarker_score"]).to_numpy(dtype=float), [1 / 3, 2 / 3])
    assert summary["cutoffs"] == pytest.approx(expected.tolist(), rel=0, abs=1e-12)


def test_median_split_rejects_single_group_and_percent_fraction_input() -> None:
    df = make_example_dataset(seed=33, n_patients=50)
    df["flag"] = [1.0] * 40 + [0.0] * 10
    with pytest.raises(ValueError, match="single group"):
        derive_group_column(df, source_column="flag", method="median_split")
    with pytest.raises(ValueError, match="fractions"):
        derive_group_column(df, source_column="biomarker_score", method="percentile_split", cutoff="0.25")


def test_derive_group_notes_non_numeric_inputs() -> None:
    df = make_example_dataset(seed=34, n_patients=50)
    df["marker"] = df["biomarker_score"].astype(object)
    df.loc[df.index[:3], "marker"] = "<5"
    _, _, summary = derive_group_column(df, source_column="marker", method="median_split")
    assert any("non-numeric" in note for note in summary["input_notes"])


def test_event_coding_handles_negated_labels_and_rejects_censor_targets() -> None:
    from survival_toolkit.analysis import coerce_event

    recurred = coerce_event(pd.Series(["0:Not Recurred", "1:Recurred", "0:Not Recurred"], name="rfs_status"))
    assert recurred.tolist() == [0.0, 1.0, 0.0]
    progression = coerce_event(pd.Series(["No progression", "Progression"], name="pfs_status"), "Progression")
    assert progression.tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="means no event"):
        coerce_event(pd.Series(["NED", "DOD", "NED"], name="os_status"), "NED")
    with pytest.raises(ValueError, match="other causes"):
        coerce_event(pd.Series(["NED", "DOD", "DOC"], name="dss_status"))
    assert coerce_event(pd.Series(["NED", "DOD", "DOC"], name="os_status")).tolist() == [0.0, 1.0, 1.0]


def test_censoring_indicator_column_is_not_used_as_event() -> None:
    df = pd.DataFrame({"time": [1.0, 2, 3, 4, 5, 6], "censored": [1, 0, 1, 1, 0, 1], "death": [0, 1, 0, 0, 1, 0]})
    assert "censored" not in suggest_columns(df)["event_columns"]
    with pytest.raises(ValueError, match="censoring indicator"):
        compute_km_analysis(df, "time", "censored", event_positive_value=1)


def test_time_column_rules_accept_futime_and_reject_scores_and_dates() -> None:
    df = pd.DataFrame(
        {"futime": [5, 6, 7], "death": [1, 0, 1], "os_risk_score": [0.1, 0.3, 0.2], "last_followup_date": ["2020-01-01"] * 3}
    )
    assert suggest_columns(df)["time_columns"] == ["futime"]
    dated = pd.DataFrame({"os_date": ["2020-01-03", "2021-05-06", "2022-01-01"], "os_event": [1, 0, 1]})
    with pytest.raises(ValueError, match="calendar dates"):
        compute_km_analysis(dated, "os_date", "os_event")


def test_time_zero_rows_are_kept() -> None:
    df = pd.DataFrame({"time": [0.0, 0.0, 1.0, 2.0, 3.0, -1.0], "event": [1, 0, 1, 0, 1, 1]})
    result = compute_km_analysis(df, "time", "event")
    assert result["cohort"]["n"] == 5
    assert result["cohort"]["dropped_nonpositive_time_rows"] == 1


def test_csv_loader_handles_single_column_decimal_comma_and_utf16() -> None:
    from survival_toolkit.analysis import load_dataframe

    assert load_dataframe(b"time\n1\n2\n3\n", "one.csv").columns.tolist() == ["time"]
    european = load_dataframe("time;event;age\n12,5;1;60,1\n7,25;0;55,0\n".encode(), "eu.csv")
    assert european["time"].tolist() == [12.5, 7.25]
    assert load_dataframe("time,event\n1,0\n2,1\n".encode("utf-16"), "u16.csv")["event"].tolist() == [0, 1]


def test_median_follow_up_and_km_median_use_s_le_half_convention() -> None:
    from survival_toolkit.analysis import _median_follow_up

    assert _median_follow_up(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([0, 0, 1, 1])) == pytest.approx(2.0)
    km = compute_km_analysis(pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0], "e": [1, 1, 0, 0]}), "t", "e")
    assert km["summary_table"][0]["Median survival"] == pytest.approx(2.0)


def test_signature_discovery_accepts_boolean_candidates() -> None:
    df = make_example_dataset(seed=51, n_patients=200)
    df["flag"] = df["biomarker_score"] > df["biomarker_score"].median()
    _, _, payload = discover_feature_signature(
        df, "os_months", "os_event", ["flag", "age"], event_positive_value=1,
        max_combination_size=1, bootstrap_iterations=0, permutation_iterations=0, validation_iterations=0,
    )
    assert payload["search_space"]["tested_combinations"] >= 1


# ---------------------------------------------------------------------------------------
# Reference values from R survival 3.8.6 (coxph ties="efron", survdiff, survfit with
# conf.type="log-log"; the proportional-hazards statistics use the classic cox.zph
# formulas on resid(fit, "schoenfeld") with a log-time transform).
# Rows: time, event, x, z, s. Event times are heavily tied.
_R_REFERENCE_ROWS = (
    "2,1,0.034,0,A;2,0,1.36,0,B;1,1,1.225,0,B;6,1,-0.51,1,B;6,0,-0.298,0,B;1,0,-0.527,0,B;"
    "1,1,0.57,0,B;8,1,-0.056,0,B;2,1,0.747,1,A;1,0,-1.847,1,B;1,1,1.567,1,A;4,1,-0.096,0,A;"
    "2,0,0.68,1,B;5,1,-0.137,0,B;2,1,-0.379,0,B;1,1,0.463,1,A;3,1,0.825,1,A;2,0,-0.203,1,A;"
    "1,1,-0.153,1,B;2,1,0.686,0,A;3,0,-0.87,0,A;2,1,-1.514,1,A;5,1,0.395,1,A;2,1,-0.671,0,B;"
    "10,1,-1.92,1,B;1,0,-0.814,0,A;1,0,-0.468,1,A;4,0,-1.193,0,A;7,1,-1.492,0,B;1,0,0.037,0,A;"
    "1,1,0.897,1,A;3,0,-0.233,1,A;4,0,-0.744,1,B;1,1,0.385,0,B;3,0,0.717,0,A;1,0,-0.3,1,B;"
    "2,0,0.545,1,A;2,1,1.043,0,A;8,0,-0.207,0,A;7,1,-0.814,0,A"
)
_R_MARTINGALE_PLAIN = [
    0.7104108398, -1.363916216, 0.7506417682, 0.1169950029, -0.8580007255, -0.09197343023,
    0.8634685579, -1.302138559, 0.3022814688, -0.03416879801, 0.5728766536, 0.4771746507,
    -0.9127574955, 0.2907639008, 0.8019202088, 0.845245736, -0.2223979314, -0.4052348001,
    0.9121731633, 0.472557396, -0.2056495622, 0.9127647566, -0.4467776843, 0.8485663159,
    -0.518607076, -0.07063851825, -0.1214419365, -0.1906520938, 0.5666230444, -0.1544935081,
    0.7693403419, -0.4620332159, -0.3603332771, 0.8848276609, -0.8849870201, -0.1417306531,
    -0.8061942551, 0.2675933079, -2.003667898, 0.1915698797,
]
_R_MARTINGALE_STRATIFIED = [
    0.7338440802, -1.391160175, 0.5910607268, 0.2200783614, -0.7698662561, -0.1430056248,
    0.7728869697, -1.434148327, 0.3813074511, -0.05357041605, 0.6772136774, 0.3491311096,
    -0.9257986981, 0.4079294701, 0.7411452813, 0.8802126009, -0.2285037773, -0.3884161397,
    0.8545835083, 0.522049356, -0.2188303536, 0.9187544474, -0.8106980857, 0.8008451437,
    -0.5594825401, -0.04830040118, -0.0807566162, -0.243063177, 0.5534949296, -0.1037044906,
    0.8231320605, -0.4751275063, -0.2577678873, 0.8076459043, -0.909823282, -0.2148703704,
    -0.7602900697, 0.3414406217, -1.493560242, 0.1339887368,
]


def _r_reference_frame() -> pd.DataFrame:
    rows = [item.split(",") for item in _R_REFERENCE_ROWS.split(";")]
    return pd.DataFrame(
        {
            "time": [float(row[0]) for row in rows],
            "event": [int(row[1]) for row in rows],
            "x": [float(row[2]) for row in rows],
            "z": [int(row[3]) for row in rows],
            "s": [row[4] for row in rows],
        }
    )


def _ph_terms(result: dict[str, Any]) -> list[float]:
    return [row["Chi-square"] for row in result["diagnostics_table"] if row.get("_kind") != "global"]


def test_cox_proportional_hazards_test_matches_r_with_tied_event_times() -> None:
    frame = _r_reference_frame()

    plain = compute_cox_analysis(frame, "time", "event", ["x", "z"])
    assert [row["Beta"] for row in plain["results_table"]] == pytest.approx([0.9195963533, 0.2236803417], abs=1e-7)
    assert _ph_terms(plain) == pytest.approx([0.4337472685, 0.02231142114], rel=1e-6)
    assert plain["model_stats"]["global_ph_statistic"] == pytest.approx(0.4661127274, rel=1e-6)
    assert plain["model_stats"]["c_index"] == pytest.approx(0.7458823529, abs=1e-9)

    stratified = compute_cox_analysis(frame, "time", "event", ["x", "z"], strata_columns=["s"])
    assert [row["Beta"] for row in stratified["results_table"]] == pytest.approx([0.8978913651, 0.2033296142], abs=1e-7)
    assert _ph_terms(stratified) == pytest.approx([0.6012703623, 0.0428374226], rel=1e-6)
    assert stratified["model_stats"]["global_ph_statistic"] == pytest.approx(0.626085916, rel=1e-6)


def test_efron_martingale_residuals_match_r() -> None:
    from survival_toolkit.analysis import _efron_martingale_residuals

    frame = _r_reference_frame()
    exog = frame[["x", "z"]].to_numpy(dtype=float)
    time = frame["time"].to_numpy(dtype=float)
    event = frame["event"].to_numpy(dtype=int)

    plain = _efron_martingale_residuals(exog, time, event, [0.9195963533, 0.2236803417])
    assert plain == pytest.approx(_R_MARTINGALE_PLAIN, abs=1e-8)
    stratified = _efron_martingale_residuals(
        exog, time, event, [0.8978913651, 0.2033296142], pd.factorize(frame["s"])[0]
    )
    assert stratified == pytest.approx(_R_MARTINGALE_STRATIFIED, abs=1e-8)


def test_km_estimates_and_weighted_tests_match_r() -> None:
    frame = _r_reference_frame()
    result = compute_km_analysis(frame, "time", "event", group_column="s")
    fleming = compute_km_analysis(
        frame, "time", "event", group_column="s", logrank_weight="fleming_harrington", fh_p=1.0
    )

    assert result["test"]["chisq"] == pytest.approx(0.1981617151, rel=1e-8)
    assert fleming["test"]["chisq"] == pytest.approx(0.09671944676, rel=1e-8)
    # RMST is truncated at the shorter group's last follow-up (group A ends at 8).
    assert result["rmst_horizon"] == pytest.approx(8.0)
    rows = {row["Group"]: row for row in result["summary_table"]}
    expected = {
        "A": (4.0, 2.0, 7.0, 4.111090067, 0.6377823875),
        "B": (6.0, 2.0, 8.0, 4.924242424, 0.7306230682),
    }
    for group, (median, lower, upper, rmst, rmst_se) in expected.items():
        row = rows[group]
        assert (row["Median survival"], row["Median CI lower"], row["Median CI upper"]) == (median, lower, upper)
        assert row["RMST"] == pytest.approx(rmst, rel=1e-9)
        assert row["RMST SE"] == pytest.approx(rmst_se, rel=1e-9)

    survival_at = {
        ("A", 2): (0.59375, 0.3466203291, 0.773767953),
        ("A", 4): (0.4222222222, 0.1710198692, 0.6564073175),
        ("B", 2): (0.6363636364, 0.3570231236, 0.8200835447),
        ("B", 4): (0.6363636364, 0.3570231236, 0.8200835447),
    }
    curves = {curve["group"]: curve for curve in result["curves"]}
    for (group, time_point), values in survival_at.items():
        curve = curves[group]
        index = int(np.searchsorted(curve["timeline"], time_point, side="right")) - 1
        observed = (curve["survival"][index], curve["ci_lower"][index], curve["ci_upper"][index])
        assert observed == pytest.approx(values, abs=1e-9)


def test_risk_table_labels_stay_unique_on_short_horizons() -> None:
    from survival_toolkit.analysis import _risk_tick_labels

    assert _risk_tick_labels([0.0, 0.5, 1.0]) == ["0", "0.5", "1"]
    assert _risk_tick_labels([0.0, 0.001, 0.002]) == ["0", "0.001", "0.002"]

    frame = pd.DataFrame({"time": np.linspace(0.001, 0.01, 30), "event": [1, 0] * 15})
    result = compute_km_analysis(frame, "time", "event", risk_table_points=6)
    columns = result["risk_table"]["columns"]
    assert len(columns) == len(set(columns)) == 7
    # At-risk counts use the exact tick times: the patient followed to the horizon is still at risk there.
    assert result["risk_table"]["rows"][0][columns[-1]] == 1


def test_km_rejects_unknown_logrank_weight() -> None:
    df = make_example_dataset(seed=21, n_patients=80)
    with pytest.raises(ValueError, match="Unknown logrank_weight 'wilcoxon'"):
        compute_km_analysis(df, "os_months", "os_event", group_column="sex", logrank_weight="wilcoxon")


def test_km_keeps_large_non_integer_group_values_apart() -> None:
    from survival_toolkit.analysis import _canonical_level_strings

    labels = _canonical_level_strings(pd.Series([100000.5, 100000.0, 100001.0, np.nan]))
    assert labels.dropna().nunique() == 3

    frame = pd.DataFrame(
        {
            "time": np.arange(1.0, 21.0),
            "event": [1, 0, 1, 1] * 5,
            "dose": [100000.5, 100000.0] * 10,
        }
    )
    result = compute_km_analysis(frame, "time", "event", group_column="dose")
    assert result["groups"] == 2


def test_km_handles_a_group_without_events() -> None:
    frame = pd.DataFrame(
        {
            "time": np.arange(1.0, 21.0),
            "event": [1, 0] * 5 + [0] * 10,
            "arm": ["treated"] * 10 + ["control"] * 10,
        }
    )
    result = compute_km_analysis(frame, "time", "event", group_column="arm")
    rows = {row["Group"]: row for row in result["summary_table"]}
    assert rows["control"]["Events"] == 0
    assert rows["control"]["Median survival"] is None
    assert rows["control"]["RMST"] == pytest.approx(result["rmst_horizon"])
    assert result["test_p_value"] is not None and 0.0 <= result["test_p_value"] <= 1.0


def test_cox_bic_uses_the_number_of_events() -> None:
    df = make_example_dataset(seed=22, n_patients=150)
    stats = compute_cox_analysis(df, "os_months", "os_event", ["age", "biomarker_score"])["model_stats"]
    expected = -2.0 * stats["partial_log_likelihood"] + stats["parameters"] * np.log(stats["events"])
    assert stats["bic"] == pytest.approx(expected)
    assert stats["bic_sample_size"] == "events"


def test_cox_treats_text_covariates_as_categorical_without_being_told() -> None:
    df = make_example_dataset(seed=23, n_patients=160)
    result = compute_cox_analysis(df, "os_months", "os_event", ["age", "stage"])
    assert result["categorical_covariates"] == ["stage"]
    assert sum(1 for row in result["results_table"] if row["Variable"] == "stage") == df["stage"].nunique() - 1
    preview = preview_cox_analysis_inputs(df, "os_months", "os_event", ["age", "stage"])
    assert preview["categorical_covariates"] == ["stage"]


def test_cox_rejects_numeric_columns_with_stray_text() -> None:
    df = make_example_dataset(seed=24, n_patients=120)
    df["age"] = df["age"].astype(object)
    df.loc[0, "age"] = "unknown"
    with pytest.raises(ValueError, match='Feature "age" looks numeric but contains 1 non-numeric value'):
        compute_cox_analysis(df, "os_months", "os_event", ["age", "sex"])


def test_optimal_cutpoint_labels_rows_without_an_outcome() -> None:
    df = make_example_dataset(seed=25, n_patients=150)
    df.loc[:9, "os_months"] = np.nan
    updated, column, summary = derive_group_column(
        df,
        source_column="biomarker_score",
        method="optimal_cutpoint",
        time_column="os_months",
        event_column="os_event",
        permutation_iterations=20,
    )
    assert summary["n_rows_labelled_without_outcome"] == 10
    assert summary["n_rows_scanned"] == 140
    assert updated.loc[:9, column].notna().all()
    above = updated["biomarker_score"] > summary["cutoff"]
    assert (updated.loc[above, column] == summary["label_above_cutpoint"]).all()
    assert (updated.loc[~above, column] == summary["label_below_cutpoint"]).all()


def test_text_loader_detects_korean_and_western_encodings() -> None:
    from survival_toolkit.analysis import load_dataframe

    korean = "환자,생존기간,사망\n김,12.5,1\n이,7,0\n".encode("cp949")
    frame = load_dataframe(korean, "korean.csv")
    assert frame.columns.tolist() == ["환자", "생존기간", "사망"]
    assert frame.attrs["source_encoding"] == "cp949"

    western = "patient,time,event\nJosé,12.5,1\nBjörn,7,0\n".encode("cp1252")
    frame = load_dataframe(western, "western.csv")
    assert frame["patient"].tolist() == ["José", "Björn"]
    assert frame.attrs["source_encoding"] == "cp1252"

    # A byte that is not UTF-8 only after the 1 MiB sniffed prefix still falls back cleanly.
    late = ("time,event,site\n" + "1,0,A\n" * 200_000 + "2,1,Zürich\n").encode("cp1252")
    frame = load_dataframe(late, "late.csv")
    assert frame["site"].iloc[-1] == "Zürich"


def test_text_loader_enforces_shape_limits_before_the_full_parse() -> None:
    from survival_toolkit.analysis import UploadShapeError, load_dataframe

    rows = "time,event\n" + "1,0\n" * 50
    assert len(load_dataframe(rows.encode(), "ok.csv", max_rows=50)) == 50
    with pytest.raises(UploadShapeError, match="more than 49 rows"):
        load_dataframe(rows.encode(), "rows.csv", max_rows=49)
    with pytest.raises(UploadShapeError, match="3 columns"):
        load_dataframe(b"a,b,c\n1,2,3\n", "cols.csv", max_columns=2)
    with pytest.raises(UploadShapeError, match="more than 99 cells"):
        load_dataframe(rows.encode(), "cells.csv", max_cells=99)


def test_risk_table_times_are_round_and_stop_at_the_horizon() -> None:
    from survival_toolkit.analysis import _nice_time_ticks

    assert _nice_time_ticks(238.11, 6) == [0, 40, 80, 120, 160, 200]
    assert _nice_time_ticks(60.0, 6) == [0, 12, 24, 36, 48, 60]
    # Steps of 600 (7 times) and 800 (5 times) are equally close to 6; the finer one is kept.
    assert _nice_time_ticks(3650.0, 6) == [0, 600, 1200, 1800, 2400, 3000, 3600]
    # More points asked, more times given: the setting stays responsive.
    assert len(_nice_time_ticks(71.35, 6)) == 6 and len(_nice_time_ticks(71.35, 10)) == 9
    assert _nice_time_ticks(0.01, 6) == [0, 0.002, 0.004, 0.006, 0.008, 0.01]
    assert _nice_time_ticks(0.0, 6) == [0.0]

    df = load_tcga_luad_upload_ready_dataset()
    result = compute_km_analysis(df, "os_months", "os_event", group_column="stage_group")
    times = result["risk_table"]["times"]
    assert result["risk_table"]["columns"][1:] == [f"{time:g}" for time in times]
    assert times[-1] <= result["display_horizon"] and all(float(time).is_integer() for time in times)


def test_standing_assumption_notes_do_not_lower_the_status() -> None:
    df = load_tcga_luad_upload_ready_dataset()
    km = compute_km_analysis(df, "os_months", "os_event", group_column="stage_group")
    assert km["scientific_summary"]["status"] == "robust"
    assert any("non-informative" in caution for caution in km["scientific_summary"]["cautions"])

    cox = compute_cox_analysis(
        df,
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        covariates=["age", "stage_group"],
        categorical_covariates=["stage_group"],
    )
    summary = cox["scientific_summary"]
    assert any("non-informative" in caution for caution in summary["cautions"])
    assert summary["status"] in {"robust", "review"}
    if summary["status"] == "review":
        from survival_toolkit.analysis import _COX_STANDING_NOTES

        assert any(caution not in _COX_STANDING_NOTES for caution in summary["cautions"])
        # The data-driven cautions come first, ahead of the notes every Cox fit carries.
        assert summary["cautions"][0] not in _COX_STANDING_NOTES


def test_data_driven_km_cautions_lead_the_standing_notes() -> None:
    rng = np.random.default_rng(4)
    # One group of 10 patients raises the data-driven "fewer than 15 patients" caution.
    df = pd.DataFrame({"time": rng.exponential(10.0, 50), "event": rng.integers(0, 2, 50), "arm": ["a"] * 40 + ["b"] * 10})
    df.loc[[0, 40], "event"] = 1
    km = compute_km_analysis(df, "time", "event", group_column="arm")
    cautions = km["scientific_summary"]["cautions"]
    assert km["scientific_summary"]["status"] != "robust"
    assert "non-informative" not in cautions[0]
    assert "non-informative" in " ".join(cautions)

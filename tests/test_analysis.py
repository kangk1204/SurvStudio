from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from survival_toolkit.__main__ import main as cli_main
from survival_toolkit.analysis import (
    MAX_MODEL_FEATURE_CANDIDATES,
    _ordered_reference_categories,
    _prepare_cox_frame,
    _reference_levels,
    _pointwise_km_ci,
    _signature_scientific_summary,
    _safe_float,
    compute_cohort_table,
    compute_cox_analysis,
    compute_km_analysis,
    coerce_event,
    discover_feature_signature,
    derive_group_column,
    ensure_model_feature_candidate_limit,
    load_dataframe_from_path,
    looks_binary,
)
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


def test_make_example_dataset_small_n_is_supported() -> None:
    df = make_example_dataset(seed=3, n_patients=10)
    assert df.shape[0] == 10


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


def test_coerce_event_rejects_multistate_numeric_status_columns() -> None:
    series = pd.Series([0, 1, 2, 1, 0])
    with pytest.raises(ValueError, match="more than two distinct states|pre-binarized"):
        coerce_event(series, event_positive_value=1)


def test_looks_binary_accepts_nonstandard_two_value_numeric_status() -> None:
    series = pd.Series([1, 2, 1, 2, None])

    assert looks_binary(series) is True


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
    assert result["model_stats"]["c_index_label"] == "Apparent C-index"
    assert result["model_stats"]["apparent_c_index"] == result["model_stats"]["c_index"]
    assert result["scientific_summary"]["headline"]
    assert result["scientific_summary"]["status"] in {"robust", "review", "caution"}
    assert result["scientific_summary"]["metrics"]
    assert any(metric["label"] == "Apparent C-index" for metric in result["scientific_summary"]["metrics"])
    assert any("apparent" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("analyzable cohort" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("spearman" in strength.lower() and "schoenfeld" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert any("external-cohort apply workflow" in caution.lower() for caution in result["scientific_summary"]["cautions"])
    assert any("changing the covariate set" in caution.lower() for caution in result["scientific_summary"]["cautions"])


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

    assert table["columns"] == ["Variable", "Statistic", "Overall", "Stage I", "Stage II", "Stage III", "Stage IV", "unknown"]


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

    assert cohort_size_row["Overall"] == 3
    assert cohort_size_row["A"] == 2
    assert cohort_size_row["B"] == 1


def test_cox_analysis_caps_extreme_hazard_ratios_instead_of_crashing(monkeypatch) -> None:
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
    assert np.isfinite(row["Hazard ratio"])
    assert row["Hazard ratio"] == np.finfo(float).max
    assert row["CI lower"] == np.finfo(float).max
    assert row["CI upper"] == np.finfo(float).max


def test_cox_analysis_rejects_non_finite_model_fit(monkeypatch) -> None:
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
        params = np.asarray([np.nan], dtype=float)
        bse = np.asarray([np.nan], dtype=float)
        tvalues = np.asarray([np.nan], dtype=float)
        pvalues = np.asarray([np.nan], dtype=float)
        schoenfeld_residuals = np.ones((40, 1), dtype=float)
        llf = np.nan
        model = type("_FakeModelMeta", (), {"exog_names": ["Q(\"age\")"], "exog": np.ones((40, 1), dtype=float)})()

        def conf_int(self):
            return np.asarray([[np.nan, np.nan]], dtype=float)

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

    with pytest.raises(ValueError, match="non-finite estimates"):
        compute_cox_analysis(
            df,
            time_column="os_months",
            event_column="os_event",
            event_positive_value=1,
            covariates=["age"],
        )


def test_cohort_table_includes_overall_column() -> None:
    df = make_example_dataset(seed=5, n_patients=120)
    table = compute_cohort_table(df, variables=["age", "sex", "stage"], group_column="treatment")
    assert "Overall" in table["columns"]
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

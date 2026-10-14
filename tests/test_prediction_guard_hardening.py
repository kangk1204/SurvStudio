from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.prediction_guard import (
    check_training_outcome_proxies, guarded_prediction_inputs, prediction_input_audit,
)


def cohort():
    rng = np.random.default_rng(2026101411)
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(120)],
        "time": rng.exponential(20, 120), "event": np.arange(120) % 2,
        "marker": rng.normal(size=120),
    })


@pytest.mark.parametrize("column", ["patient_id", "SubjectID", "TCGA barcode", "case_number"])
@pytest.mark.parametrize("padding", [" ", "\t", "\n", "\u00a0"])
def test_repeated_recognizable_ids_ignore_surrounding_whitespace_only(column, padding):
    df = cohort().rename(columns={"patient_id": column})
    df.loc[1, column] = padding + df.loc[0, column] + padding
    before = df.copy(deep=True)
    with pytest.raises(ValueError, match="Repeated patient IDs.*1 repeated ID.*1 extra row"):
        prediction_input_audit(df, "time", "event", ["marker"], 1)
    pd.testing.assert_frame_equal(df, before)


def test_missing_id_markers_are_not_shared_patient_keys_and_are_reported():
    df = cohort()
    missing = [pd.NA, None, np.nan, "", " NA ", "[Not Available]", "?", " -- ", "n/a"]
    df.loc[:len(missing) - 1, "patient_id"] = missing
    before = df.copy(deep=True)
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    check = audit["identifier_checks"][0]
    assert check["missing_identifier_rows"] == len(missing)
    assert check["n_repeated_ids"] == 0
    assert audit["analyzed_rows"] == 120
    assert any("only recorded IDs" in note for note in audit["notes"])
    pd.testing.assert_frame_equal(df, before)


def test_missing_ids_cannot_hide_a_real_padded_duplicate():
    df = cohort()
    df.loc[:19, "patient_id"] = [" NA ", "[Not Available]"] * 10
    df.loc[21, "patient_id"] = "\t" + df.loc[20, "patient_id"] + " "
    with pytest.raises(ValueError, match="Repeated patient IDs.*1 repeated ID"):
        prediction_input_audit(df, "time", "event", ["marker"], 1)


@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64"])
def test_numeric_ids_keep_numeric_identity_and_detect_real_duplicates(dtype):
    df = cohort()
    df["patient_id"] = pd.Series(np.arange(len(df)), dtype=dtype)
    before = df.copy(deep=True)
    assert prediction_input_audit(df, "time", "event", ["marker"], 1)["analyzed_rows"] == 120
    pd.testing.assert_frame_equal(df, before)
    df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="Repeated patient IDs"):
        prediction_input_audit(df, "time", "event", ["marker"], 1)


def test_id_comparison_preserves_case_leading_zeros_internal_spaces_and_value_types():
    df = cohort()
    # The mixed-type native input must stay object-valued on pandas 3 too.
    df["patient_id"] = df["patient_id"].astype(object)
    distinct = [1, "1", "001", "p1", "P1", "P 1", "P1A"]
    df.loc[:len(distinct) - 1, "patient_id"] = distinct
    before = df.copy(deep=True)
    assert prediction_input_audit(df, "time", "event", ["marker"], 1)["analyzed_rows"] == 120
    pd.testing.assert_frame_equal(df, before)


def test_repeated_baseline_categories_and_decimal_measurements_are_not_patient_ids():
    df = cohort()
    df["patient_group"] = [" A ", "A"] * 60
    df["sample_purity"] = [0.1, 0.2] * 60
    df["measurement_id"] = [0.1, 0.2] * 60
    before = df.copy(deep=True)
    assert prediction_input_audit(df, "time", "event", ["patient_group", "sample_purity"], 1)["analyzed_rows"] == 120
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("column,values", [
    ("dfs_status", [0, 1]), ("DFS.STATUS", ["0:LIVING", "1:RECURRED"]),
    ("RFSStatus", ["No recurrence", "Recurred"]),
    ("os_censor", [0, 1]), ("PFS_CENS", [False, True]),
    ("vital_status", ["Alive", "Dead"]), ("death_event", [1, 2]),
    ("OS_MONTHS", [12.0, 24.0]), ("OSMonths", [12.0, 24.0]),
    ("days_to_last_followup", [35.0, 73.0]),
    ("disease_free_months", [12.0, 24.0]), ("futime", [35.0, 73.0]),
])
def test_other_recognizable_outcomes_are_rejected_as_predictors(column, values):
    df = cohort()
    df[column] = values * 60
    before = df.copy(deep=True)
    with pytest.raises(ValueError, match="Survival outcome columns cannot") as caught:
        prediction_input_audit(df, "time", "event", [column], 1)
    assert column in str(caught.value)
    assert "another endpoint or follow-up" in str(caught.value)
    pd.testing.assert_frame_equal(df, before)


def test_missing_marker_does_not_hide_a_named_outcome_role():
    df = cohort()
    df["dfs_status"] = df["event"].astype(object)
    df.loc[0, "dfs_status"] = "[Not Available]"
    with pytest.raises(ValueError, match="dfs_status"):
        prediction_input_audit(df, "time", "event", ["dfs_status"], 1)


@pytest.mark.parametrize("column", [
    "status", "status2", "diabetes_status", "marital_status", "HER2_status", "MSI_status",
    "stage", "MKI67", "MORF4L1", "OS", "PFS", "DFS", "treatment_duration_months",
    "time_to_surgery_days",
])
def test_ambiguous_codes_genes_and_baseline_characteristics_remain_eligible(column):
    df = cohort()
    df[column] = [0, 1] * 60
    before = df.copy(deep=True)
    audit = prediction_input_audit(df, "time", "event", [column], 1)
    assert audit["analyzed_rows"] == 120
    assert audit["predictor_availability"] == "user_review_required"
    pd.testing.assert_frame_equal(df, before)


def test_status_with_explicit_dead_alive_values_is_not_ambiguous():
    df = cohort()
    df["status"] = ["Dead", "Alive"] * 60
    with pytest.raises(ValueError, match="status"):
        prediction_input_audit(df, "time", "event", ["status"], 1)


def test_comparison_notes_retain_missing_id_coverage_and_raw_input():
    calls = []

    @guarded_prediction_inputs
    def model(df, time_column, event_column, features, event_positive_value=1):
        calls.append(True)
        return {"comparison_table": [{"model": "Cox PH", "c_index": 0.5}]}

    df = cohort()
    df.loc[0, "patient_id"] = " NA "
    before = df.copy(deep=True)
    result = model(df, "time", "event", ["marker"])
    assert calls == [True]
    assert "only recorded IDs" in result["comparison_table"][0]["input_notes"]
    assert result["input_audit"]["version"] == "prediction-input/3"
    pd.testing.assert_frame_equal(df, before)


def test_id_comparison_does_not_align_missing_markers_by_repeated_index_labels():
    df = cohort()
    df.index = [0] * len(df)
    df.iloc[0, df.columns.get_loc("patient_id")] = " NA "
    before = df.copy(deep=True)
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    assert audit["identifier_checks"][0]["missing_identifier_rows"] == 1
    assert audit["identifier_checks"][0]["n_rows"] == 119
    pd.testing.assert_frame_equal(df, before)


def test_identical_complete_rows_without_ids_warn_and_remain_in_analysis():
    df = cohort().drop(columns="patient_id")
    df.iloc[1] = df.iloc[0]
    before = df.copy(deep=True)
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    check = audit["identical_input_rows"]
    assert check == {"status": "review_required", "scope": "all_input_columns", "matching_rows": 2,
                     "extra_rows": 1, "rows_removed": 0}
    assert audit["analyzed_rows"] == 120
    assert any("not proof of patient duplication" in note for note in audit["notes"])
    pd.testing.assert_frame_equal(df, before)


def test_identical_selected_columns_do_not_imply_identical_full_input_rows():
    df = cohort().drop(columns="patient_id")
    df.iloc[1] = df.iloc[0]
    df["unselected_baseline_value"] = np.arange(len(df))
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    assert audit["identical_input_rows"]["status"] == "no_identical_rows"


def test_identical_row_warning_does_not_hash_wide_unselected_omics_tables():
    df = cohort().drop(columns="patient_id")
    df = pd.concat([df, pd.DataFrame(np.zeros((len(df), 257)), columns=[f"GENE{i}" for i in range(257)])], axis=1)
    before = df.copy(deep=True)
    audit = prediction_input_audit(df, "time", "event", ["marker"], 1)
    assert audit["identical_input_rows"] == {"status": "not_assessed", "reason": "full_row_check_limit"}
    assert any("was not assessed" in note for note in audit["notes"])
    assert audit["analyzed_rows"] == 120
    pd.testing.assert_frame_equal(df, before)


def test_request_local_proxy_cache_distinguishes_requested_categorical_encoding(monkeypatch):
    from survival_toolkit import outcome_proxy

    calls = []

    def fingerprint(frame, categorical_features=()):
        return "encoding:" + ",".join(categorical_features)

    def screen(frame, time_column, event_column, features, categorical_features=()):
        calls.append(tuple(categorical_features))
        return {"status": "no_pattern_detected", "findings": [], "notes": [], "coverage": "screened"}

    monkeypatch.setattr(outcome_proxy, "training_fingerprint", fingerprint)
    monkeypatch.setattr(outcome_proxy, "screen_training_outcomes", screen)

    @guarded_prediction_inputs
    def model(df, time_column, event_column, features, event_positive_value=1):
        check_training_outcome_proxies(df, time_column, event_column, features, ())
        check_training_outcome_proxies(df, time_column, event_column, features, ["marker"])
        check_training_outcome_proxies(df, time_column, event_column, features, ["marker"])
        return {}

    result = model(cohort(), "time", "event", ["marker"])
    assert calls == [(), ("marker",)]
    assert len(result["input_audit"]["outcome_proxy"]["partitions"]) == 2


PUBLIC_MODEL_ROUTES = [
    ("ml_models", "train_random_survival_forest"),
    ("ml_models", "train_gradient_boosted_survival"),
    ("ml_models", "train_lasso_cox"),
    ("ml_models", "compare_survival_models"),
    ("ml_models", "cross_validate_survival_models"),
    ("deep_models", "compare_deep_survival_models"),
    ("deep_models", "train_deepsurv"),
    ("deep_models", "train_deephit"),
    ("deep_models", "train_neural_mtlr"),
    ("deep_models", "train_survival_transformer"),
    ("deep_models", "train_survival_vae"),
]


@pytest.mark.parametrize("module_name,function_name", PUBLIC_MODEL_ROUTES)
@pytest.mark.parametrize("defect", ["padded_duplicate", "other_outcome"])
def test_public_model_routes_stop_before_preprocessing_or_training(monkeypatch, module_name, function_name, defect):
    module = importlib.import_module("survival_toolkit." + module_name)
    function = getattr(module, function_name)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Invalid inputs reached model preparation")

    # These are shared preparation boundaries; no estimator is fitted in this test.
    for name in ("_prepare_training_matrices", "_prepare_model_evaluation_split", "_prepare_data"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)
    df = cohort()
    features = ["marker"]
    if defect == "padded_duplicate":
        df.loc[1, "patient_id"] = " " + df.loc[0, "patient_id"] + "\t"
        message = "Repeated patient IDs"
    else:
        df["dfs_status"] = [0, 1] * 60
        features = ["dfs_status"]
        message = "Survival outcome columns cannot"
    before = df.copy(deep=True)
    with pytest.raises(ValueError, match=message):
        function(df, "time", "event", features)
    assert calls == []
    pd.testing.assert_frame_equal(df, before)

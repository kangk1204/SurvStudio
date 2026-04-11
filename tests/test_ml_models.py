from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.sample_data import make_example_dataset


def _sksurv_available() -> bool:
    try:
        import sksurv  # noqa: F401
        return True
    except ImportError:
        return False


def test_find_optimal_cutpoint_returns_valid_split() -> None:
    from survival_toolkit.ml_models import find_optimal_cutpoint

    df = make_example_dataset(seed=10, n_patients=200)
    result = find_optimal_cutpoint(
        df,
        time_column="os_months",
        event_column="os_event",
        variable="biomarker_score",
        event_positive_value=1,
        min_group_fraction=0.1,
        permutation_iterations=25,
    )
    assert result["optimal_cutpoint"] is not None
    assert result["statistic"] > 0
    assert 0 < result["p_value"] <= 1
    assert result["raw_p_value"] is not None
    assert result["selection_adjusted_p_value"] is not None
    assert result["p_value"] == result["selection_adjusted_p_value"]
    assert result["p_value_label"] == "selection_adjusted_p_value"
    assert result["n_high"] > 0
    assert result["n_low"] > 0
    assert len(result["scan_data"]) > 2


def test_split_train_test_falls_back_to_apparent_when_stratified_split_raises(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=123, n_patients=30)

    def _boom(*args, **kwargs):
        raise ValueError("forced stratified split failure")

    monkeypatch.setattr(ml_models, "train_test_split", _boom)

    train_frame, test_frame, evaluation_mode = ml_models._split_train_test(
        df,
        event_column="os_event",
        random_state=7,
        test_size=0.3,
    )

    assert evaluation_mode == "apparent"
    assert len(train_frame) == len(df)
    assert len(test_frame) == len(df)


def test_optimal_cutpoint_derive_method() -> None:
    from survival_toolkit.analysis import derive_group_column

    df = make_example_dataset(seed=15, n_patients=200)
    updated, column_name, summary = derive_group_column(
        df,
        source_column="biomarker_score",
        method="optimal_cutpoint",
        new_column_name="optimal_group",
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
    )
    assert column_name == "optimal_group"
    assert "optimal_group" in updated.columns
    assert summary["method"] == "optimal_cutpoint"
    assert summary["cutoff"] is not None
    assert summary["p_value"] is not None
    assert summary["raw_p_value"] is not None
    assert summary["selection_adjusted_p_value"] is not None
    assert summary["p_value"] == summary["selection_adjusted_p_value"]
    assert summary["p_value_label"] == "selection_adjusted_p_value"
    assert summary["label_above_cutpoint"] in {"Low", "High"}
    assert summary["label_below_cutpoint"] in {"Low", "High"}
    assert "assignment_rule" in summary
    groups = set(updated["optimal_group"].dropna().unique())
    assert groups.issubset({"Low", "High"})


def test_find_optimal_cutpoint_can_return_split_series_for_internal_callers() -> None:
    from survival_toolkit.analysis import derive_group_column
    from survival_toolkit.ml_models import find_optimal_cutpoint

    df = make_example_dataset(seed=27, n_patients=180)
    result = find_optimal_cutpoint(
        df,
        time_column="os_months",
        event_column="os_event",
        variable="biomarker_score",
        event_positive_value=1,
        include_split_series=True,
        permutation_iterations=10,
    )
    split_series = result.get("split_series")

    assert isinstance(split_series, pd.Series)
    assert str(split_series.dtype) == "string"
    assert split_series.index.equals(df.index)

    updated, column_name, _ = derive_group_column(
        df,
        source_column="biomarker_score",
        method="optimal_cutpoint",
        new_column_name="optimal_group",
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        permutation_iterations=10,
    )

    assert column_name == "optimal_group"
    pd.testing.assert_series_equal(
        split_series.rename(column_name),
        updated[column_name],
        check_names=True,
    )


def test_find_optimal_cutpoint_excludes_invalid_permutation_resamples_from_denominator(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "marker": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        }
    )

    class _FixedRng:
        def permutation(self, values):
            return np.asarray(values, dtype=float)

    call_count = 0

    def _fake_survdiff(times, events, groups):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 5.0, 0.01
        if call_count == 2:
            return 4.0, 0.02
        if call_count in {3, 4}:
            return 4.0, 0.20
        raise ValueError("invalid permutation split")

    monkeypatch.setattr(ml_models, "survdiff", _fake_survdiff)
    monkeypatch.setattr(ml_models.np.random, "default_rng", lambda seed: _FixedRng())

    result = ml_models.find_optimal_cutpoint(
        df,
        time_column="time",
        event_column="event",
        variable="marker",
        event_positive_value=1,
        min_group_fraction=0.2,
        permutation_iterations=2,
        random_seed=7,
    )

    assert result["selection_adjustment"]["permutation_valid_resamples"] == 1
    assert result["selection_adjusted_p_value"] == pytest.approx(0.5)


def test_find_optimal_cutpoint_counts_only_valid_permutation_resamples(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=19, n_patients=120)
    observed_calls = {"count": 0}

    def _fake_survdiff(times, events, groups):
        observed_calls["count"] += 1
        if observed_calls["count"] == 1:
            return 5.0, 0.01
        raise ValueError("permutation split failed")

    monkeypatch.setattr(ml_models, "survdiff", _fake_survdiff)
    monkeypatch.setattr(ml_models, "_EXPECTED_CUTPOINT_SCAN_ERRORS", (ValueError,))

    result = ml_models.find_optimal_cutpoint(
        df,
        time_column="os_months",
        event_column="os_event",
        variable="biomarker_score",
        event_positive_value=1,
        min_group_fraction=0.1,
        permutation_iterations=3,
        random_seed=7,
    )

    assert result["selection_adjustment"]["permutation_valid_resamples"] == 0
    assert result["selection_adjusted_p_value"] is None
    assert result["p_value"] == result["raw_p_value"]


def test_scientific_summary_ml_filters_empty_extra_cautions() -> None:
    import survival_toolkit.ml_models as ml_models

    summary = ml_models._scientific_summary_ml(
        model_name="RSF",
        c_index=0.61,
        n_patients=80,
        n_events=40,
        n_features=2,
        extra_cautions=["screening only", None, ""],
    )

    assert "screening only" in summary["cautions"]
    assert None not in summary["cautions"]
    assert "" not in summary["cautions"]


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_rsf_reports_holdout_evaluation_on_large_cohort() -> None:
    from survival_toolkit.ml_models import train_random_survival_forest

    df = make_example_dataset(seed=20, n_patients=200)
    result = train_random_survival_forest(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        n_estimators=20,
        random_state=42,
    )
    assert result["model_stats"]["c_index"] is not None
    assert result["model_stats"]["ibs"] is not None
    assert result["model_stats"]["null_ibs"] is not None
    assert result["model_stats"]["brier_skill_score"] is not None
    assert result["model_stats"]["evaluation_mode"] == "holdout"
    assert result["model_stats"]["n_evaluation_patients"] < result["model_stats"]["n_patients"]
    assert "holdout" in result["scientific_summary"]["headline"].lower()
    assert len(result["feature_importance"]) == 3
    assert result["feature_importance"][0]["importance"] >= result["feature_importance"][-1]["importance"]
    assert any(metric["label"] == "Brier Skill Score" for metric in result["scientific_summary"]["metrics"])


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_gbs_reports_holdout_evaluation_on_large_cohort() -> None:
    from survival_toolkit.ml_models import train_gradient_boosted_survival

    df = make_example_dataset(seed=25, n_patients=200)
    result = train_gradient_boosted_survival(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        n_estimators=20,
        learning_rate=0.1,
        random_state=42,
    )
    assert result["model_stats"]["c_index"] is not None
    assert result["model_stats"]["ibs"] is not None
    assert result["model_stats"]["null_ibs"] is not None
    assert result["model_stats"]["brier_skill_score"] is not None
    assert result["model_stats"]["evaluation_mode"] == "holdout"
    assert result["model_stats"]["n_evaluation_patients"] < result["model_stats"]["n_patients"]
    assert "holdout" in result["scientific_summary"]["headline"].lower()
    assert len(result["feature_importance"]) == 3
    assert any(metric["label"] == "Brier Skill Score" for metric in result["scientific_summary"]["metrics"])


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_lasso_cox_reports_holdout_evaluation_on_large_cohort() -> None:
    from survival_toolkit.ml_models import train_lasso_cox

    df = make_example_dataset(seed=24, n_patients=200)
    result = train_lasso_cox(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        random_state=42,
    )
    assert result["model_stats"]["c_index"] is not None
    assert result["model_stats"]["ibs"] is not None
    assert result["model_stats"]["null_ibs"] is not None
    assert result["model_stats"]["brier_skill_score"] is not None
    assert result["model_stats"]["evaluation_mode"] == "holdout"
    assert result["model_stats"]["n_evaluation_patients"] < result["model_stats"]["n_patients"]
    assert result["model_stats"]["alpha"] is not None
    assert result["model_stats"]["n_active_features"] >= 1
    assert "LASSO-Cox" in result["scientific_summary"]["headline"]
    assert any(metric["label"] == "Brier Skill Score" for metric in result["scientific_summary"]["metrics"])


def test_select_lasso_alpha_prefers_sparser_model_within_one_se(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    class _FakeCoxnet:
        def __init__(self) -> None:
            self.alphas_ = np.asarray([0.1, 1.0], dtype=float)
            self.coef_ = np.asarray(
                [
                    [1.0, 1.0],
                    [0.5, 0.0],
                ],
                dtype=float,
            )

        def fit(self, X, y) -> "_FakeCoxnet":
            return self

        def predict(self, X, alpha: float) -> np.ndarray:
            return np.full(X.shape[0], 0.70 if float(alpha) < 0.5 else 0.69, dtype=float)

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_make_lasso_coxnet_model", lambda alpha=None: _FakeCoxnet())
    monkeypatch.setattr(
        ml_models,
        "_estimate_c_index_standard_error",
        lambda y_eval, risk_scores, random_state, n_bootstrap=30: 0.02,
    )
    monkeypatch.setattr(ml_models, "_sksurv_c_index", lambda y_eval, risk_scores: float(risk_scores[0]))

    df = pd.DataFrame(
        {
            "os_months": np.linspace(1.0, 40.0, 40),
            "os_event": [0, 1] * 20,
        }
    )
    encoded = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, 1.0, 40),
            "f2": np.linspace(1.0, -1.0, 40),
        }
    )

    result = ml_models._select_lasso_alpha(
        df,
        encoded,
        time_column="os_months",
        event_column="os_event",
        random_state=11,
    )

    assert result["alpha"] == pytest.approx(1.0)
    assert result["selection_rule"] == "one_se_bootstrap"
    assert result["n_nonzero_features"] == 1


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
@pytest.mark.parametrize(
    "trainer,kwargs",
    [
        (
            "train_random_survival_forest",
            {"n_estimators": 10, "random_state": 7},
        ),
        (
            "train_gradient_boosted_survival",
            {"n_estimators": 10, "learning_rate": 0.1, "random_state": 7},
        ),
    ],
)
def test_single_model_reports_apparent_mode_for_small_cohort(trainer: str, kwargs: dict[str, object]) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=35, n_patients=16)
    trainer_fn = getattr(ml_models, trainer)
    result = trainer_fn(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        **kwargs,
    )
    assert result["model_stats"]["evaluation_mode"] == "apparent"
    assert result["model_stats"]["n_evaluation_patients"] == result["model_stats"]["n_patients"]
    assert "apparent" in result["scientific_summary"]["headline"].lower()


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_compare_models_returns_table() -> None:
    from survival_toolkit.ml_models import compare_survival_models

    df = make_example_dataset(seed=30, n_patients=200)
    result = compare_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score"],
    )
    assert "comparison_table" in result
    assert result["evaluation_mode"] in {"holdout", "apparent"}
    models = {row["model"] for row in result["comparison_table"]}
    assert "Cox PH" in models
    assert "LASSO-Cox" in models
    assert len(result["comparison_table"]) >= 1
    assert all("evaluation_mode" in row for row in result["comparison_table"])
    assert all("brier_skill_score" in row for row in result["comparison_table"])
    assert "manuscript_tables" in result
    assert result["manuscript_tables"]["model_performance_table"]
    assert "Brier Skill Score" in result["manuscript_tables"]["model_performance_table"][0]


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_compare_models_handles_common_categorical_presets_without_cox_singularity() -> None:
    from survival_toolkit.ml_models import compare_survival_models

    df = make_example_dataset(seed=202, n_patients=200)
    result = compare_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "sex", "stage", "treatment", "biomarker_score"],
        categorical_features=["sex", "stage", "treatment"],
        n_estimators=30,
        max_depth=4,
        learning_rate=0.08,
        random_state=13,
    )

    assert {row["model"] for row in result["comparison_table"]} == {
        "Cox PH",
        "LASSO-Cox",
        "Random Survival Forest",
        "Gradient Boosted Survival",
    }
    assert result["errors"] == []


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_compare_models_keeps_cox_ph_on_tcga_xena_wide_feature_screen() -> None:
    from survival_toolkit.ml_models import compare_survival_models

    dataset_path = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "data" / "tcga_luad_xena_example.csv"
    df = pd.read_csv(dataset_path)
    features = [
        "age",
        "sex",
        "pathologic_stage",
        "stage_group",
        "smoking_status",
        "pack_years_smoked",
        "tumor_longest_dimension_cm",
        "histology",
        "kras_status",
        "egfr_status",
        "expression_subtype",
    ]
    categorical_features = [
        "sex",
        "pathologic_stage",
        "stage_group",
        "smoking_status",
        "histology",
        "kras_status",
        "egfr_status",
        "expression_subtype",
    ]

    result = compare_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=features,
        categorical_features=categorical_features,
        event_positive_value=1,
        n_estimators=30,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )

    assert "Cox PH" in {row["model"] for row in result["comparison_table"]}
    assert not any(error["model"] == "Cox PH" for error in result["errors"])


def test_compare_models_passes_requested_hyperparameters(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=32, n_patients=80)
    seen: list[tuple[str, object, object, object]] = []

    def _fake_cox(*args, **kwargs):
        return {"c_index": 0.61, "n_features": 2, "training_time_ms": 1.0}

    def _fake_rsf(*args, **kwargs):
        seen.append(("rsf", kwargs.get("n_estimators"), kwargs.get("max_depth"), kwargs.get("random_state")))
        return {"c_index": 0.62, "n_features": 2, "training_time_ms": 1.0}

    def _fake_gbs(*args, **kwargs):
        seen.append(
            ("gbs", kwargs.get("n_estimators"), kwargs.get("learning_rate"), kwargs.get("random_state"))
        )
        return {"c_index": 0.63, "n_features": 2, "training_time_ms": 1.0}

    def _fake_lasso(*args, **kwargs):
        seen.append(("lasso", kwargs.get("random_state"), kwargs.get("time_column"), kwargs.get("event_column")))
        return {"c_index": 0.64, "n_features": 2, "training_time_ms": 1.0}

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _fake_cox)
    monkeypatch.setattr(ml_models, "_fit_evaluate_lasso_cox_split", _fake_lasso)
    monkeypatch.setattr(ml_models, "_fit_evaluate_rsf_split", _fake_rsf)
    monkeypatch.setattr(ml_models, "_fit_evaluate_gbs_split", _fake_gbs)

    result = ml_models.compare_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score"],
        n_estimators=37,
        max_depth=5,
        learning_rate=0.2,
        random_state=99,
    )

    assert result["comparison_table"]
    assert ("lasso", 99, "os_months", "os_event") in seen
    assert ("rsf", 37, 5, 99) in seen
    assert ("gbs", 37, 0.2, 99) in seen


def test_fit_evaluate_cox_split_rejects_nonfinite_estimates(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    frame = make_example_dataset(seed=321, n_patients=80).loc[:, ["os_months", "os_event", "age", "biomarker_score"]]
    train = frame.iloc[:50].reset_index(drop=True)
    test = frame.iloc[50:].reset_index(drop=True)

    class _FakeResults:
        params = np.array([np.nan, 0.25], dtype=float)
        llf = -12.3

    class _FakePHReg:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, disp: bool = False):
            return _FakeResults()

    monkeypatch.setattr(ml_models, "PHReg", _FakePHReg)

    with pytest.raises(ValueError, match="non-finite estimates"):
        ml_models._fit_evaluate_cox_split(
            train,
            test,
            time_column="os_months",
            event_column="os_event",
            features=["age", "biomarker_score"],
        )


def test_compare_models_excludes_cox_when_shared_holdout_fit_is_nonfinite(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=322, n_patients=120).loc[:, ["os_months", "os_event", "age", "biomarker_score"]]

    class _FakeResults:
        params = np.array([np.nan, 0.25], dtype=float)
        llf = -9.5

    class _FakePHReg:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, disp: bool = False):
            return _FakeResults()

    def _fake_rsf(*args, **kwargs):
        return {"c_index": 0.62, "n_features": 2, "training_time_ms": 1.0}

    def _fake_gbs(*args, **kwargs):
        return {"c_index": 0.64, "n_features": 2, "training_time_ms": 1.0}

    def _fake_lasso(*args, **kwargs):
        return {"c_index": 0.63, "n_features": 2, "training_time_ms": 1.0}

    monkeypatch.setattr(ml_models, "PHReg", _FakePHReg)
    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_lasso_cox_split", _fake_lasso)
    monkeypatch.setattr(ml_models, "_fit_evaluate_rsf_split", _fake_rsf)
    monkeypatch.setattr(ml_models, "_fit_evaluate_gbs_split", _fake_gbs)

    result = ml_models.compare_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score"],
        random_state=17,
    )

    assert {row["model"] for row in result["comparison_table"]} == {
        "LASSO-Cox",
        "Random Survival Forest",
        "Gradient Boosted Survival",
    }
    assert any(error["model"] == "Cox PH" for error in result["errors"])


def test_feature_encoder_imputes_missing_numeric_and_categorical_values() -> None:
    import survival_toolkit.ml_models as ml_models

    frame = make_example_dataset(seed=77, n_patients=40).loc[:, ["age", "sex"]].copy()
    frame.loc[0, "age"] = np.nan
    frame.loc[1, "sex"] = pd.NA

    encoder = ml_models._fit_feature_encoder(frame, ["age", "sex"], categorical_features=["sex"])
    encoded = ml_models._transform_feature_encoder(frame, encoder)

    assert encoded.shape[0] == frame.shape[0]
    assert not encoded.isna().any().any()
    assert "sex__missing" in encoded.columns
    assert encoded.loc[1, "sex__missing"] == pytest.approx(1.0)
    assert encoded.loc[0, "age"] == pytest.approx(float(encoder["numeric_impute_values"]["age"]))


def test_build_manuscript_result_tables_handles_incomplete_repeated_cv() -> None:
    from survival_toolkit.ml_models import build_manuscript_result_tables

    manuscript = build_manuscript_result_tables(
        {
            "evaluation_mode": "repeated_cv_incomplete",
            "n_patients": 120,
            "n_events": 58,
            "comparison_table": [
                {
                    "model": "Survival Transformer",
                    "c_index": None,
                    "evaluation_mode": "repeated_cv_incomplete",
                    "cv_folds": 5,
                    "cv_repeats": 3,
                    "n_repeats": 3,
                    "n_evaluations": 13,
                    "n_failures": 2,
                    "n_apparent_fallbacks": 2,
                    "n_features": 10,
                    "training_samples": 96,
                    "train_events": 46,
                    "evaluation_samples": 24,
                    "test_events": 12,
                    "training_time_ms": 123.4,
                    "training_seeds": [42, 43, 44],
                    "split_seeds": [42, 43, 44],
                    "monitor_seeds": [42, 43, 44],
                }
            ],
        }
    )

    assert "incomplete repeated stratified cross-validation" in manuscript["caption"].lower()
    assert any("repeated-cv incomplete" in note.lower() for note in manuscript["table_notes"])
    first_row = manuscript["model_performance_table"][0]
    assert first_row["Validation Strategy"].endswith("(incomplete)")
    assert first_row["Patients, n"] == 120
    assert first_row["Events, n"] == 58
    assert first_row["Mean Evaluation Patients, n"] == 24
    assert first_row["Mean Evaluation Events, n"] == 12


def test_build_manuscript_result_tables_humanizes_holdout_fallback_validation_strategy() -> None:
    from survival_toolkit.ml_models import build_manuscript_result_tables

    manuscript = build_manuscript_result_tables(
        {
            "evaluation_mode": "mixed_holdout_apparent",
            "n_patients": 150,
            "n_events": 63,
            "comparison_table": [
                {
                    "model": "Random Survival Forest",
                    "rank": 1,
                    "c_index": 0.68,
                    "evaluation_mode": "holdout",
                    "n_features": 8,
                    "evaluation_samples": 30,
                    "test_events": 13,
                    "training_time_ms": 52.4,
                },
                {
                    "model": "Gradient Boosted Survival",
                    "rank": None,
                    "c_index": 0.61,
                    "evaluation_mode": "holdout_fallback_apparent",
                    "n_features": 8,
                    "evaluation_samples": 150,
                    "test_events": 63,
                    "training_time_ms": 61.7,
                },
            ],
        }
    )

    rows = manuscript["model_performance_table"]
    assert rows[0]["Validation Strategy"] == "Deterministic holdout"
    assert rows[1]["Validation Strategy"] == "Apparent fallback after holdout failure"
    assert "mixed holdout/apparent" in manuscript["caption"].lower()


def test_prepare_model_evaluation_split_keeps_outcome_valid_rows_with_missing_features() -> None:
    import survival_toolkit.ml_models as ml_models

    frame = make_example_dataset(seed=78, n_patients=80).loc[:, ["os_months", "os_event", "age", "sex"]].copy()
    frame.loc[:9, "age"] = np.nan
    frame.loc[10:19, "sex"] = pd.NA

    split = ml_models._prepare_model_evaluation_split(
        frame,
        time_column="os_months",
        event_column="os_event",
        features=["age", "sex"],
        categorical_features=["sex"],
        random_state=19,
    )

    assert split["full_frame"].shape[0] == frame.shape[0]
    assert split["train_frame"].shape[0] + split["eval_frame"].shape[0] == frame.shape[0]
    assert not split["train_encoded"].isna().any().any()
    assert not split["eval_encoded"].isna().any().any()


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_cross_validated_compare_models_returns_manuscript_tables() -> None:
    from survival_toolkit.ml_models import cross_validate_survival_models

    df = make_example_dataset(seed=31, n_patients=180)
    result = cross_validate_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        cv_folds=3,
        cv_repeats=2,
        random_state=11,
    )
    assert result["evaluation_mode"] == "repeated_cv"
    assert result["cv_folds"] == 3
    assert result["cv_repeats"] == 2
    assert result["fold_results"]
    assert result["repeat_results"]
    assert result["manuscript_tables"]["model_performance_table"]
    assert "repeated" in result["manuscript_tables"]["caption"].lower()
    assert "LASSO-Cox" in {row["model"] for row in result["comparison_table"]}
    assert all(row["evaluation_mode"] == "repeated_cv" for row in result["comparison_table"])
    assert all("c_index_std" in row for row in result["comparison_table"])
    assert all(row["n_repeats"] == 2 for row in result["comparison_table"])
    assert all(row["Repeat means, n"] == 2 for row in result["manuscript_tables"]["model_performance_table"])
    assert all(
        "Empirical repeat interval (repeat means)" in row
        for row in result["manuscript_tables"]["model_performance_table"]
    )


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_cross_validated_compare_models_handles_common_categorical_presets_without_cox_singularity() -> None:
    from survival_toolkit.ml_models import cross_validate_survival_models

    df = make_example_dataset(seed=203, n_patients=180)
    result = cross_validate_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "sex", "stage", "treatment", "biomarker_score"],
        categorical_features=["sex", "stage", "treatment"],
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        cv_folds=2,
        cv_repeats=2,
        random_state=5,
    )

    assert {row["model"] for row in result["comparison_table"]} == {
        "Cox PH",
        "LASSO-Cox",
        "Random Survival Forest",
        "Gradient Boosted Survival",
    }
    assert result["errors"] == []


def test_cross_validated_compare_models_pass_requested_hyperparameters(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=33, n_patients=80)
    seen: list[tuple[str, object, object, object]] = []

    def _fake_cox(*args, **kwargs):
        return {
            "c_index": 0.6,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    def _fake_rsf(*args, **kwargs):
        seen.append(("rsf", kwargs.get("n_estimators"), kwargs.get("max_depth"), kwargs.get("random_state")))
        return {
            "c_index": 0.61,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    def _fake_gbs(*args, **kwargs):
        seen.append(
            ("gbs", kwargs.get("n_estimators"), kwargs.get("learning_rate"), kwargs.get("random_state"))
        )
        return {
            "c_index": 0.62,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    def _fake_lasso(*args, **kwargs):
        seen.append(("lasso", kwargs.get("random_state"), kwargs.get("time_column"), kwargs.get("event_column")))
        return {
            "c_index": 0.615,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _fake_cox)
    monkeypatch.setattr(ml_models, "_fit_evaluate_lasso_cox_split", _fake_lasso)
    monkeypatch.setattr(ml_models, "_fit_evaluate_rsf_split", _fake_rsf)
    monkeypatch.setattr(ml_models, "_fit_evaluate_gbs_split", _fake_gbs)

    result = ml_models.cross_validate_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score"],
        n_estimators=41,
        max_depth=7,
        learning_rate=0.15,
        cv_folds=2,
        cv_repeats=1,
        random_state=13,
    )

    assert result["comparison_table"]
    assert ("lasso", 13, "os_months", "os_event") in seen
    assert ("rsf", 41, 7, 13) in seen
    assert ("gbs", 41, 0.15, 13) in seen


def test_cross_validated_compare_models_blanks_aggregate_if_any_fold_fails(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=34, n_patients=80)
    rsf_calls = {"count": 0}

    def _fake_cox(*args, **kwargs):
        return {
            "c_index": 0.6,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    def _fake_rsf(*args, **kwargs):
        rsf_calls["count"] += 1
        if rsf_calls["count"] == 1:
            raise RuntimeError("simulated fold failure")
        return {
            "c_index": 0.61,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    def _fake_gbs(*args, **kwargs):
        return {
            "c_index": 0.62,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    def _fake_lasso(*args, **kwargs):
        return {
            "c_index": 0.605,
            "n_features": 2,
            "training_time_ms": 1.0,
            "train_n": 40,
            "test_n": 40,
            "train_events": 10,
            "test_events": 10,
        }

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _fake_cox)
    monkeypatch.setattr(ml_models, "_fit_evaluate_lasso_cox_split", _fake_lasso)
    monkeypatch.setattr(ml_models, "_fit_evaluate_rsf_split", _fake_rsf)
    monkeypatch.setattr(ml_models, "_fit_evaluate_gbs_split", _fake_gbs)

    result = ml_models.cross_validate_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score"],
        cv_folds=2,
        cv_repeats=1,
        random_state=13,
    )

    rsf_row = next(row for row in result["comparison_table"] if row["model"] == "Random Survival Forest")
    assert rsf_row["n_failures"] == 1
    assert rsf_row["c_index"] is None
    manuscript_row = next(
        row
        for row in result["manuscript_tables"]["model_performance_table"]
        if row["Model"] == "Random Survival Forest"
    )
    assert manuscript_row["Mean C-index"] is None


def test_encode_train_test_features_preserves_unknown_bucket_for_unseen_levels() -> None:
    import pandas as pd

    from survival_toolkit.ml_models import _encode_train_test_features

    train = pd.DataFrame(
        {
            "stage": ["I", "I", "II"],
            "age": [50, 60, 70],
        }
    )
    test = pd.DataFrame(
        {
            "stage": ["III", "I"],
            "age": [55, 65],
        }
    )

    train_encoded, test_encoded, _ = _encode_train_test_features(
        train,
        test,
        features=["stage", "age"],
        categorical_features=["stage"],
    )

    assert "stage__unknown" in train_encoded.columns
    assert float(train_encoded["stage__unknown"].sum()) == pytest.approx(0.0)
    assert float(test_encoded.iloc[0]["stage__unknown"]) == pytest.approx(1.0)
    assert float(test_encoded.iloc[1]["stage__unknown"]) == pytest.approx(0.0)


def test_time_dependent_importance_returns_time_major_matrix() -> None:
    from survival_toolkit.ml_models import compute_time_dependent_importance

    df = make_example_dataset(seed=45, n_patients=160)
    result = compute_time_dependent_importance(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        eval_times=[12.0, 24.0, 36.0],
    )
    assert result["importance_matrix_orientation"] == "time_major"
    assert len(result["importance_matrix"]) == len(result["eval_times"])
    assert len(result["importance_matrix"][0]) == len(result["features"])
    assert len(result["importance_matrix_feature_major"]) == len(result["features"])


def test_counterfactual_survival_handles_zero_baseline_risk(monkeypatch) -> None:
    import numpy as np
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    class _DummyModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            return np.zeros(X.shape[0]) if self.calls == 1 else np.ones(X.shape[0])

    dummy_model = _DummyModel()

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "train_random_survival_forest",
        lambda *args, **kwargs: {
            "_model": dummy_model,
            "_X_encoded": pd.DataFrame({"age": [0.0, 0.0, 0.0]}),
        },
    )

    df = make_example_dataset(seed=51, n_patients=40)
    result = ml_models.counterfactual_survival(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age"],
        target_feature="age",
        original_value=float(df["age"].median()),
        counterfactual_value=float(df["age"].median() + 5.0),
        model_type="rsf",
    )

    assert result["risk_change_pct"] is None
    assert "undefined" in result["scientific_summary"]["headline"].lower()
    assert "causal" not in result["scientific_summary"]["headline"].lower()


def test_counterfactual_survival_uses_original_value_for_baseline_scenario(monkeypatch) -> None:
    import numpy as np
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    class _LinearAgeModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "train_random_survival_forest",
        lambda *args, **kwargs: {
            "_model": _LinearAgeModel(),
            "_X_encoded": pd.DataFrame({"age": analysis_frame["age"]}),
            "_analysis_frame": analysis_frame.copy(),
        },
    )

    df = make_example_dataset(seed=63, n_patients=40)
    result = ml_models.counterfactual_survival(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age"],
        target_feature="age",
        original_value=5.0,
        counterfactual_value=15.0,
        model_type="rsf",
    )

    assert result["original_median_risk"] == pytest.approx(5.0)
    assert result["counterfactual_median_risk"] == pytest.approx(15.0)
    assert result["risk_change_pct"] == pytest.approx(200.0)
    assert "from 5.0 to 15.0" in result["scientific_summary"]["headline"]
    assert "model-based scenario" in result["scientific_summary"]["headline"].lower()
    assert any("not a causal estimate" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_counterfactual_survival_rejects_unknown_model_type(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    df = make_example_dataset(seed=164, n_patients=40)

    with pytest.raises(ValueError, match="Unsupported counterfactual model_type"):
        ml_models.counterfactual_survival(
            df,
            time_column="os_months",
            event_column="os_event",
            features=["age"],
            target_feature="age",
            original_value=5.0,
            counterfactual_value=15.0,
            model_type="bogus",
        )


def test_compute_calibration_data_uses_descriptive_language() -> None:
    from survival_toolkit.ml_models import compute_calibration_data

    result = compute_calibration_data(
        times=[1.0, 2.0, 3.0, 4.0],
        events=[1, 0, 1, 0],
        predicted_survival_at_t=[0.9, 0.8, 0.7, 0.6],
        t=2.5,
        n_bins=2,
    )
    summary = result["scientific_summary"]
    assert "heuristic" in summary["headline"].lower()
    assert any("formal calibration test" in strength.lower() for strength in summary["strengths"])
    assert any("descriptive calibration check" in caution.lower() for caution in summary["cautions"])


@pytest.mark.skipif(
    not _sksurv_available(),
    reason="scikit-survival not installed",
)
def test_partial_dependence_supports_categorical_raw_feature() -> None:
    from survival_toolkit.ml_models import compute_partial_dependence, train_random_survival_forest

    df = make_example_dataset(seed=46, n_patients=160)
    result = train_random_survival_forest(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "stage"],
        categorical_features=["stage"],
        n_estimators=20,
        random_state=42,
    )
    pdp = compute_partial_dependence(
        result["_model"],
        result["_X_encoded"],
        feature_name="stage",
        categorical_features=["stage"],
        feature_encoder=result["_feature_encoder"],
        analysis_frame=result["_analysis_frame"],
    )

    assert pdp["feature"] == "stage"
    assert pdp["feature_type"] == "categorical"
    assert len(pdp["values"]) >= 2
    assert len(pdp["values"]) == len(pdp["mean_risk"])
    assert all(isinstance(value, str) for value in pdp["values"])


def test_partial_dependence_preserves_observed_categorical_order() -> None:
    import pandas as pd

    from survival_toolkit.ml_models import compute_partial_dependence

    class _OrdinalModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"stage": pd.Series(["Stage I", "Stage III", "Stage II"], dtype="string")})
    encoded = pd.DataFrame({"stage": [0.0, 1.0, 2.0]})

    pdp = compute_partial_dependence(
        _OrdinalModel(),
        encoded,
        feature_name="stage",
        categorical_features=["stage"],
        analysis_frame=analysis_frame,
    )

    assert pdp["values"] == ["Stage I", "Stage III", "Stage II"]


def test_partial_dependence_raises_on_internal_prediction_failure() -> None:
    import pandas as pd

    from survival_toolkit.ml_models import compute_partial_dependence

    class _BrokenModel:
        def predict(self, X):
            raise RuntimeError("broken predictor")

    with pytest.raises(ValueError, match="Partial dependence failed"):
        compute_partial_dependence(
            _BrokenModel(),
            pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            feature_name="age",
        )


def test_partial_dependence_requires_predict_callable() -> None:
    from survival_toolkit.ml_models import compute_partial_dependence

    with pytest.raises(ValueError, match="callable predict\\(\\)"):
        compute_partial_dependence(
            object(),
            pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            feature_name="age",
        )


def test_partial_dependence_rejects_n_points_below_two() -> None:
    from survival_toolkit.ml_models import compute_partial_dependence

    class _LinearModel:
        def predict(self, X):
            return np.asarray(X)[:, 0]

    with pytest.raises(ValueError, match="n_points must be at least 2"):
        compute_partial_dependence(
            _LinearModel(),
            pd.DataFrame({"age": [10.0, 20.0, 30.0]}),
            feature_name="age",
            n_points=1,
        )


def test_store_lru_eviction() -> None:
    from survival_toolkit.store import DatasetStore
    from survival_toolkit.errors import DatasetNotFoundError

    store = DatasetStore(max_datasets=3, ttl_seconds=9999)
    df = make_example_dataset(seed=1, n_patients=30)
    ids = []
    for i in range(5):
        stored = store.create(df, filename=f"test_{i}")
        ids.append(stored.dataset_id)
    assert store.count == 3
    with pytest.raises(DatasetNotFoundError):
        store.get(ids[0])
    store.get(ids[-1])


def test_store_create_and_get_are_defensive_against_shared_mutation() -> None:
    from survival_toolkit.store import DatasetStore

    original = make_example_dataset(seed=4, n_patients=30)
    store = DatasetStore()
    stored = store.create(original, filename="shared.csv")

    original.loc[:, "age"] = -1
    fetched = store.get(stored.dataset_id)
    fetched.dataframe.loc[:, "age"] = -2
    fetched.metadata["flag"] = True

    refetched = store.get(stored.dataset_id)

    assert (refetched.dataframe["age"] >= 0).all()
    assert "flag" not in refetched.metadata


def test_store_create_and_get_are_defensive_against_scalar_cell_mutation() -> None:
    from survival_toolkit.store import DatasetStore

    original = make_example_dataset(seed=5, n_patients=30)
    original_age = float(original.iloc[0]["age"])
    store = DatasetStore()
    stored = store.create(original, filename="shared_scalar.csv")

    original.iat[0, original.columns.get_loc("age")] = -999.0

    refetched = store.get(stored.dataset_id)
    assert float(refetched.dataframe.iloc[0]["age"]) == pytest.approx(original_age)


def test_store_uses_idle_ttl_and_records_dataset_hash() -> None:
    from survival_toolkit.store import DatasetStore

    store = DatasetStore(ttl_seconds=1)
    stored = store.create(make_example_dataset(seed=17, n_patients=24), filename="idle_ttl.csv")

    assert isinstance(stored.metadata.get("dataset_hash"), str)
    assert len(str(stored.metadata["dataset_hash"])) == 16

    internal = store._datasets[stored.dataset_id]
    internal.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    internal.last_accessed = datetime.now(timezone.utc)

    refetched = store.get(stored.dataset_id)
    assert refetched.dataset_id == stored.dataset_id
    assert refetched.metadata["dataset_hash"] == stored.metadata["dataset_hash"]


def test_store_delete() -> None:
    from survival_toolkit.store import DatasetStore
    from survival_toolkit.errors import DatasetNotFoundError

    store = DatasetStore()
    df = make_example_dataset(seed=2, n_patients=30)
    stored = store.create(df, filename="to_delete")
    store.delete(stored.dataset_id)
    with pytest.raises(DatasetNotFoundError):
        store.get(stored.dataset_id)


def test_store_update_dataframe_copies_input_by_default() -> None:
    from survival_toolkit.store import DatasetStore

    store = DatasetStore()
    stored = store.create(make_example_dataset(seed=6, n_patients=30), filename="baseline.csv")
    replacement = make_example_dataset(seed=7, n_patients=30)

    store.update_dataframe(stored.dataset_id, replacement)
    replacement.loc[:, "age"] = -5

    refetched = store.get(stored.dataset_id)
    assert (refetched.dataframe["age"] >= 0).all()


def test_compute_shap_values_caps_kernel_fallback_work(monkeypatch) -> None:
    import types
    import numpy as np
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    seen: dict[str, object] = {}

    class _FakeTreeExplainer:
        def __init__(self, model) -> None:
            raise RuntimeError("tree unsupported")

    class _FakeKernelExplainer:
        def __init__(self, predict_fn, background) -> None:
            seen["background_shape"] = background.shape

        def shap_values(self, eval_matrix, nsamples=None, silent=None):
            seen["eval_shape"] = eval_matrix.shape
            seen["nsamples"] = nsamples
            seen["silent"] = silent
            return np.ones((eval_matrix.shape[0], eval_matrix.shape[1]), dtype=float)

    monkeypatch.setattr(ml_models, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "shap",
        types.SimpleNamespace(TreeExplainer=_FakeTreeExplainer, KernelExplainer=_FakeKernelExplainer),
    )

    encoded = pd.DataFrame(np.arange(240 * 5, dtype=float).reshape(240, 5), columns=list("abcde"))
    model = types.SimpleNamespace(predict=lambda matrix: np.asarray(matrix).sum(axis=1))
    result = ml_models.compute_shap_values(model, encoded, feature_names=list(encoded.columns))

    assert result["method"] == "kernel"
    assert result["stability"] == "approximate_screening_only"
    assert "screening" in result["usage_note"].lower()
    assert result["background_samples"] == 40
    assert result["n_samples"] == 60
    assert seen["background_shape"] == (40, 5)
    assert seen["eval_shape"] == (60, 5)
    assert seen["nsamples"] == 40
    assert seen["silent"] is True


def test_compute_shap_values_rejects_high_dimensional_kernel_fallback(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    class _FakeTreeExplainer:
        def __init__(self, model) -> None:
            raise RuntimeError("tree unsupported")

    monkeypatch.setattr(ml_models, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "shap",
        types.SimpleNamespace(TreeExplainer=_FakeTreeExplainer, KernelExplainer=object),
    )

    encoded = pd.DataFrame(np.arange(120 * 81, dtype=float).reshape(120, 81), columns=[f"f{i}" for i in range(81)])
    model = types.SimpleNamespace(predict=lambda matrix: np.asarray(matrix).sum(axis=1))

    with pytest.raises(ValueError, match="disabled for high-dimensional inputs"):
        ml_models.compute_shap_values(model, encoded, feature_names=list(encoded.columns))


def test_compute_shap_values_accepts_list_output_from_older_shap(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    class _FakeTreeExplainer:
        def __init__(self, model) -> None:
            self.model = model

        def shap_values(self, matrix):
            arr = np.asarray(matrix, dtype=float)
            return [
                np.zeros_like(arr),
                np.ones_like(arr),
            ]

    monkeypatch.setattr(ml_models, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "shap",
        types.SimpleNamespace(TreeExplainer=_FakeTreeExplainer, KernelExplainer=object),
    )

    encoded = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3), columns=["a", "b", "c"])
    model = types.SimpleNamespace(predict=lambda matrix: np.asarray(matrix).sum(axis=1))

    result = ml_models.compute_shap_values(model, encoded, feature_names=list(encoded.columns))

    assert result["method"] == "tree"
    assert result["feature_importance"][0]["feature"] == "a"
    assert result["shap_summary"]


def test_compute_shap_values_requires_predict_callable(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    monkeypatch.setattr(ml_models, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "shap",
        types.SimpleNamespace(TreeExplainer=lambda model: (_ for _ in ()).throw(RuntimeError("tree unsupported")), KernelExplainer=object),
    )

    encoded = pd.DataFrame(np.arange(6, dtype=float).reshape(2, 3), columns=["a", "b", "c"])

    with pytest.raises(ValueError, match="callable predict\\(\\)"):
        ml_models.compute_shap_values(object(), encoded, feature_names=list(encoded.columns))


def test_rsf_permutation_importance_caps_rows_and_repeats(monkeypatch) -> None:
    import numpy as np
    import pandas as pd
    import survival_toolkit.ml_models as ml_models
    import sklearn.inspection

    seen: dict[str, object] = {}

    train_encoded = pd.DataFrame(
        np.linspace(0.0, 1.0, 160 * 24, dtype=float).reshape(160, 24),
        columns=[f"f{i}" for i in range(24)],
    )
    eval_encoded = pd.DataFrame(
        np.linspace(1.0, 2.0, 180 * 24, dtype=float).reshape(180, 24),
        columns=[f"f{i}" for i in range(24)],
    )
    full_encoded = pd.concat([train_encoded, eval_encoded], ignore_index=True)
    frame = pd.DataFrame({
        "os_months": np.linspace(1, 340, 340),
        "os_event": np.tile([0, 1], 170),
    })
    train_frame = frame.iloc[:160].reset_index(drop=True)
    eval_frame = frame.iloc[160:].reset_index(drop=True)
    full_frame = frame.reset_index(drop=True)
    y_train = np.zeros(train_frame.shape[0], dtype=float)
    y_eval = np.zeros(eval_frame.shape[0], dtype=float)

    class _DummyRSF:
        def __init__(self, **kwargs) -> None:
            pass

        @property
        def feature_importances_(self):
            raise NotImplementedError

        def fit(self, X, y):
            seen["fit_shape"] = X.shape
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=float)

    def _fake_perm(model, X, y, n_repeats, random_state, n_jobs):
        seen["perm_shape"] = X.shape
        seen["perm_y_shape"] = y.shape
        seen["n_repeats"] = n_repeats
        return type("PermResult", (), {"importances_mean": np.zeros(X.shape[1], dtype=float)})()

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "RandomSurvivalForest", _DummyRSF)
    monkeypatch.setattr(ml_models, "_prepare_model_evaluation_split", lambda *args, **kwargs: {
        "train_frame": train_frame,
        "eval_frame": eval_frame,
        "full_frame": full_frame,
        "train_encoded": train_encoded,
        "eval_encoded": eval_encoded,
        "full_encoded": full_encoded,
        "evaluation_mode": "holdout",
        "metric_name": "Holdout C-index",
        "feature_encoder": None,
    })
    monkeypatch.setattr(ml_models, "_prepare_sksurv_data", lambda frame, time_column, event_column: y_train if len(frame) == len(train_frame) else y_eval)
    monkeypatch.setattr(ml_models, "_sksurv_c_index", lambda y, scores: 0.61)
    monkeypatch.setattr(sklearn.inspection, "permutation_importance", _fake_perm)

    result = ml_models.train_random_survival_forest(
        full_frame.assign(age=np.linspace(40, 80, full_frame.shape[0])),
        time_column="os_months",
        event_column="os_event",
        features=["age"],
        n_estimators=20,
        random_state=42,
    )

    assert result["feature_importance"]
    assert seen["fit_shape"] == (160, 24)
    assert seen["perm_shape"] == (120, 24)
    assert seen["perm_y_shape"] == (120,)
    assert seen["n_repeats"] == 2


@pytest.mark.parametrize(
    ("train_fn_name", "model_attr"),
    [
        ("train_random_survival_forest", "RandomSurvivalForest"),
        ("train_gradient_boosted_survival", "GradientBoostingSurvivalAnalysis"),
    ],
)
def test_tree_model_training_keeps_encoded_matrices_and_targets_aligned(
    monkeypatch,
    train_fn_name: str,
    model_attr: str,
) -> None:
    import numpy as np
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    train_encoded = pd.DataFrame({"age": [50.0, 60.0, 70.0]})
    eval_encoded = pd.DataFrame({"age": [80.0, 90.0]})
    full_encoded = pd.concat([train_encoded, eval_encoded], ignore_index=True)
    train_frame = pd.DataFrame({"os_months": [10.0, 20.0, 30.0], "os_event": [1.0, 0.0, 1.0]})
    eval_frame = pd.DataFrame({"os_months": [40.0, 50.0], "os_event": [0.0, 1.0]})
    full_frame = pd.concat([train_frame, eval_frame], ignore_index=True)

    class _DummyTreeModel:
        def __init__(self, **kwargs) -> None:
            pass

        @property
        def feature_importances_(self):
            return np.array([1.0], dtype=float)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.linspace(0.1, 0.9, X.shape[0], dtype=float)

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, model_attr, _DummyTreeModel)
    monkeypatch.setattr(
        ml_models,
        "_prepare_model_evaluation_split",
        lambda *args, **kwargs: {
            "train_frame": train_frame,
            "eval_frame": eval_frame,
            "full_frame": full_frame,
            "train_encoded": train_encoded,
            "eval_encoded": eval_encoded,
            "full_encoded": full_encoded,
            "evaluation_mode": "holdout",
            "metric_name": "Holdout C-index",
            "feature_encoder": None,
        },
    )
    monkeypatch.setattr(
        ml_models,
        "_prepare_sksurv_data",
        lambda frame, time_column, event_column: np.array(
            list(zip(frame[event_column].astype(bool), frame[time_column].astype(float), strict=False)),
            dtype=[("event", bool), ("time", float)],
        ),
    )
    monkeypatch.setattr(ml_models, "_sksurv_c_index", lambda y, scores: 0.61)

    train_fn = getattr(ml_models, train_fn_name)
    kwargs = {
        "df": full_frame.assign(age=[50.0, 60.0, 70.0, 80.0, 90.0]),
        "time_column": "os_months",
        "event_column": "os_event",
        "features": ["age"],
        "random_state": 42,
    }
    if train_fn_name == "train_gradient_boosted_survival":
        kwargs["learning_rate"] = 0.1
    result = train_fn(**kwargs)

    assert result["_X_encoded"].shape[0] == result["_y"].shape[0] == full_frame.shape[0]
    assert result["_X_eval_encoded"].shape[0] == result["_y_eval"].shape[0] == eval_frame.shape[0]


def test_random_survival_forest_uses_parallel_tree_jobs(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    captured: dict[str, int | None] = {}

    class _DummyRSF:
        def __init__(self, **kwargs) -> None:
            captured["n_jobs"] = kwargs.get("n_jobs")

        @property
        def feature_importances_(self):
            return np.asarray([1.0], dtype=float)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.linspace(0.1, 0.9, X.shape[0], dtype=float)

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "RandomSurvivalForest", _DummyRSF)

    df = make_example_dataset(seed=41, n_patients=80)
    result = ml_models.train_random_survival_forest(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age"],
        n_estimators=8,
        random_state=13,
    )

    assert captured["n_jobs"] == -1
    assert result["model_stats"]["c_index"] is not None


def test_sksurv_c_index_reraises_memory_error(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)

    def _boom(*args, **kwargs):
        raise MemoryError("oom")

    monkeypatch.setattr(ml_models, "concordance_index_censored", _boom)

    y_true = np.array([(True, 10.0), (False, 12.0)], dtype=[("event", bool), ("time", float)])
    with pytest.raises(MemoryError, match="oom"):
        ml_models._sksurv_c_index(y_true, np.array([0.2, 0.1], dtype=float))


def test_time_dependent_importance_raises_on_internal_classifier_failure(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    class _BrokenRF:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, X, y):
            raise RuntimeError("classifier failed")

    monkeypatch.setattr(ml_models, "RandomForestClassifier", _BrokenRF)

    df = make_example_dataset(seed=14, n_patients=80)
    with pytest.raises(ValueError, match="Time-dependent importance failed"):
        ml_models.compute_time_dependent_importance(
            df,
            time_column="os_months",
            event_column="os_event",
            features=["age", "biomarker_score"],
            eval_times=[12.0],
        )


def test_find_optimal_cutpoint_does_not_swallow_unexpected_errors(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=8, n_patients=80)
    raised = False

    def _boom(*args, **kwargs):
        nonlocal raised
        if not raised:
            raised = True
            raise AssertionError("unexpected failure")
        return (1.0, 0.05)

    monkeypatch.setattr(ml_models, "survdiff", _boom)

    with pytest.raises(AssertionError, match="unexpected failure"):
        ml_models.find_optimal_cutpoint(
            df,
            time_column="os_months",
            event_column="os_event",
            variable="age",
            event_positive_value=1,
        )


def test_find_optimal_cutpoint_permutation_loop_does_not_swallow_unexpected_errors(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=18, n_patients=80)
    call_count = 0

    def _boom(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise AssertionError("unexpected permutation failure")
        return (1.0, 0.05)

    monkeypatch.setattr(ml_models, "survdiff", _boom)

    with pytest.raises(AssertionError, match="unexpected permutation failure"):
        ml_models.find_optimal_cutpoint(
            df,
            time_column="os_months",
            event_column="os_event",
            variable="age",
            event_positive_value=1,
            permutation_iterations=1,
        )


def test_time_dependent_importance_uses_proxy_language() -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=61, n_patients=60)
    result = ml_models.compute_time_dependent_importance(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
    )

    summary = result["scientific_summary"]
    assert "proxy" in summary["headline"].lower()
    assert any("not be described as formal survshap" in caution.lower() for caution in summary["cautions"])


def test_cross_validate_survival_models_flags_incomplete_screening_when_a_fold_fails(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=91, n_patients=120)

    def _ok(train_frame, test_frame, **kwargs):
        return {
            "c_index": 0.61,
            "n_features": len(kwargs["features"]),
            "training_time_ms": 1.0,
            "train_n": len(train_frame),
            "test_n": len(test_frame),
            "train_events": int(train_frame["os_event"].sum()),
            "test_events": int(test_frame["os_event"].sum()),
        }

    seen_failures = {"count": 0}

    def _sometimes_fail(train_frame, test_frame, **kwargs):
        seen_failures["count"] += 1
        if seen_failures["count"] == 1:
            raise ValueError("synthetic fold failure")
        return _ok(train_frame, test_frame, **kwargs)

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _ok)
    monkeypatch.setattr(ml_models, "_fit_evaluate_rsf_split", _sometimes_fail)
    monkeypatch.setattr(ml_models, "_fit_evaluate_gbs_split", _ok)

    result = ml_models.cross_validate_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score"],
        cv_folds=3,
        cv_repeats=2,
        random_state=17,
    )

    assert result["evaluation_mode"] == "repeated_cv_incomplete"
    assert any(row["evaluation_mode"] == "repeated_cv_incomplete" for row in result["comparison_table"])
    assert any(error["model"] == "Random Survival Forest" for error in result["errors"])


def test_partial_dependence_uses_encoder_levels_for_categorical_features() -> None:
    import survival_toolkit.ml_models as ml_models

    class _DummyModel:
        def predict(self, x):
            return np.asarray(x[:, 0], dtype=float)

    analysis_frame = pd.DataFrame({"marker": pd.Series(["a", "z", "a"], dtype="string")})
    encoder = {
        "features": ["marker"],
        "categorical_features": ["marker"],
        "categorical_mappings": {
            "marker": {
                "all_levels": ["a", "__unknown"],
                "retained_levels": ["a"],
                "missing_label": "__missing",
                "unknown_column": "marker____unknown",
                "missing_column": "marker__missing",
            }
        },
        "encoded_columns": ["marker__a", "marker____unknown", "marker__missing"],
        "feature_names": ["marker__a", "marker____unknown", "marker__missing"],
        "numeric_features": [],
        "numeric_impute_values": {},
    }
    encoded = ml_models._transform_feature_encoder(analysis_frame, encoder)

    result = ml_models.compute_partial_dependence(
        _DummyModel(),
        encoded,
        feature_name="marker",
        categorical_features=["marker"],
        feature_encoder=encoder,
        analysis_frame=analysis_frame,
    )

    assert result["feature_type"] == "categorical"
    assert result["values"] == ["a", "__unknown"]


def test_integrated_brier_score_restricts_eval_times_to_support_event_window() -> None:
    from survival_toolkit.ml_models import compute_integrated_brier_score

    times = np.array([5.0, 12.0, 18.0, 30.0], dtype=float)
    events = np.array([1, 0, 1, 0], dtype=int)
    support_times = np.array([4.0, 8.0, 10.0, 40.0], dtype=float)
    support_events = np.array([1, 1, 0, 0], dtype=int)

    def _predicted(eval_times: np.ndarray) -> np.ndarray:
        return np.full((len(times), len(eval_times)), 0.8, dtype=float)

    result = compute_integrated_brier_score(
        times,
        events,
        _predicted,
        eval_times=[2.0, 6.0, 12.0, 20.0],
        support_times=support_times,
        support_events=support_events,
    )

    assert result["eval_times"] == [2.0, 6.0]
    assert any("support window" in strength.lower() for strength in result["scientific_summary"]["strengths"])
    assert result["null_ibs"] is not None
    assert result["brier_skill_score"] == pytest.approx(1.0 - result["ibs"] / result["null_ibs"])
    assert any(metric["label"] == "Brier Skill Score" for metric in result["scientific_summary"]["metrics"])


def test_integrated_brier_score_reports_pointwise_mode_when_support_window_collapses() -> None:
    from survival_toolkit.ml_models import compute_integrated_brier_score

    times = np.array([0.0, 1.0, 2.0], dtype=float)
    events = np.array([1, 0, 1], dtype=int)
    support_times = np.array([0.0, 0.0, 0.0], dtype=float)
    support_events = np.array([1, 0, 0], dtype=int)

    def _predicted(eval_times: np.ndarray) -> np.ndarray:
        return np.full((len(times), len(eval_times)), 0.75, dtype=float)

    result = compute_integrated_brier_score(
        times,
        events,
        _predicted,
        support_times=support_times,
        support_events=support_events,
    )

    assert result["eval_times"] == [0.0]
    assert len(result["brier_scores"]) == 1
    assert result["ibs"] == pytest.approx(result["brier_scores"][0]["score"])
    assert result["null_ibs"] == pytest.approx(result["null_brier_scores"][0]["score"])
    assert "Pointwise Brier score" in result["scientific_summary"]["headline"]
    assert any("single time point" in caution.lower() for caution in result["scientific_summary"]["cautions"])


def test_integrated_brier_score_falls_back_when_numpy_has_no_trapezoid(monkeypatch: pytest.MonkeyPatch) -> None:
    import survival_toolkit.ml_models as ml_models

    monkeypatch.delattr(ml_models.np, "trapezoid", raising=False)

    times = np.array([5.0, 12.0, 18.0, 30.0], dtype=float)
    events = np.array([1, 0, 1, 0], dtype=int)

    def _predicted(eval_times: np.ndarray) -> np.ndarray:
        return np.full((len(times), len(eval_times)), 0.8, dtype=float)

    result = ml_models.compute_integrated_brier_score(
        times,
        events,
        _predicted,
        eval_times=[2.0, 6.0, 12.0],
    )

    assert result["ibs"] is not None
    assert result["null_ibs"] is not None
    assert result["eval_times"] == [2.0, 6.0, 12.0]


def test_integrated_brier_score_rejects_nonfinite_survival_matrix() -> None:
    from survival_toolkit.ml_models import compute_integrated_brier_score

    times = np.array([5.0, 12.0, 18.0], dtype=float)
    events = np.array([1, 0, 1], dtype=int)

    def _predicted(eval_times: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.9, np.nan],
                [0.8, 0.7],
                [0.7, 0.6],
            ],
            dtype=float,
        )

    with pytest.raises(ValueError, match="non-finite survival probabilities"):
        compute_integrated_brier_score(times, events, _predicted, eval_times=[2.0, 6.0])


def test_integrated_brier_score_prefers_patient_informed_predictions_over_flat_baseline() -> None:
    from survival_toolkit.ml_models import compute_integrated_brier_score

    times = np.array([4.0, 6.0, 9.0, 14.0, 18.0, 24.0], dtype=float)
    events = np.array([1, 1, 0, 1, 0, 1], dtype=int)

    def _better(eval_times: np.ndarray) -> np.ndarray:
        eval_times_arr = np.asarray(eval_times, dtype=float)
        patient_scales = np.array([0.18, 0.14, 0.11, 0.08, 0.06, 0.04], dtype=float)
        return np.exp(-patient_scales[:, None] * eval_times_arr[None, :])

    def _flat(eval_times: np.ndarray) -> np.ndarray:
        eval_times_arr = np.asarray(eval_times, dtype=float)
        return np.full((len(times), len(eval_times_arr)), 0.5, dtype=float)

    better = compute_integrated_brier_score(times, events, _better, eval_times=np.linspace(2.0, 18.0, 7))
    flat = compute_integrated_brier_score(times, events, _flat, eval_times=np.linspace(2.0, 18.0, 7))

    assert better["brier_skill_score"] is not None
    assert flat["brier_skill_score"] is not None
    assert better["brier_skill_score"] > flat["brier_skill_score"]


def test_counterfactual_survival_requires_predict_callable() -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=88, n_patients=40)
    trained_result = {
        "_model": object(),
        "_X_encoded": pd.DataFrame({"age": [1.0, 2.0, 3.0]}),
        "_analysis_frame": df[["os_months", "os_event", "age"]].copy(),
    }

    with pytest.raises(ValueError, match="callable predict\\(\\)"):
        ml_models.counterfactual_survival(
            df,
            time_column="os_months",
            event_column="os_event",
            features=["age"],
            target_feature="age",
            counterfactual_value=65,
            trained_result=trained_result,
        )


def test_cross_validate_survival_models_rejects_excessive_total_evaluations() -> None:
    import survival_toolkit.ml_models as ml_models

    df = make_example_dataset(seed=34, n_patients=80)

    with pytest.raises(ValueError, match="must not exceed 200"):
        ml_models.cross_validate_survival_models(
            df,
            time_column="os_months",
            event_column="os_event",
            features=["age", "biomarker_score"],
            cv_folds=10,
            cv_repeats=21,
        )


def test_augment_scientific_summary_with_brier_does_not_mutate_input() -> None:
    import survival_toolkit.ml_models as ml_models

    summary = {
        "status": "robust",
        "headline": "Summary",
        "strengths": [],
        "cautions": [],
        "next_steps": [],
        "metrics": [],
    }
    original = copy.deepcopy(summary)

    updated = ml_models._augment_scientific_summary_with_brier(
        summary,
        {"ibs": 0.12, "null_ibs": 0.20, "brier_skill_score": 0.40},
    )

    assert summary == original
    assert updated["metrics"]


def test_fit_feature_encoder_rejects_empty_feature_list() -> None:
    from survival_toolkit.encoding import fit_feature_encoder

    df = make_example_dataset(seed=55, n_patients=20)

    with pytest.raises(ValueError, match="Select at least one feature"):
        fit_feature_encoder(df, [])

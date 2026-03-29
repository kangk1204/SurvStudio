from __future__ import annotations

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
    assert result["model_stats"]["evaluation_mode"] == "holdout"
    assert result["model_stats"]["n_evaluation_patients"] < result["model_stats"]["n_patients"]
    assert "holdout" in result["scientific_summary"]["headline"].lower()
    assert len(result["feature_importance"]) == 3
    assert result["feature_importance"][0]["importance"] >= result["feature_importance"][-1]["importance"]


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
    assert result["model_stats"]["evaluation_mode"] == "holdout"
    assert result["model_stats"]["n_evaluation_patients"] < result["model_stats"]["n_patients"]
    assert "holdout" in result["scientific_summary"]["headline"].lower()
    assert len(result["feature_importance"]) == 3


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
    assert len(result["comparison_table"]) >= 1
    assert all("evaluation_mode" in row for row in result["comparison_table"])
    assert "manuscript_tables" in result
    assert result["manuscript_tables"]["model_performance_table"]


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
        "Random Survival Forest",
        "Gradient Boosted Survival",
    }
    assert result["errors"] == []


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

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _fake_cox)
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
    assert ("rsf", 37, 5, 99) in seen
    assert ("gbs", 37, 0.2, 99) in seen


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

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _fake_cox)
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

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml_models, "_fit_evaluate_cox_split", _fake_cox)
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


def test_store_lru_eviction() -> None:
    from survival_toolkit.store import DatasetStore

    store = DatasetStore(max_datasets=3, ttl_seconds=9999)
    df = make_example_dataset(seed=1, n_patients=30)
    ids = []
    for i in range(5):
        stored = store.create(df, filename=f"test_{i}")
        ids.append(stored.dataset_id)
    assert store.count == 3
    with pytest.raises(KeyError):
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


def test_store_delete() -> None:
    from survival_toolkit.store import DatasetStore

    store = DatasetStore()
    df = make_example_dataset(seed=2, n_patients=30)
    stored = store.create(df, filename="to_delete")
    store.delete(stored.dataset_id)
    with pytest.raises(KeyError):
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
    result = ml_models.compute_shap_values(object(), encoded, feature_names=list(encoded.columns))

    assert result["method"] == "kernel"
    assert seen["background_shape"] == (20, 5)
    assert seen["eval_shape"] == (20, 5)
    assert seen["nsamples"] == 40
    assert seen["silent"] is True


def test_rsf_permutation_importance_caps_rows_and_repeats(monkeypatch) -> None:
    import numpy as np
    import pandas as pd
    import survival_toolkit.ml_models as ml_models
    import sklearn.inspection

    seen: dict[str, object] = {}

    train_encoded = pd.DataFrame(np.ones((160, 24)), columns=[f"f{i}" for i in range(24)])
    eval_encoded = pd.DataFrame(np.ones((180, 24)), columns=[f"f{i}" for i in range(24)])
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

"""Regression tests for the 2026-09-29 second review of the deep-learning module (R8 and R9 findings).

Every network here is tiny (hidden width <= 8, <= 5 epochs, <= 200 rows) and no test starts a
real worker pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.sample_data import make_example_dataset

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import survival_toolkit.deep_models as dm  # noqa: E402
from survival_toolkit.analysis import _cohort_frame  # noqa: E402
from survival_toolkit.errors import UserInputError  # noqa: E402

FEATURES = ["age", "biomarker_score", "immune_index"]
TINY = {"hidden_layers": [4], "epochs": 2, "batch_size": 16}


def _cautions(result: dict) -> list[str]:
    return list(result["scientific_summary"]["cautions"])


def _metrics(result: dict) -> dict:
    return {metric["label"]: metric["value"] for metric in result["scientific_summary"]["metrics"]}


# R8#1 / R8#2: the deep models analyse the rows of the ML cohort builder ------------------


def _grouped_time_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    days = rng.integers(30, 3000, n)
    return pd.DataFrame({
        # Excel-style thousands separators on the long follow-up times.
        "os_days": [f"{int(value):,}" if value >= 1000 else str(int(value)) for value in days],
        "os_event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
    })


def test_grouped_times_are_parsed_and_the_rows_match_the_ml_cohort() -> None:
    df = _grouped_time_frame()
    assert df["os_days"].str.contains(",").any()
    clean = dm._coerce_deep_frame(df, "os_days", "os_event", ["age"])
    cohort = _cohort_frame(df, "os_days", "os_event", extra_columns=["age"], drop_missing_extra_columns=False)
    assert clean.shape[0] == cohort.shape[0] == len(df)
    assert clean.attrs["source_row_index"] == list(cohort.attrs["source_row_index"])
    np.testing.assert_array_equal(clean["os_days"].to_numpy(dtype=float), cohort["os_days"].to_numpy(dtype=float))


def test_text_that_is_not_a_time_is_refused_as_for_the_ml_models() -> None:
    df = _grouped_time_frame()
    df["os_days"] = df["os_days"].str.replace(",", "", regex=False)
    df.loc[3, "os_days"] = "12 months"
    with pytest.raises(UserInputError, match="not numbers"):
        dm.train_deepsurv(df, "os_days", "os_event", ["age"], **TINY)
    with pytest.raises(UserInputError, match="not numbers"):
        dm.compare_deep_survival_models(
            df, "os_days", "os_event", ["age"], epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
            evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1,
        )


def test_a_time_column_missing_for_one_outcome_is_refused() -> None:
    rng = np.random.default_rng(1)
    n = 80
    event = rng.integers(0, 2, n)
    df = pd.DataFrame({
        # TCGA-style days_to_death: missing for every censored patient but three.
        "days_to_death": np.where(event == 1, rng.integers(30, 3000, n).astype(float), np.nan),
        "vital_status": event,
        "age": rng.normal(60, 10, n),
    })
    df.loc[df.index[:3], ["vital_status", "days_to_death"]] = [0, 500.0]
    with pytest.raises(UserInputError, match="censored rows"):
        dm.train_deepsurv(df, "days_to_death", "vital_status", ["age"], **TINY)


def test_a_cleaning_that_removes_every_censored_row_is_refused() -> None:
    df = make_example_dataset(seed=3, n_patients=80)
    df.loc[df["os_event"] == 0, "os_months"] = -1.0
    with pytest.raises(UserInputError, match="censored rows were removed"):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], included_models=["DeepSurv"])


def test_times_that_are_never_positive_are_refused() -> None:
    df = make_example_dataset(seed=3, n_patients=40).assign(os_months=0.0)
    with pytest.raises(UserInputError, match="positive values"):
        dm.train_deepsurv(df, "os_months", "os_event", FEATURES, **TINY)


def test_rows_dropped_for_a_missing_outcome_are_reported() -> None:
    df = make_example_dataset(seed=5, n_patients=90)
    df.loc[df.index[:4], "os_months"] = np.nan
    single = dm.train_deepsurv(df, "os_months", "os_event", FEATURES, **TINY)
    assert _metrics(single)["Dropped for missing outcome"] == 4
    assert any(caution.startswith("4 row(s) with a missing or non-finite survival time") for caution in _cautions(single))
    compared = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1,
    )
    assert _metrics(compared)["Dropped for missing outcome"] == 4
    assert compared["n_patients"] == 86


def test_the_time_column_caution_is_shown_in_every_summary() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "os_months": rng.exponential(20, 90) + 0.1,
        "os_event": rng.integers(0, 2, 90),
        "score": rng.normal(size=90),
    })
    df["weird"] = df["os_months"]
    note = "does not look like a survival follow-up time column"
    single = dm.train_deepsurv(df, "weird", "os_event", ["score"], **TINY)
    assert any(note in caution for caution in _cautions(single))
    for strategy in ("holdout", "repeated_cv"):
        compared = dm.compare_deep_survival_models(
            df, "weird", "os_event", ["score"], epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
            evaluation_strategy=strategy, cv_folds=2, cv_repeats=1,
        )
        assert any(note in caution for caution in _cautions(compared))
        assert compared["scientific_summary"]["status"] != "robust"


# R8#3: feature types follow the shared rule, decided once on the cleaned cohort -----------


def _typed_frame(column: pd.Series | pd.api.extensions.ExtensionArray) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 80
    return pd.DataFrame({
        "os_months": rng.exponential(20, n) + 0.1,
        "os_event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
        "x": column,
    })


def _encoded_names(df: pd.DataFrame, categorical: list[str] | None = None) -> list[str]:
    data, split = dm._prepare_deep_training_inputs(
        df, time_column="os_months", event_column="os_event", features=["x", "age"],
        categorical_features=categorical, random_seed=3,
    )
    assert split["evaluation_mode"] == "holdout"
    return list(data["feature_names"])


def test_an_integer_pandas_categorical_is_one_hot_with_canonical_labels() -> None:
    rng = np.random.default_rng(4)
    codes = rng.choice([1, 2, 3], 80)
    assert _encoded_names(_typed_frame(pd.Categorical(codes))) == ["x_2", "x_3", "age"]
    with_missing = pd.Categorical(np.where(np.arange(80) % 8 == 0, np.nan, codes))  # float categories 1.0, 2.0, 3.0
    assert _encoded_names(_typed_frame(with_missing)) == ["x_2", "x_3", "x__missing", "age"]


def test_text_whose_values_all_read_as_numbers_stays_numeric() -> None:
    rng = np.random.default_rng(5)
    values = rng.choice(["1", "2", "3"], 80)
    assert _encoded_names(_typed_frame(pd.Series(values, dtype=object))) == ["x", "age"]
    assert _encoded_names(_typed_frame(pd.Series(values, dtype="string"))) == ["x", "age"]
    # Declared categorical: one-hot, whatever the values look like.
    assert _encoded_names(_typed_frame(pd.Series(values, dtype=object)), ["x"]) == ["x_2", "x_3", "age"]


def test_numeric_text_with_many_distinct_values_is_not_refused_as_text() -> None:
    rng = np.random.default_rng(6)
    values = pd.Series([f"{value:.3f}" for value in rng.normal(size=80)], dtype=object)
    assert _encoded_names(_typed_frame(values)) == ["x", "age"]


def test_training_splits_keep_categorical_text_even_when_their_levels_look_numeric(monkeypatch) -> None:
    seen: list[list[str]] = []
    original = dm._fit_shared_feature_encoder

    def _spy(frame, features, categorical_features=None, **kwargs):
        seen.append(list(categorical_features or []))
        return original(frame, features, categorical_features, **kwargs)

    monkeypatch.setattr(dm, "_fit_shared_feature_encoder", _spy)
    rng = np.random.default_rng(7)
    grade = pd.Series(rng.choice(["1", "2", "3"], 80), dtype=object)
    grade.iloc[0] = "unknown"  # the whole cohort holds text, so "grade" is categorical
    df = _typed_frame(grade)
    dm.compare_deep_survival_models(
        df, "os_months", "os_event", ["x", "age"], epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
        evaluation_strategy="repeated_cv", cv_folds=3, cv_repeats=1,
    )
    assert seen and all("x" in categorical for categorical in seen)


def test_results_name_the_categorical_features_actually_used() -> None:
    rng = np.random.default_rng(8)
    n = 90
    df = pd.DataFrame({
        "os_months": rng.exponential(20, n) + 0.1,
        "os_event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
        "sex": rng.choice(["F", "M"], n),
        "grade": rng.choice(["1", "2", "3"], n).astype(object),
        "codes": rng.choice(["1", "2", "3"], n).astype(object),
        "cat": pd.Categorical(rng.choice([1, 2], n)),
        "score_text": [f"{value:.2f}" for value in rng.normal(size=n)],
    })
    df.loc[0, "grade"] = "unknown"
    features = ["age", "sex", "grade", "codes", "cat", "score_text"]
    expected = ["sex", "grade", "codes", "cat"]  # declared ("codes"), inferred text, and a pandas Categorical
    common = dict(categorical_features=["codes"], epochs=1, hidden_layers=[4])
    assert dm.train_deepsurv(df, "os_months", "os_event", features, batch_size=16, **common)["categorical_features"] == expected
    for strategy in ("holdout", "repeated_cv"):
        compared = dm.compare_deep_survival_models(
            df, "os_months", "os_event", features, included_models=["DeepSurv"], evaluation_strategy=strategy,
            cv_folds=2, cv_repeats=1, **common,
        )
        assert compared["categorical_features"] == expected
    single = dm.evaluate_single_deep_survival_model(
        "deepsurv", df=df, time_column="os_months", event_column="os_event", features=features,
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, **common,
    )
    assert single["categorical_features"] == expected


def test_duplicate_feature_names_get_a_clear_message() -> None:
    df = make_example_dataset(seed=4, n_patients=60)
    with pytest.raises(UserInputError, match="listed more than once: age"):
        dm.compare_deep_survival_models(df, "os_months", "os_event", ["age", "age"], epochs=1, hidden_layers=[4], included_models=["DeepSurv"])


# R8#5 / R8#6 / R9#14 / R9#15: package-API arguments ---------------------------------------


def test_a_caller_split_needs_the_prepared_tensors_it_indexes() -> None:
    df = make_example_dataset(seed=5, n_patients=60).reset_index(drop=True)
    df.loc[2, "os_months"] = np.nan  # removed by cleaning, so later positions shift
    split = {
        "train_idx": np.setdiff1d(np.arange(58), np.arange(10, 20)),
        "eval_idx": np.arange(10, 20),
        "evaluation_mode": "holdout",
        "evaluation_note": "caller split",
    }
    with pytest.raises(UserInputError, match="evaluation_split can only be supplied together"):
        dm.train_deepsurv(df, "os_months", "os_event", FEATURES, evaluation_split=split, **TINY)


@pytest.mark.parametrize("fraction", [0.0, -0.3, float("nan"), 0.7, True])
def test_locked_test_fractions_outside_the_web_bounds_are_refused(fraction) -> None:
    df = make_example_dataset(seed=4, n_patients=60)
    with pytest.raises(UserInputError, match="locked_test_fraction must be None"):
        dm.compare_deep_survival_models(
            df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
            evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, locked_test_fraction=fraction,
        )
    with pytest.raises(UserInputError, match="locked_test_fraction must be None"):
        dm.evaluate_single_deep_survival_model(
            "deepsurv", df=df, time_column="os_months", event_column="os_event", features=FEATURES, epochs=1,
            hidden_layers=[4], evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, locked_test_fraction=fraction,
        )


@pytest.mark.parametrize(
    "setting, match",
    [
        ({"epochs": 0}, "number of epochs"),
        ({"epochs": -5}, "number of epochs"),
        ({"epochs": 2.5}, "number of epochs"),
        ({"batch_size": 0}, "batch size"),
        ({"learning_rate": 0.0}, "learning rate"),
        ({"learning_rate": -0.01}, "learning rate"),
        ({"learning_rate": float("nan")}, "learning rate"),
    ],
)
def test_training_settings_are_validated_at_every_entry_point(setting, match) -> None:
    df = make_example_dataset(seed=4, n_patients=60)
    kwargs = {**TINY, **setting}
    with pytest.raises(UserInputError, match=match):
        dm.train_deepsurv(df, "os_months", "os_event", FEATURES, **kwargs)
    with pytest.raises(UserInputError, match=match):
        dm.train_neural_mtlr(df, "os_months", "os_event", FEATURES, num_time_bins=5, **kwargs)
    with pytest.raises(UserInputError, match=match):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, included_models=["DeepSurv"], **kwargs)


def test_zero_attention_heads_are_refused_with_a_clear_message() -> None:
    df = make_example_dataset(seed=4, n_patients=60)
    with pytest.raises(UserInputError, match="number of attention heads"):
        dm.train_survival_transformer(df, "os_months", "os_event", FEATURES, d_model=8, n_heads=0, n_layers=1, epochs=1)
    with pytest.raises(UserInputError, match="number of attention heads"):
        dm.compare_deep_survival_models(
            df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], d_model=8, n_heads=0, n_layers=1,
            included_models=["DeepSurv", "Survival Transformer"],
        )
    # The setting is ignored (and not checked) when the transformer is not compared.
    compared = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], n_heads=0, included_models=["DeepSurv"],
    )
    assert [row["model"] for row in compared["comparison_table"]] == ["DeepSurv"]


def test_an_explicit_empty_vae_hidden_layer_list_is_not_replaced_by_the_default() -> None:
    df = make_example_dataset(seed=4, n_patients=60)
    with pytest.raises(UserInputError, match="at least one hidden layer"):
        dm.train_survival_vae(df, "os_months", "os_event", FEATURES, hidden_layers=[], latent_dim=2, epochs=1)
    with pytest.raises(ValueError, match="at least one hidden layer"):
        dm.SurvivalVAENet(4, hidden_layers=[], latent_dim=2)
    assert [module.out_features for module in dm.SurvivalVAENet(4, hidden_dim=6, latent_dim=2).encoder if isinstance(module, torch.nn.Linear)] == [6]


# R8#4 / R9#1 / R9#2 / R9#3 / R9#4 / R9#5 / R9#11: comparison summaries ----------------------


def _stub(c_index: float, *, fail_first: bool = False, mode_first: str | None = None):
    """A trainer that returns a fixed result (no network), optionally failing or falling back on its first call."""
    calls = {"n": 0}

    def _run(*args, **kwargs):
        calls["n"] += 1
        if fail_first and calls["n"] == 1:
            raise RuntimeError("simulated fold failure")
        split = kwargs["evaluation_split"]
        return {
            "c_index": c_index,
            "evaluation_mode": mode_first if mode_first and calls["n"] == 1 else "holdout",
            "epochs_trained": 1,
            "early_stopping_epochs": 1,
            "n_features": 3,
            "training_samples": len(split["train_idx"]),
            "evaluation_samples": len(split["eval_idx"]),
            "training_events": 20,
            "evaluation_events": 10,
        }

    return _run


def _install_stubs(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    trainers = {
        "train_deepsurv": _stub(0.74),
        "train_deephit": _stub(0.70),
        "train_neural_mtlr": _stub(0.69),
        "train_survival_transformer": _stub(0.66),
        "train_survival_vae": _stub(0.62),
    }
    trainers.update(overrides)
    for name, trainer in trainers.items():
        monkeypatch.setattr(dm, name, trainer)


def _cv(df: pd.DataFrame, **kwargs) -> dict:
    return dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1,
        random_seed=3, **kwargs,
    )


def test_repeated_cv_rows_without_an_aggregate_are_not_ranked(monkeypatch) -> None:
    _install_stubs(monkeypatch, train_deepsurv=_stub(0.74, fail_first=True))
    result = _cv(make_example_dataset(seed=17, n_patients=80))
    rows = {row["model"]: row for row in result["comparison_table"]}
    assert rows["DeepSurv"]["c_index"] is None
    assert rows["DeepSurv"]["rank"] is None and rows["DeepSurv"]["comparable_for_ranking"] is False
    assert [row["model"] for row in result["comparison_table"]] == [
        "DeepHit", "Neural MTLR", "Survival Transformer", "Survival VAE", "DeepSurv",
    ]
    assert [row["rank"] for row in result["comparison_table"]] == [1, 2, 3, 4, None]
    summary = result["scientific_summary"]
    # Holdout-only wording stays out of a repeated-CV summary.
    assert "holdout-evaluable" not in summary["headline"]
    assert not any("clean holdout estimate" in strength for strength in summary["strengths"])
    assert not any("apparent fallback were excluded" in caution for caution in summary["cautions"])
    assert "1 fold-level fit(s) failed; a model with a failed fold has no repeated-CV aggregate and is not ranked." in summary["cautions"]
    assert result["evaluation_mode"] == "repeated_cv_incomplete" and result["ranking_complete"] is False


def test_no_model_is_named_best_when_none_has_a_cv_aggregate(monkeypatch) -> None:
    _install_stubs(monkeypatch)
    original = dm._run_deep_compare_task

    def _task(task):
        if task["repeat"] == 1 and task["fold"] == 1:
            raise ValueError("simulated fold failure")
        return original(task)

    monkeypatch.setattr(dm, "_run_deep_compare_task", _task)
    result = _cv(make_example_dataset(seed=17, n_patients=80), included_models=["DeepSurv", "DeepHit"], locked_test_fraction=0.3)
    assert [row["rank"] for row in result["comparison_table"]] == [None, None]
    summary = result["scientific_summary"]
    assert summary["headline"].startswith("No deep model could be ranked")
    assert _metrics(result)["Best model"] is None
    assert not any("CV-selected model (" in strength for strength in summary["strengths"])
    assert summary["cautions"][0].startswith("No model completed every development-set repeated-CV fold")
    assert summary["status"] == "review"
    # The locked-test estimates are still shown, with their event counts.
    assert all(row["locked_test_c_index"] is not None and row["locked_test_events"] for row in result["comparison_table"])


def test_a_repeated_cv_run_whose_every_fit_failed_raises_like_the_holdout_comparison(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("DeepSurv loss became NaN or Inf during training.")

    _install_stubs(monkeypatch, train_deepsurv=_boom, train_deephit=_boom)
    df = make_example_dataset(seed=17, n_patients=80)
    with pytest.raises(UserInputError, match=r"All deep-learning models failed to train\. Errors: DeepSurv \(2 folds\): DeepSurv loss"):
        _cv(df, included_models=["DeepSurv", "DeepHit"])
    with pytest.raises(UserInputError, match="All deep-learning models failed to train"):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, included_models=["DeepSurv", "DeepHit"])


@pytest.mark.parametrize(
    "strategy, caution",
    [
        ("holdout", "The top-ranked model was selected and scored on the same evaluation split"),
        ("repeated_cv", "The top-ranked model was selected and scored within the same repeated-CV screening run"),
    ],
)
def test_comparisons_carry_the_ml_screening_caution(monkeypatch, strategy, caution) -> None:
    _install_stubs(monkeypatch)
    result = dm.compare_deep_survival_models(
        make_example_dataset(seed=17, n_patients=120), "os_months", "os_event", FEATURES,
        evaluation_strategy=strategy, cv_folds=2, cv_repeats=1, random_seed=3,
    )
    summary = result["scientific_summary"]
    assert summary["cautions"][0].startswith(caution)
    assert summary["status"] == "review"


def test_the_status_is_decided_after_the_late_cautions() -> None:
    df = make_example_dataset(seed=4, n_patients=120)

    def _finalize(**kwargs):
        comparison = [{"model": "DeepSurv", "c_index": 0.80, "evaluation_mode": "holdout", "training_seed": 1, "split_seed": 1, "monitor_seed": 1}]
        return dm._finalize_deep_comparison(
            comparison, [], df=df, n_selected_features=3, evaluation_mode="holdout", random_seed=1,
            cohort_counts={"n_patients": 120, "n_events": 60}, **kwargs,
        )

    assert _finalize()["scientific_summary"]["status"] == "robust"
    unseen = dm._unseen_category_cautions([(3, "evaluation")])
    flagged = _finalize(extra_cautions=unseen)
    assert flagged["scientific_summary"]["status"] == "review"
    assert flagged["scientific_summary"]["cautions"][-1] == unseen[0]
    noted = _finalize(parallel_execution_note="Parallel repeated-CV execution was disabled; folds ran sequentially.")
    assert noted["scientific_summary"]["status"] == "review"
    assert noted["parallel_execution_note"].startswith("Parallel repeated-CV execution was disabled")


def test_single_model_repeated_cv_keeps_the_fold_errors_and_does_not_repeat_cautions(monkeypatch) -> None:
    _install_stubs(monkeypatch, train_deepsurv=_stub(0.70, fail_first=True))
    monkeypatch.setattr(dm, "_available_cpu_count", lambda: 1)
    single = dm.evaluate_single_deep_survival_model(
        "deepsurv", df=make_example_dataset(seed=17, n_patients=80), time_column="os_months", event_column="os_event",
        features=FEATURES, evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, random_seed=3, parallel_jobs=2,
    )
    cautions = single["scientific_summary"]["cautions"]
    assert len(cautions) == len(set(cautions))
    assert sum("only 1 CPU" in caution for caution in cautions) == 1
    assert [error["error"] for error in single["errors"]] == ["simulated fold failure"]
    assert single["ranking_complete"] is False
    assert single["scientific_summary"]["headline"] == "The 1x2 repeated-CV C-index of DeepSurv was withheld: 1 of 2 fold(s) failed."


def test_a_fold_that_falls_back_to_apparent_evaluation_is_a_failed_fold(monkeypatch) -> None:
    _install_stubs(monkeypatch, train_deepsurv=_stub(0.74, mode_first="holdout_fallback_apparent"))
    result = _cv(make_example_dataset(seed=17, n_patients=80), included_models=["DeepSurv", "DeepHit"])
    rows = {row["model"]: row for row in result["comparison_table"]}
    assert rows["DeepSurv"]["n_failures"] == 1 and rows["DeepSurv"]["rank"] is None
    assert "n_apparent_fallbacks" not in rows["DeepSurv"]
    assert "did not retain a clean holdout evaluation" in result["errors"][0]["error"]
    summary = result["scientific_summary"]
    assert not any("apparent-fallback" in text for text in [*summary["strengths"], *summary["cautions"]])


def test_deep_rows_and_tables_carry_event_counts() -> None:
    df = make_example_dataset(seed=17, n_patients=120)
    common = dict(epochs=1, hidden_layers=[4], included_models=["DeepSurv"], random_seed=3)
    holdout = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    row = holdout["comparison_table"][0]
    block = holdout["test_predictions"]
    assert row["test_events"] == holdout["n_evaluation_events"] == int(sum(block["event"]))
    assert row["evaluation_samples"] == holdout["n_evaluation_patients"] == len(block["row_ids"])
    assert row["train_events"] == holdout["n_fit_events"] == holdout["n_events"] - holdout["n_evaluation_events"]
    assert holdout["manuscript_tables"]["model_performance_table"][0]["Evaluation Events, n"] == holdout["n_evaluation_events"]

    cv = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1,
        locked_test_fraction=0.3, **common,
    )
    cv_row = cv["comparison_table"][0]
    assert cv_row["test_events"] == int(round(np.mean([fold["evaluation_events"] for fold in cv["fold_results"]])))
    assert cv_row["train_events"] == int(round(np.mean([fold["training_events"] for fold in cv["fold_results"]])))
    assert cv_row["locked_test_events"] == cv["n_locked_test_events"]
    table_row = cv["manuscript_tables"]["model_performance_table"][0]
    assert table_row["Locked-test Events, n"] == cv["n_locked_test_events"]
    assert table_row["Mean Evaluation Events, n"] == cv_row["test_events"]

    single = dm.train_deepsurv(df, "os_months", "os_event", FEATURES, random_seed=3, **TINY)
    assert single["training_events"] + single["evaluation_events"] == holdout["n_events"]


# R8#8 / R9#12 / R9#6 / R9#7 / R9#14 and the R8 notes: fold execution -----------------------


def _seed_stub(*args, **kwargs):
    """A trainer whose C-index depends only on the fold's seed (so every fold differs)."""
    seed = int(kwargs["random_seed"])
    split = kwargs["evaluation_split"]
    return {
        "c_index": 0.55 + (seed % 97) / 1000.0 + 1e-9 * seed,
        "evaluation_mode": "holdout",
        "epochs_trained": 1,
        "n_features": 3,
        "training_samples": len(split["train_idx"]),
        "evaluation_samples": len(split["eval_idx"]),
    }


def test_fold_results_are_summarised_in_a_fixed_order(monkeypatch) -> None:
    _install_stubs(monkeypatch, train_deepsurv=_seed_stub, train_deephit=_seed_stub)
    original = dm._run_deep_fold_tasks

    def _completion_order(fold_splits, build_task, *, fold_results, errors, parallel_jobs):
        # Parallel workers finish in any order; simulate the reverse of the submission order.
        note = original(fold_splits, build_task, fold_results=fold_results, errors=errors, parallel_jobs=1)
        fold_results.reverse()
        errors.reverse()
        return note

    df = make_example_dataset(seed=12, n_patients=100)
    common = dict(included_models=["DeepSurv", "DeepHit"], evaluation_strategy="repeated_cv", cv_folds=5, cv_repeats=2, random_seed=7)
    sequential = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    monkeypatch.setattr(dm, "_run_deep_fold_tasks", _completion_order)
    shuffled = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)

    def _keys(result):
        return [(row["repeat"], row["fold"], row["model"]) for row in result["fold_results"]]

    assert _keys(shuffled) == _keys(sequential) == sorted(_keys(sequential), key=lambda key: (key[0], key[1], key[2] != "DeepSurv"))
    assert [repr(row["c_index"]) for row in shuffled["comparison_table"]] == [repr(row["c_index"]) for row in sequential["comparison_table"]]


def _optimizer_with_invalid_learning_rate(original):
    def _optimizer(model, learning_rate):
        if isinstance(model, dm.DeepSurvNet):
            return torch.optim.AdamW(model.parameters(), lr=-1.0)  # ValueError raised inside torch
        return original(model, learning_rate)

    return _optimizer


@pytest.mark.parametrize("strategy", ["holdout", "repeated_cv"])
def test_model_failures_keep_the_library_message_and_are_logged(monkeypatch, caplog, strategy) -> None:
    monkeypatch.setattr(dm, "_make_optimizer", _optimizer_with_invalid_learning_rate(dm._make_optimizer))
    df = make_example_dataset(seed=4, n_patients=80)
    with caplog.at_level("ERROR", logger="survival_toolkit.deep_models"):
        result = dm.compare_deep_survival_models(
            df, "os_months", "os_event", FEATURES, hidden_layers=[4], epochs=1, num_time_bins=5,
            included_models=["DeepSurv", "Neural MTLR"], evaluation_strategy=strategy, cv_folds=2, cv_repeats=1,
        )
    assert result["errors"] and all(error["model"] == "DeepSurv" for error in result["errors"])
    assert all("Invalid learning rate" in error["error"] for error in result["errors"])
    logged = [record for record in caplog.records if record.name == "survival_toolkit.deep_models" and record.exc_info]
    assert len(logged) == len(result["errors"])
    assert all("DeepSurv" in record.getMessage() for record in logged)


def test_memory_errors_end_the_run_instead_of_being_recorded_as_model_failures(monkeypatch) -> None:
    def _out_of_memory(*args, **kwargs):
        raise MemoryError()

    _install_stubs(monkeypatch, train_deepsurv=_out_of_memory)
    df = make_example_dataset(seed=4, n_patients=80)
    with pytest.raises(MemoryError):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, included_models=["DeepSurv", "DeepHit"])
    with pytest.raises(MemoryError):
        _cv(df, included_models=["DeepHit", "DeepSurv"])
    wrapped = dm.InternalAnalysisError()
    wrapped.__cause__ = MemoryError()
    assert dm._must_propagate_deep(MemoryError()) and dm._must_propagate_deep(wrapped)
    with pytest.raises(MemoryError):
        dm._record_fold_error([], "DeepSurv", 1, 1, MemoryError())


def test_the_first_fold_task_is_released_after_a_sequential_fallback(monkeypatch) -> None:
    import gc
    import weakref

    refs: list[weakref.ref] = []
    alive_at_start: list[list[bool]] = []

    def _fold_runner(task):
        gc.collect()
        alive_at_start.append([ref() is not None for ref in refs])
        refs.append(weakref.ref(task["prepared_data"]["X_tensor"]))
        row = {
            "model": "DeepSurv", "repeat": task["repeat"], "fold": task["fold"], "c_index": 0.6, "evaluation_mode": "holdout",
            "epochs_trained": 1, "training_samples": 50, "evaluation_samples": 20, "n_features": 3, "training_time_ms": 1.0,
        }
        return {"fold_results": [row], "errors": []}

    monkeypatch.setattr(dm, "_run_deep_compare_fold_task", _fold_runner)
    monkeypatch.setattr(dm, "_available_cpu_count", lambda: 2)
    monkeypatch.setattr(dm, "_estimate_deep_compare_task_bytes", lambda task: 10**12)  # "payload too large"
    result = _cv(make_example_dataset(seed=16, n_patients=90), included_models=["DeepSurv"], parallel_jobs=2)
    result_folds = [(row["repeat"], row["fold"]) for row in result["fold_results"]]
    assert result_folds == [(1, 1), (1, 2)]
    assert "too large" in result["parallel_execution_note"]
    assert alive_at_start == [[], [False]]


class _BreakingExecutor:
    """In-process executor whose first fold's future fails like a worker killed for memory."""

    def __init__(self, *args, **kwargs) -> None:
        self.submitted: list[tuple[int, int]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, task):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        self.submitted.append((task["repeat"], task["fold"]))
        future: Future = Future()
        if (task["repeat"], task["fold"]) == (1, 1):
            future.set_exception(BrokenProcessPool("A process in the process pool was terminated abruptly"))
        else:
            future.set_result(fn(task))
        return future


@pytest.mark.parametrize("memory_after_crash, rerun", [(64 * 1024**3, True), (1024**2, False), (None, False)])
def test_folds_of_a_crashed_worker_are_rerun_only_when_they_fit_in_memory(monkeypatch, memory_after_crash, rerun) -> None:
    probes = iter([64 * 1024**3])
    monkeypatch.setattr(dm, "_available_system_memory_bytes", lambda: next(probes, memory_after_crash))
    monkeypatch.setattr(dm, "_available_cpu_count", lambda: 2)
    monkeypatch.setattr(dm, "ProcessPoolExecutor", _BreakingExecutor)
    _install_stubs(monkeypatch)
    result = dm.compare_deep_survival_models(
        make_example_dataset(seed=16, n_patients=90), "os_months", "os_event", FEATURES, included_models=["DeepSurv"],
        evaluation_strategy="repeated_cv", cv_folds=3, cv_repeats=1, parallel_jobs=2, random_seed=5,
    )
    folds = [(row["repeat"], row["fold"]) for row in result["fold_results"]]
    if rerun:
        assert folds == [(1, 1), (1, 2), (1, 3)] and result["errors"] == []
        assert "reran the 2 unfinished fold(s) sequentially" in result["parallel_execution_note"]
    else:
        assert folds == [(1, 2)]
        assert [(error["repeat"], error["fold"]) for error in result["errors"]] == [(1, 1), (1, 3)]
        assert all("was not rerun in the SurvStudio process" in error["error"] for error in result["errors"])
        assert "were not rerun" in result["parallel_execution_note"]
        assert result["comparison_table"][0]["rank"] is None


def test_a_locked_test_refit_without_a_holdout_estimate_is_described_as_such(monkeypatch) -> None:
    def _fallback(*args, **kwargs):
        return {"c_index": 0.6, "evaluation_mode": "holdout_fallback_apparent", "n_features": 3}

    monkeypatch.setattr(dm, "train_deepsurv", _fallback)
    task = {
        "model_name": "DeepSurv", "extra_kwargs": {}, "repeat": None, "fold": None, "seed": 1, "split_seed": 1, "monitor_seed": 1,
        "time_column": "os_months", "event_column": "os_event", "features": ["age"], "categorical_features": [],
        "event_positive_value": 1, "learning_rate": 0.001, "epochs": 1, "batch_size": 8, "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0, "prepared_data": {}, "evaluation_split": {}, "monitor_indices": None,
        "require_holdout_evaluation": True,
    }
    with pytest.raises(ValueError, match="The locked test set did not give a clean holdout evaluation"):
        dm._run_deep_compare_task(task)
    with pytest.raises(ValueError, match="Deep repeated-CV fold did not retain a clean holdout evaluation"):
        dm._run_deep_compare_task({**task, "repeat": 1, "fold": 2})


# R9#10 / R8#12 / R9#9 / R9#13 and the R8 note on the Breslow floor: numerics ----------------


def _direct_breslow(times, events, risk, grid) -> np.ndarray:
    """The Breslow baseline cumulative hazard written out term by term (float64)."""
    cumulative, values = 0.0, []
    for t_k in grid:
        at_risk = times >= t_k
        deaths = float(np.sum((times == t_k) & (events == 1)))
        if deaths and at_risk.any():
            cumulative += deaths / float(np.sum(np.exp(risk[at_risk])))
        values.append(cumulative)
    return np.asarray(values)


def test_the_breslow_baseline_matches_the_direct_sum_and_ignores_a_shift_of_the_scores() -> None:
    rng = np.random.default_rng(0)
    n = 300
    times = rng.integers(1, 60, n).astype(float)  # many ties
    events = (rng.random(n) < 0.6).astype(float)
    risk = rng.normal(size=n)
    grid = np.unique(times[events == 1])
    log_h0 = dm._breslow_log_baseline_cumulative_hazard(times, events, risk, grid)
    np.testing.assert_allclose(np.exp(log_h0), _direct_breslow(times, events, risk, grid), rtol=1e-10)
    # Scores below -27 made the old risk-set floor (1e-12) bite; shifting every score by a
    # constant must leave every predicted survival curve unchanged.
    shifted = dm._breslow_log_baseline_cumulative_hazard(times, events, risk - 40.0, grid)
    for score in (risk.min(), 0.0, risk.max()):
        np.testing.assert_allclose(
            dm._survival_from_log_cumulative_hazard(shifted + (score - 40.0)),
            dm._survival_from_log_cumulative_hazard(log_h0 + score),
            rtol=1e-9,
            atol=1e-12,
        )
    # Grid times without an event add nothing; no event at all gives survival 1 everywhere.
    all_times = np.unique(times)
    assert np.array_equal(
        np.exp(dm._breslow_log_baseline_cumulative_hazard(times, events, risk, all_times))[np.isin(all_times, grid)],
        np.exp(log_h0),
    )
    assert np.all(np.isneginf(dm._breslow_log_baseline_cumulative_hazard(times, np.zeros(n), risk, all_times)))


def test_deepsurv_survival_curves_do_not_depend_on_the_breslow_floor(monkeypatch) -> None:
    df = make_example_dataset(seed=9, n_patients=90)
    common = dict(hidden_layers=[4], epochs=1, early_stopping_patience=None, random_seed=4)
    reference = dm.train_deepsurv(df, "os_months", "os_event", FEATURES, **common)
    original_net = dm.DeepSurvNet

    class _ShiftedNet(original_net):
        def forward(self, x):  # the same network with every risk score 40 lower
            return super().forward(x) - 40.0

    monkeypatch.setattr(dm, "DeepSurvNet", _ShiftedNet)
    shifted = dm.train_deepsurv(df, "os_months", "os_event", FEATURES, **common)
    assert shifted["c_index"] == pytest.approx(reference["c_index"], abs=0.01)
    for ours, theirs in zip(shifted["predicted_survival_function"], reference["predicted_survival_function"]):
        np.testing.assert_allclose(ours["curve"]["survival"], theirs["curve"]["survival"], rtol=1e-4, atol=1e-6)


def test_survival_from_the_log_cumulative_hazard_handles_infinite_and_missing_values() -> None:
    survival = dm._survival_from_log_cumulative_hazard(np.array([np.inf, np.nan, -np.inf, 60.0, 0.0]))
    assert survival[0] == 0.0
    assert np.isnan(survival[1])
    assert survival[2] == 1.0 and survival[3] == 0.0
    assert survival[4] == pytest.approx(np.exp(-1.0))


def _all_pairs_ranking_loss(pmf, bins, events) -> float:
    """DeepHit's ranking term as the mean over every comparable pair, built in one piece (float64)."""
    cif = torch.cumsum(pmf.double(), dim=1)
    total, count = 0.0, 0
    censored = events != 1
    for i in torch.where(events == 1)[0].tolist():
        later = (bins > bins[i]) | ((bins == bins[i]) & censored)
        diff = cif[later, bins[i]] - cif[i, bins[i]]
        total += float(torch.nn.functional.softplus(torch.clamp(diff, -20.0, 20.0)).sum())
        count += int(later.sum())
    return total / count


def test_the_deephit_ranking_loss_is_accumulated_per_chunk_without_joining_all_pairs(monkeypatch) -> None:
    torch.manual_seed(3)
    n, n_bins = 400, 6
    pmf = torch.softmax(torch.randn(n, n_bins + 1), dim=1)
    bins = torch.randint(0, n_bins, (n,))
    events = (torch.rand(n) < 0.75).float()  # about 300 events: three chunks of 128
    joined: list[int] = []
    original_cat = torch.cat

    def _spy_cat(tensors, *args, **kwargs):
        joined.append(sum(int(tensor.numel()) for tensor in tensors))
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", _spy_cat)
    loss = dm._deephit_loss(pmf, bins, events, alpha=0.0)
    monkeypatch.undo()
    assert float(loss) == pytest.approx(_all_pairs_ranking_loss(pmf, bins, events), rel=1e-5)
    # Only per-row tables (the survival table and the likelihood terms) are ever joined, never
    # the event-by-subject pair terms (tens of thousands here).
    assert joined and max(joined) <= n * (n_bins + 1)


def test_the_mtlr_calibration_interpolates_the_first_bin_from_its_edge() -> None:
    bin_edges = np.array([10.0, 20.0, 30.0])
    survival = np.array([[1.0, 0.5, 0.2], [1.0, 0.8, 0.6]])
    at_15 = dm._survival_at_reference_time(survival, bin_edges, 15.0, num_time_bins=2)
    np.testing.assert_allclose(at_15, [0.75, 0.9])  # halfway between the edges at 10 and 20
    np.testing.assert_allclose(dm._survival_at_reference_time(survival, bin_edges, 25.0, num_time_bins=2), [0.35, 0.7])
    np.testing.assert_allclose(dm._survival_at_reference_time(survival, bin_edges, 5.0, num_time_bins=2), [1.0, 1.0])


# R8#9 / R8#10 / R8#11: what the single-model summary says about the reported model ---------


def test_no_early_stopping_note_when_the_monitor_never_produced_a_value() -> None:
    df = make_example_dataset(seed=4, n_patients=120)
    data, split = dm._prepare_deep_training_inputs(
        df, time_column="os_months", event_column="os_event", features=FEATURES, random_seed=1
    )
    times, events = data["time_tensor"].numpy(), data["event_tensor"].numpy()
    train = np.asarray(split["train_idx"])
    train_events, train_censored = train[events[train] == 1], train[events[train] == 0]
    # One event, the latest of the training rows, plus censored rows before it: no comparable pair.
    latest_event = train_events[np.argmax(times[train_events])]
    monitor = np.concatenate([[latest_event], train_censored[times[train_censored] < times[latest_event]][:6]])
    with pytest.warns(RuntimeWarning):
        result = dm.train_deepsurv(
            None, "os_months", "os_event", FEATURES, hidden_layers=[4], epochs=3, early_stopping_patience=2,
            prepared_data=data, evaluation_split=split, monitor_indices=monitor, random_seed=1,
        )
    assert result["monitor_history"] == [] and result["best_monitor_epoch"] is None
    # The monitor rows still join the fit, for the full number of epochs.
    assert result["refit_on_training_partition"] and result["epochs_trained"] == 3
    assert not any("Early stopping picked epoch" in strength for strength in result["scientific_summary"]["strengths"])


def test_the_final_loss_is_that_of_the_reported_model() -> None:
    df = make_example_dataset(seed=7, n_patients=150)
    result = dm.train_deepsurv(df, "os_months", "os_event", FEATURES, hidden_layers=[4], epochs=4, early_stopping_patience=2, random_seed=5)
    assert result["refit_on_training_partition"]
    assert _metrics(result)["Final loss"] == pytest.approx(result["refit_loss_history"][-1])
    assert result["refit_loss_history"][-1] != pytest.approx(result["loss_history"][-1])

    first = dm._FitPhase(None, [3.0, 2.0, 1.0, 0.5], [0.6, 0.7, 0.65, 0.64], True)
    final = dm._FitPhase(None, [2.5, 1.5], [], False)
    assert dm._reported_loss_history(first, final, {"refit_on_training_partition": True}, 2) == [2.5, 1.5]
    # Without a refit the restored checkpoint is that of epoch 2.
    assert dm._reported_loss_history(first, first, {"refit_on_training_partition": False}, 2) == [3.0, 2.0]
    summary = dm._scientific_summary_dl(
        "DeepSurv", 0.7, 100, 30, 40, 3, 4, first.loss_history, "holdout", reported_epochs=2, reported_loss_history=[3.0, 2.0],
    )
    assert {metric["label"]: metric["value"] for metric in summary["metrics"]}["Final loss"] == 2.0


def test_apparent_evaluation_reports_no_holdout_c_index() -> None:
    result = dm.train_deepsurv(make_example_dataset(seed=2, n_patients=16), "os_months", "os_event", FEATURES, **TINY)
    assert result["evaluation_mode"] == "apparent"
    assert result["holdout_c_index"] is None
    assert result["c_index"] == result["apparent_c_index"]


# R8#7 / R8#13: dead helpers and optional imports -------------------------------------------


def test_the_module_loads_when_an_optional_library_fails_with_a_dll_error(monkeypatch) -> None:
    import builtins
    from pathlib import Path

    module_path = Path(dm.__file__)
    real_import = builtins.__import__

    def _dll_failure(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch.") or name.startswith("sksurv"):
            raise OSError("[WinError 126] The specified module could not be found")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _dll_failure)
    namespace: dict[str, object] = {"__name__": "deep_models_dll_failure_test", "__file__": str(module_path)}
    exec(compile(module_path.read_text(encoding="utf-8"), str(module_path), "exec"), namespace, namespace)
    assert namespace["TORCH_AVAILABLE"] is False
    assert namespace["_SKSURV_METRICS_AVAILABLE"] is False


def test_the_unused_preprocessing_helpers_are_gone() -> None:
    # The holdout path fits its encoder on the training rows (test_deep_models spies on it).
    assert not hasattr(dm, "_prepare_deep_data")
    assert not hasattr(dm, "_survival_after_event_bins")


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

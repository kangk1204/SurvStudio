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

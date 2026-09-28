from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.encoding import (
    MAX_TEXT_LEVELS,
    canonical_category_values,
    fit_feature_encoder,
    reject_numeric_text_features,
    transform_feature_encoder,
    unseen_category_counts,
)


def _grades(values: list) -> pd.DataFrame:
    return pd.DataFrame({"grade": values, "age": np.arange(len(values), dtype=float) + 50.0})


def test_integer_codes_give_the_same_levels_whether_read_as_whole_numbers_or_decimals() -> None:
    whole = _grades([1, 2, 3, 1, 2, 3, 2, 1])
    decimal = whole.assign(grade=whole["grade"].astype(float))
    decimal.loc[0, "grade"] = np.nan

    from_whole = fit_feature_encoder(whole, ["grade", "age"], ["grade"])
    from_decimal = fit_feature_encoder(decimal, ["grade", "age"], ["grade"])

    assert from_whole["categorical_all_levels"]["grade"] == ["1", "2", "3"]
    assert from_decimal["categorical_all_levels"]["grade"] == ["1", "2", "3"]
    assert from_whole["feature_names"] == ["grade_2", "grade_3", "age"]
    # A model fitted on whole numbers scores the decimal column (and the reverse) by level, not as the reference.
    expected = transform_feature_encoder(whole, from_whole).iloc[1:]
    scored = transform_feature_encoder(decimal, from_whole).iloc[1:]
    pd.testing.assert_frame_equal(scored, expected)
    assert scored[["grade_2", "grade_3"]].to_numpy().sum() == 5
    reverse = transform_feature_encoder(whole, from_decimal)
    assert reverse["grade_2"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert reverse["grade_3"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # The encoder survives a JSON round trip (it is stored inside locked recipes).
    round_trip = json.loads(json.dumps(from_whole))
    np.testing.assert_array_equal(
        transform_feature_encoder(decimal, round_trip, output="numpy"), transform_feature_encoder(decimal, from_whole, output="numpy")
    )


def test_encoders_locked_with_decimal_levels_still_score_whole_number_codes() -> None:
    # An encoder fitted before the levels were canonical, on a code column read as decimals.
    legacy = {
        "features": ["grade", "age"],
        "categorical_features": ["grade"],
        "numeric_features": ["age"],
        "categorical_mappings": {
            "grade": {
                "all_levels": ["1.0", "2.0", "3.0"],
                "baseline_level": "1.0",
                "retained_levels": ["2.0", "3.0"],
                "level_columns": {"2.0": "grade_2.0", "3.0": "grade_3.0"},
                "unknown_column": None,
                "missing_column": "grade__missing",
            }
        },
        "numeric_impute_values": {"age": 60.0},
        "scaler_params": {"age": {"mean": 0.0, "std": 1.0, "impute_value": 60.0}},
        "feature_names": ["grade_2.0", "grade_3.0", "grade__missing", "age"],
        "standardize_numeric": False,
    }

    for frame in (_grades([1, 2, 3, 2]), _grades([1.0, 2.0, 3.0, np.nan]), _grades(["1", "2", "3", None])):
        encoded = transform_feature_encoder(frame, legacy)
        assert list(encoded.columns) == legacy["feature_names"]
        assert encoded["grade_2.0"].tolist()[:3] == [0.0, 1.0, 0.0]
        assert encoded["grade_3.0"].tolist()[:3] == [0.0, 0.0, 1.0]
    assert transform_feature_encoder(_grades([1.0, np.nan]), legacy)["grade__missing"].tolist() == [0.0, 1.0]
    assert unseen_category_counts(_grades([1, 2, 3, 4]), legacy) == {"grade": 1}


def test_decimal_codes_in_a_mixed_column_match_whole_number_levels() -> None:
    encoder = fit_feature_encoder(_grades([1, 2, 3, 1, 2, 3]), ["grade", "age"], ["grade"])
    # 2.5 keeps the column from reading as whole numbers, so its labels are "1.0", "2.0", "2.5".
    mixed = _grades([1.0, 2.0, 2.5, 3.0])
    assert canonical_category_values(mixed["grade"]).tolist() == ["1.0", "2.0", "2.5", "3.0"]

    encoded = transform_feature_encoder(mixed, encoder)

    assert encoded["grade_2"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert encoded["grade_3"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert unseen_category_counts(mixed, encoder) == {"grade": 1}
    assert unseen_category_counts(_grades([1.0, 2.0, 3.0, np.nan]), encoder) == {"grade": 0}


def test_nullable_boolean_and_integer_features_are_imputed_as_numbers() -> None:
    frame = pd.DataFrame(
        {
            "flag": pd.array([True, False, None, True, True], dtype="boolean"),
            "count": pd.array([1, 2, None, 4, 3], dtype="Int64"),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    encoder = fit_feature_encoder(frame, ["flag", "count", "x"])
    encoded = transform_feature_encoder(frame, encoder)

    assert encoder["numeric_features"] == ["flag", "count", "x"]
    assert encoder["numeric_impute_values"] == {"flag": 1.0, "count": 2.5, "x": 3.0}
    assert encoded["flag"].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0]
    assert encoded["count"].tolist() == [1.0, 2.0, 2.5, 4.0, 3.0]
    assert encoded.dtypes.eq(float).all()


def test_numbers_stored_as_text_with_many_levels_are_refused() -> None:
    rng = np.random.default_rng(1)
    psa = [f"{value:.2f}" for value in rng.lognormal(1.0, 1.0, 300)]
    for index in range(0, 300, 4):  # a quarter below the detection limit
        psa[index] = "<0.1"
    frame = pd.DataFrame({"psa": psa, "age": rng.normal(60, 8, 300)})

    with pytest.raises(ValueError, match=r'"psa" is stored as text: 225 of 300 values are numbers but 75 are not \(such as "<0.1"\)'):
        fit_feature_encoder(frame, ["psa", "age"])

    commas = frame.assign(psa=[f"{value:.2f}".replace(".", ",") for value in rng.normal(25, 4, 300)])
    with pytest.raises(ValueError, match="written with commas"):
        fit_feature_encoder(commas, ["psa", "age"])

    ids = frame.assign(psa=[f"PT-{index:04d}" for index in range(300)])
    with pytest.raises(ValueError, match="300 distinct text values") as refused:
        reject_numeric_text_features(ids, ["psa", "age"])
    # A caller that does not pass the categorical list cannot promise that marking the column helps.
    assert "mark it as categorical" not in str(refused.value)

    # Marked categorical, every level is kept; a small code set stays categorical without marking.
    kept = fit_feature_encoder(ids, ["psa", "age"], ["psa"])
    assert len(kept["categorical_all_levels"]["psa"]) == 300
    small = frame.assign(psa=[f"site {index % MAX_TEXT_LEVELS}" for index in range(300)])
    assert len(fit_feature_encoder(small, ["psa"])["feature_names"]) == MAX_TEXT_LEVELS - 1


def test_a_column_marked_categorical_is_not_refused_as_numeric_text() -> None:
    codes = [str(100 + (index % 30)) for index in range(60)] + ["unknown"] * 3
    frame = pd.DataFrame({"site": codes, "age": np.linspace(40, 80, 63)})

    with pytest.raises(ValueError, match='"site" looks numeric'):
        fit_feature_encoder(frame, ["site", "age"])
    encoder = fit_feature_encoder(frame, ["site", "age"], ["site"])

    assert encoder["categorical_features"] == ["site"]
    assert len(encoder["categorical_all_levels"]["site"]) == 31
    reject_numeric_text_features(frame, ["site", "age"], ["site"])


def test_feature_list_and_column_problems_have_clear_messages() -> None:
    frame = _grades([1, 2, 3, 1])
    encoder = fit_feature_encoder(frame, ["grade", "age"], ["grade"])

    with pytest.raises(ValueError, match="listed more than once: age"):
        fit_feature_encoder(frame, ["age", "age"])
    with pytest.raises(ValueError, match="lack the feature column"):
        transform_feature_encoder(frame.drop(columns=["grade"]), encoder)
    with pytest.raises(ValueError, match="lack the feature column"):
        fit_feature_encoder(frame, ["grade", "stage"])
    doubled = pd.concat([frame, frame[["age"]]], axis=1)
    with pytest.raises(ValueError, match="more than one column named age"):
        fit_feature_encoder(doubled, ["grade", "age"], ["grade"])


def test_locked_marker_model_scores_an_external_cohort_whose_codes_read_as_decimals() -> None:
    from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers, validate_locked_recipe

    markers = [f"m{index}" for index in range(5)]
    rng = np.random.default_rng(7)
    n = 240
    grade = rng.integers(1, 4, size=n)
    values = rng.normal(size=(n, len(markers)))
    linear = 0.8 * (grade == 2) + 1.6 * (grade == 3) + 0.8 * values[:, 0]
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    development = pd.DataFrame(
        {
            "os_time": np.minimum(event_time, censor_time),
            "os_event": (event_time <= censor_time).astype(int),
            "grade": grade,
            **{name: values[:, index] for index, name in enumerate(markers)},
        }
    )
    result = evaluate_markers(
        development,
        time_column="os_time",
        event_column="os_event",
        marker_columns=markers,
        clinical_columns=["grade"],
        categorical_clinical=["grade"],
        settings=MarkerSettings(n_permutations=19, n_resamples=4, random_seed=2),
    )
    recipe = result["locked_recipe"]
    assert recipe["clinical"]["encoder"]["categorical_all_levels"] == {"grade": ["1", "2", "3"]}

    # The same patients as an external cohort, with one grade missing: pandas then reads the column as decimals.
    external = development.assign(grade=development["grade"].astype(float))
    external.loc[0, "grade"] = np.nan
    as_decimals = validate_locked_recipe(external, recipe, n_bootstrap=0)
    as_whole_numbers = validate_locked_recipe(development.drop(index=0), recipe, n_bootstrap=0)

    assert as_decimals["cohort"]["n"] == as_whole_numbers["cohort"]["n"] == n - 1
    assert as_decimals["metrics"]["c_index"] == pytest.approx(as_whole_numbers["metrics"]["c_index"], abs=1e-12)
    assert as_decimals["metrics"]["clinical_only_c_index"] == pytest.approx(
        as_whole_numbers["metrics"]["clinical_only_c_index"], abs=1e-12
    )
    assert as_decimals["metrics"]["clinical_only_c_index"] > 0.55

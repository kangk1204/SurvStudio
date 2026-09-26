from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.marker_evaluation as marker_evaluation
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers


def _simulated_cohort(seed: int, *, n: int = 400, n_noise: int = 30) -> pd.DataFrame:
    """Three true markers, one marker confounded with a prognostic clinical covariate, noise."""
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    grade = rng.integers(0, 2, size=n)
    true = rng.normal(size=(n, 3))
    confounded = 0.9 * age + np.sqrt(1 - 0.81) * rng.normal(size=n)
    noise = rng.normal(size=(n, n_noise))
    linear = 0.8 * age + 0.5 * grade + true @ np.array([0.6, -0.6, 0.5])
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(
        {
            "os_time": np.minimum(event_time, censor_time),
            "os_event": (event_time <= censor_time).astype(int),
            "age": age,
            "grade": np.where(grade == 1, "high", "low"),
            "true_up_1": true[:, 0],
            "true_down": true[:, 1],
            "true_up_2": true[:, 2],
            "confounded": confounded,
        }
    )
    for index in range(n_noise):
        frame[f"noise_{index:02d}"] = noise[:, index]
    return frame


def _markers(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"os_time", "os_event", "age", "grade"}]


_FAST = MarkerSettings(n_permutations=199, n_resamples=40, shortlist_size=10, random_seed=7)


def test_true_markers_are_robust_and_the_confounded_marker_is_marginal_only() -> None:
    frame = _simulated_cohort(1)
    result = evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=_markers(frame),
        clinical_columns=["age", "grade"],
        settings=_FAST,
    )
    tiers = {row["marker"]: row["tier"] for row in result["marker_table"]}
    assert result["primary_lens"] == "added_value"
    assert tiers["true_up_1"] == tiers["true_down"] == "robust"
    # The weakest true marker is significant but not always selected in subsamples.
    assert tiers["true_up_2"] in {"robust", "suggestive"}
    assert tiers["confounded"] == "marginal only"
    noise_tiers = [tier for marker, tier in tiers.items() if marker.startswith("noise_")]
    assert noise_tiers.count("robust") == 0
    rows = {row["marker"]: row for row in result["marker_table"]}
    assert rows["true_down"]["direction"] == "higher values, lower hazard"
    assert rows["true_up_1"]["pattern"] == "M+ A+"
    assert rows["confounded"]["pattern"] == "M+ A·"
    # Robust markers were selected in nearly every subsample, in the same direction.
    assert rows["true_up_1"]["added_value"]["selection_frequency"] >= 0.8
    assert rows["true_up_1"]["added_value"]["direction_consistency"] >= 0.95
    exact = rows["true_up_1"]["exact"]["adjusted"]
    assert exact["hazard_ratio"] > 1.0 and exact["lr_p"] < 0.001 and exact["delta_c_apparent"] > 0.0
    assert result["tier_counts"]["robust"] >= 2
    assert sum(result["tier_counts"].values()) == len(result["marker_table"])


def test_signature_and_winners_curse_are_reported_from_left_out_rows() -> None:
    frame = _simulated_cohort(2)
    result = evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=_markers(frame),
        clinical_columns=["age", "grade"],
        settings=_FAST,
    )
    signature = result["signature"]
    assert {"true_up_1", "true_down"} <= set(signature["markers"])
    assert signature["n_signature_replicates"] > 30
    # With real signals the selected signature generalises: left-out C stays close to the
    # in-subsample C and beats the clinical-only model on the same left-out rows.
    assert abs(signature["signature_optimism"]) < 0.05
    assert signature["optimism_corrected_c"] == pytest.approx(signature["apparent_c"] - signature["signature_optimism"])
    assert signature["signature_c_left_out"] > signature["clinical_c_left_out"]
    assert signature["apparent_c"] >= signature["signature_c_left_out"] - 0.05
    assert 0.0 < signature["top_marker_shrinkage"] <= 1.5
    assert signature["n_top_marker_replicates"] == 40


def test_marginal_lens_is_primary_without_clinical_covariates() -> None:
    frame = _simulated_cohort(3, n=300, n_noise=10)
    result = evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=_markers(frame),
        settings=_FAST._replace(n_permutations=99, n_resamples=20),
    )
    assert result["primary_lens"] == "marginal"
    assert "added_value" not in result["marker_table"][0]
    tiers = {row["marker"]: row["tier"] for row in result["marker_table"]}
    assert tiers["true_up_1"] == "robust"


def test_imputation_is_refitted_inside_every_subsample(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _simulated_cohort(4, n=200, n_noise=5)
    frame.loc[frame.index[:15], "true_up_1"] = np.nan
    seen_rows: list[int] = []
    original = marker_evaluation._column_medians

    def spy(block: np.ndarray) -> np.ndarray:
        seen_rows.append(block.shape[0])
        return original(block)

    monkeypatch.setattr(marker_evaluation, "_column_medians", spy)
    result = evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=_markers(frame),
        clinical_columns=["age"],
        settings=_FAST._replace(n_permutations=19, n_resamples=6),
    )
    assert result["cohort"]["n"] == 200
    # Once on the full cohort, then once per subsample on the subsample's rows only.
    assert seen_rows[0] == 200
    assert len(seen_rows) == 7 and all(rows < 200 for rows in seen_rows[1:])


def test_same_seed_gives_the_same_result() -> None:
    frame = _simulated_cohort(5, n=150, n_noise=5)
    settings = _FAST._replace(n_permutations=49, n_resamples=10)
    arguments = dict(time_column="os_time", event_column="os_event", marker_columns=_markers(frame), clinical_columns=["age"], settings=settings)
    first = evaluate_markers(frame, **arguments)
    second = evaluate_markers(frame, **arguments)
    assert first["marker_table"] == second["marker_table"]


def test_marker_inputs_are_validated() -> None:
    frame = _simulated_cohort(6, n=120, n_noise=3)
    frame["label"] = ["a", "b", "c"] * 40
    frame["mostly_missing"] = np.where(np.arange(120) < 60, np.nan, 1.0 * np.arange(120))
    frame["flat"] = 2.0
    common = dict(time_column="os_time", event_column="os_event", settings=_FAST._replace(n_permutations=9, n_resamples=2))
    with pytest.raises(ValueError, match="Markers must be numeric"):
        evaluate_markers(frame, marker_columns=["true_up_1", "label"], **common)
    with pytest.raises(ValueError, match="cannot also be"):
        evaluate_markers(frame, marker_columns=["true_up_1", "age"], clinical_columns=["age"], **common)
    result = evaluate_markers(frame, marker_columns=["true_up_1", "mostly_missing", "flat"], **common)
    dropped = {item["marker"]: item["reason"] for item in result["cohort"]["dropped_markers"]}
    assert dropped == {"mostly_missing": "50% missing", "flat": "constant"}
    with pytest.raises(ValueError, match="resample_fraction"):
        evaluate_markers(frame, marker_columns=["true_up_1"], time_column="os_time", event_column="os_event",
                         settings=MarkerSettings(resample_fraction=0.95))


@pytest.mark.skipif(not os.environ.get("SURVSTUDIO_SLOW_TESTS"), reason="set SURVSTUDIO_SLOW_TESTS=1 to run")
def test_family_wise_error_is_controlled_under_the_global_null() -> None:
    rejections = 0
    n_datasets = 200
    for seed in range(n_datasets):
        rng = np.random.default_rng(1000 + seed)
        n = 80
        age = rng.normal(size=n)
        markers = rng.normal(size=(n, 20)) + 0.5 * age[:, None]
        event_time = rng.exponential(np.exp(-0.7 * age))
        censor = rng.exponential(1.5, size=n)
        frame = pd.DataFrame(markers, columns=[f"m{index}" for index in range(20)])
        frame["os_time"] = np.minimum(event_time, censor)
        frame["os_event"] = (event_time <= censor).astype(int)
        frame["age"] = age
        result = evaluate_markers(
            frame,
            time_column="os_time",
            event_column="os_event",
            marker_columns=[f"m{index}" for index in range(20)],
            clinical_columns=["age"],
            settings=MarkerSettings(n_permutations=199, n_resamples=0, shortlist_size=0, random_seed=seed),
        )
        rejections += any((row["added_value"]["p_fwer"] or 1.0) <= 0.05 for row in result["marker_table"])
    # Markers correlate with the prognostic clinical covariate but add nothing to it.
    assert rejections / n_datasets <= 0.05 + 2.33 * np.sqrt(0.05 * 0.95 / n_datasets)


def _development_result(seed: int = 1) -> dict:
    frame = _simulated_cohort(seed)
    return evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=_markers(frame),
        clinical_columns=["age", "grade"],
        settings=_FAST,
    )


def test_locked_recipe_replicates_true_markers_in_an_independent_cohort() -> None:
    from survival_toolkit.marker_evaluation import validate_locked_recipe

    recipe = _development_result()["locked_recipe"]
    assert {"true_up_1", "true_down"} <= set(recipe["markers"])
    assert "confounded" not in recipe["markers"]
    external = _simulated_cohort(11, n=300)
    report = validate_locked_recipe(external, recipe)
    metrics = report["metrics"]
    assert report["recipe_hash"] == recipe["recipe_hash"]
    assert metrics["c_index"] > metrics["clinical_only_c_index"]
    assert metrics["delta_c_index_ci"][0] is not None and metrics["delta_c_index_ci"][0] > 0.0
    assert 0.6 < metrics["calibration_slope"] < 1.4
    assert 0.7 < metrics["observed_expected_ratio"] < 1.3
    assert 0.0 < metrics["brier_skill"] < 1.0
    replicated = {row["marker"]: row["replicated"] for row in report["markers"]}
    assert replicated["true_up_1"] and replicated["true_down"]


def test_locked_recipe_rejects_edits_and_survives_json() -> None:
    import json

    from survival_toolkit.marker_evaluation import validate_locked_recipe

    recipe = _development_result()["locked_recipe"]
    restored = json.loads(json.dumps(recipe))
    external = _simulated_cohort(12, n=200)
    assert validate_locked_recipe(external, restored)["recipe_hash"] == recipe["recipe_hash"]
    tampered = json.loads(json.dumps(recipe))
    tampered["model"]["coefficients"][0] += 0.1
    with pytest.raises(ValueError, match="edited after it was locked"):
        validate_locked_recipe(external, tampered)


def test_locked_recipe_maps_external_column_names_and_notes_imputation() -> None:
    from survival_toolkit.marker_evaluation import validate_locked_recipe

    recipe = _development_result()["locked_recipe"]
    external = _simulated_cohort(13, n=200).rename(columns={"os_time": "follow_up", "os_event": "died", "true_up_1": "gene_a"})
    external.loc[external.index[:5], "gene_a"] = np.nan
    mapping = {"os_time": "follow_up", "os_event": "died", "true_up_1": "gene_a"}
    report = validate_locked_recipe(external, recipe, column_mapping=mapping)
    assert report["cohort"]["n"] == 200
    assert any("5 missing true_up_1 value(s) imputed" in note for note in report["notes"])
    with pytest.raises(ValueError, match="lacks columns"):
        validate_locked_recipe(external, recipe)


def _sksurv_available() -> bool:
    try:
        import sksurv  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sksurv_available(), reason="scikit-survival is not installed")
def test_nonlinear_lens_flags_a_u_shaped_marker_the_cox_lenses_miss() -> None:
    rng = np.random.default_rng(21)
    n = 400
    age = rng.normal(size=n)
    u_shaped = rng.normal(size=n)
    noise = rng.normal(size=(n, 8))
    linear = 0.6 * age + 1.0 * (u_shaped**2 - 1.0)
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(noise, columns=[f"noise_{index}" for index in range(8)])
    frame["u_shaped"] = u_shaped
    frame["age"] = age
    frame["os_time"] = np.minimum(event_time, censor_time)
    frame["os_event"] = (event_time <= censor_time).astype(int)
    result = evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=["u_shaped", *[f"noise_{index}" for index in range(8)]],
        clinical_columns=["age"],
        settings=_FAST._replace(n_permutations=99, n_resamples=10, nonlinear_lens="gbs", nonlinear_replicates=12),
    )
    rows = {row["marker"]: row for row in result["marker_table"]}
    assert result["nonlinear_lens"]["available"] is True
    # A symmetric U-shape has no linear trend, so the score lenses see nothing ...
    assert rows["u_shaped"]["tier"] == "not supported"
    # ... while the tree model's out-of-sample importance is consistently positive.
    assert rows["u_shaped"]["pattern"].endswith("N+")
    assert rows["u_shaped"]["nonlinear"]["positive_fraction"] >= 0.8
    assert all(not rows[f"noise_{index}"]["pattern"].endswith("N+") for index in range(8))

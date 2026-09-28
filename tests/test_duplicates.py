from __future__ import annotations

import numpy as np
import pandas as pd

from survival_toolkit.duplicates import MIN_MARKERS, possible_duplicates
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers


def _expression(n_patients: int, n_markers: int, seed: int = 0) -> np.ndarray:
    """Expression-like data: five subtypes sharing latent programmes, so different patients correlate."""
    rng = np.random.default_rng(seed)
    subtype = rng.integers(0, 5, n_patients)
    programmes = rng.normal(size=(5, n_markers)) * 1.5
    return programmes[subtype] + rng.normal(size=(n_patients, n_markers))


def test_a_tumour_profiled_twice_is_flagged_and_different_patients_are_not() -> None:
    values = _expression(150, 1000)
    rng = np.random.default_rng(1)
    values[97] = values[12] + rng.normal(scale=0.3, size=values.shape[1])
    labels = [f"P{index:03d}" for index in range(150)]

    report = possible_duplicates(values, labels)

    assert report["checked"] and report["markers_used"] == 1000
    assert [(pair["a"], pair["b"]) for pair in report["pairs"]] == [("P012", "P097")]
    assert report["pairs"][0]["r"] > 0.9 and report["pairs"][0]["gap"] >= 0.2
    assert report["identical"] == [] and report["n_pairs"] == 1


def test_no_pair_is_flagged_among_distinct_patients_of_the_same_subtypes() -> None:
    report = possible_duplicates(_expression(300, 2000, seed=3), [str(index) for index in range(300)])
    assert report["checked"] and report["pairs"] == []


def test_identical_rows_are_reported_for_continuous_panels_only() -> None:
    rng = np.random.default_rng(2)
    continuous = rng.normal(size=(40, 30))
    continuous[20] = continuous[3]
    report = possible_duplicates(continuous, [f"row {index + 1}" for index in range(40)])
    assert report["identical"] == [["row 4", "row 21"]] and report["n_identical"] == 1
    assert not report["checked"] and str(MIN_MARKERS) in report["note"]

    # Mutation calls: patients share profiles by chance, so they are not reported.
    binary = rng.integers(0, 2, size=(200, 25)).astype(float)
    assert possible_duplicates(binary, [str(index) for index in range(200)])["identical"] == []


def test_patients_missing_most_markers_are_listed_and_not_compared() -> None:
    values = _expression(150, 400, seed=3)
    labels = [f"P{index:03d}" for index in range(150)]
    # Two patients without any measurement are "identical" and correlate perfectly once median-filled.
    values[[40, 90]] = np.nan
    values[60, :300] = np.nan  # 75% missing
    values[70, :100] = np.nan  # 25% missing: still screened

    report = possible_duplicates(values, labels)

    assert report["identical"] == [] and report["pairs"] == []
    assert report["mostly_missing"] == ["P040", "P060", "P090"] and report["n_mostly_missing"] == 3
    assert report["checked"]
    # A duplicated partly measured patient is still found as an identical row.
    values[120] = values[70]
    assert possible_duplicates(values, labels)["identical"] == [["P070", "P120"]]


def test_identical_rows_are_found_when_rounding_leaves_a_negative_zero() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(30, 25))
    values[5] = values[3]
    values[3, 0], values[5, 0] = 1e-10, -1e-10  # both round to zero, one of them to -0.0
    assert possible_duplicates(values, [str(index) for index in range(30)])["identical"] == [["3", "5"]]


def test_the_marker_evaluation_names_repeated_patients_by_their_id_column() -> None:
    rng = np.random.default_rng(5)
    n_patients, n_genes = 160, 300
    genes = _expression(n_patients, n_genes, seed=5)
    genes[140] = genes[7] + rng.normal(scale=0.3, size=n_genes)
    frame = pd.DataFrame(genes, columns=[f"G{index}" for index in range(n_genes)])
    frame.insert(0, "patient_id", [f"TCGA-{index:04d}" for index in range(n_patients)])
    frame["time"] = rng.exponential(30.0, n_patients) + 0.5
    frame["event"] = rng.integers(0, 2, n_patients)
    frame["age"] = rng.normal(60, 8, n_patients)

    result = evaluate_markers(
        frame, time_column="time", event_column="event", marker_columns=[f"G{index}" for index in range(n_genes)],
        clinical_columns=["age"], event_positive_value=1, id_column="patient_id",
        settings=MarkerSettings(n_permutations=19, n_resamples=4, random_seed=1),
    )

    pairs = result["duplicates"]["pairs"]
    assert [(pair["a"], pair["b"]) for pair in pairs] == [("TCGA-0007", "TCGA-0140")]
    # Without the ID column, patients are named by their row in the dataset.
    unnamed = evaluate_markers(
        frame, time_column="time", event_column="event", marker_columns=[f"G{index}" for index in range(n_genes)],
        clinical_columns=["age"], event_positive_value=1, settings=MarkerSettings(n_permutations=19, n_resamples=4, random_seed=1),
    )
    assert [(pair["a"], pair["b"]) for pair in unnamed["duplicates"]["pairs"]] == [("row 8", "row 141")]

from __future__ import annotations

import pytest

from survival_toolkit.design_audit import (
    GRID_K,
    GRID_M,
    GRID_N,
    MAIN_SCENARIO,
    Cohort,
    StudyDesign,
    audit_design,
    design_from_dict,
    load_map,
)
from survival_toolkit.errors import UserInputError


def _cell(features: str, m: int, k: int, n: int, with_training: bool, scenario: str = MAIN_SCENARIO) -> dict:
    grid = load_map()
    row = grid[
        (grid["scenario"] == scenario)
        & (grid["features"] == features)
        & (grid["m"] == m)
        & (grid["k"] == k)
        & (grid["validation_n"] == n)
        & (grid["with_training"] == int(with_training))
    ]
    assert len(row) == 1
    return row.iloc[0].to_dict()


def _cohorts(k: int, n: int, events: int | None = None) -> tuple[Cohort, ...]:
    return tuple(Cohort(f"cohort {index}", n, events) for index in range(k))


def _codes(result: dict) -> list[str]:
    return [flag["code"] for flag in result["flags"]]


def test_map_covers_the_pilot_grid():
    grid = load_map()
    assert len(grid) == 288
    assert set(grid["k"]) == set(GRID_K) and set(grid["validation_n"]) == set(GRID_N) and set(grid["m"]) == set(GRID_M)
    assert (grid.groupby(["scenario", "features", "m", "with_training"]).size() == len(GRID_K) * len(GRID_N)).all()
    assert (grid["replicates"] == 100).all()
    assert not grid.isna().any().any()


def test_grid_points_return_the_map_values():
    design = StudyDesign(candidate_models=40, gene_only=True, selection_cohorts=_cohorts(3, 150), sealed_cohorts=_cohorts(1, 500))
    cell = _cell("genes only", 40, 3, 150, False)
    result = audit_design(design)
    assert result["expected_optimism"]["value"] == pytest.approx(cell["optimism"], abs=1e-9)
    assert result["expected_regret"]["value"] == pytest.approx(cell["regret"], abs=1e-9)
    by_events = audit_design(design._replace(selection_cohorts=_cohorts(3, 150, round(cell["validation_events"]))))
    assert by_events["placement"]["size_unit"] == "events"
    assert by_events["expected_optimism"]["value"] == pytest.approx(cell["optimism"], abs=2e-3)


def test_headline_average_including_training_matches_the_selection_average():
    design = StudyDesign(
        candidate_models=12, gene_only=False, selection_cohorts=_cohorts(3, 300), training_in_selection=True, headline="average including training"
    )
    cell = _cell("genes + clinical", 12, 3, 300, True)
    assert audit_design(design)["expected_optimism"]["value"] == pytest.approx(cell["selection_optimism"], abs=2e-4)
    training = audit_design(design._replace(headline="training"))["expected_optimism"]["value"]
    assert training == pytest.approx(cell["training_optimism"], abs=1e-9)


def test_estimates_between_grid_points_lie_between_the_neighbours():
    design = StudyDesign(candidate_models=20, gene_only=True, selection_cohorts=_cohorts(2, 200))
    value = audit_design(design)["expected_optimism"]["value"]
    corners = [_cell("genes only", m, k, n, False)["optimism"] for m in GRID_M for k in (1, 3) for n in (150, 300)]
    assert min(corners) <= value <= max(corners)


def test_single_cohort_best_of_101_design_is_flagged():
    design = StudyDesign(
        candidate_models=101,
        gene_only=True,
        selection_cohorts=(Cohort("GSE72094", 214, 80),),
        training_in_selection=True,
        headline="average including training",
    )
    result = audit_design(design)
    assert {"NO_SEALED_COHORT", "TRAINING_IN_SELECTION", "HEADLINE_INCLUDES_TRAINING", "SINGLE_SELECTION_COHORT"} <= set(_codes(result))
    assert {"SMALL_SELECTION_COHORTS", "NO_CLINICAL_COMPARISON", "MANY_CANDIDATES"} <= set(_codes(result))
    assert result["high_risk_flags"] == 4
    assert [flag["severity"] for flag in result["flags"]][:4] == ["high"] * 4
    regret = result["expected_regret"]
    assert regret["value"] > regret["if_chosen_on_selection_cohorts_only"]
    assert result["expected_optimism"]["value"] > 0.05
    low, high = result["expected_optimism"]["range"]
    assert low <= result["expected_optimism"]["value"] <= high


def test_careful_design_has_no_high_risk_flags():
    design = StudyDesign(
        candidate_models=12,
        gene_only=False,
        selection_cohorts=_cohorts(5, 300, 150),
        sealed_cohorts=(Cohort("sealed", 400, 150),),
        compared_with_clinical=True,
    )
    result = audit_design(design)
    assert _codes(result) == ["HEADLINE_FROM_SELECTION_COHORTS"] and result["high_risk_flags"] == 0
    assert result["expected_optimism"]["value"] < 0.01
    sealed = audit_design(design._replace(headline="sealed"))
    assert sealed["flags"] == [] and sealed["expected_optimism"]["value"] == 0.0


def test_designs_outside_the_map_use_its_edge():
    design = StudyDesign(candidate_models=40, gene_only=True, selection_cohorts=_cohorts(8, 600), sealed_cohorts=_cohorts(1, 300))
    result = audit_design(design)
    assert "OUTSIDE_MAP" in _codes(result)
    assert len(result["placement"]["outside_map"]) == 2
    assert result["expected_optimism"]["value"] == pytest.approx(_cell("genes only", 40, 5, 300, False)["optimism"], abs=1e-9)


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_models": 1},
        {"selection_cohorts": ()},
        {"selection_cohorts": (Cohort("a", 100, 120),)},
        {"headline": "best"},
        {"headline": "sealed"},
    ],
)
def test_invalid_designs_are_rejected(changes):
    design = StudyDesign(candidate_models=12, gene_only=True, selection_cohorts=_cohorts(3, 150))
    with pytest.raises(UserInputError):
        audit_design(design._replace(**changes))


def test_design_from_dict_reads_json_input():
    design = design_from_dict(
        {
            "candidate_models": 101,
            "gene_only": True,
            "selection_cohorts": [{"name": "GSE72094", "n": 398, "events": 122}, {"n": "83"}],
            "training_in_selection": True,
            "headline": "average including training",
        }
    )
    assert design.selection_cohorts == (Cohort("GSE72094", 398, 122), Cohort("selection_cohorts 2", 83, None))
    assert design.sealed_cohorts == () and design.training_in_selection is True
    assert audit_design(design)["placement"]["size_unit"] == "patients"


@pytest.mark.parametrize(
    "values",
    [
        {"candidate_models": 12, "gene_only": True},
        {"candidate_models": 12, "gene_only": "false", "selection_cohorts": [{"n": 100}]},
        {"candidate_models": 12.5, "gene_only": True, "selection_cohorts": [{"n": 100}]},
        {"candidate_models": 12, "gene_only": True, "selection_cohorts": [{"n": 100}], "winner": "RSF"},
        {"candidate_models": 12, "gene_only": True, "selection_cohorts": [{"events": 40}]},
    ],
)
def test_design_from_dict_rejects_malformed_input(values):
    with pytest.raises(UserInputError):
        design_from_dict(values)

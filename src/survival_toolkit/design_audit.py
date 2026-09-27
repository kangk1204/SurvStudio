"""Audit the design of a multi-algorithm prognostic-signature study.

Many prognostic signatures are the best of tens to 101 model-building pipelines, chosen by their
C-index in other cohorts. How optimistic the reported C-index is, and how much worse the chosen model
is than the best candidate (regret), depend on the design: the number and size of the cohorts used for
the choice, whether the models use genes only, the number of candidates, and whether the training
cohort's apparent C-index entered the choice. ``audit_design`` places a design on a simulation map of
these factors and flags practices that make the reported performance unreliable.

The map (``data/design_audit_map.csv``) is the benchmark pilot of 2026-09: four data-generating
scenarios (no or linear marker effects, homogeneous or heterogeneous cohorts), 100 replicates each,
12 or 40 candidate pipelines fitted on a training cohort of 300 patients, and 1, 3 or 5 selection
cohorts of 100, 150 or 300 patients. Estimates are interpolated linearly in log K, log cohort size and
log M, held constant beyond the edges of the map, and reported for the heterogeneous linear scenario
with the range over all four.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from survival_toolkit.errors import UserInputError, user_input_boundary

MAP_PATH = Path(__file__).resolve().parent / "data" / "design_audit_map.csv"
MAP_VERSION = "benchmark pilot 2026-09: 4 scenarios x 100 replicates, M in {12, 40}"
MAIN_SCENARIO = "linear_heterogeneous"
GRID_K = (1, 3, 5)
GRID_N = (100, 150, 300)
GRID_M = (12, 40)
TRAINING_N = 300
# usual minimum for validating a survival model (Collins, Ogundimu & Altman, Stat Med 2016)
MIN_VALIDATION_EVENTS = 100
HEADLINES = ("validation", "average including training", "training", "sealed")
HEADLINE_NOTES = {
    "validation": "Mean C-index over the cohorts used to choose the model, minus the chosen model's C-index in new cohorts.",
    "average including training": "Mean C-index over the training and selection cohorts, minus the chosen model's C-index in new cohorts.",
    "training": f"Apparent C-index in the training cohort minus the chosen model's C-index in new cohorts (training cohorts of {TRAINING_N} patients in the map).",
    "sealed": "A cohort that was not used for the choice: its C-index is not inflated by it.",
}
BOOLEAN_FIELDS = ("gene_only", "training_in_selection", "prefilter_used_validation_outcomes", "refit_in_validation", "cutoff_per_cohort", "compared_with_clinical")
SEVERITY_ORDER = {"high": 0, "moderate": 1, "info": 2}


class Cohort(NamedTuple):
    name: str
    n: int
    events: int | None = None


class StudyDesign(NamedTuple):
    """How the winning model was chosen and how its performance was reported.

    selection_cohorts are the cohorts whose C-index entered the choice of the winner, not counting the
    training cohort; training_in_selection says whether the training cohort's C-index entered it too.
    sealed_cohorts were not used to choose, pre-select or tune anything. headline is the C-index the
    paper presents as the model's performance.
    """

    candidate_models: int
    gene_only: bool
    selection_cohorts: tuple[Cohort, ...]
    training_in_selection: bool = False
    sealed_cohorts: tuple[Cohort, ...] = ()
    headline: str = "validation"
    prefilter_used_validation_outcomes: bool = False
    refit_in_validation: bool = False
    cutoff_per_cohort: bool = False
    compared_with_clinical: bool = False


class Flag(NamedTuple):
    code: str
    severity: str
    message: str
    remedy: str


class _Placement(NamedTuple):
    k: int
    size: float
    axis: str
    m: int
    outside: tuple[str, ...]


@lru_cache(maxsize=1)
def load_map() -> pd.DataFrame:
    return pd.read_csv(MAP_PATH)


def _placement(design: StudyDesign, grid: pd.DataFrame) -> _Placement:
    """Where the design sits on the map: selection cohorts are sized by their median events when every
    event count is known, by their median number of patients otherwise."""
    cohorts = design.selection_cohorts
    by_events = all(cohort.events is not None for cohort in cohorts)
    axis = "validation_events" if by_events else "validation_n"
    size = float(np.median([cohort.events if by_events else cohort.n for cohort in cohorts]))
    sizes = grid.loc[grid["scenario"] == MAIN_SCENARIO, axis]
    outside = []
    if len(cohorts) > GRID_K[-1]:
        outside.append(f"{len(cohorts)} selection cohorts (map: {GRID_K[0]} to {GRID_K[-1]})")
    if not sizes.min() <= size <= sizes.max():
        unit = "events" if by_events else "patients"
        outside.append(f"median {size:.0f} {unit} per selection cohort (map: {sizes.min():.0f} to {sizes.max():.0f})")
    if design.candidate_models < GRID_M[0]:
        outside.append(f"{design.candidate_models} candidate models (map: {GRID_M[0]} to {GRID_M[-1]})")
    return _Placement(len(cohorts), size, axis, int(design.candidate_models), tuple(outside))


def _surface(cells: pd.DataFrame, column: str, place: _Placement) -> float:
    """One scenario's value at the design's K, cohort size and M."""
    by_m = []
    for m in GRID_M:
        at_m = cells[cells["m"] == m]
        table = at_m.pivot(index="k", columns="validation_n", values=column).reindex(index=GRID_K, columns=GRID_N)
        sizes = at_m.groupby("validation_n")[place.axis].mean().reindex(GRID_N).to_numpy(dtype=float)
        along_size = [np.interp(np.log(place.size), np.log(sizes), table.loc[k].to_numpy(dtype=float)) for k in GRID_K]
        by_m.append(np.interp(np.log(place.k), np.log(GRID_K), along_size))
    return float(np.interp(np.log(place.m), np.log(GRID_M), by_m))


def _summary(values: dict[str, float]) -> dict[str, Any]:
    return {"value": round(values[MAIN_SCENARIO], 4), "range": (round(min(values.values()), 4), round(max(values.values()), 4))}


def expected_effects(design: StudyDesign) -> dict[str, Any]:
    """Expected optimism of the headline C-index and expected regret of the choice."""
    grid = load_map()
    features = "genes only" if design.gene_only else "genes + clinical"
    place = _placement(design, grid)
    cells = grid[grid["features"] == features]
    scenarios = sorted(cells["scenario"].unique())

    def over_scenarios(column: str, with_training: bool) -> dict[str, float]:
        chosen = cells[cells["with_training"] == int(with_training)]
        return {scenario: _surface(chosen[chosen["scenario"] == scenario], column, place) for scenario in scenarios}

    trained = design.training_in_selection
    if design.headline == "sealed":
        optimism = dict.fromkeys(scenarios, 0.0)
    elif design.headline == "training":
        optimism = over_scenarios("training_optimism", trained)
    else:
        optimism = over_scenarios("optimism", trained)
        if design.headline == "average including training":
            training = over_scenarios("training_optimism", trained)
            optimism = {scenario: (place.k * optimism[scenario] + training[scenario]) / (place.k + 1) for scenario in scenarios}
    regret = over_scenarios("regret", trained)
    return {
        "expected_optimism": {"headline": design.headline, **_summary(optimism), "note": HEADLINE_NOTES[design.headline]},
        "expected_regret": {
            **_summary(regret),
            "if_chosen_on_selection_cohorts_only": round(over_scenarios("regret", False)[MAIN_SCENARIO], 4),
            "note": "Best candidate's C-index in new cohorts minus the chosen model's.",
        },
        "placement": {
            "selection_cohorts": place.k,
            "median_cohort_size": place.size,
            "size_unit": "events" if place.axis == "validation_events" else "patients",
            "candidate_models": place.m,
            "features": features,
            "outside_map": list(place.outside),
        },
    }


def _flags(design: StudyDesign, effects: dict[str, Any]) -> list[Flag]:
    flags: list[Flag] = []
    placement = effects["placement"]
    if not design.sealed_cohorts:
        flags.append(Flag("NO_SEALED_COHORT", "high", "No cohort was kept out of the choice, so every reported validation C-index helped pick the model and is optimistic.", "Keep at least one cohort sealed until the model is locked, and report its C-index as the validation result."))
    if design.training_in_selection:
        regret = effects["expected_regret"]
        loss = regret["value"] - regret["if_chosen_on_selection_cohorts_only"]
        flags.append(Flag("TRAINING_IN_SELECTION", "high", f"The training cohort's apparent C-index entered the choice, which favours over-fitted models: in the simulation map the chosen model's C-index in new cohorts is {loss:.3f} lower than with a choice on the selection cohorts alone.", "Choose on the selection cohorts or on cross-validated performance only."))
    if design.headline in ("average including training", "training"):
        flags.append(Flag("HEADLINE_INCLUDES_TRAINING", "high", "The headline C-index includes the training cohort, whose apparent C-index is optimistic by construction.", "Headline the sealed-cohort C-index, or the mean over cohorts that were not used for the choice."))
    elif design.headline == "validation" and design.sealed_cohorts:
        flags.append(Flag("HEADLINE_FROM_SELECTION_COHORTS", "moderate", "A sealed cohort exists, but the headline C-index comes from the cohorts used for the choice.", "Headline the sealed-cohort C-index."))
    if len(design.selection_cohorts) == 1:
        flags.append(Flag("SINGLE_SELECTION_COHORT", "high", "The model was chosen on a single cohort, the most optimistic design in the map.", "Choose on several cohorts or by cross-validation, and keep another cohort sealed."))
    if placement["size_unit"] == "events" and placement["median_cohort_size"] < MIN_VALIDATION_EVENTS:
        flags.append(Flag("SMALL_SELECTION_COHORTS", "moderate", f"The cohorts used for the choice are small (median {placement['median_cohort_size']:.0f} events; {MIN_VALIDATION_EVENTS} is the usual minimum for validation), so the choice is noisy: the winner's C-index in them is inflated and a weaker model wins more often.", "Use larger or more cohorts for the choice and report confidence intervals for the C-index."))
    elif placement["size_unit"] == "patients" and placement["median_cohort_size"] < 2 * MIN_VALIDATION_EVENTS:
        flags.append(Flag("SMALL_SELECTION_COHORTS", "moderate", f"The cohorts used for the choice are small (median {placement['median_cohort_size']:.0f} patients, event counts not given), so the choice is noisy: the winner's C-index in them is inflated and a weaker model wins more often.", "Report the number of events per cohort; use larger or more cohorts for the choice."))
    if design.prefilter_used_validation_outcomes:
        flags.append(Flag("PREFILTER_LEAK", "high", "Outcomes of validation cohorts were used to pre-select genes, so those cohorts are not independent of the model.", "Pre-select genes in the training cohort only."))
    if design.refit_in_validation:
        flags.append(Flag("REFIT_IN_VALIDATION", "high", "Coefficients were re-estimated in validation cohorts; that is re-derivation, not validation.", "Apply the locked model unchanged, for example with validate_locked_recipe."))
    if design.cutoff_per_cohort:
        flags.append(Flag("CUTOFF_PER_COHORT", "moderate", "Risk-group cut-offs chosen within each cohort exaggerate the separation of the groups.", "Fix the cut-off in the training cohort, or analyse the continuous risk score."))
    if not design.compared_with_clinical:
        flags.append(Flag("NO_CLINICAL_COMPARISON", "moderate", "The signature was not compared with a model of the clinical covariates alone.", "Report the added value over the clinical model (C-index gain or likelihood-ratio test)."))
    if design.candidate_models > GRID_M[-1]:
        flags.append(Flag("MANY_CANDIDATES", "info", f"{design.candidate_models} candidates is more than the map covers ({GRID_M[-1]}); the estimates are for {GRID_M[-1]} candidates and likely too low.", "Report how many candidates competed and how the winner was chosen."))
    if placement["outside_map"]:
        flags.append(Flag("OUTSIDE_MAP", "info", f"Outside the simulation map: {'; '.join(placement['outside_map'])}. The estimates are taken at the nearest edge.", "Treat the numeric estimates as indicative."))
    return sorted(flags, key=lambda flag: SEVERITY_ORDER[flag.severity])


def _validated(design: StudyDesign) -> StudyDesign:
    if design.candidate_models < 2:
        raise UserInputError("A choice needs at least 2 candidate models.")
    if not design.selection_cohorts:
        raise UserInputError("Give at least one cohort whose C-index was used to choose the model.")
    for cohort in (*design.selection_cohorts, *design.sealed_cohorts):
        if cohort.n < 2 or (cohort.events is not None and not 1 <= cohort.events <= cohort.n):
            raise UserInputError(f"Cohort {cohort.name!r}: it needs at least 2 patients, and events between 1 and the number of patients.")
    if design.headline not in HEADLINES:
        raise UserInputError(f"The headline must be one of: {', '.join(HEADLINES)}.")
    if design.headline == "sealed" and not design.sealed_cohorts:
        raise UserInputError("A sealed-cohort headline needs at least one sealed cohort.")
    return design


def design_from_dict(values: dict[str, Any]) -> StudyDesign:
    """A StudyDesign from JSON input, with cohorts as lists of {"name", "n", "events"}."""

    def whole(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise UserInputError(f"{label} must be a whole number.")
        try:
            number = float(value)
        except ValueError:
            raise UserInputError(f"{label} must be a whole number.") from None
        if not number.is_integer():
            raise UserInputError(f"{label} must be a whole number.")
        return int(number)

    def cohorts(items: Any, label: str) -> tuple[Cohort, ...]:
        if not isinstance(items, (list, tuple)) or not all(isinstance(item, dict) for item in items):
            raise UserInputError(f"{label} must be a list of cohorts with a size n.")
        return tuple(
            Cohort(
                str(item.get("name") or f"{label} {index + 1}"),
                whole(item.get("n"), "Cohort size n"),
                None if item.get("events") in (None, "") else whole(item["events"], "Events"),
            )
            for index, item in enumerate(items)
        )

    unknown = sorted(set(values) - set(StudyDesign._fields))
    if unknown:
        raise UserInputError(f"Unknown design fields: {', '.join(unknown)}.")
    missing = [name for name in ("candidate_models", "gene_only", "selection_cohorts") if name not in values]
    if missing:
        raise UserInputError(f"The design needs {', '.join(missing)}.")
    fields = dict(values)
    for name in BOOLEAN_FIELDS:
        if name in fields and not isinstance(fields[name], bool):
            raise UserInputError(f"{name} must be true or false.")
    fields["candidate_models"] = whole(values["candidate_models"], "candidate_models")
    fields["selection_cohorts"] = cohorts(values["selection_cohorts"], "selection_cohorts")
    fields["sealed_cohorts"] = cohorts(values.get("sealed_cohorts", []), "sealed_cohorts")
    return StudyDesign(**fields)


@user_input_boundary
def audit_design(design: StudyDesign) -> dict[str, Any]:
    """Expected optimism and regret of a multi-algorithm design, with flagged practices and remedies."""
    design = _validated(design)
    effects = expected_effects(design)
    flags = _flags(design, effects)
    return {
        **effects,
        "flags": [flag._asdict() for flag in flags],
        "high_risk_flags": sum(flag.severity == "high" for flag in flags),
        "map": {"version": MAP_VERSION, "main_scenario": MAIN_SCENARIO},
    }

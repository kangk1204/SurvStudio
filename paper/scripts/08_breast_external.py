"""Case study IV (validation): the locked METABRIC model applied unchanged to the MetaGxBreast cohorts.

External cohorts (common.screen_validation_cohorts, in alphabetical order): every MetaGxBreast cohort other than
METABRIC with overall survival, at least 30 deaths within 10 years among its complete cases for the five clinical
covariates (TCGA lacks tumour size, node status and grade), and at least half of the model's marker weight
(|coefficient| x development SD, SurvStudio's rule) measured on its platform; the screen checks the weight before the
cohort is scored and records why a cohort was left out. Patients who are, through any chain of listed duplicates, a
METABRIC development patient or a patient of a cohort used before are removed first. Each cohort is scored with the
markers rescaled within the cohort (the platforms differ from METABRIC's Illumina arrays).

Conventions of the export (common.py): METABRIC's development patients have cBioPortal's overall survival (script
07). UNC4 stores follow-up in multiples of 30 days, converted with 30 days a month (the others with 365.25 / 12),
and records tumour size in classes, treated as not recorded. STNO2 records the T category (1 to 4) in place of tumour
size; it is mapped to centimetres by the median METABRIC tumour of the matching range (T1 at most 2 cm, T2 over 2 and
up to 5 cm, T3 and T4 over 5 cm), recorded in breast_external_pooled.json.
"""

from __future__ import annotations

import json

import pandas as pd

from common import (
    RESULTS,
    check_marker_weight,
    pooled_validation,
    read_result,
    screen_validation_cohorts,
    survstudio_version,
    t_category_cm,
    validation_marker_rows,
    validation_row,
    write_csv_atomic,
    write_json,
)
from survival_toolkit.marker_evaluation import validate_locked_recipe


def main() -> None:
    recipe = read_result("breast_locked_model.json")
    screened, used = screen_validation_cohorts("IV", recipe)
    entries = {entry["cohort"]: entry for entry in screened}
    cohort_rows, marker_rows = [], []
    for cohort, external in used.items():
        report = validate_locked_recipe(external, recipe, marker_scaling="within_cohort")
        check_marker_weight(entries[cohort], report)
        cohort_rows.append({"cohort": cohort, **validation_row(report)})
        marker_rows.extend({"cohort": cohort, **row} for row in validation_marker_rows(report))
    cohorts = pd.DataFrame(cohort_rows)
    write_csv_atomic(cohorts, RESULTS / "breast_external_validation.csv")
    markers = pd.DataFrame(marker_rows)
    write_csv_atomic(markers, RESULTS / "breast_external_markers.csv")

    pooled = {
        "survstudio": survstudio_version(),
        "screened": screened,
        "tumour_size_t_category_cm": {f"T{int(category)}": size for category, size in t_category_cm().items()},
        "patients": int(cohorts["n"].sum()),
        "events": int(cohorts["events"].sum()),
        # Each C is pooled with its own interval: the model's, the clinical-only model's and the difference's.
        **pooled_validation(cohorts),
        "markers_replicated_somewhere": sorted(markers.loc[markers["replicated"], "marker"].unique().tolist()),
        "same_direction_share": float(markers.loc[markers["measured"], "same_direction"].mean()) if len(markers) else None,
    }
    write_json(RESULTS / "breast_external_pooled.json", pooled)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 100):
        print(pd.DataFrame(screened).to_string(index=False))
        print(cohorts.round(3).to_string(index=False))
    print(json.dumps({key: value for key, value in pooled.items() if key not in {"survstudio", "screened"}}, indent=1, default=float))


if __name__ == "__main__":
    main()

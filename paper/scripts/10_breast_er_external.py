"""Case study V (validation): the locked METABRIC ER-positive recurrence model applied unchanged to the
MetaGxBreast cohorts.

Rules fixed before the run (common.screen_validation_cohorts, cohorts in alphabetical order): every MetaGxBreast
cohort other than METABRIC with a recurrence endpoint (relapse-free survival where recorded, else distant
metastasis-free survival), except cohorts sampled on the outcome (EMC2 holds patients selected for metastatic
relapse); ER-positive complete cases for age, tumour size, node status and grade; at least 30 events within 10 years;
and at least half of the model's marker weight (|coefficient| x development SD, SurvStudio's rule) measured, checked
before the cohort is scored. The reason a cohort was left out is recorded. Patients who are, through any chain of
listed duplicates, a METABRIC development patient or a patient of a cohort used before are removed first; markers
are rescaled within each cohort.

Conventions of the export (common.py): DFHCC and MAINZ (distant metastasis-free survival) and UNC4 (relapse-free
survival) store follow-up in multiples of 30 days, converted with 30 days a month (the others with 365.25 / 12);
UNC4 records tumour size in classes, treated as not recorded. VDX records the T category (1 to 4) in place of tumour
size; it is mapped to centimetres by the median METABRIC tumour of the matching range (T1 at most 2 cm, T2 over 2
and up to 5 cm, T3 and T4 over 5 cm), recorded in breast_er_external_pooled.json. GSE58644 measures several genes
with more than one probe (its symbols are padded with spaces); the most variable probe is used (common.probe_gene).
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
    recipe = read_result("breast_er_locked_model.json")
    screened, used = screen_validation_cohorts("V", recipe)
    entries = {entry["cohort"]: entry for entry in screened}
    cohort_rows, marker_rows = [], []
    for cohort, external in used.items():
        report = validate_locked_recipe(external, recipe, marker_scaling="within_cohort")
        check_marker_weight(entries[cohort], report)
        cohort_rows.append({"cohort": cohort, "endpoint": entries[cohort]["endpoint"], **validation_row(report)})
        marker_rows.extend({"cohort": cohort, **row} for row in validation_marker_rows(report))
    cohorts = pd.DataFrame(cohort_rows)
    write_csv_atomic(cohorts, RESULTS / "breast_er_external_validation.csv")
    markers = pd.DataFrame(marker_rows)
    write_csv_atomic(markers, RESULTS / "breast_er_external_markers.csv")

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
    write_json(RESULTS / "breast_er_external_pooled.json", pooled)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 100):
        print(pd.DataFrame(screened).to_string(index=False))
        print(cohorts.round(3).to_string(index=False))
    print(json.dumps({key: value for key, value in pooled.items() if key not in {"survstudio", "screened"}}, indent=1, default=float))


if __name__ == "__main__":
    main()

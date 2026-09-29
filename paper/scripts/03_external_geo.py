"""Case study II: the locked TCGA-LUAD model applied unchanged to seven GEO microarray cohorts.

Each cohort is scored as measured and with the markers rescaled within the cohort (another platform). Writes
cohort-level metrics (the model's C, the clinical-only model's C and their difference, each with its bootstrap
interval), per-marker replication, and the random-effects pooled C of the model and of the clinical model and the
C gain, each cohort weighted by its own interval.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    GEO_COHORTS,
    LUAD,
    RESULTS,
    pooled_validation,
    read_result,
    survstudio_version,
    validation_marker_rows,
    validation_row,
    write_csv_atomic,
    write_json,
)
from survival_toolkit.marker_evaluation import validate_locked_recipe

recipe = read_result("tcga_locked_model.json")
truth = pd.read_csv(LUAD / "harmonized_clinical.csv")  # written by data_prep/harmonize_luad.py
cohort_rows, marker_rows = [], []
for cohort in GEO_COHORTS:
    table = truth[(truth["cohort"] == cohort) & truth["exclusion"].isna()]
    path = LUAD / cohort / "expression_genes.csv.gz"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    expression = pd.read_csv(path, usecols=["sample_id", *[gene for gene in recipe["markers"] if gene in header]])
    external = pd.DataFrame({
        "patient_id": table["sample_id"],
        "os_months": table["os_time"] * 12.0,
        "os_event": table["os_event"],
        "age": table["age"],
        "sex": table["sex"].map({"M": "Male", "F": "Female"}),
        "stage_group": table["stage_group"].map(lambda value: f"Stage {value}" if isinstance(value, str) else np.nan),
    }).merge(expression.rename(columns={"sample_id": "patient_id"}), on="patient_id")
    for scaling in ("as_measured", "within_cohort"):
        report = validate_locked_recipe(external, recipe, marker_scaling=scaling)
        cohort_rows.append({"cohort": cohort, "scaling": scaling, **validation_row(report)})
        marker_rows.extend({"cohort": cohort, "scaling": scaling, **row} for row in validation_marker_rows(report))
cohorts = pd.DataFrame(cohort_rows)
write_csv_atomic(cohorts, RESULTS / "external_validation.csv")
write_csv_atomic(pd.DataFrame(marker_rows), RESULTS / "external_markers.csv")

pooled = {"survstudio": survstudio_version(), "patients": int(cohorts.loc[cohorts["scaling"] == "within_cohort", "n"].sum()),
          "events": int(cohorts.loc[cohorts["scaling"] == "within_cohort", "events"].sum())}
for scaling, part in cohorts.groupby("scaling"):
    pooled[scaling] = pooled_validation(part)
write_json(RESULTS / "external_pooled.json", pooled)
print(cohorts.round(3).to_string(index=False))
print(json.dumps({key: value for key, value in pooled.items() if key != "survstudio"}, indent=1, default=float))

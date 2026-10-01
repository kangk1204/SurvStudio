"""Case study II: the locked TCGA-LUAD model applied unchanged to seven GEO microarray cohorts.

Each cohort is scored as measured and with the markers rescaled within the cohort (another platform), with bootstrap
intervals from 2,000 resamples. Writes the cohort-level metrics (external_validation.csv: the model's C, the
clinical-only model's C and their difference with their bootstrap intervals, the calibration slopes of the model and
of the clinical-only model, the SD of the linear predictor and of its clinical and gene parts, and the gene
component's hazard ratio per SD beyond the clinical part; common.component_row), per-marker replication
(external_markers.csv), and the random-effects pooling of each quantity per scaling (external_pooled.json: the
DerSimonian-Laird estimate with its CI, the Hartung-Knapp-Sidik-Jonkman interval and the 95% prediction interval, each
cohort weighted by its own interval; common.pooled_validation).

Sensitivity, a separate output (the primary results are Harrell's C above): Uno's C truncated at 5 years for the model
and the clinical-only model in each cohort and their paired difference, with percentile intervals from the same 2,000
bootstrap resamples (external_uno.csv), pooled in the same way (external_uno_pooled.json).
"""

from __future__ import annotations

import json

import pandas as pd

from common import (
    BOOTSTRAP_DRAWS,
    GEO_COHORTS,
    RESULTS,
    UNO_HORIZON_MONTHS,
    component_row,
    geo_cohort,
    locked_parts,
    pool_interval,
    pooled_validation,
    read_result,
    survstudio_version,
    uno_row,
    validation_marker_rows,
    validation_row,
    write_csv_atomic,
    write_json,
)
from survival_toolkit.marker_evaluation import validate_locked_recipe


def main() -> None:
    recipe = read_result("tcga_locked_model.json")
    cohort_rows, marker_rows, uno_rows = [], [], []
    for cohort in GEO_COHORTS:
        external = geo_cohort(cohort, recipe["markers"])
        for scaling in ("as_measured", "within_cohort"):
            report = validate_locked_recipe(external, recipe, marker_scaling=scaling, n_bootstrap=BOOTSTRAP_DRAWS)
            parts = locked_parts(external, recipe, report, scaling)
            cohort_rows.append({"cohort": cohort, "scaling": scaling, **validation_row(report), **component_row(parts)})
            marker_rows.extend({"cohort": cohort, "scaling": scaling, **row} for row in validation_marker_rows(report))
            uno_rows.append({"cohort": cohort, "scaling": scaling, **uno_row(parts)})
            print(cohort, scaling, "done", flush=True)
    cohorts = pd.DataFrame(cohort_rows)
    write_csv_atomic(cohorts, RESULTS / "external_validation.csv")
    write_csv_atomic(pd.DataFrame(marker_rows), RESULTS / "external_markers.csv")

    rescaled = cohorts["scaling"] == "within_cohort"
    pooled = {"survstudio": survstudio_version(), "bootstrap_draws": BOOTSTRAP_DRAWS, "patients": int(cohorts.loc[rescaled, "n"].sum()),
              "events": int(cohorts.loc[rescaled, "events"].sum())}
    for scaling, part in cohorts.groupby("scaling"):
        pooled[scaling] = pooled_validation(part)
    write_json(RESULTS / "external_pooled.json", pooled)

    uno = pd.DataFrame(uno_rows)
    write_csv_atomic(uno, RESULTS / "external_uno.csv")
    uno_pooled = {"survstudio": survstudio_version(), "horizon_months": UNO_HORIZON_MONTHS, "bootstrap_draws": BOOTSTRAP_DRAWS}
    for scaling, part in uno.groupby("scaling"):
        uno_pooled[scaling] = {"model_c": pool_interval(part, "uno_c", "uno_lower", "uno_upper"),
                               "clinical_c": pool_interval(part, "clinical_uno_c", "clinical_uno_lower", "clinical_uno_upper"),
                               "delta_c": pool_interval(part, "uno_delta", "uno_delta_lower", "uno_delta_upper")}
    write_json(RESULTS / "external_uno_pooled.json", uno_pooled)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(cohorts.round(3).to_string(index=False))
        print(uno.round(3).to_string(index=False))
    for name, value in (("Harrell's C", pooled), ("Uno's C at 5 years", uno_pooled)):
        print(name, json.dumps({key: part for key, part in value.items() if key != "survstudio"}, indent=1, default=float))


if __name__ == "__main__":
    main()

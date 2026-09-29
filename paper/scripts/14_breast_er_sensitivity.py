"""Case study V, sensitivity: why did the positive control's gain shrink from 0.045 inside METABRIC to 0.009 in the
external cohorts? Optimism of the internal estimate, or a shift between cohorts (platform, endpoint, treatment)?

Fixed before the run:
1. Leave one METABRIC site out (primary). METABRIC's ER-positive tumours come from five sites (cBioPortal COHORT).
   Each site with at least 30 relapses (sites 1 to 4) is held out once: the whole SurvStudio evaluation runs on the
   other sites with the defaults of case study V (1,000 permutations, 200 subsamples, seed 20260926), and the locked
   model is applied to the held-out site. A patient without a site would never be held out and so would stay in
   every development set; the summary counts them (development_without_site). In the current data there are none:
   the 5 ER-positive tumours cBioPortal gives no site have no cBioPortal record at all, so no relapse-free survival.
   Platform, endpoint and clinical definitions do not change, so a held-out gain close to the internal one points to
   cohort shift, not optimism, for the external shrinkage. Markers are applied as measured (same platform); rescaled
   within the site as a sensitivity. The internal gain is SurvStudio's paired left-out gain
   (signature_gain_left_out).
2. Endpoint (secondary): the five external cohorts of case study V pooled separately for relapse-free survival
   (NKI, TRANSBIG, UPP; the development endpoint) and distant metastasis-free survival (GSE58644, VDX).
Writes breast_er_sensitivity.csv and breast_er_sensitivity.json. A worker that dies (killed for memory, say) stops
the run with an error instead of leaving it waiting.
Usage: python 14_breast_er_sensitivity.py [workers]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    BREAST_ER_CATEGORICAL,
    BREAST_ER_COVARIATES,
    RESULTS,
    load_breast_cohort,
    metabric_sites,
    random_effects,
    read_result,
    survstudio_version,
    write_csv_atomic,
    write_json,
)
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers, validate_locked_recipe  # noqa: E402

MIN_EVENTS = 30

FRAME, GENES = load_breast_cohort("METABRIC", endpoint="recurrence", er_positive=True)
# Each patient's site, row by row (NaN where cBioPortal records none).
SITE = metabric_sites(FRAME["patient_id"])


def fold(site: float) -> dict:
    development = FRAME[SITE != site].reset_index(drop=True)
    held_out = FRAME[SITE == site].reset_index(drop=True)
    result = evaluate_markers(
        development, time_column="recurrence_months", event_column="recurrence_event", marker_columns=GENES,
        clinical_columns=BREAST_ER_COVARIATES, categorical_clinical=BREAST_ER_CATEGORICAL, event_positive_value=1,
        settings=MarkerSettings(), id_column="patient_id",
    )
    signature = result["signature"]
    row = {
        "held_out_site": int(site), "development_n": result["cohort"]["n"], "development_events": result["cohort"]["events"],
        "robust": int(result["tier_counts"]["robust"]), "markers": ";".join(signature["markers"]),
        "apparent_c": signature["apparent_c"], "corrected_c": signature["optimism_corrected_c"],
        "left_out_c": signature["signature_c_left_out"], "clinical_left_out_c": signature["clinical_c_left_out"],
        # Paired: the model's and the clinical model's C on the same left-out patients (the difference of the two means
        # above can average over different subsamples).
        "internal_gain": signature["signature_gain_left_out"],
        "repeated_patients_flagged": int(result["duplicates"].get("n_pairs", 0)),
    }
    for scaling in ("as_measured", "within_cohort"):
        metrics = validate_locked_recipe(held_out, result["locked_recipe"], marker_scaling=scaling)["metrics"]
        prefix = "held_out" if scaling == "as_measured" else "rescaled"
        row.update({
            f"{prefix}_n": int(len(held_out)), f"{prefix}_events": int(held_out["recurrence_event"].sum()),
            f"{prefix}_c": metrics["c_index"], f"{prefix}_clinical_c": metrics["clinical_only_c_index"],
            f"{prefix}_gain": metrics["delta_c_index"], f"{prefix}_gain_lower": metrics["delta_c_index_ci"][0],
            f"{prefix}_gain_upper": metrics["delta_c_index_ci"][1],
        })
    print({key: row[key] for key in ("held_out_site", "robust", "internal_gain", "held_out_gain", "held_out_gain_lower", "held_out_gain_upper")}, flush=True)
    return row


def pooled(part: pd.DataFrame, estimate: str, lower: str, upper: str) -> dict:
    se = (part[upper] - part[lower]).to_numpy(dtype=float) / 3.92
    return random_effects(part[estimate].to_numpy(dtype=float), se)


def main() -> None:
    external = read_result("breast_er_external_validation.csv")
    sites = sorted(site for site in np.unique(SITE[~pd.isna(SITE)]) if FRAME.loc[SITE == site, "recurrence_event"].sum() >= MIN_EVENTS)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else len(sites)
    # A worker that dies breaks the pool, and map raises BrokenProcessPool instead of waiting forever.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(fold, sites))
    folds = pd.DataFrame(rows)
    write_csv_atomic(folds, RESULTS / "breast_er_sensitivity.csv")

    summary = {
        "survstudio": survstudio_version(),
        "sites_held_out": [int(site) for site in sites],
        "development_without_site": int(pd.isna(SITE).sum()),
        "internal_gain_mean": float(folds["internal_gain"].mean()),
        "held_out_gain_pooled": pooled(folds, "held_out_gain", "held_out_gain_lower", "held_out_gain_upper"),
        "rescaled_gain_pooled": pooled(folds, "rescaled_gain", "rescaled_gain_lower", "rescaled_gain_upper"),
        "external_by_endpoint": {
            endpoint: {"cohorts": part["cohort"].tolist(), "patients": int(part["n"].sum()), "events": int(part["events"].sum()),
                       "gain": pooled(part, "delta_c", "delta_lower", "delta_upper")}
            for endpoint, part in external.groupby("endpoint")
        },
    }
    write_json(RESULTS / "breast_er_sensitivity.json", summary)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(folds.drop(columns=["markers"]).round(3).to_string(index=False))
    print({key: value for key, value in summary.items() if key != "survstudio"})


if __name__ == "__main__":
    main()

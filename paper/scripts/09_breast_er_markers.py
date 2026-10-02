"""Case study V (development): genome-wide marker evaluation for recurrence in ER-positive breast cancer.
This real-data application is not a known-truth positive control for the selected signature's incremental value.

METABRIC ER-positive tumours; relapse-free survival (cBioPortal) censored at 10 years; added value over age,
tumour size, node status and grade; one expression column per gene (the most variable probe); SurvStudio's
defaults (1,000 permutations, 200 subsamples, seed 20260926). The design was fixed before the run. The data and the
evaluation are common.development_data and common.evaluate_development ("V").
Writes the marker table, the locked model and a summary to paper/results/.
"""

from __future__ import annotations

import time

import pandas as pd

from common import BREAST, RESULTS, development_data, evaluate_development, sha256_file, survstudio_version, write_csv_atomic, write_json

began = time.time()
frame, genes = development_data("V")
loaded = time.time() - began
result = evaluate_development("V", frame, genes)
seconds = time.time() - began - loaded

rows = []
for row in result["marker_table"]:
    flat = {"marker": row["marker"], "tier": row["tier"], "pattern": row["pattern"]}
    for lens in ("marginal", "added_value"):
        for key, value in (row.get(lens) or {}).items():
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    flat[f"{lens}_{key}_{index}"] = item
            elif not isinstance(value, dict):
                flat[f"{lens}_{key}"] = value
    rows.append(flat)
table = pd.DataFrame(rows)
write_csv_atomic(table, RESULTS / "breast_er_markers.csv")

dropped = pd.DataFrame(result["cohort"]["dropped_markers"], columns=["marker", "reason"])
reasons = dropped["reason"].str.replace(r" \(.*\)", "", regex=True).str.replace(r"^\d+% missing$", "missing", regex=True)
funnel = {
    "genes_in_file": len(genes),
    "tested": int(result["cohort"]["n_markers_evaluated"]),
    "left_out": {str(reason): int(count) for reason, count in reasons.value_counts().items()},
    "unadjusted_p_below_0_05": int((table["marginal_p_value"] < 0.05).sum()),
    "adjusted_p_below_0_05": int((table["added_value_p_value"] < 0.05).sum()),
    "bh_q_below_0_05": int((table["added_value_q_bh"] <= 0.05).sum()),
    "fwer_at_most_0_05": int((table["added_value_p_fwer"] <= 0.05).sum()),
    "robust": int((table["tier"] == "robust").sum()),
    "suggestive": int((table["tier"] == "suggestive").sum()),
}
write_json(RESULTS / "breast_er_locked_model.json", result["locked_recipe"])
write_json(RESULTS / "breast_er_markers_summary.json", {
    "survstudio": survstudio_version(),
    "seconds": round(seconds),
    "endpoint": "relapse-free survival (cBioPortal brca_metabric RFS_MONTHS, RFS_STATUS), 10 years",
    "relapse_file_sha256": sha256_file(BREAST / "METABRIC" / "cbioportal_patients.csv"),
    # ER-positive tumours left out for lack of cBioPortal follow-up, before the complete-case filter.
    "cbioportal_no_record": frame.attrs.get("cbioportal_no_record"),
    "cbioportal_endpoint_missing": frame.attrs.get("cbioportal_endpoint_missing"),
    "cohort": {key: result["cohort"][key] for key in ("n", "events", "n_markers_evaluated")},
    "within_cohort_duplicates_removed": frame.attrs.get("within_cohort_duplicates_removed", 0),
    "settings": result["settings"],
    "tier_counts": result["tier_counts"],
    "robust": table.loc[table["tier"] == "robust", "marker"].tolist(),
    "marginal_only": table.loc[table["tier"] == "marginal only", "marker"].tolist(),
    "funnel": funnel,
    "signature": result["signature"],
})
print(f"loaded in {loaded:.0f}s, evaluated in {seconds:.0f}s: {funnel}")
print({key: result["signature"].get(key) for key in ("markers", "apparent_c", "optimism_corrected_c", "signature_c_left_out", "clinical_c_left_out")})

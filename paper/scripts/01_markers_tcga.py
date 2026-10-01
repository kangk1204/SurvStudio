"""Case study I: genome-wide marker evaluation in TCGA-LUAD with SurvStudio's defaults.

Development data: the UCSC Xena HiSeqV2.gz matrix as downloaded and SurvStudio's bundled TCGA-LUAD clinical
table (common.tcga_development); added value over age, sex and stage; 1,000 permutations and 200 subsamples (seed
20260926; common.evaluate_development).
Writes the marker table, the locked model and a summary to paper/results/.
"""

from __future__ import annotations

import time

import pandas as pd

from common import RESULTS, XENA_EXPRESSION, evaluate_development, survstudio_version, tcga_development, write_csv_atomic, write_json

began = time.time()
frame, genes, matrix = tcga_development()
result = evaluate_development("I", frame, genes)
seconds = time.time() - began

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
write_csv_atomic(table, RESULTS / "tcga_markers.csv")

# The columns are named, so a run that drops no marker still has a (empty) reason column.
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
write_json(RESULTS / "tcga_locked_model.json", result["locked_recipe"])
write_json(RESULTS / "tcga_markers_summary.json", {
    "survstudio": survstudio_version(),
    "seconds": round(seconds),
    "cohort": {key: result["cohort"][key] for key in ("n", "events", "n_markers_evaluated")},
    "matrix": {"file": XENA_EXPRESSION.name, "id_note": matrix.id_note, "markers": len(genes)},
    "settings": result["settings"],
    "tier_counts": result["tier_counts"],
    "robust": table.loc[table["tier"] == "robust", "marker"].tolist(),
    "funnel": funnel,
    "signature": result["signature"],
})
print(f"done in {seconds:.0f}s: {funnel}")

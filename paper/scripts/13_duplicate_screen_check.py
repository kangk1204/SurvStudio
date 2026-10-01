"""How SurvStudio's own repeated-patient screen (survival_toolkit.duplicates) does on the case-study cohorts.

Each cohort is screened on its own, every tumour included, as the Markers tab would screen an uploaded matrix.
Flagged pairs are compared with the duplicates confirmed by the audits (breast_duplicate_pairs.csv within
cohorts; for lung adenocarcinoma the two technical duplicates the QC removed from GSE50081, put back here).
Every flagged pair is counted: the screen's report lists at most MAX_LISTED pairs, so the list is lengthened here
and checked against the report's total.
Writes duplicate_screen_check.csv (one row per cohort) and duplicate_screen_check.json.
"""

from __future__ import annotations

import sys

import pandas as pd

from common import GEO_COHORTS, LUAD, PAPER, RESULTS, XENA_EXPRESSION, load_breast_cohort, write_csv_atomic, write_json
import survival_toolkit.duplicates as duplicate_screen
from survival_toolkit.marker_matrix import matrix_frame, read_marker_matrix
from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset

# The report's pair list stops at MAX_LISTED (100) pairs; only its length changes, not which pairs are flagged.
duplicate_screen.MAX_LISTED = sys.maxsize

confirmed = pd.read_csv(PAPER / "breast_duplicate_pairs.csv")
within = confirmed[confirmed["cohort_a"] == confirmed["cohort_b"]]
known = {cohort: {tuple(sorted(pair)) for pair in zip(part["sample_a"], part["sample_b"])} for cohort, part in within.groupby("cohort_a")}
known["GSE50081"] = {("GSM1213837", "GSM1213843"), ("GSM1213842", "GSM1213849")}

tables = {}
for cohort in ["METABRIC", "CAL", "GSE58644", "NKI", "STNO2", "TRANSBIG", "UCSF", "UPP", "VDX"]:
    frame, genes = load_breast_cohort(cohort, endpoint=None, dedupe=False)
    tables[cohort] = frame.set_index("patient_id")[genes]
clinical = load_tcga_luad_upload_ready_dataset()
matrix = read_marker_matrix(XENA_EXPRESSION, XENA_EXPRESSION.name, patient_ids=clinical["patient_id"].tolist())
tcga = matrix_frame(clinical, matrix, id_column="patient_id", columns=["patient_id"])
tables["TCGA-LUAD"] = tcga.set_index("patient_id")[list(matrix.marker_names)]
truth = pd.read_csv(LUAD / "harmonized_clinical.csv")
for cohort in GEO_COHORTS:
    expression = pd.read_csv(LUAD / cohort / "expression_genes.csv.gz").set_index("sample_id")
    # QC-passed samples; GSE50081 with every sample, so the two technical duplicates the QC removed (and their
    # partners, some excluded for other reasons) are in the screen as a positive control.
    keep = set(truth.loc[(truth["cohort"] == cohort) & (truth["exclusion"].isna() | (cohort == "GSE50081")), "sample_id"])
    tables[cohort] = expression.loc[expression.index.intersection(list(keep))]

rows = []
for cohort, table in tables.items():
    report = duplicate_screen.possible_duplicates(table.to_numpy(dtype=float), list(table.index))
    flagged = {tuple(sorted((pair["a"], pair["b"]))) for pair in report["pairs"]}
    if len(report["pairs"]) != report["n_pairs"] or len(flagged) != report["n_pairs"]:
        raise RuntimeError(f"{cohort}: the screen flagged {report['n_pairs']} pairs but listed {len(report['pairs'])} ({len(flagged)} distinct).")
    expected = {pair for pair in known.get(cohort, set()) if set(pair) <= set(table.index)}
    rows.append({
        "cohort": cohort, "patients": len(table), "markers_used": report["markers_used"], "flagged": report["n_pairs"],
        "known_duplicates": len(expected), "found": len(flagged & expected), "flagged_not_known": len(flagged - expected),
        "identical_groups": report["n_identical"],
    })
    print(rows[-1], flush=True)
table = pd.DataFrame(rows)
write_csv_atomic(table, RESULTS / "duplicate_screen_check.csv")
write_json(RESULTS / "duplicate_screen_check.json", {
    "cohorts": len(table), "known": int(table["known_duplicates"].sum()), "found": int(table["found"].sum()),
    "flagged_not_known": int(table["flagged_not_known"].sum()), "patients": int(table["patients"].sum()),
})
print(table.to_string(index=False))

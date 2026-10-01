"""Comparison with the pipelines commonly used to publish prognostic signatures, step 1: the data of experiment 1.

The patients, outcomes and clinical covariates of case studies I and II (TCGA-LUAD, 484 primary tumours; the seven
GEO cohorts with the QC exclusions applied), and the genes of case study I that TCGA and all seven GEO cohorts measure
(script 15's rule: at most 20% missing, the rest at the cohort median, and varying), each z-scored within its cohort.
Writes, for the R pipelines, results/competitors/input/<cohort>.csv in Mime's layout (ID, OS.time in days, OS, the
genes) and <cohort>_clinical.csv (patient_id, os_months, os_event, age, sex, stage_group); for the null simulation
(18_competitors_null.py) results/competitors/null/tcga_expression.csv, TCGA's expression of the same genes as
measured; and the gene count and cohort sizes in results/competitors_data.json.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import GEO_COHORTS, RESULTS, survstudio_version, write_json
from competitors import WORK, code_hash, development_data, ensure_folder, gene_set, measured, r_frame, validation_data, zscore

began = time.time()
patients, expression, cohort = development_data()
validation = validation_data()
genes = gene_set(expression, validation)
folder = ensure_folder(WORK / "input")
cohorts = {"TCGA": (patients, expression), **validation}
summary = {"survstudio": survstudio_version(), "code_hash": code_hash(), "genes": len(genes),
           "case_study_genes": int(expression.shape[1]), "cohorts": {}}
for name, (table, values) in cohorts.items():
    block = values[genes].to_numpy(dtype=float)
    frame = r_frame(table, zscore(block), genes)
    frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.7g")
    table.to_csv(folder / f"{name}_clinical.csv", index=False)
    missing = int(np.isnan(block).sum())
    summary["cohorts"][name] = {"n": int(len(table)), "events": int(table["os_event"].sum()),
                                "genes_measured": int(measured(values).sum()), "missing_values_imputed": missing}
    print(name, summary["cohorts"][name], flush=True)
pd.Series(genes).to_csv(folder / "genes.txt", index=False, header=False)
null_folder = ensure_folder(WORK / "null")
tcga = expression[genes].copy()
tcga.insert(0, "patient_id", patients["patient_id"].to_numpy())
tcga.to_csv(null_folder / "tcga_expression.csv", index=False, float_format="%.7g")
patients.to_csv(null_folder / "tcga_clinical.csv", index=False)
summary["geo_patients"] = int(sum(summary["cohorts"][name]["n"] for name in GEO_COHORTS))
summary["geo_events"] = int(sum(summary["cohorts"][name]["events"] for name in GEO_COHORTS))
summary["seconds"] = round(time.time() - began)
write_json(RESULTS / "competitors_data.json", summary)
print(f"{len(genes)} genes of {expression.shape[1]} measured in TCGA and all seven GEO cohorts; {summary['seconds']}s", flush=True)

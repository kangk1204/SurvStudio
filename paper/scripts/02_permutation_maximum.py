"""Why the near-constant filter exists: the permutation maximum of the added-value score statistic in TCGA-LUAD,
with every non-constant gene and with genes at one value in more than 90% of patients left out.

300 permutations (seed 20260926) of the same screen SurvStudio runs in 01_markers_tcga.py: each marker's residuals
after regression on the clinical covariates are permuted (the Smith scheme; Winkler et al., NeuroImage 2014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import CATEGORICAL, COVARIATES, RESULTS, XENA_EXPRESSION, survstudio_version, write_csv_atomic, write_json
from survival_toolkit.marker_evaluation import MarkerSettings, _impute, _lens_screens, prepare_marker_cohort
from survival_toolkit.marker_matrix import matrix_frame, read_marker_matrix
from survival_toolkit.marker_screen import residualize, stratified_permutation
from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset

PERMUTATIONS = 300
clinical = load_tcga_luad_upload_ready_dataset()
matrix = read_marker_matrix(XENA_EXPRESSION, XENA_EXPRESSION.name, patient_ids=clinical["patient_id"].tolist())
frame = matrix_frame(clinical, matrix, id_column="patient_id", columns=["os_months", "os_event", *COVARIATES])
# max_mode_fraction=1.0 keeps near-constant genes: this script compares the screen with and without them.
cohort = prepare_marker_cohort(frame, time_column="os_months", event_column="os_event", marker_columns=list(matrix.marker_names),
                               clinical_columns=COVARIATES, categorical_clinical=CATEGORICAL, event_positive_value=1, max_mode_fraction=1.0)
rows = np.arange(cohort.time.size)
block = _impute(cohort.markers, np.nanmedian(cohort.markers, axis=0))
mode_share = np.array([np.unique(column, return_counts=True)[1].max() / column.size for column in block.T])
keep = mode_share <= 0.9
screen, design = _lens_screens(cohort, rows, MarkerSettings().ties)["added_value"]
observed = screen.statistics(block).chi2
source = residualize(block, design, cohort.strata)
rng = np.random.default_rng(20260926)
permuted = []
for start in range(0, PERMUTATIONS, 16):
    permutations = [stratified_permutation(cohort.strata, rows.size, rng) for _ in range(min(16, PERMUTATIONS - start))]
    chunk = screen.permuted_chi2(source, permutations)
    permuted.append(np.where(np.isfinite(chunk), chunk, 0.0).astype(np.float32))
permuted = np.vstack(permuted)

maxima = pd.DataFrame({"permutation": np.arange(PERMUTATIONS) + 1, "all_genes": permuted.max(axis=1), "near_constant_removed": permuted[:, keep].max(axis=1)})
write_csv_atomic(maxima, RESULTS / "permutation_maxima.csv")
winners = permuted.argmax(axis=1)
names = np.asarray(cohort.marker_names)
summary = {"survstudio": survstudio_version(), "permutations": PERMUTATIONS, "genes": int(keep.size), "genes_after_filter": int(keep.sum()),
           "winners_mode_share_median": float(np.median(mode_share[winners])),
           "winners_at_least_95pct_one_value": float(np.mean(mode_share[winners] >= 0.95))}
for label, subset in (("all_genes", np.ones(keep.size, bool)), ("near_constant_removed", keep)):
    threshold = float(np.quantile(permuted[:, subset].max(axis=1), 0.95))
    summary[label] = {
        "threshold_95": threshold,
        "bonferroni_chi2": float(stats.chi2.isf(0.05 / subset.sum(), 1)),
        "genes_above": names[subset][observed[subset] > threshold].tolist(),
    }
summary["top_observed"] = [{"marker": names[index], "chi2": float(observed[index]), "mode_share": float(mode_share[index])}
                           for index in np.argsort(-observed)[:12]]
write_json(RESULTS / "permutation_maximum_summary.json", summary)
print(summary["all_genes"]["threshold_95"], summary["near_constant_removed"]["threshold_95"], summary["near_constant_removed"]["genes_above"])

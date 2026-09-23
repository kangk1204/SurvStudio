# Release Notes

## 2026-09-23 — Statistical and evaluation-design audit

These changes alter reported numbers; results produced before this release should be re-run.

### Evaluation design (ML and DL benchmarks)

- Classical ML and deep-learning models now share one stratified 70/30 holdout helper and identical repeated-CV folds for the same seed. Previously DL used a separate 80/20 split, so on GBSG2 only 39 of 137 DL evaluation patients were in the ML test set while the unified leaderboard ranked them together.
- Added `evaluation_split_fingerprint` to every comparison result; the unified leaderboard only ranks ML and DL together when fingerprints match.
- Added an optional locked independent test set (`locked_test_fraction`) for repeated-CV comparisons: CV runs on the development set only, each model is refit on the development set and scored once on the untouched test set, and manuscript tables gain locked-test columns.
- Deep-model early stopping now monitors a subset that is held out from gradient updates (it was previously part of the training rows, so early stopping tracked training fit).
- Repeated-CV "SD" is now the SD across fold-level C-indices (it was the SD of repeat means, e.g. 0.0006 vs 0.042 on GBSG2, and exactly 0 with one repeat). The Brier Skill Score is pooled (1 - mean IBS / mean null IBS).

### Data

- The bundled TCGA LUAD files contained 609 rows for 489 patients (120 patients duplicated with identical outcomes and RNA values). They now keep one row per patient. With duplicates, repeated-CV C-index was inflated by +0.135 (GBS) and +0.090 (RSF) on the RNA top-100 file.
- Uploaded datasets whose identifier-like columns repeat are flagged in the profile and in KM/Cox/ML/DL cautions.

### Statistics

- Cox PH diagnostics use the Grambsch-Therneau score test (per term and global, log time), matching lifelines. The previous rank-based Spearman screen combined with Fisher's method had a 30-50% false-positive rate for categorical covariates under true proportional hazards.
- RMST standard errors follow the survRM2 convention when the risk set is exhausted at an event (they were previously blanked for every group).
- KM medians and reverse-KM median follow-up use the S(t) <= 0.5 convention (lifelines/SAS); median CIs honor the requested confidence level; grouped curves stop at each group's last follow-up; CI bands are drawn as steps.
- Harrell's C counts event-vs-censored pairs tied in time as comparable (matching sksurv/lifelines).
- Gradient Boosted Survival defaults to depth-3 trees when max_depth is "auto" (it previously grew unlimited-depth trees).
- LASSO-Cox selects its penalty by inner stratified 5-fold CV (max mean C-index) instead of a bootstrap 1-SE rule on one small inner split, which kept ~2 features.
- Optimal-cutpoint High/Low labels follow the log-rank observed-vs-expected direction, and the derived column is aligned to the original rows even when rows were dropped (labels were previously shifted onto other patients).
- Signature-discovery permutation p-values are search-adjusted (Westfall-Young max statistic over all tested signatures); "validation" resamples are described as within-cohort stability checks.
- Cox survival predictions for IBS use the right-continuous Breslow baseline; the IBS time grid stays within the evaluation follow-up (10th-90th percentile).
- DeepHit's ranking loss compares cumulative incidence at the event bin; the MTLR censored likelihood is computed in log space (float32 NaN failures removed).
- Time 0 is a valid follow-up time (only negative times are dropped); numeric 0/1 columns named like "censored" are refused as event columns; negated status labels ("No recurrence") are coded as censoring; deaths from other causes are refused in disease-specific event columns.


## 2026-03-26

### Evaluation and Reporting

- Added repeated stratified cross-validation for `/api/ml-model` comparison runs via `evaluation_strategy="repeated_cv"`.
- Added manuscript-oriented model performance tables to comparison outputs for both deterministic holdout and repeated-CV workflows.
- Standardized evaluation metadata so comparison outputs include the evaluation mode and manuscript export payloads.
- Added dashboard controls for repeated-CV selection plus CSV/Markdown/LaTeX/DOCX export of manuscript-ready ML comparison tables.
- Added server-side `/api/export-table` formatting so ML and DL comparison tables can be exported as CSV, journal-style Markdown, LaTeX, or DOCX with approximate `default`, `NEJM`, `Lancet`, and `JCO` templates.

### Statistical and Deep-Learning Corrections

- Promoted selection-adjusted cutpoint p-values to the primary reported value while preserving raw p-values.
- Corrected time-dependent importance matrix orientation to a consistent time-major contract between analysis and plotting.
- Standardized ML and DL result reporting around explicit evaluation labels such as `holdout`, `repeated_cv`, and `apparent`.
- Added a deep-learning `Compare All` workflow so bundled DL architectures can be benchmarked side by side under the same feature set.
- Added repeated stratified CV support to the deep-learning comparison workflow, using fold-specific preprocessing fitted on the training partition.
- Added deep-learning early stopping controls plus optional parallel fold execution for repeated-CV comparison runs.

### Packaging and Documentation

- Kept contributor installs aligned with the app feature surface by documenting `.[dev]` as the default setup path.
- Removed over-claiming wording from app metadata and README guidance; the toolkit is exploratory by default and requires external validation for paper-grade claims.
- Added this release note to track changes that affect statistical interpretation and manuscript reporting.

### Verification

- Full regression suite passes locally after these changes.

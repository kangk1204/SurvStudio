# Release Notes

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

# Release Notes

## Unreleased — Prognostic marker evaluation, design check, and a simpler interface

### Interface

- One workspace replaces the guided and expert modes. The outcome is set once in a one-line bar; grouping settings appear only on the Survival curves and Table 1 tabs; tab names are plain (Survival curves, Cox model, Markers, Prediction models, Table 1, Data).
- Sample cohorts open with their recommended outcome, grouping and variable selections, so the preset bar and the auto-detection banner are gone. The landing page offers three samples (lung cancer, breast cancer, synthetic).
- Result sections stay hidden until a run, each tab exports from one menu, and a status next to each Run button says whether the result is up to date, running, or out of date after a settings change.
- Interpretation panels show the headline, key numbers and the first two cautions; the rest folds under "More detail".
- The Cox tab previews usable patients, dropped rows, parameters and events per parameter before fitting.
- New Markers tab: the honest marker evaluation with stability and rank-uncertainty plots, the marker table, export of the locked model as JSON, and validation of the locked model on another cohort file. The outcome-informed cut-point combination search moved here as an exploratory option and now searches the selected markers and clinical covariates.
- New design-check page (`/design-check`, no data needed) for studies that fit many models and keep the best.
- Prediction models: a "Train one model" entry next to Compare All, results hidden until a run, and the Survival Transformer and VAE labelled experimental.
- Reporting checklists: the Markers tab exports a REMARK checklist and the prediction-model leaderboard a TRIPOD+AI checklist, as Word or Markdown. Each holds the run's methods and results paragraphs and every guideline item, marked as filled in by SurvStudio, partly filled in, or for the authors to complete.
- The marker table adds each marker's unadjusted hazard ratio and p-value when markers are judged on added value, as REMARK asks for univariable results.
- Omics marker matrices: the Markers tab attaches a separate file of up to 60,000 markers (one row per marker or per patient), matched to the dataset's patients by an ID column, so genome-wide panels no longer have to fit into the 1,000-feature table.
- Hazard-ratio axes are labelled with round values (0.5, 1, 2, 5); the landing page shows one card per analysis tab.
- Kaplan-Meier figures print the numbers at risk under the time axis, at round times (0, 50, 100 ...) that match the axis ticks, and place the test and band notes where no curve runs.
- Marker matrices may be gzip-compressed (the .gz files GEO and UCSC Xena serve), TCGA sample barcodes are matched to patient barcodes (one tumour sample per patient, normal tissue left out), and an ID mismatch shows both files' IDs.
- Summary badges read Robust when only the method's standing assumptions apply, and the Cox headline names the estimates that look unstable rather than the significant terms.
- The prediction-model leaderboard gives every model's C-index a 95% bootstrap interval and its difference from Cox PH a paired 95% interval on the same test patients, drawn as a dot plot with intervals instead of bars from zero.
- Validating a locked marker model in another cohort can rescale the markers within the cohort (another platform), and runs when some locked markers are not measured there, reporting the share of the model's weight that was.

### New

- `survival_toolkit.marker_screen`: vectorized Cox score tests for many candidate markers, marginal and as added value over clinical covariates (Efron or Breslow ties, with strata); Freedman–Lane residual permutations; Westfall–Young step-down p-values; permutation FDR; Harrell's C for many risk scores at once. Score statistics match R `coxph` score tests to a relative 1e-8 (univariate) and 1e-6 (clinically adjusted).
- `survival_toolkit.marker_evaluation.evaluate_markers`: the whole screening procedure checked in one step. It reports family-wise error and FDR by permutation, stability over event-stratified subsamples, and pre-declared tiers. A robust marker needs a Westfall–Young p ≤ 0.05, selection in at least 50% of subsamples and the same direction in at least 90%. It also reports an optimism-corrected C-index for the selected signature, the shrinkage of the strongest marker's effect, and an optional descriptive tree-model lens.
- `survival_toolkit.marker_evaluation.validate_locked_recipe`: applies a SHA-256-locked signature unchanged to an external cohort. It reports the C-index with a bootstrap CI, the gain over the clinical-only model, the calibration slope, observed/expected risk, the Brier score and skill, and a Holm-adjusted replication test per marker.
- README: a Prognostic Marker Evaluation section with a runnable example.
- Markers with more than 90% of patients at one value (`max_mode_fraction`) are left out before the marker screen. In genome-wide RNA-seq, genes expressed in a few patients made the permutation maximum (in TCGA-LUAD its 95% point was chi-square 239 instead of about 25), so no marker could pass family-wise control; the filter never looks at the outcome. The marker screen and REMARK report also set the selected-marker model against the clinical covariates alone in the patients left out.
- `survival_toolkit.evaluation.c_index_intervals` and `POST /api/model-comparison-intervals`: bootstrap intervals for several models scored on the same test patients, with paired differences from a reference model. ML and DL comparisons return `test_predictions` (holdout) or `locked_test_predictions` (locked test) keyed by stored row label, so the two families can be paired patient by patient.
- `validate_locked_recipe(..., marker_scaling="within_cohort")`; locked recipes now record each marker's development mean and SD (`marker_scale`).

- `survival_toolkit.design_audit.audit_design`: places a multi-algorithm study design (number and size of selection cohorts, genes only or with clinical covariates, number of candidates, training C-index in the choice, headline C-index) on the benchmark pilot's simulation map and returns the expected optimism of the presented C-index, the expected regret of the choice, and flagged practices with remedies.
- `survival_toolkit.marker_matrix`: reads marker matrices, matches them to a dataset by patient ID and keeps a few in memory; API `POST /api/marker-matrix` and `DELETE /api/marker-matrix/{id}`, and `marker_matrix_id` in `POST /api/marker-evaluation`.
- The marker evaluation runs about 10 times faster with the same results: on one machine, 200 markers with the default 1,000 permutations and 200 subsamples took 37 s instead of 374 s, and 4,000 markers with 48 permutations and 10 subsamples 19 s instead of 175 s, with identical tiers. Cox fits use a Newton-Raphson fitter on the score screen's risk-set sums instead of statsmodels' PHReg (same estimates to 1e-6; the R reference tests are unchanged); the screen sums risk sets with sparse products in cache-sized blocks, and medians, p-values and input conversion are vectorized.
- `survival_toolkit.reporting`: `remark_checklist` and `tripod_ai_checklist` build the checklists from a marker evaluation or model comparisons; `checklist_markdown` renders one.
- API: `POST /api/marker-evaluation` (now with a `report` field), `POST /api/marker-validation`, `POST /api/design-audit`, `POST /api/tripod-ai-checklist` and `POST /api/checklist-export`.

### Validation

- `validation/agreement/run_agreement.py` compares SurvStudio with R `survival` (the reference), lifelines and scikit-survival on the bundled GBSG2 and TCGA-LUAD cohorts and writes `docs/validation/numerical_agreement.md` with every compared value and its difference.
- `docs/usability_study_protocol.md`: the protocol for a moderated usability study (8 to 12 participants, seven tasks, task success, time, the Single Ease Question and SUS).

### Packaging

- The distribution is now named `survstudio` (the import name stays `survival_toolkit`), with a `survstudio` command next to the old `survival-toolkit`; the package metadata carries the licence, authors, keywords and project links.
- A Dockerfile builds a container that serves the app on port 8000 (`docker run --rm -p 127.0.0.1:8000:8000 survstudio`); inside a container the start-up message explains how to publish the port on loopback only.
- `CITATION.cff` and `.zenodo.json` describe the software for GitHub's citation box and Zenodo archiving; a release workflow builds and checks the wheel, publishes it to PyPI by trusted publishing and pushes a multi-architecture image to GHCR when a GitHub release is published. `docs/releasing.md` lists the one-time account steps.

## 0.2.0 — 2026-09-26 — Full code review

Several fixes below change reported numbers (marked **changes results**). Exports now record the SurvStudio version that produced them, so re-run analyses exported by 0.1.0 before comparing them with new output.

### Statistics

- **Changes results.** Cox proportional-hazards tests: Schoenfeld residuals are now computed with the Efron tie correction within each stratum, as R's `residuals.coxph` does. statsmodels subtracts Breslow risk-set means even from an Efron fit, so the Grambsch-Therneau statistics were wrong whenever event times were tied, in stratified and unstratified models alike (for example a single-term statistic of 0.044 where R gives 0.011). Per-term and global statistics now match R `survival` 3.8 to about 1e-8.
- **Changes results.** Martingale residuals include the hazard jump at each patient's own event time (with the Efron share for tied events). statsmodels evaluated the cumulative hazard just before that time, which biased event residuals upward by up to 1.1 on tied data; the residuals now match R.
- **Changes results.** Cox BIC uses the number of events as the sample size (R `BIC(coxph)`, Volinsky & Raftery 2000) instead of the number of rows.
- **Changes results.** IPCW Brier scores and the IBS weight events by `1 / G(t_i-)` (Gerds & Schumacher 2006, as in `pec` and `riskRegression`), with a reverse Kaplan-Meier estimate that counts events before censorings at tied times. The summary states the actual evaluation window.
- **Changes results.** Calibration bins whose patients were not followed up to the evaluation time are reported as not estimable; the last Kaplan-Meier value was previously carried forward past the bin's follow-up.
- **Changes results.** The per-repeat Brier Skill Score in repeated CV is pooled (1 - mean IBS / mean null IBS) like the overall score, instead of averaging per-fold ratios.
- **Changes results.** Large non-integer group values (for example 100000.5 and 100000) are no longer merged into one Kaplan-Meier group.
- Kaplan-Meier: an unknown `logrank_weight` is refused (a typo used to run a plain log-rank test labelled with the typo); risk-table counts use the exact tick times and tick labels stay unique on short time scales.
- The Kaplan-Meier, log-rank, Fleming-Harrington, RMST, median-CI, Cox, C-index, proportional-hazards, and martingale values are now regression-tested against R `survival` 3.8.6 reference numbers.

### Cutpoints and derived groups

- **Changes results.** The optimal-cutpoint column labels every row with a usable marker value, including rows whose survival outcome is missing, as the median and percentile splits do; the summary reports how many rows were scanned and how many were labelled without an outcome.
- The maximally selected log-rank scan is vectorised (identical to `survdiff` to 1e-15, about 30 ms per permutation at n = 2000). Very large scans use an evenly spaced subset of candidate cutpoints, reported in `candidate_grid`; the permutation p-value is exact for the scanned set, and ties with the observed maximum are counted as at least as extreme.

### Machine learning and deep learning

- **Changes results.** Categorical encoding uses the same reference ordering as the Cox workflow (stage I before II, `2` before `10`), and no longer emits indicator columns that are constant in the training data (unknown-level columns always, missing-value columns when the training data have no missing values).
- **Changes results.** Random Survival Forest and Gradient Boosted Survival feature importance is out-of-sample permutation importance with all one-hot columns of a feature shuffled together, reported per raw feature.
- **Changes results.** Time-dependent importance now fits the selected RSF or GBS model and measures, per raw feature and time point, the increase in the IPCW Brier score when the feature is shuffled. It previously fitted separate random-forest classifiers on "event before t" labels (dropping patients censored before t) and ignored the selected model.
- **Changes results.** Deep models are refit on the whole training partition for the early-stopping epoch count, so they use the same rows as the classical models they are ranked against (the 20% monitor subset was previously never trained on).
- Deep models no longer fail or fall back to apparent evaluation when the event column uses text labels such as `Dead` / `Alive`.
- Deep models treat text feature columns as categorical, as the ML models and Cox PH do; a numeric column with a few stray text values is refused with the same message in every module.
- `evaluate_single_deep_survival_model` accepts `locked_test_fraction` for repeated CV.
- Programming errors inside a fold (for example a `KeyError`) now stop the run instead of being recorded as a failed fold.
- Parallel deep-learning folds estimate available memory on Windows as well.

### Data input

- Text uploads are read as UTF-8, UTF-16, Korean CP949/EUC-KR, Windows-1252, or Latin-1, and the detected encoding is shown after upload. Korean Excel "CSV" files were previously decoded as Latin-1, which garbled every Korean header and label.
- Row, column, and cell limits are checked from the header and a bounded read before the whole file is parsed; unsupported file extensions are refused before anything is written to disk.

### Server and security

- Blocking work runs in the thread pool, heavy jobs share a small number of slots (`SURVSTUDIO_MAX_HEAVY_JOBS`, default 2), and a job whose request was abandoned stops at its next checkpoint (HTTP 499).
- Cox convergence is read from the optimizer instead of process-wide warning filters, which concurrent requests could miss or steal.
- The cache of fitted models is limited by memory (1 GiB) as well as by count.
- `DELETE /api/dataset/{dataset_id}` frees a dataset and its cached models; shutdown also requires a loopback Host header; `serve` warns when bound to a non-loopback address because there is no login.
- Exports record the SurvStudio version and the dataset fingerprint; `/api/health` reports the package version.

### Front end

- `app.js` is split into eight classic scripts loaded in order by `index.html`.
- A newer request of the same kind cancels the older one, so the server stops working on results nobody will read.
- Group labels and other dataset text are escaped before they reach Plotly, so labels such as `<b>High</b>` or `%{y}` render literally.

### Packaging and CI

- The package version (0.2.0) has one source, `survival_toolkit.__version__`.
- The `all` extra now contains runtime features only (format readers, ML, DL, `kaleido`); test tools live in `dev` and `e2e`.
- CI adds Python 3.12 and 3.13, Windows, a wheel build with a clean-install smoke test, a front-end syntax check, and manual runs.
- Long analysis functions (Kaplan-Meier, Cox, signature search, derived groups, event coding) were split into smaller helpers without changing their output; this was checked on 24,537 input combinations.

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

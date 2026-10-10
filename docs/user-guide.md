# User guide

[Back to quick start](../README.md)

## Input Data Format

Supported file types:
- included in the base `pip install -e .`: `csv`, `tsv`, `txt`
- requires `pip install -e ".[formats]"`: `xlsx`, `xls`, `parquet`

If you are choosing a spreadsheet format, prefer `.xlsx` over legacy `.xls`.

Upload limits and text encodings:
- at most 200 MB per file, 100,000 rows, 5,000 columns, and 5,000,000 cells; `.xlsx` workbooks may expand to at most 256 MB when decompressed
- text and Excel files over a limit are refused from their header and a bounded read, before the whole file is parsed
- other file extensions are refused before anything is written to disk
- text files may be UTF-8 (with or without BOM), UTF-16, Korean CP949/EUC-KR (what Korean Excel saves as "CSV"), Windows-1252, or Latin-1; the encoding is detected automatically, and the upload banner and `survival-toolkit inspect` profile (`text_encoding`) show which one was used

Expected structure:
- one row per patient or subject
- one column for follow-up time
- one column for event status
- remaining columns as covariates

### Minimum Required Columns

At minimum, your file needs:

1. a time column
- numeric
- zero or positive values (time 0 is kept; negative times are dropped)
- examples: `os_months`, `followup_months`, `time_to_event`

2. an event column
- indicates whether the event happened during follow-up
- examples: `os_event`, `status`, `death`

3. optional feature columns
- age
- sex
- stage
- treatment
- biomarkers

### Simple Example

```csv
patient_id,os_months,os_event,age,stage,treatment,biomarker_score
PT-001,12.4,1,67,III,Standard,0.82
PT-002,18.0,0,59,II,Combination,-0.15
PT-003,7.2,1,72,IV,Standard,1.31
```

Meaning:
- `os_months`: follow-up time
- `os_event = 1`: event happened
- `os_event = 0`: censored

### Event Coding

The app can handle common event codings.

- default is `1 = event`, `0 = censored`
- if your event column uses another coding, select the event-positive value in the app
- by default, the `Event column` selector shows only binary columns whose names look like real event indicators
- if your true event indicator uses a non-standard name, turn on `Show all columns for Event`

Examples:
- `1 / 0`
- `yes / no`
- `death / alive`
- `R / N`

Important:
- this platform expects a **binary event indicator**
- if your status column has more than two states, recode it first
- baseline characteristics such as `egfr_status`, `kras_status`, `sex`, `stage`, or treatment labels are usually **not** event columns
- those fields should usually be used as `Group by` variables or model features, not as the survival event indicator

Examples that need recoding before analysis:
- `0 = censored`, `1 = cancer death`, `2 = non-cancer death`
- `0 = no event`, `1 = relapse`, `2 = death`

Those are not single-event binary outcomes.

### What Counts As "Time"

The time column should be:
- numeric
- measured in one consistent unit
- zero or positive for all analyzable rows (negative times are dropped)

Good examples:
- months from diagnosis to death
- days from surgery to recurrence
- weeks from enrollment to progression

Avoid mixing units like:
- some rows in days
- some rows in months

### Missing Data

The app will drop rows that become unusable after required columns are checked.

For example, rows may be removed if they have:
- missing survival time
- missing event status
- missing values in selected model covariates
- non-numeric values in a numeric time column

So before analysis, it is better if your file has:
- clean time values
- clean event values
- as little missingness as possible in the variables you plan to model

Text and numeric features:
- text columns (for example `stage` or `smoking_status`) are used as categorical variables by Cox PH, the ML models, and the deep models, even if you do not mark them categorical
- a column that is numeric except for a few stray text values (for example `unknown` in an `age` column) is refused as a model feature in every module; recode those cells as blank so the column stays numeric (a blank is handled as missing)
- a text column with more than 50 distinct values is refused as a feature unless you mark it categorical: it is usually numbers stored as text (values such as `<0.1`, or decimal commas) or a patient ID, and would otherwise add one column per value
- a column you mark categorical is used as categorical even when its values look numeric
- numeric category codes are the same levels whether a column was read as whole numbers or as decimals (`1` and `1.0`; one blank cell makes pandas read a code column as decimals), so a model locked on one cohort scores another cohort's codes correctly
- the reference (baseline) level of a categorical variable follows the clinical ordering (stage I before II, never smoker before current smoker) or numeric order for numeric-looking codes (`2` before `10`)

### One Row Per Patient

The current workflow assumes:
- one patient per row
- one survival endpoint at a time

Do not upload long-format repeated-measures tables like:
- multiple rows per patient across visits
- one row per timepoint
- one row per lesion

### Recommended Beginner Template

If you are preparing a first file, use columns like:
- `patient_id`
- `os_months`
- `os_event`
- `age`
- `sex`
- `stage`
- `treatment`
- one or more biomarker columns

### Common Mistakes

- event column contains more than two outcome states
- time column contains text like `12 months`
- one patient appears in multiple rows
- uploaded file uses mixed date/time formats instead of a numeric follow-up duration
- choosing a categorical text field as the survival time column

## Input Validation And Error Handling

The app does not silently guess around invalid survival inputs.
It performs explicit checks and returns errors when the input does not fit the supported workflow.

### What The App Validates

On upload and analysis, the app checks things like:
- supported file extension
- readable file encoding for text files
- file is not empty
- uploaded dataset stays within the 1000-feature model-input cap
- required columns exist
- survival time can be interpreted as numeric
- survival time is zero or positive
- event coding can be interpreted as a binary event indicator
- selected model covariates remain analyzable after missing values are handled

### Typical Error Cases

You should expect an error if:
- the file extension is unsupported
- the event column cannot be interpreted as binary
- the event-positive value you selected is not present
- the event column has more than two states
- you pick a baseline status field as the event column after enabling `Show all columns for Event`
- the selected time column has no positive values
- all usable rows disappear after missing-value filtering
- a model has too few analyzable samples or too few events

### Examples Of Helpful Error Messages

Examples of messages the app may return:
- `Unsupported input file extension`
- `Could not infer event coding`
- `The numeric event column has more than two distinct states`
- `No analyzable rows remain after removing missing values`
- `No events were found after preprocessing the event column`
- `Partial dependence for categorical feature ... is not supported`

### Practical Advice For Beginners

If upload or analysis fails:

1. check that the time column is numeric
2. check that the event column is truly binary
3. check that you selected the correct event-positive value
4. check that one patient appears only once
5. check missing values in the variables you selected for modeling
6. use `Synthetic demo` first to confirm the app itself is working
7. use `survival-toolkit inspect path/to/file.csv` to inspect your file before opening the UI

## Main Analyses

### Kaplan-Meier

Use this when you want:
- survival curves by group
- median survival
- RMST with delta-method confidence intervals at the display horizon
- weighted log-rank tests
- risk tables

### Cox PH

Use this when you want:
- hazard ratios
- confidence intervals
- p-values
- multivariable adjustment
- simple proportional hazards diagnostics

### Machine Learning

Implemented ML paths:
- LASSO-Cox (penalized Cox)
- Random Survival Forest
- Gradient Boosted Survival
- model comparison against Cox PH

Comparison supports:
- deterministic holdout (stratified 70/30, shared with the deep-learning models)
- repeated stratified CV
- repeated stratified CV on a development set plus a **locked independent test set** (`locked_test_fraction`)
- report result tables

Single-model ML training also supports:
- `Fast mode (skip SHAP)` for faster turnaround
- feature-importance output for trained models

LASSO-Cox note:
- use this when the feature set is too wide for stable unpenalized Cox PH
- it is a predictive penalized Cox path, not an inferential hazard-ratio workflow
- SHAP, partial dependence, and counterfactual analysis remain tree-model features only

Practical note:
- `Compare All Models` is usually faster than training one model (`Train one model`)
- `Compare All Models` focuses on cross-model scoring
- training one model may do extra post-fit work such as feature importance and optional SHAP computation
- ML result payloads now include IPCW `IBS`, a Kaplan-Meier null-model `IBS`, and `Brier Skill Score = 1 - IBS_model / IBS_null` so raw error can be interpreted relative to a no-covariate reference
- the IPCW weights follow Graf et al. (1999) with the Gerds & Schumacher (2006) convention used by `pec` and `riskRegression`: an event at `t_i` is weighted by `1 / G(t_i-)`, a patient still at risk at `t` by `1 / G(t)`, and the reverse Kaplan-Meier estimate `G` counts events before censorings at tied times. scikit-survival's `brier_score` uses `G(t_i)` instead, so the two differ slightly when censoring times coincide with event times
- Random Survival Forest and Gradient Boosted Survival feature importance is permutation importance on the evaluation rows (up to 300): the mean drop in Harrell's C when a raw feature is shuffled, with all one-hot columns of a categorical feature shuffled together
- for quick RSF checks on larger cohorts, leave `Fast mode` enabled
- if `TreeExplainer` is unsupported, SHAP falls back to a tightly capped `KernelExplainer` approximation using a small background/evaluation sample, so treat the ranking as approximate rather than a precise estimate
- if SHAP safe mode is triggered because the encoded matrix is too wide, SurvStudio explains a reduced companion tree model for interpretability only; describe that companion-model caveat explicitly if you cite SHAP outputs in a report

### Deep Learning

Implemented DL paths:
- DeepSurv
- DeepHit
- Neural MTLR
- Survival Transformer
- Survival VAE

Deep comparison supports:
- deterministic holdout (the same stratified 70/30 split as the ML comparison for the same seed)
- repeated CV, optionally with a locked independent test set
- early stopping on a monitor subset held out from gradient updates
- parallel fold execution

Single-model and comparison DL runs expose:
- epochs
- learning rate
- dropout
- batch size
- random seed
- shared ML/DL feature selection

Model-specific advanced controls are shown only when relevant:
- DeepHit / Neural MTLR:
  - `time bins`
  - `batch size`
- Survival Transformer:
  - `transformer width`
  - `attention heads`
  - `transformer layers`
- Survival VAE:
  - `latent dim`
  - `clusters`
  - `batch size` is ignored because the current VAE path uses full-batch optimization

Architecture note:
- `Hidden Layers` uses the full comma-separated stack for DeepSurv, DeepHit, Neural MTLR, and Survival VAE
- `Dropout` is applied to all current deep model paths, including Neural MTLR
- `Batch Size` currently affects DeepHit and Neural MTLR only. DeepSurv, Survival Transformer, and Survival VAE use full-batch optimization in the current implementation, and the run metadata reports the effective full-batch size for those paths.
- Adam-based DL optimizers use light L2 regularization (`weight_decay=1e-4`) and gradient clipping for stability on wider feature sets.
- DeepHit ranks the predicted cumulative incidence at each event time (Lee et al., 2018), including subjects censored in the same time bin, with a stabilized ranking-loss scale (`sigma=1.0`).
- `Neural MTLR` uses a neuralized right-cumulative MTLR parameterization for workflow comparison; its censored likelihood is evaluated in log space for numerical stability. It matches the canonical MTLR probability construction, while the surrounding network/training path is a practical SurvStudio implementation rather than a line-by-line clone of one reference codebase.
- `Survival VAE` should be interpreted as a VAE-inspired latent representation model for clustering and risk screening. SurvStudio does not claim validated generative simulation or uncertainty estimation from this path.
- Early stopping monitors a stratified 20% subset of the training partition that is **held out from gradient updates** (the model never trains on it). `DeepSurv`, `Survival Transformer`, and `Survival VAE` monitor C-index; `DeepHit` and `Neural MTLR` monitor the discrete-time loss. The monitor subset never overlaps the holdout, CV fold, or locked test set, and its curve is not a validation metric.
- After early stopping picks the best epoch, the model is refit from scratch on the **whole** training partition (monitor rows included) for that many epochs, so the reported model uses every training row. The run metadata reports `refit_epochs` (also `epochs_trained`, the epochs behind the reported model), the length of the early-stopping run (`early_stopping_epochs`), the rows used for early stopping (`early_stopping_fit_samples`, `monitor_samples`), and the final `fit_samples`. Every deep fit runs on one torch thread, so a seed gives the same numbers whether cross-validation folds run in parallel or one after another.
- Deep-model summaries currently report discrimination (`C-index`) only. SurvStudio does not yet compute IBS for deep-model outputs, so calibration/error comparisons are not directly symmetric with the ML module.
- Cox-style DL paths (`DeepSurv`, `Survival Transformer`) optimize a Breslow-ties partial-likelihood objective, while the classical Cox PH workflow reports Efron-ties estimates; this difference is intentional and should be documented in analysis notes if you compare those paths directly.

### Prognostic Marker Evaluation

Use this when you screen many candidate markers (for example gene-expression columns) for association with survival and want the claim checked the way a careful reviewer would check it. It runs in the Markers tab and from Python.

Markers can be columns of the uploaded table or, for omics data, a separate marker matrix. In the Markers tab, open `Markers in a separate file (omics)`, choose the dataset's patient ID column and attach a CSV, TSV, TXT or Parquet file with one row per marker and one column per patient (as GEO and TCGA distribute expression) or one row per patient; the layout is detected from the IDs. Text files may be gzip-compressed, so a UCSC Xena download such as `HiSeqV2.gz` attaches as it is. The matrix can hold up to 60,000 markers and 30 million values, and its patient IDs must be written exactly as in the ID column, except that TCGA sample barcodes (`TCGA-05-4244-01`) are matched to patient barcodes (`TCGA-05-4244`), one tumour sample per patient with normal tissue left out; patients without matrix values are left out of the evaluation. The clinical table stays small, so the other tabs are unaffected.

Markers with more than 90% of patients at one value are left out before testing (`max_mode_fraction`). A gene expressed in a handful of patients has a heavy-tailed test statistic; in the TCGA-LUAD RNA-seq data such genes made the permutation maximum (its 95% point was chi-square 239 instead of about 25), so no gene could pass family-wise control. The filter does not look at the outcome, so the error control holds.

For every marker it reports:
- two Cox score-test lenses: marginal association, and added value over the clinical covariates you name (the primary lens whenever clinical covariates are given)
- Westfall–Young step-down permutation p-values (family-wise error over all markers) and permutation FDR q-values. The added-value null permutes the residuals of each marker after regression on the clinical covariates (the Smith method; Winkler et al., NeuroImage 2014;92:381–397), so a marker that merely tracks a clinical factor is not called prognostic
- the whole procedure rerun on event-stratified 63.2% subsamples: selection frequency, rank interval, and direction consistency
- a tier from pre-declared rules:
  - `robust`: Westfall–Young p ≤ 0.05, selected in at least 50% of subsamples, same direction in at least 90%
  - `suggestive`: Westfall–Young p ≤ 0.05 or permutation q ≤ 0.10, but not stable enough to be robust
  - `marginal only`: associated on its own but not beyond the clinical covariates
  - `not supported`

It also screens the patients for repeated samples. Public expression cohorts often hold the same tumour twice, and a patient in the data twice can sit on both sides of a subsample split and flatter the internal estimates. On panels of at least 200 markers, two patients are flagged when their profiles over the 5,000 most variable markers are each other's best match, correlate at least 0.7, and stand 0.2 above either one's next-best match. Patients with identical values on every marker are flagged too, on panels of at least 20 markers that take many distinct values (on binary panels such as mutation calls, patients share profiles by chance). Flagged pairs lead the cautions, named by the patient ID column, and the verdict stays at review until one sample per patient is kept. On 17 public breast and lung cancer cohorts (5,955 patients), the screen found 15 of 21 confirmed repeated tumours and flagged no pair of different patients.

For a signature built from the selected markers it reports the apparent C-index, an optimism-corrected C-index, the C-index on left-out rows next to that of the clinical covariates alone, and how much the top marker's effect shrinks outside the rows that selected it (the "winner's curse" of picking the strongest marker). The signature is then locked into a recipe (encoders, coefficients, baseline survival, and a SHA-256 hash) that can be applied unchanged to an external cohort:

```python
import pandas as pd
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers, validate_locked_recipe

development = pd.read_csv("development.csv")
genes = [column for column in development.columns if column.startswith("gene_")]
result = evaluate_markers(
    development,
    time_column="os_months",
    event_column="os_event",
    marker_columns=genes,
    clinical_columns=["age", "stage"],
    categorical_clinical=["stage"],
    settings=MarkerSettings(n_permutations=1000, n_resamples=200),
)
robust = [row["marker"] for row in result["marker_table"] if row["tier"] == "robust"]

external = pd.read_csv("external.csv")
report = validate_locked_recipe(external, result["locked_recipe"], horizon=60)
print(report["metrics"]["c_index"])
```

The Markers tab opens its results with one summary figure: the number of markers that clear each bar (tested, p < 0.05, FDR q ≤ 0.05, family-wise p ≤ 0.05, robust) and the signature's C-index from apparent to optimism-corrected to the left-out rows, beside the clinical covariates alone. `survival_toolkit.plots.build_marker_summary_figure(result)` draws it from Python.

In the Markers tab, `Export` also gives a REMARK checklist (Word or Markdown): the methods and results paragraphs of the run and the 20 REMARK items, each marked as filled in by SurvStudio, partly filled in, or for the authors to complete (study design, specimens, assay, interpretation). From Python, `survival_toolkit.reporting.remark_checklist(result)` returns the same checklist.

External validation reports Harrell's C with a bootstrap CI, the C-index gain over the locked clinical-only model, the calibration slope, observed/expected risk, the Brier score and Brier skill at the horizon, and each marker's external hazard ratio with a Holm-adjusted one-sided replication test. A recipe that was edited after it was locked is rejected.

For a cohort measured on another platform (for example microarrays against an RNA-seq development set), choose `Another platform (rescale within cohort)` in the interface or pass `marker_scaling="within_cohort"`: each marker is mapped onto its development mean and SD by its z-score within the external cohort, so the model's relative weights hold; discrimination is then comparable, absolute risks only roughly. Locked markers the external dataset does not measure are held at their development median, and the report gives the share of the model's marker weight (|coefficient| x development SD) that was measured; below half, validation stops.

In the TCGA-LUAD case study, the ten-gene model locked on RNA-seq (apparent C 0.749, optimism-corrected 0.652) reached a pooled C of 0.665 in seven GEO microarray cohorts (1,509 patients, 576 deaths), against 0.660 for age, sex and stage alone: the corrected estimate, not the apparent one, anticipated the external result.

Notes:
- every threshold used for a tier is a field of `MarkerSettings`; fix them before looking at results, not after
- the robust-tier thresholds (50% selection, 90% direction) come from a simulation pilot in which stricter values (80% / 95%) found only 6% of true markers instead of 21%, with the robust tier's family-wise error at or below 1% either way
- `nonlinear_lens="gbs"` or `"rsf"` adds a descriptive tree-model permutation-importance check, shown as `N+` / `N·` in each marker's evidence pattern; it does not change the tiers
- the Cox score screen reproduces R `survival::coxph` score tests (Efron and Breslow ties, with and without strata) to a relative tolerance of 1e-6 or better

## How To Read Results

### Kaplan-Meier

- Curves farther apart usually suggest different survival experiences between groups.
- The log-rank p-value tests whether the group curves differ.
- A statistically small p-value does **not** automatically mean the effect is clinically important.

### Cox PH

- Hazard ratio `> 1`: higher hazard
- Hazard ratio `< 1`: lower hazard
- Confidence intervals crossing `1` mean the estimate is compatible with no effect
- The current Cox discrimination summary is an `Apparent C-index` on the analyzable cohort, not an externally validated performance estimate.
- PH diagnostics use the Grambsch-Therneau score test on scaled Schoenfeld residuals versus log time: one 1-df test per model term plus a global test (the classic `cox.zph` statistic of R `survival` < 3.0, also used by lifelines' `proportional_hazard_test` with a log transform).
- Schoenfeld and martingale residuals are computed with the Efron tie correction within each stratum, as R's `residuals.coxph` does, so the PH test and residual plots stay correct when event times are tied, in stratified and unstratified models alike. The test suite checks them against R `survival` 3.8 reference values.
- Continuous covariates also expose Martingale residual trend plots as a visual linearity screen; strong curvature suggests splines, transforms, or recoding before locking the Cox specification.
- AIC and BIC are reported for the fitted model; BIC uses the number of events as the sample size, as R's `BIC(coxph)` does.
- A Cox `C-index = 0.65` means the fitted model ranks about `65%` of comparable patient pairs in the observed risk order; it is not "65% accuracy."

### Model C-index

The app explicitly distinguishes different evaluation modes:

- `Holdout C-index`
  - discrimination measured on a deterministic stratified 70/30 holdout split
  - this is one split only, so no CI or SD is shown
- `Repeated-CV mean C-index`
  - average of all fold-level C-indices across repeated stratified CV; the reported SD is the SD across those fold-level estimates (folds share training data, so it is descriptive, not a confidence interval)
- `Locked-test C-index`
  - with a locked test set, models are ranked by repeated CV on the development set, refit once on the whole development set, and scored once on the untouched test set
  - report the locked-test C-index of the CV-selected (rank 1) model as the independent test performance
- `Apparent C-index`
  - measured on the training/analyzable cohort
  - optimistic
  - should not be treated as external validation

Rough interpretation:
- `0.50` is chance-level ranking
- values above about `0.70` can be useful for screening
- the evaluation design still matters more than the threshold itself

### Cutpoints

Optimal cutpoints are exploratory by nature.
If you use them in a report:
- report how the cutpoint was selected
- prefer selection-adjusted p-values when available
- validate the cutpoint on separate data

The chosen cutpoint is a fixed rule on the marker, so the derived column labels every row with a usable marker value, including rows whose survival outcome is missing (they did not help choose the cutpoint). The summary reports how many rows were scanned and how many were labelled without an outcome. On very large cohorts the scan may use a quantile grid of candidate cutpoints instead of every observed value; the result then carries a `candidate_grid` note.

Current derive-group options also include:
- `Percentile split`
  - `25` means `at/above the 75th-percentile threshold vs Rest`
  - `25,25` means `at/below the 25th-percentile threshold / between thresholds / at/above the 75th-percentile threshold`
  - ties at the percentile threshold can make the realized groups slightly larger than the nominal percentages
- `Extreme split`
  - `25` means `at/below the 25th-percentile threshold vs at/above the 75th-percentile threshold`
  - the middle `50%` is excluded from grouped analyses for that derived split
  - ties at either threshold can make the kept tails slightly larger than the nominal percentages

If you derive a `High/Low` grouping from the same cohort with optimal cutpointing or signature discovery and then send that new column back into Kaplan-Meier or grouped summaries on the same cohort:
- treat the follow-up KM/table output as descriptive
- do not treat the repeated group-separation p-value as an independent confirmatory test
- use external validation if you want inferential claims for the derived grouping
- signature stability scores are heuristic composite rankings, not independently validated statistical tests
- describe the top-ranked signature as hypothesis-generating rather than confirmatory until it has external validation

### Calibration and Time-Dependent Importance

These outputs are useful, but should be interpreted carefully:
- calibration outputs are partly descriptive; a bin whose patients were not followed up to the evaluation time is reported as not estimable instead of carrying its last Kaplan-Meier value forward
- time-dependent importance refits the selected Random Survival Forest or Gradient Boosted Survival model and reports, for each raw feature and time point, how much the IPCW Brier score on the evaluation rows increases when that feature is shuffled; it is a permutation-based check, not a formal SurvSHAP(t) implementation
- partial dependence and counterfactual outputs are model-based local utilities, not causal intervention estimates

## Export

You can save results directly from the dashboard.

Available exports (each tab's `Export` menu):
- Survival curves:
  - summary table as `CSV`
  - pairwise table as `CSV`
  - curve as `PNG`
  - curve as `SVG`
- Cox model:
  - results table as `CSV`
  - diagnostics table as `CSV`
  - forest plot as `PNG`
  - forest plot as `SVG`
- Table 1:
  - table as `CSV`
  - table as `XLSX`
- Markers:
  - marker table as `CSV`
  - locked model as `JSON` (for `validate_locked_recipe` or the in-app validation)
  - REMARK checklist as `DOCX` or `Markdown`
  - summary figure, stability and rank plots as `PNG`
- Prediction models leaderboard:
  - TRIPOD+AI checklist as `DOCX` or `Markdown`, covering the latest ML and DL comparisons: data preparation, missing data, the evaluation design and shared splits, performance, and the winner's-curse caution when the best of several models is chosen on the same data
- ML and DL comparison:
  - comparison table as `CSV`
  - comparison plot as `PNG`
  - comparison plot as `SVG`
  - report table as `CSV`
  - report table as `Markdown`
  - report table as `LaTeX`
  - report table as `DOCX`

When Group by is active in Table 1:
- `Overall` refers to the grouped non-missing subset used in that table
- it is not a separate all-rows summary outside the grouped analysis frame

You can export comparison tables and report tables as:
- CSV
- Markdown
- LaTeX
- DOCX

ML and DL report-table export supports these formatting helpers:
- `default`
- `NEJM`
- `Lancet`
- `JCO`

These apply to report table export only.
They are formatting helpers, not official publisher-certified house styles.

Analysis exports end with provenance notes: the SurvStudio version that produced them (results changed in 0.2.0, see the release notes), the dataset fingerprint, and the request settings needed to replay the run.

## Evaluation Contract

- Training one ML model (`Train one model`) uses the deterministic holdout path.
- `Compare All Models` is the screening path for shared-model comparison, including repeated cross-validation when selected.
- DL single-model runs can use holdout or repeated-CV according to the visible evaluation controls.
- Every classical ML and deep model is trained and scored on identical row partitions for the same seed: one stratified 70/30 holdout helper and identical `StratifiedKFold` folds are shared by both families.
- Each comparison result carries an `evaluation_split_fingerprint` (a hash of which source rows were trained and scored in each split). The unified ML+DL leaderboard ranks the two families together only when the fingerprints match.
- Holdout and locked-test comparisons also return each model's test-set risk scores (`test_predictions`, `locked_test_predictions`). The leaderboard sends them to `POST /api/model-comparison-intervals`, which gives every model's C-index a 95% bootstrap interval and its difference from Cox PH a paired 95% interval (every draw scores all models on the same resampled patients). A model whose difference interval contains 0 is not distinguishable from Cox PH on that split; on small test sets this is the usual outcome, so do not report the top-ranked model as better on its point estimate alone.
- For report benchmarks, use repeated CV with a locked test set: the development set is used for all fitting, preprocessing, tuning, early stopping, and model selection; the locked test set is used once. Describe the training-set composition, the CV procedure, and the locked test set in the analysis notes.
- Datasets with repeated subject identifiers (for example several tumour samples per patient) are flagged: row-level splits would place one subject in both training and test data. Keep one row per subject before benchmarking.

### Download File Names

Downloaded files now use dataset-aware names.

Typical pattern:
- `{dataset}_{time}_{event}_{analysis}.{ext}`

Examples:
- `gbsg2_upload_ready_rfs_days_rfs_event_cox_results.csv`
- `tcga_luad_upload_ready_os_months_os_event_stage_group_km_curve.png`

This makes it easier to keep multiple cohorts and endpoints organized in the same download folder.

### Practical Save Check

If you want to verify saving on your machine:
1. load `Breast cancer (GBSG2)` or `Lung cancer (TCGA-LUAD)`
2. run Survival curves once
3. open `Export` and choose `Plot (PNG)` or `Plot (SVG)`
4. run the Cox model once
5. open `Export` and choose `Hazard ratios (CSV)`
6. in Prediction models, run `Compare All Models`
7. open a model's `Export` menu and choose one of the report tables

If the browser download dialog is blocked, allow downloads for `http://127.0.0.1:8000`.

## CLI

You can inspect a dataset without opening the UI:

```bash
survival-toolkit inspect path/to/data.csv
```

This prints a file-level profile so you can quickly check:
- column names
- missingness
- likely time columns
- likely event columns
- basic variable types

This is the fastest way to catch file-format problems before uploading a cohort in the browser.

## DL Runtime Note

Deep learning comparison can take substantial time on CPU-only machines, especially with:
- `Compare All Models`
- `Repeated Stratified CV`
- large `Epochs`
- larger shared ML/DL feature sets

`Compare All Models` is the slowest path because it trains every deep model in sequence. For example, a 100+ feature input set can take noticeably longer than the same cohort with a compact feature set.

For larger cohorts, note that the current `DeepSurv` and `Survival Transformer` paths use a full-batch Cox-style objective. That is statistically fine, but it can hit memory limits sooner than mini-batch tree workflows on 10k+ rows.

If you are running on a laptop without GPU acceleration, start with:
- `Epochs = 100`
- `Holdout`
- a compact feature set
- `Train one model` before `Compare All Models`

Then increase epochs or switch to repeated CV only after the single-run workflow looks correct.

## Current Limitations

- Cox PH currently reports an apparent C-index only. If you need bootstrap optimism correction or cross-validated Cox discrimination, run that validation outside the current dashboard workflow.
- Uploaded tables are limited to 1,000 candidate model features (5,000 columns). Wider omics data go into the Markers tab as a separate marker matrix; the ML and DL panels keep the 1,000-feature limit.
- Standard unpenalized Cox PH is not the right tool for very wide `p >> n` settings. Use the ML-panel `LASSO-Cox` path for penalized predictive screening instead of forcing a classical Cox PH fit.
- External-cohort validation in the web interface covers the locked marker model (Markers tab, Validate in another cohort; the file needs the same column names). For Cox and prediction models, load the separate cohort, reproduce the endpoint and covariate specification, and rerun the analysis. From Python, `validate_locked_recipe` validates a locked marker model (see Prognostic Marker Evaluation).
- Left truncation and competing risks are outside the current scope. In analysis notes, state explicitly that these workflows assume standard cause-specific survival with independent censoring and do not estimate cumulative incidence under competing events.
- Martingale residual plots are available as a visual screening aid for continuous covariates, but SurvStudio does not yet implement richer Cox linearity tooling such as spline recommendation or automated term selection.

## Development and Testing

After installing `pip install -e ".[dev]"`, run the test suite with:

```bash
pytest -q
```

Recent regression coverage includes:
- upload and parsing, including legacy text encodings and upload limits
- Kaplan-Meier / Cox / cohort-table workflows, with Kaplan-Meier, log-rank, RMST, proportional-hazards, and residual values checked against R `survival` 3.8 reference numbers
- the marker score screen against R `coxph` score tests, and the marker evaluation's family-wise error under a global null (`SURVSTUDIO_SLOW_TESTS=1 pytest tests/test_marker_evaluation.py`)
- derive-group options
- signature-search operator combinations
- ML and DL single-model and compare flows
- XAI endpoints
- export formats
- server behavior: request cancellation, the heavy-job limit, and the model-cache memory budget

Numerical agreement with R `survival`, lifelines and scikit-survival on the bundled GBSG2 and TCGA-LUAD cohorts (Kaplan-Meier estimates and intervals, medians, RMST, log-rank, Cox coefficients, standard errors, likelihoods, concordance, proportional-hazards statistics, and the marker engine's score tests and Cox fits) is reported in [numerical agreement](validation/numerical_agreement.md); regenerate it with `pip install -e ".[validation]"` and `python validation/agreement/run_agreement.py` (needs `Rscript` with the `survival` and `jsonlite` packages).

CI runs the suite on Linux with Python 3.11, 3.12, and 3.13 and on macOS and Windows with Python 3.11, checks the front-end scripts' syntax, builds the wheel and serves the page from a clean install, and runs the browser E2E test.

The front end is split into classic scripts (`static/app_core.js` … `static/app.js`) that `templates/index.html` loads in order and that share one global scope; only `app.js`, loaded last, runs startup code.

## Release Notes

Detailed change history lives in [RELEASE_NOTES.md](../RELEASE_NOTES.md).

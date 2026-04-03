# SurvStudio

SurvStudio is a local-first survival analysis workbench for single-event right-censored tabular data.
The public UI is a guided five-step workflow: load data, confirm the endpoint, choose one analysis path, run it, then review the result.

It supports:
- Kaplan-Meier curves and weighted log-rank tests
- Cox proportional hazards models
- optional machine-learning survival models
- optional deep-learning survival models
- manuscript-oriented table export
- one unified Predictive Models workspace for ML and DL screening

## Who This Is For

This project is for users who have cohort data in a spreadsheet-like table and want to:
- upload the cohort into a local browser dashboard
- run standard survival analyses without writing much code
- compare classical, ML, and DL survival models
- export figures and tables for reports or manuscripts
- keep one analysis visible at a time instead of juggling multiple panels

This project is **not** a general survival-analysis platform for every survival setting.
The current scope is:
- single-event survival analysis
- right-censored data
- tabular cohorts
- no left-truncated entry-time handling
- no competing-risks analysis
- no built-in "apply the locked model directly to an external cohort" workflow yet

## What The Built-In Example Data Is

The app includes four built-in example datasets:

1. `Synthetic Example`
- a synthetic cohort generated inside the package
- includes `os_months`, `os_event`, `pfs_months`, `pfs_event`
- includes demographic, treatment, stage, and biomarker variables
- useful for quick demos and testing

2. `TCGA LUAD (Real)`
- a bundled public TCGA LUAD cohort curated from UCSC Xena
- useful for a more realistic demo with a real public dataset

3. `Upload-Ready TCGA`
- a compact TCGA LUAD overall-survival table intended for immediate upload-style testing
- useful when you want a smaller real cohort without the extra clinical columns

4. `GBSG2 (Real)`
- a real public breast-cancer recurrence dataset
- useful for a fast end-to-end Kaplan-Meier, Cox, and ML smoke test with no missing values

If you want a file that you can upload manually instead of clicking a built-in loader, use:
- [examples/tcga_luad_nature2014_upload_ready.csv](examples/tcga_luad_nature2014_upload_ready.csv)
- [examples/tcga_luad_rnaseq_top100_upload.csv](examples/tcga_luad_rnaseq_top100_upload.csv)
- [examples/tcga_luad_rnaseq_top500_upload.csv](examples/tcga_luad_rnaseq_top500_upload.csv)
- [examples/gbsg2_jco1994_upload_ready.csv](examples/gbsg2_jco1994_upload_ready.csv)
- dataset notes: [examples/README.md](examples/README.md)

## Requirements

- Python `3.11` or newer
- internet access during the first install so `pip` can download dependencies
- if `python --version` or `python3 --version` prints `3.10` or older, do **not** use the standard `venv` path yet; use the `Project-local Conda fallback` section below first

## 1-Minute Quick Start

Use the first block that matches your machine.

### If You Already Have Python 3.11+

On macOS or Linux:

```bash
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m survival_toolkit
```

On Windows 11 PowerShell:

```powershell
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m survival_toolkit
```

### If Your Machine Only Has Python 3.10 But You Have Conda Or Micromamba

```bash
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
conda create -y -p .conda python=3.11 pip
./.conda/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m survival_toolkit
```

Then open:

```text
http://127.0.0.1:8000
```

Click `Synthetic Example` first.

## Install

### Easiest Path For A New Mac

Tested path target:
- macOS on Apple Silicon
- Homebrew available

Before you start:
- if another virtual environment is active, run `deactivate`
- if Conda is active, run `conda deactivate`
- the commands below create a project-local `.venv` and do not overwrite your system Python
- this install path is for running the app itself
- ML, DL, pytest, and Playwright can be added later only if you need them

```bash
brew install python@3.11
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
python3.11 -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
pip install -e .
python -m survival_toolkit
```

Then open:

```text
http://127.0.0.1:8000
```

Click `Synthetic Example` first.

### Easiest Path For A New Ubuntu Machine

Test target for CI:
- `ubuntu-latest`
- recommended for users: Ubuntu `24.04` or newer

Before you start:
- if another virtual environment is active, run `deactivate`
- if Conda is active, run `conda deactivate`
- the commands below create a project-local `.venv` and do not overwrite your system Python
- this install path is for running the app itself
- ML, DL, pytest, and Playwright can be added later only if you need them
- on Ubuntu `22.04`, `python3` is often still `3.10`; if you cannot install `python3.11` system-wide, use the `Project-local Conda fallback` section below

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip build-essential git
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
python3.11 -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
pip install -e .
python -m survival_toolkit
```

Then open:

```text
http://127.0.0.1:8000
```

### Standard install

This is the easiest path once Python `3.11+` is already available. It installs the dashboard and the classical analysis stack first. ML, DL, and development tools can be added later.

Before you start:
- if another virtual environment is active, run `deactivate`
- if Conda is active, run `conda deactivate`
- if `python3 --version` prints `3.10` or older, stop here and use the `Project-local Conda fallback` section below
- after activation, confirm that `which python` points to `.venv/bin/python`

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
pip install -e .
```

This path is for running the app. It does **not** install the optional ML stack, DL stack, pytest extras, or Playwright.

### Easiest Path For A New Windows 11 Machine

Tested path target:
- Windows `11`
- PowerShell
- Python `3.11` or newer installed
- Git for Windows installed

Before you start:
- open a new PowerShell window after installing Python so the `py` launcher is available
- confirm `py -3.11 --version` works before continuing
- the commands below avoid PowerShell activation-policy issues by calling the venv Python directly
- this install path is for running the app itself
- ML, DL, pytest, and Playwright can be added later only if you need them

```powershell
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m survival_toolkit
```

Then open:

```text
http://127.0.0.1:8000
```

If you prefer to activate the environment first in PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m survival_toolkit
```

### Project-local Conda fallback

Use this when your machine has Conda or Micromamba available but the system Python is only `3.10` or older.

This is the safest fallback on:
- Ubuntu `22.04`
- shared servers where you cannot use `sudo`
- WSL setups where the OS Python is older than the project requirement

It bootstraps a project-local Python `3.11`, then creates the normal `.venv` from that interpreter.

```bash
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
conda create -y -p .conda python=3.11 pip
./.conda/bin/python -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
pip install -e .
python -m survival_toolkit
```

If you prefer to run directly from the Conda environment without creating `.venv`, this also works:

```bash
conda create -y -p .conda python=3.11 pip
conda run -p ./.conda python -m pip install --upgrade pip
conda run -p ./.conda python -m pip install -e .
conda run -p ./.conda python -m survival_toolkit
```

### Optional ML, DL, and development installs

Start with the app-only install above, then add extras only if you need them:

- Add ML models:

```bash
pip install -e ".[ml]"
```

- Add deep learning models:

```bash
pip install -e ".[dl]"
```

- Full local development stack:

```bash
pip install -e ".[dev]"
```

- Everything:

```bash
pip install -e ".[all]"
```

Notes:
- `.[ml]` adds `scikit-survival` and `shap`
- `.[dl]` adds `torch`
- `.[dev]` includes pytest, httpx, and `kaleido` plus the ML and DL extras
- `.[all]` includes the ML, DL, export, and browser-test extras
- on Linux, `.[dl]`, `.[dev]`, and `.[all]` can download a large PyTorch wheel and, depending on platform resolution, additional CUDA runtime packages
- if you only want to run the dashboard or classical survival workflows, stay with `pip install -e .`

### Optional Browser E2E Test Install

Only use this if you want to run the headless browser download test.

```bash
pip install -e ".[dev,e2e]"
python -m playwright install chromium
```

On Ubuntu CI or a fresh Linux machine, if Chromium system dependencies are missing, use:

```bash
python -m playwright install --with-deps chromium
```

## Run

Start the local app:

```bash
python -m survival_toolkit
```

Then open:

```text
http://127.0.0.1:8000
```

The app opens in the guided workflow by default. For predictive modeling, use the unified `Predictive Models` workspace to compare ML and DL models together or test one selected model at a time.

If `python -m survival_toolkit` does not start the server, check:
- the virtual environment is activated
- installation finished without errors
- you are inside the project directory

## Installation Notes

- The fastest first run is `pip install -e .`.
- Add `.[ml]` only when you need the optional machine-learning models.
- Add `.[dl]` only when you need the optional deep-learning models.
- Add `.[dev]` when you want the full local development stack and normal pytest suite.
- On Linux, `.[dl]` and `.[dev]` may be large because PyTorch can pull platform-specific runtime packages.
- If your machine only has Python `3.10`, bootstrap Python `3.11` first with the `Project-local Conda fallback` section.
- Browser E2E testing is optional and uses the separate `e2e` extra.
- The app itself does not need Playwright.

## First 5 Minutes

If this is your first time:

1. Click `Synthetic Example` or `TCGA LUAD (Real)`
2. Set:
   - time column: for example `os_months`
   - event column: for example `os_event`
3. Start with:
   - Kaplan-Meier
   - Cohort Table
   - Cox PH
4. If needed, derive a group from a biomarker using:
   - median split
   - tertile split
   - quartile split
   - percentile split
   - extreme split
   - optimal cutpoint
5. Move to ML or DL comparison only after the classical analysis makes sense
6. If you want to validate a file before opening the UI, run:

```bash
survival-toolkit inspect path/to/data.csv
```

## Recommended Real-Data Workflows

### TCGA LUAD Workflow

Best starting dataset choices:
- `Upload-Ready TCGA`
- `Load TCGA LUAD`

Recommended study columns:
- time column: `os_months`
- event column: `os_event`
- event-positive value: `1`
- group column: `stage_group`

Recommended first figures and tables:
1. Kaplan-Meier by `stage_group`
2. Cohort Table grouped by `stage_group`
3. Cox PH with:
   - covariates: `age`, `sex`, `stage_group`, `smoking_status`
   - categorical covariates: `sex`, `stage_group`, `smoking_status`
4. ML comparison with:
   - features: `age`, `sex`, `stage_group`, `smoking_status`
   - categorical features: `sex`, `stage_group`, `smoking_status`
5. DL smoke or comparison with the same feature set

Recommended manuscript outputs:
- Kaplan-Meier plot by stage
- Cox hazard-ratio forest plot
- cohort summary table stratified by stage
- ML comparison table or repeated-CV manuscript table

### GBSG2 Workflow

Best starting dataset choice:
- `GBSG2 (Real)`

Recommended study columns:
- time column: `rfs_days`
- event column: `rfs_event`
- event-positive value: `1`
- group column: `horTh`

Recommended first figures and tables:
1. Kaplan-Meier by `horTh`
2. Kaplan-Meier by `menostat`
3. Cohort Table grouped by `horTh`
4. Cox PH with:
   - covariates: `age`, `horTh`, `menostat`, `pnodes`, `tgrade`, `tsize`
   - categorical covariates: `horTh`, `menostat`, `tgrade`
5. ML comparison with:
   - features: `age`, `horTh`, `menostat`, `pnodes`, `tgrade`, `tsize`
   - categorical features: `horTh`, `menostat`, `tgrade`

Recommended manuscript outputs:
- Kaplan-Meier plot for hormonal therapy groups
- Cox forest plot for recurrence-free survival
- cohort table grouped by hormonal therapy
- ML comparison table for recurrence discrimination

### Synthetic Example Workflow

Best starting dataset choice:
- `Synthetic Example`

Recommended study columns:
- time column: `os_months`
- event column: `os_event`
- event-positive value: `1`
- group column: `stage` or `treatment`

Recommended first figures and tables:
1. Kaplan-Meier by `stage`
2. Kaplan-Meier by `treatment`
3. Cohort Table grouped by `stage`
4. Cox PH with:
   - covariates: `age`, `sex`, `stage`, `treatment`, `biomarker_score`, `immune_index`
   - categorical covariates: `sex`, `stage`, `treatment`
5. ML comparison with:
   - features: `age`, `sex`, `stage`, `treatment`, `biomarker_score`, `immune_index`
   - categorical features: `sex`, `stage`, `treatment`

This synthetic dataset does **not** use `stage_group` or `treatment_group`.
The actual column names are `stage` and `treatment`.

## Input Data Format

Supported file types:
- `csv`
- `tsv`
- `xlsx`
- `xls`
- `parquet`

Expected structure:
- one row per patient or subject
- one column for follow-up time
- one column for event status
- remaining columns as covariates

### Minimum Required Columns

At minimum, your file needs:

1. a time column
- numeric
- positive values only
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
- positive for all analyzable rows

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
- survival time is positive
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
6. use `Load Example` first to confirm the app itself is working
7. use `survival-toolkit inspect path/to/file.csv` to inspect your file before opening the UI

## Main Analyses

### Kaplan-Meier

Use this when you want:
- survival curves by group
- median survival
- RMST
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
- deterministic holdout
- repeated stratified CV
- manuscript-oriented result tables

Single-model ML training also supports:
- `Fast mode (skip SHAP)` for faster turnaround
- feature-importance output for trained models

LASSO-Cox note:
- use this when the feature set is too wide for stable unpenalized Cox PH
- it is a predictive penalized Cox path, not an inferential hazard-ratio workflow
- SHAP, partial dependence, and counterfactual analysis remain tree-model features only

Practical note:
- `Compare All` is usually faster than single-model `Train a model`
- `Compare All` focuses on cross-model scoring
- single-model `Train a model` may do extra post-fit work such as feature importance and optional SHAP computation
- for quick RSF checks on larger cohorts, leave `Fast mode` enabled
- if `TreeExplainer` is unsupported, SHAP falls back to a tightly capped `KernelExplainer` approximation using a small background/evaluation sample, so treat the ranking as approximate rather than publication-grade

### Deep Learning

Implemented DL paths:
- DeepSurv
- DeepHit
- Neural MTLR
- Survival Transformer
- Survival VAE

Deep comparison supports:
- deterministic holdout
- repeated CV
- early stopping
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
- DeepHit uses a stabilized ranking-loss scale (`sigma=1.0`) rather than the earlier sharper default.
- `Neural MTLR` is implemented as an MTLR-inspired discrete-time neural variant for workflow comparison, not as a literal reference reproduction of the original formulation.
- `Survival VAE` should be interpreted as a VAE-inspired latent representation model for clustering and risk screening. SurvStudio does not claim validated generative simulation or uncertainty estimation from this path.
- Cox-style DL paths (`DeepSurv`, `Survival Transformer`) use monitor C-index on an internal training-partition subset for early stopping. Discrete-time/VAE paths continue to use monitor loss. Neither monitor curve is the final holdout or external validation metric.
- Deep-model summaries currently report discrimination (`C-index`) only. SurvStudio does not yet compute IBS for deep-model outputs, so calibration/error comparisons are not directly symmetric with the ML module.

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
- PH diagnostics currently use rank-based Spearman correlations between Schoenfeld residuals and log time.
- That is useful for screening PH problems, but it is not the same thing as a full Grambsch-Therneau omnibus test.
- A Cox `C-index = 0.65` means the fitted model ranks about `65%` of comparable patient pairs in the observed risk order; it is not "65% accuracy."

### Model C-index

The app explicitly distinguishes different evaluation modes:

- `Holdout C-index`
  - discrimination measured on a deterministic holdout split
  - this is one split only, so no CI or SD is shown
- `Repeated-CV mean C-index`
  - average discrimination across repeated stratified CV
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
If you use them in a manuscript:
- report how the cutpoint was selected
- prefer selection-adjusted p-values when available
- validate the cutpoint on separate data

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

### Calibration and Time-Dependent Importance

These outputs are useful, but should be interpreted carefully:
- calibration outputs are partly descriptive
- time-dependent importance is an approximate proxy, not a formal SurvSHAP(t) implementation
- partial dependence and counterfactual outputs are model-based local utilities, not causal intervention estimates

## Export

You can save results directly from the dashboard.

Available exports:
- Kaplan-Meier:
  - summary table as `CSV`
  - pairwise table as `CSV`
  - curve as `PNG`
  - curve as `SVG`
- Cox PH:
  - results table as `CSV`
  - diagnostics table as `CSV`
  - forest plot as `PNG`
  - forest plot as `SVG`
- Cohort Table:
  - table as `CSV`

When Group by is active in the cohort table:
- `Overall` refers to the grouped non-missing subset used in that table
- it is not a separate all-rows summary outside the grouped analysis frame
- ML and DL comparison:
  - comparison table as `CSV`
  - comparison plot as `PNG`
  - comparison plot as `SVG`
  - manuscript table as `CSV`
  - manuscript table as `Markdown`
  - manuscript table as `LaTeX`
  - manuscript table as `DOCX`

You can export comparison tables and manuscript tables as:
- CSV
- Markdown
- LaTeX
- DOCX

ML and DL manuscript-table export supports these formatting helpers:
- `default`
- `NEJM`
- `Lancet`
- `JCO`

These apply to manuscript table export only.
They are formatting helpers, not official publisher-certified house styles.

## Evaluation Contract

- ML `Train a model` currently supports the deterministic holdout path for a single fitted model.
- ML `Compare All` is the screening path for shared-model comparison, including repeated cross-validation when selected.
- DL single-model runs can use holdout or repeated-CV according to the visible evaluation controls.

### Download File Names

Downloaded files now use dataset-aware names.

Typical pattern:
- `{dataset}_{time}_{event}_{analysis}.{ext}`

Examples:
- `gbsg2_upload_ready_rfs_days_rfs_event_cox_results.csv`
- `tcga_luad_upload_ready_os_months_os_event_stage_group_km_curve.png`
- `gbsg2_upload_ready_rfs_days_rfs_event_ml_manuscript_table_jco.docx`

This makes it easier to keep multiple cohorts and endpoints organized in the same download folder.

### Practical Save Check

If you want to verify saving on your machine:
1. load `GBSG2 (Real)` or `Upload-Ready TCGA`
2. run Kaplan-Meier once
3. click `PNG` or `SVG`
4. run Cox PH once
5. click `Results`
6. run ML `Compare All`
7. click one of the manuscript export buttons

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
- `Compare All`
- `Repeated Stratified CV`
- large `Epochs`
- larger shared ML/DL feature sets

`Compare All` is the slowest DL path because it trains all implemented deep models in sequence. For example, a 100+ feature input set can take noticeably longer than the same cohort with a compact feature set.

For larger cohorts, note that the current `DeepSurv` and `Survival Transformer` paths use a full-batch Cox-style objective. That is statistically fine, but it can hit memory limits sooner than mini-batch tree workflows on 10k+ rows.

If you are running on a laptop without GPU acceleration, start with:
- `Epochs = 100`
- `Holdout`
- a compact feature set
- `Train a model` before `Compare All`

Then increase epochs or switch to repeated CV only after the single-run workflow looks correct.

## Practical Notes

- The toolkit is exploratory by default.
- It is useful for analysis, figure generation, and workflow standardization.
- Strong manuscript claims still require:
  - external validation
  - sensitivity checks
  - disciplined model selection
  - careful interpretation of calibration and cutpointing

## Current Limitations

- Cox PH currently reports an apparent C-index only. If you need bootstrap optimism correction or cross-validated Cox discrimination, run that validation outside the current dashboard workflow.
- Standard unpenalized Cox PH is not the right tool for very wide `p >> n` settings. Use the ML-panel `LASSO-Cox` path for penalized predictive screening instead of forcing a classical Cox PH fit.
- External-cohort validation is currently a manual workflow: load the separate cohort, reproduce the endpoint and covariate specification, and rerun the analysis.
- Left truncation and competing risks are outside the current scope.

## Development and Testing

Run the test suite with:

```bash
pytest -q
```

Recent regression coverage includes:
- upload and parsing
- Kaplan-Meier / Cox / cohort-table workflows
- derive-group options
- signature-search operator combinations
- ML and DL single-model and compare flows
- XAI endpoints
- export formats

## Release Notes

Detailed change history lives in [RELEASE_NOTES.md](./RELEASE_NOTES.md).

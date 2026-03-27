# SurvStudio

SurvStudio is a local-first survival analysis workbench for single-event right-censored tabular data.

It supports:
- Kaplan-Meier curves and weighted log-rank tests
- Cox proportional hazards models
- optional machine-learning survival models
- optional deep-learning survival models
- manuscript-oriented table export

## Who This Is For

This project is for users who have cohort data in a spreadsheet-like table and want to:
- upload the cohort into a local browser dashboard
- run standard survival analyses without writing much code
- compare classical, ML, and DL survival models
- export figures and tables for reports or manuscripts

This project is **not** a general survival-analysis platform for every survival setting.
The current scope is:
- single-event survival analysis
- right-censored data
- tabular cohorts

## What The Built-In Example Data Is

The app includes four built-in example datasets:

1. `Load Example`
- a synthetic cohort generated inside the package
- includes `os_months`, `os_event`, `pfs_months`, `pfs_event`
- includes demographic, treatment, stage, and biomarker variables
- useful for quick demos and testing

2. `Load TCGA LUAD`
- a bundled public TCGA LUAD cohort curated from UCSC Xena
- useful for a more realistic demo with a real public dataset

3. `Upload-Ready TCGA`
- a compact TCGA LUAD overall-survival table intended for immediate upload-style testing
- useful when you want a smaller real cohort without the extra clinical columns

4. `GBSG2`
- a real public breast-cancer recurrence dataset
- useful for a fast end-to-end Kaplan-Meier, Cox, and ML smoke test with no missing values

If you want a file that you can upload manually instead of clicking a built-in loader, use:
- [examples/tcga_luad_nature2014_upload_ready.csv](examples/tcga_luad_nature2014_upload_ready.csv)
- [examples/gbsg2_jco1994_upload_ready.csv](examples/gbsg2_jco1994_upload_ready.csv)
- dataset notes: [examples/README.md](examples/README.md)

## Requirements

- Python `3.11` or newer
- internet access during the first install so `pip` can download dependencies

## Install

### Easiest Path For A New Mac

Tested path target:
- macOS on Apple Silicon
- Homebrew available

Before you start:
- if another virtual environment is active, run `deactivate`
- if Conda is active, run `conda deactivate`
- the commands below create a project-local `.venv` and do not overwrite your system Python

```bash
brew install python@3.11
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
python3.11 -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
pip install -e ".[dev]"
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
pip install -e ".[dev]"
python -m survival_toolkit
```

Then open:

```text
http://127.0.0.1:8000
```

### Standard install

This is the easiest path once Python is already available. It installs the dashboard plus the optional ML and DL features.

Before you start:
- if another virtual environment is active, run `deactivate`
- if Conda is active, run `conda deactivate`
- after activation, confirm that `which python` points to `.venv/bin/python`

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Smaller installs

If you only want part of the stack:

- Core dashboard only:

```bash
pip install -e .
```

- Core + ML only:

```bash
pip install -e ".[ml]"
```

- Core + DL only:

```bash
pip install -e ".[dl]"
```

- Everything:

```bash
pip install -e ".[all]"
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

If `python -m survival_toolkit` does not start the server, check:
- the virtual environment is activated
- installation finished without errors
- you are inside the project directory

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
   - custom cutoff
   - optimal cutpoint
5. Move to ML or DL comparison only after the classical analysis makes sense

## Recommended Real-Data Workflows

If you use one of the built-in real datasets, the UI now exposes dataset-specific preset buttons:
- `Apply KM/Cox Preset`
- `Apply ML/DL Preset`

These presets fill the most useful starting columns and feature sets for the selected cohort.

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

Examples:
- `1 / 0`
- `yes / no`
- `death / alive`
- `R / N`

Important:
- this platform expects a **binary event indicator**
- if your status column has more than two states, recode it first

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
- Random Survival Forest
- Gradient Boosted Survival
- model comparison against Cox PH

Comparison supports:
- deterministic holdout
- repeated stratified CV
- manuscript-oriented result tables

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

## How To Read Results

### Kaplan-Meier

- Curves farther apart usually suggest different survival experiences between groups.
- The log-rank p-value tests whether the group curves differ.
- A statistically small p-value does **not** automatically mean the effect is clinically important.

### Cox PH

- Hazard ratio `> 1`: higher hazard
- Hazard ratio `< 1`: lower hazard
- Confidence intervals crossing `1` mean the estimate is compatible with no effect

### Model C-index

The app explicitly distinguishes different evaluation modes:

- `Holdout C-index`
  - discrimination measured on a deterministic holdout split
- `Repeated-CV mean C-index`
  - average discrimination across repeated stratified CV
- `Apparent C-index`
  - measured on the training/analyzable cohort
  - optimistic
  - should not be treated as external validation

### Cutpoints

Optimal cutpoints are exploratory by nature.
If you use them in a manuscript:
- report how the cutpoint was selected
- prefer selection-adjusted p-values when available
- validate the cutpoint on separate data

### Calibration and Time-Dependent Importance

These outputs are useful, but should be interpreted carefully:
- calibration outputs are partly descriptive
- time-dependent importance is an approximate proxy, not a formal SurvSHAP(t) implementation

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

The app also supports manuscript-style templates:
- `default`
- `NEJM`
- `Lancet`
- `JCO`

These are formatting helpers, not official publisher-certified house styles.

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

## Practical Notes

- The toolkit is exploratory by default.
- It is useful for analysis, figure generation, and workflow standardization.
- Strong manuscript claims still require:
  - external validation
  - sensitivity checks
  - disciplined model selection
  - careful interpretation of calibration and cutpointing

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

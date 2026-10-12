# Your first SurvStudio analysis

[Back to quick start](../README.md) · [Installation help](installation.md) · [Detailed user guide](user-guide.md)

Start with a sample, draw survival curves, examine risk factors, then compare prediction models.
You do not need to write analysis code. After the [installation](../README.md#1-install), run `survstudio`
and open **http://127.0.0.1:8000** in your browser. Keep the terminal open while you work.

## Try one example from start to finish

### 1. Load the sample

Click **Breast cancer (GBSG2)** on the start screen. It contains 686 patients.
Check these settings before running an analysis:

| Setting | Choose |
|---|---|
| Time column | `rfs_days` |
| Event column | `rfs_event` |
| Event value | `1` |
| Time unit | `Days` |
| Group by | `horTh` |

Here the endpoint is recurrence-free survival, and `horTh` records hormone therapy (`no` / `yes`).
The sample preset fills the usual columns; still check them yourself. The time-unit field is under
**Curve options**. Changing its label does not convert days to months.

### 2. Draw the survival curves

Open **Survival curves** and click **Draw curves**. Leave the default 95% confidence interval and
log-rank test in place for this first run.

- The curve estimates the proportion remaining recurrence-free over time.
- The band shows uncertainty. The risk table counts patients still being followed without an event.
- Median survival is the time when the curve first reaches 50%. “Not reached” means it never reached
  50% during the observed follow-up; it does not mean no events occurred.
- The log-rank p-value compares the curves. It does not measure the size of a difference or prove a treatment effect.

The [recorded example](../examples/results.md#survival-curves) has medians of **1,528 days** for `no`
and **2,018 days** for `yes`. These are observed group differences, not a causal treatment comparison.

### 3. Examine risk factors with Cox

Open **Risk factors (Cox)**. Select:

| Variables | Treat as categorical |
|---|---|
| `age`, `horTh`, `menostat`, `pnodes`, `tgrade`, `tsize` | `horTh`, `menostat`, `tgrade` |

Expand the variable selector if it is collapsed, then click **Run Analysis**.
Leave patient IDs, time and event columns out of the variable list.

Read the reference category or measurement unit next to each hazard ratio (HR):

- **HR > 1:** higher estimated event rate at a given time, after adjustment for the other variables.
- **HR < 1:** lower estimated event rate. HR is not an absolute probability of recurrence.
- **95% interval includes 1:** the estimate remains compatible with no association.
- **Model checks:** a warning can limit interpretation even when a p-value is small.

Numeric variables use their stated units; for example, age is per year when recorded in years.
Cox excludes rows missing a selected variable. Its **Apparent C-index** uses the fitting patients;
it is not a held-out prediction score.

### 4. Compare ML and DL predictions

Open **Compare models**. Choose the same six variables and three categorical variables as above.
The predictive inputs are shared between ML and DL.

For a first run, click **Train one model**, choose **Random Survival Forest**, and keep
**100 trees**, **Fast mode**, **Holdout** and **seed 42**. Click **Train Random Survival Forest**.
To try DL, choose **DeepSurv** in the model workbench and keep **100 epochs**, **layers 64,64**,
**learning rate 0.001**, **dropout 0.1** and **patience 10**, then click **Train DeepSurv**.

Click **Back to leaderboard**, then **Compare All Models** when you are ready to compare the full
ML/DL set with Cox as the baseline. This takes longer than a single fit with explanations disabled.
Use the same inputs, seed and evaluation settings for a fair comparison.
Transformer and VAE are experimental options.

The [recorded single-model runs](../examples/results.md#mldl-one-gbsg2-run) use 480 training patients
and 206 holdout patients. Their C-indices are **0.688 for RSF** and **0.677 for DeepSurv**.
Small differences on one split do not establish that a model is better. Versions and settings can affect the values.

### 5. Save your work

Use **Save results** or the panel's **Save files** menu to download the figure and result table.
Keep the input file and settings with them. Downloads are not a saved browser session:
uploaded data leave memory when the local server stops.

## Prepare your own file

Use one patient per row and one variable per column, with a single header row.
Do not include merged cells, title rows, totals or repeated visits.

```csv
patient_id,os_months,os_event,age,stage,treatment
P001,12.4,1,67,III,Standard
P002,18.0,0,59,II,Combination
P003,7.2,1,72,IV,Standard
```

These three rows illustrate the format; they are not enough to evaluate a prediction model.

| Column | How to prepare it |
|---|---|
| Patient ID | A unique code per patient. Keep it for checking identity, not as a predictor. |
| Time | Numeric time from a defined starting point to the event or last follow-up; use one unit. Zero is allowed, negative durations are invalid. |
| Event | `1` if the specified event happened, `0` if censored. Confirm the positive value after upload. |
| Numeric input | Numbers without units in the cells: `67`, not `67 years`. Leave missing cells blank. |
| Categorical input | Consistent labels such as `I`, `II`, `III`, `IV`. Mark numeric category codes as categorical too. |

In Excel, save as **CSV UTF-8** or **Excel Workbook (.xlsx)**, then upload the file.
For a downloadable example, open a [sample CSV](../examples/gbsg2_jco1994_upload_ready.csv) on GitHub
and use **Download raw file**; saving the GitHub page as HTML does not produce a CSV.

Define the endpoint first. Use overall survival and recurrence-free survival in separate analyses,
with the matching time and event columns. Do not mix days and months or label another endpoint as the event.
A status field with relapse, death and censoring needs an explicit endpoint definition before recoding;
SurvStudio does not perform competing-risks analysis.

### Missing values and exclusions

| Analysis | What happens |
|---|---|
| Survival curves | Unusable time/event or missing group values can exclude rows; invalid endpoint inputs can stop the analysis. |
| Cox | Rows missing time/event or a selected variable are excluded. |
| ML/DL | Unusable outcomes are excluded or the analysis is refused. Missing numeric predictors use the training-partition median; category encoding is also fitted on the training partition. |

Missing categorical values get their own indicator when present in training; otherwise they are
encoded as the reference level. Read the reported usable-patient count and warnings rather than
assuming that every uploaded row was analysed. Correct invalid times in the source file before analysis.

### Keep future information out of predictors

Select inputs that would be available at the intended prediction time. Exclude outcome copies,
survival groups derived from outcomes, dates of death or recurrence, and measurements made after
the prediction time. Keep one row per patient before splitting data.

SurvStudio fits imputation and encoding within training partitions and shares evaluation splits
between ML and DL. These protections do not establish when a measurement was taken or find every
disguised outcome field. You still need to check the meaning of the input columns.
Do not select genes or tune settings using outcomes from the full cohort before a train/test split.

## Understand ML/DL results

ML and DL learn to rank patients by predicted event risk. They do not establish which treatment
causes a better outcome, and a more complex model is not automatically more accurate.

| Output | How to read it |
|---|---|
| C-index | Higher means better ranking of comparable patient pairs. `0.65` is roughly 65% correctly ranked pairs, not 65% prediction accuracy or a survival probability. |
| Score interval / difference from Cox | Read uncertainty with the score. If the difference interval includes 0, that comparison has not established an advantage over Cox on those evaluation data. |
| Feature importance / SHAP | Describes the fitted predictions, not a causal effect. Read any companion-model or approximation notice. |
| DL training / monitor curve | Shows fitting and early stopping. Falling loss alone does not establish good performance on patients kept out of fitting. |
| IBS / Brier skill | Available for ML: lower IBS is better; positive Brier skill improves on the no-predictor reference over the reported evaluation times. DL currently has no IBS output. |
| Blank, `N/A` or withheld result | Read the reason. Unavailable or withheld is not zero and must not be treated as a valid score. |

Always check **where** the score was measured:

| Label | Patients used for scoring |
|---|---|
| Apparent | The fitting patients. Often optimistic; small cohorts may fall back to this mode. |
| Holdout | A fixed 30% of patients kept out of that fit; one split can be unstable. |
| Repeated CV | Multiple held-out folds. Mean ± SD describes variation across folds, not a confidence interval. |
| Locked test | Patients reserved before CV. Select the model on development CV, then report its test score. |

A locked test is still a subset of the same cohort, not an external cohort.
Do not choose settings or a new winner after inspecting test scores. New external validation requires
applying the already fitted model and transformations to another cohort; refitting on that cohort
is a new analysis. The browser currently provides this locked external workflow for **Markers**,
not the general ML/DL leaderboard.

## Choose an option combination

These are starting examples, not settings shown to be optimal. Change one setting at a time.
Use the GBSG2 inputs above while learning the controls.

| Goal | Action and settings | What to expect |
|---|---|---|
| First curves | Draw curves; `horTh`; 95% interval; default log-rank | Curves, risk counts and group medians. |
| Adjusted associations | Cox; six inputs above; three categorical inputs | HRs and model checks; apparent performance. |
| Quick ML fit | Train one model; RSF; 100 trees; Fast mode; seed 42 | Holdout risk ranking without SHAP. |
| First DL fit | Train one model; DeepSurv; 100 epochs; layers `64,64`; dropout `0.1`; learning rate `0.001`; patience `10`; seed 42 | Early-stopped neural risk ranking. |
| First full comparison | Compare All Models; Holdout; seed 42; other settings at defaults | ML/DL scores on the same evaluation rows. |
| Try CV controls | Compare All Models; Repeated Stratified CV; 2 folds × 1 repeat; locked test 30%; seed 42 | A short CV trial, not evidence of stable performance. |
| More repeated evaluation | Compare All Models; Repeated Stratified CV; 5 folds × 3 repeats; locked test 30%; seed 42 | More fitting and higher runtime; report the CV-selected model's locked-test result. |

Use CV only when there are enough patients and events in each partition. If the run is refused,
read the message and reconsider the design; do not remove test protection just to get a higher score.

### Which settings affect which models?

| Setting | Applies to | Start with |
|---|---|---|
| Trees | RSF / gradient boosting | `100` |
| ML learning rate | Gradient boosting, not RSF | `0.1` |
| Fast mode | Single ML fit; skips SHAP | On |
| Epochs / patience | DL; maximum training / early stopping | `100` / `10` |
| DL learning rate / dropout | DL optimization / regularization | `0.001` / `0.1` |
| Hidden layers | DeepSurv, DeepHit, MTLR and VAE | `64,64` |
| Time bins / batch size | DeepHit and MTLR | `50` / `64` |
| Width / heads / layers | Experimental Transformer | `64` / `4` / `2`; width must be divisible by heads |
| Latent dimension / clusters | Experimental VAE | `8` / `3` |

DeepSurv, Transformer and VAE use full-batch training; changing the batch-size field does not change
those trainers. Single-model **ML** uses holdout or apparent scoring; CV controls apply to comparisons.
Single-model **DL** supports holdout or CV, but the locked test option applies to comparisons.

For detailed controls and limits, see [ML](user-guide.md#machine-learning),
[DL](user-guide.md#deep-learning) and the [evaluation contract](user-guide.md#evaluation-contract).
For startup, file or package errors, see [troubleshooting](installation.md#common-problems).


## Compare clinical inputs with added predictors

In **Prediction models**, open **Do added predictors help?**. Choose separate clinical and added
predictor sets, mark categorical variables, and choose one added predictor and a clinical group.
For a ready example, click **Try molecular demo**: it loads 480 simulated patients with age,
stage and two synthetic genes. This replaces the loaded dataset.

Click **Compare predictor sets**. Gradient boosting, survival forests and DeepSurv are available
in this explorer; the existing model board still compares all supported models. Both feature sets
use the same training and test patients. Positive change in C-index means better risk ranking in
that holdout; an interval crossing zero leaves the improvement uncertain. More inputs or a neural
model need not improve prediction.

The lower plot shows how the fixed model's predicted risk changes across categorical groups.
It uses training data as its reference and leaves unsupported combinations blank. The score is
in training standard deviations, not a survival probability. A curved or different group pattern
describes the fitted model; it does not establish a biological interaction or a causal effect.

**Save results (JSON)** retains both comparisons, settings and curves. **Save curves (CSV)** keeps
unsupported values blank. Changing inputs disables old exports and requires a new run. Repeated
choices on these test patients need validation in independent data.

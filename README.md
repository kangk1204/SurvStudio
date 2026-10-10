# SurvStudio

SurvStudio does survival analysis of your own data in your web browser, without programming; your data never
leave your computer.

![SurvStudio with the lung cancer sample open: Kaplan-Meier survival curves by tumour stage](github_images/hero.png)

## What you can do

## Install

The first install takes a few minutes and needs an internet connection. You do not need Python or git: a small
installer called [uv](https://docs.astral.sh/uv/) fetches everything SurvStudio needs and keeps it apart from the
rest of your computer.

**Windows**

1. Open PowerShell: press the Windows key, type `PowerShell` and press Enter.
2. Install uv. Copy this line, paste it into PowerShell (Ctrl+V or right-click) and press Enter:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   If you prefer winget: `winget install --id=astral-sh.uv -e`.
3. Close PowerShell and open it again, so that it finds `uv`.
4. Install SurvStudio:

   ```powershell
   uv tool install --python 3.12 "survstudio[all] @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
   ```

**macOS**

1. Open Terminal (Finder, Applications, Utilities, Terminal).
2. On a Mac with Apple silicon (M1 or later), install Apple's free command-line tools, which one small part of
   SurvStudio needs. Run the line below and click Install; if it says the tools are already installed, go on.

   ```bash
   xcode-select --install
   ```

3. Install uv (or use `brew install uv`):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. Close Terminal and open it again.
5. Install SurvStudio:

   ```bash
   uv tool install --python 3.12 "survstudio[all] @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
   ```

**Linux**

1. Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).
2. Open a new terminal.
3. Run the same `uv tool install` command as for macOS. On an ARM computer, install a compiler first
   (`sudo apt install build-essential` on Ubuntu).

**Start SurvStudio.** In the terminal, type:

```bash
survstudio
```

Then open <http://127.0.0.1:8000> in your web browser. Keep the terminal window open while you work: SurvStudio
runs there. To stop it, press Ctrl+C in the terminal or click **Shutdown** at the top right of the page. To start
again, type `survstudio`.

**What `[all]` installs.** Everything: Excel and Parquet files, the machine-learning models (scikit-survival) and
the deep-learning models (PyTorch). The download is about 300 MB on Windows and macOS and about 3 GB on Linux, where
PyTorch comes with NVIDIA's GPU libraries; plan for 1 GB of disk space (6 GB on Linux). If you do not need deep
learning, the lighter install downloads about 200 MB and has everything else:

```bash
uv tool install --python 3.12 "survstudio[formats,ml] @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
```

`--python 3.12` makes uv use Python 3.12, for which almost every part of SurvStudio comes ready-made (newer Python
versions would need a compiler). uv downloads it if needed and leaves any other Python on your computer alone.

**Update or remove.**

| To | Run |
|---|---|
| Update to the latest version | `uv tool upgrade survstudio` |
| Remove SurvStudio | `uv tool uninstall survstudio` |
| Also delete the downloaded files | `uv cache clean` |

**Other ways.** With [Docker](https://www.docker.com/): `docker build -t survstudio https://github.com/kangk1204/SurvStudio.git`,
then `docker run --rm -p 127.0.0.1:8000:8000 survstudio`. To work on the code, see the
[developer install](docs/reference.md#14-developer-install).

## Your first analysis in 10 minutes

This walkthrough uses a sample that comes with SurvStudio: 489 patients with lung adenocarcinoma from The Cancer
Genome Atlas (TCGA-LUAD), followed for overall survival.

**1. Open the sample.** On the start screen, click **Lung cancer (TCGA-LUAD)**. (For your own data, drop your file
on the left instead; see [Prepare your own data](#prepare-your-own-data).)

![Start screen with the upload area and the three sample cohorts](github_images/start_screen.png)

**2. Check the outcome.** The bar at the top says which columns hold the outcome. **Time** is `os_months`, the
months from diagnosis to death or last contact. **Event** is `os_event`, and **Event value** `1` means the patient
died; `0` means the patient was alive at last contact (censored). **Group by** is `stage_group`.

![The outcome bar: time os_months, event os_event, event value 1, grouped by stage_group](github_images/outcome_bar.png)

**3. Survival curves by stage.** On the **Survival curves** tab, click **Run Analysis**. Each line shows the share
of patients still alive over time; the shaded bands are 95% confidence intervals and the table under the plot
gives the numbers at risk. The log-rank test (p < 0.001) says the stages differ. The **KM Summary** table below
gives the median survival: 76 months in stage I, 38 in stage II, 27 in stages III and IV.

![Kaplan-Meier curves by stage with numbers at risk and the summary card](github_images/survival_curves.png)

**4. Cox model.** Open the **Cox model** tab. Age, sex, stage and smoking status are ticked; click **Run Analysis**.
Each dot is a hazard ratio with its 95% interval: stage III patients died at about 3.3 times the rate of stage I
patients (95% CI 2.2 to 4.9) at the same age, sex and smoking status. The badge says **Caution** because one
smoking category has only 4 patients, so its estimate is unstable, and one term may break the model's
proportional-hazards assumption; the notes say which. Untick `smoking_status` and run again to see the difference.

![Cox model forest plot of hazard ratios and the Caution card](github_images/cox_model.png)

**5. Table 1.** Open **Table 1** and click **Build Table**. The table describes the patients overall and for each
stage. It is wide: scroll it sideways, or choose **Export → Table (Excel)** to open it in Excel.

![Table 1 with the variables on the left and the output on the right](github_images/table1.png)

Every tab has an **Export** menu for its figures and tables, and a status next to **Run Analysis** tells you
whether the result still matches the settings.

## Check candidate markers

Use the additional marker analysis when you need to evaluate candidate variables beyond clinical factors.

1. Load your survival data and confirm the time column, event column and event value.
2. Open **More analyses** and choose the marker analysis. In older interfaces, open the **Markers** tab.
3. Select marker columns, or attach a marker matrix and check the patient ID matches.
4. Choose the clinical adjustment variables and review missing values and repeated patients.
5. Run the analysis and read its diagnostics, uncertainty and inference status before interpreting a marker.
6. Save the result table and, when estimable, the locked prediction model for external evaluation.

Withheld standard values are missing values, not zero. Exploratory calculations and predictions retain their
interpretation limits. A selected marker or an internal performance estimate requires independent assessment.

## Prepare your own data

SurvStudio reads one table: CSV, TSV, TXT, Excel (`.xlsx`, `.xls`) or Parquet. Make it like this:

- **One row per patient.** If a patient has several samples, keep one.
- **A follow-up time column.** Numbers only, in one unit for everybody (for example months), from the start point
  (diagnosis, surgery, randomisation) to the event or the last contact. If you have dates, compute it in Excel:
  `=(C2-B2)/30.44` gives months between the dates in B2 and C2.
- **An event column.** `1` if the event (for example death or relapse) happened, `0` if the patient was event-free
  at last contact. `yes`/`no` or `dead`/`alive` also work: choose the event value in the outcome bar.
- **Covariates.** Any other columns: numbers (age) or words (sex, stage, treatment). Leave unknown values empty.
- **One header row** with short column names, and no merged cells, notes or totals below the table.

| patient_id | os_months | os_event | age | sex | stage | treatment | marker_x |
|---|---|---|---|---|---|---|---|
| PT-001 | 12.4 | 1 | 67 | Female | III | Standard | 0.82 |
| PT-002 | 18.0 | 0 | 59 | Male | II | Combination | -0.15 |
| PT-003 | 7.2 | 1 | 72 | Female | IV | Standard | 1.31 |

In Excel, save with **File → Save As → CSV UTF-8**, or upload the `.xlsx` file itself.

**Many markers (omics).** Keep them in a separate file and attach it in the Markers tab: one row per marker
(first column: the marker names) and one column per patient (first row: the patient IDs, written exactly as in
your table's ID column), as GEO and UCSC Xena provide expression data. One row per patient with an ID column works
too. `.gz` files can be attached as downloaded.

**Common mistakes.**

- Text in the time column (`12 months`) or dates instead of a duration.
- An event column with three codes (`0` alive, `1` cancer death, `2` other death): recode it to one event first.
- The same patient in two rows, or the same tumour twice in a marker file (the Markers tab warns about this).
- A few words in a number column (`unknown` in `age`): leave those cells empty instead.
- A status that describes the patient at the start (`kras_status`, `egfr_status`) chosen as the event.

## How to read the results

- **Kaplan-Meier curve.** The share of patients still event-free over time. A step down is an event; a tick mark is
  a patient censored at that time. The **log-rank p-value** tests whether the curves differ; a small p-value does
  not by itself mean the difference matters clinically.
- **Hazard ratio (HR).** How much faster events happen in one group than in the reference group (or per unit of a
  number such as age), with the other covariates held equal. HR 2 means twice the rate, HR 0.5 half. If the 95%
  confidence interval (CI) includes 1, the data are compatible with no effect.
- **C-index.** How well a model ranks patients: for two patients, the chance that the one with the higher
  predicted risk has the event first. 0.5 is a coin toss and 1 is perfect. An apparent C-index, measured on the
  patients the model was built on, can be optimistic; report internal estimates and externally validated values with their limitations.
- **Marker tiers.**
  - *Robust*: passes the family-wise test (p ≤ 0.05 after allowing for all markers tested), is selected in at least
    half of 200 random subsamples of the patients, and has the same direction in at least 90% of them.
  - *Suggestive*: some evidence (family-wise p ≤ 0.05 or false discovery rate ≤ 10%), but not stable across
    subsamples. A hypothesis for another cohort.
  - *Marginal only*: linked to survival on its own, but not beyond the clinical covariates; it mostly tracks them.
  - *Not supported*: no evidence after the error control. Small real effects can still hide here.
- **Added value over clinical covariates.** Whether a marker tells you something about survival that age, sex and
  stage do not already tell you. This is the test that matters for a prognostic claim. SurvStudio measures it as
  the gain in C-index over the clinical covariates in patients left out of subsamples after repeated selection and
  refitting, with an approximate 95% interval. The display reports a small internal gain when the interval lies below
  0.02, suggests internal added discrimination when it lies above 0, and otherwise reports uncertainty. Validate the
  locked model independently before a final prognostic claim.
- **Cautions and "Needs review".** Cautions list what weakens the result (optimism, little added value, missing
  data, repeated patients). The verdict reads **Needs review** when no marker is robust or when two samples look
  like the same patient; then treat the markers as hypotheses, or keep one sample per patient and run again.

## Troubleshooting

- **`uv` or `survstudio` is not recognized / command not found.** Open a new terminal window. If it is still not
  found, run `uv tool update-shell` (for `survstudio`) and open a new window again.
- **Port 8000 is busy** ("address already in use"). SurvStudio may already be running in another window; use that
  one, or start a second copy on another port with `survstudio serve --port 8001` and open
  <http://127.0.0.1:8001>.
- **The page does not load.** SurvStudio runs only while its terminal window is open; start it again with
  `survstudio`.
- **The file is refused.** The message says why. Check the columns: one header row, a numeric time column and an
  event column with two values. `survstudio inspect yourfile.csv` prints what SurvStudio sees in a file.
- **A message says a package is missing** (for example PyTorch). Install again with `[all]` in the command.
- **Deep-learning models are slow.** They run on the processor. Start with **Train one model**, the holdout
  evaluation and 100 epochs before **Compare All Models**, which trains every model in turn.
- **Where do my data go?** Nowhere. SurvStudio reads your file on your computer and keeps it in memory while it
  runs; the page talks only to the SurvStudio program in your terminal (127.0.0.1 means this computer). Stopping
  SurvStudio discards the data.

## Learn more

- [Reference manual](docs/reference.md): all install options, input limits, every analysis in detail, exports,
  the command line and the tests.
- [Numerical agreement](docs/validation/numerical_agreement.md) with R `survival`, lifelines and scikit-survival.
- How to cite: see [CITATION.cff](CITATION.cff), or **Cite this repository** on the GitHub page.
- Licence: [MIT](LICENSE).
- Changes: [RELEASE_NOTES.md](RELEASE_NOTES.md).

The marker residual permutation assumes exchangeable residuals after linear adjustment for the clinical covariates. Nonlinear or heteroscedastic relations and censoring assumptions require sensitivity checks; a family-wise adjusted p-value does not establish universal error control. The internal gain interval is the approximate corrected resampled t interval.

# Comparison with the pipelines commonly used to publish prognostic signatures

This experiment compares SurvStudio with specified pipelines used to develop prognostic gene signatures.
`run_competitors.sh` (one folder up) develops three such pipelines on TCGA-LUAD, applies them to the seven GEO cohorts
of case study II, and sets what each would publish against what the independent cohorts show (experiment 1). It then
counts how often each makes its usual claim under script 06's plasmode null (experiment 2). SurvStudio's own numbers
come from `results/` (case studies I and II, scripts 06 and 15), so run `run_all.sh` first.

## The pipelines

| | Pipeline | What it does, with its documented defaults |
| --- | --- | --- |
| P1 | Mime (`Mime1::ML.Dev.Prog.Sig`, mode `"all"`) | Univariate-Cox candidate filter at p < 0.05 on TCGA; 10 algorithms (RSF, Enet at alpha 0.1 to 0.9, StepCox in three directions, CoxBoost, plsRcox, superpc, GBM, survival-SVM, Ridge, Lasso) and their combinations, 117 models in the pinned version; nodesize 5, seed 5201314 (the documentation's). Every model is trained on TCGA and scored in every cohort. |
| P2 | Univariate Cox -> LASSO -> Cox | Univariate Cox p < 0.05 on TCGA (Wald test, Efron ties); LASSO-Cox with 10-fold CV (`glmnet::cv.glmnet`, `lambda.min`, seed 5201314); a multivariable Cox model of the genes LASSO keeps, whose linear predictor is the risk score. |
| P3 | Stylised uncorrected best-cutoff screen | For every gene on TCGA, each value between its lower and upper quartiles as the cut-off (high expression above it), the log-rank test, and the smallest p; claims at p < 0.05 and at minimum p < 0.05 / genes. The latter corrects the gene count only, not cut-off searching. |

P3 is an explicitly uncorrected comparison procedure, not a reproduction of the current KM Plotter service or
its defaults. [KM Plotter's official description](https://kmplot.com/analysis/) documents Benjamini-Hochberg FDR
for cut-off selection, and its [update history](https://kmplot.com/analysis/index.php?p=updates) records FDR since
2018 and optional percentile/all-value searches since 2025. We have not reproduced those service procedures or
their tie rules. P3's results cannot establish that the current service ignores multiplicity or has the reported
error rate. Accessed 2026-10-02.

## Same patients, outcomes, covariates and genes

- Development: TCGA-LUAD as case study I reads it, 484 primary tumours (Xena `HiSeqV2.gz` matched to SurvStudio's
  bundled clinical table), 177 deaths, overall survival, age, sex and stage.
- Validation: the seven GEO cohorts as scripts 03 and 15 build them (QC exclusions applied, complete cases): 1,509
  patients, 573 deaths; the patient and death counts are checked against SurvStudio's `external_validation.csv`.
- Genes: the 19,112 genes case study I evaluated, restricted to those that TCGA and every GEO cohort measure by script
  15's rule (at most 20% missing, the rest at the cohort median, and varying): 9,938 genes. The other tools cannot
  score genes a cohort lacks.
- Scaling: every gene z-scored within its cohort, TCGA included (as 101-combination papers do); the R pipelines get
  the outcome in days (Mime's documented unit).

## Design decisions

- **Mime's candidates are capped at 100.** Mime's own filter keeps 2,459 genes. Mime runs StepCox as
  `step(coxph(Surv ~ ., all candidates))`, which needs more deaths than genes: with 500 candidates the full Cox model
  is not estimable (484 patients, 177 deaths) and backward elimination would take days (every step refits hundreds of
  models of hundreds of genes), and 54 of the 117 models start with StepCox. Timed on TCGA: StepCox (backward) takes 7 s
  with 50 candidates, 79 s with 100 and 374 s with 150. The primary analysis therefore gives Mime the 100 genes with the
  smallest univariate p (all passing its filter, which it repeats), so that every one of its 117 models runs; this is
  within the gene-set sizes Mime's documentation recommends (more than 50) and uses (42). A sensitivity run gives it the
  500 smallest p-values and runs every model whose first algorithm is not StepCox (63 models): Mime's own source of the
  pinned commit, checked to be identical to the installed function, with its section "3.StepCox" cut out
  (`run_mime.R plan=feasible`). Mime's single and double modes, separate copies of the code, could not stand in: in
  double mode, RSF + Enet fits alpha 0.1 whatever alpha it is given.
- **Two C-indices for Mime.** Mime reports `summary(coxph(Surv ~ RS))$concordance` in each cohort, so each cohort's own
  Cox fit sets the direction of the risk score (a score that runs backwards in a cohort still gets C > 0.5). This
  "reported C" drives the selection, as it does in papers. The "honest C" is Harrell's C of the risk score with its
  direction set once, on TCGA (a score with C < 0.5 on TCGA is reversed; survival-SVM predicts survival times), with
  SurvStudio's conventions (`harrell_c_many`) and bootstrap (2,000 draws, seed 20260926, matching script 03's
  explicit paper setting). The API default is 200; the release comparator used that smaller default. The audit
  aligns the comparison intervals before interpreting differences; this changes interval precision and pooled
  weights, without retraining the R models.
- **Selection replayed.** Models are trained on TCGA only, so selection can be replayed for any choice of cohorts: for
  each of the 35 splits of the seven cohorts into 3 selection and 4 sealed cohorts, the winner is the model with the
  highest mean reported C over the 3 (the first in Mime's order on a tie; a model without a C there cannot win). The
  reported number is that mean; the sealed truth is the winner's honest C in the 4 sealed cohorts (mean, and random-
  effects pooled with `common.random_effects`, each cohort weighted by its bootstrap interval as script 03 pools).
  The usual published design, all seven cohorts used for selection, is replayed too (no sealed cohort remains).
- **Added value over the clinical covariates** (P1 winners, P2), exactly as SurvStudio's external gain: a Cox model of
  age, sex, stage and the risk score fitted on TCGA (Efron ties) and locked as a SurvStudio recipe beside the locked
  clinical-only model (the same clinical-only model as SurvStudio's own), validated with `validate_locked_recipe` in each
  cohort (the model's C, the clinical-only C and their paired difference, 2,000 bootstrap draws) and pooled by random
  effects as script 03 pools. The risk score enters as computed from the z-scored genes (`as_measured`, the analogue of
  SurvStudio's within-cohort rescaling); rescaling it within each cohort to its TCGA mean and SD is reported as well.
- **Median splits** as papers draw them: high risk above the cohort's median risk score, the log-rank test as R's
  `survdiff` (checked to 1e-13 against it), the hazard ratio from the observed/expected ratios (Mime's `rs_sur`).
- **Replication of claimed genes**: script 15's rule (SurvStudio's clinically adjusted Cox score test in each GEO
  cohort, random-effects pooled log hazard ratio per SD, same direction as claimed and 95% CI excluding zero). The
  claimed direction is the best cut-off's (P3) or the gene's multivariable coefficient (P2); recomputed here for
  SurvStudio's tiers on the same genes, it reproduces script 15's replication flags.
- **SurvStudio on the same genes** (a sensitivity run): SurvStudio's locked model of case study I includes markers that
  some GEO cohorts lack (it ran on 58% of its marker weight in GSE68465). `18_competitors_real.py` therefore also runs
  SurvStudio's own analysis (evaluate_markers with its defaults, validate_locked_recipe with within-cohort rescaling,
  as scripts 01 and 03) on the comparison's 9,938 genes.
- **Seeds**: 5201314 for every R pipeline; the null replicates draw from `default_rng([20260929, r])`.

## Experiment 2: the null

Script 06's plasmode null (its generator is imported): the TCGA-LUAD expression and clinical data are kept and overall
survival is simulated from age, sex and stage alone. Each replicate has a development set of the 484 TCGA patients
with new outcomes, seven pseudo-cohorts of the GEO cohorts' sizes (115, 81, 204, 171, 118, 429, 391) drawn with
replacement from the TCGA patients with new outcomes, and 3,000 patients drawn the same way standing for new patients
("truth"). No gene carries information beyond the clinical covariates, but genes that go with stage are prognostic on
their own, as in real data, so a marginal C above 0.5 can be real. The conditional data-generating hazard has no
gene contribution; this does not force the predictive gain of each finite-sample fitted model to be exactly zero.
The independent 3,000-patient sample estimates that fitted model's performance with its own Monte Carlo error.
Every replicate uses all 9,938 genes (no gene sampling was needed). P2 and P3 run on 200 replicates, P1 on 50
(`MIME_REPLICATES`). The audit uses the fixed first 50 designs and retains failed fits without replacement. It reports
completed-fit rates together with bounds over all planned designs, assigning every missing outcome zero or one.
The historical release run replaced failed replicate 35 with replicate 50; its success-conditioned results must
not be interpreted as unconditional error rates. P1–P3 count different claims from SurvStudio's conditional-null
marker hypothesis, so their claim rates are not estimates of the same FWER.

Claims counted per replicate: P1, the winner's reported selection-cohort C (and whether it is at least 0.55 while its
sealed honest C is below 0.55), and a median-split log-rank p < 0.05 in at least one selection cohort; P2, at least one
gene selected with a training median-split p < 0.05, and an external median-split p < 0.05 in at least one of the seven
cohorts; P3, at least one gene with best-cutoff p < 0.05 and the number of such genes. SurvStudio: script 06's null
family-wise error from the new script 06 run (the historical release reported 0.055 over 400 replicates) and, from
a run of script 06 that has it (`results/`, else the
`simulation_summary.json` that `SUBSAMPLES_SUMMARY` names), the verdicts on the gain in its null with subsamples.
Rates over the 35 splits of the P1 replicates get their Monte Carlo SE from the replicates' own rates.

## Running it

```bash
bash paper/competitors/setup_r_env.sh          # the R environment (conda, about 30 minutes; see below)
bash paper/run_all.sh                          # SurvStudio's results, which the comparison reads
RSCRIPT=paper/competitors/renv/bin/Rscript WORKERS=8 bash paper/run_competitors.sh   # WORKERS by free memory: ~4 GB each
bash paper/run_competitors.sh real table       # selected steps: checks data real sensitivity null table
```

`setup_r_env.sh` creates a conda environment with R 4.4 (`COMPETITORS_R_ENV`, default `paper/competitors/renv`) from
conda-forge and bioconda, installs plsRcox, compareC, forestploter and snowfall from the Posit Package Manager's CRAN
snapshot of 2026-09-28, randomForestSRC 3.3.3 from the CRAN archive, and CoxBoost and Mime from git at the pinned
commits (`COMPETITORS_SRC`, default `paper/competitors/src`), then records the versions in `versions.txt` and the
conda packages in `renv-conda-explicit.txt`. It needs `mamba` (or `MAMBA=.../conda`) and network access.

Runtime and memory on a Ryzen 9 7950X (16 cores, 32 threads, 123 GB): the data step 30 s; P2 on the real data 2.5
minutes; Mime on the real data 2 hours with 6 cores (`MIME_CORES`), and the 500-candidate sensitivity run 2.2 hours
with 2; the real-data analysis 1 minute; per null replicate, P2 1 to 2 minutes and P3 20 to 40 seconds on one core,
Mime 40 minutes on one core of an idle server and up to 2 hours on a busy one. Mime keeps every fitted model: a null
replicate grows to about 4 GB,
and with `MIME_CORES=6` GBM's cross-validation starts 6 R workers of up to 3 GB each. Choose `WORKERS` by the free
memory, not only the cores: a run of 10 null Mime replicates beside the real-data run filled this server's memory
(part of it is a RAM disk) and stalled it for an hour; `null_replicate.sh` now caps each run's address space at 12 GB.

## Outputs (results/)

| File | Content |
| --- | --- |
| `competitors_data.json` | Gene count, cohort sizes |
| `competitors_checks.json` | The checks of `18_competitors_checks.py` |
| `competitors_mime_models.csv` | Every Mime model: orientation, genes, training C, reported and honest C per cohort, pooled honest C |
| `competitors_mime_replay.csv` | Every split (and all seven): winner, reported C, training C, sealed means, pooled sealed honest C, gain over the clinical covariates, median-split p-values |
| `competitors_mime_gains.csv` | Each winner's SurvStudio validation in each cohort: the C of clinical + its risk score, the clinical-only C and their paired gain, with intervals |
| `competitors_mime_*_500.csv` | The same for the 500-candidate sensitivity run |
| `competitors_p2_cohorts.csv` | P2 per cohort: C with interval, median-split p and hazard ratio, SurvStudio's external validation of clinical + P2 |
| `competitors_survstudio_same_genes.csv` | SurvStudio's locked model of the comparison's genes in each cohort (as script 03's rows) |
| `competitors_p3_genes.csv` | P3 per gene: best p, cut-off, hazard ratio, cut-offs tried, median-cut p, replication |
| `competitors_real.json` | Experiment 1, with SurvStudio's numbers and their checks |
| `competitors_null_replicates.csv`, `competitors_null_splits.csv`, `competitors_null.json` | Experiment 2 per replicate, per replicate and split, and summarised |
| `competitors_table.csv`, `competitors_null_table.csv`, `competitors_table.json` | The summary tables |

Intermediate files (the R inputs, every R run's outputs and logs, the null replicates) are in `results/competitors/`.

## Deviations from the tools' defaults

- Mime's candidates are capped (100 in the primary analysis, 500 in the sensitivity run); see above.
- randomForestSRC 3.3.3, not the current version: Mime's RSF-based models call `var.select`, which randomForestSRC
  removed in 3.4.0 (2025-05-25).
- snowfall is initialised in sequential mode before Mime runs (CoxBoost's `optimCoxBoostPenalty(parallel = TRUE)`,
  as Mime calls it, runs through snowfall); the folds and results are the same.
- `cores_for_parallel` (GBM's cores) and randomForestSRC's `rf.cores` are 6 for the real data and 1 per null replicate.

## Versions

`versions.txt` (R 4.4.3; Mime commit 9a9f6ac of 2025-09-23, CoxBoost commit 50d5fff, glmnet 5.0, survival 3.8.12,
randomForestSRC 3.3.3, gbm 2.3.1, plsRcox 1.8.2, superpc 1.12, survivalsvm 0.0.6, ...) and `renv-conda-explicit.txt`
(every conda package; conda's randomForestSRC is replaced by 3.3.3 in the same library).

# SurvStudio software paper: reproduction

Every number and figure in the SurvStudio software paper comes from the scripts in this folder, run on the SurvStudio
code of this repository. `prepare_data.sh` rebuilds the data from public sources and checks them against the files the
paper used; `run_all.sh` runs the analyses and draws the figures.

## Requirements

- Linux on x86-64. The numbers were computed there; the simulation (script 06) and scripts 14 and 16 use the fork start
  method, and `requirements-lock.txt` pins the CPU build of PyTorch for Linux.
- A git clone of this repository, not a zip download: every result records `git describe` of the checkout.
- Python 3.12 with the exact package versions of `requirements-lock.txt`, installed with [uv](https://docs.astral.sh/uv/)
  from the repository root (the scripts use `.venv` there by default):

  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv -r paper/requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
  ```

  SurvStudio itself is not installed: the scripts import it from `src/`. Case studies I to III were also run with
  Python 3.11.15 on ARM64 Linux with the same numpy, scipy and pandas versions; they agreed with the x86-64 run to about
  1e-6 (relative), except the experimental deep-learning models of script 04 (up to 0.004 in C). No other Python version
  was tested.
- For the data preparation, R 4.5.2 with Bioconductor 3.22: MetaGxBreast 1.30.0 (with ExperimentHub 3.0.0 and
  AnnotationHub 4.0.0), GEOquery 2.78.0 and data.table 1.18.4 (the exports also used Biobase 2.70.0 and
  SummarizedExperiment 1.40.0). `Rscript paper/data_prep/install_packages.R` installs the missing ones; data.table comes
  from CRAN, where `remotes::install_version("data.table", "1.18.4")` gets that version.
- About 4.5 GB of disk for the data: 1.7 GB of breast cohorts, 0.9 GB of LUAD cohorts, 0.55 GB of GEO downloads and
  the 1.3 GB ExperimentHub cache (in `~/.cache/R`), and network access for the downloads.
- About 4 to 5 hours on 24 cores for `run_all.sh`: the simulation (script 06, 2,300 replicates of about 20 to 30 CPU
  seconds each) takes about 45 minutes with `SIM_WORKERS` set to the number of cores (the default is 8, about 2
  hours); the seed variability (script 16, 60 development evaluations; the cores are shared out among `SEED_WORKERS`
  processes, default 4, and a METABRIC one needs a few GB of memory) about 2 to 3 hours; the other steps about an
  hour together. The data preparation adds the downloads.

## Running it

```bash
git clone https://github.com/kangk1204/SurvStudio.git && cd SurvStudio
# the Python environment as above, then
Rscript paper/data_prep/install_packages.R
bash paper/prepare_data.sh                         # the data, into paper/data, checked against the manifests
bash paper/run_all.sh                              # all steps, then all figures
bash paper/run_all.sh 03 04                        # selected steps, then only the figures drawn from their results
bash paper/run_step.sh 06_simulation.py 24         # one script with arguments (here 24 workers)
bash paper/run_step.sh 16_seed_variability.py 6 --seeds 2 --cases I   # a quick look: 2 seeds of case study I
bash paper/competitors/setup_r_env.sh             # the comparison's R environment (conda R 4.4, Mime at a pinned commit)
bash paper/run_competitors.sh                     # the comparison with other pipelines, after run_all.sh (see competitors/README.md)
```

The scripts read these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SURVSTUDIO_SRC` | this repository | the SurvStudio checkout whose `src/` the analyses import and whose commit they record |
| `ANALYSIS_PYTHON` | `$SURVSTUDIO_SRC/.venv/bin/python`, else `python3` | the environment built from `requirements-lock.txt` |
| `FIGURE_PYTHON` | `ANALYSIS_PYTHON` | an environment with matplotlib for `figures.py`; `run_all.sh` stops before any analysis when it cannot import it |
| `SURVSTUDIO_DATA` | `paper/data` | the data folder |
| `SIM_WORKERS` | 8 | processes for the simulation (06) in `run_all.sh` |
| `SEED_WORKERS` | 4 | processes for the seed variability (16) in `run_all.sh`; the cores are shared out among them |
| `RSCRIPT` | `Rscript` | R for `prepare_data.sh`; for `run_competitors.sh`, the comparison's environment (`paper/competitors/renv/bin/Rscript`) |
| `WORKERS`, `MIME_CORES`, `NULL_REPLICATES`, `MIME_REPLICATES` | see `competitors/README.md` | processes and replicates of `run_competitors.sh` |

`run_all.sh` leaves out the duplicate audits (scripts 11 and 12); run them with `run_step.sh` when checking the committed
`breast_duplicate_pairs.csv`.

| Script | What it computes |
| --- | --- |
| `prepare_data.sh`, `data_prep/` | The data chain (see Data): downloads, harmonization, QC and the breast export, then the checks against the manifests |
| `scripts/00_self_check.py` | Checks of the shared helpers, in seconds, each of which must run: on synthetic data a cohort without complete cases, duplicate chains and shared patient IDs, the follow-up and tumour-size conventions, METABRIC's cBioPortal endpoints (with the snapshot copy used when a download differs), the external-cohort screens of case studies IV and V, probe collapsing, the METABRIC sites of script 14, the LUAD data manifest, random-effects pooling (the Hartung-Knapp-Sidik-Jonkman and prediction intervals against worked examples computed with R 4.5.2 and metafor 5.0.1), the pooling of the validation rows, script 15's test and pooling, the simulation summary's gain verdicts, its resume filter and summary check with stamps that look like numbers, and the result stamps; SurvStudio's marker-weight rule on a model it locks, and that model's linear predictor recomputed for the extra per-cohort quantities as SurvStudio scores it; Uno's C against the estimator written out; the breast data against `breast_data_manifest.csv`, the LUAD data against `luad_data_manifest.csv` and the snapshot files against their SHA-256 |
| `scripts/01_markers_tcga.py` | Genome-wide marker evaluation in TCGA-LUAD (Xena `HiSeqV2.gz`, bundled clinical table; age, sex, stage), the locked model and the gene funnel |
| `scripts/02_permutation_maximum.py` | The permutation maximum with and without near-constant genes (300 permutations of each marker's residuals on the clinical covariates) |
| `scripts/03_external_geo.py` | The locked model in seven GEO cohorts, as measured and rescaled within each cohort, with bootstrap intervals from 2,000 resamples. Per cohort (`external_validation.csv`): the model's C, the clinical-only C and the gain, the calibration slopes of the model and of the clinical-only model, the SD of the linear predictor and of its clinical and gene parts, and the gene component's hazard ratio per SD beyond the clinical part. Pooled (`external_pooled.json`): each of these by random effects, the DerSimonian-Laird estimate with its CI, the Hartung-Knapp-Sidik-Jonkman (HKSJ) interval and the 95% prediction interval. Sensitivity (`external_uno.csv`, `external_uno_pooled.json`): Uno's C truncated at 5 years for the model and the clinical-only model and their paired difference, pooled in the same way |
| `scripts/04_models_tcga.py` | All prediction models through SurvStudio's API with the interface's Compare All settings, and the paired intervals |
| `scripts/05_agreement.py` | Largest difference from R survival per analysis, from SurvStudio's agreement report (`docs/validation/numerical_agreement.json` of this repository), with the report's date, SHA-256 and commit |
| `scripts/06_simulation.py` | Plasmode simulation on the TCGA-LUAD RNA-seq (2,300 replicates): error control, power, and each reported C-index against the C in new patients; the null with 100 subsamples (1,000 replicates) and a diffuse signal (150 true genes, log hazard ratios of 0.05 to 0.10 per SD, 300 replicates) were added after the first six scenarios, whose seeds and results they leave as they were; every replicate with subsamples records SurvStudio's paired left-out gain with its interval where SurvStudio reports one. Resumable: each replicate is stamped with the SurvStudio commit, a hash of the design and a hash of the simulation's code (`06_simulation.py` and `common.py`), and a rerun keeps only matching, error-free replicates (none when SurvStudio has uncommitted changes); a worker that dies stops the run with an error |
| `scripts/06_simulation_summary.py` | Per-scenario family-wise error with Monte Carlo SE, power, false markers, bias and RMSE of each C-index estimate, the bias of SurvStudio's paired left-out gain against the fitted model's gain in independently generated new patients (including the null; coverage of zero is a separate null-hypothesis check), the coverage of the gain's interval, and how often the verdict read from it says "adds little" (upper limit below 0.02), "adds" (lower limit above 0) or "uncertain", next to the old rule (gain below 0.02); stops unless the replicates are one complete run of the design and code in `simulation_settings.json` with every value it uses |
| `scripts/07_breast_markers.py` | Case study IV: genome-wide marker evaluation in METABRIC (overall survival from cBioPortal to 10 years; age, tumour size, nodes, grade, ER and expression from MetaGxBreast) and the locked model |
| `scripts/08_breast_external.py` | The locked METABRIC model in the MetaGxBreast cohorts with overall survival, every covariate, at least 30 deaths and at least half of the marker weight measured (see Breast cohorts); per-cohort quantities and random-effects pooling as in script 03 |
| `scripts/09_breast_er_markers.py` | Case study V, a positive control: METABRIC ER-positive tumours, relapse-free survival from cBioPortal (10 years), added value over age, size, nodes and grade |
| `scripts/10_breast_er_external.py` | The locked ER-positive recurrence model in the MetaGxBreast cohorts with relapse-free (else distant metastasis-free) survival, EMC2 left out (patients selected for relapse), rules fixed before the run (see Breast cohorts); per-cohort quantities and random-effects pooling as in script 03 |
| `scripts/11_duplicate_audit.py` | Breast duplicates within and across cohorts (curator annotations, shared Uppsala/Karolinska sample codes, expression mutual best matches, clinical identity); writes `results/breast_duplicate_pairs_recomputed.csv` and preserves the committed analysis input |
| `scripts/12_luad_duplicate_audit.py` | LUAD duplicates between TCGA-LUAD and the GEO cohorts and among the GEO cohorts, with the QC's known technical duplicates as a positive control; a flagged pair counts as one patient only when the clinical records (age within 1 year, sex, stage) do not contradict it |
| `scripts/13_duplicate_screen_check.py` | SurvStudio's own repeated-patient screen on each case-study cohort, every tumour included, against the duplicates the audits confirmed (every flagged pair counted) |
| `scripts/14_breast_er_sensitivity.py` | Positive control, sensitivity: each METABRIC site with at least 30 relapses held out in turn (whole evaluation on the other sites, locked model on the held-out one, bootstrap intervals from 2,000 resamples; internal gain = SurvStudio's paired left-out gain), and the external cohorts pooled by endpoint; pooled gains with their DerSimonian-Laird, HKSJ and prediction intervals |
| `scripts/15_tier_replication.py` | Evidence tiers against replication: for every gene of case studies I, IV and V, the clinically adjusted association in the validation cohorts, pooled by random effects (a gene replicates when the pooled estimate goes in its development direction and its DerSimonian-Laird 95% interval excludes zero), and the share replicated per SurvStudio tier (rules fixed before the run) |
| `scripts/15b_tier_replication_checks.py` | Checks of script 15's tiers, after it (`tier_replication_checks.json`, `.csv`): the replication rate of the robust genes, and of the robust and suggestive ones, against that of the same number of genes with the smallest development p-values; a logistic regression of replication on development \|z\|, the number of cohorts measuring the gene, its mean development expression and the tiers, with a likelihood-ratio test of the tiers; and, per tier, the share of genes whose pooled estimate excludes zero in the direction opposite to development |
| `scripts/16_seed_variability.py` | Monte Carlo variability of the development runs of case studies I, IV and V, rerun as scripts 01, 07 and 09 with 20 seeds (`seed_variability.csv`, `.json`): per seed the apparent, corrected and left-out C, the clinical-only left-out C, the paired gain (with its interval where SurvStudio reports one), the family-wise significant, robust and suggestive counts and the locked genes; their mean, SD and range, how often each gene was locked, and whether the default seed reproduces the development run. Arguments: workers, `--seeds N`, `--cases` |
| `scripts/17_cohort_table.py` | The cohort table (`cohort_table.csv`): the development cohorts and every cohort screened for case studies II, IV and V, with platform, source, endpoint, role, patients and events screened and used, median follow-up (reverse Kaplan-Meier), and why a cohort was left out; the counts are checked against the analyses' results. The breast platforms are the ones the MetaGxBreast datasets' probe identifiers and source series identify |
| `breast_duplicate_pairs.csv` | The committed breast duplicate pairs; compare script 11's independently regenerated identities and decisions before reviewing any input update. With the curators' annotations they define the patients (see Breast cohorts) |
| `breast_data_manifest.csv`, `luad_data_manifest.csv`, `scripts/breast_manifest.py`, `scripts/luad_manifest.py` | Every breast and LUAD data file the analyses read, with its size and SHA-256 (see Data), and the scripts that write the manifests |
| `run_competitors.sh`, `competitors/`, `scripts/competitors.py`, `scripts/18_competitors_*.py` | Comparison with the pipelines commonly used to publish prognostic signatures (Mime, univariate Cox → LASSO → Cox, KM Plotter's best cut-off): TCGA-LUAD → the seven GEO cohorts, and the claims each makes under script 06's null; needs `competitors/setup_r_env.sh` and `run_all.sh`'s results; see `competitors/README.md` |
| `scripts/figures.py` | Figures 1 to 5 and Supplementary Figures S1 to S6 (PNG and PDF, see Figures) from `results/`, each only from results of one run (see Results and stamps); `figures.py estimates simulation` draws a subset |
| `interface/markers_tab_case_study_i.png` | Figure 1b: the Markers tab after case study I, captured from the web interface (TCGA-LUAD with the Xena `HiSeqV2.gz` file, age, sex and stage, the defaults) |
| `scripts/simulation_smoke.py` | A few replicates of the simulation with timings, and the summary over them: `simulation_smoke.py 2 null_filter_subsamples diffuse_filter` runs two of each named scenario (by default one null and one alternative replicate) |

## Data

No raw data are redistributed here except the three small files of `data_snapshot/` (see `data_snapshot/SNAPSHOT.md`):
NCBI's `Homo_sapiens.gene_info.gz` of 2026-09-26, cBioPortal's METABRIC patient table and the LUAD QC decisions.
`prepare_data.sh` builds everything else in `SURVSTUDIO_DATA`, in this order:

1. `data_prep/export_tcga_luad.py`: TCGA-LUAD from UCSC Xena, the RNA-seq matrix
   ([`HiSeqV2.gz`](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FHiSeqV2.gz)), the
   clinical matrix ([`LUAD_clinicalMatrix`](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FLUAD_clinicalMatrix))
   and the TCGA-CDR survival table ([`LUAD_survival.txt`](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/survival%2FLUAD_survival.txt)).
2. `data_prep/export_geo_luad.R`: the seven GEO series with GEOquery, GSE13213 (GPL6480), GSE30219 (GPL570), GSE31210
   (GPL570), GSE41271 (GPL6884), GSE50081 (GPL570), GSE68465 (GPL96) and GSE72094 (GPL15048), with their platform
   annotations.
3. `data_snapshot/sample_decisions.csv`, the QC decisions the paper applied, copied to `cohorts/luad/qc/`.
4. `data_prep/harmonize_luad.py`: the harmonized clinical table of the eight LUAD cohorts, `cohorts/luad/harmonized_clinical.csv`
   (outcomes as the sources record them, the QC decisions applied; it stops when the decisions file is missing).
5. `data_prep/harmonize_expression.py`: one expression column per gene, `cohorts/luad/<cohort>/expression_genes.csv.gz`,
   with the gene symbols of the NCBI gene_info snapshot (NCBI replaces the file daily; the script checks its SHA-256).
6. `data_prep/qc_cohorts.py`: the LUAD QC rerun (it downloads the GDC PanCanAtlas
   [sample-quality annotations](https://api.gdc.cancer.gov/data/1a7d7be8-675d-4e60-a105-19d4121bdebf)); its decisions are
   compared with the ones the paper applied, which stay in place either way.
7. `data_prep/export_curated_cohorts.R`: the 39 MetaGxBreast datasets from Bioconductor ExperimentHub (records
   EH1076 to EH1114) and the curators' duplicate list, `cohorts/breast/` and `cohorts/breast_duplicates.csv`.

Then it checks, with SHA-256:

- the downloads against `data_prep/downloads.sha256` (the files the paper used; a difference is reported, not fatal,
  since a source may re-package unchanged data);
- the cBioPortal METABRIC table, `cohorts/breast/METABRIC/cbioportal_patients.csv`: `common.py` downloads it on first use
  from the [cBioPortal API](https://www.cbioportal.org/api/studies/brca_metabric/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY)
  and uses the snapshot copy when the download differs from it (2,509 patients, SHA-256 `f09389e3...`) or fails;
- the analysis inputs against `luad_data_manifest.csv` (9 files; a `.gz` file by its decompressed content, since its
  gzip header records when it was written; `harmonized_clinical.csv` has SHA-256 `b90ec979...`) and
  `breast_data_manifest.csv` (79 files, byte for byte).

`00_self_check.py`, the first step of `run_all.sh`, repeats the manifest and snapshot checks and stops on any missing
or changed file (and, for the breast data, on an extra one).

Data licences and the citations to give:

| Source | Licence | Cite |
| --- | --- | --- |
| TCGA-LUAD via UCSC Xena, TCGA-CDR endpoints, GDC PanCanAtlas annotations (QC only) | TCGA open-access data | TCGA Research Network, Nature 2014 (lung adenocarcinoma); Goldman et al., Nat Biotechnol 2020 (Xena); Liu et al., Cell 2018 (TCGA-CDR); <https://gdc.cancer.gov/about-data/publications/pancanatlas> |
| GEO series GSE13213, GSE30219, GSE31210, GSE41271, GSE50081, GSE68465, GSE72094 | public (NCBI GEO) | Tomida et al., J Clin Oncol 2009; Rousseaux et al., Sci Transl Med 2013; Okayama et al., Cancer Res 2012; Sato et al., Mol Cancer Res 2013; Der et al., J Thorac Oncol 2014; Shedden et al., Nat Med 2008; Schabath et al., Oncogene 2016; Barrett et al., Nucleic Acids Res 2013 (GEO); Davis and Meltzer, Bioinformatics 2007 (GEOquery) |
| NCBI Gene (`Homo_sapiens.gene_info.gz`) | public domain | Brown et al., Nucleic Acids Res 2015 |
| MetaGxBreast via Bioconductor ExperimentHub | Apache License (>= 2) | Gendoo et al., Sci Rep 2019, and the original study of each dataset as MetaGxBreast lists it |
| cBioPortal METABRIC patient table | ODbL v1.0 (notice in `data_snapshot/SNAPSHOT.md`) | Curtis et al., Nature 2012; Pereira et al., Nat Commun 2016; Cerami et al., Cancer Discov 2012; Gao et al., Sci Signal 2013; de Bruijn et al., Cancer Res 2023 |

The TCGA-LUAD clinical and outcome table of case studies I and III, the simulation and scripts 02, 12 and 13 is the one
SurvStudio ships, `src/survival_toolkit/data/tcga_luad_upload_ready.csv` (489 patients); its derivation from TCGA is
not part of this folder.

## Results and stamps

`results/` is regenerated and not committed; `figures/` holds the figures drawn from the results of one fresh-environment
run of `run_all.sh` and `run_competitors.sh` at SurvStudio commit fc0e1af, and `run_all.sh` redraws them in place (the
PDFs record the time they were drawn, so they always differ from the committed ones in their bytes).

Every file a script writes to `results/` gets a stamp in `results/stamps/` (`common.stamp_result`): the file's
SHA-256, the SurvStudio version and commit, a hash of the analysis code (`common.py`, the numbered scripts,
`breast_duplicate_pairs.csv` and the two data manifests), the script, and the results files it read with their
SHA-256. The commit is `git describe --always` of `SURVSTUDIO_SRC` with a `-dirty` suffix when a tracked file other than
`paper/figures/` has uncommitted changes; both run scripts warn about it, and the simulation reuses no replicate of such
a run. `figures.py` draws a figure only when every file it reads is stamped, unchanged since, computed from the results
files that are there now, and from one SurvStudio commit and one version of the analysis code, and holds every value
the figure draws (results of an earlier version of the scripts may lack one). After a partial rerun (after changing a
script, say), it names the files that fail; rerun the steps that write them, or everything.

For selected steps, `run_all.sh` draws available figures and reports a composite figure as pending when another
case study's input is absent. Existing mixed or stale results still cause an error. A full run requires every
input; `figures.py --available names...` applies the same partial-input rule when drawing figures directly.

## Figures

`figures.py` draws each figure at its print size, at most 170 mm wide and 200 mm high (reserving 25 mm of BMC's
225-mm combined figure/legend height for a legend), with a project readability floor of 7 pt. It refuses to save
one that is not. The native summary screenshot's fonts are checked using its capture record; raster text in other
images still needs visual inspection. A pooled estimate is drawn with its Hartung-Knapp-Sidik-Jonkman 95% interval and,
from three cohorts on, its 95% prediction interval.

| File (`.png`, `.pdf`) | Figure | Drawn from |
| --- | --- | --- |
| `fig1` | Figure 1: a, what SurvStudio checks by default (`fig1_workflow`); b, the native Markers summary after case study I (`interface/marker_summary_case_study_i.png` with capture provenance); the wider full-tab capture is retained separately | screenshot and capture record |
| `fig1_workflow` | Figure 1a alone | nothing |
| `fig2_markers` | Figure 2: genome-wide markers in TCGA-LUAD | 01, 02 |
| `fig3_estimates` | Figure 3: internal and external estimates in three settings (lung adenocarcinoma, breast cancer survival, the ER-positive positive control): a, the C-index from apparent to subsample gap-adjusted, left-out and external, for the model and the clinical-only model; b, the gain over the clinical covariates in left-out patients, held-out METABRIC sites (positive control) and the external cohorts | 01, 03, 07 to 10, 14 |
| `fig4_models` | Figure 4: prediction models on the same test patients | 04 |
| `fig5_comparison` | Figure 5: the pipelines commonly used to publish signatures against SurvStudio, under the null and from TCGA-LUAD to the seven GEO cohorts | `run_competitors.sh` (18) |
| `figS1_simulation` | Supplementary Figure S1: error control, power and the accuracy of the C-index estimates in the simulation | 06 |
| `figS2_luad_external`, `figS3_breast_external`, `figS4_breast_er_external` | Supplementary Figures S2 to S4: each external cohort's gain and each locked gene's replication in case studies II, IV and V | 01 and 03, 07 and 08, 09 and 10 |
| `figS5_positive_control_sensitivity` | Supplementary Figure S5: the positive control's gain in held-out METABRIC sites and by endpoint | 10, 14 |
| `figS6_tier_replication` | Supplementary Figure S6: evidence tiers against external replication | 15 |

## Breast cohorts (case studies IV and V)

`common.load_breast_cohort` turns a MetaGxBreast cohort into SurvStudio input and `common.screen_validation_cohorts`
chooses the external cohorts; scripts 08, 10 and 15 share both.

- Follow-up is censored at 10 years. METABRIC's endpoints both come from cBioPortal's `brca_metabric` patient table
  (see Data): overall survival (`OS_MONTHS`, `OS_STATUS`) for case study IV and relapse-free survival (`RFS_MONTHS`,
  `RFS_STATUS`) for case study V, matched exactly (sample `MB_0001` is patient `MB-0001`); the clinical covariates and
  expression come from MetaGxBreast. A METABRIC tumour without a cBioPortal record, or whose record lacks the endpoint,
  is left out and counted in the summaries of scripts 07 and 09 (case study IV: 7 tumours without a record and 4
  without overall survival). MetaGxBreast's own METABRIC follow-up (`days_to_death`) is an older one: as days / 30 it
  matches `OS_MONTHS` within 0.05 months for 1,106 of 1,978 tumours, and 249 tumours it records as living are deceased
  in cBioPortal. In the other cohorts a day column that holds multiples of 30 only (DFHCC's and MAINZ's distant
  metastasis-free survival, UNC4's overall and relapse-free survival) is converted with 30 days a month, every other
  day column with 365.25 / 12.
- Tumour size is in centimetres, except in STNO2 and VDX (and EORTC10994, which has no follow-up): every size they
  record is a whole number from 1 to 4, the T category. A T category is replaced by the median METABRIC tumour of its
  range: T1 (at most 2 cm) 1.6 cm, T2 (over 2 and up to 5 cm) 2.9 cm, T3 and T4 (over 5 cm) 6.2 cm. The medians are
  computed from METABRIC in each run and recorded in `breast_external_pooled.json` and `breast_er_external_pooled.json`.
  UNC4 records size classes (1, 3 and 6, and one 1.5), neither centimetres nor T categories, so its tumour size is
  treated as not recorded and it has no complete cases (it was left out of both case studies for other reasons before).
  The duplicate audit compares sizes only between cohorts that record the same kind.
- One expression column per gene. Where the export names a gene's further probes GENE.1, GENE.2, ... (R's
  `make.unique`; GSE58644 pads its symbols with spaces, so " RNF207 .1"), the loader keeps the gene's most variable
  probe over the patients analysed (`common.probe_gene`).
- One sample per patient. Two samples are the same patient when a chain of duplicate pairs links them (the
  curators' annotations and `breast_duplicate_pairs.csv`), within or across cohorts, and when a cohort gives them the
  same `unique_patient_ID` (DFHCC2's replicate arrays, MDA4's M323 and M323_bis; not compared across cohorts). Chains
  are not followed through KOO's samples: KOO was profiled on 280 genes, the curators list one KOO sample as the
  duplicate of 38 others, and through them 135 listed samples from at least 11 cohorts would form one patient. A cohort
  keeps the first sample of each patient.
- External cohorts, screened in alphabetical order by rules fixed before the runs: every cohort other than METABRIC
  with the endpoint (IV: overall survival; V: relapse-free survival, else distant metastasis-free survival) except
  cohorts sampled on the outcome (EMC2 holds patients selected for metastatic relapse); complete cases for the
  clinical covariates (ER-positive tumours in V); patients who are a METABRIC development patient or a patient of a
  cohort used before removed; then at least 30 events within 10 years and at least half of the locked model's marker
  weight (|coefficient| × development SD) measured. Only a used cohort's patients are checked against later cohorts.
  Every cohort's counts and, when it was left out, the reason are recorded under `screened` in the pooled JSON.

## Troubleshooting

- A snapshot file fails its SHA-256 check: restore it with `git checkout paper/data_snapshot`. The files must stay
  byte-identical (`.gitattributes` keeps this folder's line endings as committed).
- `prepare_data.sh` reports a download that differs: the source has changed since September 2026. When the manifest
  checks after it pass, the analysis inputs are the paper's all the same; when they fail, the named files differ from
  the ones the paper used.
- The breast data fail `breast_data_manifest.csv`: a fresh MetaGxBreast export with a newer Bioconductor (or another
  data.table or zlib, which can compress the same data into other bytes) may differ from the pinned one; use R 4.5.2
  with Bioconductor 3.22 and the versions above. After a deliberate re-export, `python scripts/breast_manifest.py` (and
  `scripts/luad_manifest.py` for the LUAD data) rewrites the manifest; the results then no longer come from the paper's
  data.
- The QC rerun decides differently (a `prepare_data.sh` warning, usually after GDC changed its annotations): the
  rerun's decisions are kept as `cohorts/luad/qc/sample_decisions.rerun.csv`, and the harmonized table keeps the
  decisions the paper applied.
- A warning that SurvStudio is at `unknown` or `-dirty`: run from a git clone, and commit or stash changes outside
  `paper/figures/` before a run whose results should name a commit.

## Review on 2026-10-02

The original release candidate remains available at f7d8baa. The audit branch incorporates fold-specific LASSO preprocessing and labels the existing gap adjustment explicitly. Recomputed results must use their recorded source commit; the committed release figures are preserved in the reference worktree and are not evidence for a changed analysis. `validation/publication/expanded_protocol.json` fixes additional conditional-null stress tests before their execution. The partial-null plasmode of script 06 classifies genes with partial correlation below 0.1 as unlinked; this is a descriptive threshold, not an exact conditional null, so its alternative-scenario false-discovery counts do not prove strong FWER control.

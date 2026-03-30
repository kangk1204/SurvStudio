# Upload-Ready Real Dataset

This folder contains a real public cohort that can be uploaded directly into SurvStudio for a first end-to-end test.

Files included:

- `tcga_luad_nature2014_upload_ready.csv`
- `tcga_luad_rnaseq_top100_upload.csv`
- `tcga_luad_rnaseq_top500_upload.csv`
- `gbsg2_jco1994_upload_ready.csv`

## Dataset 1: TCGA LUAD

- File: `tcga_luad_nature2014_upload_ready.csv`
- Rows: `609`
- Survival outcome:
  - time column: `os_months`
  - event column: `os_event`
  - event-positive value: `1`

## Source

This file is a simplified upload-ready subset of the bundled TCGA LUAD example cohort already shipped with SurvStudio.

Study citation:

- The Cancer Genome Atlas Research Network. *Comprehensive molecular profiling of lung adenocarcinoma*. Nature. 2014;511(7511):543-550. doi:10.1038/nature13385.

Data provenance:

- Public TCGA LUAD clinical cohort curated from UCSC Xena and reduced here to a compact survival-analysis table for direct upload testing.

## Recommended First Test In SurvStudio

1. Upload `tcga_luad_nature2014_upload_ready.csv`
2. Set:
   - time column: `os_months`
   - event column: `os_event`
   - event-positive value: `1`
3. Start with:
   - Kaplan-Meier
   - group column: `stage_group`
4. Then run Cox PH with:
   - covariates: `age`, `sex`, `stage_group`, `smoking_status`
   - categorical covariates: `sex`, `stage_group`, `smoking_status`
5. Then run ML or DL smoke tests with the same feature set

## Why This File Is Beginner-Friendly

- one row per patient
- already in wide tabular format
- explicit overall-survival time and event columns
- includes both numeric and categorical covariates
- missingness is modest in the recommended feature set

## Notes

- This is a single-event right-censored cohort.
- Some columns from the larger bundled TCGA example were intentionally removed so the upload example stays simple.
- If you want a larger feature set, use `tcga_luad_rnaseq_top100_upload.csv` or `tcga_luad_rnaseq_top500_upload.csv`.

## Dataset 2: TCGA LUAD RNA Top 100

- File: `tcga_luad_rnaseq_top100_upload.csv`
- Rows: `609`
- Columns: `112` total (`12` clinical + `100` gene-expression features)
- Survival outcome:
  - time column: `os_months`
  - event column: `os_event`
  - event-positive value: `1`

## Source

This file starts from the same 609-row TCGA LUAD upload-ready clinical cohort as `tcga_luad_nature2014_upload_ready.csv` and left-joins the first 100 RNA features from the larger RNA example.

Notes:

- the 100 genes are the first 100 features from the ranked RNA top-1000 source table
- four patients do not have matching RNA values in the source RNA table, so those gene cells remain missing
- clinical-only Kaplan-Meier and Cox settings stay aligned with the 609-row bundled TCGA cohort
- the file stays well below the SurvStudio upload cap of 1000 model feature candidates

Recommended first settings:

1. Upload `tcga_luad_rnaseq_top100_upload.csv`
2. Set:
   - time column: `os_months`
   - event column: `os_event`
   - event-positive value: `1`
3. Start with:
   - Kaplan-Meier
   - group column: `stage_group`
4. Then try:
   - age-derived cutpoint grouping
   - ML or DL with a smaller selected feature subset

## Dataset 3: TCGA LUAD RNA Top 500

- File: `tcga_luad_rnaseq_top500_upload.csv`
- Rows: `609`
- Columns: `512` total (`12` clinical + `500` gene-expression features)
- Survival outcome:
  - time column: `os_months`
  - event column: `os_event`
  - event-positive value: `1`

## Source

This file starts from the same 609-row TCGA LUAD upload-ready clinical cohort as `tcga_luad_nature2014_upload_ready.csv` and left-joins the first 500 RNA features from the same ranked RNA top-1000 source table used for the top-100 upload file.

Notes:

- the 500 genes are the first 500 features from the ranked RNA top-1000 source table
- four patients do not have matching RNA values in the source RNA table, so those gene cells remain missing
- clinical-only Kaplan-Meier and Cox settings stay aligned with the 609-row bundled TCGA cohort
- the file remains below the SurvStudio upload cap of 1000 model feature candidates

Recommended first settings:

1. Upload `tcga_luad_rnaseq_top500_upload.csv`
2. Set:
  - time column: `os_months`
  - event column: `os_event`
  - event-positive value: `1`
3. Start with:
  - Kaplan-Meier
  - group column: `stage_group`
4. Then try:
  - age-derived cutpoint grouping
  - ML or DL after narrowing the default feature subset
## Dataset 4: GBSG2

- File: `gbsg2_jco1994_upload_ready.csv`
- Rows: `686`
- Survival outcome:
  - time column: `rfs_days`
  - event column: `rfs_event`
  - event-positive value: `1`

Study citation:

- Schumacher M, Basert G, Bojar H, et al. *Randomized 2 × 2 trial evaluating hormonal treatment and the duration of chemotherapy in node-positive breast cancer patients*. Journal of Clinical Oncology. 1994;12(10):2086-2093.

Why this file is useful:

- no missing values in the upload-ready version
- one row per patient
- compact feature set
- good for a fast Kaplan-Meier, Cox, ML, or DL smoke test

Recommended first settings:

1. Upload `gbsg2_jco1994_upload_ready.csv`
2. Set:
   - time column: `rfs_days`
   - event column: `rfs_event`
   - event-positive value: `1`
3. Start with:
   - Kaplan-Meier
   - group column: `horTh` or `menostat`
4. Then run Cox PH with:
   - covariates: `age`, `horTh`, `menostat`, `pnodes`, `tgrade`, `tsize`
   - categorical covariates: `horTh`, `menostat`, `tgrade`

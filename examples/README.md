# Upload-Ready Real Dataset

This folder contains a real public cohort that can be uploaded directly into SurvStudio for a first end-to-end test.

## Dataset

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
- If you want a larger feature set, use the built-in `Load TCGA LUAD` button in the app.

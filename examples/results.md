# Example run results

[Back to quick start](../README.md) · [Raw outputs](demo_results.json)

These are outputs from the bundled examples, run through the dashboard API on 10 October 2026.
They show what to expect when trying the app; they do not establish a best model or a treatment effect.

## Survival curves

Select the sample, use event value **1**, and choose the group shown below.

| Sample | Group | Patients / events | Median survival |
|---|---|---:|---|
| Synthetic demo | `stage` | 360 / 259 | I: 32.91; II: 21.44; III: 15.45; IV: 11.72 months |
| TCGA-LUAD | `stage_group` | 489 / 178 | Stage I: 76.15; Stage II: 37.68; Stage III: 26.51; Stage IV: 27.14 months |
| GBSG2 | `horTh` | 686 / 299 | no: 1528; yes: 2018 days |

TCGA-LUAD and the synthetic sample use overall survival; GBSG2 uses recurrence-free survival.
An event-free patient at last contact is censored.

## Cox model

Use these variables; select the listed categorical variables as categories.

| Sample | Variables | Categorical variables | Usable patients | Apparent C-index |
|---|---|---|---:|---:|
| Synthetic demo | age, sex, stage, treatment, biomarker_score, immune_index | sex, stage, treatment | 342 | 0.683 |
| TCGA-LUAD | age, sex, stage_group, smoking_status | sex, stage_group, smoking_status | 476 | 0.681 |
| GBSG2 | age, horTh, menostat, pnodes, tgrade, tsize | horTh, menostat, tgrade | 686 | 0.674 |

Cox drops rows missing a selected variable: 18 synthetic rows and 13 TCGA-LUAD rows in these runs.
The apparent C-index is measured on the fitting patients and can be optimistic. Read the model checks
as well as the forest plot; the GBSG2 global proportional-hazards check has p = 0.00654.

## ML/DL: one GBSG2 run

Use the same six GBSG2 variables and categories as above. Choose **Train one model** and
**Holdout**, with seed **42**. The split has **480 training patients** and **206 evaluation patients**.

| Model | Settings | Held-out C-index |
|---|---|---:|
| Random Survival Forest | 100 trees; no depth limit; SHAP off | 0.688 |
| DeepSurv | Maximum 100 epochs; layers 64,64; dropout 0.1; learning rate 0.001; patience 10 | 0.677 |

DeepSurv selected 34 epochs by early stopping, then refit on the training partition.
These are scores from one split. Small differences do not show that one model is better.
Versions and source hashes are included in the raw outputs; values can differ with another environment.

## Reproduce

After a [developer install](../docs/installation.md#developer-install), run from the repository root:

```bash
python examples/run_demo.py --models --output demo_results.json
```

Omit `--models` to run only KM and Cox. To repeat the synthetic example in the browser,
click **Synthetic demo** or upload [synthetic_demo.csv](synthetic_demo.csv).

The script records the input columns, source hashes, versions, raw KM summaries, Cox estimates
and model checks. It raises an error if an API request fails.

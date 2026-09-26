# Training dataset composition — template for a Supplementary section

Journals that publish machine-learning results (for example *Bioinformatics*) ask for a dedicated subsection that describes:
- how the training data were composed,
- how the cross-validation sets were built,
- which independent test set was kept out of all training.

They also ask for performance on that independent test set; an average over cross-validation folds is not enough.

This template maps each required item to the SurvStudio setting or result field that documents it. Replace the bracketed parts, and delete the rows that do not apply.

## 1. Data source and unit of analysis

| Item | What to report | Where SurvStudio records it |
|---|---|---|
| Source | Dataset name, release or version, access date, licence | Your citation; the export notes (`Replay dataset: ...`) |
| Fingerprint | Hash of the exact table analysed | Export notes `Dataset fingerprint: ...` (`dataset_hash`) |
| Unit | One row per patient; how repeated samples per patient were collapsed | Dataset profile warning on repeated identifier values |
| Exclusions | Rows dropped for a missing outcome, negative follow-up time, or ±Inf values | Kaplan-Meier and Cox cohort summaries: `dropped_missing_rows`, `dropped_nonpositive_time_rows`, `rows_with_infinite_values` |

## 2. Outcome

- **Time:** `[time column]` in `[unit]`, with administrative censoring at `[horizon, if any]`.
- **Event:** `[event column]`. The event is coded as `[event-positive value]`, and every other value is censoring.
- **Totals:** `[N]` patients, `[events]` events, median follow-up `[x]` (reverse Kaplan-Meier).

## 3. Predictors and preprocessing

- **Features:** `[list or count]`. Categorical features use reference-level coding, and the reference is the first level in clinical or numeric order.
- **Fitting:** every preprocessing step is fitted on the training partition only and then applied unchanged to the evaluation rows:
  - median imputation of numeric features,
  - standardisation (deep models),
  - category levels.
- **Missing-value indicators:** a categorical feature gets a missing-value indicator only when the training data contain missing values.
- **New levels:** levels first seen at evaluation time are scored as the reference level.

## 4. Independent (locked) test set

| Item | Value | SurvStudio field |
|---|---|---|
| Share held out | `[e.g. 0.2]` | `locked_test_fraction` |
| Stratification | By event status | Built in |
| Size | `[n]` patients, `[events]` events | `n_locked_test_patients` |
| Development set | `[n]` patients | `n_development_patients` |
| Seed | `[seed]` | `random_state` (ML) or `random_seed` (deep models) in `request_config` |

The locked test set is never used for fitting, preprocessing, tuning, early stopping or model selection. Models are ranked by cross-validation on the development set. Each model is then refitted once on the whole development set and scored once on the locked test set.

**Primary result:** the locked-test C-index (`locked_test_c_index`) of the model ranked first by cross-validation. Report IBS and Brier skill score where available.

## 5. Cross-validation on the development set

- **Folds:** `[k]`-fold stratified cross-validation repeated `[r]` times (`cv_folds`, `cv_repeats`), with fold seeds `[seed, seed+1, ...]` (`split_seeds`).
- **Same folds for every model:** every classical ML and deep model is trained and scored on identical folds. The comparison records one `evaluation_split_fingerprint`, and results with equal fingerprints were evaluated on the same row partitions.
- **Preprocessing:** refitted inside each training fold.
- **Reported spread:** the mean fold-level C-index with the standard deviation across folds. The spread of repeat means understates variability, so it is not the one reported. The Brier skill score is pooled as 1 − mean IBS / mean null IBS.

## 6. Model-specific internal splits

- **LASSO-Cox:** the penalty is chosen by inner stratified 5-fold cross-validation within each training partition.
- **Deep models:** early stopping monitors a stratified 20% subset of the training partition that never receives gradient updates. The model is then refitted from scratch on the whole training partition for the selected number of epochs. The result records:
  - `refit_epochs`, the number of epochs used for the refit,
  - `early_stopping_fit_samples` and `monitor_samples`, the rows used for early stopping,
  - `fit_samples`, the rows in the final fit.
- **Holdout evaluation:** a stratified 70/30 holdout is shared by classical ML and deep models for the same seed.

## 7. Reproducibility

- SurvStudio version: the export notes start with `Generated with SurvStudio [x.y.z].`
- Settings: the export notes `Replay request_config: ...` contain every setting needed to rerun the analysis.
- Seeds: training, split and monitor seeds are listed in the comparison table (`training_seeds`, `split_seeds`, `monitor_seeds`).

## Example paragraph

> The analysis cohort comprised [N] patients ([events] events) from [source, version], after excluding [k] rows with missing outcomes. A stratified [20]% subset ([n_test] patients, [e_test] events) was locked as an independent test set before any model fitting and was not used for preprocessing, tuning, early stopping or model selection. On the remaining development set ([n_dev] patients), all models were compared by [k]-fold stratified cross-validation repeated [r] times on identical folds (evaluation split fingerprint [fingerprint]), with preprocessing refitted inside each training fold. The model ranked first by the mean fold-level C-index was refitted on the whole development set and evaluated once on the locked test set (C-index [c], IBS [ibs]).

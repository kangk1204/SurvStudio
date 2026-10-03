# Diagnostic follow-up after the failed v1 confirmation

This follow-up contains a separate development study and a newly frozen, completed v2 confirmation. The original v1 study (85,500 datasets and one retained calculation failure) remains unchanged. The v2 linear profile passes the prespecified main-study engineering gates; the spline profile remains exploratory. This result is limited to the declared main conditions and dimensions and does not establish universal error control.

The defect was a separation flag based on coefficient magnitude alone. Correlated spline columns can have large coefficients that cancel at a finite maximum. The correction removes that flag and keeps the pending Newton-step check. Fits, coefficients, covariance, log likelihood, score calculations, fixed knots, PH and HC3 chi-square diagnostics, Holm families and the 1% withholding threshold remain unchanged. The method is now `marker-inference/2`; old v3 prediction recipes preserve their original method provenance, and v1/v2 recipe predictions retain their existing calculation.

## Post-confirmation reason investigation

Read-only queries of all 2,184 v1 ledgers cover 85,500 datasets and 256,500 method records. Reason counts overlap. For the main independent condition (5,000 datasets), linear withholding included 363 residual-mean rejections, 13 residual-variance rejections and 255 functional-form diagnostic failures. Spline withholding included 192 residual-mean rejections, 667 residual-variance rejections and 255 clinical-fit failures. The same spline fits caused those 255 failures in both workflows. PH was a smaller contributor (17 linear and 35 spline datasets).

Independent R checks on three of the original failed spline fits and a separate synthetic fixture confirm a finite maximum without a separation warning. The largest coefficient per SD in the separate fixture is about 50.1; this is not separation. A genuine separation positive control continues to be flagged by Python and R. All new checks satisfy the recorded tolerances; the existing 14 numerical checks also pass on the corrected source.

## New paired development

Seed `2026100311` generated 500 datasets per original condition, totaling 6,000, with 180 subjects, 30 markers and 999 permutations. Each basis shares its raw calculation and permutation stream across v1 reconstruction, v2 and an offline HC3 F-reference sensitivity. The old diagnostic reconstruction was separately checked against 24 frozen v1 records. All 42,000 method records completed without calculation failures. These Monte Carlo results are descriptive development evidence.

| Condition | v1 linear allowed | v2 linear allowed | v1 spline allowed | v2 spline allowed |
|---|---:|---:|---:|---:|
| independent | 86.2% | 91.0% | 76.2% | 79.8% |
| linear | 86.0% | 90.4% | 78.6% | 82.2% |
| correlated | 90.8% | 96.4% | 80.6% | 86.0% |
| z_dependent_censoring | 90.2% | 92.6% | 79.6% | 81.8% |
| nonlinear_marker | 0.0% | 0.0% | 28.2% | 30.2% |
| nonlinear_clinical_risk | 0.0% | 0.0% | 26.2% | 27.6% |

| Signal | Legacy power | v1 linear retention | v2 linear retention | v2 spline retention |
|---|---:|---:|---:|---:|
| partial_weak | 0.09040 | 91.15% | 92.92% | 80.53% |
| partial_strong | 0.48400 | 90.58% | 92.81% | 78.60% |

The simple HC3 F(q,n-k) sensitivity changes the reference distribution but is not an exact robust finite-sample test. It did not resolve spline availability and power deficiencies and is excluded from the product and new confirmation. Residual diagnostic miscalibration remains a limitation; numerical agreement alone does not establish valid tail probabilities.

## Retained calculation failure

The original failure occurred at a wide-block cumulative-sum row access. A deterministic input replay succeeded; 100 additional fixed repeats on the exact frozen v1 sources/runtime also succeeded. They are diagnostic repeats of one input, not replacement confirmation observations or an estimate of the population failure rate. The original error and traceback remain published. Its root cause is unresolved. New development exception logs capture trace-local array shape, dtype and index type if the failure recurs.

## Confirmation and release boundary

`confirmation_protocol.json` fixes v2 with new main seed `2026100312` and extension seed `2026100313`. It preserves all original scenarios, supported sets, 85,500 planned datasets, 999 permutations, 1% Holm withholding and engineering gates. It excludes the offline F candidate. The sources were frozen at `1ed6c10ddad27be6bce4b0896f55ac8428e36dd9`; the seal was pushed before dispatch. Each server used its own pinned environment and source-isolation preflight. All 85,500 fixed datasets and 256,500 paired method records completed in 1,872 ledgers with no calculation failures. Base R independently recomputed all 2,826 aggregate quantities; the largest absolute difference was 8.882e-16. The complete outcomes, source/ownership hashes, failure record and audit are in `results/20261003/confirmation-v2`. No model, cutoff, supported set or DGP was changed after inspecting the v2 confirmation.

| Main condition | Linear allowed | Linear all-planned FWER | Conditional FWER | Conditional one-sided 95% upper bound |
|---|---:|---:|---:|---:|
| independent | 91.92% | 4.62% | 5.03% | 5.59% |
| linear | 91.70% | 4.54% | 4.95% | 5.51% |
| correlated | 94.94% | 4.62% | 4.87% | 5.41% |
| z_dependent_censoring | 91.58% | 4.36% | 4.76% | 5.31% |

Weak and strong partial-null linear power are 8.412% and 43.244%, retaining 90.53% and 90.33% of the corresponding original calculation. These descriptive point ratios just exceed the fixed 90% gate; they are not lower confidence bounds proving 90% retention. Both error upper bounds, availability and power criteria pass in every prespecified supported main linear condition. The source-bound product registry retains the failed v1 studies separately. Dataset diagnostic failures still withhold inference even under a passed method profile.

Spline availability is 81.00–86.54% in the four healthy main conditions, but only 28.30–28.32% in its two supported nonlinear conditions. The nonlinear clinical-risk conditional error upper bound is 6.46%, and the nonlinear-marker upper bound is 6.08%. Weak and strong spline power retention is 80.15% and 78.05%. The spline profile therefore fails as a whole; passing healthy subsets do not enable it.

The full extension grid remains part of the record. With 300 independent markers, linear availability is 78.5% and its conditional error upper bound is 7.57%. With 3,000 independent markers, linear availability is 52.0% and the upper bound is 8.74%; the corresponding linear-marker availability is 49.0%. The 500-subject extension also has limiting error bounds and strong-signal power retention. These extensions do not establish broader qualification. Healthy-condition diagnostic false alarms and approximate residual-test calibration remain limitations.

The finite main-study result supports only its stated scenarios. It does not establish universal error control, clinical utility, external prediction superiority or a human usability benefit. The v1 manuscript and review package remain historical deliverables; the new manuscript update is maintained privately.

The separation-check rationale can be inspected in the primary [R survival implementation](https://github.com/therneau/survival/blob/master/R/coxph.fit.R). The coefficient size cap was SurvStudio's additional heuristic, not R's warning rule. The classical PH implementation is retained; [R documentation](https://www.stat.ethz.ch/R-manual/R-devel/library/survival/html/cox.zph.html) distinguishes the older approximation from the modern score test.

# Conditional marker inference and the 2026 confirmation study

SurvStudio's conditional marker calculations are **exploratory only** in this branch. Both prespecified clinical-basis profiles failed at least one engineering gate. The API withholds added-value p/q values and robust tiers, retains raw calculations under explicitly exploratory fields, and carries this state into figures, CSV, reports and version 3 locked recipes. An estimable fixed prediction can still be exported and scored externally. Diagnostic non-rejection does not certify assumptions.

## What changed

The default clinical basis remains linear. The optional restricted cubic spline uses five knots at the training quantiles 0.05, 0.275, 0.50, 0.725 and 0.95, with Hmisc norm-2 normalization and linear tails. Binary and categorical variables retain their existing encoding. Median imputation, category levels, scale parameters and knots are fitted within each training subset and frozen for held-out or external rows. Duplicate knots, required rank failures and clinical model failures withhold conditional inference without substituting a marginal analysis.

Clinical diagnostics combine classical Grambsch–Therneau log-time PH tests and, for a linear baseline, a likelihood-ratio contrast against the fixed spline expansion. HC3 Wald diagnostics examine nonlinear marker-residual means and variance changes. Holm adjustment is applied separately to the clinical family and to the complete residual family over markers. The withholding threshold remains 1%. Version 1 and 2 predictions retain their numerical semantics and have unevaluated diagnostic status.

A separate product qualification policy applies the study's model-level decision. Numerical research kernels remain at the frozen revision. The API's inference mask is an output policy, not a new calibration experiment. Its absence of reported discoveries must not be called successful universal error control. The internal train-versus-left-out gap of a diagnosis-governed procedure does not validate an exploratory full-cohort marker signature; the qualified API masks that gap-adjusted C-index and retains its original calculation separately.

## Frozen study and complete results

The immutable study revision is `7bf9218ff48c8e66069f60cdad0c51bd8e873d22`. The freeze SHA256 is `7aeaa9b5b4de257a0c7a836572246ec3efebf58deb4a4f529c9af448e7387d8f`. Python 3.12.3 and numpy 2.5.3, pandas 3.0.6, scipy 1.18.1 and statsmodels 0.15.0 were used on all three execution servers.

All **85,500 fixed datasets** and 256,500 paired method records are present: 60,000 main, 24,000 extension and 1,500 large-marker datasets. The 39 study cells use the protocol's distinct development, main and extension seeds. Neither supported conditions nor decision criteria were changed after confirmation inspection. Interrupted jobs resumed only unrecorded indexes in immutable source snapshots; recorded failures were never replaced. Server ownership and hashes for all 2,184 ledgers are published with the aggregate outcomes.

Main-study values below are percentages. Conditional FWER uses only datasets allowed by diagnostics; all-planned FWER includes withheld datasets. These are **research-prototype results before the product qualification gate**.

|Condition|Original raw FWER|Guarded linear FWER / allowed|Guarded spline FWER / allowed|
|---|---:|---:|---:|
|Independent markers|4.82|4.16 / 87.30|3.66 / 77.54|
|Linear marker relation|4.60|3.98 / 87.72|3.98 / 77.40|
|Correlated markers|5.22|4.68 / 90.10|4.20 / 81.72|
|Z-dependent censoring|4.74|4.08 / 89.06|3.82 / 79.24|
|Nonlinear marker relation|5.30|0.00 / 0.00|1.24 / 27.02|
|Heteroskedastic markers|5.06|0.00 / 0.30|0.00 / 0.00|
|Time reversal with linear markers|5.34|0.62 / 9.64|2.12 / 50.88|
|Time reversal with nonlinear markers|69.64|0.00 / 0.00|0.90 / 17.26|
|Time reversal with heteroskedastic markers|24.90|0.00 / 0.02|0.00 / 0.00|
|Nonlinear clinical risk|99.14|0.00 / 0.00|1.10 / 25.54|

The complete table reports raw and guarded FWER, conditional FWER, exact binomial Monte Carlo intervals, one-sided 95% upper bounds, availability, false alarms, partial-null power and failure bounds. A conditional error rate is unavailable when no dataset was allowed. Zero reported FWER caused by zero availability is not successful inference.

Linear power was 8.20% and 40.84% under weak and strong partial-null signals, compared with 9.428% and 46.976% for the original calculation. Retention was about 87%, below the fixed 90% gate. Spline power was 7.084% and 35.596%, and its healthy-condition main availability was 77.40–81.72%. The spline profile additionally failed availability gates. Both remain exploratory; no favorable subset was used to qualify a default.

There was one numerical failure in the expanded strong partial-null spline cell (n=180, 300 markers, index 570). Its original IndexError and fixed index remain in the published failure record and denominators; the all-planned FWER bound for that cell is 3.1–3.2%. A separate diagnostic rerun of the same input did not reproduce it. Its cause remains unresolved, and the rerun does not replace the confirmation result.

Increasing sample size improved admission in healthy conditions, while increasing marker count reduced it markedly. With 3,000 independent markers, admission was 42.0% for guarded linear and 17.4% for guarded spline. Fixed spline tails also leave approximation limits for the globally quadratic marker relation. The current results do not establish adequate residual exchangeability, partial-null validity in general, or a universal 5% guarantee.

## Executed comparison and its scope

Synthetic A/B datasets and an independent R survival workflow cover eight declared tasks: positive-event coding, KM estimates, adjusted Cox coefficients and reference categories, misspecification handling, duplicate profiles and outcome-role overlap, selection on training rows with held-out scoring, frozen external prediction, and endpoint mismatch handling. All 84 numerical quantities agreed within their declared tolerances; six discrete checks required exact equality. The nested-selection comparison executes both numerical kernels on fixed inner training rows; application resampling is verified separately, not inferred from that task.

The independent spline/diagnostic reference compares 14 further quantities using Hmisc and survival. This establishes implementation agreement for the stated fixtures, not statistical adequacy. The R workflow explicitly implements the study rules and can reproduce them; no general claim that R cannot implement these checks is made. SurvStudio bundles the state propagation and versioned recipe into its own application.

KM Plotter custom-data execution is **unverified**. Its [accessed terms](https://kmplot.com/analysis/index.php?cancer=custom_plot&p=service) prohibit automated/non-human access. No upload or automated task execution was performed. Its page documents multivariate and PH-related options; unexecuted tasks are not evidence that it lacks those capabilities. This limitation prevents a complete empirical comparison with that web tool.

The same-server common numerical benchmark uses 3 warmups and 10 measured Efron Cox fits per engine. Operation time excludes prepared input and design construction; process peak RSS includes the worker process. Median times were 6.008 ms for SurvStudio and 9.000 ms for R survival, with peak RSS 215,328 and 178,472 KB. Shared server load and the R timer's millisecond resolution constrain interpretation. Neither these times nor scripted steps measure human usability.

## Existing external case reanalyses

All six development analyses were withheld by diagnostics before product qualification. No case was used to choose a basis or signature externally. Independent R checked 852 external transform, prediction, concordance, calibration and endpoint-specific pooling quantities, with a maximum absolute difference of 3.956e-9. All comparisons met the 1e-6 solver tolerance.

The default-linear, as-measured primary results are exploratory estimates. Intervals below are 95% HKSJ intervals; the complete aggregate includes every cohort, spline sensitivity and within-cohort rescaling sensitivity.

|Case / endpoint|Cohorts / patients / events|External ΔC (95% HKSJ)|Calibration slope (95% HKSJ)|
|---|---:|---:|---:|
|I / OS|7 / 1520 / 576|0.0077 (-0.0290, 0.0444)|0.7118 (0.3906, 1.0331)|
|IV / OS|3 / 598 / 173|-0.0408 (-0.1930, 0.1114)|0.3912 (-0.4553, 1.2376)|
|V / RFS|3 / 552 / 160|0.0084 (-0.1010, 0.1178)|0.9508 (0.0965, 1.8050)|
|V / DMFS|2 / 362 / 95|-0.0058 (-0.7392, 0.7277)|0.7767 (-1.1975, 2.7508)|

Every primary gain interval includes zero. Case V RFS is primary and DMFS secondary; no mixed-endpoint gain is reported. With only two DMFS cohorts, prediction intervals are unavailable. The broad intervals and failed engineering qualification preclude an external-superiority claim.

## Reproduction and preservation

Run the study at its frozen revision and numerical environment, following [the validation workflow](../validation/guarded_inference/README.md). Public [aggregate evidence](../validation/guarded_inference/results/20261003/aggregate-manifest.json) includes synthetic fixtures, per-replicate outcomes, complete summaries, source freezes, independent R outputs and figure source tables. Regenerate figures with `make_figures.py --results validation/guarded_inference/results/20261003`; audit coverage and R aggregate calculations with `audit_publication.py --results ... --with-r`.

Cases I, IV and V are reanalyses of existing external cohorts. Linear is prespecified primary and spline is a fixed sensitivity, with no external basis/signature selection. As-measured marker scaling and frozen development clinical transformations are primary; within-cohort marker rescaling is a separate sensitivity. RFS and DMFS are pooled separately. All new patient rows, score-level reference files and manuscript documents remain in private permanent evidence directories. Existing cohort QC exclusions are retained; the new v3 median rule can retain external rows formerly omitted for missing clinical values. The original 323bb5d3 checkout and previous analysis/postprocessing/final package are preserved.

The branch is a reviewable research candidate. Failed engineering profiles, incomplete web execution and the unresolved recorded runtime failure remain explicit limitations. Publication and formal release must not present it as a clinically validated or universally calibrated method.

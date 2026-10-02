# Guarded marker inference: development candidate

This branch preserves the analysis at `323bb5d3` and adds a versioned policy for withholding
conditional marker inference when the requested clinical model or required diagnostics fail.
It does not establish universal error control, clinical utility, or human usability gains.
The numerical implementation and the confirmation protocol are separate pieces of evidence.

## Clinical transformation

`/api/marker-evaluation` accepts `clinical_basis`: `linear` (default) or
`restricted_cubic_spline`. Each continuous variable uses five training quantiles
`0.05, 0.275, 0.50, 0.725, 0.95`; binary and categorical variables retain their original
encoding. The raw linear term and three nonlinear terms follow
[Hmisc's norm=2 basis](https://search.r-project.org/CRAN/refmans/Hmisc/html/rcspline.eval.html).
Median imputation, level/reference encoding, knots and nonlinear-term scaling are learned
on each training subset. Clinical Cox adjustment and Smith marker residualization use the
same basis. Predictions on held-out or external rows use the saved transformation.
Duplicate knots, rank deficiency, nonfinite inputs and unusable clinical fits withhold
conditional inference. There is no automatic replacement with a marginal screen.

## Diagnostics and output policy

The clinical family contains global/per-term classic Grambsch–Therneau log-time PH tests
(the pre-survival-3.0 approximation, not modern `cox.zph`'s score test), plus the LR
comparison with the fixed spline expansion for a linear model. The residual family
contains nonlinear-mean and variance HC3 Wald chi-square tests for every retained marker.
The variance response removes the fitted-projection effect by dividing squared OLS
residuals by `1-h`. Holm adjustment applies separately to the complete two families.
The withholding threshold is fixed at **1%**. Required calculation failures also withhold.

`inference` records `status`, `allowed`, `reasons`, test statistics, corrections,
transformation provenance and `method_version`. A permitted result is
`assumption_dependent`; it is not an assumption certificate. Withheld results have `null`
standard added-value p/q values and no robust designation. Raw statistics and the previous
classification are retained under `exploratory`. `null` does not mean zero, no association,
or a negative result. The API, browser, figure titles, CSV and reporting text use this policy.
Subsample withholding contributes to the resampling denominator and cannot select
added-value markers on that subset.

## Locked recipes

New exports use recipe **v3**, with the clinical basis, fixed transforms, inference status
and diagnostic provenance covered by the recipe hash. Prediction from estimable models
can be exported even when inference is withheld; external results retain that status.
External diagnostics use development knots, imputation and category mappings, including
the saved functional-form contrast. v1/v2 prediction calculations remain compatible and
their diagnostic status is `not_assessed`. Malformed or modified recipes are rejected.

## Confirmation and release gate

The [fixed protocol](../validation/guarded_inference/protocol.json) compares the original
linear calculation, guarded linear inference and guarded spline inference using the
same generated data and permutation stream. It separates development seed `2026100301`,
main seed `2026100302` and extension seed `2026100303`.

The main study contains 12 conditions × 5,000 datasets (`n=180`, 30 markers, 999
permutations). Two extensions contain 12 × 1,000 each (`n=500,p=30` and `n=180,p=300`);
three large-marker conditions contain 500 each (`n=180,p=3000`). In total there are
**85,500 fixed datasets**. A durable ledger records assigned indexes, settings, host,
source/runtime hashes, completion and failures. Failures are retained and never replaced.

Reports include raw FWER, post-withholding FWER over all planned datasets, conditional
FWER over allowed datasets, availability/withholding, healthy-condition false alarms,
partial-null power, Monte Carlo intervals and failure bounds. The supported conditions
are specified before confirmation. Engineering gates require one-sided 95% FWER upper
bounds ≤6% for both planned and allowed datasets, availability ≥80% in supported healthy
global-null conditions and power ≥90% of the original linear method. These are engineering
criteria for this study, not a universal 5% theorem. A failed model remains exploratory;
conditions or thresholds cannot be changed after inspecting confirmation results.

Development experiments are not confirmation evidence. The historical 68.45% experiment
is preserved as a development finding. The current development candidate has shown
substantial withholding, including losses of availability and power in some healthy
conditions. Reduced errors obtained by withholding everything would not satisfy the gates.

## Evidence still required for publication

The implementation is reviewable before completion of the confirmation study. A final
package also requires latest-commit operating-system CI, independent R numerical checks,
executed R survival/KM Plotter comparisons on eight fixed tasks, and reanalysis of existing
Case I/IV/V external data. RFS is the ER-positive primary endpoint; DMFS is separate.
Manuscript numbers and regenerated figures/source tables must agree with those results.
Existing external cohorts are reanalyses, not newly locked unseen validation.
No participant study or claim about users' time, interpretation errors or usability is included.

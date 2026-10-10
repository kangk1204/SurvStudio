# Guarded marker inference: development candidate

This branch preserves the analysis at `323bb5d3` and adds a versioned policy for withholding
conditional marker inference when the requested clinical model or required diagnostics fail.
It does not establish universal error control, clinical utility, or human usability gains.
Numerical agreement checks assess the implementation; model assumptions must be assessed separately.

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
When the full dataset is withheld, its internal gap-adjusted signature C-index is also
withheld from standard summaries and retained as exploratory, including under a passed
method profile. That procedure's subsample gap cannot validate the exploratory full-cohort signature.

## Locked recipes

New exports use recipe **v3**, with the clinical basis, fixed transforms, inference status
and diagnostic provenance covered by the recipe hash. Prediction from estimable models
can be exported even when inference is withheld; external results retain that status.
External diagnostics use development knots, imputation and category mappings, including
the saved functional-form contrast. v1/v2 prediction calculations remain compatible and
their diagnostic status is `not_assessed`. Malformed or modified recipes are rejected.

# Legends for the reproduced paper figures

These legends describe saved numerical results from `3e0c4af1`, repaired competitor and tier postprocessing
from `c1b8908a`, and rendering from `de53a1f6`. They accompany the local evidence package; they do not certify
a submission-ready manuscript or clinical utility. The additional calibration studies have their own protocols.

## Figure 1. Local survival-analysis workflow and marker summary

(a) Inputs, analysis steps and outputs of SurvStudio. Multiplicity adjustment and resampling do not verify
clinical-model adequacy or residual exchangeability. The locked recipe preserves development coefficients;
external within-cohort marker scaling uses the validation cohort's distribution. (b) Native browser capture
displaying saved TCGA-LUAD results: 484 patients, 177 deaths, 20,530 supplied markers and 19,112 tested markers.
The selection procedure was repeated on 200 subsamples with 1,000 permutations in the full screen. Apparent C,
the heuristic subsample gap adjustment, and left-out procedure C estimate different quantities. The paired
left-out gain interval is approximate and refers to repeated selection and fitting, rather than the final
locked model. The separate workflow export reproduces panel a.

## Figure 2. Filtering, multiplicity and selection stability in TCGA-LUAD

(a) Marker counts after the near-constant filter, clinical adjustment, BH adjustment and robust-tier criteria.
(b) Permutation maxima with and without the near-constant filter; dashed lines show their 95th percentiles.
The strongest retained marker is shown for reference. This diagnostic does not establish universal error control.
(c) Selection frequency and direction agreement across subsamples. The five robust markers are labelled.
Tier membership combines evidence and stability within these data and is not proof of causal effects or clinical benefit.

## Figure 3. Internal procedure estimates and locked external validation

(a) Apparent, gap-adjusted and left-out C in TCGA-LUAD, METABRIC overall survival and METABRIC ER-positive
recurrence. External model and clinical C use cohort-level random-effects pooling with 95% HKSJ intervals.
(b) Paired gain over the corresponding clinical model. Internal intervals are approximate subsample-procedure
intervals; external and site-held-out pooled intervals use HKSJ. Thin lines denote prediction intervals where
available; arrows indicate intervals extending beyond the plotting range. The dotted gain of 0.02 is a display
reference, not an established clinical-utility threshold. Case V combines RFS and DMFS cohorts; endpoint-specific
results appear in Figure S5. Site holdout and external validation involve different training samples and signatures,
so their difference does not identify a causal source of optimism or cohort shift. Pooled paired gain uses its
own weights and need not equal the difference between the separately pooled C values.

## Figure 4. Nine models evaluated on the same clinical-data holdout

(a) Harrell C and 95% bootstrap intervals for 147 TCGA-LUAD test patients with 54 deaths; training used
342 patients with 124 deaths. (b) Paired differences from Cox regression, using the same 1,000 bootstrap draws.
The DeepHit label refers to the implemented variant. All model-difference intervals include zero. The ranking
on one split with these settings does not establish general algorithm superiority or clinical utility.

## Figure 5. Different procedure claims and selection replay

(a) Claim rates under a conditional marker-null TCGA-LUAD plasmode. The procedures test different claims:
SurvStudio clinical added value, Mime selection-cohort C, a fitted-signature training test, and an uncorrected
best-cut-off screen. These rates are not interchangeable FWER estimates. Thin intervals are pointwise 95%
Monte Carlo intervals: Wilson for binary independent replicate outcomes and Hoeffding for the Mime mean of
35 dependent selection splits within each of 50 independent replicates. Thick failure bounds assign every
unobserved planned outcome zero or one; they are not confidence intervals. All fixed designs completed.
(b) Reported and externally evaluated C. (c) Gain over clinical covariates; pooled intervals use HKSJ. Mime
selection on all seven cohorts and the 35 overlapping three-selection/four-sealed replay splits do not constitute
independent prospective validation. Candidate and model restrictions, orientation handling and unresolved
training warnings are documented in the comparison records. The best-cut-off procedure is a stylised screen,
not a test of the current KM Plotter service.

## Figure S1. Plasmode rejection, power and performance diagnostics

(a) Probability of any unlinked discovery with pointwise 95% Wilson Monte Carlo intervals. Under alternatives,
the absolute partial-correlation threshold of 0.1 defines an unlinked category, not an exact conditional-null
family; these rows do not prove strong FWER control. (b) Fraction of five true markers detected by family-wise
adjustment and robust-tier selection. (c) Estimation error relative to C in separately generated patients for
the fitted full-development model, under filtered alternatives. These diagnostics have a different training-size
and fitting target from repeated subsample selection; they do not verify 95% selection-procedure coverage.
The complete simulation contains 2,300 fixed replicates in eight conditions; the panel shows the named subsets.

## Figure S2. Locked TCGA-LUAD signature in seven GEO cohorts

(a) Paired gain over age, sex and stage, with 2,000 patient bootstrap draws per cohort and the pooled HKSJ
and prediction intervals. Marker values are mapped using the external cohort distribution without outcomes;
this differs from single-patient deployment with development scaling alone. (b) Marker-level replication after
clinical adjustment. Holm correction is applied to the measured locked markers within each cohort, not to a
joint family across all cohorts. Counts show replicated/measured cohorts; 0/0 denotes an unmeasured marker.
Same direction without significance does not establish replication, and opposite direction is not a known null.

## Figure S3. Locked METABRIC overall-survival signature in external cohorts

(a) Paired gain over age, tumour size, node status, grade and ER in CAL, NKI and TRANSBIG, comprising
598 patients and 173 deaths. Intervals use 2,000 bootstrap draws and random-effects HKSJ pooling. Arrows
show prediction intervals beyond the plot range. (b) Replication of the ten locked markers using within-cohort
Holm correction as in Figure S2. The nonsignificant gains do not establish equivalence or clinical benefit.
Development proportional-hazards diagnostics and their limitations are reported separately.

## Figure S4. Locked ER-positive recurrence signature across RFS and DMFS cohorts

(a) Paired gain over age, tumour size, node status and grade in five cohorts: 914 patients and 255 events.
Each cohort's endpoint, patient count and event count are labelled. Cohort intervals use 2,000 bootstrap draws;
the mixed-endpoint pooled estimate uses HKSJ and a prediction interval. All individual gain intervals include
zero. (b) Replication of the ten locked markers, corrected within each cohort as in Figure S2. Pooling RFS and
DMFS is a mixed-endpoint summary and does not make the endpoints identical. Separate sensitivity results
appear in Figure S5. Robust-tier membership in development is not an established prognostic biomarker claim.

## Figure S5. Site holdout and endpoint sensitivity in the ER-positive recurrence application

(a) Internal procedure gain within the training sites and locked-model gain in each held-out METABRIC site.
The pooled site result uses HKSJ and a prediction interval. Training size and selected signatures differ across
site fits and from the full-development signature. (b) External gains pooled separately for three RFS cohorts
and two DMFS cohorts. Arrows indicate intervals beyond the axis limits; the DMFS HKSJ interval extends
approximately from −0.490 to +0.470 and has no prediction interval with only two cohorts. These results are
sensitivity descriptions of a real-data application, not a known-truth positive control or a causal decomposition
of optimism and population shift.

## Figure S6. Descriptive replication rates by development tier

Fractions of evaluable genes replicated in external cohorts for the three development applications. Counts
and nominal binomial intervals are descriptive: dependence among genes can invalidate nominal coverage.
Replication uses the pooled gene association rule recorded by script 15; it does not equal clinical prediction
gain. In these results the size-matched robust sets coincide with the smallest development p-values by
construction, so their rates do not demonstrate an independent benefit of tier labelling. Working-model logistic
contrasts are exploratory, assume independent genes and do not establish absence of a tier effect.

## Additional figure. Non-PH conditional-null sensitivity

Two thousand datasets per condition contain 180 patients and 30 markers, all conditionally null given Z.
Clinical hazard effects change from +0.8 to −0.8 at time 10; censoring is independent. Markers have either a
linear or quadratic relation to Z. Three prespecified procedures use 999 paired permutations. Points show
step-down rejection rates and pointwise 95% Wilson Monte Carlo intervals. This additional study was motivated
by the earlier METABRIC PH diagnostic and fixed before its own run; it is not independent confirmation or
formal preregistration. The 68.45% linear-residual rejection rate in the nonlinear-marker condition describes
this combined misspecification scenario, not the actual METABRIC FWER. Quadratic adjustment is not adopted
as a result-selected default; the earlier heteroscedastic study also showed a limitation for that specification.

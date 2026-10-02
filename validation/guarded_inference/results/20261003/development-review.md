# Pre-confirmation development review

The numerical sources and supported conditions are retained as specified before confirmation.
Development seed: 2026100301; 50 fixed datasets in each of 12 conditions; 999 permutations.
The main and extension seeds have not yet been used. No failed dataset has been replaced.

The guarded methods reduce reported errors in misspecified conditions by withholding.
Availability and power remain important unresolved limitations: the spline method has
low availability in several healthy conditions, and strong partial-null power retention
can fall below 90%. These are not confirmation outcomes or evidence of universal control.
The supported-condition mapping and 1% Holm thresholds will not be relaxed or selected
in response to these findings. The fixed main study will determine engineering status.
Models that fail any supported gate remain exploratory; no favorable-subset default.

All calculations and required diagnostic failures remain in the fixed-index ledger.
Raw calculation is performed independently of diagnostic success, preserving the legacy
paired comparator when a required diagnostic fails. Monte Carlo and failure ranges
are reported for raw and post-withholding FWER.

Independent R basis, coefficient, frozen-prediction, C, calibration, PH, LR and HC3 checks
passed at maximum absolute solver tolerance 1e-6. This validates numerical agreement,
not exchangeability, subset pivotality, or a universal 5% error guarantee.
No human participant study or new clinical-utility claim is included.

|Condition|Guarded linear allowed/50|Spline allowed/50|Linear power|Spline power|
|---|---:|---:|---:|---:|
|independent|43|37|None|None|
|linear|46|30|None|None|
|nonlinear_marker|0|11|None|None|
|heteroskedastic|0|0|None|None|
|correlated|47|38|None|None|
|z_dependent_censoring|44|43|None|None|
|nonph_linear|3|24|None|None|
|nonph_nonlinear|0|8|None|None|
|nonph_heteroskedastic|0|0|None|None|
|nonlinear_clinical_risk|0|13|None|None|
|partial_weak|47|41|0.09600000000000002|0.076|
|partial_strong|42|36|0.4|0.332|

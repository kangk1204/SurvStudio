# SurvStudio publication v3

This implements the approved 12-week plan starting from d9e8102f. The six v2
numerical source files are unchanged. `marker_evaluation_v3.py` is an explicit
versioned copy of that pipeline: only diagnostic calls, method provenance and
v3 recipe checks differ. No Cox or marker permutation engine was replaced.

The joint diagnostic is **experimental**. A provisional A policy is visible in
`marker_bootstrap_policy.json`; neither candidate is selected or qualified yet.
The API omission default stays v2. Both v3 candidate and basis results must be
retained. A new qualification report does not itself promote a production default.

`protocol.json` fixes 149500 confirmation datasets, 747500 method records,
999 marker permutations and 9999 joint diagnostic draws. Screening is descriptive
with 1999 draws. Screening uses 6000 datasets; independent selection uses 24000.
No failed dataset or diagnostic draw is replaced or removed from its denominator.

Commands (PYTHONPATH=src and single-thread BLAS):

- `verify_reference.py --output PRIVATE_NEW_DIR --r-library PINNED_R_LIBRARY`
- `study.py run --stage cost --owners 4 --owner 0 --output PRIVATE_LEDGER`
- `study.py summarize PRIVATE_LEDGERS --output PRIVATE_NEW_SUMMARY`
- `control.py cost COST_SUMMARY --output PRIVATE_NEW_DECISION`
- `study.py run --stage screen|selection ...` only after the cost and R gates.
- `control.py select SELECTION_SUMMARY --output PRIVATE_NEW_DECISION`
- Commit the selected global policy and use `control.py seal SELECTION REFERENCE`.
- `study.py run --stage main|extension|large|stress --freeze SEALED_MANIFEST ...`
- `control.py qualify MAIN EXTENSION LARGE STRESS --output PRIVATE_NEW_REPORT`

Cost pilot: 20 fixed datasets per each of four shapes, both candidates and bases,
4 owners. The estimate uses the observed 95th-percentile paired cost, all declared
confirmation counts, 48 workers and a factor of two. Recheck resources and use the
actual available worker count before full dispatch. If the estimate exceeds 21 days,
retain the pilot and complete the v2-based paper without lowering repeats or draws.

Private case steps are export, lock, evaluate. `case_protocol.json` fixes the known
Rotterdam-to-GBSG benchmark, PGR marker, clinical covariates, endpoint sensitivities
and horizons. A primary basis failure is recorded, not replaced by its sensitivity
analysis. No external model evaluation is permitted before recipe integrity checks.
The new actual data, recipes and manuscripts are not public Git outputs.

`comparison.py fixtures` prepares synthetic manual tasks. Research-team human
execution of KM Plotter and surviveR is pending until the records and supporting
screenshots/downloads are supplied. Do not automate those web tasks. The lifelines
comparison is a common numerical benchmark, not a claim of unsupported workflows.

Completion requires fresh CI on the final sources, independent R, cross-host
reproduction, all planned indexes, source tables/figures/manuscript agreement and
factually completed author declarations. A launched job is not scientific completion.
Hard stops: source freeze 2026-10-31; computation 2026-11-28; package 2026-12-26.

Persistent execution: `execute.py pipeline --output PRIVATE_DEVELOPMENT_DIR
--cost PRIVATE_COST_DIR --reference PRIVATE_R_VERIFICATION_JSON` waits for the
80-index pilot, applies the 21-day cost gate, checks identical numerical environments
and current CPU/RAM/disk availability, then runs screen and selection with 48
immutable owners. It stops at the selection/source-seal handoff. A second manager
cannot dispatch duplicate owners; interrupted owners require retained-log review.
Source/configuration changes never silently resume an existing scientific ledger.

The initial implementation is a development checkpoint, not the final review
package. Independent confirmation/aggregate R validation, actual case R validation,
fresh cross-host end-to-end reproduction, manual web records, final figures and
clean/marked manuscripts remain completion gates. The public case code stores
clinical-only and clinical-plus-PGR baselines separately, with no external
recalibration; both use 365.25 days per year and fixed 7-year administrative censoring.

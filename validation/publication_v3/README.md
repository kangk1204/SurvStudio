# SurvStudio publication v3

This implements the approved 12-week plan starting from d9e8102f. The six v2
numerical source files are unchanged. `marker_evaluation_v3.py` is an explicit
versioned copy of that pipeline: only diagnostic calls, method provenance and
v3 recipe checks differ. No Cox or marker permutation engine was replaced.

The joint diagnostic is **experimental**. The complete, independent development
selection selected B (`restricted_wild`) by the fixed rule; both candidates passed
the development point-estimate criteria. The byte-preserved full results are in
`results/20261004-candidate-selection`. The same candidate applies to both bases.
No v3 basis is qualified yet. The API omission default stays v2. Both candidate
and basis development results are retained. A qualification report does not itself
promote a production default.

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
- Commit the selected global policy, verify all ten R fixtures with 9999 draws
  and at least 30 markers, and complete all nine CI jobs on that commit.
- `control.py seal SELECTION FULL_R_REFERENCE CURRENT_CI ORIGINAL_SELECTION_SUMMARY --output PRIVATE_NEW_SEAL`
- `study.py run --stage main|extension|large|stress --freeze SEALED_MANIFEST ...`
- `control.py qualify MAIN EXTENSION LARGE STRESS --freeze ORIGINAL_SEAL --aggregate-references MAIN_R EXTENSION_R LARGE_R STRESS_R --output PRIVATE_NEW_REPORT`

Before interpreting that report, run `confirmation_audit.py MAIN EXTENSION LARGE
STRESS --freeze SEALED_MANIFEST --aggregate-references MAIN_R EXTENSION_R LARGE_R
STRESS_R --output PRIVATE_NEW_AUDIT`. It requires every
fixed cell, method and replicate count, exact source/environment/seed/seal identity,
original UTC attempt, independently verified aggregate, and consistent allowance
and null conditional values. A missing or duplicated
cell cannot pass a reduced support set. This audit does not promote a method.

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

`verify_case_reference.py --self-test --output PRIVATE_NEW_DIR` checks a generated
synthetic case, including missing inputs, tied events and fixed extrapolation.
For the actual case, pass `--data`, `--recipes` and `--evaluation` after lock and
external evaluation. R reads raw rows, prespecified settings and common bootstrap
row indices only. It independently estimates training transforms, both Cox models,
baselines, fixed predictions, C, paired C intervals and absolute-risk calibration.
R numerical warnings stop verification. A synthetic PASS is not actual-case evidence.

`verify_reference_full.py --output PRIVATE_NEW_DIR --markers 2 --r-library
PINNED_R_LIBRARY` supplements the unchanged development reference with 9999 draws
in all ten candidate/basis/design fixtures. `--markers 30` checks a larger joint
family. Its separate source footprint preserves the active development ledgers;
fresh final-source R verification is still required before confirmation seal.

`verify_aggregate_reference.py --self-test --output PRIVATE_NEW_DIR` checks all
9,999 Monte Carlo draws on synthetic raw indicator ledgers with missing and failed
indices, zero allowed analyses, undefined bootstrap denominators, a zero-power
baseline and an entirely unresolved cell. For a completed stage, pass `--ledgers`
and `--summary`. R reads raw method decisions, timings, reasons and the shared
resampling rows only; it recomputes counts, Clopper–Pearson limits, power retention,
paired differences and conservative failure bounds. Python summary values are
never provided to R. This verifies aggregation and does not independently validate
each simulated decision. Resampling streams are generated one method cell at a
time and removed after hash checks by default; seed words, encoding and byte hashes
are retained for exact regeneration. `--retain-streams` keeps the binary streams.

The development checkpoint under `results/20261003-development-checkpoint`
preserves the complete cost and screen aggregate grids and implementation checks.
It contains no selection decision, confirmation outcome, patient rows or manuscript.

`benchmark.py --inputs PRIVATE_FIXTURES --output PRIVATE_NEW_DIR` extends the
same-server common Cox benchmark to SurvStudio, R survival and lifelines. Each
engine uses three warmups and ten measured fits. Imports/input preparation are
outside the operation timer; whole-process peak RSS includes them. Shared load and
R timer resolution are recorded. Web responses and human usability are not measured.

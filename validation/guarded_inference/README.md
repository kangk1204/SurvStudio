# Reproducible guarded-inference validation

Install the repository, Python 3.12.3 and the confirmation numerical versions
`numpy==2.5.3 pandas==3.0.6 scipy==1.18.1 statsmodels==0.15.0`.
Independent R validation requires `survival` and `Hmisc`.
Use a permanent, private output directory outside the Git checkout.

1. Run `run_independent_reference.py --output OUTPUT/independent_R` (optional `--rlib DIR`).
   The script independently reconstructs R spline transforms and compares coefficients,
   frozen external prediction, concordance, calibration, classical PH, functional-form LR
   and HC3 calculations; it saves sources, inputs, R outputs and tolerances.
2. Run `launch_development.py --output OUTPUT/development --replicates 50 --workers 8`.
   Review all 12 conditions, retaining failures and availability/power losses.
3. Commit the numerical sources. Write the development review and run
   `freeze.py --reference OUTPUT/independent_R/verification.json
   --development OUTPUT/development/summary.json --review-record OUTPUT/review.md
   --output OUTPUT/freeze.json`. Stale source evidence is refused; the sealed file cannot
   be overwritten. Record its SHA256 and every server's exact runtime.
4. Assign nonoverlapping global worker IDs before dispatch. For example, a 56-worker
   inventory can assign IDs 0–15, 16–35 and 36–55 to three servers. Launch each server's
   `launch_confirmation.py --freeze OUTPUT/freeze.json --output OUTPUT/confirmation
   --total-workers 56 --offset OFFSET --slots SLOTS` under a durable process supervisor.
   Threads here supervise separate numerical processes. Each owns `index % 56 == ID`
   across all 39 study cells. BLAS threads are fixed at one.
5. Resume using the identical inventory and output directory. Completed indexes, including
   recorded failed calculations, are skipped. A process interruption may resume an index
   that has never been recorded. Do not duplicate ownership or replace failed datasets.
6. Collect each server's SQLite ledgers after it stops writing and run
   `study.py summarize LEDGER... --output OUTPUT/confirmation-summary.json`.
   Duplicate indexes/configuration mismatches are rejected. Missing cells remain pending;
   the report must cover all 85,500 indexes before declaring complete.

Live status comes from ownership/status files and committed SQLite rows. Do not aggregate
a live copy made in the middle of a transaction as final evidence. Runtime compatibility
is checked before dispatch. Any numerical repair after confirmation begins needs a new
method version and new confirmation seeds; preserve this frozen study.

No raw patient data or manuscript belongs here. Public artifacts may include synthetic
reference data, aggregate summaries and their provenance after validation.

## Completed study and independent reproduction

The complete 2026 study is in `results/20261003`; numerical confirmation sources
remain frozen at `7bf9218ff48c8e66069f60cdad0c51bd8e873d22`. The current API separately
applies the failed-profile qualification mask. See `../../docs/guarded_inference_validation.md`.

- `compare_tools.py --output PRIVATE/tool_comparison_qualified_v2 --rlib RLIB` executes
  the fixed synthetic A/B tasks against R. KM Plotter execution is unverified under
  its automated-access terms; it must not be inferred from documentation alone.
- `benchmark.py --data PRIVATE/tool_comparison_qualified_v2/A.csv --output PRIVATE/tool_benchmark` measures only the common same-server
  numerical operation, with three warmups and ten measured repetitions.
- `reanalyse_cases.py --case I --basis linear --data PRIVATE/data --output PRIVATE/cases/case-I-linear
  --development-only` fixes a case development analysis. Repeat for I/IV/V and the
  prespecified spline sensitivity, without consulting external performance.
- `external_fixed_cases.py --case I --development PRIVATE/cases/case-I-linear/development-analysis.json
  --data PRIVATE/data --output PRIVATE/cases_final/case-I-linear` applies both frozen
  primary and within-cohort sensitivity marker scalings, with endpoint-specific pooling.
- `verify_cases.py --output PRIVATE/cases_final/case-I-linear --rlib RLIB` compares
  private score-level inputs with R. `collect_case_aggregate.py --evidence PRIVATE
  --output PRIVATE/case_aggregate` checks all six references and exports only aggregate
  allowlisted files. Original case patient data and source archives are prerequisites;
  public aggregate reproduction is not a patient-level reanalysis.
- `make_figures.py --results validation/guarded_inference/results/20261003` regenerates
  six figures and source tables with matplotlib. Preserve manifests before regeneration;
  then validate regenerated hashes and inspect the rendered figures.
- `audit_publication.py --results validation/guarded_inference/results/20261003 --with-r`
  verifies the immutable source freeze, all fixed indexes, package hashes and 2,826
  independent base-R operating-characteristic quantities.

`publish_aggregate.py` expects separate latest-source numerical reference and qualified
comparison directories, preserving the preconfirmation reference used by the freeze.
Its central manifest excludes the audit outputs that refer back to that manifest.
No new patient-level input, prediction row, full real-case recipe or manuscript is copied.
Both profiles failed qualification; the zero-discovery API mask is not an error-control
claim and is not a substitute for a newly frozen statistical repair study.

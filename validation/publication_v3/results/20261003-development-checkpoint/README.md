# Development evidence checkpoint, 3 October 2026

The preserved v2 study is not re-used as v3 confirmation. These are aggregate
outputs from the new fixed development protocol and numerical implementation
checks. Neither diagnostic candidate is selected or qualified at this checkpoint.

- Cost: all 80 fixed datasets and 560 method records completed. The paired pilot
  estimate is 3.6976 days with 48 workers and a factor-of-two margin. This is a
  resource projection using a one-clinical-variable pilot, not an observed full
  confirmation runtime or statistical improvement result.
- Descriptive screen: all 6000 datasets and 42000 method records completed, with
  zero numerical failures, diagnostic calculation failures or missing records.
  All 84 method cells are retained. The separate selection study is still running.
- Joint diagnostic R reference: both candidates, five basis/design settings and
  9999 draws in every fixture. Separate complete runs use two and thirty markers per fixture (twenty checks in total).
  Every check passed; the largest maximum-statistic difference was 3.271e-12.
  Maximum-statistic agreement and exact exceedance counts establish implementation
  agreement for these inputs, not validity of the survival inference procedure.
- Synthetic public-case R check: 292 comparisons passed. These are generated rows,
  not a Rotterdam–GBSG prediction result. Actual additional-case analysis is pending.
- Independent R aggregation: 25 synthetic method cells, 1143 comparisons and
  9999 draws per applicable interval passed, with maximum difference 3.109e-15.
  Fixtures retain missing and failed repeats, zero allowed analyses, undefined
  bootstrap denominators, a zero-power baseline and a wholly unresolved cell.
  This validates aggregation from raw decisions, not each underlying decision.
- Common Cox benchmark: three engines, three warmups, ten measured fits, same
  server and input. Imports/preparation are excluded from the operation timer;
  peak RSS includes process imports. Shared load and R timer resolution limit
  comparison. No whole-tool speed, usability or prediction superiority is inferred.
- CI: all nine jobs passed on 5cf70f0, including Linux/macOS/Windows, browser, R
  and wheel checks. The subsequent 92cc602 full suite exposed a test-fixture module
  leak; related existing/new tests pass together after isolation was repaired.
  These earlier CI results do not substitute for fresh CI on the final sources.
- Initial three-host synthetic outputs matched exactly. Final selected-source
  reproduction in a fresh environment remains required.

`manifest.json` lists immutable output hashes. Input/stream and numerical-source
hashes are retained in the individual reports. Verification scripts regenerate
synthetic inputs and fixed streams. Only code, aggregate results and provenance
are public here. Manuscripts, new patient rows and actual-case recipes stay private.

Pilot/development owners have UTC start/end records; their replicate rows have
elapsed durations. Per-replicate absolute UTC and attempt records must be added
before confirmation. Earlier times will not be fabricated retrospectively.

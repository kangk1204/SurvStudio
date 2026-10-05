# Complete development candidate selection

The fixed selection study completed 24,000 datasets and 168,000 method results
across all 168 development method cells. All 48 original owners finished with
exit code zero. There were no missing datasets, calculation failures, or
diagnostic calculation failures. The original scientific source was
`ad6f09ef45d1359e0ec671b928f1152b9eb93720`; seed was `2026100322`.

Both prespecified candidates met the development point-estimate criteria. The
fixed ordering selected **B, restricted wild bootstrap**, because it had the
higher minimum allowance across the supported healthy linear conditions.
The same candidate is locked for both clinical bases.

| Development metric across supported linear cells at P=30 and P=300 | A: residual vectors | B: restricted wild |
|---|---:|---:|
| Maximum all-planned FWER point estimate | 5.40% | 5.40% |
| Maximum allowed-dataset conditional FWER point estimate | 5.5901% | 5.5158% |
| Minimum healthy analysis allowance | 97.8% | 97.9% |
| Minimum weak/strong signal power retention | 96.1009% | 96.6997% |
| Mean diagnostic seconds over the selection support cells | 35.1843 | 30.5408 |

These are **development selection results**, not independent confirmation or
production qualification. Point estimates do not establish the prespecified
one-sided confidence-bound adoption criteria. The full `summary.json` includes
all conditions and both bases, including unsupported and misspecified settings;
the table above defines the prespecified selection subset only.

The byte-preserved decision and summary are bound by SHA256. `manifest.json`
records every fixed owner and its private synthetic ledger hash. Original
ledgers, process completion times and logs remain in persistent private storage.
Dataset-level UTC attempt times were not recorded by the development runner and
are not reconstructed retrospectively. New confirmation runs record original
attempt start/end/status before and after calculation.

The product policy remains exploratory with `qualification_status=not_evaluated`.
Omitted API policy continues to use v2. Fresh CI, independent R on the complete
committed numerical footprint, and a dated source seal are required before the
149,500-dataset confirmation study starts. No patient rows, manuscript, or actual
case recipe is included here.

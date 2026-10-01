"""Write paper/breast_data_manifest.csv: every file the breast case studies read (each file under cohorts/breast,
and cohorts/breast_duplicates.csv; see common.breast_data_files), relative to the cohorts folder, with its size and
SHA-256. Commit the manifest (metadata only); 00_self_check.py stops when the data differ from it.
Run once after exporting the data with data_prep/export_curated_cohorts.R: python breast_manifest.py
"""

from __future__ import annotations

from common import BREAST, BREAST_MANIFEST, breast_manifest

manifest = breast_manifest()
if manifest.empty:
    raise SystemExit(f"No breast data under {BREAST}.")
temporary = BREAST_MANIFEST.with_name(BREAST_MANIFEST.name + ".tmp")
manifest.to_csv(temporary, index=False, lineterminator="\n")
temporary.replace(BREAST_MANIFEST)
print(f"{len(manifest)} files, {manifest['bytes'].sum() / 1e9:.2f} GB, written to {BREAST_MANIFEST}")

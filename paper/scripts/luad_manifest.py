"""Write paper/luad_data_manifest.csv: every LUAD file the analyses read (the Xena RNA-seq matrix, the harmonized
clinical table and the gene-level expression of the seven GEO cohorts; see common.luad_data_files), relative to the
cohorts folder, with its size and SHA-256 and, for a .gz file, the size and SHA-256 of its decompressed content.
Commit the manifest (metadata only); 00_self_check.py stops when the data differ from it.
Run once after rebuilding the LUAD data with prepare_data.sh: python luad_manifest.py
"""

from __future__ import annotations

from common import LUAD, LUAD_MANIFEST, luad_data_files, luad_manifest

missing = [str(path) for path in luad_data_files() if not path.is_file()]
if missing:
    raise SystemExit(f"LUAD data missing under {LUAD}: {', '.join(missing)}")
manifest = luad_manifest()
temporary = LUAD_MANIFEST.with_name(LUAD_MANIFEST.name + ".tmp")
manifest.to_csv(temporary, index=False, lineterminator="\n")
temporary.replace(LUAD_MANIFEST)
print(f"{len(manifest)} files, {manifest['bytes'].sum() / 1e6:.0f} MB, written to {LUAD_MANIFEST}")

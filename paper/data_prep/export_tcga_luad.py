"""Download TCGA-LUAD expression, clinical annotations and TCGA-CDR endpoints from UCSC Xena.

    python data_prep/export_tcga_luad.py <data_dir>

Keeps the raw files under <data_dir>/cohorts/luad/TCGA-LUAD/raw/ and records their URLs,
download dates and SHA-256 in raw/manifest.json. Writes, in the layout of the GEO series:

- clinical.csv: the Xena clinical matrix joined with the TCGA-CDR survival table
  (Liu et al., Cell 2018), primary tumours with expression only;
- expression.csv.gz: samples x genes, log2(RSEM normalized count + 1) (IlluminaHiSeq).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from downloads import fetch

XENA = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
SOURCES = {
    "HiSeqV2.gz": XENA + "TCGA.LUAD.sampleMap%2FHiSeqV2.gz",
    "LUAD_clinicalMatrix.tsv": XENA + "TCGA.LUAD.sampleMap%2FLUAD_clinicalMatrix",
    "LUAD_survival.tsv": XENA + "survival%2FLUAD_survival.txt",
}
PRIMARY_TUMOUR = "01"


def main() -> None:
    folder = Path(sys.argv[1]).expanduser() / "cohorts" / "luad" / "TCGA-LUAD"
    raw = folder / "raw"
    for name, url in SOURCES.items():
        fetch(url, raw / name)

    expression = pd.read_csv(raw / "HiSeqV2.gz", sep="\t", index_col=0).T
    expression.index.name = "sample_id"
    expression = expression[expression.index.str[13:15] == PRIMARY_TUMOUR]

    clinical = pd.read_csv(raw / "LUAD_clinicalMatrix.tsv", sep="\t", low_memory=False).rename(columns={"sampleID": "sample_id"})
    survival = pd.read_csv(raw / "LUAD_survival.tsv", sep="\t").rename(columns={"sample": "sample_id"})
    clinical = clinical.merge(survival, on="sample_id", how="left", suffixes=("", "_cdr"))
    clinical = clinical[clinical["sample_id"].isin(expression.index)].sort_values("sample_id")
    expression = expression.loc[clinical["sample_id"]]

    clinical.to_csv(folder / "clinical.csv", index=False, lineterminator="\n")
    expression.reset_index().to_csv(folder / "expression.csv.gz", index=False, lineterminator="\n")
    print(f"TCGA-LUAD: {len(clinical)} primary tumours with expression, {expression.shape[1]} genes")


if __name__ == "__main__":
    main()

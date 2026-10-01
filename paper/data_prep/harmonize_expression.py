"""Collapse each LUAD cohort's probe-level expression to one column per gene.

    python data_prep/harmonize_expression.py <data_dir> [--gene-info PATH]

Reads only probes.csv and expression.csv.gz of each cohort (no outcomes). Steps:

1. log2-transform when the values look unlogged (the GEO2R rule: 99th percentile > 100,
   or range > 50 with a positive lower quartile); non-positive values become missing.
2. Map probes to a single Entrez Gene ID (probes annotated to several genes are dropped)
   and genes to their NCBI symbol in Homo_sapiens.gene_info. NCBI replaces that file
   daily, so the snapshot the paper used (downloaded 2026-09-26) is read from
   data_snapshot/Homo_sapiens.gene_info.gz, or from --gene-info (or the GENE_INFO
   environment variable), and its SHA-256 is checked. TCGA columns are symbols; they are
   matched to Entrez IDs through the official symbol, then through synonyms that name
   exactly one gene.
3. Keep, per gene, the probe with the highest mean expression (MaxMean), preferring
   probes with at most 20% missing values.

Writes <cohort>/expression_genes.csv.gz (samples x symbols, 4 decimals) and
expression_summary.csv with the gene overlap across cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Source of the snapshot (see data_snapshot/SNAPSHOT.md); the file there is the copy of 2026-09-26.
GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
GENE_INFO = Path(__file__).resolve().parents[1] / "data_snapshot" / "Homo_sapiens.gene_info.gz"
GENE_INFO_SHA256 = "4a333625190f594abc21fd7564c7a0ee991a0331c639c6d3b7bb814b944e75b7"
MAX_MISSING = 0.2
ENTREZ_COLUMN = {
    "GSE13213": "GENE",
    "GSE30219": "ENTREZ_GENE_ID",
    "GSE31210": "ENTREZ_GENE_ID",
    "GSE41271": "Entrez_Gene_ID",
    "GSE50081": "ENTREZ_GENE_ID",
    "GSE68465": "ENTREZ_GENE_ID",
    "GSE72094": "EntrezGeneID",
    "TCGA-LUAD": None,
}


def gene_info(path: Path) -> pd.DataFrame:
    """NCBI's gene table, once its SHA-256 shows it is the snapshot the paper used."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GENE_INFO_SHA256:
        raise SystemExit(f"{path} is not the NCBI gene_info snapshot the paper used (SHA-256 {digest}, expected {GENE_INFO_SHA256}); "
                         f"gene symbols would differ. Use {GENE_INFO}.")
    print(f"gene_info: {path} (SHA-256 {digest}, the snapshot of 2026-09-26)", flush=True)
    return pd.read_csv(path, sep="\t", usecols=["GeneID", "Symbol", "Synonyms"], dtype={"GeneID": np.int64, "Symbol": str, "Synonyms": str})


def probe_entrez(probes: pd.DataFrame, column: str) -> pd.Series:
    """Probe ID -> Entrez ID for probes annotated to exactly one gene."""
    ids = probes[column].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    single = ids.str.fullmatch(r"\d+").fillna(False).astype(bool)
    return pd.Series(ids[single].astype(np.int64).to_numpy(), index=probes.loc[single, "ID"].astype(str).to_numpy())


def symbol_entrez(symbols: pd.Index, info: pd.DataFrame) -> pd.Series:
    """Symbol -> Entrez ID through unique official symbols, then unambiguous synonyms."""
    official_counts = info["Symbol"].value_counts()
    official = info[info["Symbol"].isin(official_counts[official_counts == 1].index)].set_index("Symbol")["GeneID"]
    synonyms = info.assign(synonym=info["Synonyms"].str.split("|")).explode("synonym")
    synonyms = synonyms[(synonyms["synonym"] != "-") & ~synonyms["synonym"].isin(info["Symbol"])]
    synonym_counts = synonyms.groupby("synonym")["GeneID"].nunique()
    synonym = synonyms[synonyms["synonym"].isin(synonym_counts[synonym_counts == 1].index)].drop_duplicates("synonym").set_index("synonym")["GeneID"]
    mapped = pd.Series(symbols, index=symbols).map(official)
    mapped = mapped.fillna(pd.Series(symbols, index=symbols).map(synonym))
    return mapped.dropna().astype(np.int64)


def load_expression(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, dtype=defaultdict(lambda: np.float32, sample_id=str))
    return pd.DataFrame(frame.to_numpy(dtype=np.float32), index=frame.index, columns=frame.columns)  # one block, not one per probe


def needs_log(values: np.ndarray) -> bool:
    q = np.nanquantile(values, [0.0, 0.25, 0.99, 1.0])
    return bool(q[2] > 100 or (q[3] - q[0] > 50 and q[1] > 0))


def collapse(expression: pd.DataFrame, entrez: pd.Series, symbol_of: pd.Series) -> pd.DataFrame:
    entrez = entrez[entrez.index.isin(expression.columns) & entrez.isin(symbol_of.index)]
    probes = expression[entrez.index]
    ranking = pd.DataFrame(
        {
            "probe": entrez.index,
            "entrez": entrez.to_numpy(),
            "complete": (probes.isna().mean() <= MAX_MISSING).to_numpy(),
            "mean": probes.mean().to_numpy(),
        }
    )
    best = ranking.sort_values(["entrez", "complete", "mean"], ascending=[True, False, False]).drop_duplicates("entrez")
    genes = probes[best["probe"].to_numpy()].copy()
    genes.columns = symbol_of.loc[best["entrez"].to_numpy()].to_numpy()
    return genes.loc[:, ~genes.columns.duplicated()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--gene-info", type=Path, default=Path(os.environ.get("GENE_INFO") or GENE_INFO),
                        help="NCBI Homo_sapiens.gene_info.gz of 2026-09-26 (default: the snapshot in data_snapshot/)")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    luad = data_dir / "cohorts" / "luad"
    info = gene_info(args.gene_info.expanduser())
    symbol_of = info.set_index("GeneID")["Symbol"]
    rows, gene_sets = [], {}
    for cohort, column in ENTREZ_COLUMN.items():
        expression = load_expression(luad / cohort / "expression.csv.gz")
        values = expression.to_numpy()
        logged = needs_log(values)
        if logged:
            with np.errstate(invalid="ignore", divide="ignore"):
                expression = expression.where(expression > 0).apply(np.log2)
        if column is None:
            entrez = symbol_entrez(expression.columns, info)
        else:
            entrez = probe_entrez(pd.read_csv(luad / cohort / "probes.csv", low_memory=False), column)
        genes = collapse(expression, entrez, symbol_of)
        genes.round(4).to_csv(luad / cohort / "expression_genes.csv.gz", index_label="sample_id", lineterminator="\n")
        gene_sets[cohort] = set(genes.columns)
        rows.append(
            {
                "cohort": cohort,
                "samples": len(genes),
                "features": expression.shape[1],
                "log2_applied": logged,
                "features_mapped": int(entrez.index.isin(expression.columns).sum()),
                "genes": genes.shape[1],
                "missing_fraction": round(float(genes.isna().to_numpy().mean()), 4),
            }
        )
        print(rows[-1], flush=True)
        del expression, values, genes
    summary = pd.DataFrame(rows)
    counts = pd.Series([gene for genes in gene_sets.values() for gene in genes]).value_counts()
    summary.to_csv(luad / "expression_summary.csv", index=False, lineterminator="\n")
    print(summary.to_string(index=False))
    for k in (len(gene_sets), len(gene_sets) - 1, len(gene_sets) - 2):
        print(f"genes present in >= {k} of {len(gene_sets)} cohorts: {int((counts >= k).sum())}")


if __name__ == "__main__":
    main()

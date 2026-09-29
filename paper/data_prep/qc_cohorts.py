"""Quality control of the harmonized cohorts of one cancer type.

    python data_prep/qc_cohorts.py <data_dir> luad

Reads the harmonized clinical table (harmonized_clinical.csv) and the gene-level expression.
No QC decision uses an outcome: outcome fields enter only consistency flags (checks 5 and 6)
that decide nothing. Exclusions an earlier QC added to the table ("QC: ...") are set aside,
so the samples are judged as the harmonization left them. Checks:

1. identifiers: duplicated sample IDs, samples missing from expression or clinical data,
   and patients with more than one sample;
2. array outliers: correlation of each sample with the cohort's median profile, flagged
   below Q1 - 3 IQR;
3. duplicates within and across cohorts: Pearson correlation of gene-standardized
   profiles over the 5,000 most variable shared genes. A pair is flagged when its
   Fisher-z correlation is a robust outlier (z >= 6 against all pairs of that comparison)
   and the two samples are each other's best match. A flagged pair with r >= 0.8 that
   stands clear of the next-best match (gap >= 0.2) is called a technical duplicate;
   other flagged pairs are explained by excluded tissue or clinical discordance, or
   left for review;
4. sex check: sex predicted from XIST and Y-linked genes against the annotated sex,
   which exposes sample mix-ups; samples whose XIST and Y signals disagree are called
   ambiguous rather than mismatched;
5. clinical consistency: relapse after the last overall-survival follow-up, ages outside
   18-100;
6. TCGA: PanCanAtlas sample-quality annotations (Do_not_use, AWG pathology exclusion,
   unacceptable prior treatment), other malignancies, PanCan RNA clusters that are not
   LUAD-like, and agreement between the TCGA-CDR endpoints and the clinical matrix.

Writes <cancer>/qc/sample_flags.csv, <cancer>/qc/duplicate_pairs.csv, <cancer>/qc/report.md and
<cancer>/qc/sample_decisions.csv (EXCLUDE reasons and SENSITIVITY flags per sample, which
harmonize_luad.py folds into the clinical table). The decisions the paper applied are in
data_snapshot/sample_decisions.csv; prepare_data.sh reruns the QC and compares.
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from downloads import fetch

QUALITY_URL = "https://api.gdc.cancer.gov/data/1a7d7be8-675d-4e60-a105-19d4121bdebf"
TOP_GENES = 5000
MAX_MISSING = 0.2
PAIR_Z = 6.0
DUPLICATE_R = 0.8
DUPLICATE_GAP = 0.2
YEAR = 365.25
FEMALE_GENES = ("XIST",)
MALE_GENES = ("RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY")
FLAGS = (
    "array_outlier",
    "sex_mismatch",
    "technical_duplicate",
    "age_out_of_range",
    "relapse_after_last_os",
    "repeated_patient",
    "do_not_use",
    "awg_pathology_excluded",
    "unacceptable_prior_treatment",
    "other_malignancy",
    "non_luad_rna_cluster",
    "low_purity",
    "vital_status_disagrees",
    "os_time_differs_30d",
    "dfi_after_os",
)
TCGA_FLAGS = FLAGS[6:]
# Decisions written to qc/sample_decisions.csv and applied by harmonize_luad.py: samples
# whose identity or eligibility is in doubt are excluded; the rest of the flags define
# sensitivity analyses.
EXCLUDE = {
    "technical_duplicate": "technical duplicate of another sample",
    "sex_mismatch": "annotated sex contradicts expression",
    "repeated_patient": "patient has another sample",
    "do_not_use": "PanCanAtlas Do_not_use",
    "awg_pathology_excluded": "TCGA AWG pathology exclusion",
    "unacceptable_prior_treatment": "unacceptable prior treatment for another malignancy",
}
SENSITIVITY = {
    "array_outlier": "array outlier",
    "other_malignancy": "other malignancy",
    "non_luad_rna_cluster": "non-LUAD-like RNA cluster",
    "low_purity": "ABSOLUTE purity < 0.2",
}


def load_genes(folder: Path) -> pd.DataFrame:
    frame = pd.read_csv(folder / "expression_genes.csv.gz", index_col=0, dtype=defaultdict(lambda: np.float32, sample_id=str))
    return pd.DataFrame(frame.to_numpy(dtype=np.float32), index=frame.index, columns=frame.columns)  # one block, not one per gene


def gene_standardize(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - np.nanmean(values, axis=0)) / np.nanstd(values, axis=0, ddof=1)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def row_unit(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    return centred / np.where(norm > 0, norm, 1.0)


def robust_z(r: np.ndarray) -> np.ndarray:
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    median = np.median(z)
    return (z - median) / (1.4826 * np.median(np.abs(z - median)))


def variable_genes(a: pd.DataFrame, b: pd.DataFrame) -> pd.Index:
    shared = a.columns.intersection(b.columns)
    shared = shared[(a[shared].isna().mean() <= MAX_MISSING).to_numpy() & (b[shared].isna().mean() <= MAX_MISSING).to_numpy()]
    score = np.maximum(a[shared].var().rank(ascending=False), b[shared].var().rank(ascending=False))
    return score.nsmallest(TOP_GENES).index


def array_outliers(frame: pd.DataFrame) -> tuple[pd.Series, float]:
    values = frame.to_numpy(dtype=np.float64)
    complete = ~np.isnan(values).any(axis=0)
    if complete.sum() >= 1000:
        values = values[:, complete]
    else:
        values = np.where(np.isnan(values), np.nanmedian(values, axis=0), values)
    r = (row_unit(values) @ row_unit(np.median(values, axis=0)[None, :]).T).ravel()
    q1, q3 = np.percentile(r, [25, 75])
    return pd.Series(r, index=frame.index), float(q1 - 3 * (q3 - q1))


def _two_means_high(values: pd.Series) -> pd.Series:
    """True for values in the upper group of a one-dimensional two-means split."""
    threshold = float(values.median())
    for _ in range(100):
        high, low = values[values > threshold], values[values <= threshold]
        if high.empty or low.empty:
            break
        updated = (high.mean() + low.mean()) / 2
        if abs(updated - threshold) < 1e-9:
            break
        threshold = updated
    return values > threshold


def predicted_sex(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sex from standardized XIST and the mean of standardized Y-linked genes, each split
    into high and low by two-means: M when Y is high and XIST low, F when XIST is high and
    Y low, ambiguous otherwise (tumours can lose chromosome Y or XIST). Returns the call
    and the score (Y mean minus XIST)."""
    male = [gene for gene in MALE_GENES if gene in frame]
    female = [gene for gene in FEMALE_GENES if gene in frame]
    if not male or not female:
        empty = pd.Series(np.nan, index=frame.index)
        return empty.astype(object), empty
    z = pd.DataFrame(gene_standardize(frame[male + female]), index=frame.index, columns=male + female)
    y, x = z[male].mean(axis=1), z[female].mean(axis=1)
    y_high, x_high = _two_means_high(y), _two_means_high(x)
    call = np.where(y_high & ~x_high, "M", np.where(x_high & ~y_high, "F", "ambiguous"))
    return pd.Series(call, index=frame.index, dtype=object), y - x


def verdict(pair: pd.Series) -> str:
    if not pair["flagged"]:
        return ""
    if pair["r"] >= DUPLICATE_R and pair["gap_to_next"] >= DUPLICATE_GAP:
        return "technical duplicate"
    if pair["exclusion_a"] or pair["exclusion_b"]:
        return "similar tissue (excluded sample)"
    if pair["sex_agrees"] is False or pair["age_difference"] > 2:
        return "similar profiles, clinically discordant"
    return "review"


def duplicate_pairs(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every pair that is a robust outlier or among the five highest correlations."""
    same = name_a == name_b
    genes = variable_genes(a, b)
    r = row_unit(gene_standardize(a[genes])) @ row_unit(gene_standardize(b[genes])).T
    if same:
        np.fill_diagonal(r, -1.0)
        rows, cols = np.triu_indices(len(a), 1)
    else:
        rows, cols = np.indices(r.shape).reshape(2, -1)
    values = r[rows, cols]
    z = robust_z(values)
    best_of_a, best_of_b = r.argmax(axis=1), r.argmax(axis=0)
    ordered = np.sort(r, axis=1)
    keep = np.union1d(np.flatnonzero(z >= PAIR_Z), np.argsort(values)[-5:])
    table = pd.DataFrame(
        {
            "cohort_a": name_a,
            "sample_a": a.index[rows[keep]],
            "cohort_b": name_b,
            "sample_b": b.index[cols[keep]],
            "r": values[keep].round(3),
            "robust_z": z[keep].round(1),
            "mutual_best": [best_of_a[i] == j and (best_of_a[j] == i if same else best_of_b[j] == i) for i, j in zip(rows[keep], cols[keep])],
            "gap_to_next": (ordered[rows[keep], -1] - ordered[rows[keep], -2]).round(3),
        }
    )
    table["flagged"] = (table["robust_z"] >= PAIR_Z) & table["mutual_best"]
    summary = {"comparison": f"{name_a} vs {name_b}", "genes": len(genes), "pairs": len(values), "median_r": round(float(np.median(values)), 3), "max_r": round(float(values.max()), 3), "flagged": int(table["flagged"].sum())}
    return table.sort_values("r", ascending=False), summary


def tcga_flags(folder: Path, reference: Path, samples: pd.Index) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(folder / "clinical.csv", low_memory=False).set_index("sample_id").reindex(samples)
    quality = pd.read_csv(fetch(QUALITY_URL, reference / "merged_sample_quality_annotations.tsv"), sep="\t", low_memory=False)
    quality = quality[quality["platform"].eq("IlluminaHiSeq_RNASeqV2")].assign(sample_id=lambda frame: frame["aliquot_barcode"].str[:15])
    quality = quality[quality["sample_id"].isin(samples)]
    notes = quality[["patient_annotation", "sample_annotation", "aliquot_annotation", "AWG_pathology_exclusion_reason"]].astype("string")
    quality = quality.assign(notes=notes.apply(lambda row: "; ".join(sorted({value for value in row.dropna() if value.strip()})), axis=1))
    per_sample = quality.groupby("sample_id").agg(
        do_not_use=("Do_not_use", lambda values: bool(values.astype(str).str.lower().eq("true").any())),
        awg_pathology_excluded=("AWG_excluded_because_of_pathology", lambda values: bool((pd.to_numeric(values, errors="coerce") == 1).any())),
        tcga_notes=("notes", lambda values: "; ".join(sorted({value for value in values if value}))),
    )
    flags = pd.DataFrame(index=samples).join(per_sample)
    flags["in_quality_table"] = flags["do_not_use"].notna()
    flags[["do_not_use", "awg_pathology_excluded"]] = flags[["do_not_use", "awg_pathology_excluded"]].fillna(False).astype(bool)
    flags["tcga_notes"] = flags["tcga_notes"].fillna("")
    flags["unacceptable_prior_treatment"] = flags["tcga_notes"].str.contains("History of unacceptable prior treatment", regex=False)
    flags["other_malignancy"] = raw["other_dx"].astype("string").str.lower().str.startswith("yes").fillna(False).astype(bool).to_numpy()
    cluster = raw["_PANCAN_UNC_RNAseq_PANCAN_K16"].astype("string")
    flags["non_luad_rna_cluster"] = (cluster.notna() & (cluster != "LUAD-like c10")).fillna(False).astype(bool).to_numpy()
    flags["rna_cluster"] = cluster.to_numpy()
    deceased = raw["vital_status"].astype("string").str.upper().map({"DECEASED": 1.0, "LIVING": 0.0}).astype(float)
    flags["vital_status_disagrees"] = (deceased.notna() & raw["OS"].notna() & (deceased != raw["OS"])).to_numpy()
    matrix_time = pd.to_numeric(raw["days_to_death"], errors="coerce").fillna(pd.to_numeric(raw["days_to_last_followup"], errors="coerce"))
    flags["os_time_differs_30d"] = ((matrix_time - pd.to_numeric(raw["OS.time"], errors="coerce")).abs() > 30).to_numpy()
    flags["dfi_after_os"] = (pd.to_numeric(raw["DFI.time"], errors="coerce") > pd.to_numeric(raw["OS.time"], errors="coerce") + 30).to_numpy()
    flags["absolute_purity"] = raw["ABSOLUTE_Purity"].to_numpy()
    flags["low_purity"] = (raw["ABSOLUTE_Purity"] < 0.2).to_numpy()
    notes = [
        f"Pathology (updated): {raw['Pathology_Updated'].value_counts(dropna=False).to_dict()}",
        f"ABSOLUTE purity: median {raw['ABSOLUTE_Purity'].median():.2f}, < 0.2 in {int((raw['ABSOLUTE_Purity'] < 0.2).sum())} samples, missing {int(raw['ABSOLUTE_Purity'].isna().sum())}",
        f"Tissue source sites: {raw['tissue_source_site'].nunique()} (largest {raw['tissue_source_site'].value_counts().head(3).to_dict()})",
        f"Other malignancy (other_dx): {raw['other_dx'].value_counts(dropna=False).to_dict()}",
    ]
    return flags, notes


def sample_flags(root: Path, reference: Path, cohort: str, frame: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    flags = pd.DataFrame({"cohort": cohort, "exclusion": table["exclusion"].reindex(frame.index).fillna("no clinical row")}, index=frame.index)
    flags["eligible"] = flags["exclusion"].eq("")
    correlation, fence = array_outliers(frame)
    flags["array_r"] = correlation.round(3)
    flags["array_outlier"] = correlation < fence
    call, score = predicted_sex(frame)
    flags["sex_annotated"] = table["sex"].reindex(frame.index)
    flags["sex_predicted"] = call
    flags["sex_score"] = score.round(2)
    flags["sex_mismatch"] = flags["sex_annotated"].notna() & flags["sex_predicted"].isin(["F", "M"]) & (flags["sex_annotated"] != flags["sex_predicted"])
    flags["age"] = table["age"].reindex(frame.index).round(1)
    flags["age_out_of_range"] = (flags["age"] < 18) | (flags["age"] > 100)
    flags["relapse_after_last_os"] = table["rfs_time"].reindex(frame.index) > table["os_time"].reindex(frame.index) + 30 / YEAR
    header = pd.read_csv(root / cohort / "clinical.csv", nrows=0).columns
    patient_column = next((column for column in ("_PATIENT", "patient_id:ch1") if column in header), None)
    if patient_column:
        patients = pd.read_csv(root / cohort / "clinical.csv", usecols=["sample_id", patient_column], low_memory=False).set_index("sample_id")[patient_column].reindex(frame.index)
        flags["repeated_patient"] = patients.duplicated(keep=False) & patients.notna()
    notes: list[str] = []
    if cohort.startswith("TCGA-"):
        extra, notes = tcga_flags(root / cohort, reference, frame.index)
        flags = flags.join(extra)
    identifiers = {
        "cohort": cohort,
        "expression samples": len(frame),
        "duplicated IDs": int(frame.index.duplicated().sum() + table.index.duplicated().sum()),
        "expression without clinical": int((~frame.index.isin(table.index)).sum()),
        "clinical without expression": int((~table.index.isin(frame.index)).sum()),
        "patient column": patient_column or "—",
        "samples of repeated patients": int(flags["repeated_patient"].sum()) if patient_column else 0,
        "array r fence": round(fence, 3),
        "sex genes": ",".join(gene for gene in (*FEMALE_GENES, *MALE_GENES) if gene in frame),
    }
    return flags, identifiers, notes


def main() -> None:
    data_dir, cancer = Path(sys.argv[1]).expanduser(), sys.argv[2]
    root = data_dir / "cohorts" / cancer
    out = root / "qc"
    out.mkdir(exist_ok=True)
    clinical = pd.read_csv(root / "harmonized_clinical.csv", low_memory=False)
    exclusion = clinical["exclusion"].fillna("")  # empty strings come back as missing
    # The QC judges the samples as the harmonization left them: exclusions an earlier QC added are set aside.
    clinical["exclusion"] = exclusion.where(~exclusion.str.startswith("QC: "), "")
    cohorts = sorted(clinical["cohort"].unique())
    expression = {cohort: load_genes(root / cohort) for cohort in cohorts}
    report = [f"# Quality control: {cancer.upper()} cohorts", ""]

    sample_rows, id_rows = [], []
    for cohort in cohorts:
        table = clinical[clinical["cohort"] == cohort].set_index("sample_id")
        flags, identifiers, notes = sample_flags(root, data_dir / "reference", cohort, expression[cohort], table)
        sample_rows.append(flags)
        id_rows.append(identifiers)
        if notes:
            report += [f"## {cohort} notes", ""] + [f"- {note}" for note in notes] + [""]
    samples = pd.concat(sample_rows)
    samples.index.name = "sample_id"
    for column in FLAGS:
        samples[column] = samples[column].fillna(False).astype(bool) if column in samples else False

    pair_tables, summaries = [], []
    for name_a, name_b in itertools.combinations_with_replacement(cohorts, 2):
        table, summary = duplicate_pairs(name_a, expression[name_a], name_b, expression[name_b])
        pair_tables.append(table)
        summaries.append(summary)
        print(summary, flush=True)
    pairs = pd.concat(pair_tables, ignore_index=True)
    covariates = clinical.set_index(["cohort", "sample_id"])[["age", "sex", "stage_group", "exclusion"]]
    for side in ("a", "b"):
        values = covariates.reindex(pd.MultiIndex.from_arrays([pairs[f"cohort_{side}"], pairs[f"sample_{side}"]]))
        for column in values.columns:
            pairs[f"{column}_{side}"] = values[column].to_numpy()
        pairs[f"exclusion_{side}"] = pairs[f"exclusion_{side}"].fillna("no clinical row")
    pairs["age_difference"] = (pairs["age_a"] - pairs["age_b"]).abs().round(1)
    pairs["sex_agrees"] = np.where(pairs["sex_a"].isna() | pairs["sex_b"].isna(), None, pairs["sex_a"] == pairs["sex_b"])
    pairs["stage_agrees"] = np.where(pairs["stage_group_a"].isna() | pairs["stage_group_b"].isna(), None, pairs["stage_group_a"] == pairs["stage_group_b"])
    pairs["verdict"] = pairs.apply(verdict, axis=1)
    duplicates = pairs[pairs["verdict"] == "technical duplicate"]
    members = set(zip(duplicates["cohort_a"], duplicates["sample_a"])) | set(zip(duplicates["cohort_b"], duplicates["sample_b"]))
    samples["technical_duplicate"] = [(cohort, sample) in members for cohort, sample in zip(samples["cohort"], samples.index)]

    samples.to_csv(out / "sample_flags.csv", lineterminator="\n")
    pairs.to_csv(out / "duplicate_pairs.csv", index=False, lineterminator="\n")
    labels = lambda row, names: "; ".join(label for column, label in names.items() if row[column])  # noqa: E731
    decisions = pd.DataFrame(
        {
            "cohort": samples["cohort"],
            "qc_exclusion": samples.apply(labels, axis=1, names=EXCLUDE),
            "qc_sensitivity": samples.apply(labels, axis=1, names=SENSITIVITY),
        }
    )
    decisions = decisions[(decisions["qc_exclusion"] != "") | (decisions["qc_sensitivity"] != "")]
    decisions.to_csv(out / "sample_decisions.csv", lineterminator="\n")

    fenced = lambda text: ["```", text, "```", ""]  # noqa: E731
    listed = lambda frame: fenced(frame.to_string()) if len(frame) else ["none", ""]  # noqa: E731
    flagged = pairs[pairs["flagged"]]
    shown = ["cohort_a", "sample_a", "cohort_b", "sample_b", "r", "robust_z", "gap_to_next", "age_difference", "sex_agrees", "stage_agrees", "exclusion_a", "exclusion_b", "verdict"]
    report += ["## Identifiers", ""] + fenced(pd.DataFrame(id_rows).to_string(index=False))
    report += ["## Sample flags (all samples / eligible samples)", ""]
    counts = samples.groupby("cohort")[list(FLAGS)].sum().astype(int).T
    eligible_counts = samples[samples["eligible"]].groupby("cohort")[list(FLAGS)].sum().astype(int).T
    report += fenced((counts.astype(str) + " / " + eligible_counts.reindex_like(counts).fillna(0).astype(int).astype(str)).to_string())
    report += ["## Sex check (annotated vs expression-predicted)", ""]
    report += fenced(pd.crosstab([samples["cohort"], samples["sex_annotated"].fillna("missing")], samples["sex_predicted"].fillna("no genes")).to_string())
    report += ["Mismatches:", ""] + listed(samples.loc[samples["sex_mismatch"], ["cohort", "exclusion", "sex_annotated", "sex_predicted", "sex_score", "age"]])
    report += ["## Array outliers (correlation with the cohort median profile below Q1 - 3 IQR)", ""]
    report += listed(samples.loc[samples["array_outlier"], ["cohort", "exclusion", "array_r"]])
    report += ["## Ages outside 18-100", ""] + listed(samples.loc[samples["age_out_of_range"], ["cohort", "exclusion", "age"]])
    report += ["## Duplicate screen", ""] + fenced(pd.DataFrame(summaries).to_string(index=False))
    report += ["### Flagged pairs by verdict", ""] + fenced(flagged["verdict"].value_counts().to_string())
    report += ["### Flagged pairs", ""] + (fenced(flagged[shown].to_string(index=False)) if len(flagged) else ["none", ""])
    tcga = samples[samples["cohort"].str.startswith("TCGA-") & samples[list(TCGA_FLAGS)].any(axis=1)]
    if len(tcga):
        columns = [column for column in (*TCGA_FLAGS, "exclusion", "absolute_purity", "rna_cluster", "tcga_notes") if column in tcga]
        listing = tcga[columns].copy()
        for column in TCGA_FLAGS:
            listing[column] = np.where(listing[column].to_numpy(dtype=bool), "x", "")
        report += ["## TCGA samples with a flag", ""] + fenced(listing.to_string())
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()


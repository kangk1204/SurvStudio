"""Duplicate audit for the lung adenocarcinoma case studies (I and II): could a TCGA-LUAD patient reappear in a GEO
validation cohort, or a GEO patient in two cohorts?

TCGA-LUAD (RNA-seq, the 484 development patients) and the seven GEO cohorts (arrays, QC-passed samples), compared
pair by pair with the LUAD QC's duplicate screen (common.matched_pairs): mutual best matches over the 5,000
most variable shared genes, flagged when they stand at least 0.2 above either sample's next-best match. As in the
breast audit (script 11), a flagged pair is confirmed as one patient only when its clinical records do not
contradict it: ages within 1 year, and the same sex and stage (a value missing on either side contradicts nothing).
Positive control: the two technical-duplicate pairs the QC removed from GSE50081 must be flagged when put back. Their
clinical records disagree (the QC found one's annotated sex contradicting its expression), so they test the
expression screen, not the clinical check.
Writes luad_duplicate_audit.csv (every mutual best match with its gap and clinical agreement) and
luad_duplicate_audit_summary.json.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import DUPLICATE_GAP, GEO_COHORTS, LUAD, RESULTS, XENA_EXPRESSION, matched_pairs, write_csv_atomic, write_json

AGE_TOLERANCE = 1.0
KNOWN = {("GSM1213837", "GSM1213843"), ("GSM1213842", "GSM1213849")}


def complete(values: pd.DataFrame) -> pd.DataFrame:
    values = values.astype(float)
    values = values.loc[:, values.notna().mean() > 0.9]
    return values.fillna(values.median())


def clinical_records(tcga: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Age, sex and stage of every sample the audit compares, coded alike: TCGA-LUAD's from SurvStudio's bundled table
    (by patient), the GEO cohorts' from the harmonised table (sex M/F, stage I to IV)."""
    tcga = tcga.set_index("patient_id")[["age", "sex", "stage_group"]]
    geo = truth[truth["cohort"] != "TCGA-LUAD"].set_index("sample_id")[["age", "sex", "stage_group"]].assign(
        sex=lambda table: table["sex"].map({"M": "Male", "F": "Female"}),
        stage_group=lambda table: table["stage_group"].map(lambda value: f"Stage {value}" if isinstance(value, str) else np.nan))
    records = pd.concat([tcga, geo])
    records.index = records.index.astype(str)
    if records.index.duplicated().any():
        raise RuntimeError(f"Samples with two clinical records: {sorted(set(records.index[records.index.duplicated()]))[:5]}")
    return records


def clinical_agreement(pairs: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """For each pair (sample_a, sample_b): both ages, whether they lie within AGE_TOLERANCE years, whether sex and stage
    agree (None where either is missing), and whether nothing recorded contradicts one patient."""
    a = records.reindex(pairs["sample_a"].astype(str)).reset_index(drop=True)
    b = records.reindex(pairs["sample_b"].astype(str)).reset_index(drop=True)

    def compare(first: pd.Series, second: pd.Series, same) -> pd.Series:
        known = first.notna() & second.notna()
        return pd.Series([bool(same(x, y)) if k else None for x, y, k in zip(first, second, known)], dtype=object)

    table = pd.DataFrame({
        "age_a": a["age"], "age_b": b["age"],
        "age_within_1": compare(a["age"], b["age"], lambda x, y: abs(float(x) - float(y)) <= AGE_TOLERANCE),
        "sex_equal": compare(a["sex"], b["sex"], lambda x, y: x == y),
        "stage_equal": compare(a["stage_group"], b["stage_group"], lambda x, y: x == y),
    })
    not_false = lambda values: values.map(lambda value: value is not False)  # noqa: E731
    table["clinically_consistent"] = not_false(table["age_within_1"]) & not_false(table["sex_equal"]) & not_false(table["stage_equal"])
    return table.set_axis(pairs.index)


def main() -> None:
    from survival_toolkit.marker_matrix import matrix_frame, read_marker_matrix
    from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset

    clinical = load_tcga_luad_upload_ready_dataset()
    matrix = read_marker_matrix(XENA_EXPRESSION, XENA_EXPRESSION.name, patient_ids=clinical["patient_id"].tolist())
    tcga = matrix_frame(clinical, matrix, id_column="patient_id", columns=["patient_id", "os_months", "os_event"])
    tcga = tcga.dropna(subset=["os_months", "os_event"])
    cohorts = {"TCGA-LUAD": complete(tcga.set_index("patient_id")[list(matrix.marker_names)])}
    truth = pd.read_csv(LUAD / "harmonized_clinical.csv")
    records = clinical_records(clinical, truth)
    raw = {}
    for cohort in GEO_COHORTS:
        raw[cohort] = pd.read_csv(LUAD / cohort / "expression_genes.csv.gz").set_index("sample_id")
        keep = set(truth.loc[(truth["cohort"] == cohort) & truth["exclusion"].isna(), "sample_id"])
        cohorts[cohort] = complete(raw[cohort].loc[raw[cohort].index.intersection(list(keep))])

    # Positive control: GSE50081 with every sample, the QC's technical duplicates included.
    control, _ = matched_pairs(complete(raw["GSE50081"]), complete(raw["GSE50081"]), same=True)
    control["known"] = [tuple(sorted(pair)) in KNOWN for pair in zip(control["sample_a"], control["sample_b"])]
    control = pd.concat([control, clinical_agreement(control, records)], axis=1)
    control_found = int((control["known"] & (control["gap"] >= DUPLICATE_GAP)).sum())
    print(control.sort_values("gap", ascending=False).head(4).round(3).to_string(index=False))
    print(f"positive control: {control_found} of {len(KNOWN)} known duplicate pairs flagged", flush=True)

    matches, pairs = [], []
    for first, second in itertools.combinations_with_replacement(cohorts, 2):
        found, stats = matched_pairs(cohorts[first], cohorts[second], same=first == second)
        found.insert(0, "cohort_b", second)
        found.insert(0, "cohort_a", first)
        matches.append(found)
        flagged = found[found["gap"] >= DUPLICATE_GAP]
        pairs.append({"cohorts": f"{first}~{second}", **stats, "mutual_best": int(len(found)), "max_gap": float(found["gap"].max()) if len(found) else None,
                      "flagged": int(len(flagged))})
        print(f"{first} ~ {second}: max r {stats['max_r']:.3f}, largest gap {pairs[-1]['max_gap']:.3f}, flagged {len(flagged)}", flush=True)
    table = pd.concat(matches, ignore_index=True)
    table["flagged"] = table["gap"] >= DUPLICATE_GAP
    table = pd.concat([table, clinical_agreement(table, records)], axis=1)
    table["confirmed"] = table["flagged"] & table["clinically_consistent"]
    write_csv_atomic(table, RESULTS / "luad_duplicate_audit.csv")
    known = control[control["known"]]
    write_json(RESULTS / "luad_duplicate_audit_summary.json", {
        "gap": DUPLICATE_GAP, "age_tolerance": AGE_TOLERANCE, "positive_control_found": control_found, "positive_control_known": len(KNOWN),
        "positive_control_clinically_consistent": int(known["clinically_consistent"].sum()),
        "pairs": pairs, "flagged": int(table["flagged"].sum()), "confirmed": int(table["confirmed"].sum()),
        "flagged_pairs": table[table["flagged"]].to_dict("records"),
    })
    print(f"{int(table['flagged'].sum())} flagged pairs, {int(table['confirmed'].sum())} with clinical records that agree")
    print(table[table["flagged"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

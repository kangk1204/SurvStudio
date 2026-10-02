"""Duplicate audit for the breast cancer case studies (IV and V): the same patient within a cohort, in two external
cohorts, or in METABRIC and an external cohort.

Every tumour of METABRIC and of each cohort that entered a validation is compared with every other (and within
its cohort), without deduplication:
1. identifiers: the MetaGxBreast curators' annotations (doppelgangR), and Uppsala/Karolinska sample codes such
   as 197B95 that recur across cohorts (TRANSBIG_VDXKIU_197B95 and UPP_197B95);
2. expression: mutual best matches over the 5,000 most variable shared genes (common.matched_pairs), with their
   gap to the next-best match;
3. clinical record: age, tumour size, node status and ER status of the two samples. Tumour sizes are compared only
   between two cohorts that record the same kind: STNO2 and VDX record T categories (1 to 4), the others
   centimetres, and a T category is no size in centimetres.
A pair is a duplicate when it is annotated; when it shares a sample code; when it stands 0.2 above the next-best
match and its ages (within 1 year) and ER status agree; or when it is a mutual best match whose age (within 0.5
years), tumour size, node status and ER status all agree and whose expression match is not negligible (gap at
least 0.05, or the two cohorts already share 10 or more duplicates by the other rules, as CAL and UCSF do).
Clinical identity alone with no expression signal (UPP_139B03 and UPP_84A44, r 0.19) is chance. Between METABRIC
(development) and a validation cohort the bar is lower, because a shared patient there leaks into the validation:
a gap of 0.15 with agreeing age and ER status suffices (two TRANSBIG samples from Guy's Hospital, which also
supplied METABRIC, stand at 0.185 and 0.188; the next METABRIC match, with CAL, stands at 0.103). The confirmed pairs
go to results/breast_duplicate_pairs_recomputed.csv; every mutual best match goes to
results/duplicate_audit.csv with its evidence. The committed analysis input,
paper/breast_duplicate_pairs.csv, is preserved. Compare identities and decisions before
reviewing any deliberate input update; floating-point bytes can differ across environments.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from common import BREAST, DUPLICATE_GAP, RESULTS, load_breast_cohort, matched_pairs, write_csv_atomic, write_json

COHORTS = ["METABRIC", "CAL", "GSE58644", "NKI", "STNO2", "TRANSBIG", "UCSF", "UPP", "VDX"]
DEVELOPMENT_GAP = 0.15
SAMPLE_CODE = re.compile(r"(\d+[A-Z]\d+)$")

duplicates = pd.read_csv(BREAST.parent / "breast_duplicates.csv")
annotated: dict[str, set[str]] = {}
for sample, partners in zip(duplicates["sample"].astype(str), duplicates["duplicates"].astype(str)):
    for partner in partners.split(";"):
        annotated.setdefault(sample, set()).add(partner)
        annotated.setdefault(partner, set()).add(sample)

loaded = {name: load_breast_cohort(name, endpoint=None, dedupe=False) for name in COHORTS}
tables = {name: frame.set_index("patient_id")[genes] for name, (frame, genes) in loaded.items()}
clinical = {name: frame.set_index("patient_id")[["age", "tumor_size", "node_positive", "grade", "er"]] for name, (frame, _) in loaded.items()}
# The kind of size each cohort records: T categories (STNO2, VDX), size classes (none audited here) or centimetres.
size_kind = {name: "T category" if frame.attrs["tumour_size_from_t_category"] else "classes" if frame.attrs["tumour_size_classes"] else "cm"
             for name, (frame, _) in loaded.items()}


def agree(a: float, b: float, tolerance: float) -> bool | None:
    if pd.isna(a) or pd.isna(b):
        return None
    return abs(float(a) - float(b)) <= tolerance


rows, pairs = [], []
for first, second in itertools.combinations_with_replacement(COHORTS, 2):
    found, stats = matched_pairs(tables[first], tables[second], same=first == second)
    # Sizes of two kinds (a T category and centimetres) are not compared: neither agreement nor disagreement.
    sizes_comparable = size_kind[first] == size_kind[second] != "classes"
    for match in found.itertuples(index=False):
        a, b = clinical[first].loc[match.sample_a], clinical[second].loc[match.sample_b]
        rows.append({"cohort_a": first, "sample_a": match.sample_a, "cohort_b": second, "sample_b": match.sample_b,
                     "r": match.r, "gap": match.gap, "age_a": a["age"], "age_b": b["age"],
                     "age_within_0_5": agree(a["age"], b["age"], 0.5), "age_within_1": agree(a["age"], b["age"], 1.0),
                     "size_equal": agree(a["tumor_size"], b["tumor_size"], 0.05) if sizes_comparable else None,
                     "nodes_equal": agree(a["node_positive"], b["node_positive"], 0.0),
                     "er_equal": None if pd.isna(a["er"]) or pd.isna(b["er"]) else a["er"] == b["er"]})
    pairs.append({"cohorts": f"{first}~{second}", **stats, "mutual_best": int(len(found)), "gap_at_least_0_2": int((found["gap"] >= DUPLICATE_GAP).sum())})
    print(f"{first} ~ {second}: max r {stats['max_r']:.3f}, mutual best {len(found)}, gap >= {DUPLICATE_GAP}: {pairs[-1]['gap_at_least_0_2']}", flush=True)
audit = pd.DataFrame(rows)
audit["annotated"] = [b in annotated.get(a, set()) for a, b in zip(audit["sample_a"], audit["sample_b"])]
not_false = lambda values: values.fillna(True).astype(bool)  # noqa: E731
development = (audit["cohort_a"] == "METABRIC") != (audit["cohort_b"] == "METABRIC")
threshold = np.where(development, DEVELOPMENT_GAP, DUPLICATE_GAP)
audit["by_gap"] = (audit["gap"] >= threshold) & not_false(audit["age_within_1"]) & not_false(audit["er_equal"])
clinically_identical = (
    audit["age_within_0_5"].fillna(False).astype(bool) & not_false(audit["size_equal"]) & not_false(audit["nodes_equal"]) & not_false(audit["er_equal"])
    & (audit["size_equal"].notna() | audit["nodes_equal"].notna())
)

# Shared Uppsala/Karolinska sample codes, whether or not the expression screen paired them.
codes: dict[str, list[tuple[str, str]]] = {}
for name, table in tables.items():
    for sample in table.index:
        match = SAMPLE_CODE.search(str(sample))
        if match:
            codes.setdefault(match.group(1), []).append((name, str(sample)))
coded = [(x, y) for members in codes.values() for x, y in itertools.combinations(members, 2)]
coded = pd.DataFrame([{"cohort_a": a[0], "sample_a": a[1], "cohort_b": b[0], "sample_b": b[1]} for a, b in coded])

# Cohorts that share patients by the other rules; there a clinically identical best match is accepted at any gap.
other_rules = audit["annotated"] | audit["by_gap"]
shared = audit[other_rules].groupby(["cohort_a", "cohort_b"]).size()
if len(coded):
    shared = shared.add(coded.groupby(["cohort_a", "cohort_b"]).size(), fill_value=0)
overlapping = {pair for pair, count in shared.items() if count >= 10}
audit["cohorts_overlap"] = [(a, b) in overlapping or (b, a) in overlapping for a, b in zip(audit["cohort_a"], audit["cohort_b"])]
audit["by_clinical_identity"] = clinically_identical & ((audit["gap"] >= 0.05) | audit["cohorts_overlap"])

confirmed = [audit.loc[audit["annotated"] | audit["by_gap"] | audit["by_clinical_identity"],
                       ["cohort_a", "sample_a", "cohort_b", "sample_b", "r", "gap", "annotated", "by_gap", "by_clinical_identity"]]]
if len(coded):
    confirmed.append(coded.assign(by_sample_code=True))
confirmed = pd.concat(confirmed, ignore_index=True)
confirmed["by_sample_code"] = confirmed.get("by_sample_code", False)
for column in ("annotated", "by_gap", "by_clinical_identity", "by_sample_code"):
    confirmed[column] = confirmed[column].fillna(False).astype(bool)
confirmed["key"] = [tuple(sorted((a, b))) for a, b in zip(confirmed["sample_a"], confirmed["sample_b"])]
confirmed = confirmed.groupby("key", as_index=False).agg({
    "cohort_a": "first", "sample_a": "first", "cohort_b": "first", "sample_b": "first", "r": "max", "gap": "max",
    "annotated": "max", "by_gap": "max", "by_clinical_identity": "max", "by_sample_code": "max"}).drop(columns="key")
write_csv_atomic(confirmed, RESULTS / "breast_duplicate_pairs_recomputed.csv")
write_csv_atomic(audit, RESULTS / "duplicate_audit.csv")

by_cohorts = confirmed.groupby(["cohort_a", "cohort_b"]).size().rename("pairs").reset_index()
write_json(RESULTS / "duplicate_audit_summary.json", {
    "gap": DUPLICATE_GAP, "cohorts": COHORTS, "size_kind": size_kind, "pairs": pairs,
    "analysis_input_updated": False, "recomputed_pairs": "breast_duplicate_pairs_recomputed.csv",
    "confirmed": int(len(confirmed)),
    "confirmed_by": {column: int(confirmed[column].sum()) for column in ("annotated", "by_gap", "by_clinical_identity", "by_sample_code")},
    "confirmed_by_cohorts": by_cohorts.to_dict("records"),
    "annotated_pairs_found_by_expression": int((audit["annotated"] & (audit["by_gap"] | audit["by_clinical_identity"])).sum()),
    "annotated_pairs_matched": int(audit["annotated"].sum()),
})
print(by_cohorts.to_string(index=False))
print(f"{len(confirmed)} confirmed duplicate pairs:", {column: int(confirmed[column].sum()) for column in ("annotated", "by_gap", "by_clinical_identity", "by_sample_code")})
print("annotated pairs that are mutual best matches:", int(audit["annotated"].sum()), "; of these found by the expression rules:",
      int((audit["annotated"] & (audit["by_gap"] | audit["by_clinical_identity"])).sum()))

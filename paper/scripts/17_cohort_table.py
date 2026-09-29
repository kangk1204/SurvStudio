"""The cohort table: the development cohorts of case studies I, IV and V and every cohort screened for the external
validations of case studies II (GEO lung adenocarcinoma), IV (breast cancer, overall survival) and V (ER-positive
breast cancer, recurrence), one row each with the case study, cancer, platform, source, endpoint, role, the patients
and events screened (complete cases) and used, the median follow-up of the patients used (reverse Kaplan-Meier, in
months as analysed: breast follow-up is censored at 10 years, so 120 means at least 10 years), whether the cohort was
used and, when it was not, why.

The patients used are rebuilt as the analyses build them (common.development_data for the development cohorts,
common.geo_cohort for the GEO cohorts, and the breast screen common.screen_validation_cohorts with the locked models)
and their counts checked against the results of scripts 01, 03, 07, 08, 09 and 10; the breast screens' reasons are
the ones recorded under screened in breast_external_pooled.json and breast_er_external_pooled.json. Platforms: the GEO
series' platforms (GPL), and for the MetaGxBreast datasets, which record none, the platform their probe identifiers and
source series identify (ExperimentHub records EH1076 to EH1114).
Writes cohort_table.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BREAST_VALIDATION,
    DEVELOPMENT,
    ENDPOINT_COLUMNS,
    GEO_COHORTS,
    RESULTS,
    development_data,
    geo_cohort,
    read_result,
    screen_validation_cohorts,
    write_csv_atomic,
)

LUAD_PLATFORMS = {
    "TCGA-LUAD": ("RNA-seq, Illumina HiSeq (UCSC Xena HiSeqV2)", "TCGA"),
    "GSE13213": ("GPL6480, Agilent Whole Human Genome 4x44K", "GSE13213"),
    "GSE30219": ("GPL570, Affymetrix HG-U133 Plus 2.0", "GSE30219"),
    "GSE31210": ("GPL570, Affymetrix HG-U133 Plus 2.0", "GSE31210"),
    "GSE41271": ("GPL6884, Illumina HumanWG-6 v3.0", "GSE41271"),
    "GSE50081": ("GPL570, Affymetrix HG-U133 Plus 2.0", "GSE50081"),
    "GSE68465": ("GPL96, Affymetrix HG-U133A", "GSE68465"),
    "GSE72094": ("GPL15048, Rosetta/Merck custom Affymetrix (HuRSTA)", "GSE72094"),
}
U133A, U133_PLUS, U133AB = "Affymetrix HG-U133A", "Affymetrix HG-U133 Plus 2.0", "Affymetrix HG-U133A and HG-U133B"
BREAST_PLATFORMS = {
    "CAL": (U133A, "E-TABM-158"), "DFHCC": (U133_PLUS, "GSE19615"), "DFHCC2": (U133_PLUS, "GSE18864"), "DFHCC3": (U133_PLUS, "GSE3744"),
    "DUKE": ("Affymetrix HG-U95Av2", "GSE3143"), "DUKE2": ("Affymetrix U133 X3P", "GSE6861"), "EMC2": (U133_PLUS, "GSE12276"),
    "EORTC10994": (U133A, "GSE1561"), "EXPO": (U133_PLUS, "GSE2109"), "FNCLCC": ("custom array", "GSE7017"), "GSE25066": (U133A, "GSE25066"),
    "GSE32646": (U133_PLUS, "GSE32646"), "GSE48091": ("custom Affymetrix array", "GSE48091"),
    "GSE58644": ("Affymetrix Human Gene 1.0 ST", "GSE58644"), "HLP": ("Illumina", "E-TABM-543"), "IRB": (U133_PLUS, "GSE5460"),
    "KOO": ("280 genes of Affymetrix probes", "not available"), "LUND": ("oligonucleotide array (H200 probes)", "GSE31863"),
    "LUND2": ("cDNA array", "GSE5325"), "MAINZ": (U133A, "GSE11121"), "MAQC2": (U133A, "GSE20194"), "MCCC": ("custom array", "GSE19177"),
    "MDA4": (U133A, "MD Anderson public data"), "METABRIC": ("Illumina HumanHT-12 v3", "EGAS00000000083"), "MSK": (U133A, "GSE2603"),
    "MUG": ("oligonucleotide array (H200 probes)", "GSE10510"), "NCCS": (U133A, "GSE5364"), "NCI": ("cDNA array", "original publication"),
    "NKI": ("Agilent Hu25K (Rosetta)", "not available"), "PNC": (U133_PLUS, "GSE20711"), "STK": (U133AB, "GSE1456"),
    "STNO2": ("Stanford cDNA array", "Stanford Microarray Database"), "TCGA": ("gene level (TCGA)", "TCGA"), "TRANSBIG": (U133A, "GSE7390"),
    "UCSF": ("custom array", "not available"), "UNC4": ("Agilent (UNC)", "GSE18229"), "UNT": (U133AB, "GSE2990"), "UPP": (U133AB, "GSE3494"),
    "VDX": (U133A, "GSE2034, GSE5327"),
}
ENDPOINTS = {"os": "overall survival", "rfs": "relapse-free survival", "dmfs": "distant metastasis-free survival", None: "not recorded"}
VALIDATION = {"IV": ("breast_locked_model.json", "breast_external_pooled.json", "breast_external_validation.csv"),
              "V": ("breast_er_locked_model.json", "breast_er_external_pooled.json", "breast_er_external_validation.csv")}
SUMMARIES = {"I": "tcga_markers_summary.json", "IV": "breast_markers_summary.json", "V": "breast_er_markers_summary.json"}


def median_follow_up(time: np.ndarray, event: np.ndarray) -> float | None:
    """Median follow-up by the reverse Kaplan-Meier method: the median of the Kaplan-Meier estimate in which a
    censoring is the event and an event censors; None when that estimate stays above one half."""
    time = np.asarray(time, dtype=float)
    censored = np.asarray(event, dtype=float) == 0
    survival = 1.0
    for value in np.unique(time[censored]):
        survival *= 1.0 - float(np.sum(censored & (time == value))) / float(np.sum(time >= value))
        if survival <= 0.5:
            return float(value)
    return None


def checked(label: str, found: tuple[int, int], expected: tuple[int, int]) -> None:
    if tuple(int(value) for value in found) != tuple(int(value) for value in expected):
        raise SystemExit(f"{label}: {found[0]} patients and {found[1]} events rebuilt here, {expected[0]} and {expected[1]} in its results")


def development_rows() -> list[dict]:
    rows = []
    for case, cohort in (("I", "TCGA-LUAD"), ("IV", "METABRIC"), ("V", "METABRIC")):
        spec = DEVELOPMENT[case]
        frame, _ = development_data(case)
        frame = frame.dropna(subset=[spec["time"], spec["event"], *spec["covariates"]])
        frame = frame[pd.to_numeric(frame[spec["time"]], errors="coerce") > 0]
        summary = read_result(SUMMARIES[case])["cohort"]
        found = (len(frame), int(frame[spec["event"]].sum()))
        checked(f"{cohort} (case study {case})", found, (summary["n"], summary["events"]))
        platform, source = LUAD_PLATFORMS[cohort] if case == "I" else BREAST_PLATFORMS[cohort]
        endpoint = "overall survival" if case in ("I", "IV") else "relapse-free survival"
        rows.append({"case_study": case, "cancer": "lung adenocarcinoma" if case == "I" else "breast", "cohort": cohort + (" ER-positive" if case == "V" else ""),
                     "platform": platform, "source": source, "endpoint": endpoint, "role": "development", "patients_screened": found[0],
                     "events_screened": found[1], "patients_used": found[0], "events_used": found[1],
                     "median_follow_up_months": median_follow_up(frame[spec["time"]], frame[spec["event"]]), "used": True, "reason": None})
    return rows


def geo_rows() -> list[dict]:
    validation = read_result("external_validation.csv")
    validation = validation[validation["scaling"] == "within_cohort"].set_index("cohort")
    rows = []
    for cohort in GEO_COHORTS:
        frame = geo_cohort(cohort).dropna(subset=["os_months", "os_event", "age", "sex", "stage_group"])
        frame = frame[frame["os_months"] > 0]
        found = (len(frame), int(frame["os_event"].sum()))
        checked(cohort, found, (validation.at[cohort, "n"], validation.at[cohort, "events"]))
        platform, source = LUAD_PLATFORMS[cohort]
        rows.append({"case_study": "II", "cancer": "lung adenocarcinoma", "cohort": cohort, "platform": platform, "source": source,
                     "endpoint": "overall survival", "role": "external validation", "patients_screened": found[0], "events_screened": found[1],
                     "patients_used": found[0], "events_used": found[1], "median_follow_up_months": median_follow_up(frame["os_months"], frame["os_event"]),
                     "used": True, "reason": None})
    return rows


def breast_rows(case: str) -> list[dict]:
    locked, pooled, validation = VALIDATION[case]
    recorded = {entry["cohort"]: entry for entry in read_result(pooled)["screened"]}
    counts = read_result(validation).set_index("cohort")
    screened, used = screen_validation_cohorts(case, read_result(locked))
    time_column, event_column = ENDPOINT_COLUMNS[BREAST_VALIDATION[case]["endpoint"]]
    rows = []
    for entry in screened:
        cohort = entry["cohort"]
        if {key: value for key, value in entry.items() if key != "marker_weight_measured"} != {
                key: value for key, value in recorded[cohort].items() if key != "marker_weight_measured"}:
            raise SystemExit(f"case study {case}, {cohort}: the screen differs from the one recorded in {pooled}: {entry} against {recorded[cohort]}")
        frame = used.get(cohort)
        follow_up = None
        if frame is not None:
            checked(f"{cohort} (case study {case})", (len(frame), int(frame[event_column].sum())), (counts.at[cohort, "n"], counts.at[cohort, "events"]))
            follow_up = median_follow_up(frame[time_column], frame[event_column])
        platform, source = BREAST_PLATFORMS[cohort]
        rows.append({"case_study": case, "cancer": "breast", "cohort": cohort, "platform": platform, "source": source,
                     "endpoint": ENDPOINTS[entry["endpoint"]], "role": "external validation", "patients_screened": entry.get("complete_cases"),
                     "events_screened": entry.get("events"), "patients_used": len(frame) if frame is not None else 0,
                     "events_used": int(frame[event_column].sum()) if frame is not None else 0, "median_follow_up_months": follow_up,
                     "used": bool(entry["used"]), "reason": entry.get("reason")})
    return rows


def main() -> None:
    rows = development_rows() + geo_rows() + breast_rows("IV") + breast_rows("V")
    table = pd.DataFrame(rows)
    counts = ["patients_screened", "events_screened", "patients_used", "events_used"]
    table[counts] = table[counts].astype("Int64")  # whole numbers, empty where a cohort was not screened that far
    write_csv_atomic(table, RESULTS / "cohort_table.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(table.drop(columns=["source"]).to_string(index=False))


if __name__ == "__main__":
    main()

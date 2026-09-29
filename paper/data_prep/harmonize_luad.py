"""Harmonize the clinical annotations of the lung adenocarcinoma cohorts.

    python data_prep/harmonize_luad.py <data_dir>

Reads <data_dir>/cohorts/luad/<cohort>/clinical.csv (written by export_geo_luad.R and
export_tcga_luad.py) and the QC decisions, <data_dir>/cohorts/luad/qc/sample_decisions.csv
(prepare_data.sh copies the ones the paper applied there from data_snapshot/; qc_cohorts.py
writes them), and never looks at expression values. The QC's exclusions (technical
duplicates, sex mismatches, TCGA quality annotations) are applied as "QC: ..." exclusion
reasons and its sensitivity flags fill the qc_sensitivity column; a missing decisions file
stops the run. Outputs, under <data_dir>/cohorts/luad/:

- harmonized_clinical.csv: the harmonized table the analyses read;
- eligibility.csv: sample and event counts per endpoint, follow-up, and covariate
  missingness for eligible samples;
- harmonization_manifest.json: the SHA-256 of the table and of the QC decisions applied.

Harmonized fields: cohort, sample_id, os_time and rfs_time (years), os_event and
rfs_event (1 event, 0 censored), age (years), sex (F/M), stage_group (I-IV),
smoking (ever/never) and exclusion (why a sample is left out, empty if eligible).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
HORIZON_YEARS = 5.0


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _label(series: pd.Series) -> pd.Series:
    """Lower-cased, stripped text with missing values kept missing."""
    return series.astype("string").str.strip().str.lower()


def _binary(series: pd.Series, positive: set[str], negative: set[str]) -> pd.Series:
    """1.0 / 0.0 for the listed labels (case-insensitive), NaN for anything else."""
    labels = _label(series)
    return pd.Series(np.where(labels.isin(positive), 1.0, np.where(labels.isin(negative), 0.0, np.nan)), index=series.index)


def _category(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return _label(series).map(mapping).astype(object).where(lambda values: values.notna(), None)


def _sex(series: pd.Series) -> pd.Series:
    return _category(series, {"f": "F", "female": "F", "m": "M", "male": "M"})


def _roman_stage(value: object) -> str | None:
    """I-IV from labels such as IA, Stage IIB, 1A, 2b or 3; None for missing or ambiguous labels."""
    if pd.isna(value):
        return None
    text = re.sub(r"^STAGE\s*", "", str(value).strip().upper())
    if " VS " in f" {text} ":
        return None
    match = re.match(r"^(IV|III|II|I)(?![IV])", text)
    if match:
        return match.group(1)
    match = re.match(r"^([1-4])", text)
    if match:
        return ["I", "II", "III", "IV"][int(match.group(1)) - 1]
    return None


def _tnm_stage(t: object, n: object, m: object) -> str | None:
    """Stage group from pathological T, N and M following AJCC 7 (T2a/T2b not distinguished,
    so T2N0 is grouped as I)."""
    def number(value: object) -> int | None:
        match = re.fullmatch(r"[TNM]([0-4])[A-C]?", str(value).strip().upper()) if not pd.isna(value) else None
        return int(match.group(1)) if match else None

    t_value, n_value, m_value = number(t), number(n), number(m)
    if m_value == 1:
        return "IV"
    if t_value is None or n_value is None:
        return None
    if n_value >= 2 or t_value == 4 or (t_value == 3 and n_value == 1):
        return "III"
    if n_value == 1 or t_value == 3:
        return "II"
    return "I"


def _years_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (pd.to_datetime(end, format="%Y-%m-%d", errors="coerce") - pd.to_datetime(start, format="%Y-%m-%d", errors="coerce")).dt.days / DAYS_PER_YEAR


def _frame(data_dir: Path, accession: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / "cohorts" / "luad" / accession / "clinical.csv", low_memory=False)


def gse13213(data_dir: Path) -> pd.DataFrame:
    """Tomida et al. 2009, Aichi; all adenocarcinoma. Relapse has no date, so no RFS."""
    raw = _frame(data_dir, "GSE13213")
    brinkman = _numeric(raw["Smoking (BI):ch1"])
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["Survival (days):ch1"]) / DAYS_PER_YEAR,
            "os_event": _binary(raw["Status:ch1"], {"dead"}, {"alive"}),
            "rfs_time": np.nan,
            "rfs_event": np.nan,
            "age": _numeric(raw["Age:ch1"]),
            "sex": _sex(raw["Sex:ch1"]),
            "stage_group": raw["Stage (Pathological ):ch1"].map(_roman_stage),
            "smoking": np.where(brinkman > 0, "ever", np.where(brinkman == 0, "never", None)),
            "exclusion": "",
        }
    )


def gse30219(data_dir: Path) -> pd.DataFrame:
    """Rousseaux et al. 2013; mixed histology plus non-tumoural lung (NTL)."""
    raw = _frame(data_dir, "GSE30219")
    histology = raw["histology:ch1"].astype("string").str.strip().str.upper()
    exclusion = np.where(histology == "NTL", "not tumour", np.where(histology == "ADC", "", "not adenocarcinoma"))
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["follow-up time (months):ch1"]) / 12.0,
            "os_event": _binary(raw["status:ch1"], {"dead"}, {"alive"}),
            "rfs_time": _numeric(raw["disease free survival in months:ch1"]) / 12.0,
            "rfs_event": _binary(raw["relapse (event=1; no event=0):ch1"], {"1"}, {"0"}),
            "age": _numeric(raw["age at surgery:ch1"]),
            "sex": _sex(raw["gender:ch1"]),
            "stage_group": [_tnm_stage(t, n, m) for t, n, m in zip(raw["pt stage:ch1"], raw["pn stage:ch1"], raw["pm stage:ch1"])],
            "smoking": None,
            "exclusion": exclusion,
        }
    )


def gse31210(data_dir: Path) -> pd.DataFrame:
    """Okayama et al. 2012, NCC Tokyo; stage I-II adenocarcinoma plus normal lung. The
    series flags incomplete resection or adjuvant therapy for exclusion from prognosis."""
    raw = _frame(data_dir, "GSE31210")
    flag = next(column for column in raw.columns if column.startswith("exclude for prognosis analysis"))
    tumour = _label(raw["tissue:ch1"]).eq("primary lung tumor").fillna(False)
    excluded = _label(raw[flag]).eq("exclude").fillna(False)
    exclusion = np.where(~tumour, "not tumour", np.where(excluded, "incomplete resection or adjuvant therapy", ""))
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["days before death/censor:ch1"]) / DAYS_PER_YEAR,
            "os_event": _binary(raw["death:ch1"], {"dead"}, {"alive"}),
            "rfs_time": _numeric(raw["days before relapse/censor:ch1"]) / DAYS_PER_YEAR,
            "rfs_event": _binary(raw["relapse:ch1"], {"relapsed"}, {"not relapsed"}),
            "age": _numeric(raw["age (years):ch1"]),
            "sex": _sex(raw["gender:ch1"]),
            "stage_group": raw["pathological stage:ch1"].map(_roman_stage),
            "smoking": _category(raw["smoking status:ch1"], {"ever-smoker": "ever", "never-smoker": "never"}),
            "exclusion": exclusion,
        }
    )


def gse41271(data_dir: Path) -> pd.DataFrame:
    """Sato et al. 2013, UT Southwestern; follow-up given as ISO dates."""
    raw = _frame(data_dir, "GSE41271")
    exclusion = np.where(_label(raw["histology:ch1"]).eq("adenocarcinoma").fillna(False), "", "not adenocarcinoma")
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _years_between(raw["date of surgery:ch1"], raw["last follow-up survival:ch1"]),
            "os_event": _binary(raw["vital statistics:ch1"], {"d"}, {"a"}),
            "rfs_time": _years_between(raw["date of surgery:ch1"], raw["last follow-up recurrence:ch1"]),
            "rfs_event": _binary(raw["recurrence:ch1"], {"y"}, {"n"}),
            "age": _years_between(raw["date of birth:ch1"], raw["date of surgery:ch1"]),
            "sex": _sex(raw["gender:ch1"]),
            "stage_group": raw["final patient stage:ch1"].map(_roman_stage),
            "smoking": _category(raw["tobacco history:ch1"], {"y": "ever", "n": "never"}),
            "exclusion": exclusion,
        }
    )


def gse50081(data_dir: Path) -> pd.DataFrame:
    """Der et al. 2014, UHN Toronto; times already in years."""
    raw = _frame(data_dir, "GSE50081")
    exclusion = np.where(_label(raw["histology:ch1"]).eq("adenocarcinoma").fillna(False), "", "not adenocarcinoma")
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["survival time:ch1"]),
            "os_event": _binary(raw["status:ch1"], {"dead"}, {"alive"}),
            "rfs_time": _numeric(raw["disease-free survival time:ch1"]),
            "rfs_event": _binary(raw["recurrence:ch1"], {"y"}, {"n"}),
            "age": _numeric(raw["age:ch1"]),
            "sex": _sex(raw["Sex:ch1"]),
            "stage_group": raw["Stage:ch1"].map(_roman_stage),
            "smoking": _category(raw["smoking:ch1"], {"current": "ever", "ex-smoker": "ever", "never": "never"}),
            "exclusion": exclusion,
        }
    )


def gse68465(data_dir: Path) -> pd.DataFrame:
    """Shedden et al. 2008 (Director's Challenge); stage recorded as pN?pT?. RFS is censored
    at the last clinical assessment when no progression was recorded."""
    raw = _frame(data_dir, "GSE68465")
    rfs_event = _binary(raw["first_progression_or_relapse:ch1"], {"yes"}, {"no"})
    rfs_months = np.where(rfs_event == 1.0, _numeric(raw["months_to_first_progression:ch1"]), _numeric(raw["mths_to_last_clinical_assessment:ch1"]))
    stage_codes = raw["disease_stage:ch1"].astype("string").str.extract(r"^pN(?P<n>\w)pT(?P<t>\w)$")
    exclusion = np.where(_label(raw["disease_state:ch1"]).eq("lung adenocarcinoma").fillna(False), "", "not tumour")
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["months_to_last_contact_or_death:ch1"]) / 12.0,
            "os_event": _binary(raw["vital_status:ch1"], {"dead"}, {"alive"}),
            "rfs_time": np.where(np.isnan(rfs_event), np.nan, rfs_months / 12.0),
            "rfs_event": rfs_event,
            "age": _numeric(raw["age:ch1"]),
            "sex": _sex(raw["Sex:ch1"]),
            "stage_group": [_tnm_stage(f"T{t}", f"N{n}", "M0") for t, n in zip(stage_codes["t"], stage_codes["n"])],
            "smoking": _category(raw["smoking_history:ch1"], {"smoked in the past": "ever", "currently smoking": "ever", "never smoked": "never"}),
            "exclusion": exclusion,
        }
    )


def gse72094(data_dir: Path) -> pd.DataFrame:
    """Schabath et al. 2016, Moffitt; all adenocarcinoma, no recurrence data."""
    raw = _frame(data_dir, "GSE72094")
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["survival_time_in_days:ch1"]) / DAYS_PER_YEAR,
            "os_event": _binary(raw["vital_status:ch1"], {"dead"}, {"alive"}),
            "rfs_time": np.nan,
            "rfs_event": np.nan,
            "age": _numeric(raw["age_at_diagnosis:ch1"]),
            "sex": _sex(raw["gender:ch1"]),
            "stage_group": raw["Stage:ch1"].map(_roman_stage),
            "smoking": _category(raw["smoking_status:ch1"], {"ever": "ever", "never": "never"}),
            "exclusion": "",
        }
    )


def tcga_luad(data_dir: Path) -> pd.DataFrame:
    """TCGA-LUAD primary tumours (export_tcga_luad.py) with TCGA-CDR endpoints; the
    disease-free interval (DFI) serves as RFS, as recommended for resected LUAD."""
    raw = _frame(data_dir, "TCGA-LUAD")
    ffpe = _label(raw["sample_type"]).ne("primary tumor").fillna(True)
    neoadjuvant = _label(raw["history_of_neoadjuvant_treatment"]).eq("yes").fillna(False)
    redacted = raw["Redaction"].notna()
    exclusion = np.where(ffpe, "not a frozen primary tumour", np.where(redacted, "redacted", np.where(neoadjuvant, "neoadjuvant therapy", "")))
    tobacco = _numeric(raw["tobacco_smoking_history"])
    os_event, rfs_event = _numeric(raw["OS"]), _numeric(raw["DFI"])
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"],
            "os_time": _numeric(raw["OS.time"]) / DAYS_PER_YEAR,
            "os_event": os_event.where(os_event.isin([0, 1])),
            "rfs_time": _numeric(raw["DFI.time"]) / DAYS_PER_YEAR,
            "rfs_event": rfs_event.where(rfs_event.isin([0, 1])),
            "age": _numeric(raw["age_at_initial_pathologic_diagnosis"]),
            "sex": _sex(raw["gender"]),
            "stage_group": raw["pathologic_stage"].map(_roman_stage),
            "smoking": np.where(tobacco == 1, "never", np.where(tobacco.isin([2, 3, 4, 5]), "ever", None)),
            "exclusion": exclusion,
        }
    )


HARMONIZERS = {
    "GSE13213": gse13213,
    "GSE30219": gse30219,
    "GSE31210": gse31210,
    "GSE41271": gse41271,
    "GSE50081": gse50081,
    "GSE68465": gse68465,
    "GSE72094": gse72094,
    "TCGA-LUAD": tcga_luad,
}


def decisions_file(data_dir: Path) -> Path:
    return data_dir / "cohorts" / "luad" / "qc" / "sample_decisions.csv"


def harmonize(data_dir: Path) -> pd.DataFrame:
    frames = []
    for cohort, harmonizer in HARMONIZERS.items():
        frame = harmonizer(data_dir)
        frame.insert(0, "cohort", cohort)
        frames.append(frame)
    harmonized = pd.concat(frames, ignore_index=True)
    for endpoint in ("os", "rfs"):
        invalid = ~(harmonized[f"{endpoint}_time"] > 0) | harmonized[f"{endpoint}_event"].isna()
        harmonized.loc[invalid, [f"{endpoint}_time", f"{endpoint}_event"]] = np.nan
    return apply_qc(harmonized, decisions_file(data_dir))


def apply_qc(harmonized: pd.DataFrame, decisions_path: Path) -> pd.DataFrame:
    """Fold the QC decisions (qc_cohorts.py) in: QC exclusions become the exclusion reason of
    otherwise eligible samples, and sensitivity flags go to qc_sensitivity. Without the
    decisions file the run stops, rather than keep samples the QC excludes."""
    if not decisions_path.exists():
        raise SystemExit(f"{decisions_path} is missing: copy data_snapshot/sample_decisions.csv there "
                         "(prepare_data.sh does), or write it with qc_cohorts.py.")
    harmonized["qc_sensitivity"] = ""
    decisions = pd.read_csv(decisions_path, keep_default_na=False).set_index(["cohort", "sample_id"])
    keys = pd.MultiIndex.from_frame(harmonized[["cohort", "sample_id"]])
    exclusion = decisions["qc_exclusion"].reindex(keys).fillna("").to_numpy()
    newly_excluded = (harmonized["exclusion"] == "").to_numpy() & (exclusion != "")
    harmonized.loc[newly_excluded, "exclusion"] = "QC: " + exclusion[newly_excluded]
    harmonized["qc_sensitivity"] = decisions["qc_sensitivity"].reindex(keys).fillna("").to_numpy()
    return harmonized


def eligibility(harmonized: pd.DataFrame, horizon: float = HORIZON_YEARS) -> pd.DataFrame:
    rows = []
    for cohort, group in harmonized.groupby("cohort", sort=True):
        usable = group[group["exclusion"] == ""]
        row: dict[str, object] = {
            "cohort": cohort,
            "samples": len(group),
            "qc_excluded": int(group["exclusion"].str.startswith("QC: ").sum()),
            "eligible": len(usable),
        }
        for endpoint in ("os", "rfs"):
            time, event = usable[f"{endpoint}_time"], usable[f"{endpoint}_event"]
            valid = time.notna()
            row[f"{endpoint}_n"] = int(valid.sum())
            row[f"{endpoint}_events"] = int(event[valid].sum())
            row[f"{endpoint}_events_{horizon:g}y"] = int(((event == 1) & (time <= horizon)).sum())
            row[f"{endpoint}_median_censored_y"] = round(float(time[event == 0].median()), 1) if (event == 0).any() else np.nan
        for covariate in ("age", "sex", "stage_group", "smoking"):
            row[f"missing_{covariate}"] = round(float(usable[covariate].isna().mean()), 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    events = table[f"os_events_{horizon:g}y"]
    core_missing = table[["missing_age", "missing_sex", "missing_stage_group"]].max(axis=1)
    table["os_discovery_eligible"] = (table["os_n"] >= 150) & (events >= 60) & (core_missing <= 0.2)
    table["os_replication_eligible"] = (events >= 30) & (core_missing <= 0.2)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    luad = data_dir / "cohorts" / "luad"

    harmonized = harmonize(data_dir)
    table_csv = harmonized.to_csv(index=False, lineterminator="\n")
    (luad / "harmonized_clinical.csv").write_text(table_csv, encoding="utf-8", newline="\n")

    table = eligibility(harmonized)
    table.to_csv(luad / "eligibility.csv", index=False, lineterminator="\n")
    manifest = {
        "horizon_years": HORIZON_YEARS,
        "table_sha256": hashlib.sha256(table_csv.encode("utf-8")).hexdigest(),
        "qc_decisions_sha256": hashlib.sha256(decisions_file(data_dir).read_bytes()).hexdigest(),
        "rows": len(harmonized),
        "cohorts": list(HARMONIZERS),
    }
    (luad / "harmonization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.to_string(index=False))
    print(f"\nharmonized_clinical.csv: {manifest['rows']} rows, SHA-256 {manifest['table_sha256']}")


if __name__ == "__main__":
    main()

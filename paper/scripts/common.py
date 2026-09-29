"""Paths and helpers shared by the software-paper scripts.

Run the analysis scripts through ``paper/run_all.sh`` or ``paper/run_step.sh``, which put SurvStudio's source on
the path (``PYTHONPATH=<repository>/src``). The data live in ``paper/data`` (written by ``paper/prepare_data.sh``),
or wherever the ``SURVSTUDIO_DATA`` environment variable points.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
PAPER = SCRIPTS.parent
DATA = Path(os.environ.get("SURVSTUDIO_DATA") or PAPER / "data")
LUAD = DATA / "cohorts" / "luad"
XENA_EXPRESSION = LUAD / "TCGA-LUAD" / "raw" / "HiSeqV2.gz"
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
COVARIATES = ["age", "sex", "stage_group"]
CATEGORICAL = ["sex", "stage_group"]
GEO_COHORTS = ["GSE13213", "GSE30219", "GSE31210", "GSE41271", "GSE50081", "GSE68465", "GSE72094"]
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Breast cancer (case studies IV and V): MetaGxBreast cohorts exported by data_prep/export_curated_cohorts.R.
BREAST = DATA / "cohorts" / "breast"
# Every file the breast analyses read, with its size and SHA-256 (breast_manifest.py writes it, 00_self_check.py
# checks the data against it).
BREAST_MANIFEST = PAPER / "breast_data_manifest.csv"
BREAST_COVARIATES = ["age", "tumor_size", "node_positive", "grade", "er"]
BREAST_CATEGORICAL = ["grade", "er"]
# Case study V: ER-positive tumours only, so ER status drops out of the clinical model.
BREAST_ER_COVARIATES = ["age", "tumor_size", "node_positive", "grade"]
BREAST_ER_CATEGORICAL = ["grade"]
BREAST_HORIZON_MONTHS = 120.0
DAYS_PER_MONTH = 365.25 / 12
# A day column holding multiples of 30 only stores months x 30 days and is converted back with 30 days a month
# (days_per_month): in the current export DFHCC's and MAINZ's distant metastasis-free survival and UNC4's overall and
# relapse-free survival. Every other day column is converted with 365.25 / 12 days a month (STNO2 stores whole months
# x 30.44 days, which that restores).
RECURRENCE_COLUMNS = {"rfs": ("days_to_tumor_recurrence", "recurrence_status"), "dmfs": ("dmfs_days", "dmfs_status")}
# Cohorts whose tumor_size holds size classes, neither centimetres nor T categories: UNC4 records 1, 3 and 6 (and one
# 1.5) for its sized tumours. Their tumour size is treated as not recorded.
SIZE_CLASSES = {"UNC4": "size classes (1, 3, 6), not centimetres"}
# Both METABRIC endpoints, overall survival (case study IV) and relapse-free survival (case study V), and the METABRIC
# site come from cBioPortal's brca_metabric patient table. MetaGxBreast's METABRIC days_to_death holds an older
# follow-up: as days / 30 it matches cBioPortal's OS_MONTHS to within 0.05 months for 1,106 of 1,978 tumours, and 249
# tumours it records as living are deceased in cBioPortal. The copy the paper used is pinned by its SHA-256, patient
# count and columns, and released in data_snapshot/ (see data_snapshot/SNAPSHOT.md).
CBIOPORTAL_METABRIC = "https://www.cbioportal.org/api/studies/brca_metabric/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY"
CBIOPORTAL_METABRIC_SHA256 = "f09389e31e88b658db463eb33e716fa7c0c905c4f2527c76cc87ccb5e02c0bee"
CBIOPORTAL_METABRIC_PATIENTS = 2509
CBIOPORTAL_METABRIC_COLUMNS = ["patientId", "COHORT", "OS_MONTHS", "OS_STATUS", "RFS_MONTHS", "RFS_STATUS"]
CBIOPORTAL_SNAPSHOT = PAPER / "data_snapshot" / "cbioportal_patients.csv"
# Endpoint: cBioPortal's months and status columns, and its status labels as event indicators.
CBIOPORTAL_ENDPOINTS = {
    "os": ("OS_MONTHS", "OS_STATUS", {"1:DECEASED": 1, "0:LIVING": 0}),
    "recurrence": ("RFS_MONTHS", "RFS_STATUS", {"1:Recurred": 1, "0:Not Recurred": 0}),
}
ENDPOINT_COLUMNS = {"os": ("os_months", "os_event"), "recurrence": ("recurrence_months", "recurrence_event")}
# Stamps of the simulation replicates; hexadecimal digests such as "0123456" or "12e4567" must be read as text.
STAMP_COLUMNS = ["survstudio_commit", "design_hash", "code_hash"]
STAMP_DTYPES = {column: str for column in STAMP_COLUMNS}

# External validation of case studies IV and V (scripts 08, 10 and 15; see screen_validation_cohorts).
BREAST_VALIDATION = {
    "IV": {"endpoint": "os", "er_positive": False, "events": "deaths"},
    "V": {"endpoint": "recurrence", "er_positive": True, "events": "events"},
}
MIN_VALIDATION_EVENTS = 30
# SurvStudio's MIN_MARKER_WEIGHT_AVAILABLE: validate_locked_recipe declines a cohort that measures less than half of
# the locked model's marker weight (00_self_check.py checks that the two agree).
MIN_MARKER_WEIGHT = 0.5
# Cohorts sampled on the outcome are left out: EMC2 holds patients selected for metastatic relapse.
OUTCOME_SELECTED = {"EMC2": "patients selected for relapse"}
# Bootstrap resamples behind every interval of the external validations (validate_locked_recipe's n_bootstrap) and
# of Uno's C, drawn from validate_locked_recipe's default seed, so Uno's C is resampled on the same patients.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260926
# Case study II, sensitivity: Uno's C truncated at 5 years.
UNO_HORIZON_MONTHS = 60.0

# The development runs of case studies I, IV and V (scripts 01, 07 and 09; 16 reruns them with other seeds): the
# endpoint's columns and the clinical covariates the markers are judged against.
DEVELOPMENT = {
    "I": {"label": "TCGA-LUAD, overall survival", "time": "os_months", "event": "os_event", "covariates": COVARIATES,
          "categorical": CATEGORICAL},
    "IV": {"label": "METABRIC, overall survival", "time": "os_months", "event": "os_event", "covariates": BREAST_COVARIATES,
           "categorical": BREAST_CATEGORICAL},
    "V": {"label": "METABRIC ER-positive, recurrence", "time": "recurrence_months", "event": "recurrence_event",
          "covariates": BREAST_ER_COVARIATES, "categorical": BREAST_ER_CATEGORICAL},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file and a rename, so an interrupted write never leaves a truncated file; a
    file of results/ is stamped (stamp_result)."""
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    _stamp_if_result(path)


def metabric_cbioportal() -> pd.DataFrame:
    """METABRIC patient data from cBioPortal (overall and relapse-free survival and the site among them), downloaded
    once and checked against the copy the paper used on every read. When the download fails or differs from that copy
    (cBioPortal updates brca_metabric from time to time), the copy in data_snapshot/ is used instead, and a download
    that differs is left next to it as ``cbioportal_patients.csv.download``. A cached table that differs stops the
    run."""
    path = BREAST / "METABRIC" / "cbioportal_patients.csv"
    if not path.exists():
        download = path.with_name(path.name + ".download")
        try:
            _download_cbioportal(download)
            _check_cbioportal_file(download)
            os.replace(download, path)
        except (OSError, ValueError, KeyError, RuntimeError) as problem:
            if not CBIOPORTAL_SNAPSHOT.exists():
                raise
            print(f"cBioPortal METABRIC table: {problem}\nUsing the copy the paper used, {CBIOPORTAL_SNAPSHOT}.", file=sys.stderr)
            _check_cbioportal_file(CBIOPORTAL_SNAPSHOT)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CBIOPORTAL_SNAPSHOT, path)
    _check_cbioportal_file(path)
    return pd.read_csv(path)


def _download_cbioportal(target: Path) -> None:
    """cBioPortal's brca_metabric patient table, one row per patient, written to ``target``."""
    import urllib.request

    with urllib.request.urlopen(CBIOPORTAL_METABRIC, timeout=180) as response:
        records = json.load(response)
    table = pd.DataFrame(records).pivot(index="patientId", columns="clinicalAttributeId", values="value").reset_index()
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)


def _check_cbioportal_file(path: Path) -> None:
    table = pd.read_csv(path)
    missing = [column for column in CBIOPORTAL_METABRIC_COLUMNS if column not in table.columns]
    if missing or len(table) != CBIOPORTAL_METABRIC_PATIENTS or table["patientId"].duplicated().any():
        raise RuntimeError(f"{path}: expected {CBIOPORTAL_METABRIC_PATIENTS} METABRIC patients, one row each, with the columns "
                           f"{CBIOPORTAL_METABRIC_COLUMNS}; found {len(table)} rows and lacking {missing}.")
    digest = sha256_file(path)
    if digest != CBIOPORTAL_METABRIC_SHA256:
        raise RuntimeError(f"{path} is not the cBioPortal METABRIC table the paper used (SHA-256 {digest}, expected "
                           f"{CBIOPORTAL_METABRIC_SHA256}); cBioPortal may have changed brca_metabric since.")


def metabric_patient(samples: pd.Series) -> pd.Series:
    """cBioPortal's patient ID of each METABRIC sample, exactly: sample MB_0001 is patient MB-0001; a sample named
    otherwise has none (NaN)."""
    samples = samples.astype(str)
    return samples.where(samples.str.fullmatch(r"MB_\d+")).str.replace("_", "-", n=1, regex=False)


def metabric_sites(samples: pd.Series) -> np.ndarray:
    """The METABRIC site (cBioPortal COHORT) of each sample, in the order given; NaN where cBioPortal records none."""
    site = metabric_cbioportal().set_index("patientId")["COHORT"]
    return pd.to_numeric(metabric_patient(samples).map(site), errors="coerce").to_numpy(dtype=float)


DUPLICATE_GAP = 0.2


def matched_pairs(first: pd.DataFrame, second: pd.DataFrame, *, same: bool, top_genes: int = 5000) -> tuple[pd.DataFrame, dict[str, float]]:
    """The duplicate screen of the LUAD QC (data_prep/qc_cohorts.py) for two expression tables (samples x genes, as
    measured): each gene is z-scored within its cohort over the most variable shared genes, samples are
    compared by Pearson correlation, and every mutual best match is returned with its correlation and its gap to the
    next-best match of either sample. The same tumour profiled twice, even on two platforms, stands clear of its
    next-best match; a gap of at least DUPLICATE_GAP (0.2) flags it.
    """
    common = first.columns.intersection(second.columns)
    rank = first[common].var().rank(ascending=False) + second[common].var().rank(ascending=False)
    genes = rank.nsmallest(min(top_genes, len(common))).index

    def rows(table: pd.DataFrame) -> np.ndarray:
        values = table[genes].astype(float)
        values = ((values - values.mean()) / values.std().replace(0, np.nan)).fillna(0.0).to_numpy()
        values = values - values.mean(axis=1, keepdims=True)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    correlation = rows(first) @ rows(second).T
    if same:
        np.fill_diagonal(correlation, -np.inf)
    best_in_second = correlation.argmax(axis=1)
    best_in_first = correlation.argmax(axis=0)
    records = []
    for i, j in enumerate(best_in_second):
        if best_in_first[j] != i or (same and i > j):
            continue
        row, column = np.delete(correlation[i], j), np.delete(correlation[:, j], i)
        records.append({"sample_a": first.index[i], "sample_b": second.index[j], "r": float(correlation[i, j]),
                        "gap": float(correlation[i, j] - max(row.max(), column.max()))})
    finite = correlation[np.isfinite(correlation)]
    return pd.DataFrame(records, columns=["sample_a", "sample_b", "r", "gap"]), {
        "genes": int(len(genes)), "median_r": float(np.median(finite)), "max_r": float(finite.max())}


@functools.lru_cache(maxsize=None)
def _shared_patient_ids(breast: Path) -> tuple[tuple[str, str], ...]:
    """Pairs of samples of one cohort that MetaGxBreast gives the same unique_patient_ID (in the current export
    DFHCC2's replicate arrays, such as DFHCC2_REF1 and DFHCC2_REF1rep, and MDA4_M323 and MDA4_M323_bis). IDs are
    compared within a cohort only: two cohorts may number their patients alike."""
    pairs = []
    for folder in sorted(breast.iterdir()):
        path = folder / "clinical.csv"
        if not path.exists() or "unique_patient_ID" not in pd.read_csv(path, nrows=0).columns:
            continue
        table = pd.read_csv(path, usecols=["sample_id", "unique_patient_ID"], dtype=str).dropna()
        for _, samples in table.groupby("unique_patient_ID")["sample_id"]:
            pairs += [(samples.iloc[0], other) for other in samples.iloc[1:]]
    return tuple(pairs)


def breast_duplicates() -> dict[str, set[str]]:
    """Each sample's duplicates, in both directions: the MetaGxBreast curators' annotations (doppelgangR), samples of
    a cohort with the same unique_patient_ID, and the pairs the paper's expression audit confirmed
    (paper/breast_duplicate_pairs.csv)."""
    pairs: dict[str, set[str]] = {}

    def add(first: str, second: str) -> None:
        if first and second and first != second:
            pairs.setdefault(first, set()).add(second)
            pairs.setdefault(second, set()).add(first)

    annotated = pd.read_csv(BREAST.parent / "breast_duplicates.csv", dtype=str, keep_default_na=False)
    for sample, partners in zip(annotated["sample"], annotated["duplicates"]):
        for partner in partners.split(";"):
            add(sample, partner)
    for first, second in _shared_patient_ids(BREAST):
        add(first, second)
    audited = PAPER / "breast_duplicate_pairs.csv"
    if audited.exists():
        for first, second in pd.read_csv(audited, dtype=str, keep_default_na=False)[["sample_a", "sample_b"]].itertuples(index=False):
            add(first, second)
    return pairs


# Duplicate chains are not followed through KOO's samples: KOO was profiled on a 280-gene panel, too few genes to
# single out a tumour (the curators list one KOO sample as the duplicate of 38 others), and through them 135 listed
# samples from at least 11 cohorts would join into one "patient". KOO records no follow-up, so no analysis uses
# KOO's own samples.
NOT_FOLLOWED_THROUGH = ("KOO_",)


def breast_patient_keys() -> dict[str, str]:
    """One key per patient for every sample in a duplicate pair: its connected component in the graph of duplicate
    pairs (without the pairs of NOT_FOLLOWED_THROUGH samples), named by the component's first sample in sort order.
    The lists are not transitive (A~B and B~C listed, A~C not), so a sample counts as the same patient as every sample
    it is linked to through any chain, within and across cohorts. A sample in no pair is its own key
    (``keys.get(sample, sample)``)."""
    pairs = breast_duplicates()
    parent = {sample: sample for sample in pairs}

    def root(sample: str) -> str:
        while parent[sample] != sample:
            parent[sample] = parent[parent[sample]]
            sample = parent[sample]
        return sample

    for sample, partners in pairs.items():
        for partner in partners:
            if sample.startswith(NOT_FOLLOWED_THROUGH) or partner.startswith(NOT_FOLLOWED_THROUGH):
                continue
            first, second = root(sample), root(partner)
            if first != second:
                # The smaller root absorbs the larger, so every root stays its component's first sample.
                parent[max(first, second)] = min(first, second)
    return {sample: root(sample) for sample in pairs}


def _tumours(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["sample_type"].fillna("tumor") == "tumor"] if "sample_type" in table else table


def recurrence_endpoint(name: str, table: pd.DataFrame | None = None) -> str | None:
    """The recurrence endpoint a cohort is analysed on: relapse-free survival where recorded, else distant
    metastasis-free survival (METABRIC: relapse-free survival from cBioPortal)."""
    if name == "METABRIC":
        return "rfs"
    clinical = pd.read_csv(BREAST / name / "clinical.csv", low_memory=False) if table is None else table
    for label, (days, status) in RECURRENCE_COLUMNS.items():
        if days in clinical and status in clinical and (clinical[days].notna() & clinical[status].notna()).any():
            return label
    return None


def overall_survival_recorded(table: pd.DataFrame) -> bool:
    if "days_to_death" not in table or "vital_status" not in table:
        return False
    return bool((pd.to_numeric(table["days_to_death"], errors="coerce").notna() & table["vital_status"].notna()).any())


def days_per_month(name: str, column: str, table: pd.DataFrame) -> float:
    """Days per month of follow-up for a cohort's day column (the whole clinical table): 30 where MetaGxBreast stored
    months x 30 days (at least 10 values, all multiples of 30), else 365.25 / 12. ``name`` names the cohort in the
    call only."""
    observed = pd.to_numeric(table[column], errors="coerce").dropna().to_numpy(dtype=float)
    if observed.size >= 10 and bool(np.all(observed % 30 == 0)):
        return 30.0
    return DAYS_PER_MONTH


@functools.lru_cache(maxsize=None)
def _t_category_medians(breast: Path) -> tuple[float, float, float]:
    table = pd.read_csv(breast / "METABRIC" / "clinical.csv", usecols=["sample_type", "tumor_size"], low_memory=False)
    size = pd.to_numeric(_tumours(table)["tumor_size"], errors="coerce").dropna()
    return tuple(float(size[mask].median()) for mask in (size <= 2, (size > 2) & (size <= 5), size > 5))


def t_category_cm() -> dict[float, float]:
    """Tumour size (cm) standing in for a T category: the median METABRIC tumour of the matching size range, T1 at
    most 2 cm, T2 over 2 and up to 5 cm, T3 and T4 over 5 cm (computed once per run from METABRIC's tumours; 1.6, 2.9
    and 6.2 cm in the current export)."""
    t1, t2, t3 = _t_category_medians(BREAST)
    return {1.0: t1, 2.0: t2, 3.0: t3, 4.0: t3}


def tumour_size_cm(size: pd.Series) -> tuple[pd.Series, bool]:
    """A cohort's tumour sizes (all its tumours) in centimetres, and whether they were T categories: a cohort whose
    recorded sizes are all whole numbers from 1 to 4 recorded the T category (STNO2, VDX and EORTC10994 in the current
    export), which ``t_category_cm`` maps to centimetres."""
    size = pd.to_numeric(size, errors="coerce").astype(float)
    observed = size.dropna()
    if observed.size and bool(np.all(observed == observed.round())) and bool(observed.between(1, 4).all()):
        return size.map(t_category_cm()), True
    return size, False


def load_breast_cohort(
    name: str, genes: list[str] | None = None, *, endpoint: str | None = "os", er_positive: bool = False, dedupe: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """One MetaGxBreast cohort as SurvStudio input, censored at 10 years: overall survival (``endpoint="os"``) or
    recurrence (``"recurrence"``, see ``recurrence_endpoint``), the clinical covariates, and one expression column
    per gene (the most variable probe when a gene has several).

    Tumours only, ER-positive ones when ``er_positive``, with follow-up and every covariate recorded, and one sample
    per patient (``dedupe``: the first sample of each ``breast_patient_keys`` component; the duplicate audits turn it
    off). Follow-up is converted to months by ``days_per_month`` and tumour size to centimetres by
    ``tumour_size_cm`` (not recorded for SIZE_CLASSES cohorts). METABRIC's follow-up for both endpoints comes from
    cBioPortal (CBIOPORTAL_ENDPOINTS), matched by ``metabric_patient``; its tumours without a cBioPortal record, or
    whose record lacks the endpoint, are left out and counted (attrs ``cbioportal_no_record`` and
    ``cbioportal_endpoint_missing``). A cohort without complete cases gives a table with every column and no rows.
    ``endpoint=None`` keeps every tumour, whatever its follow-up or covariates, with tumour size as recorded (for the
    duplicate audits; attrs ``tumour_size_from_t_category`` tells whether the cohort records T categories).
    ``genes`` limits the expression columns read.
    """
    folder = BREAST / name
    table = pd.read_csv(folder / "clinical.csv", low_memory=False)
    tumours = clinical = _tumours(table)
    if er_positive:
        clinical = clinical[clinical["er"] == "positive"]
    # T categories are recognised over all of the cohort's tumours, whatever subset is analysed.
    if name in SIZE_CLASSES:
        size, from_t_category = pd.Series(np.nan, index=tumours.index), False
    else:
        size, from_t_category = tumour_size_cm(tumours["tumor_size"])
    if endpoint is None:
        frame = pd.DataFrame({
            "patient_id": clinical["sample_id"].astype(str),
            "age": pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce"),
            "tumor_size": pd.to_numeric(clinical["tumor_size"], errors="coerce"),
            "node_positive": pd.to_numeric(clinical["N"], errors="coerce"),
            "grade": pd.to_numeric(clinical["grade"], errors="coerce"),
            "er": clinical["er"],
        }).reset_index(drop=True)
        return _with_expression(frame, folder, genes, {"within_cohort_duplicates_removed": 0, "tumour_size_from_t_category": from_t_category,
                                                       "tumour_size_classes": name in SIZE_CLASSES})
    time_column, event_column = ENDPOINT_COLUMNS[endpoint]
    per_month = None
    counts: dict[str, int] = {}
    if name == "METABRIC":
        months_column, status_column, events = CBIOPORTAL_ENDPOINTS[endpoint]
        patients = metabric_cbioportal().set_index("patientId")
        unknown = set(patients[status_column].dropna()) - set(events)
        if unknown:
            raise RuntimeError(f"cBioPortal {status_column} holds labels this loader does not know: {sorted(unknown)}")
        patient = metabric_patient(clinical["sample_id"])
        recorded = patient.isin(patients.index).to_numpy(dtype=bool)
        months = pd.to_numeric(patient.map(patients[months_column]), errors="coerce")
        happened = patient.map(patients[status_column]).map(events)
        counts = {"cbioportal_no_record": int((~recorded).sum()),
                  "cbioportal_endpoint_missing": int((recorded & (months.isna() | happened.isna()).to_numpy(dtype=bool)).sum())}
    elif endpoint == "os":
        per_month = days_per_month(name, "days_to_death", table)
        months = pd.to_numeric(clinical["days_to_death"], errors="coerce") / per_month
        happened = clinical["vital_status"].map({"deceased": 1, "living": 0})
    else:
        kind = recurrence_endpoint(name, table)
        if kind is None:
            raise ValueError(f"{name} records neither relapse-free nor distant metastasis-free survival.")
        days, status = RECURRENCE_COLUMNS[kind]
        per_month = days_per_month(name, days, table)
        months = pd.to_numeric(clinical[days], errors="coerce") / per_month
        happened = clinical[status].map({"recurrence": 1, "norecurrence": 0})
    covariates = BREAST_ER_COVARIATES if er_positive else BREAST_COVARIATES
    frame = pd.DataFrame({
        "patient_id": clinical["sample_id"].astype(str),
        time_column: months.clip(upper=BREAST_HORIZON_MONTHS),
        event_column: np.where(months > BREAST_HORIZON_MONTHS, 0, happened),
        "age": pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce"),
        "tumor_size": size.loc[clinical.index],
        "node_positive": pd.to_numeric(clinical["N"], errors="coerce"),
        "grade": pd.to_numeric(clinical["grade"], errors="coerce").map({1.0: "G1", 2.0: "G2", 3.0: "G3"}),
        "er": clinical["er"].map({"positive": "ER+", "negative": "ER-"}),
    })
    frame.loc[happened.isna().to_numpy(), event_column] = np.nan
    frame = frame.dropna(subset=[time_column, event_column, *covariates])
    frame = frame.loc[(frame[time_column] > 0).to_numpy(dtype=bool)].reset_index(drop=True)
    # One sample per patient: a sample of a patient already kept in this cohort is left out, so a patient can never
    # sit on both sides of a subsample split. Boolean arrays, not lists: an empty list would select no columns.
    if dedupe:
        keys = breast_patient_keys()
        first = ~frame["patient_id"].map(lambda sample: keys.get(sample, sample)).duplicated().to_numpy(dtype=bool)
    else:
        first = np.ones(len(frame), dtype=bool)
    frame = frame.loc[first].reset_index(drop=True)
    return _with_expression(frame, folder, genes, {
        "within_cohort_duplicates_removed": int((~first).sum()), "tumour_size_from_t_category": from_t_category,
        "tumour_size_classes": name in SIZE_CLASSES, "days_per_month": per_month, **counts})


def probe_gene(column: str, genes: set[str]) -> str:
    """The gene a column of an expression export measures: R's make.unique() named a gene's further probes GENE.1,
    GENE.2, ..., after the padding some exports add to symbols (GSE58644: " RNF207 " and " RNF207 .1", compared
    stripped as "RNF207" and "RNF207 .1"), so a column "<gene>.<number>" whose stem, stripped, is a gene of the export
    (``genes``) is a probe of that gene; any other column is its own gene."""
    stem, _, suffix = column.rpartition(".")
    return stem.strip() if stem and suffix.isdigit() and stem.strip() in genes else column


def _with_expression(frame: pd.DataFrame, folder: Path, genes: list[str] | None, attrs: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Add one expression column per gene (the most variable probe, see ``probe_gene``) to the patients of ``frame``."""
    raw = pd.read_csv(folder / "expression.csv.gz", nrows=0).columns.tolist()[1:]
    # Some exports pad gene symbols with spaces (GSE58644: " FAM87A "); names are compared stripped.
    header = [column.strip() for column in raw]
    if len(set(header)) != len(header):
        raise ValueError(f"{folder.name}: two expression columns have the same name once stripped of spaces.")
    original = dict(zip(header, raw))
    known = set(header)
    base = {column: probe_gene(column, known) for column in header}
    chosen = None if genes is None else set(genes)
    wanted = [column for column in header if chosen is None or base[column] in chosen]
    expression = pd.read_csv(folder / "expression.csv.gz", usecols=["sample_id", *(original[column] for column in wanted)]).set_index("sample_id")
    expression.columns = [column.strip() for column in expression.columns]
    expression.index = expression.index.astype(str)
    expression = expression.loc[expression.index.intersection(frame["patient_id"])]
    variance = expression.var(axis=0).dropna()
    keep = variance.groupby(pd.Series(base)[variance.index]).idxmax()
    collapsed = expression[keep.to_numpy()].set_axis(keep.index.tolist(), axis=1)
    frame = frame.merge(collapsed, left_on="patient_id", right_index=True)
    frame.attrs.update(attrs)
    return frame, sorted(collapsed.columns.tolist())


def tcga_development() -> tuple[pd.DataFrame, list[str], Any]:
    """Case study I's development data as script 01 evaluates it: SurvStudio's bundled TCGA-LUAD clinical table and the
    Xena HiSeqV2.gz matrix read for its patients (read_marker_matrix) and joined to them (matrix_frame), with overall
    survival, age, sex and stage. Returns the frame, the genes of the matrix and the matrix itself (for its id_note)."""
    from survival_toolkit.marker_matrix import matrix_frame, read_marker_matrix
    from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset

    clinical = load_tcga_luad_upload_ready_dataset()
    matrix = read_marker_matrix(XENA_EXPRESSION, XENA_EXPRESSION.name, patient_ids=clinical["patient_id"].tolist())
    frame = matrix_frame(clinical, matrix, id_column="patient_id", columns=["os_months", "os_event", *COVARIATES])
    return frame, list(matrix.marker_names), matrix


def development_data(case: str) -> tuple[pd.DataFrame, list[str]]:
    """The development data of case study I (tcga_development), IV (METABRIC, overall survival) or V (METABRIC's
    ER-positive tumours, recurrence) as scripts 01, 07 and 09 evaluate them: the frame and its genes."""
    if case == "I":
        frame, genes, _ = tcga_development()
        return frame, genes
    if case == "IV":
        return load_breast_cohort("METABRIC")
    if case == "V":
        return load_breast_cohort("METABRIC", endpoint="recurrence", er_positive=True)
    raise ValueError(f"No development data for case study {case!r}.")


def evaluate_development(case: str, frame: pd.DataFrame, genes: list[str], settings: Any = None) -> dict[str, Any]:
    """SurvStudio's marker evaluation of a case study's development data as scripts 01, 07 and 09 run it: the added
    value of each gene over the case study's clinical covariates (DEVELOPMENT), with SurvStudio's defaults unless
    ``settings`` (a MarkerSettings) says otherwise."""
    from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers

    spec = DEVELOPMENT[case]
    return evaluate_markers(
        frame, time_column=spec["time"], event_column=spec["event"], marker_columns=genes, clinical_columns=spec["covariates"],
        categorical_clinical=spec["categorical"], event_positive_value=1, settings=settings or MarkerSettings(),
    )


def geo_cohort(cohort: str, genes: Iterable[str] = ()) -> pd.DataFrame:
    """A GEO cohort of case study II as script 03 scores it: the patients of the harmonized clinical table the QC kept
    (overall survival in months, age, sex and stage) joined to their expression of ``genes`` (those the cohort
    measures); a patient without expression is left out."""
    truth = pd.read_csv(LUAD / "harmonized_clinical.csv")  # written by data_prep/harmonize_luad.py
    table = truth[(truth["cohort"] == cohort) & truth["exclusion"].isna()]
    path = LUAD / cohort / "expression_genes.csv.gz"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    expression = pd.read_csv(path, usecols=["sample_id", *[gene for gene in genes if gene in header]])
    return pd.DataFrame({
        "patient_id": table["sample_id"],
        "os_months": table["os_time"] * 12.0,
        "os_event": table["os_event"],
        "age": table["age"],
        "sex": table["sex"].map({"M": "Male", "F": "Female"}),
        "stage_group": table["stage_group"].map(lambda value: f"Stage {value}" if isinstance(value, str) else np.nan),
    }).merge(expression.rename(columns={"sample_id": "patient_id"}), on="patient_id")


def marker_weight_measured(recipe: dict[str, Any], columns: Iterable[str]) -> tuple[float, list[str]]:
    """The share of a locked model's marker weight (|coefficient| x development SD) that a cohort measures, and the
    markers it lacks, computed as SurvStudio's validate_locked_recipe computes them."""
    present = set(columns)
    markers = list(recipe["markers"])
    absent = [name for name in markers if name not in present]
    coefficient = dict(zip(recipe["model"]["terms"], recipe["model"]["coefficients"]))
    scale = recipe.get("marker_scale") or {}
    weight = {name: abs(float(coefficient[name])) * float((scale.get(name) or {}).get("sd", 1.0) or 1.0) for name in markers}
    total = sum(weight.values())
    share = 1.0 if total <= 0 else sum(value for name, value in weight.items() if name not in absent) / total
    return float(share), absent


def validation_candidates(case: str) -> list[tuple[str, str | None]]:
    """Every exported cohort other than METABRIC, alphabetically, with the endpoint case study IV (overall survival:
    "os") or V ("rfs", else "dmfs") would analyse it on, None where the cohort does not record it."""
    rows = []
    for folder in sorted(BREAST.iterdir()):
        if folder.name == "METABRIC" or not (folder / "clinical.csv").exists():
            continue
        table = pd.read_csv(folder / "clinical.csv", low_memory=False)
        if case == "IV":
            rows.append((folder.name, "os" if overall_survival_recorded(table) else None))
        else:
            rows.append((folder.name, recurrence_endpoint(folder.name, table)))
    return rows


def screen_validation_cohorts(case: str, recipe: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """The external cohorts of case study IV (overall survival) or V (ER-positive tumours, recurrence), screened in
    alphabetical order by the rules fixed before the runs. A cohort is used when it records the endpoint, was not
    sampled on the outcome (OUTCOME_SELECTED), and, among its complete cases and after removing every patient who is,
    through any chain of listed duplicates (breast_patient_keys), a METABRIC development patient or a patient of a
    cohort used before it, keeps at least MIN_VALIDATION_EVENTS events within 10 years and measures at least half of
    the locked model's marker weight (marker_weight_measured). Only a used cohort's patients join the ones later
    cohorts are checked against.

    Returns one entry per cohort with its counts and, when not used, the reason; and each used cohort's patients with
    the recipe's markers, ready for validate_locked_recipe.
    """
    spec = BREAST_VALIDATION[case]
    event_column = ENDPOINT_COLUMNS[spec["endpoint"]][1]
    keys = breast_patient_keys()
    development, _ = load_breast_cohort("METABRIC", genes=[], endpoint=spec["endpoint"], er_positive=spec["er_positive"])
    seen = {keys.get(sample, sample) for sample in development["patient_id"]}
    screened: list[dict[str, Any]] = []
    used: dict[str, pd.DataFrame] = {}
    for cohort, endpoint in validation_candidates(case):
        entry: dict[str, Any] = {"cohort": cohort, "endpoint": endpoint}
        screened.append(entry)
        if endpoint is None:
            entry.update(used=False, reason="no overall survival recorded" if case == "IV" else "no relapse-free or distant metastasis-free survival recorded")
            continue
        if cohort in OUTCOME_SELECTED:
            entry.update(used=False, reason=OUTCOME_SELECTED[cohort])
            continue
        frame, _ = load_breast_cohort(cohort, genes=list(recipe["markers"]), endpoint=spec["endpoint"], er_positive=spec["er_positive"])
        entry.update(complete_cases=len(frame), events=int(frame[event_column].sum()),
                     duplicates_within=int(frame.attrs["within_cohort_duplicates_removed"]))
        if frame.attrs.get("tumour_size_from_t_category"):
            entry["tumour_size"] = "T category, mapped to cm"
        if frame.attrs.get("tumour_size_classes"):
            entry["tumour_size"] = f"{SIZE_CLASSES[cohort]}: treated as not recorded"
        if frame.attrs.get("days_per_month") == 30.0:
            entry["days_per_month"] = 30
        if frame.empty:
            entry.update(used=False, reason="no complete cases")
            continue
        repeated = frame["patient_id"].map(lambda sample: keys.get(sample, sample)).isin(seen).to_numpy(dtype=bool)
        frame = frame.loc[~repeated].reset_index(drop=True)
        entry.update(duplicates_removed=int(repeated.sum()), patients_kept=len(frame), events_kept=int(frame[event_column].sum()))
        if entry["events_kept"] < MIN_VALIDATION_EVENTS:
            entry.update(used=False, reason=f"fewer than {MIN_VALIDATION_EVENTS} {spec['events']}")
            continue
        share, absent = marker_weight_measured(recipe, frame.columns)
        entry["marker_weight_measured"] = share
        if absent and (len(absent) == len(recipe["markers"]) or share < MIN_MARKER_WEIGHT):
            entry.update(used=False, reason=f"lacks locked markers carrying {1.0 - share:.0%} of the model's marker weight "
                                            f"({', '.join(absent)}); at least half must be measured")
            continue
        entry["used"] = True
        used[cohort] = frame
        seen |= {keys.get(sample, sample) for sample in frame["patient_id"]}
    return screened, used


def check_marker_weight(entry: dict[str, Any], report: dict[str, Any]) -> None:
    """Stop when SurvStudio measured a different share of the marker weight than the screen did."""
    reported = float(report["metrics"]["marker_weight_available"])
    if abs(reported - entry["marker_weight_measured"]) > 1e-9:
        raise RuntimeError(f"{entry['cohort']}: the screen measured {entry['marker_weight_measured']:.6f} of the marker weight, SurvStudio {reported:.6f}.")


def validation_row(report: dict[str, Any]) -> dict[str, Any]:
    """Cohort-level results of validate_locked_recipe: the locked model's C, the clinical-only model's C and their
    difference, each with its bootstrap 95% interval, the marker weight measured, and the calibration slope (the Cox
    coefficient of the locked linear predictor in the cohort) with its Wald 95% interval."""
    metrics = report["metrics"]
    slope_interval = metrics["calibration_slope_ci"] or [None, None]
    return {
        "n": report["cohort"]["n"], "events": report["cohort"]["events"],
        "weight_measured": metrics["marker_weight_available"], "absent": ";".join(metrics["absent_markers"]),
        "c": metrics["c_index"], "c_lower": metrics["c_index_ci"][0], "c_upper": metrics["c_index_ci"][1],
        "clinical_c": metrics["clinical_only_c_index"],
        "clinical_lower": metrics["clinical_only_c_index_ci"][0], "clinical_upper": metrics["clinical_only_c_index_ci"][1],
        "delta_c": metrics["delta_c_index"], "delta_lower": metrics["delta_c_index_ci"][0], "delta_upper": metrics["delta_c_index_ci"][1],
        "calibration_slope": metrics["calibration_slope"],
        "calibration_lower": slope_interval[0], "calibration_upper": slope_interval[1],
    }


# SurvStudio names the lens a marker's replication was tested on ("tested"); its fit is stored under this key.
TESTED_FIT = {"added_value": "adjusted", "marginal": "marginal"}


def validation_marker_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Each locked marker's external hazard ratio from the fit its Holm-adjusted replication test in the development
    direction used (``tested``: clinically adjusted where the model has clinical covariates that vary in the cohort,
    else marginal; none when that fit was not estimable), and the test."""
    rows = []
    for row in report["markers"]:
        lens = row.get("tested")
        if lens is not None and lens not in TESTED_FIT:
            raise RuntimeError(f"{row['marker']}: SurvStudio tested its replication on an unknown lens {lens!r}.")
        tested = (row.get(TESTED_FIT[lens]) or {}) if lens is not None else {}
        rows.append({
            "marker": row["marker"], "measured": not row.get("absent", False), "tested": lens,
            "hazard_ratio": tested.get("hazard_ratio"), "ci_lower": tested.get("ci_lower"), "ci_upper": tested.get("ci_upper"),
            "same_direction": row["same_direction"], "replication_p_holm": row.get("replication_p_holm"),
            "replicated": row.get("replicated", False),
        })
    return rows


def pool_interval(cohorts: pd.DataFrame, estimate: str, lower: str, upper: str) -> dict[str, Any]:
    """random_effects of one quantity over cohorts, each cohort's estimate weighted by its own 95% interval (standard
    error = interval width / 3.92)."""
    errors = (cohorts[upper] - cohorts[lower]).to_numpy(dtype=float) / 3.92
    return random_effects(cohorts[estimate].to_numpy(dtype=float), errors)


def exponentiated(pooled: dict[str, Any]) -> dict[str, float | None]:
    """A pooled log hazard ratio (random_effects) as a hazard ratio: its estimate and every interval bound exponentiated."""
    keys = ("estimate", "ci_lower", "ci_upper", "hksj_ci_lower", "hksj_ci_upper", "pi_lower", "pi_upper")
    return {key: None if pooled.get(key) is None else float(np.exp(pooled[key])) for key in keys}


def pooled_validation(cohorts: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Random-effects pooling (random_effects: the DerSimonian-Laird estimate with its CI, the HKSJ interval and the
    prediction interval) of the locked model's C, the clinical-only model's C, their difference (the gain) and the two
    calibration slopes, each cohort's estimate weighted by its own 95% interval (pool_interval: bootstrap intervals for
    the C-indices, Wald intervals for the slopes); and of the gene component's log hazard ratio per SD (component_row),
    weighted by its standard error, with the pooled hazard ratio."""
    gene = random_effects(cohorts["gene_log_hr"].to_numpy(dtype=float), cohorts["gene_log_hr_se"].to_numpy(dtype=float))
    return {
        "model_c": pool_interval(cohorts, "c", "c_lower", "c_upper"),
        "clinical_c": pool_interval(cohorts, "clinical_c", "clinical_lower", "clinical_upper"),
        "delta_c": pool_interval(cohorts, "delta_c", "delta_lower", "delta_upper"),
        "calibration_slope": pool_interval(cohorts, "calibration_slope", "calibration_lower", "calibration_upper"),
        "clinical_calibration_slope": pool_interval(cohorts, "clinical_calibration_slope", "clinical_calibration_lower", "clinical_calibration_upper"),
        "gene_log_hr": gene,
        "gene_hazard_ratio": exponentiated(gene),
    }


def _same_number(found: float | None, reported: float | None) -> bool:
    if found is None or reported is None:
        return found is None and reported is None
    return bool(np.isfinite(found)) and abs(float(found) - float(reported)) <= 1e-10 * max(1.0, abs(float(reported)))


def locked_parts(external: pd.DataFrame, recipe: dict[str, Any], report: dict[str, Any], marker_scaling: str) -> dict[str, Any]:
    """The locked model's linear predictor in an external cohort, computed as SurvStudio's validate_locked_recipe
    computes it: its rows (outcome and the clinical columns the model uses recorded, times positive), the locked
    clinical encoder, each marker as measured or rescaled within the cohort (``marker_scaling``), held at its
    development median where the cohort lacks it or it does not vary, and a missing value at that median. Returned with
    its clinical part (the locked model's clinical terms) and its marker part (the gene component), the clinical-only
    model's linear predictor, the times, events and the tie method. Stops unless the linear predictor gives the
    report's C-index and calibration slope and the clinical-only predictor the report's clinical-only C, so whatever
    is computed from the parts is computed from SurvStudio's own scores."""
    from survival_toolkit.analysis import _cohort_frame
    from survival_toolkit.encoding import transform_feature_encoder
    from survival_toolkit.marker_evaluation import _clinical_columns_behind, _external_cox, _pooled_c_index, _unique_rows

    if recipe.get("strata_columns"):
        raise ValueError("locked_parts handles locked models without strata only.")
    external = _unique_rows(external)
    outcome, model, encoder = recipe["outcome"], recipe["model"], recipe["clinical"]["encoder"]
    markers = list(recipe["markers"])
    clinical_columns = list(recipe["clinical"]["columns"])
    clinical_model = recipe.get("clinical_only_model") or {}
    used = _clinical_columns_behind(encoder, clinical_columns, {*model["terms"], *(clinical_model.get("terms") or [])})
    frame = _cohort_frame(external, time_column=outcome["time_column"], event_column=outcome["event_column"],
                          event_positive_value=outcome["event_positive_value"], extra_columns=used)
    time = frame[outcome["time_column"]].to_numpy(dtype=float)
    event = frame[outcome["event_column"]].to_numpy(dtype=int)
    rows = list(frame.attrs["source_row_index"])
    columns: dict[str, np.ndarray] = {}
    if clinical_columns:
        unused = {column: np.nan for column in clinical_columns if column not in used}
        design = transform_feature_encoder(frame.assign(**unused) if unused else frame, encoder, output="dataframe")
        columns.update({str(name): design[name].to_numpy(dtype=float) for name in design.columns})
    scale = recipe.get("marker_scale") or {}
    for name in markers:
        median = float(recipe["marker_medians"][name])
        values = None
        if name in external.columns:
            values = pd.to_numeric(external.loc[rows, name], errors="coerce").to_numpy(dtype=float, copy=True, na_value=np.nan)
        if values is not None and marker_scaling == "within_cohort":
            observed = values[~np.isnan(values)]
            spread = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
            values = None if spread <= 0 else float(scale[name]["mean"]) + float(scale[name]["sd"]) * (values - float(np.mean(observed))) / spread
        columns[name] = np.full(time.shape[0], median) if values is None else np.where(np.isnan(values), median, values)
    terms = list(model["terms"])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    matrix = np.column_stack([columns[term] for term in terms])
    linear_predictor = matrix @ coefficients
    is_marker = np.array([term in set(markers) for term in terms], dtype=bool)
    clinical_only = None
    if clinical_model:
        clinical_only = np.column_stack([columns[term] for term in clinical_model["terms"]]) @ np.asarray(clinical_model["coefficients"], dtype=float)
    ties = str(model.get("ties", "efron"))
    metrics = report["metrics"]
    slope = _external_cox(time, event, linear_predictor[:, None], None, ties)
    checks = {"c_index": (_pooled_c_index(time, event, linear_predictor, None), metrics["c_index"]),
              "calibration_slope": (None if slope is None else slope["log_hr"], metrics["calibration_slope"])}
    if clinical_only is not None:
        checks["clinical_only_c_index"] = (_pooled_c_index(time, event, clinical_only, None), metrics.get("clinical_only_c_index"))
    differ = {key: pair for key, pair in checks.items() if not _same_number(*pair)}
    if differ or time.shape[0] != int(report["cohort"]["n"]):
        raise RuntimeError(f"The locked model's linear predictor recomputed here differs from validate_locked_recipe's "
                           f"({time.shape[0]} patients, SurvStudio {report['cohort']['n']}; recomputed and reported: {differ}).")
    return {"time": time, "event": event, "linear_predictor": linear_predictor, "clinical": matrix[:, ~is_marker] @ coefficients[~is_marker],
            "marker": matrix[:, is_marker] @ coefficients[is_marker], "clinical_only": clinical_only, "ties": ties}


def _log_or_none(value: float | None) -> float | None:
    return None if value is None or value <= 0 else float(np.log(value))


def component_row(parts: dict[str, Any]) -> dict[str, Any]:
    """Per external cohort (``parts`` from locked_parts), beyond validate_locked_recipe's report: the clinical-only
    model's calibration slope, the Cox coefficient of its linear predictor with a Wald 95% interval, computed as
    SurvStudio computes the locked model's; the SD of the locked linear predictor and of its clinical and marker parts;
    and the gene component's hazard ratio: in a Cox model of the outcome on both parts, the hazard ratio per SD (in the
    cohort) of the marker part, with its Wald 95% interval and the log hazard ratio and standard error the pooling uses.
    The fits are SurvStudio's (marker_screen.fit_cox with the locked tie method); a fit that does not converge, or whose
    coefficient runs to infinity, gives no value."""
    from scipy import stats
    from survival_toolkit.marker_evaluation import _external_cox, _marker_fit_usable
    from survival_toolkit.marker_screen import fit_cox

    time, event, ties = parts["time"], parts["event"], parts["ties"]
    row: dict[str, Any] = dict.fromkeys(("clinical_calibration_slope", "clinical_calibration_lower", "clinical_calibration_upper"))
    if parts["clinical_only"] is not None:
        slope = _external_cox(time, event, parts["clinical_only"][:, None], None, ties)
        if slope is not None:
            row.update(clinical_calibration_slope=slope["log_hr"], clinical_calibration_lower=_log_or_none(slope["ci_lower"]),
                       clinical_calibration_upper=_log_or_none(slope["ci_upper"]))
    spread = {key: float(np.std(parts[part], ddof=1)) for key, part in (("lp_sd", "linear_predictor"), ("clinical_lp_sd", "clinical"),
                                                                        ("gene_lp_sd", "marker"))}
    row.update(spread)
    row.update(dict.fromkeys(("gene_log_hr", "gene_log_hr_se", "gene_hr", "gene_hr_lower", "gene_hr_upper")))
    if spread["gene_lp_sd"] > 0:
        fit = fit_cox(time, event, np.column_stack([parts["clinical"], parts["marker"] / spread["gene_lp_sd"]]), None, ties)
        variance = float(fit.covariance[-1, -1])
        if _marker_fit_usable(fit) and np.isfinite(variance) and variance > 0:
            beta, se, z = float(fit.beta[-1]), float(np.sqrt(variance)), float(stats.norm.ppf(0.975))
            row.update(gene_log_hr=beta, gene_log_hr_se=se, gene_hr=float(np.exp(beta)), gene_hr_lower=float(np.exp(beta - z * se)),
                       gene_hr_upper=float(np.exp(beta + z * se)))
    return row


def uno_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    """Uno's C (Uno et al., Stat Med 2011) of a risk score, truncated at ``horizon``: the pairs whose earlier time is an
    event before the horizon, each weighted by the inverse square of the censoring distribution (the cohort's own
    Kaplan-Meier estimate) at that time; scikit-survival's concordance_index_ipcw. NaN without an event before the
    horizon."""
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv

    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    if not (event & (time < horizon)).any():
        return float("nan")
    outcome = Surv.from_arrays(event=event, time=time)
    return float(concordance_index_ipcw(outcome, outcome, np.asarray(risk, dtype=float), tau=horizon)[0])


def uno_row(parts: dict[str, Any], horizon: float = UNO_HORIZON_MONTHS, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Uno's C truncated at ``horizon`` of the locked model and of the clinical-only model (``parts`` from locked_parts)
    and their paired difference, each with a percentile 95% interval from ``draws`` bootstrap resamples of the patients:
    the same resamples for the three, and the ones validate_locked_recipe draws from the same seed. A resample without
    an event before the horizon, or one scikit-survival refuses (a censoring distribution of zero before an event), is
    skipped; an interval needs 20 resamples, as validate_locked_recipe's."""
    time, event = parts["time"], parts["event"].astype(bool)
    model, clinical = parts["linear_predictor"], parts["clinical_only"]
    n = time.shape[0]
    rng = np.random.default_rng(int(seed))
    sampled = []
    for _ in range(int(draws)):
        rows = rng.integers(0, n, size=n)
        if not event[rows].any():
            continue
        try:
            pair = (uno_c(time[rows], event[rows], model[rows], horizon), uno_c(time[rows], event[rows], clinical[rows], horizon))
        except ValueError:
            continue
        if np.isfinite(pair).all():
            sampled.append(pair)
    sampled = np.asarray(sampled, dtype=float).reshape(-1, 2)

    def interval(values: np.ndarray) -> tuple[float | None, float | None]:
        if values.size < 20:
            return None, None
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    model_c, clinical_c = uno_c(time, event, model, horizon), uno_c(time, event, clinical, horizon)
    (model_lower, model_upper), (clinical_lower, clinical_upper) = interval(sampled[:, 0]), interval(sampled[:, 1])
    delta_lower, delta_upper = interval(sampled[:, 0] - sampled[:, 1])
    return {"n": int(n), "events": int(event.sum()), "events_before_horizon": int((event & (time < horizon)).sum()),
            "uno_c": model_c, "uno_lower": model_lower, "uno_upper": model_upper,
            "clinical_uno_c": clinical_c, "clinical_uno_lower": clinical_lower, "clinical_uno_upper": clinical_upper,
            "uno_delta": model_c - clinical_c, "uno_delta_lower": delta_lower, "uno_delta_upper": delta_upper, "draws_used": int(len(sampled))}


def git_commit(source: Path) -> str:
    """``git describe --always`` of a checkout, with a "-dirty" suffix when a tracked file other than the paper's
    figures (which figures.py redraws) has uncommitted changes; "unknown" outside a git checkout. run_all.sh and
    run_step.sh compute the same."""
    git = ["git", "-c", "safe.directory=*", "--no-optional-locks", "-C", str(source)]
    try:
        commit = subprocess.run([*git, "describe", "--always"], capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)paper/figures"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if not commit:
        return "unknown"
    return f"{commit}-dirty" if changed else commit


@functools.lru_cache(maxsize=1)
def _survstudio_version() -> tuple[str, str]:
    import survival_toolkit

    commit = os.environ.get("SURVSTUDIO_COMMIT") or git_commit(Path(survival_toolkit.__file__).resolve().parents[2])
    return survival_toolkit.__version__, commit


def survstudio_version() -> dict[str, str]:
    """SurvStudio's version and the commit of the source it was imported from (``git_commit``: a "-dirty" suffix marks
    uncommitted changes), read once per run. run_all.sh and run_step.sh pass the same in SURVSTUDIO_COMMIT."""
    version, commit = _survstudio_version()
    return {"version": version, "commit": commit}


def commit_is_clean(commit: str) -> bool:
    """A commit stamp that names the code exactly: known, and without uncommitted changes."""
    return bool(commit) and commit != "unknown" and not commit.endswith("-dirty")


def resumable_rows(previous: pd.DataFrame, key: list[str], wanted: set[tuple], stamp: dict[str, str]) -> pd.DataFrame:
    """The rows of an earlier, interrupted run that a resumed run keeps: finished without error, still part of the
    design (``key`` in ``wanted``), and stamped with the same values (``stamp``: column -> value, for the simulation
    the SurvStudio commit, the design hash and the hash of the simulation's code). Nothing is kept when the stamp holds
    a commit with uncommitted changes or an unknown one, since two such runs cannot be told apart. Read ``previous``
    with STAMP_DTYPES: a stamp such as "0123456" read as a number no longer matches."""
    if previous.empty or any(column not in previous for column in [*key, *stamp, "error"]):
        return previous.iloc[0:0]
    if "survstudio_commit" in stamp and not commit_is_clean(stamp["survstudio_commit"]):
        return previous.iloc[0:0]
    keep = previous["error"].isna().to_numpy(dtype=bool)
    for column, value in stamp.items():
        keep = keep & (previous[column].astype(str) == str(value)).to_numpy(dtype=bool)
    keep = keep & np.array([tuple(row) in wanted for row in previous[key].itertuples(index=False)], dtype=bool)
    return previous.loc[keep].drop_duplicates(key, keep="last")


def write_json(path: Path, value: Any) -> None:
    """Write JSON through a temporary file and a rename; a file of results/ is stamped (stamp_result)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    _stamp_if_result(path)


# ── Provenance of results/: which run wrote each file, and from which results files ──────────────────────────────
def analysis_files() -> list[Path]:
    """What a result depends on besides SurvStudio and the data: common.py, every numbered analysis script (not the
    self-check) and the committed tables they read (the breast duplicate pairs and the breast and LUAD data
    manifests)."""
    scripts = [path for path in SCRIPTS.glob("[0-9][0-9]_*.py") if path.name != "00_self_check.py"]
    tables = [SCRIPTS.parent / name for name in ("breast_duplicate_pairs.csv", "breast_data_manifest.csv", "luad_data_manifest.csv")]
    return sorted([SCRIPTS / "common.py", *scripts, *(path for path in tables if path.exists())], key=lambda path: path.name)


def analysis_code() -> str:
    """The first 16 hex digits of a SHA-256 over the names and bytes of the analysis_files."""
    digest = hashlib.sha256()
    for path in analysis_files():
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


# Taken when the run starts, from the code it runs.
ANALYSIS_CODE = analysis_code()
# The results files this run has read (read_result), with their SHA-256: the inputs of whatever it writes next.
_INPUTS: dict[str, str] = {}


def read_result(name: str, **options: Any) -> Any:
    """A file of results/ written by another step (JSON parsed; CSV as a data frame, read with ``options``), recorded
    as an input of every result this run writes afterwards."""
    data = (RESULTS / name).read_bytes()
    _INPUTS[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data.decode("utf-8")) if name.endswith(".json") else pd.read_csv(io.BytesIO(data), **options)


def stamp_result(path: Path) -> None:
    """Record the run that wrote a file of results/ in results/stamps/<file name>.json: the file's SHA-256, the
    SurvStudio version and commit, ANALYSIS_CODE, the script, and the results files it had read (read_result) with
    their SHA-256. figures.py draws a figure only from files that pass ``result_problems``."""
    stamp = {"file": path.name, "sha256": sha256_file(path), "survstudio": survstudio_version(), "analysis_code": ANALYSIS_CODE,
             "script": Path(sys.argv[0]).name, "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "inputs": {name: digest for name, digest in _INPUTS.items() if name != path.name}}
    stamps = RESULTS / "stamps"
    stamps.mkdir(parents=True, exist_ok=True)
    target = stamps / f"{path.name}.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(stamp, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _stamp_if_result(path: Path) -> None:
    if path.resolve().parent == RESULTS.resolve():
        stamp_result(path)


def result_problems(names: Iterable[str]) -> list[str]:
    """Why the results files ``names`` cannot be shown together (nothing when they can): a file without a stamp or
    changed since its run stamped it, a file computed from a results file that has changed since (a step rerun
    without the steps after it), or files from different SurvStudio commits or analysis code (a partial rerun)."""
    problems: list[str] = []
    runs: dict[str, tuple[str, str]] = {}
    for name in dict.fromkeys(names):
        path, stamp_path = RESULTS / name, RESULTS / "stamps" / f"{name}.json"
        if not path.exists():
            problems.append(f"{name} is missing")
            continue
        if not stamp_path.exists():
            problems.append(f"{name} has no stamp (written before stamps were kept, or not by an analysis script)")
            continue
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp.get("sha256") != sha256_file(path):
            problems.append(f"{name} changed after {stamp.get('script')} wrote it")
        for upstream, digest in (stamp.get("inputs") or {}).items():
            if not (RESULTS / upstream).exists() or sha256_file(RESULTS / upstream) != digest:
                problems.append(f"{name} was computed from an earlier {upstream}; rerun {stamp.get('script')}")
        runs[name] = ((stamp.get("survstudio") or {}).get("commit", "unknown"), str(stamp.get("analysis_code")))
    if len(set(runs.values())) > 1:
        problems.append("the files come from different runs: " + "; ".join(
            f"{name} (SurvStudio {commit}, analysis code {code})" for name, (commit, code) in runs.items()))
    return problems


# ── The breast data the paper used ──────────────────────────────────────────────────────────────────────────────
def breast_data_files() -> list[Path]:
    """Every file the breast analyses read: each file under BREAST, and breast_duplicates.csv next to it. The
    cBioPortal METABRIC table is left out (with a failed download of it): it is downloaded on first use and checked
    against its own SHA-256 (CBIOPORTAL_METABRIC_SHA256) on every read."""
    cbioportal = BREAST / "METABRIC" / "cbioportal_patients.csv"
    skipped = {cbioportal, cbioportal.with_name(cbioportal.name + ".download")}
    files = [path for path in BREAST.rglob("*") if path.is_file() and path not in skipped]
    duplicates = BREAST.parent / "breast_duplicates.csv"
    if duplicates.exists():
        files.append(duplicates)
    return sorted(files, key=lambda path: path.relative_to(BREAST.parent).as_posix())


def breast_manifest() -> pd.DataFrame:
    """Each file of breast_data_files, relative to the cohorts folder, with its size and SHA-256."""
    rows = [{"file": path.relative_to(BREAST.parent).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in breast_data_files()]
    return pd.DataFrame(rows, columns=["file", "bytes", "sha256"])


def check_breast_manifest(manifest: Path = BREAST_MANIFEST) -> int:
    """Stop unless the breast data are exactly the files of the committed manifest, with the same sizes and SHA-256
    (an extra file stops the run too: an extra cohort folder would enter the external screens). Returns the number
    of files checked."""
    expected = pd.read_csv(manifest, dtype={"file": str, "sha256": str}).set_index("file")
    found = breast_manifest().set_index("file")
    problems = [f"missing {name}" for name in expected.index.difference(found.index)]
    problems += [f"not in the manifest: {name}" for name in found.index.difference(expected.index)]
    for name in expected.index.intersection(found.index):
        if int(expected.at[name, "bytes"]) != int(found.at[name, "bytes"]) or expected.at[name, "sha256"] != found.at[name, "sha256"]:
            problems.append(f"differs: {name}")
    if problems:
        raise RuntimeError(f"The breast data in {BREAST.parent} are not the ones {manifest.name} pins ({len(problems)} problems): "
                           + "; ".join(problems[:10]) + ("; ..." if len(problems) > 10 else ""))
    return len(expected)


# ── The LUAD data and the snapshot files the paper used ────────────────────────────────────────────────────────
# Every LUAD file the analyses read, with its size and SHA-256 (luad_manifest.py writes it, 00_self_check.py checks
# the data against it).
LUAD_MANIFEST = PAPER / "luad_data_manifest.csv"
LUAD_MANIFEST_COLUMNS = ["file", "bytes", "sha256", "content_bytes", "content_sha256"]
# The three small files released with the paper (data_snapshot/SNAPSHOT.md) and their SHA-256.
SNAPSHOT = PAPER / "data_snapshot"
SNAPSHOT_SHA256 = {
    "Homo_sapiens.gene_info.gz": "4a333625190f594abc21fd7564c7a0ee991a0331c639c6d3b7bb814b944e75b7",
    "cbioportal_patients.csv": CBIOPORTAL_METABRIC_SHA256,
    "sample_decisions.csv": "529aa8bc7ab24c5065ce32e1fbddc1bd4c0c470c764a45150c0a4ac0cdc2e8a6",
}


def luad_data_files() -> list[Path]:
    """Every LUAD file the analyses read: the Xena RNA-seq matrix of case study I (scripts 01, 02, 06, 12 and 13),
    and the harmonized clinical table and gene-level expression of the seven GEO cohorts (scripts 03, 12, 13 and
    15)."""
    return [XENA_EXPRESSION, LUAD / "harmonized_clinical.csv", *(LUAD / cohort / "expression_genes.csv.gz" for cohort in GEO_COHORTS)]


def content_digest(path: Path) -> tuple[int, str]:
    """Size and SHA-256 of a gzip file's decompressed content."""
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def luad_manifest() -> pd.DataFrame:
    """Each of luad_data_files, relative to the cohorts folder, with its size and SHA-256 and, for a .gz file, the size
    and SHA-256 of its decompressed content."""
    rows = []
    for path in luad_data_files():
        content = content_digest(path) if path.suffix == ".gz" else ("", "")
        rows.append({"file": path.relative_to(LUAD.parent).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                     "content_bytes": str(content[0]), "content_sha256": content[1]})
    return pd.DataFrame(rows, columns=LUAD_MANIFEST_COLUMNS)


def check_luad_manifest(manifest: Path = LUAD_MANIFEST) -> int:
    """Stop unless every file of the committed LUAD manifest is there, unchanged: a .gz file with the same decompressed
    content (its gzip header records when it was written, so a file rewritten from the same data differs in its bytes
    only there), any other file with the same size and SHA-256. Returns the number of files checked."""
    expected = pd.read_csv(manifest, dtype=str, keep_default_na=False).set_index("file")
    problems = []
    for name, row in expected.iterrows():
        path = LUAD.parent / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        if row["content_sha256"]:
            try:
                size, digest = content_digest(path)
            except (OSError, EOFError):
                problems.append(f"not a readable gzip file: {name}")
                continue
            same = str(size) == row["content_bytes"] and digest == row["content_sha256"]
        else:
            same = str(path.stat().st_size) == row["bytes"] and sha256_file(path) == row["sha256"]
        if not same:
            problems.append(f"differs: {name}")
    if problems:
        raise RuntimeError(f"The LUAD data in {LUAD.parent} are not the ones {manifest.name} pins ({len(problems)} problems): "
                           + "; ".join(problems))
    return len(expected)


def check_snapshot(folder: Path = SNAPSHOT) -> list[str]:
    """Stop when a file of the data snapshot differs from its SHA-256 (SNAPSHOT_SHA256); returns the files checked,
    the ones present."""
    present = [name for name in SNAPSHOT_SHA256 if (folder / name).is_file()]
    changed = [name for name in present if sha256_file(folder / name) != SNAPSHOT_SHA256[name]]
    if changed:
        raise RuntimeError(f"{folder}: {', '.join(changed)} differ from the SHA-256 in SNAPSHOT.md; restore them with git checkout.")
    return present


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def random_effects(estimates: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    """Random-effects pooling of one estimate per cohort. Needs at least two estimates, each finite and with a finite,
    positive standard error; anything else raises. Returns:

    - ``estimate``, ``ci_lower``, ``ci_upper``, ``tau2``, ``q``: the DerSimonian-Laird estimate with its 95% CI
      (estimate +/- 1.96 SE, SE = 1 / sqrt(sum(w)) with the random-effects weights w = 1 / (se^2 + tau2)), the
      DerSimonian-Laird between-cohort variance and Cochran's Q; ``k``, the number of cohorts.
    - ``hksj_ci_lower``, ``hksj_ci_upper``: the Hartung-Knapp-Sidik-Jonkman 95% interval around the same estimate:
      estimate +/- t(0.975, k - 1) SE_HK, with SE_HK^2 = sum(w (y - estimate)^2) / ((k - 1) sum(w)) and the same weights
      (the DerSimonian-Laird tau2). With the usual modification, SE_HK is never taken below the DerSimonian-Laird SE
      (Knapp and Hartung's ad hoc variance correction, metafor's rma(test = "adhoc")), so the interval is never narrower
      than the DerSimonian-Laird one; ``hksj_scale`` is SE_HK^2 / SE^2 before that floor (the correction applied when it
      is below 1).
    - ``pi_lower``, ``pi_upper``: the 95% prediction interval for the value in a new cohort (Higgins, Thompson and
      Spiegelhalter 2009): estimate +/- t(0.975, k - 2) sqrt(tau2 + SE^2); None with fewer than three cohorts.
    00_self_check.py checks the intervals against R's metafor.
    """
    from scipy import stats

    estimates = np.asarray(estimates, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if estimates.ndim != 1 or estimates.shape != errors.shape:
        raise ValueError(f"random_effects needs one standard error per estimate (shapes {estimates.shape} and {errors.shape}).")
    if estimates.size < 2:
        raise ValueError(f"random_effects needs at least two estimates, got {estimates.size}.")
    if not (np.isfinite(estimates).all() and np.isfinite(errors).all() and (errors > 0).all()):
        raise ValueError(f"random_effects needs finite estimates with finite positive standard errors: {estimates.tolist()}, {errors.tolist()}.")
    k = int(estimates.size)
    weights = 1.0 / errors**2
    fixed = np.sum(weights * estimates) / np.sum(weights)
    q = float(np.sum(weights * (estimates - fixed) ** 2))
    tau2 = (q - (k - 1)) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    if not np.isfinite(tau2):
        raise ValueError(f"random_effects: the between-cohort variance is not finite (Q {q}); check the standard errors {errors.tolist()}.")
    tau2 = max(0.0, float(tau2))
    weights = 1.0 / (errors**2 + tau2)
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    se = float(np.sqrt(1.0 / np.sum(weights)))
    if not (np.isfinite(pooled) and np.isfinite(se)):
        raise ValueError(f"random_effects: the pooled estimate is not finite ({pooled} with SE {se}).")
    scale = float(np.sum(weights * (estimates - pooled) ** 2) / (k - 1))
    hksj = float(stats.t.ppf(0.975, k - 1)) * se * float(np.sqrt(max(1.0, scale)))
    result = {"estimate": pooled, "ci_lower": pooled - 1.96 * se, "ci_upper": pooled + 1.96 * se, "tau2": tau2, "q": q, "k": k,
              "hksj_ci_lower": pooled - hksj, "hksj_ci_upper": pooled + hksj, "hksj_scale": scale, "pi_lower": None, "pi_upper": None}
    if k >= 3:
        spread = float(stats.t.ppf(0.975, k - 2)) * float(np.sqrt(tau2 + se**2))
        result.update(pi_lower=pooled - spread, pi_upper=pooled + spread)
    return result

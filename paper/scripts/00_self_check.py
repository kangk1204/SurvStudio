"""Quick checks of the helpers the analyses rely on (seconds; run_all.sh runs them first). Every check runs: one that
cannot run fails the self-check.

On a small synthetic MetaGxBreast-like export in a temporary folder: a cohort without complete cases, duplicate
chains within and across cohorts (none through KOO), samples of a cohort sharing a unique_patient_ID, the follow-up
and tumour-size conventions (T categories, size classes), METABRIC's endpoints from a cBioPortal table (exact patient
matching, tumours without a record left out and counted, 10-year censoring, the pinned copy, and the snapshot copy
used when a download differs or fails), the external-cohort screens of case studies IV (overall survival) and V
(recurrence: relapse-free survival from cBioPortal in development), padded gene symbols whose extra probes carry
make.unique suffixes (the most variable probe is kept), the METABRIC sites of script 14, and the LUAD data manifest
(.gz files compared by their decompressed content). Then random-effects pooling (the Hartung-Knapp-Sidik-Jonkman and
prediction intervals against worked examples computed with R's metafor), the validation rows (the hazard ratio of the
lens a marker was tested on) and their pooling, script 15's cohort test, sign and pooling, the simulation's resume
filter and summary check on stamps that look like numbers after a CSV round trip, the stamps figures.py checks, a
locked model made by SurvStudio (the marker-weight rule, the clinical-only interval and the paired left-out gain the
scripts read; the locked linear predictor recomputed for the clinical-only calibration slope, the gene component and
Uno's C, as SurvStudio scores it), Uno's C against the estimator written out, the breast data against
breast_data_manifest.csv, the LUAD data against luad_data_manifest.csv, and the files of data_snapshot/ against their
SHA-256.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import common

GENES = ["G1", "G2", "G3"]
RECIPE = {"markers": GENES, "model": {"terms": ["age", *GENES], "coefficients": [0.01, 1.0, -1.0, 0.5]},
          "marker_scale": {gene: {"mean": 0.0, "sd": 1.0} for gene in GENES}}
REASON_NO_OS = "no overall survival recorded"
REASON_NO_RECURRENCE = "no relapse-free or distant metastasis-free survival recorded"
rng = np.random.default_rng(20260928)


def write_cohort(breast: Path, name: str, clinical: dict, genes: list[str] = GENES) -> None:
    folder = breast / name
    folder.mkdir(parents=True)
    clinical = pd.DataFrame(clinical)
    clinical.to_csv(folder / "clinical.csv", index=False)
    expression = pd.DataFrame(rng.normal(size=(len(clinical), len(genes))), columns=genes)
    expression.insert(0, "sample_id", clinical["sample_id"])
    expression.to_csv(folder / "expression.csv.gz", index=False)


def cohort(prefix: str, n: int, deaths: int, days: np.ndarray, **columns) -> dict:
    return {
        "sample_id": [f"{prefix}_{index}" for index in range(1, n + 1)], "sample_type": "tumor",
        "age_at_initial_pathologic_diagnosis": np.round(rng.uniform(35, 80, n), 1), "tumor_size": np.round(rng.uniform(0.5, 8, n), 1),
        "N": rng.integers(0, 2, n), "grade": rng.integers(1, 4, n), "er": "positive",
        "vital_status": ["deceased" if index < deaths else "living" for index in range(n)], "days_to_death": days,
        **columns,
    }


def cbioportal_table() -> pd.DataFrame:
    """cBioPortal patients MB-0001 to MB-0038 (MB_0039 and MB_0040 have no record), MB-0037 without overall survival,
    MB-0036 without a relapse status, MB-0035 without a site, and two patients of METABRIC's other samples."""
    patients = [f"MB-{index:04d}" for index in range(1, 39)]
    n = len(patients)
    os_months = np.round(rng.uniform(1, 200, n), 2)
    os_months[36] = np.nan
    rfs_status = rng.choice(["1:Recurred", "0:Not Recurred"], n).astype(object)
    rfs_status[35] = None
    site = np.resize([1.0, 2.0, 3.0, 4.0, 5.0], n)
    site[34] = np.nan
    table = pd.DataFrame({"patientId": patients, "COHORT": site, "OS_MONTHS": os_months,
                          "OS_STATUS": rng.choice(["1:DECEASED", "0:LIVING"], n), "RFS_MONTHS": np.round(rng.uniform(1, 200, n), 2),
                          "RFS_STATUS": rfs_status})
    others = pd.DataFrame({"patientId": ["MTS-T0001", "MTS-T0002"], "COHORT": [1.0, 2.0], "OS_MONTHS": [10.0, 20.0],
                           "OS_STATUS": ["0:LIVING", "1:DECEASED"], "RFS_MONTHS": [10.0, 20.0], "RFS_STATUS": ["0:Not Recurred", "1:Recurred"]})
    return pd.concat([table, others], ignore_index=True)


def synthetic_export(root: Path) -> None:
    breast = root / "cohorts" / "breast"
    sizes = np.resize([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], 40)  # T1 median 1.5, T2 4.0, T3/T4 7.0
    metabric = cohort("MB", 40, 25, 30.0 * rng.integers(1, 100, 40), tumor_size=sizes)
    metabric["sample_id"] = [f"MB_{index:04d}" for index in range(1, 41)]
    metabric = pd.DataFrame(metabric)
    healthy = metabric.iloc[:1].assign(sample_id="MB_0100", sample_type="healthy")
    write_cohort(breast, "METABRIC", pd.concat([metabric, healthy]).to_dict("list"))
    cbioportal_table().to_csv(breast / "METABRIC" / "cbioportal_patients.csv", index=False)
    days = lambda n: 30.0 * rng.integers(1, 100, n) + 7  # noqa: E731 - real days: not multiples of 30
    relapses = {"days_to_tumor_recurrence": days(40), "recurrence_status": ["recurrence"] * 35 + ["norecurrence"] * 5}
    write_cohort(breast, "AAA", cohort("AAA", 20, 15, days(20), grade=np.nan))
    # BBB_2 (a death) is BBB_1's patient: 44 patients and 34 deaths remain.
    write_cohort(breast, "BBB", cohort("BBB", 45, 35, days(45)))
    write_cohort(breast, "CCC", cohort("CCC", 12, 3, days(12), tumor_size=np.resize([1, 2, 3, 4], 12)))
    # DDD stores months x 30 days; DDD_1 and DDD_2 (living) are removed as duplicates.
    write_cohort(breast, "DDD", cohort("DDD", 40, 34, 30.0 * rng.integers(1, 100, 40), vital_status=["living"] * 3 + ["deceased"] * 34 + ["living"] * 3))
    write_cohort(breast, "EEE", cohort("EEE", 40, 35, days(40)), genes=["G3"])
    write_cohort(breast, "EMC2", cohort("EMC2", 40, 35, days(40), **relapses))
    # FFF_3 and FFF_4 share a unique_patient_ID; the two samples without one are not joined.
    write_cohort(breast, "FFF", cohort("FFF", 10, 0, np.full(10, np.nan), vital_status=np.nan,
                                       unique_patient_ID=["P1", "P2", "P3", "P3", "P5", "P6", "P7", "P8", np.nan, np.nan]))
    # GGG_1's patient ID is FFF_3's, but IDs are compared within a cohort only.
    write_cohort(breast, "GGG", cohort("GGG", 20, 0, np.full(20, np.nan), vital_status=np.nan, dmfs_days=30.0 * rng.integers(1, 200, 20),
                                       dmfs_status=["recurrence"] * 12 + ["norecurrence"] * 8, er=["positive"] * 15 + ["negative"] * 5,
                                       unique_patient_ID=["P3", *(f"G{index}" for index in range(2, 21))]))
    # HHH: relapse-free survival only; HHH_1 (a relapse) is METABRIC's MB_0005.
    write_cohort(breast, "HHH", cohort("HHH", 40, 0, np.full(40, np.nan), vital_status=np.nan, **relapses))
    # UNC4 records size classes, not centimetres: no complete cases.
    write_cohort(breast, "UNC4", cohort("UNC4", 30, 25, days(30), tumor_size=np.resize([1.0, 3.0, 6.0, 1.5], 30)))
    pd.DataFrame({"sample": ["DDD_1", "DDD_2", "FFF_1", "DDD_3", "BBB_1", "FFF_2", "BBB_3", "KOO_1", "BBB_6", "HHH_1"],
                  "duplicates": ["MB_0001", "FFF_1", "BBB_5", "CCC_1", "FFF_2", "BBB_2", "KOO_1", "BBB_4", "", "MB_0005"]}).to_csv(
        root / "cohorts" / "breast_duplicates.csv", index=False)
    (root / "paper").mkdir()
    pd.DataFrame({"cohort_a": ["METABRIC"], "sample_a": ["MB_0002"], "cohort_b": ["METABRIC"], "sample_b": ["MB_0003"]}).to_csv(
        root / "paper" / "breast_duplicate_pairs.csv", index=False)


def check_breast(root: Path) -> None:
    common.BREAST, common.PAPER = root / "cohorts" / "breast", root / "paper"
    cbioportal = common.BREAST / "METABRIC" / "cbioportal_patients.csv"
    pinned = cbioportal.read_bytes()
    common.CBIOPORTAL_METABRIC_SHA256, common.CBIOPORTAL_METABRIC_PATIENTS = common.sha256_file(cbioportal), len(pd.read_csv(cbioportal))
    keys = common.breast_patient_keys()
    assert "" not in keys, "an empty duplicate entry became a sample"
    assert keys["BBB_1"] == keys["BBB_2"] == keys["FFF_2"], "a chain within a cohort is one patient"
    assert keys["DDD_2"] == keys["BBB_5"], "a chain across cohorts is one patient"
    assert keys["BBB_3"] != keys["BBB_4"], "chains are not followed through KOO"
    assert keys["FFF_3"] == keys["FFF_4"], "samples of a cohort with the same unique_patient_ID are one patient"
    assert keys.get("FFF_9", "FFF_9") != keys.get("FFF_10", "FFF_10"), "samples without a unique_patient_ID are not joined"
    assert keys.get("GGG_1", "GGG_1") != keys["FFF_3"], "patient IDs are not compared across cohorts"

    empty, genes = common.load_breast_cohort("AAA", genes=GENES)
    assert empty.empty and {"patient_id", "os_months", "os_event", "age", "tumor_size", "node_positive", "grade", "er"} <= set(empty.columns), \
        "a cohort without complete cases gives every column and no rows"
    assert genes == [] and empty.attrs["within_cohort_duplicates_removed"] == 0

    # METABRIC: both endpoints from cBioPortal, matched exactly; no record (MB_0039, MB_0040) or no endpoint left out.
    patients = pd.read_csv(cbioportal).set_index("patientId")
    for endpoint, missing, (months_column, status_column, events) in (("os", "MB_0037", common.CBIOPORTAL_ENDPOINTS["os"]),
                                                                      ("recurrence", "MB_0036", common.CBIOPORTAL_ENDPOINTS["recurrence"])):
        development, _ = common.load_breast_cohort("METABRIC", genes=[], endpoint=endpoint, er_positive=endpoint == "recurrence")
        time_column, event_column = common.ENDPOINT_COLUMNS[endpoint]
        kept = set(development["patient_id"])
        assert not kept & {"MB_0100", "MB_0003", "MB_0039", "MB_0040", missing}, (endpoint, "tumours only, one per patient, with a cBioPortal endpoint")
        assert len(development) == 36 and development.attrs["cbioportal_no_record"] == 2 and development.attrs["cbioportal_endpoint_missing"] == 1, \
            (endpoint, len(development), development.attrs)
        assert development.attrs["days_per_month"] is None
        record = patients.loc[development["patient_id"].str.replace("_", "-")]
        months = record[months_column].to_numpy()
        assert np.allclose(development[time_column], np.minimum(months, 120.0)), f"{endpoint}: cBioPortal months, censored at 10 years"
        happened = record[status_column].map(events).to_numpy()
        assert np.array_equal(development[event_column].to_numpy(), np.where(months > 120.0, 0, happened)), f"{endpoint}: cBioPortal status"
    for rows in (3, len(patients)):  # too few patients; then the right shape but not the pinned copy
        pd.DataFrame({column: range(len(patients)) for column in common.CBIOPORTAL_METABRIC_COLUMNS}).iloc[:rows].to_csv(cbioportal, index=False)
        try:
            common.metabric_cbioportal()
        except RuntimeError:
            continue
        raise AssertionError("a cBioPortal table other than the pinned one was accepted")
    cbioportal.write_bytes(pinned)

    # A first download that differs from the pinned copy, or fails, gives way to the copy released with the paper, and
    # a differing download is kept next to the table; with no such copy a differing download still stops the run.
    snapshot, download = root / "snapshot" / "cbioportal_patients.csv", cbioportal.with_name(cbioportal.name + ".download")
    snapshot.parent.mkdir()
    snapshot.write_bytes(pinned)
    other = pd.read_csv(cbioportal).assign(OS_MONTHS=1.0)

    def differing(target: Path) -> None:
        other.to_csv(target, index=False)

    def offline(target: Path) -> None:
        raise OSError("no network")

    saved = common.CBIOPORTAL_SNAPSHOT, common._download_cbioportal
    try:
        common.CBIOPORTAL_SNAPSHOT = snapshot
        for fetch in (differing, offline):
            cbioportal.unlink()
            common._download_cbioportal = fetch
            with contextlib.redirect_stderr(io.StringIO()) as message:
                table = common.metabric_cbioportal()
            assert len(table) == len(patients) and cbioportal.read_bytes() == pinned, fetch.__name__
            assert "Using the copy the paper used" in message.getvalue(), message.getvalue()
        assert download.exists() and pd.read_csv(download)["OS_MONTHS"].eq(1.0).all(), "a differing download is kept"
        common.CBIOPORTAL_SNAPSHOT, common._download_cbioportal = snapshot.with_name("missing.csv"), differing
        cbioportal.unlink()
        try:
            common.metabric_cbioportal()
        except RuntimeError:
            pass
        else:
            raise AssertionError("a differing download was accepted without a copy to fall back to")
    finally:
        common.CBIOPORTAL_SNAPSHOT, common._download_cbioportal = saved
        download.unlink(missing_ok=True)
        cbioportal.write_bytes(pinned)

    # Script 14: each sample's site, in the order given (NaN: no site, no record, not a METABRIC sample).
    samples = pd.Series(["MB_0007", "MB_0035", "MB_0039", "MB_0001", "XX_0001", "MB_0012"])
    expected = [patients.at["MB-0007", "COHORT"], np.nan, np.nan, patients.at["MB-0001", "COHORT"], np.nan, patients.at["MB-0012", "COHORT"]]
    assert np.array_equal(common.metabric_sites(samples), np.array(expected, dtype=float), equal_nan=True), common.metabric_sites(samples)
    shuffled = development.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert np.array_equal(common.metabric_sites(shuffled["patient_id"]),
                          patients["COHORT"].reindex(shuffled["patient_id"].str.replace("_", "-")).to_numpy(dtype=float), equal_nan=True)

    assert common.t_category_cm() == {1.0: 1.5, 2.0: 4.0, 3.0: 7.0, 4.0: 7.0}, common.t_category_cm()
    mapped, from_t = common.tumour_size_cm(pd.Series([1, 2, 3, 4, np.nan]))
    assert from_t and mapped.iloc[:4].tolist() == [1.5, 4.0, 7.0, 7.0] and np.isnan(mapped.iloc[4])
    for sizes in ([1.0, 2.5, 3.0], [1, 2, 5], [0, 1, 2]):
        kept, from_t = common.tumour_size_cm(pd.Series(sizes))
        assert not from_t and kept.tolist() == [float(size) for size in sizes], sizes
    months = pd.DataFrame({"a": 30.0 * np.arange(1, 12), "b": [30.0, 60.0, 90.0, 120.0] + [np.nan] * 7, "c": np.arange(1, 12) * 30.0 + 1})
    assert common.days_per_month("X", "a", months) == 30.0
    assert common.days_per_month("X", "b", months) == common.DAYS_PER_MONTH, "fewer than 10 values decide nothing"
    assert common.days_per_month("X", "c", months) == common.DAYS_PER_MONTH

    recurrence, _ = common.load_breast_cohort("GGG", genes=GENES, endpoint="recurrence", er_positive=True)
    assert common.recurrence_endpoint("GGG") == "dmfs" and len(recurrence) == 15 and recurrence.attrs["days_per_month"] == 30.0
    try:
        common.load_breast_cohort("BBB", endpoint="recurrence")
    except ValueError:
        pass
    else:
        raise AssertionError("a cohort without a recurrence endpoint must not load one")
    audit, _ = common.load_breast_cohort("CCC", genes=[], endpoint=None, dedupe=False)
    assert audit.attrs["tumour_size_from_t_category"] and audit["tumor_size"].max() == 4, "audits see T categories as recorded, flagged"

    screened, used = common.screen_validation_cohorts("IV", RECIPE)
    entries = {entry["cohort"]: entry for entry in screened}
    reasons = {cohort: entry.get("reason") for cohort, entry in entries.items()}
    assert reasons == {"AAA": "no complete cases", "BBB": None, "CCC": "fewer than 30 deaths", "DDD": None,
                       "EEE": "lacks locked markers carrying 80% of the model's marker weight (G1, G2); at least half must be measured",
                       "EMC2": "patients selected for relapse", "FFF": REASON_NO_OS, "GGG": REASON_NO_OS, "HHH": REASON_NO_OS,
                       "UNC4": "no complete cases"}, reasons
    assert sorted(used) == ["BBB", "DDD"] and all(entries[cohort]["used"] for cohort in used)
    assert entries["BBB"]["duplicates_within"] == 1 and "BBB_2" not in set(used["BBB"]["patient_id"])
    assert entries["BBB"]["patients_kept"] == 44 and entries["BBB"]["events_kept"] == 34
    assert {"BBB_3", "BBB_4"} <= set(used["BBB"]["patient_id"])
    assert entries["CCC"]["tumour_size"] == "T category, mapped to cm" and entries["DDD"]["days_per_month"] == 30
    assert entries["UNC4"]["tumour_size"].startswith("size classes")
    assert entries["DDD"]["duplicates_removed"] == 2 and entries["DDD"]["events_kept"] == 34
    assert "DDD_3" in set(used["DDD"]["patient_id"]), "patients of a cohort that was not used are not removed later"
    assert entries["EEE"]["marker_weight_measured"] == 0.2

    # Case study V: METABRIC ER-positive development patients with cBioPortal relapse-free survival.
    screened, used = common.screen_validation_cohorts("V", RECIPE)
    entries = {entry["cohort"]: entry for entry in screened}
    reasons = {cohort: entry.get("reason") for cohort, entry in entries.items()}
    assert reasons == {"AAA": REASON_NO_RECURRENCE, "BBB": REASON_NO_RECURRENCE, "CCC": REASON_NO_RECURRENCE, "DDD": REASON_NO_RECURRENCE,
                       "EEE": REASON_NO_RECURRENCE, "EMC2": "patients selected for relapse", "FFF": REASON_NO_RECURRENCE,
                       "GGG": "fewer than 30 events", "HHH": None, "UNC4": REASON_NO_RECURRENCE}, reasons
    assert entries["GGG"]["endpoint"] == "dmfs" and entries["HHH"]["endpoint"] == "rfs"
    assert list(used) == ["HHH"] and entries["HHH"]["duplicates_removed"] == 1 and "HHH_1" not in set(used["HHH"]["patient_id"])
    assert entries["HHH"]["patients_kept"] == 39 and entries["HHH"]["events_kept"] == 34

    # The data manifest: every file but the cBioPortal table, and nothing else.
    manifest = root / "manifest.csv"
    common.breast_manifest().to_csv(manifest, index=False)
    listed = set(pd.read_csv(manifest)["file"])
    assert "breast_duplicates.csv" in listed and "breast/METABRIC/cbioportal_patients.csv" not in listed and len(listed) == 23, sorted(listed)
    assert common.check_breast_manifest(manifest) == 23
    extra = common.BREAST / "AAA" / "notes.txt"
    extra.write_text("an extra file", encoding="utf-8")
    try:
        common.check_breast_manifest(manifest)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a file missing from the manifest was accepted")
    extra.unlink()


def check_probes(root: Path) -> None:
    """Padded symbols (GSE58644: " RNF207 ") whose further probes make.unique() named " RNF207 .1": one column per gene,
    the most variable probe; a symbol with a dot whose stem is no gene of the export ("HGC6.3") stays itself."""
    common.BREAST = root / "probes" / "breast"
    folder = common.BREAST / "PRB"
    folder.mkdir(parents=True)
    n = 30
    pd.DataFrame(cohort("PRB", n, 10, 30.0 * rng.integers(1, 100, n) + 7)).to_csv(folder / "clinical.csv", index=False)
    base = rng.normal(size=(n, 6))
    expression = pd.DataFrame({" G1 ": base[:, 0], " G1 .1": 3.0 * base[:, 1], " G2 ": base[:, 2], "G3": 2.0 * base[:, 3],
                               "G3.1": 0.5 * base[:, 4], "HGC6.3": base[:, 5]})
    expression.insert(0, "sample_id", [f"PRB_{index}" for index in range(1, n + 1)])
    expression.to_csv(folder / "expression.csv.gz", index=False)
    frame, genes = common.load_breast_cohort("PRB", endpoint=None, dedupe=False)
    assert genes == ["G1", "G2", "G3", "HGC6.3"], genes
    assert np.allclose(frame["G1"], expression[" G1 .1"]) and np.allclose(frame["G3"], expression["G3"]), "the most variable probe"
    only, genes = common.load_breast_cohort("PRB", genes=["G1"], endpoint=None, dedupe=False)
    assert genes == ["G1"] and np.allclose(only["G1"], expression[" G1 .1"])
    assert common.probe_gene("OK/SW-CL.58", {"OK/SW-CL.58"}) == "OK/SW-CL.58" and common.probe_gene("RNF207 .1", {"RNF207"}) == "RNF207"


def check_luad(root: Path) -> None:
    """The LUAD data manifest: a .gz file is checked by its decompressed content, so one rewritten from the same data
    (another time in its gzip header) passes, while a changed or missing file stops the check."""
    saved = common.LUAD, common.XENA_EXPRESSION
    common.LUAD = root / "luad" / "cohorts" / "luad"
    common.XENA_EXPRESSION = common.LUAD / "TCGA-LUAD" / "raw" / "HiSeqV2.gz"
    try:
        files = common.luad_data_files()
        for index, path in enumerate(files):
            path.parent.mkdir(parents=True, exist_ok=True)
            text = f"sample_id,G{index}\nS1,{index}.5\n".encode("utf-8")
            if path.suffix == ".gz":
                with gzip.GzipFile(path, "wb", mtime=1) as handle:
                    handle.write(text)
            else:
                path.write_bytes(text)
        manifest = root / "luad" / "manifest.csv"
        common.luad_manifest().to_csv(manifest, index=False)
        assert common.check_luad_manifest(manifest) == len(files) == 9
        rewritten = files[-1]
        content = gzip.decompress(rewritten.read_bytes())
        with gzip.GzipFile(rewritten, "wb", mtime=2) as handle:
            handle.write(content)
        assert common.sha256_file(rewritten) != pd.read_csv(manifest).set_index("file").at["luad/GSE72094/expression_genes.csv.gz", "sha256"]
        assert common.check_luad_manifest(manifest) == 9, "a .gz file rewritten from the same data passes"
        for change in (lambda: rewritten.write_bytes(gzip.compress(content + b"S2,1.5\n")), lambda: files[1].unlink()):
            change()
            try:
                common.check_luad_manifest(manifest)
            except RuntimeError:
                rewritten.write_bytes(gzip.compress(content))
                continue
            raise AssertionError("a changed or missing LUAD file was accepted")
    finally:
        common.LUAD, common.XENA_EXPRESSION = saved


# Worked examples of random_effects, computed with R 4.5.2 and metafor 5.0.1: rma(yi, sei = sei, method = "DL") for the
# estimate, tau2 and Q; test = "adhoc" for the HKSJ interval with the ad hoc correction and test = "knha" without it;
# predict(rma(yi, sei = sei, method = "DL"), pi.type = "Riley") for the prediction interval (t with k - 2 df).
METAFOR = [
    {"yi": [0.10, 0.30, 0.35, -0.05], "sei": [0.10, 0.12, 0.15, 0.08], "estimate": 0.151070476076309, "tau2": 0.023433736759435,
     "q": 9.064584636230977, "adhoc": (-0.150139119694713, 0.452280071847331), "knha": (-0.144101495749406, 0.446242447902024),
     "pi": (-0.623309327852450, 0.925450280005069)},
    {"yi": [0.20, 0.21, 0.19], "sei": [0.10, 0.10, 0.10], "estimate": 0.2, "tau2": 0.0, "q": 0.02,
     "adhoc": (-0.048413771175033, 0.448413771175033), "knha": (0.175158622882497, 0.224841377117503),
     "pi": (-0.533593072480896, 0.933593072480896)},
    {"yi": [0.1, 0.3], "sei": [0.1, 0.1], "estimate": 0.2, "tau2": 0.01, "q": 2.0,
     "adhoc": (-1.070620473617471, 1.470620473617471), "knha": (-1.070620473617471, 1.470620473617471), "pi": (None, None)},
]


def check_pooling() -> None:
    from scipy import stats

    pooled = common.random_effects(np.array([0.1, 0.3]), np.array([0.1, 0.1]))
    assert np.isclose(pooled["tau2"], 0.01) and np.isclose(pooled["estimate"], 0.2) and np.isclose(pooled["ci_upper"] - 0.2, 0.196)
    assert common.random_effects(np.array([0.2, 0.2, 0.2]), np.array([0.1, 0.2, 0.3]))["tau2"] == 0.0
    for estimates, errors in (([0.1], [0.1]), ([0.1, np.nan], [0.1, 0.1]), ([0.1, 0.2], [0.0, 0.1]), ([0.1, 0.2], [np.nan, 0.1]),
                              ([0.1, 0.2], [0.1])):
        try:
            common.random_effects(np.array(estimates), np.array(errors))
        except ValueError:
            continue
        raise AssertionError(f"random_effects accepted {estimates} with {errors}")
    # The HKSJ interval (with and without the ad hoc correction) and the prediction interval against metafor.
    for example in METAFOR:
        pooled = common.random_effects(np.array(example["yi"]), np.array(example["sei"]))
        k = len(example["yi"])
        assert pooled["k"] == k and all(np.isclose(pooled[key], example[key], rtol=0, atol=1e-12) for key in ("estimate", "tau2", "q")), (example, pooled)
        assert np.allclose([pooled["hksj_ci_lower"], pooled["hksj_ci_upper"]], example["adhoc"], rtol=0, atol=1e-12), (example["adhoc"], pooled)
        se = (pooled["ci_upper"] - pooled["ci_lower"]) / 3.92
        unmodified = stats.t.ppf(0.975, k - 1) * se * np.sqrt(pooled["hksj_scale"])
        assert np.allclose([pooled["estimate"] - unmodified, pooled["estimate"] + unmodified], example["knha"], rtol=0, atol=1e-12), (example["knha"], pooled)
        assert pooled["hksj_ci_upper"] - pooled["hksj_ci_lower"] >= pooled["ci_upper"] - pooled["ci_lower"], "never narrower than DerSimonian-Laird"
        if example["pi"][0] is None:
            assert pooled["pi_lower"] is None and pooled["pi_upper"] is None, "no prediction interval with two cohorts"
        else:
            assert np.allclose([pooled["pi_lower"], pooled["pi_upper"]], example["pi"], rtol=0, atol=1e-12), (example["pi"], pooled)
    ratio = common.exponentiated(common.random_effects(np.array([0.1, 0.3]), np.array([0.1, 0.1])))
    assert np.isclose(ratio["estimate"], np.exp(0.2)) and ratio["pi_lower"] is None and ratio["hksj_ci_lower"] < ratio["ci_lower"]


def uno_by_hand(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    """Uno's C written out for distinct times: pairs whose earlier time is an event before the horizon, weighted by
    1 / G(t)^2 with G the Kaplan-Meier estimate of the censoring distribution, tied risks counting one half."""
    order = np.argsort(time)
    at_risk = time.size - np.arange(time.size)
    censoring = np.cumprod(np.where(event[order] == 0, 1.0 - 1.0 / at_risk, 1.0))
    survival = dict(zip(time[order], censoring))
    numerator = denominator = 0.0
    for i in np.flatnonzero((event == 1) & (time < horizon)):
        weight = 1.0 / survival[time[i]] ** 2
        later = time > time[i]
        denominator += weight * later.sum()
        numerator += weight * (np.sum(later & (risk[i] > risk)) + 0.5 * np.sum(later & (risk[i] == risk)))
    return numerator / denominator


def check_uno() -> None:
    """Uno's C of common.uno_c (scikit-survival) against uno_by_hand, with censoring before and after the horizon and
    risks pointing either way; without censoring and without truncation it is Harrell's C."""
    from survival_toolkit.marker_evaluation import _pooled_c_index

    draw = np.random.default_rng(11)  # its own stream: the checks after it see the same random data as before
    n = 80
    risk = draw.normal(size=n)
    times, censor = draw.exponential(30.0 / np.exp(risk)), draw.uniform(5, 90, n)
    time, event = np.minimum(times, censor), (times <= censor).astype(int)
    for horizon in (np.quantile(time, 0.6), np.inf):
        for score in (risk, -risk, draw.normal(size=n)):
            assert np.isclose(common.uno_c(time, event, score, horizon), uno_by_hand(time, event, score, horizon), rtol=0, atol=1e-10), horizon
    everyone = np.ones(n, dtype=int)
    assert np.isclose(common.uno_c(times, everyone, risk, np.inf), _pooled_c_index(times, everyone, risk, None), rtol=0, atol=1e-12)
    assert np.isnan(common.uno_c(time, event, risk, float(time.min()) / 2)), "no event before the horizon gives no value"


def check_validation_rows() -> None:
    def report(c: float, clinical: float, width: float) -> dict:
        return {"cohort": {"n": 100, "events": 40}, "markers": [], "metrics": {
            "marker_weight_available": 1.0, "absent_markers": [], "c_index": c, "c_index_ci": [c - 0.05, c + 0.05],
            "clinical_only_c_index": clinical, "clinical_only_c_index_ci": [clinical - width, clinical + width],
            "delta_c_index": c - clinical, "delta_c_index_ci": [c - clinical - 0.03, c - clinical + 0.03],
            "calibration_slope": 10 * (c - clinical), "calibration_slope_ci": [10 * (c - clinical) - 5 * width, 10 * (c - clinical) + 5 * width]}}

    rows = pd.DataFrame([common.validation_row(report(c, clinical, width)) for c, clinical, width in ((0.70, 0.66, 0.02), (0.64, 0.61, 0.08), (0.68, 0.60, 0.04))])
    assert rows.loc[1, "clinical_lower"] == 0.61 - 0.08 and rows.loc[1, "clinical_upper"] == 0.61 + 0.08
    assert rows.loc[1, "calibration_lower"] == 10 * (0.64 - 0.61) - 5 * 0.08, "the calibration slope keeps its own interval"
    # component_row's columns: the clinical-only calibration slope with its interval, the gene log hazard ratio with its SE.
    rows = rows.assign(clinical_calibration_slope=[0.9, 1.1, 1.0], clinical_calibration_lower=[0.5, 0.4, 0.8],
                       clinical_calibration_upper=[1.3, 1.8, 1.2], gene_log_hr=[0.2, 0.05, 0.3], gene_log_hr_se=[0.1, 0.12, 0.08])
    pooled = common.pooled_validation(rows)
    expected = common.random_effects(rows["clinical_c"].to_numpy(), (rows["clinical_upper"] - rows["clinical_lower"]).to_numpy() / 3.92)
    assert pooled["clinical_c"] == expected, "the clinical-only C is pooled with its own intervals"
    slope = common.random_effects(rows["calibration_slope"].to_numpy(), (rows["calibration_upper"] - rows["calibration_lower"]).to_numpy() / 3.92)
    assert pooled["calibration_slope"] == slope, "the calibration slope is pooled with its own intervals"
    assert pooled["model_c"] != common.random_effects(rows["c"].to_numpy(), (rows["clinical_upper"] - rows["clinical_lower"]).to_numpy() / 3.92)
    assert pooled["clinical_calibration_slope"] == common.random_effects(
        rows["clinical_calibration_slope"].to_numpy(), (rows["clinical_calibration_upper"] - rows["clinical_calibration_lower"]).to_numpy() / 3.92)
    gene = common.random_effects(np.array([0.2, 0.05, 0.3]), np.array([0.1, 0.12, 0.08]))
    assert pooled["gene_log_hr"] == gene and pooled["gene_hazard_ratio"] == common.exponentiated(gene), "the gene component is pooled with its SE"
    assert all(key in pooled["delta_c"] for key in ("hksj_ci_lower", "hksj_ci_upper", "pi_lower", "pi_upper", "k"))

    # The hazard ratio shown is the one of the lens SurvStudio tested the marker on.
    fit = lambda ratio: {"hazard_ratio": ratio, "ci_lower": ratio / 2, "ci_upper": ratio * 2}  # noqa: E731
    markers = [{"marker": "A", "marginal": fit(1.5), "adjusted": fit(1.2), "same_direction": True, "tested": "added_value"},
               {"marker": "B", "marginal": fit(1.5), "adjusted": fit(1.2), "same_direction": True, "tested": "marginal"},
               {"marker": "C", "marginal": fit(1.5), "adjusted": None, "same_direction": False, "tested": None},
               {"marker": "D", "marginal": None, "adjusted": None, "same_direction": False, "absent": True, "tested": None}]
    rows = common.validation_marker_rows({"markers": markers})
    assert [row["hazard_ratio"] for row in rows] == [1.2, 1.5, None, None] and [row["measured"] for row in rows] == [True, True, True, False]
    try:
        common.validation_marker_rows({"markers": [{**markers[0], "tested": "nonlinear"}]})
    except RuntimeError:
        pass
    else:
        raise AssertionError("an unknown tested lens was accepted")


def check_tier_replication() -> None:
    """Script 15: the clinically adjusted one-step log hazard ratio per SD of each measured gene, its sign, and the
    pooled replication decision in the development direction."""
    from survival_toolkit.marker_screen import fit_cox

    tiers = importlib.import_module("15_tier_replication")
    n = 500
    age, grade = rng.uniform(40, 80, n), rng.choice(["G1", "G2", "G3"], n)
    genes = pd.DataFrame({"UP": rng.normal(size=n), "DOWN": rng.normal(size=n), "NULL": rng.normal(size=n), "FLAT": np.ones(n),
                          "SPARSE": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n))}, index=[f"P{index}" for index in range(n)])
    risk = 0.02 * (age - 60) + 0.4 * (grade == "G3") + 0.5 * genes["UP"].to_numpy() - 0.5 * genes["DOWN"].to_numpy()
    times, censor = rng.exponential(12.0 / np.exp(risk)), rng.uniform(1, 40, n)
    clinical = pd.DataFrame({"patient_id": genes.index, "time": np.minimum(times, censor), "event": (times <= censor).astype(int),
                             "age": age, "grade": grade})
    table = tiers.cohort_statistics(clinical, genes, ["age", "grade"], ["grade"], ["UP", "DOWN", "NULL", "FLAT", "SPARSE"]).set_index("gene")
    assert set(table.index) == {"UP", "DOWN", "NULL"}, "a gene missing in over 20% of patients, or constant, is not measured"
    assert table.at["UP", "log_hr"] > 0 > table.at["DOWN", "log_hr"] and int(table["n"].iat[0]) == n
    design = pd.get_dummies(clinical[["age", "grade"]], columns=["grade"], drop_first=True, dtype=float).to_numpy()
    for gene in ("UP", "DOWN"):
        values = genes[gene].to_numpy()
        full = fit_cox(clinical["time"].to_numpy(), clinical["event"].to_numpy(), np.column_stack([design, (values - values.mean()) / values.std(ddof=1)]))
        mle, se = float(full.beta[-1]), float(np.sqrt(full.covariance[-1, -1]))
        assert abs(table.at[gene, "log_hr"] - mle) < 0.15 and 0.75 < table.at[gene, "se"] / se < 1.33, (gene, table.loc[gene].to_dict(), mle, se)

    development = pd.DataFrame({"tier": ["robust", "suggestive", "marginal only", "not supported", "not supported"],
                                "added_value_p_value": [0.001, 0.01, 0.2, 0.03, 0.2]})
    assert tiers.group_of(development).tolist() == ["robust", "suggestive", "marginal only", "nominal only", "no evidence"]
    part = lambda log_hr, se, direction: pd.DataFrame({"log_hr": log_hr, "se": se, "group": "robust", "direction": float(direction),  # noqa: E731
                                                       "same_direction": np.sign(log_hr) == direction})
    up = tiers.gene_record("V", "UP", part([0.30, 0.40], [0.10, 0.12], 1))
    assert up["replicated"] and up["signed_log_hr"] > 0 and up["cohorts"] == 2 and up["same_direction_share"] == 1.0
    against = tiers.gene_record("V", "UP", part([0.30, 0.40], [0.10, 0.12], -1))
    assert not against["replicated"] and against["signed_log_hr"] < 0, "an effect against the development direction does not replicate"
    assert not tiers.gene_record("V", "X", part([0.20, -0.10], [0.15, 0.15], 1))["replicated"], "a CI that includes zero does not replicate"
    assert "replicated" not in tiers.gene_record("V", "X", part([0.30], [0.10], 1)), "one cohort is not evaluable"


def check_resume(root: Path) -> None:
    stamp = {"survstudio_commit": "abc1234", "design_hash": "d1"}
    previous = pd.DataFrame({
        "scenario": ["s", "s", "s", "s", "s", "s"], "replicate": [0, 1, 2, 3, 9, 0], "error": [np.nan, "Boom", np.nan, np.nan, np.nan, np.nan],
        "survstudio_commit": ["abc1234", "abc1234", "old5678", "abc1234", "abc1234", "abc1234"],
        "design_hash": ["d1", "d1", "d1", "d0", "d1", "d1"], "value": [1, 2, 3, 4, 5, 6]})
    wanted = {("s", index) for index in range(5)}
    kept = common.resumable_rows(previous, ["scenario", "replicate"], wanted, stamp)
    assert kept["value"].tolist() == [6], kept
    assert common.resumable_rows(previous, ["scenario", "replicate"], wanted, {**stamp, "survstudio_commit": "abc1234-dirty"}).empty
    assert common.resumable_rows(previous.drop(columns="design_hash"), ["scenario", "replicate"], wanted, stamp).empty
    path = root / "table.csv"
    common.write_csv_atomic(previous, path)
    assert pd.read_csv(path).shape == previous.shape and not (root / "table.csv.tmp").exists()

    # Stamps that look like numbers ("0123456" as an integer, "12e4567" as infinity) survive the CSV round trip when
    # read as text: the resumed run keeps the replicates and the summary accepts them.
    summary = importlib.import_module("06_simulation_summary")
    for commit, design, code in (("0123456", "12e4567", "5e10abc"), ("1234567", "0000001", "12e4567")):
        stamp = {"survstudio_commit": commit, "design_hash": design, "code_hash": code}
        replicates = pd.DataFrame({"scenario": ["s", "s"], "replicate": [0, 1], "beta": [0.0, 0.0], "max_mode_fraction": [0.9, 0.9],
                                   "events": [40, 41], "tested": [100, 100], "fwer_false": [0, 1], "error": [np.nan, np.nan],
                                   "new_patients_c": [np.nan, np.nan], **{column: [value] * 2 for column, value in stamp.items()}})
        settings = {"survstudio": {"version": "x", "commit": commit}, "design_hash": design, "code_hash": code,
                    "scenarios": {"s": {"beta": 0.0, "max_mode_fraction": 0.9, "subsamples": 0, "replicates": 2}}}
        common.write_csv_atomic(replicates, path)
        back = pd.read_csv(path, dtype=common.STAMP_DTYPES)
        assert len(common.resumable_rows(back, ["scenario", "replicate"], {("s", 0), ("s", 1)}, stamp)) == 2, stamp
        summary.checked(back, settings)
        assert summary.summarise(back, settings)[0]["fwer"] == 0.5
        try:
            summary.checked(pd.read_csv(path), settings)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"stamps {stamp} read as numbers still passed; the check is not testing the round trip")


def check_gain_verdicts() -> None:
    """The simulation summary's reading of the paired left-out gain: the old rule (gain below 0.02), and from the gain's
    interval the coverage of the true gain (0 under the null) and one verdict per replicate, "adds little" (upper
    limit below 0.02), else "adds" (lower limit above 0), else "uncertain"; without intervals, only the old rule."""
    summary = importlib.import_module("06_simulation_summary")
    scored = pd.DataFrame({"left_out_gain": [0.01, 0.03, 0.05, -0.01, 0.01], "left_out_gain_lower": [-0.01, 0.005, 0.01, -0.03, 0.004],
                           "left_out_gain_upper": [0.015, 0.05, 0.08, 0.01, 0.019], "new_patients_c": 0.7, "new_patients_clinical_c": 0.68,
                           "apparent_c": 0.75, "corrected_c": 0.69, "left_out_c": 0.7, "clinical_left_out_c": 0.69})
    zero = summary.gain_verdicts(scored, pd.Series(0.0, index=scored.index))
    assert zero["gain_interval_replicates"] == 5 and np.isclose(zero["gain_coverage"], 0.4) and np.isclose(zero["old_rule_adds_little"], 0.6)
    assert np.isclose(zero["verdict_adds_little"], 0.6) and np.isclose(zero["verdict_adds"], 0.4) and zero["verdict_uncertain"] == 0.0, zero
    uncertain = summary.gain_verdicts(scored.assign(left_out_gain_upper=0.04), pd.Series([0.02, 0.0, 0.05, -0.05, 0.03], index=scored.index))
    assert np.isclose(uncertain["verdict_uncertain"], 0.4) and np.isclose(uncertain["verdict_adds"], 0.6), uncertain
    assert np.isclose(uncertain["gain_coverage"], 0.4), "each replicate's interval against its own true gain"
    bare = summary.gain_verdicts(scored.drop(columns=["left_out_gain_lower", "left_out_gain_upper"]), pd.Series(0.0, index=scored.index))
    assert bare["gain_interval_replicates"] == 0 and bare["gain_coverage"] is None and np.isclose(bare["old_rule_adds_little"], 0.6)
    # A null scenario with subsamples is scored against a true gain of 0; an alternative against its gain in new patients.
    replicates = scored.assign(scenario="n", replicate=range(5), beta=0.0, max_mode_fraction=0.9, events=40, tested=100, fwer_false=0, error=np.nan)
    settings = {"scenarios": {"n": {"beta": 0.0, "max_mode_fraction": 0.9, "subsamples": 100, "replicates": 5, "true_genes": 0}}}
    row = summary.summarise(replicates, settings)[0]
    assert row["gain_target"] == "zero" and np.isclose(row["gain_coverage"], 0.4) and np.isclose(row["true_gain_mean"], 0.02)


def check_stamps(root: Path) -> None:
    """The stamps figures.py checks: a file changed after its run, a result computed from an older input, and files
    of different runs are each refused."""
    results = root / "results"
    results.mkdir()
    saved = common.RESULTS
    common.RESULTS = results
    common._INPUTS.clear()
    try:
        common.write_json(results / "upstream.json", {"value": 1})
        common.read_result("upstream.json")
        common.write_csv_atomic(pd.DataFrame({"x": [1]}), results / "downstream.csv")
        problems = common.result_problems(["upstream.json", "downstream.csv"])
        assert problems == [], problems
        stamp = json.loads((results / "stamps" / "downstream.csv.json").read_text(encoding="utf-8"))
        assert stamp["analysis_code"] == common.ANALYSIS_CODE and set(stamp["inputs"]) == {"upstream.json"}
        common.write_json(results / "upstream.json", {"value": 2})  # the upstream step rerun, the downstream one not
        assert any("computed from an earlier upstream.json" in problem for problem in common.result_problems(["downstream.csv"]))
        common._INPUTS.clear()
        common.read_result("upstream.json")
        common.write_csv_atomic(pd.DataFrame({"x": [2]}), results / "downstream.csv")
        problems = common.result_problems(["upstream.json", "downstream.csv"])
        assert problems == [], problems
        (results / "downstream.csv").write_text("x\n3\n", encoding="utf-8")
        assert any("changed after" in problem for problem in common.result_problems(["downstream.csv"]))
        common.write_csv_atomic(pd.DataFrame({"x": [3]}), results / "downstream.csv")
        path = results / "stamps" / "upstream.json.json"
        path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "analysis_code": "other"}), encoding="utf-8")
        assert any("different runs" in problem for problem in common.result_problems(["upstream.json", "downstream.csv"]))
        (results / "unstamped.csv").write_text("x\n1\n", encoding="utf-8")
        assert any("no stamp" in problem for problem in common.result_problems(["unstamped.csv"]))
    finally:
        common.RESULTS = saved
        common._INPUTS.clear()


def check_survstudio() -> None:
    """Against SurvStudio itself, with a locked model it makes here: the marker-weight rule of the external screens, the
    clinical-only interval and paired left-out gain the scripts read, the validation rows, and the locked linear
    predictor behind the clinical-only calibration slope, the gene component and Uno's C."""
    from survival_toolkit.marker_evaluation import MIN_MARKER_WEIGHT_AVAILABLE, MarkerSettings, evaluate_markers, validate_locked_recipe
    from survival_toolkit.marker_screen import fit_cox

    assert common.MIN_MARKER_WEIGHT == MIN_MARKER_WEIGHT_AVAILABLE

    def patients(n: int, markers: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(rng.normal(size=(n, len(markers))), columns=markers)
        frame["age"], frame["grade"] = rng.uniform(35, 80, n), rng.choice(["G1", "G2", "G3"], n)
        risk = 0.8 * frame["M1"] - 0.6 * frame["M2"] + 0.5 * frame["M3"] + 0.3 * (frame["grade"] == "G3")
        times, censor = rng.exponential(30.0 / np.exp(risk)), rng.uniform(5, 120, n)
        frame["os_months"], frame["os_event"] = np.minimum(times, censor), (times <= censor).astype(int)
        return frame

    markers = [f"M{index}" for index in range(1, 9)]
    result = evaluate_markers(patients(300, markers), time_column="os_months", event_column="os_event", marker_columns=markers,
                              clinical_columns=["age", "grade"], categorical_clinical=["grade"], event_positive_value=1,
                              settings=MarkerSettings(n_permutations=20, n_resamples=6, random_seed=7))
    recipe = result["locked_recipe"]
    assert recipe and len(recipe["markers"]) >= 2, "SurvStudio locked no model with two or more markers"
    assert result["signature"].get("signature_gain_left_out") is not None, "no paired left-out gain (scripts 06 and 14, figures)"
    coefficient = dict(zip(recipe["model"]["terms"], recipe["model"]["coefficients"]))
    weight = {marker: abs(coefficient[marker]) * recipe["marker_scale"][marker]["sd"] for marker in recipe["markers"]}
    lightest = min(weight, key=weight.get)
    external = patients(200, markers).drop(columns=lightest)
    report = validate_locked_recipe(external, recipe, marker_scaling="within_cohort", n_bootstrap=25)
    share, absent = common.marker_weight_measured(recipe, external.columns)
    assert absent == [lightest] and abs(share - report["metrics"]["marker_weight_available"]) < 1e-12, (share, report["metrics"])
    interval = report["metrics"].get("clinical_only_c_index_ci") or [None, None]
    assert all(value is not None and np.isfinite(value) for value in interval), "no clinical-only C interval (scripts 03, 08 and 10)"
    row = common.validation_row(report)
    assert row["n"] == 200 and row["clinical_lower"] <= row["clinical_c"] <= row["clinical_upper"]
    assert row["calibration_lower"] < row["calibration_slope"] < row["calibration_upper"], "no calibration slope interval (scripts 03, 08 and 10)"
    rows = {item["marker"]: item for item in common.validation_marker_rows(report)}
    fits = {item["marker"]: item for item in report["markers"]}
    for marker in recipe["markers"]:
        if marker == lightest:
            assert not rows[marker]["measured"] and rows[marker]["hazard_ratio"] is None
        else:
            assert fits[marker]["tested"] == "added_value" and rows[marker]["hazard_ratio"] == fits[marker]["adjusted"]["hazard_ratio"]

    # The linear predictor the extra per-cohort quantities come from is SurvStudio's own, rescaled and as measured
    # (locked_parts stops otherwise, and does stop on a report it does not reproduce); it is the sum of its two parts.
    for scaling in ("within_cohort", "as_measured"):
        scored = report if scaling == "within_cohort" else validate_locked_recipe(external, recipe, marker_scaling=scaling, n_bootstrap=0)
        parts = common.locked_parts(external, recipe, scored, scaling)
        assert np.allclose(parts["clinical"] + parts["marker"], parts["linear_predictor"], rtol=0, atol=1e-10), scaling
    tampered = {**report, "metrics": {**report["metrics"], "c_index": report["metrics"]["c_index"] + 0.01}}
    try:
        common.locked_parts(external, recipe, tampered, "within_cohort")
    except RuntimeError:
        pass
    else:
        raise AssertionError("locked_parts accepted a C-index it does not reproduce")
    parts = common.locked_parts(external, recipe, report, "within_cohort")
    component = common.component_row(parts)
    clinical_fit = fit_cox(parts["time"], parts["event"], parts["clinical_only"][:, None])
    assert np.isclose(component["clinical_calibration_slope"], clinical_fit.beta[0])
    assert component["clinical_calibration_lower"] < clinical_fit.beta[0] < component["clinical_calibration_upper"]
    assert np.isclose(component["lp_sd"], np.std(parts["linear_predictor"], ddof=1)) and component["clinical_lp_sd"] > 0
    assert component["gene_hr_lower"] > 1.0, "the markers carry the simulated risk beyond the clinical part"
    assert np.isclose(np.log(component["gene_hr"]), component["gene_log_hr"]) and component["gene_log_hr_se"] > 0
    uno = common.uno_row(parts, horizon=float(np.quantile(parts["time"], 0.7)), draws=40)
    assert uno["n"] == 200 and uno["draws_used"] >= 20 and uno["uno_lower"] <= uno["uno_upper"], uno
    assert np.isclose(uno["uno_delta"], uno["uno_c"] - uno["clinical_uno_c"]) and uno["uno_delta_lower"] <= uno["uno_delta_upper"]

    heaviest = max(weight, key=weight.get)
    lacking = patients(200, markers).drop(columns=[marker for marker in recipe["markers"] if marker != heaviest] if len(recipe["markers"]) > 2
                                          else [heaviest])
    share, absent = common.marker_weight_measured(recipe, lacking.columns)
    declined = False
    try:
        validate_locked_recipe(lacking, recipe, marker_scaling="within_cohort", n_bootstrap=0)
    except ValueError:
        declined = True
    assert declined == bool(absent and (len(absent) == len(recipe["markers"]) or share < common.MIN_MARKER_WEIGHT)), (share, absent, declined)


def main() -> None:
    breast, paper = common.BREAST, common.PAPER
    pinned = common.CBIOPORTAL_METABRIC_SHA256, common.CBIOPORTAL_METABRIC_PATIENTS
    try:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synthetic_export(root)
            check_breast(root)
            check_probes(root)
            check_luad(root)
            check_resume(root)
            check_stamps(root)
    finally:
        common.BREAST, common.PAPER = breast, paper
        common.CBIOPORTAL_METABRIC_SHA256, common.CBIOPORTAL_METABRIC_PATIENTS = pinned
    check_pooling()
    check_validation_rows()
    check_tier_replication()
    check_gain_verdicts()
    check_survstudio()
    check_uno()
    files = common.check_breast_manifest()
    print(f"breast data: {files} files match {common.BREAST_MANIFEST.name}")
    files = common.check_luad_manifest()
    print(f"LUAD data: {files} files match {common.LUAD_MANIFEST.name} (the .gz files by their decompressed content)")
    snapshot = common.check_snapshot()
    print(f"data snapshot: {', '.join(snapshot) or 'no files'} match their SHA-256")
    print("self-check passed")


if __name__ == "__main__":
    main()

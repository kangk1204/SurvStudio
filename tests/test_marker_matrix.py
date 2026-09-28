from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.errors import NotFoundError, UserInputError
from survival_toolkit.marker_matrix import (
    MarkerMatrixStore,
    id_key,
    match_summary,
    matrix_format,
    matrix_frame,
    read_marker_matrix,
)


def _clinical(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "patient_id": [f"P{index:03d}" for index in range(n)],
            "os_time": rng.exponential(10, size=n),
            "os_event": rng.integers(0, 2, size=n),
            "age": rng.normal(60, 8, size=n),
        }
    )


def _expression(patients: list[str], n_genes: int = 5, seed: int = 2) -> pd.DataFrame:
    """One row per gene and one column per patient, as GEO and TCGA distribute expression."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_genes, len(patients))), index=[f"GENE{index}" for index in range(n_genes)], columns=patients)


def test_markers_in_rows_are_detected_and_matched_by_patient_id(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    genes = _expression(patients[5:] + ["EXTRA1", "EXTRA2"])
    path = tmp_path / "expression.tsv"
    genes.to_csv(path, sep="\t", index_label="gene")

    matrix = read_marker_matrix(path, "expression.tsv", patient_ids=patients)

    assert matrix.orientation == "markers_in_rows"
    assert matrix.marker_names == tuple(f"GENE{index}" for index in range(5))
    assert matrix.values.dtype == np.float32 and matrix.values.shape == (27, 5)
    summary = match_summary(matrix, patients)
    assert summary["n_matched"] == 25 and summary["n_unmatched_matrix_samples"] == 2
    assert summary["unmatched_matrix_samples"] == ["EXTRA1", "EXTRA2"]

    frame = matrix_frame(clinical, matrix, id_column="patient_id", columns=["os_time", "os_event", "age"])

    assert list(frame.columns) == ["os_time", "os_event", "age", *matrix.marker_names]
    assert len(frame) == 25
    assert frame["os_time"].iloc[0] == pytest.approx(clinical["os_time"].iloc[5])
    assert frame["GENE3"].iloc[0] == pytest.approx(genes.loc["GENE3", "P005"], rel=1e-6)


def test_patients_in_rows_and_numeric_ids(tmp_path: Path) -> None:
    clinical = _clinical().assign(patient_id=range(100, 130))
    wide = _expression([str(value) for value in range(100, 130)]).T
    path = tmp_path / "wide.csv"
    wide.to_csv(path, index_label="sample")

    matrix = read_marker_matrix(path, "wide.csv", patient_ids=clinical["patient_id"].tolist())

    assert matrix.orientation == "samples_in_rows"
    assert match_summary(matrix, clinical["patient_id"].tolist())["n_matched"] == 30
    assert id_key(101.0) == id_key("101") == id_key(101) == "101"


def test_matrix_problems_are_explained(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    path = tmp_path / "matrix.csv"

    _expression(["X1", "X2"]).to_csv(path, index_label="gene")
    with pytest.raises(UserInputError, match="No patient ID in the matrix matches") as mismatch:
        read_marker_matrix(path, "matrix.csv", patient_ids=patients)
    # Both files' IDs are shown, so a user can see how they differ.
    assert "'P000', 'P001', 'P002'" in str(mismatch.value) and "'X1', 'X2'" in str(mismatch.value) and "'GENE0'" in str(mismatch.value)

    repeated = _expression(patients)
    repeated.index = ["GENE0", "GENE0", "GENE2", "GENE3", "GENE4"]
    repeated.to_csv(path, index_label="gene")
    with pytest.raises(UserInputError, match="repeats these marker names: GENE0"):
        read_marker_matrix(path, "matrix.csv", patient_ids=patients)

    with_text = _expression(patients).astype(object)
    with_text.iloc[1, 3] = "high"
    with_text.to_csv(path, index_label="gene")
    with pytest.raises(UserInputError, match="contain text: P003"):
        read_marker_matrix(path, "matrix.csv", patient_ids=patients)

    path.write_text("gene," + ",".join([patients[0], patients[0]]) + "\nGENE0,1,2\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="repeats these patient IDs"):
        read_marker_matrix(path, "matrix.csv", patient_ids=patients)

    with pytest.raises(UserInputError, match="Unsupported matrix file type"):
        read_marker_matrix(path, "matrix.xlsx", patient_ids=patients)


def test_matrix_frame_refuses_name_clashes_and_too_few_patients(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    genes = _expression(patients)
    genes.index = ["age", "GENE1", "GENE2", "GENE3", "GENE4"]
    path = tmp_path / "matrix.csv"
    genes.to_csv(path, index_label="gene")
    matrix = read_marker_matrix(path, "matrix.csv", patient_ids=patients)

    with pytest.raises(UserInputError, match="same names as the outcome or clinical columns: age"):
        matrix_frame(clinical, matrix, id_column="patient_id", columns=["os_time", "os_event", "age"])
    with pytest.raises(UserInputError, match="Only 5 patients"):
        matrix_frame(clinical.iloc[:5], matrix, id_column="patient_id", columns=["os_time", "os_event"])
    with pytest.raises(UserInputError, match="repeats these values"):
        matrix_frame(pd.concat([clinical, clinical.iloc[:1]]), matrix, id_column="patient_id", columns=["os_time"])


def test_matrix_store_keeps_a_few_matrices_and_expires_them(tmp_path: Path) -> None:
    clinical = _clinical()
    path = tmp_path / "matrix.csv"
    _expression(clinical["patient_id"].tolist()).to_csv(path, index_label="gene")
    matrix = read_marker_matrix(path, "matrix.csv", patient_ids=clinical["patient_id"].tolist())
    store = MarkerMatrixStore(max_items=2)

    first, second, third = (store.add(matrix) for _ in range(3))

    with pytest.raises(NotFoundError):
        store.get(first)
    assert store.get(second) is matrix and store.get(third) is matrix
    store.remove(third)
    with pytest.raises(NotFoundError):
        store.get(third)
    expired = MarkerMatrixStore(ttl_seconds=-1)
    with pytest.raises(NotFoundError):
        expired.get(expired.add(matrix))


def test_gzip_compressed_matrices_are_read(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    genes = _expression(patients)
    text = genes.to_csv(sep="\t", index_label="gene")
    # GEO and UCSC Xena serve gzip-compressed TSV, sometimes without a table suffix (HiSeqV2.gz).
    for name in ("expression.tsv.gz", "HiSeqV2.gz"):
        path = tmp_path / name
        path.write_bytes(gzip.compress(text.encode("utf-8")))
        matrix = read_marker_matrix(path, name, patient_ids=patients)
        assert matrix.filename == name and matrix.orientation == "markers_in_rows"
        assert matrix.values[:, 2] == pytest.approx(genes.loc["GENE2", patients].to_numpy(dtype=np.float32))

    assert matrix_format("a.csv.gz") == (".csv", True) and matrix_format("HiSeqV2.gz") == ("", True)
    assert matrix_format("a.parquet") == (".parquet", False) and matrix_format("a.xlsx") == (".xlsx", False)
    broken = tmp_path / "broken.tsv.gz"
    broken.write_bytes(b"not gzip at all")
    with pytest.raises(UserInputError, match="could not be unpacked"):
        read_marker_matrix(broken, "broken.tsv.gz", patient_ids=patients)


def test_a_full_temporary_disk_is_not_reported_as_a_damaged_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno
    import gzip

    from survival_toolkit import marker_matrix

    archive = tmp_path / "expr.tsv.gz"
    with gzip.open(archive, "wt") as handle:
        handle.write("gene\tP1\tP2\nG1\t1\t2\n")

    class FullDisk:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, chunk):
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: FullDisk())
    with pytest.raises(UserInputError, match="could not be written .*Disk quota exceeded.*TMPDIR"):
        marker_matrix._gunzip(archive, tmp_path / "plain.tsv")


def test_tcga_sample_barcodes_are_matched_to_patient_barcodes(tmp_path: Path) -> None:
    patients = ["TCGA-05-4244", "TCGA-05-4249", "TCGA-05-4250", "TCGA-35-3615"]
    samples = [
        "TCGA-05-4244-01",  # primary tumour
        "TCGA-05-4244-11",  # normal tissue of the same patient
        "TCGA-05-4249-02",  # recurrent tumour, listed before the primary one
        "TCGA-05-4249-01",
        "TCGA-05-4250-01A",
        "TCGA-99-9999-01",  # a patient not in the dataset
    ]
    genes = _expression(samples, n_genes=3)
    path = tmp_path / "HiSeqV2.tsv"
    genes.to_csv(path, sep="\t", index_label="sample")

    matrix = read_marker_matrix(path, "HiSeqV2.tsv", patient_ids=patients)

    assert matrix.orientation == "markers_in_rows"
    assert matrix.sample_keys == ("TCGA-05-4244", "TCGA-05-4249", "TCGA-05-4250")
    assert matrix.values[1] == pytest.approx(genes["TCGA-05-4249-01"].to_numpy(dtype=np.float32))
    assert "used 3 primary tumour (01)" in matrix.id_note
    assert "1 further tumour" in matrix.id_note and "1 normal-tissue" in matrix.id_note
    assert match_summary(matrix, patients)["n_matched"] == 3

    # One row per sample works the same way; IDs already written alike are left alone.
    genes.T.to_csv(path, index_label="sample")
    assert read_marker_matrix(path, "wide.csv", patient_ids=patients).sample_keys == matrix.sample_keys
    exact = read_marker_matrix(path, "wide.csv", patient_ids=samples)
    assert exact.id_note == "" and len(exact.sample_keys) == len(samples)

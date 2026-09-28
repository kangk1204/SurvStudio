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


def test_tcga_sample_types_are_classified_by_range_and_barcodes_by_letters(tmp_path: Path) -> None:
    # Lower-case barcodes in the dataset; 04 and 09 are tumour types, 50 is a cell line.
    patients = ["tcga-ab-0001", "tcga-ab-0002", "tcga-ab-0003", "tcga-ab-0004"]
    samples = ["TCGA-AB-0001-09A", "TCGA-AB-0002-04", "TCGA-AB-0003-50", "TCGA-AB-0003-01", "TCGA-AB-0004-20"]
    path = tmp_path / "tcga.tsv"
    _expression(samples, n_genes=3).to_csv(path, sep="\t", index_label="sample")

    matrix = read_marker_matrix(path, "tcga.tsv", patient_ids=patients)

    # The dataset's own spelling is kept, so its rows match.
    assert matrix.sample_keys == ("tcga-ab-0001", "tcga-ab-0002", "tcga-ab-0003")
    assert match_summary(matrix, patients)["n_matched"] == 3
    assert "1 recurrent blood cancer (04)" in matrix.id_note and "(09)" in matrix.id_note
    assert "1 other" in matrix.id_note and "1 control" in matrix.id_note


def test_patient_ids_with_leading_zeros_match_the_numbers_the_loader_reads(tmp_path: Path) -> None:
    from survival_toolkit.analysis import load_dataframe

    ids = [f"{index:05d}" for index in range(1, 31)]
    clinical = load_dataframe(_clinical().assign(patient_id=ids).to_csv(index=False).encode("utf-8"), "clinical.csv")
    # The loader reads "00001" as the number 1 ...
    assert clinical["patient_id"].iloc[0] == 1
    genes = _expression(ids)
    for layout, frame in (("markers_in_rows", genes), ("samples_in_rows", genes.T)):
        path = tmp_path / f"{layout}.csv"
        frame.to_csv(path, index_label="id")
        # ... while the matrix keeps the text; both sides compare as the same key.
        matrix = read_marker_matrix(path, "m.csv", patient_ids=clinical["patient_id"].tolist())
        assert matrix.orientation == layout
        assert matrix.sample_keys[:2] == ("1", "2")
        assert match_summary(matrix, clinical["patient_id"].tolist())["n_matched"] == 30
    assert id_key("00101") == id_key(101) == id_key(101.0) == id_key("101.0") == id_key(" 101 ") == "101"
    assert id_key("A0101") == "A0101" and id_key("1.5") == "1.5" and id_key(pd.NA) is None and id_key(" ") is None


def test_parquet_matrices_keep_a_stored_pandas_index(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    genes = _expression(patients, n_genes=6).round(3)
    genes.index.name = "gene"

    # to_parquet stores the index: here the gene names, one row per gene.
    genes.to_parquet(tmp_path / "indexed.parquet")
    matrix = read_marker_matrix(tmp_path / "indexed.parquet", "indexed.parquet", patient_ids=patients)
    assert matrix.orientation == "markers_in_rows"
    assert matrix.marker_names == tuple(f"GENE{index}" for index in range(6))
    assert matrix.sample_keys == tuple(patients)
    assert matrix.values[0] == pytest.approx(genes["P000"].to_numpy(dtype=np.float32))

    # One row per patient with the patient IDs as an unnamed index.
    by_patient = genes.T
    by_patient.index.name = None
    by_patient.to_parquet(tmp_path / "patients.parquet")
    matrix = read_marker_matrix(tmp_path / "patients.parquet", "patients.parquet", patient_ids=patients)
    assert matrix.orientation == "samples_in_rows" and matrix.sample_keys == tuple(patients)

    # A filtered frame stores its row numbers as an unnamed integer index; they are not IDs.
    filtered = genes.reset_index().iloc[[0, 2, 3, 5]]
    filtered.to_parquet(tmp_path / "filtered.parquet")
    matrix = read_marker_matrix(tmp_path / "filtered.parquet", "filtered.parquet", patient_ids=patients)
    assert matrix.marker_names == ("GENE0", "GENE2", "GENE3", "GENE5") and len(matrix.sample_keys) == 30

    # Numeric patient IDs in an unnamed index look like row numbers; the error says how to fix that.
    numeric = _expression([str(index) for index in range(100, 130)]).T
    numeric.index = list(range(100, 130))
    numeric.to_parquet(tmp_path / "numeric.parquet")
    with pytest.raises(UserInputError, match="unnamed integer index was read as row numbers"):
        read_marker_matrix(tmp_path / "numeric.parquet", "numeric.parquet", patient_ids=list(range(100, 130)))


def test_r_write_table_matrices_without_a_row_name_header_are_read(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    genes = _expression(patients).round(3)
    # R's write.table default: the header lists the column names only, one field short of each row.
    text = "\t".join(patients) + "\n" + "".join(f"{gene}\t" + "\t".join(f"{value}" for value in row) + "\n" for gene, row in genes.iterrows())
    path = tmp_path / "r_default.tsv"
    path.write_text(text, encoding="utf-8")

    matrix = read_marker_matrix(path, "r_default.tsv", patient_ids=patients)

    assert matrix.orientation == "markers_in_rows"
    assert matrix.sample_keys == tuple(patients) and matrix.marker_names[:2] == ("GENE0", "GENE1")
    assert matrix.values[:, 3] == pytest.approx(genes.loc["GENE3"].to_numpy(dtype=np.float32))


def test_matrices_in_other_text_encodings_are_read(tmp_path: Path) -> None:
    clinical = _clinical()
    patients = clinical["patient_id"].tolist()
    genes = _expression(patients, n_genes=4)
    genes.index = ["GENE0", "CD274 (PD-L1) µg", "Café", "GENE3"]
    for encoding in ("cp1252", "utf-16", "utf-8-sig"):
        path = tmp_path / f"{encoding}.csv"
        path.write_bytes(genes.to_csv(index_label="gene").encode(encoding))
        matrix = read_marker_matrix(path, "m.csv", patient_ids=patients)
        assert matrix.marker_names == ("GENE0", "CD274 (PD-L1) µg", "Café", "GENE3"), encoding
        assert matrix.values[:, 2] == pytest.approx(genes.loc["Café"].to_numpy(dtype=np.float32))


def test_text_matrices_are_bounded_before_pandas_parses_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from survival_toolkit import marker_matrix

    def never(*args, **kwargs):
        raise AssertionError("pandas parsed a matrix that the bound checks should have refused")

    monkeypatch.setattr(marker_matrix.pd, "read_csv", never)
    tall = tmp_path / "tall.tsv"
    tall.write_text("gene\tP000\n" + "g\t1\n" * 250_000, encoding="utf-8")
    with pytest.raises(UserInputError, match="more than 100,000 rows"):
        read_marker_matrix(tall, "tall.tsv", patient_ids=["P000"])
    # Carriage returns alone end lines too (old Mac files), and pandas would read every one of them.
    tall.write_bytes(b"gene\tP000\r" + b"g\t1\r" * 250_000)
    with pytest.raises(UserInputError, match="more than 100,000 rows"):
        read_marker_matrix(tall, "tall.tsv", patient_ids=["P000"])
    # Windows line ends count once each, also where a read chunk splits one; mixed ends all count.
    counted = tmp_path / "counted.tsv"
    counted.write_bytes(b"gene\tP000\r\n" + b"g\t1\r\n" * 300_000)
    assert marker_matrix._count_data_lines(counted) == 100_001
    counted.write_bytes(b"gene\tP\r\n" + b"g\t1\r\n" * 1000 + b"g\t2\r" * 10 + b"g\t3\n" * 5 + b"g\t4")
    assert marker_matrix._count_data_lines(counted) == 1016
    split = tmp_path / "split.tsv"
    split.write_bytes(b"h" * ((1 << 20) - 1) + b"\r\n" + b"g\t1\r\n" * 3)
    assert marker_matrix._count_data_lines(split) == 3
    wide = tmp_path / "wide.tsv"
    wide.write_text("gene\t" + "\t".join(f"P{index}" for index in range(100_001)) + "\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="more than 100,000 data columns"):
        read_marker_matrix(wide, "wide.tsv", patient_ids=["P0"])
    monkeypatch.setattr(marker_matrix, "MAX_MATRIX_CELLS", 500)
    cells = tmp_path / "cells.tsv"
    _expression([f"P{index:03d}" for index in range(30)], n_genes=20).to_csv(cells, sep="\t", index_label="gene")
    with pytest.raises(UserInputError, match="at most 500 values"):
        read_marker_matrix(cells, "cells.tsv", patient_ids=["P000"])

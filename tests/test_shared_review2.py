"""Second review of the marker-matrix reader, the shared feature encoder, the evaluation helpers, the
exception policy and the dataset store."""

from __future__ import annotations

import gzip
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_toolkit import marker_matrix
from survival_toolkit.errors import UserInputError
from survival_toolkit.marker_matrix import match_summary, read_marker_matrix


def _patients(n: int = 30) -> list[str]:
    return [f"P{index:03d}" for index in range(n)]


def _genes(patients: list, n_genes: int = 5, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_genes, len(patients))).round(3),
        index=[f"GENE{index}" for index in range(n_genes)],
        columns=[str(patient) for patient in patients],
    )


# ── Marker matrix: layout detection ──────────────────────────────


def test_numeric_ids_on_both_axes_are_not_guessed(tmp_path: Path) -> None:
    # Entrez gene IDs 1..2000 as row names and sequential numeric patient IDs 1..300: the first 300
    # genes "match" patients, more of them than the 150 real patients in the header.
    rng = np.random.default_rng(0)
    patient_ids = list(range(1, 301))
    samples = patient_ids[::2]
    values = rng.normal(size=(2000, len(samples))).round(3)
    frame = pd.DataFrame(values, index=np.arange(1, 2001), columns=[str(sample) for sample in samples])
    path = tmp_path / "entrez.tsv"
    frame.to_csv(path, sep="\t", index_label="GeneID")

    with pytest.raises(UserInputError, match="Choose the layout") as refused:
        read_marker_matrix(path, "entrez.tsv", patient_ids=patient_ids)
    assert "150 of 150" in str(refused.value) and "300 of 2,000" in str(refused.value)

    matrix = read_marker_matrix(path, "entrez.tsv", patient_ids=patient_ids, orientation="markers_in_rows")
    assert matrix.orientation == "markers_in_rows"
    assert len(matrix.marker_names) == 2000 and matrix.sample_keys == tuple(str(sample) for sample in samples)
    # Patient 3 (the second column), gene 2 (the second row).
    assert matrix.values[1, 1] == pytest.approx(values[1, 1])
    assert match_summary(matrix, patient_ids)["n_matched"] == 150


def test_layout_follows_the_share_of_matching_ids_not_their_count(tmp_path: Path) -> None:
    # 30 patients, all in the dataset, against 5,000 numeric gene IDs of which 40 happen to be patient IDs too.
    rng = np.random.default_rng(1)
    patient_ids = list(range(1, 2001))
    samples = list(range(1001, 1031))
    genes = list(range(1, 41)) + list(range(100_001, 104_961))
    frame = pd.DataFrame(rng.normal(size=(len(genes), len(samples))).round(3), index=genes, columns=[str(sample) for sample in samples])

    by_marker = tmp_path / "by_marker.tsv"
    frame.to_csv(by_marker, sep="\t", index_label="gene")
    matrix = read_marker_matrix(by_marker, "by_marker.tsv", patient_ids=patient_ids)
    assert matrix.orientation == "markers_in_rows" and matrix.sample_keys == tuple(str(sample) for sample in samples)

    by_patient = tmp_path / "by_patient.csv"
    frame.T.to_csv(by_patient, index_label="sample")
    matrix = read_marker_matrix(by_patient, "by_patient.csv", patient_ids=patient_ids)
    assert matrix.orientation == "samples_in_rows" and len(matrix.marker_names) == len(genes)


# ── Marker matrix: header and row shapes ─────────────────────────


def _trailing_separator_text(header: list[str], rows: list[list[str]], *, header_too: bool = False) -> str:
    """Every data line (and optionally the header) ends with a tab, as some writers produce."""
    head = "\t".join(header) + ("\t" if header_too else "")
    return head + "\n" + "".join("\t".join(row) + "\t\n" for row in rows)


def test_rows_ending_with_a_separator_do_not_shift_the_column_names(tmp_path: Path) -> None:
    patients = [f"P{index:02d}" for index in range(12)]
    genes = ["TP53", "EGFR", "KRAS"]
    values = np.random.default_rng(5).integers(0, 1000, size=(len(genes), len(patients)))
    path = tmp_path / "trailing.tsv"

    for header_too in (False, True):
        rows = [[gene, *map(str, row)] for gene, row in zip(genes, values)]
        path.write_text(_trailing_separator_text(["gene", *patients], rows, header_too=header_too), encoding="utf-8")
        matrix = read_marker_matrix(path, "trailing.tsv", patient_ids=patients)
        assert matrix.orientation == "markers_in_rows" and matrix.sample_keys == tuple(patients)
        assert matrix.marker_names == tuple(genes)
        # values holds one row per patient: TP53 of every patient is the first gene row.
        assert matrix.values[:, 0].tolist() == values[0].tolist()
        assert not np.isnan(matrix.values).any()

        rows = [[patient, *map(str, values[:, index])] for index, patient in enumerate(patients)]
        path.write_text(_trailing_separator_text(["id", *genes], rows, header_too=header_too), encoding="utf-8")
        matrix = read_marker_matrix(path, "trailing.tsv", patient_ids=patients)
        assert matrix.orientation == "samples_in_rows" and matrix.marker_names == tuple(genes)
        assert matrix.values[0].tolist() == values[:, 0].tolist()


def test_r_write_table_layout_is_still_read_with_missing_values_in_the_last_column(tmp_path: Path) -> None:
    patients = _patients()
    genes = _genes(patients)
    lines = ["\t".join(patients)]
    for index, (gene, row) in enumerate(genes.iterrows()):
        cells = [f"{value}" for value in row]
        if index in (2, 4):
            cells[-1] = "NA"  # R writes a missing value as NA
        if index == 3:
            cells[-1] = ""  # or, with na = "", as an empty field
        lines.append("\t".join([gene, *cells]))
    path = tmp_path / "r.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    matrix = read_marker_matrix(path, "r.tsv", patient_ids=patients)

    assert matrix.orientation == "markers_in_rows" and matrix.sample_keys == tuple(patients)
    assert matrix.values[0, :].tolist() == pytest.approx(genes[patients[0]].to_numpy(dtype=np.float32))
    assert np.isnan(matrix.values[-1, [2, 3, 4]]).all() and not np.isnan(matrix.values[:-1]).any()


def test_a_header_one_field_short_is_refused_when_the_first_row_ends_with_an_empty_field(tmp_path: Path) -> None:
    patients = _patients()
    genes = _genes(patients)
    lines = ["\t".join(patients)]
    for index, (gene, row) in enumerate(genes.iterrows()):
        cells = [f"{value}" for value in row]
        if index == 0:
            cells[-1] = ""
        lines.append("\t".join([gene, *cells]))
    path = tmp_path / "unclear.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="one field more than the header"):
        read_marker_matrix(path, "unclear.tsv", patient_ids=patients)


def test_rows_with_another_number_of_fields_than_the_header_are_refused(tmp_path: Path) -> None:
    patients = _patients()
    genes = _genes(patients, n_genes=40, seed=1)
    lines = genes.to_csv(index_label="gene").splitlines()
    path = tmp_path / "short.csv"

    # The last line cut off after nine values, as by an interrupted download.
    cut = [*lines[:-1], ",".join(lines[-1].split(",")[:10])]
    path.write_text("\n".join(cut) + "\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="line 41 has 10 fields"):
        read_marker_matrix(path, "short.csv", patient_ids=patients)

    # A short line in the middle, and a long one.
    short = [*lines]
    short[7] = ",".join(short[7].split(",")[:-1])
    path.write_text("\n".join(short) + "\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="line 8 has 30 fields"):
        read_marker_matrix(path, "short.csv", patient_ids=patients)
    long = [*lines]
    long[12] += ",1.5"
    path.write_text("\n".join(long) + "\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="line 13 has 32 fields"):
        read_marker_matrix(path, "short.csv", patient_ids=patients)

    # Blank lines are skipped as pandas skips them, and quoted names may hold separators.
    quoted = [lines[0], *[f'"{line.split(",", 1)[0]}, x",{line.split(",", 1)[1]}' for line in lines[1:]]]
    path.write_text("\n\n".join(quoted) + "\n\n", encoding="utf-8")
    matrix = read_marker_matrix(path, "short.csv", patient_ids=patients)
    assert matrix.marker_names[:2] == ("GENE0, x", "GENE1, x") and matrix.values.shape == (30, 40)


# ── Marker matrix: text encodings and file names ─────────────────


def test_utf16_matrices_count_each_line_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patients = _patients()
    genes = _genes(patients, n_genes=40, seed=1)
    # U+0A0A is written as the bytes 0A 0A in UTF-16: two "line feeds" to a byte counter.
    genes.index = [*genes.index[:-1], "ਊਊ"]
    text = genes.to_csv(sep="\t", index_label="gene", lineterminator="\r\n")
    path = tmp_path / "unicode.txt"
    # Exactly at the limits: counting each "\r\n" twice (and U+0A0A as line ends) would refuse it.
    monkeypatch.setattr(marker_matrix, "MAX_MATRIX_CELLS", 40 * 30)
    monkeypatch.setattr(marker_matrix, "MAX_MATRIX_MARKERS", 40)
    for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
        path.write_bytes(text.encode(encoding))
        assert marker_matrix._count_data_lines(path, encoding) == 40, encoding
        matrix = read_marker_matrix(path, "unicode.txt", patient_ids=patients)
        assert matrix.values.shape == (30, 40) and matrix.marker_names[-1] == "ਊਊ", encoding
        assert matrix.values[:, 3] == pytest.approx(genes.iloc[3].to_numpy(dtype=np.float32))
    # One row over the limit is still refused.
    monkeypatch.setattr(marker_matrix, "MAX_MATRIX_MARKERS", 39)
    with pytest.raises(UserInputError, match="40 markers"):
        read_marker_matrix(path, "unicode.txt", patient_ids=patients)


def test_a_file_name_without_a_suffix_gets_its_separator_from_the_header(tmp_path: Path) -> None:
    patients = _patients()
    genes = _genes(patients, n_genes=3)
    # UCSC Xena names its files without a suffix (HiSeqV2); the separator is read from the first line.
    for name, separator in (("HiSeqV2", "\t"), ("expression", ",")):
        path = tmp_path / name
        genes.to_csv(path, sep=separator, index_label="sample")
        matrix = read_marker_matrix(path, name, patient_ids=patients)
        assert matrix.orientation == "markers_in_rows" and matrix.values.shape == (30, 3), name
        assert matrix.values[:, 1] == pytest.approx(genes.loc["GENE1"].to_numpy(dtype=np.float32))


def test_sniffing_the_separator_reads_a_bounded_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(marker_matrix, "MAX_HEADER_CHARS", 1000)
    path = tmp_path / "one_long_line"
    path.write_bytes(b"x" * 5_000_000)
    tracemalloc.start()
    try:
        assert marker_matrix._sniffed_suffix(path) == ".csv"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    compressed = tmp_path / "HiSeqV2.gz"
    compressed.write_bytes(gzip.compress(("x" * 2000 + "\ty\n").encode("utf-8")))
    with pytest.raises(UserInputError):
        read_marker_matrix(compressed, "HiSeqV2.gz", patient_ids=["P0"])


# ── Marker matrix: Parquet ───────────────────────────────────────


def test_parquet_matrices_are_bounded_by_their_own_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyarrow")
    patients = _patients()
    monkeypatch.setattr(marker_matrix, "MAX_MATRIX_MARKERS", 20)
    wide = _genes(patients, n_genes=5)  # 5 markers and 30 patient columns
    wide.index.name = "gene"
    wide.to_parquet(tmp_path / "wide.parquet")
    matrix = read_marker_matrix(tmp_path / "wide.parquet", "wide.parquet", patient_ids=patients)
    assert matrix.orientation == "markers_in_rows" and matrix.values.shape == (30, 5)

    tall = _genes(patients, n_genes=25)
    tall.index.name = "gene"
    tall.to_parquet(tmp_path / "tall.parquet")
    with pytest.raises(UserInputError, match="25 markers"):
        read_marker_matrix(tmp_path / "tall.parquet", "tall.parquet", patient_ids=patients)


def test_a_parquet_file_with_unreadable_data_is_a_user_error(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    patients = _patients()
    frame = _genes(patients, n_genes=200, seed=4)
    frame.index.name = "gene"
    path = tmp_path / "damaged.parquet"
    frame.to_parquet(path, compression="snappy")
    raw = bytearray(path.read_bytes())
    # Damage the column data between the leading magic bytes and the footer the pre-check reads.
    for offset in range(64, len(raw) // 2, 7):
        raw[offset] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(UserInputError, match="Parquet file could not be read"):
        read_marker_matrix(path, "damaged.parquet", patient_ids=patients)

"""Wide marker matrices (for example RNA-seq) kept apart from the clinical table.

A clinical table holds one row per patient and a manageable number of columns, which every
tab lists and profiles. An omics matrix with thousands of markers is uploaded separately,
stored here, and matched to the patients of whichever dataset the marker evaluation runs
on through an ID column. Two layouts are read: one row per marker with one column per
patient (as GEO and TCGA distribute expression), or one row per patient with one column
per marker. The first column holds the marker names or patient IDs. Text files may be
gzip-compressed, as GEO and UCSC Xena serve them. TCGA sample barcodes (TCGA-05-4244-01A)
are matched to patient barcodes (TCGA-05-4244) when the dataset holds the latter.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
import secrets
import shutil
import tempfile
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd

from survival_toolkit.errors import NotFoundError, UserInputError

MAX_MATRIX_MARKERS = 60_000
MAX_MATRIX_SAMPLES = 100_000
MAX_MATRIX_CELLS = 30_000_000
MATRIX_SUFFIXES = frozenset({".csv", ".tsv", ".txt", ".parquet"})
ORIENTATIONS = ("auto", "markers_in_rows", "samples_in_rows")
# A gzip-compressed text matrix is unpacked to a temporary file first; this bounds what it may unpack to.
MAX_DECOMPRESSED_BYTES = 1024 * 1024 * 1024

_TCGA_PATIENT = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", re.IGNORECASE)
_TCGA_SAMPLE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})([A-Z]?)(?:-.*)?$", re.IGNORECASE)
_TCGA_TUMOUR_TYPES = {"01": "primary tumour", "02": "recurrent tumour", "03": "primary blood cancer", "05": "additional primary", "06": "metastatic", "07": "additional metastatic"}


class MarkerMatrix(NamedTuple):
    filename: str
    orientation: str
    marker_names: tuple[str, ...]
    sample_keys: tuple[str, ...]
    values: np.ndarray  # float32, one row per sample and one column per marker
    fingerprint: str
    id_note: str = ""  # how the matrix IDs were matched when not written as in the dataset


class _SampleMap(NamedTuple):
    """Which samples of the matrix to keep and the patient key each one stands for."""

    positions: list[int]
    keys: list[str]
    note: str


def matrix_format(filename: str) -> tuple[str, bool]:
    """The table suffix of a matrix file name and whether it is gzip-compressed.

    "expr.tsv.gz" gives (".tsv", True); a bare "HiSeqV2.gz" gives ("", True), and its separator is
    then read from the header line.
    """
    suffixes = [suffix.lower() for suffix in Path(filename).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        inner = suffixes[-2] if len(suffixes) >= 2 and suffixes[-2] in MATRIX_SUFFIXES - {".parquet"} else ""
        return inner, True
    return (suffixes[-1] if suffixes else ".csv"), False


def _gunzip(source: Path, target: Path) -> None:
    # A full or over-quota temporary disk must not be reported as a damaged upload, so reading
    # the archive and writing the unpacked copy fail with different messages.
    def write_failed(exc: OSError) -> UserInputError:
        return UserInputError(
            f"The unpacked matrix could not be written to {target.parent} ({exc.strerror or exc}). "
            "Free space there, or set TMPDIR to a folder on a larger disk and restart SurvStudio."
        )

    written = 0
    try:
        plain = target.open("wb")
    except OSError as exc:
        raise write_failed(exc) from exc
    with gzip.open(source, "rb") as packed, plain:
        while True:
            try:
                chunk = packed.read(1 << 20)
            except (OSError, EOFError, zlib.error) as exc:
                raise UserInputError("The .gz file could not be unpacked; is it a gzip-compressed text file?") from exc
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_DECOMPRESSED_BYTES:
                raise UserInputError(f"The compressed matrix unpacks to more than {MAX_DECOMPRESSED_BYTES // 1024 ** 3} GB.")
            try:
                plain.write(chunk)
            except OSError as exc:
                raise write_failed(exc) from exc


def _sniffed_suffix(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline()
    return ".tsv" if "\t" in first else ".csv"


def _tcga_sample_map(ids: Sequence[Any], patient_keys: set[str]) -> _SampleMap | None:
    """TCGA sample barcodes matched to the dataset's patient barcodes: one tumour sample per patient.

    Normal-tissue (10-19) and control (20-29) samples are left out. A patient with several tumour
    samples keeps the lowest sample type (01, primary tumour, first) and then the first vial.
    """
    if not patient_keys or sum(1 for key in patient_keys if _TCGA_PATIENT.match(key)) < 0.8 * len(patient_keys):
        return None
    chosen: dict[str, tuple[str, str, int]] = {}
    left_out: dict[str, int] = {}
    parsed = 0
    for position, value in enumerate(ids):
        match = _TCGA_SAMPLE.match(str(value).strip())
        if not match:
            continue
        parsed += 1
        patient, sample_type, vial = match.group(1).upper(), match.group(2), match.group(3).upper()
        if sample_type not in _TCGA_TUMOUR_TYPES:
            kind = "normal-tissue" if sample_type.startswith("1") else "control"
            left_out[kind] = left_out.get(kind, 0) + 1
            continue
        if patient not in patient_keys:
            continue
        current = chosen.get(patient)
        if current is None or (sample_type, vial) < current[:2]:
            if current is not None:
                left_out["further tumour"] = left_out.get("further tumour", 0) + 1
            chosen[patient] = (sample_type, vial, position)
        else:
            left_out["further tumour"] = left_out.get("further tumour", 0) + 1
    if parsed < 0.8 * max(len(ids), 1) or not chosen:
        return None
    ordered = sorted(chosen.items(), key=lambda item: item[1][2])
    used: dict[str, int] = {}
    for _, (sample_type, _, _) in ordered:
        used[sample_type] = used.get(sample_type, 0) + 1
    used_text = ", ".join(f"{count} {_TCGA_TUMOUR_TYPES[code]} ({code})" for code, count in sorted(used.items()))
    left_text = ", ".join(f"{count} {kind}" for kind, count in sorted(left_out.items()))
    note = f"TCGA sample barcodes were matched to patient barcodes: used {used_text} sample(s)" + (f"; left out {left_text} sample(s)." if left_text else ".")
    return _SampleMap(positions=[item[1][2] for item in ordered], keys=[item[0] for item in ordered], note=note)


def _examples(values: Sequence[Any], limit: int = 3) -> str:
    shown = [str(value) for value in list(values)[:limit]]
    return ", ".join(f"'{value}'" for value in shown) if shown else "(none)"


def id_key(value: Any) -> str | None:
    """Patient IDs compared as text, so 101, 101.0 and "101" match; blanks match nothing."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value).strip()
    return text or None


def _separator(suffix: str) -> str:
    return "\t" if suffix in {".tsv", ".txt"} else ","


def _text_shape(path: Path, separator: str) -> tuple[list[str], int]:
    """Header fields and number of data lines, read without parsing the table."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=separator), [])
    lines = 0
    last = b"\n"
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        lines += 1
    return [field.strip() for field in header], max(lines - 1, 0)


def _check_shape(n_markers: int, n_samples: int) -> None:
    if n_markers > MAX_MATRIX_MARKERS:
        raise UserInputError(f"The matrix has {n_markers:,} markers; SurvStudio reads at most {MAX_MATRIX_MARKERS:,}.")
    if n_samples > MAX_MATRIX_SAMPLES:
        raise UserInputError(f"The matrix has {n_samples:,} patients; SurvStudio reads at most {MAX_MATRIX_SAMPLES:,}.")
    if n_markers * n_samples > MAX_MATRIX_CELLS:
        raise UserInputError(
            f"The matrix has {n_markers * n_samples:,} values; SurvStudio reads at most {MAX_MATRIX_CELLS:,}. "
            "Keep fewer markers, for example the most variable ones."
        )


def _orientation(
    header_ids: Sequence[Any],
    first_column_ids: Sequence[Any],
    patient_keys: set[str],
    requested: str,
    patient_examples: Sequence[Any] = (),
) -> tuple[str, _SampleMap | None]:
    """The layout, and a sample map when the matrix IDs match only as TCGA sample barcodes."""
    in_header = sum(1 for value in header_ids if id_key(value) in patient_keys)
    in_column = sum(1 for value in first_column_ids if id_key(value) in patient_keys)
    if in_header or in_column:
        if requested != "auto":
            return requested, None
        return ("markers_in_rows" if in_header > in_column else "samples_in_rows"), None
    candidates = (("markers_in_rows", header_ids), ("samples_in_rows", first_column_ids))
    for layout, ids in candidates:
        if requested in ("auto", layout):
            sample_map = _tcga_sample_map(ids, patient_keys)
            if sample_map is not None:
                return layout, sample_map
    raise UserInputError(
        "No patient ID in the matrix matches the chosen ID column. The dataset's IDs look like "
        f"{_examples(patient_examples)}; the matrix's first row holds {_examples(header_ids)} and its first column "
        f"{_examples(first_column_ids)}. The IDs must be written the same way in both files."
    )


def _unique_names(values: Sequence[Any], what: str) -> tuple[str, ...]:
    names = []
    for value in values:
        key = id_key(value)
        if key is None:
            raise UserInputError(f"The matrix has a blank {what}.")
        names.append(key)
    seen: set[str] = set()
    duplicates = sorted({name for name in names if name in seen or seen.add(name)})
    if duplicates:
        raise UserInputError(f"The matrix repeats these {what}s: " + ", ".join(duplicates[:5]) + (" ..." if len(duplicates) > 5 else "") + ".")
    return tuple(names)


def _numeric_block(frame: pd.DataFrame, what: str) -> np.ndarray:
    text_columns = [str(column) for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if text_columns:
        converted = frame[text_columns].apply(pd.to_numeric, errors="coerce")
        bad = [column for column in text_columns if bool((frame[column].notna() & converted[column].isna()).any())]
        if bad:
            raise UserInputError(
                f"The matrix must hold numbers only; these {what}s contain text: " + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "") + "."
            )
        frame = frame.copy()
        frame[text_columns] = converted
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def read_marker_matrix(path: str | Path, filename: str, *, patient_ids: Sequence[Any], orientation: str = "auto") -> MarkerMatrix:
    """Read a marker matrix file; ``patient_ids`` (the dataset's ID column) decide the layout when it is "auto"."""
    if orientation not in ORIENTATIONS:
        raise UserInputError(f"Unknown matrix layout '{orientation}'.")
    suffix, compressed = matrix_format(filename)
    if compressed:
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / "matrix"
            _gunzip(Path(path), plain)
            suffix = suffix or _sniffed_suffix(plain)
            unpacked = plain.with_suffix(suffix)
            shutil.move(plain, unpacked)
            return _read_matrix_file(unpacked, suffix, filename, patient_ids=patient_ids, orientation=orientation)
    if suffix not in MATRIX_SUFFIXES:
        raise UserInputError(f"Unsupported matrix file type '{suffix}'. Use CSV, TSV, TXT or Parquet, optionally gzip-compressed (.gz).")
    return _read_matrix_file(Path(path), suffix, filename, patient_ids=patient_ids, orientation=orientation)


def _read_matrix_file(path: Path, suffix: str, filename: str, *, patient_ids: Sequence[Any], orientation: str) -> MarkerMatrix:
    patient_keys = {key for key in (id_key(value) for value in patient_ids) if key is not None}
    patient_examples = [key for key in (id_key(value) for value in patient_ids[:3]) if key is not None]

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            metadata = pq.ParquetFile(path).metadata
        except ImportError:  # pragma: no cover - pandas reports the missing engine itself
            metadata = None
        except Exception as exc:
            raise UserInputError("The Parquet file could not be read.") from exc
        if metadata is not None:
            # Checked before decompressing: the layout is not known yet, so both dimensions are bounded.
            n_rows, n_columns = int(metadata.num_rows), max(int(metadata.num_columns) - 1, 0)
            if n_rows * n_columns > MAX_MATRIX_CELLS or max(n_rows, n_columns) > MAX_MATRIX_SAMPLES:
                raise UserInputError(
                    f"The matrix has {n_rows:,} rows and {n_columns:,} data columns; SurvStudio reads at most "
                    f"{MAX_MATRIX_CELLS:,} values. Keep fewer markers, for example the most variable ones."
                )
        frame = pd.read_parquet(path)
        if frame.shape[1] < 2:
            raise UserInputError("The matrix needs an ID column and at least one data column.")
        frame = frame.set_index(frame.columns[0])
        _check_shape(frame.shape[1], frame.shape[0])
        column_names = list(frame.columns)
        layout, sample_map = _orientation(column_names, list(frame.index), patient_keys, orientation, patient_examples)
    else:
        separator = _separator(suffix)
        header, n_lines = _text_shape(path, separator)
        if len(header) < 2:
            raise UserInputError("The matrix needs an ID column and at least one data column.")
        first_column = pd.read_csv(path, sep=separator, usecols=[0], dtype=str, encoding="utf-8-sig").iloc[:, 0].tolist()
        layout, sample_map = _orientation(header[1:], first_column, patient_keys, orientation, patient_examples)
        n_data_columns = len(header) - 1
        _check_shape(*((n_lines, n_data_columns) if layout == "markers_in_rows" else (n_data_columns, n_lines)))
        frame = pd.read_csv(path, sep=separator, index_col=0, encoding="utf-8-sig", low_memory=False)
        # pandas renames repeated header fields ("A", "A.1"), so names come from the raw header.
        column_names = header[1:]
        if len(column_names) != frame.shape[1]:
            raise UserInputError("Every row of the matrix must have as many fields as the header.")

    if sample_map is not None:
        # Only the matched tumour samples are kept, under the patients' barcodes.
        if layout == "markers_in_rows":
            frame = frame.iloc[:, sample_map.positions]
            column_names = list(sample_map.keys)
        else:
            frame = frame.iloc[sample_map.positions]
            frame.index = list(sample_map.keys)
    if layout == "markers_in_rows":
        marker_names = _unique_names(list(frame.index), "marker name")
        sample_keys = _unique_names(column_names, "patient ID")
        values = _numeric_block(frame, "patient").T.copy()
    else:
        sample_keys = _unique_names(list(frame.index), "patient ID")
        marker_names = _unique_names(column_names, "marker name")
        values = _numeric_block(frame, "marker")
    _check_shape(len(marker_names), len(sample_keys))
    digest = hashlib.sha256()
    digest.update("\x1f".join(marker_names).encode("utf-8"))
    digest.update("\x1f".join(sample_keys).encode("utf-8"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return MarkerMatrix(
        filename=filename,
        orientation=layout,
        marker_names=marker_names,
        sample_keys=sample_keys,
        values=np.ascontiguousarray(values),
        fingerprint=digest.hexdigest()[:16],
        id_note=sample_map.note if sample_map is not None else "",
    )


def match_rows(matrix: MarkerMatrix, patient_ids: Sequence[Any]) -> np.ndarray:
    """For each dataset row, the matrix row with the same patient ID, or -1."""
    position = {key: index for index, key in enumerate(matrix.sample_keys)}
    keys = [id_key(value) for value in patient_ids]
    seen: set[str] = set()
    duplicates = sorted({key for key in keys if key is not None and (key in seen or seen.add(key))})
    if duplicates:
        raise UserInputError("The ID column repeats these values: " + ", ".join(duplicates[:5]) + ". Choose a column with one value per patient.")
    return np.asarray([position.get(key, -1) if key is not None else -1 for key in keys], dtype=np.int64)


def match_summary(matrix: MarkerMatrix, patient_ids: Sequence[Any]) -> dict[str, Any]:
    rows = match_rows(matrix, patient_ids)
    matched = int(np.sum(rows >= 0))
    used = set(rows[rows >= 0].tolist())
    unmatched_samples = [key for index, key in enumerate(matrix.sample_keys) if index not in used]
    return {
        "n_patients": len(rows),
        "n_matched": matched,
        "n_matrix_samples": len(matrix.sample_keys),
        "unmatched_matrix_samples": unmatched_samples[:10],
        "n_unmatched_matrix_samples": len(unmatched_samples),
    }


def matrix_frame(
    table: pd.DataFrame,
    matrix: MarkerMatrix,
    *,
    id_column: str,
    columns: Sequence[str],
    min_patients: int = 10,
) -> pd.DataFrame:
    """The dataset's ``columns`` for the patients found in the matrix, joined with every matrix marker."""
    if id_column not in table.columns:
        raise UserInputError(f"The ID column '{id_column}' is not in the dataset.")
    needed = list(dict.fromkeys(str(column) for column in columns))
    clash = sorted(set(needed) & set(matrix.marker_names))
    if clash:
        raise UserInputError(
            "These matrix markers have the same names as the outcome or clinical columns: " + ", ".join(clash[:5]) + ". Rename them in one of the files."
        )
    rows = match_rows(matrix, table[id_column].tolist())
    kept = np.flatnonzero(rows >= 0)
    if kept.size < min_patients:
        raise UserInputError(
            f"Only {kept.size} patients of the dataset are in the matrix; at least {min_patients} are needed. Check that the IDs match."
        )
    base = table.iloc[kept][needed].reset_index(drop=True)
    markers = pd.DataFrame(matrix.values[rows[kept]], columns=list(matrix.marker_names))
    return pd.concat([base, markers], axis=1)


class MarkerMatrixStore:
    """A few uploaded matrices, dropped after ``ttl_seconds`` without use or when the byte budget is full."""

    def __init__(self, *, max_items: int = 4, max_bytes: int = 800 * 1024 * 1024, ttl_seconds: int = 3600) -> None:
        self._items: OrderedDict[str, tuple[MarkerMatrix, float]] = OrderedDict()
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        for key in [key for key, (_, used) in self._items.items() if now - used > self._ttl_seconds]:
            del self._items[key]

    def add(self, matrix: MarkerMatrix) -> str:
        if matrix.values.nbytes > self._max_bytes:
            raise UserInputError("The matrix is too large to keep in memory.")
        matrix_id = secrets.token_hex(12)
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            self._items[matrix_id] = (matrix, now)
            while len(self._items) > self._max_items or sum(item.values.nbytes for item, _ in self._items.values()) > self._max_bytes:
                self._items.popitem(last=False)
        return matrix_id

    def get(self, matrix_id: str) -> MarkerMatrix:
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            item = self._items.get(matrix_id)
            if item is None:
                raise NotFoundError("The marker matrix has expired or was removed; attach the file again.")
            self._items[matrix_id] = (item[0], now)
            self._items.move_to_end(matrix_id)
            return item[0]

    def remove(self, matrix_id: str) -> None:
        with self._lock:
            self._items.pop(matrix_id, None)

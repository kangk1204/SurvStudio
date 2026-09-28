from __future__ import annotations

import codecs
import csv
import hashlib
import io
import itertools
import math
import re
import warnings
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy.special import ndtri
from scipy import stats
from statsmodels.base.model import LikelihoodModel
from statsmodels.duration.hazard_regression import PHReg, PHRegResults
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.nonparametric.smoothers_lowess import lowess

from survival_toolkit.concurrency import raise_if_cancelled, suppressed_warnings
from survival_toolkit.encoding import reject_numeric_text_features
from survival_toolkit.errors import ColumnNotFoundError, must_propagate, user_input_boundary

TRUE_TOKENS = {
    "1",
    "true",
    "yes",
    "y",
    "event",
    "dead",
    "death",
    "deceased",
    "progressed",
    "progression",
    "relapse",
    "failure",
}
FALSE_TOKENS = {
    "0",
    "false",
    "no",
    "n",
    "censored",
    "alive",
    "living",
    "diseasefree",
    "disease-free",
    "nonevent",
    "non-event",
}
_GENERIC_TRUE_TOKENS = {"1", "true", "yes", "y", "event"}
_DEATH_TRUE_TOKENS = {"dead", "death", "deceased"}
_PROGRESSION_TRUE_TOKENS = {"progressed", "progression", "relapse", "failure"}
_EVENT_TOKEN_FAMILIES = {
    **{token: "event_generic" for token in _GENERIC_TRUE_TOKENS},
    **{token: "event_death" for token in _DEATH_TRUE_TOKENS},
    **{token: "event_progression" for token in _PROGRESSION_TRUE_TOKENS},
    **{token: "censor" for token in FALSE_TOKENS},
}
KM_WEIGHT_MAP = {
    "logrank": None,
    "gehan_breslow": "gb",
    "tarone_ware": "tw",
    "fleming_harrington": "fh",
}
# DOTALL: category levels may contain line breaks.
TERM_CATEGORICAL_PATTERN = re.compile(r'C\(Q\("(?P<var>.+)"\)\)\[T\.(?P<level>.+)\]', re.DOTALL)
TERM_NUMERIC_PATTERN = re.compile(r'Q\("(?P<var>.+)"\)', re.DOTALL)
_MAX_EXP_INPUT = math.log(np.finfo(float).max)
_MIN_EXP_INPUT = math.log(np.finfo(float).tiny)
MAX_MODEL_FEATURE_CANDIDATES = 1000
COX_CONDITION_NUMBER_WARN_THRESHOLD = 1e10
# Same bound as the Kaplan-Meier group limit: more levels means an identifier or free text.
_COX_MAX_CATEGORICAL_LEVELS = 50
_MODEL_FEATURE_ID_PATTERN = re.compile(r"^(patient_id|sample_id|subject_id|id|barcode|uuid|cohort)$", re.IGNORECASE)
_EVENT_NAME_PATTERNS = (
    re.compile(r"event"),
    re.compile(r"death"),
    re.compile(r"deceas"),
    re.compile(r"mort"),
    re.compile(r"status$"),
    re.compile(r"vital_?status"),
    re.compile(r"survival_status"),
    re.compile(r"outcome_status"),
    re.compile(r"relapse"),
    re.compile(r"recur"),
    re.compile(r"progress"),
    re.compile(r"failure"),
    re.compile(r"censor"),
)
# Outcome words matched as whole name tokens: as substrings they would also match "studied",
# "deadline" or "vitals". "cens" is a common short name of a censoring flag.
_OUTCOME_NAME_TOKENS = {"dead", "died", "alive", "survived", "vital", "cens"}
_SURVIVAL_ENDPOINT_ABBREVIATIONS = {
    "os",
    "pfs",
    "dfs",
    "rfs",
    "efs",
    "tts",
    "tte",
    "dss",
    "css",
    "mfs",
    "dmfs",
    # TCGA Pan-Cancer Clinical Data Resource endpoints and common time-to-X abbreviations.
    "pfi",
    "dfi",
    "ttp",
    "ttf",
    "ttr",
    "ttnt",
    "ffs",
    "lrfs",
    "drfs",
}
# "<word> free" names a survival endpoint: disease-free, progression-free, ... survival.
_ENDPOINT_FREE_PHRASE_WORDS = ("disease", "progression", "recurrence", "relapse", "event", "metastasis", "failure")
# Followed by one of these, "<word> free" names the endpoint ("progression-free survival", "disease
# free months"); otherwise it says the patient is still free of the event ("progression_free" = 1).
_FREE_PHRASE_ENDPOINT_TOKENS = {
    "survival",
    "surv",
    "interval",
    "time",
    "period",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
}
# Names that say the patient has not died ("alive", "is_alive", "living", "survived").
_ALIVE_NAME_TOKENS = {"alive", "living", "survived", "survivor"}
_DEATH_NAME_TOKENS = {"dead", "death", "died", "deceased"}
_OBSERVATION_TOKENS = {"obs", "observation", "observed"}
_TIME_UNIT_TOKENS = {"day", "days", "week", "weeks", "month", "months", "year", "years"}
_TIME_CONTEXT_TOKENS = {
    "time",
    "duration",
    "survival",
    "follow",
    "followup",
    "follow_up",
    "fup",
}
_OUTCOME_CONTEXT_TOKENS = {
    "event",
    "death",
    "dead",
    "died",
    "deceased",
    "alive",
    "vital",
    "mort",
    "status",
    "progress",
    "progression",
    "progressed",
    "relapse",
    "relapsed",
    "recur",
    "recurrence",
    "recurred",
    "metastasis",
    "metastases",
    "failure",
    "censor",
    "censored",
    "censoring",
    "cens",
}
_TIME_NAME_TOKENS = (
    "time",
    "os",
    "pfs",
    "dfs",
    "rfs",
    "efs",
    "tts",
    "tte",
    "dss",
    "css",
    "follow",
    "followup",
    "follow_up",
    "fup",
    "survival",
)
_OUTCOME_STATUS_VALUE_FAMILIES = {
    "alive": "censor",
    "alivewithoutdisease": "censor",
    "censored": "censor",
    "diseasefree": "censor",
    "living": "censor",
    "ned": "censor",
    "noevidenceofdisease": "censor",
    "nonevent": "censor",
    "awd": "event_progression",
    "alivewithdisease": "event_progression",
    "deceased": "event_death",
    "dead": "event_death",
    "deadofdisease": "event_death",
    "death": "event_death",
    "deathofdisease": "event_death",
    "died": "event_death",
    "doc": "event_death",
    "dod": "event_death",
    "failure": "event_progression",
    "progressed": "event_progression",
    "progression": "event_progression",
    "recurrence": "event_progression",
    "recurred": "event_progression",
    "relapse": "event_progression",
    "relapsed": "event_progression",
}
_BASELINE_STATUS_PATTERNS = (
    re.compile(r"egfr"),
    re.compile(r"kras"),
    re.compile(r"braf"),
    re.compile(r"alk"),
    re.compile(r"ros1"),
    re.compile(r"erbb2"),
    re.compile(r"mutation"),
    re.compile(r"mutated"),
    re.compile(r"wildtype"),
    re.compile(r"sex"),
    re.compile(r"gender"),
    re.compile(r"stage"),
    re.compile(r"grade"),
    re.compile(r"treat"),
    re.compile(r"therapy"),
    re.compile(r"drug"),
    re.compile(r"smok"),
    re.compile(r"histolog"),
    re.compile(r"subtype"),
    re.compile(r"cluster"),
    re.compile(r"group"),
    re.compile(r"arm"),
    re.compile(r"cohort"),
    re.compile(r"horth"),
    # Biomarker, receptor, and clinical-state "status" columns (her2_status, msi_status,
    # nodal_status, performance_status, ...) are baseline characteristics, not outcomes.
    re.compile(r"her2"),
    re.compile(r"(^|[^a-z0-9])(er|pr|pgr|ar)([^a-z0-9]|$)"),
    re.compile(r"(^|[^a-z0-9])(msi|mss|mmr|dmmr|pmmr|idh|idh1|idh2|ebv|p16|cimp|atrx|tert)([^a-z0-9]|$)"),
    re.compile(r"mgmt"),
    re.compile(r"hpv"),
    re.compile(r"p53"),
    re.compile(r"brca"),
    re.compile(r"pd.?l1"),
    re.compile(r"codel"),
    re.compile(r"methyl"),
    re.compile(r"amplif"),
    re.compile(r"fusion"),
    re.compile(r"express"),
    re.compile(r"marker"),
    re.compile(r"mutant"),
    re.compile(r"receptor"),
    re.compile(r"hormone"),
    re.compile(r"nodal"),
    re.compile(r"(^|[^a-z0-9])(node|nodes|ln)([^a-z0-9]|$)"),
    re.compile(r"lymph"),
    re.compile(r"menopaus"),
    re.compile(r"performance"),
    re.compile(r"ecog"),
    re.compile(r"karnofsky"),
    re.compile(r"(^|[^a-z0-9])(kps|ps)([^a-z0-9]|$)"),
    re.compile(r"(^|[^a-z0-9])race([^a-z0-9]|$)"),
    re.compile(r"ethnic"),
)
# Words that name the event itself. A baseline-looking name that also carries one of these
# ("treatment_failure", "progression_on_therapy") can still be an event column.
_STRONG_EVENT_NAME_PATTERNS = (
    re.compile(r"event"),
    re.compile(r"death"),
    re.compile(r"dead"),
    re.compile(r"died"),
    re.compile(r"deceas"),
    re.compile(r"mort"),
    re.compile(r"relaps"),
    re.compile(r"recur"),
    re.compile(r"progress"),
    re.compile(r"failure"),
    re.compile(r"vital"),
)


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    used: set[str] = set()
    output: list[str] = []
    for raw_name in columns:
        name = str(raw_name).strip() or "unnamed"
        counter = seen.get(name, 0)
        unique_name = name if counter == 0 else f"{name}_{counter + 1}"
        while unique_name in used:
            counter += 1
            unique_name = f"{name}_{counter + 1}"
        seen[name] = counter + 1
        used.add(unique_name)
        output.append(unique_name)
    return output


_CSV_DELIMITER_CANDIDATES = ",;\t|"
# A number written with "," thousands separators and an optional "." decimal part ("12,345.5").
_COMMA_GROUPED_NUMBER_PATTERN = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
# A value of a column formatted with "," thousands separators (plain numbers are allowed too).
_COMMA_THOUSANDS_VALUE_PATTERN = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")
# Shapes of number text in files not separated by commas. "1.234" and "1,234" (a non-zero lead of
# one to three digits and one group of three) read either as a decimal or as a thousands-grouped
# integer; each other shape reads one way only.
_PLAIN_NUMBER_PATTERN = re.compile(r"^[+-]?\d+([eE][+-]?\d+)?$")
_AMBIGUOUS_DOT_NUMBER_PATTERN = re.compile(r"^[+-]?[1-9]\d{0,2}\.\d{3}$")
_AMBIGUOUS_COMMA_NUMBER_PATTERN = re.compile(r"^[+-]?[1-9]\d{0,2},\d{3}$")
# "." as the decimal mark: "12.5", ".5", "1.5e-3", or "," thousands groups ("1,234,567", "1,234.5").
_DOT_DECIMAL_NUMBER_PATTERN = re.compile(r"^[+-]?((\d+\.\d*|\.\d+)([eE][+-]?\d+)?|[1-9]\d{0,2}(,\d{3})+(\.\d+)?)$")
# "," as the decimal mark: "12,5", "1,5E-3", or "." thousands groups ("1.234.567", "1.234,5").
_COMMA_DECIMAL_NUMBER_PATTERN = re.compile(r"^[+-]?((\d+,\d+)([eE][+-]?\d+)?|[1-9]\d{0,2}(\.\d{3})+(,\d+)?)$")
_NUMBER_SHAPE_PATTERNS = (
    ("plain", _PLAIN_NUMBER_PATTERN),
    ("ambiguous_dot", _AMBIGUOUS_DOT_NUMBER_PATTERN),
    ("ambiguous_comma", _AMBIGUOUS_COMMA_NUMBER_PATTERN),
    ("dot", _DOT_DECIMAL_NUMBER_PATTERN),
    ("comma", _COMMA_DECIMAL_NUMBER_PATTERN),
)
_ANY_NUMBER_SHAPE_PATTERN = re.compile("|".join(f"(?:{pattern.pattern})" for _, pattern in _NUMBER_SHAPE_PATTERNS))
# A number written with "," or "." separators in either convention ("1,234", "12,5", "1.234,5").
_SEPARATED_NUMBER_PATTERN = re.compile(r"^[+-]?(\d{1,3}([.,]\d{3})+([.,]\d+)?|\d+[.,]\d+)$")
# Text that means "no value" (TCGA exports write "[Not Available]" and similar). Time columns
# read it as missing, and the loader leaves it out when it decides how a column writes numbers.
_MISSING_VALUE_TOKENS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "nan",
    "null",
    "none",
    "nr",
    ".",
    "-",
    "--",
    "?",
    "#n/a",
    "missing",
    "unknown",
    "no data",
    "not available",
    "not applicable",
    "not assessed",
    "not evaluated",
    "never assessed",
    "not reported",
    "[not available]",
    "[not applicable]",
    "[not reported]",
    "[not evaluated]",
    "[unknown]",
    "[discrepancy]",
}
# Bytes inspected to pick the text encoding and the delimiter; the full file is then
# parsed by pandas in streaming mode instead of being decoded into one Python string.
_TEXT_SNIFF_BYTES = 1024 * 1024
# The delimiter is chosen from at most this many records and characters of that sample.
_DELIMITER_SNIFF_MAX_RECORDS = 50
_DELIMITER_SNIFF_MAX_CHARS = 64 * 1024
# Share of non-ASCII characters that must be Hangul syllables before a byte stream that
# is not UTF-8 is read as Korean CP949 (the default "CSV" export of Korean Excel).
_CP949_HANGUL_SHARE = 0.5
# Bytes from the first byte that is not UTF-8 that decide the fallback encoding when a file is
# UTF-8 only in its sniffed prefix.
_LATE_ENCODING_PROBE_BYTES = 64 * 1024
TEXT_ENCODING_LABELS = {
    "utf-8": "UTF-8",
    "utf-8-sig": "UTF-8",
    "utf-16": "UTF-16",
    "utf-16-le": "UTF-16 (little-endian)",
    "utf-16-be": "UTF-16 (big-endian)",
    "cp949": "Korean CP949 (EUC-KR)",
    "cp1252": "Windows-1252 (Western European)",
    "latin-1": "Latin-1 (ISO-8859-1)",
}


class UploadShapeError(ValueError):
    """The upload exceeds a row, column, or cell limit (raised before the full parse)."""


def _trim_to_line_boundary(sample: bytes, *, complete: bool) -> bytes:
    """Cut a byte sample after its last newline so no multibyte character is split.

    Only valid for encodings in which the byte 0x0A always means a newline (UTF-8, CP949,
    single-byte code pages); UTF-16 samples are decoded with ``_decode_text_sample``.
    """
    if complete:
        return sample
    cut = sample.rfind(b"\n")
    return sample[: cut + 1] if cut >= 0 else sample


def _decode_text_sample(sample: bytes, encoding: str, *, complete: bool) -> str:
    """Decode the sniffed prefix of a text file and cut it after its last complete line.

    An incremental decoder leaves a character split by the sample boundary undecoded instead
    of failing, which works for every encoding, including UTF-16, where a newline is two
    bytes and cutting after a single 0x0A byte would split a code unit.
    """
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    text = decoder.decode(sample, final=complete)
    if complete:
        return text
    cut = text.rfind("\n")
    return text[: cut + 1] if cut >= 0 else text


@lru_cache(maxsize=1)
def _ks_x_1001_hangul() -> frozenset[str]:
    """The 2,350 Hangul syllables and the Hangul letters of KS X 1001, the core of CP949.

    Korean text uses these almost exclusively. The other CP949 syllables (the Unified Hangul
    Code extension) include pairs of a byte 0x81-0xC6 and an ASCII letter, such as 0xC4 0x72,
    which is also "Är" in Windows-1252: Western text read as CP949 becomes those rare syllables.
    """
    characters: set[str] = set()
    for lead in (0xA4, *range(0xB0, 0xC9)):
        for trail in range(0xA1, 0xFF):
            try:
                characters.add(bytes((lead, trail)).decode("cp949"))
            except UnicodeDecodeError:
                continue
    return frozenset(char for char in characters if "가" <= char <= "힣" or "ㄱ" <= char <= "ㆎ")


def _hangul_share(text: str) -> float:
    """Share of the non-ASCII characters that are KS X 1001 Hangul (what CP949 Korean text is made of)."""
    non_ascii = [char for char in text if ord(char) > 127]
    if not non_ascii:
        return 0.0
    hangul = _ks_x_1001_hangul()
    return sum(1 for char in non_ascii if char in hangul) / len(non_ascii)


def _bytes_after_first_non_utf8(open_source: Any) -> bytes:
    """Up to 64 KiB of the source from its first byte that is not UTF-8 (b"" when every byte is)."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    with open_source() as handle:
        while True:
            chunk = handle.read(_TEXT_SNIFF_BYTES)
            if not chunk:
                return b""
            try:
                decoder.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                window = bytes(exc.object[exc.start : exc.start + _LATE_ENCODING_PROBE_BYTES])
                if len(window) < _LATE_ENCODING_PROBE_BYTES:
                    window += handle.read(_LATE_ENCODING_PROBE_BYTES - len(window))
                return window


def _encodings_after_late_utf8_error(open_source: Any, remaining: Sequence[str]) -> list[str]:
    """The encodings to try when a file is UTF-8 in its sniffed prefix but not further on.

    Windows-1252 comes first unless the undecodable bytes read as Korean CP949 text (mostly
    KS X 1001 Hangul); trying CP949 first would turn Western text such as "Ärzte" into Hangul.
    """
    window = _bytes_after_first_non_utf8(open_source)
    try:
        text = codecs.getincrementaldecoder("cp949")(errors="strict").decode(window, final=False)
    except UnicodeDecodeError:
        text = ""
    korean = bool(text) and _hangul_share(text) >= _CP949_HANGUL_SHARE
    order = ("cp949", "cp1252", "latin-1") if korean else ("cp1252", "latin-1")
    return [encoding for encoding in order if encoding in remaining]


def _text_encoding_candidates(sample: bytes, *, complete: bool) -> list[str]:
    """Ordered encodings to try for a text upload, judged from its first bytes.

    UTF-8 (with or without BOM) and UTF-16 are recognised directly. Other byte streams
    are read as Korean CP949 when the sample decodes as mostly Hangul, and otherwise as
    Windows-1252 with Latin-1 as the final fallback (Latin-1 accepts any byte).
    """
    if sample.startswith(b"\xef\xbb\xbf"):
        return ["utf-8-sig", "cp949", "cp1252", "latin-1"]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ["utf-16"]
    probe = sample[:4096]
    # UTF-16 without a BOM shows up as many NUL bytes; latin-1 would "decode"
    # it into garbage without an error.
    if probe and probe.count(b"\x00") > len(probe) // 4:
        nul_at_odd = probe[1::2].count(b"\x00") >= probe[0::2].count(b"\x00")
        return ["utf-16-le", "utf-16-be"] if nul_at_odd else ["utf-16-be", "utf-16-le"]
    trimmed = _trim_to_line_boundary(sample, complete=complete)
    try:
        trimmed.decode("utf-8")
        return ["utf-8", "cp949", "cp1252", "latin-1"]
    except UnicodeDecodeError:
        pass
    try:
        decoded = trimmed.decode("cp949")
    except UnicodeDecodeError:
        decoded = None
    if decoded is not None and _hangul_share(decoded) >= _CP949_HANGUL_SHARE:
        return ["cp949", "cp1252", "latin-1"]
    return ["cp1252", "latin-1"]


def _delimiter_counts_per_record(text: str) -> list[dict[str, int]]:
    """Delimiter counts outside double quotes for each record of a text sample.

    A single linear pass (quoted fields may span lines). ``csv.Sniffer`` was used before, but
    its quote-detection regexes scale quadratically on crafted input.
    """
    truncated = len(text) > _DELIMITER_SNIFF_MAX_CHARS
    sample = text[:_DELIMITER_SNIFF_MAX_CHARS].replace("\r\n", "\n").replace("\r", "\n")
    records: list[dict[str, int]] = []
    counts = dict.fromkeys(_CSV_DELIMITER_CANDIDATES, 0)
    in_quotes = False
    has_content = False
    for char in sample:
        if char == '"':
            in_quotes = not in_quotes
            has_content = True
        elif char == "\n" and not in_quotes:
            if has_content:
                records.append(counts)
                if len(records) >= _DELIMITER_SNIFF_MAX_RECORDS:
                    return records
            counts = dict.fromkeys(_CSV_DELIMITER_CANDIDATES, 0)
            has_content = False
        elif not in_quotes and char in counts:
            counts[char] += 1
            has_content = True
        elif not char.isspace():
            has_content = True
    # A record cut by the character cap is incomplete, so it is left out.
    if has_content and not truncated:
        records.append(counts)
    return records


def _sniff_delimiter(text: str, default: str) -> str:
    """Pick the delimiter whose count per record is the most consistent.

    Ties go to the delimiter with more fields per record, then to ``default``. Only the
    candidates in ``_CSV_DELIMITER_CANDIDATES`` are considered, so a single-column file is
    never split on a letter; without any candidate the default is returned.
    """
    records = _delimiter_counts_per_record(text)
    if not records:
        return default
    preference = [default, *(candidate for candidate in _CSV_DELIMITER_CANDIDATES if candidate != default)]
    best_key: tuple[float, int, int] | None = None
    best_delimiter = default
    for rank, delimiter in enumerate(preference):
        if delimiter not in _CSV_DELIMITER_CANDIDATES:
            continue
        per_record = [record[delimiter] for record in records]
        positive = [count for count in per_record if count > 0]
        if not positive:
            continue
        frequency: dict[int, int] = {}
        for count in positive:
            frequency[count] = frequency.get(count, 0) + 1
        mode_count = max(frequency, key=lambda count: (frequency[count], count))
        consistency = sum(1 for count in per_record if count == mode_count) / len(per_record)
        if consistency < 0.5:
            continue
        key = (consistency, mode_count, -rank)
        if best_key is None or key > best_key:
            best_key = key
            best_delimiter = delimiter
    return best_delimiter


def _data_row_layout(text: str, delimiter: str) -> tuple[str, int]:
    """How the data rows of a text table line up with its header row.

    Returns ``("header", 0)`` when no sampled data row has more fields than the header,
    ``("trailing", k)`` when the extra fields (at most ``k``) are all empty because the writer
    ends every row with a separator, and ``("row_names", 1)`` when every data row has exactly one
    field more, as R's ``write.table`` writes row names without a header field. pandas would use
    the first column as the index in both cases, which in the trailing case shifts every column
    label one place. Rows with extra fields that fit neither case are refused.
    """
    header: list[str] | None = None
    widths: list[int] = []
    extras_empty = True
    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter):
            if not row:
                continue  # blank line, skipped by pandas as well
            if header is None:
                header = row
                continue
            widths.append(len(row))
            if len(row) > len(header) and any(field.strip() for field in row[len(header) :]):
                extras_empty = False
            if len(widths) >= _DELIMITER_SNIFF_MAX_RECORDS:
                break
    except csv.Error:
        return "header", 0
    if header is None or not widths or max(widths) <= len(header):
        return "header", 0
    if extras_empty:
        return "trailing", max(widths) - len(header)
    if set(widths) == {len(header) + 1}:
        return "row_names", 1
    raise ValueError(
        f"The data rows have more fields than the header row ({len(header)} column names, up to {max(widths)} "
        "fields per row), so the values cannot be matched to their columns. Add the missing column names to the "
        "header row, or remove the extra separators, and upload the file again."
    )


def _is_text_series(series: pd.Series) -> bool:
    # pandas >= 3 reads text as the dedicated "str" dtype instead of object.
    return is_object_dtype(series) or is_string_dtype(series)


def _reject_upload_shape(
    *,
    n_rows: int | None = None,
    n_columns: int | None = None,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_cells: int | None = None,
) -> None:
    if max_columns is not None and n_columns is not None and n_columns > max_columns:
        raise UploadShapeError(
            f"Upload has {n_columns:,} columns. SurvStudio currently supports at most {max_columns:,} columns per uploaded cohort."
        )
    if n_rows is None:
        return
    if max_rows is not None and n_rows > max_rows:
        raise UploadShapeError(
            f"Upload has more than {max_rows:,} rows. SurvStudio currently supports at most {max_rows:,} rows per uploaded cohort."
        )
    if max_cells is not None and n_columns is not None and n_rows * n_columns > max_cells:
        raise UploadShapeError(
            f"Upload expands to more than {max_cells:,} cells after parsing. SurvStudio currently supports at most "
            f"{max_cells:,} parsed cells per uploaded cohort."
        )


def _row_read_limit(
    n_columns: int,
    *,
    max_rows: int | None,
    max_cells: int | None,
) -> int | None:
    """Rows to parse so that one row beyond any limit is seen without reading the rest."""
    limits = []
    if max_rows is not None:
        limits.append(int(max_rows))
    if max_cells is not None and n_columns > 0:
        limits.append(int(max_cells) // int(n_columns))
    return (min(limits) + 1) if limits else None


def _read_csv_with_fallback(
    source: io.BytesIO | str | Path,
    *,
    default_delimiter: str = ",",
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_cells: int | None = None,
) -> pd.DataFrame:
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        content = source.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        sample = content[:_TEXT_SNIFF_BYTES]
        complete = len(content) <= _TEXT_SNIFF_BYTES

        def _open() -> Any:
            return io.BytesIO(content)
    else:
        path = Path(source)
        with path.open("rb") as handle:
            sample = handle.read(_TEXT_SNIFF_BYTES + 1)
        complete = len(sample) <= _TEXT_SNIFF_BYTES
        sample = sample[:_TEXT_SNIFF_BYTES]

        def _open() -> Any:
            return path.open("rb")

    if not sample.strip():
        raise ValueError("The uploaded file is empty. Add a header row and at least one data row.")

    last_error: Exception | None = None
    candidates = _text_encoding_candidates(sample, complete=complete)
    while candidates:
        encoding = candidates.pop(0)
        try:
            sample_text = _decode_text_sample(sample, encoding, complete=complete)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if not sample_text.strip():
            raise ValueError("The uploaded file is empty. Add a header row and at least one data row.")
        delimiter = _sniff_delimiter(sample_text, default_delimiter)
        layout, extra_fields = _data_row_layout(sample_text, delimiter)
        try:
            frame = _parse_delimited_text(
                _open,
                encoding=encoding,
                delimiter=delimiter,
                max_rows=max_rows,
                max_columns=max_columns,
                max_cells=max_cells,
                layout=layout,
                extra_fields=extra_fields,
            )
        except UnicodeDecodeError as exc:
            # A later part of the file does not fit this encoding; try the next one.
            last_error = exc
            if encoding in {"utf-8", "utf-8-sig"}:
                # The sniffed prefix was UTF-8, so the fallback is judged from the bytes that are not.
                candidates = _encodings_after_late_utf8_error(_open, candidates)
            continue
        frame.attrs["source_encoding"] = encoding
        return frame
    raise ValueError(
        "Could not decode the file. Supported encodings: UTF-8, UTF-16, CP949, Windows-1252, Latin-1."
    ) from last_error


def _parse_delimited_text(
    open_source: Any,
    *,
    encoding: str,
    delimiter: str,
    max_rows: int | None,
    max_columns: int | None,
    max_cells: int | None,
    layout: str = "header",
    extra_fields: int = 0,
) -> pd.DataFrame:
    """Parse a delimited text table with one column per field and a 0..n-1 row index.

    ``layout`` comes from ``_data_row_layout``. Data rows with a field more than the header are
    read with explicit column names, so pandas never takes the first column as the index: R row
    names become a "row_names" column, and the empty fields after a trailing separator an
    unnamed column that is dropped when it holds nothing.
    """
    def _read(**kwargs: Any) -> pd.DataFrame:
        with open_source() as handle:
            return pd.read_csv(handle, sep=delimiter, encoding=encoding, encoding_errors="strict", **kwargs)

    try:
        header = _read(nrows=0)
        header_names = [str(name) for name in header.columns]
        names: list[str] | None = None
        if layout == "row_names":
            names = [_next_available_column_name(header_names, "row_names"), *header_names]
        elif layout == "trailing" and extra_fields > 0:
            # pandas names a header field without a name "Unnamed: <position>".
            names = [*header_names, *(f"Unnamed: {len(header_names) + offset}" for offset in range(extra_fields))]
        read_options: dict[str, Any] = {"header": 0, "names": names} if names is not None else {}
        n_columns = len(names) if names is not None else int(header.shape[1])
        _reject_upload_shape(n_columns=n_columns, max_columns=max_columns)
        nrows = _row_read_limit(n_columns, max_rows=max_rows, max_cells=max_cells)
        frame = _read(nrows=nrows, **read_options)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file is empty. Add a header row and at least one data row.") from exc
    except (csv.Error, pd.errors.ParserError) as exc:
        raise ValueError(
            "The uploaded text file is empty or malformed. Add a header row and at least one data row."
        ) from exc
    if not isinstance(frame.index, pd.RangeIndex):
        # pandas moved the first field of each row into the index because a row beyond the
        # sampled lines has more fields than the header; the column labels would be shifted.
        raise ValueError(
            "Some data rows have more fields than the header row, so the values cannot be matched to their "
            "columns. Add the missing column names to the header row, or remove the extra separators, and "
            "upload the file again."
        )
    _reject_upload_shape(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        max_rows=max_rows,
        max_columns=max_columns,
        max_cells=max_cells,
    )

    def _read_raw_text(positions: list[int]) -> pd.DataFrame:
        return _read(nrows=nrows, dtype=str, usecols=positions, **read_options)

    frame = _convert_formatted_number_columns(frame, delimiter=delimiter, read_raw_text=_read_raw_text)
    if layout == "trailing" and names is not None:
        empty_extras = [
            name
            for name in names[len(header_names) :]
            if bool((frame[name].isna() | frame[name].astype(str).str.strip().eq("")).all())
        ]
        frame = frame.drop(columns=empty_extras)
    return frame


def _number_text(series: pd.Series) -> pd.Series:
    """Stripped text of the values that are not missing and not a missing-value marker ("NA", "-")."""
    text = series.dropna().astype(str).str.strip()
    text = text[text != ""]
    return text[~text.str.lower().isin(_MISSING_VALUE_TOKENS)]


def _number_shapes(text: pd.Series) -> set[str] | None:
    """The number shapes (``_NUMBER_SHAPE_PATTERNS``) of a column's text; None when a value is no number."""
    values = [str(value) for value in pd.unique(text.to_numpy(dtype=object))]
    # A column of words fails on its first values, before every value is matched five times.
    if not all(_ANY_NUMBER_SHAPE_PATTERN.fullmatch(value) for value in values[:50]):
        return None
    shapes: set[str] = set()
    for value in values:
        for shape, pattern in _NUMBER_SHAPE_PATTERNS:
            if pattern.fullmatch(value):
                shapes.add(shape)
                break
        else:
            return None
    return shapes


def _float_column_may_hide_dot_groups(series: pd.Series) -> bool:
    """True when every value of a float column could have been written as "1.234" or as a whole number.

    pandas reads "1.234" as 1.234; whether the file meant a decimal or a thousands group is only
    known from the raw text, which is read again for such columns alone. A value such as 0.5,
    12.3456 or 1234.5 can only have been written with "." as the decimal mark.
    """
    values = series.to_numpy(dtype=float, na_value=np.nan)
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return False
    whole = values == np.round(values)
    scaled = values * 1000.0
    grouped = (values >= 1.0) & (values < 1000.0) & np.isclose(scaled, np.round(scaled), rtol=0.0, atol=1e-6)
    return bool(np.all(whole | grouped))


def _ambiguous_number_message(column: Any, text: pd.Series, mixed_evidence: tuple[Any, Any] | None) -> str:
    patterns = (_AMBIGUOUS_COMMA_NUMBER_PATTERN, _AMBIGUOUS_DOT_NUMBER_PATTERN)
    example = next(
        (str(value) for value in text if any(pattern.fullmatch(str(value)) for pattern in patterns)),
        str(text.iloc[0]),
    )
    separator = "," if "," in example else "."
    if mixed_evidence is not None:
        reason = (
            f'the file writes decimals both with "," (column "{mixed_evidence[0]}") and with "." '
            f'(column "{mixed_evidence[1]}")'
        )
    else:
        reason = 'nothing else in the file shows whether "," is its decimal mark or its thousands separator'
    return (
        f'Column "{column}" holds numbers such as "{example}", which read either as {example.replace(separator, ".")} '
        f'or as {example.replace(separator, "")}, and {reason}. Save the file again with one decimal mark and '
        "without thousands separators, then upload it again."
    )


def _set_parsed_column(frame: pd.DataFrame, column: Any, numbers: pd.Series) -> bool:
    """Store ``numbers`` (normalized text of the non-missing values) as a numeric column.

    Nothing changes unless every value parses. Like pandas, a column without missing values
    whose numbers have no decimal part becomes int64; everything else becomes float64.
    """
    parsed = pd.to_numeric(numbers.astype(object), errors="coerce")
    parsed_values = parsed.to_numpy(dtype=float, na_value=np.nan)
    if not bool(np.isfinite(parsed_values).all()):
        return False
    integers = (
        len(parsed_values) == len(frame)
        and not bool(numbers.str.contains(".", regex=False).any())
        and bool(np.all(np.abs(parsed_values) < 2**53))
    )
    if integers:
        frame[column] = parsed_values.astype(np.int64)
        return True
    values = np.full(len(frame), np.nan, dtype=float)
    values[frame.index.get_indexer(parsed.index)] = parsed_values
    frame[column] = values
    return True


def _convert_formatted_number_columns(
    frame: pd.DataFrame,
    *,
    delimiter: str,
    read_raw_text: Any = None,
) -> pd.DataFrame:
    """Parse the columns that hold formatted numbers, deciding each column's decimal mark once.

    Missing-value markers ("NA", "-", "[Not Available]") are left out of every test and become
    missing values in a converted column, so a stray marker never changes how the numbers read.

    * In comma-separated files, a column of quoted numbers with thousands separators
      ("1,234", as Excel exports formatted numbers) is parsed as numbers.
    * In other files (European ";" exports, tab-separated text), a column whose values are all
      numbers is read with its own decimal mark when one of its values shows it ("12,5" or
      "1.234,5" for ",", "12.5" or "1,234.5" for "."). A column of values such as "1,234" or
      "1.234", which read either way, follows the file: the decimal mark its other columns
      show, else "," in a ";" file (the European convention, so "1.234" days are 1234). Such a
      column is refused when the file shows both marks, or when a tab-separated file shows
      neither and the column holds "1,234" values; "1.234" alone stays the decimal pandas read.
      ``read_raw_text(positions)`` returns the raw text of the given columns, which is needed
      for float columns because pandas already parsed "1.234" as 1.234.

    Dot-decimal columns without "," values and other text (dotted dates such as "01.02.2020")
    are left as read. A column is converted only when every value parses.
    """
    if delimiter == ",":
        for column in list(frame.columns):
            series = frame[column]
            if not _is_text_series(series):
                continue
            text = _number_text(series)
            if text.empty or not bool(text.str.contains(",", regex=False).any()):
                continue
            if bool(text.str.fullmatch(_COMMA_THOUSANDS_VALUE_PATTERN).all()):
                _set_parsed_column(frame, column, text.str.replace(",", "", regex=False))
        return frame

    shaped: dict[Any, tuple[pd.Series, set[str]]] = {}
    dot_columns: list[Any] = []  # columns whose own values show "." as the decimal mark
    float_candidates: list[Any] = []
    for column in frame.columns:
        series = frame[column]
        if is_bool_dtype(series):
            continue
        if _is_text_series(series):
            text = _number_text(series)
            shapes = _number_shapes(text) if not text.empty else None
            if shapes is not None and shapes != {"plain"}:
                shaped[column] = (text, shapes)
        elif pd.api.types.is_float_dtype(series):
            if _float_column_may_hide_dot_groups(series):
                float_candidates.append(column)
            elif bool(series.notna().any()):
                dot_columns.append(column)

    def _own_mark(shapes: set[str]) -> str | None:
        if "dot" in shapes and "comma" not in shapes:
            return "."
        if "comma" in shapes and "dot" not in shapes:
            return ","
        return None

    comma_columns = [column for column, (_, shapes) in shaped.items() if _own_mark(shapes) == ","]
    dot_columns += [column for column, (_, shapes) in shaped.items() if _own_mark(shapes) == "."]
    has_comma_values = any(shapes & {"comma", "ambiguous_comma"} for _, shapes in shaped.values())
    if float_candidates and read_raw_text is not None and (
        comma_columns or (not dot_columns and (delimiter == ";" or has_comma_values))
    ):
        positions = sorted(int(frame.columns.get_loc(column)) for column in float_candidates)
        try:
            raw = read_raw_text(positions)
        except (ValueError, pd.errors.ParserError, csv.Error):
            raw = None
        # Only trust the raw text when it lines up with the parsed columns.
        if raw is not None and [str(name) for name in raw.columns] == [str(frame.columns[position]) for position in positions]:
            for offset, position in enumerate(positions):
                text = _number_text(raw.iloc[:, offset])
                shapes = _number_shapes(text) if not text.empty else None
                if shapes is None or shapes == {"plain"}:
                    continue
                column = frame.columns[position]
                shaped[column] = (text, shapes)
                if _own_mark(shapes) == ".":
                    dot_columns.append(column)
    if not shaped:
        return frame

    mixed_evidence = (comma_columns[0], dot_columns[0]) if comma_columns and dot_columns else None
    if mixed_evidence is not None:
        file_mark = None
    elif comma_columns:
        file_mark = ","
    elif dot_columns:
        file_mark = "."
    else:
        file_mark = "," if delimiter == ";" else None

    ambiguous: list[Any] = []
    for column, (text, shapes) in shaped.items():
        if "dot" in shapes and "comma" in shapes:
            continue  # both decimal marks in one column: left as read
        mark = _own_mark(shapes) or file_mark
        if mark is None:
            if mixed_evidence is None and "ambiguous_comma" not in shapes:
                continue  # "1.234" in a file without any "," number stays the decimal pandas read
            ambiguous.append(column)
            continue
        if mark == ".":
            if not bool(text.str.contains(",", regex=False).any()):
                continue  # pandas (or the time-column parser) reads these values the same way
            numbers = text.str.replace(",", "", regex=False)
        else:
            numbers = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        _set_parsed_column(frame, column, numbers)
    if ambiguous:
        raise ValueError(_ambiguous_number_message(ambiguous[0], shaped[ambiguous[0]][0], mixed_evidence))
    return frame


def _read_excel_frame(source: io.BytesIO | str | Path, **kwargs: Any) -> pd.DataFrame:
    """``pd.read_excel`` with a damaged or mislabelled workbook reported as a ValueError.

    Only the reader call is guarded: openpyxl, xlrd and zipfile raise many unrelated exception
    types for unreadable files, while an error in SurvStudio's own code must not be reported as
    an unreadable file.
    """
    try:
        return pd.read_excel(source, **kwargs)
    except MemoryError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read Excel file: {exc}") from exc


def _read_excel_limited(
    source: io.BytesIO | str | Path,
    *,
    max_rows: int | None,
    max_columns: int | None,
    max_cells: int | None,
) -> pd.DataFrame:
    if max_rows is None and max_columns is None and max_cells is None:
        return _read_excel_frame(source)
    header = _read_excel_frame(source, nrows=0)
    n_columns = int(header.shape[1])
    _reject_upload_shape(n_columns=n_columns, max_columns=max_columns)
    if hasattr(source, "seek"):
        source.seek(0)
    frame = _read_excel_frame(source, nrows=_row_read_limit(n_columns, max_rows=max_rows, max_cells=max_cells))
    _reject_upload_shape(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        max_rows=max_rows,
        max_columns=max_columns,
        max_cells=max_cells,
    )
    return frame


def _read_parquet_frame(source: io.BytesIO | str | Path) -> pd.DataFrame:
    """``pd.read_parquet`` with a damaged file reported as a ValueError (only the reader call is guarded)."""
    try:
        return pd.read_parquet(source)
    except MemoryError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read Parquet file: {exc}") from exc


def _with_default_row_index(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with a unique 0..n-1 row index; a meaningful stored index becomes ordinary columns.

    Parquet files keep a pandas index: patient IDs set as the index, or the row numbers of a
    filtered or concatenated frame (which can repeat). Analysis rows are matched by index later,
    so an unnamed integer index is taken as row numbers and dropped, and any other index (a
    named one, text labels, several levels) is kept as named columns in front.
    """
    index = frame.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        return frame
    if index.nlevels == 1 and index.name is None and pd.api.types.is_integer_dtype(index.dtype):
        return frame.reset_index(drop=True)
    used = [str(column) for column in frame.columns]
    names: list[str] = []
    for level, name in enumerate(index.names):
        base = str(name) if name is not None else ("index" if index.nlevels == 1 else f"level_{level}")
        names.append(_next_available_column_name([*used, *names], base))
    index_columns = index.to_frame(index=False)
    index_columns.columns = names
    return pd.concat([index_columns, frame.reset_index(drop=True)], axis=1)


def _load_dataframe_source(
    source: io.BytesIO | str | Path,
    filename: str,
    *,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_cells: int | None = None,
) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    limits = {"max_rows": max_rows, "max_columns": max_columns, "max_cells": max_cells}
    if suffix in {".csv", ".txt", ".tsv"}:
        df = _read_csv_with_fallback(source, default_delimiter="\t" if suffix == ".tsv" else ",", **limits)
    elif suffix in {".xlsx", ".xls"}:
        df = _read_excel_limited(source, **limits)
    elif suffix == ".parquet":
        df = _read_parquet_frame(source)
    else:
        raise ValueError(
            f"Unsupported input file extension '{suffix or '<none>'}' for '{filename}'. "
            "Supported formats are CSV, TSV, XLSX, XLS, and Parquet."
        )

    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")
    source_encoding = df.attrs.get("source_encoding")
    # Every load path returns a unique 0..n-1 row index: cohort rows are matched to the stored
    # table by index (source_row_index), which a stored or repeated index would break.
    df = _with_default_row_index(df)
    df.columns = make_unique_columns(df.columns)
    if source_encoding:
        df.attrs["source_encoding"] = source_encoding
    return df


def load_dataframe(
    file_bytes: bytes,
    filename: str,
    *,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_cells: int | None = None,
) -> pd.DataFrame:
    return _load_dataframe_source(
        io.BytesIO(file_bytes),
        filename,
        max_rows=max_rows,
        max_columns=max_columns,
        max_cells=max_cells,
    )


@user_input_boundary
def load_dataframe_from_path(
    path: str | Path,
    *,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_cells: int | None = None,
) -> pd.DataFrame:
    """Load a CSV/TSV/Excel/Parquet table.

    With ``max_rows`` / ``max_columns`` / ``max_cells`` set, text and Excel inputs are
    checked against the limits from the header and a bounded read (one row past the
    limit) instead of parsing the whole file first.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"Input path is not a file: {path_obj}")
    return _load_dataframe_source(
        path_obj,
        path_obj.name,
        max_rows=max_rows,
        max_columns=max_columns,
        max_cells=max_cells,
    )


def serialize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def preview_rows(df: pd.DataFrame, n_rows: int = 12) -> list[dict[str, Any]]:
    preview = df.head(n_rows).copy()
    return [
        {column: serialize_value(value) for column, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]


def _column_kind(series: pd.Series) -> str:
    unique_count = int(series.nunique(dropna=True))
    if unique_count <= 2 and not is_datetime64_any_dtype(series):
        return "binary"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _column_matches_keyword(column: str, token: str) -> bool:
    lowered = str(column).lower()
    normalized_token = str(token).lower()
    if not lowered or not normalized_token:
        return False
    if len(normalized_token) >= 4:
        return normalized_token in lowered
    pluralizable_time_units = {"day", "week", "month", "year"}
    suffix = "s?" if normalized_token in pluralizable_time_units else ""
    pattern = rf"(^|[^a-z0-9]){re.escape(normalized_token)}{suffix}([^a-z0-9]|$)"
    return re.search(pattern, lowered) is not None


def _column_keywords(columns: Sequence[str], tokens: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        if any(_column_matches_keyword(column, token) for token in tokens):
            matches.append(column)
    return matches


def _normalize_column_label(name: str) -> str:
    return str(name or "").strip().lower()


def _token_variants(token: str | None) -> tuple[str, ...]:
    raw = str(token or "").strip().lower()
    if not raw:
        return ()
    variants: list[str] = []

    def _add(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    _add(raw)
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    _add(compact)
    for part in re.split(r"[^a-z0-9]+", raw):
        _add(part)
    return tuple(variants)


def _column_name_tokens(name: str) -> list[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name or ""))
    return [token for token in re.split(r"[^a-z0-9]+", expanded.lower()) if token]


def _tokens_contain_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not tokens or not phrase or len(tokens) < len(phrase):
        return False
    width = len(phrase)
    return any(tuple(tokens[idx:idx + width]) == tuple(phrase) for idx in range(len(tokens) - width + 1))


def _endpoint_family_from_column_name(name: str) -> str | None:
    tokens = _column_name_tokens(name)
    if not tokens:
        return None
    token_set = set(tokens)
    phrase_families = (
        ("os", ("overall", "survival")),
        ("pfs", ("progression", "free")),
        ("dfs", ("disease", "free")),
        ("rfs", ("recurrence", "free")),
        ("rfs", ("relapse", "free")),
        ("efs", ("event", "free")),
        ("dss", ("disease", "specific")),
        ("css", ("cancer", "specific")),
    )
    for family, phrase in phrase_families:
        if _tokens_contain_phrase(tokens, phrase):
            return family
    # "tte" and "tts" (time to event, time to ...) are generic, so they pair with any endpoint.
    for family in ("os", "pfs", "dfs", "rfs", "efs", "dss", "css", "pfi", "dfi"):
        if family in token_set:
            return family
    return None


def _validate_endpoint_family_pair(time_column: str, event_column: str) -> None:
    time_family = _endpoint_family_from_column_name(time_column)
    event_family = _endpoint_family_from_column_name(event_column)
    if time_family and event_family and time_family != event_family:
        raise ValueError(
            f'"{time_column}" and "{event_column}" look like different survival endpoints. '
            "Choose a matched time/event pair from the same endpoint family."
        )


_COMPACT_TIME_COLUMN_NAMES = {
    "futime",
    "stime",
    "ftime",
    "survtime",
    "ostime",
    "pfstime",
    "dfstime",
    "rfstime",
    "efstime",
    "dsstime",
    "tte",
    "t2e",
}
_NON_TIME_QUANTITY_TOKENS = {
    "score",
    "scores",
    "risk",
    "index",
    "prob",
    "probability",
    "group",
    "grade",
    "stage",
    "category",
    "class",
    "cluster",
    "signature",
    "prediction",
    "predicted",
    "ratio",
    "rate",
}


def _looks_like_survival_time_column_name(name: str) -> bool:
    normalized = _normalize_column_label(name)
    tokens = _column_name_tokens(name)
    if not normalized or not tokens:
        return False

    token_set = set(tokens)
    if token_set & {"date", "dt", "datetime", "timestamp"}:
        # Calendar dates (e.g. last_followup_date) are not durations.
        return False
    has_time_units = bool(token_set & _TIME_UNIT_TOKENS)
    has_abbreviation = bool(token_set & _SURVIVAL_ENDPOINT_ABBREVIATIONS)
    has_time_keyword = "time" in token_set
    has_followup = (
        "followup" in token_set
        or _tokens_contain_phrase(tokens, ("follow", "up"))
        or "fu" in token_set
        or "fup" in token_set
        # "days_to_last_contact", "days_to_last_known_alive": the last follow-up by another name.
        or any(_tokens_contain_phrase(tokens, ("last", word)) for word in ("contact", "known", "seen"))
    )
    # "Disease Free (Months)", "Recurrence Free Months": "<word> free" names an endpoint.
    has_free_phrase = any(_tokens_contain_phrase(tokens, (word, "free")) for word in _ENDPOINT_FREE_PHRASE_WORDS)
    # "overall_time", "obs_time": an overall or observation time, but only together with a
    # time word ("overall_response" or "observed_value" are not times).
    has_qualified_time = bool(token_set & ({"overall"} | _OBSERVATION_TOKENS)) and (has_time_keyword or has_time_units)
    has_survival_context = (
        has_abbreviation
        or "survival" in token_set
        or "surv" in token_set
        or has_followup
        or has_free_phrase
        or has_qualified_time
        # Time to next treatment (TTNT) spelled out.
        or _tokens_contain_phrase(tokens, ("next", "treatment"))
    )
    has_event_context = bool(token_set & _OUTCOME_CONTEXT_TOKENS)
    has_duration_keyword = "duration" in token_set
    has_generic_time_context = has_time_keyword or has_duration_keyword
    # "time_to_surgery_days", "time_since_diagnosis_years": a time to or since a clinical milestone
    # that is neither an outcome nor follow-up is a baseline interval.
    names_other_interval = bool(token_set & {"to", "since"}) and not (has_survival_context or has_event_context)

    if normalized in {"time", "survival_time", "event_time", "time_to_event", "followup_time"}:
        return True
    # Common one-word conventions (R survival: futime, stime; tte = time to event).
    if normalized in _COMPACT_TIME_COLUMN_NAMES or (len(tokens) == 1 and tokens[0] in _COMPACT_TIME_COLUMN_NAMES):
        return True
    # Scores, risks, indices, and groupings are derived quantities, not follow-up
    # times, even when they carry an endpoint prefix such as "os_risk_score".
    if token_set & _NON_TIME_QUANTITY_TOKENS and not (has_time_units or has_generic_time_context or has_followup):
        return False
    if _looks_like_baseline_status_column(name) and not (has_survival_context or has_event_context):
        return False
    if has_duration_keyword and has_time_units and not (has_survival_context or has_event_context):
        return False
    if has_survival_context and (has_time_units or has_generic_time_context or not has_event_context):
        return True
    if has_generic_time_context and (len(tokens) == 1 or has_survival_context or has_event_context):
        return True
    if has_time_units and (has_survival_context or has_event_context or (has_time_keyword and not names_other_interval)):
        return True
    return False


def _absence_of_event_name(name: str) -> str | None:
    """"free" or "alive" when a column name says the event has not happened, else None.

    "progression_free", "event_free", "disease_free", "relapse_free_status": the patient is
    still free of the event. "alive", "is_alive", "living", "survived": the patient has not
    died. "Progression-free survival" (or "... months") names an endpoint instead, and a name
    that also names death ("alive_or_dead") is left alone.
    """
    tokens = _column_name_tokens(name)
    if set(tokens) & _DEATH_NAME_TOKENS:
        return None
    for position, token in enumerate(tokens):
        joined = token.endswith("free") and token[: -len("free")] in _ENDPOINT_FREE_PHRASE_WORDS  # "EventFree"
        phrase = token == "free" and position > 0 and tokens[position - 1] in _ENDPOINT_FREE_PHRASE_WORDS
        if (joined or phrase) and not set(tokens[position + 1 :]) & _FREE_PHRASE_ENDPOINT_TOKENS:
            return "free"
    if set(tokens) & _ALIVE_NAME_TOKENS:
        return "alive"
    return None


def _is_event_like_column_name(name: str) -> bool:
    normalized = _normalize_column_label(name)
    if not normalized:
        return False
    if normalized in {"event", "status"}:
        return True
    if any(pattern.search(normalized) for pattern in _EVENT_NAME_PATTERNS):
        return True
    # Death, "alive" and censoring flags, and "free of the event" flags, restate the outcome too.
    return bool(set(_column_name_tokens(name)) & _OUTCOME_NAME_TOKENS) or _absence_of_event_name(name) == "free"


# Words that make a "status" column name an outcome name ("vital_status", "os_status", "survival_status").
_OUTCOME_STATUS_NAME_TOKENS = {
    "survival",
    "surv",
    "outcome",
    "vital",
    "censor",
    "censored",
    "censoring",
    "cens",
    "alive",
    "living",
    "follow",
    "followup",
    "fu",
    "fup",
    *_SURVIVAL_ENDPOINT_ABBREVIATIONS,
}


def _is_generic_status_name(name: str) -> bool:
    """A "<something>_status" name whose only outcome word is "status" ("diabetes_status", "marital_status").

    Such columns are usually baseline characteristics. A bare "status" ("status", "status2") and
    names that also name the outcome or an endpoint ("vital_status", "os_status",
    "death_status", "Disease Free Status") are outcome names.
    """
    normalized = _normalize_column_label(name)
    if "status" not in normalized:
        return False
    tokens = _column_name_tokens(name)
    if not [token for token in tokens if token != "status" and not token.isdigit()]:
        return False
    if set(tokens) & _OUTCOME_STATUS_NAME_TOKENS or _name_has_strong_event_word(name):
        return False
    return _absence_of_event_name(name) is None


def _values_name_the_outcome(series: pd.Series) -> bool:
    """True when a value names an outcome state itself (Dead, Alive, Relapsed, NED, "No recurrence").

    Yes/No, True/False and 0/1 codes do not: under a name such as "diabetes_status" they describe
    a baseline characteristic.
    """
    for value in _unique_non_missing_values(series):
        normalized = _normalize_token(value)
        if normalized is None or normalized in _GENERIC_BINARY_CODE_TOKENS:
            continue
        family, _exact_match = _outcome_status_family_match(value)
        if family in {"event_death", "event_progression", "censor"}:
            return True
    return False


def _looks_like_baseline_status_column(name: str) -> bool:
    normalized = _normalize_column_label(name)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _BASELINE_STATUS_PATTERNS)


def _name_has_strong_event_word(name: str) -> bool:
    normalized = _normalize_column_label(name)
    return bool(normalized) and any(pattern.search(normalized) for pattern in _STRONG_EVENT_NAME_PATTERNS)


def _is_baseline_named_column(name: str) -> bool:
    """A baseline-looking name without a word that names the event itself.

    "her2_status" or "treatment_arm" are baseline characteristics; "treatment_failure" and
    "progression_on_therapy" name an event despite their baseline words.
    """
    return _looks_like_baseline_status_column(name) and not _name_has_strong_event_word(name)


def _unique_non_missing_values(series: pd.Series) -> list[Any]:
    """Distinct non-missing values, so per-value Python checks run once per level, not per cell."""
    valid = series.dropna()
    try:
        return list(pd.unique(valid))
    except TypeError:  # unhashable cells
        return valid.tolist()


def _numeric_unique_values(series: pd.Series) -> np.ndarray | None:
    """Sorted distinct values when every non-missing value is a number, else None."""
    if not isinstance(series, pd.Series):  # a duplicated column label selects a frame
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    if is_bool_dtype(series):
        return np.unique(valid.to_numpy(dtype=float))
    numeric = pd.to_numeric(valid, errors="coerce")
    if bool(numeric.isna().any()):
        return None
    return np.unique(numeric.to_numpy(dtype=float, na_value=np.nan))


def _values_decode_as_event_coding(series: pd.Series) -> bool:
    """True when the values read as an event indicator.

    That is booleans, numbers coded 0/1 (or 1/2), or labels with a recognized outcome meaning
    (dead/alive, relapse, progression, censored, ...). Two-valued biomarker labels such as
    Positive/Negative, MSI-H/MSS, or Mutant/Wildtype do not qualify.
    """
    if series.dropna().empty:
        return False
    if is_bool_dtype(series):
        return True
    numeric_values = _numeric_unique_values(series)
    if numeric_values is not None:
        observed = set(numeric_values.tolist()) if numeric_values.size <= 2 else None
        return observed is not None and (observed <= {0.0, 1.0} or observed <= {1.0, 2.0})
    families = {_outcome_status_value_family(value) for value in _unique_non_missing_values(series)}
    return bool(families - {None})


def _outcome_status_value_family(value: Any) -> str | None:
    family, _exact_match = _outcome_status_family_match(value)
    return family


_NEGATED_STATUS_PATTERN = re.compile(r"^(no|not|non|never|without)\b")
# A negation word followed by more words ("No recurrence", "Not reported").
_NEGATED_PHRASE_PATTERN = re.compile(r"^(no|not|non|never|without)\b\W*[a-z]")
# The event words a negation must apply to: "No recurrence" is censoring, while "Not reported",
# "No data" or "Never assessed" are missing values.
_NEGATABLE_EVENT_WORD_PATTERN = re.compile(
    r"event|death|dead|died|deceas|mort|relaps|recur|progress|fail|metasta|diseas|tumou?r|evidence"
)
_LEADING_STATUS_CODE_PATTERN = re.compile(r"^\s*\d+\s*[:=.)\-]\s*")


def _outcome_status_family_match(value: Any) -> tuple[str | None, bool]:
    normalized = _normalize_token(value)
    if normalized is None:
        return None, False
    # Negated labels ("No recurrence", "0:Not Recurred", "non-progressor",
    # "relapse-free") describe the absence of the event, i.e. censoring.
    label_text = _LEADING_STATUS_CODE_PATTERN.sub("", normalized).strip()
    if label_text and (
        (_NEGATED_STATUS_PATTERN.match(label_text) and _NEGATABLE_EVENT_WORD_PATTERN.search(label_text))
        or re.search(r"(^|[^a-z])free($|[^a-z])", label_text)
    ):
        return "censor", True
    if label_text and _NEGATED_PHRASE_PATTERN.match(label_text):
        # A negated phrase without an event word ("Not reported", "No data", "Never married") is no
        # outcome label; its "no" part must not read as censoring below.
        return None, False
    for candidate in _token_variants(normalized):
        family = _OUTCOME_STATUS_VALUE_FAMILIES.get(candidate)
        if family is not None:
            return family, candidate == normalized
    for candidate in _token_variants(normalized):
        if candidate in _EVENT_TOKEN_FAMILIES:
            return _EVENT_TOKEN_FAMILIES[candidate], candidate == normalized
    return None, False


def _has_ambiguous_competing_event_tokens(tokens: Sequence[str]) -> bool:
    token_details = {
        token: _outcome_status_family_match(token)
        for token in tokens
    }
    recognized_non_censor_families = {
        family
        for family, _exact_match in token_details.values()
        if family is not None and family != "censor"
    }
    specific_non_censor_families = {
        family
        for family in recognized_non_censor_families
        if family != "event_generic"
    }
    recognized_non_censor = [
        token
        for token, (family, _exact_match) in token_details.items()
        if family is not None and family != "censor"
    ]
    inferred_non_censor = [
        token
        for token, (family, exact_match) in token_details.items()
        if family is not None and family != "censor" and not exact_match
    ]
    return len(specific_non_censor_families) > 1 or (bool(inferred_non_censor) and len(recognized_non_censor) > 1)


def _looks_like_event_outcome_column(name: str, series: pd.Series) -> bool:
    """An event/status-named column whose values decode as event coding.

    Baseline-looking names (biomarker, receptor, or clinical "status" columns such as
    her2_status or nodal_status) are not outcomes unless the name also names the event
    ("treatment_failure"), and two-valued labels that are not event coding
    (Positive/Negative, MSI-H/MSS) never make a column an outcome. Other "<something>_status"
    names (diabetes_status, marital_status, hiv_status) are outcomes only when their values
    name an outcome state (Dead/Alive, Relapsed, NED), not with Yes/No or 0/1 codes.
    """
    if not _is_event_like_column_name(name) or _is_baseline_named_column(name):
        return False
    if _is_generic_status_name(name):
        return _values_name_the_outcome(series)
    return _values_decode_as_event_coding(series)


def _has_recognizable_event_coding(series: pd.Series) -> bool:
    valid = series.dropna()
    if valid.empty:
        return False

    numeric_values = _numeric_unique_values(series)
    if numeric_values is not None:
        if numeric_values.size <= 2:
            observed_numeric = set(numeric_values.tolist())
            if observed_numeric in ({0.0, 1.0}, {1.0, 2.0}):
                return True
            if len(observed_numeric) == 1 and next(iter(observed_numeric)) in {0.0, 1.0, 2.0}:
                return True
        # Among plain numbers only 1 reads as an event token (0 means censored).
        return bool(np.any(numeric_values == 1.0))

    families = {_outcome_status_value_family(value) for value in _unique_non_missing_values(series)}
    return bool(families - {None, "censor"})


def _model_feature_candidate_columns_from_metadata(
    columns: Sequence[str],
    *,
    suggested_time_columns: Sequence[str],
    binary_candidate_columns: Sequence[str],
) -> list[str]:
    suggested_time_set = set(suggested_time_columns)
    binary_set = set(binary_candidate_columns)
    candidates: list[str] = []
    for column in columns:
        if _MODEL_FEATURE_ID_PATTERN.fullmatch(column):
            continue
        # Time suggestions leave out 0/1 columns, so time-like names are checked directly too.
        if column in suggested_time_set or _looks_like_survival_time_column_name(column):
            continue
        # Only the names are known here, so a two-valued "<something>_status" column counts as the
        # baseline characteristic it usually is.
        if (
            column in binary_set
            and _is_event_like_column_name(column)
            and not _is_baseline_named_column(column)
            and not _is_generic_status_name(column)
        ):
            continue
        candidates.append(column)
    return candidates


def _survival_outcome_like_columns(df: pd.DataFrame) -> set[str]:
    likely_time_columns = {column for column in df.columns if _looks_like_survival_time_column_name(column)}
    likely_event_columns = {
        column for column in df.columns if _looks_like_event_outcome_column(column, df[column])
    }
    return likely_time_columns | likely_event_columns


def _normalize_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return str(int(value))
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value) in (0.0, 1.0):
            return str(int(value))
        return None
    return str(value).strip().lower()


_OTHER_CAUSE_DEATH_TOKENS = {
    "doc",
    "diedofothercause",
    "diedofothercauses",
    "deadofothercause",
    "deadofothercauses",
    "deathofothercause",
    "deathfromothercauses",
    "othercausedeath",
}


def _reject_other_cause_deaths_for_cause_specific_endpoint(series: pd.Series) -> None:
    """Deaths from other causes are competing events for disease-specific endpoints.

    Coding them as events (as for overall survival) would silently turn a
    disease-specific analysis into an all-cause one, so require explicit recoding.
    """
    if _endpoint_family_from_column_name(str(series.name or "")) not in {"dss", "css"}:
        return
    for value in series.dropna().unique().tolist():
        normalized = _normalize_token(value)
        if normalized is None:
            continue
        if any(variant in _OTHER_CAUSE_DEATH_TOKENS for variant in _token_variants(normalized)):
            raise ValueError(
                f'The disease-specific event column "{series.name}" contains deaths from other causes '
                f'("{value}"). These are competing events, not disease-specific events: recode them explicitly '
                "(censored for a cause-specific analysis) before survival analysis."
            )


_MULTIPLE_EVENT_STATES_MESSAGE = (
    "The event column contains more than one recognized event state. "
    "Recode it to a binary event indicator before survival analysis."
)


def _explicit_event_token(value: Any) -> str | None:
    """Normalize one event value for matching against a user-chosen event label."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        return str(float_value).strip().lower()
    return str(value).strip().lower()


def _numeric_event_indicator(
    numeric_series: pd.Series,
    valid: pd.Series,
    target: Any,
    target_numeric: float,
) -> pd.Series:
    """Code a fully numeric event column against a numeric event-positive value.

    Only truly binary coding is allowed: multi-state numeric status columns are
    rejected rather than silently collapsing extra states into censoring.
    """
    if _normalize_token(target) in FALSE_TOKENS:
        raise ValueError(
            f"The selected event-positive value '{target}' maps to censoring, not the event."
        )
    observed_numeric = sorted({float(value) for value in numeric_series.loc[valid].astype(float).tolist()})
    if target_numeric not in observed_numeric:
        raise ValueError(
            f"The selected event-positive value '{target}' is not present in the numeric event column."
        )
    if len(observed_numeric) > 2:
        raise ValueError(
            "The numeric event column has more than two distinct states. "
            "Provide a pre-binarized event indicator before survival analysis."
        )
    return (numeric_series.loc[valid] == target_numeric).astype(float)


def _raise_if_unrecognized_event_tokens(observed_family_map: dict[str, str | None]) -> None:
    unknown = [token for token, family in observed_family_map.items() if family is None]
    if unknown:
        raise ValueError(
            "Event coding contains unrecognized tokens alongside standard event/censor labels: "
            + ", ".join(unknown[:6])
            + (" ..." if len(unknown) > 6 else "")
        )


def _event_tokens_for_label(target: Any, target_token: str, observed_tokens: list[str]) -> set[str]:
    """The observed tokens that mean "event" when the user picked the label `target`."""
    if _has_ambiguous_competing_event_tokens(observed_tokens):
        raise ValueError(_MULTIPLE_EVENT_STATES_MESSAGE)
    observed_family_map = {token: _outcome_status_family_match(token)[0] for token in observed_tokens}
    observed_families = {family for family in observed_family_map.values() if family is not None}

    if target_token in TRUE_TOKENS:
        # A standard event label: decode the column with the full token vocabulary.
        _raise_if_unrecognized_event_tokens(observed_family_map)
        target_family = _outcome_status_value_family(target_token)
        if target_family is None:
            raise ValueError("The selected event-positive value could not be mapped to a supported event family.")
        if target_family == "event_generic":
            concrete_event_families = sorted(observed_families - {"censor"})
            if len(concrete_event_families) == 1:
                target_family = concrete_event_families[0]
        if observed_families - {target_family, "censor"}:
            raise ValueError(_MULTIPLE_EVENT_STATES_MESSAGE)
        return {token for token, family in observed_family_map.items() if family == target_family}

    if target_token in FALSE_TOKENS:
        _raise_if_unrecognized_event_tokens(observed_family_map)
        raise ValueError(
            f"The selected event-positive value '{target}' maps to censoring, not the event. "
            "Choose the value that means the event happened."
        )

    # Custom label: allow mapping only for true binary columns to avoid masking typos.
    target_family_match, _ = _outcome_status_family_match(target_token)
    if target_family_match == "censor":
        raise ValueError(
            f"The selected event-positive value '{target}' means no event (censoring), not the event. "
            "Choose the value that means the event happened."
        )
    if target_token not in observed_tokens:
        raise ValueError(
            f"The selected event-positive value '{target}' is not present in the event column."
        )
    if len(observed_tokens) > 2:
        raise ValueError(
            "The event column has more than two distinct values after normalization. "
            "For multi-class status columns, please recode to a binary event indicator."
        )
    return {target_token}


# Markers that mean "not recorded" in an event column, as in time columns. "None" and "-" are left
# out: in an event column they can mean "no event" ("Recurrence: None") or a negative result.
_EVENT_MISSING_VALUE_TOKENS = frozenset(_MISSING_VALUE_TOKENS - {"none", "-", "--"})


def _without_missing_event_markers(series: pd.Series) -> pd.Series:
    """The event column with missing-value markers ("[Not Available]", "Not reported", blank text) as missing."""
    if is_bool_dtype(series) or is_numeric_dtype(series):
        return series
    markers = [
        value
        for value in _unique_non_missing_values(series)
        if isinstance(value, str) and value.strip().lower() in _EVENT_MISSING_VALUE_TOKENS
    ]
    if not markers:
        return series
    return series.mask(series.isin(markers))


def coerce_event(series: pd.Series, event_positive_value: Any = None) -> pd.Series:
    # Missing-value markers are missing before any coding, as they are in the time column; the
    # negation rule would otherwise read "Not reported" as censored.
    series = _without_missing_event_markers(series)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()
    if not valid.any():
        raise ValueError("The event column contains only missing values.")
    _reject_other_cause_deaths_for_cause_specific_endpoint(series)

    if event_positive_value is not None and event_positive_value != "":
        target = event_positive_value
        numeric_series = pd.to_numeric(series, errors="coerce")
        try:
            target_numeric = float(target)
        except (TypeError, ValueError):
            target_numeric = None
        if target_numeric is not None and numeric_series.notna().sum() == valid.sum():
            out.loc[valid] = _numeric_event_indicator(numeric_series, valid, target, target_numeric)
            return out

        target_token = _explicit_event_token(target)
        if target_token is None:
            raise ValueError("The selected event-positive value could not be parsed.")
        value_tokens = series.map(_explicit_event_token)
        if value_tokens.loc[valid].isna().any():
            raise ValueError(
                "The event column contains non-missing values that cannot be normalized for event coding."
            )
        observed_tokens = sorted(set(value_tokens.loc[valid].astype(str).tolist()))
        event_tokens = _event_tokens_for_label(target, target_token, observed_tokens)
        out.loc[valid] = value_tokens.loc[valid].isin(event_tokens).astype(float)
        return out

    inferred, inference_error = _try_coerce_binary_event(series)
    if inferred is not None:
        return inferred
    if inference_error == "multistate":
        raise ValueError(_MULTIPLE_EVENT_STATES_MESSAGE)

    raise ValueError(
        "Could not infer event coding. Select the value that represents the event in the dashboard."
    )


def looks_binary(series: pd.Series) -> bool:
    valid = series.dropna()
    if valid.empty:
        return False

    inferred, inference_error = _try_coerce_binary_event(series)
    if inferred is not None:
        non_missing = inferred.dropna()
        return not non_missing.empty and set(non_missing.unique()).issubset({0.0, 1.0})
    if inference_error == "multistate":
        return False

    numeric_values = _numeric_unique_values(series)
    if numeric_values is not None:
        return int(numeric_values.size) == 2

    # Per-value checks run once per distinct value, not once per cell.
    tokens = [_normalize_token(value) for value in _unique_non_missing_values(series)]
    if all(token is not None for token in tokens):
        token_set = set(tokens)
        observed_families = {_outcome_status_value_family(token) for token in token_set} - {None}
        if not observed_families:
            return len(token_set) == 2
        event_families = observed_families - {"censor"}
        return len(event_families) == 1

    return int(valid.nunique(dropna=True)) == 2


def _try_coerce_binary_event(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    valid = series.notna()
    if not valid.any():
        return pd.Series(np.nan, index=series.index, dtype=float), None

    out = pd.Series(np.nan, index=series.index, dtype=float)
    if is_bool_dtype(series):
        out.loc[valid] = series.loc[valid].astype(int).astype(float)
        return out, None

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() == int(valid.sum()):
        unique_floats = set(numeric_series.loc[valid].unique().tolist())
        if unique_floats.issubset({0.0, 1.0}):
            out.loc[valid] = numeric_series.loc[valid].astype(float)
            return out, None
        # Every value is a number, and numbers other than 0 and 1 carry no event meaning
        # (a decimal such as "1.5" must not be split into the tokens "1" and "5").
        return None, "unrecognized"

    # Families are looked up once per distinct value instead of once per cell.
    unique_values = _unique_non_missing_values(series)
    try:
        family_by_value = {value: _outcome_status_value_family(value) for value in unique_values}
        normalized_families = series.map(family_by_value)
    except TypeError:  # unhashable cells
        family_by_value = None
        normalized_families = series.map(_outcome_status_value_family)
    mapped = pd.Series(np.nan, index=series.index, dtype=float)
    mapped.loc[normalized_families.eq("censor").to_numpy(dtype=bool, na_value=False)] = 0.0
    mapped.loc[(normalized_families.notna() & normalized_families.ne("censor")).to_numpy(dtype=bool, na_value=False)] = 1.0
    if mapped.loc[valid].notna().sum() == int(valid.sum()):
        observed_tokens = sorted(
            {token for token in (_normalize_token(value) for value in unique_values) if token is not None}
        )
        if _has_ambiguous_competing_event_tokens(observed_tokens):
            return None, "multistate"
        if family_by_value is not None:
            observed_families = {str(family) for family in family_by_value.values() if family is not None}
        else:
            observed_families = {family for family in normalized_families.loc[valid].dropna().astype(str).tolist()}
        event_families = observed_families - {"censor"}
        if len(event_families) > 1:
            return None, "multistate"
        return mapped, None

    return None, "unrecognized"


@user_input_boundary
def find_event_equivalent_columns(
    df: pd.DataFrame,
    event_column: str,
    event_positive_value: Any = None,
) -> set[str]:
    if event_column not in df.columns:
        return set()
    try:
        reference = coerce_event(df[event_column], event_positive_value=event_positive_value)
    except ValueError:
        return set()

    reference_valid = reference.notna()
    if not reference_valid.any():
        return set()

    equivalents: set[str] = set()
    reference_values = reference.to_numpy(dtype=float, na_value=np.nan)
    for column in df.columns:
        if column == event_column or _is_baseline_named_column(str(column)):
            continue
        series = df[column]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            # A numeric column can only restate the event when it is coded 0/1; checking that
            # first keeps wide numeric uploads (thousands of gene columns) cheap.
            numeric_values = _numeric_unique_values(series)
            if numeric_values is None or numeric_values.size > 2 or not set(numeric_values.tolist()) <= {0.0, 1.0}:
                continue
        elif not looks_binary(series) and not _has_recognizable_event_coding(series):
            continue
        try:
            candidate = coerce_event(series)
        except ValueError:
            continue
        overlap = (reference_valid & candidate.notna()).to_numpy(dtype=bool)
        if int(overlap.sum()) < 3:
            continue
        candidate_overlap = candidate.to_numpy(dtype=float, na_value=np.nan)[overlap]
        reference_overlap = reference_values[overlap]
        # A copy of the event, or its complement (a censoring or "alive" flag, 1 - event),
        # restates the outcome.
        if np.array_equal(candidate_overlap, reference_overlap) or np.array_equal(candidate_overlap, 1.0 - reference_overlap):
            equivalents.add(str(column))
    return equivalents


def _is_zero_one_indicator(series: pd.Series) -> bool:
    """True when every non-missing value is 0 or 1: an indicator, never a follow-up duration."""
    numeric_values = _numeric_unique_values(series)
    return numeric_values is not None and 0 < numeric_values.size <= 2 and set(numeric_values.tolist()) <= {0.0, 1.0}


def _is_endpoint_abbreviation_name(name: str) -> bool:
    """A name that is just an endpoint abbreviation, such as the TCGA CDR columns OS, PFI, DSS."""
    tokens = _column_name_tokens(name)
    return len(tokens) == 1 and tokens[0] in _SURVIVAL_ENDPOINT_ABBREVIATIONS


def suggest_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(df.columns)
    event_tokens = ("event", "status", "death", "progress", "relapse")
    # A 0/1 column is an event indicator even when its name reads like an endpoint ("OS").
    time_columns = [
        column
        for column in columns
        if _looks_like_survival_time_column_name(column) and not _is_zero_one_indicator(df[column])
    ]
    keyword_event_columns = set(_column_keywords(columns, event_tokens))
    event_columns = []
    for column in columns:
        name = str(column)
        if _is_endpoint_abbreviation_name(name) and _is_zero_one_indicator(df[column]):
            # TCGA Pan-Cancer CDR style: "OS", "PFI", "DFI", "DSS" hold the 0/1 events.
            event_columns.append(column)
        elif (
            column in keyword_event_columns
            and not _is_baseline_named_column(name)
            # "diabetes_status" coded Yes/No is a baseline characteristic.
            and not (_is_generic_status_name(name) and not _values_name_the_outcome(df[column]))
            # Censoring or "free of the event" flags coded 0/1 or Yes/No mean 1 = no event.
            and _no_event_indicator_message(df, column) is None
        ):
            event_columns.append(column)
    suggestions = {
        "time_columns": time_columns,
        "event_columns": event_columns,
        "group_columns": _column_keywords(columns, ("group", "arm", "treatment", "stage", "sex", "risk", "cluster")),
    }
    return suggestions


def _profile_dataframe_column(column: str, series: pd.Series) -> tuple[dict[str, Any], bool, bool]:
    kind = _column_kind(series)
    profile = {
        "name": str(column),
        "kind": kind,
        "missing": int(series.isna().sum()),
        "non_missing": int(series.notna().sum()),
        "n_unique": int(series.nunique(dropna=True)),
        "unique_preview": [serialize_value(value) for value in series.dropna().unique()[:8]],
    }
    is_numeric = bool(is_numeric_dtype(series))
    if is_numeric:
        profile["min"] = serialize_value(series.min())
        profile["max"] = serialize_value(series.max())
    is_binary = looks_binary(series)
    return profile, is_numeric, is_binary


def _require_dataframe_columns(df: pd.DataFrame, columns: Sequence[str | None]) -> None:
    requested: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column is None:
            continue
        name = str(column)
        if not name or name in seen:
            continue
        requested.append(name)
        seen.add(name)
    missing = [name for name in requested if name not in df.columns]
    if not missing:
        return
    if len(missing) == 1:
        raise ColumnNotFoundError(f'Column not found in dataset: "{missing[0]}".')
    quoted = ", ".join(f'"{name}"' for name in missing)
    raise ColumnNotFoundError(f"Columns not found in dataset: {quoted}.")


# Identifier names end in an identifier word ("patient_id", "tcga_barcode", "SubjectID"), name the
# entity itself ("patient", "sample"), or number it ("case_number", "sample_name"). The entity word
# alone at the front does not make an identifier: "sample_purity" or "patient_age" are measurements.
_IDENTIFIER_HEAD_TOKENS = {"id", "ids", "barcode", "uuid", "mrn", "identifier"}
_IDENTIFIER_ENTITY_TOKENS = {"patient", "subject", "sample", "case", "participant"}
_IDENTIFIER_NUMBER_TOKENS = {"no", "nr", "number", "num", "code", "name", "key"}


def _is_identifier_like_name(name: str) -> bool:
    tokens = _column_name_tokens(name)
    while tokens and tokens[-1].isdigit():  # "patient_id_2" after de-duplicated column names
        tokens.pop()
    if not tokens:
        return False
    head = tokens[-1]
    if head in _IDENTIFIER_HEAD_TOKENS or head in _IDENTIFIER_ENTITY_TOKENS:
        return True
    return head in _IDENTIFIER_NUMBER_TOKENS and any(token in _IDENTIFIER_ENTITY_TOKENS for token in tokens[:-1])


def detect_duplicate_identifier_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Find identifier-like columns whose values repeat across rows.

    Repeated subject identifiers usually mean one subject contributes several
    rows (for example multiple tumour samples). Survival estimators then
    double-count that subject, and row-level train/test splits can place
    copies of the same subject on both sides of the split. Decimal measurements
    are never identifiers, whatever their name.
    """
    findings: list[dict[str, Any]] = []
    for column in df.columns:
        if not _is_identifier_like_name(str(column)):
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        if pd.api.types.is_float_dtype(values.dtype):
            fractional = np.mod(values.to_numpy(dtype=float), 1.0)
            if not bool(np.all(fractional == 0.0)):
                continue
        n_unique = int(values.nunique())
        # Identifier columns are mostly unique; low-cardinality columns such as
        # "sample_type" are attributes, not identifiers.
        if n_unique < 0.5 * len(values) or n_unique == len(values):
            continue
        counts = values.value_counts()
        repeated = counts[counts > 1]
        findings.append(
            {
                "column": str(column),
                "n_rows": int(len(values)),
                "n_unique": n_unique,
                "n_repeated_ids": int(repeated.shape[0]),
                "n_extra_rows": int((repeated - 1).sum()),
            }
        )
    return findings


def duplicate_identifier_caution(df: pd.DataFrame) -> str | None:
    findings = detect_duplicate_identifier_columns(df)
    if not findings:
        return None
    first = findings[0]
    return (
        f"Identifier column '{first['column']}' repeats {first['n_repeated_ids']} value(s) "
        f"({first['n_rows']} rows but {first['n_unique']} unique IDs). If these rows belong to the same subject, "
        "survival estimates double-count them and row-level train/test splits can leak the same subject "
        "into both partitions; keep one row per subject before analysis."
    )


@user_input_boundary
def profile_dataframe(df: pd.DataFrame, dataset_id: str, filename: str) -> dict[str, Any]:
    column_profiles: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    binary_candidate_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        profile, is_numeric, is_binary = _profile_dataframe_column(column, series)
        if is_numeric:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
        if is_binary:
            binary_candidate_columns.append(column)
        column_profiles.append(profile)

    suggestions = suggest_columns(df)
    model_feature_candidates = _model_feature_candidate_columns_from_metadata(
        list(df.columns),
        suggested_time_columns=suggestions.get("time_columns", []),
        binary_candidate_columns=binary_candidate_columns,
    )

    source_encoding = df.attrs.get("source_encoding")
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        # Text uploads record the encoding used to read them, so a legacy encoding
        # (Korean CP949, Windows-1252) is visible instead of silently guessed.
        "text_encoding": source_encoding,
        "text_encoding_label": TEXT_ENCODING_LABELS.get(str(source_encoding)) if source_encoding else None,
        "columns": column_profiles,
        "preview": preview_rows(df),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_candidate_columns": binary_candidate_columns,
        "model_feature_candidate_count": len(model_feature_candidates),
        "suggestions": suggestions,
        "duplicate_identifier_columns": detect_duplicate_identifier_columns(df),
    }


def model_feature_candidate_columns(df: pd.DataFrame) -> list[str]:
    suggestions = suggest_columns(df)
    binary_candidate_columns = [column for column in df.columns if looks_binary(df[column])]
    return _model_feature_candidate_columns_from_metadata(
        list(df.columns),
        suggested_time_columns=suggestions.get("time_columns", []),
        binary_candidate_columns=binary_candidate_columns,
    )


@user_input_boundary
def ensure_model_feature_candidate_limit(
    df: pd.DataFrame,
    *,
    max_features: int = MAX_MODEL_FEATURE_CANDIDATES,
) -> int:
    candidate_count = len(model_feature_candidate_columns(df))
    if candidate_count > max_features:
        raise ValueError(
            f"Dataset exposes {candidate_count} model feature candidates after excluding likely survival endpoint columns. "
            f"SurvStudio supports at most {max_features} model features per dataset upload."
        )
    return candidate_count


def quote_name(name: str) -> str:
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{escaped}")'


def _next_available_column_name(existing_columns: Iterable[Any], base_name: str) -> str:
    used = {str(column) for column in existing_columns}
    if base_name not in used:
        return base_name
    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in used:
            return candidate
        suffix += 1


def _ensure_positive_times(time_values: pd.Series) -> pd.Series:
    """Mask of usable follow-up times.

    Time 0 is valid (for example death on the day of diagnosis) and is kept, as
    in R ``survival`` and lifelines; only negative times are invalid. At least
    one strictly positive time is required.
    """
    valid = time_values >= 0
    if (time_values > 0).sum() == 0:
        raise ValueError("The survival time column must contain positive values.")
    return valid

def _reject_calendar_date_time_column(series: pd.Series, time_column: str) -> None:
    message = (
        f'"{time_column}" contains calendar dates, not follow-up durations. Compute the elapsed time '
        "(for example days from diagnosis to event or last follow-up) before survival analysis."
    )
    if is_datetime64_any_dtype(series):
        raise ValueError(message)
    if _is_text_series(series):
        # Missing-value markers are neither dates nor numbers, and numbers written with separators
        # ("1,234", "12,5", "1.234,5") are numbers: the date parser would read "1,234" as the year 234.
        sample = _number_text(series).head(200)
        if sample.empty:
            return
        numbers = pd.to_numeric(sample, errors="coerce").notna() | sample.str.fullmatch(_SEPARATED_NUMBER_PATTERN)
        if numbers.mean() > 0.5:
            return
        with suppressed_warnings():
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.8:
            raise ValueError(message)


def _reject_event_indicator_time_column(series: pd.Series, time_column: str) -> None:
    """A "time" column that holds only 0 and 1 is an event indicator, not follow-up time."""
    numeric_values = _numeric_unique_values(series)
    if numeric_values is not None and numeric_values.size == 2 and set(numeric_values.tolist()) == {0.0, 1.0}:
        raise ValueError(
            f'"{time_column}" holds only the values 0 and 1, which looks like an event indicator rather than '
            "follow-up time. Choose the column with the time to the event or last follow-up."
        )


def _validate_time_column_choice(df: pd.DataFrame, time_column: str) -> str | None:
    """Check the chosen time column; return a caution when its name does not look like time.

    Calendar dates and 0/1 indicators are refused. A name that the heuristics do not
    recognize as follow-up time is allowed (standard endpoints are named in many ways), but
    the returned note is shown with the results so an unusual choice stays visible, also when
    no other column looks like a follow-up time.
    """
    _require_dataframe_columns(df, [time_column])
    _reject_calendar_date_time_column(df[time_column], time_column)
    _reject_event_indicator_time_column(df[time_column], time_column)
    if _looks_like_survival_time_column_name(time_column):
        return None
    likely_time_columns = [column for column in suggest_columns(df).get("time_columns", []) if column != time_column]
    alternatives = ""
    if likely_time_columns:
        alternatives = f" (columns that do: {', '.join(str(column) for column in likely_time_columns[:3])})"
    return (
        f'"{time_column}" does not look like a survival follow-up time column{alternatives}. '
        "Check that it holds the time from the start of follow-up to the event or last follow-up."
    )


_CENSORING_NAME_TOKENS = {"censor", "censored", "cens", "censoring"}
_EVENT_NAME_OVERRIDE_TOKENS = {"event", "death", "dead", "died", "status", "vital"}
# Codes that carry no event meaning of their own: under a "censored" name, 1/Yes/True means censored.
_GENERIC_BINARY_CODE_TOKENS = {"0", "1", "yes", "no", "y", "n", "true", "false", "t", "f"}


def _has_only_generic_binary_codes(series: pd.Series) -> bool:
    series = _without_missing_event_markers(series)
    if is_bool_dtype(series):
        return not series.dropna().empty
    numeric_values = _numeric_unique_values(series)
    if numeric_values is not None:
        return numeric_values.size <= 2 and set(numeric_values.tolist()) <= {0.0, 1.0}
    tokens = {str(value).strip().lower() for value in _unique_non_missing_values(series)}
    return bool(tokens) and tokens <= _GENERIC_BINARY_CODE_TOKENS


def _no_event_indicator_message(df: pd.DataFrame, event_column: str) -> str | None:
    """Why a generically coded column cannot be the event column, or None when it can.

    SurvStudio treats 1 as the event. A column named like "censored", or like the absence of
    the event ("progression_free", "event_free", "is_alive", "survived"), coded 0/1, Yes/No,
    Y/N, or True/False usually means 1/Yes/True = no event, so reading it directly would
    invert the whole analysis. Columns whose values name the outcome themselves
    ("Dead"/"Alive") are left to the regular event checks.
    """
    name = str(event_column)
    tokens = set(_column_name_tokens(name))
    censoring = bool(tokens & _CENSORING_NAME_TOKENS) and not tokens & _EVENT_NAME_OVERRIDE_TOKENS
    absence = None if censoring else _absence_of_event_name(name)
    if not censoring and absence is None:
        return None
    if not _has_only_generic_binary_codes(df[event_column]):
        return None
    if censoring:
        return (
            f'"{event_column}" looks like a censoring indicator (usually 1, Yes, or True = censored), but SurvStudio '
            "needs an event indicator (1 = event). Add a recoded column such as event = 1 - "
            f"{event_column} (Yes -> 0, No -> 1) and select that instead."
        )
    state = "being free of the event" if absence == "free" else "being alive"
    return (
        f'"{event_column}" looks like it records {state} (usually 1, Yes, or True = no event), but SurvStudio '
        "needs an event indicator (1 = event). Choose the event indicator column instead, or add a recoded column "
        f"such as event = 1 - {event_column} (Yes -> 0, No -> 1) and select that."
    )


def reject_censoring_indicator_event_column(df: pd.DataFrame, event_column: str) -> None:
    """Refuse a generically coded column whose name says it flags censoring or the absence of the event."""
    message = _no_event_indicator_message(df, event_column)
    if message is not None:
        raise ValueError(message)


def _validate_event_column_choice(df: pd.DataFrame, event_column: str) -> None:
    _require_dataframe_columns(df, [event_column])
    reject_censoring_indicator_event_column(df, event_column)
    series = df[event_column]
    if _looks_like_event_outcome_column(event_column, series):
        return

    def _likely_event_examples() -> str:
        # Scanning every column is only needed for the error message.
        likely_event_columns = [
            column
            for column in df.columns
            if column != event_column and _looks_like_event_outcome_column(column, df[column])
        ]
        return ", ".join(str(column) for column in likely_event_columns[:3])

    if not looks_binary(series):
        if _is_event_like_column_name(event_column):
            return
        raise ValueError(
            f'"{event_column}" is not a binary event column. '
            "Choose a 0/1-style event column or recode it before survival analysis."
        )

    # The baseline veto applies unless the name also names the event and the values decode
    # as event coding ("treatment_failure" coded 0/1 is an event column).
    if _looks_like_baseline_status_column(event_column) and not (
        _name_has_strong_event_word(event_column) and _values_decode_as_event_coding(series)
    ):
        examples = _likely_event_examples()
        hint = f" Choose one of the likely event columns instead: {examples}." if examples else ""
        raise ValueError(
            f'"{event_column}" looks more like a baseline characteristic than a survival event column.{hint}'
        )

    if _has_recognizable_event_coding(series):
        return

    if _is_event_like_column_name(event_column):
        return

    examples = _likely_event_examples()
    if examples:
        raise ValueError(
            f'"{event_column}" does not look like a survival event column. '
            f"Choose one of the likely event columns instead: {examples}."
        )
    raise ValueError(
        f'"{event_column}" does not look like a survival event column. '
        "Use a true event indicator or recode the dataset before survival analysis."
    )


def _coerce_survival_time_values(series: pd.Series, time_column: str) -> pd.Series:
    """Follow-up times as floats; text that is not a number is an error, never "missing".

    Numbers written with thousands separators ("1,234", as Excel exports formatted numbers)
    are parsed, and common missing-value markers ("NA", "[Not Available]") are missing.
    Anything else (for example "12 months" or a decimal comma "12,5") is refused with
    examples, instead of silently dropping those patients as rows with missing time.
    """
    if is_bool_dtype(series) or is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.Series(numeric.to_numpy(dtype=float, na_value=np.nan), index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    values = pd.Series(numeric.to_numpy(dtype=float, na_value=np.nan), index=series.index)
    failed = (series.notna() & values.isna()).to_numpy(dtype=bool)
    if not failed.any():
        return values
    text = series[failed].astype(str).str.strip()
    positions = np.flatnonzero(failed)
    stripped_numbers = pd.to_numeric(text, errors="coerce")
    grouped = text.str.fullmatch(_COMMA_GROUPED_NUMBER_PATTERN).to_numpy(dtype=bool, na_value=False)
    grouped_numbers = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    recovered = np.where(grouped, grouped_numbers.to_numpy(dtype=float, na_value=np.nan), stripped_numbers.to_numpy(dtype=float, na_value=np.nan))
    missing_like = text.str.lower().isin(_MISSING_VALUE_TOKENS).to_numpy(dtype=bool)
    unparsed = np.isnan(recovered) & ~missing_like
    if unparsed.any():
        bad_values = list(dict.fromkeys(text.to_numpy()[unparsed].tolist()))
        examples = ", ".join(f'"{value}"' for value in bad_values[:5])
        raise ValueError(
            f'"{time_column}" has {int(unparsed.sum())} value(s) that are not numbers, such as {examples}. '
            "Recode them as numbers (leave a cell blank when the time is missing) before survival analysis; "
            "they are not treated as missing values."
        )
    parsed_values = values.to_numpy(dtype=float, copy=True)
    parsed_values[positions] = recovered
    return pd.Series(parsed_values, index=series.index)


# Missing follow-up times that depend on the outcome. Refused: at least half of one outcome's rows
# lack a time and that share is at least 4 times the other outcome's. Cautioned: one share is at
# least 20% and at least twice the other.
_OUTCOME_MISSING_TIME_REFUSE_SHARE = 0.5
_OUTCOME_MISSING_TIME_REFUSE_RATIO = 4.0
_OUTCOME_MISSING_TIME_CAUTION_SHARE = 0.2
_OUTCOME_MISSING_TIME_CAUTION_RATIO = 2.0


def _reject_outcome_dependent_missing_time(frame: pd.DataFrame, time_column: str, event_column: str) -> str | None:
    """Refuse a time column that is missing mostly for one outcome; return a caution for a milder imbalance.

    GDC/TCGA exports carry days_to_death only for patients who died; selecting it as the time
    column would keep only deaths and drive the survival curve to zero. The mirror case
    (a last-follow-up column that is empty for the deaths) drops the events instead. The share
    of rows without a time is compared between censored rows and rows with an event, so a few
    missing event times cannot hide that most censored rows lack one.
    """
    event = frame[event_column]
    valid_event = event.notna().to_numpy(dtype=bool)
    missing_time = frame[time_column].isna().to_numpy(dtype=bool) & valid_event
    n_missing = int(missing_time.sum())
    if n_missing < 5:
        return None
    event_values = event.to_numpy(dtype=float, na_value=np.nan)
    censored = valid_event & (event_values == 0.0)
    events = valid_event & (event_values == 1.0)
    n_censored, n_events = int(censored.sum()), int(events.sum())
    missing_censored = int((missing_time & censored).sum())
    missing_events = int((missing_time & events).sum())
    censored_share = missing_censored / n_censored if n_censored else 0.0
    event_share = missing_events / n_events if n_events else 0.0
    if (
        censored_share >= _OUTCOME_MISSING_TIME_REFUSE_SHARE
        and censored_share >= _OUTCOME_MISSING_TIME_REFUSE_RATIO * event_share
    ):
        raise ValueError(
            f'"{time_column}" is missing for {missing_censored} of {n_censored} censored rows but for only '
            f"{missing_events} of {n_events} rows with an event, so the analysis would keep mostly events and "
            "underestimate survival. Censored patients need their last follow-up time: build one follow-up time "
            "column first (for example days_to_death for patients who died and days_to_last_follow_up for everyone "
            "else) and select that column."
        )
    if event_share >= _OUTCOME_MISSING_TIME_REFUSE_SHARE and event_share >= _OUTCOME_MISSING_TIME_REFUSE_RATIO * censored_share:
        raise ValueError(
            f'"{time_column}" is missing for {missing_events} of {n_events} rows with an event but for only '
            f"{missing_censored} of {n_censored} censored rows, so the analysis would drop most events and "
            "overestimate survival. Build one follow-up time column first (the event time for patients with the "
            "event and the last follow-up time for everyone else) and select that column."
        )
    higher, lower = max(censored_share, event_share), min(censored_share, event_share)
    if higher >= _OUTCOME_MISSING_TIME_CAUTION_SHARE and higher >= _OUTCOME_MISSING_TIME_CAUTION_RATIO * lower:
        direction = "underestimated" if censored_share > event_share else "overestimated"
        return (
            f'"{time_column}" is missing for {missing_censored} of {n_censored} censored rows and for '
            f"{missing_events} of {n_events} rows with an event. Rows without a time are left out, so survival may "
            f"be {direction}; check that every patient has a follow-up time (the event time or the last follow-up)."
        )
    return None


def _cohort_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    event_positive_value: Any = None,
    extra_columns: Sequence[str] | None = None,
    drop_missing_extra_columns: bool = True,
) -> pd.DataFrame:
    if time_column == event_column:
        raise ValueError("The survival time column and event column must be different.")
    _validate_endpoint_family_pair(time_column, event_column)
    extra_columns = list(dict.fromkeys(extra_columns or []))
    for column in extra_columns:
        if column in (time_column, event_column):
            role = "survival time" if column == time_column else "event"
            raise ValueError(
                f'"{column}" is the {role} column, so it cannot also be used as a grouping variable, '
                "covariate, stratum, or candidate feature."
            )
    required_columns = [time_column, event_column, *extra_columns]
    _require_dataframe_columns(df, required_columns)
    time_column_note = _validate_time_column_choice(df, time_column)
    _validate_event_column_choice(df, event_column)
    frame = df[required_columns].copy()
    frame[time_column] = _coerce_survival_time_values(frame[time_column], time_column)
    if not bool(frame[time_column].notna().any()):
        raise ValueError(f'"{time_column}" has no usable values: every follow-up time is missing.')
    frame[event_column] = coerce_event(frame[event_column], event_positive_value=event_positive_value)
    missing_time_note = _reject_outcome_dependent_missing_time(frame, time_column, event_column)
    time_column_note = " ".join(note for note in (time_column_note, missing_time_note) if note) or None
    raw_censored_rows = int((frame[event_column] == 0).sum())
    for column in extra_columns:
        if not is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("string")
    drop_subset = required_columns if drop_missing_extra_columns else [time_column, event_column]
    time_values = frame[time_column].to_numpy(dtype=float)
    outcome_inf_mask = np.isinf(time_values)
    extra_inf_mask = np.zeros(len(frame), dtype=bool)
    for column in extra_columns:
        series = frame[column]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            extra_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            extra_inf_mask |= np.isinf(extra_values)
    inf_row_mask = outcome_inf_mask | extra_inf_mask if drop_missing_extra_columns else outcome_inf_mask
    # Rows with a usable outcome whose selected inputs hold +/-Inf (for example log(0) of an
    # expression value); Cox reports them apart from rows dropped for the outcome itself.
    outcome_valid = np.isfinite(time_values) & (time_values >= 0) & frame[event_column].notna().to_numpy(dtype=bool)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # +/-Inf values were just coerced to missing, so rows_with_infinite_values is a
    # subset of dropped_missing_rows (summaries report it as "including ...").
    missing_row_mask = frame[drop_subset].isna().any(axis=1).to_numpy(dtype=bool)
    if drop_missing_extra_columns:
        frame = frame.dropna()
    else:
        frame = frame.dropna(subset=[time_column, event_column])
    if frame.empty:
        raise ValueError("No analyzable rows remain after removing missing values.")
    positive_mask = _ensure_positive_times(frame[time_column])
    frame = frame.loc[positive_mask].copy()
    source_row_index = frame.index.copy()
    frame = frame.reset_index(drop=True)
    frame.attrs["rows_with_infinite_values"] = int(inf_row_mask.sum())
    frame.attrs["rows_with_infinite_outcome"] = int(outcome_inf_mask.sum())
    frame.attrs["rows_with_infinite_extra_values"] = int((extra_inf_mask & outcome_valid).sum())
    frame.attrs["dropped_missing_rows"] = int(missing_row_mask.sum())
    frame.attrs["dropped_nonpositive_time_rows"] = int((~positive_mask).sum())
    frame.attrs["source_row_index"] = source_row_index.tolist()
    frame.attrs["row_mask_hash"] = _row_mask_hash(source_row_index)
    if time_column_note:
        frame.attrs["time_column_note"] = time_column_note
    if frame.empty:
        raise ValueError("No analyzable rows remain after removing missing values.")
    if frame[event_column].sum() == 0:
        raise ValueError("No events were found after preprocessing the event column.")
    if raw_censored_rows >= 3 and int((frame[event_column] == 0).sum()) == 0:
        raise ValueError(
            f"All {raw_censored_rows} censored rows were removed (missing or negative {time_column} values, or "
            "missing values in the selected columns), so only events would remain and the survival estimates would "
            f"be biased toward zero. Check how {time_column} was built: censored patients need their last follow-up time."
        )
    return frame


def _row_mask_hash(row_index: Sequence[Any] | pd.Index) -> str:
    index = row_index if isinstance(row_index, pd.Index) else pd.Index(list(row_index))
    hashed = pd.util.hash_pandas_object(index, index=False, categorize=True).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray([len(index)], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(hashed).tobytes())
    return digest.hexdigest()[:16]


def _frame_source_row_index(frame: pd.DataFrame) -> pd.Index:
    source_rows = frame.attrs.get("source_row_index")
    if source_rows is None:
        return pd.Index(frame.index.tolist())
    return pd.Index(list(source_rows))


def _ordered_unique_level_strings(series: pd.Series, column_name: str | None = None) -> list[str]:
    non_missing = series.dropna()
    if non_missing.empty:
        return []
    level_strings = [str(value) for value in non_missing.unique().tolist()]
    numeric_values = pd.to_numeric(pd.Series(level_strings, dtype="string"), errors="coerce")
    if numeric_values.notna().all():
        # Order numerically but return the labels exactly as they appear in the
        # data, so callers can match rows with ``series == label``.
        ordered = sorted(zip(numeric_values.astype(float).tolist(), level_strings), key=lambda item: (item[0], item[1]))
        return list(dict.fromkeys(label for _, label in ordered))
    return _ordered_reference_categories(level_strings, column_name or str(series.name or ""))


def _canonical_level_strings(series: pd.Series) -> pd.Series:
    """String labels for grouping: booleans as True/False, integer-valued numbers
    without a trailing ".0" (so a float column with missing values groups as
    "1", "2"), everything else via pandas' string dtype."""
    if is_bool_dtype(series):
        return series.astype("string")
    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric.astype(float))] if numeric.notna().any() else numeric.dropna()
        # Absolute tolerance only: np.isclose's default relative tolerance (1e-5 * |x|)
        # treats 100000.5 as integer-like and would merge distinct large values.
        if (
            not finite.empty
            and bool(np.all(np.isclose(finite.astype(float), np.round(finite.astype(float)), rtol=0.0, atol=1e-9)))
            and float(np.abs(finite.astype(float)).max()) < 1e15
        ):
            return numeric.where(np.isfinite(numeric.astype(float))).round().astype("Int64").astype("string")
        return numeric.astype("string")
    return series.astype("string")


def _sorted_group_labels(series: pd.Series, column_name: str | None = None) -> list[str]:
    return _ordered_unique_level_strings(series.astype("string"), column_name)


def _ordered_level_strings(series: pd.Series, column_name: str | None = None) -> list[str]:
    return _ordered_unique_level_strings(series, column_name)


def _is_binary_numeric_series(series: pd.Series) -> bool:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_values.empty:
        return False
    return int(numeric_values.nunique()) == 2


def _pointwise_km_ci(survival: np.ndarray, se: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    survival = np.asarray(survival, dtype=float)
    se = np.asarray(se, dtype=float)
    z_value = float(ndtri(1 - alpha / 2))
    lower = survival.copy()
    upper = survival.copy()

    finite_mask = np.isfinite(survival)
    lower[~finite_mask] = np.nan
    upper[~finite_mask] = np.nan

    zero_mask = finite_mask & (survival <= 0.0)
    lower[zero_mask] = 0.0
    upper[zero_mask] = 0.0
    one_mask = finite_mask & (survival >= 1.0 - 1e-12)
    lower[one_mask] = np.clip(survival[one_mask], 0.0, 1.0)
    upper[one_mask] = np.clip(survival[one_mask], 0.0, 1.0)

    log_s = np.full_like(survival, np.nan, dtype=float)
    candidate_mask = finite_mask & ~zero_mask & ~one_mask
    log_s[candidate_mask] = np.log(survival[candidate_mask])
    denominator = survival * log_s
    valid_mask = (
        candidate_mask
        & (survival < 1 - 1e-12)
        & np.isfinite(se)
        & (se > 0)
        & np.isfinite(log_s)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )

    transformed = np.full_like(survival, np.nan, dtype=float)
    transformed[valid_mask] = np.log(-log_s[valid_mask])
    transformed_se = np.full_like(survival, np.nan, dtype=float)
    transformed_se[valid_mask] = np.abs(se[valid_mask] / denominator[valid_mask])
    valid_mask &= np.isfinite(transformed_se)

    low = np.exp(-np.exp(transformed[valid_mask] + z_value * transformed_se[valid_mask]))
    high = np.exp(-np.exp(transformed[valid_mask] - z_value * transformed_se[valid_mask]))
    lower[valid_mask] = np.clip(low, 0.0, 1.0)
    upper[valid_mask] = np.clip(high, 0.0, 1.0)
    return lower, upper


def _step_values(event_times: np.ndarray, survival: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(event_times, query_times, side="right") - 1
    output = np.ones_like(query_times, dtype=float)
    valid = indices >= 0
    output[valid] = survival[indices[valid]]
    return output


def _restricted_mean_survival_time_delta_stats(
    event_times: np.ndarray,
    survival_after_event: np.ndarray,
    n_risk: np.ndarray,
    n_events: np.ndarray,
    horizon: float,
    *,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    """Estimate RMST uncertainty using a Greenwood/delta-method approximation.

    Parameters
    ----------
    event_times, survival_after_event, n_risk, n_events
        Per-event Kaplan-Meier quantities from ``SurvfuncRight``.
    horizon
        Truncation time for RMST.
    alpha
        Two-sided error rate for the confidence interval.
    """
    event_times_arr = np.asarray(event_times, dtype=float).reshape(-1)
    survival_arr = np.asarray(survival_after_event, dtype=float).reshape(-1)
    n_risk_arr = np.asarray(n_risk, dtype=float).reshape(-1)
    n_events_arr = np.asarray(n_events, dtype=float).reshape(-1)
    if not (
        event_times_arr.size == survival_arr.size == n_risk_arr.size == n_events_arr.size
    ):
        raise ValueError("RMST delta-method inputs must have matching lengths.")

    tau = float(max(horizon, 0.0))
    if tau <= 0.0:
        return {
            "rmst": 0.0,
            "variance": 0.0,
            "se": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    if event_times_arr.size == 0:
        return {
            "rmst": tau,
            "variance": 0.0,
            "se": 0.0,
            "ci_lower": tau,
            "ci_upper": tau,
        }

    within_horizon = event_times_arr <= tau + 1e-12
    event_times_arr = event_times_arr[within_horizon]
    survival_arr = survival_arr[within_horizon]
    n_risk_arr = n_risk_arr[within_horizon]
    n_events_arr = n_events_arr[within_horizon]

    if event_times_arr.size == 0:
        return {
            "rmst": tau,
            "variance": 0.0,
            "se": 0.0,
            "ci_lower": tau,
            "ci_upper": tau,
        }

    clipped_times = np.clip(event_times_arr, 0.0, tau)
    interval_grid = np.concatenate(([0.0], clipped_times, [tau]))
    interval_widths = np.maximum(np.diff(interval_grid), 0.0)
    interval_survival = np.concatenate(([1.0], survival_arr))
    rmst = float(np.dot(interval_survival, interval_widths))

    # Delta-method variance:
    # Var(RMST) ~= sum_k A_k^2 * d_k / (n_k (n_k - d_k)),
    # where A_k is the post-event tail area carried by the KM estimate from
    # event k onward and d_k / (n_k (n_k - d_k)) is the Greenwood increment.
    post_event_area_terms = interval_widths[1:] * survival_arr
    tail_areas = np.cumsum(post_event_area_terms[::-1])[::-1]
    greenwood_increments = np.zeros_like(tail_areas, dtype=float)
    # When the risk set is exhausted at an event time (n_k == d_k), the KM
    # estimate drops to zero and the tail area carried from that time onward is
    # zero, so the product A_k^2 * d_k / (n_k (n_k - d_k)) is 0 * inf. Following
    # survRM2 (Uno et al.), that term contributes zero instead of voiding the
    # whole variance estimate.
    valid = (
        np.isfinite(n_risk_arr)
        & np.isfinite(n_events_arr)
        & (n_risk_arr > 0.0)
        & (n_events_arr > 0.0)
        & (n_risk_arr > n_events_arr)
    )
    greenwood_increments[valid] = (
        n_events_arr[valid] / (n_risk_arr[valid] * (n_risk_arr[valid] - n_events_arr[valid]))
    )
    variance = float(np.sum((tail_areas ** 2) * greenwood_increments))
    variance = max(variance, 0.0)
    se = math.sqrt(variance)
    z_value = float(ndtri(1.0 - alpha / 2.0))
    ci_lower = max(rmst - z_value * se, 0.0)
    ci_upper = min(rmst + z_value * se, tau)
    return {
        "rmst": float(rmst),
        "variance": float(variance),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def _diagnostic_lowess_trend(y_sorted: np.ndarray, x_sorted: np.ndarray) -> np.ndarray:
    """LOWESS trend line for a residual panel whose x values are sorted ascending.

    With heavily tied x (tied event times on the log scale, or a covariate with few
    values) a local window can hold a single distinct x, where LOWESS divides by zero;
    those points fall back to the mean residual at the same x.
    """
    if x_sorted.shape[0] < 4:
        return y_sorted
    frac = min(0.8, max(0.35, 6.0 / float(x_sorted.shape[0])))
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = np.asarray(lowess(y_sorted, x_sorted, frac=frac, it=0, return_sorted=False), dtype=float)
    undefined = ~np.isfinite(trend)
    if undefined.any():
        _, inverse = np.unique(x_sorted, return_inverse=True)
        means_at_x = np.bincount(inverse, weights=y_sorted) / np.bincount(inverse)
        trend[undefined] = means_at_x[inverse[undefined]]
    return trend


def _is_martingale_screen_candidate(series: pd.Series) -> bool:
    """A continuous covariate: numeric, not boolean, and with more than two distinct values.

    The functional-form screen asks whether the log hazard is linear in the covariate, which
    is moot for a 0/1 or other two-valued variable.
    """
    if not pd.api.types.is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    return int(series.dropna().nunique()) > 2


def _cox_martingale_plot_data(
    frame: pd.DataFrame,
    martingale_residuals: np.ndarray,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> list[dict[str, Any]]:
    categorical_set = {str(value) for value in categorical_covariates}
    residual_array = np.asarray(martingale_residuals, dtype=float).reshape(-1)
    if residual_array.size == 0:
        return []
    if residual_array.size != len(frame):
        warnings.warn(
            "Martingale residual diagnostics were skipped because the residual vector length did not match the analyzable cohort.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    panels: list[dict[str, Any]] = []
    for covariate in covariates:
        if covariate in categorical_set or covariate not in frame.columns:
            continue
        series = frame[covariate]
        if not _is_martingale_screen_candidate(series):
            continue
        x_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(residual_array)
        if int(valid.sum()) < 4:
            continue
        x_valid = x_values[valid]
        y_valid = residual_array[valid]
        if np.allclose(np.nanstd(x_valid), 0.0):
            continue
        order = np.argsort(x_valid, kind="mergesort")
        x_sorted = x_valid[order]
        y_sorted = y_valid[order]
        trend_y = _diagnostic_lowess_trend(y_sorted, x_sorted)
        panels.append(
            {
                "term": covariate,
                "value": [_safe_float(value) for value in x_sorted.tolist()],
                "residual": [_safe_float(value) for value in y_sorted.tolist()],
                "trend_value": [_safe_float(value) for value in x_sorted.tolist()],
                "trend_residual": [_safe_float(value) for value in np.asarray(trend_y, dtype=float).tolist()],
            }
        )
    return panels


def _safe_float(value: Any) -> float | None:
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(float_value):
        return None
    return float_value


def _safe_exp_or_none(value: Any) -> float | None:
    """Exponentiate only when safely representable; otherwise return None."""
    try:
        exponent = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(exponent):
        return None
    if exponent >= _MAX_EXP_INPUT or exponent <= _MIN_EXP_INPUT:
        return None
    return math.exp(exponent)


def _hazard_ratio_effect_size(value: Any) -> float:
    hr = _safe_float(value)
    if hr is None or hr <= 0.0:
        return 0.0
    return abs(math.log(max(hr, 1e-12)))


def _cox_estimated_parameter_count(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> int:
    estimated_parameters = 0
    categorical_set = {str(column) for column in categorical_covariates}
    for column in covariates:
        if column in categorical_set:
            observed_levels = int(frame[column].dropna().nunique()) if column in frame.columns else 0
            estimated_parameters += max(observed_levels - 1, 0)
        else:
            estimated_parameters += 1
    return int(estimated_parameters)


def _summarize_labels(labels: Sequence[str], max_items: int = 3) -> str:
    cleaned = [str(label) for label in labels if label]
    if not cleaned:
        return "none"
    if len(cleaned) <= max_items:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:max_items])} +{len(cleaned) - max_items} more"


def _format_cox_strata_label(columns: Sequence[str], values: Sequence[Any]) -> str:
    return " | ".join(
        f"{column}={value}"
        for column, value in zip(columns, values, strict=True)
    )


def _build_cox_strata_payload(
    frame: pd.DataFrame,
    strata_columns: Sequence[str],
) -> dict[str, Any]:
    normalized = [str(column) for column in strata_columns if str(column)]
    if not normalized:
        return {
            "codes": None,
            "row_labels": None,
            "display_labels": [],
            "n_strata": None,
        }

    strata_frame = frame[normalized].copy()
    for column in normalized:
        strata_frame[column] = strata_frame[column].astype("string")

    tuple_rows = [
        tuple(str(value) for value in row)
        for row in strata_frame.itertuples(index=False, name=None)
    ]
    tuple_index = pd.Index(tuple_rows, tupleize_cols=False)
    codes, unique_rows = pd.factorize(tuple_index, sort=False)
    display_labels = [
        _format_cox_strata_label(normalized, unique_row if isinstance(unique_row, tuple) else (unique_row,))
        for unique_row in unique_rows.tolist()
    ]
    row_labels = np.asarray([display_labels[int(code)] for code in codes], dtype=object)
    return {
        "codes": np.asarray(codes, dtype=np.int64),
        "row_labels": row_labels,
        "display_labels": display_labels,
        "n_strata": int(len(display_labels)),
    }


def _cox_strata_snapshot(
    frame: pd.DataFrame,
    event_column: str,
    strata_columns: Sequence[str],
    strata_row_labels: np.ndarray | None,
) -> dict[str, Any]:
    normalized = [str(column) for column in strata_columns if str(column)]
    if not normalized or strata_row_labels is None:
        return {
            "n_strata": None,
            "zero_event_strata_count": 0,
            "zero_event_strata_examples": [],
            "sparse_event_strata_count": 0,
            "sparse_event_strata_examples": [],
            "high_cardinality_columns": [],
            "high_cardinality_numeric_columns": [],
            "stability_warnings": [],
        }

    n_rows = int(frame.shape[0])
    high_cardinality_columns: list[str] = []
    high_cardinality_numeric_columns: list[str] = []
    stability_warnings: list[str] = []
    numeric_level_cutoff = max(10, min(30, int(math.ceil(max(n_rows, 1) * 0.10))))
    generic_level_cutoff = max(20, min(60, int(math.ceil(max(n_rows, 1) * 0.20))))

    for column in normalized:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        unique_count = int(series.nunique())
        if is_numeric_dtype(series) and unique_count > numeric_level_cutoff:
            high_cardinality_numeric_columns.append(f"{column} ({unique_count} unique values)")
        elif unique_count > generic_level_cutoff:
            high_cardinality_columns.append(f"{column} ({unique_count} observed levels)")

    if high_cardinality_numeric_columns:
        stability_warnings.append(
            "Selected strata include high-cardinality numeric column(s): "
            f"{_summarize_labels(high_cardinality_numeric_columns, max_items=3)}. "
            "Recode continuous variables into a small number of clinically meaningful levels before stratifying."
        )
    if high_cardinality_columns:
        stability_warnings.append(
            "Selected strata include many observed levels: "
            f"{_summarize_labels(high_cardinality_columns, max_items=3)}. "
            "Too many strata can make a stratified Cox model unstable."
        )

    grouped = pd.DataFrame(
        {
            "__stratum": np.asarray(strata_row_labels, dtype=object),
            "__event": frame[event_column].astype(int).to_numpy(),
        }
    ).groupby("__stratum", observed=False)["__event"].agg(["size", "sum"])

    zero_event_examples = grouped[grouped["sum"] == 0].index.astype(str).tolist()
    sparse_event_examples = grouped[(grouped["sum"] > 0) & (grouped["sum"] <= 1)].index.astype(str).tolist()
    n_strata = int(grouped.shape[0])
    if zero_event_examples:
        stability_warnings.append(
            f"{len(zero_event_examples)} observed stratum/strata have zero events, including "
            f"{_summarize_labels(zero_event_examples, max_items=3)}. "
            "Collapse sparse strata before treating this stratified fit as stable."
        )
    if sparse_event_examples:
        stability_warnings.append(
            f"{len(sparse_event_examples)} observed stratum/strata have only one event, including "
            f"{_summarize_labels(sparse_event_examples, max_items=3)}. "
            "Term-specific hazard ratios can look unstable when strata are this sparse."
        )
    if n_strata > max(12, min(40, int(math.ceil(max(n_rows, 1) * 0.20)))):
        stability_warnings.append(
            f"The fitted stratified Cox specification created {n_strata} observed strata. "
            "Large numbers of strata reduce the information available within each baseline-hazard stratum."
        )

    return {
        "n_strata": n_strata,
        "zero_event_strata_count": int(len(zero_event_examples)),
        "zero_event_strata_examples": zero_event_examples,
        "sparse_event_strata_count": int(len(sparse_event_examples)),
        "sparse_event_strata_examples": sparse_event_examples,
        "high_cardinality_columns": high_cardinality_columns,
        "high_cardinality_numeric_columns": high_cardinality_numeric_columns,
        "stability_warnings": stability_warnings,
    }


def _cox_stability_snapshot(
    frame: pd.DataFrame,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
    strata_columns: Sequence[str] | None = None,
    strata_row_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    estimated_parameters = _cox_estimated_parameter_count(frame, covariates, categorical_covariates)
    event_count = int(frame[event_column].sum())
    events_per_parameter = (
        float(event_count / estimated_parameters)
        if estimated_parameters > 0
        else None
    )
    stability_warnings: list[str] = []
    risky_levels: list[dict[str, Any]] = []
    constant_design_columns: list[dict[str, str]] = []
    high_cardinality_categoricals: list[dict[str, Any]] = []
    categorical_set = {str(column) for column in categorical_covariates}
    for column in covariates:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        if str(column) in categorical_set:
            observed_levels = series.astype("string").dropna().nunique()
            if int(observed_levels) < 2:
                constant_design_columns.append({"column": str(column), "kind": "categorical"})
                stability_warnings.append(
                    f'{column} has only one observed level after missing-value filtering, so Cox PH cannot estimate a contrast for it.'
                )
            elif int(observed_levels) > _COX_MAX_CATEGORICAL_LEVELS:
                high_cardinality_categoricals.append({"column": str(column), "levels": int(observed_levels)})
                stability_warnings.append(
                    f"{column} has {int(observed_levels)} observed levels, which looks like an identifier or free text rather "
                    f"than a categorical covariate; Cox fitting accepts at most {_COX_MAX_CATEGORICAL_LEVELS} levels per categorical covariate."
                )
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.size and np.allclose(np.nanmax(finite), np.nanmin(finite), atol=1e-12, rtol=0.0):
            constant_design_columns.append({"column": str(column), "kind": "numeric"})
            stability_warnings.append(
                f"{column} is constant after missing-value filtering, so Cox PH cannot estimate a coefficient for it."
            )
    for column in categorical_covariates:
        if column not in frame.columns:
            continue
        # Observed levels only: a category with no rows left after missing-value filtering is
        # neither the fitted reference nor a level with one-sided outcomes.
        grouped = frame.groupby(column, dropna=True, observed=True)[event_column].agg(["size", "sum"])
        grouped = grouped[grouped["size"] > 0]
        categories = list(frame[column].cat.categories) if hasattr(frame[column], "cat") else []
        observed_categories = [category for category in categories if category in grouped.index]
        if observed_categories:
            reference_level = str(observed_categories[0])
            if reference_level in grouped.index:
                reference_row = grouped.loc[reference_level]
                reference_count = int(reference_row["size"])
                reference_events = int(reference_row["sum"])
                reference_censored = reference_count - reference_events
                if reference_count < 5:
                    risky_levels.append(
                        {
                            "column": str(column),
                            "level": reference_level,
                            "rows": reference_count,
                            "events": reference_events,
                            "censored": reference_censored,
                            "issue": "small_reference_level",
                        }
                    )
                    stability_warnings.append(
                        f'{column} uses "{reference_level}" as the Cox reference level, but only {reference_count} row'
                        f'{"s" if reference_count != 1 else ""} remain after missing-value filtering. Comparisons against this reference can look unstable.'
                    )
        one_sided_levels: list[str] = []
        for level, row in grouped.iterrows():
            total_count = int(row["size"])
            level_event_count = int(row["sum"])
            censored_count = total_count - level_event_count
            if level_event_count == 0 or censored_count == 0:
                risky_levels.append(
                    {
                        "column": str(column),
                        "level": str(level),
                        "rows": total_count,
                        "events": level_event_count,
                        "censored": censored_count,
                        "issue": "one_sided_outcome",
                    }
                )
                one_sided_levels.append(f"{level} ({total_count} rows)")
        if one_sided_levels:
            preview_levels = ", ".join(one_sided_levels[:3])
            stability_warnings.append(
                f'{column} has level(s) with only events or only censored rows after missing-value filtering: {preview_levels}'
                f'{" ..." if len(one_sided_levels) > 3 else ""}. This can produce non-finite Cox estimates.'
            )
    if events_per_parameter is not None:
        if events_per_parameter < 5:
            stability_warnings.append(
                f"Events per parameter is {events_per_parameter:.2f}, which is extremely low for Cox regression. Reduce covariates or add more events before treating the fit as stable."
            )
        elif events_per_parameter < 10:
            stability_warnings.append(
                f"Events per parameter is {events_per_parameter:.2f}, so coefficient estimates may still be unstable."
            )
    strata_snapshot = _cox_strata_snapshot(
        frame,
        event_column,
        strata_columns or [],
        strata_row_labels,
    )
    stability_warnings.extend(strata_snapshot["stability_warnings"])
    return {
        "events": event_count,
        "estimated_parameters": int(estimated_parameters),
        "events_per_parameter": events_per_parameter,
        "stability_warnings": stability_warnings,
        "risky_levels": risky_levels,
        "constant_design_columns": constant_design_columns,
        "high_cardinality_categoricals": high_cardinality_categoricals,
        "strata_snapshot": strata_snapshot,
    }


def _reject_oversized_cox_design(stability_snapshot: dict[str, Any]) -> None:
    """Refuse a Cox design that cannot be estimated before the design matrix is built.

    An identifier or free-text column used as a categorical covariate would expand into one
    dummy column per row (an n x n design that takes minutes and gigabytes to fit), and a
    model with at least as many coefficients as events cannot be estimated at all.
    """
    too_many_levels = list(stability_snapshot.get("high_cardinality_categoricals") or [])
    if too_many_levels:
        first = too_many_levels[0]
        raise ValueError(
            f'"{first["column"]}" has {int(first["levels"])} observed levels, which looks like an identifier or free text '
            f"rather than a categorical covariate. Cox regression accepts at most {_COX_MAX_CATEGORICAL_LEVELS} levels per "
            "categorical covariate: remove it or recode it into a few clinically meaningful groups."
        )
    parameters = int(stability_snapshot.get("estimated_parameters") or 0)
    events = int(stability_snapshot.get("events") or 0)
    if parameters >= events:
        raise ValueError(
            f"The Cox model would estimate {parameters} coefficient(s) from only {events} event(s). Remove covariates "
            "or collapse categorical levels so that there are fewer coefficients than events (ideally at least 10 "
            "events per coefficient)."
        )


def _cox_constant_design_message(stability_snapshot: dict[str, Any]) -> str | None:
    constant_columns = list(stability_snapshot.get("constant_design_columns") or [])
    if not constant_columns:
        return None
    labels = [
        f'{item["column"]} ({"single observed level" if item.get("kind") == "categorical" else "constant value"})'
        for item in constant_columns
        if item.get("column")
    ]
    if not labels:
        return None
    return (
        "Cox PH requires variation in every selected covariate after missing-value filtering. "
        f"Remove or recode {_summarize_labels(labels, max_items=4)} before fitting the model."
    )


def _cox_problem_signals(stability_snapshot: dict[str, Any]) -> str:
    """The stability signals of the analyzable cohort as one sentence, or "" when there are none."""
    details: list[str] = []
    risky_levels = list(stability_snapshot.get("risky_levels") or [])
    epv = _safe_float(stability_snapshot.get("events_per_parameter"))
    if epv is not None and epv < 10:
        details.append(f"EPV={epv:.2f}")

    small_reference_examples = [
        f'{item["column"]}="{item["level"]}" (n={int(item["rows"])})'
        for item in sorted(
            (level for level in risky_levels if level.get("issue") == "small_reference_level"),
            key=lambda level: (
                int(level.get("rows", 0)),
                str(level.get("column", "")),
                str(level.get("level", "")),
            ),
        )
    ]
    if small_reference_examples:
        details.append(
            f"sparse reference levels such as {_summarize_labels(small_reference_examples, max_items=3)}"
        )

    one_sided_examples = [
        f'{item["column"]}="{item["level"]}" (n={int(item["rows"])})'
        for item in sorted(
            (level for level in risky_levels if level.get("issue") == "one_sided_outcome"),
            key=lambda level: (
                int(level.get("rows", 0)),
                str(level.get("column", "")),
                str(level.get("level", "")),
            ),
        )
    ]
    if one_sided_examples:
        details.append(
            f'levels with only events or only censored rows such as {_summarize_labels(one_sided_examples, max_items=3)}'
        )

    return f" Problem signals in the analyzable cohort: {'; '.join(details)}." if details else ""


def _cox_nonfinite_estimate_message(stability_snapshot: dict[str, Any]) -> str:
    return (
        "Cox PH fit produced non-finite estimates. This usually means redundant covariates, sparse categories, "
        "or quasi-complete separation. Remove overlapping variables or collapse sparse levels."
        f"{_cox_problem_signals(stability_snapshot)}"
    )


def _cox_fit_failure_message(exc: Exception, stability_snapshot: dict[str, Any]) -> str:
    raw = str(exc).strip()
    lowered = raw.lower()
    if "converg" in lowered:
        return (
            "Cox PH fit did not converge cleanly. This usually means sparse strata, sparse categories, "
            "quasi-separation (a level or covariate whose rows all have, or all lack, the event early), or an "
            "over-specified model for the available events. "
            "Collapse sparse levels, reduce strata complexity, or simplify the covariate set."
            f"{_cox_problem_signals(stability_snapshot)}"
        )
    if "singular matrix" in lowered:
        return (
            "Cox PH fit failed because the design matrix is singular. This usually means redundant covariates, "
            "overlapping encodings of the same signal, or sparse categorical levels. "
            "Remove one of the overlapping variables or collapse sparse levels."
            f"{_cox_problem_signals(stability_snapshot)}"
        )
    return _cox_nonfinite_estimate_message(stability_snapshot)


# Newton-Raphson settings of R's coxph (coxph.control): the fit has converged when a full
# Newton step changes the partial log-likelihood by at most eps relative to its value. R allows
# 20 iterations, each step halving counting as one; a little more room is left here.
_COX_NEWTON_EPS = 1e-9
_COX_NEWTON_MAX_ITERATIONS = 50
# A Cholesky pivot of the unit-diagonal information matrix below this marks the design as
# singular (R's coxph.control toler.chol).
_COX_CHOLESKY_TOLERANCE = float(np.finfo(float).eps) ** 0.75
# R's check for coefficients that run to infinity after the log-likelihood has converged
# (toler.inf = sqrt(eps)), with the step and the coefficient measured per SD of the covariate
# as in the marker screen; a log hazard ratio above 10 per SD is taken to run away whatever the step.
_COX_TOLER_INF = math.sqrt(_COX_NEWTON_EPS)
_COX_MAX_LOG_HR_PER_SD = 10.0


def _phreg_score_information(model: PHReg, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Score vector and information matrix (minus the Hessian) of ``model`` at ``beta``; None when not finite."""
    score = np.asarray(model.score(beta), dtype=float).reshape(-1)
    information = -np.atleast_2d(np.asarray(model.hessian(beta), dtype=float))
    if not (np.all(np.isfinite(score)) and np.all(np.isfinite(information))):
        return None
    return score, information


def _phreg_information_factor(information: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Cholesky factor of the information matrix scaled to a unit diagonal, and the scale.

    Scaling each coefficient by the square root of its information (as R's coxph scales the
    covariates) keeps covariates on very different scales from spoiling the solve. None when
    the matrix is not positive definite or a pivot falls below R's singularity tolerance.
    """
    diagonal = np.diag(information)
    if diagonal.size == 0 or not np.all(diagonal > 0.0):
        return None
    scale = 1.0 / np.sqrt(diagonal)
    scaled = information * np.outer(scale, scale)
    try:
        factor = np.linalg.cholesky((scaled + scaled.T) / 2.0)
    except np.linalg.LinAlgError:
        return None
    if float(np.min(np.diag(factor))) ** 2 < _COX_CHOLESKY_TOLERANCE:
        return None
    return factor, scale


def _phreg_solve(factor: tuple[np.ndarray, np.ndarray], right_hand_side: np.ndarray) -> np.ndarray:
    """Solve ``information @ x = right_hand_side`` from the scaled Cholesky factor."""
    lower, scale = factor
    rhs = np.asarray(right_hand_side, dtype=float)
    scaled_rhs = scale[:, None] * rhs if rhs.ndim == 2 else scale * rhs
    solution = np.linalg.solve(lower.T, np.linalg.solve(lower, scaled_rhs))
    return scale[:, None] * solution if rhs.ndim == 2 else scale * solution


class _PHRegNewton(NamedTuple):
    beta: np.ndarray
    covariance: np.ndarray
    converged: bool
    iterations: int
    # "converged", "singular_information", "non_finite", "iteration_limit", or "separated".
    reason: str
    separated: np.ndarray


def _phreg_newton_raphson(model: PHReg) -> _PHRegNewton:
    """Maximum partial-likelihood fit of ``model`` by Newton-Raphson with step halving, as R's coxph.

    Starts at 0 and takes full Newton steps; whenever a step lowers the log-likelihood or makes it
    non-finite, the step is halved (each halving uses one iteration, as in R). The fit has
    converged when a full step changes the log-likelihood by at most ``_COX_NEWTON_EPS`` relative
    to its value. Ties and strata are handled by the model's own ``loglike``, ``score``, and
    ``hessian``. A singular or non-finite information matrix, the iteration limit, or a coefficient
    that runs to infinity (monotone likelihood, where the log-likelihood converges but the
    estimate does not exist) ends the fit as not converged.
    """
    n_params = int(model.exog.shape[1])
    beta = np.zeros(n_params, dtype=float)
    nan_covariance = np.full((n_params, n_params), np.nan)
    no_flags = np.zeros(n_params, dtype=bool)

    def _failed(reason: str, iterations: int, at: np.ndarray) -> _PHRegNewton:
        return _PHRegNewton(at, nan_covariance, False, iterations, reason, no_flags)

    if n_params == 0:
        # The null model has nothing to estimate.
        return _PHRegNewton(beta, np.zeros((0, 0)), True, 0, "converged", no_flags)
    loglik = float(model.loglike(beta))
    derivatives = _phreg_score_information(model, beta)
    if not math.isfinite(loglik) or derivatives is None:
        return _failed("non_finite", 0, beta)
    factor = _phreg_information_factor(derivatives[1])
    if factor is None:
        return _failed("singular_information", 0, beta)
    candidate = beta + _phreg_solve(factor, derivatives[0])
    halving = False
    converged = False
    iterations = 0
    for iterations in range(1, _COX_NEWTON_MAX_ITERATIONS + 1):
        if not np.all(np.isfinite(candidate)):
            return _failed("non_finite", iterations, beta)
        candidate_loglik = float(model.loglike(candidate))
        if (
            math.isfinite(candidate_loglik)
            and not halving
            and abs(candidate_loglik - loglik) <= _COX_NEWTON_EPS * abs(candidate_loglik)
        ):
            beta, loglik = candidate, candidate_loglik
            converged = True
            break
        if iterations == _COX_NEWTON_MAX_ITERATIONS:
            break
        if not math.isfinite(candidate_loglik) or candidate_loglik < loglik:
            # Half of the previous increment, as R's coxph.
            halving = True
            candidate = (candidate + beta) / 2.0
            continue
        halving = False
        beta, loglik = candidate, candidate_loglik
        derivatives = _phreg_score_information(model, beta)
        if derivatives is None:
            return _failed("non_finite", iterations, beta)
        factor = _phreg_information_factor(derivatives[1])
        if factor is None:
            return _failed("singular_information", iterations, beta)
        candidate = beta + _phreg_solve(factor, derivatives[0])
    if not converged:
        return _failed("iteration_limit", iterations, beta)

    derivatives = _phreg_score_information(model, beta)
    factor = None if derivatives is None else _phreg_information_factor(derivatives[1])
    if derivatives is None or factor is None:
        return _failed("singular_information", iterations, beta)
    covariance = _phreg_solve(factor, np.eye(n_params))
    covariance = (covariance + covariance.T) / 2.0
    pending = _phreg_solve(factor, derivatives[0])
    spread = np.std(np.asarray(model.exog, dtype=float), axis=0)
    with np.errstate(invalid="ignore", over="ignore"):
        size = np.abs(beta) * spread
        pending_size = np.abs(pending) * spread
        separated = (size > _COX_MAX_LOG_HR_PER_SD) | (pending_size > _COX_TOLER_INF * np.maximum(size, 1.0))
    separated &= spread > 0.0
    if separated.any():
        return _PHRegNewton(beta, covariance, False, iterations, "separated", separated)
    return _PHRegNewton(beta, covariance, True, iterations, "converged", no_flags)


def fit_phreg(model: PHReg) -> tuple[PHRegResults, bool]:
    """Fit ``model`` by maximum partial likelihood and return ``(results, converged)``.

    statsmodels' own ``PHReg.fit`` takes plain Newton steps from 0 without step halving, so a
    strong effect of a rare binary covariate can overshoot and diverge to NaN coefficients that
    were still flagged as converged. The fit here is a safeguarded Newton-Raphson that follows
    R's coxph (see ``_phreg_newton_raphson``) on the model's own log-likelihood, score, and
    Hessian, so Efron or Breslow ties and strata are handled exactly as statsmodels computes
    them. ``converged`` is False for a singular information matrix, non-finite values, the
    iteration limit, or a coefficient that runs to infinity; the results then hold the last
    accepted coefficients. ``results.mle_retvals`` records the reason. The results carry the
    inverse information as covariance, as ``PHReg.fit`` does. Objects that only expose a
    statsmodels-style ``fit()`` are fitted as-is and treated as converged.
    """
    if not isinstance(model, PHReg):
        return model.fit(disp=False), True
    model.groups = None
    fit = _phreg_newton_raphson(model)
    results = PHRegResults(model, fit.beta, fit.covariance)
    results.mle_retvals = {
        "converged": fit.converged,
        "iterations": fit.iterations,
        "reason": fit.reason,
        "separated": fit.separated.tolist(),
    }
    return results, fit.converged


def _fit_cox_model(model: PHReg, stability_snapshot: dict[str, Any]):
    try:
        results, converged = fit_phreg(model)
    except MemoryError:
        raise
    except Exception as exc:
        if must_propagate(exc):
            raise
        raise ValueError(_cox_fit_failure_message(exc, stability_snapshot)) from exc
    if not converged:
        retvals = getattr(results, "mle_retvals", None)
        reason = retvals.get("reason") if isinstance(retvals, dict) else None
        cause: Exception = (
            np.linalg.LinAlgError("Singular matrix")
            if reason == "singular_information"
            else RuntimeError("Cox PH fit did not converge cleanly.")
        )
        raise ValueError(_cox_fit_failure_message(cause, stability_snapshot))
    return results


class _EfronTieGroups(NamedTuple):
    """One stratum's rows sorted by time, with the Efron bookkeeping of its tied events."""

    order: np.ndarray
    sorted_time: np.ndarray
    risk: np.ndarray
    dead_pos: np.ndarray
    tie_starts: np.ndarray
    tie_index: np.ndarray
    deaths: np.ndarray
    fraction: np.ndarray
    denominator: np.ndarray


def _efron_tie_groups(
    rows: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    linear_predictor: np.ndarray,
) -> _EfronTieGroups | None:
    """Sort one stratum and compute the Efron risk-set denominators of its events.

    For d events tied at time t, event k = 0..d-1 of the tie uses the denominator
    S0 - k/d * S0_tied, where S0 sums exp(eta) over the risk set (time >= t) and
    S0_tied over the tied events. Returns None for a stratum without events.
    """
    order = rows[np.argsort(time[rows], kind="mergesort")]
    dead_pos = np.flatnonzero(event[order])
    if dead_pos.size == 0:
        return None
    sorted_time = time[order]
    lp = linear_predictor[order]
    # Shifting eta by a constant cancels in every quantity built from these sums.
    risk = np.exp(lp - lp.max())
    s0_from = np.cumsum(risk[::-1])[::-1]
    # Tied events share the first sorted position of their time as the risk-set start.
    tie_start = np.searchsorted(sorted_time, sorted_time[dead_pos], side="left")
    tie_starts, tie_index, deaths = np.unique(tie_start, return_inverse=True, return_counts=True)
    tied_s0 = np.bincount(tie_index, weights=risk[dead_pos])
    step = np.arange(dead_pos.size) - np.searchsorted(tie_index, tie_index, side="left")
    fraction = step / deaths[tie_index]
    denominator = s0_from[tie_starts][tie_index] - fraction * tied_s0[tie_index]
    return _EfronTieGroups(
        order=order,
        sorted_time=sorted_time,
        risk=risk,
        dead_pos=dead_pos,
        tie_starts=tie_starts,
        tie_index=tie_index,
        deaths=deaths,
        fraction=fraction,
        denominator=denominator,
    )


def _efron_martingale_residuals(
    exog: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    params: np.ndarray,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Martingale residuals of an Efron-tie Cox fit, computed as R's ``residuals.coxph`` does.

    statsmodels' ``PHRegResults.martingale_residuals`` evaluates the cumulative hazard
    just before each subject's time, which leaves out the jump at the subject's own
    event and biases event residuals upward. Here the cumulative hazard includes it:
    each event time adds sum_k 1 / (S0 - k/d * S0_tied) for everyone at risk, and each
    of the d tied events itself receives the Efron share sum_k (1 - k/d) / (...).
    """
    exog = np.asarray(exog, dtype=float)
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(bool)
    linear_predictor = exog @ np.asarray(params, dtype=float).reshape(-1)
    strata_codes = np.zeros(time.shape[0], dtype=np.int64) if strata is None else np.asarray(strata).reshape(-1)
    # A stratum without events has zero cumulative hazard, so its residuals are 0.
    residuals = np.zeros(time.shape[0], dtype=float)
    for code in np.unique(strata_codes):
        groups = _efron_tie_groups(np.flatnonzero(strata_codes == code), time, event, linear_predictor)
        if groups is None:
            continue
        hazard = np.bincount(groups.tie_index, weights=1.0 / groups.denominator)
        event_hazard = np.bincount(groups.tie_index, weights=(1.0 - groups.fraction) / groups.denominator)
        cumulative = np.cumsum(hazard)
        event_times = groups.sorted_time[groups.tie_starts]
        n_passed = np.searchsorted(event_times, groups.sorted_time, side="right")
        subject_hazard = np.where(n_passed > 0, cumulative[np.maximum(n_passed - 1, 0)], 0.0)
        own = groups.tie_index
        subject_hazard[groups.dead_pos] = cumulative[own] - hazard[own] + event_hazard[own]
        residuals[groups.order] = event[groups.order].astype(float) - groups.risk * subject_hazard
    return residuals


def _efron_schoenfeld_residuals(
    exog: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    params: np.ndarray,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Schoenfeld residuals of an Efron-tie Cox fit, computed as R's ``residuals.coxph`` does.

    statsmodels' ``PHRegResults.schoenfeld_residuals`` subtracts Breslow risk-set means
    even from an Efron fit, which shifts the proportional-hazards test whenever event
    times are tied (with Breslow ties it matches R, stratified or not). Here, within each
    stratum, the expected covariate for d events tied at time t is the mean over
    k = 0..d-1 of (S1 - k/d * S1_tied) / (S0 - k/d * S0_tied), where S1 sums
    exp(eta) * x over the risk set and S1_tied over the tied events (S0 as in
    ``_efron_tie_groups``). Rows without an event are NaN.
    """
    exog = np.asarray(exog, dtype=float)
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event).reshape(-1).astype(bool)
    linear_predictor = exog @ np.asarray(params, dtype=float).reshape(-1)
    strata_codes = np.zeros(time.shape[0], dtype=np.int64) if strata is None else np.asarray(strata).reshape(-1)
    n_terms = exog.shape[1]
    residuals = np.full(exog.shape, np.nan)
    for code in np.unique(strata_codes):
        groups = _efron_tie_groups(np.flatnonzero(strata_codes == code), time, event, linear_predictor)
        if groups is None:
            continue
        x = exog[groups.order]
        weighted_x = groups.risk[:, None] * x
        # Covariate sums over everyone at risk from each sorted position on (time >= t).
        s1_from = np.cumsum(weighted_x[::-1], axis=0)[::-1]
        dead_pos, tie_index = groups.dead_pos, groups.tie_index
        tied_s1 = np.column_stack(
            [np.bincount(tie_index, weights=weighted_x[dead_pos, term]) for term in range(n_terms)]
        )
        numerator = s1_from[groups.tie_starts][tie_index] - groups.fraction[:, None] * tied_s1[tie_index]
        efron_terms = numerator / groups.denominator[:, None]
        expected = np.column_stack(
            [np.bincount(tie_index, weights=efron_terms[:, term]) for term in range(n_terms)]
        ) / groups.deaths[:, None]
        residuals[groups.order[dead_pos]] = x[dead_pos] - expected[tie_index]
    return residuals


def _cox_grambsch_therneau_test(
    schoenfeld: np.ndarray,
    cov_matrix: np.ndarray,
    transformed_time: np.ndarray,
) -> dict[str, Any]:
    """Grambsch-Therneau score test for non-proportional hazards.

    Implements the classic ``cox.zph`` statistics (survival < 3.0, also used
    by lifelines): for centered time transform ``g`` over event rows and raw
    Schoenfeld residuals ``r``,

    * per-term: ``T_j = (sum g * r*_j)^2 / (d * V_jj * sum g^2)`` with
      ``r* = d * r V`` (1 df each);
    * global: ``T = (g' r) V (g' r)' * d / sum g^2`` on ``p`` df.

    Rows that are not events (NaN residuals) are ignored.
    """
    residuals = np.atleast_2d(np.asarray(schoenfeld, dtype=float))
    if residuals.ndim == 2 and residuals.shape[0] == 1 and np.asarray(schoenfeld).ndim == 1:
        residuals = residuals.reshape(-1, 1)
    variance = np.atleast_2d(np.asarray(cov_matrix, dtype=float))
    g_values = np.asarray(transformed_time, dtype=float).reshape(-1)
    n_terms = int(variance.shape[0])
    empty = {
        "method": "grambsch_therneau_log_time",
        "terms_tested": 0,
        "statistic": None,
        "df": None,
        "p_value": None,
        "term_statistics": [None] * n_terms,
        "term_p_values": [None] * n_terms,
        "term_rho": [None] * n_terms,
        "n_events_used": 0,
    }
    if residuals.shape[0] != g_values.shape[0] or residuals.shape[1] != n_terms:
        return empty
    event_rows = np.isfinite(residuals).all(axis=1) & np.isfinite(g_values)
    n_events = int(event_rows.sum())
    if n_events < 3 or not np.isfinite(variance).all():
        return empty
    resid_events = residuals[event_rows]
    g_centered = g_values[event_rows] - float(np.mean(g_values[event_rows]))
    g_ss = float(np.sum(g_centered ** 2))
    if g_ss <= 0.0:
        return empty
    scaled = n_events * (resid_events @ variance)
    term_scores = g_centered @ scaled
    diag = np.diag(variance)
    term_statistics: list[float | None] = []
    term_p_values: list[float | None] = []
    term_rho: list[float | None] = []
    for idx in range(n_terms):
        denominator = float(n_events * diag[idx] * g_ss)
        if not np.isfinite(denominator) or denominator <= 0.0:
            term_statistics.append(None)
            term_p_values.append(None)
            term_rho.append(None)
            continue
        statistic = float(term_scores[idx] ** 2 / denominator)
        term_statistics.append(statistic)
        term_p_values.append(float(stats.chi2.sf(statistic, df=1)))
        column = scaled[:, idx]
        if np.allclose(np.std(column), 0.0):
            term_rho.append(None)
        else:
            term_rho.append(float(np.corrcoef(g_centered, column)[0, 1]))
    raw_scores = g_centered @ resid_events
    global_statistic = float(raw_scores @ variance @ raw_scores * n_events / g_ss)
    global_ok = np.isfinite(global_statistic) and global_statistic >= 0.0
    return {
        "method": "grambsch_therneau_log_time",
        "terms_tested": int(sum(value is not None for value in term_statistics)),
        "statistic": global_statistic if global_ok else None,
        "df": n_terms if global_ok else None,
        "p_value": float(stats.chi2.sf(global_statistic, df=n_terms)) if global_ok else None,
        "term_statistics": term_statistics,
        "term_p_values": term_p_values,
        "term_rho": term_rho,
        "n_events_used": n_events,
    }

def _cox_categorical_stability_alerts(
    frame: pd.DataFrame,
    categorical_covariates: Sequence[str],
    *,
    min_level_n: int = 5,
) -> dict[str, list[str]]:
    reference_alerts: list[str] = []
    sparse_level_alerts: list[str] = []
    for column in categorical_covariates:
        if column not in frame.columns:
            continue
        series = frame[column]
        counts = series.value_counts(dropna=True)
        if counts.empty:
            continue
        if hasattr(series, "cat") and len(series.cat.categories) > 0:
            reference_level = str(series.cat.categories[0])
        else:
            reference_level = str(counts.index[0])
        reference_count = int(counts.get(reference_level, 0))
        if 0 < reference_count < min_level_n:
            reference_alerts.append(f'{column} reference "{reference_level}" (n={reference_count})')
        for level, count in counts.items():
            if 0 < int(count) < min_level_n:
                sparse_level_alerts.append(f"{column}={level} (n={int(count)})")
    return {
        "reference_levels": reference_alerts,
        "sparse_levels": sparse_level_alerts,
    }


def _cox_wide_ci_alert_terms(
    model_rows: Sequence[dict[str, Any]],
    *,
    width_ratio_threshold: float = 8.0,
) -> list[str]:
    wide_terms: list[str] = []
    for row in model_rows:
        ci_low = _safe_float(row.get("CI lower"))
        ci_high = _safe_float(row.get("CI upper"))
        if ci_low is None or ci_high is None or ci_low <= 0:
            continue
        if (ci_high / ci_low) >= width_ratio_threshold:
            wide_terms.append(str(row.get("Label", row.get("Variable", ""))))
    return wide_terms


def _weighted_test_label(test_name: str | None, *, fh_p: float | None = None) -> str:
    label_map = {
        "logrank": "log-rank",
        "gehan_breslow": "Gehan-Breslow",
        "tarone_ware": "Tarone-Ware",
        "fleming_harrington": (
            f"Fleming-Harrington (fh_p={fh_p:g})"
            if fh_p is not None and np.isfinite(float(fh_p))
            else "Fleming-Harrington (fh_p-only)"
        ),
    }
    return label_map.get(str(test_name), str(test_name))


def _km_scientific_summary(
    summary_rows: Sequence[dict[str, Any]],
    cohort_summary: dict[str, Any],
    group_column: str | None,
    test_payload: dict[str, Any] | None,
    pairwise_rows: Sequence[dict[str, Any]],
    *,
    fh_p: float | None = None,
    outcome_informed_group: bool = False,
    rmst_contrast: dict[str, Any] | None = None,
    rmst_horizon: float | None = None,
    display_horizon: float | None = None,
    global_test_untestable: bool = False,
) -> dict[str, Any]:
    group_count = len(summary_rows)
    min_group_n = min(int(row["N"]) for row in summary_rows)
    min_group_events = min(int(row["Events"]) for row in summary_rows)
    total_events = int(cohort_summary["events"])
    dropped_nonpositive_time_rows = int(cohort_summary.get("dropped_nonpositive_time_rows") or 0)
    dropped_missing_rows = int(cohort_summary.get("dropped_missing_rows") or 0)
    rows_with_infinite_values = int(cohort_summary.get("rows_with_infinite_values") or 0)
    time_column_note = str(cohort_summary.get("time_column_note") or "").strip()

    strengths = [
        "Kaplan-Meier estimation uses Greenwood standard errors with log-log confidence intervals.",
        "Groupwise RMST summaries include Greenwood/delta-method confidence intervals at the RMST truncation horizon when estimable.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []
    cautions.append("All KM outputs assume non-informative (independent) censoring.")
    cautions.append(
        "Competing risks are not modeled here, so 1-KM should not be interpreted as cumulative incidence when other event types can preclude the event of interest."
    )
    cautions.append(
        "Left truncation (delayed entry) is not supported; if patients entered the risk set after time 0, survival estimates can be biased."
    )
    # Notes that hold for every KM run are shown but do not lower the status; only data-driven cautions do.
    standing_cautions = len(cautions)

    single_group = bool(group_column) and group_count < 2
    if group_column and not outcome_informed_group:
        if test_payload is not None:
            test_name = _weighted_test_label(test_payload["test"], fh_p=fh_p)
            strengths.append(f"Global {test_name} comparison was run across {group_count} groups.")
        elif not single_group and not global_test_untestable:
            strengths.append("Survival was estimated per group; between-group hypothesis tests were not run for this grouping.")
        if (
            rmst_horizon is not None
            and display_horizon is not None
            and rmst_horizon + 1e-12 < display_horizon
        ):
            strengths.append(
                "RMST was truncated at the shared observed follow-up support across groups rather than the full plot horizon."
            )
        if pairwise_rows:
            strengths.append("Pairwise group comparisons include Benjamini-Hochberg adjusted p-values.")
            cautions.append(
                "Pairwise BH-adjusted comparisons are shown alongside the global test, so pre-specify in the manuscript how omnibus and pairwise evidence will be interpreted."
            )
            standing_cautions += 1
        # Data-driven cautions follow the standing notes counted above.
        if global_test_untestable:
            cautions.append(
                "The global between-group test could not be computed because its log-rank variance is singular "
                "(for example every event of the compared groups falls at one time that nobody at risk survives); "
                "R's survdiff stops with an error on such data."
            )
        untestable_pairs = [str(row.get("Comparison")) for row in pairwise_rows if row.get("P value") is None]
        if untestable_pairs:
            cautions.append(
                f"{len(untestable_pairs)} pairwise comparison(s) could not be tested because their log-rank variance is zero "
                f"(for example every event of both groups at one time that nobody at risk survives): "
                f"{_summarize_labels(untestable_pairs, max_items=3)}. They are left out of the BH adjustment."
            )
    elif group_column and outcome_informed_group:
        strengths.append("Outcome-informed groups were visualized descriptively without a fresh between-group hypothesis test.")
    else:
        strengths.append("This run estimates a single cohort without between-group multiple testing.")

    if single_group:
        cautions.append(
            f"Only one group of {group_column} remained in the analyzable cohort, so no between-group comparison was run."
        )
    if time_column_note:
        cautions.append(time_column_note)
    if total_events < 20:
        cautions.append("Fewer than 20 total events limits precision of survival contrasts.")
    if min_group_n < 15:
        cautions.append("At least one group has fewer than 15 patients.")
    if min_group_events < 5:
        cautions.append("At least one group has fewer than 5 events, so median survival and p-values may be unstable.")
    if cohort_summary["median_follow_up"] is None:
        cautions.append("Median follow-up could not be estimated from the censoring distribution.")
    if dropped_nonpositive_time_rows:
        cautions.append(
            f"{dropped_nonpositive_time_rows} row(s) with negative survival time were excluded before KM estimation."
        )
    if dropped_missing_rows:
        missing_message = f"{dropped_missing_rows} row(s) with missing KM inputs were excluded before KM estimation."
        if rows_with_infinite_values:
            missing_message = (
                f"{dropped_missing_rows} row(s) with missing KM inputs were excluded before KM estimation, "
                f"including {rows_with_infinite_values} row(s) where +/-Inf values were coerced to missing."
            )
        cautions.append(missing_message)

    if group_column and outcome_informed_group:
        headline = "Outcome-informed groups were visualized descriptively; a fresh log-rank p-value is not reported on the same selected split."
        cautions.append("This grouping was derived using outcome information, so treat the KM figure as exploratory rather than confirmatory.")
        next_steps.append("Report the selection procedure and any selection-adjusted p-value from the cutpoint or signature workflow instead of a fresh raw log-rank test.")
        if rmst_contrast is not None and rmst_contrast.get("estimate") is not None:
            cautions.append(
                "Two-group RMST contrast is shown descriptively on the selected split; a fresh Wald p-value is intentionally withheld, "
                "and so is its confidence interval, because the split was chosen using the outcome."
            )
    elif group_column and test_payload:
        if float(test_payload["p_value"]) < 0.05:
            headline = f"Global {_weighted_test_label(test_payload['test'], fh_p=fh_p)} testing suggests survival differs across groups."
            next_steps.append("Inspect groupwise medians, RMST summaries with delta-method confidence intervals when estimable, and adjusted pairwise comparisons before making a manuscript claim.")
        else:
            headline = f"Global {_weighted_test_label(test_payload['test'], fh_p=fh_p)} testing does not show clear survival separation across groups."
            next_steps.append("Do not treat visual curve separation alone as evidence if the global test is not significant.")
    elif single_group:
        headline = f"Only one group of {group_column} was observed, so survival was estimated for a single cohort without a between-group test."
        next_steps.append(f"Check {group_column}: every analyzable row has the same value, so it cannot compare subcohorts.")
    elif group_column and global_test_untestable:
        headline = "Survival was estimated per group, but the global between-group test could not be computed on these data."
        next_steps.append("Merge or leave out the groups whose events all fall at one time before testing the curves formally.")
    elif group_column:
        headline = "Survival was estimated per group without a between-group hypothesis test."
        next_steps.append("Run the between-group test before comparing the curves formally.")
    else:
        headline = "Single-cohort survival was estimated without a between-group hypothesis test."
        next_steps.append("Use a grouping variable to compare subcohorts formally.")

    if min_group_events < 5 or min_group_n < 15:
        next_steps.append("Avoid anchoring on median survival in sparse groups; emphasize confidence intervals and follow-up maturity.")
    if (
        rmst_contrast is not None
        and rmst_contrast.get("estimate") is not None
        and rmst_contrast.get("ci_lower") is not None
        and rmst_contrast.get("ci_upper") is not None
    ):
        strengths.append(
            f"Two-group RMST contrast ({rmst_contrast['comparison']}) was summarized with a delta-method confidence interval."
        )
    if rmst_contrast is not None and rmst_contrast.get("p_value") is not None:
        strengths.append(
            f"Two-group RMST contrast ({rmst_contrast['comparison']}) also reports a Wald p-value from the delta-method standard error."
        )

    status = "robust"
    if len(cautions) > standing_cautions:
        status = "review"
    if total_events < 10 or min_group_events < 3:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        # Data-driven cautions lead; the notes that hold for every KM run follow.
        "cautions": cautions[standing_cautions:] + cautions[:standing_cautions],
        "next_steps": next_steps,
        "metrics": [
            {"label": "Patients", "value": int(cohort_summary["n"])},
            {"label": "Events", "value": total_events},
            {"label": "Groups", "value": group_count},
            {"label": "Median follow-up", "value": _safe_float(cohort_summary["median_follow_up"])},
            *(
                [{"label": "Dropped for negative time", "value": dropped_nonpositive_time_rows}]
                if dropped_nonpositive_time_rows
                else []
            ),
            *(
                [{"label": "RMST difference", "value": _safe_float(rmst_contrast["estimate"])}]
                if rmst_contrast is not None and rmst_contrast.get("estimate") is not None
                else []
            ),
            *(
                [{"label": "RMST difference p", "value": _safe_float(rmst_contrast["p_value"])}]
                if rmst_contrast is not None and rmst_contrast.get("p_value") is not None
                else []
            ),
        ],
    }


class _SummaryNotes(NamedTuple):
    """The ordered strengths, cautions, and next steps of a scientific summary."""

    strengths: list[str]
    cautions: list[str]
    next_steps: list[str]


# Notes that hold for every Cox fit: shown with the cautions, but they do not lower the status.
_COX_ASSUMPTION_NOTES = (
    "All Cox outputs assume non-informative (independent) censoring.",
    "Competing risks are not modeled in this Cox workflow, so cause-specific questions need dedicated competing-risk methods rather than treating other event types as ordinary censoring.",
    "Left truncation (delayed entry) is not supported; if patients entered the risk set after time 0, coefficient estimates and survival summaries can be biased.",
)
_COX_APPARENT_C_NOTE = "The Cox C-index is apparent, so it is optimistic and should not be treated as external validation."
_COX_C_INTERVAL_NOTE = (
    "The reported C-index confidence interval is a bootstrap interval on the same cohort using fixed fitted risk scores, "
    "so it understates full model-building uncertainty and does not replace external validation."
)
_COX_EXTERNAL_NOTE = (
    "The current dashboard does not yet provide a built-in external-cohort apply workflow for Cox validation; "
    "validate the final specification on a separate cohort outside this run."
)
_COX_COMPLETE_CASE_NOTE = (
    "Changing the covariate set can change the analyzable cohort because Cox fitting uses complete-case rows for the selected covariates."
)
_COX_STRATA_COMPLETE_CASE_NOTE = (
    "Changing the covariate set or strata set can change the analyzable cohort because Cox fitting uses complete-case rows for the selected inputs."
)
_COX_STRATA_TERMS_NOTE = (
    "Stratified variables are used only for stratum-specific baseline hazards and do not receive hazard-ratio estimates."
)
# How the complete-case rule and stratification work holds for every such fit; rows actually
# dropped are reported by their own data-driven caution.
_COX_STANDING_NOTES = frozenset(
    {
        *_COX_ASSUMPTION_NOTES,
        _COX_APPARENT_C_NOTE,
        _COX_C_INTERVAL_NOTE,
        _COX_EXTERNAL_NOTE,
        _COX_COMPLETE_CASE_NOTE,
        _COX_STRATA_COMPLETE_CASE_NOTE,
        _COX_STRATA_TERMS_NOTE,
    }
)


class _CoxTermAlerts(NamedTuple):
    significant: list[str]
    proportional_hazards: list[str]
    reference_levels: list[str]
    sparse_levels: list[str]
    wide_ci: list[str]
    non_estimable: list[str]


def _nonempty_strings(values: Any) -> list[str]:
    return [str(value) for value in values or [] if value]


def _cox_term_alerts(
    model_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
    categorical_alerts: dict[str, list[str]] | None,
) -> _CoxTermAlerts:
    categorical_alerts = categorical_alerts or {"reference_levels": [], "sparse_levels": []}
    return _CoxTermAlerts(
        significant=[
            row["Label"]
            for row in model_rows
            if row["P value"] is not None
            and row["CI lower"] is not None
            and row["CI upper"] is not None
            and float(row["P value"]) < 0.05
            and not (float(row["CI lower"]) <= 1.0 <= float(row["CI upper"]))
        ],
        proportional_hazards=[
            str(row["Term"])
            for row in diagnostic_rows
            if row.get("_kind") != "global"
            if row["P value"] is not None and float(row["P value"]) < 0.05
        ],
        reference_levels=_nonempty_strings(categorical_alerts.get("reference_levels", [])),
        sparse_levels=_nonempty_strings(categorical_alerts.get("sparse_levels", [])),
        wide_ci=_cox_wide_ci_alert_terms(model_rows),
        non_estimable=[
            str(row.get("Label", row.get("Variable", "")))
            for row in model_rows
            if row.get("Hazard ratio") is None or row.get("CI lower") is None or row.get("CI upper") is None
        ],
    )


def _cox_ill_conditioned_number(model_stats: dict[str, Any]) -> float | None:
    """The design-matrix condition number when it exceeds the warning threshold, else None."""
    design_condition_number = _safe_float(model_stats.get("design_condition_number"))
    warning_threshold = _safe_float(
        model_stats.get("design_condition_warning_threshold") or COX_CONDITION_NUMBER_WARN_THRESHOLD
    )
    if (
        design_condition_number is not None
        and warning_threshold is not None
        and design_condition_number > warning_threshold
    ):
        return design_condition_number
    return None


def _cox_strata_notes(model_stats: dict[str, Any], strata_columns: list[str], notes: _SummaryNotes) -> None:
    strengths, cautions, next_steps = notes
    if not strata_columns:
        cautions.append(_COX_COMPLETE_CASE_NOTE)
        return
    n_strata = _safe_float(model_stats.get("n_strata"))
    zero_event_strata_count = int(model_stats.get("zero_event_strata_count") or 0)
    sparse_event_strata_count = int(model_stats.get("sparse_event_strata_count") or 0)
    high_cardinality_strata_columns = _nonempty_strings(model_stats.get("high_cardinality_strata_columns"))
    high_cardinality_numeric_strata_columns = _nonempty_strings(
        model_stats.get("high_cardinality_numeric_strata_columns")
    )
    cautions.append(_COX_STRATA_COMPLETE_CASE_NOTE)
    strengths.append(
        f"Baseline hazards were stratified by {_summarize_labels(strata_columns, max_items=3)}, so those variables are not reported as hazard-ratio terms."
    )
    if n_strata is not None:
        strengths.append(f"The fitted stratified Cox specification used {int(n_strata)} observed strata combination(s).")
    cautions.append(_COX_STRATA_TERMS_NOTE)
    if zero_event_strata_count:
        cautions.append(
            f"{zero_event_strata_count} observed stratum/strata had zero events, so the stratified Cox fit can look more stable than the within-stratum information actually supports."
        )
        next_steps.append("Collapse sparse strata before treating a stratified Cox fit as manuscript-ready.")
    if sparse_event_strata_count:
        cautions.append(
            f"{sparse_event_strata_count} observed stratum/strata had only one event, which makes within-stratum information thin for a stratified Cox fit."
        )
    if high_cardinality_numeric_strata_columns:
        cautions.append(
            "Selected strata include numeric/high-cardinality columns: "
            f"{_summarize_labels(high_cardinality_numeric_strata_columns, max_items=3)}."
        )
        next_steps.append("Recode continuous strata into a small number of clinically meaningful levels before fitting stratified Cox.")
    elif high_cardinality_strata_columns:
        cautions.append(
            "Selected strata include many observed levels: "
            f"{_summarize_labels(high_cardinality_strata_columns, max_items=3)}."
        )


def _cox_input_quality_notes(
    model_stats: dict[str, Any],
    *,
    outcome_rows: int | None,
    dropped_rows: int | None,
    alerts: _CoxTermAlerts,
    ill_conditioned_number: float | None,
    notes: _SummaryNotes,
) -> None:
    _, cautions, next_steps = notes
    dropped_nonpositive_time_rows = int(model_stats.get("dropped_nonpositive_time_rows") or 0)
    rows_with_infinite_values = int(model_stats.get("rows_with_infinite_values") or 0)
    rows_with_infinite_outcome = int(model_stats.get("rows_with_infinite_outcome_values") or 0)
    time_column_note = str(model_stats.get("time_column_note") or "").strip()
    if time_column_note:
        cautions.append(time_column_note)
    if rows_with_infinite_outcome:
        cautions.append(
            f"{rows_with_infinite_outcome} row(s) with a +/-Inf survival time were excluded before the outcome-valid cohort was formed."
        )
    if dropped_rows:
        drop_message = f"{int(dropped_rows)} outcome-valid rows were excluded because at least one selected Cox input was missing."
        if outcome_rows is not None and outcome_rows > 0:
            dropped_fraction = float(dropped_rows / outcome_rows)
            drop_message = (
                f"{int(dropped_rows)} outcome-valid rows ({dropped_fraction:.1%}) were excluded "
                "because at least one selected Cox input was missing."
            )
        cautions.append(drop_message)
        next_steps.append("Review missingness patterns or use an imputation strategy before treating the fitted cohort as representative.")
    if dropped_nonpositive_time_rows:
        cautions.append(
            f"{dropped_nonpositive_time_rows} outcome-valid row(s) with negative survival time were excluded before Cox fitting."
        )
    if rows_with_infinite_values:
        cautions.append(
            f"{rows_with_infinite_values} outcome-valid row(s) contained +/-Inf values in selected Cox inputs; those values were coerced to missing before complete-case filtering."
        )
    if ill_conditioned_number is not None:
        cautions.append(
            f"The Cox design matrix is poorly conditioned (condition number {ill_conditioned_number:.2e}), "
            "so hazard ratios can become numerically unstable when selected inputs vary on very different scales or are nearly collinear."
        )
        next_steps.append(
            "Standardize extreme-scale continuous covariates and review collinearity before treating Cox coefficients as stable."
        )
    if alerts.proportional_hazards:
        cautions.append(
            f"Possible proportional-hazards violations detected for: {', '.join(alerts.proportional_hazards)}."
        )
        next_steps.append("Consider stratification or time-varying effects for PH-violating terms.")
    if alerts.reference_levels:
        cautions.append(
            f"Some Cox reference levels are very small after missing-value filtering: {_summarize_labels(alerts.reference_levels, max_items=4)}."
        )
        next_steps.append("Use a more stable reference level or collapse sparse categories before interpreting reference-based contrasts.")
    if alerts.sparse_levels:
        cautions.append(
            f"Sparse categorical levels remain in the analyzable cohort: {_summarize_labels(alerts.sparse_levels, max_items=4)}."
        )
        next_steps.append("Collapse rare categorical levels before treating term-specific hazard ratios as stable.")
    if alerts.wide_ci:
        cautions.append(
            f"Some hazard-ratio intervals are very wide, which suggests unstable estimates: {_summarize_labels(alerts.wide_ci, max_items=4)}."
        )
        next_steps.append("Treat wide-interval terms as unstable unless the category encoding or cohort size is improved.")
    if alerts.non_estimable:
        cautions.append(
            f"Some Cox contrasts produced non-estimable hazard ratios or confidence intervals: {_summarize_labels(alerts.non_estimable, max_items=4)}."
        )
        next_steps.append("Collapse sparse categories or remove quasi-separated terms before interpreting those contrasts.")


def _cox_fit_statistic_notes(
    model_stats: dict[str, Any],
    *,
    c_index: float | None,
    c_index_label: str,
    notes: _SummaryNotes,
) -> None:
    strengths, cautions, next_steps = notes
    lr_statistic = _safe_float(model_stats.get("lr_statistic"))
    lr_pvalue = _safe_float(model_stats.get("lr_pvalue"))
    lr_note = str(model_stats.get("lr_note") or "").strip()
    global_ph_statistic = _safe_float(model_stats.get("global_ph_statistic"))
    global_ph_df = _safe_float(model_stats.get("global_ph_df"))
    global_ph_pvalue = _safe_float(model_stats.get("global_ph_pvalue"))
    global_ph_terms_tested = int(model_stats.get("global_ph_terms_tested") or 0)
    martingale_terms = _nonempty_strings(model_stats.get("martingale_terms"))
    martingale_note = str(model_stats.get("martingale_note") or "").strip()

    if c_index is not None:
        if c_index < 0.6:
            cautions.append("Apparent model discrimination is modest (C-index below 0.60).")
        cautions.append(_COX_APPARENT_C_NOTE)
        strengths.append(
            f"A C-index of {c_index:.3f} means the fitted model correctly orders approximately {c_index * 100:.1f}% of evaluable patient pairs by predicted risk."
        )
    if lr_statistic is not None and lr_pvalue is not None:
        strengths.append(
            f"The overall likelihood-ratio test compares the fitted Cox model with a null model (LR chi-square {lr_statistic:.2f}, p={lr_pvalue:.3g})."
        )
    elif lr_note:
        cautions.append(lr_note)
    if global_ph_pvalue is not None:
        label = "Global proportional-hazards test (Grambsch-Therneau, log time)"
        if global_ph_statistic is not None and global_ph_df is not None:
            strengths.append(
                f"{label}: chi-square {global_ph_statistic:.2f} on {int(global_ph_df)} df (p={global_ph_pvalue:.3g}) across {global_ph_terms_tested} model term(s)."
            )
        else:
            strengths.append(f"{label}: p={global_ph_pvalue:.3g} across {global_ph_terms_tested} model term(s).")
        if global_ph_pvalue < 0.05:
            cautions.append(
                "The global Grambsch-Therneau test suggests at least one term departs from proportional hazards, even if not every single term crosses the nominal cutoff."
            )
            next_steps.append(
                "Inspect the per-term PH table and Schoenfeld residual plots before writing that the proportional-hazards assumption was fully satisfied."
            )
    if martingale_terms:
        strengths.append(
            f"Martingale residual screening is available for continuous covariates not marked categorical: {_summarize_labels(martingale_terms, max_items=4)}."
        )
        next_steps.append(
            "If martingale LOWESS trends curve strongly away from zero, consider splines, nonlinear transforms, or recoding the continuous covariate."
        )
    elif martingale_note:
        if "unavailable" in martingale_note.lower():
            cautions.append(martingale_note)
        else:
            strengths.append(martingale_note)
    ci_low = _safe_float(model_stats.get("c_index_ci_lower"))
    ci_high = _safe_float(model_stats.get("c_index_ci_upper"))
    ci_level = _safe_float(model_stats.get("c_index_ci_level"))
    if ci_low is not None and ci_high is not None:
        level_pct = int(round((ci_level or 0.95) * 100.0))
        strengths.append(
            f"{c_index_label} includes an internal bootstrap percentile {level_pct}% CI ({ci_low:.3f} to {ci_high:.3f}) on the fixed fitted risk scores."
        )
        cautions.append(_COX_C_INTERVAL_NOTE)


def _cox_summary_headline(alerts: _CoxTermAlerts, structural_instability: bool, next_steps: list[str]) -> str:
    significant_terms = alerts.significant
    # Name the estimates that look unstable, not the significant ones: causes (a small reference or
    # sparse level) before the wide intervals they produce. Instability from the model as a whole
    # (few events per parameter, collinearity, sparse strata) has no term to name.
    unstable_terms = list(dict.fromkeys(alerts.reference_levels + alerts.sparse_levels + alerts.non_estimable + alerts.wide_ci))
    if significant_terms:
        if structural_instability:
            headline = (
                f"Model fit shows {len(significant_terms)} term(s) with nominal hazard association "
                f"({_summarize_labels(significant_terms)}), but "
                + (
                    f"some estimates appear unstable: {_summarize_labels(unstable_terms)}."
                    if unstable_terms
                    else "the estimates may be unstable; see the cautions."
                )
            )
        elif alerts.proportional_hazards:
            headline = (
                f"Model fit shows {len(significant_terms)} term(s) with nominal hazard association, "
                f"but some terms need closer proportional-hazards review: {_summarize_labels(alerts.proportional_hazards)}."
            )
        else:
            headline = (
                f"Model fit identified {len(significant_terms)} term(s) with nominal hazard association: "
                f"{_summarize_labels(significant_terms)}."
            )
        next_steps.append("Interpret hazard ratios together with confidence intervals, not p-values alone.")
        return headline
    if structural_instability:
        headline = (
            "Model fit completed, but no term shows clear nominal hazard association and some estimates remain unstable under the current specification."
        )
    elif alerts.proportional_hazards:
        headline = (
            "Model fit completed, but no term shows clear nominal hazard association and some terms still need closer proportional-hazards review under the current specification."
        )
    else:
        headline = "Model fit completed, but no term shows clear nominal hazard association under the current specification."
    next_steps.append("Revisit covariate selection, encoding, and cohort size before forcing interpretation.")
    return headline


def _cox_summary_metrics(
    model_stats: dict[str, Any],
    *,
    outcome_rows: int | None,
    dropped_rows: int | None,
    epv: float | None,
    c_index: float | None,
    c_index_label: str,
    strata_columns: list[str],
    ill_conditioned_number: float | None,
) -> list[dict[str, Any]]:
    n_strata = _safe_float(model_stats.get("n_strata"))
    zero_event_strata_count = int(model_stats.get("zero_event_strata_count") or 0)
    sparse_event_strata_count = int(model_stats.get("sparse_event_strata_count") or 0)
    lr_statistic = _safe_float(model_stats.get("lr_statistic"))
    lr_pvalue = _safe_float(model_stats.get("lr_pvalue"))
    ci_low = _safe_float(model_stats.get("c_index_ci_lower"))
    ci_high = _safe_float(model_stats.get("c_index_ci_upper"))
    ci_level = _safe_float(model_stats.get("c_index_ci_level"))
    metrics = [
        {"label": "Outcome-valid rows", "value": outcome_rows},
        {"label": "Dropped for missing Cox inputs", "value": dropped_rows},
        {"label": "Dropped for negative time", "value": int(model_stats.get("dropped_nonpositive_time_rows") or 0)},
        {"label": "Events", "value": int(model_stats["events"])},
        {"label": "Parameters", "value": int(model_stats["parameters"])},
        {"label": "EPV", "value": epv},
    ]
    if c_index is not None:
        metrics.append({"label": c_index_label, "value": c_index})
    if strata_columns:
        metrics.append({"label": "Strata variables", "value": len(strata_columns)})
    if ill_conditioned_number is not None:
        metrics.append({"label": "Condition number", "value": f"{ill_conditioned_number:.2e}"})
    if n_strata is not None:
        metrics.append({"label": "Observed strata", "value": int(n_strata)})
    if zero_event_strata_count:
        metrics.append({"label": "Zero-event strata", "value": zero_event_strata_count})
    if sparse_event_strata_count:
        metrics.append({"label": "One-event strata", "value": sparse_event_strata_count})
    if lr_statistic is not None:
        metrics.append({"label": "LR chi-square", "value": lr_statistic})
    if lr_pvalue is not None:
        metrics.append({"label": "LR test p-value", "value": lr_pvalue})
    if c_index is not None and ci_low is not None and ci_high is not None:
        level_pct = int(round(float(ci_level or 0.95) * 100.0))
        ci_metric_label = (
            f"Apparent C-index {level_pct}% CI"
            if c_index_label == "Apparent C-index (training cohort)"
            else f"{c_index_label} {level_pct}% CI"
        )
        metrics.append(
            {
                "label": ci_metric_label,
                "value": f"{float(ci_low):.3f} to {float(ci_high):.3f}",
            }
        )
    return metrics


def _cox_scientific_summary(
    model_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
    model_stats: dict[str, Any],
    *,
    categorical_alerts: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    alerts = _cox_term_alerts(model_rows, diagnostic_rows, categorical_alerts)
    complete_case_n = int(model_stats["n"])
    outcome_rows_raw = model_stats.get("outcome_rows")
    dropped_rows_raw = model_stats.get("dropped_rows")
    outcome_rows = int(outcome_rows_raw) if outcome_rows_raw is not None else None
    dropped_rows = int(dropped_rows_raw) if dropped_rows_raw is not None else None
    strata_columns = _nonempty_strings(model_stats.get("strata_columns"))
    c_index_label = str(model_stats.get("c_index_label") or "Apparent C-index (training cohort)")
    epv = _safe_float(model_stats.get("events_per_parameter"))
    c_index = _safe_float(model_stats.get("c_index"))
    ill_conditioned_number = _cox_ill_conditioned_number(model_stats)
    zero_event_strata_count = int(model_stats.get("zero_event_strata_count") or 0)
    high_cardinality_numeric_strata_columns = _nonempty_strings(
        model_stats.get("high_cardinality_numeric_strata_columns")
    )

    if outcome_rows is None:
        cohort_statement = f"Model estimates use the analyzable cohort after dropping rows with missing selected Cox inputs (N = {complete_case_n})."
    else:
        cohort_statement = (
            "Model estimates use the analyzable cohort after dropping rows with missing selected Cox inputs "
            f"(N = {complete_case_n} of {int(outcome_rows)} outcome-valid rows)."
        )
    notes = _SummaryNotes(
        strengths=[
            "Cox regression was fit with the Efron tie method.",
            cohort_statement,
            "Proportional-hazards checks use the Grambsch-Therneau score test on scaled Schoenfeld residuals versus log time (per term, 1 df, plus a global test); the plot overlay uses LOWESS smoothing for visual inspection only.",
            (
                "Discrimination was not reported for the stratified Cox fit because pooled cross-stratum ranking is not directly interpretable."
                if strata_columns
                else "The reported discrimination metric is an apparent C-index on the fitted cohort, so it reflects training-cohort ranking only."
            ),
        ],
        cautions=[],
        next_steps=[],
    )
    if epv is not None and epv < 10:
        notes.cautions.append("Events per parameter is below 10, so coefficients may be unstable or overfit.")
        notes.next_steps.append("Reduce model complexity or increase the event count before treating estimates as final.")
    _cox_strata_notes(model_stats, strata_columns, notes)
    notes.cautions.extend(_COX_ASSUMPTION_NOTES)
    _cox_input_quality_notes(
        model_stats,
        outcome_rows=outcome_rows,
        dropped_rows=dropped_rows,
        alerts=alerts,
        ill_conditioned_number=ill_conditioned_number,
        notes=notes,
    )
    _cox_fit_statistic_notes(model_stats, c_index=c_index, c_index_label=c_index_label, notes=notes)
    notes.cautions.append(_COX_EXTERNAL_NOTE)
    if not alerts.significant:
        notes.cautions.append("No model term shows clear nominal evidence at p < 0.05.")

    structural_instability = bool(
        (epv is not None and epv < 10)
        or alerts.reference_levels
        or alerts.sparse_levels
        or alerts.wide_ci
        or alerts.non_estimable
        or zero_event_strata_count
        or high_cardinality_numeric_strata_columns
        or ill_conditioned_number is not None
    )
    headline = _cox_summary_headline(alerts, structural_instability, notes.next_steps)

    status = "robust"
    if any(caution not in _COX_STANDING_NOTES for caution in notes.cautions):
        status = "review"
    if (
        (epv is not None and epv < 5)
        or len(alerts.proportional_hazards) >= 2
        or alerts.reference_levels
        or alerts.sparse_levels
        or zero_event_strata_count
        or high_cardinality_numeric_strata_columns
    ):
        status = "caution"

    # Data-driven cautions lead; the notes that hold for every Cox fit follow.
    cautions = [caution for caution in notes.cautions if caution not in _COX_STANDING_NOTES]
    cautions += [caution for caution in notes.cautions if caution in _COX_STANDING_NOTES]
    return {
        "status": status,
        "headline": headline,
        "strengths": notes.strengths,
        "cautions": cautions,
        "next_steps": notes.next_steps,
        "metrics": _cox_summary_metrics(
            model_stats,
            outcome_rows=outcome_rows,
            dropped_rows=dropped_rows,
            epv=epv,
            c_index=c_index,
            c_index_label=c_index_label,
            strata_columns=strata_columns,
            ill_conditioned_number=ill_conditioned_number,
        ),
    }


def _signature_scientific_summary(
    best_split: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    strengths = [
        "Signatures are ordered by the significance rule first (BH-adjusted p, CI excluding 1, and any enabled permutation, bootstrap, and resampling filters), then by the heuristic stability score.",
    ]
    cautions: list[str] = []
    next_steps: list[str] = []
    # Notes that hold for every signature search are shown but do not lower the status.
    standing_notes: list[str] = [
        "Stability score is a composite heuristic that mixes significance, effect size, replication support, and parsimony; its weights are expert-set and not independently validated."
    ]

    permutation_iterations = int(search_space.get("permutation_iterations") or 0)
    validation_iterations = int(search_space.get("validation_iterations") or 0)
    bootstrap_iterations = int(search_space.get("bootstrap_iterations") or 0)

    if permutation_iterations > 0:
        strengths.append(
            "Permutation p-values are search-adjusted: each shuffle re-scores every tested signature that meets the group-size rule "
            "(applying the event rule within the shuffle) and keeps the maximum statistic (Westfall-Young), so they account for selecting the best rule."
        )
        # The p-value is (exceedances + 1) / (valid shuffles + 1), so its floor follows the
        # shuffles that produced a statistic, not the number requested.
        used_permutations = int(best_split.get("Permutation valid resamples") or 0) or permutation_iterations
        min_attainable = 1.0 / (used_permutations + 1.0)
        if min_attainable > float(search_space["significance_level"]):
            cautions.append(
                f"With {used_permutations} valid permutations the smallest attainable permutation p is {min_attainable:.3g}, above alpha; increase permutations or no signature can pass."
            )
    if validation_iterations > 0:
        strengths.append("Within-cohort resampling replication was checked for top-ranked candidates.")
        standing_notes.append(
            "Resampling replication folds are subsamples of the same cohort used to discover the signature, so they measure stability, not independent validation."
        )
    time_column_note = str(search_space.get("time_column_note") or "").strip()
    if time_column_note:
        cautions.append(time_column_note)
    skipped_candidates = [item for item in search_space.get("skipped_candidates") or [] if isinstance(item, dict)]
    if skipped_candidates:
        described = "; ".join(f"{item.get('column')} ({item.get('reason')})" for item in skipped_candidates[:4])
        more = f" and {len(skipped_candidates) - 4} more" if len(skipped_candidates) > 4 else ""
        cautions.append(f"Some candidates produced no usable rule and were not searched: {described}{more}.")
    unevaluated = int(search_space.get("robustness_unevaluated_signatures") or 0)
    if unevaluated:
        cautions.append(
            f"{unevaluated} further rule(s) passed the BH threshold but were not screened for robustness, because enough "
            "significant signatures were already found in rank order or the robustness cap was reached; they are listed as not significant."
        )

    support = _safe_float(best_split.get("Bootstrap support (p<alpha)"))
    direction_consistency = _safe_float(best_split.get("Bootstrap HR direction consistency"))
    validation_support = _safe_float(best_split.get("Validation support (p<alpha)"))
    permutation_p = _safe_float(best_split.get("Permutation p"))
    bootstrap_valid_resamples = int(best_split.get("Bootstrap valid resamples") or 0)
    bootstrap_skipped_resamples = int(best_split.get("Bootstrap skipped resamples") or 0)
    permutation_valid_resamples = int(best_split.get("Permutation valid resamples") or 0)
    permutation_skipped_resamples = int(best_split.get("Permutation skipped resamples") or 0)
    validation_valid_folds = int(best_split.get("Validation valid folds") or 0)
    validation_skipped_folds = int(best_split.get("Validation skipped folds") or 0)
    signature_n = int(best_split["N signature+"])
    is_significant = bool(best_split["Statistically significant"])
    alpha = float(search_space["significance_level"])

    if search_space["truncated"]:
        cautions.append("Search space hit the internal combination cap, so discovery was not exhaustive.")
        cautions.append("Adjusted p-values only account for the tested subset of combinations under the cap.")
        cautions.append(
            "When the cap is hit, the retained rules depend on deterministic candidate order, so earlier input columns can be overrepresented."
        )
    analyzable_n = search_space.get("n_rows_analyzed")
    if analyzable_n is not None:
        standing_notes.append(
            f"Signature discovery used {int(analyzable_n)} analyzable rows after dropping missing candidate values; changing the candidate set can change the search cohort."
        )
    else:
        standing_notes.append(
            "Signature discovery uses the analyzable subset after dropping missing candidate values; changing the candidate set can change the search cohort."
        )
    if permutation_iterations == 0 and validation_iterations == 0:
        cautions.append(
            "Because the signature is selected from many tested rules in the same cohort, "
            "reported p-values (including BH-adjusted) can be optimistic; enable search-adjusted permutations and validate externally."
        )
    if not is_significant:
        cautions.append("Top-ranked signature remains exploratory under the current significance rules.")
        next_steps.append("Do not lock this signature for biological interpretation without further validation.")
    if signature_n < max(15, int(search_space["min_group_size"])):
        cautions.append("The signature-positive subgroup is small, which can inflate apparent separation.")
    if support is not None and support < 0.7:
        cautions.append("Bootstrap support is below 0.70, so the signal may be unstable.")
    if bootstrap_iterations > 0 and bootstrap_valid_resamples == 0:
        cautions.append("Bootstrap resampling did not yield any valid signature resamples, so bootstrap stability support is unavailable.")
    elif bootstrap_iterations > 0 and bootstrap_skipped_resamples > 0:
        cautions.append(
            f"Bootstrap screening skipped {bootstrap_skipped_resamples} of {bootstrap_iterations} resamples (a side with fewer "
            f"than {_RESAMPLE_MIN_ROWS_PER_GROUP} rows or without an event, or a failed test); skipped resamples count against "
            "bootstrap support and direction consistency."
        )
    if direction_consistency is not None and direction_consistency < 0.75:
        cautions.append("Bootstrap hazard-ratio direction is not consistently preserved.")
    if validation_support is not None and validation_support < 0.5:
        cautions.append("Within-cohort resampling replication support is below 0.50.")
    if validation_iterations > 0 and validation_valid_folds == 0:
        cautions.append("Resampling replication did not yield any analyzable subsamples, so replication support is unavailable.")
    elif validation_iterations > 0 and validation_skipped_folds > 0:
        cautions.append(
            f"Resampling replication skipped {validation_skipped_folds} of {validation_iterations} subsamples (a side with "
            f"fewer than {_RESAMPLE_MIN_ROWS_PER_GROUP} rows or without an event, or a failed test); skipped subsamples count "
            "against replication support."
        )
    if permutation_p is not None and permutation_p > alpha:
        cautions.append("The search-adjusted permutation p-value exceeds alpha: the top signature is not distinguishable from the best rule expected under no association.")
    if permutation_iterations > 0 and permutation_valid_resamples == 0:
        cautions.append("Permutation screening did not yield any valid resamples, so the empirical p-value is unavailable.")
    elif permutation_iterations > 0 and permutation_skipped_resamples > 0:
        cautions.append(
            f"Permutation screening used {permutation_valid_resamples} valid shuffles and skipped {permutation_skipped_resamples}, so the empirical p-value is based on fewer resamples than requested."
        )

    has_internal_confirmation = (
        permutation_iterations > 0 or validation_iterations > 0
    )
    if is_significant and has_internal_confirmation:
        headline = (
            "Top-ranked signature passed the within-cohort screening rules "
            "but still needs external validation before it is framed as a biomarker claim."
        )
        next_steps.append("Validate the locked signature on an external cohort before presenting it as a biomarker claim.")
    elif is_significant:
        headline = (
            "Top-ranked signature clears the current ranking rules but remains exploratory "
            "without a search-adjusted permutation test."
        )
        next_steps.append("Run search-adjusted permutations and external validation before framing this signature as a robust biomarker claim.")
    else:
        headline = "Top-ranked signature is still exploratory and should be treated as hypothesis-generating."
        next_steps.append("Narrow the candidate list or increase sample size before claiming a stable signature.")

    status = "robust"
    if cautions:
        status = "review"
    if not is_significant or signature_n < 10:
        status = "caution"

    return {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        # Data-driven cautions lead; the notes that hold for every search follow.
        "cautions": cautions + standing_notes,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Tested combos", "value": int(search_space["tested_combinations"])},
            {"label": "Significant combos", "value": int(search_space["significant_signatures"])},
            {"label": "Signature+ N", "value": signature_n},
            {"label": "Permutation p (search-adjusted)", "value": permutation_p},
        ],
    }


def _km_median_time(times: np.ndarray, events: np.ndarray) -> float | None:
    """Smallest time with Kaplan-Meier S(t) <= 0.5 (lifelines/SAS convention).

    ``SurvfuncRight.quantile`` requires S(t) < 0.5 strictly, so it skips a time
    where the curve lands exactly on 0.5 (common with small or even-sized groups).
    """
    sf = SurvfuncRight(np.asarray(times, dtype=float), np.asarray(events, dtype=float))
    survival = np.asarray(sf.surv_prob, dtype=float)
    reached = np.flatnonzero(survival <= 0.5 + 1e-12)
    if reached.size == 0:
        return None
    return _safe_float(float(np.asarray(sf.surv_times, dtype=float)[reached[0]]))


def _median_follow_up(time_values: pd.Series, event_values: pd.Series) -> float | None:
    """Reverse Kaplan-Meier median follow-up (censoring treated as the event)."""
    censor_status = 1 - event_values.astype(int)
    if censor_status.sum() == 0:
        return None
    return _km_median_time(time_values.to_numpy(dtype=float), censor_status.to_numpy(dtype=float))


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    p_array = np.asarray(p_values, dtype=float)
    if p_array.size == 0:
        return []
    adjusted = np.full(p_array.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p_array)
    if not valid_mask.any():
        return [float(value) for value in adjusted]
    valid_p = p_array[valid_mask]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    n_tests = valid_p.size
    candidate = np.minimum(1.0, sorted_p * n_tests / np.arange(1, valid_p.size + 1, dtype=float))
    monotone = np.minimum.accumulate(candidate[::-1])[::-1]
    valid_adjusted = np.empty_like(valid_p)
    valid_adjusted[order] = monotone
    adjusted[valid_mask] = valid_adjusted
    return [float(value) for value in adjusted]



def _format_percent_value(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _parse_percentile_values(cutoff: str | float | None, *, mode: str) -> list[float]:
    if cutoff is None:
        raise ValueError("Enter percentile value(s) before creating a grouped column.")
    if isinstance(cutoff, (int, float, np.integer, np.floating)):
        tokens = [str(float(cutoff))]
    else:
        tokens = [token.strip() for token in str(cutoff).split(",") if token.strip()]
    if not tokens:
        raise ValueError("Enter percentile value(s) before creating a grouped column.")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Percentile values must be numeric, for example 25 or 25,25.") from exc
    if any((not math.isfinite(value)) or value <= 0 or value >= 100 for value in values):
        raise ValueError("Percentile values must be greater than 0 and less than 100.")
    if all(value < 1.0 for value in values):
        # 0.25 almost always means "25%", but would select the top 0.25%.
        raise ValueError(
            "Percentile values are percentages (for example 25 for the top 25%); values below 1 look like fractions."
        )
    if mode == "percentile_split":
        if len(values) not in {1, 2}:
            raise ValueError("Percentile split accepts one value (25) or two values (25,25).")
        if len(values) == 2 and sum(values) >= 100:
            raise ValueError("Two percentile values must sum to less than 100 so a middle group remains.")
    elif mode == "extreme_split":
        if len(values) != 1:
            raise ValueError("Extreme split accepts one value, for example 25.")
        if values[0] >= 50:
            raise ValueError("Extreme split percentile must be less than 50 so a middle range remains excluded.")
    return values


def _append_realized_group_share_note(
    summary: dict[str, Any],
    counts: Sequence[dict[str, Any]],
) -> None:
    non_missing = [row for row in counts if not row.get("unassigned") and str(row.get("group")) not in _UNASSIGNED_GROUP_LABELS]
    total = sum(int(row.get("n", 0) or 0) for row in non_missing)
    if total <= 0:
        return
    realized = [
        {
            "group": str(row.get("group")),
            "n": int(row.get("n", 0) or 0),
            "fraction": float((int(row.get("n", 0) or 0) / total) * 100.0),
        }
        for row in non_missing
    ]
    summary["realized_group_shares"] = realized
    realized_text = ", ".join(
        f"{item['group']} = {item['fraction']:.1f}%"
        for item in realized
    )
    summary["assignment_rule"] = (
        f"{summary.get('assignment_rule', '')} "
        f"Realized non-missing shares: {realized_text}. Ties at the threshold can shift these from the nominal target."
    ).strip()


def _percentile_threshold_label(percentile: float, direction: str) -> str:
    value = _format_percent_value(percentile)
    if direction == "above":
        return f"At/above {value}th percentile threshold"
    if direction == "below":
        return f"At/below {value}th percentile threshold"
    if direction == "strict_above":
        return f"Above {value}th percentile threshold"
    if direction == "strict_below":
        return f"Below {value}th percentile threshold"
    raise ValueError(f"Unsupported percentile-threshold direction: {direction}")


def _quantile_split(
    numeric_series: pd.Series,
    source_column: str,
    *,
    n_bins: int,
    prefix: str,
    method: str,
) -> tuple[pd.Series, dict[str, Any]]:
    try:
        split_raw, bin_edges = pd.qcut(numeric_series, n_bins, retbins=True, duplicates="drop")
    except ValueError as exc:
        raise ValueError(f"{source_column} cannot be split with {method}: {exc}") from exc
    if not hasattr(split_raw, "cat"):
        raise ValueError(f"{source_column} cannot be split with {method}.")
    n_groups = int(len(split_raw.cat.categories))
    if n_groups < n_bins:
        # Tied values collapse quantile bins; silently renumbering the
        # survivors (e.g. calling the top third "T2") is misleading.
        raise ValueError(
            f"{source_column} does not have enough unique values for a {n_bins}-group {method}: "
            f"tied values leave only {n_groups} distinct quantile bin(s). Use a median or percentile split, "
            "or a variable with more distinct values."
        )
    labels = [f"{prefix}{idx}" for idx in range(1, n_groups + 1)]
    codes = split_raw.cat.codes.to_numpy()
    label_values = np.array(labels, dtype=object)
    mapped = np.where(codes >= 0, label_values[codes], pd.NA)
    split = pd.Series(mapped, index=numeric_series.index, dtype="string")
    # Exact bin edges (the interval labels pandas displays are rounded).
    cutoffs = [float(edge) for edge in np.asarray(bin_edges, dtype=float)[1:-1]]
    return split, {
        "method": method,
        "cutoffs": cutoffs,
        "n_groups": n_groups,
        "assignment_rule": (
            f"{source_column} is cut at the exact quantile edges {', '.join(f'{value:.6g}' for value in cutoffs)} "
            f"(right-closed bins) into {', '.join(labels)}."
        ),
    }


def _median_split(
    numeric_series: pd.Series,
    usable: pd.Series,
    source_column: str,
    *,
    lower_label: str,
    upper_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    split_point = float(usable.median())
    labels = np.where(numeric_series <= split_point, lower_label, upper_label)
    return labels, {
        "method": "median_split",
        "cutoff": split_point,
        "assignment_rule": f"{source_column} <= median ({split_point:.6g}) -> {lower_label}, else -> {upper_label}.",
    }


def _percentile_split(
    numeric_series: pd.Series,
    usable: pd.Series,
    source_column: str,
    *,
    cutoff: str | float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    method = "percentile_split"
    percentiles = _parse_percentile_values(cutoff, mode=method)
    cutoff_spec = ",".join(_format_percent_value(value) for value in percentiles)
    if len(percentiles) == 1:
        top_percent = percentiles[0]
        split_point = float(usable.quantile(1 - (top_percent / 100.0)))
        threshold_percent = 100.0 - top_percent
        rest_label = "Rest"
        # Match median_split at 50th percentile so ties at the threshold stay in the lower/rest group.
        if math.isclose(top_percent, 50.0):
            top_label = _percentile_threshold_label(threshold_percent, "strict_above")
            rest_label = _percentile_threshold_label(threshold_percent, "below")
            labels = np.where(numeric_series <= split_point, rest_label, top_label)
            assignment_rule = (
                f"{source_column} > percentile threshold ({split_point:.3f}) -> {top_label}, else -> {rest_label}"
            )
        else:
            top_label = _percentile_threshold_label(threshold_percent, "above")
            labels = np.where(numeric_series >= split_point, top_label, rest_label)
            assignment_rule = (
                f"{source_column} >= percentile threshold ({split_point:.3f}) -> {top_label}, else -> {rest_label}"
            )
        return labels, {
            "method": method,
            "cutoff_spec": cutoff_spec,
            "percentiles": percentiles,
            "cutoffs": [split_point],
            "n_groups": 2,
            "assignment_rule": assignment_rule,
        }

    bottom_percent, top_percent = percentiles
    low_threshold = float(usable.quantile(bottom_percent / 100.0))
    high_threshold = float(usable.quantile(1 - (top_percent / 100.0)))
    if not low_threshold < high_threshold:
        raise ValueError("Percentile split thresholds overlap. Choose less aggressive percentiles or a variable with more distinct values.")
    bottom_label = _percentile_threshold_label(bottom_percent, "below")
    middle_label = "Between percentile thresholds"
    top_label = _percentile_threshold_label(100.0 - top_percent, "above")
    labels = np.where(
        numeric_series <= low_threshold,
        bottom_label,
        np.where(numeric_series >= high_threshold, top_label, middle_label),
    )
    return labels, {
        "method": method,
        "cutoff_spec": cutoff_spec,
        "percentiles": percentiles,
        "cutoffs": [low_threshold, high_threshold],
        "n_groups": 3,
        "assignment_rule": (
            f"{source_column} <= lower percentile threshold ({low_threshold:.3f}) -> {bottom_label}; "
            f"{source_column} >= upper percentile threshold ({high_threshold:.3f}) -> {top_label}; "
            f"else -> {middle_label}"
        ),
    }


def _extreme_split(
    numeric_series: pd.Series,
    usable: pd.Series,
    source_column: str,
    *,
    cutoff: str | float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    method = "extreme_split"
    percentiles = _parse_percentile_values(cutoff, mode=method)
    tail_percent = percentiles[0]
    low_threshold = float(usable.quantile(tail_percent / 100.0))
    high_threshold = float(usable.quantile(1 - (tail_percent / 100.0)))
    if not low_threshold < high_threshold:
        raise ValueError("Extreme split thresholds overlap. Choose a smaller percentile or a variable with more distinct values.")
    bottom_label = _percentile_threshold_label(tail_percent, "below")
    top_label = _percentile_threshold_label(100.0 - tail_percent, "above")
    labels = np.full(len(numeric_series), pd.NA, dtype=object)
    low_mask = (numeric_series <= low_threshold).fillna(False).to_numpy()
    high_mask = (numeric_series >= high_threshold).fillna(False).to_numpy()
    labels[low_mask] = bottom_label
    labels[high_mask] = top_label
    excluded_middle_count = int((numeric_series.notna() & ~pd.Series(low_mask | high_mask, index=numeric_series.index)).sum())
    return labels, {
        "method": method,
        "cutoff_spec": _format_percent_value(tail_percent),
        "percentiles": percentiles,
        "cutoffs": [low_threshold, high_threshold],
        "n_groups": 2,
        "excluded_count": excluded_middle_count,
        "assignment_rule": (
            f"{source_column} <= lower percentile threshold ({low_threshold:.3f}) -> {bottom_label}; "
            f"{source_column} >= upper percentile threshold ({high_threshold:.3f}) -> {top_label}; "
            "else -> excluded middle range"
        ),
    }


def _optimal_cutpoint_split(
    df: pd.DataFrame,
    numeric_series: pd.Series,
    source_column: str,
    *,
    time_column: str | None,
    event_column: str | None,
    event_positive_value: Any,
    min_group_fraction: float,
    lower_label: str,
    upper_label: str,
    permutation_iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not time_column or not event_column:
        raise ValueError("optimal_cutpoint requires time_column and event_column.")
    from survival_toolkit.ml_models import find_optimal_cutpoint

    result = find_optimal_cutpoint(
        df,
        time_column=time_column,
        event_column=event_column,
        variable=source_column,
        event_positive_value=event_positive_value,
        min_group_fraction=min_group_fraction,
        lower_label=lower_label,
        upper_label=upper_label,
        permutation_iterations=permutation_iterations,
        random_seed=random_seed,
    )
    split_point = result["optimal_cutpoint"]
    label_above = result["label_above_cutpoint"]
    label_below = result["label_below_cutpoint"]
    # The chosen cutpoint is a fixed rule on the marker, so it labels every row with a
    # usable marker value, as the median/quantile/percentile splits do. Rows whose
    # survival outcome is missing did not help choose the cutpoint but still get a group.
    labels = np.where(numeric_series > split_point, label_above, label_below)
    n_scanned = int(result["n_above_cutpoint"]) + int(result["n_below_cutpoint"])
    return labels, {
        "method": "optimal_cutpoint",
        "cutoff": split_point,
        "label_above_cutpoint": label_above,
        "label_below_cutpoint": label_below,
        "assignment_rule": f"{source_column} > cutoff -> {label_above}, else -> {label_below}",
        "n_rows_scanned": n_scanned,
        "n_rows_labelled_without_outcome": max(int(numeric_series.notna().sum()) - n_scanned, 0),
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "p_value_label": result.get("p_value_label"),
        "raw_p_value": result.get("raw_p_value"),
        "selection_adjusted_p_value": result.get("selection_adjusted_p_value"),
        "selection_adjustment": result.get("selection_adjustment"),
        "min_group_fraction": float(min_group_fraction),
        "permutation_iterations": int(permutation_iterations),
        "random_seed": int(random_seed),
        "scan_data": result["scan_data"],
        "candidate_grid": result.get("candidate_grid"),
    }


def _derived_group_column_name(df: pd.DataFrame, new_column_name: str | None, default_name: str) -> str:
    requested_name = (new_column_name or "").strip() or None
    if requested_name is None:
        return _next_available_column_name(df.columns, default_name)
    if requested_name in df.columns:
        raise ValueError(
            f'"{requested_name}" already exists. Choose a new derived-column name instead of overwriting an existing field.'
        )
    return requested_name


_MISSING_GROUP_LABEL = "Missing"
_EXCLUDED_GROUP_LABEL = "Excluded (middle range)"
# Count entries for rows without a group; group labels may not reuse these names.
_UNASSIGNED_GROUP_LABELS = frozenset({_MISSING_GROUP_LABEL, _EXCLUDED_GROUP_LABEL})


def _validate_group_labels(lower_label: Any, upper_label: Any) -> None:
    labels = [str(lower_label if lower_label is not None else "").strip(), str(upper_label if upper_label is not None else "").strip()]
    if not all(labels):
        raise ValueError("Group labels must not be empty.")
    if labels[0] == labels[1]:
        raise ValueError(f'The lower and upper group labels must differ (both are "{labels[0]}").')
    reserved = {label.lower() for label in _UNASSIGNED_GROUP_LABELS}
    for label in labels:
        if label.lower() in reserved:
            raise ValueError(
                f'"{label}" is reserved for rows without a group. Choose another group label.'
            )


def _group_counts(labels: pd.Series, *, excluded_mask: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Rows per group, then separate entries for rows without a group.

    Rows whose source value is missing are counted as "Missing" and rows left out by design
    (the middle range of an extreme split) as "Excluded (middle range)"; both entries carry
    ``"unassigned": True`` and never merge with a group of the same name.
    """
    present = labels.dropna()
    counts: list[dict[str, Any]] = [
        {"group": str(group), "n": int(n)}
        for group, n in present.astype(str).value_counts().sort_index().items()
    ]
    unassigned = labels.isna().to_numpy(dtype=bool)
    excluded = unassigned & excluded_mask if excluded_mask is not None else np.zeros(len(labels), dtype=bool)
    missing = unassigned & ~excluded
    if excluded.any():
        counts.append({"group": _EXCLUDED_GROUP_LABEL, "n": int(excluded.sum()), "unassigned": True})
    if missing.any():
        counts.append({"group": _MISSING_GROUP_LABEL, "n": int(missing.sum()), "unassigned": True})
    return counts


@user_input_boundary
def derive_group_column(
    df: pd.DataFrame,
    source_column: str,
    method: str,
    new_column_name: str | None = None,
    cutoff: str | float | None = None,
    lower_label: str = "Low",
    upper_label: str = "High",
    time_column: str | None = None,
    event_column: str | None = None,
    event_positive_value: Any = None,
    min_group_fraction: float = 0.1,
    permutation_iterations: int = 500,
    random_seed: int = 20260311,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    _require_dataframe_columns(df, [source_column])
    if source_column in _survival_outcome_like_columns(df):
        raise ValueError(
            f'"{source_column}" looks like a survival endpoint column. '
            "Do not derive groups from survival time or event indicators."
        )
    if method in {"median_split", "optimal_cutpoint"}:
        _validate_group_labels(lower_label, upper_label)
    raw_source = df[source_column]
    coerced_source = pd.to_numeric(raw_source, errors="coerce")
    # Plain float64 (nullable Int64/Float64/boolean NA -> NaN), so comparisons stay boolean.
    numeric_series = pd.Series(coerced_source.to_numpy(dtype=float, na_value=np.nan), index=raw_source.index)
    non_finite_count = int(np.isinf(numeric_series.to_numpy()).sum())
    numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    non_numeric_count = int((raw_source.notna().to_numpy(dtype=bool) & coerced_source.isna().to_numpy(dtype=bool)).sum())
    usable = numeric_series.dropna()
    if usable.empty:
        raise ValueError(f"{source_column} does not contain numeric values that can be split.")

    if method == "optimal_cutpoint":
        labels, summary = _optimal_cutpoint_split(
            df,
            numeric_series,
            source_column,
            time_column=time_column,
            event_column=event_column,
            event_positive_value=event_positive_value,
            min_group_fraction=min_group_fraction,
            lower_label=lower_label,
            upper_label=upper_label,
            permutation_iterations=permutation_iterations,
            random_seed=random_seed,
        )
    elif method == "median_split":
        labels, summary = _median_split(
            numeric_series, usable, source_column, lower_label=lower_label, upper_label=upper_label
        )
    elif method == "tertile_split":
        labels, summary = _quantile_split(numeric_series, source_column, n_bins=3, prefix="T", method=method)
    elif method == "quartile_split":
        labels, summary = _quantile_split(numeric_series, source_column, n_bins=4, prefix="Q", method=method)
    elif method == "percentile_split":
        labels, summary = _percentile_split(numeric_series, usable, source_column, cutoff=cutoff)
    elif method == "extreme_split":
        labels, summary = _extreme_split(numeric_series, usable, source_column, cutoff=cutoff)
    else:
        raise ValueError(f"Unsupported derive-group method: {method}")
    outcome_informed = method == "optimal_cutpoint"

    label_series = pd.Series(labels, index=df.index, dtype="string")
    label_series.loc[numeric_series.isna()] = pd.NA
    observed_groups = int(label_series.dropna().nunique())
    if method == "median_split" and observed_groups < 2:
        raise ValueError(
            f"Median split of {source_column} produced a single group (too many values tied at the median). "
            "Choose another split or variable."
        )
    input_notes: list[str] = []
    if non_numeric_count:
        input_notes.append(f"{non_numeric_count} non-numeric value(s) in {source_column} were treated as missing.")
    if non_finite_count:
        input_notes.append(f"{non_finite_count} infinite value(s) in {source_column} were treated as missing.")
    if input_notes:
        summary["input_notes"] = input_notes
    if method == "percentile_split" and summary.get("n_groups") == 3 and observed_groups < 3:
        raise ValueError("Percentile split did not produce three distinct groups. Choose a less aggressive percentile setting or another variable.")
    if method in {"percentile_split", "extreme_split"} and observed_groups < 2:
        raise ValueError("Selected percentile thresholds did not produce at least two non-empty groups.")

    column_name = _derived_group_column_name(df, new_column_name, f"{source_column}__{method}")
    updated = df.copy()
    updated[column_name] = label_series

    # Rows with a usable source value but no label were left out by the method (the middle
    # range of an extreme split); they are counted apart from rows with a missing source value.
    excluded_by_method = (numeric_series.notna() & label_series.isna()).to_numpy(dtype=bool)
    counts = _group_counts(updated[column_name], excluded_mask=excluded_by_method)
    summary["missing_count"] = int(numeric_series.isna().sum())
    summary["column_name"] = column_name
    summary["outcome_informed"] = outcome_informed
    summary["counts"] = counts
    if method in {"percentile_split", "extreme_split", "median_split", "tertile_split", "quartile_split"}:
        _append_realized_group_share_note(summary, counts)
    summary["recipe"] = {
        "source_column": source_column,
        "column_name": column_name,
        "method": method,
        "cutoff": summary.get("cutoff"),
        "cutoff_spec": summary.get("cutoff_spec"),
        "cutoffs": list(summary.get("cutoffs", [])),
        "percentiles": list(summary.get("percentiles", [])),
        "lower_label": lower_label,
        "upper_label": upper_label,
        "time_column": time_column,
        "event_column": event_column,
        "event_positive_value": event_positive_value,
        "min_group_fraction": summary.get("min_group_fraction"),
        "permutation_iterations": summary.get("permutation_iterations"),
        "random_seed": summary.get("random_seed"),
        "outcome_informed": outcome_informed,
    }
    return updated, column_name, summary


_SIGNATURE_MAX_CATEGORICAL_LEVELS = 8


def _build_candidate_indicators(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
    min_group_size: int,
    skipped: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Binary rules for each candidate: numeric quartile thresholds or one level vs the rest.

    When ``skipped`` is given, every candidate that yields no rule is appended to it with the
    reason, so the search can report it instead of dropping the column silently.
    """
    indicators: list[dict[str, Any]] = []
    for column in candidate_columns:
        series = frame[column]
        n_before = len(indicators)
        reason = ""
        # Booleans are categorical flags (quantiles of True/False are undefined).
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric = pd.Series(
                pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan), index=series.index
            )
            quantile_candidates = (0.25, 0.5, 0.75)
            seen_thresholds: set[float] = set()
            for quantile in quantile_candidates:
                cutoff = float(numeric.quantile(quantile))
                if not np.isfinite(cutoff):
                    continue
                threshold_key = round(cutoff, 10)
                if threshold_key in seen_thresholds:
                    continue
                seen_thresholds.add(threshold_key)
                mask = numeric > cutoff
                n_positive = int(mask.sum())
                if min_group_size <= n_positive <= (len(mask) - min_group_size):
                    indicators.append(
                        {
                            "column": column,
                            "kind": "numeric_gt",
                            "threshold": cutoff,
                            "quantile": float(quantile),
                            "label": f"{column} > Q{int(quantile * 100)}({cutoff:.3f})",
                        }
                    )
            reason = (
                f"no quartile threshold leaves at least {int(min_group_size)} rows on both sides"
                if numeric.notna().any()
                else "no numeric values"
            )
        else:
            values = series.astype("string")
            counts = values.value_counts(dropna=True)
            if counts.empty:
                reason = "no non-missing values"
            elif int(counts.shape[0]) > _SIGNATURE_MAX_CATEGORICAL_LEVELS:
                reason = (
                    f"{int(counts.shape[0])} levels; categorical candidates may have at most "
                    f"{_SIGNATURE_MAX_CATEGORICAL_LEVELS}"
                )
            elif int(counts.shape[0]) < 2:
                reason = "only one observed level"
            else:
                # Most common level first; ties follow the reference order, so neither the
                # reference nor the rule order depends on the order of the rows.
                reference_order = {
                    level: position
                    for position, level in enumerate(_ordered_reference_categories(counts.index.tolist(), str(column)))
                }
                levels = sorted((str(level) for level in counts.index), key=lambda level: (-int(counts[level]), reference_order[level]))
                reference = levels[0]
                # With two levels "most common level vs rest" is the same split as the other level;
                # with more it is a distinct rule (for example wild type vs any mutation).
                tested_levels = levels[1:] if len(levels) == 2 else levels
                for level_text in tested_levels:
                    mask = values == level_text
                    n_positive = int(mask.sum())
                    if min_group_size <= n_positive <= (len(mask) - min_group_size):
                        indicators.append(
                            {
                                "column": column,
                                "kind": "categorical_level",
                                "level": level_text,
                                "reference": reference if level_text != reference else levels[1],
                                "label": f'{column} == "{level_text}"',
                            }
                        )
                reason = (
                    f"no level{' other than the most common one' if len(levels) == 2 else ''} has between "
                    f"{int(min_group_size)} and {len(values) - int(min_group_size)} rows"
                )
        if skipped is not None and len(indicators) == n_before:
            skipped.append({"column": str(column), "reason": reason})
    return indicators


def _indicator_definition(indicator: dict[str, Any]) -> dict[str, Any]:
    """A JSON-ready copy of one signature rule with its exact threshold or level."""
    definition: dict[str, Any] = {
        "column": str(indicator["column"]),
        "kind": str(indicator["kind"]),
        "label": str(indicator.get("label", "")),
    }
    if indicator["kind"] == "numeric_gt":
        definition["threshold"] = float(indicator["threshold"])
        definition["quantile"] = float(indicator["quantile"])
    elif indicator["kind"] == "categorical_level":
        definition["level"] = str(indicator["level"])
        definition["reference"] = str(indicator["reference"])
    return definition


def _evaluate_indicator(df: pd.DataFrame, indicator: dict[str, Any]) -> pd.Series:
    column = indicator["column"]
    if indicator["kind"] == "numeric_gt":
        numeric = pd.to_numeric(df[column], errors="coerce")
        out = pd.Series(pd.NA, index=df.index, dtype="boolean")
        valid = numeric.notna()
        out.loc[valid] = (numeric.loc[valid] > float(indicator["threshold"])).astype(bool)
        return out
    if indicator["kind"] == "categorical_level":
        values = df[column].astype("string")
        out = pd.Series(pd.NA, index=df.index, dtype="boolean")
        valid = values.notna()
        out.loc[valid] = (values.loc[valid] == str(indicator["level"])).astype(bool)
        return out
    raise ValueError(f"Unsupported indicator kind: {indicator['kind']}")


def _signature_mask(
    frame: pd.DataFrame, combo: Sequence[dict[str, Any]], operator: str = "AND"
) -> np.ndarray:
    if not combo:
        return np.zeros(frame.shape[0], dtype=bool)
    if operator not in {"AND", "OR"}:
        raise ValueError(f"Unsupported signature operator: {operator}")

    values = [
        _evaluate_indicator(frame, indicator).fillna(False).to_numpy(dtype=bool)
        for indicator in combo
    ]
    if operator == "AND":
        mask = np.logical_and.reduce(values)
    else:
        mask = np.logical_or.reduce(values)
    return np.asarray(mask, dtype=bool)


def _signature_hazard_ratio(
    times: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> tuple[float | None, float | None, float | None]:
    """Signature+ vs signature- Cox hazard ratio (Efron ties) and its (1 - alpha) confidence interval.

    A fit that does not converge (for example a hazard ratio that runs to 0 or infinity because
    one side has every event before the other side's) gives ``(None, None, None)``.
    """
    model = PHReg(
        np.asarray(times, dtype=float),
        np.asarray(mask, dtype=bool).astype(float)[:, np.newaxis],
        status=np.asarray(events, dtype=int),
        ties="efron",
    )
    results, converged = fit_phreg(model)
    if not converged:
        return None, None, None
    conf_int = np.asarray(results.conf_int(alpha=alpha), dtype=float)
    return (
        _safe_exp_or_none(np.asarray(results.params, dtype=float)[0]),
        _safe_exp_or_none(conf_int[0, 0]),
        _safe_exp_or_none(conf_int[0, 1]),
    )


def _signature_cox_metrics(
    times: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    """Signature+ vs signature- hazard ratio with a (1 - alpha) confidence interval."""
    hazard_ratio, ci_lower, ci_upper = _signature_hazard_ratio(times, events, mask, alpha)
    return {
        "Hazard ratio (signature+ vs -)": hazard_ratio,
        "HR CI lower": ci_lower,
        "HR CI upper": ci_upper,
    }


def _signature_logrank(times: np.ndarray, events: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Log-rank chi-square of signature+ vs signature- and its upper-tail p-value (1 df).

    The p-value comes from ``chi2.sf``: ``survdiff`` returns ``1 - chi2.cdf``, which rounds to
    0 once the chi-square passes about 75.
    """
    chisq, _ = survdiff(times, events, np.where(mask, "Signature+", "Signature-"))
    chisq = float(chisq)
    return chisq, float(stats.chi2.sf(chisq, 1))


def _stability_score(row: dict[str, Any], significance_level: float = 0.05) -> float:
    # Composite score balancing significance, robustness, effect size, and parsimony.
    bh_p = max(float(row["BH adjusted p"]), 1e-12)
    significance_score = min(-math.log10(bh_p), 10.0)
    effect = _hazard_ratio_effect_size(row.get("Hazard ratio (signature+ vs -)"))
    support = row["Bootstrap support (p<alpha)"]
    support_value = float(support) if support is not None else 0.0
    direction_consistency = row.get("Bootstrap HR direction consistency")
    direction_consistency_value = (
        float(direction_consistency) if direction_consistency is not None else 0.0
    )
    validation_support = row.get("Validation support (p<alpha)")
    validation_support_value = (
        float(validation_support) if validation_support is not None else 0.0
    )
    permutation_p = row["Permutation p"]
    permutation_penalty = 0.0
    if permutation_p is not None:
        # Only the part of the permutation p-value above the chosen alpha is penalized.
        permutation_penalty = max(float(permutation_p) - float(significance_level), 0.0)
    complexity_penalty = 0.18 * max(int(row["Rule count"]) - 1, 0)
    return float(
        significance_score
        + (0.35 * effect)
        + (0.85 * support_value)
        + (0.55 * direction_consistency_value)
        + (1.05 * validation_support_value)
        - complexity_penalty
        - (0.65 * permutation_penalty)
    )


def _signature_is_significant(
    row: dict[str, Any],
    significance_level: float,
    require_permutation: bool,
    require_validation: bool,
    min_validation_support: float,
    require_bootstrap_consistency: bool,
    min_bootstrap_consistency: float,
) -> bool:
    bh_p = float(row["BH adjusted p"])
    if bh_p > significance_level:
        return False

    permutation_p = row.get("Permutation p")
    if require_permutation and permutation_p is None:
        return False
    if permutation_p is not None and float(permutation_p) > significance_level:
        return False

    ci_low = _safe_float(row.get("HR CI lower"))
    ci_high = _safe_float(row.get("HR CI upper"))
    if ci_low is None or ci_high is None:
        return False
    if ci_low <= 1.0 <= ci_high:
        return False
    if require_bootstrap_consistency:
        direction_consistency = _safe_float(row.get("Bootstrap HR direction consistency"))
        if direction_consistency is None:
            return False
        if direction_consistency < min_bootstrap_consistency:
            return False
    if require_validation:
        validation_support = _safe_float(row.get("Validation support (p<alpha)"))
        if validation_support is None:
            return False
        if validation_support < min_validation_support:
            return False
    return True


# A bootstrap resample or replication fold is scored when each side of the signature has at
# least this many rows and events: the log-rank test and the Cox hazard ratio need an event on
# each side, and a side of a single row cannot vary. The discovery group-size and event rules
# are not re-applied to the smaller draws (a rule at the minimum group size would then fail about
# half of them by chance alone); skipped draws count against support instead.
_RESAMPLE_MIN_ROWS_PER_GROUP = 2
_RESAMPLE_MIN_EVENTS_PER_GROUP = 1


def _resample_is_estimable(mask_values: np.ndarray, events: np.ndarray) -> bool:
    n_positive = int(mask_values.sum())
    n_negative = int(mask_values.shape[0]) - n_positive
    if min(n_positive, n_negative) < _RESAMPLE_MIN_ROWS_PER_GROUP:
        return False
    events_positive = int(events[mask_values].sum())
    events_negative = int(events[~mask_values].sum())
    return min(events_positive, events_negative) >= _RESAMPLE_MIN_EVENTS_PER_GROUP


def _bootstrap_signature_metrics(
    frame: pd.DataFrame,
    time_column: str,
    event_column: str,
    combo: Sequence[dict[str, Any]],
    combo_operator: str,
    min_group_size: int,
    n_iterations: int,
    sample_fraction: float,
    random_seed: int,
    significance_level: float,
    observed_hazard_ratio: float | None = None,
) -> dict[str, float | int | None]:
    """Bootstrap support for one signature.

    Each resample is scored when both sides of the signature have at least
    ``_RESAMPLE_MIN_ROWS_PER_GROUP`` rows and ``_RESAMPLE_MIN_EVENTS_PER_GROUP`` event (the
    minimum the log-rank test and the Cox hazard ratio need); the discovery group-size and
    event rules are not re-applied to resamples. Support and direction consistency are shares
    of all requested resamples: a resample that had to be skipped (too few rows or events on
    one side, or a failed test) counts as a failure, so frequent skipping cannot inflate the
    support of an unstable rule, and the number skipped is reported. A resample whose Cox fit
    does not converge adds no hazard ratio, so it counts against direction consistency.
    """
    if n_iterations <= 0:
        return {
            "Bootstrap support (p<alpha)": None,
            "Bootstrap median HR": None,
            "Bootstrap median p": None,
            "Bootstrap HR direction consistency": None,
            "Bootstrap valid resamples": 0,
            "Bootstrap skipped resamples": 0,
        }

    n_obs = int(frame.shape[0])
    sample_size = int(math.ceil(n_obs * sample_fraction))
    sample_size = min(max(sample_size, min_group_size * 2), n_obs)
    rng = np.random.default_rng(random_seed)
    significant_count = 0
    valid_resamples = 0
    skipped_resamples = 0
    p_values: list[float] = []
    hazard_ratios: list[float] = []
    # The rules are row-wise, so a resampled row's membership is its cohort row's membership:
    # the mask is evaluated once and indexed per resample.
    cohort_mask = np.asarray(_signature_mask(frame, combo, operator=combo_operator), dtype=bool)
    cohort_events = frame[event_column].to_numpy(dtype=int)
    cohort_times = frame[time_column].to_numpy(dtype=float)

    for _ in range(n_iterations):
        raise_if_cancelled()
        sampled_idx = rng.integers(0, n_obs, size=sample_size)
        mask_values = cohort_mask[sampled_idx]
        events = cohort_events[sampled_idx]
        if not _resample_is_estimable(mask_values, events):
            skipped_resamples += 1
            continue
        times = cohort_times[sampled_idx]

        try:
            _, p_float = _signature_logrank(times, events, mask_values)
            hr, _, _ = _signature_hazard_ratio(times, events, mask_values, significance_level)
        except MemoryError:
            raise
        except Exception as exc:
            if must_propagate(exc):
                raise
            skipped_resamples += 1
            continue

        valid_resamples += 1
        p_values.append(p_float)
        if hr is not None:
            hazard_ratios.append(hr)
        if p_float <= significance_level:
            significant_count += 1

    if valid_resamples == 0:
        return {
            "Bootstrap support (p<alpha)": None,
            "Bootstrap median HR": None,
            "Bootstrap median p": None,
            "Bootstrap HR direction consistency": None,
            "Bootstrap valid resamples": 0,
            "Bootstrap skipped resamples": int(skipped_resamples),
        }

    requested = int(n_iterations)
    direction_consistency = None
    if hazard_ratios:
        hr_array = np.asarray(hazard_ratios, dtype=float)
        if observed_hazard_ratio is not None and math.isfinite(float(observed_hazard_ratio)):
            # Share of requested resamples that keep the observed effect direction.
            observed_harmful = float(observed_hazard_ratio) >= 1.0
            direction_consistency = float(np.sum((hr_array >= 1.0) == observed_harmful) / requested)
        else:
            direction_consistency = float(
                max(float(np.sum(hr_array >= 1.0)), float(np.sum(hr_array < 1.0))) / requested
            )

    return {
        "Bootstrap support (p<alpha)": float(significant_count / requested),
        "Bootstrap median HR": float(np.median(hazard_ratios)) if hazard_ratios else None,
        "Bootstrap median p": float(np.median(p_values)),
        "Bootstrap HR direction consistency": direction_consistency,
        "Bootstrap valid resamples": int(valid_resamples),
        "Bootstrap skipped resamples": int(skipped_resamples),
    }


# Upper bound on rows x groupings materialised at once by the vectorized log-rank helpers
# (each block creates a few float64 arrays of this many cells).
_LOGRANK_BLOCK_CELLS = 2_000_000


class _LogrankOutcome(NamedTuple):
    """The outcome side of a two-group log-rank test, sorted by time once."""

    order: np.ndarray
    sorted_events: np.ndarray
    first_index: np.ndarray
    has_event: np.ndarray
    n_at_risk: np.ndarray
    deaths: np.ndarray


def _logrank_outcome(times: np.ndarray, events: np.ndarray) -> _LogrankOutcome:
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=float)
    order = np.argsort(times_arr, kind="mergesort")
    sorted_times = times_arr[order]
    sorted_events = events_arr[order]
    _, first_index = np.unique(sorted_times, return_index=True)
    at_risk_total = sorted_times.size - first_index
    deaths_total = np.add.reduceat(sorted_events, first_index)
    has_event = deaths_total > 0
    return _LogrankOutcome(
        order=order,
        sorted_events=sorted_events,
        first_index=first_index,
        has_event=has_event,
        n_at_risk=at_risk_total[has_event].astype(float)[:, np.newaxis],
        deaths=deaths_total[has_event].astype(float)[:, np.newaxis],
    )


def _logrank_chisq_sorted(outcome: _LogrankOutcome, sorted_masks: np.ndarray) -> np.ndarray:
    """Chi-square per column of masks whose rows are already in ``outcome.order``."""
    mask_arr = np.asarray(sorted_masks, dtype=float)
    at_risk_group = np.cumsum(mask_arr[::-1], axis=0)[::-1][outcome.first_index]
    deaths_group = np.add.reduceat(mask_arr * outcome.sorted_events[:, np.newaxis], outcome.first_index, axis=0)
    n = outcome.n_at_risk
    d = outcome.deaths
    n1 = at_risk_group[outcome.has_event]
    o1 = deaths_group[outcome.has_event]
    share = n1 / n
    expected = d * share
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = d * share * (1.0 - share) * np.where(n > 1.0, (n - d) / (n - 1.0), 0.0)
    numerator = np.sum(o1 - expected, axis=0) ** 2
    denominator = np.sum(variance, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.where(denominator > 0.0, numerator / denominator, 0.0)
    return np.asarray(statistic, dtype=float)


def _vectorized_logrank_chisq(
    times: np.ndarray,
    events: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Two-group log-rank chi-square for many binary groupings at once.

    ``masks`` is an ``(n, k)`` boolean matrix; column ``j`` defines group 1 of
    the ``j``-th comparison. Matches ``statsmodels.duration.survfunc.survdiff``
    (unweighted log-rank with the hypergeometric variance). Columns are processed in
    blocks so memory stays bounded for large ``n * k``.
    """
    mask_arr = np.asarray(masks)
    if mask_arr.ndim == 1:
        mask_arr = mask_arr[:, np.newaxis]
    outcome = _logrank_outcome(times, events)
    n_rows, n_columns = mask_arr.shape
    block = max(1, _LOGRANK_BLOCK_CELLS // max(int(n_rows), 1))
    sorted_masks = mask_arr[outcome.order]
    if n_columns <= block:
        return _logrank_chisq_sorted(outcome, sorted_masks)
    return np.concatenate(
        [_logrank_chisq_sorted(outcome, sorted_masks[:, start : start + block]) for start in range(0, n_columns, block)]
    )


class ThresholdLogrankScan:
    """Two-group log-rank chi-square for every split "marker > c" over many cutpoints c.

    Equivalent to calling ``statsmodels.duration.survfunc.survdiff`` once per cutpoint
    (unweighted log-rank, hypergeometric variance), but the outcome structure is built
    once and each marker ordering is scored with histograms and cumulative sums, so an
    exhaustive cutpoint scan and its permutation null are cheap enough to run together.
    """

    # Upper bound on event-time x cutpoint cells materialised at once.
    _MAX_BLOCK_CELLS = 2_000_000

    def __init__(self, times: np.ndarray, events: np.ndarray) -> None:
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        event_mask = np.asarray(events, dtype=float).reshape(-1) > 0
        self.n = int(times_arr.size)
        self.event_times = np.unique(times_arr[event_mask])
        n_event_times = int(self.event_times.size)
        # Subject i is at risk at event times 0..risk_index[i] (times >= that event time).
        self._risk_index = np.searchsorted(self.event_times, times_arr, side="right") - 1
        self._event_rows = np.flatnonzero(event_mask)
        self._event_index = np.searchsorted(self.event_times, times_arr[self._event_rows], side="left")
        at_risk_rows = self._risk_index >= 0
        at_risk = np.bincount(self._risk_index[at_risk_rows], minlength=n_event_times)[::-1].cumsum()[::-1]
        deaths = np.bincount(self._event_index, minlength=n_event_times)
        self._n_at_risk = at_risk.astype(float)
        self._deaths = deaths.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._variance_factor = np.where(
                self._n_at_risk > 1.0,
                self._deaths * (self._n_at_risk - self._deaths) / (self._n_at_risk - 1.0),
                0.0,
            )
        self.total_events = int(self._event_rows.size)

    def scan(self, marker: np.ndarray, cutpoints: np.ndarray) -> dict[str, np.ndarray]:
        """Statistics, group sizes, and event counts of "marker > c" for each cutpoint.

        ``cutpoints`` must be sorted ascending. Returns arrays of length ``len(cutpoints)``:
        ``statistic`` (chi-square, 0 where the variance is zero), ``n_high`` and
        ``events_high`` (rows and events above the cutpoint).
        """
        marker_arr = np.asarray(marker, dtype=float).reshape(-1)
        cut_arr = np.asarray(cutpoints, dtype=float).reshape(-1)
        n_cut = int(cut_arr.size)
        # rank_i = number of cutpoints strictly below marker_i, so marker_i > c_j iff j < rank_i.
        rank = np.searchsorted(cut_arr, marker_arr, side="left")
        rank_counts = np.bincount(rank, minlength=n_cut + 1)
        n_high = rank_counts[::-1].cumsum()[::-1][1:]
        event_rank_counts = np.bincount(rank[self._event_rows], minlength=n_cut + 1)
        events_high = event_rank_counts[::-1].cumsum()[::-1][1:]
        statistic = np.zeros(n_cut, dtype=float)
        n_event_times = int(self.event_times.size)
        if n_event_times == 0 or n_cut == 0:
            return {"statistic": statistic, "n_high": n_high, "events_high": events_high}

        at_risk_rows = self._risk_index >= 0
        risk_index = self._risk_index[at_risk_rows]
        risk_rank = rank[at_risk_rows]
        event_rank = rank[self._event_rows]
        n_total = self._n_at_risk[:, np.newaxis]
        deaths = self._deaths[:, np.newaxis]
        variance_factor = self._variance_factor[:, np.newaxis]
        block = max(1, min(n_cut, self._MAX_BLOCK_CELLS // max(n_event_times, 1) - 1))
        for start in range(0, n_cut, block):
            stop = min(start + block, n_cut)
            width = stop - start
            # Ranks clipped into block coordinates; a reverse cumulative sum over the rank
            # axis then counts subjects with rank >= start + m at column m.
            local_rank = np.clip(risk_rank - start, 0, width)
            histogram = np.bincount(
                risk_index * (width + 1) + local_rank,
                minlength=n_event_times * (width + 1),
            ).reshape(n_event_times, width + 1)
            at_risk_high = histogram[::-1].cumsum(axis=0)[::-1]
            at_risk_high = at_risk_high[:, ::-1].cumsum(axis=1)[:, ::-1][:, 1:].astype(float)
            local_event_rank = np.clip(event_rank - start, 0, width)
            event_histogram = np.bincount(
                self._event_index * (width + 1) + local_event_rank,
                minlength=n_event_times * (width + 1),
            ).reshape(n_event_times, width + 1)
            deaths_high = event_histogram[:, ::-1].cumsum(axis=1)[:, ::-1][:, 1:].astype(float)
            share = at_risk_high / n_total
            numerator = np.sum(deaths_high - deaths * share, axis=0) ** 2
            denominator = np.sum(variance_factor * share * (1.0 - share), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                statistic[start:stop] = np.where(denominator > 0.0, numerator / denominator, 0.0)
        return {"statistic": statistic, "n_high": n_high, "events_high": events_high}


def _search_adjusted_permutation_p_values(
    times: np.ndarray,
    events: np.ndarray,
    masks: np.ndarray,
    observed_stats: np.ndarray,
    *,
    min_events_per_group: int,
    n_iterations: int,
    random_seed: int,
) -> tuple[np.ndarray | None, int]:
    """Westfall-Young max-statistic permutation p-values over the whole search.

    Each permutation shuffles outcomes against the covariates, re-scores every
    candidate signature in ``masks`` that passes the event rule under the shuffled
    outcome, and records the maximum log-rank statistic. A signature's p-value is the
    share of permutations whose best-of-search statistic reaches its own statistic,
    so it accounts for having picked the best rule from many (FWER control).

    ``masks`` must be the whole search family defined by the covariates alone (every
    combination that meets the group-size rule), not only the combinations that met the
    event rule on the observed outcome: that subset depends on the outcome, and using it
    as the null family makes the p-values anti-conservative.

    The outcome is sorted once and each shuffle reorders the mask rows instead; mask
    columns are scored in blocks so memory stays bounded for large searches.
    """
    if n_iterations <= 0:
        return None, 0
    mask_arr = np.asarray(masks, dtype=bool)
    if mask_arr.ndim == 1:
        mask_arr = mask_arr[:, np.newaxis]
    events_arr = np.asarray(events, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    observed = np.asarray(observed_stats, dtype=float)
    if mask_arr.shape[1] == 0:
        return None, 0
    total_events = float(events_arr.sum())
    outcome = _logrank_outcome(times_arr, events_arr)
    n_rows, n_columns = mask_arr.shape
    block = max(1, _LOGRANK_BLOCK_CELLS // max(int(n_rows), 1))
    positions = np.arange(n_rows)
    inverse = np.empty(n_rows, dtype=np.intp)
    rng = np.random.default_rng(random_seed)
    max_stats: list[float] = []
    for _ in range(int(n_iterations)):
        raise_if_cancelled()
        permutation = rng.permutation(times_arr.size)
        # Mask row i meets outcome row permutation[i]; with the outcome in time order,
        # sorted outcome position j (row order[j]) meets mask row inverse[order[j]].
        inverse[permutation] = positions
        mask_rows = inverse[outcome.order]
        best = -np.inf
        for start in range(0, n_columns, block):
            sorted_block = mask_arr[mask_rows, start : start + block]
            group_events = outcome.sorted_events @ sorted_block
            eligible = (group_events >= min_events_per_group) & ((total_events - group_events) >= min_events_per_group)
            if not np.any(eligible):
                continue
            statistics = _logrank_chisq_sorted(outcome, sorted_block[:, eligible])
            finite = statistics[np.isfinite(statistics)]
            if finite.size:
                best = max(best, float(np.max(finite)))
        if np.isfinite(best):
            max_stats.append(best)
    if not max_stats:
        return None, 0
    max_array = np.asarray(max_stats, dtype=float)
    exceed = (max_array[np.newaxis, :] >= observed[:, np.newaxis] - 1e-9).sum(axis=1)
    return (exceed + 1.0) / (max_array.size + 1.0), int(max_array.size)


def _validation_signature_metrics(
    frame: pd.DataFrame,
    time_column: str,
    event_column: str,
    combo: Sequence[dict[str, Any]],
    combo_operator: str,
    min_group_size: int,
    n_iterations: int,
    validation_fraction: float,
    significance_level: float,
    random_seed: int,
    observed_hazard_ratio: float | None = None,
) -> dict[str, float | int | None]:
    """Within-cohort replication support for one signature.

    Each of ``n_iterations`` random subsamples (``validation_fraction`` of the cohort, drawn
    without replacement) is scored under the same estimability floor as the bootstrap. A fold
    supports the signature when its log-rank p is at most alpha and its hazard-ratio interval
    excludes 1 on the side of the discovery hazard ratio (``observed_hazard_ratio``; either
    side when it is unknown). Support is the share of all requested folds, so skipped folds and
    folds significant in the opposite direction count against it, as in the bootstrap.
    """
    if n_iterations <= 0:
        return {
            "Validation support (p<alpha)": None,
            "Validation median HR": None,
            "Validation median p": None,
            "Validation valid folds": 0,
            "Validation skipped folds": 0,
        }

    n_obs = int(frame.shape[0])
    min_holdout = min_group_size * 2
    max_holdout = n_obs - 1
    if max_holdout < min_holdout:
        return {
            "Validation support (p<alpha)": None,
            "Validation median HR": None,
            "Validation median p": None,
            "Validation valid folds": 0,
            "Validation skipped folds": int(n_iterations),
        }
    holdout_size = int(math.ceil(n_obs * validation_fraction))
    holdout_size = min(max(holdout_size, min_holdout), max_holdout)
    observed_harmful: bool | None = None
    if observed_hazard_ratio is not None and math.isfinite(float(observed_hazard_ratio)):
        observed_harmful = float(observed_hazard_ratio) >= 1.0
    rng = np.random.default_rng(random_seed)
    significant_count = 0
    valid_folds = 0
    skipped_folds = 0
    p_values: list[float] = []
    hazard_ratios: list[float] = []
    # Row-wise rules: evaluate the mask once on the cohort and index it per subsample.
    cohort_mask = np.asarray(_signature_mask(frame, combo, operator=combo_operator), dtype=bool)
    cohort_events = frame[event_column].to_numpy(dtype=int)
    cohort_times = frame[time_column].to_numpy(dtype=float)

    for _ in range(n_iterations):
        raise_if_cancelled()
        holdout_idx = rng.choice(n_obs, size=holdout_size, replace=False)
        mask_values = cohort_mask[holdout_idx]
        events = cohort_events[holdout_idx]
        if not _resample_is_estimable(mask_values, events):
            skipped_folds += 1
            continue
        times = cohort_times[holdout_idx]

        try:
            _, p_float = _signature_logrank(times, events, mask_values)
            hr, ci_low, ci_high = _signature_hazard_ratio(times, events, mask_values, significance_level)
        except MemoryError:
            raise
        except Exception as exc:
            if must_propagate(exc):
                raise
            skipped_folds += 1
            continue

        valid_folds += 1
        p_values.append(p_float)
        if hr is not None:
            hazard_ratios.append(hr)
        if p_float > significance_level or ci_low is None or ci_high is None or ci_low <= 1.0 <= ci_high:
            continue
        if observed_harmful is None or (ci_low > 1.0) == observed_harmful:
            significant_count += 1

    if valid_folds == 0:
        return {
            "Validation support (p<alpha)": None,
            "Validation median HR": None,
            "Validation median p": None,
            "Validation valid folds": 0,
            "Validation skipped folds": int(skipped_folds),
        }

    return {
        "Validation support (p<alpha)": float(significant_count / int(n_iterations)),
        "Validation median HR": float(np.median(hazard_ratios)) if hazard_ratios else None,
        "Validation median p": float(np.median(p_values)),
        "Validation valid folds": int(valid_folds),
        "Validation skipped folds": int(skipped_folds),
    }


# Tested family: combinations that meet the group-size rule, which depends on the covariates
# only, so the cap (and the permutation null family) never depends on the outcome.
_SIGNATURE_MAX_TESTED_COMBINATIONS = 5000
# Every (combination, operator) evaluated counts toward this cap, so searches whose
# combinations mostly fail the size rule ("and" rules on small groups) stay bounded too.
_SIGNATURE_MAX_EVALUATED_COMBINATIONS = 100_000
# Rows beyond the always-scored top of the ranking whose robustness is checked lazily, in
# rank order, until enough significant signatures are found.
_SIGNATURE_MAX_EXTRA_ROBUSTNESS_ROWS = 200
_SIGNATURE_OPERATORS = {
    "and": ["AND"],
    "or": ["OR"],
    "mixed": ["AND", "OR"],
}


def _validate_signature_search_settings(
    candidates: Sequence[str],
    *,
    max_combination_size: int,
    bootstrap_iterations: int,
    bootstrap_sample_fraction: float,
    permutation_iterations: int,
    validation_iterations: int,
    validation_fraction: float,
    significance_level: float,
    combination_operator: str,
    random_seed: int,
) -> str:
    """Check the signature-search settings and return the normalized combination operator."""
    if not candidates:
        raise ValueError("Select at least one candidate feature for signature discovery.")
    if max_combination_size < 1:
        raise ValueError("max_combination_size must be at least 1.")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be at least 0.")
    if bootstrap_sample_fraction < 0.4 or bootstrap_sample_fraction > 1.0:
        raise ValueError("bootstrap_sample_fraction must be between 0.4 and 1.0.")
    if permutation_iterations < 0:
        raise ValueError("permutation_iterations must be at least 0.")
    if validation_iterations < 0:
        raise ValueError("validation_iterations must be at least 0.")
    if validation_fraction < 0.2 or validation_fraction > 0.6:
        raise ValueError("validation_fraction must be between 0.2 and 0.6.")
    if significance_level <= 0.0 or significance_level > 0.2:
        raise ValueError("significance_level must be within (0, 0.2].")
    normalized_operator = str(combination_operator).strip().lower()
    if normalized_operator not in _SIGNATURE_OPERATORS:
        raise ValueError("combination_operator must be one of: and, or, mixed.")
    if random_seed < 0:
        raise ValueError("random_seed must be >= 0.")
    return normalized_operator


def _signature_result_row(
    combo: Sequence[dict[str, Any]],
    combo_operator: str,
    mask: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    chisq: float,
    p_value: float,
) -> dict[str, Any]:
    """One screened signature, with robustness metrics left empty until they are computed."""
    n_high = int(mask.sum())
    return {
        "Signature": f" {combo_operator} ".join(part["label"] for part in combo),
        "Combination operator": combo_operator,
        "Features": [part["column"] for part in combo],
        "Rule count": int(len(combo)),
        "N signature+": n_high,
        "N signature-": int(mask.shape[0]) - n_high,
        "Events signature+": int(events[mask].sum()),
        "Events signature-": int(events[~mask].sum()),
        "Chi-square": float(chisq),
        "P value": float(p_value),
        "Hazard ratio (signature+ vs -)": None,
        "HR CI lower": None,
        "HR CI upper": None,
        "Median signature+": _km_median_time(times[mask], events[mask]),
        "Median signature-": _km_median_time(times[~mask], events[~mask]),
        "Bootstrap support (p<alpha)": None,
        "Bootstrap median HR": None,
        "Bootstrap median p": None,
        "Bootstrap HR direction consistency": None,
        "Bootstrap valid resamples": 0,
        "Bootstrap skipped resamples": 0,
        "Permutation p": None,
        "Permutation valid resamples": 0,
        "Permutation skipped resamples": 0,
        "Validation support (p<alpha)": None,
        "Validation median HR": None,
        "Validation median p": None,
        "Validation valid folds": 0,
        "Validation skipped folds": 0,
        "Stability score": None,
        "Statistically significant": False,
    }


def _combine_indicator_masks(
    indicator_masks: Sequence[np.ndarray],
    indices: Sequence[int],
    operator: str,
) -> np.ndarray:
    parts = [indicator_masks[index] for index in indices]
    combined = np.logical_and.reduce(parts) if operator == "AND" else np.logical_or.reduce(parts)
    return np.asarray(combined, dtype=bool)


class _SignatureScreen(NamedTuple):
    """Result of screening every indicator combination."""

    rows: list[dict[str, Any]]
    combinations: list[dict[str, Any]]
    truncated: bool
    indicator_masks: list[np.ndarray]
    # Every combination that met the group-size rule (the tested family), as
    # (indicator indices, operator). The rule depends on the covariates only, so this family
    # is the permutation null family.
    family: list[tuple[tuple[int, ...], str]]
    evaluated_combinations: int

    def family_matrix(self) -> np.ndarray:
        """Masks of the tested family as an (n, k) boolean matrix."""
        n_rows = int(self.indicator_masks[0].shape[0]) if self.indicator_masks else 0
        if not self.family:
            return np.zeros((n_rows, 0), dtype=bool)
        matrix = np.empty((n_rows, len(self.family)), dtype=bool)
        for position, (indices, operator) in enumerate(self.family):
            matrix[:, position] = _combine_indicator_masks(self.indicator_masks, indices, operator)
        return matrix


def _screen_signature_combinations(
    frame: pd.DataFrame,
    indicators: Sequence[dict[str, Any]],
    *,
    time_column: str,
    event_column: str,
    max_size: int,
    operator: str,
    min_group_size: int,
    min_events_per_group: int,
) -> _SignatureScreen:
    """Log-rank screen of every indicator combination up to max_size.

    Each indicator is evaluated once and combinations are built from those masks. The search
    stops (``truncated``) when the size-feasible family reaches the tested-combination cap or
    when the number of evaluated combinations reaches the evaluation cap; both counts depend
    on the covariates only. Rows are the size-feasible combinations that also meet the event
    rule on the observed outcome.
    """
    n_obs = int(frame.shape[0])
    times = frame[time_column].to_numpy(dtype=float)
    events = frame[event_column].to_numpy(dtype=int)
    indicator_masks = [
        _evaluate_indicator(frame, indicator).fillna(False).to_numpy(dtype=bool) for indicator in indicators
    ]
    rows: list[dict[str, Any]] = []
    valid_combinations: list[dict[str, Any]] = []
    family: list[tuple[tuple[int, ...], str]] = []
    evaluated = 0

    def _result(truncated: bool) -> _SignatureScreen:
        return _SignatureScreen(rows, valid_combinations, truncated, indicator_masks, family, evaluated)

    for size in range(1, max_size + 1):
        # A single rule reads the same under AND and OR, so mixed mode tests it once.
        operator_list = ["AND"] if size == 1 and operator == "mixed" else _SIGNATURE_OPERATORS[operator]
        for combo_index in itertools.combinations(range(len(indicators)), size):
            raise_if_cancelled()
            combo = tuple(indicators[index] for index in combo_index)
            if len({part["column"] for part in combo}) != len(combo):
                continue
            for combo_operator in operator_list:
                if len(family) >= _SIGNATURE_MAX_TESTED_COMBINATIONS or evaluated >= _SIGNATURE_MAX_EVALUATED_COMBINATIONS:
                    return _result(True)
                evaluated += 1
                mask = _combine_indicator_masks(indicator_masks, combo_index, combo_operator)
                n_high = int(mask.sum())
                if n_high < min_group_size or n_obs - n_high < min_group_size:
                    continue
                family.append((tuple(combo_index), combo_operator))
                if events[mask].sum() < min_events_per_group or events[~mask].sum() < min_events_per_group:
                    continue
                try:
                    chisq, p_value = _signature_logrank(times, events, mask)
                except MemoryError:
                    raise
                except Exception as exc:
                    if must_propagate(exc):
                        raise
                    continue
                valid_combinations.append(
                    {"combo": combo, "operator": combo_operator, "indicator_indices": tuple(combo_index)}
                )
                rows.append(_signature_result_row(combo, combo_operator, mask, times, events, chisq, p_value))
    return _result(False)


def _signature_metric_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    significance_level: float,
) -> tuple[list[int], list[int]]:
    """Rows that get the expensive robustness metrics, in primary rank order.

    ``always`` is the top of the ranking by (BH p, p, -chi-square), which holds the output
    candidates. ``conditional`` holds the further rows that pass the BH threshold, the only
    other rows that can be significant; they are checked lazily, in rank order, until enough
    significant signatures are found or a cap is reached.
    """
    primary_ranked_idx = sorted(
        range(len(rows)),
        key=lambda idx: (
            rows[idx]["BH adjusted p"],
            rows[idx]["P value"],
            -float(rows[idx]["Chi-square"]),
        ),
    )
    n_always = min(len(primary_ranked_idx), max(80, top_k * 4))
    always = primary_ranked_idx[:n_always]
    conditional = [
        idx
        for idx in primary_ranked_idx[n_always:]
        if float(rows[idx].get("BH adjusted p", 1.0)) <= float(significance_level)
    ]
    return always, conditional


def _add_signature_robustness_metrics(
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    valid_combinations: Sequence[dict[str, Any]],
    metric_candidates: tuple[Sequence[int], Sequence[int]] | Sequence[int],
    *,
    time_column: str,
    event_column: str,
    min_group_size: int,
    min_events_per_group: int,
    bootstrap_iterations: int,
    bootstrap_sample_fraction: float,
    permutation_iterations: int,
    validation_iterations: int,
    validation_fraction: float,
    significance_level: float,
    random_seed: int,
    family_masks: np.ndarray | None = None,
    top_k: int | None = None,
    indicator_masks: Sequence[np.ndarray] | None = None,
) -> dict[str, int]:
    """Fill in the Cox, bootstrap, permutation, and validation metrics of the screened rows.

    Permutation p-values are computed for every row against ``family_masks`` (all
    size-feasible combinations). The other metrics are computed for the ``always`` rows and
    then, lazily and in rank order, for the ``conditional`` rows until ``top_k`` significant
    signatures are known or ``_SIGNATURE_MAX_EXTRA_ROBUSTNESS_ROWS`` extra rows were scored.
    ``indicator_masks`` (from the screen) rebuild each row's mask without re-evaluating
    its rules on the frame. Returns how many rows were scored and how many BH-passing rows
    were left unscored.
    """
    times = frame[time_column].to_numpy(dtype=float)
    events = frame[event_column].to_numpy(dtype=int)
    if isinstance(metric_candidates, tuple) and len(metric_candidates) == 2 and not isinstance(metric_candidates[0], int):
        always, conditional = list(metric_candidates[0]), list(metric_candidates[1])
    else:
        always, conditional = list(metric_candidates), []

    if permutation_iterations > 0:
        if family_masks is None:
            family_masks = np.column_stack(
                [_signature_mask(frame, combo["combo"], operator=combo["operator"]) for combo in valid_combinations]
            )
        adjusted_perm_p, valid_perm = _search_adjusted_permutation_p_values(
            times,
            events,
            family_masks,
            np.asarray([float(row["Chi-square"]) for row in rows], dtype=float),
            min_events_per_group=min_events_per_group,
            n_iterations=permutation_iterations,
            random_seed=random_seed + 2000,
        )
        for idx, row in enumerate(rows):
            row["Permutation p"] = None if adjusted_perm_p is None else float(adjusted_perm_p[idx])
            row["Permutation valid resamples"] = int(valid_perm)
            row["Permutation skipped resamples"] = max(0, int(permutation_iterations) - int(valid_perm))

    def _is_significant(row: dict[str, Any]) -> bool:
        return _signature_is_significant(
            row,
            significance_level=significance_level,
            require_permutation=permutation_iterations > 0,
            require_validation=validation_iterations > 0,
            min_validation_support=0.5,
            require_bootstrap_consistency=bootstrap_iterations > 0,
            min_bootstrap_consistency=0.6,
        )

    def _score(idx: int) -> None:
        combo = valid_combinations[idx]
        if indicator_masks is not None and combo.get("indicator_indices") is not None:
            mask = _combine_indicator_masks(indicator_masks, combo["indicator_indices"], combo["operator"])
        else:
            mask = _signature_mask(frame, combo["combo"], operator=combo["operator"])
        try:
            rows[idx].update(_signature_cox_metrics(times, events, mask, alpha=significance_level))
        except MemoryError:
            raise
        except Exception as exc:
            if must_propagate(exc):
                raise
            warnings.warn(
                f"Skipping Cox robustness metrics for signature row {idx} due to: {exc}",
                RuntimeWarning,
            )
        observed_hazard_ratio = _safe_float(rows[idx].get("Hazard ratio (signature+ vs -)"))
        if bootstrap_iterations > 0:
            rows[idx].update(
                _bootstrap_signature_metrics(
                    frame=frame,
                    time_column=time_column,
                    event_column=event_column,
                    combo=combo["combo"],
                    combo_operator=combo["operator"],
                    min_group_size=min_group_size,
                    n_iterations=bootstrap_iterations,
                    sample_fraction=bootstrap_sample_fraction,
                    random_seed=random_seed + 1000 + idx,
                    significance_level=significance_level,
                    observed_hazard_ratio=observed_hazard_ratio,
                )
            )
        if validation_iterations > 0:
            rows[idx].update(
                _validation_signature_metrics(
                    frame=frame,
                    time_column=time_column,
                    event_column=event_column,
                    combo=combo["combo"],
                    combo_operator=combo["operator"],
                    min_group_size=min_group_size,
                    n_iterations=validation_iterations,
                    validation_fraction=validation_fraction,
                    significance_level=significance_level,
                    random_seed=random_seed + 3000 + idx,
                    observed_hazard_ratio=observed_hazard_ratio,
                )
            )

    for idx in always:
        _score(idx)
    significant_found = sum(1 for idx in always if _is_significant(rows[idx]))
    wanted = int(top_k) if top_k is not None else len(rows)
    extra_scored = 0
    for idx in conditional:
        if significant_found >= wanted or extra_scored >= _SIGNATURE_MAX_EXTRA_ROBUSTNESS_ROWS:
            break
        _score(idx)
        extra_scored += 1
        if _is_significant(rows[idx]):
            significant_found += 1

    for row in rows:
        row["Stability score"] = _stability_score(row, significance_level)
        row["Statistically significant"] = _is_significant(row)
    return {
        "robustness_evaluated_signatures": len(always) + extra_scored,
        "robustness_unevaluated_signatures": len(conditional) - extra_scored,
    }


def _apply_signature_column(
    df: pd.DataFrame,
    column_name: str,
    combo: Sequence[dict[str, Any]],
    operator: str,
) -> pd.DataFrame:
    """Copy of df with a Signature+/Signature- column; rows missing any rule input stay missing."""
    output_df = df.copy()
    indicator_values_list = [_evaluate_indicator(output_df, indicator) for indicator in combo]
    missing = pd.Series(False, index=output_df.index, dtype=bool)
    for indicator_values in indicator_values_list:
        missing = missing | indicator_values.isna().to_numpy(dtype=bool)
    bool_arrays = [item.fillna(False).to_numpy(dtype=bool) for item in indicator_values_list]
    if operator == "OR":
        combined = np.logical_or.reduce(bool_arrays)
    else:
        combined = np.logical_and.reduce(bool_arrays)
    labels = pd.Series(np.where(combined, "Signature+", "Signature-"), index=output_df.index, dtype="string")
    labels.loc[missing] = pd.NA
    output_df[column_name] = labels
    return output_df


@user_input_boundary
def discover_feature_signature(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    candidate_columns: Sequence[str],
    event_positive_value: Any = None,
    max_combination_size: int = 3,
    top_k: int = 15,
    min_group_fraction: float = 0.1,
    bootstrap_iterations: int = 30,
    bootstrap_sample_fraction: float = 0.8,
    permutation_iterations: int = 0,
    validation_iterations: int = 0,
    validation_fraction: float = 0.35,
    significance_level: float = 0.05,
    combination_operator: str = "mixed",
    random_seed: int = 20260311,
    new_column_name: str | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    unique_candidates = list(dict.fromkeys(candidate_columns))
    normalized_operator = _validate_signature_search_settings(
        unique_candidates,
        max_combination_size=max_combination_size,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_sample_fraction=bootstrap_sample_fraction,
        permutation_iterations=permutation_iterations,
        validation_iterations=validation_iterations,
        validation_fraction=validation_fraction,
        significance_level=significance_level,
        combination_operator=combination_operator,
        random_seed=random_seed,
    )

    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=unique_candidates,
    )

    n_obs = int(frame.shape[0])
    min_group_size = max(8, int(math.ceil(n_obs * min_group_fraction)))
    min_events_per_group = max(3, int(math.ceil(n_obs * 0.03)))
    skipped_candidates: list[dict[str, str]] = []
    indicators = _build_candidate_indicators(
        frame, unique_candidates, min_group_size=min_group_size, skipped=skipped_candidates
    )
    if not indicators:
        reasons = "; ".join(f"{item['column']}: {item['reason']}" for item in skipped_candidates[:5])
        raise ValueError(
            "No valid binary indicators could be generated from the selected features"
            + (f" ({reasons})." if reasons else ".")
        )

    max_size = min(max_combination_size, len(indicators))
    screen = _screen_signature_combinations(
        frame,
        indicators,
        time_column=time_column,
        event_column=event_column,
        max_size=max_size,
        operator=normalized_operator,
        min_group_size=min_group_size,
        min_events_per_group=min_events_per_group,
    )
    rows, valid_combinations, truncated = screen.rows, screen.combinations, screen.truncated
    if not rows:
        raise ValueError("No analyzable feature combinations passed minimum group/event requirements.")

    adjusted = _bh_adjust([row["P value"] for row in rows])
    for row, adj in zip(rows, adjusted, strict=True):
        row["BH adjusted p"] = adj

    robustness_counts = _add_signature_robustness_metrics(
        frame,
        rows,
        valid_combinations,
        _signature_metric_candidates(rows, top_k=top_k, significance_level=significance_level),
        time_column=time_column,
        event_column=event_column,
        min_group_size=min_group_size,
        min_events_per_group=min_events_per_group,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_sample_fraction=bootstrap_sample_fraction,
        permutation_iterations=permutation_iterations,
        validation_iterations=validation_iterations,
        validation_fraction=validation_fraction,
        significance_level=significance_level,
        random_seed=random_seed,
        # The permutation null family is every size-feasible combination, materialized only
        # when permutations were requested.
        family_masks=screen.family_matrix() if permutation_iterations > 0 else None,
        top_k=top_k,
        indicator_masks=screen.indicator_masks,
    )

    ranked_idx = sorted(
        range(len(rows)),
        key=lambda idx: (
            not rows[idx]["Statistically significant"],
            -rows[idx]["Stability score"],
            rows[idx]["BH adjusted p"],
            rows[idx]["P value"],
        ),
    )
    ranked_rows = [rows[idx] for idx in ranked_idx][:top_k]
    best_idx = ranked_idx[0]
    best_row = rows[best_idx]

    best_column_name = _derived_group_column_name(df, new_column_name, "auto_signature_group")
    output_df = _apply_signature_column(
        df,
        best_column_name,
        valid_combinations[best_idx]["combo"],
        valid_combinations[best_idx]["operator"],
    )
    counts = _group_counts(output_df[best_column_name])
    search_space = {
        "n_rows_analyzed": n_obs,
        "row_mask_hash": str(frame.attrs.get("row_mask_hash") or ""),
        "candidate_columns": unique_candidates,
        "skipped_candidates": skipped_candidates,
        "generated_indicators": int(len(indicators)),
        "tested_combinations": int(len(rows)),
        "evaluated_combinations": int(screen.evaluated_combinations),
        "size_feasible_combinations": int(len(screen.family)),
        "permutation_family_size": int(len(screen.family)) if permutation_iterations > 0 else None,
        "robustness_evaluated_signatures": int(robustness_counts["robustness_evaluated_signatures"]),
        "robustness_unevaluated_signatures": int(robustness_counts["robustness_unevaluated_signatures"]),
        "time_column_note": frame.attrs.get("time_column_note"),
        "truncated": truncated,
        "min_group_size": int(min_group_size),
        "min_events_per_group": int(min_events_per_group),
        "max_combination_size": int(max_size),
        "combination_operator": normalized_operator,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_sample_fraction": float(bootstrap_sample_fraction),
        "bootstrap_scored_signatures": int(
            sum(1 for row in rows if row["Bootstrap valid resamples"] > 0)
        ),
        "bootstrap_skipped_resamples_total": int(
            sum(int(row.get("Bootstrap skipped resamples") or 0) for row in rows)
        ),
        "permutation_iterations": int(permutation_iterations),
        "permutation_method": "max_statistic_over_search" if permutation_iterations > 0 else None,
        "permutation_scored_signatures": int(
            sum(1 for row in rows if row["Permutation valid resamples"] > 0)
        ),
        "permutation_skipped_resamples_total": int(
            sum(int(row.get("Permutation skipped resamples") or 0) for row in rows)
        ),
        "validation_iterations": int(validation_iterations),
        "validation_fraction": float(validation_fraction),
        "validation_scored_signatures": int(
            sum(1 for row in rows if row["Validation valid folds"] > 0)
        ),
        "validation_skipped_folds_total": int(
            sum(int(row.get("Validation skipped folds") or 0) for row in rows)
        ),
        "significance_level": float(significance_level),
        "permutation_required_for_significance": bool(permutation_iterations > 0),
        "validation_required_for_significance": bool(validation_iterations > 0),
        "validation_min_support": 0.5,
        "bootstrap_min_direction_consistency": 0.6,
        "random_seed": int(random_seed),
        "significant_signatures": int(sum(1 for row in rows if row["Statistically significant"])),
    }
    scientific_summary = _signature_scientific_summary(
        best_split=best_row,
        search_space=search_space,
    )
    signature_recipe = {
        "column_name": best_column_name,
        "operator": str(valid_combinations[best_idx]["operator"]),
        "features": list(best_row.get("Features", [])),
        "signature": str(best_row.get("Signature", "")),
        # The exact rule definitions (full-precision thresholds), so the locked signature can
        # be re-applied; the "signature" text rounds thresholds for display.
        "rules": [_indicator_definition(indicator) for indicator in valid_combinations[best_idx]["combo"]],
        "positive_label": "Signature+",
        "negative_label": "Signature-",
        "statistically_significant": bool(best_row.get("Statistically significant")),
        "outcome_informed": True,
        "random_seed": int(random_seed),
    }

    payload = {
        "results_table": ranked_rows,
        "best_split": best_row,
        "search_space": search_space,
        "signature_recipe": signature_recipe,
        "derived_group": {
            "column_name": best_column_name,
            "counts": counts,
            "outcome_informed": True,
            "auto_apply_recommended": bool(best_row.get("Statistically significant")),
            "recipe": signature_recipe,
        },
        "scientific_summary": scientific_summary,
    }
    return output_df, best_column_name, payload


def _nice_time_ticks(horizon: float, points: int) -> list[float]:
    """Round times from 0 up to the horizon for the risk table and the time axis.

    The step is 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten, whichever gives the tick
    count closest to ``points`` (the finer step on a tie), so the columns read 0, 12, 24 ... or
    0, 40, 80 ... rather than 0, 47.62, 95.24 ..., line up with the axis ticks, and follow the
    requested number of points closely.
    """
    if not np.isfinite(horizon) or horizon <= 0:
        return [0.0]
    raw_step = horizon / max(int(points) - 1, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    best: tuple[int, float] | None = None
    for scale in (magnitude / 10.0, magnitude, magnitude * 10.0):
        for multiple in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
            step = multiple * scale
            count = int(np.floor(horizon / step * (1.0 + 1e-9))) + 1
            if best is None or abs(count - points) < abs(best[0] - points):
                best = (count, step)
    count, step = best
    # Tick times are rounded to the step's precision; the last one never passes the horizon, so a
    # patient followed exactly to it is still counted at risk there.
    decimals = max(0, int(-np.floor(np.log10(step))) + 2)
    return [min(round(index * step, decimals), float(horizon)) for index in range(count)]


def _risk_tick_labels(ticks: Sequence[float]) -> list[str]:
    """Readable, unique column labels for risk-table tick times.

    Labels use two decimals when that keeps them distinct and more digits otherwise, so
    ticks on a very short horizon never collapse onto one column key.
    """
    values = [float(tick) for tick in ticks]
    for decimals in range(2, 16):
        labels = [f"{round(value, decimals):g}" for value in values]
        if len(set(labels)) == len(labels):
            return labels
    return [f"{value:.17g}" for value in values]


def _km_group_estimates(
    group_frame: pd.DataFrame,
    label: str,
    *,
    time_column: str,
    event_column: str,
    alpha: float,
    display_horizon: float,
    rmst_horizon: float,
    risk_ticks: Sequence[float],
    risk_tick_labels: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
    """One group's summary row, risk-table row, plotted curve, and RMST statistics."""
    t_group = group_frame[time_column].to_numpy(dtype=float)
    e_group = group_frame[event_column].to_numpy(dtype=int)
    sf = SurvfuncRight(t_group, e_group)

    event_times = sf.surv_times.astype(float)
    step_timeline = np.concatenate(([0.0], event_times))
    step_survival = np.concatenate(([1.0], sf.surv_prob.astype(float)))
    lower, upper = _pointwise_km_ci(sf.surv_prob.astype(float), sf.surv_prob_se.astype(float), alpha=alpha)
    lower = np.concatenate(([1.0], lower))
    upper = np.concatenate(([1.0], upper))
    censor_times = group_frame.loc[group_frame[event_column] == 0, time_column].to_numpy(dtype=float)
    censor_survival = _step_values(event_times, sf.surv_prob.astype(float), censor_times) if censor_times.size else np.array([])

    # Carry the last KM level only to this group's own last follow-up time;
    # extending it to the pooled horizon would draw survival where the
    # group has no observations.
    group_curve_end = float(min(display_horizon, float(np.max(t_group))))
    if step_timeline[-1] < group_curve_end:
        step_timeline = np.concatenate((step_timeline, [group_curve_end]))
        step_survival = np.concatenate((step_survival, [step_survival[-1]]))
        lower = np.concatenate((lower, [lower[-1]]))
        upper = np.concatenate((upper, [upper[-1]]))

    with suppressed_warnings(RuntimeWarning):
        median_ci = sf.quantile_ci(0.5, alpha=alpha)
    rmst_stats = _restricted_mean_survival_time_delta_stats(
        event_times,
        sf.surv_prob.astype(float),
        sf.n_risk.astype(float),
        sf.n_events.astype(float),
        rmst_horizon,
        alpha=alpha,
    )
    summary_row = {
        "Group": label,
        "N": int(group_frame.shape[0]),
        "Events": int(group_frame[event_column].sum()),
        "Censored": int((group_frame[event_column] == 0).sum()),
        "Median survival": _km_median_time(t_group, e_group),
        "Median CI lower": _safe_float(median_ci[0]) if isinstance(median_ci, tuple) else None,
        "Median CI upper": _safe_float(median_ci[1]) if isinstance(median_ci, tuple) else None,
        "RMST": _safe_float(rmst_stats["rmst"]),
        "RMST SE": _safe_float(rmst_stats["se"]),
        "RMST CI lower": _safe_float(rmst_stats["ci_lower"]),
        "RMST CI upper": _safe_float(rmst_stats["ci_upper"]),
    }
    risk_row = OrderedDict({"Group": label})
    for tick, tick_label in zip(risk_ticks, risk_tick_labels, strict=True):
        risk_row[tick_label] = int((group_frame[time_column] >= tick).sum())
    curve = {
        "group": label,
        "timeline": step_timeline.tolist(),
        "survival": step_survival.tolist(),
        "ci_lower": lower.tolist(),
        "ci_upper": upper.tolist(),
        "censor_times": censor_times.tolist(),
        "censor_survival": censor_survival.tolist(),
    }
    return summary_row, dict(risk_row), curve, rmst_stats


_KM_NOT_TESTABLE_NOTE = "Not testable: zero log-rank variance"


def _km_group_tests(
    frame: pd.DataFrame,
    group_labels: Sequence[str],
    *,
    time_column: str,
    event_column: str,
    group_column: str,
    logrank_weight: str,
    fh_p: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The global weighted log-rank test plus BH-adjusted pairwise tests.

    A comparison whose log-rank variance is zero (for example two groups whose every event
    falls at one time that nobody at risk survives) cannot be tested; R's survdiff stops with an
    error there. Such a pair is reported with no statistic and a note, and left out of the BH
    adjustment; when the global test itself cannot be computed it is returned as None. P-values
    are upper chi-square tails (``chi2.sf``), which stay positive for strong separation where
    ``1 - chi2.cdf`` rounds to 0.
    """
    time_values = frame[time_column].to_numpy(dtype=float)
    event_values = frame[event_column].to_numpy(dtype=int)
    group_values = frame[group_column].astype(str).to_numpy()
    weight_type = KM_WEIGHT_MAP[logrank_weight]
    kwargs: dict[str, Any] = {"fh_p": fh_p} if weight_type == "fh" else {}

    def _weighted_test(mask: np.ndarray | None) -> tuple[float, float] | None:
        rows = np.ones(time_values.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        times, events, groups = time_values[rows], event_values[rows], group_values[rows]
        # As R's survdiff does, groups with no one at risk at any event time (zero expected
        # events) are left out of the test; with fewer than two groups left the statistic is 0.
        event_times = times[events == 1]
        if event_times.size == 0:
            return 0.0, 1.0
        first_event_time = float(event_times.min())
        at_risk_groups = {str(label) for label in groups[times >= first_event_time]}
        if len(at_risk_groups) < 2:
            return 0.0, 1.0
        if len(at_risk_groups) < len(set(groups.tolist())):
            keep = np.isin(groups, sorted(at_risk_groups))
            times, events, groups = times[keep], events[keep], groups[keep]
        try:
            # Fleming-Harrington weights take log(1 - d/n); when everyone still at risk has the
            # event that is log(0) = -inf, so S is 0 from then on, as intended.
            with np.errstate(divide="ignore"):
                chisq, _ = survdiff(times, events, groups, weight_type=weight_type, **kwargs)
        except np.linalg.LinAlgError:
            return None
        chisq = float(chisq)
        if not math.isfinite(chisq):
            return None
        return chisq, float(stats.chi2.sf(chisq, len(at_risk_groups) - 1))

    global_test = _weighted_test(None)
    test_payload = (
        None
        if global_test is None
        else {"test": logrank_weight, "chisq": global_test[0], "p_value": global_test[1]}
    )
    pairwise_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(group_labels, 2):
        raise_if_cancelled()
        outcome = _weighted_test(frame[group_column].isin([left, right]).to_numpy())
        pairwise_rows.append(
            {
                "Comparison": f"{left} vs {right}",
                "Chi-square": None if outcome is None else outcome[0],
                "P value": None if outcome is None else outcome[1],
            }
        )
    adjusted = _bh_adjust([np.nan if row["P value"] is None else row["P value"] for row in pairwise_rows])
    for row, adjusted_p in zip(pairwise_rows, adjusted, strict=True):
        row["BH adjusted p"] = _safe_float(adjusted_p)
    if any(row["P value"] is None for row in pairwise_rows):
        # Every row carries the note column, so tables built from the first row show it.
        for row in pairwise_rows:
            row["Note"] = _KM_NOT_TESTABLE_NOTE if row["P value"] is None else ""
    return test_payload, pairwise_rows


def _km_cohort_summary(frame: pd.DataFrame, *, time_column: str, event_column: str) -> dict[str, Any]:
    return {
        "n": int(frame.shape[0]),
        "events": int(frame[event_column].sum()),
        "censored": int((1 - frame[event_column]).sum()),
        "median_follow_up": _median_follow_up(frame[time_column], frame[event_column]),
        "time_max": float(np.nanmax(frame[time_column])),
        "row_mask_hash": str(frame.attrs.get("row_mask_hash") or ""),
        "dropped_missing_rows": int(frame.attrs.get("dropped_missing_rows", 0)),
        "dropped_nonpositive_time_rows": int(frame.attrs.get("dropped_nonpositive_time_rows", 0)),
        "rows_with_infinite_values": int(frame.attrs.get("rows_with_infinite_values", 0)),
        "time_column_note": frame.attrs.get("time_column_note"),
    }


def _km_rmst_contrast(
    summary_rows: Sequence[dict[str, Any]],
    rmst_stats_by_group: Sequence[dict[str, float]],
    *,
    rmst_horizon: float,
    alpha: float,
    outcome_informed_group: bool,
) -> dict[str, Any] | None:
    """Two-group RMST difference with a delta-method CI and Wald test (None otherwise).

    For outcome-informed groups, whose split was chosen on the same outcome, only the point
    estimate is kept: the standard error, confidence interval, and Wald test would all treat
    the selected split as if it had been fixed in advance.
    """
    if len(summary_rows) != 2 or len(rmst_stats_by_group) != 2:
        return None
    left_row, right_row = summary_rows
    left_stats, right_stats = rmst_stats_by_group
    estimate = float(left_stats["rmst"] - right_stats["rmst"])
    rmst_contrast: dict[str, Any] = {
        "comparison": f"{left_row['Group']} minus {right_row['Group']}",
        "estimate": _safe_float(estimate),
        "se": None,
        "z_statistic": None,
        "p_value": None,
        "p_value_method": None,
        "ci_lower": None,
        "ci_upper": None,
        "horizon": _safe_float(rmst_horizon),
    }
    left_variance = left_stats.get("variance")
    right_variance = right_stats.get("variance")
    if (
        left_variance is not None
        and right_variance is not None
        and np.isfinite(left_variance)
        and np.isfinite(right_variance)
    ):
        variance = float(left_variance + right_variance)
        se = math.sqrt(max(variance, 0.0))
        z_value = float(ndtri(1.0 - alpha / 2.0))
        rmst_contrast["se"] = _safe_float(se)
        if se > 0.0 and np.isfinite(se):
            z_statistic = float(estimate / se)
            rmst_contrast["z_statistic"] = _safe_float(z_statistic)
            rmst_contrast["p_value"] = _safe_float(2.0 * stats.norm.sf(abs(z_statistic)))
            rmst_contrast["p_value_method"] = "wald_normal"
        elif estimate == 0.0:
            rmst_contrast["z_statistic"] = 0.0
            rmst_contrast["p_value"] = 1.0
            rmst_contrast["p_value_method"] = "wald_normal"
        rmst_contrast["ci_lower"] = _safe_float(estimate - z_value * se)
        rmst_contrast["ci_upper"] = _safe_float(estimate + z_value * se)
    if outcome_informed_group:
        rmst_contrast["se"] = None
        rmst_contrast["z_statistic"] = None
        rmst_contrast["p_value"] = None
        rmst_contrast["p_value_method"] = None
        rmst_contrast["ci_lower"] = None
        rmst_contrast["ci_upper"] = None
        rmst_contrast["inference_withheld"] = "outcome_informed_group"
    return rmst_contrast


@user_input_boundary
def compute_km_analysis(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    group_column: str | None = None,
    event_positive_value: Any = None,
    confidence_level: float = 0.95,
    max_time: float | None = None,
    risk_table_points: int = 6,
    logrank_weight: str = "logrank",
    fh_p: float = 1.0,
    suppress_group_inference: bool = False,
    outcome_informed_group: bool = False,
) -> dict[str, Any]:
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")
    if max_time is not None and not (math.isfinite(float(max_time)) and float(max_time) > 0.0):
        raise ValueError("max_time must be positive and finite when provided.")
    if int(risk_table_points) < 1:
        raise ValueError("risk_table_points must be at least 1.")
    if logrank_weight not in KM_WEIGHT_MAP:
        # A typo would otherwise run a plain log-rank test labelled with the typo.
        raise ValueError(
            f"Unknown logrank_weight '{logrank_weight}'. Choose one of: {', '.join(KM_WEIGHT_MAP)}."
        )

    extra_columns = [group_column] if group_column else []
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=extra_columns,
    )
    if group_column:
        frame[group_column] = _canonical_level_strings(frame[group_column])
        frame = frame.dropna(subset=[group_column]).reset_index(drop=True)
        group_labels = _sorted_group_labels(frame[group_column], group_column)
    else:
        group_labels = ["Overall"]

    if len(group_labels) == 0:
        raise ValueError("No groups remain after removing missing values.")
    if group_column and len(group_labels) > 50:
        raise ValueError(
            "Kaplan-Meier can plot at most 50 groups at once. "
            "Choose a grouped clinical variable instead of an identifier-like column."
        )

    time_values = frame[time_column].to_numpy(dtype=float)
    alpha = 1.0 - confidence_level
    display_horizon = float(np.nanmax(time_values) if max_time is None else min(max_time, np.nanmax(time_values)))
    display_horizon = max(display_horizon, 1e-6)
    if group_column:
        group_max_times = [
            float(frame.loc[frame[group_column] == label, time_column].max())
            for label in group_labels
        ]
        common_group_horizon = min(group_max_times) if group_max_times else display_horizon
        rmst_horizon = max(min(display_horizon, common_group_horizon), 1e-6)
    else:
        rmst_horizon = display_horizon
    risk_ticks = _nice_time_ticks(display_horizon, risk_table_points)
    risk_tick_labels = _risk_tick_labels(risk_ticks)

    summary_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    curve_payloads: list[dict[str, Any]] = []
    rmst_stats_by_group: list[dict[str, float]] = []
    for label in group_labels:
        group_frame = frame if label == "Overall" and not group_column else frame.loc[frame[group_column] == label].copy()
        summary_row, risk_row, curve, rmst_stats = _km_group_estimates(
            group_frame,
            label,
            time_column=time_column,
            event_column=event_column,
            alpha=alpha,
            display_horizon=display_horizon,
            rmst_horizon=rmst_horizon,
            risk_ticks=risk_ticks,
            risk_tick_labels=risk_tick_labels,
        )
        summary_rows.append(summary_row)
        risk_rows.append(risk_row)
        curve_payloads.append(curve)
        rmst_stats_by_group.append(rmst_stats)

    # A split chosen on the same outcome makes a fresh log-rank p-value invalid, so the test is
    # never run for outcome-informed groups, whatever suppress_group_inference says.
    suppress_group_inference = bool(suppress_group_inference or outcome_informed_group)
    test_payload = None
    pairwise_rows: list[dict[str, Any]] = []
    global_test_untestable = False
    if group_column and len(group_labels) >= 2 and not suppress_group_inference:
        test_payload, pairwise_rows = _km_group_tests(
            frame,
            group_labels,
            time_column=time_column,
            event_column=event_column,
            group_column=group_column,
            logrank_weight=logrank_weight,
            fh_p=fh_p,
        )
        global_test_untestable = test_payload is None

    cohort_summary = _km_cohort_summary(frame, time_column=time_column, event_column=event_column)
    rmst_contrast = _km_rmst_contrast(
        summary_rows,
        rmst_stats_by_group,
        rmst_horizon=rmst_horizon,
        alpha=alpha,
        outcome_informed_group=outcome_informed_group,
    )

    scientific_summary = _km_scientific_summary(
        summary_rows=summary_rows,
        cohort_summary=cohort_summary,
        group_column=group_column,
        test_payload=test_payload,
        pairwise_rows=pairwise_rows,
        fh_p=fh_p,
        outcome_informed_group=outcome_informed_group,
        rmst_contrast=rmst_contrast,
        rmst_horizon=rmst_horizon,
        display_horizon=display_horizon,
        global_test_untestable=global_test_untestable,
    )
    duplicate_caution = duplicate_identifier_caution(df)
    if duplicate_caution and isinstance(scientific_summary, dict):
        scientific_summary.setdefault("cautions", []).append(duplicate_caution)

    test_label = (
        None
        if test_payload is None
        else _weighted_test_label(str(test_payload["test"]), fh_p=fh_p)
    )

    return {
        "curves": curve_payloads,
        "summary_table": summary_rows,
        "groups": int(len(summary_rows)),
        "logrank_p": (
            None
            if test_payload is None or logrank_weight != "logrank"
            else float(test_payload["p_value"])
        ),
        "risk_table": {
            "columns": ["Group", *risk_tick_labels],
            "rows": risk_rows,
            "times": risk_ticks,
        },
        "pairwise_table": pairwise_rows,
        "test": test_payload,
        "test_p_value": None if test_payload is None else float(test_payload["p_value"]),
        "test_p_value_label": test_label,
        "cohort": cohort_summary,
        "confidence_level": float(confidence_level),
        "display_horizon": display_horizon,
        "rmst_horizon": rmst_horizon,
        "group_column": group_column,
        "outcome_informed_group": bool(outcome_informed_group),
        "rmst_contrast": rmst_contrast,
        "scientific_summary": scientific_summary,
    }


def _is_number_text(series: pd.Series) -> bool:
    """True when every non-missing value of a text column reads as a finite number once trimmed."""
    text = series.dropna().astype(str).str.strip()
    numbers = pd.to_numeric(text, errors="coerce")
    if bool(numbers.isna().any()):
        return False
    return bool(np.isfinite(numbers.to_numpy(dtype=float)).all())


def _categorical_candidates(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Covariates encoded as categorical without being declared.

    The typing rule shared with the ML and deep-learning encoders: a column with a pandas
    categorical dtype, or a non-numeric column whose non-missing values do not all read as
    finite numbers, is categorical. A text column whose values are all finite numbers (numbers
    stored as text, as a CSV load would have read them) is numeric, and so are booleans.
    """
    candidates: list[str] = []
    for column in columns:
        series = df[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            candidates.append(column)
        elif not is_numeric_dtype(series) and not is_bool_dtype(series) and not _is_number_text(series):
            candidates.append(column)
    return candidates


def _resolve_cox_categorical_covariates(
    df: pd.DataFrame,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str] | None,
) -> list[str]:
    """Explicitly marked categorical covariates plus every undeclared covariate typed as categorical.

    Text covariates are categorical (see ``_categorical_candidates``): coercing labels to numbers
    would silently turn them into missing values and drop those rows from the complete-case
    fit. A text column that holds only numbers is used as a number. Continuous numbers with a
    few stray text values are refused instead of becoming a categorical term with one level per
    distinct number.
    """
    # Missing columns are reported together with the outcome columns by _cohort_frame.
    present = [column for column in covariates if column in df.columns]
    explicit = list(dict.fromkeys(categorical_covariates or []))
    # Covariates the user marked categorical are used as they are (the Cox design check still refuses
    # identifier-like ones); the others get the shared text-feature checks.
    reject_numeric_text_features(df, present, explicit)
    covariate_set = set(covariates)
    unselected = [str(column) for column in explicit if column not in covariate_set]
    if unselected:
        raise ValueError(
            "Categorical covariates must also be selected as covariates: "
            + ", ".join(unselected)
            + ". Select them as covariates or remove the categorical flag."
        )
    return explicit + [column for column in _categorical_candidates(df, present) if column not in explicit]


def _cox_categorical(values: pd.Series, column: str) -> pd.Categorical:
    """Categorical covariate whose first category is the Cox reference level.

    Numeric-coded levels (60, 70, ..., 100) are ordered numerically, so the reference is the
    smallest value rather than the lexicographically first label ("100").
    """
    string_values = values.astype("string")
    categories = _ordered_unique_level_strings(string_values, column)
    return pd.Categorical(string_values, categories=categories)


def _as_float_covariate(values: pd.Series) -> pd.Series:
    """A numeric covariate as float64; nullable Int64/Float64/boolean NA becomes NaN.

    Numbers stored as text are read after trimming surrounding spaces, as the typing rule reads them.
    """
    if not is_numeric_dtype(values) and not is_bool_dtype(values):
        values = values.astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(numeric.to_numpy(dtype=float, na_value=np.nan), index=values.index)


def _prepare_cox_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
    strata_columns: Sequence[str] | None = None,
    event_positive_value: Any = None,
    *,
    drop_missing_covariates: bool = True,
) -> pd.DataFrame:
    required_columns = list(dict.fromkeys([*covariates, *(strata_columns or [])]))
    frame = _cohort_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        event_positive_value=event_positive_value,
        extra_columns=required_columns,
        drop_missing_extra_columns=drop_missing_covariates,
    )
    source_row_index = _frame_source_row_index(frame)
    for column in covariates:
        if column in categorical_covariates:
            frame[column] = _cox_categorical(frame[column], column)
        else:
            frame[column] = _as_float_covariate(frame[column])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if drop_missing_covariates:
        base_attrs = dict(frame.attrs)
        keep_mask = ~frame.isna().any(axis=1).to_numpy(dtype=bool)
        frame = frame.loc[keep_mask].copy().reset_index(drop=True)
        frame.attrs.update(base_attrs)
        kept_source_rows = source_row_index[keep_mask]
        frame.attrs["source_row_index"] = kept_source_rows.tolist()
        frame.attrs["row_mask_hash"] = _row_mask_hash(kept_source_rows)
    if frame.empty:
        raise ValueError("No rows remain after removing missing values for the Cox model.")
    return frame


def _validate_cox_covariates(covariates: Sequence[str]) -> None:
    covariate_set = {str(column) for column in covariates}
    overlapping_stage_sets = [
        {"stage", "stage_group"},
        {"pathologic_stage", "stage_group"},
        {"stage", "pathologic_stage"},
    ]
    for overlap in overlapping_stage_sets:
        if overlap <= covariate_set:
            joined = ", ".join(sorted(overlap))
            raise ValueError(
                f"Cox PH cannot fit overlapping stage representations together ({joined}). "
                "Select one stage variable to avoid redundant encoding."
            )


def _normalize_cox_strata_columns(
    strata_columns: Sequence[str] | None,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> list[str]:
    normalized = list(dict.fromkeys(str(column) for column in (strata_columns or []) if str(column).strip()))
    covariate_set = {str(column) for column in covariates}
    categorical_set = {str(column) for column in categorical_covariates}
    overlap = sorted(set(normalized) & covariate_set)
    if overlap:
        joined = ", ".join(overlap)
        raise ValueError(
            f"Strata variables cannot also be included as Cox covariates ({joined}). "
            "Use each variable only once in the Cox specification."
        )
    categorical_overlap = sorted(set(normalized) & categorical_set)
    if categorical_overlap:
        joined = ", ".join(categorical_overlap)
        raise ValueError(
            f"Strata variables cannot also be marked as categorical covariates ({joined}). "
            "Use each variable only once in the Cox specification."
        )
    return normalized


def _build_cox_formula(time_column: str, covariates: Sequence[str], categorical_covariates: Sequence[str]) -> str:
    if not covariates:
        raise ValueError("Select at least one covariate for the Cox PH model.")
    terms: list[str] = []
    for column in covariates:
        if column in categorical_covariates:
            terms.append(f"C({quote_name(column)})")
        else:
            terms.append(quote_name(column))
    return f"{quote_name(time_column)} ~ {' + '.join(terms)}"


def _is_formula_safe_name(name: str) -> bool:
    """A column name that round-trips through a patsy ``Q("...")`` term unchanged."""
    text = str(name)
    return '"' not in text and "\\" not in text and text.isprintable()


def _cox_model_frame(
    frame: pd.DataFrame,
    time_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    """The data, formula, and alias map the Cox model is fitted on.

    Plain names are used as they are. When a name holds a quote, a backslash, or a line
    break (which the formula parser mangles or rejects), every term is fitted on a
    formula-safe alias and ``column_by_alias`` maps the fitted terms back to the column names.
    """
    names = [time_column, *covariates]
    if all(_is_formula_safe_name(name) for name in names):
        return frame, _build_cox_formula(time_column, covariates, categorical_covariates), {}
    alias_by_column = {column: f"__cox_term_{index}" for index, column in enumerate(covariates)}
    time_alias = "__cox_time"
    model_frame = pd.DataFrame({time_alias: frame[time_column]}, index=frame.index)
    for column, alias in alias_by_column.items():
        model_frame[alias] = frame[column]
    categorical_set = set(categorical_covariates)
    fit_formula = _build_cox_formula(
        time_alias,
        [alias_by_column[column] for column in covariates],
        [alias_by_column[column] for column in covariates if column in categorical_set],
    )
    return model_frame, fit_formula, {alias: str(column) for column, alias in alias_by_column.items()}


def _cox_design_condition_number(exog: Any) -> float | None:
    try:
        design = np.asarray(exog, dtype=float)
    except Exception:
        return None
    if design.ndim != 2 or design.size == 0 or not np.isfinite(design).all():
        return None
    try:
        condition_number = float(np.linalg.cond(design))
    except Exception:
        return None
    if not math.isfinite(condition_number):
        return None
    return condition_number


def _normalize_category_text(value: Any) -> tuple[str, str]:
    """Lower-case words of a label with punctuation turned into single spaces, and the same without spaces.

    Letters and digits of every script count as word characters, so non-Latin labels keep their text.
    """
    text = str(value).strip().lower()
    normalized = re.sub(r"[\W_]+", " ", text).strip()
    compact = normalized.replace(" ", "")
    return normalized, compact


# Labels that stand for a missing or unusable value; they sort after every other level.
_UNKNOWN_LEVEL_TEXTS = frozenset(
    {
        "unknown",
        "unk",
        "missing",
        "not available",
        "na",
        "n a",
        "nan",
        "not applicable",
        "not reported",
        "not evaluated",
        "not assessed",
        "not specified",
        "unspecified",
        "undetermined",
        "indeterminate",
        "discrepancy",
    }
)
_PLAIN_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
# AJCC/FIGO stage tokens after "stage" (or bare Roman numerals): numeral, sub-stage letter, digit.
_ROMAN_STAGE_PATTERN = re.compile(r"(0|iv|i{1,3})([abc]?)(\d?)")
_ARABIC_STAGE_PATTERN = re.compile(r"([0-4])([abc]?)(\d?)")
_ROMAN_STAGE_VALUES = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4}
# Negations may follow a prefix ("EGFR non-mutated", "MGMT unmethylated").
_NEGATED_MUTATION_LEVEL_PATTERN = re.compile(r"(?:^|\s)(?:non|not|no|un)\s?(?:mutated|mutant|mutation)\b")
_NEGATED_WILDTYPE_LEVEL_PATTERN = re.compile(r"(?:^|\s)(?:non|not|no)\s?(?:wild\s?type|wt)\b")
_NEGATED_FINDING_LEVEL_PATTERN = re.compile(
    r"(?:^|\s)(?:non|not|no|un)\s?(?:detected|detectable|expressed|expression|methylated|amplified|amplification|altered)\b"
)
_NEGATIVE_LEVEL_TEXTS = frozenset({"negative", "neg", "no", "absent", "control", "reference", "baseline", "normal", "none"})
_POSITIVE_LEVEL_TEXTS = frozenset({"positive", "pos", "yes", "present", "case", "abnormal"})
# A label ending in a sign reads as positive or negative ("ER+", "HER2-", "PD-L1(+)", "Signature-").
_TRAILING_SIGN_PATTERN = re.compile(r"(.*[^\W_])\s*\(?([+＋\-−–])\)?", re.DOTALL)
_BARE_EXPOSURE_LEVELS = frozenset({"never", "former", "current", "ex"})
_ORDINAL_LEVEL_RANKS = {
    "low": 0,
    "lower": 0,
    "mild": 0,
    "intermediate": 1,
    "medium": 1,
    "mid": 1,
    "middle": 1,
    "moderate": 1,
    "high": 2,
    "higher": 2,
    "severe": 2,
}
# Group labels written by derive_group_column's percentile splits.
_PERCENTILE_GROUP_PATTERN = re.compile(r"(?:at )?(below|above) .*percentile threshold")
_DIGIT_RUN_PATTERN = re.compile(r"(\d+)")


def _plain_number(text: str) -> float | None:
    stripped = text.strip()
    if not _PLAIN_NUMBER_PATTERN.fullmatch(stripped):
        return None
    number = float(stripped)
    return number if math.isfinite(number) else None


def _exposure_rank(tokens: Sequence[str], compact: str) -> int | None:
    """Never (0) < former (1) < current (2) exposure, e.g. smoking status."""
    token_set = set(tokens)
    if "never" in token_set or "nonsmoker" in compact:
        return 0
    if token_set & {"former", "formerly", "ex", "reformed", "quit", "past", "previous"} or compact.startswith("exsmok"):
        return 1
    if token_set & {"current", "currently", "active", "smoker"}:
        return 2
    return None


def _stage_rank(compact: str) -> tuple[int, int, int] | None:
    """(numeral, sub-stage letter, digit) of a stage label such as "Stage IA2", "IIIB", or "Stage 2A"."""
    token = compact
    prefixed = token.startswith("stage")
    if prefixed:
        token = token[5:]
    if not token:
        return None
    match = _ROMAN_STAGE_PATTERN.fullmatch(token)
    if match is not None:
        numeral = _ROMAN_STAGE_VALUES[match.group(1)]
    else:
        # Arabic numerals read as stages only after the word "stage".
        match = _ARABIC_STAGE_PATTERN.fullmatch(token) if prefixed else None
        if match is None:
            return None
        numeral = int(match.group(1))
    letter = "abc".index(match.group(2)) + 1 if match.group(2) else 0
    digit = int(match.group(3)) if match.group(3) else 0
    return numeral, letter, digit


def _trailing_sign_polarity(text: str) -> int | None:
    """0 for a label ending in a minus sign, 1 for a plus sign, None otherwise (and for combinations like "ER+/PR-")."""
    stripped = text.strip()
    if "/" in stripped:
        return None
    match = _TRAILING_SIGN_PATTERN.fullmatch(stripped)
    if match is None:
        return None
    return 1 if match.group(2) in {"+", "＋"} else 0


def _ordinal_level_rank(normalized: str, tokens: Sequence[str]) -> int | None:
    """Low (0) < intermediate (1) < high (2), including the percentile-split group labels."""
    if normalized == "rest":
        return 0
    if normalized == "between percentile thresholds":
        return 1
    percentile = _PERCENTILE_GROUP_PATTERN.fullmatch(normalized)
    if percentile is not None:
        return 0 if percentile.group(1) == "below" else 2
    return _ORDINAL_LEVEL_RANKS.get(tokens[0]) if tokens else None


def _natural_text_key(text: str) -> tuple[tuple[int, int, str], ...]:
    """Sort key that compares digit runs as numbers ("T2" before "T10")."""
    parts: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(_DIGIT_RUN_PATTERN.split(text)):
        if index % 2:
            parts.append((0, int(chunk), ""))
        elif chunk:
            parts.append((1, 0, chunk))
    return tuple(parts)


def _category_reference_rank(raw: str, column: str) -> tuple[int, tuple[Any, ...]]:
    """Reference-order group and rank of one category label (lower comes first)."""
    normalized, compact = _normalize_category_text(raw)
    column_text, column_compact = _normalize_category_text(column)
    if not normalized or normalized in _UNKNOWN_LEVEL_TEXTS or "unknown" in normalized or "missing" in normalized:
        return 9, ()
    number = _plain_number(raw)
    if number is not None:
        # Numeric codes (60, 70, ..., 100) keep numeric order ahead of text levels.
        return 0, (number,)
    tokens = normalized.split()
    exposure_context = any(word in column_compact for word in ("smok", "tobacco", "cigar")) or "smok" in compact
    if exposure_context or normalized in _BARE_EXPOSURE_LEVELS:
        exposure = _exposure_rank(tokens, compact)
        if exposure is not None:
            return 1, (exposure,)
    stage = _stage_rank(compact)
    if stage is not None:
        return 2, stage
    # Negated labels are checked first: "Non-mutated" contains "mutated" but is the wild-type
    # reference, "Non-wildtype" is the mutant level, and "Not detected" is the negative level.
    if _NEGATED_MUTATION_LEVEL_PATTERN.search(normalized):
        return 3, (0,)
    if _NEGATED_WILDTYPE_LEVEL_PATTERN.search(normalized):
        return 3, (1,)
    if _NEGATED_FINDING_LEVEL_PATTERN.search(normalized):
        return 4, (0,)
    if "wildtype" in compact or compact == "wt":
        return 3, (0,)
    if "mutated" in normalized or "mutant" in normalized or "mutation" in normalized or compact == "mut":
        return 3, (1,)
    if normalized in _NEGATIVE_LEVEL_TEXTS:
        return 4, (0,)
    if normalized in _POSITIVE_LEVEL_TEXTS:
        return 4, (1,)
    polarity = _trailing_sign_polarity(raw)
    if polarity is not None:
        return 4, (polarity,)
    if "sex" in column_text or "gender" in column_text:
        if normalized == "female":
            return 5, (0,)
        if normalized == "male":
            return 5, (1,)
    ordinal = _ordinal_level_rank(normalized, tokens)
    if ordinal is not None:
        return 6, (ordinal,)
    return 8, ()


def _category_reference_sort_key(column: str, value: Any) -> tuple[Any, ...]:
    """Sort key that puts the natural reference level of a categorical variable first.

    Clinical conventions come first (numeric codes in numeric order, never < former < current
    exposure, AJCC stages, wild type before mutant, negative before positive, female before
    male, low < intermediate < high), then the remaining labels in natural text order, and
    unknown-like labels last. Labels that normalise alike ("<65" and ">=65") are ordered by
    their exact text, so the order never depends on the order of the rows.
    """
    raw = str(value)
    group, rank = _category_reference_rank(raw, column)
    normalized, _ = _normalize_category_text(raw)
    return (group, rank, _natural_text_key(normalized), raw)


def _ordered_reference_categories(values: Sequence[Any], column: str) -> list[str]:
    unique_values = [str(item) for item in values]
    return sorted(unique_values, key=lambda item: _category_reference_sort_key(column, item))


def _reference_levels(frame: pd.DataFrame, categorical_covariates: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for column in categorical_covariates:
        categories = [str(item) for item in frame[column].cat.categories]
        if categories:
            output[column] = categories[0]
    return output


def _clean_term(
    term: str,
    reference_levels: dict[str, str],
    column_by_alias: dict[str, str] | None = None,
) -> tuple[str, str, str | None]:
    """Variable, display label, and reference level of a fitted design term.

    ``column_by_alias`` maps the formula-safe aliases the model was fitted on back to the
    column names, so names with quotes, backslashes, or line breaks display as written.
    """
    aliases = column_by_alias or {}
    categorical_match = TERM_CATEGORICAL_PATTERN.match(term)
    if categorical_match:
        variable = aliases.get(categorical_match.group("var"), categorical_match.group("var"))
        level = categorical_match.group("level")
        reference = reference_levels.get(variable)
        label = f"{variable}: {level} vs {reference}" if reference else f"{variable}: {level}"
        return variable, label, reference
    numeric_match = TERM_NUMERIC_PATTERN.match(term)
    if numeric_match:
        variable = aliases.get(numeric_match.group("var"), numeric_match.group("var"))
        return variable, variable, None
    return term, term, None


def _harrell_c_index_counts(
    time_values: np.ndarray,
    event_values: np.ndarray,
    risk_score: np.ndarray,
) -> tuple[float, float]:
    event_mask = event_values == 1
    if int(np.sum(event_mask)) == 0:
        return 0.0, 0.0

    order = np.argsort(time_values, kind="mergesort")
    unique_risks, inverse = np.unique(risk_score.astype(float), return_inverse=True)
    ranks = inverse + 1  # Fenwick tree is 1-indexed.
    tree = np.zeros(len(unique_risks) + 1, dtype=np.int64)

    def _fenwick_add(index: int, delta: int) -> None:
        while index < tree.size:
            tree[index] += delta
            index += index & -index

    def _fenwick_prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    concordant = 0.0
    comparable = 0.0
    later_count = 0
    cursor = len(order) - 1

    while cursor >= 0:
        time_value = time_values[order[cursor]]
        group_end = cursor
        while cursor >= 0 and time_values[order[cursor]] == time_value:
            cursor -= 1
        group_indices = order[cursor + 1:group_end + 1]

        # Harrell/sksurv/R convention: a subject censored at the same time as
        # an event is known to outlive it, so the pair is comparable. Add the
        # tied censored subjects before scoring the tied events.
        for idx in group_indices:
            if event_values[idx] != 1:
                _fenwick_add(int(ranks[idx]), 1)
                later_count += 1

        if later_count:
            for idx in group_indices:
                if event_values[idx] != 1:
                    continue
                rank = int(ranks[idx])
                lower = _fenwick_prefix(rank - 1)
                equal = _fenwick_prefix(rank) - lower
                concordant += float(lower) + 0.5 * float(equal)
                comparable += float(later_count)

        for idx in group_indices:
            if event_values[idx] == 1:
                _fenwick_add(int(ranks[idx]), 1)
                later_count += 1

    return concordant, comparable


def _harrell_c_index(time_values: np.ndarray, event_values: np.ndarray, risk_score: np.ndarray) -> float | None:
    concordant, comparable = _harrell_c_index_counts(time_values, event_values, risk_score)
    if comparable <= 0.0:
        return None
    return concordant / comparable


def _harrell_c_index_bootstrap_ci(
    time_values: np.ndarray,
    event_values: np.ndarray,
    risk_score: np.ndarray,
    *,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_seed: int = 20260311,
) -> dict[str, float | None]:
    if n_bootstrap <= 1:
        return {"c_index_std": None, "c_index_ci_lower": None, "c_index_ci_upper": None}

    time_array = np.asarray(time_values, dtype=float)
    event_array = np.asarray(event_values, dtype=int)
    risk_array = np.asarray(risk_score, dtype=float)
    n_obs = int(time_array.shape[0])
    if n_obs < 10 or int(event_array.sum()) < 5:
        warnings.warn(
            "C-index bootstrap CI skipped because the analyzable cohort is too small for stable internal resampling.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"c_index_std": None, "c_index_ci_lower": None, "c_index_ci_upper": None}

    rng = np.random.default_rng(int(random_seed))
    boot_scores: list[float] = []
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n_obs, size=n_obs)
        sample_events = event_array[sample_idx]
        if int(sample_events.sum()) == 0:
            continue
        score = _harrell_c_index(time_array[sample_idx], sample_events, risk_array[sample_idx])
        if score is not None and math.isfinite(score):
            boot_scores.append(float(score))

    min_valid_boot = max(10, int(n_bootstrap * 0.3))
    if len(boot_scores) < min_valid_boot:
        warnings.warn(
            f"C-index bootstrap CI skipped: only {len(boot_scores)} valid bootstrap samples out of {n_bootstrap} requested (threshold: {min_valid_boot}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"c_index_std": None, "c_index_ci_lower": None, "c_index_ci_upper": None}

    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(np.asarray(boot_scores, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    std = float(np.std(np.asarray(boot_scores, dtype=float), ddof=1)) if len(boot_scores) > 1 else None
    return {
        "c_index_std": std,
        "c_index_ci_lower": float(lower),
        "c_index_ci_upper": float(upper),
    }


def _cox_complete_case_frame(preview_frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Rows with every selected Cox input present, keeping the source-row bookkeeping."""
    columns = list(dict.fromkeys(columns))
    complete_case_mask = ~preview_frame[columns].isna().any(axis=1).to_numpy(dtype=bool)
    frame = preview_frame.loc[complete_case_mask].copy().reset_index(drop=True)
    frame.attrs.update(dict(preview_frame.attrs))
    complete_case_source_rows = _frame_source_row_index(preview_frame)[complete_case_mask]
    frame.attrs["source_row_index"] = complete_case_source_rows.tolist()
    frame.attrs["row_mask_hash"] = _row_mask_hash(complete_case_source_rows)
    if frame.empty:
        raise ValueError("No rows remain after removing missing values for the Cox model.")
    return frame


def _cox_coefficient_rows(
    results: Any,
    reference_levels: dict[str, str],
    stability_snapshot: dict[str, Any],
    column_by_alias: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    """Hazard-ratio table rows, fitted risk scores, and the partial log-likelihood.

    A fit with any non-finite estimate is rejected with a stability explanation.
    """
    conf_int = np.asarray(results.conf_int(), dtype=float)
    param_vector = np.asarray(results.params, dtype=float)
    bse_vector = np.asarray(results.bse, dtype=float)
    z_vector = np.asarray(results.tvalues, dtype=float)
    p_vector = np.asarray(results.pvalues, dtype=float)
    llf_value = float(results.llf) if results.llf is not None else np.nan
    risk_score = np.asarray(results.model.exog @ results.params, dtype=float)
    fit_components = [param_vector, conf_int.reshape(-1), bse_vector, z_vector, p_vector, risk_score]
    if (not np.isfinite(llf_value)) or any(not np.isfinite(component).all() for component in fit_components):
        raise ValueError(_cox_nonfinite_estimate_message(stability_snapshot))

    model_rows: list[dict[str, Any]] = []
    for idx, term in enumerate(results.model.exog_names):
        variable, label, reference = _clean_term(term, reference_levels, column_by_alias)
        beta = float(param_vector[idx])
        model_rows.append(
            {
                "Variable": variable,
                "Label": label,
                "Reference": reference,
                "Beta": beta,
                "Hazard ratio": _safe_exp_or_none(beta),
                "CI lower": _safe_exp_or_none(conf_int[idx, 0]),
                "CI upper": _safe_exp_or_none(conf_int[idx, 1]),
                "SE": float(bse_vector[idx]),
                "Z": float(z_vector[idx]),
                "P value": float(p_vector[idx]),
            }
        )
    return model_rows, risk_score, llf_value


def _cox_ph_diagnostics(
    results: Any,
    frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    strata_codes: np.ndarray | None,
    reference_levels: dict[str, str],
    stability_snapshot: dict[str, Any],
    column_by_alias: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Grambsch-Therneau PH tests (per term and global) and scaled Schoenfeld residual panels."""
    raw_times = frame[time_column].to_numpy(dtype=float)
    schoenfeld = _efron_schoenfeld_residuals(
        np.asarray(results.model.exog, dtype=float),
        raw_times,
        frame[event_column].to_numpy(dtype=int),
        np.asarray(results.params, dtype=float),
        strata_codes,
    )
    cov_matrix = np.atleast_2d(np.asarray(results.cov_params(), dtype=float))
    if schoenfeld.shape[1] != cov_matrix.shape[0] or not np.isfinite(cov_matrix).all():
        raise ValueError(_cox_nonfinite_estimate_message(stability_snapshot))
    positive_times = raw_times[raw_times > 0]
    # log(0) is undefined; place time-0 events just below the smallest positive time.
    floor_time = float(positive_times.min()) / 2.0 if positive_times.size else 1.0
    log_time = np.log(np.where(raw_times > 0, raw_times, floor_time))
    scaled_schoenfeld = schoenfeld @ cov_matrix
    scaled_schoenfeld *= float(max(int(frame[event_column].sum()), 1))
    ph_test = _cox_grambsch_therneau_test(schoenfeld, cov_matrix, log_time)
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_plot_data: list[dict[str, Any]] = []
    for idx, term in enumerate(results.model.exog_names):
        valid = np.isfinite(scaled_schoenfeld[:, idx]) & np.isfinite(log_time)
        rho = ph_test["term_rho"][idx] if idx < len(ph_test["term_rho"]) else None
        chi_square = ph_test["term_statistics"][idx] if idx < len(ph_test["term_statistics"]) else None
        p_value = ph_test["term_p_values"][idx] if idx < len(ph_test["term_p_values"]) else None
        _, label, _ = _clean_term(term, reference_levels, column_by_alias)
        diagnostic_rows.append(
            {
                "Term": label,
                "Schoenfeld rho": _safe_float(rho),
                "Chi-square": _safe_float(chi_square),
                "P value": _safe_float(p_value),
            }
        )
        if int(valid.sum()) >= 4:
            x_values = log_time[valid].astype(float)
            y_values = scaled_schoenfeld[valid, idx].astype(float)
            order = np.argsort(x_values, kind="mergesort")
            x_sorted = x_values[order]
            y_sorted = y_values[order]
            trend_y = _diagnostic_lowess_trend(y_sorted, x_sorted)
            diagnostic_plot_data.append(
                {
                    "term": label,
                    "schoenfeld_rho": _safe_float(rho),
                    "p_value": _safe_float(p_value),
                    "log_time": [_safe_float(value) for value in x_sorted.tolist()],
                    "residual": [_safe_float(value) for value in y_sorted.tolist()],
                    "trend_log_time": [_safe_float(value) for value in x_sorted.tolist()],
                    "trend_residual": [_safe_float(value) for value in np.asarray(trend_y, dtype=float).tolist()],
                }
            )
    if ph_test.get("p_value") is not None:
        diagnostic_rows.append(
            {
                "Term": "Global PH test (Grambsch-Therneau)",
                "Schoenfeld rho": None,
                "Chi-square": _safe_float(ph_test.get("statistic")),
                "P value": _safe_float(ph_test["p_value"]),
                "_kind": "global",
            }
        )
    return diagnostic_rows, diagnostic_plot_data, ph_test


def _cox_martingale_screen(
    results: Any,
    frame: pd.DataFrame,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str],
    *,
    time_column: str,
    event_column: str,
    strata_codes: np.ndarray | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Martingale residual panels for continuous covariates, or a note on why there are none."""
    categorical_names = {str(value) for value in categorical_covariates}
    candidate_terms = [
        str(covariate)
        for covariate in covariates
        if covariate not in categorical_names
        and covariate in frame.columns
        and _is_martingale_screen_candidate(frame[covariate])
    ]
    martingale_residuals = _efron_martingale_residuals(
        np.asarray(results.model.exog, dtype=float),
        frame[time_column].to_numpy(dtype=float),
        frame[event_column].to_numpy(dtype=int),
        np.asarray(results.params, dtype=float),
        strata_codes,
    )
    plot_data = _cox_martingale_plot_data(
        frame,
        martingale_residuals,
        covariates,
        categorical_covariates,
    )
    if plot_data:
        return plot_data, None
    if candidate_terms:
        return plot_data, "Martingale residual screening was unavailable for this fit."
    return plot_data, "No continuous covariates were eligible for martingale screening."


def _cox_likelihood_ratio_test(results: Any, llf_value: float, k_params: int) -> dict[str, Any]:
    """Overall likelihood-ratio test of the fitted model against the null model."""
    llnull_raw = getattr(results, "llnull", None)
    if llnull_raw is None:
        try:
            llnull_raw = results.model.loglike(np.zeros(k_params, dtype=float))
        except MemoryError:
            raise
        except Exception:
            llnull_raw = None
    llnull_value = _safe_float(llnull_raw)
    lr_statistic = None
    lr_pvalue = None
    lr_note = None
    if llnull_value is not None and math.isfinite(llf_value) and k_params > 0:
        lr_candidate = -2.0 * (llnull_value - llf_value)
        if math.isfinite(lr_candidate) and lr_candidate >= 0.0:
            lr_statistic = float(lr_candidate)
            lr_pvalue = float(stats.chi2.sf(lr_statistic, df=k_params))
        else:
            lr_note = (
                "Overall likelihood-ratio test was not reportable because the null-model comparison "
                "returned an invalid negative chi-square candidate."
            )
    elif llnull_value is None:
        lr_note = "Overall likelihood-ratio test was not reportable because the null-model log-likelihood was unavailable."
    return {
        "null_log_likelihood": llnull_value,
        "lr_statistic": _safe_float(lr_statistic),
        "lr_pvalue": _safe_float(lr_pvalue),
        "lr_note": lr_note,
    }


def _cox_apparent_c_index(
    frame: pd.DataFrame,
    risk_score: np.ndarray,
    *,
    time_column: str,
    event_column: str,
    stratified: bool,
) -> dict[str, Any]:
    """Apparent Harrell C-index with a bootstrap CI; not reported for a stratified fit."""
    if stratified:
        return {
            "c_index": None,
            "c_index_std": None,
            "c_index_ci_lower": None,
            "c_index_ci_upper": None,
            "c_index_label": "C-index not reported for stratified Cox",
            "evaluation_mode": "stratified_not_reported",
        }
    time_values = frame[time_column].to_numpy(dtype=float)
    event_values = frame[event_column].to_numpy(dtype=int)
    c_index = _harrell_c_index(time_values, event_values, risk_score.astype(float))
    c_index_ci = _harrell_c_index_bootstrap_ci(time_values, event_values, risk_score.astype(float))
    return {
        "c_index": c_index,
        **c_index_ci,
        "c_index_label": "Apparent C-index (training cohort)",
        "evaluation_mode": "apparent",
    }


@user_input_boundary
def compute_cox_analysis(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str] | None = None,
    strata_columns: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> dict[str, Any]:
    covariates = list(dict.fromkeys(covariates))
    _validate_cox_covariates(covariates)
    categorical_covariates = _resolve_cox_categorical_covariates(df, covariates, categorical_covariates)
    strata_columns = _normalize_cox_strata_columns(strata_columns, covariates, categorical_covariates)
    preview_frame = _prepare_cox_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        covariates=covariates,
        categorical_covariates=categorical_covariates,
        strata_columns=strata_columns,
        event_positive_value=event_positive_value,
        drop_missing_covariates=False,
    )
    frame = _cox_complete_case_frame(preview_frame, [*covariates, *strata_columns])
    for column in categorical_covariates:
        frame[column] = _cox_categorical(frame[column], column)
    formula = _build_cox_formula(time_column, covariates, categorical_covariates)
    status = frame[event_column].astype(int).to_numpy()
    strata_payload = _build_cox_strata_payload(frame, strata_columns)
    stability_snapshot = _cox_stability_snapshot(
        frame,
        event_column,
        covariates,
        categorical_covariates,
        strata_columns=strata_columns,
        strata_row_labels=strata_payload["row_labels"],
    )
    constant_design_message = _cox_constant_design_message(stability_snapshot)
    if constant_design_message:
        raise ValueError(constant_design_message)
    _reject_oversized_cox_design(stability_snapshot)
    model_frame, fit_formula, column_by_alias = _cox_model_frame(frame, time_column, covariates, categorical_covariates)
    model = PHReg.from_formula(fit_formula, data=model_frame, status=status, strata=strata_payload["codes"], ties="efron")
    design_condition_number = _cox_design_condition_number(getattr(model, "exog", None))
    results = _fit_cox_model(model, stability_snapshot)

    reference_levels = _reference_levels(frame, categorical_covariates)
    model_rows, risk_score, llf_value = _cox_coefficient_rows(results, reference_levels, stability_snapshot, column_by_alias)
    diagnostic_rows, diagnostic_plot_data, global_ph_screen = _cox_ph_diagnostics(
        results,
        frame,
        time_column=time_column,
        event_column=event_column,
        strata_codes=strata_payload["codes"],
        reference_levels=reference_levels,
        stability_snapshot=stability_snapshot,
        column_by_alias=column_by_alias,
    )
    martingale_plot_data, martingale_note = _cox_martingale_screen(
        results,
        frame,
        covariates,
        categorical_covariates,
        time_column=time_column,
        event_column=event_column,
        strata_codes=strata_payload["codes"],
    )

    n_obs = int(frame.shape[0])
    outcome_rows = int(preview_frame.shape[0])
    n_events = int(frame[event_column].sum())
    k_params = len(results.params)
    strata_snapshot = dict(stability_snapshot.get("strata_snapshot") or {})
    c_index_fields = _cox_apparent_c_index(
        frame,
        risk_score,
        time_column=time_column,
        event_column=event_column,
        stratified=bool(strata_columns),
    )
    c_index = c_index_fields["c_index"]

    model_stats = {
        "n": n_obs,
        "outcome_rows": outcome_rows,
        "dropped_rows": int(outcome_rows - n_obs),
        "row_mask_hash": str(frame.attrs.get("row_mask_hash") or ""),
        "outcome_row_mask_hash": str(preview_frame.attrs.get("row_mask_hash") or ""),
        "dropped_nonpositive_time_rows": int(frame.attrs.get("dropped_nonpositive_time_rows", 0)),
        # Outcome-valid rows whose selected covariates or strata held +/-Inf (coerced to missing
        # before complete-case filtering); rows with +/-Inf time never became outcome-valid.
        "rows_with_infinite_values": int(frame.attrs.get("rows_with_infinite_extra_values", 0)),
        "rows_with_infinite_outcome_values": int(frame.attrs.get("rows_with_infinite_outcome", 0)),
        "time_column_note": frame.attrs.get("time_column_note"),
        "events": n_events,
        "parameters": k_params,
        "events_per_parameter": float(n_events / k_params) if k_params else None,
        "partial_log_likelihood": _safe_float(llf_value),
        **_cox_likelihood_ratio_test(results, llf_value, k_params),
        "global_ph_statistic": _safe_float(global_ph_screen.get("statistic")),
        "global_ph_df": _safe_float(global_ph_screen.get("df")),
        "global_ph_pvalue": _safe_float(global_ph_screen.get("p_value")),
        "global_ph_terms_tested": int(global_ph_screen.get("terms_tested") or 0),
        "global_ph_method": global_ph_screen.get("method"),
        "martingale_terms": [panel["term"] for panel in martingale_plot_data],
        "martingale_note": martingale_note,
        "aic": _safe_float(-2 * llf_value + 2 * k_params),
        # For a partial likelihood the effective sample size is the number of events,
        # as in R's BIC(coxph) (survival >= 2.41) and Volinsky & Raftery (2000).
        "bic": _safe_float(-2 * llf_value + k_params * np.log(max(n_events, 1))),
        "bic_sample_size": "events",
        "c_index": _safe_float(c_index),
        "c_index_std": _safe_float(c_index_fields["c_index_std"]),
        "c_index_ci_lower": _safe_float(c_index_fields["c_index_ci_lower"]),
        "c_index_ci_upper": _safe_float(c_index_fields["c_index_ci_upper"]),
        "c_index_ci_level": 0.95 if c_index is not None else None,
        "c_index_ci_method": "bootstrap_percentile_fixed_score" if c_index is not None else None,
        "c_index_label": c_index_fields["c_index_label"],
        "evaluation_mode": c_index_fields["evaluation_mode"],
        "tie_method": "efron",
        "strata_columns": list(strata_columns),
        "n_strata": strata_snapshot.get("n_strata"),
        "zero_event_strata_count": int(strata_snapshot.get("zero_event_strata_count") or 0),
        "sparse_event_strata_count": int(strata_snapshot.get("sparse_event_strata_count") or 0),
        "high_cardinality_strata_columns": list(strata_snapshot.get("high_cardinality_columns") or []),
        "high_cardinality_numeric_strata_columns": list(strata_snapshot.get("high_cardinality_numeric_columns") or []),
        "design_condition_number": _safe_float(design_condition_number),
        "design_condition_warning_threshold": COX_CONDITION_NUMBER_WARN_THRESHOLD,
    }
    scientific_summary = _cox_scientific_summary(
        model_rows=model_rows,
        diagnostic_rows=diagnostic_rows,
        model_stats=model_stats,
        categorical_alerts=_cox_categorical_stability_alerts(frame, categorical_covariates),
    )
    duplicate_caution = duplicate_identifier_caution(df)
    if duplicate_caution and isinstance(scientific_summary, dict):
        scientific_summary.setdefault("cautions", []).append(duplicate_caution)

    return {
        "formula": formula,
        "results_table": model_rows,
        "diagnostics_table": diagnostic_rows,
        "diagnostics_plot_data": diagnostic_plot_data,
        "martingale_plot_data": martingale_plot_data,
        "model_stats": {
            **model_stats,
            "apparent_c_index": _safe_float(c_index),
        },
        "categorical_covariates": categorical_covariates,
        "strata_columns": list(strata_columns),
        "scientific_summary": scientific_summary,
    }


@user_input_boundary
def preview_cox_analysis_inputs(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    covariates: Sequence[str],
    categorical_covariates: Sequence[str] | None = None,
    strata_columns: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> dict[str, Any]:
    if not covariates:
        raise ValueError("Select at least one covariate for the Cox model.")
    covariates = list(dict.fromkeys(covariates))
    _validate_cox_covariates(covariates)
    categorical_covariates = _resolve_cox_categorical_covariates(df, covariates, categorical_covariates)
    strata_columns = _normalize_cox_strata_columns(strata_columns, covariates, categorical_covariates)
    preview_frame = _prepare_cox_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        covariates=covariates,
        categorical_covariates=categorical_covariates,
        strata_columns=strata_columns,
        event_positive_value=event_positive_value,
        drop_missing_covariates=False,
    )
    input_columns = list(dict.fromkeys([*covariates, *strata_columns]))
    missing_by_covariate: list[dict[str, Any]] = []
    for column in input_columns:
        missing_count = int(preview_frame[column].isna().sum())
        if missing_count > 0:
            missing_by_covariate.append({"column": column, "missing_rows": missing_count})
    complete_case = _cox_complete_case_frame(preview_frame, input_columns)
    # Levels are rebuilt on the complete-case rows, as the fit does, so the preview names the
    # reference level the fit will use instead of a level whose rows were all dropped.
    for column in categorical_covariates:
        complete_case[column] = _cox_categorical(complete_case[column], column)
    strata_payload = _build_cox_strata_payload(complete_case, strata_columns)
    stability_snapshot = _cox_stability_snapshot(
        complete_case,
        event_column,
        covariates,
        categorical_covariates,
        strata_columns=strata_columns,
        strata_row_labels=strata_payload["row_labels"],
    )
    missing_by_covariate.sort(key=lambda item: (-int(item["missing_rows"]), str(item["column"])))
    strata_snapshot = dict(stability_snapshot.get("strata_snapshot") or {})
    return {
        "outcome_rows": int(preview_frame.shape[0]),
        "analyzable_rows": int(complete_case.shape[0]),
        "dropped_rows": int(preview_frame.shape[0] - complete_case.shape[0]),
        "events": int(stability_snapshot["events"]),
        "estimated_parameters": int(stability_snapshot["estimated_parameters"]),
        "events_per_parameter": stability_snapshot["events_per_parameter"],
        "row_mask_hash": str(complete_case.attrs.get("row_mask_hash") or ""),
        "outcome_row_mask_hash": str(preview_frame.attrs.get("row_mask_hash") or ""),
        "covariates": list(covariates),
        "categorical_covariates": list(categorical_covariates),
        "strata_columns": list(strata_columns),
        "missing_by_covariate": missing_by_covariate,
        "stability_warnings": list(stability_snapshot["stability_warnings"]),
        "risky_levels": list(stability_snapshot["risky_levels"]),
        "n_strata": strata_snapshot.get("n_strata"),
        "zero_event_strata_count": int(strata_snapshot.get("zero_event_strata_count") or 0),
        "sparse_event_strata_count": int(strata_snapshot.get("sparse_event_strata_count") or 0),
    }


@user_input_boundary
def compute_cohort_table(df: pd.DataFrame, variables: Sequence[str], group_column: str | None = None) -> dict[str, Any]:
    variables = list(dict.fromkeys(variable for variable in variables if variable != group_column))
    if not variables:
        raise ValueError("Select at least one variable for the cohort summary table.")
    overall_label = "Overall (grouped subset)" if group_column else "Overall"
    columns = [*variables]
    if group_column:
        columns.append(group_column)
    _require_dataframe_columns(df, columns)
    frame = df[columns].copy()

    # Groups are row masks over one frame, so every variable is canonicalized once over the
    # whole cohort and the per-group counts slice the same labels (a group whose values happen
    # to be integer-valued must still match the "1.0" level of the cohort).
    group_masks: OrderedDict[str, np.ndarray] = OrderedDict()
    if group_column:
        string_group = _canonical_level_strings(frame[group_column])
        keep = string_group.notna().to_numpy(dtype=bool)
        frame = frame.loc[keep].copy()
        string_group = string_group[keep]
        group_masks[overall_label] = np.ones(len(frame), dtype=bool)
        group_labels = _sorted_group_labels(string_group, group_column)
        for label in group_labels:
            group_masks[label] = (string_group == label).to_numpy(dtype=bool, na_value=False)
    else:
        group_masks[overall_label] = np.ones(len(frame), dtype=bool)
        group_labels = []

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "Variable": "Cohort size",
            "Statistic": "N",
            **{label: int(mask.sum()) for label, mask in group_masks.items()},
        }
    )

    for variable in variables:
        series = frame[variable]
        is_boolean = is_bool_dtype(series)
        is_binary_numeric = (not is_boolean) and is_numeric_dtype(series) and _is_binary_numeric_series(series)
        missing_row = {
            "Variable": variable,
            "Statistic": "Missing",
            **{label: int(series[mask].isna().sum()) for label, mask in group_masks.items()},
        }
        if is_numeric_dtype(series) and not is_binary_numeric and not is_boolean:
            row = {"Variable": variable, "Statistic": "Mean ± SD | Median [IQR]"}
            for label, mask in group_masks.items():
                values = pd.to_numeric(series[mask], errors="coerce").dropna()
                if values.empty:
                    row[label] = "NA"
                    continue
                sd_value = values.std(ddof=1)
                # A single observation has no SD; do not print it as 0.00.
                sd_text = f"{sd_value:.2f}" if len(values) > 1 and np.isfinite(sd_value) else "NA"
                row[label] = (
                    f"{values.mean():.2f} ± {sd_text} | "
                    f"{values.median():.2f} [{values.quantile(0.25):.2f}, {values.quantile(0.75):.2f}]"
                )
            rows.append(row)
            rows.append(missing_row)
            continue

        source_series = _canonical_level_strings(series)
        levels = _ordered_level_strings(source_series, variable)
        for level in levels:
            row = {"Variable": variable, "Statistic": str(level)}
            for label, mask in group_masks.items():
                group_values = source_series[mask]
                denominator = int(group_values.notna().sum())
                numerator = int((group_values == level).sum())
                row[label] = f"{numerator} ({100.0 * numerator / denominator:.1f}%)" if denominator else f"{numerator} (NA)"
            rows.append(row)
        rows.append(missing_row)

    return {
        "columns": ["Variable", "Statistic", overall_label, *group_labels],
        "rows": rows,
        "row_mask_hash": _row_mask_hash(frame.index),
        "overall_scope": (
            "Overall summarizes the non-missing grouped subset."
            if group_column
            else "Overall summarizes the full analyzable table cohort."
        ),
    }

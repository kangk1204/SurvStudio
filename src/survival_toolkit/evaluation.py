from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.errors import InternalAnalysisError, UserInputError

# One holdout convention for every model family. Classical ML and deep models
# must be scored on the same evaluation rows to be rank-comparable, so both
# call :func:`stratified_holdout_indices` with the same seed.
DEFAULT_HOLDOUT_FRACTION = 0.3
MIN_SAMPLES_FOR_HOLDOUT = 20
MIN_STRATUM_FOR_HOLDOUT = 4


def metric_name_for_evaluation(evaluation_mode: str) -> str:
    if evaluation_mode == "holdout":
        return "Holdout C-index"
    if evaluation_mode == "holdout_fallback_apparent":
        return "Apparent fallback C-index"
    if evaluation_mode == "repeated_cv":
        return "Repeated-CV mean C-index"
    if evaluation_mode == "repeated_cv_incomplete":
        return "Repeated-CV mean C-index (incomplete)"
    return "Apparent C-index"


def stratified_holdout_indices(
    events: Sequence[Any] | np.ndarray,
    *,
    random_state: int,
    test_size: float = DEFAULT_HOLDOUT_FRACTION,
    min_samples: int = MIN_SAMPLES_FOR_HOLDOUT,
    min_stratum: int = MIN_STRATUM_FOR_HOLDOUT,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(train_positions, eval_positions, mode)`` for a stratified holdout.

    Positions index the cleaned analysis frame. ``mode`` is ``"holdout"`` for a
    real split and ``"apparent"`` when the cohort is too small or an outcome
    stratum is too sparse, in which case both position arrays cover every row.
    """
    from sklearn.model_selection import train_test_split

    event_array = np.asarray(events).astype(int).reshape(-1)
    n_samples = int(event_array.shape[0])
    all_positions = np.arange(n_samples, dtype=int)
    if n_samples < min_samples:
        return all_positions, all_positions, "apparent"
    unique, counts = np.unique(event_array, return_counts=True)
    if unique.size < 2 or int(counts.min()) < min_stratum:
        return all_positions, all_positions, "apparent"
    try:
        train_positions, eval_positions = train_test_split(
            all_positions,
            test_size=test_size,
            random_state=random_state,
            stratify=event_array,
        )
    except ValueError:
        return all_positions, all_positions, "apparent"
    train_positions = np.asarray(train_positions, dtype=int)
    eval_positions = np.asarray(eval_positions, dtype=int)
    if int(event_array[train_positions].sum()) == 0 or int(event_array[eval_positions].sum()) == 0:
        return all_positions, all_positions, "apparent"
    return train_positions, eval_positions, "holdout"


def locked_test_split(
    events: Sequence[Any] | np.ndarray,
    *,
    random_state: int,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the cohort into a development set and an untouched locked test set.

    The locked test set is never used for fitting, preprocessing, tuning, or
    model selection; it is scored exactly once after models are refit on the
    whole development set.
    """
    if not (0.05 <= float(test_fraction) <= 0.5):
        raise ValueError("locked_test_fraction must be between 0.05 and 0.5.")
    dev_positions, test_positions, mode = stratified_holdout_indices(
        events,
        random_state=random_state,
        test_size=float(test_fraction),
    )
    if mode != "holdout":
        raise ValueError(
            "A locked independent test set could not be reserved: the cohort is too small or one "
            "outcome stratum is too sparse for a stratified development/test split."
        )
    return dev_positions, test_positions


def _source_labels(source_row_index: Sequence[Any] | None, positions: np.ndarray) -> list[str]:
    positions = np.asarray(positions, dtype=int).reshape(-1)
    if source_row_index is None:
        labels = [str(int(position)) for position in positions]
    else:
        labels = [str(source_row_index[int(position)]) for position in positions]
    return sorted(labels)


def evaluation_split_fingerprint(
    source_row_index: Sequence[Any] | None,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    kind: str,
) -> str:
    """Hash the exact evaluation design (which source rows were scored in which split).

    Two results with the same fingerprint were trained and evaluated on the same
    row partitions of the same stored dataset, so their metrics are directly
    comparable. ``source_row_index`` maps analysis-frame positions back to the
    stored dataset's row labels.
    """
    payload = {
        "kind": str(kind),
        "n_rows": None if source_row_index is None else int(len(source_row_index)),
        "splits": [
            {
                "train": _source_labels(source_row_index, train_positions),
                "eval": _source_labels(source_row_index, eval_positions),
            }
            for train_positions, eval_positions in splits
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest[:16]


def prediction_block(
    row_ids: Sequence[Any],
    time: Sequence[float],
    event: Sequence[int],
    risks: Mapping[str, tuple[Sequence[int] | None, Sequence[float]]],
) -> dict[str, Any] | None:
    """Test-set risk scores of several models on the test patients all of them scored.

    ``row_ids``, ``time`` and ``event`` describe every test patient in order; each model
    gives the positions it scored (None for all) and its risk scores there, higher
    meaning earlier events. Row IDs are the stored dataset's row labels as text, so
    blocks from different model families can be matched patient by patient. Scores that
    do not line up with their positions are an internal error, never a model left out.
    """
    n = len(row_ids)
    aligned: dict[str, dict[int, float]] = {}
    common = set(range(n))
    for name, (positions, values) in risks.items():
        scored = list(range(n)) if positions is None else [int(position) for position in positions]
        values = [float(value) for value in values]
        if len(scored) != len(values):
            raise InternalAnalysisError(
                f"The test-set predictions of {name} could not be paired with the test patients: {len(values)} risk "
                f"scores for {len(scored)} scored patients. This is an internal error; please report it with the server log."
            )
        lookup = {position: value for position, value in zip(scored, values) if np.isfinite(value)}
        aligned[str(name)] = lookup
        common &= set(lookup)
    keep = sorted(common)
    if not aligned or not keep:
        return None
    return {
        "row_ids": [str(row_ids[position]) for position in keep],
        "time": [float(time[position]) for position in keep],
        "event": [int(event[position]) for position in keep],
        "risk": {name: [lookup[position] for position in keep] for name, lookup in aligned.items()},
    }


def _is_list_like(value: Any) -> bool:
    return hasattr(value, "__len__") and hasattr(value, "__iter__") and not isinstance(value, (str, bytes, Mapping))


def _block_numbers(values: Any, *, label: str, field: str, length: int) -> np.ndarray:
    """One list of a prediction block as floats, with its length checked against the row IDs."""
    if not _is_list_like(values):
        raise UserInputError(f"{label}: '{field}' must be a list of numbers, one per row ID.")
    if len(values) != length:
        raise UserInputError(f"{label}: '{field}' has {len(values)} values for {length} row IDs.")
    try:
        numbers = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise UserInputError(f"{label}: '{field}' must contain only numbers.") from exc
    if numbers.shape[0] != length:
        raise UserInputError(f"{label}: '{field}' must be a flat list of numbers, one per row ID.")
    return numbers


def _validated_prediction_block(block: Any, index: int) -> dict[str, Any] | None:
    """A prediction block checked field by field; None for an empty block (skipped)."""
    label = f"Prediction block {index + 1}"
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise UserInputError(f"{label} must be an object with row_ids, time, event and risk.")
    row_ids = block.get("row_ids")
    if row_ids is None or (_is_list_like(row_ids) and len(row_ids) == 0):
        return None
    if not _is_list_like(row_ids):
        raise UserInputError(f"{label}: 'row_ids' must be a list of row labels.")
    missing = [key for key in ("time", "event", "risk") if key not in block]
    if missing:
        raise UserInputError(f"{label} lacks {', '.join(repr(key) for key in missing)}.")
    rows = [str(row) for row in row_ids]
    if len(set(rows)) != len(rows):
        raise UserInputError(f"{label}: 'row_ids' repeats a row label; each test patient must appear once.")
    n = len(rows)
    time = _block_numbers(block["time"], label=label, field="time", length=n)
    if not np.isfinite(time).all():
        raise UserInputError(f"{label}: every 'time' must be a finite number.")
    event_values = _block_numbers(block["event"], label=label, field="event", length=n)
    if not np.isin(event_values, (0.0, 1.0)).all():
        raise UserInputError(f"{label}: every 'event' must be 0 or 1.")
    risk = block["risk"]
    if not isinstance(risk, Mapping) or not risk:
        raise UserInputError(f"{label}: 'risk' must map each model name to its list of risk scores.")
    risks = {
        str(name): _block_numbers(values, label=label, field=f"risk of {name}", length=n)
        for name, values in risk.items()
    }
    return {"row_ids": rows, "time": time, "event": event_values.astype(int), "risk": risks}


def merge_prediction_blocks(blocks: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Time, event and every model's risk on the patients present in all blocks.

    Blocks come from comparisons of different model families on the same split; a
    patient's time and event must agree between blocks. A model in several blocks is
    taken from the first. Blocks may come from a client, so each one is validated and
    problems raise a :class:`UserInputError` that names the block and field.
    """
    if not _is_list_like(blocks):
        raise UserInputError("Prediction blocks must be given as a list.")
    validated = [_validated_prediction_block(block, index) for index, block in enumerate(blocks)]
    usable = [block for block in validated if block is not None]
    if not usable:
        raise UserInputError("No test-set predictions to compare.")
    shared = set(usable[0]["row_ids"])
    for block in usable[1:]:
        shared &= set(block["row_ids"])
    rows = sorted(shared)
    if not rows:
        raise UserInputError("The comparisons share no test patients; run them on the same split.")
    time = np.empty(len(rows))
    event = np.empty(len(rows), dtype=int)
    risks: dict[str, np.ndarray] = {}
    for index, block in enumerate(usable):
        position = {row: offset for offset, row in enumerate(block["row_ids"])}
        order = [position[row] for row in rows]
        block_time = block["time"][order]
        block_event = block["event"][order]
        if index == 0:
            time, event = block_time, block_event
        elif not (np.allclose(time, block_time) and np.array_equal(event, block_event)):
            raise UserInputError("The comparisons disagree on the outcome of shared test patients; run them on the same dataset.")
        for name, values in block["risk"].items():
            if name not in risks:
                risks[name] = values[order]
    return time, event, risks, rows


def c_index_intervals(
    time: Sequence[float] | np.ndarray,
    event: Sequence[int] | np.ndarray,
    risks: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reference: str | None = "Cox PH",
    n_bootstrap: int = 1000,
    random_seed: int = 20260926,
    level: float = 0.95,
) -> dict[str, Any]:
    """Harrell's C of each model on the same test patients, with bootstrap intervals.

    Every bootstrap draw resamples patients once and scores all models on them, so the
    interval for a model's difference from the reference is paired: it accounts for the
    models being scored on the same patients, which separate intervals do not.
    """
    from survival_toolkit.marker_screen import harrell_c_many

    if not isinstance(risks, Mapping):
        raise UserInputError("Model risk scores must map each model name to its risk scores.")
    names = [str(name) for name in risks]
    if not names:
        raise UserInputError("No model risk scores to compare.")
    try:
        time = np.asarray(time, dtype=float).reshape(-1)
        event_values = np.asarray(event, dtype=float).reshape(-1)
        columns = [np.asarray(risks[name], dtype=float).reshape(-1) for name in risks]
    except (TypeError, ValueError) as exc:
        raise UserInputError("Test-set times, events and risk scores must be numbers.") from exc
    if event_values.shape[0] != time.shape[0] or not np.isin(event_values, (0.0, 1.0)).all():
        raise UserInputError("Every test patient needs one event indicator of 0 or 1.")
    if not np.isfinite(time).all():
        raise UserInputError("Every test patient needs a finite follow-up time.")
    event = event_values.astype(int)
    for name, column in zip(names, columns):
        if column.shape[0] != time.shape[0]:
            raise UserInputError(
                f"Every model needs one risk score per test patient: {name} has {column.shape[0]} for {time.shape[0]} patients."
            )
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise UserInputError("Every model needs a finite risk score for every test patient.")
    point = harrell_c_many(time, event, matrix)
    rng = np.random.default_rng(int(random_seed))
    draws = []
    for _ in range(int(n_bootstrap)):
        raise_if_cancelled()
        rows = rng.integers(0, time.shape[0], size=time.shape[0])
        if event[rows].any():
            draws.append(harrell_c_many(time[rows], event[rows], matrix[rows]))
    samples = np.asarray(draws, dtype=float).reshape(-1, len(names))
    tail = (1.0 - float(level)) / 2.0

    def interval(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        if finite.size < 20:
            return [None, None]
        return [float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))]

    reference_index = names.index(reference) if reference in names else None
    rows_out: list[dict[str, Any]] = []
    for column, name in enumerate(names):
        row: dict[str, Any] = {
            "model": name,
            "c_index": None if not np.isfinite(point[column]) else float(point[column]),
            "c_index_ci": interval(samples[:, column]),
        }
        if reference_index is not None and column != reference_index:
            difference = float(point[column] - point[reference_index])
            row["delta_vs_reference"] = difference if np.isfinite(difference) else None
            row["delta_ci"] = interval(samples[:, column] - samples[:, reference_index])
        rows_out.append(row)
    return {
        "n": int(time.shape[0]),
        "events": int(event.sum()),
        "reference": names[reference_index] if reference_index is not None else None,
        "n_bootstrap": int(samples.shape[0]),
        "level": float(level),
        "rows": rows_out,
    }

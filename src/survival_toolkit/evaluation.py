from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np

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

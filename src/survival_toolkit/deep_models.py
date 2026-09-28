"""Deep learning survival analysis models.

Provides PyTorch-based models for survival analysis:
- DeepSurv (Neural Cox Proportional Hazards)
- DeepHit (Discrete-time single-event survival)
- Neural MTLR (Multi-Task Logistic Regression)
- Survival Transformer (Self-attention for feature interactions)
- Survival VAE (VAE-inspired latent model for risk group discovery)

All functions return plain dicts (JSON-serializable) suitable for FastAPI responses.
"""

from __future__ import annotations

import gc
import logging
import math
import multiprocessing as mp
import os
import subprocess
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from survival_toolkit.encoding import (
    _checked_features,
    canonical_category_values,
    fit_feature_encoder as _fit_shared_feature_encoder,
    reject_numeric_text_features,
    transform_feature_encoder as _transform_shared_feature_encoder,
)
from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.errors import InternalAnalysisError, must_propagate, user_input_boundary
from survival_toolkit.evaluation import (
    DEFAULT_HOLDOUT_FRACTION,
    evaluation_split_fingerprint,
    locked_test_split,
    prediction_block,
    stratified_holdout_indices,
)
from survival_toolkit.evaluation import metric_name_for_evaluation as _metric_name_for_evaluation

try:
    from sklearn.model_selection import StratifiedKFold

    _SKLEARN_AVAILABLE = True
except ImportError:
    StratifiedKFold = None
    _SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sksurv.metrics import concordance_index_censored as _sksurv_concordance

    _SKSURV_METRICS_AVAILABLE = True
except ImportError:
    _SKSURV_METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch.nn as _torch_nn

    _TorchModuleBase = _torch_nn.Module
elif TORCH_AVAILABLE:
    _TorchModuleBase = nn.Module
else:
    _TorchModuleBase = object

_TORCH_INSTALL_MSG = (
    "PyTorch is required for deep learning models. "
    "Install with: pip install torch  (see https://pytorch.org for GPU options)"
)
_SKLEARN_INSTALL_MSG = (
    "scikit-learn is required for deep-learning holdout splits and repeated CV. "
    "Install with: pip install 'survstudio[dl]'"
)
_ADAM_WEIGHT_DECAY = 1e-4
_DEEPHIT_RANKING_SIGMA = 1.0
_DEEP_COMPARE_PARALLEL_MAX_INFLIGHT_BYTES = 512 * 1024 * 1024
_DEEP_COMPARE_PARALLEL_WORKER_OVERHEAD_BYTES = 650 * 1024 * 1024
_DEEP_COMPARE_PARALLEL_MEMORY_RESERVE_BYTES = 256 * 1024 * 1024
_TRANSFORMER_MAX_ATTENTION_BYTES = 2 * 1024 * 1024 * 1024
# Most trainable parameters a deep network may have. Training keeps about five float32 copies
# (weights, gradients, two AdamW moments, best-epoch checkpoint): roughly 400 MB at this limit.
_DEEP_MAX_PARAMETERS = 20_000_000
# Hidden layers used when a caller passes none, shared by the single-model and comparison paths.
_DEFAULT_HIDDEN_LAYERS: tuple[int, ...] = (64, 64)
# Every deep-learning fit runs with this many intra-op torch threads, in this process and in
# parallel repeated-CV workers alike: floating-point reductions depend on the thread count, so
# a machine-dependent default would make the same seed give different results.
_DEEP_TORCH_NUM_THREADS = 1
# How often a waiting job re-checks cancellation (training lock, parallel repeated-CV folds).
_CANCELLATION_POLL_SECONDS = 0.5
# Seeds derived as seed + offset wrap into the range numpy and scikit-learn accept.
_SEED_MODULUS = 2**32
# Serialises deep-learning training in this process (see _serialized_torch_training).
_TORCH_TRAINING_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(_TORCH_INSTALL_MSG)


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(_SKLEARN_INSTALL_MSG)


def _seed_torch(random_seed: int) -> None:
    """Seed torch for one training run.

    NumPy's and Python's global random states are left alone: the deep models draw only
    from torch generators and seeded local NumPy generators, and reseeding the process-wide
    state would disturb concurrent analyses that sample from it (for example Kernel SHAP).
    """
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError as exc:
        warnings.warn(f"Could not enable deterministic algorithms: {exc}", RuntimeWarning)


def _clip_gradients(model: nn.Module, max_norm: float = 1.0) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _derived_seed(base_seed: int, offset: int = 0) -> int:
    """``base_seed + offset`` wrapped into [0, 2**32), the range numpy and scikit-learn accept.

    Equal to the plain sum whenever it does not overflow, so ordinary seeds keep their fold
    assignments and training seeds.
    """
    return int((int(base_seed) + int(offset)) % _SEED_MODULUS)


_MUST_PROPAGATE_MARK = "_survstudio_must_propagate"


def _must_propagate_deep(exc: BaseException) -> bool:
    """``must_propagate`` for the deep-model fallbacks, seeing through ``user_input_boundary``.

    The public trainers turn a ``TypeError`` raised by SurvStudio code (a coding bug) into an
    ``InternalAnalysisError``; per-model and per-fold fallbacks must re-raise it instead of
    recording an ordinary model failure. Running out of memory also ends the run: the next
    model or fold would only exhaust the memory again. Worker processes mark such errors
    before they are pickled back to the parent, because pickling drops ``__cause__``.
    """
    if isinstance(exc, MemoryError) or getattr(exc, _MUST_PROPAGATE_MARK, False) or must_propagate(exc):
        return True
    cause = exc.__cause__
    return (
        isinstance(exc, InternalAnalysisError)
        and cause is not None
        and (isinstance(cause, MemoryError) or must_propagate(cause))
    )


def _failure_message(exc: BaseException) -> str:
    """The message recorded for a failed fit.

    The trainers' input boundary replaces an error raised inside a library with a generic
    internal-error message that points to the server log; the library's own message (the
    cause) says what went wrong, as the ML module's model failures do. The full traceback is
    logged where the failure is recorded.
    """
    cause = exc.__cause__ if isinstance(exc, InternalAnalysisError) else None
    source = cause if cause is not None else exc
    return str(source).strip() or type(source).__name__


def _mark_must_propagate(exc: BaseException) -> None:
    try:
        setattr(exc, _MUST_PROPAGATE_MARK, True)
    except (AttributeError, TypeError):
        pass


@contextmanager
def _pinned_torch_threads() -> Iterator[None]:
    """Train with ``_DEEP_TORCH_NUM_THREADS`` intra-op torch threads and restore the previous count."""
    if not TORCH_AVAILABLE:
        yield
        return
    previous = int(torch.get_num_threads())
    changed = previous != _DEEP_TORCH_NUM_THREADS
    if changed:
        torch.set_num_threads(_DEEP_TORCH_NUM_THREADS)
    try:
        yield
    finally:
        if changed:
            torch.set_num_threads(previous)


def _run_deep_compare_task(task: dict[str, Any]) -> dict[str, Any]:
    trainer_name = str(task["model_name"])
    trainer_map = {
        "DeepSurv": train_deepsurv,
        "DeepHit": train_deephit,
        "Neural MTLR": train_neural_mtlr,
        "Survival Transformer": train_survival_transformer,
        "Survival VAE": train_survival_vae,
    }
    trainer = trainer_map[trainer_name]
    started = time.monotonic()
    result = trainer(
        None,
        time_column=task["time_column"],
        event_column=task["event_column"],
        features=task["features"],
        categorical_features=task["categorical_features"],
        event_positive_value=task["event_positive_value"],
        learning_rate=task["learning_rate"],
        epochs=task["epochs"],
        batch_size=task["batch_size"],
        random_seed=task["seed"],
        prepared_data=task["prepared_data"],
        evaluation_split=task["evaluation_split"],
        monitor_indices=task["monitor_indices"],
        early_stopping_patience=task["early_stopping_patience"],
        early_stopping_min_delta=task["early_stopping_min_delta"],
        **task["extra_kwargs"],
    )
    evaluation_mode = str(result.get("evaluation_mode", "unknown"))
    if task.get("require_holdout_evaluation") and evaluation_mode != "holdout":
        if task.get("repeat") is None:
            # The refit on the development set, scored on the locked test set.
            raise ValueError(
                "The locked test set did not give a clean holdout evaluation of the model refit on the development set "
                f"(reported '{evaluation_mode}'), so its locked-test C-index is blank."
            )
        raise ValueError(
            "Deep repeated-CV fold did not retain a clean holdout evaluation "
            f"(reported '{evaluation_mode}')."
        )
    return {
        "model": trainer_name,
        "repeat": task["repeat"],
        "fold": task["fold"],
        "c_index": result.get("c_index"),
        "evaluation_mode": evaluation_mode,
        "evaluation_note": result.get("evaluation_note"),
        "training_seed": int(task["seed"]),
        "split_seed": None if task.get("split_seed") is None else int(task["split_seed"]),
        "monitor_seed": None if task.get("monitor_seed") is None else int(task["monitor_seed"]),
        "n_features": result.get("n_features"),
        "training_time_ms": round((time.monotonic() - started) * 1000, 1),
        "training_samples": result.get("training_samples"),
        "evaluation_samples": result.get("evaluation_samples"),
        "training_events": result.get("training_events"),
        "evaluation_events": result.get("evaluation_events"),
        "epochs_trained": result.get("epochs_trained"),
        "early_stopping_epochs": result.get("early_stopping_epochs"),
        **({"holdout_risk": result.get("holdout_risk")} if task.get("keep_holdout_risk") else {}),
    }


def _run_deep_compare_fold_task(task: dict[str, Any]) -> dict[str, Any]:
    """Train every model of one repeated-CV fold (in this process or in a worker process).

    Torch threads are pinned by the trainers themselves (``_serialized_torch_training``), so a
    fold gives the same result in a worker as in the sequential path.
    """
    fold_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for model_spec in task["model_specs"]:
        model_name = str(model_spec["model_name"])
        try:
            fold_results.append(
                _run_deep_compare_task(
                    {
                        "model_name": model_name,
                        "extra_kwargs": model_spec["extra_kwargs"],
                        "repeat": task["repeat"],
                        "fold": task["fold"],
                        "seed": int(task["seed_base"]),
                        "split_seed": task.get("split_seed"),
                        "monitor_seed": task.get("monitor_seed"),
                        "time_column": task["time_column"],
                        "event_column": task["event_column"],
                        "features": task["features"],
                        "categorical_features": task["categorical_features"],
                        "event_positive_value": task["event_positive_value"],
                        "learning_rate": task["learning_rate"],
                        "epochs": task["epochs"],
                        "batch_size": task["batch_size"],
                        "early_stopping_patience": task["early_stopping_patience"],
                        "early_stopping_min_delta": task["early_stopping_min_delta"],
                        "prepared_data": task["prepared_data"],
                        "evaluation_split": task["evaluation_split"],
                        "monitor_indices": task["monitor_indices"],
                        "require_holdout_evaluation": task.get("require_holdout_evaluation", False),
                    }
                )
            )
        except Exception as exc:
            if _must_propagate_deep(exc):
                _mark_must_propagate(exc)
                raise
            # Logged here, where the traceback and the cause still exist (a worker's exceptions
            # lose their cause when they are pickled back to the parent).
            logger.exception(
                "Deep-learning model %s failed in repeated-CV repeat %s, fold %s.", model_name, task["repeat"], task["fold"]
            )
            errors.append(
                {
                    "model": model_name,
                    "repeat": task["repeat"],
                    "fold": task["fold"],
                    "error": _failure_message(exc),
                }
            )
    return {"fold_results": fold_results, "errors": errors}


def _deep_trainer_specs(
    *,
    hidden_layers: list[int],
    dropout: float,
    num_time_bins: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    latent_dim: int,
    n_clusters: int,
) -> list[tuple[str, Any, dict[str, Any]]]:
    return [
        ("DeepSurv", train_deepsurv, {"hidden_layers": hidden_layers, "dropout": dropout}),
        ("DeepHit", train_deephit, {"hidden_layers": hidden_layers, "dropout": dropout, "num_time_bins": num_time_bins}),
        ("Neural MTLR", train_neural_mtlr, {"hidden_layers": hidden_layers, "dropout": dropout, "num_time_bins": num_time_bins}),
        (
            "Survival Transformer",
            train_survival_transformer,
            {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "dropout": dropout},
        ),
        (
            "Survival VAE",
            train_survival_vae,
            {
                "hidden_layers": hidden_layers,
                "latent_dim": latent_dim,
                "n_clusters": n_clusters,
                "dropout": dropout,
            },
        ),
    ]


def _canonical_deep_model_name(model_type: str) -> str:
    mapping = {
        "deepsurv": "DeepSurv",
        "deephit": "DeepHit",
        "mtlr": "Neural MTLR",
        "transformer": "Survival Transformer",
        "vae": "Survival VAE",
    }
    if model_type not in mapping:
        raise ValueError(f"Unknown deep model type: {model_type}")
    return mapping[model_type]


class InsufficientDeepSampleError(ValueError):
    """Too few analyzable rows or events remain for a deep-learning fit."""


def _coerce_deep_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    min_samples: int = 10,
    require_event: bool = True,
    event_already_coded: bool = False,
) -> pd.DataFrame:
    """Clean raw data for deep models before fitting an encoder.

    The full pass builds the cohort with the ML cohort builder (``_deep_cohort_frame``), so the
    deep models analyse exactly the rows, and apply exactly the outcome checks, of the ML models.

    ``event_already_coded`` marks frames that already went through this function (their
    event column is 0/1). Re-reading a 0/1 column against the user's original event label
    (for example "Dead") would fail or mis-code the events. Such split passes keep the rows,
    outcome coding, and feature types of the full cleaned frame.
    """
    _require_torch()

    if not features:
        raise ValueError("Select at least one feature for deep learning models.")
    if time_column == event_column:
        raise ValueError("The survival time column and event column must be different.")
    overlapping_outcomes = sorted({str(feature) for feature in features if str(feature) in {str(time_column), str(event_column)}})
    if overlapping_outcomes:
        raise ValueError(
            "Survival outcome columns cannot be used as deep-learning features: "
            + ", ".join(overlapping_outcomes)
            + "."
        )
    required_columns = [*features, time_column, event_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_preview = ", ".join(str(column) for column in missing_columns[:5])
        raise ValueError(
            "Deep learning input is missing required columns: "
            f"{missing_preview}."
        )

    if event_already_coded:
        frame = df[required_columns].copy()
        coded_events = pd.to_numeric(frame[event_column], errors="coerce")
        if not bool(coded_events.dropna().isin([0.0, 1.0]).all()):
            raise ValueError("Internal error: a cleaned deep-learning frame must carry a 0/1 event column.")
        frame[event_column] = coded_events.astype(float)
        frame[time_column] = pd.to_numeric(frame[time_column], errors="coerce")
        frame = frame.dropna(subset=[time_column, event_column])
        frame = frame.loc[frame[time_column] >= 0]
        source_row_index = frame.index.tolist()
        frame = frame.reset_index(drop=True)
        # Positions in this slice; the slice inherits the full frame's attrs, which do not apply to it.
        frame.attrs = {"source_row_index": source_row_index, "dropped_nonpositive_time_rows": 0}
    else:
        frame = _deep_cohort_frame(
            df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=list(categorical_features or []),
            event_positive_value=event_positive_value,
        )

    if frame.empty:
        raise InsufficientDeepSampleError("No analyzable rows remain after removing missing/invalid values.")
    if require_event and float(frame[event_column].sum()) <= 0:
        raise InsufficientDeepSampleError("No events were found after preprocessing the event column.")
    if frame.shape[0] < min_samples:
        raise InsufficientDeepSampleError(
            f"Only {frame.shape[0]} analyzable rows remain after validating outcome columns. "
            f"Need at least {min_samples} samples for deep learning models."
        )
    return frame


def _deep_cohort_frame(
    df: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str],
    event_positive_value: Any,
) -> pd.DataFrame:
    """The analysed cohort of the ML models (``analysis._cohort_frame``) with deep-model feature types.

    The ML cohort builder parses follow-up times ("1,234"), refuses text that is not a time
    ("12 months"), a time column missing for (nearly) every row of one outcome, times that are
    never positive, and a cleaning that removes every censored row, and returns the caution
    for a time column whose name does not look like follow-up time. Rows with a missing
    feature are kept (the encoder imputes them), as for the ML models.

    Feature types are decided once, here, on the whole cleaned cohort, and split passes keep
    them: a feature is categorical when it is declared categorical, is a pandas Categorical,
    or is text whose non-missing values do not all read as finite numbers. Categorical
    features are stored as canonical text labels (``_category_labels``); text features whose
    values all read as numbers are numeric.
    """
    from survival_toolkit.analysis import _cohort_frame

    features = _checked_features(df, features)
    declared = list(categorical_features)
    pandas_categoricals = [column for column in features if isinstance(df[column].dtype, pd.CategoricalDtype)]
    source = df
    if pandas_categoricals:
        # Decoded before the cohort builder turns them into text, so integer categories read
        # "1", not "1.0". A shallow copy: the caller's frame is left unchanged.
        source = df.copy(deep=False)
        for column in pandas_categoricals:
            source[column] = _category_labels(df[column])
    cohort = _cohort_frame(
        source,
        time_column,
        event_column,
        event_positive_value=event_positive_value,
        extra_columns=features,
        drop_missing_extra_columns=False,
    )
    frame = cohort[[*features, time_column, event_column]].copy()
    text_features: list[Any] = []
    for column in features:
        values = frame[column]
        if column in declared or column in pandas_categoricals:
            # The encoder's canonical labels ("1", not "1.0" when a blank made the codes decimal), as for ML.
            frame[column] = _category_labels(values)
        elif is_numeric_dtype(values.dtype):
            continue
        elif _all_finite_numbers(values):
            frame[column] = pd.Series(
                pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan),
                index=values.index,
            )
        else:
            text_features.append(column)
    # Continuous numbers with a few stray text values, and text with too many levels, are
    # refused (the shared rule of the ML, deep-learning, and Cox paths); other text is categorical.
    reject_numeric_text_features(frame, text_features, declared)
    for column in text_features:
        frame[column] = _category_labels(frame[column])
    frame.attrs = {
        "source_row_index": list(cohort.attrs.get("source_row_index", [])),
        "dropped_nonpositive_time_rows": int(cohort.attrs.get("dropped_nonpositive_time_rows", 0)),
        # Rows without a usable time or event (``drop_missing_extra_columns=False``: features are kept).
        "dropped_missing_outcome_rows": int(cohort.attrs.get("dropped_missing_rows", 0)),
        "time_column_note": cohort.attrs.get("time_column_note"),
    }
    return frame


def _all_finite_numbers(values: pd.Series) -> bool:
    """Whether every non-missing value of a text column reads as a finite number."""
    numbers = pd.to_numeric(values.dropna(), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return bool(np.isfinite(numbers).all())


def _category_labels(values: pd.Series) -> pd.Series:
    """Canonical text labels of a categorical feature (``encoding.canonical_category_values``).

    A pandas Categorical is decoded to its category values first, so integer categories read
    "1", "2" even when a missing value made them decimal, as for a declared code column.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = pd.Series(
            pd.api.extensions.take(np.asarray(values.cat.categories), values.cat.codes.to_numpy(), allow_fill=True),
            index=values.index,
            name=values.name,
        )
    return canonical_category_values(values).astype("string")


def _cohort_summary_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    """Cohort-building counts and the time-column caution, for the scientific summaries.

    ``source`` is a cleaned frame's ``attrs`` or prepared tensors that carry the same keys.
    """
    return {
        "dropped_nonpositive_time_rows": int(source.get("dropped_nonpositive_time_rows", 0) or 0),
        "dropped_missing_outcome_rows": int(source.get("dropped_missing_outcome_rows", 0) or 0),
        "time_column_note": source.get("time_column_note") or None,
    }


def _cohort_cautions(
    dropped_nonpositive_time_rows: int,
    dropped_missing_outcome_rows: int,
    time_column_note: str | None,
) -> list[str]:
    """Cautions about rows the cohort builder removed and about an unusual time column."""
    cautions: list[str] = []
    if int(dropped_missing_outcome_rows) > 0:
        cautions.append(
            f"{int(dropped_missing_outcome_rows)} row(s) with a missing or non-finite survival time or a missing event "
            "were excluded before deep-model preprocessing."
        )
    if int(dropped_nonpositive_time_rows) > 0:
        cautions.append(
            f"{int(dropped_nonpositive_time_rows)} row(s) with negative survival time were excluded before deep-model preprocessing."
        )
    if time_column_note:
        cautions.append(str(time_column_note))
    return cautions


def _categorical_feature_columns(frame: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """Features a cleaned frame treats as categorical (``_coerce_deep_frame`` stores them as text)."""
    return [column for column in features if isinstance(frame[column].dtype, pd.StringDtype)]


def _fit_deep_encoder(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the shared tabular encoder with numeric standardization enabled.

    Every feature the cleaned frame stores as text is passed as categorical, so a training
    split keeps the type decided on the whole cohort even when its rows happen to hold only
    number-like levels.
    """
    categorical = list(dict.fromkeys([*(categorical_features or []), *_categorical_feature_columns(frame, features)]))
    return _fit_shared_feature_encoder(
        frame,
        features,
        categorical,
        standardize_numeric=True,
    )


def _transform_deep_frame(
    frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    encoder: dict[str, Any],
) -> dict[str, Any]:
    """Transform a cleaned frame with a previously fitted encoder."""
    encoded = _transform_shared_feature_encoder(frame, encoder, output="dataframe")
    x_array = encoded.to_numpy(dtype=np.float32, copy=False)

    return {
        "X_tensor": torch.from_numpy(np.ascontiguousarray(x_array)),
        "time_tensor": torch.from_numpy(frame[time_column].values.astype(np.float32)),
        "event_tensor": torch.from_numpy(frame[event_column].values.astype(np.float32)),
        "feature_names": list(encoder["feature_names"]),
        "scaler_params": dict(encoder["scaler_params"]),
        # Input columns the encoder one-hot codes (declared plus auto-coded), for reporting.
        "categorical_features": list(encoder.get("categorical_features", [])),
        "categorical_feature_indices": list(encoder.get("categorical_feature_indices", [])),
        "numeric_feature_indices": list(encoder.get("numeric_feature_indices", [])),
        "n_samples": int(x_array.shape[0]),
        "n_features": int(x_array.shape[1]),
    }


def _prepare_deep_data(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> dict[str, Any]:
    """Prepare data for deep models using a single-cohort fitted encoder."""
    frame = _coerce_deep_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
    )
    encoder = _fit_deep_encoder(frame, features, categorical_features)
    return _transform_deep_frame(
        frame,
        time_column=time_column,
        event_column=event_column,
        encoder=encoder,
    )


def _prepare_deep_split_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare fold-specific deep-learning data with preprocessing fitted on the training fold.

    ``train_df`` and ``eval_df`` are slices of a frame already cleaned by
    ``_coerce_deep_frame``, so their event column is 0/1; ``event_positive_value`` is
    accepted for signature compatibility and not re-applied. The returned split records how
    many evaluation rows carry a categorical level the training rows never show
    (``unseen_category_rows``).
    """
    from survival_toolkit.ml_models import _unseen_category_rows

    del event_positive_value
    train_frame = _coerce_deep_frame(
        train_df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        min_samples=10,
        require_event=True,
        event_already_coded=True,
    )
    eval_frame = _coerce_deep_frame(
        eval_df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        min_samples=1,
        require_event=False,
        event_already_coded=True,
    )
    encoder = _fit_deep_encoder(train_frame, features, categorical_features)
    train_data = _transform_deep_frame(
        train_frame,
        time_column=time_column,
        event_column=event_column,
        encoder=encoder,
    )
    eval_data = _transform_deep_frame(
        eval_frame,
        time_column=time_column,
        event_column=event_column,
        encoder=encoder,
    )
    combined_x = torch.cat([train_data["X_tensor"], eval_data["X_tensor"]], dim=0)
    combined_t = torch.cat([train_data["time_tensor"], eval_data["time_tensor"]], dim=0)
    combined_e = torch.cat([train_data["event_tensor"], eval_data["event_tensor"]], dim=0)
    train_n = int(train_data["n_samples"])
    eval_n = int(eval_data["n_samples"])
    feature_meta = {
        "feature_names": list(train_data["feature_names"]),
        "scaler_params": dict(train_data["scaler_params"]),
        "categorical_features": list(train_data.get("categorical_features", [])),
        "categorical_feature_indices": list(train_data.get("categorical_feature_indices", [])),
        "numeric_feature_indices": list(train_data.get("numeric_feature_indices", [])),
    }
    del train_data, eval_data  # free per-split tensors now that combined tensors are built
    categorical_columns = _categorical_feature_columns(train_frame, features)
    evaluation_split = {
        "train_idx": np.arange(train_n, dtype=int),
        "eval_idx": np.arange(train_n, train_n + eval_n, dtype=int),
        # Positions in ``eval_df`` of the evaluation rows, in tensor order.
        "eval_source_positions": [int(label) for label in eval_frame.attrs.get("source_row_index", range(eval_n))],
        "evaluation_mode": "holdout",
        "evaluation_note": (
            f"Reported C-index is computed on an external fold with {train_n} training samples "
            f"and {eval_n} evaluation samples."
        ),
        # The encoder scores these rows as the reference level (see ml_models._unseen_category_rows).
        "unseen_category_rows": _unseen_category_rows(train_frame, eval_frame, categorical_columns),
    }
    return (
        {
            "X_tensor": combined_x,
            "time_tensor": combined_t,
            "event_tensor": combined_e,
            **feature_meta,
            "n_samples": train_n + eval_n,
            "n_features": int(combined_x.shape[1]),
            "dropped_nonpositive_time_rows": int(train_frame.attrs.get("dropped_nonpositive_time_rows", 0))
            + int(eval_frame.attrs.get("dropped_nonpositive_time_rows", 0)),
        },
        evaluation_split,
    )


def _estimate_array_like_nbytes(value: Any) -> int:
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return int(value.element_size() * value.numel())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    return 0


def _estimate_deep_compare_task_bytes(task: dict[str, Any]) -> int:
    total = 0
    prepared_data = task.get("prepared_data")
    if isinstance(prepared_data, dict):
        total += sum(_estimate_array_like_nbytes(value) for value in prepared_data.values())
    evaluation_split = task.get("evaluation_split")
    if isinstance(evaluation_split, dict):
        total += sum(_estimate_array_like_nbytes(value) for value in evaluation_split.values())
    total += _estimate_array_like_nbytes(task.get("monitor_indices"))
    return int(total)


def _proc_meminfo_available_bytes() -> int | None:
    """Linux ``MemAvailable`` (free pages plus reclaimable cache), in bytes."""
    meminfo = Path("/proc/meminfo")
    try:
        if not meminfo.is_file():
            return None
        for line in meminfo.read_text(encoding="ascii", errors="ignore").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _windows_available_memory_bytes() -> int | None:
    """Available physical memory from ``GlobalMemoryStatusEx`` on Windows."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except (AttributeError, OSError, ValueError):
        return None


def _vm_stat_available_bytes() -> int | None:
    """Free, speculative, and purgeable pages reported by macOS ``vm_stat``."""
    try:
        vm_stat = subprocess.check_output(["vm_stat"], text=True)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    page_size_match = None
    free_pages = 0
    for raw_line in vm_stat.splitlines():
        line = raw_line.strip()
        if line.startswith("Mach Virtual Memory Statistics:"):
            page_size_match = line
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        value = value.strip().rstrip(".")
        try:
            pages = int(value)
        except ValueError:
            continue
        if key in {"Pages free", "Pages speculative", "Pages purgeable"}:
            free_pages += pages
    if page_size_match is None:
        return None
    try:
        page_size = int(page_size_match.split("page size of", maxsplit=1)[1].split("bytes", maxsplit=1)[0].strip())
    except (IndexError, ValueError):
        return None
    return free_pages * page_size


def _sysconf_available_bytes() -> int | None:
    """Free physical pages from ``sysconf`` (POSIX systems that expose SC_AVPHYS_PAGES)."""
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


_CGROUP_MEMORY_FILES = (
    # cgroup v2: the group's limit ("max" when unlimited) and its current usage.
    (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
    # cgroup v1: an unlimited group reports a huge page-counter maximum instead.
    (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
)


def _cgroup_available_memory_bytes() -> int | None:
    """Memory left under a cgroup (container) memory limit, or ``None`` when no limit applies.

    Inside a container ``MemAvailable`` describes the host, so a worker pool sized from it
    alone can exceed the container's limit and be killed.
    """
    for limit_path, usage_path in _CGROUP_MEMORY_FILES:
        try:
            raw_limit = limit_path.read_text(encoding="ascii", errors="ignore").strip()
        except OSError:
            continue
        try:
            limit = int(raw_limit)
        except ValueError:
            continue  # "max": no limit at this level
        if limit <= 0 or limit >= 1 << 60:
            continue
        try:
            usage = int(usage_path.read_text(encoding="ascii", errors="ignore").strip())
        except (OSError, ValueError):
            usage = 0
        return max(0, limit - usage)
    return None


def _available_system_memory_bytes() -> int | None:
    """Memory available to new worker processes, or ``None`` when it cannot be determined.

    Probes run from the most to the least accurate; each returns ``None`` on platforms
    where it does not apply (for example ``vm_stat`` exists only on macOS). A cgroup
    memory limit, when present, caps the result.
    """
    system_available: int | None = None
    for probe in (
        _proc_meminfo_available_bytes,
        _windows_available_memory_bytes,
        _sysconf_available_bytes,
        _vm_stat_available_bytes,
    ):
        available = probe()
        if available is not None:
            system_available = available
            break
    cgroup_available = _cgroup_available_memory_bytes()
    if cgroup_available is None:
        return system_available
    if system_available is None:
        return cgroup_available
    return min(system_available, cgroup_available)


def _available_cpu_count() -> int:
    """CPUs this process may run on (its affinity mask where the platform exposes one)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _estimate_parallel_deep_compare_memory_bytes(
    *,
    payload_bytes: int,
    max_workers: int,
    training_bytes: int = 0,
) -> int:
    """Memory for ``max_workers`` workers, each holding one fold payload and training one model."""
    per_worker = int(payload_bytes) + _DEEP_COMPARE_PARALLEL_WORKER_OVERHEAD_BYTES + max(0, int(training_bytes))
    return int(per_worker * max(1, max_workers) + _DEEP_COMPARE_PARALLEL_MEMORY_RESERVE_BYTES)


def _transformer_feedforward_dim(d_model: int) -> int:
    return int(4 * max(1, int(d_model)))


def _estimate_transformer_attention_bytes(
    *,
    training_samples: int,
    n_features: int,
    n_heads: int,
    n_layers: int,
    d_model: int = 64,
) -> int:
    """Rough full-batch activation footprint (float32) of the Survival Transformer.

    Per layer and token it counts the attention weights (``heads x tokens``,
    kept for backward plus softmax output), the feed-forward hidden layer
    (activation, ReLU and dropout masks), and the ``d_model``-wide residual,
    norm, projection and dropout buffers, with 50% headroom for autograd.
    """
    tokens = max(1, int(n_features))
    per_token_per_layer = (
        2 * max(1, int(n_heads)) * tokens
        + 3 * _transformer_feedforward_dim(d_model)
        + 12 * max(1, int(d_model))
    )
    activations = max(1, int(training_samples)) * tokens * max(1, int(n_layers)) * per_token_per_layer * 4
    return int(activations * 1.5)


def _guard_transformer_attention_budget(
    *,
    training_samples: int,
    n_features: int,
    n_heads: int,
    n_layers: int,
    d_model: int = 64,
) -> None:
    estimated_bytes = _estimate_transformer_attention_bytes(
        training_samples=training_samples,
        n_features=n_features,
        n_heads=n_heads,
        n_layers=n_layers,
        d_model=d_model,
    )
    if estimated_bytes <= _TRANSFORMER_MAX_ATTENTION_BYTES:
        return
    estimated_gib = estimated_bytes / (1024 ** 3)
    raise ValueError(
        "Survival Transformer is disabled for this feature set because full-batch training on "
        f"the training split is estimated at {estimated_gib:.1f} GiB of activations "
        f"({int(training_samples)} training samples x {int(n_features)} encoded features x "
        f"{int(n_heads)} head(s) x {int(n_layers)} layer(s), width {int(d_model)}). Reduce the shared feature set before "
        "rerunning Compare All or testing Survival Transformer directly."
    )


def _validated_hidden_layers(hidden_layers: Sequence[Any] | None) -> list[int]:
    """Hidden-layer widths as positive ints; ``None`` gives the shared default."""
    if hidden_layers is None:
        return list(_DEFAULT_HIDDEN_LAYERS)
    widths: list[int] = []
    for width in hidden_layers:
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or int(width) <= 0:
            raise ValueError("Hidden layers must contain positive integers only.")
        widths.append(int(width))
    return widths


def _validated_positive_integer(value: Any, label: str) -> int:
    """A whole number of at least 1 (an integral float such as 10.0 is accepted)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"{label} must be a whole number of at least 1 (got {value!r}).")
    number = float(value)
    if not math.isfinite(number) or number != math.floor(number) or number < 1:
        raise ValueError(f"{label} must be a whole number of at least 1 (got {value!r}).")
    return int(number)


def _validated_training_settings(*, epochs: Any, batch_size: Any, learning_rate: Any) -> tuple[int, int, float]:
    """Epochs, batch size, and learning rate checked at the Python entry points.

    The web form enforces its own bounds; package callers passing 0 epochs (or a negative
    count) would otherwise get an untrained network reported with a C-index.
    """
    epochs = _validated_positive_integer(epochs, "The number of epochs")
    batch_size = _validated_positive_integer(batch_size, "The batch size")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, np.integer, float, np.floating)):
        raise ValueError(f"The learning rate must be a positive finite number (got {learning_rate!r}).")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError(f"The learning rate must be a positive finite number (got {learning_rate!r}).")
    return epochs, batch_size, float(learning_rate)


_LOCKED_TEST_FRACTION_BOUNDS = (0.05, 0.5)


def _validated_locked_test_fraction(value: Any) -> float | None:
    """``None`` (no locked test set) or a fraction within the web form's bounds.

    0, a negative value, or NaN used to leave the run without a locked test set silently.
    """
    if value is None:
        return None
    low, high = _LOCKED_TEST_FRACTION_BOUNDS
    valid_type = not isinstance(value, bool) and isinstance(value, (int, np.integer, float, np.floating))
    if not valid_type or not (low <= float(value) <= high):
        raise ValueError(
            f"locked_test_fraction must be None (no locked test set) or a fraction between {low} and {high} "
            f"(got {value!r})."
        )
    return float(value)


def _mlp_parameter_count(in_features: int, widths: Sequence[int]) -> tuple[int, int]:
    """Parameters of a stack of Linear layers (weights and biases) and its output width."""
    total = 0
    previous = int(in_features)
    for width in widths:
        total += previous * int(width) + int(width)
        previous = int(width)
    return total, previous


def _estimate_deep_parameter_count(
    model_name: str,
    *,
    n_features: int,
    hidden_layers: Sequence[Any] | None = None,
    num_time_bins: int = 50,
    d_model: int = 64,
    n_layers: int = 2,
    latent_dim: int = 8,
) -> int:
    """Exact trainable-parameter count of the network a trainer would build (Python ints, no allocation)."""
    n_features = int(n_features)
    if model_name == "Survival Transformer":
        width = int(d_model)
        # Per encoder layer: attention in/out projections (4d^2 + 4d), the 4d feed-forward
        # block (8d^2 + 5d) and two layer norms (4d).
        per_layer = 12 * width * width + 13 * width
        return 2 * width + n_features * width + max(0, int(n_layers)) * per_layer + width + 1
    widths = _validated_hidden_layers(hidden_layers)
    body, last_width = _mlp_parameter_count(n_features, widths)
    if model_name == "DeepSurv":
        return body + last_width + 1
    if model_name in {"DeepHit", "Neural MTLR"}:
        outputs = int(num_time_bins) + 1  # observed bins plus the tail bucket
        return body + last_width * outputs + outputs
    if model_name == "Survival VAE":
        if not widths:
            raise ValueError("Survival VAE needs at least one hidden layer.")
        latent = int(latent_dim)
        posterior_heads = 2 * (last_width * latent + latent)
        decoder, decoder_width = _mlp_parameter_count(latent, list(reversed(widths)))
        decoder += decoder_width * n_features + n_features
        risk_hidden = max(widths[-1] // 2, 1)
        survival_head = latent * risk_hidden + risk_hidden + risk_hidden + 1
        return body + posterior_heads + decoder + survival_head
    raise ValueError(f"Unknown deep model type: {model_name}")


def _guard_deep_parameter_budget(model_name: str, *, n_features: int, **architecture: Any) -> None:
    """Refuse a network above ``_DEEP_MAX_PARAMETERS`` before any of it is allocated."""
    n_parameters = _estimate_deep_parameter_count(model_name, n_features=n_features, **architecture)
    if n_parameters <= _DEEP_MAX_PARAMETERS:
        return
    if model_name == "Survival Transformer":
        shape = f"width {int(architecture.get('d_model', 64))} and {int(architecture.get('n_layers', 2))} layer(s)"
        advice = "Use a smaller transformer width or fewer layers"
    else:
        shape = f"hidden layers {_validated_hidden_layers(architecture.get('hidden_layers'))}"
        advice = "Use fewer or narrower hidden layers"
    raise ValueError(
        f"{model_name} with {shape} on {int(n_features)} encoded input feature(s) would have "
        f"{n_parameters:,} trainable parameters; SurvStudio trains at most {_DEEP_MAX_PARAMETERS:,}. {advice}."
    )


def _estimate_deep_training_bytes(
    model_name: str,
    extra_kwargs: dict[str, Any],
    *,
    training_samples: int,
    total_samples: int,
    n_features: int,
    batch_size: int,
) -> int:
    """Rough float32 peak of training one model in a worker.

    Parameters count five times (weights, gradients, two AdamW moments, best-epoch
    checkpoint). Activations are those of the largest batch the model back-propagates
    through: the whole training partition for the full-batch models (DeepSurv, VAE,
    Transformer), all rows scored at once for monitoring and evaluation otherwise.
    """
    architecture_keys = {"hidden_layers", "num_time_bins", "d_model", "n_layers", "latent_dim"}
    architecture = {key: value for key, value in extra_kwargs.items() if key in architecture_keys}
    n_parameters = _estimate_deep_parameter_count(model_name, n_features=n_features, **architecture)
    parameter_bytes = 5 * 4 * n_parameters
    if model_name == "Survival Transformer":
        activation_bytes = _estimate_transformer_attention_bytes(
            training_samples=training_samples,
            n_features=n_features,
            n_heads=int(extra_kwargs.get("n_heads", 4)),
            n_layers=int(extra_kwargs.get("n_layers", 2)),
            d_model=int(extra_kwargs.get("d_model", 64)),
        )
    else:
        hidden_width = sum(_validated_hidden_layers(extra_kwargs.get("hidden_layers")))
        if model_name == "Survival VAE":
            row_width = 2 * n_features + 2 * hidden_width + 4 * int(extra_kwargs.get("latent_dim", 8))
            rows = training_samples
        elif model_name == "DeepSurv":
            row_width = n_features + hidden_width + 1
            rows = training_samples
        else:
            row_width = n_features + hidden_width + 3 * (int(extra_kwargs.get("num_time_bins", 50)) + 1)
            rows = max(int(batch_size), int(total_samples))
        # Layer outputs, ReLU/dropout masks and their gradients.
        activation_bytes = 3 * 4 * max(1, int(rows)) * row_width
    return int(parameter_bytes + activation_bytes)


def _reject_leaky_evaluation_split(evaluation_split: dict[str, Any], *, n_samples: int) -> None:
    """Refuse a caller-supplied split whose evaluation rows are also training rows.

    Apparent evaluation (training rows = evaluation rows by design) is exempt; every other
    mode needs disjoint partitions inside the ``n_samples`` rows they index.
    """
    train_rows = np.asarray(evaluation_split["train_idx"], dtype=int).ravel()
    eval_rows = np.asarray(evaluation_split["eval_idx"], dtype=int).ravel()
    for name, rows in (("train_idx", train_rows), ("eval_idx", eval_rows)):
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= int(n_samples)):
            raise ValueError(f"evaluation_split {name} addresses rows outside the {int(n_samples)} prepared rows.")
    if str(evaluation_split.get("evaluation_mode")) == "apparent":
        return
    shared_rows = np.intersect1d(train_rows, eval_rows)
    if shared_rows.size:
        raise ValueError(
            f"evaluation_split puts {int(shared_rows.size)} row(s) in both train_idx and eval_idx; "
            "a holdout evaluation needs disjoint training and evaluation rows."
        )


def _prepare_deep_training_inputs(
    df: pd.DataFrame | None,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    random_seed: int = 42,
    prepared_data: dict[str, Any] | None = None,
    evaluation_split: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare deep-model tensors with holdout preprocessing fit on train rows only."""
    if prepared_data is not None:
        if evaluation_split is None:
            # A split drawn now could not have been respected by the preprocessing already
            # fitted into ``prepared_data`` (scaling and levels would include evaluation rows).
            raise ValueError(
                "prepared_data must be passed together with the evaluation_split it was prepared for."
            )
        _reject_leaky_evaluation_split(evaluation_split, n_samples=int(prepared_data["X_tensor"].shape[0]))
        return prepared_data, evaluation_split
    if df is None:
        raise ValueError("Raw dataframe input is required when prepared_data is not provided.")
    if evaluation_split is not None:
        # Its positions would index the cleaned frame, not the caller's rows: rows removed for
        # a missing outcome shift every later position onto a different patient.
        raise ValueError(
            "evaluation_split can only be supplied together with the prepared_data it was built for; "
            "with a raw dataframe the trainer draws the shared holdout split itself."
        )

    clean_frame = _coerce_deep_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
    )
    resolved_split = _build_holdout_split(
        clean_frame[event_column].astype(int).to_numpy(),
        random_seed,
        source_rows=clean_frame.attrs.get("source_row_index"),
    )

    if str(resolved_split.get("evaluation_mode")) == "holdout":
        train_idx = np.asarray(resolved_split["train_idx"], dtype=int)
        eval_idx = np.asarray(resolved_split["eval_idx"], dtype=int)
        train_frame = clean_frame.iloc[train_idx].reset_index(drop=True)
        eval_frame = clean_frame.iloc[eval_idx].reset_index(drop=True)
        try:
            split_data, split_eval = _prepare_deep_split_data(
                train_frame,
                eval_frame,
                time_column=time_column,
                event_column=event_column,
                features=features,
                categorical_features=categorical_features,
                event_positive_value=event_positive_value,
            )
            split_eval["evaluation_note"] = str(
                resolved_split.get("evaluation_note", split_eval["evaluation_note"])
            )
            if resolved_split.get("evaluation_split_fingerprint"):
                split_eval["evaluation_split_fingerprint"] = resolved_split["evaluation_split_fingerprint"]
            clean_rows = clean_frame.attrs.get("source_row_index")
            split_eval["eval_row_ids"] = [
                clean_rows[int(eval_idx[position])] if clean_rows is not None else int(eval_idx[position])
                for position in split_eval["eval_source_positions"]
            ]
            split_data.update(_cohort_summary_fields(clean_frame.attrs))
            return split_data, split_eval
        except InsufficientDeepSampleError:
            # Only a too-small or event-free training split falls back to apparent
            # evaluation; any other preprocessing error is reported to the user.
            resolved_split = {
                "train_idx": np.arange(clean_frame.shape[0], dtype=int),
                "eval_idx": np.arange(clean_frame.shape[0], dtype=int),
                "evaluation_mode": "apparent",
                "evaluation_note": (
                    "A deterministic holdout split was available, but the holdout subset did not "
                    "retain enough analyzable samples after preprocessing. Reported C-index falls "
                    "back to the analyzable cohort."
                ),
                "evaluation_split_fingerprint": evaluation_split_fingerprint(
                    clean_frame.attrs.get("source_row_index"),
                    [(np.arange(clean_frame.shape[0]), np.arange(clean_frame.shape[0]))],
                    kind="apparent",
                ),
            }

    encoder = _fit_deep_encoder(clean_frame, features, categorical_features)
    full_data = _transform_deep_frame(
        clean_frame,
        time_column=time_column,
        event_column=event_column,
        encoder=encoder,
    )
    full_data.update(_cohort_summary_fields(clean_frame.attrs))
    return full_data, resolved_split


def _compute_c_index_torch(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> float | None:
    """Compute Harrell's concordance index from torch tensors.

    Delegates to sksurv.metrics.concordance_index_censored when available and
    otherwise (or if sksurv fails) to SurvStudio's own O(n log n) Harrell C
    (``analysis._harrell_c_index``), which uses the same tie convention.
    """
    risk_np = risk_scores.detach().cpu().numpy().ravel()
    time_np = times.detach().cpu().numpy().ravel()
    event_np = events.detach().cpu().numpy().ravel()
    if not np.all(np.isfinite(risk_np)):
        raise ValueError("Deep learning risk scores contain NaN or Inf values.")
    if not np.all(np.isfinite(time_np)):
        raise ValueError("Evaluation times contain NaN or Inf values.")
    if not np.all(np.isfinite(event_np)):
        raise ValueError("Evaluation events contain NaN or Inf values.")

    event_bool = event_np.astype(bool)
    if not event_bool.any():
        return None

    if _SKSURV_METRICS_AVAILABLE:
        try:
            result = _sksurv_concordance(event_bool, time_np, risk_np)
            return float(result[0])
        except (ImportError, RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
            warnings.warn(
                "scikit-survival concordance computation failed; falling back to SurvStudio's internal "
                f"O(n log n) Harrell C. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # Fallback without scikit-survival: the shared O(n log n) Harrell C, which
    # uses the same tie convention as sksurv (event vs. censoring at the same
    # time is a comparable pair).
    from survival_toolkit.analysis import _harrell_c_index

    return _harrell_c_index(time_np.astype(float), event_np.astype(int), risk_np.astype(float))


def _logsumexp_numpy(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        return max_value
    stable = np.exp(values - max_value)
    return float(max_value + np.log(np.sum(stable)))


def _survival_from_log_cumulative_hazard(log_cumulative_hazard: np.ndarray) -> np.ndarray:
    safe_log = np.asarray(log_cumulative_hazard, dtype=float)
    survival = np.ones_like(safe_log)
    positive_mask = np.isfinite(safe_log)
    if not positive_mask.any():
        return survival
    very_large = safe_log >= 50.0
    moderate = positive_mask & ~very_large
    survival[very_large] = 0.0
    survival[moderate] = np.exp(-np.exp(safe_log[moderate]))
    survival[~positive_mask] = 1.0
    return survival


def _expected_time_risk(pmf_with_tail: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
    """Fixed prognostic risk score from a discrete-time PMF with tail mass."""
    return -(pmf_with_tail * time_grid.reshape(1, -1)).sum(dim=1)


def _survival_after_event_bins(pmf_with_tail: torch.Tensor) -> torch.Tensor:
    """Return survival after each event bin end, preserving tail mass."""
    event_pmf = pmf_with_tail[:, :-1]
    return 1.0 - torch.cumsum(event_pmf, dim=1)


def _build_evaluation_split(
    events: np.ndarray,
    random_seed: int,
    holdout_fraction: float = 0.2,
    min_samples_for_holdout: int = 20,
    min_group_size: int = 4,
) -> dict[str, Any]:
    """Create a deterministic train/holdout split when the data support it.

    The split is stratified by event indicator so that concordance can be
    computed on the holdout subset when possible. If the cohort is too small
    or a stratum is too sparse, the function falls back to apparent evaluation.
    """
    n_samples = int(events.shape[0])
    all_indices = np.arange(n_samples, dtype=int)
    event_idx = np.flatnonzero(events == 1)
    censored_idx = np.flatnonzero(events != 1)

    if (
        n_samples < min_samples_for_holdout
        or event_idx.size < min_group_size
        or censored_idx.size < min_group_size
    ):
        return {
            "train_idx": all_indices,
            "eval_idx": all_indices,
            "evaluation_mode": "apparent",
            "evaluation_note": (
                "Holdout evaluation was skipped because the analyzable cohort was too small "
                "or one outcome stratum was too sparse."
            ),
        }

    rng = np.random.default_rng(random_seed)

    def _sample_eval(indices: np.ndarray) -> np.ndarray:
        shuffled = rng.permutation(indices)
        n_eval = max(1, int(round(indices.size * holdout_fraction)))
        n_eval = min(n_eval, indices.size - 1)
        return shuffled[:n_eval]

    eval_idx = np.unique(np.concatenate([_sample_eval(event_idx), _sample_eval(censored_idx)]))
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[eval_idx] = False
    train_idx = np.flatnonzero(train_mask)

    train_event_count = int(np.sum(events[train_idx] == 1))
    eval_event_count = int(np.sum(events[eval_idx] == 1))
    if train_idx.size < 10 or train_event_count == 0 or eval_event_count == 0:
        return {
            "train_idx": all_indices,
            "eval_idx": all_indices,
            "evaluation_mode": "apparent",
            "evaluation_note": (
                "Holdout evaluation was skipped because the deterministic split did not "
                "leave enough comparable events in both partitions."
            ),
        }

    return {
        "train_idx": train_idx,
        "eval_idx": eval_idx,
        "evaluation_mode": "holdout",
        "evaluation_note": (
            f"Reported C-index is computed on a deterministic holdout split with "
            f"{train_idx.size} training samples and {eval_idx.size} evaluation samples."
        ),
    }


def _build_holdout_split(
    events: np.ndarray,
    random_seed: int,
    *,
    source_rows: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """External holdout shared with the classical ML models.

    Uses the same stratified 70/30 split (``survival_toolkit.evaluation``) as
    the ML comparison, so ML and DL rows scored with the same seed come from
    identical evaluation patients.
    """
    event_array = np.asarray(events).astype(int).ravel()
    train_positions, eval_positions, mode = stratified_holdout_indices(event_array, random_state=int(random_seed))
    fingerprint = evaluation_split_fingerprint(
        None if source_rows is None else list(source_rows),
        [(train_positions, eval_positions)],
        kind=mode,
    )
    if mode != "holdout":
        return {
            "train_idx": train_positions,
            "eval_idx": eval_positions,
            "evaluation_mode": "apparent",
            "evaluation_note": (
                "Holdout evaluation was skipped because the analyzable cohort was too small "
                "or one outcome stratum was too sparse."
            ),
            "evaluation_split_fingerprint": fingerprint,
        }
    return {
        "train_idx": np.asarray(train_positions, dtype=int),
        "eval_idx": np.asarray(eval_positions, dtype=int),
        "evaluation_mode": "holdout",
        "evaluation_note": (
            f"Reported C-index is computed on the shared stratified holdout split "
            f"({train_positions.size} training and {eval_positions.size} evaluation samples; "
            f"{DEFAULT_HOLDOUT_FRACTION:.0%} holdout, identical to the ML comparison for the same seed)."
        ),
        "evaluation_split_fingerprint": fingerprint,
    }


def _build_monitor_indices(
    train_idx: Sequence[int] | torch.Tensor,
    events: Sequence[int] | np.ndarray | torch.Tensor,
    random_seed: int,
    holdout_fraction: float = 0.2,
) -> np.ndarray | None:
    """Create an internal monitoring subset drawn from the training partition.

    This subset is used only for checkpoint selection. It must never overlap
    with the external evaluation fold.
    """
    if TORCH_AVAILABLE and isinstance(train_idx, torch.Tensor):
        train_idx_np = train_idx.detach().cpu().numpy().astype(int).ravel()
    else:
        train_idx_np = np.asarray(train_idx, dtype=int).ravel()

    if train_idx_np.size == 0:
        return None

    if TORCH_AVAILABLE and isinstance(events, torch.Tensor):
        event_values = events.detach().cpu().numpy().astype(int).ravel()
    else:
        event_values = np.asarray(events, dtype=int).ravel()

    local_split = _build_evaluation_split(
        event_values[train_idx_np],
        random_seed=random_seed,
        holdout_fraction=holdout_fraction,
        min_samples_for_holdout=24,
        min_group_size=2,
    )
    if str(local_split.get("evaluation_mode")) != "holdout":
        return None

    return train_idx_np[np.asarray(local_split["eval_idx"], dtype=int)]


def _reject_unaligned_monitor_indices(
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None,
    prepared_data: dict[str, Any] | None,
) -> None:
    """Caller-supplied monitor rows are only meaningful for caller-prepared tensors.

    Raw-dataframe inputs are cleaned and (for holdout) re-ordered train-first
    inside the trainer, so external row positions would silently point at the
    wrong patients, possibly evaluation rows.
    """
    if monitor_indices is not None and prepared_data is None:
        raise ValueError(
            "monitor_indices can only be supplied together with prepared_data; "
            "with a raw dataframe the trainer builds its own monitor subset."
        )


def _resolve_monitor_indices(
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None,
    *,
    train_idx: torch.Tensor,
    events: torch.Tensor,
    random_seed: int,
) -> torch.Tensor | None:
    if monitor_indices is None:
        monitor_np = _build_monitor_indices(train_idx, events, random_seed)
    elif TORCH_AVAILABLE and isinstance(monitor_indices, torch.Tensor):
        monitor_np = monitor_indices.detach().cpu().numpy().astype(int).ravel()
    else:
        monitor_np = np.asarray(monitor_indices, dtype=int).ravel()

    if monitor_np is None or monitor_np.size == 0:
        return None
    return torch.as_tensor(monitor_np, dtype=torch.long)


def _select_artifact_indices(
    *,
    total_n: int,
    eval_idx: torch.Tensor,
    evaluation_mode: str,
) -> torch.Tensor:
    """Use holdout rows for descriptive artifacts when a true holdout exists."""
    if evaluation_mode == "holdout" and eval_idx.numel() > 0:
        return eval_idx
    return torch.arange(total_n, dtype=torch.long)


def _artifact_scope_label(evaluation_mode: str) -> str:
    return "evaluation_subset" if evaluation_mode == "holdout" else "analyzable_cohort"


def _batching_metadata(
    *,
    requested_batch_size: int,
    effective_batch_size: int,
    optimization_mode: str,
    note: str,
) -> dict[str, Any]:
    return {
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "optimization_mode": optimization_mode,
        "batching_note": note,
    }


def _monitor_c_index(
    model: nn.Module,
    x_all: torch.Tensor,
    t_all: torch.Tensor,
    e_all: torch.Tensor,
    monitor_idx: torch.Tensor,
) -> float | None:
    with torch.inference_mode():
        monitor_risk = model(x_all[monitor_idx])
    return _compute_c_index_torch(monitor_risk, t_all[monitor_idx], e_all[monitor_idx])


def _discrete_survival_from_pmf(
    pmf: torch.Tensor,
    bin_widths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return survival at bin edges and a restricted-mean survival score.

    The last PMF column is treated as a tail bucket beyond the observed horizon.
    Survival at the final observed edge therefore remains positive unless the
    model assigns that bucket vanishing probability.
    """
    if pmf.shape[1] < 2:
        raise ValueError("Discrete survival outputs require at least one observed bin and one tail bin.")

    log_pmf = torch.log(torch.clamp(pmf, min=1e-12))
    raw_log_survival = torch.flip(
        torch.logcumsumexp(torch.flip(log_pmf, dims=[1]), dim=1),
        dims=[1],
    )
    log_survival_at_edges = torch.cat(
        [
            torch.zeros((pmf.shape[0], 1), device=pmf.device, dtype=pmf.dtype),
            raw_log_survival[:, 1:],
        ],
        dim=1,
    )
    survival_at_edges = torch.exp(log_survival_at_edges)
    rmst = torch.sum(survival_at_edges[:, :-1] * bin_widths.view(1, -1), dim=1)
    return survival_at_edges, -rmst


def _digitize_time_bins(
    time_values: Sequence[float] | np.ndarray,
    bin_edges: np.ndarray,
    num_time_bins: int,
    *,
    preserve_tail_overflow: bool = False,
) -> np.ndarray:
    """Map observed times into discrete bins with optional tail-bucket support.

    The observed horizon is defined by ``bin_edges[-1]``. When
    ``preserve_tail_overflow`` is true, times beyond that horizon are assigned
    to the explicit tail bucket at index ``num_time_bins`` instead of being
    clipped into the last in-horizon event bin.
    """
    values = np.asarray(time_values, dtype=float).reshape(-1)
    indices = np.digitize(values, bin_edges[1:-1]).astype(int, copy=False)
    indices = np.clip(indices, 0, num_time_bins - 1)
    if preserve_tail_overflow:
        indices = indices.copy()
        indices[values > float(bin_edges[-1])] = num_time_bins
    return indices


def _append_evaluation_note(note: str | None, extra_note: str | None) -> str:
    base = str(note or "").strip()
    extra = str(extra_note or "").strip()
    if not base:
        return extra
    if not extra or extra in base:
        return base
    return f"{base} {extra}"


def _discrete_time_tail_bucket_note(
    time_values: torch.Tensor | np.ndarray | Sequence[float],
    indices: torch.Tensor | np.ndarray | Sequence[int] | None,
    bin_edges: np.ndarray,
    *,
    scope_label: str,
) -> str | None:
    if indices is None:
        return None
    index_arr = np.asarray(indices, dtype=int).reshape(-1)
    if index_arr.size == 0:
        return None
    time_arr = np.asarray(time_values, dtype=float).reshape(-1)
    scoped_times = time_arr[index_arr]
    overflow_count = int(np.sum(scoped_times > float(bin_edges[-1])))
    if overflow_count <= 0:
        return None
    return (
        f"{overflow_count} {scope_label} sample(s) exceeded the training-time horizon used for discrete bins "
        "and were assigned to the tail bucket; interpret late-time tail-bin estimates cautiously."
    )


def _gradient_feature_importance(
    model: Any,
    x_tensor: torch.Tensor,
    *,
    output_to_score: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[float]:
    """Compute gradient-based feature importance: mean |grad of output w.r.t. input|."""
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        x_input = x_tensor.clone().detach().requires_grad_(True)
        output = model(x_input)
        if output_to_score is not None:
            output = output_to_score(output)
        elif output.dim() > 1:
            output = output.sum(dim=1)
        grad = torch.autograd.grad(output.sum(), x_input, retain_graph=False, create_graph=False)[0]
        if grad is None:
            return [0.0] * x_tensor.shape[1]
        importance = grad.abs().mean(dim=0).detach().cpu().numpy()
        # A column that never varies in these rows (for example an all-zero
        # "__missing"/"__unknown" indicator) cannot drive any prediction here;
        # its gradient only reflects untrained weights, so it gets no salience.
        if x_tensor.shape[0] > 1:
            column_spread = (x_tensor.max(dim=0).values - x_tensor.min(dim=0).values).detach().cpu().numpy()
            importance = np.where(column_spread > 0.0, importance, 0.0)
        return [float(v) for v in importance]
    finally:
        if was_training:
            model.train()


def _require_finite_loss(loss: torch.Tensor, *, context: str) -> None:
    if not torch.isfinite(loss):
        raise ValueError(f"{context} became NaN or Inf during training.")


def _scientific_summary_dl(
    model_name: str,
    c_index: float | None,
    train_samples: int,
    eval_samples: int,
    train_events: int | None,
    n_features: int,
    epochs: int,
    loss_history: list[float],
    evaluation_mode: str,
    evaluation_note: str | None = None,
    dropped_nonpositive_time_rows: int = 0,
    refit_note: str | None = None,
    reported_epochs: int | None = None,
    unseen_category_rows: int = 0,
    dropped_missing_outcome_rows: int = 0,
    time_column_note: str | None = None,
) -> dict[str, Any]:
    """Build an insight board dict for deep learning models.

    ``loss_history`` is the early-stopping run; ``reported_epochs`` (when given) is the
    number of epochs behind the reported weights, which differs after a refit.
    ``dropped_missing_outcome_rows`` and ``time_column_note`` come from the cohort builder
    (``_cohort_summary_fields``).
    """
    metric_name = _metric_name_for_evaluation(evaluation_mode)
    c_val = float(c_index) if c_index is not None else None

    early_stopping_epochs = int(len(loss_history))
    epochs_trained = int(reported_epochs) if reported_epochs is not None else early_stopping_epochs

    if c_val is None:
        status = "review"
    elif c_val > 0.65:
        status = "robust"
    elif c_val >= 0.55:
        status = "review"
    else:
        status = "caution"
    # An apparent (resubstitution) C-index is optimistic by construction and
    # can never earn a "robust" badge, however high it is.
    if status == "robust" and evaluation_mode not in {"holdout", "repeated_cv"}:
        status = "review"

    strengths: list[str] = [
        f"{model_name} trained for {epochs_trained or epochs} epoch(s) on {train_samples} sample(s)"
        + (
            f" ({int(train_events)} events)"
            if train_events is not None
            else ""
        )
        + f" with {n_features} features.",
    ]
    if len(loss_history) >= 2 and loss_history[-1] < loss_history[0]:
        strengths.append("Training loss decreased over epochs, indicating successful optimization.")
    if refit_note:
        strengths.append(refit_note)

    cautions: list[str] = []
    next_steps: list[str] = []

    if evaluation_note:
        cautions.append(evaluation_note)
    cautions.extend(_cohort_cautions(dropped_nonpositive_time_rows, dropped_missing_outcome_rows, time_column_note))
    if evaluation_mode == "holdout" and int(unseen_category_rows) > 0:
        # Imported only when needed: repeated-CV worker processes need not load the ML module.
        from survival_toolkit.ml_models import _unseen_category_caution

        unseen_caution = _unseen_category_caution(int(unseen_category_rows), "evaluation")
        if unseen_caution:
            cautions.append(unseen_caution)

    if train_samples < 100:
        cautions.append("Sample size is small for a deep learning model; results may be unreliable.")
        next_steps.append("Consider using classical survival models (Cox PH, KM) for small datasets.")
    if train_events is not None:
        events_per_feature = float(train_events) / max(float(n_features), 1.0)
        if events_per_feature < 10:
            cautions.append(
                f"Training events per feature is {events_per_feature:.1f}; deep models can overfit quickly when feature count is high relative to events."
            )
    if c_val is None:
        cautions.append(
            f"{metric_name} could not be computed for this run, usually because the evaluation subset had no usable event/comparison pairs."
        )
        next_steps.append("Use a larger evaluation split or a comparison path that preserves evaluable events before interpreting discrimination.")
    elif c_val < 0.55:
        cautions.append(f"{metric_name} ({c_val:.3f}) suggests weak discrimination.")
        next_steps.append("Try different architectures, hyperparameters, or additional features.")
    elif c_val < 0.65:
        cautions.append(f"{metric_name} ({c_val:.3f}) indicates moderate discrimination.")
        next_steps.append("Validate on held-out data; consider ensemble approaches.")
    elif c_val >= 0.70:
        strengths.append(
            "C-index is well above chance-level ranking (0.50), which can be useful for screening if independent validation agrees."
        )

    if evaluation_mode == "holdout":
        cautions.append(
            "A deterministic holdout scores one split, and the point estimate alone does not show its uncertainty; "
            "judge it by the bootstrap intervals over the test patients of a model comparison or by repeated cross-validation."
        )

    if len(loss_history) >= 5:
        tail = loss_history[-5:]
        if max(tail) - min(tail) < 1e-6:
            cautions.append("Loss plateaued in the final epochs; model may benefit from more epochs or a learning rate change.")
    if model_name == "DeepSurv":
        cautions.append(
            "This DeepSurv path uses full-batch Cox optimization; the batch-size control is recorded for reproducibility but does not change optimization."
        )
    if model_name == "Survival Transformer":
        cautions.append(
            "This Survival Transformer path uses full-batch Cox optimization; the batch-size control is recorded for reproducibility but does not change optimization."
        )
        cautions.append(
            "The transformer treats each feature as a token with a learned feature-identity embedding. Interpret it as an exploratory tabular attention model and validate against simpler baselines."
        )
    if model_name == "DeepHit":
        cautions.append(
            "DeepHit uses a stabilized ranking-loss term; wide discrete-time bin grids still need review on strongly right-skewed survival data."
        )
        cautions.append(
            "This implementation uses a smooth stabilized ranking-loss surrogate rather than a literal step-function reference formulation."
        )
    if model_name == "Neural MTLR":
        cautions.append(
            "This path uses a neuralized right-cumulative MTLR parameterization; treat it as a flexible discrete-time extension rather than a line-by-line clone of any single reference codebase."
        )
    if model_name == "Survival VAE":
        cautions.append(
            "This path should be interpreted as a VAE-inspired latent representation model for clustering and risk screening, not as a validated generative simulator or uncertainty estimator."
        )
    cautions.append(
        "Deep-model summaries currently report discrimination (C-index) only; SurvStudio does not yet compute IBS for these paths, so calibration/error comparisons are not symmetric with the ML module."
    )

    if not next_steps:
        next_steps.append("Validate with external data or cross-validation to confirm generalizability.")

    return {
        "status": status,
        "headline": (
            f"{model_name} estimated a {metric_name.lower()} of {c_val:.3f} on {evaluation_mode.replace('_', ' ')} evaluation."
            if c_val is not None
            else f"{model_name} completed {evaluation_mode.replace('_', ' ')} evaluation, but the {metric_name.lower()} was not estimable for this run."
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": metric_name, "value": c_val},
            {"label": "Evaluation mode", "value": evaluation_mode},
            {"label": "Training samples", "value": train_samples},
            {"label": "Training events", "value": train_events},
            {"label": "Dropped for negative time", "value": int(dropped_nonpositive_time_rows) or None},
            {"label": "Dropped for missing outcome", "value": int(dropped_missing_outcome_rows) or None},
            {"label": "Evaluation samples", "value": eval_samples},
            {"label": "Features", "value": n_features},
            {"label": "Epochs", "value": epochs_trained or epochs},
            {"label": "Final loss", "value": float(loss_history[-1]) if loss_history else None},
            *(
                [{"label": "Early-stopping epochs", "value": early_stopping_epochs}]
                if early_stopping_epochs and early_stopping_epochs != epochs_trained
                else []
            ),
        ],
    }


def _make_survival_curve(timeline: list[float], survival: list[float]) -> dict[str, list[float]]:
    """Build a KM-style survival curve dict."""
    return {"timeline": timeline, "survival": survival}


def _update_early_stopping(
    monitor_value: float,
    *,
    best_value: float | None,
    wait_count: int,
    patience: int | None,
    min_delta: float,
    goal: str = "min",
    model: Any,
    best_state: dict[str, torch.Tensor] | None,
) -> tuple[float | None, int, dict[str, torch.Tensor] | None, bool]:
    """Track the best monitored metric and decide whether to stop."""
    if patience is None or patience <= 0:
        return best_value, wait_count, best_state, False
    if goal not in {"min", "max"}:
        raise ValueError("Early-stopping goal must be 'min' or 'max'.")
    if not math.isfinite(monitor_value):
        wait_count += 1
        return best_value, wait_count, best_state, wait_count >= patience

    improved = best_value is None or (
        monitor_value < (best_value - min_delta)
        if goal == "min"
        else monitor_value > (best_value + min_delta)
    )
    if improved:
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        return float(monitor_value), 0, state, False

    wait_count += 1
    return best_value, wait_count, best_state, wait_count >= patience


def _early_stopping_active(patience: int | None) -> bool:
    return patience is not None and int(patience) > 0


def _fit_rows_excluding_monitor(
    train_idx: torch.Tensor,
    monitor_idx: torch.Tensor | None,
    events: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Rows used for gradient updates: the training partition minus the monitor subset.

    The early-stopping monitor must be held out from fitting; otherwise the
    monitored metric tracks training fit and never signals overfitting.
    """
    if monitor_idx is None or monitor_idx.numel() == 0:
        return train_idx, None
    keep = ~torch.isin(train_idx, monitor_idx)
    fit_idx = train_idx[keep]
    if fit_idx.numel() < 10 or float(events[fit_idx].sum().item()) <= 0.0:
        return train_idx, None
    return fit_idx, monitor_idx


def _training_run_metadata(
    loss_history: list[float],
    monitor_loss_history: list[float] | None,
    requested_epochs: int,
    *,
    monitor_goal: str = "min",
    patience: int | None = None,
    min_delta: float = 0.0,
) -> dict[str, int | bool | None]:
    monitor_loss_history = list(monitor_loss_history or [])
    if monitor_goal not in {"min", "max"}:
        raise ValueError("Monitor goal must be 'min' or 'max'.")
    epochs_trained = int(len(loss_history))
    # Replay the checkpoint rule of _update_early_stopping so the reported
    # epoch is the one whose weights were restored (improvement > min_delta).
    best_monitor_epoch: int | None = None
    if monitor_loss_history and _early_stopping_active(patience):
        best_value: float | None = None
        for epoch_number, value in enumerate(monitor_loss_history, start=1):
            if not math.isfinite(float(value)):
                continue
            improved = best_value is None or (
                float(value) < best_value - float(min_delta)
                if monitor_goal == "min"
                else float(value) > best_value + float(min_delta)
            )
            if improved:
                best_value = float(value)
                best_monitor_epoch = epoch_number
    stopped_early = bool(monitor_loss_history) and epochs_trained < int(requested_epochs)
    return {
        "epochs_trained": epochs_trained,
        "best_monitor_epoch": best_monitor_epoch,
        "restored_best_checkpoint": best_monitor_epoch is not None,
        "stopped_early": stopped_early,
        "max_epochs_requested": int(requested_epochs),
    }


def _serialized_torch_training(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run deep-learning training one call at a time in this process, on pinned torch threads.

    ``_seed_torch`` seeds process-wide generators (numpy, ``random``, torch) and weight
    initialisation and dropout draw from them, so two trainings interleaved on server
    threads would consume each other's random numbers and the same seed would not
    reproduce. Worker processes of the parallel repeated-CV path each have their own lock.

    A job queued behind another training keeps polling its cancellation signal while it
    waits, so a cancelled request does not hold a heavy-job slot until the lock frees.
    Training runs with ``_DEEP_TORCH_NUM_THREADS`` intra-op threads in every path.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        raise_if_cancelled()
        while not _TORCH_TRAINING_LOCK.acquire(timeout=_CANCELLATION_POLL_SECONDS):
            raise_if_cancelled()
        try:
            with _pinned_torch_threads():
                return func(*args, **kwargs)
        finally:
            _TORCH_TRAINING_LOCK.release()

    return wrapper


class _DeepTrainingContext(NamedTuple):
    """Prepared tensors and row partitions for one deep-model training run."""

    data: dict[str, Any]
    train_idx: Any
    eval_idx: Any
    fit_idx: Any
    monitor_idx: Any | None
    evaluation_mode: str
    evaluation_note: str
    # Evaluation rows with a categorical level the training rows never show (holdout only).
    unseen_category_rows: int = 0

    @property
    def x_all(self) -> Any:
        return self.data["X_tensor"]

    @property
    def t_all(self) -> Any:
        return self.data["time_tensor"]

    @property
    def e_all(self) -> Any:
        return self.data["event_tensor"]


def _prepare_deep_training_context(
    df: pd.DataFrame | None,
    *,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None,
    event_positive_value: Any,
    random_seed: int,
    prepared_data: dict[str, Any] | None,
    evaluation_split: dict[str, Any] | None,
    monitor_indices: Sequence[int] | np.ndarray | Any | None,
    early_stopping_patience: int | None,
) -> _DeepTrainingContext:
    _reject_unaligned_monitor_indices(monitor_indices, prepared_data)
    data, eval_split = _prepare_deep_training_inputs(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
        prepared_data=prepared_data,
        evaluation_split=evaluation_split,
    )
    train_idx = torch.as_tensor(eval_split["train_idx"], dtype=torch.long)
    eval_idx = torch.as_tensor(eval_split["eval_idx"], dtype=torch.long)
    monitor_idx = _resolve_monitor_indices(
        monitor_indices,
        train_idx=train_idx,
        events=data["event_tensor"],
        random_seed=random_seed,
    )
    if monitor_idx is not None:
        outside_training = ~torch.isin(monitor_idx, train_idx)
        if bool(outside_training.any()):
            # Early stopping on evaluation rows would select the epoch on the rows that are
            # then scored, an optimistic (leaky) holdout estimate.
            raise ValueError(
                f"monitor_indices contain {int(outside_training.sum())} row(s) outside the training "
                "partition; early-stopping monitor rows must be a subset of train_idx."
            )
    if not _early_stopping_active(early_stopping_patience):
        monitor_idx = None
    fit_idx, monitor_idx = _fit_rows_excluding_monitor(train_idx, monitor_idx, data["event_tensor"])
    evaluation_mode = str(eval_split["evaluation_mode"])
    return _DeepTrainingContext(
        data=data,
        train_idx=train_idx,
        eval_idx=eval_idx,
        fit_idx=fit_idx,
        monitor_idx=monitor_idx,
        evaluation_mode=evaluation_mode,
        evaluation_note=str(eval_split["evaluation_note"]),
        unseen_category_rows=int(eval_split.get("unseen_category_rows", 0) or 0) if evaluation_mode == "holdout" else 0,
    )


class _FitPhase(NamedTuple):
    """One optimisation run: the fitted model, its histories, and model-specific extras."""

    model: Any
    loss_history: list[float]
    monitor_history: list[float]
    monitor_used: bool
    aux: dict[str, Any] | None = None


def _make_optimizer(model: Any, learning_rate: float) -> Any:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_ADAM_WEIGHT_DECAY)


def _train_epochs(
    model: Any,
    *,
    epochs: int,
    step: Callable[[], float],
    monitor: Callable[[], float | None] | None,
    monitor_goal: str,
    patience: int | None,
    min_delta: float,
) -> tuple[list[float], list[float], bool]:
    """Run ``step`` once per epoch with optional early stopping on ``monitor``.

    ``monitor`` returns the held-out metric after each epoch (``None`` when the monitor
    rows cannot be scored, which switches monitoring off for the run). The weights of
    the best monitored epoch are restored at the end. Returns the training-loss history,
    the monitor history, and whether the monitor produced any value.
    """
    loss_history: list[float] = []
    monitor_history: list[float] = []
    best_value: float | None = None
    best_state: dict[str, Any] | None = None
    wait_count = 0
    for _epoch in range(int(epochs)):
        raise_if_cancelled()
        model.train()
        loss_history.append(float(step()))
        if monitor is None:
            continue
        model.eval()
        value = monitor()
        if value is None:
            monitor = None
            continue
        monitor_history.append(float(value))
        best_value, wait_count, best_state, should_stop = _update_early_stopping(
            float(value),
            best_value=best_value,
            wait_count=wait_count,
            patience=patience,
            min_delta=min_delta,
            goal=monitor_goal,
            model=model,
            best_state=best_state,
        )
        if should_stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return loss_history, monitor_history, bool(monitor_history)


def _fit_with_refit(
    fit_phase: Callable[[Any, Any | None, int, int | None], _FitPhase],
    context: _DeepTrainingContext,
    *,
    epochs: int,
    patience: int | None,
    min_delta: float,
    monitor_goal: str,
    refit: bool,
) -> tuple[_FitPhase, _FitPhase, dict[str, Any]]:
    """Early stopping on a held-out monitor subset, then a refit on all training rows.

    The monitor subset must stay out of gradient updates for early stopping to measure
    generalisation, but keeping it out of the final model would train deep models on
    about 20% fewer rows than the classical models they are ranked against. After the
    first run selects the number of epochs, a fresh model (same seed) is fitted on the
    whole training partition for that many epochs.

    Returns ``(final_phase, early_stopping_phase, refit_info)``.
    """
    first = fit_phase(context.fit_idx, context.monitor_idx, epochs, patience)
    held_out_rows = context.monitor_idx is not None and int(context.fit_idx.numel()) < int(context.train_idx.numel())
    if not (refit and held_out_rows):
        return first, first, {"refit_on_training_partition": False, "refit_epochs": None}
    meta = _training_run_metadata(
        first.loss_history,
        first.monitor_history,
        epochs,
        monitor_goal=monitor_goal,
        patience=patience,
        min_delta=min_delta,
    )
    refit_epochs = int(meta["best_monitor_epoch"] or len(first.loss_history))
    final = fit_phase(context.train_idx, None, refit_epochs, None)
    return final, first, {"refit_on_training_partition": True, "refit_epochs": refit_epochs}


def _deep_training_fields(
    context: _DeepTrainingContext,
    final: _FitPhase,
    first: _FitPhase,
    refit_info: dict[str, Any],
    *,
    epochs: int,
    patience: int | None,
    min_delta: float,
    monitor_goal: str,
    random_seed: int,
) -> dict[str, Any]:
    """Seeds, histories, early-stopping metadata, and sample counts shared by every trainer.

    ``epochs_trained`` counts the epochs behind the reported weights (the refit, or the
    restored best epoch); ``early_stopping_epochs`` is how long the early-stopping run
    lasted, the length of ``loss_history`` and ``monitor_history``.
    """
    meta = _training_run_metadata(
        first.loss_history,
        first.monitor_history,
        epochs,
        monitor_goal=monitor_goal,
        patience=patience,
        min_delta=min_delta,
    )
    refitted = bool(refit_info["refit_on_training_partition"])
    final_rows = context.train_idx if refitted else context.fit_idx
    if refitted:
        reported_epochs = int(len(final.loss_history))
    elif meta["restored_best_checkpoint"]:
        reported_epochs = int(meta["best_monitor_epoch"])
    else:
        reported_epochs = int(meta["epochs_trained"])
    return {
        "training_seed": random_seed,
        "split_seed": random_seed,
        "monitor_seed": random_seed,
        "loss_history": first.loss_history,
        "monitor_history": first.monitor_history,
        "refit_loss_history": final.loss_history if refitted else [],
        "best_monitor_epoch": meta["best_monitor_epoch"],
        "stopped_early": meta["stopped_early"],
        "max_epochs_requested": meta["max_epochs_requested"],
        "epochs_trained": reported_epochs,
        "early_stopping_epochs": meta["epochs_trained"],
        "refit_on_training_partition": refitted,
        "refit_epochs": refit_info["refit_epochs"],
        "torch_num_threads": int(torch.get_num_threads()),
        "n_samples": context.data["n_samples"],
        "training_samples": int(context.train_idx.numel()),
        # Rows behind the reported weights (the whole training partition after a refit).
        "fit_samples": int(final_rows.numel()),
        "early_stopping_fit_samples": int(context.fit_idx.numel()),
        "monitor_samples": int(context.monitor_idx.numel()) if first.monitor_used and context.monitor_idx is not None else 0,
        "evaluation_samples": int(context.eval_idx.numel()),
        "training_events": int(context.e_all[context.train_idx].sum().item()),
        "evaluation_events": int(context.e_all[context.eval_idx].sum().item()),
        "n_features": context.data["n_features"],
        # Input columns that were one-hot coded (declared plus auto-coded); None for
        # caller-prepared tensors that do not record them.
        "categorical_features": (
            None if context.data.get("categorical_features") is None else list(context.data["categorical_features"])
        ),
    }


def _deep_fit_summary_counts(
    context: _DeepTrainingContext,
    refit_info: dict[str, Any],
) -> tuple[int, int, str | None]:
    """Rows, events, and a refit note for the scientific summary of the reported model."""
    refitted = bool(refit_info["refit_on_training_partition"])
    final_rows = context.train_idx if refitted else context.fit_idx
    events = int(context.e_all[final_rows].sum().item())
    note = None
    if refitted:
        note = (
            f"Early stopping picked epoch {refit_info['refit_epochs']} on a held-out monitor subset of "
            f"{int(context.monitor_idx.numel())} training rows; the reported model was then refit on all "
            f"{int(context.train_idx.numel())} training rows for that many epochs."
        )
    return int(final_rows.numel()), events, note


def _holdout_and_apparent_c_index(
    context: _DeepTrainingContext,
    risk_scores: Any,
) -> tuple[float | None, float | None, float | None, str, str, list[float] | None]:
    """Holdout C-index (apparent fallback) with the matching evaluation mode and note.

    The last value is the evaluation rows' risk scores when the C-index is a holdout
    estimate, for paired comparisons with other models; comparisons remove it.
    """
    # Apparent = resubstitution on the training partition only.
    apparent_c_index = _compute_c_index_torch(
        risk_scores[context.train_idx], context.t_all[context.train_idx], context.e_all[context.train_idx]
    )
    holdout_c_index = _compute_c_index_torch(
        risk_scores[context.eval_idx], context.t_all[context.eval_idx], context.e_all[context.eval_idx]
    )
    c_index = holdout_c_index if holdout_c_index is not None else apparent_c_index
    evaluation_mode = context.evaluation_mode
    evaluation_note = context.evaluation_note
    if holdout_c_index is None and evaluation_mode == "holdout":
        evaluation_mode = "holdout_fallback_apparent"
        evaluation_note = (
            "A deterministic holdout split was available, but the holdout subset did not "
            "support a comparable concordance estimate; the reported C-index is apparent."
        )
    holdout_risk = None
    if holdout_c_index is not None and evaluation_mode == "holdout":
        holdout_risk = risk_scores[context.eval_idx].detach().cpu().numpy().reshape(-1).astype(float).tolist()
    return c_index, apparent_c_index, holdout_c_index, evaluation_mode, evaluation_note, holdout_risk


def _time_bin_grid(time_values: np.ndarray, num_time_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width discrete-time bin edges and centres over the fitting rows' time range."""
    t_min, t_max = float(np.min(time_values)), float(np.max(time_values))
    if t_max <= t_min:
        t_max = t_min + 1e-6
    bin_edges = np.linspace(t_min, t_max, num_time_bins + 1)
    return bin_edges, (bin_edges[:-1] + bin_edges[1:]) / 2.0


def _minibatch_step(
    model: Any,
    optimizer: Any,
    loader: Any,
    loss_fn: Callable[[Any, Any, Any], Any],
    *,
    context_label: str,
) -> Callable[[], float]:
    def _step() -> float:
        epoch_losses: list[float] = []
        for x_batch, bin_batch, event_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), bin_batch, event_batch)
            _require_finite_loss(loss, context=context_label)
            loss.backward()
            _clip_gradients(model)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        return float(np.mean(epoch_losses))

    return _step


def _full_batch_step(
    model: Any,
    optimizer: Any,
    loss_fn: Callable[[], Any],
    *,
    context_label: str,
) -> Callable[[], float]:
    def _step() -> float:
        optimizer.zero_grad()
        loss = loss_fn()
        _require_finite_loss(loss, context=context_label)
        loss.backward()
        _clip_gradients(model)
        optimizer.step()
        return float(loss.item())

    return _step


class _DeepRunSettings(NamedTuple):
    """Training settings shared by every model of a deep-learning comparison."""

    time_column: str
    event_column: str
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    event_positive_value: Any
    learning_rate: float
    epochs: int
    batch_size: int
    early_stopping_patience: int | None
    early_stopping_min_delta: float

    def task_fields(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "event_column": self.event_column,
            "features": list(self.features),
            "categorical_features": list(self.categorical_features),
            "event_positive_value": self.event_positive_value,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }


def _record_fold_failure(
    errors: list[dict[str, Any]],
    model_names: Sequence[str],
    repeat: Any,
    fold: Any,
    exc: BaseException,
) -> None:
    """Record one failed fold for each named model; errors that must propagate are re-raised.

    The failure is logged once with its traceback, and the recorded message keeps the cause
    of a library error (``_failure_message``).
    """
    if _must_propagate_deep(exc):
        raise exc
    logger.exception("Deep-learning repeated-CV repeat %s, fold %s failed.", repeat, fold, exc_info=exc)
    message = _failure_message(exc)
    errors.extend({"model": str(name), "repeat": repeat, "fold": fold, "error": message} for name in model_names)


def _record_fold_error(errors: list[dict[str, Any]], model_name: str, repeat: Any, fold: Any, exc: BaseException) -> None:
    _record_fold_failure(errors, [model_name], repeat, fold, exc)


@user_input_boundary
@_serialized_torch_training
def compare_deep_survival_models(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    num_time_bins: int = 50,
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    latent_dim: int = 8,
    n_clusters: int = 3,
    evaluation_strategy: str = "holdout",
    cv_folds: int = 5,
    cv_repeats: int = 3,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    parallel_jobs: int = 1,
    included_models: Sequence[str] | None = None,
    locked_test_fraction: float | None = None,
) -> dict[str, Any]:
    """Train the bundled deep survival models on one feature set and rank them.

    Holdout uses the same stratified split as the ML comparison. With
    ``evaluation_strategy="repeated_cv"`` and ``locked_test_fraction`` set, a
    stratified test set is reserved first, CV runs on the development set only,
    and every model is refit on the development set and scored once on the
    untouched test set.
    """
    _require_torch()

    if evaluation_strategy not in {"holdout", "repeated_cv"}:
        raise ValueError(
            f"Unknown deep-learning evaluation strategy '{evaluation_strategy}'; use 'holdout' or 'repeated_cv'."
        )
    if evaluation_strategy != "repeated_cv" and locked_test_fraction is not None:
        raise ValueError("A locked test set is only available with repeated cross-validation.")
    locked_test_fraction = _validated_locked_test_fraction(locked_test_fraction)
    epochs, batch_size, learning_rate = _validated_training_settings(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    hidden_layers = _validated_hidden_layers(hidden_layers)
    trainer_specs = _deep_trainer_specs(
        hidden_layers=hidden_layers,
        dropout=dropout,
        num_time_bins=num_time_bins,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        latent_dim=latent_dim,
        n_clusters=n_clusters,
    )
    if included_models is not None:
        included = {str(name) for name in included_models}
        trainer_specs = [spec for spec in trainer_specs if spec[0] in included]
        if not trainer_specs:
            raise ValueError("No deep-learning models remain after applying the requested model filter.")
    if any(spec[0] == "Survival Transformer" for spec in trainer_specs):
        _validated_positive_integer(n_heads, "The number of attention heads")
    settings = _DeepRunSettings(
        time_column=time_column,
        event_column=event_column,
        features=tuple(features),
        categorical_features=tuple(categorical_features or []),
        event_positive_value=event_positive_value,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
    if evaluation_strategy == "repeated_cv":
        return _deep_repeated_cv_comparison(
            df,
            trainer_specs,
            settings,
            random_seed=random_seed,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
            parallel_jobs=parallel_jobs,
            locked_test_fraction=locked_test_fraction,
        )
    return _deep_holdout_comparison(df, trainer_specs, settings, random_seed=random_seed)


def _deep_holdout_comparison(
    df: pd.DataFrame,
    trainer_specs: list[tuple[str, Any, dict[str, Any]]],
    settings: _DeepRunSettings,
    *,
    random_seed: int,
) -> dict[str, Any]:
    shared_data, shared_eval_split = _prepare_deep_training_inputs(
        df,
        time_column=settings.time_column,
        event_column=settings.event_column,
        features=settings.features,
        categorical_features=settings.categorical_features,
        event_positive_value=settings.event_positive_value,
        random_seed=random_seed,
    )
    shared_monitor_indices = _build_monitor_indices(
        shared_eval_split["train_idx"],
        shared_data["event_tensor"],
        random_seed,
    )
    comparison: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    holdout_risks: dict[str, tuple[None, list[float]]] = {}
    for model_name, trainer, extra_kwargs in trainer_specs:
        raise_if_cancelled()
        try:
            started = time.monotonic()
            result = trainer(
                df,
                time_column=settings.time_column,
                event_column=settings.event_column,
                features=list(settings.features),
                categorical_features=list(settings.categorical_features),
                event_positive_value=settings.event_positive_value,
                learning_rate=settings.learning_rate,
                epochs=settings.epochs,
                batch_size=settings.batch_size,
                random_seed=random_seed,
                prepared_data=shared_data,
                evaluation_split=shared_eval_split,
                monitor_indices=shared_monitor_indices,
                early_stopping_patience=settings.early_stopping_patience,
                early_stopping_min_delta=settings.early_stopping_min_delta,
                **extra_kwargs,
            )
            training_time_ms = round((time.monotonic() - started) * 1000, 1)
            risk = result.pop("holdout_risk", None)
            if risk is not None:
                holdout_risks[str(result.get("model") or model_name)] = (None, risk)
            comparison.append({
                "model": str(result.get("model") or model_name or "Unknown model"),
                "c_index": result.get("c_index"),
                "apparent_c_index": result.get("apparent_c_index"),
                "holdout_c_index": result.get("holdout_c_index"),
                "evaluation_mode": result.get("evaluation_mode"),
                "training_seed": result.get("training_seed"),
                "split_seed": result.get("split_seed"),
                "monitor_seed": result.get("monitor_seed"),
                "epochs_trained": result.get("epochs_trained"),
                "early_stopping_epochs": result.get("early_stopping_epochs"),
                "n_features": result.get("n_features"),
                "training_samples": result.get("training_samples"),
                "evaluation_samples": result.get("evaluation_samples"),
                # The ML comparison rows' names for the event counts (manuscript tables read them).
                "train_events": result.get("training_events"),
                "test_events": result.get("evaluation_events"),
                "training_time_ms": training_time_ms,
            })
        except Exception as exc:
            if _must_propagate_deep(exc):
                raise
            logger.exception("Deep-learning model %s failed in the holdout comparison.", model_name)
            errors.append({"model": model_name, "error": _failure_message(exc)})
    events = shared_data["event_tensor"].detach().cpu().numpy().reshape(-1)
    train_rows = np.asarray(shared_eval_split["train_idx"], dtype=int)
    eval_rows = np.asarray(shared_eval_split["eval_idx"], dtype=int)
    holdout_split = str(shared_eval_split.get("evaluation_mode")) == "holdout"
    result = _finalize_deep_comparison(
        comparison,
        errors,
        df=df,
        n_selected_features=len(settings.features),
        evaluation_mode="holdout",
        random_seed=random_seed,
        **_cohort_summary_fields(shared_data),
        cohort_counts={
            "n_patients": int(shared_data["n_samples"]),
            "n_events": int(float(shared_data["event_tensor"].sum().item())),
            "n_fit_patients": int(train_rows.size),
            "n_fit_events": int(events[train_rows].sum()),
            "n_evaluation_patients": int(eval_rows.size),
            "n_evaluation_events": int(events[eval_rows].sum()),
            "evaluation_split_fingerprint": shared_eval_split.get("evaluation_split_fingerprint"),
        },
        extra_cautions=_unseen_category_cautions(
            [(int(shared_eval_split.get("unseen_category_rows", 0) or 0) if holdout_split else 0, "evaluation")]
        ),
    )
    # The features one-hot coded after the whole-cohort typing decision (declared plus auto-coded).
    result["categorical_features"] = list(shared_data.get("categorical_features") or [])
    result["test_predictions"] = _deep_prediction_block(shared_data, shared_eval_split, holdout_risks)
    return result


def _unseen_category_cautions(counts: Sequence[tuple[int, str]]) -> list[str]:
    """The ML module's unseen-category caution for each (row count, scope) pair with rows."""
    from survival_toolkit.ml_models import _unseen_category_caution

    cautions = [_unseen_category_caution(int(n_rows), scope) for n_rows, scope in counts]
    return [caution for caution in cautions if caution]


def _deep_prediction_block(
    data: dict[str, Any],
    split: dict[str, Any],
    risks: dict[str, tuple[None, list[float]]],
) -> dict[str, Any] | None:
    """The models' risk scores on the evaluation rows, keyed by stored row label."""
    row_ids = split.get("eval_row_ids")
    if not risks or row_ids is None or str(split.get("evaluation_mode")) != "holdout":
        return None
    eval_idx = np.asarray(split["eval_idx"], dtype=int)
    return prediction_block(
        row_ids,
        data["time_tensor"].detach().cpu().numpy().reshape(-1)[eval_idx],
        data["event_tensor"].detach().cpu().numpy().reshape(-1)[eval_idx].astype(int),
        risks,
    )


def _deep_repeated_cv_comparison(
    df: pd.DataFrame,
    trainer_specs: list[tuple[str, Any, dict[str, Any]]],
    settings: _DeepRunSettings,
    *,
    random_seed: int,
    cv_folds: int,
    cv_repeats: int,
    parallel_jobs: int,
    locked_test_fraction: float | None,
) -> dict[str, Any]:
    _require_sklearn()
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2 for deep-learning repeated CV.")
    if cv_repeats < 1:
        raise ValueError("cv_repeats must be at least 1 for deep-learning repeated CV.")
    if cv_folds * cv_repeats > 200:
        raise ValueError("cv_folds * cv_repeats must not exceed 200 total evaluations for deep-learning repeated CV.")

    event_column = settings.event_column
    clean_frame = _coerce_deep_frame(
        df,
        time_column=settings.time_column,
        event_column=event_column,
        features=settings.features,
        categorical_features=settings.categorical_features,
        event_positive_value=settings.event_positive_value,
    )
    cohort_fields = _cohort_summary_fields(clean_frame.attrs)
    source_rows = clean_frame.attrs.get("source_row_index")
    all_events = clean_frame[event_column].astype(int).to_numpy()
    # Validated by the caller: None, or a fraction between 0.05 and 0.5.
    use_locked_test = locked_test_fraction is not None
    if use_locked_test:
        dev_positions, test_positions = locked_test_split(
            all_events,
            random_state=random_seed,
            test_fraction=float(locked_test_fraction),
        )
    else:
        dev_positions = np.arange(clean_frame.shape[0], dtype=int)
        test_positions = np.array([], dtype=int)
    dev_frame = clean_frame.iloc[dev_positions].reset_index(drop=True)
    events = dev_frame[event_column].astype(int).to_numpy()
    unique, counts = np.unique(events, return_counts=True)
    if len(unique) < 2 or counts.min() < cv_folds:
        raise ValueError(
            f"Repeated CV requires at least {cv_folds} analyzable samples in each event stratum"
            + (" of the development set." if use_locked_test else ".")
        )

    from survival_toolkit.ml_models import _unseen_category_rows

    model_specs = [
        {"model_name": model_name, "extra_kwargs": extra_kwargs}
        for model_name, _trainer, extra_kwargs in trainer_specs
    ]
    categorical_columns = _categorical_feature_columns(clean_frame, settings.features)
    design_splits: list[tuple[np.ndarray, np.ndarray]] = []
    unseen_fold_rows = 0
    # Collect only split indices (cheap numpy arrays - no tensors).
    fold_splits: list[dict[str, Any]] = []
    for repeat_idx in range(cv_repeats):
        # Derived seeds wrap at 2**32 so a seed near the maximum still works; ordinary seeds
        # give the same folds as the ML module's repeated CV.
        repeat_seed = _derived_seed(random_seed, repeat_idx)
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_idx, (train_rows, eval_rows) in enumerate(splitter.split(dev_frame, events), start=1):
            design_splits.append((dev_positions[train_rows], dev_positions[eval_rows]))
            unseen_fold_rows += _unseen_category_rows(
                dev_frame.iloc[train_rows], dev_frame.iloc[eval_rows], categorical_columns
            )
            fold_splits.append({
                "repeat": repeat_idx + 1,
                "fold": fold_idx,
                "seed_base": _derived_seed(random_seed, repeat_idx * cv_folds + fold_idx),
                # The StratifiedKFold random_state that produced this fold.
                "split_seed": repeat_seed,
                "monitor_seed": repeat_seed,
                "train_rows": train_rows,
                "eval_rows": eval_rows,
            })

    fold_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def _build_fold_task(split: dict[str, Any]) -> dict[str, Any] | None:
        """Build a single fold task (with tensors). Records prep errors; returns None on failure."""
        try:
            prepared_data, fold_split = _prepare_deep_split_data(
                dev_frame.iloc[split["train_rows"]].reset_index(drop=True),
                dev_frame.iloc[split["eval_rows"]].reset_index(drop=True),
                time_column=settings.time_column,
                event_column=event_column,
                features=settings.features,
                categorical_features=settings.categorical_features,
                event_positive_value=settings.event_positive_value,
            )
        except Exception as exc:
            _record_fold_failure(
                errors, [spec["model_name"] for spec in model_specs], split["repeat"], split["fold"], exc
            )
            return None
        return {
            "repeat": split["repeat"],
            "fold": split["fold"],
            "seed_base": split["seed_base"],
            "split_seed": split["split_seed"],
            **settings.task_fields(),
            "prepared_data": prepared_data,
            "evaluation_split": fold_split,
            "monitor_indices": _build_monitor_indices(
                fold_split["train_idx"],
                prepared_data["event_tensor"],
                split["monitor_seed"],
            ),
            "monitor_seed": split["monitor_seed"],
            "model_specs": model_specs,
            "require_holdout_evaluation": True,
        }

    parallel_execution_note = _run_deep_fold_tasks(
        fold_splits,
        _build_fold_task,
        fold_results=fold_results,
        errors=errors,
        parallel_jobs=parallel_jobs,
    )
    # Parallel folds finish in any order. Summing the fold C-indices in a fixed order (repeat,
    # fold, model) makes parallel and sequential runs give bit-identical aggregates.
    model_order = {str(spec["model_name"]): position for position, spec in enumerate(model_specs)}

    def _fold_order(item: dict[str, Any]) -> tuple[int, int, int]:
        return (int(item.get("repeat") or 0), int(item.get("fold") or 0), model_order.get(str(item.get("model")), len(model_order)))

    fold_results.sort(key=_fold_order)
    errors.sort(key=_fold_order)

    locked_results: dict[str, dict[str, Any]] = {}
    locked_predictions: dict[str, Any] | None = None
    locked_note: str | None = None
    locked_test_frame: pd.DataFrame | None = None
    if use_locked_test:
        design_splits.append((dev_positions, test_positions))
        locked_test_frame = clean_frame.iloc[test_positions].reset_index(drop=True)
        locked_note = (
            f"A stratified locked test set ({int(locked_test_frame.shape[0])} patients, "
            f"{int(locked_test_frame[event_column].sum())} events; {float(locked_test_fraction):.0%} of the cohort) "
            "was reserved before any fitting. Repeated CV, preprocessing, and early stopping used only the development set "
            f"({int(dev_frame.shape[0])} patients, {int(dev_frame[event_column].sum())} events); each model was then refit "
            "once on the full development set and scored once on the locked test set."
        )
        locked_labels = [source_rows[int(position)] if source_rows is not None else int(position) for position in test_positions]
        locked_results, locked_predictions = _deep_locked_test_results(
            dev_frame,
            locked_test_frame,
            model_specs,
            settings,
            random_seed=random_seed,
            row_labels=locked_labels,
        )

    comparison = _summarize_deep_cv_rows(
        trainer_specs,
        fold_results,
        errors,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        locked_results=locked_results if use_locked_test else None,
    )
    # A model whose refit on the development set fails has no locked-test estimate. That is an
    # error of the run, but its cross-validation result still ranks it.
    locked_errors = [
        {"model": model_name, "stage": "locked_test", "error": str(locked_results[model_name]["error"])}
        for model_name, _, _ in trainer_specs
        if (locked_results.get(model_name) or {}).get("error") is not None
    ]
    cohort_counts: dict[str, Any] = {
        "n_patients": int(clean_frame.shape[0]),
        "n_events": int(clean_frame[event_column].sum()),
        "evaluation_split_fingerprint": evaluation_split_fingerprint(
            source_rows,
            design_splits,
            kind="repeated_cv+locked_test" if use_locked_test else "repeated_cv",
        ),
    }
    if use_locked_test:
        cohort_counts.update({
            "locked_test_fraction": float(locked_test_fraction),
            "n_development_patients": int(dev_frame.shape[0]),
            "n_development_events": int(dev_frame[event_column].sum()),
            "n_locked_test_patients": int(test_positions.size),
            "n_locked_test_events": int(all_events[test_positions].sum()),
            "locked_test_note": locked_note,
        })
    result = _finalize_deep_comparison(
        comparison,
        errors,
        df=df,
        n_selected_features=len(settings.features),
        evaluation_mode="repeated_cv",
        random_seed=random_seed,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        fold_results=fold_results,
        **cohort_fields,
        cohort_counts=cohort_counts,
        locked_errors=locked_errors,
        parallel_execution_note=parallel_execution_note,
        extra_cautions=_unseen_category_cautions(
            [
                (unseen_fold_rows, "cross-validation evaluation (summed over folds)"),
                (
                    _unseen_category_rows(dev_frame, locked_test_frame, categorical_columns)
                    if locked_test_frame is not None
                    else 0,
                    "locked-test",
                ),
            ]
        ),
    )
    # The features one-hot coded after the whole-cohort typing decision (declared plus auto-coded).
    result["categorical_features"] = list(categorical_columns)
    result["locked_test_predictions"] = locked_predictions
    return result


def _collect_fold_task_result(
    task_meta: dict[str, Any],
    *,
    fold_results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    task_result: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    if exc is None and task_result is not None:
        fold_results.extend(task_result["fold_results"])
        errors.extend(task_result["errors"])
        return
    if exc is None:
        return
    _record_fold_failure(
        errors, [spec["model_name"] for spec in task_meta["model_specs"]], task_meta["repeat"], task_meta["fold"], exc
    )


def _run_fold_tasks_sequentially(
    fold_splits: list[dict[str, Any]],
    build_task: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    fold_results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    initial_tasks: list[dict[str, Any]] | None = None,
) -> None:
    """Run fold tasks one at a time, building each task's tensors only when it is due.

    ``initial_tasks`` holds tasks already built (the first fold, when a parallel run fell
    back to sequential folds). They are popped from the list, so once run their tensors are
    freed instead of staying referenced for the rest of the run.
    """

    def _run(task: dict[str, Any]) -> None:
        raise_if_cancelled()
        try:
            task_result = _run_deep_compare_fold_task(task)
        except Exception as exc:
            _collect_fold_task_result(task, fold_results=fold_results, errors=errors, exc=exc)
        else:
            _collect_fold_task_result(task, fold_results=fold_results, errors=errors, task_result=task_result)

    while initial_tasks:
        task = initial_tasks.pop(0)
        _run(task)
        del task
        gc.collect()
    for split in fold_splits:
        task = build_task(split)
        if task is None:
            continue
        _run(task)
        del task
        gc.collect()


def _unfinished_fold_splits(
    fold_splits: Sequence[dict[str, Any]],
    fold_results: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Folds with neither a result nor a recorded error, so a rerun counts no fold twice."""
    finished = {
        (int(item["repeat"]), int(item["fold"]))
        for item in [*fold_results, *errors]
        if item.get("repeat") is not None and item.get("fold") is not None
    }
    return [split for split in fold_splits if (int(split["repeat"]), int(split["fold"])) not in finished]


def _estimate_fold_task_training_bytes(task: dict[str, Any], fold_splits: Sequence[dict[str, Any]]) -> int:
    """Largest single-model training footprint of a fold task, sized for the largest training fold."""
    prepared = task.get("prepared_data")
    split = task.get("evaluation_split")
    if not isinstance(prepared, dict) or not isinstance(split, dict):
        return 0
    try:
        n_features = int(prepared["n_features"])
        total_samples = int(prepared["n_samples"])
        training_samples = int(np.asarray(split["train_idx"]).size)
    except (KeyError, TypeError, ValueError):
        return 0
    largest_training = max([training_samples, *(len(item["train_rows"]) for item in fold_splits if "train_rows" in item)])
    estimates = [0]
    for spec in task.get("model_specs") or []:
        try:
            estimates.append(
                _estimate_deep_training_bytes(
                    str(spec["model_name"]),
                    dict(spec.get("extra_kwargs") or {}),
                    training_samples=largest_training,
                    total_samples=total_samples + (largest_training - training_samples),
                    n_features=n_features,
                    batch_size=int(task.get("batch_size", 64)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # a model this estimate does not know; its trainer validates it
    return max(estimates)


def _abandon_process_pool(executor: Any) -> None:
    """Stop a worker pool now: cancel queued folds and terminate the running workers.

    Used when a run ends early (cancellation, a crashed worker, an error that must
    propagate) so abandoned folds stop using CPU and memory instead of running to the end.
    """
    processes = list((getattr(executor, "_processes", None) or {}).values())
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            continue  # already exited or closed
    deadline = time.monotonic() + 5.0
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            # The pool's manager thread may reap the process first; wait until the exit is recorded.
            while process.exitcode is None and time.monotonic() < deadline:
                time.sleep(0.05)
        except (AssertionError, OSError, ValueError):
            continue


def _run_deep_fold_tasks(
    fold_splits: list[dict[str, Any]],
    build_task: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    fold_results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    parallel_jobs: int,
) -> str | None:
    """Run every repeated-CV fold, in worker processes when it is safe to.

    Returns a note when parallel execution was requested but fell back to sequential. A
    worker that dies (for example killed for memory) does not end the run: the folds it
    left unfinished are rerun sequentially when the memory guard says one fold fits in this
    process, and are otherwise reported as failed folds. Cancellation is checked every
    ``_CANCELLATION_POLL_SECONDS`` and terminates the workers.
    """
    remaining = list(fold_splits)
    if parallel_jobs <= 1 or len(remaining) <= 1:
        _run_fold_tasks_sequentially(remaining, build_task, fold_results=fold_results, errors=errors)
        return None

    cpu_count = _available_cpu_count()
    max_workers = min(max(1, parallel_jobs), len(remaining), cpu_count)
    if max_workers <= 1:
        _run_fold_tasks_sequentially(remaining, build_task, fold_results=fold_results, errors=errors)
        return (
            f"Parallel repeated-CV execution was disabled because only {cpu_count} CPU is available "
            "to SurvStudio; folds ran sequentially."
        )
    first_task: dict[str, Any] | None = None
    while remaining and first_task is None:
        first_task = build_task(remaining.pop(0))
    if first_task is None:
        return None

    model_names = [str(spec["model_name"]) for spec in first_task.get("model_specs") or []]
    estimated_task_bytes = _estimate_deep_compare_task_bytes(first_task)
    training_bytes = _estimate_fold_task_training_bytes(first_task, fold_splits)

    def _sequential_from_first_task(note: str) -> str:
        nonlocal first_task
        # Handed over in a list and released here, so the first fold's tensors are freed
        # once it has run instead of staying referenced for the whole sequential run.
        initial_tasks = [first_task]
        first_task = None
        _run_fold_tasks_sequentially(
            remaining, build_task, fold_results=fold_results, errors=errors, initial_tasks=initial_tasks
        )
        return note

    estimated_inflight_bytes = estimated_task_bytes * max_workers
    if estimated_inflight_bytes >= _DEEP_COMPARE_PARALLEL_MAX_INFLIGHT_BYTES:
        return _sequential_from_first_task(
            "Parallel repeated-CV execution was disabled because fold payloads were too large "
            f"for safe multi-process buffering ({estimated_inflight_bytes / (1024 ** 2):.1f} MiB estimated in flight "
            f"across {max_workers} worker(s)); SurvStudio fell back to sequential folds."
        )
    available_memory = _available_system_memory_bytes()
    if available_memory is None:
        return _sequential_from_first_task(
            "Parallel repeated-CV execution was disabled because available system memory "
            "could not be determined for this runtime; SurvStudio fell back to sequential folds."
        )
    estimated_parallel_bytes = _estimate_parallel_deep_compare_memory_bytes(
        payload_bytes=estimated_task_bytes,
        max_workers=max_workers,
        training_bytes=training_bytes,
    )
    if estimated_parallel_bytes >= available_memory:
        return _sequential_from_first_task(
            "Parallel repeated-CV execution was disabled because available system memory "
            f"({available_memory / (1024 ** 2):.1f} MiB) was below the estimated worker footprint "
            f"({estimated_parallel_bytes / (1024 ** 2):.1f} MiB including process startup overhead and model training); "
            "SurvStudio fell back to sequential folds."
        )

    pool_broken = False
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            future_meta: dict[Any, dict[str, Any]] = {}

            def _submit(task: dict[str, Any]) -> None:
                future = executor.submit(_run_deep_compare_fold_task, task)
                future_meta[future] = {
                    "repeat": task["repeat"],
                    "fold": task["fold"],
                    "model_specs": task["model_specs"],
                }

            def _refill() -> None:
                while remaining and len(future_meta) < max_workers:
                    task = build_task(remaining.pop(0))
                    if task is None:
                        continue
                    _submit(task)
                    del task
                    gc.collect()

            try:
                _submit(first_task)
                first_task = None
                gc.collect()
                _refill()
                while future_meta:
                    raise_if_cancelled()  # the handler below terminates the workers
                    done, _pending = wait(
                        set(future_meta), timeout=_CANCELLATION_POLL_SECONDS, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        meta = future_meta.pop(future)
                        try:
                            task_result = future.result()
                        except BrokenExecutor:
                            # The worker died; this fold is rerun sequentially below.
                            pool_broken = True
                        except Exception as exc:
                            _collect_fold_task_result(meta, fold_results=fold_results, errors=errors, exc=exc)
                        else:
                            _collect_fold_task_result(meta, fold_results=fold_results, errors=errors, task_result=task_result)
                    if pool_broken:
                        break
                    _refill()
            except BrokenExecutor:
                pool_broken = True  # submit() refuses work once a worker has died
            except BaseException:
                _abandon_process_pool(executor)
                raise
            if pool_broken:
                _abandon_process_pool(executor)
    except (NotImplementedError, PermissionError, OSError) as exc:
        first_task = None  # an unsubmitted first fold is rebuilt below when it is due
        # Keep folds that already finished and rerun only the rest, so no fold is counted twice.
        _run_fold_tasks_sequentially(
            _unfinished_fold_splits(fold_splits, fold_results, errors),
            build_task,
            fold_results=fold_results,
            errors=errors,
        )
        return (
            "Parallel repeated-CV execution was unavailable in this runtime; "
            f"SurvStudio fell back to sequential folds ({type(exc).__name__})."
        )
    if pool_broken:
        first_task = None
        unfinished = _unfinished_fold_splits(fold_splits, fold_results, errors)
        # The worker was most likely killed for memory. Rerunning its folds in this (server)
        # process is only safe when the memory guard says one fold fits here.
        needed_bytes = estimated_task_bytes + training_bytes + _DEEP_COMPARE_PARALLEL_MEMORY_RESERVE_BYTES
        available_after = _available_system_memory_bytes()
        if unfinished and (available_after is None or needed_bytes >= available_after):
            memory_text = (
                "could not be determined"
                if available_after is None
                else f"({available_after / (1024 ** 2):.1f} MiB) was below the estimated footprint of one fold "
                f"({needed_bytes / (1024 ** 2):.1f} MiB)"
            )
            message = (
                "A parallel repeated-CV worker process stopped unexpectedly (for example, it ran out of memory), and this "
                f"fold was not rerun in the SurvStudio process because the available memory {memory_text}."
            )
            logger.error(
                "Deep-learning repeated CV: %d unfinished fold(s) not rerun after a worker process stopped: %s",
                len(unfinished),
                message,
            )
            for split in unfinished:
                errors.extend(
                    {"model": name, "repeat": split["repeat"], "fold": split["fold"], "error": message} for name in model_names
                )
            return (
                "A parallel repeated-CV worker process stopped unexpectedly (for example, it ran out of memory); its "
                f"{len(unfinished)} unfinished fold(s) were not rerun because the available memory {memory_text}, "
                "and they are reported as failed folds."
            )
        _run_fold_tasks_sequentially(unfinished, build_task, fold_results=fold_results, errors=errors)
        return (
            "A parallel repeated-CV worker process stopped unexpectedly (for example, it ran out of memory); "
            f"SurvStudio reran the {len(unfinished)} unfinished fold(s) sequentially."
        )
    return None


def _deep_locked_test_results(
    dev_frame: pd.DataFrame,
    locked_test_frame: pd.DataFrame,
    model_specs: list[dict[str, Any]],
    settings: _DeepRunSettings,
    *,
    random_seed: int,
    row_labels: Sequence[Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Refit every model on the development set and score it once on the locked test set.

    Also returns the models' locked-test risk scores keyed by stored row label
    (``row_labels`` gives the label of each ``locked_test_frame`` row).
    """
    locked_results: dict[str, dict[str, Any]] = {}
    try:
        locked_data, locked_split = _prepare_deep_split_data(
            dev_frame,
            locked_test_frame,
            time_column=settings.time_column,
            event_column=settings.event_column,
            features=settings.features,
            categorical_features=settings.categorical_features,
            event_positive_value=settings.event_positive_value,
        )
        locked_monitor = _build_monitor_indices(locked_split["train_idx"], locked_data["event_tensor"], random_seed)
    except Exception as exc:
        if _must_propagate_deep(exc):
            raise
        logger.exception("Preparing the deep-learning locked-test evaluation failed.")
        message = _failure_message(exc)
        return {str(spec["model_name"]): {"error": message} for spec in model_specs}, None
    for model_spec in model_specs:
        model_name = str(model_spec["model_name"])
        try:
            locked_results[model_name] = _run_deep_compare_task(
                {
                    "model_name": model_name,
                    "extra_kwargs": model_spec["extra_kwargs"],
                    "repeat": None,
                    "fold": None,
                    "seed": int(random_seed),
                    "split_seed": int(random_seed),
                    "monitor_seed": int(random_seed),
                    **settings.task_fields(),
                    "prepared_data": locked_data,
                    "evaluation_split": locked_split,
                    "monitor_indices": locked_monitor,
                    "require_holdout_evaluation": True,
                    "keep_holdout_risk": True,
                }
            )
        except Exception as exc:
            if _must_propagate_deep(exc):
                raise
            logger.exception("Deep-learning model %s failed when refit on the development set for the locked test.", model_name)
            locked_results[model_name] = {"error": _failure_message(exc)}
    risks = {
        name: (None, risk)
        for name, result in locked_results.items()
        if (risk := result.pop("holdout_risk", None)) is not None
    }
    if row_labels is not None:
        locked_split = {
            **locked_split,
            "eval_row_ids": [row_labels[position] for position in locked_split["eval_source_positions"]],
        }
    return locked_results, _deep_prediction_block(locked_data, locked_split, risks)


def _mean_epochs(rows: Sequence[dict[str, Any]], key: str) -> int | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return int(round(float(np.mean(values)))) if values else None


def _failure_summary(errors: Sequence[dict[str, Any]], *, unit: str) -> str:
    """Each model's failure messages, once per distinct message with how many ``unit`` failed with it."""
    counts: dict[tuple[str, str], int] = {}
    for item in errors:
        if isinstance(item, dict) and "error" in item:
            key = (str(item.get("model")), str(item["error"]))
            counts[key] = counts.get(key, 0) + 1
    return "; ".join(
        f"{model}{f' ({count} {unit})' if count > 1 else ''}: {message}" for (model, message), count in counts.items()
    )


def _summarize_deep_cv_rows(
    trainer_specs: list[tuple[str, Any, dict[str, Any]]],
    fold_results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    cv_folds: int,
    cv_repeats: int,
    locked_results: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """One comparison row per model from its repeated-CV folds (and locked test, if any).

    Fold rows are always clean holdout estimates: ``_run_deep_compare_task`` records a fold
    that fell back to apparent evaluation as a failed fold, so failures are the only reason a
    model misses folds.
    """
    from survival_toolkit.ml_models import _summarize_repeated_cv_rows, repeated_cv_row_fields

    comparison: list[dict[str, Any]] = []
    for model_name, _, _ in trainer_specs:
        model_rows = [
            row
            for row in fold_results
            if row["model"] == model_name
            and row.get("c_index") is not None
            and str(row.get("evaluation_mode", "holdout")) == "holdout"
        ]
        n_failures = sum(1 for err in errors if err["model"] == model_name)
        expected_evaluations = cv_folds * cv_repeats
        summary = (
            _summarize_repeated_cv_rows(
                model_rows,
                train_n_key="training_samples",
                test_n_key="evaluation_samples",
                train_events_key="training_events",
                test_events_key="evaluation_events",
            )
            if model_rows
            else None
        )
        incomplete = (len(model_rows) + n_failures) < expected_evaluations or n_failures > 0
        if summary is None and n_failures == 0:
            continue

        def _single_seed(key: str) -> int | None:
            values = {int(row[key]) for row in model_rows if row.get(key) is not None}
            return next(iter(values)) if len(values) == 1 else None

        row = {
            "model": model_name,
            **repeated_cv_row_fields(summary, incomplete=incomplete),
            "n_evaluations": len(model_rows),
            "n_failures": n_failures,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "training_seed": _single_seed("training_seed"),
            "split_seed": _single_seed("split_seed"),
            "monitor_seed": _single_seed("monitor_seed"),
            "training_seeds": sorted({int(item["training_seed"]) for item in model_rows if item.get("training_seed") is not None}),
            "split_seeds": sorted({int(item["split_seed"]) for item in model_rows if item.get("split_seed") is not None}),
            "monitor_seeds": sorted({int(item["monitor_seed"]) for item in model_rows if item.get("monitor_seed") is not None}),
            "epochs_trained": _mean_epochs(model_rows, "epochs_trained"),
            "early_stopping_epochs": _mean_epochs(model_rows, "early_stopping_epochs"),
        }
        if locked_results is not None:
            locked = locked_results.get(model_name) or {}
            row.update({
                "locked_test_c_index": None if locked.get("c_index") is None else float(locked["c_index"]),
                "locked_test_samples": locked.get("evaluation_samples"),
                "locked_test_events": locked.get("evaluation_events"),
                "locked_test_training_samples": locked.get("training_samples"),
                "locked_test_error": locked.get("error"),
            })
        comparison.append(row)
    return comparison


def _finalize_deep_comparison(
    comparison: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    df: pd.DataFrame,
    n_selected_features: int,
    evaluation_mode: str,
    random_seed: int,
    cv_folds: int | None = None,
    cv_repeats: int | None = None,
    fold_results: list[dict[str, Any]] | None = None,
    dropped_nonpositive_time_rows: int = 0,
    dropped_missing_outcome_rows: int = 0,
    time_column_note: str | None = None,
    cohort_counts: dict[str, Any] | None = None,
    locked_errors: Sequence[dict[str, Any]] = (),
    parallel_execution_note: str | None = None,
    extra_cautions: Sequence[str] = (),
) -> dict[str, Any]:
    """Rank the compared models and build the shared comparison payload and summary.

    ``errors`` are fits left out of the ranking (or of a model's CV aggregate);
    ``locked_errors`` are locked-test refits that failed for models that stay ranked. Both
    are reported in ``errors`` and make the ranking incomplete. A row without a C-index (a
    repeated-CV model with a failed fold, whose aggregate is withheld) is not ranked, and
    when no row can be ranked no model is named best. The parallel-execution note and
    ``extra_cautions`` (unseen categorical levels) are cautions like the others, so they
    count towards the status.
    """
    cohort_counts = dict(cohort_counts or {})
    locked_errors = list(locked_errors)
    repeated_cv = evaluation_mode == "repeated_cv"
    if not comparison or (repeated_cv and not fold_results):
        # Every fit failed (in repeated CV: every fold of every model).
        raise ValueError(
            "All deep-learning models failed to train. Errors: "
            + _failure_summary(errors, unit="folds" if repeated_cv else "fits")
        )

    for row in comparison:
        row["model"] = str(row.get("model") or "Unknown model")

    evaluation_modes = sorted({str(row.get("evaluation_mode", "unknown")) for row in comparison})
    # Holdout rows that fell back to apparent evaluation; repeated-CV rows are instead complete
    # or incomplete, which the ranking below handles.
    mixed_evaluation = not repeated_cv and len(evaluation_modes) > 1
    result_evaluation_mode = evaluation_mode
    ranked_rows: list[dict[str, Any]]
    unranked_rows: list[dict[str, Any]]
    if not repeated_cv:
        if mixed_evaluation:
            result_evaluation_mode = "mixed_holdout_apparent"
            ranked_rows = [row for row in comparison if str(row.get("evaluation_mode")) == "holdout"]
            unranked_rows = [row for row in comparison if str(row.get("evaluation_mode")) != "holdout"]
        elif all(str(row.get("evaluation_mode")) != "holdout" for row in comparison):
            result_evaluation_mode = "apparent"
            ranked_rows = list(comparison)
            unranked_rows = []
        else:
            ranked_rows = list(comparison)
            unranked_rows = []
    else:
        ranked_rows = list(comparison)
        unranked_rows = []
        if any(str(row.get("evaluation_mode")) != "repeated_cv" for row in comparison):
            result_evaluation_mode = "repeated_cv_incomplete"
    unranked_rows.extend(row for row in ranked_rows if row.get("c_index") is None)
    ranked_rows = [row for row in ranked_rows if row.get("c_index") is not None]

    ranked_rows.sort(key=lambda row: float(row["c_index"]), reverse=True)
    unranked_rows.sort(key=lambda row: row["model"])
    comparison = ranked_rows + unranked_rows
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
        row["comparable_for_ranking"] = True
    for row in unranked_rows:
        row["rank"] = None
        row["comparable_for_ranking"] = False
    best = ranked_rows[0] if ranked_rows else None
    locked_test = bool(cohort_counts.get("locked_test_note"))

    if repeated_cv:
        metric_mode = "repeated_cv"
    elif best is not None:
        metric_mode = "holdout" if best.get("evaluation_mode") == "holdout" else "apparent"
    else:
        metric_mode = "holdout" if result_evaluation_mode in {"holdout", "mixed_holdout_apparent"} else "apparent"
    metric_name = _metric_name_for_evaluation(metric_mode)

    strengths = [
        f"{len(comparison)} deep model(s) were trained on the same feature set ({n_selected_features} selected input columns).",
    ]
    shared_training_seeds = {
        int(row["training_seed"])
        for row in comparison
        if row.get("training_seed") is not None
    }
    if repeated_cv:
        strengths.append(
            f"Each model was evaluated across {cv_repeats} repeat(s) of {cv_folds}-fold stratified cross-validation."
        )
        if len(comparison) == 1:
            strengths.append(
                "The same repeated-CV settings can be rerun with Train Model when the evaluation strategy and seed are left unchanged."
            )
    elif len(shared_training_seeds) == 1:
        shared_seed = next(iter(shared_training_seeds))
        strengths.append(
            f"All holdout comparisons used the same split and seed ({shared_seed}), so the top model can be rerun directly under the same settings."
        )
    if mixed_evaluation:
        strengths.append(
            f"{len(ranked_rows)} model(s) retained a clean holdout estimate and remained rank-comparable."
        )
    if best is not None:
        strengths.append(f"Screening top deep model was {best['model']} with {metric_name} = {best['c_index']:.3f}.")
    strengths.append(
        "When early stopping held out a monitor subset, each model's reported weights were refit on the whole training partition for the selected number of epochs, so deep and classical models are fitted on the same rows."
    )

    cautions: list[str] = []
    if mixed_evaluation:
        cautions.append(
            "Rows with apparent fallback were excluded from the rank ordering because they are not directly comparable to holdout-evaluated rows."
        )
    if errors:
        cautions.append(
            f"{len(errors)} fold-level fit(s) failed; a model with a failed fold has no repeated-CV aggregate and is not ranked."
            if repeated_cv
            else f"{len(errors)} deep model fit(s) failed and were excluded from the ranking."
        )
    cautions.extend(_cohort_cautions(dropped_nonpositive_time_rows, dropped_missing_outcome_rows, time_column_note))
    if result_evaluation_mode == "repeated_cv_incomplete":
        cautions.append(
            "Repeated-CV incomplete means one or more folds were excluded because they failed or fell back to apparent evaluation."
        )
    if best is None:
        cautions.append(
            "No model completed every repeated-CV fold, so none was ranked and no repeated-CV C-index is reported; "
            "review the fold errors before rerunning."
            if repeated_cv
            else "No model reported a concordance estimate, so none was ranked."
        )
    elif best.get("evaluation_mode") != "holdout" and not repeated_cv:
        cautions.append(
            "The top-ranked model did not report a clean holdout C-index, so the ranking is optimistic."
        )

    if locked_test:
        strengths.append(str(cohort_counts["locked_test_note"]))
        best_locked = None if best is None else best.get("locked_test_c_index")
        if best_locked is not None:
            strengths.append(
                f"The CV-selected model ({best['model']}) reached a locked-test C-index of {float(best_locked):.3f}; "
                "this single untouched-test estimate is the performance to report."
            )
        cautions.insert(
            0,
            "Models were ranked by development-set repeated CV; report the locked-test C-index of the CV-selected model as the independent performance estimate, not the best locked-test value across models."
            if best is not None
            else "No model completed every development-set repeated-CV fold, so no model was CV-selected and none of the locked-test C-indices is a performance estimate to report.",
        )
    elif best is not None and len(comparison) > 1:
        # The ML comparisons' screening caution: the top model was chosen on the estimates it is reported with.
        cautions.insert(
            0,
            "The top-ranked model was selected and scored within the same repeated-CV screening run; treat this as model screening rather than final external validation. Reserve a locked test set or use an external cohort for the performance you report."
            if repeated_cv
            else "The top-ranked model was selected and scored on the same evaluation split; treat this as screening rather than final external validation.",
        )
    if locked_errors:
        failed_names = ", ".join(str(error["model"]) for error in locked_errors)
        selected_failed = best is not None and any(str(error["model"]) == best["model"] for error in locked_errors)
        locked_caution = (
            f"{len(locked_errors)} model(s) failed when refit on the development set and scored on the locked test set "
            f"({failed_names}); their locked-test C-index is blank"
            + (
                f", including the CV-selected model ({best['model']}), so this run has no untouched-test estimate to report."
                if selected_failed
                else "."
            )
        )
        if selected_failed:
            cautions.insert(0, locked_caution)
        else:
            cautions.append(locked_caution)
    from survival_toolkit.analysis import duplicate_identifier_caution

    duplicate_caution = duplicate_identifier_caution(df)
    if duplicate_caution:
        cautions.insert(0, duplicate_caution)
    if parallel_execution_note:
        cautions.append(str(parallel_execution_note))
    cautions.extend(str(caution) for caution in extra_cautions if caution)

    next_steps = [
        "Use the ranking to narrow candidates, then rerun the strongest architecture with external validation or repeated resampling.",
        "Prefer simpler models if the best deep model only matches the apparent-performance range of classical methods.",
    ]

    # Decided after every caution is in place (including the parallel-execution note and the
    # unseen-level cautions), so a caution can never sit next to a "robust" badge.
    best_c = None if best is None else float(best["c_index"])
    if best_c is None:
        status = "review"
    elif best_c < 0.55:
        status = "caution"
    elif best_c <= 0.65 or cautions or result_evaluation_mode not in {"holdout", "repeated_cv"}:
        status = "review"
    else:
        status = "robust"

    if best is None:
        headline = (
            "No deep model could be ranked: every model lost at least one repeated-CV fold, so no repeated-CV C-index is reported."
            if repeated_cv
            else "No deep model could be ranked because no concordance estimate was available."
        )
    else:
        headline = (
            f"Deep model screening placed {best['model']} first"
            + (" among holdout-evaluable models" if mixed_evaluation else "")
            + f" with {metric_name.lower()} of {best['c_index']:.3f}."
        )
    summary = {
        "status": status,
        "headline": headline,
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": "Models compared", "value": len(comparison)},
            {"label": "Best model", "value": None if best is None else best["model"]},
            {"label": metric_name, "value": None if best is None else best.get("c_index")},
            {"label": "Evaluation mode", "value": result_evaluation_mode},
            {"label": "Dropped for negative time", "value": int(dropped_nonpositive_time_rows) or None},
            {"label": "Dropped for missing outcome", "value": int(dropped_missing_outcome_rows) or None},
            {"label": "Failures", "value": len(errors) + len(locked_errors)},
        ],
    }
    all_errors = [*errors, *locked_errors]
    result = {
        "comparison_table": comparison,
        "errors": all_errors,
        "ranking_complete": not all_errors and not unranked_rows and all(row.get("c_index") is not None for row in ranked_rows),
        "evaluation_mode": result_evaluation_mode,
        "n_patients": cohort_counts.get("n_patients"),
        "n_events": cohort_counts.get("n_events"),
        "split_seed": int(random_seed),
        "evaluation_split_fingerprint": cohort_counts.get("evaluation_split_fingerprint"),
        # Intra-op torch threads of every fit (fixed, so results do not depend on the machine
        # or on parallel versus sequential folds).
        "torch_num_threads": _DEEP_TORCH_NUM_THREADS,
        "scientific_summary": summary,
        "insight_board": summary,
    }
    if parallel_execution_note:
        result["parallel_execution_note"] = parallel_execution_note
    for key in (
        "n_fit_patients",
        "n_fit_events",
        "n_evaluation_patients",
        "n_evaluation_events",
        "locked_test_fraction",
        "n_development_patients",
        "n_development_events",
        "n_locked_test_patients",
        "n_locked_test_events",
        "locked_test_note",
    ):
        if key in cohort_counts:
            result[key] = cohort_counts[key]
    if len(shared_training_seeds) == 1:
        result["shared_training_seed"] = next(iter(shared_training_seeds))
    shared_split_seeds = {
        int(row["split_seed"])
        for row in comparison
        if row.get("split_seed") is not None
    }
    if len(shared_split_seeds) == 1:
        result["shared_split_seed"] = next(iter(shared_split_seeds))
    shared_monitor_seeds = {
        int(row["monitor_seed"])
        for row in comparison
        if row.get("monitor_seed") is not None
    }
    if len(shared_monitor_seeds) == 1:
        result["shared_monitor_seed"] = next(iter(shared_monitor_seeds))
    if fold_results is not None:
        result["fold_results"] = fold_results
        result["cv_folds"] = cv_folds
        result["cv_repeats"] = cv_repeats
        result["repeat_results"] = [row.get("repeat_results") for row in comparison]
    from survival_toolkit.ml_models import build_manuscript_result_tables

    result["manuscript_tables"] = build_manuscript_result_tables(result)
    return result


@user_input_boundary
@_serialized_torch_training
def evaluate_single_deep_survival_model(
    model_type: str,
    *,
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    num_time_bins: int = 50,
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    latent_dim: int = 8,
    n_clusters: int = 3,
    evaluation_strategy: str = "holdout",
    cv_folds: int = 5,
    cv_repeats: int = 3,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    parallel_jobs: int = 1,
    locked_test_fraction: float | None = None,
) -> dict[str, Any]:
    """Evaluate one deep model with either holdout or repeated CV semantics.

    Repeated CV can reserve a locked test set (``locked_test_fraction``) exactly as the
    multi-model comparison does.
    """
    canonical_name = _canonical_deep_model_name(model_type)
    if evaluation_strategy not in {"holdout", "repeated_cv"}:
        raise ValueError(
            f"Unknown deep-learning evaluation strategy '{evaluation_strategy}'; use 'holdout' or 'repeated_cv'."
        )
    if evaluation_strategy == "repeated_cv":
        compare_result = compare_deep_survival_models(
            df=df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=categorical_features,
            event_positive_value=event_positive_value,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            num_time_bins=num_time_bins,
            n_heads=n_heads,
            d_model=d_model,
            n_layers=n_layers,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            evaluation_strategy=evaluation_strategy,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            parallel_jobs=parallel_jobs,
            included_models=[canonical_name],
            locked_test_fraction=locked_test_fraction,
        )
        row = compare_result["comparison_table"][0]
        aggregate_mode = str(compare_result.get("evaluation_mode", row.get("evaluation_mode", "repeated_cv")))
        mode_label = (
            "repeated-CV"
            if aggregate_mode == "repeated_cv"
            else (
                "repeated-CV incomplete"
                if aggregate_mode == "repeated_cv_incomplete"
                else aggregate_mode.replace("_", " ")
            )
        )
        n_failed_folds = int(row.get("n_failures") or 0)
        summary = {
            "status": compare_result["scientific_summary"]["status"],
            "headline": (
                f"{canonical_name} completed {cv_repeats}x{cv_folds} {mode_label} with mean C-index "
                f"of {row['c_index']:.3f}."
                if row.get("c_index") is not None
                else (
                    f"The {cv_repeats}x{cv_folds} repeated-CV C-index of {canonical_name} was withheld: "
                    f"{n_failed_folds} of {cv_repeats * cv_folds} fold(s) failed."
                    if n_failed_folds
                    else f"{canonical_name} completed {cv_repeats}x{cv_folds} {mode_label}, but the aggregate C-index could not be computed."
                )
            ),
            "strengths": list(compare_result["scientific_summary"].get("strengths", [])),
            "cautions": list(compare_result["scientific_summary"].get("cautions", [])),
            "next_steps": [
                "Use this repeated-CV estimate to judge this architecture on its own rather than as part of a cross-model screen.",
                "Run a separate single-fit analysis only if you need loss curves, feature-importance outputs, or deployment-ready artifacts.",
            ],
            "metrics": [
                {"label": "Model", "value": canonical_name},
                {"label": "Mean C-index", "value": row.get("c_index")},
                {"label": "Evaluation mode", "value": aggregate_mode},
                {"label": "CV folds", "value": cv_folds},
                {"label": "CV repeats", "value": cv_repeats},
            ],
        }
        if row.get("locked_test_c_index") is not None:
            summary["metrics"].append({"label": "Locked-test C-index", "value": row.get("locked_test_c_index")})
        # The comparison's cautions already carry the parallel-execution note and the
        # incomplete-CV explanation.
        summary["cautions"].append(
            "This result is an aggregate repeated-CV estimate. Feature-importance and loss-curve outputs require a separate single-fit run."
        )
        result = {
            "model": canonical_name,
            "model_type": model_type,
            "c_index": row.get("c_index"),
            "evaluation_mode": aggregate_mode,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "n_evaluations": row.get("n_evaluations"),
            "n_failures": row.get("n_failures"),
            # The failed folds' messages (and locked-test refit failures), as in the comparison.
            "errors": list(compare_result.get("errors") or []),
            "ranking_complete": compare_result.get("ranking_complete"),
            "n_features": row.get("n_features"),
            "epochs_trained": row.get("epochs_trained"),
            "early_stopping_epochs": row.get("early_stopping_epochs"),
            "training_time_ms": row.get("training_time_ms"),
            "training_seed": row.get("training_seed"),
            "split_seed": row.get("split_seed"),
            "monitor_seed": row.get("monitor_seed"),
            "training_seeds": row.get("training_seeds", []),
            "split_seeds": row.get("split_seeds", []),
            "monitor_seeds": row.get("monitor_seeds", []),
            "repeat_results": row.get("repeat_results", []),
            "parallel_execution_note": compare_result.get("parallel_execution_note"),
            "categorical_features": compare_result.get("categorical_features"),
            "torch_num_threads": compare_result.get("torch_num_threads"),
            "comparison_table": [dict(row)],
            "fold_results": [
                fold_row for fold_row in compare_result.get("fold_results", [])
                if fold_row.get("model") == canonical_name
            ],
            "manuscript_tables": compare_result.get("manuscript_tables"),
            "scientific_summary": summary,
            "insight_board": summary,
        }
        for key in (
            "locked_test_fraction",
            "n_development_patients",
            "n_development_events",
            "n_locked_test_patients",
            "n_locked_test_events",
            "locked_test_note",
        ):
            if key in compare_result:
                result[key] = compare_result[key]
        if "locked_test_c_index" in row:
            result["locked_test_c_index"] = row.get("locked_test_c_index")
            result["locked_test_samples"] = row.get("locked_test_samples")
            result["locked_test_events"] = row.get("locked_test_events")
            result["locked_test_error"] = row.get("locked_test_error")
        return result

    if locked_test_fraction is not None:
        raise ValueError("A locked test set is only available with repeated cross-validation.")
    shared_kwargs = {
        "time_column": time_column,
        "event_column": event_column,
        "features": features,
        "categorical_features": categorical_features,
        "event_positive_value": event_positive_value,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    trainer_map: dict[str, Callable[[], dict[str, Any]]] = {
        "deepsurv": lambda: train_deepsurv(df, hidden_layers=hidden_layers, **shared_kwargs),
        "deephit": lambda: train_deephit(df, hidden_layers=hidden_layers, num_time_bins=num_time_bins, **shared_kwargs),
        "mtlr": lambda: train_neural_mtlr(df, hidden_layers=hidden_layers, num_time_bins=num_time_bins, **shared_kwargs),
        "transformer": lambda: train_survival_transformer(
            df, d_model=d_model, n_heads=n_heads, n_layers=n_layers, **shared_kwargs
        ),
        "vae": lambda: train_survival_vae(
            df, latent_dim=latent_dim, hidden_layers=hidden_layers, n_clusters=n_clusters, **shared_kwargs
        ),
    }
    if model_type not in trainer_map:
        raise ValueError(f"Unknown model type: {model_type}")
    result = trainer_map[model_type]()
    result.pop("holdout_risk", None)
    return result


# ---------------------------------------------------------------------------
# 1. DeepSurv (Neural Cox PH)
# ---------------------------------------------------------------------------


def _cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood loss (Breslow ties).

    IMPORTANT: This must be computed on the full cohort so each event sees the
    correct risk set. Mini-batching breaks the Cox objective.
    """
    risk = risk_scores.squeeze(-1)
    t = times.reshape(-1)
    e = events.reshape(-1)

    order = torch.argsort(t, descending=True)
    t_sorted = t[order]
    r_sorted = risk[order]
    e_sorted = (e[order] == 1)

    if torch.sum(e_sorted) == 0:
        return r_sorted.sum() * 0.0  # preserve gradient path through model params

    log_cumsum_exp = torch.logcumsumexp(r_sorted, dim=0)
    _, counts = torch.unique_consecutive(t_sorted, return_counts=True)
    end_idx = torch.cumsum(counts, dim=0) - 1  # inclusive indices
    start_idx = end_idx - counts + 1

    event_values = r_sorted * e_sorted.to(dtype=r_sorted.dtype)
    cumulative_event_values = torch.cumsum(event_values, dim=0)
    cumulative_event_counts = torch.cumsum(e_sorted.to(dtype=r_sorted.dtype), dim=0)

    start_prev = start_idx - 1
    start_prev_valid = start_prev >= 0

    event_sum = cumulative_event_values[end_idx]
    event_sum = event_sum - torch.where(
        start_prev_valid,
        cumulative_event_values[start_prev.clamp(min=0)],
        torch.zeros_like(event_sum),
    )
    event_count = cumulative_event_counts[end_idx]
    event_count = event_count - torch.where(
        start_prev_valid,
        cumulative_event_counts[start_prev.clamp(min=0)],
        torch.zeros_like(event_count),
    )

    active_groups = event_count > 0
    total_events = torch.sum(event_count[active_groups])
    contributions = event_sum[active_groups] - event_count[active_groups] * log_cumsum_exp[end_idx[active_groups]]
    return -torch.sum(contributions) / torch.clamp(total_events, min=1.0)


class DeepSurvNet(_TorchModuleBase):
    """MLP that outputs a single risk score for Cox PH."""

    def __init__(self, in_features: int, hidden_layers: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for layer_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = layer_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@user_input_boundary
@_serialized_torch_training
def train_deepsurv(
    df: pd.DataFrame | None,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    prepared_data: dict[str, Any] | None = None,
    evaluation_split: dict[str, Any] | None = None,
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    refit_on_training_partition: bool = True,
) -> dict[str, Any]:
    """Train a DeepSurv (Neural Cox PH) model.

    Returns a JSON-serializable dict with c_index, loss_history,
    feature_importance, risk_scores, predicted_survival_function, and insight_board.

    Notes:
    - ``batch_size`` is accepted for API consistency, but Cox partial likelihood
      is optimized on the full training risk set each epoch.
    - With early stopping, the selected number of epochs is refit on the whole
      training partition (``refit_on_training_partition``).
    """
    _require_torch()
    epochs, batch_size, learning_rate = _validated_training_settings(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    hidden_layers = _validated_hidden_layers(hidden_layers)
    context = _prepare_deep_training_context(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
        prepared_data=prepared_data,
        evaluation_split=evaluation_split,
        monitor_indices=monitor_indices,
        early_stopping_patience=early_stopping_patience,
    )
    data = context.data
    x_all, t_all, e_all = context.x_all, context.t_all, context.e_all
    _guard_deep_parameter_budget("DeepSurv", n_features=int(data["n_features"]), hidden_layers=hidden_layers)

    def _fit_phase(rows: torch.Tensor, monitor_rows: torch.Tensor | None, max_epochs: int, patience: int | None) -> _FitPhase:
        _seed_torch(random_seed)
        model = DeepSurvNet(data["n_features"], hidden_layers, dropout)
        optimizer = _make_optimizer(model, learning_rate)
        x_fit, t_fit, e_fit = x_all[rows], t_all[rows], e_all[rows]
        step = _full_batch_step(
            model,
            optimizer,
            lambda: _cox_partial_likelihood_loss(model(x_fit), t_fit, e_fit),
            context_label="DeepSurv loss",
        )
        monitor = None if monitor_rows is None else (lambda: _monitor_c_index(model, x_all, t_all, e_all, monitor_rows))
        losses, monitor_values, monitor_used = _train_epochs(
            model,
            epochs=max_epochs,
            step=step,
            monitor=monitor,
            monitor_goal="max",
            patience=patience,
            min_delta=early_stopping_min_delta,
        )
        return _FitPhase(model, losses, monitor_values, monitor_used)

    final, first, refit_info = _fit_with_refit(
        _fit_phase,
        context,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="max",
        refit=refit_on_training_partition,
    )
    model = final.model

    # Evaluation
    model.eval()
    with torch.inference_mode():
        risk_scores_tensor = model(x_all)
    c_index, apparent_c_index, holdout_c_index, evaluation_mode, evaluation_note, holdout_risk = _holdout_and_apparent_c_index(
        context, risk_scores_tensor
    )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=context.eval_idx,
        evaluation_mode=evaluation_mode,
    )
    artifact_scope = _artifact_scope_label(evaluation_mode)

    # Feature importance (gradient-based)
    importance = _gradient_feature_importance(model, x_all[artifact_idx])
    feature_importance = [
        {"feature": name, "importance": imp}
        for name, imp in sorted(
            zip(data["feature_names"], importance), key=lambda p: p[1], reverse=True
        )
    ]

    # Risk scores
    risk_np = risk_scores_tensor.detach().cpu().numpy().ravel()
    artifact_idx_np = artifact_idx.detach().cpu().numpy()
    artifact_risk_np = risk_np[artifact_idx_np]
    risk_list = [float(v) for v in artifact_risk_np]

    # Predicted survival function for representative patients (low / median / high risk)
    sorted_risk_indices = np.argsort(artifact_risk_np)
    representative_indices = [
        int(artifact_idx_np[sorted_risk_indices[0]]),
        int(artifact_idx_np[sorted_risk_indices[len(sorted_risk_indices) // 2]]),
        int(artifact_idx_np[sorted_risk_indices[-1]]),
    ]

    train_idx_np = context.train_idx.detach().cpu().numpy()
    time_np = t_all.detach().cpu().numpy().ravel()
    event_np = e_all.detach().cpu().numpy().ravel()
    train_time_np = time_np[train_idx_np]
    train_event_np = event_np[train_idx_np]
    unique_times = np.sort(np.unique(train_time_np[train_event_np == 1]))
    if len(unique_times) == 0:
        unique_times = np.sort(np.unique(train_time_np))

    # Breslow baseline hazard estimation
    # Sort training samples once; use searchsorted to avoid O(N) boolean mask per time point.
    train_risk_np = risk_np[train_idx_np]
    sort_order = np.argsort(train_time_np, kind="stable")
    sorted_times_train = train_time_np[sort_order]
    sorted_risk_train = train_risk_np[sort_order]
    sorted_events_train = train_event_np[sort_order]
    # Precompute event counts at each unique time (O(N) via np.unique).
    event_times_train = sorted_times_train[sorted_events_train == 1]
    event_uniq, event_counts_arr = np.unique(event_times_train, return_counts=True)
    event_count_map: dict[float, float] = dict(zip(event_uniq.tolist(), event_counts_arr.tolist()))
    baseline_cumhaz = np.zeros(len(unique_times))
    for k, t_k in enumerate(unique_times):
        first_at_risk = int(np.searchsorted(sorted_times_train, t_k, side="left"))
        at_risk_risk = sorted_risk_train[first_at_risk:]
        if at_risk_risk.size == 0:
            baseline_cumhaz[k] = baseline_cumhaz[k - 1] if k > 0 else 0.0
            continue
        d_k = event_count_map.get(float(t_k), 0.0)
        log_risk_sum = _logsumexp_numpy(at_risk_risk)
        risk_sum = 0.0 if not np.isfinite(log_risk_sum) else float(np.exp(min(log_risk_sum, 700.0)))
        h0_k = d_k / max(risk_sum, 1e-12)
        baseline_cumhaz[k] = (baseline_cumhaz[k - 1] if k > 0 else 0.0) + h0_k

    predicted_survival_function: list[dict[str, Any]] = []
    for idx in representative_indices:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_cumhaz_i = np.where(
                baseline_cumhaz > 0.0,
                np.log(baseline_cumhaz) + float(risk_np[idx]),
                -np.inf,
            )
        surv_i = _survival_from_log_cumulative_hazard(log_cumhaz_i)
        timeline = [0.0] + [float(t) for t in unique_times]
        survival = [1.0] + [float(s) for s in surv_i]
        predicted_survival_function.append({
            "patient_index": idx,
            "risk_score": float(risk_np[idx]),
            "curve": _make_survival_curve(timeline, survival),
        })

    training_fields = _deep_training_fields(
        context,
        final,
        first,
        refit_info,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="max",
        random_seed=random_seed,
    )
    fit_rows, fit_events, refit_note = _deep_fit_summary_counts(context, refit_info)
    insight = _scientific_summary_dl(
        "DeepSurv",
        c_index,
        fit_rows,
        int(context.eval_idx.numel()),
        fit_events,
        data["n_features"],
        epochs,
        first.loss_history,
        evaluation_mode,
        evaluation_note,
        refit_note=refit_note,
        reported_epochs=int(training_fields["epochs_trained"]),
        unseen_category_rows=context.unseen_category_rows,
        **_cohort_summary_fields(data),
    )
    batching_meta = _batching_metadata(
        requested_batch_size=batch_size,
        effective_batch_size=fit_rows,
        optimization_mode="full_batch_cox",
        note=(
            "DeepSurv uses the full training partition each epoch because the Cox risk set must be evaluated "
            "in full; the requested batch size is recorded but not applied."
        ),
    )

    return {
        "model": "DeepSurv",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "holdout_risk": holdout_risk,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "tie_method": "breslow",
        **training_fields,
        "monitor_metric_label": "Monitor C-index",
        "monitor_metric_goal": "max",
        "feature_importance": feature_importance,
        "risk_scores": risk_list,
        "predicted_survival_function": predicted_survival_function,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "insight_board": insight,
        "scientific_summary": insight,
        **batching_meta,
    }


# ---------------------------------------------------------------------------
# 2. DeepHit
# ---------------------------------------------------------------------------


class DeepHitNet(_TorchModuleBase):
    """MLP that outputs discrete-time hazard probabilities."""

    def __init__(
        self, in_features: int, hidden_layers: list[int], num_time_bins: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for layer_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = layer_dim
        self.shared = nn.Sequential(*layers)
        # Add a tail bucket beyond the last observed horizon bin.
        self.output_layer = nn.Linear(prev_dim, num_time_bins + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.shared(x)
        logits = self.output_layer(hidden)
        return torch.softmax(logits, dim=1)


def _deephit_loss(
    pmf: torch.Tensor,
    time_bin_indices: torch.Tensor,
    events: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Combined DeepHit loss: log-likelihood + ranking loss."""
    n = pmf.shape[0]

    bin_idx = time_bin_indices.long()
    event_mask = (events == 1)
    censor_mask = ~event_mask

    log_pmf = torch.log(torch.clamp(pmf, min=1e-12))
    # Survival at each observed bin edge, computed in log-space so censored
    # terms do not lose precision when the tail probability is tiny.
    raw_log_survival = torch.flip(
        torch.logcumsumexp(torch.flip(log_pmf, dims=[1]), dim=1),
        dims=[1],
    )
    log_survival_at_edges = torch.cat(
        [
            torch.zeros((n, 1), device=pmf.device, dtype=pmf.dtype),
            raw_log_survival[:, 1:],
        ],
        dim=1,
    )
    # Cumulative incidence F(k) = sum_{j<=k} pmf[j], the quantity DeepHit
    # (Lee et al., 2018) ranks at each event time.
    cumulative_incidence = torch.cumsum(pmf, dim=1)
    # Likelihood:
    # - event in bin k: -log pmf[k]
    # - censored in bin k: -log S_start[k]
    terms: list[torch.Tensor] = []
    if torch.any(event_mask):
        selected_log_pmf = log_pmf[torch.arange(n, device=pmf.device), bin_idx]
        terms.append(-selected_log_pmf[event_mask])
    if torch.any(censor_mask):
        cens_log_surv = log_survival_at_edges[torch.arange(n, device=pmf.device), bin_idx]
        terms.append(-cens_log_surv[censor_mask])
    log_likelihood = torch.cat(terms).mean() if terms else torch.tensor(0.0, device=pmf.device)

    # Ranking loss component (pairwise)
    if event_mask.sum() > 0 and n > 1:
        event_indices = torch.where(event_mask)[0]
        ranking_terms: list[torch.Tensor] = []
        # Evaluate all event-driven pairings, but chunk the event dimension so
        # monitoring on larger holdout sets does not allocate one huge matrix.
        chunk_size = 128
        for start in range(0, int(event_indices.shape[0]), chunk_size):
            chunk = event_indices[start:start + chunk_size]
            event_times = bin_idx[chunk]
            # Comparable pairs: subjects observed beyond the event's bin, plus
            # subjects censored in the same bin (known to outlive the event).
            later_mask = (bin_idx.unsqueeze(1) > event_times.unsqueeze(0)) | (
                (bin_idx.unsqueeze(1) == event_times.unsqueeze(0)) & censor_mask.unsqueeze(1)
            )
            if not torch.any(later_mask):
                continue
            subject_cif = cumulative_incidence[:, event_times]
            event_cif = cumulative_incidence[chunk, event_times]
            # Penalize when a longer-surviving subject has at least as much
            # predicted incidence by the event time as the subject who failed.
            diff = subject_cif - event_cif.unsqueeze(0)
            scaled_diff = torch.clamp(diff / _DEEPHIT_RANKING_SIGMA, min=-20.0, max=20.0)
            ranking_terms.append(F.softplus(scaled_diff)[later_mask])
        ranking_loss = (
            torch.cat(ranking_terms).mean()
            if ranking_terms
            else torch.tensor(0.0, device=pmf.device)
        )
    else:
        ranking_loss = torch.tensor(0.0, device=pmf.device)

    return alpha * log_likelihood + (1.0 - alpha) * ranking_loss


@user_input_boundary
@_serialized_torch_training
def train_deephit(
    df: pd.DataFrame | None,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    hidden_layers: list[int] | None = None,
    num_time_bins: int = 50,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    alpha: float = 0.5,
    random_seed: int = 42,
    prepared_data: dict[str, Any] | None = None,
    evaluation_split: dict[str, Any] | None = None,
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    refit_on_training_partition: bool = True,
) -> dict[str, Any]:
    """Train a DeepHit model for discrete-time survival prediction.

    Returns a JSON-serializable dict with c_index, loss_history,
    predicted_survival_curves, and feature_importance.
    """
    _require_torch()
    epochs, batch_size, learning_rate = _validated_training_settings(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    hidden_layers = _validated_hidden_layers(hidden_layers)
    context = _prepare_deep_training_context(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
        prepared_data=prepared_data,
        evaluation_split=evaluation_split,
        monitor_indices=monitor_indices,
        early_stopping_patience=early_stopping_patience,
    )
    data = context.data
    x_all, t_all, e_all = context.x_all, context.t_all, context.e_all
    all_times_np = t_all.detach().cpu().numpy()
    _guard_deep_parameter_budget(
        "DeepHit", n_features=int(data["n_features"]), hidden_layers=hidden_layers, num_time_bins=num_time_bins
    )

    def _fit_phase(rows: torch.Tensor, monitor_rows: torch.Tensor | None, max_epochs: int, patience: int | None) -> _FitPhase:
        _seed_torch(random_seed)
        # Discretize time over the rows being fitted.
        bin_edges, bin_centers = _time_bin_grid(all_times_np[rows.detach().cpu().numpy()], num_time_bins)
        bin_tensor = torch.tensor(
            _digitize_time_bins(all_times_np[rows.detach().cpu().numpy()], bin_edges, num_time_bins, preserve_tail_overflow=True),
            dtype=torch.long,
        )
        monitor_bin_tensor: torch.Tensor | None = None
        if monitor_rows is not None:
            monitor_bin_tensor = torch.tensor(
                _digitize_time_bins(
                    all_times_np[monitor_rows.detach().cpu().numpy()],
                    bin_edges,
                    num_time_bins,
                    preserve_tail_overflow=True,
                ),
                dtype=torch.long,
            )
        model = DeepHitNet(data["n_features"], hidden_layers, num_time_bins, dropout)
        optimizer = _make_optimizer(model, learning_rate)
        loader = DataLoader(
            TensorDataset(x_all[rows], bin_tensor, e_all[rows]),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(random_seed),
        )
        step = _minibatch_step(
            model,
            optimizer,
            loader,
            lambda pmf, bins, events: _deephit_loss(pmf, bins, events, alpha),
            context_label="DeepHit loss",
        )

        def _monitor() -> float:
            with torch.inference_mode():
                monitor_pmf = model(x_all[monitor_rows])
                return float(_deephit_loss(monitor_pmf, monitor_bin_tensor, e_all[monitor_rows], alpha).item())

        losses, monitor_values, monitor_used = _train_epochs(
            model,
            epochs=max_epochs,
            step=step,
            monitor=None if monitor_rows is None else _monitor,
            monitor_goal="min",
            patience=patience,
            min_delta=early_stopping_min_delta,
        )
        return _FitPhase(model, losses, monitor_values, monitor_used, {"bin_edges": bin_edges, "bin_centers": bin_centers})

    final, first, refit_info = _fit_with_refit(
        _fit_phase,
        context,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="min",
        refit=refit_on_training_partition,
    )
    model = final.model
    bin_edges = final.aux["bin_edges"]
    bin_centers = final.aux["bin_centers"]
    bin_widths = torch.tensor(np.diff(bin_edges), dtype=torch.float32)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        pmf_all = model(x_all)
    survival_all, rmst_risk_all = _discrete_survival_from_pmf(pmf_all, bin_widths)
    risk_scores_tensor = rmst_risk_all
    c_index, apparent_c_index, holdout_c_index, evaluation_mode, evaluation_note, holdout_risk = _holdout_and_apparent_c_index(
        context, risk_scores_tensor
    )
    evaluation_note = _append_evaluation_note(
        evaluation_note,
        _discrete_time_tail_bucket_note(all_times_np, context.eval_idx.detach().cpu().numpy(), bin_edges, scope_label="evaluation"),
    )
    if first.monitor_used and context.monitor_idx is not None:
        evaluation_note = _append_evaluation_note(
            evaluation_note,
            _discrete_time_tail_bucket_note(
                all_times_np,
                context.monitor_idx.detach().cpu().numpy(),
                first.aux["bin_edges"],
                scope_label="monitor",
            ),
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=context.eval_idx,
        evaluation_mode=evaluation_mode,
    )
    artifact_scope = _artifact_scope_label(evaluation_mode)

    # Feature importance
    time_grid = torch.cat(
        [
            torch.as_tensor(bin_centers, dtype=torch.float32),
            torch.as_tensor([float(bin_edges[-1])], dtype=torch.float32),
        ]
    )
    importance = _gradient_feature_importance(
        model,
        x_all[artifact_idx],
        output_to_score=lambda pmf: _expected_time_risk(pmf, time_grid),
    )
    feature_importance = [
        {"feature": name, "importance": imp}
        for name, imp in sorted(
            zip(data["feature_names"], importance), key=lambda p: p[1], reverse=True
        )
    ]

    # Predicted survival curves for representative patients
    survival_np = survival_all.detach().cpu().numpy()
    risk_scores_np = risk_scores_tensor.detach().cpu().numpy().ravel()
    artifact_idx_np = artifact_idx.detach().cpu().numpy()
    artifact_risk_np = risk_scores_np[artifact_idx_np]
    sorted_idx = np.argsort(artifact_risk_np)
    representative = [
        int(artifact_idx_np[sorted_idx[0]]),
        int(artifact_idx_np[sorted_idx[len(sorted_idx) // 2]]),
        int(artifact_idx_np[sorted_idx[-1]]),
    ]

    timeline = [0.0] + [float(edge) for edge in bin_edges[1:]]
    predicted_survival_curves: list[dict[str, Any]] = []
    for idx in representative:
        surv_values = [float(v) for v in survival_np[idx]]
        predicted_survival_curves.append({
            "patient_index": idx,
            "curve": _make_survival_curve(timeline, surv_values),
        })

    training_fields = _deep_training_fields(
        context,
        final,
        first,
        refit_info,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="min",
        random_seed=random_seed,
    )
    fit_rows, fit_events, refit_note = _deep_fit_summary_counts(context, refit_info)
    insight = _scientific_summary_dl(
        "DeepHit",
        c_index,
        fit_rows,
        int(context.eval_idx.numel()),
        fit_events,
        data["n_features"],
        epochs,
        first.loss_history,
        evaluation_mode,
        evaluation_note,
        refit_note=refit_note,
        reported_epochs=int(training_fields["epochs_trained"]),
        unseen_category_rows=context.unseen_category_rows,
        **_cohort_summary_fields(data),
    )

    return {
        "model": "DeepHit",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "holdout_risk": holdout_risk,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        **training_fields,
        "monitor_metric_label": "Monitor loss",
        "monitor_metric_goal": "min",
        "predicted_survival_curves": predicted_survival_curves,
        "feature_importance": feature_importance,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "time_bins": [float(c) for c in bin_centers],
        "time_bin_edges": [float(edge) for edge in bin_edges],
        "insight_board": insight,
        "scientific_summary": insight,
    }


# ---------------------------------------------------------------------------
# 3. Neural MTLR (Multi-Task Logistic Regression)
# ---------------------------------------------------------------------------


class NeuralMTLRNet(_TorchModuleBase):
    """Neural network version of Multi-Task Logistic Regression."""

    def __init__(
        self, in_features: int, hidden_layers: list[int], num_time_bins: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for layer_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = layer_dim
        self.encoder = nn.Sequential(*layers)
        # Add a tail bucket beyond the observed horizon.
        self.output_layer = nn.Linear(prev_dim, num_time_bins + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        logits = self.output_layer(hidden)
        # Canonical MTLR normalizes right-cumulative interval scores rather
        # than the raw per-bin logits.
        cumsum_logits = torch.flip(
            torch.cumsum(torch.flip(logits, dims=[1]), dim=1),
            dims=[1],
        )
        return cumsum_logits


def _mtlr_loss(
    cumsum_logits: torch.Tensor,
    time_bin_indices: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Canonical right-cumulative MTLR negative log-likelihood.

    We normalize the right-cumulative interval scores with ``softmax`` to
    obtain a PMF over time bins, which is algebraically equivalent to the
    canonical MTLR parameterization.
    - Event in bin k: -log pmf[k]
    - Censored in bin k: -log S_start[k] where S_start[k] = sum_{j>=k} pmf[j]
    """
    n = cumsum_logits.shape[0]
    bin_idx = time_bin_indices.long()
    event_mask = (events == 1)
    censor_mask = ~event_mask

    log_pmf = torch.log_softmax(cumsum_logits, dim=1)

    # log S_start[k] = log sum_{j>=k} pmf[j], computed in log space: 1 - cdf in
    # float32 can go negative (NaN loss) or saturate when survival is tiny.
    log_survival_at_edges = torch.flip(
        torch.logcumsumexp(torch.flip(log_pmf, dims=[1]), dim=1),
        dims=[1],
    )
    terms: list[torch.Tensor] = []
    if torch.any(event_mask):
        terms.append(-log_pmf[torch.arange(n, device=cumsum_logits.device), bin_idx][event_mask])
    if torch.any(censor_mask):
        terms.append(-log_survival_at_edges[torch.arange(n, device=cumsum_logits.device), bin_idx][censor_mask])
    return torch.cat(terms).mean() if terms else torch.tensor(0.0, device=cumsum_logits.device)


@user_input_boundary
@_serialized_torch_training
def train_neural_mtlr(
    df: pd.DataFrame | None,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.1,
    num_time_bins: int = 50,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    prepared_data: dict[str, Any] | None = None,
    evaluation_split: dict[str, Any] | None = None,
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    refit_on_training_partition: bool = True,
) -> dict[str, Any]:
    """Train a Neural MTLR model.

    Returns a JSON-serializable dict with c_index, loss_history,
    predicted_survival_curves, calibration_data, and feature_importance.
    """
    _require_torch()
    epochs, batch_size, learning_rate = _validated_training_settings(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    # The same default architecture as the multi-model comparison.
    hidden_layers = _validated_hidden_layers(hidden_layers)
    context = _prepare_deep_training_context(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
        prepared_data=prepared_data,
        evaluation_split=evaluation_split,
        monitor_indices=monitor_indices,
        early_stopping_patience=early_stopping_patience,
    )
    data = context.data
    x_all, t_all, e_all = context.x_all, context.t_all, context.e_all
    all_times_np = t_all.detach().cpu().numpy()
    _guard_deep_parameter_budget(
        "Neural MTLR", n_features=int(data["n_features"]), hidden_layers=hidden_layers, num_time_bins=num_time_bins
    )

    def _fit_phase(rows: torch.Tensor, monitor_rows: torch.Tensor | None, max_epochs: int, patience: int | None) -> _FitPhase:
        _seed_torch(random_seed)
        # Discretize time over the rows being fitted.
        bin_edges, bin_centers = _time_bin_grid(all_times_np[rows.detach().cpu().numpy()], num_time_bins)
        bin_tensor = torch.tensor(
            _digitize_time_bins(all_times_np[rows.detach().cpu().numpy()], bin_edges, num_time_bins, preserve_tail_overflow=True),
            dtype=torch.long,
        )
        monitor_bin_tensor: torch.Tensor | None = None
        if monitor_rows is not None:
            monitor_bin_tensor = torch.tensor(
                _digitize_time_bins(
                    all_times_np[monitor_rows.detach().cpu().numpy()],
                    bin_edges,
                    num_time_bins,
                    preserve_tail_overflow=True,
                ),
                dtype=torch.long,
            )
        model = NeuralMTLRNet(data["n_features"], hidden_layers, num_time_bins, dropout=dropout)
        optimizer = _make_optimizer(model, learning_rate)
        loader = DataLoader(
            TensorDataset(x_all[rows], bin_tensor, e_all[rows]),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(random_seed),
        )
        step = _minibatch_step(model, optimizer, loader, _mtlr_loss, context_label="Neural MTLR loss")

        def _monitor() -> float:
            with torch.inference_mode():
                cumsum_logits_monitor = model(x_all[monitor_rows])
                return float(_mtlr_loss(cumsum_logits_monitor, monitor_bin_tensor, e_all[monitor_rows]).item())

        losses, monitor_values, monitor_used = _train_epochs(
            model,
            epochs=max_epochs,
            step=step,
            monitor=None if monitor_rows is None else _monitor,
            monitor_goal="min",
            patience=patience,
            min_delta=early_stopping_min_delta,
        )
        return _FitPhase(model, losses, monitor_values, monitor_used, {"bin_edges": bin_edges, "bin_centers": bin_centers})

    final, first, refit_info = _fit_with_refit(
        _fit_phase,
        context,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="min",
        refit=refit_on_training_partition,
    )
    model = final.model
    bin_edges = final.aux["bin_edges"]
    bin_centers = final.aux["bin_centers"]
    bin_widths = torch.tensor(np.diff(bin_edges), dtype=torch.float32)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        cumsum_logits_all = model(x_all)
    # Convert to survival probabilities
    log_pmf = torch.log_softmax(cumsum_logits_all, dim=1)
    pmf = torch.exp(log_pmf)
    survival_all, rmst_risk_all = _discrete_survival_from_pmf(pmf, bin_widths)
    c_index, apparent_c_index, holdout_c_index, evaluation_mode, evaluation_note, holdout_risk = _holdout_and_apparent_c_index(
        context, rmst_risk_all
    )
    evaluation_note = _append_evaluation_note(
        evaluation_note,
        _discrete_time_tail_bucket_note(all_times_np, context.eval_idx.detach().cpu().numpy(), bin_edges, scope_label="evaluation"),
    )
    if first.monitor_used and context.monitor_idx is not None:
        evaluation_note = _append_evaluation_note(
            evaluation_note,
            _discrete_time_tail_bucket_note(
                all_times_np,
                context.monitor_idx.detach().cpu().numpy(),
                first.aux["bin_edges"],
                scope_label="monitor",
            ),
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=context.eval_idx,
        evaluation_mode=evaluation_mode,
    )
    artifact_scope = _artifact_scope_label(evaluation_mode)

    # Feature importance (gradient-based salience on expected-time risk)
    time_grid = torch.cat(
        [
            torch.as_tensor(bin_centers, dtype=torch.float32),
            torch.as_tensor([float(bin_edges[-1])], dtype=torch.float32),
        ]
    )
    importance = _gradient_feature_importance(
        model,
        x_all[artifact_idx],
        output_to_score=lambda cumsum_logits: _expected_time_risk(
            torch.exp(torch.log_softmax(cumsum_logits, dim=1)),
            time_grid,
        ),
    )
    feature_importance = [
        {"feature": name, "importance": imp}
        for name, imp in sorted(
            zip(data["feature_names"], importance), key=lambda p: p[1], reverse=True
        )
    ]

    # Predicted survival curves
    survival_np = survival_all.detach().cpu().numpy()
    risk_np = rmst_risk_all.detach().cpu().numpy().ravel()
    artifact_idx_np = artifact_idx.detach().cpu().numpy()
    artifact_risk_np = risk_np[artifact_idx_np]
    sorted_idx = np.argsort(artifact_risk_np)
    representative = [
        int(artifact_idx_np[sorted_idx[0]]),
        int(artifact_idx_np[sorted_idx[len(sorted_idx) // 2]]),
        int(artifact_idx_np[sorted_idx[-1]]),
    ]

    timeline = [0.0] + [float(edge) for edge in bin_edges[1:]]
    predicted_survival_curves: list[dict[str, Any]] = []
    for idx in representative:
        surv_values = [float(v) for v in survival_np[idx]]
        predicted_survival_curves.append({
            "patient_index": idx,
            "curve": _make_survival_curve(timeline, surv_values),
        })

    # Calibration data: predicted vs observed event rates per decile
    n_deciles = min(10, int(artifact_idx.numel()) // 5)
    calibration_data: list[dict[str, float]] = []
    if n_deciles >= 2:
        # Use a reference time inside the observed horizon (median event time if possible).
        t_np = t_all[artifact_idx].detach().cpu().numpy().ravel()
        e_np = e_all[artifact_idx].detach().cpu().numpy().ravel()
        evt_times = t_np[e_np == 1]
        t_ref = float(np.median(evt_times)) if evt_times.size else float(np.median(t_np))
        t_ref = min(t_ref, float(bin_edges[-1]))
        ref_bin = int(_digitize_time_bins(np.asarray([t_ref]), bin_edges, num_time_bins)[0])

        # Predicted survival at t_ref: interpolate between the edges of the
        # reference bin instead of taking the value at the end of the bin.
        edge_low = float(timeline[ref_bin])
        edge_high = float(timeline[min(ref_bin + 1, len(timeline) - 1)])
        weight = 0.0 if edge_high <= edge_low else min(max((t_ref - edge_low) / (edge_high - edge_low), 0.0), 1.0)
        survival_at_ref = (
            (1.0 - weight) * survival_np[artifact_idx_np, ref_bin]
            + weight * survival_np[artifact_idx_np, min(ref_bin + 1, survival_np.shape[1] - 1)]
        )
        predicted_event_prob = 1.0 - survival_at_ref
        decile_indices = np.argsort(predicted_event_prob)
        chunk_size = len(decile_indices) // n_deciles
        for d in range(n_deciles):
            start = d * chunk_size
            end = start + chunk_size if d < n_deciles - 1 else len(decile_indices)
            idx_slice = decile_indices[start:end]
            pred_mean = float(np.mean(predicted_event_prob[idx_slice]))
            # Observed risk by t_ref from a Kaplan-Meier estimate within the
            # decile, so patients censored before t_ref still contribute
            # (dropping them biases the observed rate upward).
            obs_mean = _km_event_probability(t_np[idx_slice], e_np[idx_slice], t_ref)
            calibration_data.append({
                "decile": d + 1,
                "predicted_event_rate": pred_mean,
                "observed_event_rate": obs_mean,
            })

    training_fields = _deep_training_fields(
        context,
        final,
        first,
        refit_info,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="min",
        random_seed=random_seed,
    )
    fit_rows, fit_events, refit_note = _deep_fit_summary_counts(context, refit_info)
    insight = _scientific_summary_dl(
        "Neural MTLR",
        c_index,
        fit_rows,
        int(context.eval_idx.numel()),
        fit_events,
        data["n_features"],
        epochs,
        first.loss_history,
        evaluation_mode,
        evaluation_note,
        refit_note=refit_note,
        reported_epochs=int(training_fields["epochs_trained"]),
        unseen_category_rows=context.unseen_category_rows,
        **_cohort_summary_fields(data),
    )

    return {
        "model": "Neural MTLR",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "holdout_risk": holdout_risk,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        **training_fields,
        "monitor_metric_label": "Monitor loss",
        "monitor_metric_goal": "min",
        "predicted_survival_curves": predicted_survival_curves,
        "feature_importance": feature_importance,
        "calibration_data": calibration_data,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "time_bins": [float(c) for c in bin_centers],
        "time_bin_edges": [float(edge) for edge in bin_edges],
        "insight_board": insight,
        "scientific_summary": insight,
    }


# ---------------------------------------------------------------------------
# 4. Survival Transformer
# ---------------------------------------------------------------------------


class _FeatureIdentityEncoding(_TorchModuleBase):
    """Learned feature-identity embedding for tabular feature tokens."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_ids = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(feature_ids).unsqueeze(0)


class SurvivalTransformerNet(_TorchModuleBase):
    """Transformer encoder for survival risk prediction.

    Each feature is treated as a token: Linear embed -> Feature identity embed ->
    TransformerEncoder -> Mean pool -> Linear(1).
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.feature_embed = nn.Linear(1, d_model)
        self.feature_identity = _FeatureIdentityEncoding(in_features, d_model)
        # Standard 4x feed-forward width; the PyTorch default (2048) is far
        # wider than a tabular token model needs and dominated memory use.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=_transformer_feedforward_dim(d_model),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features) -> treat each feature as a token
        tokens = x.unsqueeze(-1)  # (batch, seq_len, 1)
        embedded = self.feature_embed(tokens)  # (batch, seq_len, d_model)
        embedded = self.feature_identity(embedded)
        encoded = self.transformer(embedded)  # (batch, seq_len, d_model)
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        return self.output_layer(pooled)  # (batch, 1)

    def get_attention_weights(self, x: torch.Tensor) -> list[list[list[float]]]:
        """Extract attention weights while reusing the normal encoder forward path."""
        was_training = self.training
        self.eval()
        tokens = x.unsqueeze(-1)
        embedded = self.feature_embed(tokens)
        embedded = self.feature_identity(embedded)

        captured: dict[int, torch.Tensor] = {}
        pre_handles: list[Any] = []
        hook_handles: list[Any] = []
        fastpath_enabled = (
            bool(torch.backends.mha.get_fastpath_enabled())
            if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled")
            else None
        )
        for layer_index, layer in enumerate(self.transformer.layers):
            def _capturing_pre_hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                next_kwargs = dict(kwargs)
                next_kwargs["need_weights"] = True
                next_kwargs["average_attn_weights"] = False
                return args, next_kwargs

            def _capturing_hook(
                _module: Any,
                _args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                output: Any,
                *,
                _layer_index: int = layer_index,
            ) -> Any:
                attn_output, attn_weights = output
                if attn_weights is not None:
                    captured[_layer_index] = attn_weights.detach().cpu()
                return output

            pre_handles.append(layer.self_attn.register_forward_pre_hook(_capturing_pre_hook, with_kwargs=True))
            hook_handles.append(layer.self_attn.register_forward_hook(_capturing_hook, with_kwargs=True))

        try:
            if fastpath_enabled is not None:
                torch.backends.mha.set_fastpath_enabled(False)
            with torch.inference_mode():
                output = embedded
                for layer in self.transformer.layers:
                    output = layer(output)
                if self.transformer.norm is not None:
                    _ = self.transformer.norm(output)
        finally:
            if fastpath_enabled is not None:
                torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            for handle in pre_handles:
                handle.remove()
            for handle in hook_handles:
                handle.remove()
            if was_training:
                self.train()

        attention_maps: list[list[list[float]]] = []
        for layer_index in range(len(self.transformer.layers)):
            attn_weights = captured.get(layer_index)
            if attn_weights is None:
                continue
            attn_np = attn_weights.numpy()
            while attn_np.ndim > 2:
                attn_np = attn_np.mean(axis=0)
            attention_maps.append([[float(v) for v in row] for row in attn_np])
        return attention_maps


@user_input_boundary
@_serialized_torch_training
def train_survival_transformer(
    df: pd.DataFrame | None,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    prepared_data: dict[str, Any] | None = None,
    evaluation_split: dict[str, Any] | None = None,
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    refit_on_training_partition: bool = True,
) -> dict[str, Any]:
    """Train a Survival Transformer model.

    Returns a JSON-serializable dict with c_index, loss_history,
    attention_weights, feature_importance, and risk_scores.

    Notes:
    - ``batch_size`` is accepted for API consistency, but Cox partial likelihood
      is optimized on the full training risk set each epoch.
    """
    _require_torch()
    epochs, batch_size, learning_rate = _validated_training_settings(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    n_heads = _validated_positive_integer(n_heads, "The number of attention heads")
    if d_model % n_heads != 0:
        raise ValueError("Transformer width must be divisible by attention heads.")
    context = _prepare_deep_training_context(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
        prepared_data=prepared_data,
        evaluation_split=evaluation_split,
        monitor_indices=monitor_indices,
        early_stopping_patience=early_stopping_patience,
    )
    data = context.data
    x_all, t_all, e_all = context.x_all, context.t_all, context.e_all
    # The largest full batch is the whole training partition (used by the refit).
    _guard_transformer_attention_budget(
        training_samples=int(context.train_idx.numel()) if refit_on_training_partition else int(context.fit_idx.numel()),
        n_features=int(data["n_features"]),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        d_model=int(d_model),
    )
    _guard_deep_parameter_budget(
        "Survival Transformer", n_features=int(data["n_features"]), d_model=int(d_model), n_layers=int(n_layers)
    )

    def _fit_phase(rows: torch.Tensor, monitor_rows: torch.Tensor | None, max_epochs: int, patience: int | None) -> _FitPhase:
        _seed_torch(random_seed)
        model = SurvivalTransformerNet(
            data["n_features"], d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        optimizer = _make_optimizer(model, learning_rate)
        x_fit, t_fit, e_fit = x_all[rows], t_all[rows], e_all[rows]
        step = _full_batch_step(
            model,
            optimizer,
            lambda: _cox_partial_likelihood_loss(model(x_fit), t_fit, e_fit),
            context_label="Survival Transformer loss",
        )
        monitor = None if monitor_rows is None else (lambda: _monitor_c_index(model, x_all, t_all, e_all, monitor_rows))
        losses, monitor_values, monitor_used = _train_epochs(
            model,
            epochs=max_epochs,
            step=step,
            monitor=monitor,
            monitor_goal="max",
            patience=patience,
            min_delta=early_stopping_min_delta,
        )
        return _FitPhase(model, losses, monitor_values, monitor_used)

    final, first, refit_info = _fit_with_refit(
        _fit_phase,
        context,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="max",
        refit=refit_on_training_partition,
    )
    model = final.model

    # Evaluation
    model.eval()
    with torch.inference_mode():
        risk_scores_tensor = model(x_all)
    c_index, apparent_c_index, holdout_c_index, evaluation_mode, evaluation_note, holdout_risk = _holdout_and_apparent_c_index(
        context, risk_scores_tensor
    )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=context.eval_idx,
        evaluation_mode=evaluation_mode,
    )
    artifact_scope = _artifact_scope_label(evaluation_mode)

    # Feature importance (gradient-based)
    importance = _gradient_feature_importance(model, x_all[artifact_idx])
    feature_importance = [
        {"feature": name, "importance": imp}
        for name, imp in sorted(
            zip(data["feature_names"], importance), key=lambda p: p[1], reverse=True
        )
    ]

    # Risk scores
    risk_np = risk_scores_tensor.detach().cpu().numpy().ravel()
    artifact_idx_np = artifact_idx.detach().cpu().numpy()
    risk_list = [float(v) for v in risk_np[artifact_idx_np]]

    # Attention weights (from a sample of patients to keep response size reasonable)
    sample_size = min(32, int(artifact_idx.numel()))
    sample_indices = artifact_idx_np[
        np.linspace(0, int(artifact_idx.numel()) - 1, sample_size, dtype=int)
    ]
    x_sample = x_all[sample_indices]
    with torch.inference_mode():
        attention_weights = model.get_attention_weights(x_sample)

    # Per-feature attention score: average attention received by each feature across layers
    feature_attention: list[dict[str, Any]] = []
    if attention_weights:
        last_layer_attn = np.array(attention_weights[-1])  # (n_features, n_features)
        col_sums = last_layer_attn.sum(axis=0)
        col_sums_norm = col_sums / max(col_sums.sum(), 1e-12)
        for idx, name in enumerate(data["feature_names"]):
            feature_attention.append({
                "feature": name,
                "attention_score": float(col_sums_norm[idx]) if idx < len(col_sums_norm) else 0.0,
            })
        feature_attention.sort(key=lambda d: d["attention_score"], reverse=True)

    training_fields = _deep_training_fields(
        context,
        final,
        first,
        refit_info,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="max",
        random_seed=random_seed,
    )
    fit_rows, fit_events, refit_note = _deep_fit_summary_counts(context, refit_info)
    insight = _scientific_summary_dl(
        "Survival Transformer",
        c_index,
        fit_rows,
        int(context.eval_idx.numel()),
        fit_events,
        data["n_features"],
        epochs,
        first.loss_history,
        evaluation_mode,
        evaluation_note,
        refit_note=refit_note,
        reported_epochs=int(training_fields["epochs_trained"]),
        unseen_category_rows=context.unseen_category_rows,
        **_cohort_summary_fields(data),
    )
    batching_meta = _batching_metadata(
        requested_batch_size=batch_size,
        effective_batch_size=fit_rows,
        optimization_mode="full_batch_cox",
        note=(
            "Survival Transformer uses the full training partition each epoch because the Cox risk set must "
            "be evaluated in full; the requested batch size is recorded but not applied."
        ),
    )

    return {
        "model": "Survival Transformer",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "holdout_risk": holdout_risk,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "tie_method": "breslow",
        **training_fields,
        "monitor_metric_label": "Monitor C-index",
        "monitor_metric_goal": "max",
        "attention_weights": attention_weights,
        "feature_attention": feature_attention,
        "feature_importance": feature_importance,
        "risk_scores": risk_list,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "insight_board": insight,
        "scientific_summary": insight,
        **batching_meta,
    }


# ---------------------------------------------------------------------------
# 5. Survival VAE (VAE-inspired latent model)
# ---------------------------------------------------------------------------


class SurvivalVAENet(_TorchModuleBase):
    """VAE-inspired autoencoder with a survival risk head.

    Encoder: Input -> Hidden -> (mu, log_var)
    Decoder: Latent -> Hidden -> Reconstructed input
    Survival head: Latent mean -> Risk score
    """

    def __init__(
        self,
        in_features: int,
        hidden_layers: list[int] | None = None,
        hidden_dim: int | None = None,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_layers = list(
            hidden_layers if hidden_layers is not None else ([hidden_dim] if hidden_dim is not None else _DEFAULT_HIDDEN_LAYERS)
        )
        if not hidden_layers:
            raise ValueError("Survival VAE needs at least one hidden layer.")
        encoder_layers: list[nn.Module] = []
        prev_dim = in_features
        for layer_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = layer_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        prev_decoder_dim = latent_dim
        for layer_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(prev_decoder_dim, layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_decoder_dim = layer_dim
        decoder_layers.append(nn.Linear(prev_decoder_dim, in_features))
        self.decoder = nn.Sequential(*decoder_layers)

        risk_hidden = max(hidden_layers[-1] // 2, 1)
        self.survival_head = nn.Sequential(
            nn.Linear(latent_dim, risk_hidden),
            nn.ReLU(),
            nn.Linear(risk_hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_log_var(hidden)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        safe_log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        std = torch.exp(0.5 * safe_log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        # Use the deterministic posterior mean for the Cox head so ranking
        # comparisons are stable across stochastic VAE samples.
        risk = self.survival_head(mu)
        return x_recon, mu, log_var, risk

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


def _vae_combined_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    risk: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    recon_weight: float = 1.0,
    kl_weight: float = 0.1,
    cox_weight: float = 1.0,
    categorical_feature_indices: Sequence[int] | None = None,
    numeric_feature_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Combined VAE loss: reconstruction + KL divergence + Cox loss."""
    categorical_indices = [
        int(index)
        for index in (list(categorical_feature_indices) if categorical_feature_indices is not None else [])
    ]
    numeric_indices = [
        int(index)
        for index in (list(numeric_feature_indices) if numeric_feature_indices is not None else [])
    ]
    total_recon_loss = x_recon.new_tensor(0.0)
    total_recon_elements = 0

    if categorical_indices:
        categorical_index_tensor = torch.as_tensor(categorical_indices, dtype=torch.long, device=x.device)
        categorical_targets = x.index_select(1, categorical_index_tensor)
        categorical_logits = x_recon.index_select(1, categorical_index_tensor)
        total_recon_loss = total_recon_loss + F.binary_cross_entropy_with_logits(
            categorical_logits,
            categorical_targets,
            reduction="sum",
        )
        total_recon_elements += int(categorical_targets.numel())

    if numeric_indices:
        numeric_index_tensor = torch.as_tensor(numeric_indices, dtype=torch.long, device=x.device)
        numeric_targets = x.index_select(1, numeric_index_tensor)
        numeric_reconstruction = x_recon.index_select(1, numeric_index_tensor)
        total_recon_loss = total_recon_loss + F.mse_loss(
            numeric_reconstruction,
            numeric_targets,
            reduction="sum",
        )
        total_recon_elements += int(numeric_targets.numel())

    if total_recon_elements > 0:
        recon_loss = total_recon_loss / total_recon_elements
    else:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    log_var_clamped = torch.clamp(log_var, min=-10.0, max=10.0)
    kl_loss = -0.5 * torch.mean(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())

    # Cox partial likelihood loss
    cox_loss = _cox_partial_likelihood_loss(risk, times, events)

    return recon_weight * recon_loss + kl_weight * kl_loss + cox_weight * cox_loss


def _km_event_probability(times: np.ndarray, events: np.ndarray, horizon: float) -> float | None:
    """Kaplan-Meier estimate of P(T <= horizon) = 1 - S_KM(horizon)."""
    times_arr = np.asarray(times, dtype=float).ravel()
    events_arr = np.asarray(events, dtype=float).ravel() > 0
    if times_arr.size == 0:
        return None
    survival = 1.0
    for event_time in np.unique(times_arr[events_arr & (times_arr <= horizon)]):
        at_risk = float(np.sum(times_arr >= event_time))
        if at_risk <= 0.0:
            break
        deaths = float(np.sum(events_arr & (times_arr == event_time)))
        survival *= 1.0 - deaths / at_risk
    return float(1.0 - survival)


def _simple_pca_2d(data: np.ndarray) -> np.ndarray:
    """Reduce data to 2D using PCA via SVD (no sklearn dependency)."""
    centered = data - data.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or centered.shape[1] < 2:
        # Fallback: just take first two dimensions or pad
        result = np.zeros((centered.shape[0], 2))
        for d in range(min(2, centered.shape[1])):
            result[:, d] = centered[:, d]
        return result
    try:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        return u[:, :2] * s[:2]
    except np.linalg.LinAlgError:
        return centered[:, :2] if centered.shape[1] >= 2 else np.zeros((centered.shape[0], 2))


def _simple_kmeans(data: np.ndarray, n_clusters: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Simple K-means clustering (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n <= n_clusters:
        return np.arange(n, dtype=int) % n_clusters

    # K-means++ initialization
    centers = np.empty((n_clusters, data.shape[1]))
    idx = int(rng.integers(0, n))
    centers[0] = data[idx]
    for c in range(1, n_clusters):
        dists = np.min(
            np.sum((data[:, np.newaxis, :] - centers[:c][np.newaxis, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = float(dists.sum())
        if total <= 0.0 or not np.isfinite(total):
            # Fewer distinct points than clusters: every point already sits on a
            # center, so fall back to a uniform draw instead of an all-zero
            # probability vector (which made rng.choice raise).
            idx = int(rng.integers(0, n))
        else:
            idx = int(rng.choice(n, p=dists / total))
        centers[c] = data[idx]

    labels = np.zeros(n, dtype=int)
    for _it in range(max_iter):
        # Assign
        dists = np.sum((data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = data[mask].mean(axis=0)
    return labels


@user_input_boundary
@_serialized_torch_training
def train_survival_vae(
    df: pd.DataFrame | None,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    latent_dim: int = 8,
    hidden_layers: list[int] | None = None,
    hidden_dim: int | None = None,
    n_clusters: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    prepared_data: dict[str, Any] | None = None,
    evaluation_split: dict[str, Any] | None = None,
    monitor_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 1e-4,
    refit_on_training_partition: bool = True,
) -> dict[str, Any]:
    """Train a Survival VAE model.

    Returns a JSON-serializable dict with c_index, loss_history,
    latent_embeddings, cluster_labels, cluster_survival_curves, and risk_scores.

    Notes:
    - ``batch_size`` is accepted for API consistency, but the current VAE survival
      objective is optimized on the full training partition each epoch.
    """
    _require_torch()
    epochs, batch_size, learning_rate = _validated_training_settings(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    # ``hidden_dim`` is the older single-layer spelling; the default matches the comparison's.
    hidden_layers = _validated_hidden_layers(
        hidden_layers if hidden_layers is not None else ([hidden_dim] if hidden_dim is not None else _DEFAULT_HIDDEN_LAYERS)
    )
    if not hidden_layers:
        # An explicit empty list is refused rather than silently replaced by the default.
        raise ValueError("Survival VAE needs at least one hidden layer.")
    context = _prepare_deep_training_context(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
        prepared_data=prepared_data,
        evaluation_split=evaluation_split,
        monitor_indices=monitor_indices,
        early_stopping_patience=early_stopping_patience,
    )
    data = context.data
    x_all, t_all, e_all = context.x_all, context.t_all, context.e_all
    categorical_feature_indices = list(data.get("categorical_feature_indices", []))
    numeric_feature_indices = list(data.get("numeric_feature_indices", []))
    _guard_deep_parameter_budget(
        "Survival VAE", n_features=int(data["n_features"]), hidden_layers=hidden_layers, latent_dim=latent_dim
    )

    def _fit_phase(rows: torch.Tensor, monitor_rows: torch.Tensor | None, max_epochs: int, patience: int | None) -> _FitPhase:
        _seed_torch(random_seed)
        model = SurvivalVAENet(data["n_features"], hidden_layers=hidden_layers, latent_dim=latent_dim, dropout=dropout)
        optimizer = _make_optimizer(model, learning_rate)
        x_fit, t_fit, e_fit = x_all[rows], t_all[rows], e_all[rows]

        def _loss() -> torch.Tensor:
            x_recon, mu, log_var, risk = model(x_fit)
            return _vae_combined_loss(
                x_fit,
                x_recon,
                mu,
                log_var,
                risk,
                t_fit,
                e_fit,
                categorical_feature_indices=categorical_feature_indices,
                numeric_feature_indices=numeric_feature_indices,
            )

        def _monitor() -> float | None:
            with torch.inference_mode():
                _, _, _, risk_monitor = model(x_all[monitor_rows])
            return _compute_c_index_torch(risk_monitor, t_all[monitor_rows], e_all[monitor_rows])

        losses, monitor_values, monitor_used = _train_epochs(
            model,
            epochs=max_epochs,
            step=_full_batch_step(model, optimizer, _loss, context_label="Survival VAE loss"),
            monitor=None if monitor_rows is None else _monitor,
            monitor_goal="max",
            patience=patience,
            min_delta=early_stopping_min_delta,
        )
        return _FitPhase(model, losses, monitor_values, monitor_used)

    final, first, refit_info = _fit_with_refit(
        _fit_phase,
        context,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="max",
        refit=refit_on_training_partition,
    )
    model = final.model

    # Evaluation
    model.eval()
    with torch.inference_mode():
        x_recon_all, mu_all, log_var_all, risk_all = model(x_all)
        latent_all = mu_all  # get_latent returns mu; reuse from the forward pass above
    c_index, apparent_c_index, holdout_c_index, evaluation_mode, evaluation_note, holdout_risk = _holdout_and_apparent_c_index(
        context, risk_all
    )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=context.eval_idx,
        evaluation_mode=evaluation_mode,
    )
    artifact_scope = _artifact_scope_label(evaluation_mode)

    # Risk scores
    risk_np = risk_all.detach().cpu().numpy().ravel()
    artifact_idx_np = artifact_idx.detach().cpu().numpy()
    risk_list = [float(v) for v in risk_np[artifact_idx_np]]

    # Latent embeddings -> 2D for visualization
    latent_np = latent_all[artifact_idx].detach().cpu().numpy()
    latent_2d = _simple_pca_2d(latent_np)
    latent_embeddings = [
        {"x": float(latent_2d[i, 0]), "y": float(latent_2d[i, 1])}
        for i in range(latent_2d.shape[0])
    ]

    # Cluster latent space
    cluster_labels_np = _simple_kmeans(latent_np, n_clusters, seed=random_seed)
    cluster_labels = [int(c) for c in cluster_labels_np]

    # Add cluster info to embeddings
    for i, emb in enumerate(latent_embeddings):
        emb["cluster"] = cluster_labels[i]

    # Cluster survival curves (simple KM per cluster)
    time_np = t_all[artifact_idx].detach().cpu().numpy().ravel()
    event_np = e_all[artifact_idx].detach().cpu().numpy().ravel()
    cluster_survival_curves: list[dict[str, Any]] = []
    for c in range(n_clusters):
        mask = cluster_labels_np == c
        if mask.sum() < 2:
            continue
        c_times = np.asarray(time_np[mask], dtype=float)
        c_events = np.asarray(event_np[mask], dtype=float)

        # Simple KM estimator
        unique_times, inverse_index, exit_counts = np.unique(
            c_times,
            return_inverse=True,
            return_counts=True,
        )
        event_counts = np.bincount(
            inverse_index,
            weights=(c_events == 1).astype(float),
            minlength=unique_times.size,
        ).astype(float)
        survival = np.ones(len(unique_times))
        n_at_risk = float(mask.sum())
        for j, (d_j, exits_at_time) in enumerate(zip(event_counts, exit_counts, strict=True)):
            previous_survival = survival[j - 1] if j > 0 else 1.0
            if n_at_risk > 0:
                survival[j] = min(previous_survival, previous_survival * max(0.0, 1.0 - d_j / n_at_risk))
            else:
                survival[j] = previous_survival
            n_at_risk = max(0.0, n_at_risk - float(exits_at_time))

        timeline = [0.0] + [float(t) for t in unique_times]
        surv_values = [1.0] + [float(s) for s in survival]
        cluster_survival_curves.append({
            "cluster": int(c),
            "n_patients": int(mask.sum()),
            "curve": _make_survival_curve(timeline, surv_values),
        })

    # Feature importance (gradient-based salience of the survival head risk output)
    importance = _gradient_feature_importance(
        model,
        x_all[artifact_idx],
        output_to_score=lambda output: output[3].reshape(-1),
    )
    feature_importance = [
        {"feature": name, "importance": imp}
        for name, imp in sorted(
            zip(data["feature_names"], importance), key=lambda p: p[1], reverse=True
        )
    ]

    training_fields = _deep_training_fields(
        context,
        final,
        first,
        refit_info,
        epochs=epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor_goal="max",
        random_seed=random_seed,
    )
    fit_rows, fit_events, refit_note = _deep_fit_summary_counts(context, refit_info)
    insight = _scientific_summary_dl(
        "Survival VAE",
        c_index,
        fit_rows,
        int(context.eval_idx.numel()),
        fit_events,
        data["n_features"],
        epochs,
        first.loss_history,
        evaluation_mode,
        evaluation_note,
        refit_note=refit_note,
        reported_epochs=int(training_fields["epochs_trained"]),
        unseen_category_rows=context.unseen_category_rows,
        **_cohort_summary_fields(data),
    )
    batching_meta = _batching_metadata(
        requested_batch_size=batch_size,
        effective_batch_size=fit_rows,
        optimization_mode="full_batch_vae",
        note=(
            "Survival VAE currently uses the full training partition each epoch; the requested batch size is "
            "recorded for reproducibility but not applied."
        ),
    )

    return {
        "model": "Survival VAE",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "holdout_risk": holdout_risk,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "tie_method": "breslow",
        **training_fields,
        "monitor_metric_label": "Monitor C-index",
        "monitor_metric_goal": "max",
        "latent_embeddings": latent_embeddings,
        "cluster_labels": cluster_labels,
        "cluster_survival_curves": cluster_survival_curves,
        "feature_importance": feature_importance,
        "risk_scores": risk_list,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "insight_board": insight,
        "scientific_summary": insight,
        "n_clusters": n_clusters,
        "latent_dim": latent_dim,
        **batching_meta,
    }

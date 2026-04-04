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

import multiprocessing as mp
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd

from survival_toolkit.encoding import (
    fit_feature_encoder as _fit_shared_feature_encoder,
    transform_feature_encoder as _transform_shared_feature_encoder,
)
from survival_toolkit.errors import user_input_boundary
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
    "Install with: pip install 'survival-toolkit[dl]'"
)
_ADAM_WEIGHT_DECAY = 1e-4
_DEEPHIT_RANKING_SIGMA = 1.0


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
        "epochs_trained": result.get("epochs_trained"),
    }


def _run_deep_compare_fold_task(task: dict[str, Any]) -> dict[str, Any]:
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
            errors.append(
                {
                    "model": model_name,
                    "repeat": task["repeat"],
                    "fold": task["fold"],
                    "error": str(exc),
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


def _coerce_deep_frame(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    event_positive_value: Any = None,
    min_samples: int = 10,
    require_event: bool = True,
) -> pd.DataFrame:
    """Clean raw data for deep models before fitting an encoder."""
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

    categorical_features = list(categorical_features or [])
    frame = df[required_columns].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame[time_column] = pd.to_numeric(frame[time_column], errors="coerce")

    from survival_toolkit.analysis import coerce_event

    frame[event_column] = coerce_event(frame[event_column], event_positive_value=event_positive_value)
    for col in features:
        if col in categorical_features:
            frame[col] = frame[col].astype("string")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=[time_column, event_column]).copy()
    frame = frame.loc[frame[time_column] > 0].reset_index(drop=True)

    if frame.empty:
        raise ValueError("No analyzable rows remain after removing missing/invalid values.")
    if require_event and float(frame[event_column].sum()) <= 0:
        raise ValueError("No events were found after preprocessing the event column.")
    if frame.shape[0] < min_samples:
        raise ValueError(
            f"Only {frame.shape[0]} analyzable rows remain after validating outcome columns. "
            f"Need at least {min_samples} samples for deep learning models."
        )
    return frame


def _fit_deep_encoder(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the shared tabular encoder with numeric standardization enabled."""
    return _fit_shared_feature_encoder(
        frame,
        features,
        categorical_features,
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
    """Prepare fold-specific deep-learning data with preprocessing fitted on the training fold."""
    train_frame = _coerce_deep_frame(
        train_df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        min_samples=10,
        require_event=True,
    )
    eval_frame = _coerce_deep_frame(
        eval_df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        min_samples=1,
        require_event=False,
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
        "categorical_feature_indices": list(train_data.get("categorical_feature_indices", [])),
        "numeric_feature_indices": list(train_data.get("numeric_feature_indices", [])),
    }
    del train_data, eval_data  # free per-split tensors now that combined tensors are built
    evaluation_split = {
        "train_idx": np.arange(train_n, dtype=int),
        "eval_idx": np.arange(train_n, train_n + eval_n, dtype=int),
        "evaluation_mode": "holdout",
        "evaluation_note": (
            f"Reported C-index is computed on an external fold with {train_n} training samples "
            f"and {eval_n} evaluation samples."
        ),
    }
    return (
        {
            "X_tensor": combined_x,
            "time_tensor": combined_t,
            "event_tensor": combined_e,
            **feature_meta,
            "n_samples": train_n + eval_n,
            "n_features": int(combined_x.shape[1]),
        },
        evaluation_split,
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
        resolved_split = evaluation_split if evaluation_split is not None else _build_evaluation_split(
            prepared_data["event_tensor"].detach().cpu().numpy().astype(int).ravel(),
            random_seed,
        )
        return prepared_data, resolved_split
    if df is None:
        raise ValueError("Raw dataframe input is required when prepared_data is not provided.")

    clean_frame = _coerce_deep_frame(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
    )
    resolved_split = dict(evaluation_split) if evaluation_split is not None else _build_evaluation_split(
        clean_frame[event_column].astype(int).to_numpy(),
        random_seed,
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
            return split_data, split_eval
        except ValueError:
            resolved_split = {
                "train_idx": np.arange(clean_frame.shape[0], dtype=int),
                "eval_idx": np.arange(clean_frame.shape[0], dtype=int),
                "evaluation_mode": "apparent",
                "evaluation_note": (
                    "A deterministic holdout split was available, but the holdout subset did not "
                    "retain enough analyzable samples after preprocessing. Reported C-index falls "
                    "back to the analyzable cohort."
                ),
            }

    encoder = _fit_deep_encoder(clean_frame, features, categorical_features)
    full_data = _transform_deep_frame(
        clean_frame,
        time_column=time_column,
        event_column=event_column,
        encoder=encoder,
    )
    return full_data, resolved_split


def _compute_c_index_torch(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> float | None:
    """Compute Harrell's concordance index from torch tensors.

    Delegates to sksurv.metrics.concordance_index_censored when available
    (O(n log n) via sorted comparisons), falling back to a pure-NumPy O(n²)
    loop only when sksurv is not installed.
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
        except Exception:
            pass

    # Fallback: pure-NumPy O(n²) implementation
    if len(time_np) > 2000:
        warnings.warn(
            f"sksurv is not installed; falling back to an O(n\u00b2) concordance computation "
            f"for n={len(time_np)} samples. This may be slow. "
            "Install scikit-survival for O(n log n) performance: pip install scikit-survival",
            RuntimeWarning,
            stacklevel=3,
        )
    concordant = 0.0
    comparable = 0.0
    for idx in range(len(time_np)):
        if event_np[idx] != 1:
            continue
        later = time_np > time_np[idx]
        if not later.any():
            continue
        comparable += float(later.sum())
        concordant += float(np.sum(risk_np[idx] > risk_np[later]))
        concordant += 0.5 * float(np.sum(risk_np[idx] == risk_np[later]))
    if comparable == 0:
        return None
    return float(concordant / comparable)


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


def _gradient_feature_importance(
    model: Any,
    x_tensor: torch.Tensor,
    *,
    output_to_score: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[float]:
    """Compute gradient-based feature importance: mean |grad of output w.r.t. input|."""
    model.eval()
    x_input = x_tensor.clone().detach().requires_grad_(True)
    output = model(x_input)
    if output_to_score is not None:
        output = output_to_score(output)
    elif output.dim() > 1:
        output = output.sum(dim=1)
    output.sum().backward()
    grad = x_input.grad
    if grad is None:
        return [0.0] * x_tensor.shape[1]
    importance = grad.abs().mean(dim=0).detach().cpu().numpy()
    return [float(v) for v in importance]


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
) -> dict[str, Any]:
    """Build an insight board dict for deep learning models."""
    c_val = c_index if c_index is not None else 0.5
    metric_name = _metric_name_for_evaluation(evaluation_mode)

    epochs_trained = int(len(loss_history))

    if c_val > 0.65:
        status = "robust"
    elif c_val >= 0.55:
        status = "review"
    else:
        status = "caution"

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

    cautions: list[str] = []
    next_steps: list[str] = []

    if evaluation_note:
        cautions.append(evaluation_note)

    if train_samples < 100:
        cautions.append("Sample size is small for a deep learning model; results may be unreliable.")
        next_steps.append("Consider using classical survival models (Cox PH, KM) for small datasets.")
    if train_events is not None:
        events_per_feature = float(train_events) / max(float(n_features), 1.0)
        if events_per_feature < 10:
            cautions.append(
                f"Training events per feature is {events_per_feature:.1f}; deep models can overfit quickly when feature count is high relative to events."
            )
    if c_val < 0.55:
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
            "Deterministic holdout reports one split only, so no confidence interval or standard deviation is shown."
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
            "This path is a Neural MTLR-inspired discrete-time variant, not a literal reference implementation of the original MTLR parameterization."
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
            f"{model_name} estimated a {metric_name.lower()} of {c_val:.3f} on "
            f"{evaluation_mode.replace('_', ' ')} evaluation."
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": next_steps,
        "metrics": [
            {"label": metric_name, "value": c_val},
            {"label": "Evaluation mode", "value": evaluation_mode},
            {"label": "Training samples", "value": train_samples},
            {"label": "Training events", "value": train_events},
            {"label": "Evaluation samples", "value": eval_samples},
            {"label": "Features", "value": n_features},
            {"label": "Epochs", "value": epochs_trained or epochs},
            {"label": "Final loss", "value": float(loss_history[-1]) if loss_history else None},
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


def _training_run_metadata(
    loss_history: list[float],
    monitor_loss_history: list[float] | None,
    requested_epochs: int,
    *,
    monitor_goal: str = "min",
) -> dict[str, int | bool | None]:
    monitor_loss_history = list(monitor_loss_history or [])
    if monitor_goal not in {"min", "max"}:
        raise ValueError("Monitor goal must be 'min' or 'max'.")
    epochs_trained = int(len(loss_history))
    best_monitor_epoch = (
        (
            int(np.argmin(np.asarray(monitor_loss_history, dtype=float))) + 1
            if monitor_goal == "min"
            else int(np.argmax(np.asarray(monitor_loss_history, dtype=float))) + 1
        )
        if monitor_loss_history
        else None
    )
    stopped_early = bool(monitor_loss_history) and epochs_trained < int(requested_epochs)
    return {
        "epochs_trained": epochs_trained,
        "best_monitor_epoch": best_monitor_epoch,
        "stopped_early": stopped_early,
        "max_epochs_requested": int(requested_epochs),
    }


@user_input_boundary
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
) -> dict[str, Any]:
    """Train all deep survival models with shared inputs and compare them."""
    _require_torch()

    hidden_layers = hidden_layers if hidden_layers is not None else [64, 64]
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
    def _finalize_result(
        comparison: list[dict[str, Any]],
        errors: list[dict[str, Any]],
        *,
        evaluation_mode: str,
        fold_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not comparison:
            raise ValueError(
                "All deep-learning models failed to train. Errors: "
                + "; ".join(
                    f"{item['model']}: {item['error']}"
                    for item in errors
                    if isinstance(item, dict) and "error" in item
                )
            )

        for row in comparison:
            row["model"] = str(row.get("model") or "Unknown model")

        evaluation_modes = sorted({str(row.get("evaluation_mode", "unknown")) for row in comparison})
        mixed_evaluation = len(evaluation_modes) > 1
        result_evaluation_mode = evaluation_mode
        ranked_rows: list[dict[str, Any]]
        unranked_rows: list[dict[str, Any]]
        if evaluation_mode != "repeated_cv":
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

        ranked_rows.sort(key=lambda row: row["c_index"] if row["c_index"] is not None else -1.0, reverse=True)
        if unranked_rows:
            unranked_rows.sort(key=lambda row: row["model"])
        comparison = ranked_rows + unranked_rows
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank"] = rank
            row["comparable_for_ranking"] = True
        for row in unranked_rows:
            row["rank"] = None
            row["comparable_for_ranking"] = False
        best = ranked_rows[0] if ranked_rows else comparison[0]

        metric_name = _metric_name_for_evaluation(
            "holdout"
            if best.get("evaluation_mode") == "holdout"
            else ("repeated_cv" if evaluation_mode == "repeated_cv" else "apparent")
        )

        strengths = [
            f"{len(comparison)} deep model(s) were trained on the same feature set ({len(features)} selected input columns).",
        ]
        shared_training_seeds = {
            int(row["training_seed"])
            for row in comparison
            if row.get("training_seed") is not None
        }
        if evaluation_mode == "repeated_cv":
            strengths.append(
                f"Each model was evaluated across {cv_repeats} repeat(s) of {cv_folds}-fold stratified cross-validation."
            )
            if len(comparison) == 1:
                strengths.append(
                    "The same repeated-CV settings can be rerun with Train Model when the evaluation strategy and seed are left unchanged."
                )
            fallback_models = [
                row["model"] for row in comparison if int(row.get("n_apparent_fallbacks", 0) or 0) > 0
            ]
            if fallback_models:
                strengths.append(
                    f"{len(fallback_models)} model(s) retained at least one clean fold but had additional apparent-fallback folds excluded from the repeated-CV aggregate."
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
        if best.get("c_index") is not None:
            strengths.append(f"Screening top deep model was {best['model']} with {metric_name} = {best['c_index']:.3f}.")
        else:
            strengths.append(f"Best-ranked model was {best['model']}, but the concordance estimate was not available.")

        cautions: list[str] = []
        if mixed_evaluation:
            cautions.append(
                "Rows with apparent fallback were excluded from the rank ordering because they are not directly comparable to holdout-evaluated rows."
            )
        if errors:
            cautions.append(f"{len(errors)} deep model fit(s) failed and were excluded from the ranking.")
        if evaluation_mode == "repeated_cv" and any(int(row.get("n_apparent_fallbacks", 0) or 0) > 0 for row in comparison):
            cautions.append(
                "Some repeated-CV folds fell back to apparent evaluation inside model training and were excluded from the repeated-CV aggregate."
            )
        if result_evaluation_mode == "repeated_cv_incomplete":
            cautions.append(
                "Repeated-CV incomplete means one or more folds were excluded because they failed or fell back to apparent evaluation."
            )
        if best.get("evaluation_mode") != "holdout" and evaluation_mode != "repeated_cv":
            cautions.append(
                "The top-ranked model did not report a clean holdout C-index, so the ranking is optimistic."
            )

        next_steps = [
            "Use the ranking to narrow candidates, then rerun the strongest architecture with external validation or repeated resampling.",
            "Prefer simpler models if the best deep model only matches the apparent-performance range of classical methods.",
        ]

        status = "robust"
        if cautions:
            status = "review"
        if best.get("c_index") is not None and float(best["c_index"]) < 0.55:
            status = "caution"

        summary = {
            "status": status,
            "headline": (
                f"Deep model screening placed {best['model']} first"
                + (" among holdout-evaluable models" if mixed_evaluation else "")
                + f" with {metric_name.lower()} "
                f"of {best['c_index']:.3f}."
                if best.get("c_index") is not None
                else (
                    f"Deep model comparison ranked {best['model']} first"
                    + (" among holdout-evaluable models" if mixed_evaluation else "")
                    + ", but concordance could not be estimated."
                )
            ),
            "strengths": strengths,
            "cautions": cautions,
            "next_steps": next_steps,
            "metrics": [
                {"label": "Models compared", "value": len(comparison)},
                {"label": "Best model", "value": best["model"]},
                {"label": metric_name, "value": best.get("c_index")},
                {"label": "Evaluation mode", "value": result_evaluation_mode},
                {"label": "Failures", "value": len(errors)},
            ],
        }
        result = {
            "comparison_table": comparison,
            "errors": errors,
            "ranking_complete": not errors and not unranked_rows and all(row.get("c_index") is not None for row in ranked_rows),
            "evaluation_mode": result_evaluation_mode,
            "scientific_summary": summary,
            "insight_board": summary,
        }
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

    if evaluation_strategy == "repeated_cv":
        _require_sklearn()
        if cv_folds < 2:
            raise ValueError("cv_folds must be at least 2 for deep-learning repeated CV.")
        if cv_repeats < 1:
            raise ValueError("cv_repeats must be at least 1 for deep-learning repeated CV.")

        clean_frame = _coerce_deep_frame(
            df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=categorical_features,
            event_positive_value=event_positive_value,
        )
        events = clean_frame[event_column].astype(int).to_numpy()
        unique, counts = np.unique(events, return_counts=True)
        if len(unique) < 2 or counts.min() < cv_folds:
            raise ValueError(
                f"Repeated CV requires at least {cv_folds} analyzable samples in each event stratum."
            )

        fold_results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        model_specs = [
            {
                "model_name": model_name,
                "extra_kwargs": extra_kwargs,
            }
            for model_name, _trainer, extra_kwargs in trainer_specs
        ]
        # Collect only split indices (cheap numpy arrays — no tensors).
        fold_splits: list[dict[str, Any]] = []
        for repeat_idx in range(cv_repeats):
            splitter = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_seed + repeat_idx,
            )
            for fold_idx, (train_rows, eval_rows) in enumerate(splitter.split(clean_frame, events), start=1):
                fold_splits.append({
                    "repeat": repeat_idx + 1,
                    "fold": fold_idx,
                    "seed_base": random_seed + repeat_idx * cv_folds + fold_idx,
                    "split_seed": random_seed + repeat_idx * cv_folds + fold_idx,
                    "monitor_seed": random_seed + repeat_idx,
                    "train_rows": train_rows,
                    "eval_rows": eval_rows,
                })

        def _build_fold_task(split: dict[str, Any]) -> dict[str, Any] | None:
            """Build a single fold task (with tensors). Logs prep errors; returns None on failure."""
            train_frame = clean_frame.iloc[split["train_rows"]].reset_index(drop=True)
            eval_frame = clean_frame.iloc[split["eval_rows"]].reset_index(drop=True)
            try:
                prepared_data, fold_split = _prepare_deep_split_data(
                    train_frame,
                    eval_frame,
                    time_column=time_column,
                    event_column=event_column,
                    features=features,
                    categorical_features=categorical_features,
                    event_positive_value=event_positive_value,
                )
            except Exception as exc:
                for model_name, _, _ in trainer_specs:
                    errors.append({
                        "model": model_name,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "error": str(exc),
                    })
                return None
            return {
                "repeat": split["repeat"],
                "fold": split["fold"],
                "seed_base": split["seed_base"],
                "split_seed": split["split_seed"],
                "time_column": time_column,
                "event_column": event_column,
                "features": list(features),
                "categorical_features": list(categorical_features or []),
                "event_positive_value": event_positive_value,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
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

        parallel_execution_note: str | None = None

        def _run_folds_sequentially() -> None:
            for split in fold_splits:
                task = _build_fold_task(split)
                if task is None:
                    continue
                try:
                    task_result = _run_deep_compare_fold_task(task)
                    fold_results.extend(task_result["fold_results"])
                    errors.extend(task_result["errors"])
                except Exception as exc:
                    for model_spec in task["model_specs"]:
                        errors.append({
                            "model": model_spec["model_name"],
                            "repeat": task["repeat"],
                            "fold": task["fold"],
                            "error": str(exc),
                        })
                # task goes out of scope here: tensors freed immediately

        if parallel_jobs > 1 and len(fold_splits) > 1:
            try:
                with ProcessPoolExecutor(
                    max_workers=max(1, parallel_jobs),
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    future_meta: dict[Any, dict[str, int]] = {}
                    for split in fold_splits:
                        task = _build_fold_task(split)
                        if task is None:
                            continue
                        future = executor.submit(_run_deep_compare_fold_task, task)
                        # Store only lightweight fold metadata, not the task with tensors.
                        future_meta[future] = {"repeat": split["repeat"], "fold": split["fold"]}
                        # task goes out of scope: tensors freed after pickling for subprocess
                    for future in as_completed(future_meta):
                        meta = future_meta.pop(future)
                        try:
                            task_result = future.result()
                            fold_results.extend(task_result["fold_results"])
                            errors.extend(task_result["errors"])
                        except Exception as exc:
                            for model_spec in model_specs:
                                errors.append({
                                    "model": model_spec["model_name"],
                                    "repeat": meta["repeat"],
                                    "fold": meta["fold"],
                                    "error": str(exc),
                                })
            except (NotImplementedError, PermissionError, OSError) as exc:
                parallel_execution_note = (
                    "Parallel repeated-CV execution was unavailable in this runtime; "
                    f"SurvStudio fell back to sequential folds ({type(exc).__name__})."
                )
                _run_folds_sequentially()
        else:
            _run_folds_sequentially()

        comparison: list[dict[str, Any]] = []
        from survival_toolkit.ml_models import _summarize_repeated_cv_rows

        for model_name, _, _ in trainer_specs:
            model_rows = [row for row in fold_results if row["model"] == model_name and row["c_index"] is not None]
            n_failures = sum(1 for err in errors if err["model"] == model_name)
            expected_evaluations = cv_folds * cv_repeats
            holdout_rows = [row for row in model_rows if str(row.get("evaluation_mode")) == "holdout"]
            fallback_rows = [row for row in model_rows if str(row.get("evaluation_mode")) != "holdout"]
            summary = (
                _summarize_repeated_cv_rows(
                    holdout_rows,
                    train_n_key="training_samples",
                    test_n_key="evaluation_samples",
                    train_events_key=None,
                    test_events_key=None,
                )
                if holdout_rows
                else None
            )
            n_failures += len(fallback_rows)
            incomplete = (len(holdout_rows) + n_failures) < expected_evaluations or n_failures > 0
            if summary is None and n_failures == 0:
                continue
            row_evaluation_mode = "repeated_cv_incomplete" if incomplete else "repeated_cv"
            comparison.append({
                "model": model_name,
                "c_index": None if incomplete or summary is None else float(summary["c_index"]),
                "c_index_std": None if incomplete or summary is None else float(summary["c_index_std"]),
                "c_index_median": None if incomplete or summary is None else float(summary["c_index_median"]),
                "c_index_interval_lower": None if incomplete or summary is None else summary["c_index_interval_lower"],
                "c_index_interval_upper": None if incomplete or summary is None else summary["c_index_interval_upper"],
                "c_index_interval_label": None if summary is None else summary["c_index_interval_label"],
                "n_features": None if summary is None else int(summary["n_features"]),
                "training_time_ms": None if summary is None else float(summary["training_time_ms"]),
                "n_evaluations": len(holdout_rows),
                "n_repeats": 0 if summary is None else int(summary["n_repeats"]),
                "n_failures": n_failures,
                "n_apparent_fallbacks": len(fallback_rows),
                "cv_folds": cv_folds,
                "cv_repeats": cv_repeats,
                "evaluation_mode": row_evaluation_mode,
                "training_seed": None if not holdout_rows else (int(holdout_rows[0]["training_seed"]) if len({int(row["training_seed"]) for row in holdout_rows if row.get("training_seed") is not None}) == 1 else None),
                "split_seed": None if not holdout_rows else (int(holdout_rows[0]["split_seed"]) if len({int(row["split_seed"]) for row in holdout_rows if row.get("split_seed") is not None}) == 1 else None),
                "monitor_seed": None if not holdout_rows else (int(holdout_rows[0]["monitor_seed"]) if len({int(row["monitor_seed"]) for row in holdout_rows if row.get("monitor_seed") is not None}) == 1 else None),
                "training_seeds": sorted({int(row["training_seed"]) for row in holdout_rows if row.get("training_seed") is not None}),
                "split_seeds": sorted({int(row["split_seed"]) for row in holdout_rows if row.get("split_seed") is not None}),
                "monitor_seeds": sorted({int(row["monitor_seed"]) for row in holdout_rows if row.get("monitor_seed") is not None}),
                "epochs_trained": int(round(np.mean([row["epochs_trained"] for row in holdout_rows]))) if holdout_rows else None,
                "training_samples": None if summary is None else int(summary["train_n"]),
                "evaluation_samples": None if summary is None else int(summary["test_n"]),
                "train_events": None if summary is None else summary["train_events"],
                "test_events": None if summary is None else summary["test_events"],
                "repeat_results": [] if summary is None else summary["repeat_results"],
            })
        result = _finalize_result(comparison, errors, evaluation_mode="repeated_cv", fold_results=fold_results)
        if parallel_execution_note:
            result["parallel_execution_note"] = parallel_execution_note
            result["scientific_summary"]["cautions"].append(parallel_execution_note)
        return result

    shared_data, shared_eval_split = _prepare_deep_training_inputs(
        df,
        time_column=time_column,
        event_column=event_column,
        features=features,
        categorical_features=categorical_features,
        event_positive_value=event_positive_value,
        random_seed=random_seed,
    )
    shared_monitor_indices = _build_monitor_indices(
        shared_eval_split["train_idx"],
        shared_data["event_tensor"],
        random_seed,
    )
    comparison: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for model_name, trainer, extra_kwargs in trainer_specs:
        try:
            started = time.monotonic()
            result = trainer(
                df,
                time_column=time_column,
                event_column=event_column,
                features=features,
                categorical_features=categorical_features,
                event_positive_value=event_positive_value,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                random_seed=random_seed,
                prepared_data=shared_data,
                evaluation_split=shared_eval_split,
                monitor_indices=shared_monitor_indices,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                **extra_kwargs,
            )
            training_time_ms = round((time.monotonic() - started) * 1000, 1)
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
                "n_features": result.get("n_features"),
                "training_samples": result.get("training_samples"),
                "evaluation_samples": result.get("evaluation_samples"),
                "training_time_ms": training_time_ms,
            })
        except Exception as exc:
            errors.append({"model": model_name, "error": str(exc)})
    return _finalize_result(comparison, errors, evaluation_mode="holdout")


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
) -> dict[str, Any]:
    """Evaluate one deep model with either holdout or repeated CV semantics."""
    canonical_name = _canonical_deep_model_name(model_type)
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
        summary = {
            "status": compare_result["scientific_summary"]["status"],
            "headline": (
                f"{canonical_name} completed {cv_repeats}x{cv_folds} {mode_label} with mean C-index "
                f"of {row['c_index']:.3f}."
                if row.get("c_index") is not None
                else f"{canonical_name} completed {cv_repeats}x{cv_folds} {mode_label}, but the aggregate C-index could not be computed."
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
        summary["cautions"].append(
            "This result is an aggregate repeated-CV estimate. Feature-importance and loss-curve outputs require a separate single-fit run."
        )
        if compare_result.get("parallel_execution_note"):
            summary["cautions"].append(str(compare_result["parallel_execution_note"]))
        if aggregate_mode == "repeated_cv_incomplete":
            summary["cautions"].append(
                "Repeated-CV incomplete means one or more folds were excluded because they failed or fell back to apparent evaluation."
            )
        return {
            "model": canonical_name,
            "model_type": model_type,
            "c_index": row.get("c_index"),
            "evaluation_mode": aggregate_mode,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "n_evaluations": row.get("n_evaluations"),
            "n_failures": row.get("n_failures"),
            "n_features": row.get("n_features"),
            "epochs_trained": row.get("epochs_trained"),
            "training_time_ms": row.get("training_time_ms"),
            "training_seed": row.get("training_seed"),
            "split_seed": row.get("split_seed"),
            "monitor_seed": row.get("monitor_seed"),
            "training_seeds": row.get("training_seeds", []),
            "split_seeds": row.get("split_seeds", []),
            "monitor_seeds": row.get("monitor_seeds", []),
            "repeat_results": row.get("repeat_results", []),
            "parallel_execution_note": compare_result.get("parallel_execution_note"),
            "comparison_table": [dict(row)],
            "fold_results": [
                fold_row for fold_row in compare_result.get("fold_results", [])
                if fold_row.get("model") == canonical_name
            ],
            "manuscript_tables": compare_result.get("manuscript_tables"),
            "scientific_summary": summary,
            "insight_board": summary,
        }

    trainer_map = {
        "deepsurv": lambda: train_deepsurv(
            df,
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
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        ),
        "deephit": lambda: train_deephit(
            df,
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
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        ),
        "mtlr": lambda: train_neural_mtlr(
            df,
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
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        ),
        "transformer": lambda: train_survival_transformer(
            df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=categorical_features,
            event_positive_value=event_positive_value,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        ),
        "vae": lambda: train_survival_vae(
            df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            categorical_features=categorical_features,
            event_positive_value=event_positive_value,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            n_clusters=n_clusters,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        ),
    }
    if model_type not in trainer_map:
        raise ValueError(f"Unknown model type: {model_type}")
    return trainer_map[model_type]()


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
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@user_input_boundary
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
) -> dict[str, Any]:
    """Train a DeepSurv (Neural Cox PH) model.

    Returns a JSON-serializable dict with c_index, loss_history,
    feature_importance, risk_scores, predicted_survival_function, and insight_board.

    Notes:
    - ``batch_size`` is accepted for API consistency, but Cox partial likelihood
      is optimized on the full training risk set each epoch.
    """
    _require_torch()
    _seed_torch(random_seed)

    hidden_layers = hidden_layers if hidden_layers is not None else [64, 64]
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
    x_all, t_all, e_all = data["X_tensor"], data["time_tensor"], data["event_tensor"]
    train_idx = torch.as_tensor(eval_split["train_idx"], dtype=torch.long)
    eval_idx = torch.as_tensor(eval_split["eval_idx"], dtype=torch.long)
    monitor_idx = _resolve_monitor_indices(
        monitor_indices,
        train_idx=train_idx,
        events=e_all,
        random_seed=random_seed,
    )
    evaluation_mode = str(eval_split["evaluation_mode"])
    evaluation_note = str(eval_split["evaluation_note"])
    x_train, t_train, e_train = x_all[train_idx], t_all[train_idx], e_all[train_idx]

    model = DeepSurvNet(data["n_features"], hidden_layers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_ADAM_WEIGHT_DECAY)

    loss_history: list[float] = []
    monitor_loss_history: list[float] = []
    best_monitor: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    early_wait = 0
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        risk = model(x_train)
        loss = _cox_partial_likelihood_loss(risk, t_train, e_train)
        _require_finite_loss(loss, context="DeepSurv loss")
        loss.backward()
        _clip_gradients(model)
        optimizer.step()
        loss_history.append(float(loss.item()))
        if monitor_idx is not None:
            model.eval()
            monitor_c_index = _monitor_c_index(model, x_all, t_all, e_all, monitor_idx)
            if monitor_c_index is None:
                monitor_idx = None  # monitor subset has no events; disable for this run
            else:
                monitor_loss_history.append(float(monitor_c_index))
                best_monitor, early_wait, best_state, should_stop = _update_early_stopping(
                    float(monitor_c_index),
                    best_value=best_monitor,
                    wait_count=early_wait,
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    goal="max",
                    model=model,
                    best_state=best_state,
                )
                if should_stop:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        risk_scores_tensor = model(x_all)
    apparent_c_index = _compute_c_index_torch(risk_scores_tensor, t_all, e_all)
    holdout_c_index = _compute_c_index_torch(risk_scores_tensor[eval_idx], t_all[eval_idx], e_all[eval_idx])
    c_index = holdout_c_index if holdout_c_index is not None else apparent_c_index
    if holdout_c_index is None and evaluation_mode == "holdout":
        evaluation_mode = "holdout_fallback_apparent"
        evaluation_note = (
            "A deterministic holdout split was available, but the holdout subset did not "
            "support a comparable concordance estimate; the reported C-index is apparent."
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=eval_idx,
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

    train_idx_np = train_idx.detach().cpu().numpy()
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

    training_meta = _training_run_metadata(loss_history, monitor_loss_history, epochs, monitor_goal="max")
    insight = _scientific_summary_dl(
        "DeepSurv",
        c_index,
        int(train_idx.numel()),
        int(eval_idx.numel()),
        int(e_all[train_idx].sum().item()),
        data["n_features"],
        epochs,
        loss_history,
        evaluation_mode,
        evaluation_note,
    )
    batching_meta = _batching_metadata(
        requested_batch_size=batch_size,
        effective_batch_size=int(train_idx.numel()),
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
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "tie_method": "breslow",
        "training_seed": random_seed,
        "split_seed": random_seed,
        "monitor_seed": random_seed,
        "loss_history": loss_history,
        "monitor_history": monitor_loss_history,
        "monitor_metric_label": "Monitor C-index",
        "monitor_metric_goal": "max",
        "best_monitor_epoch": training_meta["best_monitor_epoch"],
        "stopped_early": training_meta["stopped_early"],
        "max_epochs_requested": training_meta["max_epochs_requested"],
        "feature_importance": feature_importance,
        "risk_scores": risk_list,
        "predicted_survival_function": predicted_survival_function,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "insight_board": insight,
        "scientific_summary": insight,
        "epochs_trained": training_meta["epochs_trained"],
        "n_samples": data["n_samples"],
        "training_samples": int(train_idx.numel()),
        "evaluation_samples": int(eval_idx.numel()),
        "n_features": data["n_features"],
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
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
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
    num_bins = pmf.shape[1] - 1

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
    survival_at_edges = torch.exp(log_survival_at_edges)
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
            later_mask = bin_idx.unsqueeze(1) > event_times.unsqueeze(0)
            if not torch.any(later_mask):
                continue
            subject_survival = survival_at_edges[:, event_times]
            event_survival = survival_at_edges[chunk, event_times]
            diff = event_survival.unsqueeze(0) - subject_survival
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
) -> dict[str, Any]:
    """Train a DeepHit model for discrete-time survival prediction.

    Returns a JSON-serializable dict with c_index, loss_history,
    predicted_survival_curves, and feature_importance.
    """
    _require_torch()
    _seed_torch(random_seed)

    hidden_layers = hidden_layers if hidden_layers is not None else [64, 64]
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
    x_all, t_all, e_all = data["X_tensor"], data["time_tensor"], data["event_tensor"]
    train_idx = torch.as_tensor(eval_split["train_idx"], dtype=torch.long)
    eval_idx = torch.as_tensor(eval_split["eval_idx"], dtype=torch.long)
    monitor_idx = _resolve_monitor_indices(
        monitor_indices,
        train_idx=train_idx,
        events=e_all,
        random_seed=random_seed,
    )
    evaluation_mode = str(eval_split["evaluation_mode"])
    evaluation_note = str(eval_split["evaluation_note"])
    x_train, t_train, e_train = x_all[train_idx], t_all[train_idx], e_all[train_idx]

    # Discretize time into bins
    time_np = t_train.numpy()
    t_min, t_max = float(time_np.min()), float(time_np.max())
    if t_max <= t_min:
        t_max = t_min + 1e-6
    bin_edges = np.linspace(t_min, t_max, num_time_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = _digitize_time_bins(
        time_np,
        bin_edges,
        num_time_bins,
        preserve_tail_overflow=True,
    )
    bin_tensor = torch.tensor(bin_indices, dtype=torch.long)
    bin_widths = torch.tensor(np.diff(bin_edges), dtype=torch.float32)
    monitor_bin_tensor: torch.Tensor | None = None
    if monitor_idx is not None:
        monitor_bin_indices = _digitize_time_bins(
            t_all[monitor_idx].detach().cpu().numpy(),
            bin_edges,
            num_time_bins,
            preserve_tail_overflow=True,
        )
        monitor_bin_tensor = torch.tensor(monitor_bin_indices, dtype=torch.long)

    model = DeepHitNet(data["n_features"], hidden_layers, num_time_bins, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_ADAM_WEIGHT_DECAY)
    dataset = TensorDataset(x_train, bin_tensor, e_train)
    loader_generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)

    loss_history: list[float] = []
    monitor_loss_history: list[float] = []
    best_monitor: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    early_wait = 0
    for _epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for x_b, bin_b, e_b in loader:
            optimizer.zero_grad()
            pmf = model(x_b)
            loss = _deephit_loss(pmf, bin_b, e_b, alpha)
            _require_finite_loss(loss, context="DeepHit loss")
            loss.backward()
            _clip_gradients(model)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(float(np.mean(epoch_losses)))
        if monitor_idx is not None and monitor_bin_tensor is not None:
            model.eval()
            with torch.inference_mode():
                monitor_pmf = model(x_all[monitor_idx])
                monitor_loss = float(_deephit_loss(monitor_pmf, monitor_bin_tensor, e_all[monitor_idx], alpha).item())
            monitor_loss_history.append(monitor_loss)
            best_monitor, early_wait, best_state, should_stop = _update_early_stopping(
                monitor_loss,
                best_value=best_monitor,
                wait_count=early_wait,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                model=model,
                best_state=best_state,
            )
            if should_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        pmf_all = model(x_all)
    survival_all, rmst_risk_all = _discrete_survival_from_pmf(pmf_all, bin_widths)
    risk_scores_tensor = rmst_risk_all

    apparent_c_index = _compute_c_index_torch(risk_scores_tensor, t_all, e_all)
    holdout_c_index = _compute_c_index_torch(risk_scores_tensor[eval_idx], t_all[eval_idx], e_all[eval_idx])
    c_index = holdout_c_index if holdout_c_index is not None else apparent_c_index
    if holdout_c_index is None and evaluation_mode == "holdout":
        evaluation_mode = "holdout_fallback_apparent"
        evaluation_note = (
            "A deterministic holdout split was available, but the holdout subset did not "
            "support a comparable concordance estimate; the reported C-index is apparent."
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=eval_idx,
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

    training_meta = _training_run_metadata(loss_history, monitor_loss_history, epochs)
    insight = _scientific_summary_dl(
        "DeepHit",
        c_index,
        int(train_idx.numel()),
        int(eval_idx.numel()),
        int(e_all[train_idx].sum().item()),
        data["n_features"],
        epochs,
        loss_history,
        evaluation_mode,
        evaluation_note,
    )

    return {
        "model": "DeepHit",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "training_seed": random_seed,
        "split_seed": random_seed,
        "monitor_seed": random_seed,
        "loss_history": loss_history,
        "monitor_history": monitor_loss_history,
        "monitor_metric_label": "Monitor loss",
        "monitor_metric_goal": "min",
        "best_monitor_epoch": training_meta["best_monitor_epoch"],
        "stopped_early": training_meta["stopped_early"],
        "max_epochs_requested": training_meta["max_epochs_requested"],
        "predicted_survival_curves": predicted_survival_curves,
        "feature_importance": feature_importance,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "time_bins": [float(c) for c in bin_centers],
        "time_bin_edges": [float(edge) for edge in bin_edges],
        "insight_board": insight,
        "scientific_summary": insight,
        "epochs_trained": training_meta["epochs_trained"],
        "n_samples": data["n_samples"],
        "training_samples": int(train_idx.numel()),
        "evaluation_samples": int(eval_idx.numel()),
        "n_features": data["n_features"],
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
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
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
    num_time_bins: int,
) -> torch.Tensor:
    """Discrete-time survival NLL with right censoring.

    We treat the network output as logits for a PMF over time bins (softmax).
    - Event in bin k: -log pmf[k]
    - Censored in bin k: -log S_start[k] where S_start[k] = sum_{j>=k} pmf[j]
    """
    n = cumsum_logits.shape[0]
    bin_idx = time_bin_indices.long()
    event_mask = (events == 1)
    censor_mask = ~event_mask

    log_pmf = torch.log_softmax(cumsum_logits, dim=1)
    pmf = torch.exp(log_pmf)

    # Survival at bin edges, with a tail bucket beyond the last observed horizon.
    cdf = torch.cumsum(pmf, dim=1)
    survival_at_edges = torch.cat(
        [
            torch.ones((n, 1), device=cumsum_logits.device, dtype=cumsum_logits.dtype),
            1.0 - cdf[:, :-1],
        ],
        dim=1,
    )
    terms: list[torch.Tensor] = []
    if torch.any(event_mask):
        terms.append(-log_pmf[torch.arange(n, device=cumsum_logits.device), bin_idx][event_mask])
    if torch.any(censor_mask):
        terms.append(
            -torch.log(survival_at_edges[torch.arange(n, device=cumsum_logits.device), bin_idx][censor_mask] + 1e-12)
        )
    return torch.cat(terms).mean() if terms else torch.tensor(0.0, device=cumsum_logits.device)


@user_input_boundary
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
) -> dict[str, Any]:
    """Train a Neural MTLR model.

    Returns a JSON-serializable dict with c_index, loss_history,
    predicted_survival_curves, and calibration_data.
    """
    _require_torch()
    _seed_torch(random_seed)

    hidden_layers = hidden_layers if hidden_layers is not None else [64]
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
    x_all, t_all, e_all = data["X_tensor"], data["time_tensor"], data["event_tensor"]
    train_idx = torch.as_tensor(eval_split["train_idx"], dtype=torch.long)
    eval_idx = torch.as_tensor(eval_split["eval_idx"], dtype=torch.long)
    monitor_idx = _resolve_monitor_indices(
        monitor_indices,
        train_idx=train_idx,
        events=e_all,
        random_seed=random_seed,
    )
    evaluation_mode = str(eval_split["evaluation_mode"])
    evaluation_note = str(eval_split["evaluation_note"])
    x_train, t_train, e_train = x_all[train_idx], t_all[train_idx], e_all[train_idx]

    # Discretize time
    time_np = t_train.numpy()
    t_min, t_max = float(time_np.min()), float(time_np.max())
    if t_max <= t_min:
        t_max = t_min + 1e-6
    bin_edges = np.linspace(t_min, t_max, num_time_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = _digitize_time_bins(
        time_np,
        bin_edges,
        num_time_bins,
        preserve_tail_overflow=True,
    )
    bin_tensor = torch.tensor(bin_indices, dtype=torch.long)
    bin_widths = torch.tensor(np.diff(bin_edges), dtype=torch.float32)
    monitor_bin_tensor: torch.Tensor | None = None
    if monitor_idx is not None:
        monitor_bin_indices = _digitize_time_bins(
            t_all[monitor_idx].detach().cpu().numpy(),
            bin_edges,
            num_time_bins,
            preserve_tail_overflow=True,
        )
        monitor_bin_tensor = torch.tensor(monitor_bin_indices, dtype=torch.long)

    model = NeuralMTLRNet(data["n_features"], hidden_layers, num_time_bins, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_ADAM_WEIGHT_DECAY)
    dataset = TensorDataset(x_train, bin_tensor, e_train)
    loader_generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)

    loss_history: list[float] = []
    monitor_loss_history: list[float] = []
    best_monitor: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    early_wait = 0
    for _epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for x_b, bin_b, e_b in loader:
            optimizer.zero_grad()
            cumsum_logits = model(x_b)
            loss = _mtlr_loss(cumsum_logits, bin_b, e_b, num_time_bins)
            _require_finite_loss(loss, context="Neural MTLR loss")
            loss.backward()
            _clip_gradients(model)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(float(np.mean(epoch_losses)))
        if monitor_idx is not None and monitor_bin_tensor is not None:
            model.eval()
            with torch.inference_mode():
                cumsum_logits_monitor = model(x_all[monitor_idx])
                monitor_loss = float(_mtlr_loss(cumsum_logits_monitor, monitor_bin_tensor, e_all[monitor_idx], num_time_bins).item())
            monitor_loss_history.append(monitor_loss)
            best_monitor, early_wait, best_state, should_stop = _update_early_stopping(
                monitor_loss,
                best_value=best_monitor,
                wait_count=early_wait,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                model=model,
                best_state=best_state,
            )
            if should_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        cumsum_logits_all = model(x_all)
    # Convert to survival probabilities
    log_pmf = torch.log_softmax(cumsum_logits_all, dim=1)
    pmf = torch.exp(log_pmf)
    survival_all, rmst_risk_all = _discrete_survival_from_pmf(pmf, bin_widths)

    apparent_c_index = _compute_c_index_torch(rmst_risk_all, t_all, e_all)
    holdout_c_index = _compute_c_index_torch(rmst_risk_all[eval_idx], t_all[eval_idx], e_all[eval_idx])
    c_index = holdout_c_index if holdout_c_index is not None else apparent_c_index
    if holdout_c_index is None and evaluation_mode == "holdout":
        evaluation_mode = "holdout_fallback_apparent"
        evaluation_note = (
            "A deterministic holdout split was available, but the holdout subset did not "
            "support a comparable concordance estimate; the reported C-index is apparent."
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=eval_idx,
        evaluation_mode=evaluation_mode,
    )
    artifact_scope = _artifact_scope_label(evaluation_mode)

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

        predicted_event_prob = 1.0 - survival_np[artifact_idx_np, ref_bin + 1]
        # Use evaluable subjects only: exclude those censored before t_ref
        evaluable = (t_np > t_ref) | ((t_np <= t_ref) & (e_np == 1))
        observed_event = ((t_np <= t_ref) & (e_np == 1)).astype(float)
        decile_indices = np.argsort(predicted_event_prob)
        chunk_size = len(decile_indices) // n_deciles
        for d in range(n_deciles):
            start = d * chunk_size
            end = start + chunk_size if d < n_deciles - 1 else len(decile_indices)
            idx_slice = decile_indices[start:end]
            eval_mask = evaluable[idx_slice]
            pred_mean = float(np.mean(predicted_event_prob[idx_slice]))
            if eval_mask.sum() > 0:
                obs_mean = float(np.sum(observed_event[idx_slice][eval_mask]) / eval_mask.sum())
            else:
                obs_mean = None
            calibration_data.append({
                "decile": d + 1,
                "predicted_event_rate": pred_mean,
                "observed_event_rate": obs_mean,
            })

    training_meta = _training_run_metadata(loss_history, monitor_loss_history, epochs)
    insight = _scientific_summary_dl(
        "Neural MTLR",
        c_index,
        int(train_idx.numel()),
        int(eval_idx.numel()),
        int(e_all[train_idx].sum().item()),
        data["n_features"],
        epochs,
        loss_history,
        evaluation_mode,
        evaluation_note,
    )

    return {
        "model": "Neural MTLR",
        "c_index": c_index,
        "apparent_c_index": apparent_c_index,
        "holdout_c_index": holdout_c_index,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "training_seed": random_seed,
        "split_seed": random_seed,
        "monitor_seed": random_seed,
        "loss_history": loss_history,
        "monitor_history": monitor_loss_history,
        "monitor_metric_label": "Monitor loss",
        "monitor_metric_goal": "min",
        "best_monitor_epoch": training_meta["best_monitor_epoch"],
        "stopped_early": training_meta["stopped_early"],
        "max_epochs_requested": training_meta["max_epochs_requested"],
        "predicted_survival_curves": predicted_survival_curves,
        "calibration_data": calibration_data,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "time_bins": [float(c) for c in bin_centers],
        "time_bin_edges": [float(edge) for edge in bin_edges],
        "insight_board": insight,
        "scientific_summary": insight,
        "epochs_trained": training_meta["epochs_trained"],
        "n_samples": data["n_samples"],
        "training_samples": int(train_idx.numel()),
        "evaluation_samples": int(eval_idx.numel()),
        "n_features": data["n_features"],
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
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
) -> dict[str, Any]:
    """Train a Survival Transformer model.

    Returns a JSON-serializable dict with c_index, loss_history,
    attention_weights, feature_importance, and risk_scores.

    Notes:
    - ``batch_size`` is accepted for API consistency, but Cox partial likelihood
      is optimized on the full training risk set each epoch.
    """
    _require_torch()
    _seed_torch(random_seed)

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
    x_all, t_all, e_all = data["X_tensor"], data["time_tensor"], data["event_tensor"]
    train_idx = torch.as_tensor(eval_split["train_idx"], dtype=torch.long)
    eval_idx = torch.as_tensor(eval_split["eval_idx"], dtype=torch.long)
    monitor_idx = _resolve_monitor_indices(
        monitor_indices,
        train_idx=train_idx,
        events=e_all,
        random_seed=random_seed,
    )
    evaluation_mode = str(eval_split["evaluation_mode"])
    evaluation_note = str(eval_split["evaluation_note"])
    x_train, t_train, e_train = x_all[train_idx], t_all[train_idx], e_all[train_idx]

    if d_model % n_heads != 0:
        raise ValueError("Transformer width must be divisible by attention heads.")

    model = SurvivalTransformerNet(
        data["n_features"], d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_ADAM_WEIGHT_DECAY)

    loss_history: list[float] = []
    monitor_loss_history: list[float] = []
    best_monitor: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    early_wait = 0
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        risk = model(x_train)
        loss = _cox_partial_likelihood_loss(risk, t_train, e_train)
        _require_finite_loss(loss, context="Survival Transformer loss")
        loss.backward()
        _clip_gradients(model)
        optimizer.step()
        loss_history.append(float(loss.item()))
        if monitor_idx is not None:
            model.eval()
            monitor_c_index = _monitor_c_index(model, x_all, t_all, e_all, monitor_idx)
            if monitor_c_index is None:
                monitor_idx = None  # monitor subset has no events; disable for this run
            else:
                monitor_loss_history.append(float(monitor_c_index))
                best_monitor, early_wait, best_state, should_stop = _update_early_stopping(
                    float(monitor_c_index),
                    best_value=best_monitor,
                    wait_count=early_wait,
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    goal="max",
                    model=model,
                    best_state=best_state,
                )
                if should_stop:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        risk_scores_tensor = model(x_all)
    apparent_c_index = _compute_c_index_torch(risk_scores_tensor, t_all, e_all)
    holdout_c_index = _compute_c_index_torch(risk_scores_tensor[eval_idx], t_all[eval_idx], e_all[eval_idx])
    c_index = holdout_c_index if holdout_c_index is not None else apparent_c_index
    if holdout_c_index is None and evaluation_mode == "holdout":
        evaluation_mode = "holdout_fallback_apparent"
        evaluation_note = (
            "A deterministic holdout split was available, but the holdout subset did not "
            "support a comparable concordance estimate; the reported C-index is apparent."
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=eval_idx,
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

    training_meta = _training_run_metadata(loss_history, monitor_loss_history, epochs, monitor_goal="max")
    insight = _scientific_summary_dl(
        "Survival Transformer",
        c_index,
        int(train_idx.numel()),
        int(eval_idx.numel()),
        int(e_all[train_idx].sum().item()),
        data["n_features"],
        epochs,
        loss_history,
        evaluation_mode,
        evaluation_note,
    )
    batching_meta = _batching_metadata(
        requested_batch_size=batch_size,
        effective_batch_size=int(train_idx.numel()),
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
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "tie_method": "breslow",
        "training_seed": random_seed,
        "split_seed": random_seed,
        "monitor_seed": random_seed,
        "loss_history": loss_history,
        "monitor_history": monitor_loss_history,
        "monitor_metric_label": "Monitor C-index",
        "monitor_metric_goal": "max",
        "best_monitor_epoch": training_meta["best_monitor_epoch"],
        "stopped_early": training_meta["stopped_early"],
        "max_epochs_requested": training_meta["max_epochs_requested"],
        "attention_weights": attention_weights,
        "feature_attention": feature_attention,
        "feature_importance": feature_importance,
        "risk_scores": risk_list,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "insight_board": insight,
        "scientific_summary": insight,
        "epochs_trained": training_meta["epochs_trained"],
        "n_samples": data["n_samples"],
        "training_samples": int(train_idx.numel()),
        "evaluation_samples": int(eval_idx.numel()),
        "n_features": data["n_features"],
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
        hidden_layers = list(hidden_layers or ([hidden_dim] if hidden_dim is not None else [64]))
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
        probs = dists / max(dists.sum(), 1e-12)
        idx = rng.choice(n, p=probs)
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
) -> dict[str, Any]:
    """Train a Survival VAE model.

    Returns a JSON-serializable dict with c_index, loss_history,
    latent_embeddings, cluster_labels, cluster_survival_curves, and risk_scores.

    Notes:
    - ``batch_size`` is accepted for API consistency, but the current VAE survival
      objective is optimized on the full training partition each epoch.
    """
    _require_torch()
    _seed_torch(random_seed)
    hidden_layers = list(hidden_layers or ([hidden_dim] if hidden_dim is not None else [64]))

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
    x_all, t_all, e_all = data["X_tensor"], data["time_tensor"], data["event_tensor"]
    train_idx = torch.as_tensor(eval_split["train_idx"], dtype=torch.long)
    eval_idx = torch.as_tensor(eval_split["eval_idx"], dtype=torch.long)
    monitor_idx = _resolve_monitor_indices(
        monitor_indices,
        train_idx=train_idx,
        events=e_all,
        random_seed=random_seed,
    )
    evaluation_mode = str(eval_split["evaluation_mode"])
    evaluation_note = str(eval_split["evaluation_note"])
    x_train, t_train, e_train = x_all[train_idx], t_all[train_idx], e_all[train_idx]
    categorical_feature_indices = list(data.get("categorical_feature_indices", []))
    numeric_feature_indices = list(data.get("numeric_feature_indices", []))

    model = SurvivalVAENet(data["n_features"], hidden_layers=hidden_layers, latent_dim=latent_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=_ADAM_WEIGHT_DECAY)

    loss_history: list[float] = []
    monitor_loss_history: list[float] = []
    best_monitor: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    early_wait = 0
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_recon, mu, log_var, risk = model(x_train)
        loss = _vae_combined_loss(
            x_train,
            x_recon,
            mu,
            log_var,
            risk,
            t_train,
            e_train,
            categorical_feature_indices=categorical_feature_indices,
            numeric_feature_indices=numeric_feature_indices,
        )
        _require_finite_loss(loss, context="Survival VAE loss")
        loss.backward()
        _clip_gradients(model)
        optimizer.step()
        loss_history.append(float(loss.item()))
        if monitor_idx is not None:
            model.eval()
            with torch.inference_mode():
                _, _, _, risk_monitor = model(x_all[monitor_idx])
            monitor_c_index = _compute_c_index_torch(risk_monitor, t_all[monitor_idx], e_all[monitor_idx])
            if monitor_c_index is None:
                monitor_idx = None  # monitor subset has no events; disable for this run
            else:
                monitor_loss_history.append(float(monitor_c_index))
                best_monitor, early_wait, best_state, should_stop = _update_early_stopping(
                    float(monitor_c_index),
                    best_value=best_monitor,
                    wait_count=early_wait,
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    goal="max",
                    model=model,
                    best_state=best_state,
                )
                if should_stop:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.inference_mode():
        x_recon_all, mu_all, log_var_all, risk_all = model(x_all)
        latent_all = mu_all  # get_latent returns mu; reuse from the forward pass above

    apparent_c_index = _compute_c_index_torch(risk_all, t_all, e_all)
    holdout_c_index = _compute_c_index_torch(risk_all[eval_idx], t_all[eval_idx], e_all[eval_idx])
    c_index = holdout_c_index if holdout_c_index is not None else apparent_c_index
    if holdout_c_index is None and evaluation_mode == "holdout":
        evaluation_mode = "holdout_fallback_apparent"
        evaluation_note = (
            "A deterministic holdout split was available, but the holdout subset did not "
            "support a comparable concordance estimate; the reported C-index is apparent."
        )
    artifact_idx = _select_artifact_indices(
        total_n=data["n_samples"],
        eval_idx=eval_idx,
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
        c_times = time_np[mask]
        c_events = event_np[mask]

        # Simple KM estimator
        unique_times = np.sort(np.unique(c_times))
        survival = np.ones(len(unique_times))
        n_at_risk = float(mask.sum())
        for j, t_j in enumerate(unique_times):
            d_j = float(((c_times == t_j) & (c_events == 1)).sum())
            if n_at_risk > 0:
                survival[j] = (survival[j - 1] if j > 0 else 1.0) * (1.0 - d_j / n_at_risk)
            else:
                survival[j] = survival[j - 1] if j > 0 else 1.0
            # Update at-risk count: subtract all who had event or were censored at this time
            n_at_risk -= float((c_times == t_j).sum())

        timeline = [0.0] + [float(t) for t in unique_times]
        surv_values = [1.0] + [float(s) for s in survival]
        cluster_survival_curves.append({
            "cluster": int(c),
            "n_patients": int(mask.sum()),
            "curve": _make_survival_curve(timeline, surv_values),
        })

    training_meta = _training_run_metadata(loss_history, monitor_loss_history, epochs, monitor_goal="max")
    insight = _scientific_summary_dl(
        "Survival VAE",
        c_index,
        int(train_idx.numel()),
        int(eval_idx.numel()),
        int(e_all[train_idx].sum().item()),
        data["n_features"],
        epochs,
        loss_history,
        evaluation_mode,
        evaluation_note,
    )
    batching_meta = _batching_metadata(
        requested_batch_size=batch_size,
        effective_batch_size=int(train_idx.numel()),
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
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "tie_method": "breslow",
        "training_seed": random_seed,
        "split_seed": random_seed,
        "monitor_seed": random_seed,
        "loss_history": loss_history,
        "monitor_history": monitor_loss_history,
        "monitor_metric_label": "Monitor C-index",
        "monitor_metric_goal": "max",
        "best_monitor_epoch": training_meta["best_monitor_epoch"],
        "stopped_early": training_meta["stopped_early"],
        "max_epochs_requested": training_meta["max_epochs_requested"],
        "latent_embeddings": latent_embeddings,
        "cluster_labels": cluster_labels,
        "cluster_survival_curves": cluster_survival_curves,
        "risk_scores": risk_list,
        "artifact_scope": artifact_scope,
        "artifact_samples": int(artifact_idx.numel()),
        "insight_board": insight,
        "scientific_summary": insight,
        "epochs_trained": training_meta["epochs_trained"],
        "n_samples": data["n_samples"],
        "training_samples": int(train_idx.numel()),
        "evaluation_samples": int(eval_idx.numel()),
        "n_features": data["n_features"],
        "n_clusters": n_clusters,
        "latent_dim": latent_dim,
        **batching_meta,
    }

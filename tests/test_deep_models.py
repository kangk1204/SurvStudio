from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.sample_data import make_example_dataset


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
@pytest.mark.parametrize(
    "trainer_name, trainer_kwargs, result_key",
    [
        ("deep_surv", {"hidden_layers": [8], "epochs": 1, "batch_size": 8, "random_seed": 11}, None),
        ("deep_hit", {"hidden_layers": [8], "num_time_bins": 6, "epochs": 1, "batch_size": 8, "random_seed": 11}, "predicted_survival_curves"),
        ("neural_mtlr", {"hidden_layers": [8], "num_time_bins": 6, "epochs": 1, "batch_size": 8, "random_seed": 11}, "predicted_survival_curves"),
        ("transformer", {"d_model": 16, "n_heads": 4, "n_layers": 1, "epochs": 1, "batch_size": 8, "random_seed": 11}, None),
        ("vae", {"hidden_dim": 16, "latent_dim": 4, "epochs": 1, "batch_size": 8, "random_seed": 11}, "cluster_survival_curves"),
    ],
)
def test_deep_trainers_report_explicit_evaluation_metadata(
    trainer_name: str,
    trainer_kwargs: dict[str, object],
    result_key: str | None,
) -> None:
    from survival_toolkit.deep_models import (
        train_deepsurv,
        train_deephit,
        train_neural_mtlr,
        train_survival_transformer,
        train_survival_vae,
    )

    df = make_example_dataset(seed=9, n_patients=60)
    features = ["age", "biomarker_score", "immune_index"]
    trainers = {
        "deep_surv": train_deepsurv,
        "deep_hit": train_deephit,
        "neural_mtlr": train_neural_mtlr,
        "transformer": train_survival_transformer,
        "vae": train_survival_vae,
    }

    result = trainers[trainer_name](
        df,
        time_column="os_months",
        event_column="os_event",
        features=features,
        **trainer_kwargs,
    )

    assert result["c_index"] is not None
    assert result["apparent_c_index"] is not None
    assert result["evaluation_mode"] != "apparent"
    assert result["evaluation_note"]
    assert result["training_samples"] + result["evaluation_samples"] == result["n_samples"]
    assert result["scientific_summary"]["metrics"][1]["label"] == "Evaluation mode"
    assert result["evaluation_mode"].replace("_", " ") in result["scientific_summary"]["headline"]

    if result_key is not None:
        assert result[result_key]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
@pytest.mark.parametrize(
    "trainer_name, trainer_kwargs",
    [
        ("deep_surv", {"hidden_layers": [8], "epochs": 1, "batch_size": 8, "random_seed": 21}),
        ("deep_hit", {"hidden_layers": [8], "num_time_bins": 6, "epochs": 1, "batch_size": 8, "random_seed": 21}),
        ("neural_mtlr", {"hidden_layers": [8], "num_time_bins": 6, "epochs": 1, "batch_size": 8, "random_seed": 21}),
        ("transformer", {"d_model": 16, "n_heads": 4, "n_layers": 1, "epochs": 1, "batch_size": 8, "random_seed": 21}),
        ("vae", {"hidden_dim": 16, "latent_dim": 4, "epochs": 1, "batch_size": 8, "random_seed": 21}),
    ],
)
def test_deep_trainers_avoid_full_cohort_preprocessing_for_holdout(
    monkeypatch,
    trainer_name: str,
    trainer_kwargs: dict[str, object],
) -> None:
    import survival_toolkit.deep_models as deep_models

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("_prepare_deep_data should not be used for holdout preprocessing.")

    monkeypatch.setattr(deep_models, "_prepare_deep_data", _raise_if_called)

    df = make_example_dataset(seed=19, n_patients=60)
    features = ["age", "biomarker_score", "immune_index"]
    trainers = {
        "deep_surv": deep_models.train_deepsurv,
        "deep_hit": deep_models.train_deephit,
        "neural_mtlr": deep_models.train_neural_mtlr,
        "transformer": deep_models.train_survival_transformer,
        "vae": deep_models.train_survival_vae,
    }

    result = trainers[trainer_name](
        df,
        time_column="os_months",
        event_column="os_event",
        features=features,
        **trainer_kwargs,
    )

    assert result["evaluation_mode"].startswith("holdout")
    assert result["training_samples"] < result["n_samples"]
    assert result["evaluation_samples"] < result["n_samples"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_models_fall_back_to_apparent_evaluation_for_small_cohorts() -> None:
    from survival_toolkit.deep_models import train_deepsurv

    df = make_example_dataset(seed=3, n_patients=12)
    result = train_deepsurv(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=4,
        random_seed=11,
    )

    assert result["evaluation_mode"] == "apparent"
    assert "skipped" in result["evaluation_note"].lower() or "too small" in result["evaluation_note"].lower()
    assert result["training_samples"] == result["n_samples"]
    assert result["evaluation_samples"] == result["n_samples"]


def test_run_deep_compare_task_passes_none_dataframe_when_prepared_data_is_supplied(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    seen: dict[str, object] = {}

    def _fake_train(
        df,
        **kwargs,
    ):
        seen["df"] = df
        seen["prepared_data"] = kwargs["prepared_data"]
        return {
            "c_index": 0.61,
            "n_features": 3,
            "training_samples": 20,
            "evaluation_samples": 10,
            "epochs_trained": 1,
        }

    monkeypatch.setattr(deep_models, "train_deepsurv", _fake_train)

    result = deep_models._run_deep_compare_task(
        {
            "model_name": "DeepSurv",
            "repeat": 1,
            "fold": 2,
            "seed": 123,
            "split_seed": 456,
            "monitor_seed": 789,
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "event_positive_value": 1,
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 8,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 1e-4,
            "prepared_data": {"n_features": 3},
            "evaluation_split": {"train_idx": np.array([0, 1]), "eval_idx": np.array([2]), "evaluation_mode": "holdout", "evaluation_note": "ok"},
            "monitor_indices": np.array([0, 1]),
            "extra_kwargs": {"hidden_layers": [8], "dropout": 0.1},
        }
    )

    assert seen["df"] is None
    assert seen["prepared_data"] == {"n_features": 3}
    assert result["model"] == "DeepSurv"


def test_run_deep_compare_task_rejects_apparent_fallback_when_holdout_is_required(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    def _fake_train(df, **kwargs):
        return {
            "c_index": 0.61,
            "n_features": 3,
            "training_samples": 20,
            "evaluation_samples": 10,
            "epochs_trained": 1,
            "evaluation_mode": "holdout_fallback_apparent",
            "evaluation_note": "fallback",
        }

    monkeypatch.setattr(deep_models, "train_deepsurv", _fake_train)

    with pytest.raises(ValueError, match="clean holdout evaluation"):
        deep_models._run_deep_compare_task(
            {
                "model_name": "DeepSurv",
                "repeat": 1,
                "fold": 1,
                "seed": 123,
                "split_seed": 456,
                "monitor_seed": 789,
                "time_column": "os_months",
                "event_column": "os_event",
                "features": ["age"],
                "categorical_features": [],
                "event_positive_value": 1,
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 8,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 1e-4,
                "prepared_data": {"n_features": 1},
                "evaluation_split": {"train_idx": np.array([0, 1]), "eval_idx": np.array([2]), "evaluation_mode": "holdout", "evaluation_note": "ok"},
                "monitor_indices": np.array([0, 1]),
                "extra_kwargs": {"hidden_layers": [8], "dropout": 0.1},
                "require_holdout_evaluation": True,
            }
        )


def test_survival_from_log_cumulative_hazard_handles_extreme_values() -> None:
    import survival_toolkit.deep_models as deep_models

    survival = deep_models._survival_from_log_cumulative_hazard(np.array([-np.inf, 0.0, 60.0], dtype=float))

    assert survival[0] == pytest.approx(1.0)
    assert 0.0 < survival[1] < 1.0
    assert survival[2] == pytest.approx(0.0)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_vae_reparameterize_clamps_extreme_log_variance() -> None:
    import torch
    import survival_toolkit.deep_models as deep_models

    model = deep_models.SurvivalVAENet(in_features=4, hidden_layers=[8], latent_dim=2, dropout=0.1)
    model.train()
    mu = torch.zeros((2, 2), dtype=torch.float32)
    log_var = torch.full((2, 2), 1_000.0, dtype=torch.float32)

    sample = model.reparameterize(mu, log_var)

    assert torch.isfinite(sample).all()


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_survival_vae_uses_posterior_mean_for_risk_head_during_training() -> None:
    import torch
    import survival_toolkit.deep_models as deep_models

    torch.manual_seed(3)
    model = deep_models.SurvivalVAENet(in_features=4, hidden_layers=[8], latent_dim=2, dropout=0.0)
    model.train()
    x = torch.randn((5, 4), dtype=torch.float32)

    x_recon_a, mu_a, _, risk_a = model(x)
    x_recon_b, mu_b, _, risk_b = model(x)

    assert not torch.allclose(x_recon_a, x_recon_b)
    assert torch.allclose(mu_a, mu_b)
    assert torch.allclose(risk_a, risk_b)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_prepare_deep_inputs_reports_missing_columns_cleanly() -> None:
    import survival_toolkit.deep_models as deep_models

    with pytest.raises(ValueError, match="missing required columns"):
        deep_models._prepare_deep_training_inputs(
            pd.DataFrame(),
            time_column="os_months",
            event_column="os_event",
            features=["age"],
        )


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_coerce_deep_frame_keeps_rows_with_missing_features_when_outcome_is_valid() -> None:
    import survival_toolkit.deep_models as deep_models

    frame = make_example_dataset(seed=81, n_patients=40).loc[:, ["os_months", "os_event", "age", "sex"]].copy()
    frame.loc[0, "age"] = np.nan
    frame.loc[1, "sex"] = pd.NA

    cleaned = deep_models._coerce_deep_frame(
        frame,
        time_column="os_months",
        event_column="os_event",
        features=["age", "sex"],
        categorical_features=["sex"],
        event_positive_value=1,
    )

    assert cleaned.shape[0] == frame.shape[0]
    assert cleaned["age"].isna().sum() == 1
    assert cleaned["sex"].isna().sum() == 1


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_encoder_imputes_missing_numeric_and_categorical_values() -> None:
    import torch
    import survival_toolkit.deep_models as deep_models

    frame = make_example_dataset(seed=82, n_patients=40).loc[:, ["os_months", "os_event", "age", "sex"]].copy()
    frame.loc[0, "age"] = np.nan
    frame.loc[1, "sex"] = pd.NA

    cleaned = deep_models._coerce_deep_frame(
        frame,
        time_column="os_months",
        event_column="os_event",
        features=["age", "sex"],
        categorical_features=["sex"],
        event_positive_value=1,
    )
    encoder = deep_models._fit_deep_encoder(cleaned, ["age", "sex"], ["sex"])
    transformed = deep_models._transform_deep_frame(
        cleaned,
        time_column="os_months",
        event_column="os_event",
        encoder=encoder,
    )

    assert transformed["n_samples"] == cleaned.shape[0]
    assert torch.isfinite(transformed["X_tensor"]).all()
    assert "sex__missing" in transformed["feature_names"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_encoder_records_categorical_and_numeric_reconstruction_indices() -> None:
    import survival_toolkit.deep_models as deep_models

    frame = pd.DataFrame(
        {
            "os_months": [5, 6, 7, 8],
            "os_event": [1, 0, 1, 0],
            "age": [62, 58, 55, 67],
            "sex": ["female", "male", pd.NA, "female"],
        }
    )
    cleaned = deep_models._coerce_deep_frame(
        frame,
        time_column="os_months",
        event_column="os_event",
        features=["age", "sex"],
        categorical_features=["sex"],
        event_positive_value=1,
        min_samples=1,
    )
    encoder = deep_models._fit_deep_encoder(cleaned, ["age", "sex"], ["sex"])
    transformed = deep_models._transform_deep_frame(
        cleaned,
        time_column="os_months",
        event_column="os_event",
        encoder=encoder,
    )

    assert transformed["feature_names"] == ["sex_male", "sex__unknown", "sex__missing", "age"]
    assert transformed["categorical_feature_indices"] == [0, 1, 2]
    assert transformed["numeric_feature_indices"] == [3]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_vae_combined_loss_uses_bce_for_categorical_reconstruction_terms() -> None:
    import torch
    import torch.nn.functional as F

    from survival_toolkit.deep_models import _vae_combined_loss

    x = torch.tensor(
        [
            [1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, -1.0, 0.25],
        ],
        dtype=torch.float32,
    )
    x_recon = torch.tensor(
        [
            [2.0, -1.0, 0.0, -1.0],
            [-2.0, 3.0, -0.5, 0.75],
        ],
        dtype=torch.float32,
    )
    mu = torch.zeros((2, 2), dtype=torch.float32)
    log_var = torch.zeros((2, 2), dtype=torch.float32)
    risk = torch.zeros((2, 1), dtype=torch.float32)
    times = torch.tensor([1.0, 2.0], dtype=torch.float32)
    events = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = _vae_combined_loss(
        x,
        x_recon,
        mu,
        log_var,
        risk,
        times,
        events,
        kl_weight=0.0,
        cox_weight=0.0,
        categorical_feature_indices=[0, 1],
        numeric_feature_indices=[2, 3],
    )

    expected = (
        F.binary_cross_entropy_with_logits(x_recon[:, :2], x[:, :2], reduction="sum")
        + F.mse_loss(x_recon[:, 2:], x[:, 2:], reduction="sum")
    ) / x.numel()
    assert torch.allclose(loss, expected)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
@pytest.mark.parametrize(
    "trainer_name, trainer_kwargs, curve_key",
    [
        ("deep_hit", {"hidden_layers": [8], "num_time_bins": 6, "epochs": 1, "batch_size": 8, "random_seed": 11}, "predicted_survival_curves"),
        ("neural_mtlr", {"hidden_layers": [8], "num_time_bins": 6, "epochs": 1, "batch_size": 8, "random_seed": 11}, "predicted_survival_curves"),
    ],
)
def test_discrete_survival_outputs_preserve_tail_mass(
    trainer_name: str,
    trainer_kwargs: dict[str, object],
    curve_key: str,
) -> None:
    from survival_toolkit.deep_models import train_deephit, train_neural_mtlr

    df = make_example_dataset(seed=9, n_patients=60)
    features = ["age", "biomarker_score", "immune_index"]
    trainers = {
        "deep_hit": train_deephit,
        "neural_mtlr": train_neural_mtlr,
    }

    result = trainers[trainer_name](
        df,
        time_column="os_months",
        event_column="os_event",
        features=features,
        **trainer_kwargs,
    )

    curves = result[curve_key]
    assert curves
    first_curve = curves[0]["curve"]
    assert len(first_curve["timeline"]) == len(result["time_bin_edges"])
    assert first_curve["survival"][0] == pytest.approx(1.0)
    assert first_curve["survival"][-1] > 0.0
    assert len(first_curve["timeline"]) == len(first_curve["survival"])


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deephit_ranking_loss_prefers_correct_survival_ordering() -> None:
    import torch

    from survival_toolkit.deep_models import _deephit_loss

    events = torch.tensor([1.0, 0.0], dtype=torch.float32)
    time_bins = torch.tensor([1, 2], dtype=torch.long)

    good = torch.tensor(
        [[0.4, 0.5, 0.1, 0.0], [0.1, 0.2, 0.2, 0.5]],
        dtype=torch.float32,
    )
    bad = torch.tensor(
        [[0.1, 0.2, 0.2, 0.5], [0.4, 0.5, 0.1, 0.0]],
        dtype=torch.float32,
    )

    good_loss = _deephit_loss(good, time_bins, events, alpha=0.0).item()
    bad_loss = _deephit_loss(bad, time_bins, events, alpha=0.0).item()

    assert good_loss < bad_loss


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deephit_ranking_loss_is_invariant_to_row_order_when_many_events() -> None:
    import torch

    from survival_toolkit.deep_models import _deephit_loss

    torch.manual_seed(7)
    logits = torch.randn(180, 5, dtype=torch.float32)
    pmf = torch.softmax(logits, dim=1)
    time_bins = torch.tensor([(idx % 4) for idx in range(180)], dtype=torch.long)
    events = torch.tensor([1.0] * 150 + [0.0] * 30, dtype=torch.float32)
    perm = torch.randperm(180)

    baseline = _deephit_loss(pmf, time_bins, events, alpha=0.0).item()
    permuted = _deephit_loss(pmf[perm], time_bins[perm], events[perm], alpha=0.0).item()

    # The ranking-loss reduction is mathematically order-invariant, but tiny
    # float32 accumulation differences can appear across platforms.
    assert baseline == pytest.approx(permuted, rel=2e-7, abs=2e-6)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deephit_ranking_loss_stays_finite_for_large_pairwise_gaps() -> None:
    import torch

    from survival_toolkit.deep_models import _deephit_loss

    events = torch.tensor([1.0, 0.0], dtype=torch.float32)
    time_bins = torch.tensor([0, 1], dtype=torch.long)
    pmf = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    loss = _deephit_loss(pmf, time_bins, events, alpha=0.0)

    assert torch.isfinite(loss)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deephit_loss_stays_finite_for_near_zero_tail_survival() -> None:
    import torch

    from survival_toolkit.deep_models import _deephit_loss

    pmf = torch.tensor(
        [
            [0.5, 0.5, 1e-12],
            [0.2, 0.3, 0.5],
        ],
        dtype=torch.float32,
    )
    time_bins = torch.tensor([2, 1], dtype=torch.long)
    events = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss = _deephit_loss(pmf, time_bins, events, alpha=1.0)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compute_c_index_torch_rejects_nonfinite_risk_scores() -> None:
    import torch

    from survival_toolkit.deep_models import _compute_c_index_torch

    with pytest.raises(ValueError, match="NaN or Inf"):
        _compute_c_index_torch(
            torch.tensor([0.1, float("nan")], dtype=torch.float32),
            torch.tensor([5.0, 8.0], dtype=torch.float32),
            torch.tensor([1.0, 0.0], dtype=torch.float32),
        )


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_transformer_attention_extraction_calls_self_attention_once_per_layer() -> None:
    import torch

    from survival_toolkit.deep_models import SurvivalTransformerNet

    model = SurvivalTransformerNet(in_features=4, d_model=8, n_heads=2, n_layers=2, dropout=0.0)
    x = torch.randn(3, 4, dtype=torch.float32)
    calls = {"count": 0}

    for layer in model.transformer.layers:
        original_forward = layer.self_attn.forward

        def _counting_forward(*args, _original=original_forward, **kwargs):
            calls["count"] += 1
            return _original(*args, **kwargs)

        layer.self_attn.forward = _counting_forward

    attention = model.get_attention_weights(x)

    assert len(attention) == 2
    assert calls["count"] == 2


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_discrete_survival_helper_keeps_tail_mass() -> None:
    import torch

    from survival_toolkit.deep_models import _discrete_survival_from_pmf

    pmf = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    bin_widths = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    survival_at_edges, risk_scores = _discrete_survival_from_pmf(pmf, bin_widths)

    assert survival_at_edges.shape == (1, 4)
    assert survival_at_edges[0, -1].item() == pytest.approx(0.4)
    assert survival_at_edges[0, -1].item() > 0.0
    assert risk_scores.shape == (1,)


def test_digitize_time_bins_preserves_tail_bucket_for_overflow_times() -> None:
    from survival_toolkit.deep_models import _digitize_time_bins

    bin_edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    indices = _digitize_time_bins(
        np.array([5.0, 15.0, 30.0, 42.0], dtype=float),
        bin_edges,
        num_time_bins=3,
        preserve_tail_overflow=True,
    )

    assert indices.tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_survival_transformer_and_vae_train_without_batch_splitting() -> None:
    from survival_toolkit.deep_models import train_survival_transformer, train_survival_vae

    df = make_example_dataset(seed=9, n_patients=24)
    features = ["age", "biomarker_score", "immune_index"]

    transformer = train_survival_transformer(
        df,
        time_column="os_months",
        event_column="os_event",
        features=features,
        epochs=1,
        batch_size=1,
        d_model=16,
        n_heads=4,
        n_layers=1,
        random_seed=11,
    )
    assert transformer["c_index"] is not None
    assert transformer["requested_batch_size"] == 1
    assert transformer["effective_batch_size"] == transformer["training_samples"]
    assert transformer["optimization_mode"] == "full_batch_cox"
    assert transformer["monitor_metric_label"] == "Monitor C-index"
    assert transformer["monitor_metric_goal"] == "max"
    assert "full training partition" in transformer["batching_note"]

    vae = train_survival_vae(
        df,
        time_column="os_months",
        event_column="os_event",
        features=features,
        epochs=1,
        batch_size=1,
        hidden_dim=16,
        latent_dim=4,
        random_seed=11,
    )
    assert vae["c_index"] is not None
    assert vae["requested_batch_size"] == 1
    assert vae["effective_batch_size"] == vae["training_samples"]
    assert vae["optimization_mode"] == "full_batch_vae"
    assert vae["monitor_metric_label"] == "Monitor loss"
    assert vae["monitor_metric_goal"] == "min"
    assert "requested batch size" in vae["batching_note"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deepsurv_reports_full_batch_metadata_and_monitor_c_index() -> None:
    from survival_toolkit.deep_models import train_deepsurv

    df = make_example_dataset(seed=17, n_patients=36)
    result = train_deepsurv(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=2,
        batch_size=3,
        random_seed=7,
    )

    assert result["c_index"] is not None
    assert result["requested_batch_size"] == 3
    assert result["effective_batch_size"] == result["training_samples"]
    assert result["optimization_mode"] == "full_batch_cox"
    assert result["monitor_metric_label"] == "Monitor C-index"
    assert result["monitor_metric_goal"] == "max"
    assert "IBS" in " ".join(result["scientific_summary"]["cautions"])


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_survival_vae_eval_is_deterministic() -> None:
    import torch

    from survival_toolkit.deep_models import SurvivalVAENet, _prepare_deep_data

    df = make_example_dataset(seed=13, n_patients=20)
    data = _prepare_deep_data(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
    )
    model = SurvivalVAENet(data["n_features"], hidden_dim=16, latent_dim=4, dropout=0.1)
    model.eval()
    x = data["X_tensor"][:4]

    with torch.no_grad():
        first = model(x)
        second = model(x)

    for lhs, rhs in zip(first, second, strict=False):
        assert torch.allclose(lhs, rhs)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_neural_mtlr_net_uses_dropout_layers() -> None:
    import torch.nn as nn

    from survival_toolkit.deep_models import NeuralMTLRNet

    model = NeuralMTLRNet(in_features=6, hidden_layers=[16, 8], num_time_bins=10, dropout=0.2)

    dropout_layers = [module for module in model.encoder if isinstance(module, nn.Dropout)]
    assert len(dropout_layers) == 2
    assert all(layer.p == pytest.approx(0.2) for layer in dropout_layers)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_neural_mtlr_uses_right_cumulative_parameterization() -> None:
    import torch
    import torch.nn as nn

    from survival_toolkit.deep_models import NeuralMTLRNet

    model = NeuralMTLRNet(in_features=2, hidden_layers=[2], num_time_bins=3, dropout=0.0)
    model.encoder = nn.Identity()
    model.output_layer = nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        model.output_layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, -1.0],
                ],
                dtype=torch.float32,
            )
        )

    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    logits = model.output_layer(x)
    expected = torch.flip(torch.cumsum(torch.flip(logits, dims=[1]), dim=1), dims=[1])

    assert torch.allclose(model(x), expected)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_survival_vae_uses_full_hidden_layer_stack() -> None:
    import torch.nn as nn

    from survival_toolkit.deep_models import SurvivalVAENet

    model = SurvivalVAENet(in_features=5, hidden_layers=[16, 8], latent_dim=4, dropout=0.1)

    encoder_linears = [module for module in model.encoder if isinstance(module, nn.Linear)]
    decoder_linears = [module for module in model.decoder if isinstance(module, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in encoder_linears] == [(5, 16), (16, 8)]
    assert model.fc_mu.in_features == 8
    assert model.fc_log_var.in_features == 8
    assert [(layer.in_features, layer.out_features) for layer in decoder_linears] == [(4, 8), (8, 16), (16, 5)]


def test_evaluate_single_deep_survival_model_passes_dropout_to_mtlr(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    seen: dict[str, object] = {}

    def _fake_train(*args, **kwargs):
        seen.update(kwargs)
        return {"model": "Neural MTLR", "c_index": 0.6}

    monkeypatch.setattr(deep_models, "train_neural_mtlr", _fake_train)

    result = deep_models.evaluate_single_deep_survival_model(
        "mtlr",
        time_column="os_months",
        event_column="os_event",
        df=pd.DataFrame({"os_months": [1, 2], "os_event": [1, 0], "age": [50, 60]}),
        features=["age"],
        hidden_layers=[16, 8],
        dropout=0.25,
        epochs=10,
        batch_size=8,
    )

    assert result["model"] == "Neural MTLR"
    assert seen["dropout"] == pytest.approx(0.25)
    assert seen["hidden_layers"] == [16, 8]


def test_evaluate_single_deep_survival_model_passes_full_hidden_layers_to_vae(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    seen: dict[str, object] = {}

    def _fake_train(*args, **kwargs):
        seen.update(kwargs)
        return {"model": "Survival VAE", "c_index": 0.6}

    monkeypatch.setattr(deep_models, "train_survival_vae", _fake_train)

    result = deep_models.evaluate_single_deep_survival_model(
        "vae",
        time_column="os_months",
        event_column="os_event",
        df=pd.DataFrame({"os_months": [1, 2], "os_event": [1, 0], "age": [50, 60]}),
        features=["age"],
        hidden_layers=[32, 16],
        latent_dim=6,
        n_clusters=4,
        dropout=0.15,
        epochs=10,
        batch_size=8,
    )

    assert result["model"] == "Survival VAE"
    assert seen["hidden_layers"] == [32, 16]
    assert seen["latent_dim"] == 6
    assert seen["n_clusters"] == 4


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_uses_shared_holdout_and_monitor_splits(monkeypatch) -> None:
    import numpy as np
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=14, n_patients=60)
    seen: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def _fake_trainer(*args, **kwargs):
        evaluation_split = kwargs["evaluation_split"]
        monitor_indices = tuple(int(v) for v in np.asarray(kwargs["monitor_indices"], dtype=int).tolist())
        seen.append((
            tuple(int(v) for v in np.asarray(evaluation_split["eval_idx"], dtype=int).tolist()),
            monitor_indices,
            int(kwargs["random_seed"]),
        ))
        return {
            "c_index": 0.61,
            "apparent_c_index": 0.61,
            "holdout_c_index": 0.61,
            "evaluation_mode": "holdout",
            "training_seed": int(kwargs["random_seed"]),
            "split_seed": 21,
            "monitor_seed": 21,
            "epochs_trained": 1,
            "n_features": 3,
            "training_samples": len(evaluation_split["train_idx"]),
            "evaluation_samples": len(evaluation_split["eval_idx"]),
        }

    monkeypatch.setattr(deep_models, "train_deepsurv", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_deephit", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_neural_mtlr", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_survival_transformer", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_survival_vae", _fake_trainer)

    result = deep_models.compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        random_seed=21,
    )

    assert result["comparison_table"]
    assert len(seen) == 5
    assert len({item[0] for item in seen}) == 1
    assert len({item[1] for item in seen}) == 1
    assert len({item[2] for item in seen}) == 1
    assert result["shared_training_seed"] == 21
    assert result["shared_split_seed"] == 21
    assert result["shared_monitor_seed"] == 21
    assert all(row["training_seed"] == 21 for row in result["comparison_table"])


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_neural_mtlr_single_run_matches_compare_all_holdout_when_settings_match(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=88, n_patients=48)
    original_mtlr = deep_models.train_neural_mtlr

    def _stub_model(name: str):
        def _inner(*args, **kwargs):
            return {
                "model": name,
                "c_index": 0.50,
                "apparent_c_index": 0.50,
                "holdout_c_index": 0.50,
                "evaluation_mode": "holdout",
                "training_seed": kwargs["random_seed"],
                "split_seed": kwargs["random_seed"],
                "monitor_seed": kwargs["random_seed"],
                "epochs_trained": 1,
                "n_features": 3,
                "training_samples": len(kwargs["evaluation_split"]["train_idx"]),
                "evaluation_samples": len(kwargs["evaluation_split"]["eval_idx"]),
            }
        return _inner

    monkeypatch.setattr(deep_models, "train_deepsurv", _stub_model("DeepSurv"))
    monkeypatch.setattr(deep_models, "train_deephit", _stub_model("DeepHit"))
    monkeypatch.setattr(deep_models, "train_survival_transformer", _stub_model("Survival Transformer"))
    monkeypatch.setattr(deep_models, "train_survival_vae", _stub_model("Survival VAE"))

    compare_result = deep_models.compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        random_seed=33,
    )

    compare_row = next(row for row in compare_result["comparison_table"] if row["model"] == "Neural MTLR")
    single_result = original_mtlr(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        num_time_bins=6,
        epochs=1,
        batch_size=8,
        random_seed=33,
    )

    assert compare_row["training_seed"] == single_result["training_seed"] == 33
    assert compare_row["split_seed"] == single_result["split_seed"] == 33
    assert compare_row["monitor_seed"] == single_result["monitor_seed"] == 33
    assert compare_row["c_index"] == pytest.approx(single_result["c_index"], rel=0.0, abs=1e-9)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deepsurv_uses_internal_monitor_subset_not_eval_fold(monkeypatch) -> None:
    import torch
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=52, n_patients=20)
    prepared = deep_models._prepare_deep_data(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
    )
    n_samples = int(prepared["n_samples"])
    evaluation_split = {
        "train_idx": np.arange(0, n_samples - 4, dtype=int),
        "eval_idx": np.arange(n_samples - 4, n_samples, dtype=int),
        "evaluation_mode": "holdout",
        "evaluation_note": "test split",
    }
    monitor_indices = np.array([2, 3, 4, 5], dtype=int)
    seen_times: list[tuple[float, ...]] = []

    def _recording_monitor(model, x_all, times, events, monitor_idx):
        seen_times.append(
            tuple(float(v) for v in times[monitor_idx].detach().cpu().numpy().ravel().tolist())
        )
        return 0.61

    monkeypatch.setattr(deep_models, "_monitor_c_index", _recording_monitor)

    deep_models.train_deepsurv(
        pd.DataFrame(),
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        random_seed=11,
        prepared_data=prepared,
        evaluation_split=evaluation_split,
        monitor_indices=monitor_indices,
        early_stopping_patience=2,
    )

    monitor_times = tuple(
        float(v) for v in prepared["time_tensor"][torch.as_tensor(monitor_indices)].detach().cpu().numpy().tolist()
    )
    assert seen_times == [monitor_times]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_encoder_preserves_unknown_bucket_for_unseen_levels() -> None:
    import survival_toolkit.deep_models as deep_models

    train = pd.DataFrame(
        {
            "stage": ["I", "I", "II"],
            "age": [50, 60, 70],
            "os_months": [10.0, 12.0, 14.0],
            "os_event": [1, 0, 1],
        }
    )
    eval_frame = pd.DataFrame(
        {
            "stage": ["III", "I"],
            "age": [55, 65],
            "os_months": [11.0, 13.0],
            "os_event": [0, 1],
        }
    )

    encoder = deep_models._fit_deep_encoder(train, ["stage", "age"], ["stage"])
    transformed = deep_models._transform_deep_frame(
        eval_frame,
        time_column="os_months",
        event_column="os_event",
        encoder=encoder,
    )

    unknown_idx = transformed["feature_names"].index("stage__unknown")
    X = transformed["X_tensor"].detach().cpu().numpy()
    assert X[0, unknown_idx] == pytest.approx(1.0)
    assert X[1, unknown_idx] == pytest.approx(0.0)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deepsurv_holdout_artifacts_use_evaluation_subset(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=52, n_patients=20)
    prepared = deep_models._prepare_deep_data(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
    )
    n_samples = int(prepared["n_samples"])
    eval_idx = np.arange(n_samples - 4, n_samples, dtype=int)
    evaluation_split = {
        "train_idx": np.arange(0, n_samples - 4, dtype=int),
        "eval_idx": eval_idx,
        "evaluation_mode": "holdout",
        "evaluation_note": "test split",
    }
    seen = {"artifact_rows": None}

    def _fake_importance(model, x_tensor):
        seen["artifact_rows"] = int(x_tensor.shape[0])
        return [0.1] * int(x_tensor.shape[1])

    monkeypatch.setattr(deep_models, "_gradient_feature_importance", _fake_importance)

    result = deep_models.train_deepsurv(
        pd.DataFrame(),
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        random_seed=11,
        prepared_data=prepared,
        evaluation_split=evaluation_split,
        early_stopping_patience=None,
    )

    assert result["artifact_scope"] == "evaluation_subset"
    assert result["artifact_samples"] == len(eval_idx)
    assert seen["artifact_rows"] == len(eval_idx)
    assert len(result["risk_scores"]) == len(eval_idx)
    assert {row["patient_index"] for row in result["predicted_survival_function"]}.issubset(set(eval_idx.tolist()))


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_returns_ranked_table() -> None:
    from survival_toolkit.deep_models import compare_deep_survival_models

    df = make_example_dataset(seed=14, n_patients=60)
    result = compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        random_seed=21,
    )

    assert result["comparison_table"]
    assert result["scientific_summary"]["headline"]
    assert result["manuscript_tables"]["model_performance_table"]
    assert len(result["comparison_table"]) >= 3
    assert all("model" in row and "c_index" in row for row in result["comparison_table"])


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_normalizes_blank_model_labels(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=113, n_patients=60)

    def _stub(model_label: str):
        def _run(*args, **kwargs):
            return {
                "model": model_label,
                "c_index": 0.61,
                "apparent_c_index": 0.59,
                "holdout_c_index": 0.61,
                "evaluation_mode": "holdout",
                "epochs_trained": 1,
                "n_features": 3,
                "training_samples": 48,
                "evaluation_samples": 12,
                "training_time_ms": 1.0,
            }
        return _run

    monkeypatch.setattr(deep_models, "train_deepsurv", _stub(""))
    monkeypatch.setattr(deep_models, "train_deephit", _stub("DeepHit"))
    monkeypatch.setattr(deep_models, "train_neural_mtlr", _stub("Neural MTLR"))
    monkeypatch.setattr(deep_models, "train_survival_transformer", _stub("Survival Transformer"))
    monkeypatch.setattr(deep_models, "train_survival_vae", _stub("Survival VAE"))

    result = deep_models.compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        random_seed=21,
    )

    models = {row["model"] for row in result["comparison_table"]}
    assert "DeepSurv" in models
    assert "" not in models


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_supports_repeated_cv() -> None:
    from survival_toolkit.deep_models import compare_deep_survival_models

    df = make_example_dataset(seed=15, n_patients=60)
    result = compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        evaluation_strategy="repeated_cv",
        cv_folds=2,
        cv_repeats=2,
        random_seed=22,
    )

    assert result["evaluation_mode"] == "repeated_cv"
    assert result["cv_folds"] == 2
    assert result["cv_repeats"] == 2
    assert result["fold_results"]
    assert result["comparison_table"]
    assert all(row["n_repeats"] == 2 for row in result["comparison_table"])
    assert result["manuscript_tables"]["model_performance_table"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_marks_incomplete_repeated_cv_rows(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=17, n_patients=60)
    calls = {"deepsurv": 0}

    def _succeeding_trainer(*args, **kwargs):
        return {
            "c_index": 0.62,
            "evaluation_mode": "holdout",
            "epochs_trained": 1,
            "n_features": 3,
            "training_samples": 20,
            "evaluation_samples": 10,
        }

    def _flaky_deepsurv(*args, **kwargs):
        calls["deepsurv"] += 1
        if calls["deepsurv"] == 1:
            raise RuntimeError("simulated fold failure")
        return _succeeding_trainer(*args, **kwargs)

    monkeypatch.setattr(deep_models, "train_deepsurv", _flaky_deepsurv)
    monkeypatch.setattr(deep_models, "train_deephit", _succeeding_trainer)
    monkeypatch.setattr(deep_models, "train_neural_mtlr", _succeeding_trainer)
    monkeypatch.setattr(deep_models, "train_survival_transformer", _succeeding_trainer)
    monkeypatch.setattr(deep_models, "train_survival_vae", _succeeding_trainer)

    result = deep_models.compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        evaluation_strategy="repeated_cv",
        cv_folds=2,
        cv_repeats=1,
        random_seed=22,
    )

    row = next(row for row in result["comparison_table"] if row["model"] == "DeepSurv")
    assert row["n_failures"] == 1
    assert row["c_index"] is None
    assert row["evaluation_mode"] == "repeated_cv_incomplete"
    assert result["evaluation_mode"] == "repeated_cv_incomplete"
    assert "incomplete" in result["manuscript_tables"]["caption"].lower()


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_reports_mixed_holdout_apparent_caption(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    df = make_example_dataset(seed=24, n_patients=60)
    call_index = {"count": 0}

    def _fake_trainer(*args, **kwargs):
        call_index["count"] += 1
        mode = "holdout" if call_index["count"] == 1 else "holdout_fallback_apparent"
        return {
            "c_index": 0.63 if mode == "holdout" else 0.58,
            "apparent_c_index": 0.63,
            "holdout_c_index": 0.63 if mode == "holdout" else None,
            "evaluation_mode": mode,
            "epochs_trained": 1,
            "n_features": 3,
            "training_samples": 40,
            "evaluation_samples": 20,
        }

    monkeypatch.setattr(deep_models, "train_deepsurv", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_deephit", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_neural_mtlr", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_survival_transformer", _fake_trainer)
    monkeypatch.setattr(deep_models, "train_survival_vae", _fake_trainer)

    result = deep_models.compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        random_seed=21,
    )

    assert result["evaluation_mode"] == "mixed_holdout_apparent"
    assert "mixed holdout/apparent" in result["manuscript_tables"]["caption"].lower()
    assert result["comparison_table"][0]["evaluation_mode"] == "holdout"
    assert result["comparison_table"][0]["rank"] == 1
    assert result["comparison_table"][0]["comparable_for_ranking"] is True
    fallback_rows = [row for row in result["comparison_table"] if row["evaluation_mode"] != "holdout"]
    assert fallback_rows
    assert all(row["rank"] is None for row in fallback_rows)
    assert all(row["comparable_for_ranking"] is False for row in fallback_rows)


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_compare_deep_survival_models_supports_parallel_repeated_cv() -> None:
    from survival_toolkit.deep_models import compare_deep_survival_models

    df = make_example_dataset(seed=16, n_patients=48)
    result = compare_deep_survival_models(
        df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        evaluation_strategy="repeated_cv",
        cv_folds=2,
        cv_repeats=1,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        parallel_jobs=2,
        random_seed=23,
    )

    assert result["evaluation_mode"] == "repeated_cv"
    assert result["comparison_table"]
    assert result["fold_results"]
    assert result["manuscript_tables"]["model_performance_table"]
    first_row = result["comparison_table"][0]
    assert first_row["training_seeds"]
    assert first_row["split_seeds"]
    assert first_row["monitor_seeds"]
    assert "Training seeds" in result["manuscript_tables"]["model_performance_table"][0]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_parallel_repeated_cv_is_reproducible_with_fixed_seed() -> None:
    from survival_toolkit.deep_models import compare_deep_survival_models

    df = make_example_dataset(seed=18, n_patients=48)
    common = dict(
        df=df,
        time_column="os_months",
        event_column="os_event",
        features=["age", "biomarker_score", "immune_index"],
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        num_time_bins=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        latent_dim=4,
        n_clusters=3,
        evaluation_strategy="repeated_cv",
        cv_folds=2,
        cv_repeats=1,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        parallel_jobs=2,
        random_seed=29,
    )

    first = compare_deep_survival_models(**common)
    second = compare_deep_survival_models(**common)

    first_rows = [(row["model"], row["c_index"]) for row in first["comparison_table"]]
    second_rows = [(row["model"], row["c_index"]) for row in second["comparison_table"]]
    assert first_rows == second_rows


def test_evaluate_single_deep_survival_model_repeated_cv_reuses_compare_path(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    def _fake_compare(*args, **kwargs):
        assert kwargs["included_models"] == ["Neural MTLR"]
        assert kwargs["evaluation_strategy"] == "repeated_cv"
        return {
            "comparison_table": [{
                "model": "Neural MTLR",
                "c_index": 0.713,
                "evaluation_mode": "repeated_cv",
                "n_features": 4,
                "epochs_trained": 8,
                "training_time_ms": 123.4,
                "training_seed": None,
                "split_seed": None,
                "monitor_seed": None,
                "training_seeds": [42, 43],
                "split_seeds": [42, 43],
                "monitor_seeds": [42, 43],
                "n_evaluations": 4,
                "n_failures": 0,
                "repeat_results": [{"repeat": 1, "c_index": 0.71}, {"repeat": 2, "c_index": 0.716}],
            }],
            "fold_results": [
                {"model": "Neural MTLR", "repeat": 1, "fold": 1, "c_index": 0.71},
                {"model": "DeepSurv", "repeat": 1, "fold": 1, "c_index": 0.69},
            ],
            "manuscript_tables": {"model_performance_table": [{"Model": "Neural MTLR"}]},
            "scientific_summary": {
                "status": "review",
                "strengths": ["summary"],
                "cautions": [],
                "next_steps": ["Use the ranking to narrow candidates."],
            },
        }

    monkeypatch.setattr(deep_models, "compare_deep_survival_models", _fake_compare)

    result = deep_models.evaluate_single_deep_survival_model(
        "mtlr",
        df=pd.DataFrame(),
        time_column="os_months",
        event_column="os_event",
        features=["age", "stage"],
        evaluation_strategy="repeated_cv",
        cv_folds=2,
        cv_repeats=2,
    )

    assert result["evaluation_mode"] == "repeated_cv"
    assert result["model"] == "Neural MTLR"
    assert result["comparison_table"][0]["model"] == "Neural MTLR"
    assert result["training_seeds"] == [42, 43]
    assert result["fold_results"] == [{"model": "Neural MTLR", "repeat": 1, "fold": 1, "c_index": 0.71}]
    assert "aggregate repeated-cv estimate" in result["scientific_summary"]["cautions"][-1].lower()
    assert all("ranking" not in step.lower() for step in result["scientific_summary"]["next_steps"])


def test_evaluate_single_deep_survival_model_repeated_cv_preserves_incomplete_state(monkeypatch) -> None:
    import survival_toolkit.deep_models as deep_models

    monkeypatch.setattr(
        deep_models,
        "compare_deep_survival_models",
        lambda *args, **kwargs: {
            "evaluation_mode": "repeated_cv_incomplete",
            "comparison_table": [{
                "model": "Neural MTLR",
                "c_index": 0.611,
                "evaluation_mode": "repeated_cv_incomplete",
                "n_features": 4,
                "epochs_trained": 8,
                "training_time_ms": 123.4,
                "training_seed": None,
                "split_seed": None,
                "monitor_seed": None,
                "training_seeds": [42, 43],
                "split_seeds": [42, 43],
                "monitor_seeds": [42, 43],
                "n_evaluations": 3,
                "n_failures": 1,
                "repeat_results": [{"repeat": 1, "c_index": 0.61}],
            }],
            "fold_results": [],
            "manuscript_tables": {"model_performance_table": [{"Model": "Neural MTLR"}]},
            "scientific_summary": {"status": "review", "strengths": [], "cautions": ["fallbacks"], "next_steps": []},
        },
    )

    result = deep_models.evaluate_single_deep_survival_model(
        "mtlr",
        df=pd.DataFrame(),
        time_column="os_months",
        event_column="os_event",
        features=["age", "stage"],
        evaluation_strategy="repeated_cv",
        cv_folds=2,
        cv_repeats=2,
    )

    assert result["evaluation_mode"] == "repeated_cv_incomplete"
    evaluation_metric = next(metric for metric in result["scientific_summary"]["metrics"] if metric["label"] == "Evaluation mode")
    assert evaluation_metric["value"] == "repeated_cv_incomplete"


def test_scientific_summary_dl_preserves_fallback_metric_name() -> None:
    from survival_toolkit.deep_models import _scientific_summary_dl

    summary = _scientific_summary_dl(
        "DeepSurv",
        c_index=0.58,
        train_samples=100,
        eval_samples=20,
        train_events=42,
        n_features=5,
        epochs=10,
        loss_history=[1.2, 1.0, 0.9],
        evaluation_mode="holdout_fallback_apparent",
    )

    assert summary["metrics"][0]["label"] == "Apparent fallback C-index"
    assert "apparent fallback c-index" in summary["headline"].lower()


def test_scientific_summary_dl_flags_transformer_as_exploratory_tabular_attention() -> None:
    from survival_toolkit.deep_models import _scientific_summary_dl

    summary = _scientific_summary_dl(
        "Survival Transformer",
        c_index=0.61,
        train_samples=180,
        eval_samples=40,
        train_events=70,
        n_features=24,
        epochs=12,
        loss_history=[1.4, 1.0, 0.8],
        evaluation_mode="holdout",
    )

    cautions_text = " ".join(summary["cautions"]).lower()
    assert "full-batch cox optimization" in cautions_text
    assert "feature-identity embedding" in cautions_text


def test_deep_models_module_can_load_when_torch_is_missing(monkeypatch) -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "deep_models.py"
    source = module_path.read_text(encoding="utf-8")
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch intentionally unavailable for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    module_globals: dict[str, object] = {
        "__name__": "deep_models_no_torch_test",
        "__file__": str(module_path),
    }
    exec(compile(source, str(module_path), "exec"), module_globals, module_globals)

    assert module_globals["TORCH_AVAILABLE"] is False
    assert module_globals["DeepSurvNet"]
    assert module_globals["SurvivalTransformerNet"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_require_finite_loss_rejects_nan_before_optimizer_step() -> None:
    import torch
    import survival_toolkit.deep_models as deep_models

    with pytest.raises(ValueError, match="NaN or Inf"):
        deep_models._require_finite_loss(torch.tensor(float("nan")), context="unit test loss")


def test_deephit_feature_importance_uses_expected_time_risk_target() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "deep_models.py"
    ).read_text(encoding="utf-8")

    assert "output_to_score=lambda pmf: _expected_time_risk(pmf, time_grid)" in source

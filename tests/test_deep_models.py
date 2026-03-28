from __future__ import annotations

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
    assert len({item[2] for item in seen}) == 5


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
    original_loss = deep_models._cox_partial_likelihood_loss

    def _recording_loss(risk_scores, times, events):
        seen_times.append(tuple(float(v) for v in times.detach().cpu().numpy().ravel().tolist()))
        return original_loss(risk_scores, times, events)

    monkeypatch.setattr(deep_models, "_cox_partial_likelihood_loss", _recording_loss)

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

    monitor_times = tuple(float(v) for v in prepared["time_tensor"][torch.as_tensor(monitor_indices)].detach().cpu().numpy().tolist())
    eval_times = tuple(float(v) for v in prepared["time_tensor"][torch.as_tensor(evaluation_split["eval_idx"])].detach().cpu().numpy().tolist())
    assert monitor_times in seen_times
    assert eval_times not in seen_times


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
            "evaluation_mode": "repeated_cv",
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

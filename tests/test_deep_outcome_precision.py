"""Original outcome precision in final evaluation, independent of float32 training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import survival_toolkit.deep_models as dm
from survival_toolkit.analysis import _harrell_c_index
from survival_toolkit.errors import UserInputError
from survival_toolkit.evaluation import c_index_intervals, merge_prediction_blocks, prediction_block
from survival_toolkit.sample_data import make_example_dataset


def _frame(n: int = 84) -> pd.DataFrame:
    rng = np.random.default_rng(125)
    return pd.DataFrame({
        "time": 2.**24 + rng.permutation(np.arange(n)) / 4.,
        "event": np.resize([1, 0, 1, 1], n),
        "x": rng.normal(size=n),
    }, index=np.arange(1000, 1000+n))


def test_split_preparation_keeps_original_times_aligned_and_independent() -> None:
    frame = _frame(42).reset_index(drop=True)
    expected = frame.time.to_numpy(copy=True)
    data, split = dm._prepare_deep_split_data(frame.iloc[:30], frame.iloc[30:],
        time_column="time", event_column="event", features=["x"])
    np.testing.assert_array_equal(data["outcome_time_values"], expected)
    assert data["outcome_time_values"].dtype == np.float64
    assert data["time_tensor"].dtype == torch.float32
    np.testing.assert_array_equal(data["time_tensor"].numpy(), expected.astype(np.float32))
    assert np.any(expected != data["time_tensor"].numpy().astype(float))
    frame.loc[:, "time"] = 0.
    np.testing.assert_array_equal(data["outcome_time_values"], expected)
    split["eval_row_ids"] = list(range(30,42))
    block = dm._deep_prediction_block(data, split, {"DL": (None, list(range(12)))})
    np.testing.assert_array_equal(block["time"], expected[30:])


def test_holdout_export_uses_original_source_rows_and_time_values() -> None:
    frame = _frame()
    data, split = dm._prepare_deep_training_inputs(frame,
        time_column="time", event_column="event", features=["x"], random_seed=23)
    n = len(split["eval_idx"])
    block = dm._deep_prediction_block(data, split, {"DL": (None, list(range(n)))})
    expected = frame.loc[[int(row) for row in block["row_ids"]], "time"].to_numpy()
    np.testing.assert_array_equal(block["time"], expected)
    assert data["time_tensor"].dtype == torch.float32


def test_final_c_index_and_export_do_not_collapse_close_event_times() -> None:
    times = np.array([16777216., 16777217., 16777218.])
    events = np.array([1., 1., 0.])
    risk = torch.tensor([2., 3., 1.])
    rows = torch.arange(3)
    data = {"time_tensor": torch.tensor(times, dtype=torch.float32),
            "event_tensor": torch.tensor(events, dtype=torch.float32),
            "outcome_time_values": times.copy()}
    context = dm._DeepTrainingContext(data, rows, rows, rows, None, "holdout", "test")
    result = dm._holdout_and_apparent_c_index(context, risk)
    assert result[:3] == pytest.approx([2/3, 2/3, 2/3])
    assert dm._compute_c_index_torch(risk, data["time_tensor"], data["event_tensor"]) == 1.
    block = dm._deep_prediction_block(data,
        {"eval_row_ids": [0,1,2], "eval_idx": rows.numpy(), "evaluation_mode": "holdout"},
        {"DL": (None, risk.tolist())})
    np.testing.assert_array_equal(block["time"], times)
    assert c_index_intervals(block["time"], block["event"], block["risk"], n_bootstrap=100)["rows"][0]["c_index"] == 2/3


@pytest.mark.parametrize("events,expected", [([1,1], True), ([0,1], False)])
def test_comparable_fold_check_uses_original_order_not_float32_ties(events, expected) -> None:
    assert dm._has_comparable_pair([16777216.,16777217.], events) is expected


def test_undefined_c_returns_none_even_if_optional_metric_returns_nan(monkeypatch) -> None:
    monkeypatch.setattr(dm, "_SKSURV_METRICS_AVAILABLE", True)
    monkeypatch.setattr(dm, "_sksurv_concordance", lambda *args, **kwargs: (float("nan"),))
    assert dm._compute_evaluation_c_index(torch.tensor([0.,1.]), torch.tensor([1.,1.]), torch.tensor([1.,1.])) is None


def test_undefined_training_monitor_keeps_legacy_patience_behavior(monkeypatch) -> None:
    monkeypatch.setattr(dm, "_SKSURV_METRICS_AVAILABLE", True)
    monkeypatch.setattr(dm, "_sksurv_concordance", lambda *args, **kwargs: (float("nan"),))
    model = torch.nn.Linear(1, 1)
    monitor = lambda: dm._monitor_c_index(model, torch.tensor([[0.],[1.]]),
        torch.tensor([1.,1.]), torch.tensor([1.,1.]), torch.arange(2))
    losses, values, used = dm._train_epochs(model, epochs=3, step=lambda: 0.,
        monitor=monitor, monitor_goal="max", patience=1, min_delta=0.)
    assert losses == [0.] and used and len(values) == 1 and np.isnan(values[0])


@pytest.mark.parametrize("values", [[1.], [1.,float("nan")], [-1.,2.]])
def test_malformed_original_outcome_array_is_not_silently_used(values) -> None:
    with pytest.raises(ValueError, match="aligned with the training tensors"):
        dm._evaluation_time_values({"time_tensor":torch.tensor([1.,2.]),"outcome_time_values":np.array(values)})


@pytest.mark.parametrize("altered", [
    [16777216.,16777216.,16777218.],
    [16777217.,16777216.,16777218.],
])
@pytest.mark.parametrize("reverse", [False,True])
def test_merge_rejects_close_but_order_or_tie_altering_times(altered, reverse) -> None:
    original = prediction_block([0,1,2], [16777216.,16777217.,16777218.], [1,1,0], {"A":(None,[2.,3.,1.])})
    other = prediction_block([0,1,2], altered, [1,1,0], {"B":(None,[2.,3.,1.])})
    assert np.allclose(original["time"], other["time"])
    blocks = [original,other] if not reverse else [other,original]
    with pytest.raises(UserInputError, match="ordering or exact ties"):
        merge_prediction_blocks(blocks)


def test_corrected_prediction_blocks_are_invariant_to_family_order() -> None:
    times = [16777216.,16777217.,16777218.]
    first = prediction_block([0,1,2], times, [1,1,0], {"A":(None,[2.,3.,1.])})
    second = prediction_block([2,0,1], [times[2],times[0],times[1]], [0,1,1], {"B":(None,[1.,2.,3.])})
    outputs=[]
    for blocks in [[first,second],[second,first]]:
        t,e,risks,rows = merge_prediction_blocks(blocks)
        result=c_index_intervals(t,e,risks,reference="A",n_bootstrap=100,random_seed=7)
        outputs.append({row["model"]:row for row in result["rows"]})
    assert outputs[0]==outputs[1]
    assert outputs[0]["A"]["c_index"]==outputs[0]["B"]["c_index"]==2/3


def test_merge_keeps_harmless_roundoff_compatibility_without_relaxing_ties() -> None:
    first=prediction_block([0,1,2],[1.,2.,3.],[1,0,1],{"A":(None,[3.,2.,1.])})
    other=prediction_block([0,1,2],[1.+1e-9,2.,3.],[1,0,1],{"B":(None,[3.,2.,1.])})
    assert merge_prediction_blocks([first,other])[0].tolist()==[1.,2.,3.]


@pytest.mark.parametrize("strategy", ["holdout","repeated_cv"])
def test_reported_c_and_holdout_or_locked_export_use_same_original_outcomes(strategy) -> None:
    frame=make_example_dataset(seed=39,n_patients=100)
    frame.index=np.arange(2000,2100)
    settings={"evaluation_strategy":strategy}
    if strategy=="repeated_cv":settings.update(cv_folds=2,cv_repeats=1,locked_test_fraction=.2)
    result=dm.compare_deep_survival_models(frame,"os_months","os_event",["age","biomarker_score"],
        included_models=["DeepSurv"],hidden_layers=[4],epochs=1,batch_size=16,random_seed=23,**settings)
    block=result["test_predictions" if strategy=="holdout" else "locked_test_predictions"]
    selected=frame.loc[[int(row) for row in block["row_ids"]]]
    np.testing.assert_array_equal(block["time"],selected.os_months.to_numpy(float))
    np.testing.assert_array_equal(block["event"],selected.os_event.to_numpy(int))
    row=result["comparison_table"][0]
    expected=_harrell_c_index(block["time"],block["event"],block["risk"]["DeepSurv"])
    assert row["c_index" if strategy=="holdout" else "locked_test_c_index"]==expected


@pytest.mark.parametrize("name,settings", [
    ("train_deepsurv",{"hidden_layers":[4]}),
    ("train_deephit",{"hidden_layers":[4],"num_time_bins":6}),
    ("train_neural_mtlr",{"hidden_layers":[4],"num_time_bins":6}),
    ("train_survival_transformer",{"d_model":16,"n_heads":2,"n_layers":1}),
    ("train_survival_vae",{"hidden_layers":[4],"latent_dim":2,"n_clusters":2}),
])
def test_original_outcome_sidecar_does_not_change_fixed_seed_training(name, settings, monkeypatch) -> None:
    frame=make_example_dataset(seed=47,n_patients=80)
    features=["age","biomarker_score"]
    data,split=dm._prepare_deep_training_inputs(frame,time_column="os_months",event_column="os_event",features=features,random_seed=23)
    assert np.any(data["outcome_time_values"]!=data["time_tensor"].numpy().astype(float))
    legacy={key:value for key,value in data.items() if key!="outcome_time_values"}
    snapshots=[];original=dm._fit_with_refit
    def capture(*args,**kwargs):
        assert args[1].t_all.dtype==torch.float32
        result=original(*args,**kwargs)
        snapshots.append({key:value.detach().cpu().clone() for key,value in result[0].model.state_dict().items()})
        return result
    monkeypatch.setattr(dm,"_fit_with_refit",capture)
    trainer=getattr(dm,name)
    common=dict(time_column="os_months",event_column="os_event",features=features,evaluation_split=split,
        epochs=3,batch_size=16,random_seed=23,early_stopping_patience=1,**settings)
    raw=trainer(None,prepared_data=data,**common)
    old_precision=trainer(None,prepared_data=legacy,**common)
    assert len(snapshots)==2 and snapshots[0].keys()==snapshots[1].keys()
    assert all(torch.equal(snapshots[0][key],snapshots[1][key]) for key in snapshots[0])
    for key in ["loss_history","monitor_history","refit_loss_history","best_monitor_epoch","epochs_trained","refit_epochs","stopped_early","holdout_risk"]:
        assert raw[key]==old_precision[key],key
    for key in ["risk_scores","predicted_survival_curves","predicted_survival_function","time_bins","time_bin_edges","calibration_data"]:
        if key in raw:
            assert raw[key]==old_precision[key],key
    assert raw["monitor_history"]

"""Regression tests for the 2026-09-28 review of the deep-learning module (findings E1-E17).

The ``test_real_*``/crash/cancel tests start a spawn-context worker pool, so they exercise
the code that runs in worker processes rather than an in-process stand-in.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import threading
import time

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.sample_data import make_example_dataset

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import survival_toolkit.deep_models as dm  # noqa: E402
from survival_toolkit.concurrency import cancellation_scope  # noqa: E402
from survival_toolkit.errors import InternalAnalysisError, JobCancelledError, UserInputError  # noqa: E402
from survival_toolkit.evaluation import locked_test_split, stratified_holdout_indices  # noqa: E402

FEATURES = ["age", "biomarker_score", "immune_index"]
TINY = {"hidden_layers": [4], "epochs": 2, "batch_size": 16}


def _grade_frame(n: int = 120) -> pd.DataFrame:
    """Numeric-looking text levels plus one stray text level in row 0."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "time": rng.exponential(20, n).round(1) + 0.1,
        "event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
        "grade": rng.choice(["1", "2", "3"], n).astype(object),
    })
    frame.loc[0, "grade"] = "unknown"
    return frame


def _holdout_seed_with_row_in_eval(events: np.ndarray, row: int = 0) -> int:
    for seed in range(200):
        _train, eval_rows, mode = stratified_holdout_indices(events, random_state=seed)
        if mode == "holdout" and row in set(np.asarray(eval_rows).tolist()):
            return seed
    raise AssertionError("no seed puts the row in the evaluation split")


def _force_parallel(monkeypatch: pytest.MonkeyPatch, workers: int = 2) -> None:
    monkeypatch.setattr(dm, "_available_system_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(dm, "_available_cpu_count", lambda: workers)


class _WorkerKiller:
    """Unpickling this object in a worker ends the process abruptly (like an OOM kill)."""

    def __reduce__(self):
        return (os._exit, (137,))


class _RecordingExecutor:
    """In-process executor whose futures and submits can be made to fail like a broken pool."""

    def __init__(self, *args, fail_results=(), fail_submit_after=None, **kwargs) -> None:
        self.max_workers = kwargs.get("max_workers")
        self._fail_results = set(fail_results)
        self._fail_submit_after = fail_submit_after
        self.submitted: list[tuple[int, int]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, task):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        key = (task["repeat"], task["fold"])
        if self._fail_submit_after is not None and len(self.submitted) >= self._fail_submit_after:
            raise BrokenProcessPool("A child process terminated abruptly")
        self.submitted.append(key)
        future: Future = Future()
        if key in self._fail_results:
            future.set_exception(BrokenProcessPool("A process in the process pool was terminated abruptly"))
        else:
            future.set_result(fn(task))
        return future


# E1 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name, build, architecture",
    [
        ("DeepSurv", lambda n: dm.DeepSurvNet(n, [16, 8], 0.1), {"hidden_layers": [16, 8]}),
        ("DeepSurv", lambda n: dm.DeepSurvNet(n, [], 0.1), {"hidden_layers": []}),
        ("DeepHit", lambda n: dm.DeepHitNet(n, [12], 7, 0.1), {"hidden_layers": [12], "num_time_bins": 7}),
        ("Neural MTLR", lambda n: dm.NeuralMTLRNet(n, [9, 5], 11, 0.1), {"hidden_layers": [9, 5], "num_time_bins": 11}),
        (
            "Survival Transformer",
            lambda n: dm.SurvivalTransformerNet(n, d_model=16, n_heads=4, n_layers=3),
            {"d_model": 16, "n_layers": 3},
        ),
        (
            "Survival VAE",
            lambda n: dm.SurvivalVAENet(n, hidden_layers=[10, 6], latent_dim=3),
            {"hidden_layers": [10, 6], "latent_dim": 3},
        ),
    ],
)
def test_parameter_count_estimate_matches_the_built_network(model_name, build, architecture) -> None:
    n_features = 7
    built = sum(parameter.numel() for parameter in build(n_features).parameters())
    assert dm._estimate_deep_parameter_count(model_name, n_features=n_features, **architecture) == built


def _refuse_construction(monkeypatch: pytest.MonkeyPatch, *network_names: str) -> None:
    """Make building these networks fail loudly, so a missing guard cannot allocate gigabytes."""

    def _must_not_build(*args, **kwargs):
        raise AssertionError("the oversized network must be refused before it is built")

    for name in network_names:
        monkeypatch.setattr(dm, name, _must_not_build)


def test_trainers_refuse_oversized_networks_before_building_them(monkeypatch) -> None:
    _refuse_construction(monkeypatch, "DeepSurvNet", "SurvivalTransformerNet")
    df = make_example_dataset(seed=4, n_patients=80)
    with pytest.raises(UserInputError, match="trainable parameters"):
        dm.train_deepsurv(df, "os_months", "os_event", FEATURES, hidden_layers=[2_000_000_000, 2_000_000_000], epochs=1)
    with pytest.raises(UserInputError, match="trainable parameters"):
        dm.train_survival_transformer(df, "os_months", "os_event", FEATURES, d_model=4096, n_heads=4, n_layers=2, epochs=1)
    with pytest.raises(UserInputError, match="positive integers"):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, hidden_layers=[64, 0], epochs=1)


def test_comparison_reports_the_parameter_budget_as_a_model_failure(monkeypatch) -> None:
    _refuse_construction(monkeypatch, "DeepSurvNet")
    df = make_example_dataset(seed=4, n_patients=80)
    result = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, hidden_layers=[40_000] * 20, epochs=1,
        included_models=["DeepSurv", "Survival Transformer"], d_model=16, n_heads=4, n_layers=1,
    )
    assert [row["model"] for row in result["comparison_table"]] == ["Survival Transformer"]
    assert "trainable parameters" in result["errors"][0]["error"]


# E2 -------------------------------------------------------------------------


def test_split_passes_keep_the_feature_types_of_the_full_cohort() -> None:
    from sklearn.model_selection import StratifiedKFold

    df = _grade_frame()
    seed = _holdout_seed_with_row_in_eval(df["event"].to_numpy())
    data, split = dm._prepare_deep_training_inputs(
        df, time_column="time", event_column="event", features=["age", "grade"], random_seed=seed
    )
    assert split["evaluation_mode"] == "holdout"
    # Row 0 ("unknown") is only in the evaluation split; "grade" stays one-hot, as for the ML models.
    assert data["feature_names"] == ["grade_2", "grade_3", "age"]

    clean = dm._coerce_deep_frame(df, "time", "event", ["age", "grade"])
    events = clean["event"].astype(int).to_numpy()
    for repeat in range(3):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat)
        for train_rows, eval_rows in splitter.split(clean, events):
            fold_data, _ = dm._prepare_deep_split_data(
                clean.iloc[train_rows].reset_index(drop=True),
                clean.iloc[eval_rows].reset_index(drop=True),
                time_column="time",
                event_column="event",
                features=["age", "grade"],
            )
            assert "grade" not in fold_data["feature_names"]
            assert {"grade_2", "grade_3"} <= set(fold_data["feature_names"])


# E3 -------------------------------------------------------------------------


def test_deep_models_apply_the_ml_outcome_checks() -> None:
    df = make_example_dataset(seed=3, n_patients=150)
    with pytest.raises(UserInputError, match="different survival endpoints"):
        dm.compare_deep_survival_models(df, "os_months", "pfs_event", FEATURES, epochs=1, included_models=["DeepSurv"])
    dated = df.assign(diagnosis_date=pd.Timestamp("2015-01-01") + pd.to_timedelta(np.arange(len(df)) * 7, unit="D"))
    with pytest.raises(UserInputError, match="calendar dates"):
        dm.train_deepsurv(dated, "diagnosis_date", "os_event", FEATURES, **TINY)
    with pytest.raises(UserInputError, match="calendar dates"):
        dm.compare_deep_survival_models(
            dated, "diagnosis_date", "os_event", FEATURES, epochs=1, included_models=["DeepSurv"],
            evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1,
        )


# E4 -------------------------------------------------------------------------


def test_cgroup_memory_limit_caps_the_available_memory(monkeypatch, tmp_path) -> None:
    limit, usage = tmp_path / "memory.max", tmp_path / "memory.current"
    monkeypatch.setattr(dm, "_CGROUP_MEMORY_FILES", ((limit, usage),))
    monkeypatch.setattr(dm, "_proc_meminfo_available_bytes", lambda: 64 * 1024**3)

    limit.write_text("2147483648\n", encoding="ascii")
    usage.write_text("536870912\n", encoding="ascii")
    assert dm._available_system_memory_bytes() == 2147483648 - 536870912

    limit.write_text("max\n", encoding="ascii")
    assert dm._available_system_memory_bytes() == 64 * 1024**3

    limit.write_text("9223372036854771712\n", encoding="ascii")  # cgroup v1 "unlimited"
    assert dm._available_system_memory_bytes() == 64 * 1024**3


def test_parallel_workers_are_capped_by_the_cpu_count(monkeypatch) -> None:
    created: list[_RecordingExecutor] = []

    def _executor(*args, **kwargs):
        created.append(_RecordingExecutor(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(dm, "ProcessPoolExecutor", _executor)
    monkeypatch.setattr(dm, "_available_system_memory_bytes", lambda: 64 * 1024**3)
    df = make_example_dataset(seed=16, n_patients=60)
    common = dict(
        epochs=1, hidden_layers=[4], included_models=["DeepSurv"], evaluation_strategy="repeated_cv",
        cv_folds=3, cv_repeats=2, parallel_jobs=8, random_seed=3,
    )

    monkeypatch.setattr(dm, "_available_cpu_count", lambda: 2)
    result = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    assert created[-1].max_workers == 2
    assert "parallel_execution_note" not in result

    monkeypatch.setattr(dm, "_available_cpu_count", lambda: 1)
    result = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    assert len(created) == 1
    assert "only 1 CPU" in result["parallel_execution_note"]
    assert len(result["fold_results"]) == 6


def test_parallel_memory_guard_counts_model_training_memory(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    rng = np.random.default_rng(0)
    n, p = 1000, 100
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    df["time"] = rng.exponential(10, n) + 0.1
    df["event"] = rng.integers(0, 2, n)
    features = [f"x{i}" for i in range(p)]

    class _UnexpectedExecutor:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("the workers would not fit next to the transformer's training memory")

    def _fake_fold_runner(task):
        row = {
            "model": "Survival Transformer", "repeat": task["repeat"], "fold": task["fold"], "c_index": 0.6,
            "evaluation_mode": "holdout", "training_seed": task["seed_base"], "split_seed": task["split_seed"],
            "monitor_seed": task["monitor_seed"], "epochs_trained": 1, "training_samples": 500,
            "evaluation_samples": 500, "n_features": p, "training_time_ms": 1.0,
        }
        return {"fold_results": [row], "errors": []}

    monkeypatch.setattr(dm, "ProcessPoolExecutor", _UnexpectedExecutor)
    monkeypatch.setattr(dm, "_run_deep_compare_fold_task", _fake_fold_runner)
    monkeypatch.setattr(dm, "_estimate_deep_compare_task_bytes", lambda task: 1024)
    monkeypatch.setattr(dm, "_available_cpu_count", lambda: 2)
    # Enough for two idle workers (about 1.5 GiB), not for two transformer fits on 500 x 100 tokens.
    monkeypatch.setattr(dm, "_available_system_memory_bytes", lambda: 3 * 1024**3)
    monkeypatch.setattr(ml_models, "build_manuscript_result_tables", lambda result: {})

    result = dm.compare_deep_survival_models(
        df, "time", "event", features, epochs=1, included_models=["Survival Transformer"],
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, parallel_jobs=2,
    )
    assert "model training" in result["parallel_execution_note"]
    task = {
        "prepared_data": {"n_features": p, "n_samples": 1000},
        "evaluation_split": {"train_idx": np.arange(500)},
        "model_specs": [{"model_name": "Survival Transformer", "extra_kwargs": {"d_model": 64, "n_heads": 4, "n_layers": 2}}],
        "batch_size": 64,
    }
    attention = dm._estimate_transformer_attention_bytes(training_samples=500, n_features=p, n_heads=4, n_layers=2, d_model=64)
    assert dm._estimate_fold_task_training_bytes(task, []) >= attention


# E5 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail_results, fail_submit_after, n_unfinished",
    [
        ({(1, 1)}, None, 5),  # the first fold's worker dies while it runs
        (set(), 2, 4),  # the pool refuses the third fold after it was built (and popped)
    ],
)
def test_broken_worker_pool_reruns_every_unfinished_fold_once(
    monkeypatch, fail_results, fail_submit_after, n_unfinished
) -> None:
    def _executor(*args, **kwargs):
        return _RecordingExecutor(*args, fail_results=fail_results, fail_submit_after=fail_submit_after, **kwargs)

    monkeypatch.setattr(dm, "ProcessPoolExecutor", _executor)
    _force_parallel(monkeypatch)
    df = make_example_dataset(seed=16, n_patients=60)
    result = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
        evaluation_strategy="repeated_cv", cv_folds=3, cv_repeats=2, parallel_jobs=2, random_seed=5,
    )
    assert result["errors"] == []
    folds = sorted((row["repeat"], row["fold"]) for row in result["fold_results"])
    assert folds == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert "stopped unexpectedly" in result["parallel_execution_note"]
    assert f"{n_unfinished} unfinished fold(s)" in result["parallel_execution_note"]


# E6 -------------------------------------------------------------------------


def test_trainers_reject_leaky_monitor_rows_and_splits() -> None:
    df = make_example_dataset(seed=4, n_patients=120)
    data, split = dm._prepare_deep_training_inputs(
        df, time_column="os_months", event_column="os_event", features=FEATURES, random_seed=1
    )
    common = dict(time_column="os_months", event_column="os_event", features=FEATURES, **TINY)
    with pytest.raises(UserInputError, match="monitor_indices"):
        dm.train_deepsurv(
            None, prepared_data=data, evaluation_split=split, monitor_indices=split["eval_idx"],
            early_stopping_patience=2, **common,
        )
    with pytest.raises(UserInputError, match="evaluation_split"):
        dm.train_deepsurv(None, prepared_data=data, **common)
    overlapping = {**split, "eval_idx": np.concatenate([split["eval_idx"], split["train_idx"][:3]])}
    with pytest.raises(UserInputError, match="both train_idx and eval_idx"):
        dm.train_deephit(None, prepared_data=data, evaluation_split=overlapping, num_time_bins=5, **common)
    outside = {**split, "eval_idx": np.append(split["eval_idx"], data["n_samples"])}
    with pytest.raises(UserInputError, match="outside"):
        dm.train_deepsurv(None, prepared_data=data, evaluation_split=outside, **common)
    everything = np.arange(int(data["n_samples"]))
    apparent = {"train_idx": everything, "eval_idx": everything, "evaluation_mode": "apparent", "evaluation_note": "Apparent."}
    result = dm.train_deepsurv(None, prepared_data=data, evaluation_split=apparent, **common)
    assert result["evaluation_mode"] == "apparent"


# E7 / E12 ---------------------------------------------------------------------


def test_training_pins_torch_threads_and_restores_them() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        df = make_example_dataset(seed=4, n_patients=80)
        result = dm.train_survival_vae(df, "os_months", "os_event", FEATURES, hidden_layers=[8], latent_dim=2, epochs=2)
        assert result["torch_num_threads"] == dm._DEEP_TORCH_NUM_THREADS == 1
        assert torch.get_num_threads() == 2
        compared = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, epochs=1, included_models=["DeepSurv"])
        assert compared["torch_num_threads"] == 1
    finally:
        torch.set_num_threads(previous)


def test_real_worker_processes_match_the_sequential_folds(monkeypatch) -> None:
    pools: list[object] = []

    class _TrackedPool(dm.ProcessPoolExecutor):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            pools.append(self)

    monkeypatch.setattr(dm, "ProcessPoolExecutor", _TrackedPool)
    _force_parallel(monkeypatch)
    df = make_example_dataset(seed=12, n_patients=90)
    common = dict(
        epochs=4, hidden_layers=[8], latent_dim=2, included_models=["DeepSurv", "Survival VAE"],
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, random_seed=17,
    )
    parallel = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, parallel_jobs=2, **common)
    sequential = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, parallel_jobs=1, **common)

    assert len(pools) == 1
    assert "parallel_execution_note" not in parallel

    def _folds(result):
        return sorted((row["model"], row["repeat"], row["fold"], row["c_index"]) for row in result["fold_results"])

    assert _folds(parallel) == _folds(sequential)
    assert len(_folds(parallel)) == 4


def test_a_crashed_worker_process_does_not_end_the_run(monkeypatch) -> None:
    original = dm._run_deep_fold_tasks

    def _poison_first_fold(fold_splits, build_task, **kwargs):
        poisoned: list[bool] = []

        def _build(split):
            task = build_task(split)
            if task is not None and not poisoned:
                task["worker_killer"] = _WorkerKiller()
                poisoned.append(True)
            return task

        return original(fold_splits, _build, **kwargs)

    _force_parallel(monkeypatch)
    df = make_example_dataset(seed=12, n_patients=90)
    common = dict(
        epochs=3, hidden_layers=[8], included_models=["DeepSurv"], evaluation_strategy="repeated_cv",
        cv_folds=3, cv_repeats=1, random_seed=21,
    )
    sequential = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, parallel_jobs=1, **common)
    monkeypatch.setattr(dm, "_run_deep_fold_tasks", _poison_first_fold)
    crashed = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, parallel_jobs=2, **common)

    assert crashed["errors"] == []
    assert "stopped unexpectedly" in crashed["parallel_execution_note"]
    folds = sorted((row["fold"], row["c_index"]) for row in crashed["fold_results"])
    assert folds == sorted((row["fold"], row["c_index"]) for row in sequential["fold_results"])


# E8 -------------------------------------------------------------------------


def test_cancelling_parallel_cv_terminates_the_running_workers(monkeypatch) -> None:
    _force_parallel(monkeypatch)
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"x{i}" for i in range(10)])
    df["time"] = rng.exponential(10, n) + 0.1
    df["event"] = rng.integers(0, 2, n)
    cancel = threading.Event()
    timer = threading.Timer(3.0, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(JobCancelledError):
            with cancellation_scope(cancel):
                # Each fold trains for minutes (bounded, so a regression fails instead of hanging).
                dm.compare_deep_survival_models(
                    df, "time", "event", [f"x{i}" for i in range(10)], hidden_layers=[64, 64], epochs=300_000,
                    early_stopping_patience=None, evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1,
                    parallel_jobs=2, included_models=["DeepSurv"], random_seed=3,
                )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 30
    deadline = time.monotonic() + 10
    while mp.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not mp.active_children()


# E9 -------------------------------------------------------------------------


def test_queued_training_stops_waiting_for_the_lock_when_cancelled() -> None:
    held, release = threading.Event(), threading.Event()

    def _holder() -> None:
        with dm._TORCH_TRAINING_LOCK:
            held.set()
            release.wait(20)

    holder = threading.Thread(target=_holder, daemon=True)
    holder.start()
    held.wait(5)
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    df = make_example_dataset(seed=4, n_patients=80)
    started = time.monotonic()
    try:
        with pytest.raises(JobCancelledError):
            with cancellation_scope(cancel):
                dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, epochs=1, included_models=["DeepSurv"])
        assert time.monotonic() - started < 5
    finally:
        timer.cancel()
        release.set()
        holder.join(5)


# E10 ------------------------------------------------------------------------


def test_deep_comparisons_caution_about_unseen_category_levels() -> None:
    df = _grade_frame()
    features = ["age", "grade"]
    common = dict(epochs=1, hidden_layers=[4], included_models=["DeepSurv"])
    seed = _holdout_seed_with_row_in_eval(df["event"].to_numpy())
    holdout = dm.compare_deep_survival_models(df, "time", "event", features, random_seed=seed, **common)
    assert any(
        caution.startswith("1 evaluation row(s) had a categorical level that never occurs")
        for caution in holdout["scientific_summary"]["cautions"]
    )

    events = df["event"].to_numpy()
    locked_seed = next(
        seed for seed in range(200) if 0 in set(locked_test_split(events, random_state=seed, test_fraction=0.25)[1].tolist())
    )
    cv = dm.compare_deep_survival_models(
        df, "time", "event", features, evaluation_strategy="repeated_cv", cv_folds=3, cv_repeats=1,
        random_seed=locked_seed, locked_test_fraction=0.25, **common,
    )
    assert any(caution.startswith("1 locked-test row(s)") for caution in cv["scientific_summary"]["cautions"])

    cv_only = dm.compare_deep_survival_models(
        df, "time", "event", features, evaluation_strategy="repeated_cv", cv_folds=3, cv_repeats=2, random_seed=1, **common,
    )
    assert any(
        caution.startswith("2 cross-validation evaluation (summed over folds) row(s)")
        for caution in cv_only["scientific_summary"]["cautions"]
    )
    single = dm.train_deepsurv(df, "time", "event", features, random_seed=seed, **TINY)
    assert any("never occurs" in caution for caution in single["scientific_summary"]["cautions"])


# E11 ------------------------------------------------------------------------


def test_comparison_rejects_settings_it_would_ignore() -> None:
    df = make_example_dataset(seed=4, n_patients=80)
    with pytest.raises(UserInputError, match="locked test set is only available"):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, epochs=1, locked_test_fraction=0.3)
    with pytest.raises(UserInputError, match="Unknown deep-learning evaluation strategy"):
        dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, epochs=1, evaluation_strategy="bootstrap")
    with pytest.raises(UserInputError, match="Unknown deep-learning evaluation strategy"):
        dm.evaluate_single_deep_survival_model(
            "deepsurv", df=df, time_column="os_months", event_column="os_event", features=FEATURES,
            epochs=1, evaluation_strategy="bootstrap",
        )


# Locked-test refit failures (the D20 fix of the ML module, applied to the deep models) -----


def test_locked_test_refit_failures_are_errors_but_keep_the_model_ranked(monkeypatch) -> None:
    original = dm._run_deep_compare_task
    failing: set[str] = set()

    def _task(task):
        if task["repeat"] is None and task["model_name"] in failing:
            raise ValueError("refit on the development set failed")
        return original(task)

    monkeypatch.setattr(dm, "_run_deep_compare_task", _task)
    df = make_example_dataset(seed=5, n_patients=100)
    common = dict(
        epochs=2, hidden_layers=[4], num_time_bins=5, included_models=["DeepSurv", "Neural MTLR"],
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, locked_test_fraction=0.3, random_seed=4,
    )
    clean = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    assert clean["errors"] == [] and clean["ranking_complete"]
    selected, other = (row["model"] for row in clean["comparison_table"])

    failing.update({other})
    result = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    assert result["errors"] == [{"model": other, "stage": "locked_test", "error": "refit on the development set failed"}]
    assert result["ranking_complete"] is False
    assert result["evaluation_mode"] == "repeated_cv"
    rows = {row["model"]: row for row in result["comparison_table"]}
    assert [row["model"] for row in result["comparison_table"]] == [selected, other]
    assert rows[other]["rank"] == 2 and rows[other]["n_failures"] == 0 and rows[other]["c_index"] is not None
    assert rows[other]["locked_test_c_index"] is None and rows[other]["locked_test_error"]
    cautions = result["scientific_summary"]["cautions"]
    assert any(caution.startswith(f"1 model(s) failed when refit on the development set") and caution.endswith("is blank.") for caution in cautions)

    failing.update({selected})
    both = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, **common)
    assert {error["model"] for error in both["errors"]} == {selected, other}
    assert all(error["stage"] == "locked_test" for error in both["errors"])
    first_cautions = both["scientific_summary"]["cautions"][:2]
    assert any(f"including the CV-selected model ({selected})" in caution for caution in first_cautions)


def test_holdout_summary_points_to_bootstrap_intervals() -> None:
    summary = dm._scientific_summary_dl("DeepSurv", 0.7, 100, 40, 30, 3, 10, [1.0, 0.9], "holdout")
    assert any("bootstrap intervals over the test patients" in caution for caution in summary["cautions"])
    assert not any("no confidence interval" in caution for caution in summary["cautions"])


# E13 ------------------------------------------------------------------------


def test_reported_epochs_are_those_of_the_refit_model() -> None:
    df = make_example_dataset(seed=7, n_patients=220)
    result = dm.train_deepsurv(
        df, "os_months", "os_event", FEATURES, hidden_layers=[8], epochs=80, early_stopping_patience=3, random_seed=5,
    )
    assert result["refit_on_training_partition"]
    assert result["epochs_trained"] == result["refit_epochs"] == len(result["refit_loss_history"])
    assert result["early_stopping_epochs"] == len(result["loss_history"]) > result["epochs_trained"]
    metrics = {metric["label"]: metric["value"] for metric in result["scientific_summary"]["metrics"]}
    assert metrics["Epochs"] == result["epochs_trained"]
    assert metrics["Early-stopping epochs"] == result["early_stopping_epochs"]
    assert f"trained for {result['epochs_trained']} epoch(s)" in result["scientific_summary"]["strengths"][0]

    compared = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, hidden_layers=[8], epochs=80, early_stopping_patience=3,
        random_seed=5, included_models=["DeepSurv"],
    )
    row = compared["comparison_table"][0]
    assert (row["epochs_trained"], row["early_stopping_epochs"]) == (result["epochs_trained"], result["early_stopping_epochs"])


# E14 ------------------------------------------------------------------------


def test_text_infinite_times_are_dropped_like_the_ml_cohort() -> None:
    from survival_toolkit.analysis import _cohort_frame

    df = make_example_dataset(seed=4, n_patients=120)
    df["os_months"] = df["os_months"].astype(object)
    df.loc[5, "os_months"] = "inf"
    clean = dm._coerce_deep_frame(df, "os_months", "os_event", FEATURES)
    cohort = _cohort_frame(df, "os_months", "os_event", extra_columns=FEATURES, drop_missing_extra_columns=False)
    assert clean.attrs["source_row_index"] == list(cohort.attrs["source_row_index"])
    assert np.isfinite(clean["os_months"]).all()

    with_inf = dm.compare_deep_survival_models(df, "os_months", "os_event", FEATURES, epochs=2, random_seed=9, included_models=["DeepSurv"])
    without = dm.compare_deep_survival_models(
        df.drop(index=5), "os_months", "os_event", FEATURES, epochs=2, random_seed=9, included_models=["DeepSurv"]
    )
    assert with_inf["evaluation_split_fingerprint"] == without["evaluation_split_fingerprint"]
    assert with_inf["comparison_table"][0]["c_index"] == without["comparison_table"][0]["c_index"]


# E15 ------------------------------------------------------------------------


def test_repeated_cv_accepts_the_largest_seed() -> None:
    df = make_example_dataset(seed=4, n_patients=80)
    result = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=2, random_seed=2**32 - 1,
    )
    row = result["comparison_table"][0]
    assert row["split_seeds"] == [0, 2**32 - 1]
    assert row["n_evaluations"] == 4
    assert dm._derived_seed(42, 3) == 45


# E16 ------------------------------------------------------------------------


@pytest.mark.parametrize("model_type, network", [("mtlr", "NeuralMTLRNet"), ("vae", "SurvivalVAENet")])
def test_single_model_and_comparison_defaults_build_the_same_network(monkeypatch, model_type, network) -> None:
    built: list[list[int]] = []
    original = getattr(dm, network)

    class _Spy(original):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            built.append([module.out_features for module in self.modules() if isinstance(module, torch.nn.Linear)])

    monkeypatch.setattr(dm, network, _Spy)
    df = make_example_dataset(seed=4, n_patients=80)
    dm.evaluate_single_deep_survival_model(
        model_type, df=df, time_column="os_months", event_column="os_event", features=FEATURES, epochs=1,
        early_stopping_patience=None,
    )
    name = dm._canonical_deep_model_name(model_type)
    dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, epochs=1, included_models=[name], early_stopping_patience=None,
    )
    assert len(built) == 2
    assert built[0] == built[1]


# E17 ------------------------------------------------------------------------


def test_coding_errors_inside_a_model_are_not_recorded_as_model_failures(monkeypatch) -> None:
    # A TypeError raised by SurvStudio code inside the trainer (here: unpacking None).
    monkeypatch.setattr(dm, "_deep_fit_summary_counts", lambda *args, **kwargs: None)
    df = make_example_dataset(seed=4, n_patients=80)
    for strategy in ("holdout", "repeated_cv"):
        with pytest.raises(InternalAnalysisError) as raised:
            dm.compare_deep_survival_models(
                df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], included_models=["DeepSurv"],
                evaluation_strategy=strategy, cv_folds=2, cv_repeats=1,
            )
        assert isinstance(raised.value.__cause__, TypeError)


def test_library_type_errors_stay_ordinary_model_failures(monkeypatch) -> None:
    original = dm._make_optimizer

    def _optimizer(model, learning_rate):
        if isinstance(model, dm.DeepSurvNet):
            return torch.optim.AdamW(model.parameters(), lr="not a number")  # TypeError inside torch
        return original(model, learning_rate)

    monkeypatch.setattr(dm, "_make_optimizer", _optimizer)
    df = make_example_dataset(seed=4, n_patients=80)
    result = dm.compare_deep_survival_models(
        df, "os_months", "os_event", FEATURES, epochs=1, hidden_layers=[4], num_time_bins=5,
        included_models=["DeepSurv", "Neural MTLR"],
    )
    assert [error["model"] for error in result["errors"]] == ["DeepSurv"]
    assert [row["model"] for row in result["comparison_table"]] == ["Neural MTLR"]


def test_worker_marks_survive_pickling_so_the_parent_re_raises() -> None:
    exc = InternalAnalysisError()
    assert not dm._must_propagate_deep(exc)
    dm._mark_must_propagate(exc)
    restored = pickle.loads(pickle.dumps(exc))
    assert dm._must_propagate_deep(restored)
    with pytest.raises(InternalAnalysisError):
        dm._record_fold_error([], "DeepSurv", 1, 1, restored)

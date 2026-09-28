"""Regression tests for the review of the ML models and the model-comparison intervals."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.errors import JobCancelledError, UserInputError
from survival_toolkit.sample_data import make_example_dataset


def _sksurv_available() -> bool:
    try:
        import sksurv  # noqa: F401
    except ImportError:
        return False
    return True


requires_sksurv = pytest.mark.skipif(not _sksurv_available(), reason="scikit-survival not installed")


def _graded_cohort(n: int = 260) -> pd.DataFrame:
    """Example cohort with a float-coded grade (1.0/2.0/3.0 and some missing) that is prognostic."""
    df = make_example_dataset(seed=20, n_patients=n).copy()
    rng = np.random.default_rng(3)
    grade = rng.integers(1, 4, size=len(df)).astype(float)
    grade[rng.random(len(df)) < 0.05] = np.nan
    df["grade"] = grade
    df["os_months"] = df["os_months"] / np.where(np.nan_to_num(grade, nan=1.0) == 3.0, 3.0, 1.0)
    return df


# ── Counterfactual and partial dependence on categorical targets ────────────


@requires_sksurv
def test_counterfactual_resolves_numeric_level_text_and_refuses_unknown_levels() -> None:
    from survival_toolkit import ml_models as ml

    df = _graded_cohort()
    features = ["age", "biomarker_score", "grade"]
    fitted = ml.train_random_survival_forest(
        df, "os_months", "os_event", features, categorical_features=["grade"], n_estimators=30, random_state=1
    )
    # The shared encoder stores whole-number codes of a float column as canonical labels ("3", not "3.0").
    assert fitted["_feature_encoder"]["categorical_mappings"]["grade"]["all_levels"] == ["1", "2", "3"]

    def run(value, original=None):
        return ml.counterfactual_survival(
            df, "os_months", "os_event", features, categorical_features=["grade"], target_feature="grade",
            original_value=original, counterfactual_value=value, trained_result=fitted,
        )

    as_level = run("3")
    # 3, 3.0 and "3.0" name the level "3" of the float-coded column instead of falling back to the reference level.
    for value in (3, "3.0", 3.0, " 3.0 ", " 3 "):
        result = run(value)
        assert result["risk_change_pct"] == pytest.approx(as_level["risk_change_pct"])
        assert "to 3" in result["scientific_summary"]["headline"]
    assert as_level["risk_change_pct"] > 5.0

    for bad in ("9 (does not exist)", "Stage 4", 4):
        with pytest.raises(ValueError, match="is not a level of 'grade'.*'1', '2', '3'"):
            run(bad)
    with pytest.raises(ValueError, match="is not a level of 'grade'"):
        run("3", original="unknown")


@requires_sksurv
def test_counterfactual_and_partial_dependence_follow_the_encoder_for_undeclared_text_features() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=200)
    features = ["age", "biomarker_score", "stage"]
    # "stage" is text, so the encoder one-hot encodes it although the request lists no categorical feature.
    fitted = ml.train_random_survival_forest(df, "os_months", "os_event", features, categorical_features=[],
                                             n_estimators=20, random_state=1)
    assert fitted["_feature_encoder"]["categorical_features"] == ["stage"]

    pdp = ml.compute_partial_dependence(
        fitted["_model"], fitted["_X_encoded"], "stage", categorical_features=[],
        feature_encoder=fitted["_feature_encoder"], analysis_frame=fitted["_analysis_frame"],
    )
    assert pdp["feature_type"] == "categorical" and pdp["values"] == ["I", "II", "III", "IV"]

    result = ml.counterfactual_survival(df, "os_months", "os_event", features, categorical_features=[],
                                        target_feature="stage", counterfactual_value="IV", trained_result=fitted)
    assert result["risk_change_pct"] is not None and result["risk_change_pct"] > 0.0
    with pytest.raises(ValueError, match="is not a level of 'stage'"):
        ml.counterfactual_survival(df, "os_months", "os_event", features, categorical_features=[],
                                   target_feature="stage", counterfactual_value="Stage 4", trained_result=fitted)


@requires_sksurv
def test_counterfactual_labels_the_random_survival_forest_score_as_summed_cumulative_hazard() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=21, n_patients=160)
    result = ml.counterfactual_survival(df, "os_months", "os_event", ["age", "biomarker_score"], target_feature="age",
                                        original_value=50, counterfactual_value=70, model_type="rsf", n_estimators=15)
    assert "summed over the training event times" in result["risk_change_method"]
    assert "cumulative-hazard ratio" not in result["risk_change_method"]
    assert "summed over the training event times" in result["scientific_summary"]["headline"]


# ── Random Survival Forest memory ───────────────────────────────────────────


@requires_sksurv
def test_rsf_memory_estimate_tracks_the_fitted_forest() -> None:
    from sksurv.ensemble import RandomSurvivalForest

    from survival_toolkit.ml_models import rsf_memory_estimate_bytes

    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 5))
    y = np.empty(n, dtype=[("event", bool), ("time", float)])
    y["event"] = rng.random(n) < 0.6
    y["time"] = np.round(rng.exponential(30.0, size=n), 3) + 0.001
    for leaf, depth in ((6, None), (3, None), (6, 4)):
        forest = RandomSurvivalForest(n_estimators=5, min_samples_leaf=leaf, max_depth=depth, random_state=1).fit(X, y)
        actual = sum(tree.tree_.value.nbytes for tree in forest.estimators_)
        estimate = rsf_memory_estimate_bytes(n, forest.unique_times_.size, n_estimators=5, min_samples_leaf=leaf, max_depth=depth)
        assert 0.8 * actual <= estimate <= 1.6 * actual


@requires_sksurv
def test_rsf_refuses_a_forest_beyond_the_memory_budget(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=22, n_patients=200)
    features = ["age", "biomarker_score", "stage"]
    monkeypatch.setenv("SURVSTUDIO_RSF_MEMORY_BUDGET_GB", "0.001")
    with pytest.raises(ValueError, match="Random Survival Forest would need about .* GB of memory: 100 trees"):
        ml.train_random_survival_forest(df, "os_months", "os_event", features)
    # A comparison records the refusal for the forest and keeps the other models.
    comparison = ml.compare_survival_models(df, "os_months", "os_event", features, n_estimators=100)
    assert [error["model"] for error in comparison["errors"]] == ["Random Survival Forest"]
    assert "GB of memory" in comparison["errors"][0]["error"]
    assert {row["model"] for row in comparison["comparison_table"]} == {"Cox PH", "LASSO-Cox", "Gradient Boosted Survival"}
    monkeypatch.delenv("SURVSTUDIO_RSF_MEMORY_BUDGET_GB")
    assert ml.train_random_survival_forest(df, "os_months", "os_event", features, n_estimators=10)["model_stats"]["c_index"]


@requires_sksurv
def test_forest_predictions_in_row_chunks_match_one_call(monkeypatch) -> None:
    from sksurv.ensemble import RandomSurvivalForest

    from survival_toolkit import ml_models as ml

    rng = np.random.default_rng(5)
    n = 300
    X = rng.normal(size=(n, 4))
    y = np.empty(n, dtype=[("event", bool), ("time", float)])
    y["event"] = rng.random(n) < 0.7
    y["time"] = rng.exponential(np.exp(-X[:, 0]) * 20.0)
    forest = RandomSurvivalForest(n_estimators=12, min_samples_leaf=5, random_state=2, n_jobs=2).fit(X, y)
    whole = forest.predict(X)
    grid = np.quantile(y["time"], [0.2, 0.5, 0.8])
    whole_survival = ml._step_function_matrix(forest.predict_survival_function(X), grid)

    monkeypatch.setattr(ml, "_RSF_PREDICTION_CHUNK_BYTES", 1)
    assert ml._prediction_row_chunk(forest) == 64
    np.testing.assert_allclose(ml._predict_risk_scores(forest, X), whole, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(ml._sksurv_survival_predictor(forest, X)(grid), whole_survival, rtol=1e-12, atol=0.0)


# ── Integrated Brier score ──────────────────────────────────────────────────


@requires_sksurv
@pytest.mark.parametrize("model_name", ["gbs", "coxnet", "rsf"])
def test_survival_predictor_keeps_only_grid_columns_with_identical_values(model_name: str) -> None:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=23, n_patients=180)
    X = df[["age", "biomarker_score", "immune_index"]].fillna(0.0).to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = ml._prepare_sksurv_data(df, "os_months", "os_event")
    alpha = None
    if model_name == "gbs":
        model = GradientBoostingSurvivalAnalysis(n_estimators=20, max_depth=2, random_state=0).fit(X, y)
    elif model_name == "coxnet":
        model = ml._make_lasso_coxnet_model(alpha=0.02).fit(X, y)
        alpha = 0.02
    else:
        model = RandomSurvivalForest(n_estimators=10, min_samples_leaf=6, random_state=0, n_jobs=1).fit(X, y)
    frame = pd.DataFrame(X)
    grid = np.concatenate([[0.0, 1e-9], np.quantile(df["os_months"], [0.1, 0.35, 0.6, 0.9]), [float(df["os_months"].min())]])
    grid = np.unique(grid)
    kwargs = {"return_array": False} if alpha is None else {"return_array": False, "alpha": alpha}
    reference = ml._step_function_matrix(model.predict_survival_function(X, **kwargs), grid)

    predicted = ml._sksurv_survival_predictor(model, frame, alpha=alpha)(grid)

    assert predicted.shape == (X.shape[0], grid.size)
    np.testing.assert_allclose(predicted, reference, rtol=1e-14, atol=0.0)
    # Times past the last training time are refused, as sksurv's step functions refuse them.
    with pytest.raises(ValueError, match="x must be within"):
        ml._sksurv_survival_predictor(model, frame, alpha=alpha)(np.array([float(df["os_months"].max()) + 1.0]))


def test_integrated_brier_score_stops_before_the_censoring_survival_reaches_zero() -> None:
    from survival_toolkit.ml_models import _censoring_survival_zero_time, compute_integrated_brier_score

    rng = np.random.default_rng(0)
    # The last support time carries an event tied with a censoring, so G(60) = 0.
    support_times = np.concatenate([rng.uniform(1, 59, 60), [60.0, 60.0]])
    support_events = np.concatenate([rng.integers(0, 2, 60), [1, 0]])
    eval_times = np.concatenate([rng.uniform(1, 59, 30), rng.uniform(61, 80, 6)])
    eval_events = np.concatenate([rng.integers(0, 2, 30), np.zeros(6, dtype=int)])

    def survival(grid):
        return np.clip(1.0 - np.asarray(grid)[None, :] / 100.0, 0, 1).repeat(eval_times.size, axis=0)

    assert _censoring_survival_zero_time(support_times, support_events) == 60.0
    result = compute_integrated_brier_score(eval_times, eval_events, survival,
                                            support_times=support_times, support_events=support_events)
    assert result["eval_times"][-1] < 60.0
    assert 0.0 <= result["ibs"] < 1.0 and max(row["score"] for row in result["brier_scores"]) < 1.0
    assert any("censoring survival reaches zero at 60" in text for text in result["scientific_summary"]["strengths"])

    requested = compute_integrated_brier_score(eval_times, eval_events, survival, eval_times=[20.0, 40.0, 60.0],
                                               support_times=support_times, support_events=support_events)
    assert requested["eval_times"] == [20.0, 40.0]
    with pytest.raises(ValueError, match="censoring survival of the support cohort reaches zero at t=60"):
        compute_integrated_brier_score(eval_times, eval_events, survival, eval_times=[60.0],
                                       support_times=support_times, support_events=support_events)


@requires_sksurv
def test_time_dependent_importance_skips_times_where_the_censoring_survival_is_zero() -> None:
    from survival_toolkit.ml_models import compute_time_dependent_importance

    rng = np.random.default_rng(1)
    # 18 patients (too few for a holdout, so the fitting rows are the evaluation rows); the last
    # time carries an event tied with a censoring, so the censoring survival is zero from t=20 on.
    df = pd.DataFrame({
        "time": np.concatenate([np.arange(1.0, 17.0), [20.0, 20.0]]),
        "event": np.concatenate([np.tile([1, 0], 8), [1, 0]]),
        "x": rng.normal(size=18),
    })
    result = compute_time_dependent_importance(df, "time", "event", ["x"], eval_times=[5.0, 20.0], model_type="gbs",
                                               n_estimators=10, event_positive_value=1)
    assert result["eval_times"] == [5.0]
    assert result["skipped_time_points"] == [{"time": 20.0, "reason": "censoring_survival_zero", "evaluable_patients": None}]
    assert all(np.isfinite(value) for value in result["baseline_brier_scores"])


def test_brier_helpers_propagate_bugs_cancellation_and_memory_errors(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1, 0, 1, 0])

    def failing(exc):
        def _raise(*args, **kwargs):
            raise exc
        return _raise

    for exc in (MemoryError("oom"), KeyError("bug"), JobCancelledError("stop")):
        monkeypatch.setattr(ml, "compute_integrated_brier_score", failing(exc))
        with pytest.raises(type(exc)):
            ml._maybe_compute_brier_metrics(times, events, lambda grid: None)
    monkeypatch.setattr(ml, "compute_integrated_brier_score", failing(ValueError("no support")))
    with pytest.warns(RuntimeWarning, match="no support"):
        assert ml._maybe_compute_brier_metrics(times, events, lambda grid: None) is None

    class _BrokenSurvival:
        def predict_survival_function(self, X, return_array=False):
            raise AttributeError("bug inside the model code")

    with pytest.raises(AttributeError):
        ml._maybe_compute_sksurv_brier_metrics(times, events, _BrokenSurvival(), pd.DataFrame({"x": times}))


# ── Cox baseline rank pruning ───────────────────────────────────────────────


def test_cox_rank_pruning_keeps_tiny_scale_features_and_drops_constant_combinations() -> None:
    from survival_toolkit import ml_models as ml

    rng = np.random.default_rng(7)
    n = 400
    tiny = rng.normal(size=n) * 1e-9
    huge = rng.normal(1e8, 1e7, size=n)
    t_event = rng.exponential(np.exp(-1.2 * tiny / 1e-9) * 10)
    t_cens = rng.exponential(12, n)
    df = pd.DataFrame({"t": np.minimum(t_event, t_cens) + 0.01, "e": (t_event <= t_cens).astype(int), "tiny": tiny, "huge": huge})
    train, test = df.iloc[:280].reset_index(drop=True), df.iloc[280:].reset_index(drop=True)
    result = ml._fit_evaluate_cox_split(train, test, time_column="t", event_column="e", features=["tiny", "huge"])
    assert result["n_features"] == 2
    assert result["c_index"] > 0.7

    # Fractions that sum to one are not identifiable in a model without intercept.
    frac_a = rng.uniform(0.1, 0.6, n)
    frac_b = rng.uniform(0.1, 0.3, n)
    composition = df.assign(frac_a=frac_a, frac_b=frac_b, frac_c=1.0 - frac_a - frac_b, age=rng.normal(60, 10, n))
    result = ml._fit_evaluate_cox_split(composition.iloc[:280].reset_index(drop=True), composition.iloc[280:].reset_index(drop=True),
                                        time_column="t", event_column="e", features=["age", "frac_a", "frac_b", "frac_c"])
    assert result["n_features"] == 3


def test_cox_rank_pruning_order_does_not_change_a_full_rank_design() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=24, n_patients=200)
    encoded, held_out, _ = ml._encode_train_test_features(df.iloc[:140], df.iloc[140:], ["age", "biomarker_score", "stage", "sex"], ["stage", "sex"])
    old_train, old_eval = ml._drop_rank_deficient_train_columns(encoded, held_out)
    old_train, old_eval, _, _ = ml._standardize_encoded_matrices(old_train, old_eval)
    new_train, new_eval, _, _ = ml._standardize_encoded_matrices(encoded, held_out)
    new_train, new_eval = ml._drop_rank_deficient_train_columns(new_train, new_eval)
    pd.testing.assert_frame_equal(old_train, new_train, check_exact=True)
    pd.testing.assert_frame_equal(old_eval, new_eval, check_exact=True)


# ── Validation of requests and settings ─────────────────────────────────────


@requires_sksurv
@pytest.mark.parametrize("depth", [0, -3, 2.5, True])
def test_invalid_max_depth_is_refused_before_any_fit(depth) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=25, n_patients=120)
    features = ["age", "biomarker_score"]
    for call in (
        lambda: ml.train_random_survival_forest(df, "os_months", "os_event", features, max_depth=depth, n_estimators=10),
        lambda: ml.train_gradient_boosted_survival(df, "os_months", "os_event", features, max_depth=depth, n_estimators=10),
        lambda: ml.compare_survival_models(df, "os_months", "os_event", features, max_depth=depth, n_estimators=10),
        lambda: ml.cross_validate_survival_models(df, "os_months", "os_event", features, max_depth=depth, n_estimators=10,
                                                  cv_folds=2, cv_repeats=1),
    ):
        with pytest.raises(UserInputError, match="max_depth must be a whole number of at least 1"):
            call()


def test_cutpoint_scan_refuses_an_outcome_column_as_the_marker() -> None:
    from survival_toolkit.ml_models import find_optimal_cutpoint

    df = make_example_dataset(seed=10, n_patients=120)
    with pytest.raises(UserInputError, match="'os_months' is the survival time column"):
        find_optimal_cutpoint(df, "os_months", "os_event", "os_months", event_positive_value=1, permutation_iterations=0)
    with pytest.raises(UserInputError, match="'os_event' is the survival event column"):
        find_optimal_cutpoint(df, "os_months", "os_event", "os_event", event_positive_value=1, permutation_iterations=0)


@requires_sksurv
def test_constant_feature_errors_name_the_model() -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=10, n_patients=120).assign(site="A", batch=1.0)
    with pytest.raises(UserInputError, match="remain for Random Survival Forest"):
        ml.train_random_survival_forest(df, "os_months", "os_event", ["site", "batch"], n_estimators=10)
    with pytest.raises(UserInputError, match="remain for Gradient Boosted Survival"):
        ml.train_gradient_boosted_survival(df, "os_months", "os_event", ["site", "batch"], n_estimators=10)


@requires_sksurv
def test_unseen_level_caution_covers_text_features_not_declared_categorical() -> None:
    from survival_toolkit import ml_models as ml
    from survival_toolkit.analysis import _cohort_frame

    df = make_example_dataset(seed=20, n_patients=260).copy()
    features = ["age", "biomarker_score", "stage"]
    frame = _cohort_frame(df, "os_months", "os_event", extra_columns=features, drop_missing_extra_columns=False)
    _, eval_positions, mode = ml._split_train_test_positions(frame, "os_event", random_state=42)
    assert mode == "holdout"
    held_out = [frame.attrs["source_row_index"][position] for position in eval_positions[:4]]
    df.loc[held_out, "stage"] = "Stage X"

    def unseen_cautions(result):
        return [text for text in result["scientific_summary"]["cautions"] if "never occurs" in text]

    for declared in (["stage"], []):
        result = ml.compare_survival_models(df, "os_months", "os_event", features, categorical_features=declared, n_estimators=10)
        assert unseen_cautions(result) and unseen_cautions(result)[0].startswith("4 evaluation row(s)")


# ── Model comparison intervals ──────────────────────────────────────────────


def _outcomes(n: int = 60, seed: int = 4):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-signal))
    censor = rng.exponential(1.5, size=n)
    return np.minimum(event_time, censor), (event_time <= censor).astype(int), signal


def test_prediction_blocks_from_a_client_are_validated_with_specific_messages() -> None:
    from survival_toolkit.evaluation import merge_prediction_blocks

    time, event, signal = _outcomes()
    rows = [f"r{i}" for i in range(time.size)]
    good = {"row_ids": rows, "time": time.tolist(), "event": event.tolist(), "risk": {"Cox PH": signal.tolist()}}
    cases = [
        ({"row_ids": rows, "event": event.tolist(), "risk": {"DeepSurv": signal.tolist()}}, "Prediction block 2 lacks 'time'"),
        ({**good, "time": time[:10].tolist(), "risk": {"DeepSurv": signal.tolist()}}, "'time' has 10 values for 60 row IDs"),
        ({**good, "risk": {"DeepSurv": signal[:5].tolist()}}, "'risk of DeepSurv' has 5 values for 60 row IDs"),
        ({**good, "risk": [1, 2, 3]}, "'risk' must map each model name"),
        ({**good, "event": [2] * time.size, "risk": {"DeepSurv": signal.tolist()}}, "every 'event' must be 0 or 1"),
        ({**good, "row_ids": ["same"] * time.size}, "repeats a row label"),
        ({**good, "time": ["soon"] * time.size}, "'time' must contain only numbers"),
        ("not a block", "Prediction block 2 must be an object"),
    ]
    for bad, message in cases:
        with pytest.raises(UserInputError, match=message):
            merge_prediction_blocks([good, bad])
    with pytest.raises(UserInputError, match="No test-set predictions"):
        merge_prediction_blocks([None, {"row_ids": []}])


def test_interval_endpoint_reports_block_problems_as_bad_requests() -> None:
    from fastapi.testclient import TestClient

    from survival_toolkit.app import app

    time, event, signal = _outcomes()
    rows = [f"r{i}" for i in range(time.size)]
    good = {"row_ids": rows, "time": time.tolist(), "event": event.tolist(), "risk": {"Cox PH": signal.tolist()}}
    client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    # Blocks that cannot be merged reach the evaluation module, which names the problem (400).
    merge_cases = [
        ([good, {**good, "row_ids": [f"x{i}" for i in range(time.size)], "risk": {"DeepSurv": signal.tolist()}}], "share no test patients"),
        ([good, {**good, "event": [1] * time.size, "risk": {"DeepSurv": signal.tolist()}}], "disagree on the outcome"),
    ]
    for predictions, message in merge_cases:
        response = client.post("/api/model-comparison-intervals", json={"predictions": predictions})
        assert response.status_code == 400, response.text
        assert message in response.json()["detail"]
    # Malformed blocks are refused by the request model (422) or the evaluation module (400), never with a 500,
    # and the message names the block field at fault.
    shape_cases = [
        ([good, {"row_ids": rows, "event": event.tolist(), "risk": {"DeepSurv": signal.tolist()}}], "time"),
        ([{**good, "risk": {"Cox PH": signal[:5].tolist()}}], "risk"),
        ([{**good, "risk": [1, 2, 3]}], "risk"),
    ]
    for predictions, field in shape_cases:
        response = client.post("/api/model-comparison-intervals", json={"predictions": predictions})
        assert response.status_code in (400, 422), response.text
        detail = str(response.json()["detail"])
        assert field in detail and ("block" in detail.lower()), detail


def test_interval_bootstrap_stops_when_its_request_is_cancelled() -> None:
    from survival_toolkit.concurrency import cancellation_scope
    from survival_toolkit.evaluation import c_index_intervals

    time, event, signal = _outcomes()
    cancelled = threading.Event()
    cancelled.set()
    with cancellation_scope(cancelled), pytest.raises(JobCancelledError):
        c_index_intervals(time, event, {"Cox PH": signal}, n_bootstrap=200)
    with pytest.raises(UserInputError, match="finite risk score"):
        c_index_intervals(time, event, {"Cox PH": signal[:10]})
    with pytest.raises(UserInputError, match="event indicator of 0 or 1"):
        c_index_intervals(time, event * 2, {"Cox PH": signal})


# ── SHAP ────────────────────────────────────────────────────────────────────


@requires_sksurv
@pytest.mark.parametrize("trainer_name", ["train_random_survival_forest", "train_gradient_boosted_survival"])
def test_kernel_shap_for_sksurv_models_is_reproducible_and_leaves_numpy_state_alone(trainer_name: str) -> None:
    pytest.importorskip("shap")
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=20, n_patients=140)
    fitted = getattr(ml, trainer_name)(df, "os_months", "os_event", ["age", "biomarker_score", "immune_index", "stage"],
                                       categorical_features=["stage"], n_estimators=10, random_state=5)
    model, X = fitted["_model"], fitted["_X_eval_encoded"]
    np.random.seed(123)
    state_before = np.random.get_state()[1].copy()

    first = ml.compute_shap_values(model, X, fitted["feature_names"])
    assert np.array_equal(np.random.get_state()[1], state_before)
    second = ml.compute_shap_values(model, X, fitted["feature_names"])

    assert first["method"] == "kernel" and first["random_state"] == 5
    assert [row["feature"] for row in first["feature_importance"]] == [row["feature"] for row in second["feature_importance"]]
    for one, two in zip(first["shap_summary"], second["shap_summary"], strict=True):
        np.testing.assert_allclose(one["shap_values"], two["shap_values"], rtol=1e-9, atol=1e-12)
    other_seed = ml.compute_shap_values(model, X, fitted["feature_names"], random_state=6)
    assert other_seed["random_state"] == 6


# ── Time-dependent importance and trainer diagnostics ───────────────────────


@requires_sksurv
def test_internal_callers_skip_the_trainers_importance_and_brier_metrics(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=26, n_patients=150)
    features = ["age", "biomarker_score", "immune_index"]
    reference = ml.compute_time_dependent_importance(df, "os_months", "os_event", features, eval_times=[12.0, 24.0],
                                                     model_type="gbs", n_estimators=15, random_state=3)

    def _unexpected(*args, **kwargs):
        raise AssertionError("internal callers must not compute the trainer's diagnostics")

    monkeypatch.setattr(ml, "_grouped_permutation_importance", _unexpected)
    monkeypatch.setattr(ml, "_holdout_brier_metrics", _unexpected)
    result = ml.compute_time_dependent_importance(df, "os_months", "os_event", features, eval_times=[12.0, 24.0],
                                                  model_type="gbs", n_estimators=15, random_state=3)
    assert result["importance_matrix"] == reference["importance_matrix"]
    ml.counterfactual_survival(df, "os_months", "os_event", features, target_feature="age", original_value=50,
                               counterfactual_value=70, model_type="gbs", n_estimators=15)

    bare = ml.train_gradient_boosted_survival(df, "os_months", "os_event", features, n_estimators=15,
                                              compute_importance=False, compute_brier=False)
    assert bare["feature_importance"] == [] and bare["calibration_metrics"] is None
    assert not any("could not be computed" in text for text in bare["scientific_summary"]["cautions"])


# ── Cutpoint orientation ────────────────────────────────────────────────────


def _loop_observed_expected(times, events, mask):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float) > 0
    mask = np.asarray(mask, dtype=bool)
    observed = float(events[mask].sum())
    expected = 0.0
    for event_time in np.unique(times[events]):
        at_risk = times >= event_time
        deaths = float((events & (times == event_time)).sum())
        expected += deaths * float((at_risk & mask).sum()) / float(at_risk.sum())
    return observed, expected


def test_logrank_observed_expected_matches_the_per_event_time_scan_exactly() -> None:
    from survival_toolkit.ml_models import _logrank_observed_expected

    rng = np.random.default_rng(2)
    for n in (5, 60, 900):
        times = np.round(rng.exponential(10.0, size=n), 0)  # many tied times
        events = (rng.random(n) < 0.6).astype(float)
        mask = rng.random(n) < 0.4
        assert _logrank_observed_expected(times, events, mask) == _loop_observed_expected(times, events, mask)
    assert _logrank_observed_expected(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([True, False])) == (0.0, 0.0)


# ── Summaries and locked test ───────────────────────────────────────────────


def test_summaries_drop_stale_statements() -> None:
    from survival_toolkit import ml_models as ml

    summary = ml._scientific_summary_ml(model_name="RSF", c_index=0.7, n_patients=200, n_events=80, n_features=3,
                                        evaluation_mode="holdout")
    assert not any("no confidence interval" in text for text in summary["cautions"])
    assert any("bootstrap intervals" in text for text in summary["cautions"])
    times = np.array([4.0, 6.0, 9.0, 14.0, 18.0, 24.0])
    events = np.array([1, 1, 0, 1, 0, 1])
    brier = ml.compute_integrated_brier_score(times, events, lambda grid: np.full((6, len(grid)), 0.5))
    assert not any("compute_calibration_data" in text for text in brier["scientific_summary"]["next_steps"])
    assert not hasattr(ml, "_estimate_c_index_standard_error")


def test_locked_test_failures_are_errors_that_leave_the_cross_validation_complete(monkeypatch) -> None:
    from survival_toolkit import ml_models as ml

    df = make_example_dataset(seed=27, n_patients=150)

    def _fit(train_frame, test_frame, **kwargs):
        return {
            "c_index": 0.6 + 0.001 * len(train_frame) / 100.0,
            "n_features": len(kwargs["features"]),
            "training_time_ms": 1.0,
            "train_n": len(train_frame),
            "test_n": len(test_frame),
            "train_events": int(train_frame["os_event"].sum()),
            "test_events": int(test_frame["os_event"].sum()),
        }

    dev_size: dict[str, int] = {}

    def _fails_on_locked_refit(train_frame, test_frame, **kwargs):
        # The locked-test refit is the only fit on the whole development set.
        if len(train_frame) == dev_size["n"]:
            raise ValueError("synthetic locked-test failure")
        return _fit(train_frame, test_frame, **kwargs)

    from survival_toolkit.evaluation import locked_test_split

    dev_positions, _ = locked_test_split(df["os_event"].to_numpy(), random_state=11, test_fraction=0.3)
    dev_size["n"] = len(dev_positions)
    monkeypatch.setattr(ml, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(ml, "_fit_evaluate_cox_split", _fit)
    monkeypatch.setattr(ml, "_fit_evaluate_lasso_cox_split", _fit)
    monkeypatch.setattr(ml, "_fit_evaluate_rsf_split", _fit)
    monkeypatch.setattr(ml, "_fit_evaluate_gbs_split", _fails_on_locked_refit)

    result = ml.cross_validate_survival_models(df, "os_months", "os_event", ["age", "biomarker_score"], cv_folds=3,
                                               cv_repeats=1, random_state=11, locked_test_fraction=0.3)

    assert result["evaluation_mode"] == "repeated_cv"
    assert result["errors"] == [{"model": "Gradient Boosted Survival", "stage": "locked_test", "error": "synthetic locked-test failure"}]
    assert result["ranking_complete"] is False
    # The model is still ranked by cross-validation, so it is not listed as excluded.
    assert result["excluded_models"] == []
    gbs = next(row for row in result["comparison_table"] if row["model"] == "Gradient Boosted Survival")
    assert gbs["c_index"] is not None and gbs["locked_test_c_index"] is None
    assert any("failed when refit on the development set" in text for text in result["scientific_summary"]["cautions"])
    assert not any("fold-level fit" in text for text in result["scientific_summary"]["cautions"])

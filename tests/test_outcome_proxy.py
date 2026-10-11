from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.errors import OutcomeProxyError, must_propagate
from survival_toolkit.outcome_proxy import screen_training_outcomes
from survival_toolkit.prediction_guard import check_training_outcome_proxies, guarded_prediction_inputs


def fixture_frame(n=240):
    rng = np.random.default_rng(2026101112)
    return pd.DataFrame({"time": rng.exponential(10, n), "event": rng.binomial(1, .6, n),
                         "x": rng.normal(size=n)})


@pytest.mark.parametrize("pattern", ["event_copy", "noisy_event", "log_time", "noisy_time", "xor", "split_event"])
def test_hidden_patterns_require_review(pattern):
    df = fixture_frame()
    features = ["x"]
    if pattern == "event_copy":
        df["x"] = 13 - 7 * df.event
    elif pattern == "noisy_event":
        df["x"] = df.event + .05 * df.x
    elif pattern == "log_time":
        df["x"] = np.log1p(df.time)
    elif pattern == "noisy_time":
        df["x"] = np.log1p(df.time) + .001 * df.x
    elif pattern == "xor":
        df["x"] = np.arange(len(df)) % 2
        df["y"] = np.bitwise_xor(df.x.astype(int), df.event)
        features = ["x", "y"]
    else:
        df["y"] = df.event - df.x
        features = ["x", "y"]
    before = df.copy(deep=True)
    audit = screen_training_outcomes(df, "time", "event", features)
    assert audit["status"] == "review_required"
    assert audit["findings"]
    if pattern in {"xor", "split_event", "noisy_time"}:
        assert any(f.get("method") == "neural" for f in audit["findings"])
    pd.testing.assert_frame_equal(df, before)


def test_strong_baseline_survival_signal_is_not_automatically_rejected():
    rng = np.random.default_rng(2026101115)
    df = fixture_frame()
    failure = rng.exponential(10 * np.exp(-1.5 * df.x))
    censor = rng.exponential(20, len(df))
    df["time"] = np.minimum(failure, censor)
    df["event"] = (failure <= censor).astype(int)
    audit = screen_training_outcomes(df, "time", "event", ["x"])
    assert audit["status"] == "no_pattern_detected"
    assert not audit["findings"]
    assert "cannot" in audit["notes"][0]


def test_small_or_rare_event_data_are_explicitly_not_fully_assessed():
    assert screen_training_outcomes(fixture_frame(40), "time", "event", ["x"])["status"] == "not_assessed"
    df = fixture_frame()
    df["event"] = 0
    df.loc[:3, "event"] = 1
    audit = screen_training_outcomes(df, "time", "event", ["x"])
    assert audit["status"] == "partially_assessed"
    assert {s["target"] for s in audit["scores"]} == {"log_follow_up"}


def test_inner_scaling_and_categories_are_fitted_on_inner_training_rows():
    from survival_toolkit.outcome_proxy import _inner_design
    train = pd.DataFrame({"x": [1., 2., np.nan, 3.], "group": ["a", "a", "b", "b"]})
    validation = pd.DataFrame({"x": [100., np.nan], "group": ["new", "b"]})
    x_fit, x_score = _inner_design(train, validation, ["x", "group"], ["group"])
    changed = validation.assign(x=[-1e5, 1e5], group=["other", "other"])
    repeated, _ = _inner_design(train, changed, ["x", "group"], ["group"])
    np.testing.assert_array_equal(x_fit, repeated)
    assert x_fit.shape[1] == 3 and np.isfinite(x_score).all()
    typed = train.copy()
    typed["group"] = pd.Categorical(["a", "a", "b", None])
    categorical_fit, _ = _inner_design(typed, validation, ["x", "group"], ["group"])
    assert np.isfinite(categorical_fit).all()


def test_review_is_deliberate_recorded_and_carried_into_exports():
    fits = []

    @guarded_prediction_inputs
    def model(df, time_column, event_column, features, event_positive_value=1, outcome_proxy_review=None):
        check_training_outcome_proxies(df, time_column, event_column, features)
        fits.append(True)
        return {"comparison_table": [{"model": "fixture", "c_index": None}],
                "scientific_summary": {"cautions": []}, "manuscript_tables": {"table_notes": []}}

    df = fixture_frame()
    df["x"] = df.event + .05 * df.x
    with pytest.raises(OutcomeProxyError) as error:
        model(df, "time", "event", ["x"])
    assert fits == [] and must_propagate(error.value)
    with pytest.raises(ValueError, match="20–1000"):
        model(df, "time", "event", ["x"], outcome_proxy_review="OK")
    reason = "Baseline assay from the enrollment visit; checked against the source table."
    result = model(df, "time", "event", ["x"], outcome_proxy_review=reason)
    audit = result["input_audit"]["outcome_proxy"]
    assert audit["status"] == "reviewed_by_user" and audit["partitions"][0]["findings"]
    assert result["comparison_table"][0]["outcome_proxy_review"] == reason
    assert result["comparison_table"][0]["c_index"] is None
    assert any(reason in note for note in result["manuscript_tables"]["table_notes"])


@pytest.mark.parametrize("route", ["ml_holdout", "ml_cv", "ml_single", "deep_holdout", "deep_cv", "deep_single"])
def test_public_paths_pause_before_survival_fit(route, monkeypatch):
    from survival_toolkit import ml_models, deep_models
    df = fixture_frame()
    df["x"] = df.event + .05 * df.x
    fits = []

    def fail_fit(*args, **kwargs):
        fits.append(True)
        raise AssertionError("A survival model must not fit before the proxy review.")

    monkeypatch.setattr(ml_models, "_ml_model_specs", lambda **kwargs: [("Cox", fail_fit, {})])
    monkeypatch.setattr(deep_models, "_deep_trainer_specs", lambda *args, **kwargs: [("DeepSurv", fail_fit, {})])
    monkeypatch.setattr(deep_models, "_fit_with_refit", fail_fit)
    monkeypatch.setattr(ml_models.RandomSurvivalForest, "fit", fail_fit)
    functions = {"ml_holdout": ml_models.compare_survival_models,
                 "ml_cv": ml_models.cross_validate_survival_models,
                 "ml_single": ml_models.train_random_survival_forest,
                 "deep_holdout": deep_models.compare_deep_survival_models,
                 "deep_cv": deep_models.compare_deep_survival_models,
                 "deep_single": deep_models.train_deepsurv}
    extra = {"cv_folds": 2, "cv_repeats": 1} if route.endswith("cv") else {}
    if route == "deep_cv":
        extra["evaluation_strategy"] = "repeated_cv"
    with pytest.raises(OutcomeProxyError):
        functions[route](df, "time", "event", ["x"], **extra)
    assert fits == []


def test_holdout_outcomes_and_features_do_not_enter_the_pattern_screen(monkeypatch):
    from survival_toolkit import ml_models
    df = fixture_frame()
    train = np.arange(180)
    test = np.arange(180, 240)
    monkeypatch.setattr(ml_models, "_split_train_test_positions", lambda *args, **kwargs: (train, test, "holdout"))
    monkeypatch.setattr(ml_models, "_ml_model_specs", lambda **kwargs: [("Cox PH", ml_models._fit_evaluate_cox_split, {})])
    first = ml_models.compare_survival_models(df, "time", "event", ["x"])["input_audit"]["outcome_proxy"]
    changed = df.copy()
    changed.loc[test, "time"] = 1000
    changed.loc[test, "event"] = 1 - changed.loc[test, "event"]
    changed.loc[test, "x"] = changed.loc[test, "event"]
    repeated = ml_models.compare_survival_models(changed, "time", "event", ["x"])["input_audit"]["outcome_proxy"]
    assert first == repeated
    assert first["partitions"][0]["training_rows"] == 180


def test_screen_failure_stops_the_whole_request(monkeypatch):
    import survival_toolkit.outcome_proxy as proxy
    def failed(*args, **kwargs):
        raise ValueError("diagnostic solver failed")
    monkeypatch.setattr(proxy, "screen_training_outcomes", failed)
    with pytest.raises(OutcomeProxyError) as caught:
        check_training_outcome_proxies(fixture_frame(), "time", "event", ["x"])
    assert caught.value.audit["status"] == "screen_failed"


def test_guard_reproducibility_does_not_modify_global_random_stream():
    df = fixture_frame()
    np.random.seed(912)
    before = np.random.get_state()
    first = screen_training_outcomes(df, "time", "event", ["x"])
    repeated = screen_training_outcomes(df, "time", "event", ["x"])
    after = np.random.get_state()
    assert first == repeated
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_wide_inputs_report_limited_neural_coverage():
    df = fixture_frame()
    rng = np.random.default_rng(2026101128)
    features = [f"x{i}" for i in range(70)]
    df = pd.concat([df, pd.DataFrame(rng.normal(size=(len(df), 70)), columns=features)], axis=1)
    audit = screen_training_outcomes(df, "time", "event", features)
    assert audit["status"] == "partially_assessed"
    assert any("coverage" in note for note in audit["notes"])


def test_prepared_deep_matrices_are_checked_before_neural_training(monkeypatch):
    import torch
    from survival_toolkit import deep_models
    df = fixture_frame()
    prepared = {"X_tensor": torch.tensor(df.event.to_numpy()[:, None], dtype=torch.float32),
                "time_tensor": torch.tensor(df.time.to_numpy(), dtype=torch.float32),
                "event_tensor": torch.tensor(df.event.to_numpy(), dtype=torch.float32)}
    split = {"train_idx": np.arange(180), "eval_idx": np.arange(180, 240), "evaluation_mode": "holdout"}
    with pytest.raises(OutcomeProxyError) as caught:
        deep_models.train_deepsurv(None, "time", "event", ["x"], prepared_data=prepared, evaluation_split=split)
    assert caught.value.audit["training_rows"] == 180


def test_reviewed_deep_cv_keeps_audit_and_fold_review():
    from survival_toolkit.deep_models import evaluate_single_deep_survival_model
    df = fixture_frame(120)
    df["x"] = df.event + .05 * df.x
    reason = "Enrollment-time assay; the source and acquisition dates were reviewed."
    result = evaluate_single_deep_survival_model("deepsurv", df=df, time_column="time", event_column="event",
        features=["x"], epochs=1, hidden_layers=[4], early_stopping_patience=None,
        evaluation_strategy="repeated_cv", cv_folds=2, cv_repeats=1, outcome_proxy_review=reason)
    assert result["input_audit"]["outcome_proxy"]["status"] == "reviewed_by_user"
    assert result["comparison_table"][0]["outcome_proxy_review"] == reason
    assert len(result["fold_results"]) == 2


def test_api_checkbox_does_not_clear_findings_and_error_is_structured():
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app, store
    df = fixture_frame()
    df["x"] = df.event + .05 * df.x
    stored = store.create(df, filename="fixture.csv", source="upload")
    client = TestClient(app, base_url="http://127.0.0.1")
    request = {"dataset_id": stored.dataset_id, "time_column": "time", "event_column": "event",
               "features": ["x"], "model_type": "compare", "predictor_availability_confirmed": True,
               "n_estimators": 10}
    response = client.post("/api/ml-model", json=request)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "outcome_proxy_review_required"
    assert detail["outcome_proxy"]["findings"]
    assert "paused" in detail["message"]

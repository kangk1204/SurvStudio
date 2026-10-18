"""Regression checks for interpretation boundaries without changing fitted scores."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from survival_toolkit import analysis
from survival_toolkit.prediction_interpretation import apply_prediction_interpretation
from survival_toolkit.prediction_guard import prediction_input_audit


def signature_frame():
    rng = np.random.default_rng(719)
    group = np.repeat([0, 1], 60)
    return pd.DataFrame({"time": rng.exponential(np.where(group, 2., 40.)),
                         "event": np.ones(120), "group": group,
                         "rare": np.arange(120) % 11 == 0})


def test_default_signature_search_never_confirms_without_permutations():
    _, _, result = analysis.discover_feature_signature(
        signature_frame(), "time", "event", ["group", "rare"],
        bootstrap_iterations=0, permutation_iterations=0, max_combination_size=1)
    assert result["best_split"]["P value"] < .001
    assert result["search_space"]["significant_signatures"] == 0
    assert all(row["Statistically significant"] is False for row in result["results_table"])
    assert result["search_space"]["permutation_required_for_significance"]
    assert result["signature_recipe"]["inference_status"] == "exploratory"
    assert result["derived_group"]["auto_apply_recommended"] is False


def test_bh_retains_size_feasible_rules_removed_by_event_filter(monkeypatch):
    original = analysis._screen_signature_combinations
    captured = {}
    def screen(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["family"] = len(result.family)
        captured["p"] = [row["P value"] for row in result.rows]
        return result
    monkeypatch.setattr(analysis, "_screen_signature_combinations", screen)
    frame = signature_frame()
    # A size-feasible subgroup with no events is retained in the family.
    frame.loc[frame.group == 0, "event"] = 0
    frame["other"] = np.arange(120) % 2
    _, _, result = analysis.discover_feature_signature(frame, "time", "event", ["group", "other"],
        max_combination_size=2, bootstrap_iterations=0, top_k=100)
    assert captured["family"] > len(captured["p"])
    # R p.adjust(c(.01,.04), 'BH', n=5) uses the full n in rank scaling.
    p = np.array(captured["p"])
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * captured["family"] / np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    expected = dict(zip(p[order], adjusted))
    for row in result["results_table"]:
        assert row["BH adjusted p"] == pytest.approx(expected[row["P value"]])
    assert result["search_space"]["bh_family_size"] == captured["family"]


def test_completed_search_permutation_can_pass_but_failed_draw_cannot():
    _, _, result = analysis.discover_feature_signature(signature_frame(), "time", "event", ["group"],
        max_combination_size=1, bootstrap_iterations=0, permutation_iterations=39)
    row = result["best_split"]
    assert row["Statistically significant"]
    assert row["Permutation p"] <= .05
    row = {**row, "Permutation skipped resamples": 1}
    assert not analysis._signature_is_significant(row, .05, True, False, .5, False, .6)


@pytest.mark.parametrize("mode,train,test", [("apparent", 1, 1), ("holdout", 9, 4), ("holdout", 30, 3)])
def test_high_score_is_preserved_but_not_ranked_with_sparse_or_apparent_evaluation(mode, train, test):
    result = {"evaluation_mode": mode, "comparison_table": [{"model": "Model", "rank": 1,
        "c_index": 1., "train_events": train, "test_events": test}], "ranking_complete": True,
        "scientific_summary": {"strengths": ["Excellent"], "headline": "Top Model", "metrics": []},
        "manuscript_tables": {"model_performance_table": [{"Model": "Model", "Rank": 1, "C-index": 1.}]}}
    apply_prediction_interpretation(result)
    row = result["comparison_table"][0]
    assert row["c_index"] == 1. and row["rank"] is None
    assert not row["comparable_for_ranking"] and not result["ranking_complete"]
    assert result["scientific_summary"]["status"] == "caution"
    assert not result["scientific_summary"]["strengths"]
    assert result["manuscript_tables"]["model_performance_table"][0]["Rank"] is None


def test_repeated_cv_uses_smallest_fold_not_rounded_mean():
    result = {"evaluation_mode": "repeated_cv", "comparison_table": [{"model": "Model", "rank": 1,
        "c_index": .8}], "fold_results": [{"model": "Model", "train_events": 25, "test_events": n} for n in [3, 15]]}
    apply_prediction_interpretation(result)
    assert result["comparison_table"][0]["rank"] is None


def test_adequate_held_out_score_and_rank_stay_unchanged():
    result = {"evaluation_mode": "holdout", "comparison_table": [{"model": "Model", "rank": 1,
        "c_index": .71, "train_events": 40, "test_events": 20}], "scientific_summary": {"headline": "original"}}
    apply_prediction_interpretation(result)
    assert result["comparison_table"][0]["rank"] == 1
    assert result["comparison_table"][0]["c_index"] == .71
    assert result["scientific_summary"]["headline"] == "original"


@pytest.mark.parametrize("values", [list(range(120)), [f"P{i}" for i in range(120)]])
def test_id_column_cannot_be_selected_as_a_predictor(values):
    frame = signature_frame().assign(patient_id=values)
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="identifier columns cannot"):
        prediction_input_audit(frame, "time", "event", ["patient_id"], 1)
    assert prediction_input_audit(frame, "time", "event", ["group"], 1)["analyzed_rows"] == 120
    pd.testing.assert_frame_equal(frame, before)


def test_one_event_api_keeps_scores_without_a_winner():
    pytest.importorskip("sksurv")
    from fastapi.testclient import TestClient
    from survival_toolkit.app import app, store
    rng = np.random.default_rng(72)
    frame = pd.DataFrame({"time": np.arange(1., 81.), "event": [1] + [0] * 79,
                          "marker": rng.normal(size=80)})
    item = store.create(frame, filename="one-event.csv", source="upload")
    response = TestClient(app, base_url="http://127.0.0.1").post("/api/ml-model", json={
        "dataset_id": item.dataset_id, "time_column": "time", "event_column": "event",
        "event_positive_value": 1, "features": ["marker"], "model_type": "compare", "n_estimators": 10,
        "predictor_availability_confirmed": True})
    assert response.status_code == 200, response.text
    result = response.json()["analysis"]
    assert result["comparison_table"]
    assert all(row["rank"] is None for row in result["comparison_table"])
    assert all(row["Rank"] is None for row in result["manuscript_tables"]["model_performance_table"])
    assert result["scientific_summary"]["status"] == "caution"
    assert not any("above chance" in s for s in result["scientific_summary"]["strengths"])

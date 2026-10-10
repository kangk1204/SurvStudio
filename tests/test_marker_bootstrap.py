import copy
import numpy as np
import pandas as pd
import pytest

from survival_toolkit.marker_bootstrap import (WaldDesign, diagnose_markers, joint_maxima,
    plus_one_interval, uniform_stream)
from survival_toolkit.marker_diagnostics import hc3_wald
from survival_toolkit.marker_evaluation import prepare_marker_cohort


def cohort():
    rng = np.random.default_rng(921)
    z = rng.normal(size=90)
    t = rng.exponential(size=90) / (.06 * np.exp(.8 * z))
    c = rng.exponential(size=90) / .04
    frame = pd.DataFrame(dict(time=np.minimum(t, c), event=(t <= c).astype(int), Z=z,
        X=.8*z+rng.normal(size=90), Y=.8*z+rng.normal(size=90)))
    return prepare_marker_cohort(frame, time_column="time", event_column="event",
        marker_columns=["X", "Y"], clinical_columns=["Z"])


def test_batched_hc3_matches_scalar():
    rng = np.random.default_rng(11)
    d = np.column_stack([np.ones(80), rng.normal(size=(80, 3))])
    y = rng.normal(size=(3, 4, 80))
    calculated = WaldDesign(d, 2).statistic(y)
    expected = np.array([[hc3_wald(one, d, 2)["statistic"] / 2 for one in draw] for draw in y])
    np.testing.assert_allclose(calculated, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("candidate", ["residual_vector", "restricted_wild"])
def test_chunks_and_marker_order_keep_common_draws(candidate):
    data = cohort()
    u = uniform_stream(32, 19, len(data.time))
    a = joint_maxima(data, data.markers, candidate=candidate, uniforms=u, draw_chunk=1, marker_chunk=1)
    b = joint_maxima(data, data.markers, candidate=candidate, uniforms=u, draw_chunk=8, marker_chunk=2)
    c = joint_maxima(data._replace(marker_names=data.marker_names[::-1]), data.markers[:, ::-1],
        candidate=candidate, uniforms=u, draw_chunk=5, marker_chunk=1)
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(a, c, atol=1e-10, rtol=1e-10)


def test_required_draw_failure_is_not_removed(monkeypatch):
    import survival_toolkit.marker_bootstrap as m
    monkeypatch.setattr(m, "joint_maxima", lambda *a, **k: (_ for _ in ()).throw(ValueError("draw failure")))
    data = cohort()
    result = diagnose_markers(data, data.markers, draws=19)
    assert not result["allowed"]
    assert result["bootstrap"]["status"] == "failed"
    assert all(t["p_maxT"] is None for t in result["residual_tests"])


def test_mc_boundary_withholds(monkeypatch):
    import survival_toolkit.marker_bootstrap as m
    data = cohort()
    original = m.legacy.diagnose_markers
    def without_clinical_failure(*a, **k):
        value = original(*a, **k)
        value["reasons"] = []
        return value
    monkeypatch.setattr(m.legacy, "diagnose_markers", without_clinical_failure)
    observed = max(t["statistic"] / t["df"] for t in original(data, data.markers)["residual_tests"] if t.get("status") == "calculated")
    maxima = np.full(9999, observed - 1.)
    maxima[:99] = observed + 1.
    monkeypatch.setattr(m, "joint_maxima", lambda *a, **k: maxima)
    result = diagnose_markers(data, data.markers)
    assert result["bootstrap"]["global_p"] == .01
    assert any(r.startswith("diagnostic_mc_uncertain") for r in result["reasons"])
    assert not result["allowed"]
    assert plus_one_interval(99, 9999)[0] < .01 < plus_one_interval(99, 9999)[1]


def test_wild_variance_null_changes_squared_response():
    data = cohort()
    u = uniform_stream(88, 30, len(data.time))
    maxima = joint_maxima(data, data.markers, candidate="restricted_wild", uniforms=u)
    assert np.std(maxima) > .01


def test_api_policy_defaults_and_unknown_rejected():
    from survival_toolkit.app import MarkerEvaluationRequest
    request = dict(dataset_id="synthetic", time_column="time", event_column="event", marker_columns=["X"])
    assert MarkerEvaluationRequest(**request).diagnostic_policy == "v2_holm"
    assert MarkerEvaluationRequest(**request, diagnostic_policy="v3_joint_bootstrap").marker_settings().diagnostic_bootstraps == 9999
    with pytest.raises(ValueError):
        MarkerEvaluationRequest(**request, diagnostic_policy="auto_best")


def test_v2_source_bound_qualification_is_preserved():
    from survival_toolkit.marker_qualification import qualification
    actual=qualification("linear","marker-inference/2")
    assert actual["status"]=="passed_supported_conditions_only"
    assert actual["kernel_sources_match"] is True
    assert qualification("linear","marker-inference/3")["status"]=="not_evaluated"


def test_v3_recipe_state_and_predictions_survive_external_validation():
    import survival_toolkit.marker_evaluation as old
    import survival_toolkit.marker_evaluation_v3 as new
    from survival_toolkit.marker_qualification import qualify_marker_result
    data=cohort()
    frame=pd.DataFrame(dict(time=data.time,event=data.event,Z=data.clinical_frame.Z,
        X=data.markers[:,0],Y=data.markers[:,1]))
    kwargs=dict(time_column="time",event_column="event",marker_columns=["X","Y"],clinical_columns=["Z"])
    options=dict(n_permutations=19,n_resamples=0,random_seed=33)
    previous=old.evaluate_markers(frame,**kwargs,settings=old.MarkerSettings(**options))
    result=qualify_marker_result(new.evaluate_markers(frame,**kwargs,settings=new.MarkerSettings(**options)))
    assert result["method_version"]=="marker-inference/3"
    assert result["inference"]["status"]=="withheld"
    assert all(row["added_value"]["p_value"] is None for row in result["marker_table"])
    assert all("exploratory" in row for row in result["marker_table"])
    recipe=result["locked_recipe"]
    assert recipe["recipe_version"]==3 and recipe["inference"]["bootstrap"]["draws"]==9999
    np.testing.assert_allclose(recipe["model"]["coefficients"],previous["locked_recipe"]["model"]["coefficients"],atol=1e-10)
    external=new.validate_locked_recipe(frame,recipe,n_bootstrap=0)
    assert external["inference"]["status"]=="withheld"
    historical=new.validate_locked_recipe(frame,previous["locked_recipe"],n_bootstrap=0)
    assert abs(external["metrics"]["c_index"]-historical["metrics"]["c_index"])<1e-10
    altered=copy.deepcopy(recipe);altered["inference"]["bootstrap"]["draws"]=1999
    altered["recipe_hash"]=new.recipe_hash(altered)
    with pytest.raises(ValueError): new.validate_locked_recipe(frame,altered,n_bootstrap=0)

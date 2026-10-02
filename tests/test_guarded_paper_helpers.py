import importlib.util
from pathlib import Path

from test_guarded_marker_inference import cohort,evaluate
from survival_toolkit.marker_evaluation import validate_locked_recipe

spec=importlib.util.spec_from_file_location('guarded_paper_common',Path(__file__).parents[1]/'paper/scripts/common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


def test_case_reference_scores_preserve_v3_splines_and_missing_values():
    recipe=evaluate(cohort(),'restricted_cubic_spline')['locked_recipe']
    external=cohort(12);external.loc[:11,'z']=float('nan')
    report=validate_locked_recipe(external,recipe,n_bootstrap=0)
    parts=common.locked_parts(external,recipe,report,'as_measured')
    assert len(parts['time'])==300==report['cohort']['n']
    assert parts['linear_predictor'].shape==(300,)

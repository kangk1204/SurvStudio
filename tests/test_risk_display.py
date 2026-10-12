import numpy as np
import pytest

from survival_toolkit.errors import UserInputError
from survival_toolkit.risk_display import describe_heldout_risk


def display(**changes):
    args = dict(training_risk=[0, 1, 2, 3], test_risk=[3, 3, 0, 1],
                time=[1, 2, 4, 5], event=[1, 0, 1, 1],
                source_rows=[8, 20, 9, 30])
    args.update(changes)
    return describe_heldout_risk(**args)


def test_groups_and_ranks_do_not_depend_on_test_outcomes():
    first = display()
    second = display(time=[9, 8, 1, 2], event=[0, 1, 1, 0])
    assert first['rows'] == second['rows']
    assert first['cutoff'] == second['cutoff'] == 1.5
    assert first['cutoff_optimized_on_outcomes'] is False
    assert first['km']['curves'] != second['km']['curves']


def test_threshold_is_training_median_and_source_rows_stay_aligned():
    result = display(test_risk=[9, 8, 0, -1])
    assert result['cutoff'] == 1.5  # Test median is 4.
    assert [r['source_row'] for r in result['rows']] == ['8', '20', '9', '30']
    tied = display()['rows']
    assert [r['test_risk_rank'] for r in tied] == [1, 1, 3, 4]
    assert tied[0]['source_row'] == '8' and tied[1]['source_row'] == '20'


def test_group_curves_do_not_export_association_tests():
    result = display()
    assert result['km']['test'] is None and result['km']['logrank_p'] is None
    assert 'rmst_contrast' not in result['km']
    assert 'scientific_summary' not in result['km']
    assert result['logrank_reported'] is False


def test_one_group_keeps_scores_and_explains_missing_curves():
    result = display(test_risk=[10, 10, 10, 10])
    assert len(result['rows']) == 4
    assert result['km'] is None
    assert 'one predicted risk group' in result['curve_error'].lower()


@pytest.mark.parametrize('changes', [dict(training_risk=[]),
    dict(training_risk=[np.nan]), dict(test_risk=[1, np.inf, 2, 3]),
    dict(source_rows=[1, 2])])
def test_invalid_scores_or_alignment_are_rejected(changes):
    with pytest.raises(UserInputError):
        display(**changes)

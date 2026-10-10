import pandas as pd
import pytest
from survival_toolkit.analysis import compute_km_analysis,_validate_endpoint_family_pair


@pytest.mark.parametrize('time,event',[
    ('rfs_months','dmfs_event'),('DMFS_months','RFS_status'),
    ('distant_metastasis_free_days','relapse_free_status')])
def test_recognized_rfs_and_dmfs_cannot_share_one_time_event_pair(time,event):
    with pytest.raises(ValueError,match='different survival endpoints'):
        _validate_endpoint_family_pair(time,event)


def test_matched_dmfs_and_generic_columns_remain_usable():
    _validate_endpoint_family_pair('dmfs_months','dmfs_status')
    _validate_endpoint_family_pair('time','event')
    frame=pd.DataFrame({'dmfs_months':[1.,2.,3.,4.],'dmfs_event':[1,0,1,0]})
    result=compute_km_analysis(frame,'dmfs_months','dmfs_event',event_positive_value=1)
    assert result['summary_table'][0]['Events']==2

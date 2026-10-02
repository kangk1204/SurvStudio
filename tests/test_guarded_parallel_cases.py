import importlib.util
import multiprocessing
import sys
from pathlib import Path
import numpy as np
import pytest
import survival_toolkit.marker_evaluation as engine
from test_guarded_marker_inference import cohort

spec=importlib.util.spec_from_file_location('parallel_case_resampling',Path(__file__).parents[1]/'validation/guarded_inference/parallel_case_resampling.py')
parallel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parallel;spec.loader.exec_module(parallel)


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(),reason='Private case acceleration requires fork')
@pytest.mark.parametrize('basis',['linear','restricted_cubic_spline'])
@pytest.mark.parametrize('nonlinear',[False,True])
def test_parallel_case_resampling_preserves_ordered_serial_procedure(basis,nonlinear,tmp_path,monkeypatch):
    data=cohort(n=180,nonlinear=nonlinear)
    prepared=engine.prepare_marker_cohort(data,time_column='time',event_column='event',marker_columns=['m0','m1','m2'],clinical_columns=['z'],clinical_basis=basis)
    settings=engine.MarkerSettings(n_permutations=19,n_resamples=6,random_seed=42,clinical_basis=basis)
    full=engine.run_procedure(prepared,np.arange(len(data)),settings)
    original=engine.run_procedure
    def occasionally_unestimable(c,rows,s):
        if int(rows.sum())%3==0: raise engine.ClinicalModelNotConvergedError('fixed synthetic failure')
        return original(c,rows,s)
    monkeypatch.setattr(engine,'run_procedure',occasionally_unestimable)
    serial=engine.resample_procedure(prepared,full,settings,np.random.default_rng(43),'added_value')
    faster=parallel.parallel_resample(prepared,full,settings,np.random.default_rng(43),'added_value',workers=2,progress=tmp_path)
    assert serial.n_valid>=2 and serial.n_failed>=1
    assert (serial.n_valid,serial.n_failed,serial.n_withheld)==(faster.n_valid,faster.n_failed,faster.n_withheld)
    for field in ('selection_frequency','direction_consistency','median_rank','rank_low','rank_high'):
        for lens in full.lenses: np.testing.assert_allclose(getattr(serial,field)[lens],getattr(faster,field)[lens],atol=0,rtol=0,equal_nan=True)
    assert faster.optimism==serial.optimism
    assert len(list(tmp_path.glob('*.json')))==6

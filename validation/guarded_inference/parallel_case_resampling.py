"""Fork-only case-study acceleration without changing the numerical procedure.

Draw all fixed subsample rows in parent order. Each worker calls the original
one-replicate engine, captures its unaggregated optimism inputs, and returns its
summary. Aggregate in original replicate order, including failed/withheld rows.
This helper is never used by the frozen confirmation study or the public API.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import survival_toolkit.marker_evaluation as engine

_ORIGINAL = engine.resample_procedure
_STATE = None


def _replicate(task):
    index, rows = task
    cohort, full, settings, primary = _STATE
    captured = []
    draw = engine._event_stratified_subsample
    aggregate = engine._optimism_summary
    engine._event_stratified_subsample = lambda *args: rows
    def capture(*args):
        captured.append(args)
        return aggregate(*args)
    engine._optimism_summary = capture
    started = time.time()
    try:
        result = _ORIGINAL(cohort, full, settings._replace(n_resamples=1), np.random.default_rng(0), primary)
        return index, result, captured[0], time.time()-started
    finally:
        engine._event_stratified_subsample = draw
        engine._optimism_summary = aggregate


def parallel_resample(cohort, full, settings, rng, primary, *, workers, progress=None):
    global _STATE
    if workers < 2:
        return _ORIGINAL(cohort, full, settings, rng, primary)
    if 'fork' not in multiprocessing.get_all_start_methods():
        raise ValueError('Case acceleration requires a fork runtime')
    rows = [engine._event_stratified_subsample(cohort.event, settings.resample_fraction, rng)
            for _ in range(int(settings.n_resamples))]
    _STATE = (cohort, full, settings, primary)
    progress = Path(progress) if progress else None
    if progress:
        progress.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        results = []
        for index, result, raw, elapsed in pool.map(_replicate, enumerate(rows)):
            results.append((result, raw))
            if progress:
                record = {'index':index, 'row_sha256':hashlib.sha256(np.asarray(rows[index],dtype='<i8').tobytes()).hexdigest(),
                          'n_valid':result.n_valid,'n_failed':result.n_failed,'n_withheld':result.n_withheld,'elapsed':elapsed}
                (progress/f'{index:04d}.json').write_text(json.dumps(record,sort_keys=True)+'\n')
    _STATE = None
    n_valid = sum(r.n_valid for r, _ in results)
    lenses = list(full.lenses); p = len(cohort.marker_names)
    frequency = {}; direction = {}; median = {}; lower = {}; upper = {}
    for lens in lenses:
        valid = [r for r, _ in results if r.n_valid]
        counts = sum((r.selection_frequency[lens] for r in valid), np.zeros(p))
        finite = sum((np.isfinite(r.direction_consistency[lens]) for r in valid), np.zeros(p))
        agree = sum((np.nan_to_num(r.direction_consistency[lens],nan=0.) for r in valid), np.zeros(p))
        frequency[lens] = counts/n_valid if n_valid else np.full(p,np.nan)
        direction[lens] = np.where(finite>0,agree/np.maximum(finite,1),np.nan)
        if valid:
            matrix = np.vstack([r.median_rank[lens] for r in valid]).astype(float)
            median[lens] = np.median(matrix,axis=0)
            lower[lens] = np.quantile(matrix,.025,axis=0)
            upper[lens] = np.quantile(matrix,.975,axis=0)
        else:
            median[lens] = lower[lens] = upper[lens] = np.full(p,np.nan)
    combined = [[] for _ in range(6)]
    for _, raw in results:
        for i, values in enumerate(raw): combined[i].extend(values)
    return engine.ResamplingSummary(n_valid=n_valid,n_failed=sum(r.n_failed for r,_ in results),
        selection_frequency=frequency,direction_consistency=direction,median_rank=median,rank_low=lower,rank_high=upper,
        optimism=engine._optimism_summary(*combined),n_withheld=sum(r.n_withheld for r,_ in results))


def install(workers,progress=None):
    engine.resample_procedure = lambda *args: parallel_resample(*args,workers=workers,progress=progress)

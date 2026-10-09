"""Deterministic synthetic recipe-to-figure reproduction on a fresh server."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from comparison import save,sha


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    from survival_toolkit.marker_evaluation import MarkerSettings,evaluate_markers,validate_locked_recipe
    from survival_toolkit.marker_qualification import qualify_marker_result,qualify_external_result
    from survival_toolkit.clinical_basis import transform_clinical_encoder
    from survival_toolkit.plots import build_marker_summary_figure
    rng=np.random.default_rng(2026100907)
    def data(n,shift):
        z=rng.normal(shift,size=n);e=rng.normal(size=(n,6));t=rng.exponential(size=n)/(.06*np.exp(.6*z+.9*e[:,0]));c=rng.exponential(size=n)/.04
        return pd.DataFrame(dict(time=np.minimum(t,c),event=(t<=c).astype(int),z=z,**{f'm{i}':.5*z+e[:,i] for i in range(6)}))
    a=data(100,0);b=data(65,.6)
    a.to_csv(output/'development.csv',index=False);b.to_csv(output/'external.csv',index=False)
    result=qualify_marker_result(evaluate_markers(a,time_column='time',event_column='event',marker_columns=[f'm{i}' for i in range(6)],clinical_columns=['z'],
        settings=MarkerSettings(n_permutations=999,n_resamples=3,random_seed=2026100907)))
    recipe=result['locked_recipe']
    if recipe is None:raise ValueError('Fixed synthetic reproduction has no estimable recipe')
    external=qualify_external_result(validate_locked_recipe(b,recipe,n_bootstrap=20,random_seed=2026100907),recipe)
    save(output/'analysis.json',result);save(output/'recipe.json',recipe);save(output/'external.json',external)
    save(output/'marker-figure.json',build_marker_summary_figure(result))
    design=transform_clinical_encoder(b,recipe['clinical']['encoder'],output='dataframe')
    for marker in recipe['markers']:design[marker]=b[marker].fillna(recipe['marker_medians'][marker])
    lp=design[recipe['model']['terms']].to_numpy()@np.asarray(recipe['model']['coefficients'])
    table=pd.DataFrame(dict(row=range(len(lp)),fixed_lp=lp));table.to_csv(output/'figure-source.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,3));ax.plot(table.row,table.fixed_lp,'.',color='#0072B2');ax.set(xlabel='Synthetic external row',ylabel='Fixed linear predictor')
    fig.tight_layout();fig.savefig(output/'figure.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(output/'figure.png',dpi=180);plt.close(fig)
    files={p.name:sha(p) for p in output.glob('*') if p.is_file()}
    save(output/'manifest.json',dict(files=files,source_sha256=sha(__file__),host=platform.node(),python=platform.python_version(),
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','matplotlib']},
        method='v2 default synthetic reproduction; 999 permutations and 20 external draws are implementation checks, not a qualification study',
        source_files={str(p.relative_to(Path(__file__).resolve().parents[2])):sha(p) for p in (Path(__file__).resolve().parents[2]/'src/survival_toolkit').glob('*.py')}))
    print(json.dumps({'host':platform.node(),'outputs':len(files)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)

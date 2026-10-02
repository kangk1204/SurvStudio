"""Publish a fixed allowlist of synthetic and aggregate evidence, never real-case rows."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import io
from reanalyse_cases import save


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(evidence,output):
    evidence=Path(evidence).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((evidence/'confirmation-summary.json').read_text())
    if not summary['complete']: raise ValueError('Every fixed confirmation index must be accounted for')
    for name in ('confirmation-summary.json','freeze.json','development-seal-summary.json','development-review.md','confirmation-failures.json'):
        target=out/name
        if target.exists() and digest(target)!=digest(evidence/name): raise ValueError('Preserve existing evidence '+name)
        shutil.copyfile(evidence/name,target)
    flat=[]
    for row in summary['summaries']:
        item={k:v for k,v in row.items() if k not in ('source_hashes',)}
        for k,v in list(item.items()):
            if isinstance(v,list):
                del item[k];item[k+'_lower']=v[0];item[k+'_upper']=v[1]
        flat.append(item)
    keys=list(dict.fromkeys(k for row in flat for k in row))
    with (out/'confirmation-cells.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(flat)
    fields=['stage','condition','n','p','index','method','host','worker','allowed','calculation_failed','diagnostic_failed',
            'raw_null_rejected','guarded_null_rejected','raw_true_rejected','guarded_true_rejected','n_true','error']
    ownership=[];count=0;seen=set();failed=0
    with (out/'replicate-outcomes.csv.gz').open('wb') as file:
        with gzip.GzipFile(filename='',fileobj=file,mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fields);writer.writeheader()
                for path in sorted((evidence/'confirmation').rglob('*.sqlite')):
                    db=sqlite3.connect(path);cfg=json.loads(db.execute("select value from metadata where key='configuration'").fetchone()[0])
                    worker=int(re.search(r'worker(\d+)$',path.stem)[1]);n_records=0;hosts=set()
                    for index,host,text in db.execute('select idx,host,result from replicates order by idx'):
                        identity=(cfg['stage'],cfg['condition'],cfg['n'],cfg['p'],index)
                        if identity in seen: raise ValueError('Duplicate fixed index')
                        seen.add(identity);n_records+=1;hosts.add(host)
                        for row in json.loads(text):
                            error=row.get('failure',False);failed+=int(error);count+=1
                            writer.writerow({'stage':cfg['stage'],'condition':cfg['condition'],'n':cfg['n'],'p':cfg['p'],'index':index,
                                'method':row['method'],'host':host,'worker':worker,'allowed':int(row['allowed'] and not error),
                                'calculation_failed':int(error),'diagnostic_failed':int(row.get('diagnostic_failed',False)),
                                'raw_null_rejected':None if error else int(row.get('raw_null_rejected',row['null_rejected'])),
                                'guarded_null_rejected':int(row['null_rejected']),'raw_true_rejected':None if error else row.get('raw_true_rejected',row['true_rejected']),
                                'guarded_true_rejected':row['true_rejected'],'n_true':row['n_true'],'error':row.get('error','')})
                    db.close()
                    ownership.append({'ledger':str(path.relative_to(evidence/'confirmation')),'sha256':digest(path),'worker':worker,
                                      'stage':cfg['stage'],'condition':cfg['condition'],'n':cfg['n'],'p':cfg['p'],'records':n_records,'hosts':';'.join(sorted(hosts))})
    if len(seen)!=85500 or count!=256500: raise ValueError('Incomplete fixed coverage')
    with (out/'ownership-ledgers.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(ownership[0]));writer.writeheader();writer.writerows(ownership)
    for (directory,destination_name),allowlist in {
        ('independent_R_latest','independent_R'):['verification.json','R_session.txt','training.csv','external.csv','knots.csv','scaling.csv',
                         'r_HC3.csv','r_classic_ph.csv','r_coefficients.csv','r_external_basis.csv','r_external_prediction.csv',
                         'r_functional_LR.csv','r_linear_coefficients.csv','r_metrics.csv','r_training_basis.csv'],
        ('tool_comparison_qualified_v2','tool_comparison'):['A.csv','B.csv','duplicate.csv','misspecified.csv','recipe.json','survstudio-task-analysis.json',
                           'survstudio-external.json','survstudio-api-result.json','numerical-comparison.csv','r_values.csv',
                           'r_locked_prediction.csv','task-comparison.csv','verification.json','R_session.txt'],
        ('tool_benchmark','tool_benchmark'):['verification.json','timing-memory-summary.csv','SurvStudio.csv','R_survival.csv',
                          'SurvStudio.csv.coefficients.csv','R_survival.csv.coefficients.csv',
                          'SurvStudio-peak-rss-kb.txt','R_survival-peak-rss-kb.txt']}.items():
        destination=out/destination_name;destination.mkdir(exist_ok=True)
        for name in allowlist:
            source=evidence/directory/name
            if not source.exists(): raise FileNotFoundError(source)
            shutil.copyfile(source,destination/name)
    for p in (evidence/'confirmation').glob('*/source-isolation.json'):
        destination=out/'source-isolation';destination.mkdir(exist_ok=True)
        shutil.copyfile(p,destination/(p.parent.name+'.json'))
    manifest={'scope':'synthetic fixtures, per-replicate aggregate outcomes and provenance only',
              'patient_level_case_data_included':False,'manuscript_included':False,
              'frozen_study_git_revision':json.loads((out/'freeze.json').read_text())['git_revision'],
              'fixed_datasets':len(seen),'paired_method_records':count,'calculation_failures':failed,
              'ledger_files':len(ownership),'model_release_status':summary['model_release_status'],
              'note':'confirmation-summary.json is retained byte-for-byte; its generic pending note is a template, while complete and missing-index fields establish final status.',
              # Derived audit outputs refer to this manifest and must not create a hash cycle.
              'audit_outputs_excluded':['aggregate-audit.json','independent-R-aggregate-values.csv',
                                        'independent-R-aggregate-comparison.csv','aggregate-R-session.txt'],
              'files':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in
                       ('aggregate-manifest.json','aggregate-audit.json','independent-R-aggregate-values.csv',
                        'independent-R-aggregate-comparison.csv','aggregate-R-session.txt')}}
    save(out/'aggregate-manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','note')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.evidence,a.output)

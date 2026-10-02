"""Seal a reviewed development specification before confirmation; refuse stale R evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from study import ROOT, hashes

p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--development',required=True)
p.add_argument('--output',required=True);p.add_argument('--review-record',required=True);a=p.parse_args()
reference=json.loads(Path(a.reference).read_text());development=json.loads(Path(a.development).read_text())
if not reference.get('passed'): raise SystemExit('Independent R reference failed')
for name,expected in reference['source_hashes'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise SystemExit('R reference source is stale: '+name)
sources=hashes()
if any(row['source_hashes']!=sources for row in development['summaries']): raise SystemExit('Development source is stale')
if len(development['summaries'])!=36 or any(row['stage']!='development' or row['completed']<50 for row in development['summaries']):
    raise SystemExit('All 12 development conditions require at least 50 fixed replicates')
revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain',*sources],text=True).strip():
    raise SystemExit('Commit the numerical study sources before sealing')
out=Path(a.output)
if out.exists(): raise SystemExit('A sealed manifest cannot be overwritten')
manifest={'git_revision':revision,'source_hashes':sources,'python_version':platform.python_version(),
          'package_versions':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','statsmodels')},
          'reference_verification':reference,'development_review_complete':True,
          'development_summary_sha256':hashlib.sha256(Path(a.development).read_bytes()).hexdigest(),
          'development_review_record':Path(a.review_record).read_text(),
          'confirmation_policy':'No threshold, method, condition or supported-scenario changes after confirmation inspection. No replacement of failed fixed indexes.'}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'frozen_manifest':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'revision':revision}))

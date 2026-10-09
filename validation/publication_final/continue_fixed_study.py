"""Run only demonstrably unattempted indices with the original sealed kernel.

Original ledgers remain read-only. A sidecar preserves the original logical
owner (index modulo 48), while recording its new physical host. Completed,
failed, unreadable and uncertain original attempts cannot enter the plan.
This operational wrapper changes no DGP, statistic, draw count or gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def untouched_extension_indices(audit):
    excluded = {(r['condition'], r['n'], r['p'], r['idx'])
                for r in audit['unresolved_original_attempts'] + audit['invalid_records']}
    absent = {tuple(r) for r in audit['absent_indices']}
    if len(absent) != audit['absent_rows']:
        raise ValueError('Absent index count or uniqueness mismatch')
    return sorted(absent - excluded)


def prepare(args):
    if not 1 <= args.workers <= 12:
        raise ValueError('This measured single-host schedule permits 1–12 concurrent workers')
    audit = json.loads(args.audit.read_text())
    cfg = json.loads(args.protocol.read_text())
    main = json.loads(args.main.read_text())
    pilot = json.loads(args.pilot.read_text())
    owners = json.loads(args.owner_audit.read_text())
    if not main['complete'] or not pilot['complete']:
        raise ValueError('Observed complete cost/main evidence required')
    durations = {}
    for result in (main, pilot):
        for row in result['summaries']:
            key = (row['n'], row['p'])
            durations[key] = max(durations.get(key, 0), row['elapsed_p95_seconds'] or 0)
    # The extension uses the worst observed owner maximum, not a selected mean.
    extension_times = [o['elapsed_max'] for h in owners['hosts'] for o in h['owners']
                       if o['stage'] == 'extension' and o.get('elapsed_max')]
    durations[(500, 30)] = max(durations[(500, 30)], max(extension_times))
    extension = untouched_extension_indices(audit)
    planned = {'extension': extension, 'large': [], 'stress': []}
    if any(audit['other_stages'][s]['dispatched'] for s in ('large', 'stress')):
        raise ValueError('An existing large/stress dispatch must first be audited')
    for cell in cfg['confirmation']:
        if cell['stage'] in ('large', 'stress'):
            planned[cell['stage']].extend((c, cell['n'], cell['p'], i)
                                         for c in cell['conditions'] for i in range(cell['replicates']))
    seconds = 0.
    for stage, indices in planned.items():
        for condition, n, p, _ in indices:
            # The pilot has one clinical variable: conservatively double this
            # cell's observed cost for the fixed two-variable stress condition.
            seconds += durations[(n, p)] * (2 if condition == 'two_clinical' else 1)
    days = 2 * seconds / args.workers / 86400
    result = dict(version='fixed-study-continuation/1', prepared_utc=now(),
                  status='feasible' if days <= 21 else 'v2_fallback_resource_limit',
                  conservative_days=days, safety_factor=2, concurrent_workers=args.workers,
                  owners=48, indices=planned, counts={s:len(v) for s,v in planned.items()},
                  excluded_uncertain_original=audit['unresolved_original_attempts'],
                  excluded_unreadable_original=audit['invalid_records'],
                  observed_seconds={f'{n},{p}':v for (n,p),v in durations.items()},
                  limitation='Observed timing is an estimate, not a runtime guarantee. Counts, draws and gates remain fixed.',
                  inputs={str(p):sha(p) for p in (args.audit,args.protocol,args.main,args.pilot,args.owner_audit)},
                  originals_read_only=True, no_original_attempt_replacement=True)
    write_new(args.output, result)
    print(json.dumps({k:result[k] for k in ('status','conservative_days','counts')}))


def load_kernel(source, freeze):
    sys.path.insert(0, str(source/'src'))
    sys.path.insert(0, str(source/'validation/publication_v3'))
    spec = importlib.util.spec_from_file_location('sealed_fixed_study', source/'validation/publication_v3/study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    seal = json.loads(freeze.read_text())
    if study.hashes() != seal['source_hashes'] or study.environment() != seal['environment']:
        raise ValueError('Original numerical source/environment seal mismatch')
    if not all(seal[k] is True for k in ('selection_eligible','reference_passed','source_ci_passed')):
        raise ValueError('Original seal prerequisite missing')
    return study, seal


def worker(args):
    plan = json.loads(args.plan.read_text())
    if plan['status'] != 'feasible':
        raise ValueError('Cost gate does not permit execution')
    if not 0 <= args.owner < plan['owners']:
        raise ValueError('Invalid original logical owner')
    study, seal = load_kernel(args.source, args.freeze)
    indices = [tuple(k) for k in plan['indices'][args.stage] if k[3] % plan['owners'] == args.owner]
    if args.ledger.exists():
        raise ValueError('A sidecar ledger already exists: no automatic resume/replacement')
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    seed = study.protocol()[('extension' if args.stage == 'large' else args.stage)+'_seed']
    configuration = dict(stage=args.stage, seed=seed, owner=args.owner, owners=plan['owners'],
        host=__import__('socket').gethostname(), source_hashes=study.hashes(), environment=study.environment(),
        freeze_sha256=sha(args.freeze), candidate=seal['candidate'],
        continuation_plan_sha256=sha(args.plan), continuation_wrapper_sha256=sha(__file__),
        original_logical_owner_preserved=True)
    with sqlite3.connect(args.ledger) as db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        db.execute('INSERT INTO metadata VALUES(?,?)', ('configuration',json.dumps(configuration,sort_keys=True)))
        db.execute('CREATE TABLE attempts(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,started_utc TEXT NOT NULL,ended_utc TEXT,status TEXT NOT NULL,error TEXT,PRIMARY KEY(condition,n,p,idx))')
        db.execute('CREATE TABLE replicates(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,elapsed REAL,result TEXT,PRIMARY KEY(condition,n,p,idx))')
        db.commit()
        for key in indices:
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,NULL,'started',NULL)", (*key,now()))
            db.commit()
            begin = time.perf_counter()
            try:
                rows = study.paired(args.stage,*key[:3],seed,key[3],seal['candidate'])
                raw = json.dumps(rows)
                db.execute('INSERT INTO replicates VALUES(?,?,?,?,?,?)', (*key,time.perf_counter()-begin,raw))
                status = 'completed_with_scientific_failure' if any(r.get('failure') or r.get('diagnostic_failure') for r in rows) else 'completed'
                db.execute('UPDATE attempts SET ended_utc=?,status=? WHERE condition=? AND n=? AND p=? AND idx=?',(now(),status,*key))
                db.commit()
                stored = db.execute('SELECT result FROM replicates WHERE condition=? AND n=? AND p=? AND idx=?',key).fetchone()[0]
                if stored != raw:
                    raise ValueError('Stored paired record differs from serialized output; preserve it without replacement')
            except BaseException as exc:
                db.rollback()
                # A committed but subsequently unreadable row stays committed.
                if not db.execute('SELECT 1 FROM replicates WHERE condition=? AND n=? AND p=? AND idx=?',key).fetchone():
                    db.execute("UPDATE attempts SET ended_utc=?,status='interrupted',error=? WHERE condition=? AND n=? AND p=? AND idx=?",(now(),type(exc).__name__+': '+str(exc),*key))
                    db.commit()
                raise


def resources():
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(utc=now(), cpus=os.cpu_count(), load1=os.getloadavg()[0],
        available_bytes=int(mem['MemAvailable'].split()[0])*1024,
        free_disk_bytes=shutil.disk_usage('/home/keunsoo/projects').free)


def launch(args):
    import fcntl
    plan = json.loads(args.plan.read_text())
    if plan['status'] != 'feasible':
        raise ValueError('The fixed 21-day cost gate failed')
    load_kernel(args.source,args.freeze)
    args.output.mkdir(exist_ok=False)
    lock = (args.output/'controller.lock').open('x')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs = [(stage,owner) for stage in ('extension','large','stress') for owner in range(plan['owners'])
            if any(k[3] % plan['owners'] == owner for k in plan['indices'][stage])]
    write_new(args.output/'ownership.json',dict(host=__import__('socket').gethostname(),jobs=jobs,
        logical_owners=48,concurrent_workers=plan['concurrent_workers'],plan_sha256=sha(args.plan),
        source_seal_sha256=sha(args.freeze),wrapper_sha256=sha(__file__),resources=resources()))
    def run(job):
        stage,owner=job
        while True:
            measured=resources()
            if measured['available_bytes'] >= 2*plan['concurrent_workers']*1024**3 and measured['free_disk_bytes'] >= 20*1024**3 and measured['load1'] < measured['cpus']:
                break
            time.sleep(15)
        folder=args.output/stage;folder.mkdir(exist_ok=True)
        stem=f'owner-{owner:02d}'
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--source',str(args.source),
                 '--freeze',str(args.freeze),'--plan',str(args.plan),'--stage',stage,'--owner',str(owner),
                 '--ledger',str(folder/(stem+'.sqlite'))]
        status=dict(stage=stage,owner=owner,started_utc=now(),command=command,resources=measured)
        write_new(folder/(stem+'.started.json'),status)
        with (folder/(stem+'.log')).open('xb') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            write_new(folder/(stem+'.pid.json'),dict(pid=proc.pid,command=command))
            returncode=proc.wait()
        write_new(folder/(stem+'.done.json'),{**status,'ended_utc':now(),'returncode':returncode})
        return dict(stage=stage,owner=owner,returncode=returncode)
    with ThreadPoolExecutor(max_workers=plan['concurrent_workers']) as pool:
        results=list(pool.map(run,jobs))
    write_new(args.output/'completion.json',dict(completed_utc=now(),jobs=results,
        all_workers_exited_successfully=all(r['returncode']==0 for r in results),
        adoption_status='NO_GO: completed main sensitivity gate failed; no promotion by continuation'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan')
    for name in ('audit','protocol','main','pilot','owner-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    for name in ('worker','launch'):
        p=sub.add_parser(name)
        for flag in ('source','freeze','plan'):
            p.add_argument('--'+flag,type=Path,required=True)
        if name=='launch':p.add_argument('--output',type=Path,required=True)
        else:
            p.add_argument('--stage',choices=['extension','large','stress'],required=True)
            p.add_argument('--owner',type=int,required=True);p.add_argument('--ledger',type=Path,required=True)
    arguments=parser.parse_args()
    {'plan':prepare,'worker':worker,'launch':launch}[arguments.command](arguments)

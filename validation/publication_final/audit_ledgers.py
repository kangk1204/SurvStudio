"""Read-only audit of damaged original ledgers, with all planned denominators.

This descriptive audit is separate from the sealed confirmation summarizer.
It never repairs bytes, salvages part of a paired replicate, or repeats an index.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_record(raw, methods):
    """Reject a whole paired row when strict decoding or its schema fails."""
    rows = json.loads(raw.decode('utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))
    if not isinstance(rows, list) or len(rows) != len(methods):
        raise ValueError('paired method count')
    if {r.get('method') for r in rows} != set(methods):
        raise ValueError('missing or duplicate method')
    for r in rows:
        for key in ('failure', 'allowed', 'null_rejected'):
            if type(r.get(key)) is not bool:
                raise ValueError('missing or nonboolean '+key)
        for key in ('true_rejected', 'n_true'):
            if type(r.get(key)) is not int or r[key] < 0:
                raise ValueError('invalid '+key)
        if r['true_rejected'] > r['n_true']:
            raise ValueError('signal count exceeds denominator')
        if (r['failure'] or not r['allowed']) and (r['null_rejected'] or r['true_rejected']):
            raise ValueError('withheld/failed row contains a standard rejection')
        for key in ('diagnostic_failure', 'raw_null_rejected'):
            if key in r and type(r[key]) is not bool:
                raise ValueError('nonboolean '+key)
        for key in ('raw_seconds', 'diagnostic_seconds'):
            if key in r and (not isinstance(r[key], (float, int)) or not math.isfinite(r[key]) or r[key] < 0):
                raise ValueError('invalid '+key)
        if not r['failure'] and r['method'] != 'legacy_linear':
            for key in ('raw_null_rejected', 'raw_true_rejected', 'diagnostic_failure', 'reasons'):
                if key not in r:
                    raise ValueError('missing '+key)
    return rows


def audit(ledgers, output, protocol_path=ROOT/'validation/publication_v3/protocol.json'):
    cfg = json.loads(Path(protocol_path).read_text())
    methods = cfg['methods']
    expected = {(c, cell['n'], cell['p'], i) for cell in cfg['confirmation'] if cell['stage']=='extension'
                for c in cell['conditions'] for i in range(cell['replicates'])}
    output.mkdir(parents=True, exist_ok=False)
    records, seen, attempts, invalid, manifests = {}, set(), {}, [], []
    common = None
    for path in sorted(ledgers):
        before = digest(path)
        with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro', uri=True) as db:
            db.text_factory = bytes
            integrity = [v[0].decode('utf-8') for v in db.execute('PRAGMA integrity_check')]
            if integrity != ['ok']:
                raise ValueError('SQLite structural integrity failed: '+str(path))
            configuration = json.loads(db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()[0])
            keys = ('stage','seed','owners','source_hashes','environment','freeze_sha256','candidate')
            comparable = {k:configuration[k] for k in keys}
            if comparable['stage']!='extension': raise ValueError('Expected original extension ledger')
            if common is None: common = comparable
            elif common != comparable: raise ValueError('Mixed original configurations')
            local_attempts = {}
            for condition,n,p,index,started,ended,status,error in db.execute('SELECT * FROM attempts'):
                key = (condition.decode('utf-8'),n,p,index)
                if key not in expected or index%configuration['owners']!=configuration['owner'] or key in attempts:
                    raise ValueError('Unexpected/duplicate/wrong-owner attempt')
                start = datetime.fromisoformat(started.decode('utf-8'))
                end = datetime.fromisoformat(ended.decode('utf-8')) if ended is not None else None
                if start.utcoffset()!=timezone.utc.utcoffset(start) or end is not None and (end.utcoffset()!=timezone.utc.utcoffset(end) or end<start):
                    raise ValueError('Invalid original UTC timestamps')
                row = dict(status=status.decode('utf-8'),started_utc=start.isoformat(),ended_utc=None if end is None else end.isoformat())
                attempts[key]=row; local_attempts[key]=row
            for condition,n,p,index,elapsed,raw in db.execute('SELECT * FROM replicates'):
                key = (condition.decode('utf-8'),n,p,index)
                if key not in expected or index%configuration['owners']!=configuration['owner'] or key in seen:
                    raise ValueError('Unexpected/duplicate/wrong-owner replicate')
                seen.add(key)
                try:
                    rows = decode_record(raw,methods)
                    original = local_attempts[key]
                    if original['status'] not in ('completed','completed_with_scientific_failure') or original['ended_utc'] is None:
                        raise ValueError('Committed replicate has no original completed attempt')
                    if not math.isfinite(elapsed) or elapsed<0: raise ValueError('invalid elapsed time')
                except (UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    invalid.append(dict(condition=key[0],n=n,p=p,idx=index,owner=configuration['owner'],
                        ledger=path.name,raw_result_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                        reason=type(exc).__name__+': '+str(exc),original_attempt=local_attempts.get(key)))
                else: records[key] = {r['method']:r for r in rows}
        if digest(path)!=before: raise ValueError('Read-only audit changed an original copy')
        manifests.append(dict(file=path.name,sha256=before,owner=configuration['owner'],integrity=integrity))
    if not manifests: raise ValueError('No original ledgers')
    summaries, flat = [], []
    for cell in cfg['confirmation']:
        if cell['stage']!='extension': continue
        count=cell['replicates']
        for condition in cell['conditions']:
            data=[(i,records[(condition,cell['n'],cell['p'],i)]) for i in range(count) if (condition,cell['n'],cell['p'],i) in records]
            missing=count-len(data)
            for method in methods:
                rows=[r[method] for _,r in data]
                failures=sum(r['failure'] for r in rows);unknown=missing+failures
                known=[r for r in rows if not r['failure']]
                allowed=[r for r in known if r['allowed']]
                hits=sum(r['null_rejected'] for r in known)
                raw_hits=sum(r.get('raw_null_rejected',r['null_rejected']) for r in known)
                conditional=sum(r['null_rejected'] for r in allowed)
                a=len(allowed)
                row=dict(stage='extension',condition=condition,n=cell['n'],p=cell['p'],method=method,
                    planned=count,valid_paired_rows=len(data),unusable_or_absent=missing,calculation_failures=failures,
                    diagnostic_failures=sum(r.get('diagnostic_failure',False) for r in rows),allowed=a,
                    allowance_bounds=[a/count,(a+unknown)/count],raw_fwer_bounds=[raw_hits/count,(raw_hits+unknown)/count],
                    fwer_bounds=[hits/count,(hits+unknown)/count],
                    observed_conditional_fwer=None if not a else conditional/a,
                    conditional_fwer_failure_bounds=None if not (a+unknown) else [conditional/(a+unknown),(conditional+unknown)/(a+unknown)],
                    withheld_reason_dataset_counts=dict(Counter(k for r in rows for k in r.get('reasons',{}))),
                    boundary='Descriptive bounds on planned indices; no requalification, no imputation or replacement')
                summaries.append(row)
                for i,r in data:
                    v=r[method]
                    flat.append(dict(condition=condition,n=cell['n'],p=cell['p'],idx=i,method=method,
                        failure=int(v['failure']),allowed=int(v['allowed']),null_rejected=int(v['null_rejected']),
                        raw_null_rejected=int(v.get('raw_null_rejected',v['null_rejected'])),diagnostic_failure=int(v.get('diagnostic_failure',False))))
    absent=sorted(expected-seen)
    result=dict(audit_version='original-ledger-audit/1',status='INCOMPLETE_CORRUPTED_EVIDENCE' if invalid else 'INCOMPLETE',
        audit_utc=datetime.now(timezone.utc).isoformat(),planned=len(expected),committed_rows=len(seen),
        valid_paired_rows=len(records),invalid_paired_rows=len(invalid),absent_rows=len(absent),
        attempt_status_counts=dict(Counter(a['status'] for a in attempts.values())),
        unresolved_original_attempts=[dict(condition=k[0],n=k[1],p=k[2],idx=k[3],**a) for k,a in attempts.items() if k not in seen],
        invalid_records=invalid,absent_indices=[list(k) for k in absent],summaries=summaries,ledgers=manifests,
        configuration=common,original_copies_unchanged=True,no_worker_relaunches=True,
        source_sha256=digest(__file__),protocol_sha256=digest(protocol_path),
        sealed_summary='failed strict UTF-8 decoding; this audit is a separate descriptive output',
        other_stages={'large':{'planned':1500,'dispatched':False,'result_bounds':[0,1]},
                      'stress':{'planned':16000,'dispatched':False,'result_bounds':[0,1]}},
        adoption_status='NO_GO based on completed main sensitivity criterion; unresolved extensions cannot reverse it')
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'valid-paired-decisions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]) if flat else ['condition','n','p','idx','method','failure','allowed','null_rejected','raw_null_rejected','diagnostic_failure'])
        writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:result[k] for k in ('status','planned','committed_rows','valid_paired_rows','invalid_paired_rows','absent_rows')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ledgers',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.ledgers,args.output)

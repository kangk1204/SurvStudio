"""Evidence-integrity failures cannot be silently dropped or replaced."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import pytest

DIRECTORY=Path(__file__).resolve().parents[1]/'validation/publication_final'


def module(name):
    spec=importlib.util.spec_from_file_location(name,DIRECTORY/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_corrupt_original_row_is_unresolved_without_partial_salvage(tmp_path):
    audit=module('audit_ledgers');methods=['legacy_linear','v2_linear']
    cfg=dict(methods=methods,confirmation=[dict(stage='extension',n=10,p=2,replicates=2,conditions=['independent'])])
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(cfg))
    ledger=tmp_path/'owner.sqlite'
    r=dict(method='legacy_linear',failure=False,allowed=True,null_rejected=True,true_rejected=0,n_true=0)
    c={**r,'method':'v2_linear','raw_null_rejected':True,'raw_true_rejected':0,'diagnostic_failure':False,'reasons':{}}
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
        db.execute('INSERT INTO metadata VALUES(?,?)',('configuration',json.dumps(dict(stage='extension',seed=1,owners=1,owner=0,source_hashes={},environment={},freeze_sha256='fixture',candidate='restricted_wild'))))
        db.execute('CREATE TABLE attempts(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,started_utc TEXT,ended_utc TEXT,status TEXT,error TEXT)')
        db.execute('CREATE TABLE replicates(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,elapsed REAL,result TEXT)')
        for i,raw in enumerate((json.dumps([r,c]).encode(),b'[{"method":"legacy_linear"},\xff]')):
            db.execute('INSERT INTO replicates VALUES(?,?,?,?,?,?)',('independent',10,2,i,1.,raw))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,NULL)',('independent',10,2,i,'2026-10-09T00:00:00+00:00','2026-10-09T00:00:01+00:00','completed'))
    before=audit.digest(ledger);result=audit.audit([ledger],tmp_path/'audit',protocol)
    assert audit.digest(ledger)==before
    assert result['valid_paired_rows']==1 and result['invalid_paired_rows']==1
    assert result['absent_rows']==0 and result['no_worker_relaunches']
    assert result['summaries'][0]['fwer_bounds']==[.5,1.]
    assert result['summaries'][0]['conditional_fwer_failure_bounds']==[.5,1.]
    with pytest.raises(FileExistsError):audit.audit([ledger],tmp_path/'audit',protocol)


def test_missing_boolean_and_withheld_rejection_are_not_valid_rows():
    audit=module('audit_ledgers')
    with pytest.raises(ValueError):audit.decode_record(b'[{"method":"legacy_linear"}]',['legacy_linear'])
    r=dict(method='legacy_linear',failure=False,allowed=False,null_rejected=True,true_rejected=0,n_true=0)
    with pytest.raises(ValueError,match='withheld'):audit.decode_record(json.dumps([r]).encode(),['legacy_linear'])


def test_frozen_comparison_rejects_input_edit(tmp_path):
    comparison=module('comparison');p=tmp_path/'A.csv';p.write_text('event,time\n1,2\n')
    contract=dict(input_hashes={'A.csv':comparison.sha(p)},source_hashes={'validation/publication_final/comparison.py':comparison.sha(DIRECTORY/'comparison.py')})
    (tmp_path/'contract.json').write_text(json.dumps(contract));comparison.verify_inputs(tmp_path)
    p.write_text('event,time\n0,2\n')
    with pytest.raises(ValueError,match='input changed'):comparison.verify_inputs(tmp_path)


def test_continuation_excludes_uncertain_unreadable_and_completed_indices():
    continuation=module('continue_fixed_study')
    audit=dict(absent_indices=[['independent',500,30,2],['independent',500,30,3]],absent_rows=2,
        unresolved_original_attempts=[dict(condition='independent',n=500,p=30,idx=2)],
        invalid_records=[dict(condition='independent',n=500,p=30,idx=1)])
    assert continuation.untouched_extension_indices(audit)==[('independent',500,30,3)]
    audit['absent_indices'].append(['independent',500,30,3])
    audit['absent_rows']=3
    with pytest.raises(ValueError,match='uniqueness'):continuation.untouched_extension_indices(audit)

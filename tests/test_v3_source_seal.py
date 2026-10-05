"""Confirmation logging preserves original attempts and incomplete evidence."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

DIRECTORY=Path(__file__).resolve().parents[1]/"validation/publication_v3"


def load(name):
    spec=importlib.util.spec_from_file_location("staged_"+name,DIRECTORY/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(monkeypatch):
    m=load("study");cfg=m.protocol()
    cfg["screen"]={"cells":[[40,2]],"replicates_per_condition":2}
    cfg["conditions"]=["linear"];cfg["stress_conditions"]=[]
    monkeypatch.setattr(m,"protocol",lambda:cfg)
    def paired(*args):
        return [dict(method=name,failure=False,allowed=True,null_rejected=False,true_rejected=0,n_true=0)
                for name in cfg["development_methods"]]
    monkeypatch.setattr(m,"paired",paired)
    return m


def args(path): return SimpleNamespace(stage="screen",owner=0,owners=1,output=path,freeze=None)


def test_original_attempt_times_and_resume_without_replacement(study,tmp_path):
    path=tmp_path/"owner.sqlite";study.run(args(path));study.run(args(path))
    with sqlite3.connect(path) as db:
        rows=db.execute("SELECT started_utc,ended_utc,status FROM attempts").fetchall()
        assert db.execute("SELECT count(*) FROM replicates").fetchone()[0]==2 and len(rows)==2
    for start,end,status in rows:
        a=datetime.fromisoformat(start);b=datetime.fromisoformat(end)
        assert a.utcoffset().total_seconds()==0 and b>=a and status=="completed"
    out=tmp_path/"summary.json";study.summarize(SimpleNamespace(ledgers=[path],output=out))
    summary=json.loads(out.read_text())
    assert summary["original_utc_completed_replicates"]==2 and summary["original_utc_coverage_complete"]
    assert summary["attempt_status_counts"]=={"completed":2} and summary["unresolved_attempts"]==0


def test_interrupted_and_pending_attempts_are_not_rerun(study,tmp_path,monkeypatch):
    path=tmp_path/"failed.sqlite";calls=[]
    def failure(*unused): calls.append(True);raise RuntimeError("fixture abort")
    monkeypatch.setattr(study,"paired",failure)
    with pytest.raises(RuntimeError,match="fixture abort"):study.run(args(path))
    with sqlite3.connect(path) as db:
        start,end,status,error=db.execute("SELECT started_utc,ended_utc,status,error FROM attempts").fetchone()
        assert end is not None and status=="interrupted" and error=="RuntimeError: fixture abort"
        assert db.execute("SELECT count(*) FROM replicates").fetchone()[0]==0
    with pytest.raises(ValueError,match="prior attempt"):study.run(args(path))
    with sqlite3.connect(path) as db:db.execute("UPDATE attempts SET status='started',ended_utc=NULL,error=NULL")
    with pytest.raises(ValueError,match="prior attempt"):study.run(args(path))
    assert len(calls)==1


def test_scientific_failure_is_retained_with_original_utc(study,tmp_path,monkeypatch):
    path=tmp_path/"scientific.sqlite";original=study.paired
    def paired(*a):
        result=original(*a);result[0].update(failure=True,allowed=False);return result
    monkeypatch.setattr(study,"paired",paired);study.run(args(path))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT distinct status FROM attempts").fetchall()==[("completed_with_scientific_failure",)]
        assert db.execute("SELECT count(*) FROM replicates").fetchone()[0]==2


def test_missing_historical_utc_is_not_reconstructed(study,tmp_path):
    path=tmp_path/"historical.sqlite";study.run(args(path))
    with sqlite3.connect(path) as db:db.execute("DROP TABLE attempts")
    out=tmp_path/"summary.json";study.summarize(SimpleNamespace(ledgers=[path],output=out))
    summary=json.loads(out.read_text())
    assert summary["original_utc_completed_replicates"]==0 and summary["original_utc_coverage_complete"] is False


@pytest.mark.parametrize("defect",["invalid_utc","missing_end","unexpected_index"])
def test_invalid_attempt_records_are_rejected(study,tmp_path,defect):
    path=tmp_path/"invalid.sqlite";study.run(args(path))
    with sqlite3.connect(path) as db:
        if defect=="invalid_utc":db.execute("UPDATE attempts SET started_utc='2026-10-03T00:00:00'")
        elif defect=="missing_end":db.execute("UPDATE attempts SET ended_utc=NULL")
        else:db.execute("UPDATE attempts SET idx=999 WHERE idx=0")
    with pytest.raises(ValueError):study.summarize(SimpleNamespace(ledgers=[path],output=tmp_path/"summary.json"))


def test_direct_qualification_without_original_seal_is_rejected(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY));monkeypatch.setitem(sys.modules,"study",load("study"))
    control=load("control")
    with pytest.raises(ValueError,match="immutable seal"):control.qualify([])

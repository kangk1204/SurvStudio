"""Confirmation safeguards and failure denominators, independent of expensive simulation."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

spec=importlib.util.spec_from_file_location("guarded_study",Path(__file__).parents[1]/"validation/guarded_inference/study.py")
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_diagnostic_crash_preserves_paired_raw_calculation(monkeypatch):
    monkeypatch.setattr(study,"prepare_marker_cohort",lambda *a,**k:type("C",(),{"markers":None})())
    monkeypatch.setattr(study,"calculation",lambda *a,**k:{"null_rejected":True,"true_rejected":2,"n_true":5,"failure":False})
    def fail(*args,**kwargs):
        raise ArithmeticError("required diagnostic unavailable")
    monkeypatch.setattr(study,"diagnose_markers",fail)
    result=study.paired("partial_weak",180,30,2026100301,0,19)
    assert result[0]["null_rejected"] and not result[0]["failure"]
    for row in result[1:]:
        assert row["raw_null_rejected"] and row["raw_true_rejected"]==2
        assert row["diagnostic_failed"] and not row["failure"]
        assert not row["allowed"] and not row["null_rejected"] and row["true_rejected"]==0


def test_confirmation_refuses_stale_sources_and_incomplete_reference(tmp_path):
    path=tmp_path/"freeze.json"
    path.write_text(json.dumps({"source_hashes":{}}))
    with pytest.raises(ValueError,match="sources changed"):
        study.verify_freeze(path)
    path.write_text(json.dumps({"source_hashes":study.hashes()}))
    with pytest.raises(ValueError,match="Independent R"):
        study.verify_freeze(path)


def test_missing_and_failed_indexes_stay_in_denominator(tmp_path):
    path=tmp_path/"main.sqlite"
    config={"stage":"main","condition":"independent","n":180,"p":30,"source_hashes":study.hashes(),"freeze_hash":"test-only"}
    db=sqlite3.connect(path)
    db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
    db.execute("INSERT INTO metadata VALUES ('configuration',?)",(json.dumps(config),))
    db.execute("CREATE TABLE replicates (idx INTEGER PRIMARY KEY,result TEXT)")
    rows=[{"method":m,"failure":True,"allowed":False,"null_rejected":False,"true_rejected":0,"n_true":0}
          for m in ("legacy_linear","guarded_linear","guarded_spline")]
    db.execute("INSERT INTO replicates VALUES (0,?)",(json.dumps(rows),));db.commit();db.close()
    output=tmp_path/"summary.json";study.summary([path],output)
    result=json.loads(output.read_text())
    assert result["complete"] is False
    for row in result["summaries"]:
        assert row["planned"]==5000 and row["completed"]==1 and row["failures"]==1 and row["missing"]==4999
        assert row["fwer_failure_bounds"]==[0.,1.]
        assert row["engineering_status"] in {"pending","not_applicable"}
    with pytest.raises(ValueError,match="Duplicate index"):
        study.summary([path,path],output)

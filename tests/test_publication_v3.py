"""Failure accounting and immutable ownership for the publication study."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

DIRECTORY=Path(__file__).resolve().parents[1]/"validation/publication_v3"


def load(name):
    spec=importlib.util.spec_from_file_location("publication_"+name,DIRECTORY/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_confirmation_counts_and_stress_streams():
    study=load("study");cfg=study.protocol()
    assert sum(c["replicates"]*len(c["conditions"]) for c in cfg["confirmation"])==149500
    assert sum(len(c["conditions"]) for c in cfg["confirmation"])==55
    for condition in cfg["stress_conditions"]:
        a=study.dataset(condition,70,12,cfg["stress_seed"],4)
        b=study.dataset(condition,70,12,cfg["stress_seed"],4)
        pd.testing.assert_frame_equal(a[0],b[0]);assert not a[2].any()
        assert a[4]==(["Z1","Z2"] if condition=="two_clinical" else ["Z"])
        assert a[3].generate_state(4).tolist()==b[3].generate_state(4).tolist()
        assert not a[0].equals(study.dataset(condition,70,12,cfg["stress_seed"],5)[0])


def test_missing_and_failed_replicates_cannot_improve_retention(tmp_path,monkeypatch):
    study=load("study");cfg=study.protocol();cfg["conditions"]=["partial_weak"];cfg["stress_conditions"]=[]
    cfg["development_methods"]=["legacy_linear","candidate_A_linear"]
    cfg["selection"]={"cells":[[40,2]],"replicates_per_condition":3}
    monkeypatch.setattr(study,"protocol",lambda:cfg)
    ledger=tmp_path/"owner.sqlite"
    metadata=dict(stage="selection",seed=1,owners=1,owner=0,host="fixture",source_hashes={},environment={},freeze_sha256=None,candidate=None)
    failed=dict(method="legacy_linear",failure=True,allowed=False,null_rejected=False,true_rejected=0,n_true=1)
    candidate=dict(method="candidate_A_linear",failure=False,allowed=True,null_rejected=False,true_rejected=1,n_true=1)
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        db.execute("INSERT INTO metadata VALUES('configuration',?)",(json.dumps(metadata),))
        db.execute("CREATE TABLE replicates(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,elapsed REAL,result TEXT)")
        db.execute("INSERT INTO replicates VALUES('partial_weak',40,2,0,1,?)",(json.dumps([failed,candidate]),))
    output=tmp_path/"summary.json";study.summarize(SimpleNamespace(ledgers=[ledger],output=output))
    result=json.loads(output.read_text());row=result["summaries"][1]
    assert not result["complete"] and row["missing"]==2
    assert row["power_ratio"]["point"]==pytest.approx(1/3)
    assert row["paired_difference_from_legacy"]["complete_pairs"]==0
    assert row["fwer_failure_bounds"]==[0,2/3]
    assert result["summaries"][0]["conditional_fwer"] is None
    with pytest.raises(ValueError,match="immutable"):
        study.summarize(SimpleNamespace(ledgers=[ledger],output=output))
    with pytest.raises(ValueError,match="Duplicate"):
        study.summarize(SimpleNamespace(ledgers=[ledger,ledger],output=tmp_path/"duplicate.json"))


def test_cost_gate_never_reduces_draws(monkeypatch):
    study=load("study");monkeypatch.setitem(sys.modules,"study",study);control=load("control")
    summary=dict(stage="cost",complete=True,summaries=[dict(n=n,p=p,elapsed_p95_seconds=10000) for n,p in study.protocol()["cost"]["cells"]])
    result=control.cost(summary)
    assert result["status"]=="v2_fallback_resource_limit" and result["safety_factor"]==2
    assert study.protocol()["diagnostic_draws"]==9999
    summary["complete"]=False
    with pytest.raises(ValueError,match="Complete"):
        control.cost(summary)


def test_fixed_owner_interruption_is_detected(tmp_path):
    execute=load("execute")
    def probe():
        return json.loads(subprocess.check_output([sys.executable,"-c",execute.owner_status_code(tmp_path,7,[sys.executable,"fixture.py"])],text=True))
    assert probe()["status"]=="new"
    (tmp_path/"owner-07.sqlite").touch();assert probe()["status"]=="interrupted"
    (tmp_path/"owner-07.done.json").write_text(json.dumps(dict(returncode=1,owner=7)))
    assert probe()["returncode"]==1


def test_public_endpoint_gap_and_external_missing_size():
    case=load("public_case")
    development=pd.DataFrame(dict(nodes=[1,1,0],er=[0,2,1],pgr=[1,2,1],size=["<=20","20-50",">50"],
        recur=[0,1,0],death=[1,1,0],rtime=[100,200,300],dtime=[500,600,300]))
    conservative=case.prepare_frame(development,True,"conservative_RFS")
    liberal=case.prepare_frame(development,True,"liberal_RFS")
    assert conservative["event"].tolist()==[0,1] and conservative["time"].tolist()==[100,200]
    assert liberal["event"].tolist()==[1,1] and liberal["time"].tolist()==[500,200]
    external=pd.DataFrame(dict(nodes=[1,1],er=[0,1],pgr=[1,2],size=[20,np.nan],rfstime=[3000,100],status=[1,0]))
    transformed=case.prepare_frame(external,False,"conservative_RFS")
    assert transformed.loc[0,"event"]==0 and transformed.loc[0,"time"]==2556.75
    assert transformed.loc[0,"size"]=="0" and pd.isna(transformed.loc[1,"size"])
    with pytest.raises(ValueError,match="endpoint"):
        case.prepare_frame(external,False,"choose_after_results")

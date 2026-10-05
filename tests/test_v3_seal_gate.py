"""Synthetic tests for source-bound seals; no real CI or selection fabricated."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

DIRECTORY=Path(__file__).resolve().parents[1]/"validation/publication_v3"


@pytest.fixture
def setup(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    study_spec=importlib.util.spec_from_file_location("staged_seal_study",DIRECTORY/"study.py")
    study=importlib.util.module_from_spec(study_spec);study_spec.loader.exec_module(study)
    monkeypatch.setitem(sys.modules,"study",study)
    spec=importlib.util.spec_from_file_location("staged_seal_control",DIRECTORY/"control.py")
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    cfg=study.protocol();root=tmp_path/"source";root.mkdir()
    relative="validation/publication_v3/verify_reference_full.py";p=root/relative;p.parent.mkdir(parents=True);p.write_text("synthetic oracle fixture")
    oracle_hash=hashlib.sha256(p.read_bytes()).hexdigest()
    policy_path=root/"src/survival_toolkit/data/marker_bootstrap_policy.json";policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps(dict(candidate=cfg["candidates"][0],selection_status="selected_before_confirmation")))
    protocol_path=root/"validation/publication_v3/protocol.json";protocol_path.write_text(json.dumps(cfg))
    footprints={relative:oracle_hash,"validation/publication_v3/protocol.json":hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    monkeypatch.setattr(control,"ROOT",root);monkeypatch.setattr(control,"hashes",lambda:deepcopy(footprints))
    monkeypatch.setattr(control,"environment",lambda:{"synthetic":True})
    revision="b"*40
    def command(args,**unused):
        if "status" in args:return ""
        return revision+"\n"
    monkeypatch.setattr(control.subprocess,"check_output",command)
    monkeypatch.setattr(control.subprocess,"run",lambda *a,**kw:None)
    rows=[]
    for p in (30,300):
        for condition in cfg["supported"]["guarded_linear"]:
            for label in ("A","B"):
                rows.append(dict(method=f"candidate_{label}_linear",condition=condition,n=180,p=p,missing=0,failures=0,
                                 fwer=0.,conditional_fwer=0.,allowed_fraction=1.,power_ratio={"point":1.},mean_diagnostic_seconds=1.))
    summary=dict(stage="selection",complete=True,summaries=rows,
                 configuration=dict(seed=cfg["selection_seed"],source_hashes=deepcopy(footprints)))
    selected=control.select(summary)
    reference=dict(passed=True,full_draw_coverage=True,diagnostic_draws=9999,markers=30,environment={"synthetic":True},
                   source_hashes=deepcopy(footprints),checks=[dict(passed=True,draws=9999,exceedance_count_exact=True) for _ in range(10)])
    names=["Numerical agreement with R survival","Front-end script syntax","Wheel install smoke test",
           "ubuntu-latest / Python 3.11 / browser-e2e","macos-14 / Python 3.11","windows-latest / Python 3.11",
           "ubuntu-latest / Python 3.11","ubuntu-latest / Python 3.12","ubuntu-latest / Python 3.13"]
    ci=dict(revision=revision,passed=True,run_id=123,jobs=[dict(name=n,status="completed",conclusion="success") for n in names])
    return control,selected,reference,ci,summary


def test_complete_source_bound_seal(setup):
    control,*values=setup;result=control.seal(*values)
    assert result["source_ci_passed"] and result["source_ci_run_id"]==123
    stamp=datetime.fromisoformat(result["sealed_at_utc"])
    assert stamp.utcoffset().total_seconds()==0 and result["freeze_deadline"]=="2026-10-31"
    assert result["qualification_status"]=="pending_confirmation"


@pytest.mark.parametrize("defect",["old_ci","missing_job","failed_job","short_draws","small_family","missing_r_check","different_r_source","selection_seed","changed_protocol","selection_tamper"])
def test_incomplete_or_changed_seal_is_rejected(setup,defect):
    control,selection,reference,ci,summary=setup
    if defect=="old_ci":ci["revision"]="c"*40
    elif defect=="missing_job":ci["jobs"].pop()
    elif defect=="failed_job":ci["jobs"][0]["conclusion"]="failure"
    elif defect=="short_draws":reference["diagnostic_draws"]=1999
    elif defect=="small_family":reference["markers"]=2
    elif defect=="missing_r_check":reference["checks"].pop()
    elif defect=="different_r_source":reference["source_hashes"]["validation/publication_v3/protocol.json"]="c"*64
    elif defect=="selection_seed":summary["configuration"]["seed"]+=1
    elif defect=="changed_protocol":summary["configuration"]["source_hashes"]["validation/publication_v3/protocol.json"]="c"*64
    else:selection["candidate"]=None
    with pytest.raises(ValueError):control.seal(selection,reference,ci,summary)


def test_source_freeze_deadline_is_enforced(setup,monkeypatch):
    control,*values=setup
    class LateClock:
        @staticmethod
        def now(tz):return datetime(2026,10,31,15,0,0,tzinfo=timezone.utc)
    monkeypatch.setattr(control,"datetime",LateClock)
    with pytest.raises(ValueError,match="deadline"):control.seal(*values)

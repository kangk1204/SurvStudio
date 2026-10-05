"""Source, environment and incremental resources are checked before confirmation."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def execute():
    path=Path(__file__).resolve().parents[1]/'validation/publication_v3/execute.py'
    spec=importlib.util.spec_from_file_location('preflight_execute',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('defect',[None,'source_hashes','environment'])
def test_fresh_confirmation_hosts_must_match_seal(execute,tmp_path,monkeypatch,defect):
    freeze=tmp_path/'seal.json';freeze.write_text(json.dumps(dict(source_hashes={'x':'a'},environment={'python':'fixture'})))
    def probe(*args):
        result=dict(source_hashes={'x':'a'},environment={'python':'fixture'},load1=1,cpus=32,available_bytes=64*1024**3,free_disk_bytes=40*1024**3,new_workers=18)
        if defect:result[defect]={}
        return json.dumps(result)
    monkeypatch.setattr(execute,'remote',probe)
    if defect:
        with pytest.raises(ValueError,match='differs'):execute.confirmation_preflight('main',tmp_path,freeze)
    else:
        execute.confirmation_preflight('main',tmp_path,freeze)
        assert len(json.loads((tmp_path/'execution-state.json').read_text())['confirmation_resource_snapshots'])==3


def test_existing_workers_are_not_reserved_twice(execute,tmp_path,monkeypatch):
    freeze=tmp_path/'seal.json';freeze.write_text(json.dumps(dict(source_hashes={},environment={})))
    monkeypatch.setattr(execute,'remote',lambda *a:json.dumps(dict(source_hashes={},environment={},load1=100,cpus=32,available_bytes=0,free_disk_bytes=0,new_workers=0)))
    monkeypatch.setattr(execute.time,'sleep',lambda *a:pytest.fail('Existing owners must not wait for duplicate CPU reservations'))
    execute.confirmation_preflight('main',tmp_path,freeze)

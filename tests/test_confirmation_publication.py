"""Evidence must be complete and independently audited before product qualification."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).parents[1]


def module(name):
    path = ROOT / "validation/diagnostic_followup" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_incomplete_confirmation_cannot_be_exported(tmp_path):
    summary = tmp_path / "summary.json"
    freeze = tmp_path / "freeze.json"
    summary.write_text(json.dumps({"complete": False}))
    freeze.write_text("{}")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Complete v2"):
        module("publish_confirmation").publish(tmp_path, summary, freeze, output)
    assert not output.exists()


def test_missing_independent_audit_cannot_qualify(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name, content in {"confirmation-summary.json": {"complete": True}, "freeze.json": {},
                          "aggregate-audit.json": {"passed": False}, "aggregate-manifest.json": {}}.items():
        (results / name).write_text(json.dumps(content))
    with pytest.raises(ValueError, match="independently audited"):
        module("qualify_confirmation").build(results, tmp_path / "previous.json", tmp_path / "registry.json")
    assert not (tmp_path / "registry.json").exists()


def test_current_registry_is_reproducible_and_preserves_v1(tmp_path):
    registry_path = ROOT / "src/survival_toolkit/data/marker_qualification.json"
    current = json.loads(registry_path.read_text())
    old = tmp_path / "previous.json"
    old.write_text(json.dumps(current["studies"]["marker-inference/1"]))
    output = tmp_path / "registry.json"
    module("qualify_confirmation").build(ROOT / "validation/diagnostic_followup/results/20261003/confirmation-v2", old, output)
    rebuilt=json.loads(output.read_text())
    historical={name:current["studies"][name] for name in rebuilt["studies"]}
    assert rebuilt["studies"]==historical
    assert {k:v for k,v in rebuilt.items() if k!="studies"}=={k:v for k,v in current.items() if k!="studies"}
    assert current["studies"]["marker-inference/3"]["study_complete"] is False
    assert all(profile["status"]=="not_evaluated" for profile in current["studies"]["marker-inference/3"]["profiles"].values())
    assert current["studies"]["marker-inference/1"]["profiles"]["linear"]["status"] == "failed_exploratory_only"
    assert current["profiles"]["restricted_cubic_spline"]["status"] == "failed_exploratory_only"


def test_evidence_tampering_cannot_qualify(tmp_path):
    source = ROOT / "validation/diagnostic_followup/results/20261003/confirmation-v2"
    results = tmp_path / "results"
    shutil.copytree(source, results)
    summary = json.loads((results / "confirmation-summary.json").read_text())
    summary["model_release_status"]["guarded_spline"] = "passed_supported_conditions_only"
    (results / "confirmation-summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="evidence changed"):
        module("qualify_confirmation").build(results, ROOT / "src/survival_toolkit/data/marker_qualification.json", tmp_path / "registry.json")

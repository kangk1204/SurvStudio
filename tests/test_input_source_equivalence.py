"""The source bridge must reject statistical tampering without scientific imports."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "input_source_equivalence", _ROOT / "scripts" / "verify_input_source_equivalence.py",
)
scope = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scope)


@pytest.fixture(scope="module")
def sources():
    # No historical Git object is needed in ordinary shallow-checkout CI. The
    # dedicated source-verification job fetches and compares the actual baseline.
    return {name: (_ROOT / name).read_bytes() for name in scope.BOUND_FILES}


def _changed_function(sources, name, statement):
    changed = dict(sources)
    tree = ast.parse(changed[scope.ANALYSIS].decode("utf-8"))
    node = scope._definitions(tree)[name]
    node.body.insert(0, ast.parse(statement).body[0])
    changed[scope.ANALYSIS] = ast.unparse(tree).encode("utf-8")
    return changed


def test_identical_sources_have_deterministic_path_free_report(sources):
    first = scope.verify_sources(sources, sources)
    second = scope.verify_sources(sources, sources)
    assert first["passed"] and first == second
    assert all(first["protected_ast_checks"].values())
    serialized = json.dumps(first, sort_keys=True)
    assert str(_ROOT) not in serialized
    assert "timestamp" not in serialized


@pytest.mark.parametrize("name", ["_harrell_c_index", "compute_km_analysis", "compute_cox_analysis"])
def test_synthetic_statistical_function_tamper_is_rejected(sources, name):
    result = scope.verify_sources(sources, _changed_function(sources, name, "return 0.999"))
    assert not result["passed"]
    assert "Unreviewed definition change: " + name in result["errors"]
    assert any(error.startswith("Protected AST comparison failed:") for error in result["errors"])


def test_same_parser_name_does_not_authorize_unreviewed_changes(sources):
    result = scope.verify_sources(sources, _changed_function(sources, "_read_excel_frame", "return None"))
    assert not result["passed"]
    assert "Unreviewed definition change: _read_excel_frame" in result["errors"]


def test_unrecognized_new_definition_is_rejected(sources):
    candidate = dict(sources)
    candidate[scope.ANALYSIS] += b"\ndef unreviewed_statistic():\n    return 0.999\n"
    result = scope.verify_sources(sources, candidate)
    assert not result["passed"]
    assert "Unreviewed added definition: unreviewed_statistic" in result["errors"]


def test_each_other_bound_module_requires_exact_bytes(sources):
    for name in scope.BOUND_FILES[:-1]:
        candidate = dict(sources)
        candidate[name] += b"\n# even a nonfunctional edit needs a new source review\n"
        result = scope.verify_sources(sources, candidate)
        assert not result["passed"]
        assert "Bound numerical module changed: " + name in result["errors"]


def test_module_constant_tamper_is_rejected(sources):
    candidate = dict(sources)
    tree = ast.parse(candidate[scope.ANALYSIS].decode("utf-8"))
    target = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(name, ast.Name) and name.id == "_MAX_EXP_INPUT" for name in node.targets))
    target.value = ast.Constant(value=0)
    candidate[scope.ANALYSIS] = ast.unparse(tree).encode("utf-8")
    result = scope.verify_sources(sources, candidate)
    assert not result["passed"]
    assert "Unreviewed module-level import, constant or statement change" in result["errors"]


def test_only_exact_event_metadata_is_removed_from_cohort_comparison(sources):
    candidate = _changed_function(sources, "_cohort_frame", "frame = df.iloc[:1].copy()")
    result = scope.verify_sources(sources, candidate)
    assert not result["passed"]
    assert not result["protected_ast_checks"]["cohort_after_exact_metadata_removal"]


def test_empty_python_type_parameter_fields_do_not_change_canonical_hash():
    # Python 3.12 introduced this empty field. Its absence on 3.11 must not make
    # otherwise identical source receipts differ between CI platforms.
    class Definition(ast.AST):
        _fields = ("name", "type_params")
    first = Definition()
    first.name = "example"
    second = copy.deepcopy(first)
    second.type_params = []
    assert scope._ast_hash(first) == scope._ast_hash(second)


def test_cli_json_file_and_stdout_are_identical_without_historical_git_object(sources, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(scope, "_resolve_ref", lambda *args: scope.BASELINE)
    monkeypatch.setattr(scope, "_git_sources", lambda *args: sources)
    output = tmp_path / "receipt.json"
    assert scope.main(["--candidate", "HEAD", "--output", str(output)]) == 0
    stdout = capsys.readouterr().out
    assert stdout == output.read_text(encoding="utf-8")
    assert json.loads(stdout)["passed"]

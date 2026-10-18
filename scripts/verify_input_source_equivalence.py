"""Check a reviewed input-layer source bridge, separately from numerical validation.

Example: python scripts/verify_input_source_equivalence.py --baseline 302b70e --candidate HEAD

The default candidate is WORKTREE. JSON goes to stdout and optionally --output.
Reports contain no machine paths or timestamps. Fetch the baseline before running
in a shallow checkout. No product module or optional scientific package is imported.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASELINE = "302b70eb97297df86ee34e484f9afadbad2f13a7"
PREFIX = "src/survival_toolkit/"
BOUND_FILES = tuple(PREFIX + name + ".py" for name in (
    "clinical_basis", "marker_diagnostics", "marker_evaluation", "marker_screen", "encoding", "analysis",
))
ANALYSIS = PREFIX + "analysis.py"
# Exact reviewed ASTs, not permission to change arbitrary code under these names.
REVIEWED_DEFINITION_SHA256 = {
    "_ambiguous_number_message": "0c66beccc23927fda4e5edcb0b8bc7bb87a7509967bfbbb76fc888f9e3fa51de",
    "_coerce_survival_time_values": "20b4d1cfae0e0cdae883fbf78ad2f975777c10932ca3595aef9b22727bacb072",
    "_cohort_frame": "8f17525d9b722c0a99cd42b12bd8450487007e175c15f673bdbd324dfee04521",
    "_convert_declared_number_format": "f799d746c7cb897751424950b2c8529aca7c8a5074e2774c88c508ae774a3162",
    "_convert_formatted_number_columns": "4934606b6c2a2581be539f39d031ab20cf0a01a9bc8f8e6e70a96c217708386b",
    "_cox_input_quality_notes": "ff344de476a9e841dbb89de4951ced301548e8b184e5e62cddda81494bb93aef",
    "_event_tokens_for_label": "47ca2a2ef715f4341173b8fba226bdc7f2b78b6e910df076fef69ef113f134b3",
    "_km_cohort_summary": "16f2cff2964347703cc4d876561c6ee522ca51dba612118a808bb5500da67562",
    "_km_scientific_summary": "9940bbe7ed7aa704c7ea21baae643766657edd0ef792dce0f1decfd265200963",
    "_load_dataframe_source": "0caeccb664afa68e3c487a3e6fd5acd012c7c0a6d479827ca0a19fb9e95e4180",
    "_numeric_text_input_reviews": "b89642e95433377cae414cdd1632a3e330ab4da98d534d452f921004ec4e1851",
    "_parse_delimited_text": "ae58f7f2ad4a80e42a3c10f5d1ab8076d932f91284da2c41f75903fa0d586721",
    "_parse_delimited_text._read": "c68e1693a076c3bb09b147f0b942a85394b91f4965580547d9c140319c962437",
    "_read_csv_with_fallback": "f8803790ca162bc23d02444c5d899472b1718d0e19d4fd20d061f5b4f6407ddf",
    "_read_excel_frame": "7e6527efb6816af097ab00a4277ddd3b2e9609b9aaf79c61ef20cac337b15a89",
    "_record_numeric_missing_normalization": "9828a787694424758a31d5a819caa3ef26ac96aba19a8745bbd3dfaa208313cd",
    "_set_parsed_column": "5a7c7ee02c34de3b18bc9ed2d9cf244802a18faf87515ffc19c83bbc0266308d",
    "coerce_event": "ebc365b61745b6f93f228f5fa2158745cd74060823f5795bef5d6cb79045e7d3",
    "compute_cox_analysis": "c5ba1b5f28aed494196652acf44d7dd8bc28904e1971b77d55806d7a36768107",
    "load_dataframe": "fae935da7aec7da7c18e72b9c8c1c089f55bb7fb101aca1c471b704594385243",
    "load_dataframe_from_path": "fd983a9fcfc60e8fa7ea37cab6994506bb78e9366e996877f8b56e628d38e083",
    "profile_dataframe": "f44a2b29a2fa59212dd6d322f3acaca2fe44d096a14f03a0ff620ffd5fe8a90e"
}
REVIEWED_MODULE_SETUP_SHA256 = "a976483029a9056bf8cceaf2bc2ebed54f4c9da8422c11b4063d832355d5f6f1"
ALLOWED_ADDED = frozenset({
    "_convert_declared_number_format", "_numeric_text_input_reviews", "_record_numeric_missing_normalization",
})
PROTECTED_CORE = (
    "_numeric_event_indicator", "_try_coerce_binary_event", "compute_km_analysis",
    "_efron_schoenfeld_residuals", "_cox_grambsch_therneau_test", "_harrell_c_index",
)


def _ast_value(value):
    if isinstance(value, ast.AST):
        return {"_type": type(value).__name__, **{
            field: _ast_value(item) for field, item in ast.iter_fields(value)
            if not (field == "type_params" and not item)
        }}
    if isinstance(value, list):
        return [_ast_value(item) for item in value]
    if value is Ellipsis:
        return {"_literal": "Ellipsis"}
    if isinstance(value, bytes):
        return {"_literal": "bytes", "hex": value.hex()}
    if isinstance(value, complex):
        return {"_literal": "complex", "value": repr(value)}
    return value


def _canonical_ast(value):
    # Python 3.12 added empty type_params fields. Excluding only those keeps the
    # representation stable across supported Python versions without losing code.
    return json.dumps(_ast_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _ast_hash(value):
    return hashlib.sha256(_canonical_ast(value).encode("utf-8")).hexdigest()


def _definitions(tree):
    found = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.parents = []

        def visit_FunctionDef(self, node):
            name = ".".join([*self.parents, node.name])
            suffix = 2
            original = name
            while name in found:
                name = f"{original}[{suffix}]"
                suffix += 1
            found[name] = node
            self.parents.append(name.rsplit(".", 1)[-1])
            self.generic_visit(node)
            self.parents.pop()

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

    Visitor().visit(tree)
    return found


def _module_setup(tree):
    return [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _same(left, right):
    return _canonical_ast(left) == _canonical_ast(right)


def _strip_cohort_metadata(node):
    node = copy.deepcopy(node)
    replacement = ast.parse("frame[event_column] = coerce_event(frame[event_column], event_positive_value=event_positive_value)").body[0]
    sequence = ast.parse("""coded_event = coerce_event(frame[event_column], event_positive_value=event_positive_value)
event_coding = coded_event.attrs.get("event_coding")
frame[event_column] = coded_event""").body
    attachment = ast.parse("""if event_coding:
    frame.attrs["event_coding"] = event_coding""").body[0]
    result = []
    index = 0
    while index < len(node.body):
        if _same(node.body[index:index + len(sequence)], sequence):
            result.append(copy.deepcopy(replacement))
            index += len(sequence)
        elif _same(node.body[index], attachment):
            index += 1
        else:
            result.append(node.body[index])
            index += 1
    node.body = result
    return node


def _strip_output_metadata(node):
    """Remove only the reviewed optional event-coding dictionary expansion."""
    expected = ast.parse('{**({"event_coding": frame.attrs["event_coding"]} if frame.attrs.get("event_coding") else {})}', mode="eval").body.values[0]

    class Visitor(ast.NodeTransformer):
        def visit_Dict(self, value):
            value = self.generic_visit(value)
            pairs = [(key, item) for key, item in zip(value.keys, value.values)
                     if not (key is None and _same(item, expected))]
            value.keys = [key for key, _ in pairs]
            value.values = [item for _, item in pairs]
            return value

    return Visitor().visit(copy.deepcopy(node))


def _numeric_event_branches(node):
    explicit = next(index for index, item in enumerate(node.body)
                    if isinstance(item, ast.If) and "event_positive_value is not None" in ast.unparse(item.test))
    branch = node.body[explicit]
    numeric_return = next(index for index, item in enumerate(branch.body)
                          if isinstance(item, ast.If) and any(
                              isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                              and child.func.id == "_numeric_event_indicator" for child in ast.walk(item)))
    # Compare the initial checks, explicit numeric mapping, and successful automatic
    # inference path. New text-label policies and refusal messages are separate.
    return [node.body[:explicit], branch.test, branch.body[:numeric_return + 1], node.body[explicit + 1:explicit + 4]]


def verify_sources(baseline_sources, candidate_sources):
    errors = []
    files = []
    for name in BOUND_FILES:
        before, after = baseline_sources[name], candidate_sources[name]
        identical = before == after
        files.append({"path": name, "baseline_sha256": hashlib.sha256(before).hexdigest(),
                      "candidate_sha256": hashlib.sha256(after).hexdigest(), "byte_identical": identical})
        if name != ANALYSIS and not identical:
            errors.append("Bound numerical module changed: " + name)
    before_tree = ast.parse(baseline_sources[ANALYSIS].decode("utf-8"))
    after_tree = ast.parse(candidate_sources[ANALYSIS].decode("utf-8"))
    before, after = _definitions(before_tree), _definitions(after_tree)
    common = before.keys() & after.keys()
    changed = sorted(name for name in common if not _same(before[name], after[name]))
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    for name in changed:
        if REVIEWED_DEFINITION_SHA256.get(name) != _ast_hash(after[name]):
            errors.append("Unreviewed definition change: " + name)
    for name in added:
        if name not in ALLOWED_ADDED or REVIEWED_DEFINITION_SHA256.get(name) != _ast_hash(after[name]):
            errors.append("Unreviewed added definition: " + name)
    errors.extend("Removed definition: " + name for name in removed)
    setup_changed = not _same(_module_setup(before_tree), _module_setup(after_tree))
    if setup_changed and _ast_hash(_module_setup(after_tree)) != REVIEWED_MODULE_SETUP_SHA256:
        errors.append("Unreviewed module-level import, constant or statement change")
    checks = {}
    for name in PROTECTED_CORE:
        checks[name] = name in common and _same(before[name], after[name])
    checks["numeric_time_early_branch"] = "_coerce_survival_time_values" in common and _same(
        next(item for item in before["_coerce_survival_time_values"].body if isinstance(item, ast.If)),
        next(item for item in after["_coerce_survival_time_values"].body if isinstance(item, ast.If)),
    )
    checks["numeric_event_mapping_branches"] = "coerce_event" in common and _same(
        _numeric_event_branches(before["coerce_event"]), _numeric_event_branches(after["coerce_event"]),
    )
    checks["cohort_after_exact_metadata_removal"] = "_cohort_frame" in common and _same(
        _strip_cohort_metadata(before["_cohort_frame"]), _strip_cohort_metadata(after["_cohort_frame"]),
    )
    for name in ("compute_cox_analysis", "_km_cohort_summary"):
        checks[name + "_after_exact_output_metadata_removal"] = name in common and _same(
            _strip_output_metadata(before[name]), _strip_output_metadata(after[name]),
        )
    errors.extend("Protected AST comparison failed: " + name for name, passed in checks.items() if not passed)
    imports = {}
    for name in BOUND_FILES[:-1]:
        tree = ast.parse(candidate_sources[name].decode("utf-8"))
        imports[name] = sorted({alias.name for node in ast.walk(tree)
                                if isinstance(node, ast.ImportFrom) and node.module == "survival_toolkit.analysis"
                                for alias in node.names})
    return {
        "schema_version": "marker-source-scope/1", "baseline_revision": BASELINE,
        "passed": not errors, "errors": sorted(errors), "bound_files": files,
        "analysis_definition_inventory": {"baseline_count": len(before), "candidate_count": len(after),
            "changed": changed, "added": added, "removed": removed},
        "reviewed_candidate_definition_sha256": {name: _ast_hash(after[name]) for name in sorted(set(changed + added))},
        "module_setup": {"changed": setup_changed, "candidate_ast_sha256": _ast_hash(_module_setup(after_tree))},
        "protected_ast_checks": checks, "analysis_imports_from_bound_modules": imports,
        "scope": "Numeric time and event paths; unchanged bound numerical modules; cohort and Cox ASTs after exact optional event-coding metadata removal. Reviewed parsing, label policies, notices and summaries are explicitly changed.",
        "limitations": ["This is source-scope verification, not numerical validation or a new confirmation study.",
                        "Whole event/cohort functions are not claimed byte-identical. Text-file conventions and literal-label mappings require their own tests.",
                        "Historical study criteria and results are not rerun or amended by this tool."],
        "independent_validation_commands": [
            "python validation/agreement/run_agreement.py --check --output validation-output/agreement",
            "python validation/clinical/run_independent_reference.py --output validation-output/clinical",
        ],
    }


def _resolve_ref(repo, ref):
    command = ["git", "-C", str(repo), "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError("Git reference is unavailable; fetch the specified baseline/candidate commit first.")
    return result.stdout.strip()


def _git_sources(repo, revision):
    return {name: subprocess.run(["git", "-C", str(repo), "show", revision + ":" + name],
            capture_output=True, check=True).stdout for name in BOUND_FILES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default="WORKTREE", help="Git reference, or WORKTREE (default)")
    parser.add_argument("--output", type=Path, help="Optional copy of the JSON report")
    args = parser.parse_args(argv)
    try:
        baseline = _resolve_ref(args.repo, args.baseline)
        previous = _git_sources(args.repo, baseline)
        current = ({name: (args.repo / name).read_bytes() for name in BOUND_FILES}
                   if args.candidate == "WORKTREE" else _git_sources(args.repo, _resolve_ref(args.repo, args.candidate)))
        report = verify_sources(previous, current)
        report["baseline_revision"] = baseline
    except (OSError, ValueError, SyntaxError, StopIteration, subprocess.SubprocessError) as exc:
        report = {"schema_version": "marker-source-scope/1", "passed": False,
                  "errors": ["Source verification could not complete: " + type(exc).__name__]}
    serialized = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    sys.stdout.write(serialized)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

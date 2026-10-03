"""Connect independently audited confirmation to versioned product evidence.

This records the unchanged main-study gates. Extension findings remain limitations,
not additional fitted criteria or grounds for selecting favorable support conditions.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(results, previous, output):
    summary_path = results / "confirmation-summary.json"
    summary = json.loads(summary_path.read_text())
    freeze = json.loads((results / "freeze.json").read_text())
    audit = json.loads((results / "aggregate-audit.json").read_text())
    manifest = json.loads((results / "aggregate-manifest.json").read_text())
    if not summary["complete"] or not audit["passed"] or audit["independent_R_quantities"] != 2826:
        raise ValueError("Complete independently audited confirmation is required")
    if audit["manifest_sha256"] != digest(results / "aggregate-manifest.json"):
        raise ValueError("Audit is not bound to the publication manifest")
    for name, expected in manifest["files"].items():
        if digest(results / name) != expected:
            raise ValueError("Published evidence changed: " + name)
    if summary["protocol"]["method_version"] != "marker-inference/2":
        raise ValueError("Unexpected method version")
    old = json.loads(previous.read_text())
    historical = old.get("studies", {}).get("marker-inference/1", old)
    if historical["method_version"] != "marker-inference/1" or not historical["study_complete"]:
        raise ValueError("Original completed v1 evidence must be preserved")
    rows = summary["summaries"]
    profiles = {}
    for basis, method in (("linear", "guarded_linear"), ("restricted_cubic_spline", "guarded_spline")):
        required = [r for r in rows if r["stage"] == "main" and r["method"] == method
                    and r["condition"] in summary["protocol"]["supported"][method]]
        if len(required) != len(summary["protocol"]["supported"][method]):
            raise ValueError("Missing prespecified supported condition")
        gates = []
        for r in required:
            baseline = next(b for b in rows if b["stage"] == "main" and b["condition"] == r["condition"]
                            and b["method"] == "legacy_linear")
            gates.append({"condition": r["condition"], "status": r["engineering_status"],
                          "allowed_fraction": r["allowed_fraction"],
                          "fwer_upper95": r["fwer_worst_case_upper95"],
                          "conditional_fwer_upper95": r["conditional_upper95"],
                          "power": r["power"], "power_retention": None if r["power"] is None else r["power"] / baseline["power"]})
        profiles[basis] = {"status": summary["model_release_status"][method],
                           "failed_conditions": [r["condition"] for r in required if r["engineering_status"] != "passed"],
                           "supported_conditions": summary["protocol"]["supported"][method], "main_gates": gates}
    kernel_names = ("clinical_basis", "marker_diagnostics", "marker_evaluation", "marker_screen", "encoding", "analysis")
    current = {"policy_version": "marker-qualification/1", "method_version": "marker-inference/2", "study_complete": True,
               "study_git_revision": freeze["git_revision"], "fixed_datasets": 85500,
               "freeze_sha256": digest(results / "freeze.json"), "summary_sha256": digest(summary_path),
               "aggregate_audit_sha256": digest(results / "aggregate-audit.json"),
               "kernel_source_hashes": {f"src/survival_toolkit/{name}.py": freeze["source_hashes"][f"src/survival_toolkit/{name}.py"]
                                        for name in kernel_names},
               "qualification_dimensions": {"n": 180, "markers": 30, "permutations": 999, "datasets_per_condition": 5000},
               "claim_boundary": "Finite main-study engineering gates in prespecified supported conditions; not universal 5% error control or certification of an individual dataset.",
               "extension_limitations": "The 500-subject, 300-marker and 3000-marker extensions do not establish broader qualification. In particular, large independent and linear marker families substantially reduce availability; power retention and conditional error bounds also remain limiting.",
               "profiles": profiles}
    registry = {**current, "studies": {"marker-inference/1": historical, "marker-inference/2": current}}
    output.write_text(json.dumps(registry, indent=2) + "\n")
    print(json.dumps({"profiles": {k: v["status"] for k, v in profiles.items()}, "registry_sha256": digest(output)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.results, a.previous, a.output)

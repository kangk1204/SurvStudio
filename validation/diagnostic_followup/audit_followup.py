"""Verify the current review evidence without changing frozen statistical decisions.

This audit checks provenance, public aggregate scope, and figure source numbers.
Visual inspection and author review remain separate requirements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_hashes(base, hashes):
    for name, expected in hashes.items():
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()) or digest(path) != expected:
            raise ValueError("Missing, changed or escaping evidence: " + str(path))


def source_numbers(path, expected, keys, columns):
    actual = pd.read_csv(path, float_precision="round_trip").set_index(keys)
    expected = expected.set_index(keys)
    if not actual.index.is_unique or set(actual.index) != set(expected.index):
        raise ValueError("Figure source coverage differs: " + str(path))
    for column in columns:
        a = actual.loc[expected.index, column].to_numpy(dtype=float)
        b = expected[column].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError("Figure source numbers differ: " + str(path) + ":" + column)


def run(results):
    results = results.resolve()
    freeze = read(results / "v2-freeze.json")
    check_hashes(ROOT, freeze["source_hashes"])
    study = results / "confirmation-v2"
    manifest = read(study / "aggregate-manifest.json")
    check_hashes(study, manifest["files"])
    audit = read(study / "aggregate-audit.json")
    if not audit["passed"] or digest(study / "aggregate-manifest.json") != audit["manifest_sha256"]:
        raise ValueError("Confirmation audit does not bind the current manifest")
    if (audit["fixed_datasets"], audit["paired_method_records"], audit["calculation_failures"]) != (85500, 256500, 0):
        raise ValueError("Unexpected confirmation coverage or failures")
    summary = read(study / "confirmation-summary.json")
    if not summary["complete"] or len(summary["summaries"]) != 117:
        raise ValueError("Full confirmation grid is required")
    decisions = read(results / "independent-gate-decision-check.json")
    if not decisions["passed"] or len(decisions["checks"]) != 14 or not all(c["matches"] for c in decisions["checks"]):
        raise ValueError("Independent frozen gate decision check failed")
    retention = read(results / "power-retention-reference-audit.json")
    if not retention["independent_R_passed"] or not retention["postconfirmation_descriptive_only"]:
        raise ValueError("Independent descriptive uncertainty check failed")
    check_hashes(study, retention["input_hashes"])
    check_hashes(results, retention["output_hashes"])
    check_hashes(Path(__file__).parent, retention["source_hashes"])
    for directory in ("independent_R_full_v2", "independent_R_v2", "tool-comparison-v2"):
        reference = read(results / directory / "verification.json")
        if not reference["passed"]:
            raise ValueError("Independent reference failed: " + directory)
        if "hashes" in reference:
            check_hashes(results / directory, reference["hashes"])
        if "input_hashes" in reference:
            check_hashes(results / directory, reference["input_hashes"])
        if "files" in reference:
            check_hashes(results / directory, reference["files"])
        check_hashes(ROOT, reference["source_hashes"])
    case_root = results / "cases-v2"
    case = read(case_root / "case-summary.json")
    if not case["independent_R_passed"] or case["patient_level_data_included"] or case["external_basis_or_signature_selection"]:
        raise ValueError("Case reference, privacy or external selection boundary failed")
    if len(case["analyses"]) != 6 or case["primary_case_V_endpoint"] != "rfs" or case["secondary_case_V_endpoint"] != "dmfs":
        raise ValueError("All case/basis analyses and separate endpoints are required")
    for analysis in case["analyses"]:
        if not analysis["qualified_allowed"] and analysis["optimism_corrected_c"] is not None:
            raise ValueError("Withheld case retained a standard internal signature summary")
        state = analysis["external_configuration"]["state_refresh"]
        if state["numerical_quantities_recomputed"] or not state["prediction_model_and_metrics_unchanged"]:
            raise ValueError("Product-state refresh changed a fixed case prediction")
        if state["product_policy_sha256"] != digest(ROOT / "src/survival_toolkit/marker_qualification.py"):
            raise ValueError("Case state is not bound to the current product policy")
    allowed_names = {"external-aggregate.csv", "pooled-aggregate.json", "configuration.json", "eligibility.json",
                     "independent-R-verification.json", "independent-R-comparison.csv", "independent-R-pooling.csv",
                     "independent-R-values.csv", "R-session.txt"}
    expected_keys = {f"case-{c}-{b}" for c in ("I", "IV", "V") for b in ("linear", "restricted_cubic_spline")}
    if {p.name for p in case_root.iterdir()} != expected_keys | {"case-summary.json", "pooled-case-results.csv"}:
        raise ValueError("Case export does not match the public aggregate allowlist")
    for key in expected_keys:
        directory = case_root / key
        if {p.name for p in directory.iterdir()} != allowed_names:
            raise ValueError("Unexpected case artifact: " + key)
        reference = read(directory / "independent-R-verification.json")
        if not reference["passed"] or reference["unavailable"]:
            raise ValueError("Case reference incomplete: " + key)
        check_hashes(directory, {n: h for n, h in reference["hashes"].items() if n in allowed_names})
    keys = ["stage", "condition", "n", "p", "method"]
    numbers = ["planned", "completed", "failures", "missing", "allowed", "fwer_raw_lower", "fwer", "conditional_fwer", "allowed_fraction", "power"]
    cells = pd.DataFrame(summary["summaries"])
    main_null = cells[cells.stage.eq("main") & ~cells.condition.str.startswith("partial_")]
    shape_conditions = ("independent", "linear", "correlated")
    power = cells[cells.condition.str.startswith("partial_") | (cells.n.eq(180) & cells.condition.isin(shape_conditions))]
    figures = results / "figures"
    source_numbers(figures / "Figure_2_source.csv", main_null, keys, numbers)
    source_numbers(figures / "Figure_3_source.csv", power, keys, numbers)
    source_numbers(figures / "Figure_S1_source.csv", cells, keys, numbers)
    ratios = pd.read_csv(results / "power-retention-uncertainty.csv", float_precision="round_trip")
    source_numbers(figures / "Figure_3_ratio_source.csv", ratios, ["condition", "method"],
                   ["ratio", "paired_delta_MC95_lower", "paired_delta_MC95_upper"])
    pooled = pd.read_csv(case_root / "pooled-case-results.csv", float_precision="round_trip")
    source_numbers(figures / "Figure_5_source.csv", pooled[pooled.scaling.eq("as_measured") & pooled.metric.isin(["delta_c", "calibration_slope"])],
                   ["case", "clinical_basis", "endpoint", "metric"], ["n", "events", "cohorts", "estimate", "hksj_ci_lower", "hksj_ci_upper"])
    primary = pooled[pooled.clinical_basis.eq("linear") & pooled.scaling.eq("as_measured") & pooled.metric.eq("delta_c")]
    if len(primary) != 4 or not ((primary.hksj_ci_lower <= 0) & (primary.hksj_ci_upper >= 0)).all():
        raise ValueError("Recheck the current manuscript claim about primary gain intervals")
    names = ["Figure_1_workflow", "Figure_2_error_and_availability", "Figure_3_power_and_dimension",
             "Figure_4_independent_numerical_validation", "Figure_5_external_reanalysis", "Figure_S1_full_availability"]
    for name in names:
        for suffix in ("png", "pdf", "svg"):
            if not (figures / f"{name}.{suffix}").is_file():
                raise ValueError("Incomplete figure formats: " + name)
    report = {"passed": True, "frozen_numerical_sources_unchanged": True,
              "fixed_datasets": audit["fixed_datasets"], "calculation_failures": audit["calculation_failures"],
              "independent_gate_decisions": 14, "paired_power_uncertainty_reference_passed": True,
              "case_reference_quantities": case["independent_R_quantities"], "case_reference_maximum_difference": case["maximum_absolute_difference"],
              "case_aggregate_allowlist_passed": True, "figure_source_number_checks_passed": True,
              "all_primary_external_gain_intervals_include_zero": True,
              "scope": "Evidence and source-number audit; separate visual inspection, CI and author review are required.",
              "audit_script_sha256": digest(Path(__file__)),
              "input_hashes": {str(p.relative_to(results)): digest(p) for p in sorted(results.rglob("*"))
                               if p.is_file() and p.name != "followup-evidence-audit.json"}}
    (results / "followup-evidence-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "input_hashes"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    run(parser.parse_args().results)

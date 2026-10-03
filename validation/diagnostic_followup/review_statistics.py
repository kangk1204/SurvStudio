"""Post-confirmation descriptive uncertainty and an independent gate-decision audit.

Paired delta intervals describe uncertainty in power ratios. They are not new
qualification gates: the frozen study uses the original point-estimate criterion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd


def run(root, with_r=False):
    study = root / "confirmation-v2"
    summary = json.loads((study / "confirmation-summary.json").read_text())
    records = pd.read_csv(study / "replicate-outcomes.csv.gz", dtype={"error": "string"})
    rows = []
    for condition in ("partial_weak", "partial_strong"):
        table = records[records.stage.eq("main") & records.condition.eq(condition)].pivot(index="index", columns="method", values="guarded_true_rejected") / 5
        if len(table) != 5000 or table.isna().any().any():
            raise ValueError("Complete paired main records are required")
        for method in ("guarded_linear", "guarded_spline"):
            a, b = table[method].to_numpy(), table["legacy_linear"].to_numpy()
            ratio = a.mean() / b.mean()
            se = np.std(a - ratio * b, ddof=1) / np.sqrt(len(a)) / b.mean()
            rows.append({"condition": condition, "method": method, "datasets": len(a), "ratio": ratio,
                         "paired_delta_MC95_lower": ratio - 1.96 * se, "paired_delta_MC95_upper": ratio + 1.96 * se,
                         "preplanned_gate_uses_point_estimate": True, "postconfirmation_descriptive_uncertainty_not_a_new_gate": True})
    results = pd.DataFrame(rows)
    results.to_csv(root / "power-retention-uncertainty.csv", index=False)
    reference = pd.read_csv(study / "independent-R-aggregate-values.csv").pivot(index=["stage", "condition", "n", "p", "method"], columns="quantity", values="value")
    checks = []
    for method, conditions in summary["protocol"]["supported"].items():
        for condition in conditions:
            row = reference.loc["main", condition, 180, 30, method]
            baseline = reference.loc["main", condition, 180, 30, "legacy_linear"]
            partial = condition.startswith("partial_")
            passed = bool(row.fwer_worst_case_upper95 <= .06 and row.conditional_upper95 <= .06
                          and (partial or row.allowed_fraction >= .8)
                          and (not partial or row.power >= .9 * baseline.power_failure_bounds_upper))
            recorded = next(r for r in summary["summaries"] if (r["stage"], r["condition"], r["method"]) == ("main", condition, method))
            decision = "passed" if passed else "failed_exploratory_only"
            checks.append({"method": method, "condition": condition, "R_quantity_based_gate": decision,
                           "recorded_gate": recorded["engineering_status"], "matches": decision == recorded["engineering_status"]})
    if not all(c["matches"] for c in checks):
        raise ValueError("Frozen gate decisions do not match the independent quantities")
    (root / "independent-gate-decision-check.json").write_text(json.dumps({"passed": True, "checks": checks,
        "scope": "Separate decision code applied to independently reconstructed R quantities; original gates unchanged."}, indent=2) + "\n")
    if with_r:
        subprocess.run(["Rscript", str(Path(__file__).with_name("review_statistics.R")), str(root)], check=True)
        r = pd.read_csv(root / "power-retention-independent-R.csv")
        merged = results.merge(r, on=["condition", "method"], suffixes=("_Python", "_R"), validate="one_to_one")
        differences = {name: float(np.max(np.abs(merged[name + "_Python"] - merged[name + "_R"])))
                       for name in ("datasets", "ratio", "paired_delta_MC95_lower", "paired_delta_MC95_upper")}
        report = {"independent_R_passed": all(v <= 1e-12 for v in differences.values()), "differences": differences,
                  "tolerance": 1e-12, "postconfirmation_descriptive_only": True,
                  "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    (Path(__file__), Path(__file__).with_suffix(".R"))},
                  "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (study / "replicate-outcomes.csv.gz", study / "confirmation-summary.json",
                                    study / "independent-R-aggregate-values.csv")},
                  "output_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    (root / "power-retention-uncertainty.csv", root / "power-retention-independent-R.csv",
                                     root / "power-retention-R-session.txt", root / "independent-gate-decision-check.json")}}
        (root / "power-retention-reference-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["independent_R_passed"]:
            raise ValueError("Independent power-ratio uncertainty differs")
    print(json.dumps({"gate_decisions_verified": len(checks), "power_ratio_rows": len(rows), "independent_R_rerun": with_r}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--with-r", action="store_true")
    a = p.parse_args()
    run(a.results, a.with_r)

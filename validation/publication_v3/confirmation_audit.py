"""Check the complete fixed confirmation grid before evaluating adoption.

This gate is separate from the immutable development controller. It does not
change statistics, thresholds, candidate selection or a production registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from study import protocol, cells, binomial
from control import qualify


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative integer: {name}")
    return value


def audit(summaries, freeze):
    cfg = protocol()
    stages = {c["stage"] for c in cfg["confirmation"]}
    if len(summaries) != len(stages) or {s.get("stage") for s in summaries} != stages:
        raise ValueError("Exactly one summary for each fixed confirmation stage required")
    if freeze.get("selection_eligible") is not True or freeze.get("reference_passed") is not True:
        raise ValueError("Eligible selection and R-verified seal required")
    if freeze.get("candidate") not in cfg["candidates"]:
        raise ValueError("Unknown sealed diagnostic candidate")
    expected_freeze_hash = hashlib.sha256(
        json.dumps({k: v for k, v in freeze.items() if k != "_input_file_sha256"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # The caller supplies the digest of the original, immutable file. Canonical
    # JSON is only a logical fingerprint, never substituted for its byte digest.
    expected_file_hash = freeze.get("_input_file_sha256")
    if expected_file_hash is None:
        raise ValueError("Original seal byte digest required")
    cells_checked = 0
    records_checked = 0
    for summary in summaries:
        stage = summary["stage"]
        if summary.get("complete") is not True:
            raise ValueError("Incomplete confirmation stage")
        configuration = summary["configuration"]
        seed = cfg[("extension" if stage == "large" else stage) + "_seed"]
        if configuration.get("stage") != stage or configuration.get("seed") != seed:
            raise ValueError("Wrong stage or independent confirmation seed")
        if configuration.get("freeze_sha256") != expected_file_hash:
            raise ValueError("Summary does not belong to the supplied seal")
        for key in ("source_hashes", "environment", "candidate"):
            if configuration.get(key) != freeze.get(key):
                raise ValueError(f"Sealed {key} mismatch")
        if summary.get("methods") != cfg["methods"]:
            raise ValueError("All five fixed methods in fixed order required")
        expected = {
            (stage, condition, cell["n"], cell["p"], method): cell["replicates"]
            for cell in cells(stage)
            for condition in cell["conditions"]
            for method in cfg["methods"]
        }
        planned = sum(c["replicates"] * len(c["conditions"]) for c in cells(stage))
        if summary.get("planned_datasets") != planned or summary.get("completed_datasets") != planned:
            raise ValueError("Wrong fixed dataset count")
        found = set()
        for row in summary["summaries"]:
            key = tuple(row[k] for k in ("stage", "condition", "n", "p", "method"))
            if key in found or key not in expected:
                raise ValueError("Duplicate or unplanned method cell")
            found.add(key)
            n = integer(row["planned"], "planned")
            if n != expected[key] or integer(row["completed"], "completed") != n or integer(row["missing"], "missing") != 0:
                raise ValueError("Missing or wrong fixed repetitions")
            failed = integer(row["failures"], "failures")
            allowed = integer(row["allowed"], "allowed")
            diagnostic = integer(row["diagnostic_failures"], "diagnostic_failures")
            if failed + allowed > n or diagnostic > n:
                raise ValueError("Inconsistent failure/allowance counts")
            for field in ("fwer", "fwer_lower95", "fwer_upper95", "allowed_fraction", "allowed_lower95", "withhold_fraction"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid probability: {field}")
            if abs(row["allowed_fraction"] - allowed / n) > 1e-12 or abs(row["withhold_fraction"] - (1 - allowed / n)) > 1e-12:
                raise ValueError("Allowance fraction differs from counts")
            hits = row["fwer"] * n
            if abs(hits - round(hits)) > 1e-9 or round(hits) > allowed:
                raise ValueError("False-discovery rate differs from integer counts")
            hits = round(hits)
            for field, expected_value in (
                ("fwer_lower95", binomial(hits, n, "lower")),
                ("fwer_upper95", binomial(hits + failed, n, "upper")),
                ("allowed_lower95", binomial(allowed, n, "lower")),
            ):
                if abs(row[field] - expected_value) > 1e-12:
                    raise ValueError(f"Monte Carlo bound differs from counts: {field}")
            conditional_fields = ("conditional_fwer", "conditional_lower95", "conditional_upper95")
            if allowed == 0:
                if any(row[field] is not None for field in conditional_fields):
                    raise ValueError("Zero allowed analyses must have null conditional quantities")
            else:
                if any(isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(row[field]) or not 0 <= row[field] <= 1 for field in conditional_fields):
                    raise ValueError("Missing or invalid conditional quantity")
                if abs(row["conditional_fwer"] - round(hits) / allowed) > 1e-12:
                    raise ValueError("Conditional FWER differs from allowed counts")
                for field, side in (("conditional_lower95", "lower"), ("conditional_upper95", "upper")):
                    if abs(row[field] - binomial(hits, allowed, side)) > 1e-12:
                        raise ValueError("Conditional bound differs from counts")
            ratio = row["power_ratio"]
            if ratio is not None:
                for field in ("point", "lower95"):
                    value = ratio[field]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid paired power ratio")
                lower, upper = ratio["mc95"]
                if not isinstance(lower, (int, float)) or not math.isfinite(lower) or lower < 0:
                    raise ValueError("Invalid paired power interval")
                if upper is None:
                    if integer(ratio["undefined_denominator_draws"], "undefined denominator draws") == 0:
                        raise ValueError("Undefined upper interval without undefined draws")
                elif not isinstance(upper, (int, float)) or not math.isfinite(upper) or upper < lower:
                    raise ValueError("Invalid paired power interval")
                if not row["condition"].startswith("partial_"):
                    raise ValueError("Unexpected non-null partial-null power ratio")
            records_checked += n
        if found != set(expected):
            raise ValueError("Required method cells absent")
        cells_checked += len(expected) // len(cfg["methods"])
    if cells_checked != 55 or records_checked != 747500:
        raise ValueError("Wrong complete confirmation grid")
    return {
        "passed": True, "dataset_cells": cells_checked, "method_records": records_checked,
        "freeze_sha256": expected_file_hash, "logical_seal_sha256": expected_freeze_hash,
        "production_promotion": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs=4, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit evidence cannot be overwritten")
    summaries = [json.loads(p.read_text()) for p in args.summaries]
    freeze = json.loads(args.freeze.read_text())
    freeze["_input_file_sha256"] = hashlib.sha256(args.freeze.read_bytes()).hexdigest()
    result = audit(summaries, freeze)
    result["finite_qualification"] = qualify(summaries)
    result["input_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.freeze, *args.summaries]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_hashes"}))


if __name__ == "__main__":
    main()

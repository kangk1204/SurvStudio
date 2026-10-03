"""Carry current product provenance into a fresh private copy of v2 case evidence.

The original completed case run is preserved. Predictions, bootstrap draws, metrics,
transformations and resampling calculations are reused unchanged; only inference
provenance and its recipe hash are refreshed. Independent R is rerun on the copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/guarded_inference"))
from reanalyse_cases import save
from survival_toolkit.marker_qualification import qualify_marker_result, qualify_external_result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(original, output, rlib):
    original, output = original.resolve(), output.resolve()
    if output.is_relative_to(ROOT) or original.is_relative_to(ROOT):
        raise ValueError("Real-case evidence must remain outside Git")
    if not (original / "complete.json").exists():
        raise ValueError("All original v2 case computations must be complete")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original / "cases", output / "cases")
    for case in ("I", "IV", "V"):
        for basis in ("linear", "restricted_cubic_spline"):
            key = f"case-{case}-{basis}"
            old = original / "cases_final" / key
            reference = json.loads((old / "independent-R-verification.json").read_text())
            if not reference["passed"]:
                raise ValueError("Original independent reference failed")
            for name, expected in reference["hashes"].items():
                if digest(old / name) != expected:
                    raise ValueError("Original case evidence changed: " + key + "/" + name)
            out = output / "cases_final" / key
            excluded = {"independent-R-verification.json", "independent-R-comparison.csv", "independent-R-pooling.csv", "independent-R-values.csv", "R-session.txt"}
            shutil.copytree(old, out, ignore=lambda directory, names: [n for n in names if n in excluded])
            raw = json.loads((output / "cases" / key / "development-analysis.json").read_text())
            if raw["method_version"] != "marker-inference/2":
                raise ValueError("Do not relabel historical v1 calculations")
            result = qualify_marker_result(raw)
            recipe = result["locked_recipe"]
            original_recipe = json.loads((old / "recipe.json").read_text())
            for name in ("model", "clinical", "markers", "outcome"):
                if recipe[name] != original_recipe[name]:
                    raise ValueError("State refresh cannot change the fixed prediction")
            save(out / "recipe.json", recipe)
            for path in sorted(out.glob("*-report.json")):
                report = json.loads(path.read_text())
                metrics = json.dumps(report["metrics"], sort_keys=True)
                report = qualify_external_result(report, recipe)
                report["recipe_hash"] = recipe["recipe_hash"]
                if report.get("notes"):
                    report["notes"] = list(dict.fromkeys(report["notes"]))
                if json.dumps(report["metrics"], sort_keys=True) != metrics:
                    raise ValueError("State refresh changed prediction metrics")
                save(path, report)
            config = json.loads((out / "configuration.json").read_text())
            config["engineering_qualification"] = result["inference"]["engineering_qualification"]
            config["state_refresh"] = {"original_configuration_sha256": digest(old / "configuration.json"),
                                       "original_recipe_sha256": digest(old / "recipe.json"),
                                       "script_sha256": digest(Path(__file__)),
                                       "product_policy_sha256": digest(ROOT / "src/survival_toolkit/marker_qualification.py"),
                                       "product_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                       "numerical_quantities_recomputed": False,
                                       "prediction_model_and_metrics_unchanged": True}
            config["source_hashes"]["src/survival_toolkit/marker_qualification.py"] = config["state_refresh"]["product_policy_sha256"]
            save(out / "configuration.json", config)
            complete = json.loads((out / "complete.json").read_text())
            complete.update(configuration=config, development_inference=result["inference"], independent_R_status="see_independent-R-verification.json")
            save(out / "complete.json", complete)
            subprocess.run([sys.executable, str(ROOT / "validation/guarded_inference/verify_cases.py"), "--output", str(out), "--rlib", str(rlib)], check=True)
    subprocess.run([sys.executable, str(ROOT / "validation/guarded_inference/collect_case_aggregate.py"), "--evidence", str(output), "--output", str(output / "aggregate")], check=True)
    save(output / "complete.json", {"state_refresh_only": True, "original_completion_sha256": digest(original / "complete.json"),
                                   "original_preserved": True, "independent_R_rerun": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rlib", type=Path, required=True)
    a = p.parse_args()
    run(a.original, a.output, a.rlib)

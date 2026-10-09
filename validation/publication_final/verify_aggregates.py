"""Verify stored aggregate integrity and key reported boundaries without rerunning studies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    def read_csv(name):
        with (root / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    manifest = json.loads((root / "aggregate-manifest.json").read_text())
    for name, expected in manifest.items():
        check("aggregate hash: " + name,
              hashlib.sha256((root / name).read_bytes()).hexdigest() == expected)
    summary = json.loads((root / "evidence-summary.json").read_text())
    main = read_csv("v3_main.csv")
    check("120 paired main method cells", len(main) == 120)
    check("main fixed planned count", all(int(row["planned"]) == 5000 for row in main))
    check("main complete and no reported failures", all(
        int(row["completed"]) == 5000 and int(row["missing"]) == 0
        and int(row["failures"]) == 0 and int(row["diagnostic_failures"]) == 0
        for row in main))
    by_key = {(row["condition"], int(row["p"]), row["method"]): row for row in main}
    sensitivity_losses = {}
    for markers in (30, 300):
        a = by_key[("heteroskedastic", markers, "v2_linear")]
        b = by_key[("heteroskedastic", markers, "v3_linear")]
        loss = float(a["withhold_fraction"]) - float(b["withhold_fraction"])
        sensitivity_losses[str(markers)] = loss
        check(f"v3 sensitivity adoption failure P{markers}", loss > 0.05)
    check("v3 not adopted", summary["v3_adoption"] == "NO_GO")
    check("extension original planned denominator", summary["v3_extension_valid"]
          + summary["v3_extension_unusable"] + summary["v3_extension_absent"] == 12000)
    check("unreadable original result retained", len(read_csv("invalid_original_records.csv")) == 1)
    grid = read_csv("v3_completeness.csv")
    check("149500 planned indices preserved", sum(int(row["planned"]) for row in grid) == 149500)
    tasks = read_csv("tool_tasks.csv")
    check("six tools by eight tasks retained", len(tasks) == 48)
    check("web pending not treated as failed feature", all(
        row["display_code"] == ("P" if row["tool"] == "surviveR" else "L")
        for row in tasks if row["tool"] in ("surviveR", "KM Plotter")))
    check("no unconfirmed submission claim", summary["submission_ready"] is False)
    states = json.loads((root / "state_surfaces-verification.json").read_text())
    fresh = json.loads((root / "fresh_environment-verification.json").read_text())
    check("surface verification", states["passed"])
    check("fresh artifact verification", fresh["passed"] and len(fresh["file_hash_checks"]) == 9
          and all(fresh["file_hash_checks"].values()))
    public = [row for row in read_csv("public_external_case.csv") if float(row["horizon_years"]) == 5]
    check("all four public benchmark sensitivities retained", len(public) == 4)
    check("external results do not lift withholding", all(row["inference_status"] == "withheld" for row in public))
    return {"passed": all(row["passed"] for row in checks), "checks": checks,
            "sensitivity_loss": sensitivity_losses,
            "scope": "Integrity and stated aggregate boundaries; not an independent statistical recalculation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.inputs)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

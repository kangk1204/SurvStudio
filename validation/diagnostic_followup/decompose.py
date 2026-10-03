"""Read-only decomposition of the completed, failed v1 confirmation study.

This is a post-confirmation failure investigation, not a new confirmation study.
Overlapping reasons are counted once per dataset within each diagnostic category.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(reason: str) -> str:
    if reason.startswith("clinical_misspecification: PH"):
        return "PH"
    if reason.startswith("clinical_misspecification: functional_form_LR"):
        return "functional_form"
    if reason.startswith("residual_misspecification:"):
        return "residual_mean" if reason.endswith("/nonlinear_mean") else "residual_variance"
    if "diagnostic_failed" in reason:
        return "diagnostic_failure"
    return "clinical_failure_or_other"


def investigate(root: Path, output: Path) -> dict:
    cells = defaultdict(Counter)
    overlaps = defaultdict(Counter)
    failures, sources, seen = [], [], set()
    for path in sorted(root.rglob("*.sqlite")):
        digest = sha256(path)
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            config = json.loads(db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()[0])
            cell = tuple(config[k] for k in ("stage", "condition", "n", "p"))
            count = 0
            for index, host, result in db.execute("SELECT idx, host, result FROM replicates ORDER BY idx"):
                identity = (*cell, index)
                if identity in seen:
                    raise ValueError(f"Duplicate dataset ownership: {identity}")
                seen.add(identity)
                count += 1
                rows = json.loads(result)
                if sorted(row["method"] for row in rows) != ["guarded_linear", "guarded_spline", "legacy_linear"]:
                    raise ValueError(f"Missing or duplicate v1 method: {identity}")
                for row in rows:
                    key = (*cell, row["method"])
                    totals = cells[key]
                    totals["completed"] += 1
                    totals["allowed"] += bool(row["allowed"] and not row["failure"])
                    totals["failure"] += bool(row["failure"])
                    categories = {category(reason) for reason in row.get("reasons", [])}
                    if row["failure"]:
                        categories.add("calculation_failure")
                        failures.append(dict(stage=cell[0], condition=cell[1], n=cell[2], p=cell[3],
                                             index=index, host=host, ledger=str(path.relative_to(root)), **row))
                    totals.update(categories)
                    overlaps[key]["+".join(sorted(categories)) or "none"] += 1
                    if not row["allowed"] and not categories:
                        raise ValueError(f"Unexplained withholding: {identity}, {row['method']}")
        if sha256(path) != digest:
            raise ValueError(f"Ledger changed while being read: {path}")
        sources.append({"ledger": str(path.relative_to(root)), "sha256": digest,
                        "datasets": count, "source_hashes": config["source_hashes"],
                        "freeze_hash": config["freeze_hash"]})
    if not seen:
        raise ValueError("No ledgers found")
    output.mkdir(parents=True, exist_ok=False)
    categories = ["PH", "functional_form", "residual_mean", "residual_variance",
                  "diagnostic_failure", "clinical_failure_or_other", "calculation_failure"]
    columns = ["stage", "condition", "n", "p", "method", "completed", "allowed", "failure", *categories]
    with (output / "reason-counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, totals in sorted(cells.items()):
            writer.writerow({**dict(zip(columns[:5], key)), **{k: totals[k] for k in columns[5:]}})
    with (output / "reason-overlaps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*columns[:5], "reason_combination", "datasets"])
        for key, counts in sorted(overlaps.items()):
            for reason, count in sorted(counts.items()):
                writer.writerow([*key, reason, count])
    (output / "original-failure-context.json").write_text(json.dumps(failures, indent=2) + "\n")
    (output / "source-ledgers.json").write_text(json.dumps(sources, indent=2) + "\n")
    summary = {"purpose": "post-confirmation v1 failure investigation; descriptive only",
               "datasets": len(seen), "method_records": sum(t["completed"] for t in cells.values()),
               "ledgers": len(sources), "failed_method_records": len(failures),
               "reason_counts_overlap": True, "originals_unchanged": True,
               "script_sha256": sha256(Path(__file__)),
               "files": {p.name: sha256(p) for p in sorted(output.iterdir())}}
    (output / "investigation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(investigate(args.ledgers, args.output)))

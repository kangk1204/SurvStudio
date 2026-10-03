"""Publish only complete synthetic v2 outcomes, ledger ownership and aggregate evidence.

The original v1 publisher and results remain untouched. The output schema permits
the existing independent base-R audit to verify the new study without new formulas.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(ledgers, summary_path, freeze_path, output):
    summary = json.loads(summary_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if not summary.get("complete") or summary["protocol"].get("method_version") != "marker-inference/2":
        raise ValueError("Complete v2 confirmation is required")
    if freeze.get("method_version") != "marker-inference/2":
        raise ValueError("Wrong frozen method")
    expected = {(r["stage"], r["condition"], r["n"], r["p"], r["method"]): r["planned"] for r in summary["summaries"]}
    if len(expected) != 117 or any(r["missing"] for r in summary["summaries"]):
        raise ValueError("Incomplete or duplicate confirmation cells")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(summary_path, output / "confirmation-summary.json")
    shutil.copyfile(freeze_path, output / "freeze.json")
    flat = []
    for row in summary["summaries"]:
        item = {k: v for k, v in row.items() if k != "source_hashes"}
        for key, value in list(item.items()):
            if isinstance(value, list):
                del item[key]
                item[key + "_lower"], item[key + "_upper"] = value
        flat.append(item)
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with (output / "confirmation-cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(flat)
    outcome_fields = ["stage", "condition", "n", "p", "index", "method", "host", "worker", "allowed", "calculation_failed", "diagnostic_failed",
                      "raw_null_rejected", "guarded_null_rejected", "raw_true_rejected", "guarded_true_rejected", "n_true", "error"]
    seen, ownership, failures, sizes = set(), [], [], {}
    freeze_hash = digest(freeze_path)
    inventories = [json.loads(path.read_text()) for path in sorted(ledgers.glob("*/ownership.json"))]
    owners = [worker for inventory in inventories for worker in inventory["workers"]]
    totals = {inventory["total_workers"] for inventory in inventories}
    if not inventories or len(totals) != 1 or sorted(owners) != list(range(next(iter(totals)))):
        raise ValueError("Incomplete or duplicate global worker ownership")
    workers = next(iter(totals))
    if any(inventory["freeze_hash"] != freeze_hash for inventory in inventories):
        raise ValueError("Mixed ownership freeze")
    with (output / "replicate-outcomes.csv.gz").open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, outcome_fields)
                writer.writeheader()
                for path in sorted(ledgers.rglob("*.sqlite")):
                    original_hash = digest(path)
                    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                        config = json.loads(db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()[0])
                        if config["freeze_hash"] != freeze_hash or config["source_hashes"] != freeze["source_hashes"]:
                            raise ValueError("Mixed numerical sources or freeze: " + str(path))
                        if config["python_version"] != freeze["python_version"] or config["package_versions"] != freeze["package_versions"]:
                            raise ValueError("Mixed frozen environment")
                        worker = int(re.search(r"worker(\d+)$", path.stem)[1])
                        cell = tuple(config[k] for k in ("stage", "condition", "n", "p"))
                        count, hosts = 0, set()
                        for index, host, result in db.execute("SELECT idx,host,result FROM replicates ORDER BY idx"):
                            identity = (*cell, index)
                            planned = expected.get((*cell, "legacy_linear"))
                            if planned is None or not 0 <= index < planned or index % workers != worker or identity in seen:
                                raise ValueError("Duplicate, unowned or out-of-protocol fixed index")
                            seen.add(identity)
                            count += 1
                            hosts.add(host)
                            rows = json.loads(result)
                            if sorted(r["method"] for r in rows) != sorted(summary["protocol"]["methods"]):
                                raise ValueError("Missing or duplicate method")
                            for row in rows:
                                key = (*cell, row["method"])
                                sizes[key] = sizes.get(key, 0) + 1
                                failure = bool(row["failure"])
                                if failure:
                                    failures.append({"stage": cell[0], "condition": cell[1], "n": cell[2], "p": cell[3],
                                                     "index": index, "ledger": str(path.relative_to(ledgers)), **row})
                                writer.writerow({**dict(zip(outcome_fields[:4], cell)), "index": index, "method": row["method"], "host": host,
                                                 "worker": worker, "allowed": int(row["allowed"] and not failure), "calculation_failed": int(failure),
                                                 "diagnostic_failed": int(row.get("diagnostic_failed", False)),
                                                 "raw_null_rejected": None if failure else int(row.get("raw_null_rejected", row["null_rejected"])),
                                                 "guarded_null_rejected": int(row["null_rejected"]),
                                                 "raw_true_rejected": None if failure else row.get("raw_true_rejected", row["true_rejected"]),
                                                 "guarded_true_rejected": row["true_rejected"], "n_true": row["n_true"], "error": row.get("error", "")})
                    if digest(path) != original_hash:
                        raise ValueError("Ledger changed during read")
                    ownership.append({"ledger": str(path.relative_to(ledgers)), "sha256": original_hash, "worker": worker,
                                      **dict(zip(outcome_fields[:4], cell)), "records": count, "hosts": ";".join(sorted(hosts))})
    if len(seen) != 85500 or sizes != expected:
        raise ValueError("Incomplete or out-of-protocol index coverage")
    with (output / "ownership-ledgers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(ownership[0]))
        writer.writeheader()
        writer.writerows(ownership)
    (output / "confirmation-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    isolation = output / "source-isolation"
    isolation.mkdir()
    for path in sorted(ledgers.glob("*/preflight.json")):
        preflight = json.loads(path.read_text())
        if preflight["freeze_sha256"] != freeze_hash or preflight["sources"] != freeze["source_hashes"]:
            raise ValueError("Source-isolation check mismatch")
        shutil.copyfile(path, isolation / (path.parent.name + ".json"))
    if len(list(isolation.iterdir())) != len(inventories):
        raise ValueError("Every host needs a source-isolation preflight")
    manifest = {"scope": "synthetic fixed-index outcomes, aggregate numerical results and provenance only",
                "patient_level_case_data_included": False, "manuscript_included": False,
                "frozen_study_git_revision": freeze["git_revision"], "freeze_sha256": freeze_hash,
                "fixed_datasets": len(seen), "paired_method_records": sum(sizes.values()),
                "calculation_failures": len(failures), "ledger_files": len(ownership),
                "model_release_status": summary["model_release_status"],
                "export_script_sha256": digest(Path(__file__)),
                "audit_outputs_excluded": ["aggregate-audit.json", "independent-R-aggregate-values.csv", "independent-R-aggregate-comparison.csv", "aggregate-R-session.txt"],
                "files": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "aggregate-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.ledgers, args.summary, args.freeze, args.output)

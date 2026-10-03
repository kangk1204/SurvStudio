"""Seal v2 only after complete new development and fresh independent R checks."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from confirmation import ROOT, PROTOCOL, hashes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--development", required=True, type=Path)
parser.add_argument("--reference", required=True, type=Path)
parser.add_argument("--separation-reference", required=True, type=Path)
parser.add_argument("--review-record", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("A sealed specification cannot be overwritten")
development = json.loads(args.development.read_text())
reference = json.loads(args.reference.read_text())
separation = json.loads(args.separation_reference.read_text())
if not development.get("complete") or development.get("datasets") != 6000 or development.get("status") != "development_only_not_qualification":
    raise SystemExit("All 6000 new development datasets are required")
for name, digest in development["source_hashes"].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise SystemExit("Development source changed: " + name)
for evidence in (reference, separation):
    if evidence.get("passed") is not True:
        raise SystemExit("Both independent R checks must pass")
    for name, digest in evidence["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit("Independent R source changed: " + name)
sources = hashes()
if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", *sources], text=True).strip():
    raise SystemExit("Commit numerical sources before sealing")
protocol = json.loads(PROTOCOL.read_text())
if {protocol[k] for k in ("main_seed", "extension_seed")} & {2026100301, 2026100302, 2026100303, 2026100311}:
    raise SystemExit("Confirmation seeds must be unused")
manifest = {"git_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_hashes": sources, "method_version": protocol["method_version"],
            "python_version": platform.python_version(),
            "package_versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "statsmodels")},
            "reference_verification": reference, "separation_verification": separation,
            "development_review_complete": True,
            "development_summary_sha256": hashlib.sha256(args.development.read_bytes()).hexdigest(),
            "development_review_record": args.review_record.read_text(),
            "confirmation_policy": protocol["confirmation_policy"]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"revision": manifest["git_revision"], "freeze_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))

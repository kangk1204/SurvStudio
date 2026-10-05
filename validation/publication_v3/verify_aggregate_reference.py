"""Recompute study aggregates in R from raw decisions and fixed resampling rows.

This checks aggregation, not the statistical validity of each Python decision.
The separate raw-data R oracles check basis, Cox and diagnostic computations.
R never reads the Python aggregate values. All 9,999 Monte Carlo draws are kept.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("aggregate_fixed_study", DIRECTORY / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
DRAW_COUNT = 9999
RAW_FIELDS = ("failure", "allowed", "null_rejected", "true_rejected", "n_true")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_stream(path, columns, seed):
    """Preserve NumPy's default integer stream before portable int32 encoding."""
    rng = np.random.default_rng(seed)
    with Path(path).open("xb") as stream:
        for start in range(0, DRAW_COUNT, 128):
            indices = rng.integers(0, columns, size=(min(128, DRAW_COUNT-start), columns))
            indices.astype("<i4").tofile(stream)
    return dict(sha256=digest(path), bytes=Path(path).stat().st_size, rows=DRAW_COUNT,
                columns=columns, encoding="little-endian signed int32, row-major, zero-based")


def read_ledgers(paths):
    records = {}; configurations = []; sources = {}
    for path in paths:
        path = Path(path).resolve(); before = digest(path)
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
            found = db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()
            if found is None:
                raise ValueError("Missing ledger configuration")
            cfg = json.loads(found[0]); configurations.append(cfg)
            for condition, n, p, index, elapsed, encoded in db.execute("SELECT condition,n,p,idx,elapsed,result FROM replicates"):
                key = (condition, n, p, index)
                if key in records or index % cfg["owners"] != cfg["owner"]:
                    raise ValueError("Duplicate fixed index or wrong owner")
                records[key] = (json.loads(encoded), elapsed)
        if digest(path) != before:
            raise ValueError("Ledger changed during independent verification")
        sources[str(path)] = before
    if not configurations:
        raise ValueError("No raw ledgers supplied")
    first = configurations[0]
    for cfg in configurations:
        for name in ("stage", "seed", "owners", "source_hashes", "environment", "freeze_sha256", "candidate"):
            if cfg[name] != first[name]:
                raise ValueError("Mixed raw ledger configurations")
    return records, first, sources


def raw_cell(records, cell, condition, method, methods):
    values = []; reasons = []
    for index in range(cell["replicates"]):
        found = records.get((condition, cell["n"], cell["p"], index))
        if found is None:
            continue
        result, elapsed = found
        by_method = {r["method"]: r for r in result}
        if len(by_method) != len(result) or set(by_method) != set(methods):
            raise ValueError("Missing or duplicated raw method")
        selected = by_method[method]; baseline = by_method["legacy_linear"]
        for row in (selected, baseline):
            for flag in ("failure", "allowed", "null_rejected"):
                if type(row[flag]) is not bool:
                    raise ValueError("Raw indicators must be booleans")
            for count in ("true_rejected", "n_true"):
                if type(row[count]) is not int or not 0 <= row[count]:
                    raise ValueError("Raw marker counts must be nonnegative integers")
            if row["true_rejected"] > row["n_true"]:
                raise ValueError("Raw true-hit count exceeds signal count")
            if row["failure"] and (row["allowed"] or row["null_rejected"] or row["true_rejected"]):
                raise ValueError("Failed computation cannot contribute a discovery")
            if row["null_rejected"] and not row["allowed"]:
                raise ValueError("Withheld computation cannot contribute a discovery")
        value = dict(index=index, elapsed=elapsed, diagnostic_seconds=selected.get("diagnostic_seconds", 0.),
                     diagnostic_failure=int(selected.get("diagnostic_failure", False)),
                     raw_null_rejected=int(selected.get("raw_null_rejected", selected["null_rejected"])))
        for name in RAW_FIELDS:
            value[name] = int(selected[name]); value["baseline_"+name] = int(baseline[name])
        values.append(value)
        for reason, count in selected.get("reasons", {}).items():
            if not isinstance(reason, str) or type(count) is not int or count <= 0:
                raise ValueError("Invalid raw withholding reason")
            reasons.append(dict(index=index, reason=reason, count=count))
    return values, reasons


def compare(actual, expected, path="", tolerance=1e-12):
    """Keep null, zero, integer indicators and missing fields distinct."""
    if isinstance(expected, dict):
        if not expected and actual == []:  # jsonlite serializes an empty named list as [].
            actual = {}
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return [dict(quantity=path, passed=False, reason="object keys differ")]
        return [c for key in expected for c in compare(actual[key], expected[key], path+"/"+key, tolerance)]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [dict(quantity=path, passed=False, reason="array shape differs")]
        return [c for i, value in enumerate(expected) for c in compare(actual[i], value, path+f"/{i}", tolerance)]
    if expected is None or isinstance(expected, (str, bool, int)):
        return [dict(quantity=path, passed=type(actual) is type(expected) and actual == expected, exact=True)]
    if isinstance(expected, float):
        finite = type(actual) in (int, float) and np.isfinite(actual) and np.isfinite(expected)
        difference = abs(float(actual)-expected) if finite else None
        return [dict(quantity=path, passed=bool(finite and difference <= tolerance), maximum_difference=difference)]
    raise TypeError("Unsupported aggregate value")


def verify(paths, summary_path, output, r_library=None, retain_streams=False, synthetic=False):
    records, configuration, ledger_hashes = read_ledgers(paths)
    summary_path = Path(summary_path)
    summary_hash = digest(summary_path)
    source_hashes = study.hashes()
    source_hashes.update({str(p.relative_to(study.ROOT)):digest(p) for p in (Path(__file__), DIRECTORY/"aggregate_reference.R")})
    expected = json.loads(summary_path.read_text())
    if expected["configuration"] != configuration:
        raise ValueError("Summary does not bind the supplied first raw ledger")
    cfg = study.protocol(); stage = configuration["stage"]
    methods = cfg["development_methods"] if stage in {"cost", "screen", "selection"} else cfg["methods"]
    if expected["methods"] != methods or expected["stage"] != stage:
        raise ValueError("Summary method order or stage differs from fixed raw study")
    cells = study.cells(stage)
    planned_keys = {(condition, c["n"], c["p"], i) for c in cells for condition in c["conditions"] for i in range(c["replicates"])}
    if set(records)-planned_keys:
        raise ValueError("Unexpected raw fixed index")
    expected_rows = {(r["condition"], r["n"], r["p"], r["method"]): r for r in expected["summaries"]}
    row_keys = {(condition, c["n"], c["p"], method) for c in cells for condition in c["conditions"] for method in methods}
    if set(expected_rows) != row_keys or len(expected_rows) != len(expected["summaries"]):
        raise ValueError("Incomplete or duplicated summary cells")
    checks = []; manifests = []
    for name, actual in (("planned_datasets", len(planned_keys)), ("completed_datasets", len(records)),
                         ("complete", set(records) == planned_keys)):
        checks.extend(compare(actual, expected[name], name))
    for cell in cells:
        for condition in cell["conditions"]:
            condition_index = (cfg["conditions"]+cfg["stress_conditions"]).index(condition)
            for method_index, method in enumerate(methods):
                name = f"{condition}-N{cell['n']}-P{cell['p']}-{method}"
                directory = output/name; directory.mkdir()
                values, reasons = raw_cell(records, cell, condition, method, methods)
                columns = ["index", "elapsed", "diagnostic_seconds", "diagnostic_failure", "raw_null_rejected",
                           *[key for field in RAW_FIELDS for key in (field, "baseline_"+field)]]
                for filename, rows, fields in (("raw-indicators.csv", values, columns), ("raw-reasons.csv", reasons, ["index", "reason", "count"])):
                    with (directory/filename).open("x", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
                partial = condition.startswith("partial_")
                settings = dict(stage=stage, condition=condition, n=cell["n"], p=cell["p"], method=method,
                                planned=cell["replicates"], partial_null=partial,
                                healthy=condition in cfg["healthy_original"]+cfg["healthy_stress"], draws=DRAW_COUNT)
                (directory/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
                base_seed = [cfg["mc_seed"], cell["n"], cell["p"], condition_index, method_index]
                streams = {}
                if partial:
                    streams["ratio-rows.i32"] = row_stream(directory/"ratio-rows.i32", cell["replicates"], np.random.SeedSequence(base_seed))
                pairs = sum(not v["failure"] and not v["baseline_failure"] for v in values)
                if pairs:
                    streams["difference-rows.i32"] = row_stream(directory/"difference-rows.i32", pairs, np.random.SeedSequence(base_seed+[11]))
                command = ["Rscript", str(DIRECTORY/"aggregate_reference.R"), str(directory)]
                if r_library: command.append(r_library)
                subprocess.run(command, check=True)
                actual = json.loads((directory/"r-summary.json").read_text())
                key = (condition, cell["n"], cell["p"], method)
                checks.extend(compare(actual, expected_rows[key], name))
                manifests.append(dict(cell=name, resampling_seed_words=base_seed, difference_seed_suffix=11,
                                      streams=streams, retained_streams=retain_streams,
                                      input_hashes={f:digest(directory/f) for f in ("settings.json", "raw-indicators.csv", "raw-reasons.csv")},
                                      r_output_sha256=digest(directory/"r-summary.json")))
                if not retain_streams:
                    for filename, info in streams.items():
                        if digest(directory/filename) != info["sha256"]:
                            raise ValueError("Resampling stream changed after R verification")
                        (directory/filename).unlink()
    for filename, expected_hash in ledger_hashes.items():
        if digest(filename) != expected_hash:
            raise ValueError("Raw ledger changed during verification")
    if digest(summary_path) != summary_hash:
        raise ValueError("Summary changed during verification")
    for name, expected_hash in source_hashes.items():
        if digest(study.ROOT/name) != expected_hash:
            raise ValueError("Numerical reference source changed during verification")
    report = dict(passed=all(c["passed"] for c in checks), checks=checks, cells=len(manifests), draws=DRAW_COUNT,
                  synthetic_fixture_only=synthetic, stage=stage, summary_sha256=summary_hash, ledger_hashes=ledger_hashes,
                  source_hashes=source_hashes, environment=study.environment(), resampling_manifests=manifests,
                  tolerance=1e-12, boundary="Independent R aggregation uses raw replicate decisions and shared row streams only. It does not independently recompute every simulated diagnostic or certify statistical assumptions.")
    (output/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    failures = [c for c in checks if not c["passed"]]
    print(json.dumps(dict(passed=report["passed"], cells=len(manifests), checks=len(checks), failed=failures[:20])))
    if failures: raise SystemExit(1)
    return report


def self_test(output):
    """Deliberately include missing, failed, withheld and undefined-denominator cells."""
    cfg = study.protocol()
    cfg["confirmation"] = [dict(stage="main", n=40, p=3, replicates=9, conditions=["independent", "partial_weak", "partial_strong"]),
                           dict(stage="main", n=40, p=4, replicates=9, conditions=["partial_strong"]),
                           dict(stage="main", n=40, p=5, replicates=9, conditions=["independent"])]
    study.protocol = lambda: cfg
    ledger = output/"synthetic-raw.sqlite"
    configuration = dict(stage="main", seed=cfg["main_seed"], owner=0, owners=1, host="synthetic-only",
                         source_hashes=study.hashes(), environment=study.environment(), freeze_sha256="synthetic-not-a-seal", candidate="residual_vector")
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        db.execute("INSERT INTO metadata VALUES('configuration',?)", (json.dumps(configuration, sort_keys=True),))
        db.execute("CREATE TABLE replicates(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,elapsed REAL,result TEXT)")
        db.execute("CREATE TABLE attempts(condition TEXT,n INTEGER,p INTEGER,idx INTEGER,started_utc TEXT,ended_utc TEXT,status TEXT,error TEXT)")
        for cell in cfg["confirmation"]:
            for condition in cell["conditions"]:
                for index in range(9):
                    if cell["p"] == 5: continue  # Entire planned cell unresolved.
                    # One partial cell has a complete but identically zero-power baseline.
                    if index == 8 and cell["p"] != 4: continue
                    results = []
                    for j, method in enumerate(cfg["methods"]):
                        n_true = 0 if condition == "independent" else 2
                        failure = (index == 6 and method == "legacy_linear" and cell["p"] != 4) or (index == 5 and method == "v3_spline")
                        diagnostic_failure = index == 4 and method == "v3_linear"
                        allowed = not failure and not diagnostic_failure and not (method == "v3_spline" and condition == "independent")
                        raw_null = index in (1, 5) and not failure
                        raw_true = 0 if not n_true or cell["p"] == 4 else ((index+j) % 3 if condition == "partial_weak" else int(index == 0))
                        results.append(dict(method=method, failure=failure, diagnostic_failure=diagnostic_failure,
                                            allowed=allowed, null_rejected=bool(raw_null and allowed), raw_null_rejected=raw_null,
                                            true_rejected=raw_true if allowed else 0, raw_true_rejected=raw_true,
                                            n_true=n_true, reasons={"diagnostic_exception":1} if diagnostic_failure else {"fixture_misfit":2} if not allowed and not failure else {},
                                            diagnostic_seconds=0. if method == "legacy_linear" else .1*j))
                    db.execute("INSERT INTO replicates VALUES(?,?,?,?,?,?)", (condition, cell["n"], cell["p"], index, .5+.01*index, json.dumps(results)))
                    db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,NULL)", (condition,cell["n"],cell["p"],index,
                        "2026-10-03T00:00:00+00:00","2026-10-03T00:00:01+00:00","completed_with_scientific_failure" if any(r["failure"] or r["diagnostic_failure"] for r in results) else "completed"))
    summary = output/"synthetic-summary.json"
    study.summarize(SimpleNamespace(ledgers=[ledger], output=summary))
    return [ledger], summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", type=Path, nargs="+"); parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--r-library")
    parser.add_argument("--retain-streams", action="store_true"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    ledgers, summary = self_test(args.output) if args.self_test else (args.ledgers, args.summary)
    if not ledgers or not summary: parser.error("Actual verification requires --ledgers and --summary")
    verify(ledgers, summary, args.output, args.r_library, args.retain_streams, args.self_test)

"""Paired, append-only diagnostic development; never a confirmation launcher."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import sqlite3
import sys
import time
import traceback

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation" / "guarded_inference"))
from study import dataset, calculation, cp_interval
from survival_toolkit.marker_evaluation import prepare_marker_cohort
from survival_toolkit.marker_diagnostics import diagnose_markers, holm, METHOD_VERSION
import survival_toolkit.marker_screen as screen_module

PROTOCOL = Path(__file__).with_name("protocol.json")
SOURCE_FILES = ["validation/diagnostic_followup/development.py", "validation/diagnostic_followup/protocol.json",
                "validation/guarded_inference/study.py", "validation/guarded_inference/protocol.json",
                *[f"src/survival_toolkit/{name}.py" for name in
                  ("marker_screen", "marker_diagnostics", "marker_evaluation", "clinical_basis", "encoding", "analysis")]]


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


@contextmanager
def original_separation():
    """Isolated-process reconstruction of the original v1 flag, not its qualification."""
    current = screen_module._runaway_coefficients

    def original(design, beta, score, information):
        flags = current(design, beta, score, information)
        if np.isfinite(beta).all() and np.isfinite(score).all() and np.isfinite(information).all():
            scale = np.std(design, axis=0)
            flags = flags | ((np.abs(beta) * scale > 10.0) & (scale > 0))
        return flags

    screen_module._runaway_coefficients = original
    try:
        yield
    finally:
        screen_module._runaway_coefficients = current


def f_sensitivity(diagnostic):
    """F(q,n-k) sensitivity only: HC3 makes this an approximation, not an exact F test."""
    result = copy.deepcopy(diagnostic)
    result["method_version"] = "candidate-HC3-F/1"
    result["reasons"] = [r for r in result["reasons"] if not r.startswith("residual_misspecification:")]
    tests = result["residual_tests"]
    for test in tests:
        if test["status"] == "calculated":
            test["p_value"] = float(stats.f.sf(test["statistic"] / test["df"], test["df"], test["df_residual"]))
    for test, adjusted in zip(tests, holm([t["p_value"] for t in tests])):
        test["p_holm"] = adjusted
        if adjusted is not None and adjusted <= result["threshold"]:
            result["reasons"].append("residual_misspecification: " + test["marker"] + "/" + test["diagnostic"])
    result.update(allowed=not result["reasons"], status="withheld" if result["reasons"] else "assumption_dependent")
    return result


def exception_context(exc):
    """Synthetic-only exception frame metadata, including the original unmodified traceback."""
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        values = {}
        for name in ("block", "sorted_block", "row", "index", "source_rows", "permutation", "running"):
            if name not in frame.f_locals:
                continue
            value = frame.f_locals[name]
            if isinstance(value, np.ndarray):
                values[name] = {"type": type(value).__name__, "dtype": str(value.dtype), "shape": list(value.shape)}
            elif isinstance(value, (int, np.integer)):
                values[name] = {"type": type(value).__name__, "value": int(value)}
            else:
                values[name] = {"type": type(value).__name__}
        frames.append({"file": frame.f_code.co_filename, "function": frame.f_code.co_name,
                       "line": tb.tb_lineno, "locals": values})
        tb = tb.tb_next
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "frames": frames}


def paired(condition, index, protocol):
    frame, names, signal, perm_seed = dataset(condition, protocol["n"], protocol["p"],
                                            protocol["development_seed"], index)
    results = []
    for basis, suffix in (("linear", "linear"), ("restricted_cubic_spline", "spline")):
        methods = [f"{prefix}_{suffix}" for prefix in ("v1_guarded", "v2_guarded", "candidate_F")]
        try:
            cohort = prepare_marker_cohort(frame, time_column="time", event_column="event",
                                          marker_columns=names, clinical_columns=["Z"], clinical_basis=basis)
            raw = calculation(cohort, signal, perm_seed, protocol["permutations"])
        except Exception as exc:
            context = exception_context(exc)
            for method in (["legacy_linear"] if suffix == "linear" else []) + methods:
                results.append({"method": method, "allowed": False, "failure": True, "n_true": int(signal.sum()),
                                "null_rejected": False, "true_rejected": 0, **context})
            continue
        if suffix == "linear":
            results.append({"method": "legacy_linear", "allowed": True, **raw})
        try:
            with original_separation():
                v1 = diagnose_markers(cohort, cohort.markers)
            v2 = diagnose_markers(cohort, cohort.markers)
            diagnostics = [v1, v2, f_sensitivity(v2)]
        except Exception as exc:
            context = exception_context(exc)
            diagnostics = [{"allowed": False, "reasons": ["diagnostic_exception"], **context}] * 3
        for method, diagnostic in zip(methods, diagnostics):
            allowed = bool(diagnostic["allowed"])
            results.append({"method": method, "allowed": allowed, "failure": False,
                            "diagnostic_exception": "error" in diagnostic, "reasons": diagnostic["reasons"],
                            "raw_null_rejected": raw["null_rejected"], "raw_true_rejected": raw["true_rejected"],
                            "null_rejected": allowed and raw["null_rejected"],
                            "true_rejected": raw["true_rejected"] if allowed else 0, "n_true": raw["n_true"],
                            **({"exception_context": diagnostic} if "error" in diagnostic else {})})
    return results


def run(worker, workers, output):
    protocol = json.loads(PROTOCOL.read_text())
    if METHOD_VERSION != "marker-inference/2":
        raise ValueError("This development protocol requires the unqualified corrected v2 method")
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("Invalid ownership")
    configuration = {"protocol": protocol, "workers": workers, "worker": worker,
                     "source_hashes": source_hashes(), "python": platform.python_version(),
                     "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS replicates (condition TEXT, idx INTEGER, host TEXT, elapsed REAL, result TEXT, PRIMARY KEY(condition,idx))")
        serialized = json.dumps(configuration, sort_keys=True)
        previous = db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()
        if previous and previous[0] != serialized:
            raise ValueError("Source, protocol, environment or ownership changed; ledger cannot be reused")
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('configuration', ?)", (serialized,))
        db.commit()
        for condition in protocol["conditions"]:
            for index in range(protocol["datasets_per_condition"]):
                if index % workers != worker or db.execute("SELECT 1 FROM replicates WHERE condition=? AND idx=?", (condition, index)).fetchone():
                    continue
                started = time.time()
                result = paired(condition, index, protocol)
                db.execute("INSERT INTO replicates VALUES (?,?,?,?,?)",
                           (condition, index, socket.gethostname(), time.time() - started, json.dumps(result)))
                db.commit()
            print(json.dumps({"worker": worker, "completed_condition": condition}), flush=True)


def summarize(paths, output):
    protocol = json.loads(PROTOCOL.read_text())
    records, sources, configurations = {}, [], []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            config = json.loads(db.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()[0])
            if config["protocol"] != protocol:
                raise ValueError("Protocol mismatch")
            for condition, index, result in db.execute("SELECT condition,idx,result FROM replicates"):
                if (condition, index) in records or condition not in protocol["conditions"] or not 0 <= index < protocol["datasets_per_condition"]:
                    raise ValueError("Duplicate or out-of-protocol dataset")
                if index % config["workers"] != config["worker"]:
                    raise ValueError("Wrong worker ownership")
                rows = json.loads(result)
                if sorted(r["method"] for r in rows) != sorted(protocol["methods"]):
                    raise ValueError("Missing or duplicate method")
                records[(condition, index)] = rows
        configurations.append(config)
        sources.append({"ledger": path.name, "sha256": digest})
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Ledger changed during read")
    if not configurations or any(c["source_hashes"] != configurations[0]["source_hashes"] or c["versions"] != configurations[0]["versions"] or c["python"] != configurations[0]["python"] for c in configurations):
        raise ValueError("Inconsistent source or environment")
    rows = []
    planned = protocol["datasets_per_condition"]
    for condition in protocol["conditions"]:
        for method in protocol["methods"]:
            data = [next(r for r in v if r["method"] == method) for (c, _), v in records.items() if c == condition]
            allowed = [r for r in data if r["allowed"] and not r["failure"]]
            hits = sum(r["null_rejected"] for r in data)
            failed = sum(r["failure"] for r in data)
            missing = planned - len(data)
            power = [r["true_rejected"] / r["n_true"] for r in data if r["n_true"]]
            power_mean = sum(power) / planned if power else None
            power_half = 1.96 * float(np.std(power, ddof=1)) / np.sqrt(planned) if len(power) > 1 else None
            raw_hits = sum(r.get("raw_null_rejected", r["null_rejected"]) for r in data)
            reasons = Counter(reason.split(": ")[0] + ("/" + reason.rsplit("/", 1)[1] if "/" in reason else "")
                              for r in data for reason in set(r.get("reasons", [])))
            rows.append({"condition": condition, "method": method, "planned": planned, "completed": len(data),
                         "failures": failed, "missing": missing, "allowed": len(allowed), "allowed_fraction": len(allowed) / planned,
                         "fwer_raw": raw_hits / planned, "fwer": hits / planned,
                         "fwer_raw_mc95": cp_interval(raw_hits, planned),
                         "fwer_raw_failure_bounds": [raw_hits / planned, (raw_hits + failed + missing) / planned],
                         "fwer_mc95": cp_interval(hits, planned),
                         "fwer_failure_bounds": [hits / planned, (hits + failed + missing) / planned],
                         "conditional_fwer": hits / len(allowed) if allowed else None,
                         "conditional_mc95": cp_interval(hits, len(allowed)),
                         "power": power_mean,
                         "power_mc95": [max(0., power_mean - power_half), min(1., power_mean + power_half)] if power_half is not None else None,
                         "power_failure_bounds": [sum(power) / planned, min(1., (sum(power) + failed + missing) / planned)] if power else None,
                         "reason_counts_overlap": dict(reasons)})
    result = {"status": "development_only_not_qualification", "protocol": protocol,
              "datasets": len(records), "complete": len(records) == len(protocol["conditions"]) * planned,
              "source_hashes": configurations[0]["source_hashes"], "environment": configurations[0]["versions"],
              "ledgers": sources, "summaries": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Do not overwrite an existing development summary")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"datasets": len(records), "complete": result["complete"], "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--worker", required=True, type=int)
    worker.add_argument("--workers", required=True, type=int)
    worker.add_argument("--output", required=True, type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("ledgers", nargs="+", type=Path)
    summary.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.worker, args.workers, args.output)
    else:
        summarize(args.ledgers, args.output)

"""Repeat the one failed synthetic input on the immutable original numerical kernel.

Repeated successes cannot resolve the original failure or estimate a population
failure rate. No original ledger or source file is modified by this investigation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def run(historical_root, freeze, output, attempts):
    if output.exists() or attempts < 1:
        raise ValueError("Use a fresh output and positive fixed attempt count")
    sys.path.insert(0, str(historical_root / "src"))
    sys.path.insert(0, str(historical_root / "validation/guarded_inference"))
    import numpy as np
    from study import verify_freeze, dataset, calculation, hashes
    from survival_toolkit.marker_evaluation import prepare_marker_cohort
    import survival_toolkit.marker_screen as screen
    freeze_hash = verify_freeze(freeze)
    if not Path(screen.__file__).is_relative_to(historical_root):
        raise ValueError("Historical numerical kernel was not imported")
    frame, names, signal, seed = dataset("partial_strong", 180, 300, 2026100303, 570)
    cohort = prepare_marker_cohort(frame, time_column="time", event_column="event",
                                  marker_columns=names, clinical_columns=["Z"],
                                  clinical_basis="restricted_cubic_spline")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for attempt in range(attempts):
        started = time.time()
        try:
            result = calculation(cohort, signal, seed, 999)
            rows.append({"attempt": attempt, "elapsed": time.time()-started, **result})
        except Exception as exc:
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                frame_locals = tb.tb_frame.f_locals
                values = {}
                for name in ("block", "sorted_block", "row", "index", "running"):
                    if name not in frame_locals:
                        continue
                    value = frame_locals[name]
                    info = {"type": type(value).__name__}
                    if isinstance(value, np.ndarray):
                        info.update(dtype=str(value.dtype), shape=list(value.shape))
                    elif isinstance(value, (int, np.integer)):
                        info["value"] = int(value)
                    values[name] = info
                frames.append({"function": tb.tb_frame.f_code.co_name, "line": tb.tb_lineno, "locals": values})
                tb = tb.tb_next
            rows.append({"attempt": attempt, "elapsed": time.time()-started, "failure": True,
                         "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "frames": frames})
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(rows[-1]) + "\n")
    result = {"role": "same-input diagnostic repeat; never replacement confirmation observations",
              "original_failure_retained": True, "root_cause_resolved": False,
              "attempts": attempts, "failures_reproduced": sum(r["failure"] for r in rows),
              "freeze_hash": freeze_hash, "source_hashes": hashes(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input": {"condition": "partial_strong", "n": 180, "p": 300, "index": 570, "seed": 2026100303, "permutations": 999},
              "attempts_sha256": hashlib.sha256((output / "attempts.jsonl").read_bytes()).hexdigest()}
    (output / "investigation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts", default=100, type=int)
    args = parser.parse_args()
    run(args.historical_root, args.freeze, args.output, args.attempts)

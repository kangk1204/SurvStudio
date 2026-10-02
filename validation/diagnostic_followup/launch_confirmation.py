"""Durable v2 processes with fixed modulo ownership and append-only ledgers."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from confirmation import PROTOCOL, verify_freeze


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def launch(args):
    freeze_hash = verify_freeze(args.freeze)
    if args.offset < 0 or args.slots < 1 or args.offset + args.slots > args.total_workers:
        raise ValueError("Invalid ownership allocation")
    protocol = json.loads(PROTOCOL.read_text())
    cells = [dict(stage="main", condition=c, **protocol["main"]) for c in protocol["conditions"]]
    cells += [dict(stage="extension", condition=c, **e) for e in protocol["extensions"] for c in protocol["conditions"]]
    cells += [dict(stage="large", condition=c, **{k: v for k, v in protocol["large_marker"].items() if k != "conditions"})
              for c in protocol["large_marker"]["conditions"]]
    inventory = {"host": socket.gethostname(), "total_workers": args.total_workers,
                 "workers": list(range(args.offset, args.offset + args.slots)),
                 "freeze_hash": freeze_hash, "cells": cells}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ownership = out / "ownership.json"
    if ownership.exists() and json.loads(ownership.read_text()) != inventory:
        raise ValueError("Ownership cannot be silently reassigned")
    if not ownership.exists():
        write_status(ownership, inventory)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    def worker(worker_id):
        status = out / f"worker-{worker_id}-status.json"
        for cell_index, cell in enumerate(cells):
            name = f"{cell['stage']}-{cell['condition']}-n{cell['n']}-p{cell['p']}-worker{worker_id}"
            command = [sys.executable, str(Path(__file__).with_name("confirmation.py")), "run",
                       "--stage", cell["stage"], "--condition", cell["condition"], "--n", str(cell["n"]),
                       "--p", str(cell["p"]), "--stop", str(cell["replicates_per_condition"]),
                       "--workers", str(args.total_workers), "--worker", str(worker_id),
                       "--output", str(out / (name + ".sqlite")), "--freeze", args.freeze]
            state = {"worker": worker_id, "cell": cell, "cell_index": cell_index,
                     "command": command, "status": "running", "updated": time.time()}
            write_status(status, state)
            with (out / (name + ".log")).open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            state.update(status="completed" if result.returncode == 0 else "process_failed",
                         returncode=result.returncode, updated=time.time())
            write_status(status, state)
            if result.returncode:
                raise RuntimeError(f"Worker {worker_id} process failed; ledger preserved: {name}")
        state.update(status="all_assigned_cells_completed", updated=time.time())
        write_status(status, state)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.slots) as pool:
        list(pool.map(worker, inventory["workers"]))
    write_status(out / "server-complete.json", {**inventory, "completed": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--slots", type=int, required=True)
    launch(parser.parse_args())

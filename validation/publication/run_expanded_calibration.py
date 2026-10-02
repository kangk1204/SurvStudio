"""Execute the frozen conditional-null stress test, with resumable per-replicate evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from survival_toolkit.marker_evaluation import MarkerSettings, permutation_null, prepare_marker_cohort, run_procedure


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / total
    centre = (rate + z*z/(2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [centre-radius, centre+radius]


def replicate(task: tuple[dict, int, int]) -> list[dict]:
    config, scenario_index, index = task
    scenario = config["scenarios"][scenario_index]
    rng = np.random.default_rng([config["seed"], scenario_index, index])
    n, p = config["patients"], config["markers"]
    z = rng.normal(size=n)
    noise = rng.normal(size=(n, p))
    if scenario == "correlated_null":
        noise = math.sqrt(.7)*rng.normal(size=(n, 1)) + math.sqrt(.3)*noise
    x = noise if scenario == "independent_null" else .8*z[:, None] + noise
    if scenario == "heteroscedastic_null":
        x = .8*z[:, None] + np.exp(.8*z[:, None])*noise
    if scenario == "nonlinear_null":
        x = z[:, None]**2 + noise
    truth = np.zeros(p, dtype=bool)
    beta = .2 if scenario == "partial_null_weak" else .4 if scenario == "partial_null_strong" else 0.
    if beta:
        truth[:5] = True
    eta = .8*z + beta*noise[:, truth].sum(axis=1)
    failure = rng.exponential(size=n) / (.06*np.exp(eta))
    censor_eta = .8*z if scenario == "covariate_dependent_censoring" else np.zeros(n)
    censor = rng.exponential(size=n) / (.04*np.exp(censor_eta))
    markers = [f"marker_{j}" for j in range(p)]
    frame = pd.DataFrame(x, columns=markers)
    frame["z"], frame["z_squared"] = z, z*z
    frame["time"], frame["event"] = np.minimum(failure, censor), (failure <= censor).astype(int)
    rows = []
    for method in config["methods"]:
        record = {"scenario": scenario, "replicate": index, "method": method, "events": int(frame.event.sum()), "error": None}
        try:
            clinical = ["z", "z_squared"] if method == "smith_quadratic" else ["z"]
            cohort = prepare_marker_cohort(frame, time_column="time", event_column="event", marker_columns=markers,
                                           clinical_columns=clinical, event_positive_value=1)
            if cohort.marker_names != tuple(markers) and list(cohort.marker_names) != markers:
                raise RuntimeError("The prespecified marker family changed.")
            settings = MarkerSettings(n_permutations=config["permutations"], n_resamples=0,
                                      lens2_null="raw" if method == "raw_linear" else "smith")
            full = run_procedure(cohort, np.arange(n), settings)
            # Identical permutation stream across methods; generator draws cannot depend on the method.
            perm_rng = np.random.default_rng([config["seed"], scenario_index, index, 991])
            adjusted = permutation_null(cohort, full, settings, perm_rng)["added_value"]
            for label, values in (("step_down", adjusted["p_fwer"]), ("single_step", adjusted["p_fwer_single_step"]),
                                  ("permutation_q", adjusted["q_perm"]), ("bh", full.lenses["added_value"].q_bh)):
                values = np.asarray(values, dtype=float)
                if values.shape != (p,) or not np.all(np.isfinite(values)):
                    raise RuntimeError(f"Non-estimable or malformed {label} values.")
                rejected = values <= config["alpha"]
                false = int(np.sum(rejected & ~truth))
                true = int(np.sum(rejected & truth))
                record[f"{label}_any_false"] = bool(false)
                record[f"{label}_fdp"] = false/max(1, false+true)
                record[f"{label}_power"] = true/int(truth.sum()) if truth.any() else None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("expanded_protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--engineering-pilot", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    if args.engineering_pilot:
        config.update(seed=config["engineering_seed"], replicates=3, permutations=99)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), args.protocol, *sorted((root/"src/survival_toolkit").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    stamp = hashlib.sha256(json.dumps({"config": config, "sources": hashes}, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output/"replicates.jsonl"
    settings = {"config": config, "stamp": stamp, "source_sha256": hashes,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    settings_path = args.output/"settings.json"
    existing = []
    if settings_path.exists():
        if json.loads(settings_path.read_text())["stamp"] != stamp:
            raise RuntimeError("Existing evidence belongs to different source/configuration; use a new output directory.")
        if journal.exists():
            existing = [json.loads(line) for line in journal.read_text().splitlines()]
    else:
        settings_path.write_text(json.dumps(settings, indent=2)+"\n")
    completed = {(row["scenario"], row["replicate"]) for row in existing}
    expected_methods = set(config["methods"])
    for key in completed:
        matching = [r for r in existing if (r["scenario"], r["replicate"]) == key]
        if len(matching) != len(expected_methods) or {r["method"] for r in matching} != expected_methods:
            raise RuntimeError(f"Incomplete or duplicate journal entry {key}; preserve and repair the journal before resuming.")
    tasks = [(config, s, i) for s, name in enumerate(config["scenarios"]) for i in range(config["replicates"]) if (name, i) not in completed]
    rows = existing[:]
    with journal.open("a") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replicate, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            batch = future.result()
            stream.write("".join(json.dumps(row, allow_nan=False)+"\n" for row in batch)); stream.flush()
            rows.extend(batch)
            if number % 100 == 0:
                print(f"Completed {len(rows)//len(config['methods'])}/{len(config['scenarios'])*config['replicates']} datasets", flush=True)
    summary = []
    for scenario in config["scenarios"]:
        for method in config["methods"]:
            group = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            valid = [r for r in group if r["error"] is None]
            failed = len(group)-len(valid)
            entry = {"scenario": scenario, "method": method, "planned": config["replicates"], "completed": len(group), "invalid": failed}
            for metric in ("step_down", "single_step", "permutation_q", "bh"):
                count = sum(r[f"{metric}_any_false"] for r in valid)
                entry[metric] = {"any_false_count": count, "fwer_valid": count/len(valid) if valid else None,
                                 "fwer_mc95": wilson(count, len(valid)) if valid else None,
                                 "fwer_failure_bounds": [count/len(group), (count+failed)/len(group)],
                                 "mean_fdp": float(np.mean([r[f"{metric}_fdp"] for r in valid])) if valid else None,
                                 "mean_power": float(np.mean([r[f"{metric}_power"] for r in valid])) if valid and valid[0][f"{metric}_power"] is not None else None}
            summary.append(entry)
    pd.DataFrame(rows).sort_values(["scenario", "replicate", "method"]).to_csv(args.output/"replicates.csv", index=False)
    (args.output/"summary.json").write_text(json.dumps({"settings": settings, "summary": summary}, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    if any(entry["invalid"] or entry["completed"] != entry["planned"] for entry in summary):
        raise RuntimeError("The planned simulation has failures or is incomplete; see recorded evidence.")


if __name__ == "__main__":
    main()

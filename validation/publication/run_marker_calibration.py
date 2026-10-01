"""Seeded global-null calibration audit of the marker permutation screen.

All outcomes depend on the clinical covariate only. Marker relations and censoring
vary across scenarios to stress the Smith residual-permutation assumptions.
This pilot estimates family-wise rejection rates with Monte Carlo intervals;
it does not establish universal error control or replace external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from survival_toolkit.marker_evaluation import MarkerSettings, permutation_null, prepare_marker_cohort, run_procedure


SCENARIOS = ("independent_markers", "linear_marker_relation", "heteroscedastic_markers", "nonlinear_marker_relation")


def one_replicate(task: tuple[int, int, int, int, int, int]) -> dict:
    scenario, replicate, seed, n, p, permutations = task
    rng = np.random.default_rng([seed, scenario, replicate])
    z = rng.normal(size=n)
    noise = rng.normal(size=(n, p))
    if scenario == 0:
        x = noise
    elif scenario == 1:
        x = 0.8 * z[:, None] + noise
    elif scenario == 2:
        x = 0.8 * z[:, None] + np.exp(0.8 * z[:, None]) * noise
    else:
        x = z[:, None] ** 2 + noise
    failure = rng.exponential(size=n) / (0.06 * np.exp(0.8 * z))
    censoring = rng.exponential(size=n) / 0.04
    frame = pd.DataFrame(x, columns=[f"marker_{j}" for j in range(p)])
    frame["clinical_z"] = z
    frame["followup"] = np.minimum(failure, censoring)
    frame["event"] = (failure <= censoring).astype(int)
    cohort = prepare_marker_cohort(frame, time_column="followup", event_column="event", marker_columns=list(frame.columns[:p]),
                                   clinical_columns=["clinical_z"], event_positive_value=1)
    settings = MarkerSettings(n_permutations=permutations, n_resamples=0)
    full = run_procedure(cohort, np.arange(n), settings)
    adjusted = permutation_null(cohort, full, settings, rng)
    primary = adjusted["added_value"]
    return {"scenario": SCENARIOS[scenario], "replicate": replicate, "events": int(frame.event.sum()),
            "fwer_rejected": bool(np.any(primary["p_fwer"] <= settings.alpha)),
            "single_step_rejected": bool(np.any(primary["p_fwer_single_step"] <= settings.alpha)),
            "bh_rejected": bool(np.any(full.lenses["added_value"].q_bh <= settings.alpha)),
            "minimum_p_fwer": float(np.min(primary["p_fwer"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--patients", type=int, default=180)
    parser.add_argument("--markers", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if min(args.replicates, args.permutations, args.markers, args.workers) < 1 or args.patients < 30 or args.seed < 0:
        parser.error("Use positive counts, at least 30 patients and a nonnegative seed.")
    tasks = [(s, r, args.seed, args.patients, args.markers, args.permutations) for s in range(len(SCENARIOS)) for r in range(args.replicates)]
    if args.workers == 1:
        rows = [one_replicate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one_replicate, tasks))
    summaries = []
    for scenario in SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario]
        summary = {"scenario": scenario, "n_replicates": len(subset)}
        for metric in ("fwer_rejected", "single_step_rejected", "bh_rejected"):
            count = sum(row[metric] for row in subset)
            summary[metric] = {"count": count, "rate": count / len(subset),
                               "monte_carlo_wilson_95": list(proportion_confint(count, len(subset), method="wilson"))}
        summaries.append(summary)
    root = Path(__file__).resolve().parents[2]
    source = [Path(__file__).resolve(), *sorted((root / "src" / "survival_toolkit").glob("*.py"))]
    manifest = {"settings": {**vars(args), "output": str(args.output)},
                "software": {name: importlib.metadata.version(name) for name in ("survstudio", "numpy", "pandas", "scipy", "statsmodels")},
                "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source},
                "interpretation": "Monte Carlo pilot under four specified conditional nulls. The intervals quantify simulation uncertainty; general family-wise control is not established.",
                "summary": summaries, "replicates": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "marker_calibration.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(rows).to_csv(args.output / "marker_calibration_replicates.csv", index=False)
    for summary in summaries:
        print(summary["scenario"], json.dumps(summary["fwer_rejected"]), flush=True)


if __name__ == "__main__":
    main()

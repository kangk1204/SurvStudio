"""Monte Carlo variability of the development evaluations: case studies I (TCGA-LUAD, overall survival), IV (METABRIC,
overall survival) and V (METABRIC ER-positive, recurrence) rerun exactly as scripts 01, 07 and 09 run them
(common.development_data and common.evaluate_development: SurvStudio's defaults, 1,000 permutations and 200
subsamples), once with each of 20 random seeds, the default seed 20260926 and the 19 fixed others of SEEDS.

The seed sets the permutations (the family-wise p-values, and through them the tiers) and the subsamples (selection
frequency and direction consistency, and through them the tiers; the optimism-corrected C, the left-out C of the model
and of the clinical-only model, and the paired left-out gain). It does not set the locked model: SurvStudio locks the
markers with Benjamini-Hochberg q <= 0.05 on the analytic score test in all patients, so the locked genes and the
apparent C should be the same for every seed; they are recorded to show it.

Per case study and seed (seed_variability.csv): the apparent C, the optimism-corrected C, the left-out C of the model
and of the clinical-only model, the paired left-out gain (signature_gain_left_out) with its interval when SurvStudio
reports one (signature_gain_left_out_ci, read when present), the numbers of markers with family-wise p <= 0.05 on the
added-value lens, robust and suggestive, and the locked genes. Per case study (seed_variability.json): the mean, SD,
minimum and maximum of each number over the seeds, how many seeds locked each gene, and whether the default seed
reproduces the development run (the summaries of scripts 01, 07 and 09, when they are there).

One process per case study and seed, with the cores shared out among the workers (METABRIC workers need a few GB of
memory each); a worker that dies stops the run with an error.
Usage: python 16_seed_variability.py [workers] [--seeds N] [--cases I IV V]   (the first N seeds; all 20 by default)
"""

from __future__ import annotations

import argparse
import os

SEEDS = [20260926, 11, 23, 37, 41, 53, 67, 79, 83, 97, 101, 113, 127, 131, 149, 151, 163, 173, 181, 191]
CASES = ["I", "IV", "V"]
parser = argparse.ArgumentParser(description="Rerun the development evaluations of case studies I, IV and V with 20 seeds.")
parser.add_argument("workers", nargs="?", type=int, default=4, help="processes (default 4)")
parser.add_argument("--seeds", type=int, default=len(SEEDS), help=f"run the first N seeds of the list (default all {len(SEEDS)})")
parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES, help="case studies (default I IV V)")
ARGUMENTS = parser.parse_args()
THREADS = str(max(1, (os.cpu_count() or 1) // max(1, ARGUMENTS.workers)))
os.environ.setdefault("OMP_NUM_THREADS", THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", THREADS)

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import pandas as pd  # noqa: E402

from common import DEVELOPMENT, RESULTS, development_data, evaluate_development, read_result, survstudio_version, write_csv_atomic, write_json  # noqa: E402
from survival_toolkit.marker_evaluation import MarkerSettings  # noqa: E402

NUMBERS = ["apparent_c", "corrected_c", "left_out_c", "clinical_left_out_c", "gain", "gain_lower", "gain_upper", "fwer", "robust", "suggestive"]
SUMMARIES = {"I": "tcga_markers_summary.json", "IV": "breast_markers_summary.json", "V": "breast_er_markers_summary.json"}
# Built once in the parent and inherited by the forked workers.
DATA = {case: development_data(case) for case in ARGUMENTS.cases}


def interval_bounds(value) -> tuple[float | None, float | None]:
    """The two bounds of an interval SurvStudio reports as [lower, upper] or {"lower": ..., "upper": ...}; none when absent."""
    if isinstance(value, dict):
        value = [value.get("lower"), value.get("upper")]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(None if bound is None else float(bound) for bound in value)
    return None, None


def run(task: tuple[str, int]) -> dict:
    case, seed = task
    frame, genes = DATA[case]
    began = time.time()
    result = evaluate_development(case, frame, genes, MarkerSettings(random_seed=seed))
    signature = result["signature"]
    lower, upper = interval_bounds(signature.get("signature_gain_left_out_ci"))
    fwer = sum(1 for row in result["marker_table"] if row["added_value"]["p_fwer"] is not None and row["added_value"]["p_fwer"] <= 0.05)
    return {
        "case": case, "seed": seed, "apparent_c": signature["apparent_c"], "corrected_c": signature["optimism_corrected_c"],
        "left_out_c": signature["signature_c_left_out"], "clinical_left_out_c": signature["clinical_c_left_out"],
        "gain": signature["signature_gain_left_out"], "gain_lower": lower, "gain_upper": upper, "fwer": fwer,
        "robust": int(result["tier_counts"]["robust"]), "suggestive": int(result["tier_counts"]["suggestive"]),
        "locked_genes": ";".join(signature["markers"]), "n": result["cohort"]["n"], "events": result["cohort"]["events"],
        "seconds": round(time.time() - began),
    }


def development_match(case: str, row: pd.Series) -> dict:
    """The default seed's numbers against the development run's summary (scripts 01, 07 and 09): each difference, and
    whether all agree: the C-indices and the gain to 1e-6 (the agreement across machines and thread counts; the
    README's), the counts and the locked genes exactly."""
    try:
        summary = read_result(SUMMARIES[case])
    except FileNotFoundError:
        return {"compared": False, "note": f"{SUMMARIES[case]} is not in results/"}
    signature = summary["signature"]
    development = {"apparent_c": signature["apparent_c"], "corrected_c": signature["optimism_corrected_c"],
                   "left_out_c": signature["signature_c_left_out"], "clinical_left_out_c": signature["clinical_c_left_out"],
                   "gain": signature["signature_gain_left_out"], "fwer": summary["funnel"]["fwer_at_most_0_05"],
                   "robust": summary["tier_counts"]["robust"], "suggestive": summary["tier_counts"]["suggestive"]}
    differences = {key: float(row[key]) - float(value) for key, value in development.items() if value is not None and pd.notna(row[key])}
    same_genes = row["locked_genes"] == ";".join(signature["markers"])
    counts = ("fwer", "robust", "suggestive")
    agree = all(abs(value) <= (0 if key in counts else 1e-6) for key, value in differences.items())
    return {"compared": True, "matches": bool(same_genes and agree), "differences": differences, "same_locked_genes": bool(same_genes)}


def summarise(rows: pd.DataFrame) -> dict:
    summary = {}
    for case in ARGUMENTS.cases:
        part = rows[rows["case"] == case]
        numbers = {}
        for name in NUMBERS:
            values = pd.to_numeric(part[name], errors="coerce").dropna()
            if len(values):
                numbers[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                                 "min": float(values.min()), "max": float(values.max()), "seeds": int(len(values))}
        genes = pd.Series([gene for listed in part["locked_genes"] for gene in listed.split(";") if gene]).value_counts()
        default = part[part["seed"] == SEEDS[0]]
        summary[case] = {
            "label": DEVELOPMENT[case]["label"], "seeds": int(len(part)), "patients": int(part["n"].iat[0]), "events": int(part["events"].iat[0]),
            "numbers": numbers, "gain_interval_reported": bool(part["gain_lower"].notna().any()),
            "locked_gene_seeds": {str(gene): int(count) for gene, count in genes.items()},
            "distinct_locked_models": int(part["locked_genes"].nunique()),
            "default_seed": development_match(case, default.iloc[0]) if len(default) else {"compared": False, "note": "default seed not run"},
        }
    return summary


def main() -> None:
    seeds = SEEDS[: max(1, min(ARGUMENTS.seeds, len(SEEDS)))]
    # The slowest case studies first, so the workers stay busy to the end.
    order = {"IV": 0, "V": 1, "I": 2}
    tasks = sorted(((case, seed) for case in ARGUMENTS.cases for seed in seeds), key=lambda task: order[task[0]])
    print(f"{len(tasks)} evaluations ({', '.join(ARGUMENTS.cases)}; {len(seeds)} seeds) with {ARGUMENTS.workers} workers, "
          f"{THREADS} threads each", flush=True)
    rows = []
    began = time.time()
    # A worker that dies breaks the pool: the next result raises BrokenProcessPool instead of waiting forever.
    with ProcessPoolExecutor(max_workers=ARGUMENTS.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            gain = "none" if row["gain"] is None else f"{row['gain']:+.4f}"
            print(f"{count}/{len(tasks)} case {row['case']} seed {row['seed']}: gain {gain}, robust {row['robust']}, "
                  f"{time.time() - began:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table["case"] = pd.Categorical(table["case"], categories=CASES, ordered=True)
    table["position"] = table["seed"].map({seed: position for position, seed in enumerate(SEEDS)})
    table = table.sort_values(["case", "position"]).drop(columns="position").reset_index(drop=True)
    table["case"] = table["case"].astype(str)
    write_csv_atomic(table, RESULTS / "seed_variability.csv")
    settings = {key: value for key, value in MarkerSettings()._asdict().items() if key != "random_seed"}
    summary = {"survstudio": survstudio_version(), "seeds": seeds, "cases": ARGUMENTS.cases, "settings_besides_the_seed": settings,
               "summary": summarise(table)}
    write_json(RESULTS / "seed_variability.json", summary)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(table.drop(columns=["locked_genes"]).round(4).to_string(index=False))
    for case, value in summary["summary"].items():
        print(case, {name: {key: round(number, 4) if isinstance(number, float) else number for key, number in stats.items()}
                     for name, stats in value["numbers"].items()}, value["default_seed"], flush=True)


if __name__ == "__main__":
    main()

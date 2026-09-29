"""Supplementary simulation: error control, power and optimism of SurvStudio's marker evaluation on real RNA-seq.

Plasmode design: the TCGA-LUAD expression and clinical data are kept as they are (484 patients) and only the
outcome is simulated, so marker distributions, zero inflation and correlations with the clinical covariates are
real. Each replicate draws 2,000 of the 20,191 non-constant genes. Survival follows a Weibull proportional-hazards
model (shape 1.2) with the clinical effects fitted to TCGA-LUAD overall survival (age, sex, stage), and, under the
alternative, five genes with a log hazard ratio of +/-beta per SD beyond them. Exponential censoring gives about
the observed 37% deaths.

Scenarios: the global null with and without the near-constant filter (family-wise error), and the alternative
at beta 0.3 and 0.45 with and without the filter (power, false discoveries, and the optimism-corrected C against
the locked model's C in 3,000 new patients from the same design).

Resumable: each replicate is written as it finishes (the whole CSV rewritten through a temporary file), stamped with
the SurvStudio commit, a hash of the design and a hash of the simulation's code (this script and common.py). A rerun
keeps only the replicates with the same stamps that finished without error, and none when SurvStudio has uncommitted
changes; the rest run again. A worker that dies (killed for memory, say) stops the run with an error; the replicates
finished so far are kept for the next run.
Usage: python 06_simulation.py [workers]
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    CATEGORICAL,
    COVARIATES,
    RESULTS,
    SCRIPTS,
    STAMP_DTYPES,
    XENA_EXPRESSION,
    commit_is_clean,
    resumable_rows,
    survstudio_version,
    write_csv_atomic,
    write_json,
)
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers, validate_locked_recipe  # noqa: E402
from survival_toolkit.marker_matrix import matrix_frame, read_marker_matrix  # noqa: E402
from survival_toolkit.marker_screen import fit_cox  # noqa: E402
from survival_toolkit.sample_data import load_tcga_luad_upload_ready_dataset  # noqa: E402

GENES_PER_REPLICATE = 2000
TRUE_MARKERS = 5
SHAPE = 1.2
MEDIAN_MONTHS = 40.0
ADMIN_MONTHS = 150.0
TARGET_EVENT_RATE = 0.37
# Replicate r of scenario s draws from default_rng([BASE_SEED, s, r]), s being the scenario's position in SCENARIOS.
BASE_SEED = 20260928
PERMUTATIONS = 1000
NEW_PATIENTS = 3000
SCENARIOS = {
    # name: (beta per SD for the true markers, max_mode_fraction, subsamples, replicates)
    "null_filter": (0.0, 0.9, 0, 400),
    "null_no_filter": (0.0, 1.0, 0, 200),
    "alt_0.30_filter": (0.30, 0.9, 100, 100),
    "alt_0.30_no_filter": (0.30, 1.0, 100, 100),
    "alt_0.45_filter": (0.45, 0.9, 100, 100),
    "alt_0.45_no_filter": (0.45, 1.0, 100, 100),
}
# A gene outside the five true markers still carries their signal when it is co-expressed with them: its own
# hypothesis ("no association beyond the clinical covariates") is then false. Genes whose partial correlation
# with the true markers' linear predictor, given the clinical covariates, reaches LINKED_R are counted as linked;
# only the others are false discoveries. Under the null every gene is unlinked.
LINKED_R = 0.1
# Every record is written with the same columns; the C-index ones stay empty under the null. left_out_gain is
# SurvStudio's paired left-out gain (signature_gain_left_out: the model's C minus the clinical model's C on the same
# left-out patients, averaged over the subsamples where both exist).
COLUMNS = [
    "scenario", "replicate", "beta", "max_mode_fraction", "subsamples", "events", "tested", "near_constant_in_draw",
    "fwer_true", "fwer_false", "robust_true", "robust_false", "marginal_hits", "seconds",
    "apparent_c", "corrected_c", "left_out_c", "clinical_left_out_c", "left_out_gain", "new_patients_c", "new_patients_clinical_c",
    "error", "linked_genes", "fwer_linked", "fwer_unlinked", "robust_linked", "robust_unlinked",
    "survstudio_commit", "design_hash", "code_hash",
]

# ── Shared, read-only data (built once, inherited by forked workers) ──────────
clinical_all = load_tcga_luad_upload_ready_dataset()
matrix = read_marker_matrix(XENA_EXPRESSION, XENA_EXPRESSION.name, patient_ids=clinical_all["patient_id"].tolist())
frame = matrix_frame(clinical_all, matrix, id_column="patient_id", columns=["patient_id", "os_months", "os_event", *COVARIATES])
frame = frame.dropna(subset=["os_months", "os_event", *COVARIATES]).reset_index(drop=True)
frame = frame[frame["os_months"] > 0].reset_index(drop=True)
GENES = np.array([name for name in matrix.marker_names if np.unique(frame[name]).size > 1])
EXPRESSION = frame[GENES].to_numpy(dtype=np.float64)
MODE_SHARE = np.array([np.unique(column, return_counts=True)[1].max() / column.size for column in EXPRESSION.T])
STANDARDISED = (EXPRESSION - EXPRESSION.mean(axis=0)) / np.where(EXPRESSION.std(axis=0) > 0, EXPRESSION.std(axis=0), 1.0)
CLINICAL = frame[COVARIATES].reset_index(drop=True)
DESIGN = pd.get_dummies(CLINICAL, columns=CATEGORICAL, drop_first=True, dtype=float)
CLINICAL_FIT = fit_cox(frame["os_months"].to_numpy(), frame["os_event"].to_numpy(), DESIGN.to_numpy())
CLINICAL_BASIS = np.column_stack([np.ones(DESIGN.shape[0]), DESIGN.to_numpy()])
ETA_CLINICAL = DESIGN.to_numpy() @ CLINICAL_FIT.beta
ETA_CLINICAL = ETA_CLINICAL - ETA_CLINICAL.mean()
N = frame.shape[0]
SCALE = MEDIAN_MONTHS / np.log(2.0) ** (1.0 / SHAPE)


def event_times(eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return SCALE * (-np.log(rng.random(eta.size)) / np.exp(eta)) ** (1.0 / SHAPE)


def censoring_rate() -> float:
    """Exponential censoring rate that gives the target death fraction under the clinical-only model."""
    rng = np.random.default_rng(1)
    eta = np.tile(ETA_CLINICAL, 40)
    times = event_times(eta, rng)
    low, high = 1e-4, 1.0
    for _ in range(60):
        rate = (low + high) / 2
        censor = np.minimum(np.random.default_rng(2).exponential(1.0 / rate, eta.size), ADMIN_MONTHS)
        if np.mean(times <= censor) > TARGET_EVENT_RATE:
            low = rate
        else:
            high = rate
    return (low + high) / 2


CENSOR_RATE = censoring_rate()
# Everything a replicate's result depends on besides SurvStudio's code: a stored replicate is reused only under the
# same hash. The replicate counts are left out (adding replicates leaves the earlier ones valid); the scenarios'
# order is in (it sets their seeds).
SIMULATION_DESIGN = {
    "genes_per_replicate": GENES_PER_REPLICATE, "true_markers": TRUE_MARKERS, "weibull_shape": SHAPE,
    "median_months": MEDIAN_MONTHS, "administrative_months": ADMIN_MONTHS, "target_event_rate": TARGET_EVENT_RATE,
    "linked_r": LINKED_R, "base_seed": BASE_SEED, "permutations": PERMUTATIONS, "new_patients": NEW_PATIENTS,
    "patients": int(N), "genes": int(GENES.size),
    "scenarios": [[name, beta, max_mode, subsamples] for name, (beta, max_mode, subsamples, _) in SCENARIOS.items()],
}
DESIGN_HASH = hashlib.sha256(json.dumps(SIMULATION_DESIGN, sort_keys=True).encode("utf-8")).hexdigest()[:16]
# The simulation's own code: a stored replicate is reused only while this script and common.py are unchanged, so a
# fix to either runs every replicate again.
CODE_FILES = [Path(__file__).resolve(), SCRIPTS / "common.py"]
CODE_HASH = hashlib.sha256(b"".join(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0" for path in CODE_FILES)).hexdigest()[:16]
SURVSTUDIO = survstudio_version()
STAMP = {"survstudio_commit": SURVSTUDIO["commit"], "design_hash": DESIGN_HASH, "code_hash": CODE_HASH}


def simulate(eta: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    times = event_times(eta, rng)
    censor = np.minimum(rng.exponential(1.0 / CENSOR_RATE, eta.size), ADMIN_MONTHS)
    return np.minimum(times, censor), (times <= censor).astype(int)


def replicate(task: tuple[str, int]) -> dict:
    """One replicate; a failure is a result too (recorded with its error) and does not stop the run."""
    name, index = task
    beta, max_mode, subsamples, _ = SCENARIOS[name]
    try:
        return {**_replicate(name, index), **STAMP}
    except Exception as exc:  # noqa: BLE001
        return {"scenario": name, "replicate": index, "beta": beta, "max_mode_fraction": max_mode, "subsamples": subsamples,
                "error": f"{type(exc).__name__}: {exc}"[:300], **STAMP}


def linked_genes(columns: np.ndarray, truth: list[int], effects: np.ndarray) -> set[str]:
    """Genes of the draw whose partial correlation with the true markers' linear predictor, given the clinical
    covariates, is at least LINKED_R in absolute value (none under the null)."""
    if not truth:
        return set()
    residual = lambda values: values - CLINICAL_BASIS @ np.linalg.lstsq(CLINICAL_BASIS, values, rcond=None)[0]  # noqa: E731
    signal = residual(STANDARDISED[:, truth] @ effects)
    genes = residual(STANDARDISED[:, columns])
    norms = np.linalg.norm(genes, axis=0) * np.linalg.norm(signal)
    correlation = np.divide(signal @ genes, norms, out=np.zeros(columns.size), where=norms > 0)
    return set(GENES[columns][np.abs(correlation) >= LINKED_R].tolist())


def _replicate(name: str, index: int) -> dict:
    beta, max_mode, subsamples, _ = SCENARIOS[name]
    rng = np.random.default_rng([BASE_SEED, list(SCENARIOS).index(name), index])
    columns = np.sort(rng.choice(GENES.size, GENES_PER_REPLICATE, replace=False))
    truth: list[int] = []
    eta = ETA_CLINICAL.copy()
    effects = np.zeros(0)
    if beta > 0:
        candidates = columns[MODE_SHARE[columns] <= 0.5]
        truth = sorted(rng.choice(candidates, TRUE_MARKERS, replace=False).tolist())
        effects = beta * rng.choice([-1.0, 1.0], TRUE_MARKERS)
        eta = eta + STANDARDISED[:, truth] @ effects
    time_, event = simulate(eta, rng)
    data = pd.DataFrame(EXPRESSION[:, columns], columns=GENES[columns])
    data[COVARIATES] = CLINICAL
    data["time"], data["event"] = time_, event
    began = time.time()
    result = evaluate_markers(
        data, time_column="time", event_column="event", marker_columns=list(GENES[columns]),
        clinical_columns=COVARIATES, categorical_clinical=CATEGORICAL, event_positive_value=1,
        settings=MarkerSettings(n_permutations=PERMUTATIONS, n_resamples=subsamples, max_mode_fraction=max_mode, random_seed=int(index)),
    )
    table = result["marker_table"]
    true_names = set(GENES[truth].tolist())
    fwer = {row["marker"] for row in table if (row["added_value"]["p_fwer"] or 1.0) <= 0.05}
    robust = {row["marker"] for row in table if row["tier"] == "robust"}
    marginal = {row["marker"] for row in table if (row["marginal"]["p_fwer"] or 1.0) <= 0.05}
    linked = linked_genes(columns, truth, effects) - true_names
    record = {
        "scenario": name, "replicate": index, "beta": beta, "max_mode_fraction": max_mode, "subsamples": subsamples,
        "events": int(event.sum()), "tested": int(result["cohort"]["n_markers_evaluated"]),
        "near_constant_in_draw": int(np.sum(MODE_SHARE[columns] > 0.9)),
        "fwer_true": len(fwer & true_names), "fwer_false": len(fwer - true_names),
        "robust_true": len(robust & true_names), "robust_false": len(robust - true_names),
        "marginal_hits": len(marginal), "seconds": round(time.time() - began, 1),
        "linked_genes": len(linked),
        "fwer_linked": len((fwer - true_names) & linked), "fwer_unlinked": len(fwer - true_names - linked),
        "robust_linked": len((robust - true_names) & linked), "robust_unlinked": len(robust - true_names - linked),
    }
    signature = result.get("signature") or {}
    recipe = result.get("locked_recipe")
    if beta > 0 and recipe and signature.get("apparent_c") is not None:
        rows = rng.integers(0, N, NEW_PATIENTS)
        new_eta = ETA_CLINICAL[rows] + STANDARDISED[np.ix_(rows, truth)] @ effects
        new_time, new_event = simulate(new_eta, rng)
        external = pd.DataFrame(EXPRESSION[np.ix_(rows, columns)], columns=GENES[columns])
        external[COVARIATES] = CLINICAL.iloc[rows].reset_index(drop=True)
        external["time"], external["event"] = new_time, new_event
        truth_report = validate_locked_recipe(external, recipe, n_bootstrap=0)
        record.update({
            "apparent_c": signature["apparent_c"], "corrected_c": signature.get("optimism_corrected_c"),
            "left_out_c": signature.get("signature_c_left_out"), "clinical_left_out_c": signature.get("clinical_c_left_out"),
            "left_out_gain": signature.get("signature_gain_left_out"),
            "new_patients_c": truth_report["metrics"]["c_index"], "new_patients_clinical_c": truth_report["metrics"].get("clinical_only_c_index"),
        })
    return record


def write_replicates(path, rows: list[dict]) -> None:
    """All replicates so far, in design order, written through a temporary file."""
    order = {name: position for position, name in enumerate(SCENARIOS)}
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    table = table.sort_values(["scenario", "replicate"], key=lambda column: column.map(order) if column.name == "scenario" else column)
    write_csv_atomic(table, path)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tasks = [(name, index) for name, (*_, replicates) in SCENARIOS.items() for index in range(replicates)]
    out = RESULTS / "simulation_replicates.csv"
    settings = RESULTS / "simulation_settings.json"
    if not commit_is_clean(STAMP["survstudio_commit"]):
        print(f"SurvStudio is at {STAMP['survstudio_commit']}: earlier replicates cannot be matched to this code and all run again.", flush=True)
    kept = pd.DataFrame(columns=COLUMNS)
    if out.exists():
        # Stamps as text: a commit such as 0123456 or a hash such as 12e4567 would otherwise be read as a number.
        previous = pd.read_csv(out, dtype=STAMP_DTYPES)
        kept = resumable_rows(previous, ["scenario", "replicate"], set(tasks), STAMP)
        print(f"{len(kept)} of {len(previous)} stored replicates kept (SurvStudio {STAMP['survstudio_commit']}, design {DESIGN_HASH}, "
              f"code {CODE_HASH})", flush=True)
        if len(kept) < len(previous):
            # Replicates from other code or another design, failed ones and ones outside the design are set aside,
            # never mixed with this run's.
            write_csv_atomic(previous.drop(index=kept.index), RESULTS / "simulation_replicates_stale.csv")
            print(f"{len(previous) - len(kept)} moved to simulation_replicates_stale.csv", flush=True)
    rows = kept.to_dict("records")
    done = set(zip(kept["scenario"], kept["replicate"]))
    tasks = [task for task in tasks if task not in done]
    print(f"{len(tasks)} replicates to run with {workers} workers; censoring rate {CENSOR_RATE:.4f}", flush=True)
    # Written even when nothing is left to run, so the kept replicates carry this run's stamp (common.stamp_result).
    write_replicates(out, rows)
    began = time.time()
    if tasks:
        # A worker that dies breaks the pool: the next result raises BrokenProcessPool instead of waiting forever.
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            futures = [pool.submit(replicate, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                write_replicates(out, rows)
                if count % 25 == 0:
                    print(f"{count}/{len(tasks)} done, {time.time() - began:.0f}s", flush=True)
    write_json(settings, {
        "survstudio": SURVSTUDIO, "design_hash": DESIGN_HASH, "code_hash": CODE_HASH,
        "code_files": [path.name for path in CODE_FILES], "genes_per_replicate": GENES_PER_REPLICATE, "true_markers": TRUE_MARKERS,
        "weibull_shape": SHAPE, "median_months": MEDIAN_MONTHS, "administrative_months": ADMIN_MONTHS,
        "target_event_rate": TARGET_EVENT_RATE, "censoring_rate": CENSOR_RATE, "linked_r": LINKED_R, "base_seed": BASE_SEED,
        "permutations": PERMUTATIONS, "new_patients": NEW_PATIENTS,
        "clinical_log_hr": dict(zip(DESIGN.columns, CLINICAL_FIT.beta.tolist())),
        "patients": N, "genes": int(GENES.size),
        "scenarios": {name: dict(zip(("beta", "max_mode_fraction", "subsamples", "replicates"), values)) for name, values in SCENARIOS.items()},
    })
    print(f"all done in {time.time() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()

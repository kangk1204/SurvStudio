"""Comparison with the pipelines commonly used to publish prognostic signatures, experiment 2: claims under a null.

Script 06's plasmode null: the TCGA-LUAD expression and clinical data are kept and overall survival is simulated from
age, sex and stage alone (Weibull proportional hazards, shape 1.2, the clinical effects fitted to TCGA-LUAD,
exponential censoring to about 37% deaths; script 06's generator, imported from it). No gene carries information
beyond the clinical covariates, but genes that go with stage are prognostic on their own, as in real data.

Replicate r (random stream default_rng([NULL_SEED, r])): the 484 TCGA patients with new null outcomes (development);
seven pseudo-cohorts of the GEO cohorts' sizes (115, 81, 204, 171, 118, 429, 391) drawn with replacement from the TCGA
patients, with new null outcomes (validation, named sim_<GEO cohort>); and 3,000 patients drawn the same way ("truth":
a model's C and gain there stand for their values in new patients). The genes are experiment 1's (case study I's genes
that all seven GEO cohorts measure), z-scored within each cohort; the R pipelines z-score each cohort themselves
(paper/competitors/common.R), P3 needs no scaling (its cut-offs are ranks) and script 15's test standardises within
the cohort.

  python 18_competitors_null.py generate [replicates]   the replicate designs, results/competitors/null/rep_NNNN/design.csv
  python 18_competitors_null.py p3 [workers]            P3 on every replicate (rep_NNNN/p3.json)
  python 18_competitors_null.py summarise              P1 and P2 from the R runs (rep_NNNN/mime, rep_NNNN/p2) and P3:
                                                        competitors_null_replicates.csv, competitors_null_splits.csv and
                                                        competitors_null.json in results/
SurvStudio's numbers under the same null come from script 06 (results/simulation_summary.json); when that run lacks
the null with subsamples, SUBSAMPLES_SUMMARY may name the simulation_summary.json of a newer run that has it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import CATEGORICAL, COVARIATES, GEO_COHORTS, RESULTS, SCRIPTS, read_result, survstudio_version, write_csv_atomic, write_json  # noqa: E402
from competitors import (  # noqa: E402
    CANDIDATE_CAP,
    DAYS_PER_MONTH,
    WORK,
    RiskSets,
    adjusted_statistics,
    best_cutoff_scan,
    choose_winner,
    clean,
    code_hash,
    ensure_folder,
    external_gain_point,
    harrell_c,
    median_split,
    r_versions,
    replicated,
    risk_score_recipe,
    splits,
    summarise as describe,
)

NULL = WORK / "null"
NULL_SEED = 20260929
REPLICATES = 200
MIME_REPLICATES = 50
TRUTH_PATIENTS = 3000
SIZES = {"GSE13213": 115, "GSE30219": 81, "GSE31210": 204, "GSE41271": 171, "GSE50081": 118, "GSE68465": 429, "GSE72094": 391}
PSEUDO = [f"sim_{cohort}" for cohort in GEO_COHORTS]
CLAIM_C = 0.55


def generator():
    """Script 06, imported for its null generator (it reads TCGA-LUAD as case study I does)."""
    spec = importlib.util.spec_from_file_location("simulation06", SCRIPTS / "06_simulation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tcga_patients() -> pd.DataFrame:
    """The development patients in the order of null/tcga_expression.csv (18_competitors_data.py)."""
    return pd.read_csv(NULL / "tcga_clinical.csv", dtype={"patient_id": str})


def generate(replicates: int) -> None:
    sim = generator()
    patients = tcga_patients()
    position = {patient: index for index, patient in enumerate(sim.frame["patient_id"].astype(str))}
    if sorted(position) != sorted(patients["patient_id"]):
        raise SystemExit("Script 06's patients differ from the development patients of 18_competitors_data.py.")
    # Script 06's clinical linear predictor, in this comparison's patient order.
    eta = sim.ETA_CLINICAL[[position[patient] for patient in patients["patient_id"]]]
    settings = {"null_seed": NULL_SEED, "replicates": replicates,
                "mime_replicates": min(replicates, int(os.environ.get("MIME_REPLICATES", MIME_REPLICATES))),
                "truth_patients": TRUTH_PATIENTS, "sizes": SIZES,
                "censoring_rate": float(sim.CENSOR_RATE), "weibull_shape": float(sim.SHAPE), "median_months": float(sim.MEDIAN_MONTHS),
                "administrative_months": float(sim.ADMIN_MONTHS), "clinical_log_hr": dict(zip(sim.DESIGN.columns, map(float, sim.CLINICAL_FIT.beta)))}
    for replicate in range(replicates):
        rng = np.random.default_rng([NULL_SEED, replicate])
        parts = []
        rows = np.arange(len(patients))
        time_, event = sim.simulate(eta, rng)
        parts.append(pd.DataFrame({"cohort": "DEV", "row": rows + 1, "time_days": time_ * DAYS_PER_MONTH, "event": event}))
        for cohort, size in [*((f"sim_{name}", size) for name, size in SIZES.items()), ("truth", TRUTH_PATIENTS)]:
            rows = rng.integers(0, len(patients), size)
            time_, event = sim.simulate(eta[rows], rng)
            parts.append(pd.DataFrame({"cohort": cohort, "row": rows + 1, "time_days": time_ * DAYS_PER_MONTH, "event": event}))
        folder = ensure_folder(NULL / f"rep_{replicate:04d}")
        pd.concat(parts, ignore_index=True).to_csv(folder / "design.csv", index=False, float_format="%.10g")
    (NULL / "settings.json").write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")
    print(f"{replicates} replicate designs in {NULL}", flush=True)


def cohort_frames(design: pd.DataFrame, patients: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each cohort of a replicate: its patients' covariates (from TCGA) with the simulated outcomes, in months, and
    the TCGA row (0-based) of each."""
    frames = {}
    for cohort, part in design.groupby("cohort", sort=False):
        rows = part["row"].to_numpy(dtype=int) - 1
        frame = patients.iloc[rows][COVARIATES].reset_index(drop=True)
        frame.insert(0, "os_event", part["event"].to_numpy(dtype=int))
        frame.insert(0, "os_months", part["time_days"].to_numpy(dtype=float) / DAYS_PER_MONTH)
        frame.insert(0, "patient_id", [f"{cohort}_{index + 1}" for index in range(len(part))])
        frame["row"] = rows
        frames[str(cohort)] = frame
    return frames


# ── P3 on one replicate ─────────────────────────────────────────────────────────────────────────────────────────
_EXPRESSION: np.ndarray | None = None
_GENES: list[str] = []
_COLUMN: dict[str, int] = {}
_PATIENTS: pd.DataFrame | None = None


def _load_shared() -> None:
    """TCGA's expression of the comparison's genes and its patients, read once and shared with forked workers."""
    global _EXPRESSION, _GENES, _COLUMN, _PATIENTS
    table = pd.read_csv(NULL / "tcga_expression.csv", dtype={"patient_id": str})
    _PATIENTS = tcga_patients()
    if not table["patient_id"].equals(_PATIENTS["patient_id"]):
        raise SystemExit("tcga_expression.csv and tcga_clinical.csv list the patients in different orders.")
    _GENES = [column for column in table.columns if column != "patient_id"]
    _COLUMN = {gene: index for index, gene in enumerate(_GENES)}
    _EXPRESSION = table[_GENES].to_numpy(dtype=float)


def p3_replicate(replicate: int) -> dict:
    folder = NULL / f"rep_{replicate:04d}"
    frames = cohort_frames(pd.read_csv(folder / "design.csv"), _PATIENTS)
    development = frames["DEV"]
    began = time.time()
    scan = best_cutoff_scan(development["os_months"], development["os_event"], _EXPRESSION[development["row"]])
    scan.index = _GENES
    genes = len(_GENES)
    claimed = scan[scan["best_p"] < 0.05]
    bonferroni = scan[scan["best_p"] < 0.05 / genes]
    tests = []
    columns = [_COLUMN[gene] for gene in claimed.index]
    for cohort in PSEUDO:
        frame = frames[cohort]
        expression = pd.DataFrame(_EXPRESSION[frame["row"]][:, columns], columns=list(claimed.index))
        tests.append(adjusted_statistics(frame, expression).assign(cohort=cohort))
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(columns=["gene", "log_hr", "se", "cohort"])
    direction = np.sign(np.log(claimed["hazard_ratio"])).astype(int)
    replication = replicated(tests, direction[direction != 0]) if len(claimed) else pd.DataFrame(columns=["gene", "replicated"])
    record = {
        "replicate": replicate, "genes": genes, "events": int(development["os_event"].sum()),
        "best_cutoff_p05": int(len(claimed)), "best_cutoff_bonferroni": int(len(bonferroni)),
        "median_cut_p05": int((scan["median_p"] < 0.05).sum()), "median_cut_bonferroni": int((scan["median_p"] < 0.05 / genes).sum()),
        "claimed_replicated": int(replication["replicated"].fillna(False).sum()) if len(replication) else 0,
        "claimed_evaluable": int(replication["replicated"].notna().sum()) if len(replication) else 0,
        "bonferroni_replicated": int(replication.loc[replication["gene"].isin(bonferroni.index), "replicated"].fillna(False).sum()) if len(replication) else 0,
        "seconds": round(time.time() - began, 1),
    }
    (folder / "p3.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


def run_p3(workers: int) -> None:
    _load_shared()
    replicates = sorted(int(path.name[4:]) for path in NULL.glob("rep_*") if (path / "design.csv").exists())
    todo = [replicate for replicate in replicates if not (NULL / f"rep_{replicate:04d}" / "p3.json").exists()]
    print(f"P3 on {len(todo)} of {len(replicates)} replicates with {workers} workers", flush=True)
    began = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        for count, record in enumerate(pool.map(p3_replicate, todo), start=1):
            if count % 20 == 0:
                print(f"{count}/{len(todo)} done, {time.time() - began:.0f}s; last: {record}", flush=True)


# ── P1 and P2 from the R runs ────────────────────────────────────────────────────────────────────────────────────
def risk_table(path: Path, value: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"ID": str})
    return table.rename(columns={value: "risk"})


def aligned(risk: pd.DataFrame, patients: pd.DataFrame, value: str) -> np.ndarray:
    """A cohort's risk scores in the order of its patients (by ID)."""
    scores = risk.set_index(risk["ID"].astype(str))[value]
    if scores.index.duplicated().any() or not set(patients["patient_id"]).issubset(scores.index):
        raise RuntimeError("The risk scores do not name each patient of the cohort once.")
    return scores.loc[patients["patient_id"]].to_numpy(dtype=float)


def p2_replicate(replicate: int, frames: dict[str, pd.DataFrame]) -> dict:
    folder = NULL / f"rep_{replicate:04d}" / "p2"
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    record = {"replicate": replicate, "p2_unicox_kept": info["unicox_kept"], "p2_selected": info["selected"]}
    if not info["selected"]:
        return record
    risk = risk_table(folder / "risk.csv", "risk")
    scores = {cohort: aligned(part, frames[cohort], "risk") for cohort, part in risk.groupby("cohort", sort=False)}
    development = frames["DEV"]
    record["p2_training_c"] = float(harrell_c(development["os_months"], development["os_event"], scores["DEV"])[0])
    record["p2_training_p"], record["p2_training_hr"] = median_split(development["os_months"], development["os_event"], scores["DEV"])
    external = [median_split(frames[cohort]["os_months"], frames[cohort]["os_event"], scores[cohort]) for cohort in PSEUDO]
    record["p2_external_p05_any"] = bool(any(p < 0.05 for p, _ in external))
    record["p2_external_p05_claimed_direction"] = bool(any(p < 0.05 and hazard > 1 for p, hazard in external))
    record["p2_external_p05_count"] = int(sum(p < 0.05 for p, _ in external))
    record["p2_external_mean_c"] = float(np.mean([harrell_c(frames[c]["os_months"], frames[c]["os_event"], scores[c])[0] for c in PSEUDO]))
    truth = frames["truth"]
    record["p2_truth_c"] = float(harrell_c(truth["os_months"], truth["os_event"], scores["truth"])[0])
    record["p2_truth_gain"] = truth_gain(development, scores["DEV"], truth, scores["truth"], "P2")
    record["p2_claim"] = bool(record["p2_training_p"] < 0.05)
    return record


def truth_gain(development: pd.DataFrame, development_risk: np.ndarray, truth: pd.DataFrame, truth_risk: np.ndarray, label: str) -> float:
    """The paired C gain over the clinical-only model (SurvStudio's definition) of the clinical covariates plus the risk
    score, fitted on the development cohort, in the 3,000 truth patients."""
    from survival_toolkit.marker_evaluation import prepare_marker_cohort

    frame = development[["os_months", "os_event", *COVARIATES]].assign(risk_score=np.asarray(development_risk, dtype=float))
    cohort = prepare_marker_cohort(frame, time_column="os_months", event_column="os_event", marker_columns=["risk_score"],
                                   clinical_columns=COVARIATES, categorical_clinical=CATEGORICAL, event_positive_value=1)
    recipe = risk_score_recipe(cohort, frame["risk_score"].to_numpy(dtype=float), label)
    return external_gain_point(recipe, truth, truth_risk)


def p1_replicate(replicate: int, frames: dict[str, pd.DataFrame]) -> list[dict]:
    """The selection replay of one replicate: every split of the seven pseudo-cohorts into three selection and four
    sealed cohorts, and the design with all seven for selection."""
    folder = NULL / f"rep_{replicate:04d}" / "mime"
    cindex = pd.read_csv(folder / "cindex.csv")
    reported = cindex.pivot_table(index="model", columns="cohort", values="cindex", sort=False)
    models = list(dict.fromkeys(cindex["model"]))
    reported = reported.reindex(models)
    risk = pd.read_csv(folder / "risk.csv.gz", dtype={"ID": str})
    development = frames["DEV"]
    scores: dict[str, dict[str, np.ndarray]] = {}
    for (model, cohort), part in risk.groupby(["model", "cohort"], sort=False):
        scores.setdefault(model, {})[cohort] = aligned(part, frames[cohort], "RS")
    orientation = {}
    honest = pd.DataFrame(index=models, columns=[*PSEUDO, "truth"], dtype=float)
    for model in models:
        if model not in scores or "DEV" not in scores[model]:
            continue
        training = harrell_c(development["os_months"], development["os_event"], scores[model]["DEV"])[0]
        orientation[model] = 1.0 if not np.isfinite(training) or training >= 0.5 else -1.0
        for cohort in [*PSEUDO, "truth"]:
            frame = frames[cohort]
            honest.loc[model, cohort] = harrell_c(frame["os_months"], frame["os_event"], orientation[model] * scores[model][cohort])[0]
    rows = []
    designs = [*splits(PSEUDO), (tuple(PSEUDO), ())]
    gains: dict[str, float] = {}
    for chosen, sealed in designs:
        winner, value = choose_winner(reported[list(PSEUDO)], chosen)
        record = {"replicate": replicate, "design": "all seven" if not sealed else "3 selection + 4 sealed",
                  "selection": ";".join(chosen), "winner": winner, "reported_c": value}
        if winner is not None:
            record["training_c"] = float(reported.loc[winner, "DEV"])
            if sealed:
                record["sealed_reported_mean"] = float(reported.loc[winner, list(sealed)].mean())
                record["sealed_honest_mean"] = float(honest.loc[winner, list(sealed)].mean())
                sealed_means = honest[list(sealed)].mean(axis=1)
                record["sealed_honest_mean_all_models"] = float(sealed_means.mean())
                record["winner_beats_share_of_models_sealed"] = float((sealed_means < sealed_means[winner]).mean())
            record["truth_honest_c"] = float(honest.loc[winner, "truth"])
            record["truth_reported_c"] = float(reported.loc[winner, "truth"]) if "truth" in reported.columns else np.nan
            tests = [median_split(frames[c]["os_months"], frames[c]["os_event"], orientation[winner] * scores[winner][c]) for c in chosen]
            record["selection_km_p05_any"] = bool(any(p < 0.05 for p, _ in tests))
            record["selection_km_p05_claimed_direction"] = bool(any(p < 0.05 and hazard > 1 for p, hazard in tests))
            if winner not in gains:
                gains[winner] = truth_gain(development, orientation[winner] * scores[winner]["DEV"], frames["truth"],
                                           orientation[winner] * scores[winner]["truth"], f"Mime {winner}")
            record["truth_gain"] = gains[winner]
        rows.append(record)
    return rows


def summarise() -> None:
    patients = tcga_patients()
    replicates = sorted(int(path.name[4:]) for path in NULL.glob("rep_*") if (path / "design.csv").exists())
    per_replicate, split_rows = [], []
    for replicate in replicates:
        folder = NULL / f"rep_{replicate:04d}"
        frames = cohort_frames(pd.read_csv(folder / "design.csv"), patients)
        record: dict = {"replicate": replicate, "events": int(frames["DEV"]["os_event"].sum())}
        if (folder / "p3.json").exists():
            record.update({f"p3_{key}": value for key, value in json.loads((folder / "p3.json").read_text(encoding="utf-8")).items() if key != "replicate"})
        if (folder / "p2" / "run.json").exists():
            record.update(p2_replicate(replicate, frames))
        elif (folder / "p2.log").exists():
            record["p2_failed"] = True
        if (folder / "mime" / "run.json").exists():
            split_rows.extend(p1_replicate(replicate, frames))
            record["p1_done"] = True
        elif (folder / "mime.log").exists():
            # Record a missing outcome; it is retained in the planned-denominator bounds.
            record["p1_failed"] = True
        per_replicate.append(record)
        print(f"replicate {replicate} summarised", flush=True)
    table = pd.DataFrame(per_replicate)
    splits_table = pd.DataFrame(split_rows)
    write_csv_atomic(table, RESULTS / "competitors_null_replicates.csv")
    write_csv_atomic(splits_table, RESULTS / "competitors_null_splits.csv")
    write_json(RESULTS / "competitors_null.json", clean(null_summary(table, splits_table)))


def rate(values: pd.Series, planned: int | None = None) -> dict:
    values = values.dropna().astype(bool)
    n = int(values.size)
    share = float(values.mean()) if n else None
    result = {"n": n, "rate": share, "mcse": float(np.sqrt(share * (1 - share) / n)) if n else None}
    if planned is not None:
        if planned <= 0 or n > planned:
            raise ValueError("Invalid planned denominator")
        successes = int(values.sum())
        result.update(planned_replicates=planned, failure_bounds=[successes / planned, (successes + planned - n) / planned])
    return result


def null_summary(table: pd.DataFrame, splits_table: pd.DataFrame) -> dict:
    summary: dict = {"survstudio": survstudio_version(), "code_hash": code_hash(), "r": r_versions(),
                     "design": json.loads((NULL / "settings.json").read_text(encoding="utf-8")), "claim_c": CLAIM_C,
                     "candidate_cap": CANDIDATE_CAP}
    planned = int(summary["design"]["replicates"])
    planned_mime = int(summary["design"].get("mime_replicates", MIME_REPLICATES))
    if planned <= 0 or not 0 < planned_mime <= planned:
        raise ValueError("Invalid planned replicate counts")
    mime_table = table[table["replicate"] < planned_mime]
    summary["failure_reporting"] = {
        "P1": {"planned": planned_mime, "completed": int(mime_table.get("p1_done", pd.Series(False, index=mime_table.index)).fillna(False).sum()),
               "failed_or_missing_ids": [i for i in range(planned_mime) if i not in set(mime_table.loc[mime_table.get("p1_done", pd.Series(False, index=mime_table.index)).fillna(False).astype(bool), "replicate"])]},
        "P2": {"planned": planned, "completed": int(table.get("p2_selected", pd.Series(index=table.index, dtype=float)).notna().sum())},
        "P3": {"planned": planned, "completed": int(table.get("p3_best_cutoff_p05", pd.Series(index=table.index, dtype=float)).notna().sum())},
        "interpretation": "Reported rates condition on completed fits; failure_bounds include every planned design with missing outcomes assigned zero or one. Failed fits are not replaced.",
    }
    if "p3_best_cutoff_p05" in table:
        p3 = table.dropna(subset=["p3_best_cutoff_p05"])
        summary["P3"] = {
            "replicates": int(len(p3)), "genes": int(p3["p3_genes"].iat[0]),
            "any_gene_p05": rate(p3["p3_best_cutoff_p05"] > 0, planned),
            "genes_p05": describe(p3["p3_best_cutoff_p05"]),
            "any_gene_bonferroni": rate(p3["p3_best_cutoff_bonferroni"] > 0, planned),
            "genes_bonferroni": describe(p3["p3_best_cutoff_bonferroni"]),
            "median_cut_genes_p05": describe(p3["p3_median_cut_p05"]),
            "median_cut_any_bonferroni": rate(p3["p3_median_cut_bonferroni"] > 0),
            "claimed_replicated_share": float(p3["p3_claimed_replicated"].sum() / max(p3["p3_claimed_evaluable"].sum(), 1)),
        }
    if "p2_selected" in table:
        p2 = table.dropna(subset=["p2_selected"])
        with_genes = p2[p2["p2_selected"] > 0]
        summary["P2"] = {
            "replicates": int(len(p2)), "any_gene_selected": rate(p2["p2_selected"] > 0, planned),
            "selected_genes": describe(p2["p2_selected"]),
            "claim_training_p05": rate(p2["p2_selected"].gt(0) & p2.get("p2_training_p", pd.Series(np.nan, index=p2.index)).lt(0.05), planned),
            "external_p05_any": rate(with_genes["p2_external_p05_any"]) if len(with_genes) else None,
            "external_p05_claimed_direction": rate(with_genes["p2_external_p05_claimed_direction"]) if len(with_genes) else None,
            "training_c": describe(with_genes.get("p2_training_c", [])), "external_mean_c": describe(with_genes.get("p2_external_mean_c", [])),
            "truth_c": describe(with_genes.get("p2_truth_c", [])), "truth_gain": describe(with_genes.get("p2_truth_gain", [])),
        }
    if len(splits_table):
        summary["P1"] = {}
        failed = table.loc[table.get("p1_failed", pd.Series(False, index=table.index)).fillna(False).astype(bool), "replicate"]
        summary["P1_failed_replicates"] = [int(value) for value in failed]
        for design, part in splits_table[splits_table["replicate"] < planned_mime].groupby("design"):
            splits_per_design = 35 if str(design).startswith("3 selection") else 1
            part = part.dropna(subset=["winner"])
            if not len(part):
                continue

            def by_replicate(flags: pd.Series) -> dict:
                """A rate over every split of every replicate, with its Monte Carlo SE from the replicates' own rates
                (the splits of one replicate share its data)."""
                means = flags.astype(float).groupby(part["replicate"]).mean()
                denominator = planned_mime * splits_per_design
                if len(flags) > denominator:
                    raise ValueError("More split outcomes than planned")
                successes = int(flags.astype(bool).sum())
                return {"n": int(len(flags)), "replicates": int(len(means)), "rate": float(means.mean()),
                        "planned_replicates": planned_mime,
                        "failure_bounds": [successes / denominator, (successes + denominator - len(flags)) / denominator],
                        "rate_conditions_on_completed_fits": True,
                        "mcse": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None}

            entry = {
                "replicates": int(part["replicate"].nunique()), "cases": int(len(part)),
                "reported_c": describe(part["reported_c"]), "reported_c_at_least_claim": by_replicate(part["reported_c"] >= CLAIM_C),
                "training_c": describe(part["training_c"]), "truth_honest_c": describe(part["truth_honest_c"]),
                "truth_gain": describe(part["truth_gain"]),
                "selection_km_p05_any": by_replicate(part["selection_km_p05_any"]),
                "selection_km_p05_claimed_direction": by_replicate(part["selection_km_p05_claimed_direction"]),
                "winners": part["winner"].value_counts().head(10).to_dict(),
            }
            if "sealed_honest_mean" in part and part["sealed_honest_mean"].notna().any():
                entry["sealed_honest_mean"] = describe(part["sealed_honest_mean"])
                entry["sealed_reported_mean"] = describe(part["sealed_reported_mean"])
                entry["optimism_vs_sealed"] = describe(part["reported_c"] - part["sealed_honest_mean"])
                entry["claim_not_holding"] = by_replicate((part["reported_c"] >= CLAIM_C) & (part["sealed_honest_mean"] < CLAIM_C))
                # Mime's C above the honest C in the sealed cohorts: each cohort's own Cox fit sets the score's direction.
                entry["reported_minus_honest_sealed"] = describe(part["sealed_reported_mean"] - part["sealed_honest_mean"])
                entry["sealed_honest_mean_all_models"] = describe(part["sealed_honest_mean_all_models"])
                entry["winner_beats_share_of_models_sealed"] = describe(part["winner_beats_share_of_models_sealed"])
            summary["P1"][design] = entry
    try:
        simulation = read_result("simulation_summary.json")
        null = next(row for row in simulation["scenarios"] if row["scenario"] == "null_filter")
        summary["SurvStudio"] = {"source": "script 06, scenario null_filter", "replicates": null["replicates"], "fwer": null["fwer"],
                                 "fwer_mcse": null["fwer_mcse"], "false_per_replicate": null["false_per_replicate"],
                                 "planned_replicates": null.get("planned_replicates"),
                                 "fwer_failure_bounds": null.get("fwer_failure_bounds"),
                                 "null_with_subsamples": subsample_scenario(simulation)}
        # The null with subsamples (script 06's null_filter_subsamples, with SurvStudio's gain verdicts) may come from a
        # newer run of script 06 than results/ holds: SUBSAMPLES_SUMMARY names its simulation_summary.json.
        other = os.environ.get("SUBSAMPLES_SUMMARY")
        if summary["SurvStudio"]["null_with_subsamples"] is None and other:
            text = Path(other).read_bytes()
            summary["SurvStudio"]["null_with_subsamples"] = subsample_scenario(json.loads(text.decode("utf-8")))
            summary["SurvStudio"]["null_with_subsamples_source"] = {"file": other, "sha256": hashlib.sha256(text).hexdigest()}
    except (OSError, StopIteration, KeyError) as problem:
        summary["SurvStudio"] = {"missing": str(problem)}
    return summary


def subsample_scenario(simulation: dict) -> dict | None:
    """Script 06's null scenario with subsamples (null_filter_subsamples), with its survstudio stamp; None when absent."""
    row = next((row for row in simulation.get("scenarios", []) if row.get("scenario") == "null_filter_subsamples"), None)
    return None if row is None else {**row, "survstudio": simulation.get("survstudio")}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "summarise"
    if command == "generate":
        generate(int(sys.argv[2]) if len(sys.argv) > 2 else REPLICATES)
    elif command == "p3":
        run_p3(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif command == "summarise":
        summarise()
    else:
        raise SystemExit(__doc__)

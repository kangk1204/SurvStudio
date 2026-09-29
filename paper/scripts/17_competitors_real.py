"""Comparison with the pipelines commonly used to publish prognostic signatures, experiment 1: developed on TCGA-LUAD,
validated in the seven GEO cohorts of case study II, with the patients, outcomes, covariates and genes of
17_competitors_data.py and the R runs of paper/competitors/ (run_competitors.sh runs them in order).

P1 Mime: every model's C as Mime reports it (summary(coxph(Surv ~ RS))$concordance, whose direction the cohort's own
fit sets) and its honest C (Harrell's C of the risk score oriented once, on TCGA, with SurvStudio's conventions and
bootstrap), then the selection replayed for every split of the seven cohorts into three selection and four sealed
cohorts (the winner has the highest mean reported C over the selection cohorts) and for all seven used for selection.
For each winner, SurvStudio's external gain: a Cox model of age, sex, stage and the winner's risk score fitted on TCGA,
validated with validate_locked_recipe as script 03 validates SurvStudio's locked model, and pooled by random effects.
P2 univariate Cox -> LASSO -> Cox and P3 KM Plotter's best cut-off (with script 15's replication rule) likewise.
SurvStudio's own numbers come from case studies I and II and script 15 (results/), checked against the paper's.

Writes competitors_mime_models.csv, competitors_mime_replay.csv, competitors_p2_cohorts.csv, competitors_p3_genes.csv
and competitors_real.json to results/.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import GEO_COHORTS, RESULTS, read_result, survstudio_version, write_csv_atomic, write_json
from competitors import (
    CANDIDATE_CAP,
    SELECTION_COHORTS,
    SENSITIVITY_CAP,
    WORK,
    adjusted_statistics,
    best_cutoff_scan,
    bootstrap_c,
    clean,
    code_hash,
    development_data,
    external_gain,
    harrell_c,
    median_split,
    pooled,
    pooled_c_gain,
    r_versions,
    read_json,
    records,
    replay,
    replicated,
    risk_score_recipe,
    splits,
    summarise,
)

INPUT = WORK / "input"
REAL = WORK / "real"
COHORTS = ["TCGA", *GEO_COHORTS]
# The paper's SurvStudio numbers for case study II (within-cohort scaling), as rounded in the text.
PAPER_SURVSTUDIO = {"model_c": 0.665, "clinical_c": 0.661, "delta_c": 0.000, "delta_lower": -0.032, "delta_upper": 0.032}


def aligned(risk: pd.DataFrame, patients: pd.DataFrame, value: str) -> np.ndarray:
    """A cohort's risk scores in the order of its patients (by ID)."""
    scores = risk.set_index(risk["ID"].astype(str))[value]
    if scores.index.duplicated().any() or not set(patients["patient_id"]).issubset(scores.index):
        raise RuntimeError("The risk scores do not name each patient of the cohort once.")
    return scores.loc[patients["patient_id"]].to_numpy(dtype=float)


def gain_rows(label: str, recipe: dict, patients: dict[str, pd.DataFrame], scores: dict[str, np.ndarray], scaling: str = "as_measured") -> pd.DataFrame:
    rows = [{"cohort": cohort, **external_gain(recipe, patients[cohort], scores[cohort], scaling)} for cohort in GEO_COHORTS]
    return pd.DataFrame(rows).assign(model=label, scaling=scaling)


def pooled_gain(rows: pd.DataFrame, cohorts: list[str]) -> dict:
    part = rows[rows["cohort"].isin(cohorts)]
    return pooled_c_gain(part)


def survstudio_section(patients: dict[str, pd.DataFrame]) -> dict:
    """SurvStudio's numbers from case studies I and II and script 15, with the checks: the same patients per cohort,
    and the pooled numbers recomputed from external_validation.csv equal to external_pooled.json and to the paper's."""
    validation = read_result("external_validation.csv")
    within = validation[validation["scaling"] == "within_cohort"].set_index("cohort")
    for cohort in GEO_COHORTS:
        mine = {"n": len(patients[cohort]), "events": int(patients[cohort]["os_event"].sum())}
        theirs = {"n": int(within.loc[cohort, "n"]), "events": int(within.loc[cohort, "events"])}
        if mine != theirs:
            raise SystemExit(f"{cohort}: the comparison has {mine}, SurvStudio's validation {theirs}.")
    recomputed = pooled_c_gain(within.reset_index())
    stored = read_result("external_pooled.json")["within_cohort"]
    for key in ("model_c", "clinical_c", "delta_c"):
        for field in ("estimate", "ci_lower", "ci_upper"):
            if abs(recomputed[key][field] - stored[key][field]) > 1e-12:
                raise SystemExit(f"Pooled {key} {field}: recomputed {recomputed[key][field]}, stored {stored[key][field]}.")
    rounded = {"model_c": round(recomputed["model_c"]["estimate"], 3), "clinical_c": round(recomputed["clinical_c"]["estimate"], 3),
               "delta_c": round(recomputed["delta_c"]["estimate"], 3) + 0.0, "delta_lower": round(recomputed["delta_c"]["ci_lower"], 3),
               "delta_upper": round(recomputed["delta_c"]["ci_upper"], 3)}
    if rounded != PAPER_SURVSTUDIO:
        raise SystemExit(f"SurvStudio's pooled numbers {rounded} are not the paper's {PAPER_SURVSTUDIO}.")
    summary = read_result("tcga_markers_summary.json")
    tiers = read_result("tier_replication.json")["cases"]["I"]["groups"]
    signature = summary["signature"]
    return {
        "robust_genes": summary["robust"], "tier_counts": summary["tier_counts"], "funnel": summary["funnel"],
        "signature_markers": signature["markers"], "apparent_c": signature["apparent_c"],
        "optimism_corrected_c": signature["optimism_corrected_c"], "left_out_c": signature["signature_c_left_out"],
        "clinical_left_out_c": signature["clinical_c_left_out"], "left_out_gain": signature["signature_gain_left_out"],
        "external_within_cohort": recomputed, "external_matches_paper": rounded,
        "tiers_replication": {group: {key: value[key] for key in ("genes", "evaluable", "replicated", "rate", "rate_ci")} for group, value in tiers.items()},
    }


def mime_section(folder, patients: dict[str, pd.DataFrame], development, label: str, with_gains: bool = True) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    info = read_json(folder / "run.json")
    cindex = pd.read_csv(folder / "cindex.csv")
    models = list(dict.fromkeys(cindex["model"]))
    reported = cindex.pivot_table(index="model", columns="cohort", values="cindex", sort=False).reindex(models)
    risk = pd.read_csv(folder / "risk.csv.gz", dtype={"ID": str})
    genes = pd.read_csv(folder / "genes.csv").set_index("model")
    scores: dict[str, dict[str, np.ndarray]] = {}
    for (model, cohort), part in risk.groupby(["model", "cohort"], sort=False):
        scores.setdefault(model, {})[cohort] = aligned(part, patients[cohort], "RS")
    tcga = patients["TCGA"]
    orientation = {model: 1.0 for model in models}
    for model in models:
        training = harrell_c(tcga["os_months"], tcga["os_event"], scores[model]["TCGA"])[0]
        orientation[model] = 1.0 if not np.isfinite(training) or training >= 0.5 else -1.0
    honest = pd.DataFrame(index=models, columns=["TCGA", *GEO_COHORTS], dtype=float)
    lower = pd.DataFrame(index=models, columns=GEO_COHORTS, dtype=float)
    upper = pd.DataFrame(index=models, columns=GEO_COHORTS, dtype=float)
    for cohort in COHORTS:
        frame = patients[cohort]
        matrix = np.column_stack([orientation[model] * scores[model][cohort] for model in models])
        honest[cohort] = harrell_c(frame["os_months"], frame["os_event"], matrix)
        if cohort != "TCGA":
            lower[cohort], upper[cohort] = bootstrap_c(frame["os_months"], frame["os_event"], matrix)
    table = pd.DataFrame({"model": models, "orientation": [orientation[model] for model in models],
                          "n_genes": [genes["n_genes"].get(model, np.nan) for model in models],
                          "training_reported_c": reported["TCGA"].to_numpy(), "training_honest_c": honest["TCGA"].to_numpy()})
    for cohort in GEO_COHORTS:
        table[f"reported_c_{cohort}"] = reported[cohort].to_numpy()
        table[f"honest_c_{cohort}"] = honest[cohort].to_numpy()
    pooled_rows = []
    for model in models:
        try:
            result = pooled(honest.loc[model, GEO_COHORTS], lower.loc[model, GEO_COHORTS], upper.loc[model, GEO_COHORTS])
            pooled_rows.append((result["estimate"], result["ci_lower"], result["ci_upper"]))
        except ValueError:
            pooled_rows.append((np.nan, np.nan, np.nan))
    table["mean_reported_c_geo"] = reported[GEO_COHORTS].mean(axis=1).to_numpy()
    table["pooled_honest_c_geo"], table["pooled_honest_lower"], table["pooled_honest_upper"] = zip(*pooled_rows)
    table["genes"] = [genes["genes"].get(model, "") for model in models]

    replay_table = replay(reported[GEO_COHORTS], honest[GEO_COHORTS], lower, upper, training=reported["TCGA"])
    everything = replay(reported[GEO_COHORTS], honest[GEO_COHORTS], lower, upper, training=reported["TCGA"], selection=len(GEO_COHORTS))
    replay_table["design"] = f"{SELECTION_COHORTS} selection + {len(GEO_COHORTS) - SELECTION_COHORTS} sealed"
    everything["design"] = "all seven"
    replay_table = pd.concat([replay_table, everything], ignore_index=True)
    # Median-split log-rank tests of each winner in its selection cohorts (as a paper's KM plots would show them).
    km = {}
    for winner in replay_table["winner"].dropna().unique():
        km[winner] = {cohort: median_split(patients[cohort]["os_months"], patients[cohort]["os_event"], orientation[winner] * scores[winner][cohort])
                      for cohort in GEO_COHORTS}
    replay_table["selection_km_p05_any"] = [any(km[w][c][0] < 0.05 for c in s.split(";")) if isinstance(w, str) else np.nan
                                            for w, s in zip(replay_table["winner"], replay_table["selection"])]
    replay_table["selection_km_p05_claimed_direction"] = [any(km[w][c][0] < 0.05 and km[w][c][1] > 1 for c in s.split(";")) if isinstance(w, str) else np.nan
                                                          for w, s in zip(replay_table["winner"], replay_table["selection"])]
    replay_table["selection_km_p05_count"] = [sum(km[w][c][0] < 0.05 for c in s.split(";")) if isinstance(w, str) else np.nan
                                              for w, s in zip(replay_table["winner"], replay_table["selection"])]
    replay_table["winner_genes"] = [genes["n_genes"].get(w, np.nan) if isinstance(w, str) else np.nan for w in replay_table["winner"]]
    gains = pd.DataFrame()
    if with_gains:
        rows = []
        for winner in replay_table["winner"].dropna().unique():
            oriented = {cohort: orientation[winner] * scores[winner][cohort] for cohort in COHORTS}
            recipe = risk_score_recipe(development, oriented["TCGA"], f"Mime {winner}")
            rows.append(gain_rows(winner, recipe, patients, oriented))
        gains = pd.concat(rows, ignore_index=True)
        pooled_rows = []
        for winner, sealed in zip(replay_table["winner"], replay_table["sealed"]):
            cohorts = sealed.split(";") if isinstance(sealed, str) and sealed else GEO_COHORTS
            result = pooled_gain(gains[gains["model"] == winner], cohorts) if isinstance(winner, str) else None
            pooled_rows.append({} if result is None else {
                "gain": result["delta_c"]["estimate"], "gain_lower": result["delta_c"]["ci_lower"], "gain_upper": result["delta_c"]["ci_upper"],
                "combined_c": result["model_c"]["estimate"], "clinical_c": result["clinical_c"]["estimate"]})
        replay_table = pd.concat([replay_table, pd.DataFrame(pooled_rows)], axis=1)
    split_part = replay_table[replay_table["design"] != "all seven"]
    usual = replay_table[replay_table["design"] == "all seven"].iloc[0]
    section = {
        "label": label, "run": info, "models": len(models), "candidates": info["candidates"], "unicox_passing": info["unicox_passing"],
        "models_with_reversed_orientation": int(sum(value < 0 for value in orientation.values())),
        "splits": {
            "count": int(len(split_part)), "distinct_winners": split_part["winner"].value_counts().to_dict(),
            "reported_c": summarise(split_part["reported_c"]), "training_c": summarise(split_part["training_c"]),
            "sealed_reported_mean": summarise(split_part["sealed_reported_mean"]), "sealed_honest_mean": summarise(split_part["sealed_honest_mean"]),
            "sealed_honest_pooled": summarise(split_part["sealed_honest_pooled"]),
            "optimism_vs_sealed_mean": summarise(split_part["optimism_vs_sealed_mean"]),
            "optimism_vs_sealed_pooled": summarise(split_part["reported_c"] - split_part["sealed_honest_pooled"]),
            "selection_optimism": summarise(split_part["selection_optimism"]),
            "selection_km_p05_any": float(split_part["selection_km_p05_any"].mean()),
        },
        "all_seven": {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in usual.items()},
    }
    if with_gains:
        section["splits"]["gain"] = summarise(split_part["gain"])
        section["splits"]["gain_interval_excludes_zero"] = float(((split_part["gain_lower"] > 0) | (split_part["gain_upper"] < 0)).mean())
        section["splits"]["gain_interval_above_zero"] = float((split_part["gain_lower"] > 0).mean())
    return section, table, replay_table


def p2_section(patients: dict[str, pd.DataFrame], development, tests: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    folder = REAL / "p2"
    info = read_json(folder / "run.json")
    risk = pd.read_csv(folder / "risk.csv", dtype={"ID": str})
    cox = pd.read_csv(folder / "cox.csv")
    scores = {cohort: aligned(part, patients[cohort], "risk") for cohort, part in risk.groupby("cohort", sort=False)}
    tcga = patients["TCGA"]
    training_p, training_hr = median_split(tcga["os_months"], tcga["os_event"], scores["TCGA"])
    rows = []
    for cohort in GEO_COHORTS:
        frame = patients[cohort]
        c_value = harrell_c(frame["os_months"], frame["os_event"], scores[cohort])[0]
        c_lower, c_upper = bootstrap_c(frame["os_months"], frame["os_event"], scores[cohort])
        p_value, hazard_ratio = median_split(frame["os_months"], frame["os_event"], scores[cohort])
        rows.append({"cohort": cohort, "n": len(frame), "events": int(frame["os_event"].sum()), "c": c_value, "c_lower": c_lower[0],
                     "c_upper": c_upper[0], "median_split_p": p_value, "median_split_hr": hazard_ratio})
    cohorts = pd.DataFrame(rows)
    pooled_c = pooled(cohorts["c"], cohorts["c_lower"], cohorts["c_upper"])
    recipe = risk_score_recipe(development, scores["TCGA"], "P2 uni-Cox -> LASSO -> Cox")
    gains = gain_rows("P2", recipe, patients, scores)
    gains_within = gain_rows("P2", recipe, patients, scores, scaling="within_cohort")
    cohorts = cohorts.merge(gains.drop(columns=["model", "scaling", "n", "events"]).add_prefix("gain_").rename(columns={"gain_cohort": "cohort"}), on="cohort")
    direction = pd.Series(np.sign(cox["coefficient"]).astype(int).to_numpy(), index=cox["gene"])
    replication = replicated(tests, direction[direction != 0])
    section = {
        "run": info, "genes_selected": cox["gene"].tolist(), "cox_coefficients": dict(zip(cox["gene"], cox["coefficient"])),
        "training_c": float(harrell_c(tcga["os_months"], tcga["os_event"], scores["TCGA"])[0]),
        "training_median_split_p": training_p, "training_median_split_hr": training_hr,
        "external_c_pooled": pooled_c, "external_c_mean": float(cohorts["c"].mean()),
        "external_median_split_p": dict(zip(cohorts["cohort"], cohorts["median_split_p"])),
        "external_median_split_p05": int((cohorts["median_split_p"] < 0.05).sum()),
        "external_median_split_p05_claimed_direction": int(((cohorts["median_split_p"] < 0.05) & (cohorts["median_split_hr"] > 1)).sum()),
        "gain": pooled_gain(gains, GEO_COHORTS), "gain_within_cohort_scaling": pooled_gain(gains_within, GEO_COHORTS),
        "genes_replicated": int(replication["replicated"].fillna(False).sum()), "genes_evaluable": int(replication["replicated"].notna().sum()),
    }
    return section, cohorts


def p3_section(patients: dict[str, pd.DataFrame], expression: dict[str, pd.DataFrame], tests: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tcga = patients["TCGA"]
    genes = list(expression["TCGA"].columns)
    began = time.time()
    scan = best_cutoff_scan(tcga["os_months"], tcga["os_event"], expression["TCGA"].to_numpy(dtype=float))
    scan.insert(0, "gene", genes)
    seconds = time.time() - began
    scan["direction"] = np.sign(np.log(scan["hazard_ratio"])).fillna(0).astype(int)
    scan["claimed"] = scan["best_p"] < 0.05
    scan["claimed_bonferroni"] = scan["best_p"] < 0.05 / len(genes)
    replication = replicated(tests, scan.loc[scan["claimed"] & (scan["direction"] != 0)].set_index("gene")["direction"])
    scan = scan.merge(replication[["gene", "cohorts", "pooled_log_hr", "ci_lower", "ci_upper", "replicated"]], on="gene", how="left")

    def share(mask: pd.Series) -> dict:
        part = scan[mask & scan["replicated"].notna()]
        return {"claimed": int(mask.sum()), "evaluable": int(len(part)), "replicated": int(part["replicated"].astype(bool).sum()),
                "rate": float(part["replicated"].astype(bool).mean()) if len(part) else None}

    section = {"genes_scanned": len(genes), "seconds": round(seconds, 1),
               "claimed_p05": share(scan["claimed"]), "claimed_bonferroni": share(scan["claimed_bonferroni"]),
               "median_cut_p05": int((scan["median_p"] < 0.05).sum()), "median_cut_bonferroni": int((scan["median_p"] < 0.05 / len(genes)).sum()),
               "cutoffs_tried": summarise(scan["cutoffs_tried"])}
    return section, scan


def tier_check(tests: pd.DataFrame, genes: list[str]) -> dict:
    """SurvStudio's tiers on the comparison's genes (script 15's output), and script 15's rule recomputed here on the
    same genes: the replication flags must agree."""
    markers = read_result("tcga_markers.csv", usecols=["marker", "tier", "added_value_p_value", "added_value_log_hr_one_step"])
    stored = read_result("tier_replication_genes.csv")
    stored = stored[(stored["case"] == "I") & stored["gene"].isin(genes)].set_index("gene")
    direction = pd.Series(np.sign(markers["added_value_log_hr_one_step"]).to_numpy(), index=markers["marker"]).reindex(stored.index).astype(int)
    mine = replicated(tests, direction[direction != 0]).set_index("gene")
    both = stored.join(mine[["replicated", "pooled_log_hr"]], rsuffix="_here", how="inner").dropna(subset=["replicated", "replicated_here"])
    disagree = both[both["replicated"].astype(bool) != both["replicated_here"].astype(bool)]
    groups = {}
    for group, part in stored.groupby("group"):
        evaluable = part[part["cohorts"] >= 2]
        groups[group] = {"genes": int(len(part)), "evaluable": int(len(evaluable)), "replicated": int(evaluable["replicated"].astype(bool).sum()),
                         "rate": float(evaluable["replicated"].astype(bool).mean()) if len(evaluable) else None}
    return {"groups_on_comparison_genes": groups, "genes_compared": int(len(both)), "replication_flags_disagree": int(len(disagree)),
            "largest_pooled_log_hr_difference": float(np.nanmax(np.abs(both["pooled_log_hr"] - both["pooled_log_hr_here"]))) if len(both) else None}


def main() -> None:
    began = time.time()
    patients = {cohort: pd.read_csv(INPUT / f"{cohort}_clinical.csv", dtype={"patient_id": str}) for cohort in COHORTS}
    expression = {}
    for cohort in COHORTS:
        frame = pd.read_csv(INPUT / f"{cohort}.csv", dtype={"ID": str})
        if not frame["ID"].equals(patients[cohort]["patient_id"]):
            raise SystemExit(f"{cohort}: the expression and clinical inputs list different patients.")
        expression[cohort] = frame.drop(columns=["ID", "OS.time", "OS"])
    genes = list(expression["TCGA"].columns)
    _, _, development = development_data()
    if not np.allclose(development.time, patients["TCGA"]["os_months"].to_numpy(dtype=float)):
        raise SystemExit("SurvStudio's prepared TCGA cohort is not in the order of the comparison's TCGA patients.")
    # Script 15's clinically adjusted test of every gene in every GEO cohort (the inputs are already z-scored within
    # the cohort after median imputation, as script 15 standardises).
    tests = pd.concat([adjusted_statistics(patients[cohort], expression[cohort]).assign(cohort=cohort) for cohort in GEO_COHORTS], ignore_index=True)
    result = {"survstudio": survstudio_version(), "code_hash": code_hash(), "r": r_versions(), "genes": len(genes),
              "candidate_cap": CANDIDATE_CAP, "cohorts": {cohort: {"n": len(patients[cohort]), "events": int(patients[cohort]["os_event"].sum())} for cohort in COHORTS}}
    result["SurvStudio"] = survstudio_section(patients)
    result["SurvStudio"]["tiers_on_comparison_genes"] = tier_check(tests, genes)
    print("SurvStudio checked", flush=True)
    result["P2"], p2_cohorts = p2_section(patients, development, tests)
    write_csv_atomic(p2_cohorts, RESULTS / "competitors_p2_cohorts.csv")
    print("P2 done", flush=True)
    result["P3"], p3_genes = p3_section(patients, expression, tests)
    write_csv_atomic(p3_genes, RESULTS / "competitors_p3_genes.csv")
    print("P3 done", flush=True)
    result["P1"], models, replay_table = mime_section(REAL / f"mime_cap{CANDIDATE_CAP}", patients, development, f"Mime, {CANDIDATE_CAP} candidates")
    write_csv_atomic(models, RESULTS / "competitors_mime_models.csv")
    write_csv_atomic(replay_table, RESULTS / "competitors_mime_replay.csv")
    sensitivity = REAL / f"mime_cap{SENSITIVITY_CAP}"
    if (sensitivity / "run.json").exists():
        result["P1_sensitivity"], models_500, replay_500 = mime_section(sensitivity, patients, development, f"Mime, {SENSITIVITY_CAP} candidates, StepCox-first models left out")
        write_csv_atomic(models_500, RESULTS / "competitors_mime_models_500.csv")
        write_csv_atomic(replay_500, RESULTS / "competitors_mime_replay_500.csv")
    print("P1 done", flush=True)
    result["splits"] = [";".join(chosen) for chosen, _ in splits(GEO_COHORTS)]
    result["seconds"] = round(time.time() - began)
    write_json(RESULTS / "competitors_real.json", clean(result))
    print(f"done in {result['seconds']}s", flush=True)


if __name__ == "__main__":
    main()

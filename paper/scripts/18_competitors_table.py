"""Comparison with the pipelines commonly used to publish prognostic signatures, the summary: what each approach would
report against what the independent cohorts show (experiment 1), and the rate of the claim a paper would make under
the plasmode null (experiment 2), from competitors_real.json and competitors_null.json.

Writes competitors_table.csv (experiment 1), competitors_null_table.csv (experiment 2) and competitors_table.json
(both, with the definitions) to results/.
"""

from __future__ import annotations

import pandas as pd

from common import RESULTS, read_result, survstudio_version, write_csv_atomic, write_json
from competitors import clean, code_hash


def interval(entry: dict | None) -> tuple:
    entry = entry or {}
    return entry.get("estimate"), entry.get("ci_lower"), entry.get("ci_upper")


def experiment_1(real: dict) -> list[dict]:
    rows = []
    survstudio = real["SurvStudio"]
    external = survstudio["external_within_cohort"]
    robust = survstudio["tiers_replication"].get("robust", {})
    suggestive = survstudio["tiers_replication"].get("suggestive", {})
    rows.append({
        "approach": "SurvStudio", "design": "case studies I-II: marker evaluation on TCGA, locked model applied unchanged to the seven cohorts",
        "claim": "robust markers (FWER <= 0.05 beyond age, sex and stage, stable over 200 subsamples); locked 10-marker model with its optimism-corrected C and left-out gain over the clinical model",
        "genes_claimed": len(survstudio["robust_genes"]), "training_c": survstudio["apparent_c"],
        "reported_c": survstudio["optimism_corrected_c"], "reported_c_meaning": "optimism-corrected C (apparent 0.749)",
        "reported_gain": survstudio["left_out_gain"], "reported_gain_meaning": "paired left-out gain over the clinical model in subsamples",
        "external_c": external["model_c"]["estimate"], "external_c_lower": external["model_c"]["ci_lower"], "external_c_upper": external["model_c"]["ci_upper"],
        "external_clinical_c": external["clinical_c"]["estimate"],
        "gain": external["delta_c"]["estimate"], "gain_lower": external["delta_c"]["ci_lower"], "gain_upper": external["delta_c"]["ci_upper"],
        "genes_replicated": robust.get("replicated"), "genes_evaluable": robust.get("evaluable"),
        "replicated_share": robust.get("rate"),
        "warns_about_optimism": "yes: reports the optimism-corrected C, the paired left-out gain over the clinical model and evidence tiers",
        "notes": f"suggestive tier: {suggestive.get('replicated')}/{suggestive.get('evaluable')} replicated; external C with markers rescaled within each cohort",
    })
    for key, label in (("P1", "P1 Mime"), ("P1_sensitivity", "P1 Mime (500 candidates)")):
        if key not in real:
            continue
        mime = real[key]
        usual = mime["all_seven"]
        rows.append({
            "approach": label, "design": f"all seven cohorts used to pick the best of {mime['models']} models (the usual published design; no sealed cohort)",
            "claim": f"best model {usual['winner']}: mean C {usual['reported_c']:.3f} over seven validation cohorts",
            "genes_claimed": usual.get("winner_genes"), "training_c": usual.get("training_c"),
            "reported_c": usual["reported_c"], "reported_c_meaning": "mean of Mime's C over the cohorts used for selection (direction set by each cohort's own Cox fit)",
            "external_c": mime["all_seven_pooled_honest_c"]["estimate"], "external_c_lower": mime["all_seven_pooled_honest_c"]["ci_lower"],
            "external_c_upper": mime["all_seven_pooled_honest_c"]["ci_upper"],
            "external_clinical_c": usual.get("clinical_c"),
            "gain": usual.get("gain"), "gain_lower": usual.get("gain_lower"), "gain_upper": usual.get("gain_upper"),
            "median_split_significant": f"{usual.get('selection_km_p05_count')} of 7 cohorts",
            "genes_replicated": (mime.get("all_seven_genes") or {}).get("replicated"),
            "genes_evaluable": (mime.get("all_seven_genes") or {}).get("evaluable"),
            "replicated_share": ((mime["all_seven_genes"]["replicated"] / mime["all_seven_genes"]["evaluable"])
                                 if (mime.get("all_seven_genes") or {}).get("evaluable") else None),
            "warns_about_optimism": "no",
            "notes": ("external C and gain here are in the cohorts used for selection, so they are not independent of it; the winner's "
                      "genes replicate by script 15's rule in the direction of their univariate Cox fit on TCGA"),
        })
        splits = mime["splits"]
        rows.append({
            "approach": label, "design": f"3 selection + 4 sealed cohorts, all {splits['count']} splits (winner: highest mean C over the 3)",
            "claim": "the winner's mean C over the 3 selection cohorts (median over splits)",
            "genes_claimed": None, "training_c": splits["training_c"]["median"],
            "reported_c": splits["reported_c"]["median"], "reported_c_meaning": "median over splits of the reported mean C (range {:.3f} to {:.3f})".format(splits["reported_c"]["min"], splits["reported_c"]["max"]),
            "external_c": splits["sealed_honest_pooled"]["median"], "external_c_lower": splits["sealed_honest_pooled"]["min"], "external_c_upper": splits["sealed_honest_pooled"]["max"],
            "gain": (splits.get("gain") or {}).get("median"), "gain_lower": (splits.get("gain") or {}).get("min"), "gain_upper": (splits.get("gain") or {}).get("max"),
            "optimism": splits["optimism_vs_sealed_pooled"]["median"],
            "warns_about_optimism": "no",
            "notes": ("external C: median (min to max over splits) of the random-effects pooled honest C in the 4 sealed cohorts; gain: median (min to max) of the "
                      f"pooled gain there, its interval excluding zero in {splits.get('gain_interval_excludes_zero', float('nan')):.0%} of splits; "
                      f"selection-cohort KM p < 0.05 in {splits['selection_km_p05_any']:.0%} of splits"),
        })
    p2 = real["P2"]
    external = p2["external_c_pooled"]
    gain = p2["gain"]["delta_c"]
    rows.append({
        "approach": "P2 uni-Cox -> LASSO -> Cox", "design": "univariate Cox p < 0.05 on TCGA, LASSO-Cox (10-fold CV, lambda.min), multivariable Cox; the seven cohorts for validation",
        "claim": f"{len(p2['genes_selected'])}-gene signature; training median-split log-rank p = {p2['training_median_split_p']:.2g}",
        "genes_claimed": len(p2["genes_selected"]), "training_c": p2["training_c"], "reported_c": p2["training_c"], "reported_c_meaning": "training (apparent) C",
        "median_split_p_training": p2["training_median_split_p"], "median_split_significant": f"{p2['external_median_split_p05']} of 7 cohorts",
        "external_c": external["estimate"], "external_c_lower": external["ci_lower"], "external_c_upper": external["ci_upper"],
        "external_clinical_c": p2["gain"]["clinical_c"]["estimate"],
        "gain": gain["estimate"], "gain_lower": gain["ci_lower"], "gain_upper": gain["ci_upper"],
        "genes_replicated": p2["genes_replicated"], "genes_evaluable": p2["genes_evaluable"],
        "replicated_share": p2["genes_replicated"] / p2["genes_evaluable"] if p2["genes_evaluable"] else None,
        "warns_about_optimism": "no",
        "notes": "gene replication by script 15's rule in the direction of each gene's multivariable coefficient",
    })
    p3 = real["P3"]
    for key, label, rule in (("claimed_p05", "P3 best cut-off (p < 0.05)", "uncorrected, KM Plotter's default"),
                             ("claimed_bonferroni", "P3 best cut-off (Bonferroni)", f"p < 0.05 / {p3['genes_scanned']}")):
        share = p3[key]
        rows.append({
            "approach": label, "design": f"minimum log-rank p over the cut-offs between the quartiles of each of {p3['genes_scanned']} genes on TCGA",
            "claim": f"{share['claimed']} prognostic genes ({rule})", "genes_claimed": share["claimed"],
            "genes_replicated": share["replicated"], "genes_evaluable": share["evaluable"], "replicated_share": share["rate"],
            "warns_about_optimism": "no",
            "notes": "replication: script 15's rule (clinically adjusted, pooled over the seven cohorts, the cut-off's direction)",
        })
    return rows


def experiment_2(null: dict) -> list[dict]:
    rows = []
    survstudio = null.get("SurvStudio", {})
    subsamples = survstudio.get("null_with_subsamples")
    rows.append({
        "approach": "SurvStudio", "claim": "at least one marker declared (FWER <= 0.05 beyond the clinical covariates)",
        "claim_rate": survstudio.get("fwer"), "claim_rate_mcse": survstudio.get("fwer_mcse"), "replicates": survstudio.get("replicates"),
        "mean_claimed": survstudio.get("false_per_replicate"),
        "notes": "script 06 (null with the near-constant filter, 2,000 genes per replicate); the null-with-subsamples scenario: "
                 + ("see null_with_subsamples" if subsamples else "not yet in results/"),
    })
    if "P1" in null:
        for design, entry in null["P1"].items():
            rows.append({
                "approach": "P1 Mime", "design": design,
                "claim": f"the winner's mean selection-cohort C >= {null['claim_c']}",
                "claim_rate": entry["reported_c_at_least_claim"]["rate"], "claim_rate_mcse": entry["reported_c_at_least_claim"]["mcse"],
                "replicates": entry["replicates"], "cases": entry["cases"],
                "reported_c_mean": entry["reported_c"]["mean"], "sealed_c_mean": (entry.get("sealed_honest_mean") or {}).get("mean"),
                "truth_c_mean": entry["truth_honest_c"]["mean"], "truth_gain_mean": entry["truth_gain"]["mean"],
                "km_p05_in_a_selection_cohort": entry["selection_km_p05_any"]["rate"],
                "claim_not_holding": (entry.get("claim_not_holding") or {}).get("rate"),
            })
    if "P2" in null:
        entry = null["P2"]
        rows.append({
            "approach": "P2 uni-Cox -> LASSO -> Cox", "claim": "at least one gene selected and training median-split log-rank p < 0.05",
            "claim_rate": entry["claim_training_p05"]["rate"], "claim_rate_mcse": entry["claim_training_p05"]["mcse"], "replicates": entry["replicates"],
            "mean_claimed": entry["selected_genes"]["mean"],
            "external_p05_any": (entry.get("external_p05_any") or {}).get("rate"),
            "external_p05_claimed_direction": (entry.get("external_p05_claimed_direction") or {}).get("rate"),
            "reported_c_mean": entry["training_c"]["mean"], "sealed_c_mean": entry["external_mean_c"]["mean"],
            "truth_c_mean": entry["truth_c"]["mean"], "truth_gain_mean": entry["truth_gain"]["mean"],
        })
    if "P3" in null:
        entry = null["P3"]
        rows.append({
            "approach": "P3 best cut-off", "claim": "at least one gene with best-cutoff log-rank p < 0.05",
            "claim_rate": entry["any_gene_p05"]["rate"], "claim_rate_mcse": entry["any_gene_p05"]["mcse"], "replicates": entry["replicates"],
            "mean_claimed": entry["genes_p05"]["mean"],
            "notes": (f"of {entry['genes']} genes; Bonferroni: at least one gene in {entry['any_gene_bonferroni']['rate']:.0%} of replicates "
                      f"(mean {entry['genes_bonferroni']['mean']:.1f}); median cut instead: mean {entry['median_cut_genes_p05']['mean']:.0f} genes at p < 0.05; "
                      f"claimed genes replicated by script 15's rule: {entry['claimed_replicated_share']:.1%}"),
        })
    return rows


def main() -> None:
    result = {"survstudio": survstudio_version(), "code_hash": code_hash()}
    real = read_result("competitors_real.json")
    table = pd.DataFrame(experiment_1(real))
    write_csv_atomic(table, RESULTS / "competitors_table.csv")
    result["experiment_1"] = table.to_dict("records")
    result["genes"] = real["genes"]
    try:
        null = read_result("competitors_null.json")
    except OSError:
        null = None
    if null:
        null_table = pd.DataFrame(experiment_2(null))
        write_csv_atomic(null_table, RESULTS / "competitors_null_table.csv")
        result["experiment_2"] = null_table.to_dict("records")
    write_json(RESULTS / "competitors_table.json", clean(result))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

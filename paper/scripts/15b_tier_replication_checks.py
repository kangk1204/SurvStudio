"""Evidence tiers against replication, the checks statistical reviewers ask for (post hoc, after script 15, whose rules
and results stay as they are): per case study (I, IV, V), from script 15's gene table (tier_replication_genes.csv),
the development marker tables (scripts 01, 07 and 09) and the development expression (common.development_data).

1. Size-matched comparison. Among the genes evaluable for replication (measured in at least two validation cohorts),
   the replication rate of the robust genes against that of the same number of genes with the smallest development
   p-values (added-value lens, ranked by the score chi-square, which orders them as the p-values do without their
   underflow; ties by name), and the same for the robust and suggestive genes together; with the overlap of each pair
   of sets and Wilson 95% intervals.
2. Logistic regression of replication (script 15's primary definition) on the development |z| (the square root of the
   added-value score chi-square), the number of validation cohorts measuring the gene, the gene's mean expression in
   the development cohort (the patients the evaluation analysed: endpoint and clinical covariates recorded, time
   above zero) and indicators of SurvStudio's tiers (robust, suggestive, and marginal only where it occurs; not
   supported is the reference). Reported: every coefficient as an odds ratio with its Wald 95% interval, a
   likelihood-ratio test of the tier indicators together against the model without them, whether the tiers add
   nothing detectable beyond |z| (that test's p >= 0.05 and every tier's interval including 1), and a plain statement
   of whether robust genes replicate more often than their |z|, cohorts and expression predict.
3. Empirical null. Per group of script 15, the share of evaluable genes whose pooled log hazard ratio has a 95% CI
   excluding zero in the direction opposite to development (the rate at which "replication" happens the wrong way),
   next to the share replicated in the development direction; Wilson 95% intervals.
Co-expressed genes share their evidence, so genes are not independent: every interval here is too narrow and the
likelihood-ratio p-value too small.
Writes tier_replication_checks.json and tier_replication_checks.csv (one row per case study, analysis and group or
term).
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd

from common import DEVELOPMENT, RESULTS, development_data, read_result, survstudio_version, write_csv_atomic, write_json

TIERS = importlib.import_module("15_tier_replication")
TIER_TERMS = ["robust", "suggestive", "marginal only"]
COVARIATES = ["abs_z", "cohorts", "mean_expression"]


def development_means(case: str) -> pd.Series:
    """Each gene's mean expression over the development patients the evaluation analysed."""
    spec = DEVELOPMENT[case]
    frame, genes = development_data(case)
    frame = frame.dropna(subset=[spec["time"], spec["event"], *spec["covariates"]])
    frame = frame[pd.to_numeric(frame[spec["time"]], errors="coerce") > 0]
    return frame[genes].apply(pd.to_numeric, errors="coerce").mean(axis=0)


def gene_table(case: str, genes: pd.DataFrame) -> pd.DataFrame:
    """Script 15's evaluable genes of one case study with their development |z|, tier and mean expression."""
    development = read_result(TIERS.CASES[case]["markers"], usecols=["marker", "tier", "added_value_chi2"])
    table = genes[(genes["case"] == case) & (genes["cohorts"] >= 2)].merge(development, left_on="gene", right_on="marker", how="left")
    if table["tier"].isna().any():
        raise SystemExit(f"case {case}: {int(table['tier'].isna().sum())} genes of tier_replication_genes.csv are not in {TIERS.CASES[case]['markers']}")
    table["abs_z"] = np.sqrt(table["added_value_chi2"].astype(float))
    table["mean_expression"] = table["gene"].map(development_means(case))
    table["replicated"] = table["replicated"].astype(bool)
    excludes = (table["ci_lower"] > 0) | (table["ci_upper"] < 0)
    table["opposite"] = excludes & (table["signed_log_hr"] < 0)
    return table


def share(count: int, total: int) -> dict:
    lower, upper = TIERS.wilson(count, total)
    return {"genes": int(total), "count": int(count), "share": count / total if total else None, "share_lower": lower, "share_upper": upper}


def size_matched(table: pd.DataFrame) -> dict:
    ranked = table.sort_values(["added_value_chi2", "gene"], ascending=[False, True])
    comparisons = {}
    for label, members in (("robust", {"robust"}), ("robust or suggestive", {"robust", "suggestive"})):
        chosen = table[table["tier"].isin(members)]
        top = ranked.head(len(chosen))
        overlap = int(len(set(chosen["gene"]) & set(top["gene"])))
        comparisons[label] = {
            "tier": share(int(chosen["replicated"].sum()), len(chosen)),
            "top_by_p_value": share(int(top["replicated"].sum()), len(top)),
            "overlap": overlap,
            "smallest_chi2_in_top": float(top["added_value_chi2"].min()) if len(top) else None,
            "statement": (f"The {len(chosen)} {label} genes are the {len(chosen)} evaluable genes with the smallest development p-values, so "
                          "the two rates are the same by construction." if overlap == len(chosen) else
                          f"{overlap} of the {len(chosen)} {label} genes are among the {len(chosen)} evaluable genes with the smallest "
                          "development p-values."),
        }
    return comparisons


def logistic(table: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    from scipy import stats

    data = table.dropna(subset=COVARIATES)
    terms = [term for term in TIER_TERMS if (data["tier"] == term).any()]
    design = data[COVARIATES].astype(float).assign(**{term: (data["tier"] == term).astype(float) for term in terms})
    outcome = data["replicated"].astype(float)
    fits = {}
    for name, columns in (("with_tiers", [*COVARIATES, *terms]), ("without_tiers", COVARIATES)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(outcome, sm.add_constant(design[columns])).fit(disp=0, maxiter=200)
        fits[name] = fit
    full, reduced = fits["with_tiers"], fits["without_tiers"]
    interval = full.conf_int()
    coefficients = {
        term: {"odds_ratio": float(np.exp(full.params[term])), "or_lower": float(np.exp(interval.loc[term, 0])),
               "or_upper": float(np.exp(interval.loc[term, 1])), "p_value": float(full.pvalues[term]), "genes": int((data["tier"] == term).sum())
               if term in terms else int(len(data))}
        for term in [*COVARIATES, *terms]
    }
    statistic = float(2.0 * (full.llf - reduced.llf))
    p_value = float(stats.chi2.sf(statistic, len(terms))) if terms else None
    converged = bool(full.mle_retvals.get("converged")) and bool(reduced.mle_retvals.get("converged"))
    excludes_one = {term: not coefficients[term]["or_lower"] <= 1.0 <= coefficients[term]["or_upper"] for term in terms}
    nothing = p_value is not None and p_value >= 0.05 and not any(excludes_one.values())

    def odds(term: str) -> str:
        value = coefficients[term]
        return f"odds ratio {value['odds_ratio']:.2f}, {value['or_lower']:.2f} to {value['or_upper']:.2f}"

    predicted = "their development |z|, the number of cohorts measuring them and their mean expression predict"
    if "robust" in terms:
        how = "more often" if coefficients["robust"]["odds_ratio"] > 1 else "less often"
        verdict = f"Robust genes replicate {how if excludes_one['robust'] else 'no more often'} than {predicted} ({odds('robust')}); "
    else:
        verdict = "No robust gene is evaluable; "
    others = [f"{term}: {odds(term)}" for term in terms if term != "robust" and excludes_one[term]]
    verdict += (f"all tier indicators together: likelihood-ratio p = {p_value:.3g}" + (f" ({'; '.join(others)})" if others else "") + "."
                if terms else "no tier indicator.")
    return {"genes": int(len(data)), "replicated": int(outcome.sum()), "left_out_missing_covariates": int(len(table) - len(data)),
            "converged": converged, "coefficients": coefficients, "likelihood_ratio": statistic, "likelihood_ratio_df": len(terms),
            "likelihood_ratio_p": p_value, "tiers_add_nothing_beyond_z": bool(nothing), "statement": verdict}


def empirical_null(table: pd.DataFrame) -> dict:
    groups = {}
    for group in TIERS.GROUPS:
        part = table[table["group"] == group]
        if len(part):
            groups[group] = {"replicated": share(int(part["replicated"].sum()), len(part)),
                             "opposite_direction": share(int(part["opposite"].sum()), len(part))}
    return groups


def csv_rows(case: str, result: dict) -> list[dict]:
    rows = []
    for label, value in result["size_matched"].items():
        for side in ("tier", "top_by_p_value"):
            rows.append({"case": case, "analysis": "size-matched", "item": f"{label}: {'tier' if side == 'tier' else 'top genes by development p'}",
                         **value[side], "note": value["statement"]})
    for term, value in result["logistic"]["coefficients"].items():
        rows.append({"case": case, "analysis": "logistic", "item": term, "genes": value["genes"], "odds_ratio": value["odds_ratio"],
                     "or_lower": value["or_lower"], "or_upper": value["or_upper"], "p_value": value["p_value"]})
    rows.append({"case": case, "analysis": "logistic", "item": "tiers together (likelihood ratio)", "genes": result["logistic"]["genes"],
                 "p_value": result["logistic"]["likelihood_ratio_p"], "note": result["logistic"]["statement"]})
    for group, value in result["empirical_null"].items():
        for kind in ("replicated", "opposite_direction"):
            rows.append({"case": case, "analysis": "empirical null", "item": f"{group}: {kind.replace('_', ' ')}", **value[kind]})
    return rows


def main() -> None:
    genes = read_result("tier_replication_genes.csv")
    summary = {"survstudio": survstudio_version(), "cases": {}}
    rows = []
    for case in TIERS.CASES:
        table = gene_table(case, genes)
        result = {"label": TIERS.CASES[case]["label"], "evaluable_genes": int(len(table)), "size_matched": size_matched(table),
                  "logistic": logistic(table), "empirical_null": empirical_null(table)}
        summary["cases"][case] = result
        rows.extend(csv_rows(case, result))
        print(case, result["logistic"]["statement"], flush=True)
    columns = ["case", "analysis", "item", "genes", "count", "share", "share_lower", "share_upper", "odds_ratio", "or_lower", "or_upper", "p_value", "note"]
    write_csv_atomic(pd.DataFrame(rows).reindex(columns=columns), RESULTS / "tier_replication_checks.csv")
    write_json(RESULTS / "tier_replication_checks.json", summary)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 70):
        print(pd.DataFrame(rows).reindex(columns=columns).drop(columns="note").round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Evidence tiers against replication: do the genes SurvStudio calls robust replicate in the external cohorts more
often than the genes it calls suggestive, and those more often than genes with only a nominal p-value?

Rules fixed before the run (later changes moved the choice of the breast cohorts into the screen that scripts 08 and
10 share and changed the shared loaders, not these rules):
1. Genes and groups: every gene evaluated in the development runs of case studies I (TCGA-LUAD, overall survival),
   IV (METABRIC, overall survival) and V (METABRIC ER-positive, recurrence), grouped by SurvStudio's tier on the
   added-value lens: robust, suggestive, marginal only (IV only), and "not supported" split into nominal only
   (clinically adjusted p < 0.05, the usual reporting threshold) and no evidence (the rest).
2. External cohorts, patients, endpoints and clinical covariates: those of each case study's validation (scripts 03,
   08 and 10), with duplicates removed as there; the patient and event counts are checked against their results.
3. Test: in each cohort, SurvStudio's clinically adjusted Cox score test (Efron ties) of the gene standardised within
   the cohort; the one-step log hazard ratio per SD with standard error 1/sqrt(information). A cohort measures a gene
   when at most 20% of its values are missing (the rest take the cohort median) and it varies.
4. Replication (primary): over the cohorts measuring the gene (at least two), the random-effects pooled log hazard
   ratio has the development direction and its 95% CI excludes zero. Secondary: the share of cohort tests in the
   development direction, and the pooled log hazard ratio signed by the development direction.
5. Reported per case study and group, with Wilson 95% intervals; nothing else is tested. (The Wilson intervals treat
   genes as independent, which co-expressed genes are not, so they are too narrow.)
Writes tier_replication_genes.csv and tier_replication.json.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BREAST_CATEGORICAL,
    BREAST_COVARIATES,
    BREAST_ER_CATEGORICAL,
    BREAST_ER_COVARIATES,
    BREAST_VALIDATION,
    CATEGORICAL,
    COVARIATES,
    ENDPOINT_COLUMNS,
    GEO_COHORTS,
    LUAD,
    RESULTS,
    load_breast_cohort,
    random_effects,
    read_result,
    screen_validation_cohorts,
    survstudio_version,
    write_csv_atomic,
    write_json,
)
from survival_toolkit.marker_screen import CoxScoreScreen, fit_cox_null

MAX_MISSING = 0.2
GROUPS = ["robust", "suggestive", "marginal only", "nominal only", "no evidence"]
CASES = {
    "I": {"label": "TCGA-LUAD, overall survival", "markers": "tcga_markers.csv", "validation": "external_validation.csv"},
    "IV": {"label": "METABRIC, overall survival", "markers": "breast_markers.csv", "validation": "breast_external_validation.csv",
           "locked": "breast_locked_model.json"},
    "V": {"label": "METABRIC ER-positive, recurrence", "markers": "breast_er_markers.csv", "validation": "breast_er_external_validation.csv",
          "locked": "breast_er_locked_model.json"},
}


def luad_cohorts():
    """Case study I's seven GEO cohorts, as script 03 builds them."""
    truth = pd.read_csv(LUAD / "harmonized_clinical.csv")
    for cohort in GEO_COHORTS:
        table = truth[(truth["cohort"] == cohort) & truth["exclusion"].isna()]
        clinical = pd.DataFrame({
            "patient_id": table["sample_id"].astype(str),
            "time": table["os_time"] * 12.0,
            "event": table["os_event"],
            "age": table["age"],
            "sex": table["sex"].map({"M": "Male", "F": "Female"}),
            "stage_group": table["stage_group"].map(lambda value: f"Stage {value}" if isinstance(value, str) else np.nan),
        })
        expression = pd.read_csv(LUAD / cohort / "expression_genes.csv.gz").set_index("sample_id")
        expression.index = expression.index.astype(str)
        clinical = clinical[clinical["patient_id"].isin(expression.index)]
        yield cohort, clinical, expression, COVARIATES, CATEGORICAL


def breast_cohorts(case: str):
    """Case study IV's or V's external cohorts: the ones scripts 08 and 10 used, with the same patients (the same
    screen, common.screen_validation_cohorts, run with the same locked model), with every gene."""
    spec = BREAST_VALIDATION[case]
    covariates, categorical = (BREAST_COVARIATES, BREAST_CATEGORICAL) if case == "IV" else (BREAST_ER_COVARIATES, BREAST_ER_CATEGORICAL)
    time_column, event_column = ENDPOINT_COLUMNS[spec["endpoint"]]
    recipe = read_result(CASES[case]["locked"])
    _, used = screen_validation_cohorts(case, recipe)
    for cohort, patients in used.items():
        frame, genes = load_breast_cohort(cohort, endpoint=spec["endpoint"], er_positive=spec["er_positive"])
        frame = frame[frame["patient_id"].isin(set(patients["patient_id"]))]
        clinical = frame[["patient_id", time_column, event_column, *covariates]].rename(columns={time_column: "time", event_column: "event"})
        yield cohort, clinical, frame.set_index("patient_id")[genes], covariates, categorical


def cohort_statistics(clinical: pd.DataFrame, expression: pd.DataFrame, covariates: list[str], categorical: list[str], genes: list[str]) -> pd.DataFrame:
    """One-step log hazard ratio per SD and its standard error for each measured gene, adjusted for the covariates."""
    clinical = clinical.dropna(subset=["time", "event", *covariates])
    clinical = clinical[pd.to_numeric(clinical["time"], errors="coerce") > 0].reset_index(drop=True)
    design = pd.get_dummies(clinical[covariates], columns=categorical, drop_first=True, dtype=float).to_numpy(dtype=float)
    design = design[:, np.ptp(design, axis=0) > 0]
    time = clinical["time"].to_numpy(dtype=float)
    event = clinical["event"].to_numpy(dtype=float).astype(int)
    block = expression.reindex(index=clinical["patient_id"], columns=[gene for gene in genes if gene in expression.columns])
    # A copy: under pandas' copy-on-write, to_numpy can return a read-only view of the frame.
    values = block.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing = np.isnan(values).mean(axis=0)
    medians = np.nanmedian(np.where(np.isnan(values).all(axis=0), 0.0, values), axis=0)
    values = np.where(np.isnan(values), medians, values)
    spread = values.std(axis=0, ddof=1)
    measured = (missing <= MAX_MISSING) & (spread > 0)
    standardised = (values[:, measured] - values[:, measured].mean(axis=0)) / spread[measured]
    null = fit_cox_null(time, event, design)
    stats = CoxScoreScreen(time, event, null=null, Z=design).statistics(standardised)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = 1.0 / np.sqrt(stats.information)
    table = pd.DataFrame({"gene": np.asarray(block.columns)[measured], "log_hr": stats.beta_one_step, "se": se})
    # A gene collinear with the clinical covariates in this cohort has no estimate.
    table = table[np.isfinite(table["log_hr"]) & np.isfinite(table["se"]) & (table["se"] > 0)]
    return table.assign(n=len(time), events=int(event.sum()))


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    p, z = successes / total, 1.959963984540054
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(centre - half), float(centre + half)]


def group_of(table: pd.DataFrame) -> pd.Series:
    nominal = table["added_value_p_value"] < 0.05
    return pd.Series(np.select(
        [table["tier"] == "robust", table["tier"] == "suggestive", table["tier"] == "marginal only", nominal],
        ["robust", "suggestive", "marginal only", "nominal only"], default="no evidence"), index=table.index)


def gene_record(case: str, gene: str, part: pd.DataFrame) -> dict:
    """One gene's replication from its cohort tests (``part``: log_hr, se, same_direction, and its group and
    development direction): with at least two cohorts, the random-effects pooled log hazard ratio, signed by the
    development direction, and whether it replicated (same direction, 95% CI excluding zero)."""
    record = {"case": case, "gene": gene, "group": part["group"].iat[0], "direction": int(part["direction"].iat[0]),
              "cohorts": len(part), "same_direction_share": float(part["same_direction"].mean())}
    if len(part) >= 2:
        pooled = random_effects(part["log_hr"].to_numpy(), part["se"].to_numpy())
        signed = pooled["estimate"] * record["direction"]
        record.update(pooled_log_hr=pooled["estimate"], ci_lower=pooled["ci_lower"], ci_upper=pooled["ci_upper"], tau2=pooled["tau2"],
                      signed_log_hr=signed,
                      replicated=bool(signed > 0 and (pooled["ci_lower"] > 0 or pooled["ci_upper"] < 0)))
    return record


def main() -> None:
    rows, summary = [], {"survstudio": survstudio_version(), "cases": {}}
    for case, spec in CASES.items():
        development = read_result(spec["markers"])
        development = development[np.isfinite(development["added_value_log_hr_one_step"])].copy()
        development["group"] = group_of(development)
        development["direction"] = np.sign(development["added_value_log_hr_one_step"])
        genes = development["marker"].tolist()
        per_cohort, checks = [], []
        cohorts = luad_cohorts() if case == "I" else breast_cohorts(case)
        for cohort, clinical, expression, covariates, categorical in cohorts:
            table = cohort_statistics(clinical, expression, covariates, categorical, genes)
            checks.append({"cohort": cohort, "n": int(table["n"].iat[0]), "events": int(table["events"].iat[0]), "genes_measured": len(table)})
            per_cohort.append(table.assign(cohort=cohort))
            print(case, checks[-1], flush=True)
        validation = read_result(spec["validation"])
        if "scaling" in validation:
            validation = validation[validation["scaling"] == "within_cohort"]
        expected = validation.set_index("cohort")[["n", "events"]].astype(int).to_dict("index")
        mismatched = [check for check in checks if expected.get(check["cohort"]) != {"n": check["n"], "events": check["events"]}]
        if mismatched or len(checks) != len(expected):
            raise SystemExit(f"case {case}: patients differ from the validation run: {mismatched} (expected {expected})")

        tests = pd.concat(per_cohort, ignore_index=True).merge(development[["marker", "group", "direction"]], left_on="gene", right_on="marker")
        tests["same_direction"] = np.sign(tests["log_hr"]) == tests["direction"]
        rows.extend(gene_record(case, gene, part) for gene, part in tests.groupby("gene", sort=False))

        genes_table = pd.DataFrame([row for row in rows if row["case"] == case])
        groups = {}
        for group in GROUPS:
            part = genes_table[genes_table["group"] == group]
            evaluable = part[part["cohorts"] >= 2]
            replicated = int(evaluable["replicated"].sum()) if len(evaluable) else 0
            groups[group] = {
                "genes": int((development["group"] == group).sum()), "evaluable": int(len(evaluable)), "replicated": replicated,
                "rate": replicated / len(evaluable) if len(evaluable) else None, "rate_ci": wilson(replicated, len(evaluable)),
                "same_direction_share": float(tests.loc[tests["group"] == group, "same_direction"].mean()) if (tests["group"] == group).any() else None,
                "median_signed_log_hr": float(evaluable["signed_log_hr"].median()) if len(evaluable) else None,
            }
        summary["cases"][case] = {"label": spec["label"], "cohorts": checks, "groups": {key: value for key, value in groups.items() if value["genes"]}}

    write_csv_atomic(pd.DataFrame(rows), RESULTS / "tier_replication_genes.csv")
    write_json(RESULTS / "tier_replication.json", summary)
    for case, result in summary["cases"].items():
        print(f"\ncase {case}: {result['label']}")
        for group, value in result["groups"].items():
            low, high = value["rate_ci"]
            rate = "" if value["rate"] is None else f"{value['rate']:.1%} ({low:.1%} to {high:.1%})"
            print(f"  {group:14s} {value['replicated']:5d} of {value['evaluable']:5d} replicated {rate:28s} same direction {value['same_direction_share']:.1%}")


if __name__ == "__main__":
    main()

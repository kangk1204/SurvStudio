"""Summarise the plasmode simulation (06_simulation.py): error control, power and the accuracy of the C-index
estimates SurvStudio reports.

Per scenario: the family-wise error rate (any false marker with family-wise p <= 0.05) with its Monte Carlo
standard error. A false marker is one unlinked to the true markers (06_simulation.LINKED_R); under the null
every gene is. Under the alternative: the share of the five true markers found (family-wise and robust tier),
linked and false markers per replicate, and each C-index estimate minus the locked model's C in 3,000 new
patients from the same design, and SurvStudio's paired left-out gain (signature_gain_left_out) minus the model's
gain over the clinical model in the new patients. Writes simulation_summary.csv and simulation_summary.json.

Stops, rather than summarising, when the replicates do not all come from one SurvStudio commit, one design and one
version of the simulation's code (the ones in simulation_settings.json), when a scenario lacks replicates, or when a
value the summary uses is missing: under the null the event and discovery counts, under the alternative also the
linked-gene counts, and for every replicate scored in new patients all six C-indices and the paired gain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, STAMP_DTYPES, read_result, write_csv_atomic, write_json

TRUE_MARKERS = 5
ESTIMATES = {"apparent_c": "apparent", "corrected_c": "optimism-corrected", "left_out_c": "left-out patients"}
NEEDED = ["beta", "max_mode_fraction", "events", "tested", "fwer_false"]
NEEDED_ALTERNATIVE = ["fwer_true", "robust_true", "linked_genes", "fwer_linked", "fwer_unlinked", "robust_linked", "robust_unlinked"]
NEEDED_SCORED = ["new_patients_c", "new_patients_clinical_c", "apparent_c", "corrected_c", "left_out_c", "clinical_left_out_c", "left_out_gain"]
# Replicate column -> the settings value every replicate must carry.
STAMPED = {"survstudio_commit": lambda settings: settings["survstudio"]["commit"], "design_hash": lambda settings: settings.get("design_hash"),
           "code_hash": lambda settings: settings.get("code_hash")}


def checked(replicates: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """The replicates, once they are shown to be one complete run of the design in simulation_settings.json. Read the
    replicates with STAMP_DTYPES, so a stamp that looks like a number stays the text it was written as."""
    problems = []
    for column, expected in STAMPED.items():
        expected = expected(settings)
        found = sorted(replicates[column].astype(str).unique()) if column in replicates else ["<no column>"]
        if expected is None or found != [str(expected)]:
            problems.append(f"{column} {found}, settings {expected}")
    design = {name: spec["replicates"] for name, spec in settings["scenarios"].items()}
    counts = replicates.groupby("scenario")["replicate"].agg(["size", "nunique"])
    for name, wanted in design.items():
        size, unique = counts.loc[name] if name in counts.index else (0, 0)
        if size != wanted or unique != wanted:
            problems.append(f"{name}: {size} rows for {unique} of {wanted} replicates")
    problems += [f"{name}: not in the design" for name in counts.index if name not in design]
    fine = replicates[replicates["error"].isna()]
    alternative = fine["beta"] > 0
    scored = alternative & fine["new_patients_c"].notna()
    for columns, rows in ((NEEDED, fine.index), (NEEDED_ALTERNATIVE, fine.index[alternative]), (NEEDED_SCORED, fine.index[scored])):
        if len(rows) == 0:
            continue
        problems += [f"no column {column}" for column in columns if column not in fine]
        missing = fine.loc[rows, [column for column in columns if column in fine]].isna().sum()
        problems += [f"{count} missing {column}" for column, count in missing.items() if count]
    if problems:
        raise SystemExit("simulation_replicates.csv is not one complete run of the design: " + "; ".join(problems))
    return replicates


def summarise(replicates: pd.DataFrame, settings: dict) -> list[dict]:
    rows = []
    for scenario in settings["scenarios"]:
        everything = replicates[replicates["scenario"] == scenario]
        # A replicate whose evaluation failed is reported as such and left out of the rates.
        failed = int(everything["error"].notna().sum())
        part = everything[everything["error"].isna()]
        count = len(part)
        beta = float(settings["scenarios"][scenario]["beta"])
        # Under the null every discovery is false; under the alternative only the unlinked ones are.
        false = part["fwer_false"] if beta == 0 else part["fwer_unlinked"]
        fwer = float((false > 0).mean())
        row = {
            "scenario": scenario, "beta": beta, "filter": bool(settings["scenarios"][scenario]["max_mode_fraction"] < 1.0),
            "replicates": count, "failed": failed, "events_mean": float(part["events"].mean()), "tested_mean": float(part["tested"].mean()),
            "fwer": fwer, "fwer_mcse": float(np.sqrt(fwer * (1 - fwer) / count)),
            "false_per_replicate": float(false.mean()),
        }
        if beta > 0:
            found = part["fwer_true"] + part["fwer_false"]
            row.update({
                "linked_genes_mean": float(part["linked_genes"].mean()),
                "linked_found_per_replicate": float(part["fwer_linked"].mean()),
                "power_fwer": float(part["fwer_true"].mean() / TRUE_MARKERS),
                "power_robust": float(part["robust_true"].mean() / TRUE_MARKERS),
                "robust_linked_per_replicate": float(part["robust_linked"].mean()),
                "robust_false_per_replicate": float(part["robust_unlinked"].mean()),
                "robust_fwer": float((part["robust_unlinked"] > 0).mean()),
                "any_true_found": float((part["fwer_true"] > 0).mean()),
                "false_discovery_proportion": float(np.where(found > 0, part["fwer_unlinked"] / found.where(found > 0, 1), 0).mean()),
            })
            scored = part.dropna(subset=["new_patients_c"])
            row["replicates_scored"] = len(scored)
            row["new_patients_c_mean"] = float(scored["new_patients_c"].mean())
            row["new_patients_clinical_c_mean"] = float(scored["new_patients_clinical_c"].mean())
            for column, _ in ESTIMATES.items():
                error = scored[column] - scored["new_patients_c"]
                row[f"{column}_bias"] = float(error.mean())
                row[f"{column}_rmse"] = float(np.sqrt((error**2).mean()))
            # Both gains are paired: SurvStudio's on the same left-out patients, the true one on the same new patients.
            gain = scored["new_patients_c"] - scored["new_patients_clinical_c"]
            row["true_gain_mean"] = float(gain.mean())
            row["left_out_gain_bias"] = float((scored["left_out_gain"] - gain).mean())
        rows.append(row)
    return rows


def main() -> None:
    settings = read_result("simulation_settings.json")
    replicates = checked(read_result("simulation_replicates.csv", dtype=STAMP_DTYPES), settings)
    rows = summarise(replicates, settings)
    summary = pd.DataFrame(rows)
    write_csv_atomic(summary, RESULTS / "simulation_summary.csv")
    write_json(RESULTS / "simulation_summary.json", {"scenarios": rows, "total_replicates": int(len(replicates)),
                                                      "survstudio": settings["survstudio"], "design_hash": settings["design_hash"],
                                                      "code_hash": settings["code_hash"]})
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

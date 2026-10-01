"""Checks of the comparison's own code (competitors.py and the R pipelines), each of which must pass:

1. the selection replay on a tiny synthetic C matrix: the split enumeration, the winner (ties to the first model, a
   model without a C cannot win), the reported number, the sealed means and the pooled sealed C against
   common.random_effects;
2. the log-rank test against a direct per-death-time computation and, when R is there, against survival::survdiff;
3. the best-cutoff scan against a scan of every cut-off one at a time;
4. the bootstrap intervals of Harrell's C against SurvStudio's validate_locked_recipe (same draws), and the
   risk-score recipe's clinical-only model against a direct Cox fit;
5. P2 (run_p2.R) on a toy dataset against a direct computation: univariate Cox by the formula interface, cv.glmnet
   and coxph called directly with the same seed, the risk score recomputed, and the final Cox fit against
   SurvStudio's own Cox fit (needs RSCRIPT, the competitors' R);
6. SurvStudio's pooled numbers of case study II, recomputed by competitors.pooled_c_gain from
   external_validation.csv, equal the ones script 03 wrote to external_pooled.json (the C-index of the locked and of
   the clinical-only model and their difference, each with every interval), when results/ holds them.
Usage: python 18_competitors_checks.py   (writes results/competitors_checks.json)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, random_effects, survstudio_version, write_json
from competitors import (
    R_SCRIPTS,
    RiskSets,
    best_cutoff_scan,
    bootstrap_c,
    choose_winner,
    code_hash,
    harrell_c,
    logrank,
    pooled_c_gain,
    replay,
    risk_score_recipe,
    splits,
)
from survival_toolkit.marker_evaluation import prepare_marker_cohort, validate_locked_recipe
from survival_toolkit.marker_screen import fit_cox

RSCRIPT = os.environ.get("RSCRIPT") or shutil.which("Rscript")
report: dict = {"survstudio": survstudio_version(), "code_hash": code_hash(), "checks": {}}


def passed(name: str, detail: object) -> None:
    report["checks"][name] = {"passed": True, "detail": detail}
    print(f"ok  {name}: {detail}", flush=True)


def check_replay() -> None:
    cohorts = ["A", "B", "C", "D"]
    reported = pd.DataFrame([[0.60, 0.70, 0.50, 0.55], [0.65, 0.60, 0.62, 0.50], [0.70, 0.55, 0.58, 0.66]],
                            index=["m1", "m2", "m3"], columns=cohorts)
    honest = pd.DataFrame([[0.55, 0.60, 0.48, 0.52], [0.58, 0.54, 0.57, 0.49], [0.61, 0.50, 0.53, 0.60]], index=reported.index, columns=cohorts)
    width = pd.DataFrame([[0.10, 0.12, 0.08, 0.20]] * 3, index=reported.index, columns=cohorts)
    enumerated = splits(cohorts, 2)
    assert [chosen for chosen, _ in enumerated] == [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")], enumerated
    assert all(set(chosen) | set(sealed) == set(cohorts) and not set(chosen) & set(sealed) for chosen, sealed in enumerated)
    assert len(splits(["a", "b", "c", "d", "e", "f", "g"], 3)) == 35
    table = replay(reported, honest, honest - width / 2, honest + width / 2, training=pd.Series([0.9, 0.8, 0.7], index=reported.index), selection=2)
    expected = {"A;B": ("m1", 0.65), "A;C": ("m3", 0.64), "A;D": ("m3", 0.68), "B;C": ("m2", 0.61), "B;D": ("m1", 0.625), "C;D": ("m3", 0.62)}
    for _, row in table.iterrows():
        winner, value = expected[row["selection"]]
        assert row["winner"] == winner and abs(row["reported_c"] - value) < 1e-12, row
        sealed = row["sealed"].split(";")
        assert abs(row["sealed_honest_mean"] - honest.loc[winner, sealed].mean()) < 1e-12
        assert abs(row["sealed_reported_mean"] - reported.loc[winner, sealed].mean()) < 1e-12
        assert abs(row["training_c"] - {"m1": 0.9, "m2": 0.8, "m3": 0.7}[winner]) < 1e-12
        direct = random_effects(honest.loc[winner, sealed].to_numpy(), width.loc[winner, sealed].to_numpy() / 3.92)
        assert abs(row["sealed_honest_pooled"] - direct["estimate"]) < 1e-12 and abs(row["sealed_honest_pooled_lower"] - direct["ci_lower"]) < 1e-12
        assert abs(row["optimism_vs_sealed_mean"] - (value - honest.loc[winner, sealed].mean())) < 1e-12
    # Ties go to the first model in the tool's order; a model without a C in a selection cohort cannot win.
    tied = pd.DataFrame([[0.6, 0.7], [0.7, 0.6], [np.nan, 0.9]], index=["first", "second", "missing"], columns=["X", "Y"])
    winner, value = choose_winner(tied, ["X", "Y"])
    assert winner == "first" and abs(value - 0.65) < 1e-12, (winner, value)
    assert choose_winner(tied, ["Y"])[0] == "missing"
    passed("replay", f"{len(table)} splits of 4 cohorts, winners {table['winner'].tolist()}, ties and missing C")


def toy_survival(n: int, rng: np.random.Generator, genes: int = 30, effects: int = 5) -> pd.DataFrame:
    x = rng.normal(size=(n, genes))
    beta = np.zeros(genes)
    beta[:effects] = rng.choice([-0.6, 0.6], effects)
    death = rng.weibull(1.3, n) * 900 / np.exp(x @ beta / 1.3)
    censor = rng.uniform(100, 2500, n)
    frame = pd.DataFrame(x, columns=[f"G{index + 1}" for index in range(genes)])
    frame.insert(0, "OS", (death <= censor).astype(int))
    # Whole days, so that deaths tie as in real follow-up.
    frame.insert(0, "OS.time", np.ceil(np.minimum(death, censor)))
    frame.insert(0, "ID", [f"P{index + 1}" for index in range(n)])
    return frame


def check_logrank() -> None:
    rng = np.random.default_rng(1)
    frame = toy_survival(200, rng)
    time, event = frame["OS.time"].to_numpy(), frame["OS"].to_numpy()
    groups = (rng.random((5, time.size)) < 0.4).astype(float)
    chi2, p_values, hazard = logrank(time, event, groups)
    # Direct: at each death time, observed minus expected deaths of group 1 and the hypergeometric variance.
    for row, group in enumerate(groups):
        observed = expected = variance = 0.0
        for moment in np.unique(time[event == 1]):
            at_risk = time >= moment
            n_all, n_one = at_risk.sum(), (at_risk & (group == 1)).sum()
            d_all = ((time == moment) & (event == 1)).sum()
            observed += ((time == moment) & (event == 1) & (group == 1)).sum()
            expected += d_all * n_one / n_all
            if n_all > 1:
                variance += d_all * (n_all - d_all) * n_one * (n_all - n_one) / (n_all**2 * (n_all - 1))
        assert abs(chi2[row] - (observed - expected) ** 2 / variance) < 1e-9, (chi2[row], (observed - expected) ** 2 / variance)
    detail = {"direct_max_difference": 0.0}
    if RSCRIPT:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.csv"
            pd.DataFrame({"time": time, "event": event, **{f"g{row}": groups[row] for row in range(len(groups))}}).to_csv(path, index=False)
            script = ("library(survival); d <- read.csv('%s'); cat(sprintf('%%.15g', sapply(0:%d, function(i) survdiff(Surv(d$time, d$event) ~ d[[paste0('g', i)]])$chisq)))"
                      % (path.as_posix(), len(groups) - 1))
            output = subprocess.run([RSCRIPT, "-e", script], capture_output=True, text=True, check=True).stdout.split()
            r_chi2 = np.array([float(value) for value in output])
            assert np.allclose(chi2, r_chi2, rtol=1e-9, atol=1e-9), (chi2, r_chi2)
            detail["survdiff_max_difference"] = float(np.max(np.abs(chi2 - r_chi2)))
    passed("logrank", detail)


def check_best_cutoff() -> None:
    rng = np.random.default_rng(2)
    frame = toy_survival(150, rng, genes=12)
    time, event = frame["OS.time"].to_numpy(), frame["OS"].to_numpy()
    values = frame.drop(columns=["ID", "OS.time", "OS"]).to_numpy(dtype=float, copy=True)
    values[:, 3] = np.round(values[:, 3], 1)  # tied values
    values[:40, 4] = 0.0  # a spike of zeros, as in RNA-seq
    scan = best_cutoff_scan(time, event, values, batch=5)
    sets = RiskSets(time, event)
    for column in range(values.shape[1]):
        x = values[:, column]
        low, high = np.quantile(x, [0.25, 0.75])
        best = (np.inf, None)
        for cut in np.unique(x[(x >= low) & (x <= high)]):
            if cut >= x.max():
                continue
            _, p_value, _ = sets.test((x > cut).astype(float)[None, :])
            if p_value[0] < best[0]:
                best = (p_value[0], cut)
        assert abs(scan.loc[column, "best_p"] - best[0]) < 1e-12 and scan.loc[column, "cutoff"] == best[1], (column, scan.loc[column].to_dict(), best)
        _, median_p, _ = logrank(time, event, (x > np.quantile(x, 0.5)).astype(float)[None, :])
        assert abs(scan.loc[column, "median_p"] - median_p[0]) < 1e-12
    passed("best_cutoff", f"{values.shape[1]} genes, every cut-off between the quartiles, ties and a spike of zeros")


def check_bootstrap_and_recipe() -> None:
    rng = np.random.default_rng(3)
    n = 300
    clinical = pd.DataFrame({"age": rng.normal(65, 9, n).round(), "sex": rng.choice(["Male", "Female"], n),
                             "stage_group": rng.choice(["Stage I", "Stage II", "Stage III", "Stage IV"], n, p=[0.5, 0.25, 0.18, 0.07])})
    score = rng.normal(size=n)
    hazard = 0.02 * (clinical["age"] - 65) + clinical["stage_group"].map({"Stage I": 0, "Stage II": 0.8, "Stage III": 1.2, "Stage IV": 1.3}) + 0.4 * score
    death = rng.exponential(40 / np.exp(hazard))
    censor = rng.uniform(5, 120, n)
    frame = clinical.assign(os_months=np.minimum(death, censor).round(2) + 0.01, os_event=(death <= censor).astype(int), risk_score=score,
                            patient_id=[f"T{index}" for index in range(n)])
    development, external = frame.iloc[:180].reset_index(drop=True), frame.iloc[180:].reset_index(drop=True)
    cohort = prepare_marker_cohort(development, time_column="os_months", event_column="os_event", marker_columns=["risk_score"],
                                   clinical_columns=["age", "sex", "stage_group"], categorical_clinical=["sex", "stage_group"], event_positive_value=1)
    recipe = risk_score_recipe(cohort, development["risk_score"].to_numpy(), "toy")
    direct = fit_cox(cohort.time, cohort.event, np.asarray(cohort.clinical, dtype=float), ties="efron")
    assert np.allclose(recipe["clinical_only_model"]["coefficients"], direct.beta, atol=1e-10)
    report_ = validate_locked_recipe(external, recipe)
    terms = recipe["model"]["terms"]
    design = pd.get_dummies(external[["sex", "stage_group"]], dtype=float)
    columns = {"age": external["age"].to_numpy(dtype=float), "risk_score": external["risk_score"].to_numpy(dtype=float),
               **{name: design[name].to_numpy() for name in terms if name not in ("age", "risk_score")}}
    linear = np.column_stack([columns[name] for name in terms]) @ np.asarray(recipe["model"]["coefficients"])
    c_value = harrell_c(external["os_months"], external["os_event"], linear)[0]
    lower, upper = bootstrap_c(external["os_months"], external["os_event"], linear)
    metrics = report_["metrics"]
    assert abs(metrics["c_index"] - c_value) < 1e-12, (metrics["c_index"], c_value)
    assert abs(metrics["c_index_ci"][0] - lower[0]) < 1e-12 and abs(metrics["c_index_ci"][1] - upper[0]) < 1e-12, (metrics["c_index_ci"], lower, upper)
    passed("bootstrap_and_recipe", {"c": c_value, "interval": [float(lower[0]), float(upper[0])], "gain": metrics["delta_c_index"]})


def check_p2() -> None:
    ready = RSCRIPT and subprocess.run([RSCRIPT, "-e", "stopifnot(requireNamespace('glmnet', quietly = TRUE), requireNamespace('data.table', quietly = TRUE))"],
                                       capture_output=True).returncode == 0
    if not ready:
        report["checks"]["p2_direct"] = {"passed": None, "detail": "skipped: no Rscript with glmnet (set RSCRIPT to the competitors' R)"}
        print("--  p2_direct skipped: no Rscript with glmnet", flush=True)
        return
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        toy_survival(400, rng).to_csv(folder / "TOY.csv", index=False)
        toy_survival(150, rng).to_csv(folder / "V1.csv", index=False)
        subprocess.run([RSCRIPT, str(R_SCRIPTS / "run_p2.R"), "source=real", f"input={folder}", "cohorts=TOY,V1", f"out={folder / 'p2'}"],
                       check=True, capture_output=True, text=True)
        direct = r"""
suppressPackageStartupMessages({library(survival); library(glmnet)})
d <- read.csv('%(folder)s/TOY.csv', check.names = FALSE); v <- read.csv('%(folder)s/V1.csv', check.names = FALSE)
genes <- colnames(d)[-(1:3)]
p <- sapply(genes, function(g) summary(coxph(Surv(d$OS.time, d$OS) ~ d[[g]]))$coefficients[1, 5])
kept <- genes[p < 0.05]
set.seed(5201314)
cv <- cv.glmnet(as.matrix(d[, kept]), Surv(d$OS.time, d$OS), family = "cox", alpha = 1, nfolds = 10)
b <- as.matrix(coef(cv, s = "lambda.min")); selected <- rownames(b)[b[, 1] != 0]
fit <- coxph(Surv(d$OS.time, d$OS) ~ as.matrix(d[, selected]))
write.csv(data.frame(gene = genes, p = p), '%(folder)s/direct_p.csv', row.names = FALSE)
write.csv(data.frame(gene = selected, coefficient = unname(coef(fit))), '%(folder)s/direct_cox.csv', row.names = FALSE)
write.csv(data.frame(ID = v$ID, risk = as.numeric(as.matrix(v[, selected]) %%*%% coef(fit))), '%(folder)s/direct_risk.csv', row.names = FALSE)
""" % {"folder": folder.as_posix()}
        subprocess.run([RSCRIPT, "-e", direct], check=True, capture_output=True, text=True)
        unicox = pd.read_csv(folder / "p2" / "unicox.csv").set_index("gene")
        direct_p = pd.read_csv(folder / "direct_p.csv").set_index("gene")
        assert np.allclose(unicox.loc[direct_p.index, "p"], direct_p["p"], rtol=1e-10, atol=1e-14)
        cox = pd.read_csv(folder / "p2" / "cox.csv")
        direct_cox = pd.read_csv(folder / "direct_cox.csv")
        assert cox["gene"].tolist() == direct_cox["gene"].tolist(), (cox["gene"].tolist(), direct_cox["gene"].tolist())
        assert np.allclose(cox["coefficient"], direct_cox["coefficient"], rtol=1e-10, atol=1e-12)
        risk = pd.read_csv(folder / "p2" / "risk.csv")
        external = risk[risk["cohort"] == "V1"].set_index("ID")["risk"]
        direct_risk = pd.read_csv(folder / "direct_risk.csv").set_index("ID")["risk"]
        assert np.allclose(external.loc[direct_risk.index], direct_risk, rtol=1e-10, atol=1e-12)
        # The final Cox fit against SurvStudio's own Newton fit (Efron ties): an independent implementation.
        toy = pd.read_csv(folder / "TOY.csv")
        survstudio = fit_cox(toy["OS.time"].to_numpy(dtype=float), toy["OS"].to_numpy(dtype=int), toy[cox["gene"]].to_numpy(dtype=float), ties="efron")
        difference = float(np.max(np.abs(survstudio.beta - cox["coefficient"].to_numpy())))
        assert difference < 1e-5, difference
        # The univariate Wald p-values against SurvStudio's fit of each gene alone.
        own = []
        for gene in unicox.index[:10]:
            single = fit_cox(toy["OS.time"].to_numpy(dtype=float), toy["OS"].to_numpy(dtype=int), toy[[gene]].to_numpy(dtype=float), ties="efron")
            z = single.beta[0] / np.sqrt(single.covariance[0, 0])
            own.append(abs(2 * stats.norm.sf(abs(z)) - unicox.loc[gene, "p"]))
        assert max(own) < 1e-6, max(own)
        info = json.loads((folder / "p2" / "run.json").read_text(encoding="utf-8"))
    passed("p2_direct", {"selected": cox["gene"].tolist(), "unicox_kept": info["unicox_kept"], "cox_vs_survstudio_max_difference": difference,
                         "unicox_p_vs_survstudio_max_difference": float(max(own))})


def check_survstudio_pooled() -> None:
    path, pooled_path = RESULTS / "external_validation.csv", RESULTS / "external_pooled.json"
    if not path.exists() or not pooled_path.exists():
        report["checks"]["survstudio_pooled"] = {"passed": None, "detail": "skipped: results/ lacks external_validation.csv or "
                                                                           "external_pooled.json (run step 03)"}
        print("--  survstudio_pooled skipped", flush=True)
        return
    validation = pd.read_csv(path)
    result = pooled_c_gain(validation[validation["scaling"] == "within_cohort"])
    own = json.loads(pooled_path.read_text(encoding="utf-8"))["within_cohort"]
    differences = {}
    for key in ("model_c", "clinical_c", "delta_c"):
        shared = [field for field in own[key] if field in result[key] and own[key][field] is not None and result[key][field] is not None]
        assert {"estimate", "ci_lower", "ci_upper", "hksj_ci_lower", "hksj_ci_upper"} <= set(shared), (key, shared)
        differences[key] = max(abs(float(result[key][field]) - float(own[key][field])) for field in shared)
    assert max(differences.values()) < 1e-9, differences
    passed("survstudio_pooled", {"model_c": result["model_c"]["estimate"], "clinical_c": result["clinical_c"]["estimate"],
                                 "gain": [result["delta_c"]["estimate"], result["delta_c"]["hksj_ci_lower"], result["delta_c"]["hksj_ci_upper"]],
                                 "max_difference_from_external_pooled": differences})


if __name__ == "__main__":
    for check in (check_replay, check_logrank, check_best_cutoff, check_bootstrap_and_recipe, check_p2, check_survstudio_pooled):
        check()
    write_json(RESULTS / "competitors_checks.json", report)
    print("all checks passed", flush=True)

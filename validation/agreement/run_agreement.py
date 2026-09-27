"""Numerical agreement of SurvStudio with R survival, lifelines and scikit-survival.

Runs the same analyses on two bundled cohorts (GBSG2 breast cancer and TCGA-LUAD lung
adenocarcinoma) in SurvStudio, R survival (the reference), lifelines and scikit-survival,
and writes every compared value with its difference to docs/validation/.

    python validation/agreement/run_agreement.py [--output docs/validation]

Needs Rscript with the survival and jsonlite packages. The lifelines and scikit-survival
columns are filled when those packages are installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import survival_toolkit
from survival_toolkit.analysis import compute_cox_analysis, compute_km_analysis
from survival_toolkit.marker_screen import CoxScoreScreen, fit_cox, fit_cox_null
from survival_toolkit.sample_data import load_gbsg2_upload_ready_dataset, load_tcga_luad_upload_ready_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

COHORTS: dict[str, Callable[[], pd.DataFrame]] = {
    "gbsg2": load_gbsg2_upload_ready_dataset,
    "luad": load_tcga_luad_upload_ready_dataset,
}
GBSG2 = {"cohort": "gbsg2", "time": "rfs_days", "event": "rfs_event"}
LUAD = {"cohort": "luad", "time": "os_months", "event": "os_event"}

KM_CASES = [
    {"name": "GBSG2: Kaplan-Meier by hormone therapy", **GBSG2, "group": "horTh", "times": [365, 1095, 1825]},
    {"name": "TCGA-LUAD: Kaplan-Meier by stage", **LUAD, "group": "stage_group", "times": [12, 36, 60]},
]
COX_CASES = [
    {
        "name": "GBSG2: Cox model",
        **GBSG2,
        "covariates": ["age", "tsize", "pnodes", "progrec", "estrec", "horTh", "menostat", "tgrade"],
        "categorical": ["horTh", "menostat", "tgrade"],
        "strata": [],
    },
    {
        "name": "GBSG2: Cox model stratified by menopausal status",
        **GBSG2,
        "covariates": ["age", "tsize", "pnodes", "horTh"],
        "categorical": ["horTh"],
        "strata": ["menostat"],
    },
    {
        "name": "TCGA-LUAD: Cox model (complete cases)",
        **LUAD,
        "covariates": ["age", "sex", "stage_group", "smoking_status"],
        "categorical": ["sex", "stage_group", "smoking_status"],
        "strata": [],
    },
]
SCORE_CASE = {
    "name": "GBSG2: marker score tests (marker engine)",
    **GBSG2,
    "markers": ["pnodes", "progrec", "estrec", "tsize"],
    "clinical": ["age", "tgrade"],
}
FIT_CASE = {
    "name": "GBSG2: Cox fit of the marker engine (stratified by menopausal status)",
    **GBSG2,
    "covariates": ["age", "tsize", "pnodes", "progrec"],
    "strata": ["menostat"],
}


def _step_at(timeline: np.ndarray, values: np.ndarray, time: float) -> float:
    """A right-continuous step function's value at ``time``."""
    return float(values[max(int(np.searchsorted(timeline, time, side="right")) - 1, 0)])


def _float(value: Any) -> float:
    return float("nan") if value is None else float(value)


# ── SurvStudio ───────────────────────────────────────────────────


def survstudio_km(frame: pd.DataFrame, case: dict[str, Any]) -> tuple[dict[str, float], float]:
    result = compute_km_analysis(frame, case["time"], case["event"], group_column=case["group"], event_positive_value=1)
    values: dict[str, float] = {}
    for curve in result["curves"]:
        timeline = np.asarray(curve["timeline"], dtype=float)
        for time in case["times"]:
            prefix = f"{curve['group']} @ {time}"
            values[f"{prefix} survival"] = _step_at(timeline, np.asarray(curve["survival"]), time)
            values[f"{prefix} CI lower"] = _step_at(timeline, np.asarray(curve["ci_lower"]), time)
            values[f"{prefix} CI upper"] = _step_at(timeline, np.asarray(curve["ci_upper"]), time)
    for row in result["summary_table"]:
        label = row["Group"]
        values[f"{label} median"] = _float(row["Median survival"])
        values[f"{label} median CI lower"] = _float(row["Median CI lower"])
        values[f"{label} median CI upper"] = _float(row["Median CI upper"])
        values[f"{label} RMST"] = _float(row["RMST"])
        values[f"{label} RMST SE"] = _float(row["RMST SE"])
    values["log-rank chi-square"] = float(result["test"]["chisq"])
    return values, float(result["rmst_horizon"])


def _level(label: str) -> str:
    """The level of a categorical term label "column: level vs reference"."""
    return label.split(": ", 1)[1].rsplit(" vs ", 1)[0]


def survstudio_cox(frame: pd.DataFrame, case: dict[str, Any]) -> tuple[dict[str, float], dict[str, str], list[tuple[str, str | None, str]]]:
    """SurvStudio's Cox values, the reference level of each categorical covariate, and its
    terms as (variable, level or None, label)."""
    result = compute_cox_analysis(
        frame,
        case["time"],
        case["event"],
        covariates=case["covariates"],
        categorical_covariates=case["categorical"],
        strata_columns=case["strata"],
        event_positive_value=1,
    )
    values: dict[str, float] = {}
    references: dict[str, str] = {}
    terms: list[tuple[str, str | None, str]] = []
    for row in result["results_table"]:
        label = row["Label"]
        values[f"{label} coefficient"] = float(row["Beta"])
        values[f"{label} SE"] = float(row["SE"])
        variable = row["Variable"]
        if variable in case["categorical"]:
            references[variable] = str(row["Reference"])
            terms.append((variable, _level(label), label))
        else:
            terms.append((variable, None, label))
    stats = result["model_stats"]
    values["partial log-likelihood"] = float(stats["partial_log_likelihood"])
    values["likelihood-ratio chi-square"] = float(stats["lr_statistic"])
    if not case["strata"]:
        values["concordance"] = float(stats["c_index"])
    for row in result["diagnostics_table"]:
        if row["Term"].startswith("Global PH test"):
            values["global PH chi-square"] = _float(row["Chi-square"])
        else:
            values[f"{row['Term']} PH chi-square"] = _float(row["Chi-square"])
    return values, references, terms


def _r_coefficient_labels(terms: list[tuple[str, str | None, str]]) -> dict[str, str]:
    """R names its coefficients column + level for factors and column for numbers."""
    return {variable + (level or ""): label for variable, level, label in terms}


def _complete(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.dropna(subset=columns).reset_index(drop=True)


def _clinical_design(frame: pd.DataFrame, clinical: list[str]) -> pd.DataFrame:
    """Numeric clinical design with reference-coded dummies, shared by SurvStudio and R."""
    design = pd.get_dummies(frame[clinical], drop_first=True, dtype=float)
    return design.rename(columns=lambda name: "design_" + str(name).replace(" ", "_"))


def survstudio_score(frame: pd.DataFrame, case: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame, list[str]]:
    data = _complete(frame, [case["time"], case["event"], *case["markers"], *case["clinical"]])
    design = _clinical_design(data, case["clinical"])
    time = data[case["time"]].to_numpy(dtype=float)
    event = data[case["event"]].to_numpy(dtype=int)
    markers = data[case["markers"]].to_numpy(dtype=float)
    marginal = CoxScoreScreen(time, event, null=fit_cox_null(time, event)).statistics(markers)
    clinical = design.to_numpy(dtype=float)
    adjusted = CoxScoreScreen(time, event, null=fit_cox_null(time, event, clinical), Z=clinical).statistics(markers)
    values = {}
    for index, marker in enumerate(case["markers"]):
        values[f"{marker} marginal score chi-square"] = float(marginal.chi2[index])
        values[f"{marker} added-value score chi-square"] = float(adjusted.chi2[index])
    return values, pd.concat([data, design], axis=1), list(design.columns)


def survstudio_fit(frame: pd.DataFrame, case: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame]:
    data = _complete(frame, [case["time"], case["event"], *case["covariates"], *case["strata"]])
    strata = pd.factorize(data[case["strata"][0]])[0] if case["strata"] else None
    values = {}
    for ties in ("efron", "breslow"):
        fit = fit_cox(data[case["time"]], data[case["event"]], data[case["covariates"]].to_numpy(dtype=float), strata, ties)
        for index, covariate in enumerate(case["covariates"]):
            values[f"{covariate} coefficient ({ties})"] = float(fit.beta[index])
            values[f"{covariate} SE ({ties})"] = float(math.sqrt(fit.covariance[index, index]))
        values[f"partial log-likelihood ({ties})"] = fit.loglik
    return values, data


# ── R ────────────────────────────────────────────────────────────


def r_values(frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as work:
        for name, frame in frames.items():
            frame.to_csv(Path(work) / f"{name}.csv", index=False)
        (Path(work) / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        subprocess.run(["Rscript", str(HERE / "reference.R"), work], check=True)
        table = pd.read_csv(Path(work) / "r_values.csv")
    values: dict[str, dict[str, float]] = {}
    for row in table.itertuples(index=False):
        values.setdefault(row.case, {})[row.quantity] = float(row.value)
    return values


def r_versions() -> str:
    output = subprocess.run(
        ["Rscript", "-e", 'cat(R.version.string, "; survival", as.character(packageVersion("survival")))'],
        check=True,
        capture_output=True,
        text=True,
    )
    return output.stdout.strip()


# ── lifelines and scikit-survival ────────────────────────────────


def _installed(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def lifelines_km(frame: pd.DataFrame, case: dict[str, Any], tau: float) -> dict[str, float]:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.utils import median_survival_times, restricted_mean_survival_time

    data = _complete(frame, [case["time"], case["event"], case["group"]])
    values: dict[str, float] = {}
    for label, group in data.groupby(data[case["group"]].astype(str)):
        fitter = KaplanMeierFitter().fit(group[case["time"]], group[case["event"]])
        interval = fitter.confidence_interval_
        timeline = interval.index.to_numpy(dtype=float)
        for time in case["times"]:
            prefix = f"{label} @ {time}"
            values[f"{prefix} survival"] = float(fitter.survival_function_at_times(time).iloc[0])
            values[f"{prefix} CI lower"] = _step_at(timeline, interval.iloc[:, 0].to_numpy(), time)
            values[f"{prefix} CI upper"] = _step_at(timeline, interval.iloc[:, 1].to_numpy(), time)
        values[f"{label} median"] = float(fitter.median_survival_time_)
        median_interval = median_survival_times(interval)
        values[f"{label} median CI lower"] = float(median_interval.iloc[0, 0])
        values[f"{label} median CI upper"] = float(median_interval.iloc[0, 1])
        # With return_variance=True lifelines gives the variance of min(T, tau) itself, not of the
        # RMST estimate, so only the mean is compared.
        values[f"{label} RMST"] = float(restricted_mean_survival_time(fitter, t=tau))
    test = multivariate_logrank_test(data[case["time"]], data[case["group"]].astype(str), data[case["event"]])
    values["log-rank chi-square"] = float(test.test_statistic)
    return values


def _cox_design(frame: pd.DataFrame, case: dict[str, Any], terms: list[tuple[str, str | None, str]]) -> pd.DataFrame:
    """The Cox design with SurvStudio's reference levels, one column per term named by its label."""
    data = _complete(frame, [case["time"], case["event"], *case["covariates"], *case["strata"]])
    columns = {case["time"]: data[case["time"]], case["event"]: data[case["event"]]}
    for variable, level, label in terms:
        columns[label] = (data[variable].astype(str) == level).astype(float) if level is not None else data[variable].astype(float)
    for stratum in case["strata"]:
        columns[stratum] = data[stratum].astype(str)
    return pd.DataFrame(columns)


def lifelines_cox(frame: pd.DataFrame, case: dict[str, Any], terms: list[tuple[str, str | None, str]]) -> dict[str, float]:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    design = _cox_design(frame, case, terms)
    fitter = CoxPHFitter().fit(design, duration_col=case["time"], event_col=case["event"], strata=case["strata"] or None)
    values: dict[str, float] = {}
    for label in fitter.params_.index:
        values[f"{label} coefficient"] = float(fitter.params_[label])
        values[f"{label} SE"] = float(fitter.standard_errors_[label])
    values["partial log-likelihood"] = float(fitter.log_likelihood_)
    values["likelihood-ratio chi-square"] = float(fitter.log_likelihood_ratio_test().test_statistic)
    if not case["strata"]:
        values["concordance"] = float(fitter.concordance_index_)
    test = proportional_hazard_test(fitter, design, time_transform="log")
    for label, statistic in test.summary["test_statistic"].items():
        values[f"{label[0] if isinstance(label, tuple) else label} PH chi-square"] = float(statistic)
    return values


def sksurv_km(frame: pd.DataFrame, case: dict[str, Any]) -> dict[str, float]:
    from sksurv.compare import compare_survival
    from sksurv.nonparametric import kaplan_meier_estimator

    data = _complete(frame, [case["time"], case["event"], case["group"]])
    values: dict[str, float] = {}
    for label, group in data.groupby(data[case["group"]].astype(str)):
        times, survival, interval = kaplan_meier_estimator(
            group[case["event"]].astype(bool).to_numpy(), group[case["time"]].to_numpy(dtype=float), conf_type="log-log"
        )
        timeline = np.concatenate(([0.0], times))
        for time in case["times"]:
            prefix = f"{label} @ {time}"
            values[f"{prefix} survival"] = _step_at(timeline, np.concatenate(([1.0], survival)), time)
            values[f"{prefix} CI lower"] = _step_at(timeline, np.concatenate(([1.0], interval[0])), time)
            values[f"{prefix} CI upper"] = _step_at(timeline, np.concatenate(([1.0], interval[1])), time)
    outcome = np.array(
        list(zip(data[case["event"]].astype(bool), data[case["time"]].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
    values["log-rank chi-square"] = float(compare_survival(outcome, data[case["group"]].astype(str).to_numpy())[0])
    return values


def sksurv_cox(frame: pd.DataFrame, case: dict[str, Any], terms: list[tuple[str, str | None, str]]) -> dict[str, float]:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    if case["strata"]:
        return {}
    design = _cox_design(frame, case, terms)
    event = design.pop(case["event"]).astype(bool).to_numpy()
    time = design.pop(case["time"]).astype(float).to_numpy()
    outcome = np.array(list(zip(event, time)), dtype=[("event", bool), ("time", float)])
    model = CoxPHSurvivalAnalysis(ties="efron", alpha=0.0).fit(design, outcome)
    values = {f"{label} coefficient": float(beta) for label, beta in zip(design.columns, model.coef_)}
    values["concordance"] = float(concordance_index_censored(event, time, design.to_numpy() @ model.coef_)[0])
    return values


# ── Report ───────────────────────────────────────────────────────


def compare(reference: dict[str, float], other: dict[str, float] | None) -> dict[str, float | None]:
    if other is None:
        return {}
    return {key: (abs(other[key] - value) if key in other and np.isfinite(other[key]) and np.isfinite(value) else None) for key, value in reference.items()}


def _cell(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.{digits}g}"


def _max(differences: dict[str, float | None]) -> float | None:
    finite = [value for value in differences.values() if value is not None]
    return max(finite) if finite else None


def build_report(cases: list[dict[str, Any]], environment: dict[str, str]) -> str:
    lines = [
        "# Numerical agreement with R, lifelines and scikit-survival",
        "",
        f"Generated {environment['date']} by `validation/agreement/run_agreement.py` on the bundled GBSG2 and TCGA-LUAD cohorts.",
        "R survival is the reference; the difference columns give the absolute difference from R. Empty cells: the package",
        "does not report that quantity (or the package is not installed).",
        "",
        "| Component | Version |",
        "|---|---|",
        *[f"| {name} | {version} |" for name, version in environment.items() if name != "date"],
        "",
        "## Summary",
        "",
        "| Analysis | Quantities | Largest difference from R: SurvStudio | lifelines | scikit-survival |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case['name']} | {len(case['r'])} | {_cell(_max(case['differences']['survstudio']), 3)} | "
            f"{_cell(_max(case['differences'].get('lifelines', {})), 3)} | {_cell(_max(case['differences'].get('sksurv', {})), 3)} |"
        )
    lines += [
        "",
        "Notes:",
        "",
        "- Kaplan-Meier intervals are pointwise log-log intervals (R `conf.type = \"log-log\"`); the median's interval is",
        "  where those bands cross 0.5.",
        "- When the curve stays exactly at 0.5 over an interval, SurvStudio and lifelines report its first time as the median",
        "  and R the midpoint; this does not happen in these cohorts.",
        "- The proportional-hazards statistics are the classic Grambsch-Therneau tests on the Schoenfeld residuals with log",
        "  time, which SurvStudio reports; the R values apply those formulas to `resid(fit, \"schoenfeld\")` (the newer",
        "  `cox.zph` test differs), and lifelines' per-term test uses the same formulas.",
        "- lifelines stops its Newton-Raphson iterations at a looser tolerance, so its Cox estimates differ from R's in the",
        "  fifth or sixth significant digit.",
        "- The RMST standard error is compared with R only: lifelines returns the variance of the restricted survival time",
        "  itself, not of the RMST estimate.",
        "- The marker score tests are compared with R's `coxph(..., init = ..., iter.max = 0)` score statistics; the",
        "  marker engine's Cox fit is compared for Efron and Breslow ties.",
        "",
    ]
    for case in cases:
        lines += [f"## {case['name']}", "", "| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |", "|---|---|---|---|---|---|---|---|"]
        for quantity, reference in case["r"].items():
            row = [quantity, _cell(case["survstudio"].get(quantity)), _cell(reference), _cell(case["differences"]["survstudio"].get(quantity), 3)]
            for package in ("lifelines", "sksurv"):
                values = case.get(package) or {}
                row += [_cell(values.get(quantity)), _cell(case["differences"].get(package, {}).get(quantity), 3)]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    """Missing values (NaN) as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=str(ROOT / "docs" / "validation"))
    arguments = parser.parse_args()

    frames = {name: loader() for name, loader in COHORTS.items()}
    has_lifelines, has_sksurv = _installed("lifelines"), _installed("sksurv")
    cases: list[dict[str, Any]] = []
    spec: dict[str, Any] = {"km": [], "cox": [], "score": [], "fit": []}

    for case in KM_CASES:
        values, tau = survstudio_km(frames[case["cohort"]], case)
        spec["km"].append({**case, "tau": tau})
        cases.append({
            "name": case["name"],
            "survstudio": values,
            "lifelines": lifelines_km(frames[case["cohort"]], case, tau) if has_lifelines else None,
            "sksurv": sksurv_km(frames[case["cohort"]], case) if has_sksurv else None,
        })
    for case in COX_CASES:
        values, references, terms = survstudio_cox(frames[case["cohort"]], case)
        spec["cox"].append({**case, "references": references, "labels": _r_coefficient_labels(terms)})
        cases.append({
            "name": case["name"],
            "survstudio": values,
            "lifelines": lifelines_cox(frames[case["cohort"]], case, terms) if has_lifelines else None,
            "sksurv": sksurv_cox(frames[case["cohort"]], case, terms) if has_sksurv else None,
        })
    score_values, score_frame, design_columns = survstudio_score(frames[SCORE_CASE["cohort"]], SCORE_CASE)
    frames["gbsg2_score"] = score_frame
    spec["score"].append({**SCORE_CASE, "cohort": "gbsg2_score", "clinical_design": design_columns})
    cases.append({"name": SCORE_CASE["name"], "survstudio": score_values, "lifelines": None, "sksurv": None})
    fit_values, fit_frame = survstudio_fit(frames[FIT_CASE["cohort"]], FIT_CASE)
    frames["gbsg2_fit"] = fit_frame
    spec["fit"].append({**FIT_CASE, "cohort": "gbsg2_fit"})
    cases.append({"name": FIT_CASE["name"], "survstudio": fit_values, "lifelines": None, "sksurv": None})

    reference = r_values(frames, spec)
    for case in cases:
        case["r"] = reference[case["name"]]
        case["differences"] = {"survstudio": compare(case["r"], case["survstudio"])}
        for package in ("lifelines", "sksurv"):
            if case.get(package):
                case["differences"][package] = compare(case["r"], case[package])

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    environment = {
        "date": dt.date.today().isoformat(),
        "SurvStudio": survival_toolkit.__version__,
        "Python": platform.python_version(),
        "numpy": version("numpy"),
        "statsmodels": version("statsmodels"),
        "R": r_versions(),
        "lifelines": version("lifelines"),
        "scikit-survival": version("scikit-survival"),
        "Platform": f"{platform.system()} {platform.machine()}",
    }
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "numerical_agreement.md").write_text(build_report(cases, environment) + "\n", encoding="utf-8")
    (output / "numerical_agreement.json").write_text(
        json.dumps(_json_ready({"environment": environment, "cases": cases}), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for case in cases:
        print(f"{case['name']}: {len(case['r'])} quantities, largest difference from R {_cell(_max(case['differences']['survstudio']), 3)}")


if __name__ == "__main__":
    main()

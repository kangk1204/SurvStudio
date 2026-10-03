"""Independent R checks for finite cancellation, genuine separation and unchanged HC3."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/guarded_inference"))
from study import dataset
from survival_toolkit.clinical_basis import fit_clinical_encoder, transform_clinical_encoder
from survival_toolkit.marker_screen import fit_cox
from survival_toolkit.marker_diagnostics import hc3_wald, METHOD_VERSION


def run(output, rlib=None):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(10482)
    n = 400
    z, u = rng.normal(size=(2, n))
    design = np.column_stack([z, z + .01 * u])
    time = rng.exponential(size=n) / np.exp(.8 * z + .4 * u)
    cases = {"finite_cancellation": (time, np.ones(n, int), design, False, "independent synthetic regression fixture")}
    for index in (27, 371, 1099):
        frame, _, _, _ = dataset("independent", 180, 30, 2026100302, index)
        encoder = fit_clinical_encoder(frame, ["Z"], basis="restricted_cubic_spline")
        x = transform_clinical_encoder(frame, encoder)
        cases[f"v1_failed_spline_{index}"] = (frame.time.to_numpy(), frame.event.to_numpy(), x, False,
                                                 "post-confirmation defect reproduction; not new performance evidence")
    time = np.arange(1., 21.)
    cases["genuine_separation"] = (time, np.ones(20, int), (time <= 10).astype(float)[:, None], True, "positive control")
    for name, (time, event, x, _, _) in cases.items():
        frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(x.shape[1])])
        frame.insert(0, "event", event)
        frame.insert(0, "time", time)
        frame.to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame({"name": list(cases)}).to_csv(output / "cases.csv", index=False)
    x = rng.normal(size=(180, 2))
    d = np.column_stack([np.ones(180), x])
    y = .1 * x[:, 0] + rng.normal(size=180) * np.exp(.3 * x[:, 1])
    frame = pd.DataFrame(d, columns=["intercept", "z1", "z2"])
    frame["y"] = y
    frame.to_csv(output / "hc3.csv", index=False)
    command = ["Rscript", str(Path(__file__).with_name("reference.R")), str(output)]
    if rlib:
        command.append(rlib)
    subprocess.run(command, check=True)
    states = pd.read_csv(output / "r-fit-states.csv").set_index("name")
    checks = []
    for name, (time, event, x, separated, role) in cases.items():
        fit = fit_cox(time, event, x)
        differences = {"loglik": abs(fit.loglik - states.loc[name, "loglik"])}
        if not separated:
            differences.update(coefficients=float(np.max(np.abs(fit.beta - pd.read_csv(output / f"{name}-r-coefficients.csv").coefficient))),
                               covariance=float(np.max(np.abs(fit.covariance - pd.read_csv(output / f"{name}-r-covariance.csv").to_numpy()))),
                               linear_predictor=float(np.max(np.abs(x @ fit.beta - pd.read_csv(output / f"{name}-r-lp.csv").lp))))
        passed = (fit.converged and bool(np.any(fit.separated)) == separated and
                  bool(states.loc[name, "separation_warning"]) == separated and
                  not bool(states.loc[name, "nonconvergence_warning"]) and all(v <= 1e-6 for v in differences.values()))
        checks.append({"case": name, "role": role, "passed": passed, "differences": differences,
                       "expected_separation": separated, "python_separation": bool(np.any(fit.separated)),
                       "R_separation_warning": bool(states.loc[name, "separation_warning"]),
                       "max_coefficient_per_SD": float(np.max(np.abs(fit.beta) * x.std(axis=0)))})
    hc3 = hc3_wald(y, d, 1)
    reference = pd.read_csv(output / "r-hc3.csv").iloc[0]
    differences = {"statistic": abs(hc3["statistic"] - reference.statistic),
                   "chi_square_p": abs(hc3["p_value"] - reference.chi_square_p),
                   "approximate_F_p": abs(float(stats.f.sf(hc3["statistic"] / hc3["df"], hc3["df"], hc3["df_residual"])) - reference.approximate_F_p)}
    checks.append({"case": "HC3_and_offline_F_reference", "passed": all(v <= 1e-10 for v in differences.values()),
                   "differences": differences, "F_is_exact": False})
    result = {"passed": all(c["passed"] for c in checks), "method_version": METHOD_VERSION,
              "solver_tolerance": 1e-6, "checks": checks,
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), Path(__file__).with_name("reference.R"),
                                 ROOT / "src/survival_toolkit/marker_screen.py", ROOT / "src/survival_toolkit/marker_diagnostics.py"]},
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": checks}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rlib")
    args = parser.parse_args()
    run(args.output, args.rlib)

"""Generate synthetic reference inputs and verify independently reconstructed R transforms."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import importlib.metadata
import platform
import numpy as np
import pandas as pd
from survival_toolkit.clinical_basis import fit_clinical_encoder, transform_clinical_encoder
from survival_toolkit.marker_screen import fit_cox
from survival_toolkit.marker_evaluation import _pooled_c_index
from survival_toolkit.analysis import _efron_schoenfeld_residuals, _cox_grambsch_therneau_test
from survival_toolkit.marker_diagnostics import hc3_wald
from scipy import stats


def run(out, rlib=None, verify_only=False):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    if not verify_only:
        rng=np.random.default_rng(2026100301)
        def data(n):
            z=rng.normal(size=n); grade=rng.choice(["A","B","C"],size=n); binary=rng.integers(0,2,n)
            m=rng.normal(size=(n,2)); eta=.8*z+.2*(z*z-1)+.3*binary+.4*(grade=="B")-.3*(grade=="C")+.4*m[:,0]
            t=rng.exponential(size=n)/(.06*np.exp(eta)); c=rng.exponential(size=n)/.04
            return pd.DataFrame(dict(time=np.minimum(t,c),event=(t<=c).astype(int),z=z,grade=grade,binary=binary,m0=m[:,0],m1=m[:,1]))
        training=data(260); external=data(180)
        training.loc[:9,"z"]=np.nan; external.loc[:6,"z"]=np.nan
        external.loc[7:10,"z"]=[-10.,-5.,5.,10.]
        encoder=fit_clinical_encoder(training,["z","grade","binary"],["grade"],basis="restricted_cubic_spline")
        spec=encoder["spline_specifications"]["z"]
        pd.DataFrame({"knot":spec["knots"]}).to_csv(out/"knots.csv",index=False)
        pd.DataFrame({"mean":spec["means"],"scale":spec["scales"]}).to_csv(out/"scaling.csv",index=False)
        training.to_csv(out/"training.csv",index=False); external.to_csv(out/"external.csv",index=False)
        (out/"encoder.json").write_text(json.dumps(encoder,indent=2)+"\n")
        command=["Rscript",str(Path(__file__).with_name("independent_reference.R")),str(out)]
        if rlib: command.append(rlib)
        subprocess.run(command,check=True)
    training=pd.read_csv(out/"training.csv"); external=pd.read_csv(out/"external.csv")
    encoder=json.loads((out/"encoder.json").read_text())
    inside=transform_clinical_encoder(training,encoder); outside=transform_clinical_encoder(external,encoder)
    design=np.column_stack([inside,training[["m0","m1"]]])
    fit=fit_cox(training.time.to_numpy(),training.event.to_numpy(),design)
    prediction=np.column_stack([outside,external[["m0","m1"]]])@fit.beta
    calibration=fit_cox(external.time.to_numpy(),external.event.to_numpy(),prediction[:,None])
    expected=pd.read_csv(out/"r_metrics.csv").set_index("metric").value
    clinical_fit=fit_cox(training.time.to_numpy(),training.event.to_numpy(),inside)
    ph=_cox_grambsch_therneau_test(
        _efron_schoenfeld_residuals(inside,training.time.to_numpy(),training.event.to_numpy(),clinical_fit.beta),
        clinical_fit.covariance, np.log(training.time.to_numpy()))
    reference_ph=pd.read_csv(out/"r_classic_ph.csv").iloc[0]
    linear_encoder=fit_clinical_encoder(training,["z","grade","binary"],["grade"],basis="linear")
    linear=transform_clinical_encoder(training,linear_encoder)
    linear_fit=fit_cox(training.time.to_numpy(),training.event.to_numpy(),linear)
    lr=2*(clinical_fit.loglik-linear_fit.loglik)
    reference_lr=pd.read_csv(out/"r_functional_LR.csv").iloc[0]
    z=training.z.fillna(linear_encoder["numeric_impute_values"]["z"]).to_numpy()
    hc3=hc3_wald(training.m0.to_numpy(),np.column_stack([np.ones(len(z)),z,z*z]),1)
    reference_hc3=pd.read_csv(out/"r_HC3.csv").iloc[0]
    differences={
        "training_basis":float(np.max(np.abs(inside-pd.read_csv(out/"r_training_basis.csv").to_numpy()))),
        "external_basis":float(np.max(np.abs(outside-pd.read_csv(out/"r_external_basis.csv").to_numpy()))),
        "coefficients":float(np.max(np.abs(fit.beta-pd.read_csv(out/"r_coefficients.csv").coefficient))),
        "external_prediction":float(np.max(np.abs(prediction-pd.read_csv(out/"r_external_prediction.csv").linear_predictor))),
        "C":abs(_pooled_c_index(external.time.to_numpy(),external.event.to_numpy(),prediction,None)-expected["C"]),
        "calibration_slope":abs(float(calibration.beta[0])-expected["calibration_slope"]),
        "loglik":abs(fit.loglik-expected["loglik"]),
        "classic_PH_statistic":abs(ph["statistic"]-reference_ph.statistic),
        "classic_PH_p":abs(ph["p_value"]-reference_ph.p_value),
        "linear_coefficients":float(np.max(np.abs(linear_fit.beta-pd.read_csv(out/"r_linear_coefficients.csv").coefficient))),
        "functional_LR_statistic":abs(lr-reference_lr.statistic),
        "functional_LR_p":abs(float(stats.chi2.sf(lr,inside.shape[1]-linear.shape[1]))-reference_lr.p_value),
        "HC3_statistic":abs(hc3["statistic"]-reference_hc3.statistic),
        "HC3_p":abs(hc3["p_value"]-reference_hc3.p_value)}
    tolerance={name:1e-6 for name in differences}
    result={"synthetic":True,"differences":differences,"tolerances":tolerance,
            "passed":bool(fit.converged and calibration.converged and linear_fit.converged and clinical_fit.converged
                          and all(differences[k]<=tolerance[k] for k in differences)),
            "input_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))},
            "R_session":(out/"R_session.txt").read_text(),
            "python_version":platform.python_version(),
            "package_versions":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","statsmodels")},
            "source_hashes":{str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__).resolve(), Path(__file__).with_name("independent_reference.R"),
                                       Path(__file__).resolve().parents[2]/"src/survival_toolkit/clinical_basis.py",
                                       Path(__file__).resolve().parents[2]/"src/survival_toolkit/marker_screen.py",
                                       Path(__file__).resolve().parents[2]/"src/survival_toolkit/marker_diagnostics.py",
                                       Path(__file__).resolve().parents[2]/"src/survival_toolkit/analysis.py"]}}
    (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="R_session"},indent=2))
    if not result["passed"]: raise SystemExit(1)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True); parser.add_argument("--rlib");parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args();run(args.output,args.rlib,args.verify_only)

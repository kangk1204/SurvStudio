"""Paired before/after preprocessing audit; no guarantee that a leakage fix improves realized C."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import survival_toolkit.ml_models as current


def load_reference(path: Path):
    name = "survival_toolkit._audit_reference_ml"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = load_reference(args.reference)
    rows = []
    scenarios = ["complete", "missing", "rare_category", "missing_rare_category"]
    for scenario_index, scenario in enumerate(scenarios):
        for index in range(25):
            rng = np.random.default_rng([2026100211, scenario_index, index])
            n, p = 180, 24
            x = rng.normal(size=(n, p))
            category = np.where(rng.random(n)<.5, "A", "B").astype(object)
            if "rare" in scenario:
                category[rng.choice(n, 3, replace=False)] = "Rare"
            eta = .5*x[:, 0] - .35*x[:, 1] + .2*(category == "B")
            failure, censor = rng.exponential(size=n)/(.06*np.exp(eta)), rng.exponential(size=n)/.04
            if "missing" in scenario:
                missing = rng.random(x.shape) < .25
                x[missing] = np.nan
            features = [f"x_{j}" for j in range(p)] + ["category"]
            frame = pd.DataFrame(x, columns=features[:-1]);frame["category"] = category
            frame["time"],frame["event"] = np.minimum(failure,censor),(failure<=censor).astype(int)
            # Outcomes are never used to change these outer partitions.
            order = rng.permutation(n);train,test=frame.iloc[order[:126]].reset_index(drop=True),frame.iloc[order[126:]].reset_index(drop=True)
            record={"scenario":scenario,"replicate":index,"error":None}
            try:
                for label,module in (("before",reference),("after",current)):
                    result=module._fit_evaluate_lasso_cox_split(train,test,time_column="time",event_column="event",features=features,categorical_features=["category"],random_state=42)
                    record[label+"_c"]=result["c_index"]
                    record[label+"_active"]=result["n_active_features"]
                    record[label+"_risk"]=result["test_risk"]
                record["delta_c"]=record["after_c"]-record["before_c"]
                record["maximum_risk_change"]=float(np.max(np.abs(np.asarray(record["after_risk"])-record["before_risk"])))
            except Exception as exc:
                record["error"]=f"{type(exc).__name__}: {exc}"
            rows.append(record)
    summary=[]
    for scenario in scenarios:
        subset=[r for r in rows if r["scenario"]==scenario];valid=[r for r in subset if r["error"] is None]
        summary.append({"scenario":scenario,"planned":len(subset),"failed":len(subset)-len(valid),
                        "changed_predictions":sum(r["maximum_risk_change"]>1e-10 for r in valid),
                        "mean_delta_c":float(np.mean([r["delta_c"] for r in valid])) if valid else None,
                        "mean_before_c":float(np.mean([r["before_c"] for r in valid])) if valid else None,
                        "mean_after_c":float(np.mean([r["after_c"] for r in valid])) if valid else None})
    args.output.mkdir(parents=True,exist_ok=True)
    result={"seed":2026100211,"reference_sha256":hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "current_sha256":hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),"summary":summary,"replicates":rows,
            "interpretation":"Engineering ablation on 100 paired synthetic outer splits; same outer data and outcomes. Realized accuracy can increase or decrease. This is not a clinical superiority study."}
    (args.output/"lasso_ablation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    pd.DataFrame([{k:v for k,v in r.items() if not k.endswith("_risk")} for r in rows]).to_csv(args.output/"lasso_ablation.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)
    if any(r["error"] for r in rows):raise RuntimeError("Ablation has failed replicates; see the preserved records.")


if __name__=="__main__":main()

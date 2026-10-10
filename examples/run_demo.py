"""Run the bundled examples through the same API used by the dashboard.

Install SurvStudio with [dev] first. Add --models for the GBSG2 ML/DL example.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from fastapi.testclient import TestClient

from survival_toolkit.app import app
from survival_toolkit.sample_data import (
    load_gbsg2_upload_ready_dataset,
    load_tcga_luad_upload_ready_dataset,
    make_example_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("demo_results.json"))
    args = parser.parse_args()
    cases = [
        ("Synthetic demo", "/api/load-example", make_example_dataset,
         "os_months", "os_event", "stage", "Months",
         ["age", "sex", "stage", "treatment", "biomarker_score", "immune_index"],
         ["sex", "stage", "treatment"]),
        ("TCGA-LUAD", "/api/load-tcga-upload-ready", load_tcga_luad_upload_ready_dataset,
         "os_months", "os_event", "stage_group", "Months",
         ["age", "sex", "stage_group", "smoking_status"],
         ["sex", "stage_group", "smoking_status"]),
        ("GBSG2", "/api/load-gbsg2-example", load_gbsg2_upload_ready_dataset,
         "rfs_days", "rfs_event", "horTh", "Days",
         ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
         ["horTh", "menostat", "tgrade"]),
    ]
    report = {"scope": "Bundled-example outputs; not a model-performance benchmark",
              "source_files_sha256": {}, "environment": {}, "datasets": []}
    source_root = Path(__file__).resolve().parents[1] / "src/survival_toolkit"
    for name in ["app.py", "analysis.py", "sample_data.py", "ml_models.py", "deep_models.py"]:
        report["source_files_sha256"][name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
    for name in ["numpy", "pandas", "statsmodels", "scikit-survival", "torch"]:
        try:
            report["environment"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][name] = None
    with TestClient(app, base_url="http://127.0.0.1") as client:
        def post(route, payload=None):
            response = client.post(route, json=payload)
            response.raise_for_status()
            return response.json()

        for name, route, load, time, event, group, unit, features, categorical in cases:
            frame = load()
            dataset = post(route)
            outcome = {"dataset_id": dataset["dataset_id"], "time_column": time,
                       "event_column": event, "event_positive_value": 1}
            km = post("/api/kaplan-meier", {**outcome, "group_column": group,
                       "time_unit_label": unit})["analysis"]
            cox = post("/api/cox", {**outcome, "covariates": features,
                       "categorical_covariates": categorical})["analysis"]
            row = {"name": name, "rows": len(frame), "events": int(frame[event].sum()),
                   "input_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                   "settings": {k: v for k, v in outcome.items() if k != "dataset_id"},
                   "group": group, "unit": unit, "features": features,
                   "categorical_features": categorical,
                   "km": {"summary_table": km["summary_table"], "logrank_p": km["logrank_p"]},
                   "cox": {"model_stats": cox["model_stats"],
                           "results_table": cox["results_table"],
                           "diagnostics_table": cox["diagnostics_table"],
                           "scientific_summary": cox["scientific_summary"]}}
            if args.models and name == "GBSG2":
                common = {**outcome, "features": features,
                          "categorical_features": categorical, "evaluation_strategy": "holdout"}
                ml = post("/api/ml-model", {**common, "model_type": "rsf",
                          "n_estimators": 100, "random_state": 42, "compute_shap": False})["analysis"]
                dl = post("/api/deep-model", {**common, "model_type": "deepsurv",
                          "epochs": 100, "hidden_layers": [64, 64], "dropout": 0.1,
                          "learning_rate": 0.001, "batch_size": 64, "random_seed": 42,
                          "early_stopping_patience": 10})["analysis"]
                ml_stats = {k: v for k, v in ml["model_stats"].items() if k != "training_time_ms"}
                selected = ["c_index", "holdout_c_index", "fit_samples", "evaluation_samples",
                            "epochs_trained", "evaluation_mode", "split_fingerprint",
                            "evaluation_metadata", "scientific_summary"]
                row["prediction_examples"] = {
                    "settings": {"seed": 42, "evaluation": "holdout", "rsf_trees": 100,
                                 "deepsurv_max_epochs": 100, "hidden_layers": [64, 64]},
                    "RSF": {"model_stats": ml_stats,
                            **{k: ml[k] for k in selected if k in ml}},
                    "DeepSurv": {k: dl[k] for k in selected if k in dl},
                }
            report["datasets"].append(row)
            print(f"{name}: {len(frame)} patients; KM and Cox complete", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

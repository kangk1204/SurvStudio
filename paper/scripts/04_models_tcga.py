"""Case study III: all prediction models on TCGA-LUAD clinical data, run through SurvStudio's API exactly as the
interface's Compare All Models sends them, then the bootstrap intervals the leaderboard shows.
"""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from common import RESULTS, survstudio_version, write_csv_atomic, write_json
from survival_toolkit.app import app
from survival_toolkit.sample_data import DATA_DIR

FEATURES = ["age", "sex", "stage_group", "smoking_status", "pack_years_smoked", "tumor_longest_dimension_cm",
            "kras_status", "egfr_status", "expression_subtype"]
CATEGORICAL = ["sex", "stage_group", "smoking_status", "kras_status", "egfr_status", "expression_subtype"]
OUTCOME = {"time_column": "os_months", "event_column": "os_event", "event_positive_value": "1", "features": FEATURES,
           "categorical_features": CATEGORICAL, "model_type": "compare", "evaluation_strategy": "holdout", "locked_test_fraction": None}
# The settings the interface sends for Compare All Models (recorded from its requests).
ML_REQUEST = {**OUTCOME, "random_state": 42, "n_estimators": 100, "learning_rate": 0.1}
DL_REQUEST = {**OUTCOME, "dropout": 0.1, "learning_rate": 0.001, "epochs": 100, "random_seed": 42, "early_stopping_patience": 10,
              "early_stopping_min_delta": 0.0001, "hidden_layers": [64, 64], "batch_size": 64, "num_time_bins": 50, "d_model": 64,
              "n_heads": 4, "n_layers": 2, "latent_dim": 8, "n_clusters": 3}

client = TestClient(app, base_url="http://127.0.0.1")
csv = (DATA_DIR / "tcga_luad_upload_ready.csv").read_bytes()
upload = client.post("/api/upload", files={"file": ("tcga_luad_upload_ready.csv", csv, "text/csv")})
upload.raise_for_status()
dataset_id = upload.json()["dataset_id"]
ml = client.post("/api/ml-model", json={**ML_REQUEST, "dataset_id": dataset_id})
dl = client.post("/api/deep-model", json={**DL_REQUEST, "dataset_id": dataset_id})
ml.raise_for_status()
dl.raise_for_status()
ml_analysis, dl_analysis = ml.json()["analysis"], dl.json()["analysis"]
if ml_analysis["evaluation_split_fingerprint"] != dl_analysis["evaluation_split_fingerprint"]:
    raise RuntimeError("The machine-learning and deep-learning models were tested on different patients: split fingerprints "
                       f"{ml_analysis['evaluation_split_fingerprint']} and {dl_analysis['evaluation_split_fingerprint']}.")
intervals = client.post("/api/model-comparison-intervals", json={"predictions": [ml_analysis["test_predictions"], dl_analysis["test_predictions"]]})
intervals.raise_for_status()
result = intervals.json()

family = {row["model"]: "Classical ML" for row in ml_analysis["comparison_table"]}
family.update({row["model"]: "Deep learning" for row in dl_analysis["comparison_table"]})
reported = {row["model"]: row["c_index"] for row in [*ml_analysis["comparison_table"], *dl_analysis["comparison_table"]]}
table = pd.DataFrame([
    {
        "model": row["model"], "family": family.get(row["model"]), "reported_c": reported.get(row["model"]),
        "c": row["c_index"], "c_lower": row["c_index_ci"][0], "c_upper": row["c_index_ci"][1],
        "delta_vs_cox": row.get("delta_vs_reference"),
        "delta_lower": (row.get("delta_ci") or [None, None])[0], "delta_upper": (row.get("delta_ci") or [None, None])[1],
    }
    for row in result["rows"]
]).sort_values("c", ascending=False)
write_csv_atomic(table, RESULTS / "model_comparison.csv")
write_json(RESULTS / "model_comparison_summary.json", {
    "survstudio": survstudio_version(), "test_patients": result["n"], "test_events": result["events"], "n_bootstrap": result["n_bootstrap"],
    "training_patients": ml_analysis["n_fit_patients"], "training_events": ml_analysis["n_fit_events"],
    "split_fingerprint": ml_analysis["evaluation_split_fingerprint"], "ml_request": ML_REQUEST, "dl_request": DL_REQUEST,
})
print(table.round(3).to_string(index=False))
print(f"test patients {result['n']}, events {result['events']}")

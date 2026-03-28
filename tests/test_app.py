from __future__ import annotations

import csv
import io
from pathlib import Path
import tomllib
import zipfile

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from survival_toolkit.app import DeepModelRequest, app, fail_bad_request, store
from survival_toolkit.sample_data import make_example_dataset


client = TestClient(app)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_index_uses_relative_static_assets() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert '../static/styles.css' in response.text
    assert '<script src="../static/vendor/plotly-3.4.0.min.js" defer></script>' in response.text
    assert '<script src="../static/app.js" defer></script>' in response.text
    assert "cdn.plot.ly" not in response.text
    assert 'id="mlEvaluationStrategy"' in response.text
    assert 'id="downloadMlManuscriptMarkdownButton"' in response.text
    assert 'id="downloadMlManuscriptLatexButton"' in response.text
    assert 'id="downloadMlManuscriptDocxButton"' in response.text
    assert 'id="downloadMlComparisonPngButton"' in response.text
    assert 'id="downloadMlComparisonSvgButton"' in response.text
    assert 'id="mlJournalTemplate"' in response.text
    assert 'id="downloadCoxPngButton"' in response.text
    assert 'id="downloadCoxSvgButton"' in response.text
    assert 'id="runDlCompareButton"' in response.text
    assert 'id="dlEvaluationStrategy"' in response.text
    assert 'id="downloadDlManuscriptMarkdownButton"' in response.text
    assert 'id="downloadDlManuscriptLatexButton"' in response.text
    assert 'id="downloadDlManuscriptDocxButton"' in response.text
    assert 'id="downloadDlComparisonPngButton"' in response.text
    assert 'id="downloadDlComparisonSvgButton"' in response.text
    assert 'id="dlEarlyStoppingPatience"' in response.text
    assert 'id="dlParallelJobs"' in response.text
    assert 'id="dlJournalTemplate"' in response.text
    assert "publication-ready" not in response.text
    assert "exploratory Kaplan-Meier curves" in response.text
    assert 'class="button ghost" id="loadTcgaUploadReadyButton"' in response.text
    assert 'class="button ghost" id="loadTcgaButton"' in response.text
    assert 'class="button ghost" id="loadGbsg2Button"' in response.text
    assert 'class="button ghost" id="loadExampleButton"' in response.text
    assert "Compact upload-style test cohort with fewer columns" in response.text
    assert "Broader bundled clinical cohort with extra fields" in response.text
    assert "Classic breast cancer recurrence-survival cohort" in response.text
    assert "Fastest demo path for checking the app and workflow" in response.text
    assert 'id="uploadButton" type="button"' in response.text
    assert '<span>Compare All Evaluation</span>' in response.text
    assert "Compare All Evaluation affects only <strong>Compare All</strong>" in response.text
    assert "Risk table ticks" in response.text
    assert "It does not change the Kaplan-Meier curve itself." in response.text


def test_index_exposes_dataset_preset_feedback_ui() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="datasetPresetStatusTitle"' in response.text
    assert 'id="datasetPresetStatusText"' in response.text
    assert 'id="datasetPresetChips"' in response.text
    assert 'id="mlFeatureSummaryText"' in response.text
    assert 'id="mlFeatureSummaryChips"' in response.text
    assert 'id="dlFeatureSummaryText"' in response.text
    assert 'id="dlFeatureSummaryChips"' in response.text
    assert 'id="reviewMlFeaturesButton"' in response.text
    assert 'id="reviewDlFeaturesButton"' in response.text
    assert "No preset applied yet." in response.text
    assert "Applying a preset updates recommended columns and checkbox selections only." in response.text
    assert 'class="button ghost compact-btn" id="applyBasicPresetButton"' in response.text
    assert 'class="button ghost compact-btn" id="applyModelPresetButton"' in response.text


def test_frontend_tracks_workspace_controls_in_history_state() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "captureControlSnapshot()" in text
    assert "controls: captureControlSnapshot()" in text
    assert "applyControlSnapshot(historyState.controls || null);" in text
    assert "queueHistorySync()" in text


def test_frontend_disables_ml_learning_rate_for_rsf() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function updateMlModelControlVisibility()" in text
    assert 'refs.mlModelType?.value !== "rsf"' in text
    assert "Learning rate applies to Gradient Boosted Survival only." in text
    assert 'refs.mlModelType.addEventListener("change", () => { updateMlModelControlVisibility(); queueHistorySync(); });' in text


def test_frontend_formats_validation_errors_and_guards_dl_epoch_range() -> None:
    app_js = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit" / "static" / "app.js"
    text = app_js.read_text()

    assert "function extractErrorMessage(payload)" in text
    assert 'item.loc.filter((part) => part !== "body").join(" > ")' in text
    assert "Epochs must be between 10 and 1000." in text
    assert "Hidden layers must be a comma-separated list of positive integers." in text
    assert "CV folds must be between 2 and 10." in text
    assert "Parallel jobs must be between 1 and 16." in text
    assert "validateDlControls();" in text
    assert "rerun seed=" in text
    assert "seed=" in text


def test_deep_model_request_accepts_1000_epochs() -> None:
    request_model = DeepModelRequest(
        dataset_id="demo",
        time_column="os_months",
        event_column="os_event",
        features=["age", "stage"],
        categorical_features=["stage"],
        model_type="deepsurv",
        hidden_layers=[64, 64],
        epochs=1000,
    )

    assert request_model.epochs == 1000


def test_health_allows_null_origin_for_file_preview() -> None:
    response = client.get("/api/health", headers={"Origin": "null"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "null"


def test_upload_dataset_preserves_too_large_status(monkeypatch) -> None:
    import survival_toolkit.app as app_module

    monkeypatch.setattr(app_module, "_MAX_UPLOAD_BYTES", 1)
    response = client.post(
        "/api/upload",
        files={"file": ("big.csv", b"12", "text/csv")},
    )

    assert response.status_code == 413
    assert "200 MB limit" in response.json()["detail"]


def test_upload_dataset_accepts_tsv_files() -> None:
    payload = b"os_months\tos_event\tage\n12\t1\t60\n18\t0\t55\n24\t1\t70\n"
    response = client.post(
        "/api/upload",
        files={"file": ("cohort.tsv", payload, "text/tab-separated-values")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "cohort.tsv"
    assert body["n_rows"] == 3
    assert {column["name"] for column in body["columns"]} >= {"os_months", "os_event", "age"}


def test_readme_highlights_synthetic_columns_cli_inspect_and_dl_runtime_note() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text()

    assert "Synthetic Example Workflow" in text
    assert "This synthetic dataset does **not** use `stage_group` or `treatment_group`." in text
    assert "survival-toolkit inspect path/to/data.csv" in text
    assert "This is the fastest way to catch file-format problems before uploading a cohort in the browser." in text
    assert "## DL Runtime Note" in text


def test_kaplan_meier_response_exposes_groups_and_logrank_p_summary_fields() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "stage",
            "confidence_level": 0.95,
            "max_time": None,
            "risk_table_points": 6,
            "show_confidence_bands": True,
            "logrank_weight": "logrank",
            "fh_p": 1.0,
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["groups"] == 4
    assert analysis["logrank_p"] is not None


def test_ml_compare_response_includes_request_config() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "n_estimators": 20,
            "learning_rate": 0.05,
            "evaluation_strategy": "holdout",
        },
    )

    assert response.status_code == 200
    request_config = response.json()["request_config"]
    assert request_config["dataset_id"] == dataset["dataset_id"]
    assert request_config["time_column"] == "os_months"
    assert request_config["features"] == ["age", "sex", "stage", "biomarker_score"]
    assert request_config["evaluation_strategy"] == "holdout"


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_model_response_includes_request_config_and_seed_metadata() -> None:
    dataset = client.post("/api/load-example").json()
    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "mtlr",
            "hidden_layers": [16],
            "num_time_bins": 10,
            "epochs": 10,
            "batch_size": 16,
            "random_seed": 77,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_config"]["random_seed"] == 77
    assert body["request_config"]["model_type"] == "mtlr"
    assert body["analysis"]["training_seed"] == 77
    assert body["analysis"]["split_seed"] == 77
    assert body["analysis"]["monitor_seed"] == 77


def test_endpoints_accept_boundary_configuration_values() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": None,
            "confidence_level": 0.998,
            "max_time": 36,
            "risk_table_points": 12,
            "show_confidence_bands": True,
            "logrank_weight": "fleming_harrington",
            "fh_p": 5.0,
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["summary_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "n_estimators": 10,
            "learning_rate": 0.002,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["comparison_table"]

    deep_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage", "biomarker_score"],
            "categorical_features": ["sex", "stage"],
            "model_type": "compare",
            "hidden_layers": [16],
            "dropout": 0.0,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 11,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "parallel_jobs": 1,
            "num_time_bins": 10,
            "n_heads": 1,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 2,
            "n_clusters": 2,
        },
    )
    assert deep_response.status_code == 200
    assert deep_response.json()["analysis"]["comparison_table"]


@pytest.mark.parametrize(
    ("path", "payload_overrides"),
    [
        ("/api/kaplan-meier", {"confidence_level": 0.5}),
        ("/api/ml-model", {"model_type": "rsf", "features": ["age"], "n_estimators": 9}),
        ("/api/deep-model", {"model_type": "deepsurv", "features": ["age"], "batch_size": 7}),
        ("/api/deep-model", {"model_type": "compare", "features": ["age"], "parallel_jobs": 17}),
    ],
)
def test_endpoints_reject_out_of_bounds_values(path: str, payload_overrides: dict[str, object]) -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    base_payload: dict[str, object] = {
        "dataset_id": dataset["dataset_id"],
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "features": ["age", "sex", "stage", "biomarker_score"],
        "categorical_features": ["sex", "stage"],
        "covariates": ["age", "sex", "stage", "biomarker_score"],
        "categorical_covariates": ["sex", "stage"],
        "hidden_layers": [16],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 8,
        "random_seed": 11,
        "evaluation_strategy": "holdout",
        "cv_folds": 2,
        "cv_repeats": 1,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,
        "parallel_jobs": 1,
        "num_time_bins": 10,
        "n_heads": 1,
        "d_model": 16,
        "n_layers": 1,
        "latent_dim": 2,
        "n_clusters": 2,
        "confidence_level": 0.95,
        "risk_table_points": 6,
        "show_confidence_bands": True,
        "logrank_weight": "logrank",
        "fh_p": 1.0,
        "model_type": "compare",
        "n_estimators": 10,
    }
    base_payload.update(payload_overrides)

    response = client.post(path, json=base_payload)

    assert response.status_code == 422


def test_missing_dataset_returns_404() -> None:
    response = client.get("/api/dataset/does-not-exist")

    assert response.status_code == 404
    assert "Unknown dataset id" in response.json()["detail"]


def test_fail_bad_request_reraises_server_errors() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        fail_bad_request(RuntimeError("boom"))


def test_fail_bad_request_maps_dependency_errors_to_503() -> None:
    with pytest.raises(HTTPException) as excinfo:
        fail_bad_request(ImportError("scikit-survival is required"))

    assert excinfo.value.status_code == 503
    assert "scikit-survival" in excinfo.value.detail


def test_ml_model_endpoint_reports_missing_dependency(monkeypatch) -> None:
    import survival_toolkit.ml_models as ml_models

    def _raise_missing_dependency(*args, **kwargs):
        raise ImportError("scikit-survival is required for Random Survival Forest.")

    monkeypatch.setattr(ml_models, "train_random_survival_forest", _raise_missing_dependency)

    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "rsf",
        },
    )

    assert response.status_code == 503
    assert "scikit-survival" in response.json()["detail"]


def test_discover_signature_endpoint_persists_derived_group() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "stage", "treatment", "biomarker_score", "immune_index"],
            "max_combination_size": 3,
            "top_k": 10,
            "min_group_fraction": 0.1,
            "bootstrap_iterations": 8,
            "bootstrap_sample_fraction": 0.8,
            "permutation_iterations": 16,
            "validation_iterations": 5,
            "validation_fraction": 0.35,
            "significance_level": 0.05,
            "combination_operator": "mixed",
            "random_seed": 777,
            "new_column_name": "auto_sig",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["derived_column"] == "auto_sig"
    assert payload["signature_analysis"]["results_table"]
    assert payload["signature_analysis"]["search_space"]["bootstrap_iterations"] == 8
    assert payload["signature_analysis"]["search_space"]["permutation_iterations"] == 16
    assert payload["signature_analysis"]["search_space"]["validation_iterations"] == 5
    assert payload["signature_analysis"]["search_space"]["validation_fraction"] == 0.35
    assert payload["signature_analysis"]["search_space"]["significance_level"] == 0.05
    assert payload["signature_analysis"]["search_space"]["combination_operator"] == "mixed"
    assert payload["signature_analysis"]["search_space"]["random_seed"] == 777
    assert payload["signature_analysis"]["scientific_summary"]["headline"]
    assert payload["signature_analysis"]["scientific_summary"]["status"] in {"robust", "review", "caution"}
    assert any(column["name"] == "auto_sig" for column in payload["columns"])


@pytest.mark.parametrize("operator", ["and", "or", "mixed"])
def test_discover_signature_endpoint_supports_all_combination_operators(operator: str) -> None:
    stored = store.create(
        make_example_dataset(seed=73, n_patients=84),
        filename=f"sig_{operator}.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/discover-signature",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "candidate_columns": ["age", "stage", "treatment", "biomarker_score"],
            "max_combination_size": 2,
            "top_k": 6,
            "min_group_fraction": 0.1,
            "bootstrap_iterations": 0,
            "permutation_iterations": 0,
            "validation_iterations": 0,
            "combination_operator": operator,
            "new_column_name": f"sig_{operator}",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["derived_column"] == f"sig_{operator}"
    assert payload["signature_analysis"]["search_space"]["combination_operator"] == operator
    assert payload["signature_analysis"]["results_table"]


def test_optimal_cutpoint_endpoint_reports_adjusted_p_value() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/optimal-cutpoint",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "variable": "biomarker_score",
            "min_group_fraction": 0.1,
            "permutation_iterations": 20,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert result["selection_adjusted_p_value"] is not None
    assert result["raw_p_value"] is not None
    assert result["p_value"] == result["selection_adjusted_p_value"]
    assert result["p_value_label"] == "selection_adjusted_p_value"
    annotations = payload["figure"]["layout"].get("annotations", [])
    assert any("Adj. p" in (a.get("text", "") or "") for a in annotations)


def test_optimal_cutpoint_derive_group_returns_inline_scan_figure() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/derive-group",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_column": "biomarker_score",
            "method": "optimal_cutpoint",
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cutpoint_figure"]["data"][0]["type"] == "scatter"
    assert payload["derive_summary"]["scan_data"]
    assert payload["derive_summary"]["assignment_rule"]
    assert payload["derive_summary"]["p_value_label"] in {"selection_adjusted_p_value", "raw_p_value"}


def test_time_dependent_importance_endpoint_uses_time_major_matrix() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/time-dependent-importance",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "eval_times": [12.0, 24.0, 36.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    analysis = payload["analysis"]
    assert analysis["importance_matrix_orientation"] == "time_major"
    assert len(analysis["importance_matrix"]) == len(analysis["eval_times"])
    assert len(analysis["importance_matrix"][0]) == len(analysis["features"])
    assert len(analysis["importance_matrix_feature_major"]) == len(analysis["features"])
    assert payload["figure"]["data"][0]["type"] == "heatmap"


def test_xai_endpoints_support_explicit_event_positive_value_and_categorical_counterfactual() -> None:
    df = make_example_dataset(seed=67, n_patients=96)
    df["custom_event"] = df["os_event"].map({1: "R", 0: "N"})
    stored = store.create(df, filename="xai_custom_event.csv", copy_dataframe=False)

    time_dep_response = client.post(
        "/api/time-dependent-importance",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "eval_times": [12.0, 24.0],
        },
    )

    assert time_dep_response.status_code == 200
    time_dep_analysis = time_dep_response.json()["analysis"]
    assert time_dep_analysis["importance_matrix_orientation"] == "time_major"

    pdp_response = client.post(
        "/api/pdp",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "target_feature": "age",
        },
    )

    assert pdp_response.status_code == 200
    assert pdp_response.json()["analysis"]["feature"] == "age"

    counterfactual_response = client.post(
        "/api/counterfactual",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "target_feature": "stage",
            "original_value": "I",
            "counterfactual_value": "III",
        },
    )

    assert counterfactual_response.status_code == 200
    counterfactual_analysis = counterfactual_response.json()["analysis"]
    assert counterfactual_analysis["target_feature"] == "stage"
    assert counterfactual_analysis["original_value"] == "I"
    assert counterfactual_analysis["counterfactual_value"] == "III"


def test_pdp_endpoint_rejects_categorical_target_feature() -> None:
    stored = store.create(
        make_example_dataset(seed=68, n_patients=72),
        filename="pdp_categorical.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/pdp",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment"],
            "categorical_features": ["stage", "treatment"],
            "target_feature": "stage",
        },
    )

    assert response.status_code == 400
    assert "categorical feature" in response.json()["detail"].lower()


def test_counterfactual_endpoint_uses_original_value_for_baseline(monkeypatch) -> None:
    import pandas as pd
    import survival_toolkit.ml_models as ml_models

    class _LinearAgeModel:
        def predict(self, X):
            return X[:, 0].astype(float)

    analysis_frame = pd.DataFrame({"age": [10.0, 20.0, 30.0]})

    monkeypatch.setattr(ml_models, "SKSURV_AVAILABLE", True)
    monkeypatch.setattr(
        ml_models,
        "train_random_survival_forest",
        lambda *args, **kwargs: {
            "_model": _LinearAgeModel(),
            "_X_encoded": pd.DataFrame({"age": analysis_frame["age"]}),
            "_analysis_frame": analysis_frame.copy(),
        },
    )

    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/counterfactual",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "features": ["age"],
            "categorical_features": [],
            "target_feature": "age",
            "original_value": 5.0,
            "counterfactual_value": 15.0,
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["original_median_risk"] == pytest.approx(5.0)
    assert analysis["counterfactual_median_risk"] == pytest.approx(15.0)
    assert analysis["risk_change_pct"] == pytest.approx(200.0)


def test_classical_endpoint_option_matrix_runs_without_errors() -> None:
    stored = store.create(
        make_example_dataset(seed=69, n_patients=96),
        filename="classical_matrix.csv",
        copy_dataframe=False,
    )

    derive_payloads = [
        {
            "source_column": "age",
            "method": "median_split",
            "new_column_name": "age_median_group",
        },
        {
            "source_column": "age",
            "method": "tertile_split",
            "new_column_name": "age_tertile_group",
        },
        {
            "source_column": "biomarker_score",
            "method": "quartile_split",
            "new_column_name": "biomarker_quartile_group",
        },
        {
            "source_column": "immune_index",
            "method": "custom_cutoff",
            "new_column_name": "immune_custom_group",
            "cutoff": 0.0,
        },
    ]

    for payload in derive_payloads:
        response = client.post(
            "/api/derive-group",
            json={"dataset_id": stored.dataset_id, **payload},
        )
        assert response.status_code == 200
        assert response.json()["derived_column"] == payload["new_column_name"]

    for weight in ["logrank", "gehan_breslow", "tarone_ware", "fleming_harrington"]:
        km_response = client.post(
            "/api/kaplan-meier",
            json={
                "dataset_id": stored.dataset_id,
                "time_column": "os_months",
                "event_column": "os_event",
                "event_positive_value": 1,
                "group_column": "stage",
                "time_unit_label": "Months",
                "confidence_level": 0.9,
                "risk_table_points": 5,
                "show_confidence_bands": False,
                "logrank_weight": weight,
                "fh_p": 1.5,
            },
        )
        assert km_response.status_code == 200
        km_payload = km_response.json()
        assert km_payload["analysis"]["curves"]
        assert km_payload["figure"]["data"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_covariates": ["stage", "treatment"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    cohort_response = client.post(
        "/api/cohort-table",
        json={
            "dataset_id": stored.dataset_id,
            "variables": ["age", "stage", "treatment", "biomarker_score"],
            "group_column": "stage",
        },
    )
    assert cohort_response.status_code == 200
    assert cohort_response.json()["analysis"]["rows"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_core_endpoints_support_explicit_string_event_positive_value() -> None:
    df = make_example_dataset(seed=72, n_patients=84)
    df["custom_event"] = df["os_event"].map({1: "R", 0: "N"})
    stored = store.create(df, filename="custom_event_core.csv", copy_dataframe=False)

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "group_column": "stage",
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["cohort"]["events"] > 0

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "covariates": ["age", "stage", "treatment"],
            "categorical_covariates": ["stage", "treatment"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "rsf",
            "n_estimators": 20,
            "max_depth": 3,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["model_stats"]["n_events"] > 0

    deep_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "custom_event",
            "event_positive_value": "R",
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "deepsurv",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
        },
    )
    assert deep_response.status_code == 200
    assert deep_response.json()["analysis"]["n_samples"] > 0


@pytest.mark.parametrize("model_type", ["rsf", "gbs"])
def test_single_ml_model_endpoints_handle_mixed_features(model_type: str) -> None:
    stored = store.create(
        make_example_dataset(seed=70, n_patients=96),
        filename=f"single_{model_type}.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": model_type,
            "n_estimators": 24,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["model_stats"]["evaluation_mode"] in {"holdout", "apparent"}
    assert payload["analysis"]["feature_importance"]
    assert payload["importance_figure"]["data"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
@pytest.mark.parametrize("model_type", ["deepsurv", "deephit", "mtlr", "transformer", "vae"])
def test_single_deep_model_endpoints_handle_mixed_features(model_type: str) -> None:
    stored = store.create(
        make_example_dataset(seed=71, n_patients=64),
        filename=f"single_{model_type}.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": model_type,
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 23,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["evaluation_mode"] in {"holdout", "apparent", "holdout_fallback_apparent"}
    assert payload["analysis"]["scientific_summary"]["headline"]
    assert payload["figures"]["loss"]["data"]


def test_ml_model_compare_endpoint_supports_repeated_cv_export() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "compare",
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 2,
            "random_state": 17,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    analysis = payload["analysis"]
    assert analysis["evaluation_mode"] == "repeated_cv"
    assert analysis["cv_folds"] == 2
    assert analysis["cv_repeats"] == 2
    assert analysis["fold_results"]
    assert analysis["manuscript_tables"]["model_performance_table"]
    assert payload["figure"]["data"][0]["type"] == "bar"


def test_frontend_ml_compare_forwards_visible_hyperparameters() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    run_compare_start = app_js.index("async function runCompareModels()")
    run_compare_end = app_js.index("state.ml = payload;", run_compare_start)
    run_compare_body = app_js[run_compare_start:run_compare_end]

    assert 'n_estimators: Number(refs.mlNEstimators.value)' in run_compare_body
    assert 'learning_rate: Number(refs.mlLearningRate.value)' in run_compare_body


def test_frontend_exposes_real_dataset_loader_buttons() -> None:
    index_html = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "templates"
        / "index.html"
    ).read_text(encoding="utf-8")
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'id="loadTcgaUploadReadyButton"' in index_html
    assert 'Upload-Ready TCGA' in index_html
    assert 'id="loadGbsg2Button"' in index_html
    assert 'GBSG2 (Real)' in index_html
    assert 'id="datasetPresetBar"' in index_html
    assert 'id="applyBasicPresetButton"' in index_html
    assert 'id="applyModelPresetButton"' in index_html
    assert 'fetchJSON("/api/load-tcga-upload-ready"' in app_js
    assert 'fetchJSON("/api/load-gbsg2-example"' in app_js
    assert "function datasetPresetForCurrentDataset()" in app_js
    assert 'applyDatasetPreset("basic")' in app_js
    assert 'applyDatasetPreset("models")' in app_js


def test_frontend_uses_dataset_aware_download_filenames() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function buildDownloadFilename(stem, ext" in app_js
    assert "function currentDatasetSlug()" in app_js
    assert "function currentOutcomeSlug()" in app_js
    assert 'buildDownloadFilename("km_summary", "csv", { includeGroup: true })' in app_js
    assert 'buildDownloadFilename("cox_results", "csv")' in app_js
    assert 'buildDownloadFilename("cox_forest", "png")' in app_js
    assert 'buildDownloadFilename("cox_forest", "svg")' in app_js
    assert 'buildDownloadFilename("ml_manuscript_table", "docx", { template: currentMlJournalTemplate() })' in app_js
    assert 'buildDownloadFilename("ml_model_comparison", "png")' in app_js
    assert 'buildDownloadFilename("ml_model_comparison", "svg")' in app_js
    assert 'buildDownloadFilename("dl_manuscript_table", "docx", { template: currentDlJournalTemplate() })' in app_js
    assert 'buildDownloadFilename("dl_model_comparison", "png")' in app_js
    assert 'buildDownloadFilename("dl_model_comparison", "svg")' in app_js
    assert 'buildDownloadFilename("km_curve", "png", { includeGroup: true })' in app_js


def test_frontend_csv_download_sanitizes_formula_like_cells() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "function downloadCsv(filename, rows, columns = null)" in app_js
    assert "const sanitizeCsvCell = (value) => {" in app_js
    assert 'trimmed.startsWith("=")' in app_js
    assert 'trimmed.startsWith("+")' in app_js
    assert 'trimmed.startsWith("-")' in app_js
    assert 'trimmed.startsWith("@")' in app_js
    assert "return `'${text}`;" in app_js


def test_frontend_covariate_picker_keeps_all_unique_continuous_columns() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert "col.n_unique === state.dataset.n_rows" not in app_js


def test_frontend_uses_inline_cutpoint_figure_without_refetch() -> None:
    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "survival_toolkit"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")

    derive_start = app_js.index("async function deriveGroup()")
    derive_end = app_js.index("function updateMethodVisibility()", derive_start)
    derive_body = app_js[derive_start:derive_end]
    assert "payload.cutpoint_figure" in derive_body
    assert 'fetchJSON("/api/optimal-cutpoint"' not in derive_body


def test_export_table_endpoint_returns_journal_markdown() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH", "Mean C-index": 0.7123, "95% CI": "0.650 to 0.774"},
                {"Rank": 2, "Model": "RSF", "Mean C-index": 0.7011, "95% CI": "0.640 to 0.762"},
            ],
            "format": "markdown",
            "style": "journal",
            "template": "nejm",
            "caption": "Table 1. Model discrimination summary.",
            "notes": ["C-index values are cross-validated."],
        },
    )

    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "*Table 1. Model discrimination summary.*" in response.text
    assert "| Rank | Model | Mean C-index | 95% CI |" in response.text
    assert "Notes:" in response.text


def test_export_table_endpoint_preserves_union_of_row_keys() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH"},
                {"Rank": 2, "Model": "RSF", "Mean C-index": 0.7011},
            ],
            "format": "csv",
            "style": "plain",
        },
    )

    assert response.status_code == 200
    assert "Mean C-index" in response.text.splitlines()[0]
    assert "0.7011" in response.text


def test_export_table_endpoint_sanitizes_formula_like_csv_cells() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Sample": "=cmd|'/C calc'!A0", "Model": "+SUM(1,1)", "Note": "@risk"},
                {"Sample": "-10", "Model": "RSF", "Note": "safe"},
            ],
            "format": "csv",
            "style": "plain",
        },
    )

    assert response.status_code == 200
    lines = response.text.splitlines()
    first_row = next(csv.reader([lines[1]]))
    second_row = next(csv.reader([lines[2]]))
    assert first_row[0] == "'=cmd|'/C calc'!A0"
    assert first_row[1] == "'+SUM(1,1)"
    assert first_row[2] == "'@risk"
    assert second_row[0] == "'-10"


def test_export_table_endpoint_returns_journal_latex() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "DeepSurv", "Mean C-index": 0.7342, "95% CI": "0.681 to 0.787"},
            ],
            "format": "latex",
            "style": "journal",
            "template": "jco",
            "caption": "Table 2. Deep-model comparison.",
            "notes": ["Mean C-index is averaged over repeated CV folds."],
        },
    )

    assert response.status_code == 200
    assert "text/x-tex" in response.headers["content-type"]
    assert "\\caption{Table 2. Deep-model comparison.}" in response.text
    assert "\\textit{Footnotes:}" in response.text
    assert "DeepSurv" in response.text


def test_export_table_endpoint_returns_docx_archive() -> None:
    response = client.post(
        "/api/export-table",
        json={
            "rows": [
                {"Rank": 1, "Model": "Cox PH", "C-index": 0.712},
                {"Rank": 2, "Model": "RSF", "C-index": 0.701},
            ],
            "format": "docx",
            "style": "journal",
            "template": "lancet",
            "caption": "Table 1. Model performance summary.",
            "notes": ["Lancet-style comments block."],
        },
    )

    assert response.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in response.headers["content-type"]
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert "[Content_Types].xml" in names
    assert "_rels/.rels" in names
    assert "word/document.xml" in names
    document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Table 1. Model performance summary." in document_xml
    assert "Comments:" in document_xml
    assert "Lancet-style comments block." in document_xml


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_model_compare_endpoint_returns_comparison_figure() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
            "num_time_bins": 10,
            "n_heads": 4,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["comparison_table"]
    assert payload["analysis"]["scientific_summary"]["headline"]
    assert payload["figures"]["comparison"]["data"][0]["type"] == "bar"


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_model_compare_endpoint_supports_repeated_cv() -> None:
    load_response = client.post("/api/load-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "biomarker_score", "immune_index"],
            "categorical_features": [],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 19,
            "num_time_bins": 10,
            "n_heads": 4,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "parallel_jobs": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["evaluation_mode"] == "repeated_cv"
    assert payload["analysis"]["cv_folds"] == 2
    assert payload["analysis"]["fold_results"]
    assert payload["analysis"]["manuscript_tables"]["model_performance_table"]


def test_ml_compare_workflow_with_mixed_features_exports_manuscript_table() -> None:
    stored = store.create(
        make_example_dataset(seed=41, n_patients=96),
        filename="workflow_mixed_ml.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "compare",
            "evaluation_strategy": "repeated_cv",
            "cv_folds": 2,
            "cv_repeats": 1,
            "n_estimators": 16,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 31,
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    manuscript = analysis["manuscript_tables"]
    assert analysis["comparison_table"]
    assert manuscript["model_performance_table"]

    export_response = client.post(
        "/api/export-table",
        json={
            "rows": manuscript["model_performance_table"],
            "format": "markdown",
            "style": "journal",
            "template": "nejm",
            "caption": manuscript["caption"],
            "notes": manuscript["table_notes"],
        },
    )

    assert export_response.status_code == 200
    assert "text/markdown" in export_response.headers["content-type"]
    assert manuscript["caption"] in export_response.text


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_deep_compare_workflow_with_mixed_features_exports_manuscript_table() -> None:
    stored = store.create(
        make_example_dataset(seed=42, n_patients=84),
        filename="workflow_mixed_dl.csv",
        copy_dataframe=False,
    )

    response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 33,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
            "evaluation_strategy": "holdout",
        },
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    manuscript = analysis["manuscript_tables"]
    assert analysis["comparison_table"]
    assert manuscript["model_performance_table"]

    export_response = client.post(
        "/api/export-table",
        json={
            "rows": manuscript["model_performance_table"],
            "format": "latex",
            "style": "journal",
            "template": "jco",
            "caption": manuscript["caption"],
            "notes": manuscript["table_notes"],
        },
    )

    assert export_response.status_code == 200
    assert "text/x-tex" in export_response.headers["content-type"]
    assert "\\caption{" in export_response.text


def test_load_tcga_example_endpoint_returns_real_public_cohort() -> None:
    response = client.post("/api/load-tcga-example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "tcga_luad_xena_example"
    assert payload["n_rows"] >= 500
    column_names = {column["name"] for column in payload["columns"]}
    assert {"os_months", "os_event", "age", "pathologic_stage", "stage_group", "smoking_status"}.issubset(column_names)


def test_load_tcga_upload_ready_endpoint_returns_compact_real_cohort() -> None:
    response = client.post("/api/load-tcga-upload-ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "tcga_luad_upload_ready"
    assert payload["n_rows"] == 609
    column_names = {column["name"] for column in payload["columns"]}
    assert {"os_months", "os_event", "age", "sex", "stage_group", "smoking_status"}.issubset(column_names)


def test_load_gbsg2_example_endpoint_returns_real_public_cohort() -> None:
    response = client.post("/api/load-gbsg2-example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "gbsg2_upload_ready"
    assert payload["n_rows"] == 686
    column_names = {column["name"] for column in payload["columns"]}
    assert {"rfs_days", "rfs_event", "age", "horTh", "menostat", "pnodes", "tgrade", "tsize"}.issubset(column_names)


def test_upload_ready_real_tcga_file_runs_classical_and_ml_smoke() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "tcga_luad_nature2014_upload_ready.csv"
    payload = example_path.read_bytes()

    upload_response = client.post(
        "/api/upload",
        files={"file": (example_path.name, payload, "text/csv")},
    )

    assert upload_response.status_code == 200
    dataset = upload_response.json()
    assert dataset["n_rows"] == 609

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "group_column": "stage_group",
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["curves"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "covariates": ["age", "sex", "stage_group", "smoking_status"],
            "categorical_covariates": ["sex", "stage_group", "smoking_status"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage_group", "smoking_status"],
            "categorical_features": ["sex", "stage_group", "smoking_status"],
            "model_type": "rsf",
            "n_estimators": 20,
            "max_depth": 3,
            "random_state": 13,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["model_stats"]["n_evaluation_patients"] > 0


def test_upload_ready_real_gbsg2_file_runs_classical_and_ml_smoke() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "gbsg2_jco1994_upload_ready.csv"
    payload = example_path.read_bytes()

    upload_response = client.post(
        "/api/upload",
        files={"file": (example_path.name, payload, "text/csv")},
    )

    assert upload_response.status_code == 200
    dataset = upload_response.json()
    assert dataset["n_rows"] == 686

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "group_column": "horTh",
        },
    )
    assert km_response.status_code == 200
    assert km_response.json()["analysis"]["curves"]

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "covariates": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_covariates": ["horTh", "menostat", "tgrade"],
        },
    )
    assert cox_response.status_code == 200
    assert cox_response.json()["analysis"]["results_table"]

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "features": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_features": ["horTh", "menostat", "tgrade"],
            "model_type": "rsf",
            "n_estimators": 20,
            "max_depth": 3,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    assert ml_response.json()["analysis"]["model_stats"]["n_evaluation_patients"] > 0


def test_export_endpoints_can_be_saved_to_nonempty_files(tmp_path: Path) -> None:
    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "covariates": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_covariates": ["horTh", "menostat", "tgrade"],
        },
    )
    assert cox_response.status_code == 200
    cox_rows = cox_response.json()["analysis"]["results_table"]
    cox_path = tmp_path / "cox_results.csv"
    cox_path.write_text(
        "\n".join(
            [
                ",".join(cox_rows[0].keys()),
                *[",".join(str(row.get(col, "")) for col in cox_rows[0].keys()) for row in cox_rows[:3]],
            ]
        ),
        encoding="utf-8",
    )
    assert cox_path.stat().st_size > 0

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "features": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_features": ["horTh", "menostat", "tgrade"],
            "model_type": "compare",
            "evaluation_strategy": "holdout",
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    manuscript = ml_response.json()["analysis"]["manuscript_tables"]

    for fmt, ext in [("markdown", "md"), ("latex", "tex"), ("docx", "docx")]:
        export_response = client.post(
            "/api/export-table",
            json={
                "rows": manuscript["model_performance_table"],
                "format": fmt,
                "style": "journal",
                "template": "jco",
                "caption": manuscript["caption"],
                "notes": manuscript["table_notes"],
            },
        )
        assert export_response.status_code == 200
        out_path = tmp_path / f"ml_manuscript.{ext}"
        if fmt == "docx":
            out_path.write_bytes(export_response.content)
        else:
            out_path.write_text(export_response.text, encoding="utf-8")
        assert out_path.stat().st_size > 0


def test_km_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    km_response = client.post(
        "/api/kaplan-meier",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "group_column": "horTh",
        },
    )
    assert km_response.status_code == 200
    figure = go.Figure(km_response.json()["figure"])

    png_path = tmp_path / "km_plot.png"
    svg_path = tmp_path / "km_plot.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


def test_cox_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    cox_response = client.post(
        "/api/cox",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "covariates": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_covariates": ["horTh", "menostat", "tgrade"],
        },
    )
    assert cox_response.status_code == 200
    figure = go.Figure(cox_response.json()["figure"])

    png_path = tmp_path / "cox_plot.png"
    svg_path = tmp_path / "cox_plot.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


def test_ml_comparison_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    load_response = client.post("/api/load-gbsg2-example")
    assert load_response.status_code == 200
    dataset = load_response.json()

    ml_response = client.post(
        "/api/ml-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "rfs_days",
            "event_column": "rfs_event",
            "event_positive_value": 1,
            "features": ["age", "horTh", "menostat", "pnodes", "tgrade", "tsize"],
            "categorical_features": ["horTh", "menostat", "tgrade"],
            "model_type": "compare",
            "evaluation_strategy": "holdout",
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 17,
        },
    )
    assert ml_response.status_code == 200
    figure = go.Figure(ml_response.json()["figure"])

    png_path = tmp_path / "ml_comparison.png"
    svg_path = tmp_path / "ml_comparison.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_dl_comparison_figure_can_be_rendered_to_png_and_svg_when_kaleido_available(tmp_path: Path) -> None:
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")

    stored = store.create(
        make_example_dataset(seed=84, n_patients=72),
        filename="dl_compare_render.csv",
        copy_dataframe=False,
    )

    dl_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": stored.dataset_id,
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "stage", "treatment", "biomarker_score"],
            "categorical_features": ["stage", "treatment"],
            "model_type": "compare",
            "hidden_layers": [8],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 8,
            "random_seed": 21,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )
    assert dl_response.status_code == 200
    figure = go.Figure(dl_response.json()["figures"]["comparison"])

    png_path = tmp_path / "dl_comparison.png"
    svg_path = tmp_path / "dl_comparison.svg"
    figure.write_image(str(png_path), format="png", width=1400, height=900, scale=2)
    figure.write_image(str(svg_path), format="svg", width=1400, height=900, scale=1)

    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_upload_ready_real_tcga_file_runs_deep_smoke() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "tcga_luad_nature2014_upload_ready.csv"
    payload = example_path.read_bytes()

    upload_response = client.post(
        "/api/upload",
        files={"file": (example_path.name, payload, "text/csv")},
    )

    assert upload_response.status_code == 200
    dataset = upload_response.json()

    deep_response = client.post(
        "/api/deep-model",
        json={
            "dataset_id": dataset["dataset_id"],
            "time_column": "os_months",
            "event_column": "os_event",
            "event_positive_value": 1,
            "features": ["age", "sex", "stage_group", "smoking_status"],
            "categorical_features": ["sex", "stage_group", "smoking_status"],
            "model_type": "deepsurv",
            "hidden_layers": [16],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
            "random_seed": 13,
            "num_time_bins": 10,
            "n_heads": 2,
            "d_model": 16,
            "n_layers": 1,
            "latent_dim": 4,
            "n_clusters": 3,
        },
    )
    assert deep_response.status_code == 200
    assert deep_response.json()["analysis"]["n_samples"] > 0


def test_dev_extra_includes_ml_and_dl_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dev_dependencies = set(pyproject["project"]["optional-dependencies"]["dev"])

    assert {"scikit-survival>=0.23.0", "shap>=0.45.0", "torch>=2.0.0"}.issubset(dev_dependencies)
    assert {"httpx>=0.28.1", "pytest>=8.3.5"}.issubset(dev_dependencies)

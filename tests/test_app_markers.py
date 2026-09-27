from __future__ import annotations

import copy
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from survival_toolkit.app import app
from survival_toolkit.sample_data import make_example_dataset

# The local request guard only answers loopback Host headers.
client = TestClient(app, base_url="http://127.0.0.1")

_FAST = {"n_permutations": 49, "n_resamples": 8, "random_seed": 7}


def _marker_request(dataset_id: str, **changes) -> dict:
    return {
        "dataset_id": dataset_id,
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "marker_columns": ["biomarker_score", "immune_index"],
        "clinical_columns": ["age", "stage"],
        "categorical_clinical": ["stage"],
        **_FAST,
        **changes,
    }


def test_marker_evaluation_returns_tiers_figures_and_a_locked_recipe():
    dataset = client.post("/api/load-example").json()
    response = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"]))

    assert response.status_code == 200, response.text
    payload = response.json()
    analysis = payload["analysis"]
    assert analysis["primary_lens"] == "added_value"
    assert {row["Marker"] for row in payload["display_table"]} == {"biomarker_score", "immune_index"}
    assert {"Tier", "Family-wise P", "Selection frequency", "LR test P"} <= set(payload["display_table"][0])
    assert payload["stability_figure"]["data"] and "layout" in payload["rank_figure"]
    assert analysis["locked_recipe"]["recipe_hash"]
    assert payload["request_config"]["n_permutations"] == 49
    assert payload["dataset_hash"]


def test_marker_evaluation_rejects_outcome_columns_as_markers():
    dataset = client.post("/api/load-example").json()
    response = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"], marker_columns=["os_months"]))

    assert response.status_code == 400
    assert "outcome" in response.json()["detail"].lower()


def test_marker_evaluation_requires_categorical_columns_to_be_clinical():
    dataset = client.post("/api/load-example").json()
    response = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"], categorical_clinical=["sex"]))

    assert response.status_code == 422


def test_marker_validation_applies_the_locked_recipe_to_another_dataset():
    development = client.post("/api/load-example").json()
    recipe = client.post("/api/marker-evaluation", json=_marker_request(development["dataset_id"])).json()["analysis"]["locked_recipe"]
    external = client.post("/api/load-example").json()

    response = client.post("/api/marker-validation", json={"dataset_id": external["dataset_id"], "recipe": recipe, "n_bootstrap": 30})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["validation"]["recipe_hash"] == recipe["recipe_hash"]
    assert 0.5 < payload["validation"]["metrics"]["c_index"] < 1.0
    assert "recipe" not in payload["request_config"]
    assert "data" in payload["figure"]


def test_marker_validation_rejects_an_edited_recipe():
    development = client.post("/api/load-example").json()
    recipe = client.post("/api/marker-evaluation", json=_marker_request(development["dataset_id"])).json()["analysis"]["locked_recipe"]
    edited = copy.deepcopy(recipe)
    edited["model"]["coefficients"][0] += 0.1

    response = client.post("/api/marker-validation", json={"dataset_id": development["dataset_id"], "recipe": edited})

    assert response.status_code == 400
    assert "hash" in response.json()["detail"]


def test_design_audit_flags_a_single_cohort_best_of_101_design():
    response = client.post(
        "/api/design-audit",
        json={
            "candidate_models": 101,
            "gene_only": True,
            "selection_cohorts": [{"name": "GSE72094", "n": 398, "events": 122}],
            "training_in_selection": True,
            "headline": "average including training",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    codes = [flag["code"] for flag in payload["flags"]]
    assert {"NO_SEALED_COHORT", "TRAINING_IN_SELECTION", "SINGLE_SELECTION_COHORT", "MANY_CANDIDATES"} <= set(codes)
    assert payload["high_risk_flags"] >= 3
    assert payload["expected_optimism"]["value"] > 0.05
    assert payload["request_config"]["selection_cohorts"][0]["name"] == "GSE72094"


def test_design_audit_validates_its_input():
    base = {"candidate_models": 12, "gene_only": True, "selection_cohorts": [{"n": 150}]}
    assert client.post("/api/design-audit", json=base).status_code == 200
    for changes in ({"headline": "best"}, {"candidate_models": 1}, {"selection_cohorts": []}, {"winner": "RSF"}):
        assert client.post("/api/design-audit", json={**base, **changes}).status_code == 422
    sealed_headline = client.post("/api/design-audit", json={**base, "headline": "sealed"})
    assert sealed_headline.status_code == 400
    assert "sealed" in sealed_headline.json()["detail"]


def test_marker_evaluation_attaches_a_remark_checklist_and_unadjusted_estimates():
    dataset = client.post("/api/load-example").json()
    payload = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"])).json()

    report = payload["report"]
    assert report["guideline"] == "REMARK"
    assert [item["item"] for item in report["items"]] == [str(number) for number in range(1, 21)]
    assert "example_survival_cohort" in report["items"][1]["text"]
    assert {"Unadjusted HR", "Unadjusted P"} <= set(payload["display_table"][0])


def test_checklist_export_writes_markdown_and_word():
    dataset = client.post("/api/load-example").json()
    report = client.post("/api/marker-evaluation", json=_marker_request(dataset["dataset_id"])).json()["report"]

    markdown = client.post("/api/checklist-export", json={**report, "format": "markdown"})
    assert markdown.status_code == 200, markdown.text
    assert markdown.text.startswith("# REMARK checklist")
    assert markdown.headers["content-type"].startswith("text/markdown")

    word = client.post("/api/checklist-export", json={**report, "format": "docx"})
    assert word.status_code == 200, word.text
    with zipfile.ZipFile(io.BytesIO(word.content)) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
    assert "REMARK checklist" in document and "Authors to complete" in document and "Westfall-Young" in document


def test_checklist_export_validates_the_checklist():
    item = {"item": "1", "section": "Introduction", "topic": "Markers", "status": "author", "text": "Describe."}
    base = {"format": "markdown", "guideline": "REMARK", "reference": "Ref", "software": "SurvStudio", "items": [item]}
    assert client.post("/api/checklist-export", json=base).status_code == 200
    for changes in ({"guideline": "CONSORT"}, {"items": []}, {"items": [{**item, "status": "done"}]}, {"format": "pdf"}):
        assert client.post("/api/checklist-export", json={**base, **changes}).status_code == 422


def test_tripod_ai_checklist_endpoint_reads_comparisons_and_the_dataset():
    dataset = client.post("/api/load-example").json()
    comparison = {
        "family": "ml",
        "analysis": {
            "comparison_table": [{"model": "LASSO-Cox", "c_index": 0.7}],
            "n_patients": 360,
            "n_events": 259,
            "evaluation_mode": "holdout",
        },
        "request_config": {"time_column": "os_months", "event_column": "os_event", "features": ["age", "stage"]},
    }

    response = client.post("/api/tripod-ai-checklist", json={"dataset_id": dataset["dataset_id"], "comparisons": [comparison]})

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["guideline"] == "TRIPOD+AI" and len(report["items"]) == 27
    assert "example_survival_cohort" in report["items"][4]["text"]
    assert "0.700 (LASSO-Cox)" in report["results"]
    expired = client.post("/api/tripod-ai-checklist", json={"dataset_id": "gone", "comparisons": [comparison]})
    assert expired.status_code == 200
    assert expired.json()["items"][4]["text"].startswith("The analysed table")
    assert client.post("/api/tripod-ai-checklist", json={"comparisons": []}).status_code == 422


def _expression_file(tmp_path: Path) -> Path:
    """30 noise genes and one gene equal to the example biomarker, one row per gene."""
    frame = make_example_dataset()
    rng = np.random.default_rng(4)
    genes = pd.DataFrame(rng.normal(size=(30, len(frame))), index=[f"GENE{index:02d}" for index in range(30)], columns=frame["patient_id"])
    genes.loc["SIGNAL"] = frame["biomarker_score"].to_numpy()
    path = tmp_path / "expression.tsv"
    genes.to_csv(path, sep="\t", index_label="gene")
    return path


def _attach(path: Path, dataset_id: str, **fields):
    with path.open("rb") as handle:
        return client.post(
            "/api/marker-matrix",
            data={"dataset_id": dataset_id, "id_column": "patient_id", **fields},
            files={"file": (path.name, handle, "text/tab-separated-values")},
        )


def test_marker_matrix_markers_are_evaluated_against_the_dataset(tmp_path: Path):
    dataset = client.post("/api/load-example").json()
    attached = _attach(_expression_file(tmp_path), dataset["dataset_id"])

    assert attached.status_code == 200, attached.text
    info = attached.json()
    assert info["orientation"] == "markers_in_rows"
    assert (info["n_markers"], info["n_matched"], info["n_patients"]) == (31, 360, 360)

    request = _marker_request(dataset["dataset_id"], marker_columns=[], marker_matrix_id=info["matrix_id"], marker_matrix_id_column="patient_id")
    response = client.post("/api/marker-evaluation", json=request)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["marker_matrix"]["n_markers"] == 31 and payload["marker_matrix"]["n_matched"] == 360
    assert payload["display_table"][0]["Marker"] == "SIGNAL"
    assert payload["analysis"]["cohort"]["n_markers_evaluated"] == 31
    assert "expression.tsv" in payload["report"]["items"][1]["text"]
    assert "Markers evaluated: 31 markers from expression.tsv." in payload["report"]["items"][0]["text"]

    both = client.post("/api/marker-evaluation", json={**request, "marker_columns": ["immune_index"]})
    assert both.status_code == 422
    assert client.delete(f"/api/marker-matrix/{info['matrix_id']}").status_code == 200
    expired = client.post("/api/marker-evaluation", json=request)
    assert expired.status_code == 404
    assert "attach the file again" in expired.json()["detail"]


def test_marker_matrix_upload_explains_problems(tmp_path: Path):
    dataset = client.post("/api/load-example").json()
    path = _expression_file(tmp_path)

    unknown_column = _attach(path, dataset["dataset_id"], id_column="barcode")
    assert unknown_column.status_code == 400 and "not in the dataset" in unknown_column.json()["detail"]

    wrong_layout = _attach(path, dataset["dataset_id"], orientation="samples_in_rows")
    assert wrong_layout.status_code == 400 and "Only 0 patients" in wrong_layout.json()["detail"]

    assert _attach(path, dataset["dataset_id"], orientation="sideways").status_code == 422
    excel = tmp_path / "expression.xlsx"
    excel.write_bytes(b"not a workbook")
    assert _attach(excel, dataset["dataset_id"]).status_code == 400

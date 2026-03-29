from __future__ import annotations

from survival_toolkit.plots import (
    build_km_figure,
    build_cox_forest_figure,
    build_cutpoint_scan_figure,
    build_feature_importance_figure,
    build_model_comparison_figure,
    build_loss_curve_figure,
    build_time_dependent_importance_figure,
)


def test_build_km_figure_returns_json_structure() -> None:
    km_result = {
        "curves": [
            {
                "group": "A",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.8, 0.6],
                "ci_lower": [1.0, 0.7, 0.5],
                "ci_upper": [1.0, 0.9, 0.7],
                "censor_times": [1.5],
                "censor_survival": [0.8],
            }
        ],
        "test": {"test": "logrank", "chisq": 4.2, "p_value": 0.04},
        "display_horizon": 3.0,
    }
    figure = build_km_figure(km_result)
    assert "data" in figure
    assert "layout" in figure
    assert len(figure["data"]) >= 2  # CI band + line


def test_build_cox_forest_figure_returns_json() -> None:
    cox_result = {
        "results_table": [
            {"Label": "age", "Hazard ratio": 1.05, "CI lower": 1.01, "CI upper": 1.10, "P value": 0.02},
        ],
        "model_stats": {"n": 100, "events": 50, "c_index": 0.65, "c_index_label": "Apparent C-index"},
    }
    figure = build_cox_forest_figure(cox_result)
    assert "data" in figure
    assert "layout" in figure
    shapes = figure["layout"].get("shapes", [])
    assert any(shape.get("type") == "line" and shape.get("x0") == 1.0 and shape.get("x1") == 1.0 for shape in shapes)
    annotation_text = " ".join(str(annotation.get("text", "")) for annotation in figure["layout"].get("annotations", []))
    assert "Apparent C-index = 0.650" in annotation_text


def test_build_cutpoint_scan_figure_with_data() -> None:
    result = {
        "scan_data": [
            {"cutpoint": 1.0, "statistic": 2.0, "p_value": 0.15, "n_high": 80, "n_low": 120},
            {"cutpoint": 2.0, "statistic": 5.0, "p_value": 0.03, "n_high": 100, "n_low": 100},
        ],
        "optimal_cutpoint": 2.0,
        "statistic": 5.0,
        "raw_p_value": 0.03,
        "selection_adjusted_p_value": 0.08,
        "p_value": 0.08,
        "label_below_cutpoint": "Low",
        "label_above_cutpoint": "High",
    }
    figure = build_cutpoint_scan_figure(result, variable_name="biomarker")
    assert "data" in figure
    assert len(figure["data"]) >= 2  # line + optimal marker
    # group labels and p-values are shown as stacked right-aligned annotations
    annotations = figure["layout"].get("annotations", [])
    assert any("Adj. p" in (a.get("text", "") or "") for a in annotations)
    assert any("<= cutpoint: Low" in (a.get("text", "") or "") for a in annotations)
    p_annotation = next(a for a in annotations if "Adj. p" in (a.get("text", "") or ""))
    group_annotation = next(a for a in annotations if "<= cutpoint: Low" in (a.get("text", "") or ""))
    assert p_annotation["font"]["size"] == 14
    assert group_annotation["y"] > p_annotation["y"]


def test_build_cutpoint_scan_figure_empty() -> None:
    figure = build_cutpoint_scan_figure({"scan_data": []})
    assert "data" in figure


def test_build_feature_importance_figure() -> None:
    importances = [
        {"feature": "age", "importance": 0.3},
        {"feature": "stage", "importance": 0.2},
    ]
    figure = build_feature_importance_figure(importances, model_name="RSF")
    assert "data" in figure
    assert len(figure["data"]) == 1


def test_build_feature_importance_figure_supports_custom_title_label() -> None:
    importances = [
        {"feature": "age", "importance": 0.3},
        {"feature": "stage", "importance": 0.2},
    ]
    figure = build_feature_importance_figure(
        importances,
        model_name="TRANSFORMER",
        title_label="Gradient-Based Feature Salience",
    )
    assert figure["layout"]["title"]["text"] == "TRANSFORMER Gradient-Based Feature Salience"


def test_build_model_comparison_figure() -> None:
    comparison = {
        "comparison_table": [
            {"model": "Cox PH", "c_index": 0.65},
            {"model": "RSF", "c_index": 0.70},
        ]
    }
    figure = build_model_comparison_figure(comparison)
    assert "data" in figure
    annotations = figure["layout"].get("annotations", [])
    assert any((annotation.get("text") or "") == "Reference (0.5)" for annotation in annotations)
    random_annotation = next(annotation for annotation in annotations if annotation.get("text") == "Reference (0.5)")
    assert random_annotation["xref"] == "paper"
    assert random_annotation["yref"] == "y"
    assert random_annotation["xanchor"] == "left"
    assert random_annotation["yanchor"] == "bottom"
    shapes = figure["layout"].get("shapes", [])
    assert any(shape.get("type") == "line" and shape.get("y0") == 0.5 and shape.get("y1") == 0.5 for shape in shapes)


def test_build_model_comparison_figure_handles_missing_values() -> None:
    comparison = {
        "comparison_table": [
            {"model": "Cox PH", "c_index": None},
            {"model": "RSF", "c_index": 0.70},
        ]
    }
    figure = build_model_comparison_figure(comparison)
    assert "data" in figure
    assert "layout" in figure


def test_build_time_dependent_importance_figure_orients_matrix_correctly() -> None:
    result = {
        "features": ["age", "stage", "biomarker"],
        "eval_times": [1.0, 2.0],
        "importance_matrix_orientation": "time_major",
        # time-by-feature matrix
        "importance_matrix": [
            [0.1, 0.2, 0.5],
            [0.2, 0.1, 0.4],
        ],
    }
    figure = build_time_dependent_importance_figure(result, top_n=2)
    assert "data" in figure
    assert figure["data"][0]["type"] == "heatmap"
    assert figure["data"][0]["x"] == ["1.0", "2.0"]
    assert figure["data"][0]["y"] == ["biomarker", "age"]
    assert figure["data"][0]["z"] == [[0.5, 0.4], [0.1, 0.2]]


def test_build_loss_curve_figure() -> None:
    loss_history = [1.0, 0.8, 0.6, 0.5, 0.45]
    figure = build_loss_curve_figure(loss_history, model_name="DeepSurv")
    assert "data" in figure
    assert len(figure["data"]) == 1

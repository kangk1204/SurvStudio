from __future__ import annotations

import numpy as np

from survival_toolkit.plots import (
    build_cox_diagnostics_figure,
    build_km_figure,
    build_cox_forest_figure,
    build_cutpoint_scan_figure,
    build_feature_importance_figure,
    build_model_comparison_figure,
    build_loss_curve_figure,
    build_shap_figure,
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


def test_build_km_figure_uses_fixed_high_low_colors_regardless_of_curve_order() -> None:
    km_result = {
        "curves": [
            {
                "group": "High",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.7, 0.5],
                "ci_lower": [1.0, 0.6, 0.4],
                "ci_upper": [1.0, 0.8, 0.6],
                "censor_times": [],
                "censor_survival": [],
            },
            {
                "group": "Low",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.8, 0.6],
                "ci_lower": [1.0, 0.7, 0.5],
                "ci_upper": [1.0, 0.9, 0.7],
                "censor_times": [],
                "censor_survival": [],
            },
        ],
        "test": None,
        "display_horizon": 3.0,
    }

    figure = build_km_figure(km_result)

    line_traces = [trace for trace in figure["data"] if trace.get("mode") == "lines"]
    line_colors = {trace["name"]: trace["line"]["color"] for trace in line_traces}
    assert line_colors["High"] == "#db583b"
    assert line_colors["Low"] == "#2563eb"


def test_build_km_figure_explains_hidden_p_value_for_outcome_informed_groups() -> None:
    km_result = {
        "curves": [
            {
                "group": "High",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.7, 0.5],
                "ci_lower": [1.0, 0.6, 0.4],
                "ci_upper": [1.0, 0.8, 0.6],
                "censor_times": [],
                "censor_survival": [],
            },
            {
                "group": "Low",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.8, 0.6],
                "ci_lower": [1.0, 0.7, 0.5],
                "ci_upper": [1.0, 0.9, 0.7],
                "censor_times": [],
                "censor_survival": [],
            },
        ],
        "test": None,
        "outcome_informed_group": True,
        "display_horizon": 3.0,
    }

    figure = build_km_figure(km_result)

    annotations = figure["layout"].get("annotations", [])
    annotation_text = " ".join(str(annotation.get("text", "")) for annotation in annotations)
    assert "fresh raw p-value suppressed" in annotation_text


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
    assert "Red: term p &lt; 0.05" in annotation_text


def test_build_cox_forest_figure_includes_c_index_ci_when_available() -> None:
    cox_result = {
        "results_table": [
            {"Label": "age", "Hazard ratio": 1.05, "CI lower": 1.01, "CI upper": 1.10, "P value": 0.02},
        ],
        "model_stats": {
            "n": 100,
            "events": 50,
            "c_index": 0.65,
            "c_index_label": "Apparent C-index",
            "c_index_ci_lower": 0.61,
            "c_index_ci_upper": 0.69,
            "c_index_ci_level": 0.95,
        },
    }

    figure = build_cox_forest_figure(cox_result)

    annotation_text = " ".join(str(annotation.get("text", "")) for annotation in figure["layout"].get("annotations", []))
    assert "95% CI = 0.610 to 0.690" in annotation_text


def test_build_cox_forest_figure_wraps_long_labels() -> None:
    cox_result = {
        "results_table": [
            {
                "Label": "histology: Lung Adenocarcinoma Mixed Subtype vs Lung Acinar Adenocarcinoma",
                "Hazard ratio": 1.25,
                "CI lower": 1.05,
                "CI upper": 1.52,
                "P value": 0.01,
            },
        ],
        "model_stats": {"n": 220, "events": 99, "c_index": 0.68, "c_index_label": "Apparent C-index"},
    }

    figure = build_cox_forest_figure(cox_result)

    assert "<br>" in figure["layout"]["yaxis"]["ticktext"][0] or figure["layout"]["yaxis"]["ticktext"][0].endswith("…")


def test_build_cox_diagnostics_figure_wraps_long_term_titles_and_adds_top_spacing() -> None:
    cox_result = {
        "diagnostics_plot_data": [
            {
                "term": "expression_subtype: Squamoid vs Bronchioid molecular program",
                "log_time": [0.0, 1.0, 2.0],
                "residual": [0.2, -0.1, 0.15],
                "trend_log_time": [0.0, 2.0],
                "trend_residual": [0.05, 0.12],
                "schoenfeld_rho": 0.31,
                "p_value": 0.021,
            },
            {
                "term": "smoking_status: Current smoker vs Former smoker >15 pack-years",
                "log_time": [0.2, 1.4, 2.6],
                "residual": [0.1, -0.08, -0.02],
                "trend_log_time": [0.2, 2.6],
                "trend_residual": [0.03, -0.04],
                "schoenfeld_rho": -0.22,
                "p_value": 0.044,
            },
        ],
    }

    figure = build_cox_diagnostics_figure(cox_result)

    annotations = figure["layout"].get("annotations", [])
    subplot_titles = [annotation["text"] for annotation in annotations if "Screening view only" not in annotation.get("text", "")]
    subtitle = next(
        annotation["text"]
        for annotation in annotations
        if "Screening view only" in annotation.get("text", "")
    )
    title = next(
        annotation["text"]
        for annotation in annotations
        if annotation.get("text", "") == "Scaled Schoenfeld Residual Trend Check"
    )
    assert any("<br>" in title or title.endswith("…") for title in subplot_titles)
    assert "<br>" in subtitle
    assert title == "Scaled Schoenfeld Residual Trend Check"
    assert figure["layout"]["margin"]["t"] >= 164
    assert figure["layout"]["height"] >= 420
    assert figure["layout"]["title"]["text"] == ""
    assert any("LOWESS-smoothed scaled Schoenfeld residual trends" in annotation.get("text", "") for annotation in annotations)
    assert any("Red: PH table p" in annotation.get("text", "") for annotation in annotations)
    assert figure["layout"]["yaxis"]["title"]["text"] == "Scaled residual"


def test_build_cox_diagnostics_figure_uses_scaled_residual_hover_text() -> None:
    cox_result = {
        "diagnostics_plot_data": [
            {
                "term": "age",
                "log_time": [0.0, 1.0, 2.0, 3.0],
                "residual": [0.2, -0.1, 0.15, 0.05],
                "trend_log_time": [0.0, 1.0, 2.0, 3.0],
                "trend_residual": [0.12, 0.08, 0.03, 0.01],
                "schoenfeld_rho": 0.31,
                "p_value": 0.021,
            },
        ],
    }

    figure = build_cox_diagnostics_figure(cox_result)

    marker_trace = next(trace for trace in figure["data"] if trace.get("mode") == "markers")
    assert "Scaled Schoenfeld residual" in marker_trace["hovertemplate"]


def test_build_cox_diagnostics_figure_uses_robust_y_scale_when_outliers_flatten_panel() -> None:
    cox_result = {
        "diagnostics_plot_data": [
            {
                "term": "estrec",
                "log_time": [float(value) for value in np.linspace(4.5, 8.0, 20)],
                "residual": [
                    -0.08, -0.05, -0.03, -0.02, 0.0, 0.03, 0.01, -0.01, 0.04, 0.02,
                    -0.04, 0.05, 0.08, -0.06, 0.02, 0.01, -0.03, 0.07, 0.06, 950.0,
                ],
                "trend_log_time": [float(value) for value in np.linspace(4.5, 8.0, 20)],
                "trend_residual": [
                    -0.02, -0.01, -0.01, 0.0, 0.01, 0.01, 0.0, -0.01, 0.0, 0.02,
                    0.01, 0.01, 0.02, 0.01, 0.0, -0.01, 0.0, 0.01, 0.02, 0.03,
                ],
                "schoenfeld_rho": 0.12,
                "p_value": 0.09,
            },
        ],
    }

    figure = build_cox_diagnostics_figure(cox_result)

    subtitle = next(
        annotation["text"]
        for annotation in figure["layout"]["annotations"]
        if "Screening view only" in annotation.get("text", "")
    )
    assert "clipped for readability" in subtitle
    assert figure["layout"]["yaxis"]["range"][1] < 950.0


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


def test_build_feature_importance_figure_wraps_long_feature_labels_without_losing_hover_label() -> None:
    long_name = "histology_Lung Adenocarcinoma-Not Otherwise Specified (NOS)"
    importances = [
        {"feature": long_name, "importance": 0.3},
        {"feature": "stage_group", "importance": 0.2},
    ]

    figure = build_feature_importance_figure(importances, model_name="RSF")

    assert "<br>" in figure["layout"]["yaxis"]["ticktext"][1] or figure["layout"]["yaxis"]["ticktext"][1].endswith("…")
    assert figure["data"][0]["customdata"][1] == long_name
    assert figure["layout"]["yaxis"]["tickvals"][1] == long_name
    assert figure["layout"]["margin"]["l"] >= 200


def test_build_shap_figure_wraps_long_feature_labels_without_losing_hover_label() -> None:
    long_name = "pathologic_stage: Stage IIIB vs Stage I reference group"
    shap_result = {
        "feature_importance": [
            {"feature": long_name, "mean_abs_shap": 2.5},
            {"feature": "age", "mean_abs_shap": 1.1},
        ]
    }

    figure = build_shap_figure(shap_result)

    assert "<br>" in figure["layout"]["yaxis"]["ticktext"][1] or figure["layout"]["yaxis"]["ticktext"][1].endswith("…")
    assert figure["data"][0]["customdata"][1] == long_name
    assert figure["layout"]["yaxis"]["tickvals"][1] == long_name
    assert figure["layout"]["margin"]["l"] >= 200


def test_build_feature_importance_figure_keeps_distinct_bars_when_wrapped_labels_collide() -> None:
    importances = [
        {
            "feature": "histology_Lung Signet Ring Adenocarcinoma very long label alpha",
            "importance": 0.8,
        },
        {
            "feature": "histology_Lung Signet Ring Adenocarcinoma very long label beta",
            "importance": 0.7,
        },
    ]

    figure = build_feature_importance_figure(importances, model_name="DEEPHIT")

    assert figure["data"][0]["y"][0] != figure["data"][0]["y"][1]
    assert figure["layout"]["yaxis"]["ticktext"][0] == figure["layout"]["yaxis"]["ticktext"][1]
    assert figure["layout"]["yaxis"]["tickvals"][0] != figure["layout"]["yaxis"]["tickvals"][1]


def test_build_shap_figure_keeps_distinct_bars_when_wrapped_labels_collide() -> None:
    shap_result = {
        "feature_importance": [
            {
                "feature": "expression_subtype_Very long duplicated wrapped label alpha",
                "mean_abs_shap": 2.5,
            },
            {
                "feature": "expression_subtype_Very long duplicated wrapped label beta",
                "mean_abs_shap": 2.0,
            },
        ]
    }

    figure = build_shap_figure(shap_result)

    assert figure["data"][0]["y"][0] != figure["data"][0]["y"][1]
    assert figure["layout"]["yaxis"]["ticktext"][0] == figure["layout"]["yaxis"]["ticktext"][1]
    assert figure["layout"]["yaxis"]["tickvals"][0] != figure["layout"]["yaxis"]["tickvals"][1]


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


def test_build_model_comparison_figure_treats_nan_c_index_as_missing() -> None:
    comparison = {
        "comparison_table": [
            {"model": "Cox PH", "c_index": float("nan")},
            {"model": "RSF", "c_index": 0.70},
        ]
    }

    figure = build_model_comparison_figure(comparison)

    assert figure["data"][0]["y"][0] is None
    assert figure["data"][0]["text"][0] == "NA"


def test_build_cox_forest_figure_ignores_nonfinite_rows() -> None:
    cox_result = {
        "results_table": [
            {"Label": "age", "Hazard ratio": 1.1, "CI lower": 1.01, "CI upper": 1.2, "P value": 0.01},
            {"Label": "bad", "Hazard ratio": float("inf"), "CI lower": 0.9, "CI upper": float("inf"), "P value": 0.02},
            {"Label": "non-estimable", "Hazard ratio": None, "CI lower": None, "CI upper": None, "P value": 0.03},
        ],
        "model_stats": {"n": 100, "events": 50, "c_index": 0.65, "c_index_label": "Apparent C-index"},
    }

    figure = build_cox_forest_figure(cox_result)

    assert figure["data"][0]["x"] == [1.1]
    assert figure["data"][0]["y"] == ["age"]


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


def test_build_loss_curve_figure_includes_monitor_trace_when_available() -> None:
    loss_history = [1.0, 0.8, 0.6]
    monitor_loss_history = [1.1, 0.9, 0.75]

    figure = build_loss_curve_figure(
        loss_history,
        model_name="Transformer",
        monitor_loss_history=monitor_loss_history,
        best_monitor_epoch=3,
        epochs_trained=3,
        max_epochs_requested=5,
        stopped_early=True,
    )

    assert len(figure["data"]) == 2
    assert figure["data"][0]["name"] == "Training loss"
    assert figure["data"][1]["name"] == "Monitor loss"
    assert figure["layout"]["title"]["text"] == "Transformer Training Loss and Monitor loss"
    annotations = figure["layout"].get("annotations", [])
    annotation_text = " ".join(str(annotation.get("text", "")) for annotation in annotations)
    assert "Best monitor epoch: 3" in annotation_text
    assert "Stopped early at epoch 3" in annotation_text
    shapes = figure["layout"].get("shapes", [])
    assert any(shape.get("type") == "line" and shape.get("x0") == 3 and shape.get("x1") == 3 for shape in shapes)


def test_build_loss_curve_figure_supports_maximized_monitor_metric() -> None:
    loss_history = [1.2, 1.0, 0.9]
    monitor_history = [0.54, 0.62, 0.59]

    figure = build_loss_curve_figure(
        loss_history,
        model_name="Transformer",
        monitor_loss_history=monitor_history,
        epochs_trained=3,
        max_epochs_requested=6,
        stopped_early=True,
        monitor_label="Monitor C-index",
        monitor_goal="max",
    )

    assert len(figure["data"]) == 2
    assert figure["data"][1]["name"] == "Monitor C-index"
    assert figure["layout"]["title"]["text"] == "Transformer Training Loss and Monitor C-index"
    annotations = figure["layout"].get("annotations", [])
    annotation_text = " ".join(str(annotation.get("text", "")) for annotation in annotations)
    assert "Best monitor epoch: 2" in annotation_text

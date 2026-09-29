from __future__ import annotations

import numpy as np
import pytest

from survival_toolkit.plots import (
    build_cox_diagnostics_figure,
    build_cox_martingale_figure,
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
        "confidence_level": 0.95,
        "display_horizon": 3.0,
    }
    figure = build_km_figure(km_result)
    band, line, censored = figure["data"]
    assert band["fill"] == "toself" and band["showlegend"] is False
    assert line["name"] == "A" and line["x"] == [0.0, 1.0, 2.0] and line["y"] == [1.0, 0.8, 0.6]
    assert line["line"]["shape"] == "hv"
    assert censored["x"] == [1.5] and censored["y"] == [0.8]
    notes = " ".join(annotation["text"] for annotation in figure["layout"]["annotations"])
    assert "Log-rank test: p = 0.040" in notes and "Shaded bands: 95% pointwise CI" in notes


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
    assert line_colors["High"] == "#c94e33"
    assert line_colors["Low"] == "#2563eb"
    line_dashes = {trace["name"]: trace["line"].get("dash") for trace in line_traces}
    assert line_dashes["High"] == "solid"
    assert line_dashes["Low"] == "solid"


def test_build_km_figure_cycles_dash_patterns_for_multigroup_curves() -> None:
    km_result = {
        "curves": [
            {
                "group": label,
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.9, 0.8],
                "ci_lower": [1.0, 0.85, 0.75],
                "ci_upper": [1.0, 0.95, 0.85],
                "censor_times": [],
                "censor_survival": [],
            }
            for label in ["Stage I", "Stage II", "Stage III", "Stage IV"]
        ],
        "test": None,
        "display_horizon": 3.0,
    }

    figure = build_km_figure(km_result)
    line_traces = [trace for trace in figure["data"] if trace.get("mode") == "lines"]

    assert [trace["line"].get("dash") for trace in line_traces] == ["solid", "dash", "dot", "dashdot"]


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


def test_build_km_figure_uses_journal_style_p_value_and_ci_copy() -> None:
    km_result = {
        "curves": [
            {
                "group": "Stage I",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.9, 0.8],
                "ci_lower": [1.0, 0.85, 0.75],
                "ci_upper": [1.0, 0.95, 0.85],
                "censor_times": [],
                "censor_survival": [],
            }
        ],
        "test": {"test": "logrank", "chisq": 61.2, "p_value": 7.582823258189819e-14},
        "confidence_level": 0.95,
        "display_horizon": 3.0,
    }

    figure = build_km_figure(km_result)

    annotations = figure["layout"].get("annotations", [])
    annotation_text = " ".join(str(annotation.get("text", "")) for annotation in annotations)
    assert "p < 0.001" in annotation_text
    assert "p = <" not in annotation_text
    assert "Shaded bands: 95% pointwise CI" in annotation_text


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
    assert figure["layout"]["margin"]["t"] == 28
    assert figure["layout"]["title"]["text"] == ""


def test_build_cox_forest_figure_keeps_stats_out_of_plot_annotations() -> None:
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
    assert "Apparent C-index" not in annotation_text
    assert "95% CI =" not in annotation_text
    assert "Red: term p" not in annotation_text
    assert "Points = HR; whiskers = 95% Wald CI" in annotation_text


def test_build_cox_forest_figure_labels_log_ticks_in_full() -> None:
    cox_result = {
        "results_table": [
            {"Label": "stage", "Hazard ratio": 3.9, "CI lower": 2.2, "CI upper": 6.9, "P value": 0.001},
            {"Label": "smoking", "Hazard ratio": 3.0, "CI lower": 0.4, "CI upper": 22.0, "P value": 0.3},
        ],
        "model_stats": {"n": 489, "events": 175, "c_index": 0.7, "c_index_label": "Apparent C-index"},
    }

    xaxis = build_cox_forest_figure(cox_result)["layout"]["xaxis"]

    assert xaxis["type"] == "log"
    assert xaxis["tickvals"] == [0.5, 1, 2, 5, 10, 20]
    assert xaxis["ticktext"] == ["0.5", "1", "2", "5", "10", "20"]


def test_build_cox_forest_figure_keeps_default_ticks_for_a_narrow_range() -> None:
    cox_result = {
        "results_table": [{"Label": "age", "Hazard ratio": 1.01, "CI lower": 0.99, "CI upper": 1.03, "P value": 0.2}],
        "model_stats": {"n": 100, "events": 50, "c_index": 0.6, "c_index_label": "Apparent C-index"},
    }

    xaxis = build_cox_forest_figure(cox_result)["layout"]["xaxis"]

    assert "tickvals" not in xaxis


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
    subplot_titles = [annotation["text"] for annotation in annotations]
    assert any("<br>" in title or title.endswith("…") for title in subplot_titles)
    assert figure["layout"]["margin"]["t"] >= 84
    assert figure["layout"]["height"] >= 400
    assert figure["layout"]["title"]["text"] == ""
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


def test_build_cox_diagnostics_figure_uses_bottom_row_x_titles_only_in_multirow_layout() -> None:
    cox_result = {
        "diagnostics_plot_data": [
            {
                "term": "sex: Male vs Female",
                "log_time": [0.0, 1.0, 2.0],
                "residual": [0.2, -0.1, 0.15],
                "trend_log_time": [0.0, 2.0],
                "trend_residual": [0.05, 0.12],
                "schoenfeld_rho": 0.31,
                "p_value": 0.021,
            },
            {
                "term": "stage_group: Stage II vs Stage I",
                "log_time": [0.2, 1.4, 2.6],
                "residual": [0.1, -0.08, -0.02],
                "trend_log_time": [0.2, 2.6],
                "trend_residual": [0.03, -0.04],
                "schoenfeld_rho": -0.22,
                "p_value": 0.044,
            },
            {
                "term": "age",
                "log_time": [0.1, 1.2, 2.4],
                "residual": [0.04, -0.05, 0.02],
                "trend_log_time": [0.1, 2.4],
                "trend_residual": [0.01, 0.0],
                "schoenfeld_rho": 0.11,
                "p_value": 0.19,
            },
            {
                "term": "stage_group: Stage III vs Stage I",
                "log_time": [0.3, 1.6, 2.8],
                "residual": [0.14, -0.18, -0.06],
                "trend_log_time": [0.3, 2.8],
                "trend_residual": [0.04, -0.03],
                "schoenfeld_rho": -0.18,
                "p_value": 0.08,
            },
        ],
    }

    figure = build_cox_diagnostics_figure(cox_result)

    assert figure["layout"]["xaxis"]["title"]["text"] == ""
    assert figure["layout"]["xaxis2"]["title"]["text"] == ""
    assert figure["layout"]["xaxis3"]["title"]["text"] == "log(time)"
    assert figure["layout"]["xaxis4"]["title"]["text"] == "log(time)"


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


def test_build_cox_martingale_figure_uses_separate_title_and_linearity_copy() -> None:
    cox_result = {
        "martingale_plot_data": [
            {
                "term": "age",
                "value": [45.0, 50.0, 55.0, 60.0],
                "residual": [-0.2, -0.05, 0.08, 0.16],
                "trend_value": [45.0, 50.0, 55.0, 60.0],
                "trend_residual": [-0.18, -0.06, 0.06, 0.14],
            },
        ],
    }

    figure = build_cox_martingale_figure(cox_result)

    annotations = figure["layout"].get("annotations", [])
    subplot_titles = [annotation["text"] for annotation in annotations]
    marker_trace = next(trace for trace in figure["data"] if trace.get("mode") == "markers")
    assert "age" in subplot_titles
    assert "Martingale residual" in marker_trace["hovertemplate"]
    assert figure["layout"]["margin"]["t"] >= 72


def test_build_cox_martingale_figure_reports_when_robust_y_clipping_is_applied() -> None:
    cox_result = {
        "martingale_plot_data": [
            {
                "term": "age",
                "value": [float(value) for value in np.linspace(40.0, 80.0, 20)],
                "residual": [
                    -0.2, -0.1, -0.06, -0.03, 0.0, 0.02, 0.03, -0.01, 0.05, 0.04,
                    -0.02, 0.03, 0.08, -0.05, 0.01, 0.0, -0.03, 0.07, 0.06, 150.0,
                ],
                "trend_value": [float(value) for value in np.linspace(40.0, 80.0, 20)],
                "trend_residual": [
                    -0.03, -0.02, -0.01, -0.01, 0.0, 0.01, 0.01, 0.0, 0.01, 0.02,
                    0.01, 0.01, 0.02, 0.01, 0.0, 0.0, 0.01, 0.01, 0.02, 0.03,
                ],
            },
        ],
    }

    figure = build_cox_martingale_figure(cox_result)

    subtitle = next(
        annotation["text"]
        for annotation in figure["layout"]["annotations"]
        if "Screening view only" in annotation.get("text", "")
    )
    assert "martingale residual panels were clipped for" in subtitle
    assert "readability" in subtitle


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
    # "<" is escaped so Plotly does not read "<= cutpoint ... >" as a markup tag.
    assert any("&lt;= cutpoint: Low" in (a.get("text", "") or "") for a in annotations)
    p_annotation = next(a for a in annotations if "Adj. p" in (a.get("text", "") or ""))
    group_annotation = next(a for a in annotations if "&lt;= cutpoint: Low" in (a.get("text", "") or ""))
    assert p_annotation["font"]["size"] == 14
    assert group_annotation["y"] > p_annotation["y"]


def test_build_cutpoint_scan_figure_empty() -> None:
    figure = build_cutpoint_scan_figure({"scan_data": []})
    assert figure["data"] == [] and not figure["layout"].get("annotations") and not figure["layout"].get("shapes")


def test_build_feature_importance_figure() -> None:
    importances = [
        {"feature": "age", "importance": 0.3},
        {"feature": "stage", "importance": 0.2},
    ]
    figure = build_feature_importance_figure(importances, model_name="RSF")
    (bars,) = figure["data"]
    # Horizontal bars listed least important first, so the most important one is drawn at the top.
    assert bars["type"] == "bar" and bars["orientation"] == "h"
    assert bars["y"] == ["stage", "age"] and bars["x"] == [0.2, 0.3]
    assert figure["layout"]["title"]["text"] == "RSF Feature Importance"


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


def test_build_shap_figure_without_subtitle_uses_the_standard_title() -> None:
    shap_result = {
        "method": "kernel",
        "feature_importance": [
            {"feature": "gene_a", "mean_abs_shap": 0.12},
            {"feature": "gene_b", "mean_abs_shap": 0.08},
        ],
    }

    figure = build_shap_figure(shap_result)

    assert figure["layout"]["title"]["text"] == "Approximate SHAP Screening Importance"
    assert not figure["layout"].get("annotations")
    assert figure["layout"]["margin"]["t"] == 80


def test_build_shap_figure_safe_mode_separates_title_and_subtitle() -> None:
    shap_result = {
        "safe_mode": True,
        "companion_model": {
            "selected_feature_count_raw": 30,
            "selected_feature_count_encoded": 42,
        },
        "feature_importance": [
            {"feature": "gene_a", "mean_abs_shap": 0.12},
            {"feature": "gene_b", "mean_abs_shap": 0.08},
        ],
    }

    figure = build_shap_figure(shap_result)

    annotations = figure["layout"].get("annotations", [])
    title_annotation = next(
        annotation for annotation in annotations
        if annotation.get("text", "") == "Reduced-Feature SHAP Screening Importance"
    )
    subtitle_annotation = next(
        annotation for annotation in annotations
        if "SHAP safe mode refit a reduced companion model" in annotation.get("text", "")
    )

    assert float(title_annotation["y"]) > float(subtitle_annotation["y"])
    assert figure["layout"]["margin"]["t"] >= 108


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
    (bars,) = figure["data"]
    assert bars["x"] == ["Cox PH", "RSF"]
    assert bars["y"] == [None, 0.7] and bars["text"] == ["NA", "0.700"]
    assert "C-index = NA" in bars["customdata"][0]
    # The axis still scales to the model that has a C-index.
    assert figure["layout"]["yaxis"]["range"] == [0, pytest.approx(0.84)]


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
    assert figure["data"][0]["type"] == "heatmap"
    assert figure["data"][0]["x"] == ["1.0", "2.0"]
    # Most important first, on an axis that runs top-down: biomarker is the top row.
    assert figure["data"][0]["y"] == ["biomarker", "age"]
    assert figure["layout"]["yaxis"]["autorange"] == "reversed"
    assert figure["data"][0]["z"] == [[0.5, 0.4], [0.1, 0.2]]


def test_build_time_dependent_importance_figure_rejects_ragged_feature_major_matrix() -> None:
    result = {
        "features": ["age", "stage", "biomarker"],
        "eval_times": [1.0, 2.0],
        "importance_matrix_orientation": "feature_major",
        "importance_matrix": [
            [0.1, 0.2],
            [0.3],
            [0.4, 0.5],
        ],
    }

    with pytest.raises(ValueError):
        build_time_dependent_importance_figure(result, top_n=2)


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


def test_format_p_value_never_rounds_across_the_nominal_threshold() -> None:
    from survival_toolkit.plots import _format_p_value, _p_value_expression

    assert _format_p_value(0.0496) == "0.0496"
    assert _format_p_value(0.0504) == "0.050"
    assert _format_p_value(0.2) == "0.200"
    assert _p_value_expression(0.0004) == "p < 0.001"
    assert _p_value_expression(0.012, "Adj. p") == "Adj. p = 0.012"
    # More digits until the printed value stays below 0.05.
    assert _format_p_value(0.04996) == "0.04996"
    assert _format_p_value(0.049996) == "0.049996"
    assert _p_value_expression(0.04996) == "p = 0.04996"
    assert _format_p_value(0.05) == "0.050"
    assert _format_p_value(True) == "NA"


def test_km_confidence_band_is_drawn_as_steps() -> None:
    km_result = {
        "curves": [
            {
                "group": "A",
                "timeline": [0.0, 1.0, 2.0],
                "survival": [1.0, 0.8, 0.6],
                "ci_lower": [1.0, 0.7, 0.5],
                "ci_upper": [1.0, 0.9, 0.7],
                "censor_times": [],
                "censor_survival": [],
            }
        ],
        "test": None,
        "confidence_level": 0.95,
        "display_horizon": 2.0,
    }
    figure = build_km_figure(km_result)
    band = next(trace for trace in figure["data"] if trace.get("fill") == "toself")
    xs, ys = band["x"], band["y"]
    # Upper boundary: (0,1) (1,1) (1,0.9) (2,0.9) (2,0.7) — a vertical jump at each time.
    assert xs[:5] == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert ys[:5] == [1.0, 1.0, 0.9, 0.9, 0.7]


def test_time_dependent_heatmap_keeps_distinct_time_labels() -> None:
    from survival_toolkit.plots import build_time_dependent_importance_figure

    figure = build_time_dependent_importance_figure(
        {
            "features": ["a", "b"],
            "eval_times": [1.21, 1.24, 3.0],
            "importance_matrix": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        }
    )
    labels = figure["data"][0]["x"]
    assert len(set(labels)) == 3


def test_plot_text_from_the_dataset_is_escaped() -> None:
    from survival_toolkit.plots import escape_plotly_template_text, escape_plotly_text

    assert escape_plotly_text("<b>A & B</b>") == "&lt;b&gt;A &amp; B&lt;/b&gt;"
    assert escape_plotly_template_text("50% <x> %{y}") == "50&#37; &lt;x&gt; &#37;{y}"

    group = '<a href="https://example.org">%{y}</a>'
    figure = build_km_figure(
        {
            "curves": [
                {
                    "group": group,
                    "timeline": [0.0, 1.0, 2.0],
                    "survival": [1.0, 0.8, 0.6],
                    "ci_lower": [1.0, 0.7, 0.5],
                    "ci_upper": [1.0, 0.9, 0.7],
                    "censor_times": [1.5],
                    "censor_survival": [0.8],
                }
            ],
            "test": None,
            "display_horizon": 3.0,
        }
    )
    for trace in figure["data"]:
        assert "<a" not in str(trace.get("name", ""))
        # The hover template keeps its own %{x}/%{y} fields but none from the group label.
        assert "%{y}</a>" not in str(trace.get("hovertemplate", ""))
    assert escape_plotly_text(group) in {trace.get("name") for trace in figure["data"]}


def test_build_km_figure_prints_numbers_at_risk_under_the_axis() -> None:
    curve = {
        "timeline": [0.0, 10.0, 20.0],
        "survival": [1.0, 0.8, 0.6],
        "ci_lower": [1.0, 0.7, 0.5],
        "ci_upper": [1.0, 0.9, 0.7],
        "censor_times": [],
        "censor_survival": [],
    }
    km_result = {
        "curves": [{**curve, "group": "Stage I"}, {**curve, "group": "Stage II"}],
        "test": {"test": "logrank", "chisq": 4.2, "p_value": 0.0004},
        "confidence_level": 0.95,
        "display_horizon": 24.0,
        "risk_table": {
            "columns": ["Group", "0", "10", "20"],
            "rows": [{"Group": "Stage I", "0": 50, "10": 31, "20": 12}, {"Group": "Stage II", "0": 40, "10": 20, "20": 5}],
            "times": [0.0, 10.0, 20.0],
        },
    }

    figure = build_km_figure(km_result)

    annotations = figure["layout"]["annotations"]
    texts = [annotation["text"] for annotation in annotations]
    assert "<b>Number at risk</b>" in texts and "Stage I" in texts and "Stage II" in texts
    counts = [annotation for annotation in annotations if annotation.get("xref") == "x"]
    assert [(annotation["x"], annotation["text"]) for annotation in counts[:3]] == [(0.0, "50"), (10.0, "31"), (20.0, "12")]
    assert figure["layout"]["xaxis"]["tickvals"] == [0.0, 10.0, 20.0]
    # The test result and the band note sit in the lower left, off the curves' start at 100%.
    note = next(annotation for annotation in annotations if "Log-rank test" in annotation["text"])
    assert note["y"] == 0.02 and "Shaded bands" in note["text"]
    assert figure["layout"]["margin"]["b"] > 70

    without_table = build_km_figure({key: value for key, value in km_result.items() if key != "risk_table"})
    assert not any("Number at risk" in annotation["text"] for annotation in without_table["layout"]["annotations"])


def _km_curve(group: str) -> dict:
    return {
        "group": group,
        "timeline": [0.0, 10.0, 20.0],
        "survival": [1.0, 0.8, 0.6],
        "ci_lower": [1.0, 0.7, 0.5],
        "ci_upper": [1.0, 0.9, 0.7],
        "censor_times": [],
        "censor_survival": [],
    }


def test_km_three_groups_give_intermediate_its_own_colour() -> None:
    from survival_toolkit.plots import ACCENT, SLATE, _km_group_colors

    groups = ["High", "Intermediate", "Low"]
    figure = build_km_figure(
        {
            "curves": [_km_curve(group) for group in groups],
            "test": None,
            "confidence_level": 0.975,
            "display_horizon": 24.0,
            "risk_table": {
                "columns": ["Group", "0", "10"],
                "rows": [{"Group": group, "0": 30, "10": 20} for group in groups],
                "times": [0.0, 10.0],
            },
        }
    )

    colours = {trace["name"]: trace["line"]["color"] for trace in figure["data"] if trace.get("mode") == "lines"}
    assert colours["High"] == ACCENT and colours["Low"] == SLATE
    assert colours["Intermediate"] not in {ACCENT, SLATE}
    risk_labels = {annotation["text"]: annotation["font"]["color"] for annotation in figure["layout"]["annotations"] if annotation["text"] in colours}
    assert risk_labels == colours
    # A 97.5% band is not rounded to 98%.
    assert any("Shaded bands: 97.5% pointwise CI" in annotation["text"] for annotation in figure["layout"]["annotations"])
    # Plots without a High or Low group keep the palette order.
    assert _km_group_colors(["Stage I", "Stage II"]) == [SLATE, ACCENT]
    assert _km_group_colors(["high risk", "Other", "low risk"])[1] not in {ACCENT, SLATE}


def test_cutpoint_scan_marks_the_cutoff_of_a_make_groups_summary() -> None:
    from survival_toolkit.analysis import derive_group_column
    from survival_toolkit.sample_data import make_example_dataset

    _, _, summary = derive_group_column(
        make_example_dataset(),
        source_column="biomarker_score",
        method="optimal_cutpoint",
        time_column="os_months",
        event_column="os_event",
        event_positive_value=1,
        permutation_iterations=20,
    )
    assert "optimal_cutpoint" not in summary

    figure = build_cutpoint_scan_figure(summary, variable_name="biomarker_score")

    stars = [trace for trace in figure["data"] if trace.get("mode") == "markers"]
    assert len(stars) == 1 and stars[0]["x"] == [summary["cutoff"]]
    assert figure["layout"]["shapes"][0]["x0"] == summary["cutoff"]
    assert any("Adj. p" in annotation["text"] for annotation in figure["layout"]["annotations"])


def test_marker_rank_figure_draws_the_strongest_marker_at_the_top() -> None:
    from survival_toolkit.plots import build_marker_rank_figure

    def row(name: str, tier: str, median: float, low: float, high: float) -> dict:
        return {"marker": name, "tier": tier, "marginal": {"median_rank": median, "rank_interval": [low, high]}}

    figure = build_marker_rank_figure(
        {
            "primary_lens": "marginal",
            "marker_table": [
                row("rank1", "robust", 1, 1, 2),
                row("rank2", "robust", 2, 1, 3),
                row("rank3", "suggestive", 3, 2, 5),
                row("rank4", "not supported", 4, 3, 7),
                row("rank5", "suggestive", 5, 3, 8),
                row("rank6", "not supported", 6, 4, 9),
            ],
        }
    )

    yaxis = figure["layout"]["yaxis"]
    # Categories run bottom to top, so the strongest marker is last.
    assert yaxis["categoryorder"] == "array"
    assert yaxis["categoryarray"] == ["rank6", "rank5", "rank4", "rank3", "rank2", "rank1"]


def test_marker_replication_figure_keeps_the_marker_order_and_skips_infinite_intervals() -> None:
    from survival_toolkit.plots import build_marker_replication_figure

    def row(name: str, hr: float, low: float, high: float, replicated: bool, same: bool) -> dict:
        return {
            "marker": name,
            "marginal": {"hazard_ratio": hr, "ci_lower": low, "ci_upper": high},
            "replicated": replicated,
            "same_direction": same,
            "replication_p_holm": 0.01,
        }

    figure = build_marker_replication_figure(
        {
            "markers": [
                row("lock1", 1.1, 0.9, 1.3, False, True),
                row("lock2", 1.8, 1.4, 2.3, True, True),
                row("lock3", 0.8, 0.6, 1.05, False, False),
                row("lock4", 1.5, 1.2, 1.9, True, True),
                row("lock5", 2.0, 0.5, float("inf"), False, True),
            ],
            "metrics": {"c_index": 0.7, "clinical_only_c_index": 0.62},
        }
    )

    assert figure["layout"]["yaxis2"]["categoryorder"] == "array"
    assert figure["layout"]["yaxis2"]["categoryarray"] == ["lock4", "lock3", "lock2", "lock1"]
    plotted = {label for trace in figure["data"] for label in trace.get("y") or [] if str(label).startswith("lock")}
    assert plotted == {"lock1", "lock2", "lock3", "lock4"}
    # The reference line at HR = 1 is drawn in the forest (Plotly skips lines added to a subplot without traces).
    assert any(shape.get("xref") == "x2" and shape.get("x0") == 1.0 for shape in figure["layout"]["shapes"])


def test_residual_panels_drop_points_with_a_missing_coordinate_as_pairs() -> None:
    diagnostics = build_cox_diagnostics_figure(
        {
            "diagnostics_plot_data": [
                {
                    "term": "age",
                    "log_time": [0.1, None, 0.3, 0.4, 0.5],
                    "residual": [1.0, 2.0, None, 4.0, float("nan")],
                    "trend_log_time": [0.1, 0.4],
                    "trend_residual": [0.5, None],
                    "p_value": 0.2,
                    "schoenfeld_rho": 0.1,
                }
            ]
        }
    )
    points = diagnostics["data"][0]
    assert points["x"] == [0.1, 0.4] and points["y"] == [1.0, 4.0]

    martingale = build_cox_martingale_figure(
        {"martingale_plot_data": [{"term": "age", "value": [50.0, None, 70.0], "residual": [0.2, 0.4, None]}]}
    )
    points = martingale["data"][0]
    assert points["x"] == [50.0] and points["y"] == [0.2]


def test_marker_summary_names_a_model_without_markers_as_the_clinical_model() -> None:
    from survival_toolkit.plots import build_marker_summary_figure

    base = {
        "primary_lens": "added_value",
        "marker_table": [],
        "cohort": {"n_markers_evaluated": 3},
        "tier_counts": {},
        "settings": {"alpha": 0.05},
    }

    clinical_only = build_marker_summary_figure({**base, "signature": {"markers": [], "apparent_c": 0.72, "optimism_corrected_c": 0.71}})
    with_markers = build_marker_summary_figure({**base, "signature": {"markers": ["m1"], "apparent_c": 0.75}})

    titles = [annotation["text"] for annotation in clinical_only["layout"]["annotations"]]
    assert "C-index of the clinical-only model" in titles
    assert "C-index of the selected-marker model" in [annotation["text"] for annotation in with_markers["layout"]["annotations"]]

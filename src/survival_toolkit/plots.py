from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

PAPER = "#ffffff"
INK = "#1a2332"
ACCENT = "#db583b"
SLATE = "#2563eb"
GOLD = "#d97706"
SAGE = "#059669"
PLUM = "#9333ea"
TEAL = "#0891b2"
PALETTE = [SLATE, ACCENT, SAGE, GOLD, PLUM, TEAL]

_COMMON_LAYOUT = dict(
    template="simple_white",
    paper_bgcolor=PAPER,
    plot_bgcolor="white",
    font={"family": "Sora, sans-serif", "size": 13, "color": INK},
)

_COMMON_AXES = dict(
    linecolor="rgba(0,0,0,0.15)",
    gridcolor="rgba(0,0,0,0.04)",
)


def figure_to_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(pio.to_json(fig, pretty=False))


# ── KM & Cox (existing) ────────────────────────────────────────


def build_km_figure(km_result: dict[str, Any], time_unit_label: str = "Months", show_confidence_bands: bool = True) -> dict[str, Any]:
    fig = go.Figure()
    for idx, curve in enumerate(km_result["curves"]):
        color = PALETTE[idx % len(PALETTE)]
        label = curve["group"]
        if show_confidence_bands:
            fig.add_trace(
                go.Scatter(
                    x=curve["timeline"] + curve["timeline"][::-1],
                    y=curve["ci_upper"] + curve["ci_lower"][::-1],
                    fill="toself",
                    fillcolor=color,
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    opacity=0.12,
                    showlegend=False,
                    name=f"{label} CI",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=curve["timeline"],
                y=curve["survival"],
                mode="lines",
                name=label,
                line={"shape": "hv", "width": 3, "color": color},
                hovertemplate=f"{label}<br>{time_unit_label}: %{{x:.2f}}<br>Survival: %{{y:.1%}}<extra></extra>",
            )
        )
        if curve["censor_times"]:
            fig.add_trace(
                go.Scatter(
                    x=curve["censor_times"],
                    y=curve["censor_survival"],
                    mode="markers",
                    name=f"{label} censored",
                    marker={"symbol": "line-ns-open", "size": 10, "color": color, "line": {"width": 2}},
                    hovertemplate=f"{label} censored<br>{time_unit_label}: %{{x:.2f}}<extra></extra>",
                    showlegend=False,
                )
            )

    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
        title={
            "text": "Kaplan-Meier Survival Curve",
            "font": {"family": "Source Serif 4, serif", "size": 24, "color": INK},
            "x": 0.02,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
        hovermode="x unified",
    )
    if km_result.get("test"):
        test = km_result["test"]
        fig.add_annotation(
            text=f"{test['test'].replace('_', ' ').title()} test: p = {test['p_value']:.4g}",
            xref="paper", yref="paper", x=0.98, y=0.98,
            showarrow=False, font={"size": 12, "color": INK},
            align="right", xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", borderpad=4,
        )
    fig.update_xaxes(title=f"Time ({time_unit_label})", **_COMMON_AXES, range=[0, km_result["display_horizon"]])
    fig.update_yaxes(title="Survival probability", tickformat=".0%", range=[0, 1.02], **_COMMON_AXES)
    return figure_to_json(fig)


def build_cox_forest_figure(cox_result: dict[str, Any]) -> dict[str, Any]:
    rows = list(reversed(cox_result["results_table"]))
    labels = [row["Label"] for row in rows]
    hazard_ratios = [row["Hazard ratio"] for row in rows]
    error_plus = [row["CI upper"] - row["Hazard ratio"] for row in rows]
    error_minus = [row["Hazard ratio"] - row["CI lower"] for row in rows]
    colors = [ACCENT if row["P value"] < 0.05 else SLATE for row in rows]

    fig = go.Figure()
    fig.add_vline(x=1.0, line_dash="dash", line_color=INK, opacity=0.6)
    fig.add_trace(
        go.Scatter(
            x=hazard_ratios,
            y=labels,
            mode="markers",
            marker={"size": 12, "color": colors, "line": {"width": 1, "color": INK}},
            error_x={"type": "data", "array": error_plus, "arrayminus": error_minus, "thickness": 1.5, "width": 0},
            hovertemplate="Hazard ratio: %{x:.3f}<extra></extra>",
        )
    )

    stats = cox_result["model_stats"]
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 220, "r": 40, "t": 90, "b": 70},
        title={
            "text": "Cox PH Forest Plot",
            "font": {"family": "Source Serif 4, serif", "size": 24, "color": INK},
            "x": 0.02,
        },
        height=max(420, 90 + 46 * len(rows)),
    )
    stat_parts = [f"N = {stats['n']}", f"events = {stats['events']}"]
    if stats.get("c_index") is not None:
        stat_parts.append(f"C-index = {stats['c_index']:.3f}")
    fig.add_annotation(
        text=", ".join(stat_parts),
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, font={"size": 11, "color": INK},
        align="right", xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.85)", borderpad=4,
    )
    fig.update_xaxes(title="Hazard ratio (log scale)", type="log", **_COMMON_AXES)
    fig.update_yaxes(**_COMMON_AXES)
    return figure_to_json(fig)


# ── Cutpoint scan ───────────────────────────────────────────────


def build_cutpoint_scan_figure(result: dict[str, Any], variable_name: str = "Variable") -> dict[str, Any]:
    scan = result.get("scan_data", [])
    if not scan:
        return figure_to_json(go.Figure())

    cutpoints = [row["cutpoint"] for row in scan]
    statistics = [row["statistic"] for row in scan]
    optimal = result.get("optimal_cutpoint")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cutpoints,
            y=statistics,
            mode="lines",
            line={"width": 2.5, "color": SLATE},
            name="Log-rank statistic",
            hovertemplate=f"{variable_name} = %{{x:.3f}}<br>Chi-square = %{{y:.3f}}<extra></extra>",
        )
    )
    if optimal is not None:
        opt_stat = result.get("statistic", 0)
        adjusted_p = result.get("selection_adjusted_p_value")
        raw_p = result.get("raw_p_value", result.get("p_value"))
        fig.add_trace(
            go.Scatter(
                x=[optimal],
                y=[opt_stat],
                mode="markers",
                marker={"size": 14, "color": ACCENT, "symbol": "star", "line": {"width": 2, "color": INK}},
                name=f"Optimal: {optimal:.3f}",
                hovertemplate=(
                    f"Optimal cutpoint: {optimal:.3f}<br>Chi-square: {opt_stat:.3f}"
                    + (f"<br>Selection-adjusted p: {adjusted_p:.4g}" if adjusted_p is not None else "")
                    + (f"<br>Raw p: {raw_p:.4g}" if raw_p is not None else "")
                    + "<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=optimal, line_dash="dot", line_color=ACCENT, opacity=0.5)

    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 70, "r": 30, "t": 110, "b": 70},
        title={
            "text": f"Optimal Cutpoint Scan: {variable_name}",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
    )
    if optimal is not None:
        p_parts = []
        if adjusted_p is not None:
            p_parts.append(f"Adj. p = {adjusted_p:.4g}")
        if raw_p is not None:
            p_parts.append(f"Raw p = {raw_p:.4g}")
        group_parts = []
        label_below = result.get("label_below_cutpoint")
        label_above = result.get("label_above_cutpoint")
        if label_below is not None:
            group_parts.append(f"<= cutpoint: {label_below}")
        if label_above is not None:
            group_parts.append(f"> cutpoint: {label_above}")
        if group_parts:
            fig.add_annotation(
                text=" | ".join(group_parts),
                xref="paper", yref="paper", x=0.98, y=1.13,
                showarrow=False, font={"size": 12, "color": INK},
                align="right", xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.92)", borderpad=4,
            )
        if p_parts:
            fig.add_annotation(
                text=" | ".join(p_parts),
                xref="paper", yref="paper", x=0.98, y=0.98,
                showarrow=False, font={"size": 14, "color": INK},
                align="right", xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.92)", borderpad=5,
            )
    fig.update_xaxes(title=variable_name, **_COMMON_AXES)
    fig.update_yaxes(title="Log-rank chi-square statistic", **_COMMON_AXES)
    return figure_to_json(fig)


# ── Feature importance ──────────────────────────────────────────


def build_feature_importance_figure(importances: list[dict[str, Any]], model_name: str = "Model") -> dict[str, Any]:
    if not importances:
        return figure_to_json(go.Figure())

    top = importances[:20]
    top = list(reversed(top))
    labels = [row["feature"] for row in top]
    values = [row["importance"] for row in top]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": SLATE, "line": {"width": 0}},
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 200, "r": 30, "t": 80, "b": 60},
        title={
            "text": f"{model_name} Feature Importance",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=max(400, 60 + 28 * len(top)),
    )
    fig.update_xaxes(title="Importance", **_COMMON_AXES)
    fig.update_yaxes(**_COMMON_AXES)
    return figure_to_json(fig)


# ── SHAP ────────────────────────────────────────────────────────


def build_shap_figure(shap_result: dict[str, Any]) -> dict[str, Any]:
    importance = shap_result.get("feature_importance", [])
    if not importance:
        return figure_to_json(go.Figure())

    top = importance[:15]
    top = list(reversed(top))
    labels = [row["feature"] for row in top]
    values = [row["mean_abs_shap"] for row in top]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": ACCENT, "line": {"width": 0}},
            hovertemplate="%{y}: mean|SHAP| = %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 200, "r": 30, "t": 80, "b": 60},
        title={
            "text": "SHAP Feature Importance",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=max(400, 60 + 28 * len(top)),
    )
    fig.update_xaxes(title="Mean |SHAP value|", **_COMMON_AXES)
    fig.update_yaxes(**_COMMON_AXES)
    return figure_to_json(fig)


# ── Model comparison ────────────────────────────────────────────


def build_model_comparison_figure(comparison: dict[str, Any]) -> dict[str, Any]:
    table = comparison.get("comparison_table", [])
    if not table:
        return figure_to_json(go.Figure())

    models = [row["model"] for row in table]
    c_indices = [row.get("c_index") for row in table]
    safe_c = [float(v) if isinstance(v, (int, float)) and v is not None else None for v in c_indices]
    colors = [
        (PALETTE[i % len(PALETTE)] if row.get("comparable_for_ranking", True) else "rgba(148,163,184,0.75)")
        for i, row in enumerate(table)
    ]
    labels = [
        ("NA" if v is None else f"{v:.3f}") + ("*" if not row.get("comparable_for_ranking", True) else "")
        for row, v in zip(table, safe_c, strict=False)
    ]
    hover_text = [
        f"{row['model']}: C-index = {('NA' if value is None else f'{value:.4f}')}<br>Evaluation = {row.get('evaluation_mode')}"
        + ("<br>Excluded from rank ordering" if not row.get("comparable_for_ranking", True) else "")
        for row, value in zip(table, safe_c, strict=False)
    ]

    finite_vals = [v for v in safe_c if v is not None]
    y_max = max(finite_vals) if finite_vals else 1.0
    note = "<br><sup>* apparent-fallback rows shown for transparency and excluded from rank ordering</sup>" if any(
        not row.get("comparable_for_ranking", True) for row in table
    ) else ""

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=models,
            y=safe_c,
            marker={"color": colors, "line": {"width": 1, "color": INK}},
            text=labels,
            textposition="outside",
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Random (0.5)")
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
        title={
            "text": f"Model Comparison (C-index){note}",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=420,
        yaxis_range=[0, y_max * 1.2 if y_max else 1],
    )
    fig.update_xaxes(title="Model", **_COMMON_AXES)
    fig.update_yaxes(title="Concordance Index", **_COMMON_AXES)
    return figure_to_json(fig)


# ── Loss curve (DL) ────────────────────────────────────────────


def build_loss_curve_figure(loss_history: list[float], model_name: str = "Model") -> dict[str, Any]:
    if not loss_history:
        return figure_to_json(go.Figure())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(loss_history) + 1)),
            y=loss_history,
            mode="lines",
            line={"width": 2, "color": TEAL},
            hovertemplate="Epoch %{x}: Loss = %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
        title={
            "text": f"{model_name} Training Loss",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=380,
    )
    fig.update_xaxes(title="Epoch", **_COMMON_AXES)
    fig.update_yaxes(title="Loss", **_COMMON_AXES)
    return figure_to_json(fig)


# ── XAI: Time-dependent importance ────────────────────────────


def build_time_dependent_importance_figure(
    result: dict[str, Any], top_n: int = 8
) -> dict[str, Any]:
    """Heatmap of feature importance over time.

    Parameters
    ----------
    result : dict
        Output of ``compute_time_dependent_importance`` with keys
        ``features``, ``eval_times``, and ``importance_matrix``
        (list-of-lists, shape [n_times, n_features]).
    top_n : int
        Maximum number of features to display.
    """
    features: list[str] = result.get("features", [])
    eval_times: list[float] = result.get("eval_times", [])
    matrix: list[list[float | None]] = result.get("importance_matrix", [])
    orientation = result.get("importance_matrix_orientation", "time_major")

    if not features or not eval_times or not matrix:
        return figure_to_json(go.Figure())

    if orientation == "feature_major":
        matrix = [list(row) for row in zip(*matrix, strict=False)]

    # matrix is time-by-feature. Select top features by mean importance across time.
    means: list[float] = []
    for feat_idx in range(len(features)):
        values = [
            float(row[feat_idx])
            for row in matrix
            if row and feat_idx < len(row) and row[feat_idx] is not None
        ]
        means.append(float(np.mean(values)) if values else -1.0)

    ranked_idx = sorted(range(len(features)), key=lambda idx: means[idx], reverse=True)
    selected_idx = ranked_idx[: min(top_n, len(ranked_idx))]
    selected_features = [features[idx] for idx in selected_idx]
    z = [
        [
            None
            if (not row or feat_idx >= len(row) or row[feat_idx] is None)
            else float(row[feat_idx])
            for row in matrix
        ]
        for feat_idx in selected_idx
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{t:.1f}" for t in eval_times],
            y=selected_features,
            colorscale=[[0, SLATE], [1, ACCENT]],
            hovertemplate="Feature: %{y}<br>Time: %{x}<br>Importance: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 200, "r": 30, "t": 80, "b": 70},
        title={
            "text": "Time-Dependent Feature Importance",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=max(400, 60 + 36 * len(selected_features)),
    )
    fig.update_xaxes(title="Evaluation Time", **_COMMON_AXES)
    fig.update_yaxes(**_COMMON_AXES)
    return figure_to_json(fig)


# ── XAI: Calibration plot ─────────────────────────────────────


def build_calibration_figure(calibration_data: dict[str, Any]) -> dict[str, Any]:
    """Scatter plot of predicted vs observed survival probability.

    Parameters
    ----------
    calibration_data : dict
        Output of ``compute_calibration_data`` with keys
        ``predicted`` and ``observed`` (parallel lists of floats).
    """
    predicted: list[float] = calibration_data.get("predicted", [])
    observed: list[float] = calibration_data.get("observed", [])

    if not predicted or not observed:
        return figure_to_json(go.Figure())

    fig = go.Figure()

    # 45-degree perfect calibration reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "width": 1.5, "color": INK},
            opacity=0.5,
            showlegend=False,
            name="Perfect calibration",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=observed,
            mode="markers",
            marker={"size": 8, "color": SLATE, "line": {"width": 1, "color": INK}},
            name="Calibration",
            hovertemplate="Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
        title={
            "text": "Calibration Plot",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=480,
    )
    fig.update_xaxes(
        title="Predicted survival probability",
        range=[0, 1.02],
        **_COMMON_AXES,
    )
    fig.update_yaxes(
        title="Observed survival probability",
        range=[0, 1.02],
        **_COMMON_AXES,
    )
    return figure_to_json(fig)


# ── XAI: Partial Dependence Plot ──────────────────────────────


def build_pdp_figure(pdp_data: dict[str, Any]) -> dict[str, Any]:
    """Line plot showing how risk score changes as a feature value varies.

    Parameters
    ----------
    pdp_data : dict
        Output with keys ``feature``, ``values``, ``mean_risk``.
    """
    feature: str = pdp_data.get("feature", "Feature")
    values: list[float] = pdp_data.get("values", [])
    mean_risk: list[float] = pdp_data.get("mean_risk", [])

    if not values or not mean_risk:
        return figure_to_json(go.Figure())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values,
            y=mean_risk,
            mode="lines",
            line={"width": 2.5, "color": SLATE},
            hovertemplate=f"{feature} = %{{x:.3f}}<br>Mean risk = %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
        title={
            "text": f"Partial Dependence: {feature}",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=420,
    )
    fig.update_xaxes(title=feature, **_COMMON_AXES)
    fig.update_yaxes(title="Mean predicted risk", **_COMMON_AXES)
    return figure_to_json(fig)

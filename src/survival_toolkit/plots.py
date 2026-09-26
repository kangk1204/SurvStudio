from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PAPER = "#ffffff"
INK = "#1a2332"
ACCENT = "#c94e33"
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

_FEATURE_LABEL_BREAK_PATTERN = re.compile(r"( vs |__|_|:|/|-)")


def figure_to_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(pio.to_json(fig, pretty=False))


def escape_plotly_text(value: Any) -> str:
    """Escape user-supplied text shown in Plotly titles, legends, ticks, or hover values.

    Plotly renders a subset of HTML (``<br>``, ``<b>``, ``<a href>`` ...) in any text,
    so a group label such as ``"<b>High</b>"`` or ``"<a href=...>"`` would change the
    figure. Plotly decodes these entities back to the literal characters.
    """
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_plotly_template_text(value: Any) -> str:
    """Escape user text embedded in a ``hovertemplate``, where ``%{...}`` is also special."""
    return escape_plotly_text(value).replace("%", "&#37;")


def _format_p_value(value: Any) -> str:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return "NA"
    p_value = float(value)
    if p_value < 0.0:
        return "NA"
    if p_value < 1e-16:
        return "<1e-16"
    if p_value < 0.001:
        return "<0.001"
    text = f"{p_value:.3f}"
    # Never let rounding push a p-value across the conventional 0.05 threshold
    # (for example 0.0496 -> "0.050").
    if p_value < 0.05 <= float(text):
        text = f"{p_value:.4f}"
    return text


def _p_value_expression(value: Any, label: str = "p") -> str:
    """Render ``label = value`` or ``label < bound`` without doubled operators."""
    formatted = _format_p_value(value)
    if formatted.startswith("<"):
        return f"{label} < {formatted[1:]}"
    return f"{label} = {formatted}"


def _step_polyline(x_values: list[float], y_values: list[float]) -> tuple[list[float], list[float]]:
    """Expand right-continuous step data into explicit polyline vertices."""
    if not x_values:
        return [], []
    xs: list[float] = [x_values[0]]
    ys: list[float] = [y_values[0]]
    for index in range(1, len(x_values)):
        xs.extend([x_values[index], x_values[index]])
        ys.extend([y_values[index - 1], y_values[index]])
    return xs, ys


def _truncate_label_fragment(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"
    return f"{text[: width - 1].rstrip()}…"


def _wrap_feature_axis_label(label: Any, *, width: int = 26, max_lines: int = 2) -> tuple[str, int]:
    raw = str(label)
    if len(raw) <= width:
        return escape_plotly_text(raw), 1

    tokens: list[str] = []
    cursor = 0
    for match in _FEATURE_LABEL_BREAK_PATTERN.finditer(raw):
        tokens.append(raw[cursor : match.end()])
        cursor = match.end()
    if cursor < len(raw):
        tokens.append(raw[cursor:])
    if not tokens:
        tokens = [raw]

    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{token}"
        if current and len(candidate.strip()) > width:
            lines.append(current.strip())
            current = token.lstrip()
        else:
            current = candidate
    if current.strip():
        lines.append(current.strip())

    if len(lines) > max_lines:
        remainder = "".join(lines[max_lines - 1 :]).strip()
        lines = lines[: max_lines - 1] + [_truncate_label_fragment(remainder, width)]

    lines = [_truncate_label_fragment(line.strip(), width) for line in lines if line.strip()]
    if not lines:
        lines = [_truncate_label_fragment(raw, width)]
    return "<br>".join(escape_plotly_text(line) for line in lines), len(lines)


def _wrap_annotation_text(text: Any, *, width: int = 84, max_lines: int = 3) -> tuple[str, int]:
    raw = " ".join(str(text).split())
    if len(raw) <= width:
        return raw, 1

    words = raw.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        remainder = " ".join(lines[max_lines - 1 :])
        lines = lines[: max_lines - 1] + [_truncate_label_fragment(remainder, width)]

    return "<br>".join(lines), len(lines)


def _feature_plot_axis_layout(
    labels: list[Any],
    *,
    width: int = 26,
    max_lines: int = 2,
) -> tuple[list[str], dict[str, int]]:
    wrapped: list[str] = []
    max_line_chars = 0
    total_lines = 0

    for label in labels:
        wrapped_label, line_count = _wrap_feature_axis_label(label, width=width, max_lines=max_lines)
        wrapped.append(wrapped_label)
        total_lines += line_count
        for line in wrapped_label.split("<br>"):
            max_line_chars = max(max_line_chars, len(line))

    left_margin = min(320, max(200, 70 + max_line_chars * 6))
    height = max(400, 100 + total_lines * 30)
    return wrapped, {"l": left_margin, "r": 30, "t": 80, "b": 60, "height": height}


def _diagnostic_residual_axis_range(
    residual_values: list[float],
    trend_values: list[float],
) -> list[float] | None:
    residual_array = np.asarray([value for value in residual_values if np.isfinite(value)], dtype=float)
    if residual_array.size < 12:
        return None
    trend_array = np.asarray([value for value in trend_values if np.isfinite(value)], dtype=float)
    full_min = float(np.min(residual_array))
    full_max = float(np.max(residual_array))
    full_span = full_max - full_min
    if full_span <= 0:
        return None

    q_low, q_high = np.quantile(residual_array, [0.05, 0.95])
    core_min = min(float(q_low), 0.0, float(np.min(trend_array)) if trend_array.size else 0.0)
    core_max = max(float(q_high), 0.0, float(np.max(trend_array)) if trend_array.size else 0.0)
    core_span = core_max - core_min
    if core_span <= 0 or full_span < core_span * 3.0:
        return None

    padding = max(0.06, core_span * 0.12)
    candidate_range = [core_min - padding, core_max + padding]
    outlier_count = int(np.sum((residual_array < candidate_range[0]) | (residual_array > candidate_range[1])))
    if outlier_count == 0:
        return None
    return candidate_range


def _km_group_color(label: Any, fallback_index: int) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"high", "high risk"}:
        return ACCENT
    if normalized in {"low", "low risk"}:
        return SLATE
    return PALETTE[fallback_index % len(PALETTE)]


def _km_group_dash(label: Any, fallback_index: int) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"high", "high risk", "low", "low risk"}:
        return "solid"
    return ["solid", "dash", "dot", "dashdot"][fallback_index % 4]


# ── KM & Cox (existing) ────────────────────────────────────────


def build_km_figure(km_result: dict[str, Any], time_unit_label: str = "Months", show_confidence_bands: bool = True) -> dict[str, Any]:
    fig = go.Figure()
    confidence_level = float(km_result.get("confidence_level", 0.95) or 0.95)
    confidence_percent = max(1, round(confidence_level * 100))
    unit_template = escape_plotly_template_text(time_unit_label)
    for idx, curve in enumerate(km_result["curves"]):
        label = curve["group"]
        color = _km_group_color(label, idx)
        display_label = escape_plotly_text(label)
        template_label = escape_plotly_template_text(label)
        if show_confidence_bands:
            upper_x, upper_y = _step_polyline(list(curve["timeline"]), list(curve["ci_upper"]))
            lower_x, lower_y = _step_polyline(list(curve["timeline"]), list(curve["ci_lower"]))
            fig.add_trace(
                go.Scatter(
                    x=upper_x + lower_x[::-1],
                    y=upper_y + lower_y[::-1],
                    fill="toself",
                    fillcolor=color,
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    opacity=0.12,
                    showlegend=False,
                    name=f"{display_label} CI",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=curve["timeline"],
                y=curve["survival"],
                mode="lines",
                name=display_label,
                line={"shape": "hv", "width": 3, "color": color, "dash": _km_group_dash(label, idx)},
                hovertemplate=f"{template_label}<br>{unit_template}: %{{x:.2f}}<br>Survival: %{{y:.1%}}<extra></extra>",
            )
        )
        if curve["censor_times"]:
            fig.add_trace(
                go.Scatter(
                    x=curve["censor_times"],
                    y=curve["censor_survival"],
                    mode="markers",
                    name=f"{display_label} censored",
                    marker={"symbol": "line-ns-open", "size": 10, "color": color, "line": {"width": 2}},
                    hovertemplate=f"{template_label} censored<br>{unit_template}: %{{x:.2f}}<extra></extra>",
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
            text=f"{test['test'].replace('_', ' ').title()} test: {_p_value_expression(test['p_value'])}",
            xref="paper", yref="paper", x=0.98, y=0.98,
            showarrow=False, font={"size": 12, "color": INK},
            align="right", xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", borderpad=4,
        )
    elif km_result.get("outcome_informed_group"):
        fig.add_annotation(
            text="Outcome-informed grouping: fresh raw p-value suppressed",
            xref="paper", yref="paper", x=0.98, y=0.98,
            showarrow=False, font={"size": 12, "color": INK},
            align="right", xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", borderpad=4,
        )
    if show_confidence_bands:
        fig.add_annotation(
            text=f"Shaded bands: {confidence_percent}% pointwise CI",
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.98,
            showarrow=False,
            font={"size": 12, "color": INK},
            align="left",
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=4,
        )
    fig.update_xaxes(title=f"Time ({escape_plotly_text(time_unit_label)})", **_COMMON_AXES, range=[0, km_result["display_horizon"]])
    fig.update_yaxes(title="Survival probability", tickformat=".0%", range=[0, 1.02], **_COMMON_AXES)
    return figure_to_json(fig)


def build_cox_forest_figure(cox_result: dict[str, Any]) -> dict[str, Any]:
    raw_rows = list(reversed(cox_result["results_table"]))
    rows = [
        row
        for row in raw_rows
        if isinstance(row.get("Hazard ratio"), (int, float))
        and isinstance(row.get("CI lower"), (int, float))
        and isinstance(row.get("CI upper"), (int, float))
        and np.isfinite(float(row["Hazard ratio"]))
        and np.isfinite(float(row["CI lower"]))
        and np.isfinite(float(row["CI upper"]))
        and float(row["Hazard ratio"]) > 0.0
        and float(row["CI lower"]) > 0.0
        and float(row["CI upper"]) > 0.0
    ]
    labels = [row["Label"] for row in rows]
    display_labels, axis_layout = _feature_plot_axis_layout(labels, width=34, max_lines=3)
    hazard_ratios = [row["Hazard ratio"] for row in rows]
    error_plus = [row["CI upper"] - row["Hazard ratio"] for row in rows]
    error_minus = [row["Hazard ratio"] - row["CI lower"] for row in rows]
    colors = [
        ACCENT
        if isinstance(row.get("P value"), (int, float)) and np.isfinite(float(row["P value"])) and float(row["P value"]) < 0.05
        else SLATE
        for row in rows
    ]

    fig = go.Figure()
    fig.add_vline(x=1.0, line_dash="solid", line_color=INK, line_width=1.5, opacity=0.75)
    fig.add_trace(
        go.Scatter(
            x=hazard_ratios,
            y=labels,
            mode="markers",
            customdata=[escape_plotly_text(label) for label in labels],
            marker={"size": 12, "color": colors, "line": {"width": 1, "color": INK}},
            error_x={"type": "data", "array": error_plus, "arrayminus": error_minus, "thickness": 1.5, "width": 0},
            hovertemplate="%{customdata}<br>Hazard ratio: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": axis_layout["l"], "r": 40, "t": 28, "b": 70},
        title={
            "text": "",
            "font": {"family": "Source Serif 4, serif", "size": 24, "color": INK},
            "x": 0.02,
        },
        height=max(420, axis_layout["height"]),
    )
    fig.add_annotation(
        text="Points = HR; whiskers = 95% Wald CI",
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.99,
        showarrow=False,
        font={"size": 12, "color": INK},
        align="right",
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
        borderpad=4,
    )
    if not rows:
        fig.add_annotation(
            text="No finite hazard ratios are available to plot.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": INK},
            align="center",
        )
    fig.update_xaxes(title="Hazard ratio (log scale)", type="log", **_COMMON_AXES)
    fig.update_yaxes(
        automargin=True,
        tickmode="array",
        tickvals=labels,
        ticktext=display_labels,
        **_COMMON_AXES,
    )
    return figure_to_json(fig)


def build_cox_diagnostics_figure(cox_result: dict[str, Any]) -> dict[str, Any]:
    diagnostic_series = list(cox_result.get("diagnostics_plot_data") or [])
    if not diagnostic_series:
        return figure_to_json(go.Figure())

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
        raw_p = item.get("p_value")
        raw_rho = item.get("schoenfeld_rho")
        safe_p = float(raw_p) if isinstance(raw_p, (int, float)) and np.isfinite(float(raw_p)) else float("inf")
        safe_rho = abs(float(raw_rho)) if isinstance(raw_rho, (int, float)) and np.isfinite(float(raw_rho)) else -1.0
        return (safe_p, -safe_rho, str(item.get("term") or ""))

    panels = sorted(diagnostic_series, key=_sort_key)[:4]
    panel_count = max(1, len(panels))
    cols = 2 if panel_count > 1 else 1
    rows = int(np.ceil(panel_count / cols))
    wrapped_titles: list[str] = []
    max_title_lines = 1
    for panel in panels:
        wrapped_label, line_count = _wrap_feature_axis_label(panel.get("term") or "Term", width=28, max_lines=2)
        wrapped_titles.append(wrapped_label)
        max_title_lines = max(max_title_lines, line_count)
    fig = make_subplots(
      rows=rows,
      cols=cols,
      subplot_titles=wrapped_titles,
      horizontal_spacing=0.14,
      vertical_spacing=0.3 if rows > 1 else 0.2,
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 13, "color": INK, "family": "Sora, sans-serif"}
        annotation.yshift = 12
        annotation.bgcolor = "rgba(255,255,255,0.96)"
        annotation.borderpad = 3

    clipped_for_readability = False
    for panel_index, panel in enumerate(panels):
        row = (panel_index // cols) + 1
        col = (panel_index % cols) + 1
        x_values = [float(value) for value in panel.get("log_time") or [] if value is not None]
        y_values = [float(value) for value in panel.get("residual") or [] if value is not None]
        trend_x = [float(value) for value in panel.get("trend_log_time") or [] if value is not None]
        trend_y = [float(value) for value in panel.get("trend_residual") or [] if value is not None]
        rho = panel.get("schoenfeld_rho")
        p_value = panel.get("p_value")
        marker_color = ACCENT if isinstance(p_value, (int, float)) and np.isfinite(float(p_value)) and float(p_value) < 0.05 else SLATE

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=escape_plotly_text(panel.get("term") or "Residuals"),
                marker={"size": 6, "color": marker_color, "opacity": 0.55},
                hovertemplate=(
                    f"{escape_plotly_template_text(panel.get('term') or 'Term')}<br>log(time): %{{x:.3f}}<br>Scaled Schoenfeld residual: %{{y:.3f}}"
                    + (f"<br>rho={float(rho):.3f}" if isinstance(rho, (int, float)) and np.isfinite(float(rho)) else "")
                    + (f"<br>{_p_value_expression(p_value)}" if isinstance(p_value, (int, float)) and np.isfinite(float(p_value)) else "")
                    + "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        if trend_x and trend_y:
            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode="lines",
                    line={"width": 2.5, "color": marker_color},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="rgba(90, 103, 118, 0.7)", row=row, col=col)
        robust_range = _diagnostic_residual_axis_range(y_values, trend_y)
        clipped_for_readability = clipped_for_readability or (robust_range is not None)
        fig.update_xaxes(
            title="log(time)" if row == rows else "",
            title_standoff=14,
            row=row,
            col=col,
            **_COMMON_AXES,
        )
        fig.update_yaxes(
            title="Scaled residual" if col == 1 else "",
            title_standoff=12,
            row=row,
            col=col,
            range=robust_range,
            **_COMMON_AXES,
        )

    top_margin = 84 + ((max_title_lines - 1) * 18)
    if clipped_for_readability:
        subtitle_text, subtitle_lines = _wrap_annotation_text(
            "Screening view only: one or more residual panels were clipped for readability. "
            "Inspect hover values and raw outputs before drawing PH conclusions.",
            width=86,
            max_lines=2,
        )
        fig.add_annotation(
            text=subtitle_text,
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.12,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            align="left",
            font={"size": 12, "color": INK, "family": "Sora, sans-serif"},
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=4,
        )
        top_margin += 24 + ((subtitle_lines - 1) * 16)
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 60, "r": 30, "t": top_margin, "b": 68},
        title={"text": ""},
        height=max(400, rows * 300 + ((max_title_lines - 1) * 24)),
    )
    return figure_to_json(fig)


def build_cox_martingale_figure(cox_result: dict[str, Any]) -> dict[str, Any]:
    diagnostic_series = list(cox_result.get("martingale_plot_data") or [])
    if not diagnostic_series:
        return figure_to_json(go.Figure())

    panels = diagnostic_series[:4]
    panel_count = max(1, len(panels))
    cols = 2 if panel_count > 1 else 1
    rows = int(np.ceil(panel_count / cols))
    wrapped_titles: list[str] = []
    max_title_lines = 1
    for panel in panels:
        wrapped_label, line_count = _wrap_feature_axis_label(panel.get("term") or "Covariate", width=28, max_lines=2)
        wrapped_titles.append(wrapped_label)
        max_title_lines = max(max_title_lines, line_count)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=wrapped_titles,
        horizontal_spacing=0.14,
        vertical_spacing=0.22,
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 13, "color": INK, "family": "Sora, sans-serif"}
        annotation.yshift = 4

    clipped_for_readability = False
    for panel_index, panel in enumerate(panels):
        row = (panel_index // cols) + 1
        col = (panel_index % cols) + 1
        x_values = [float(value) for value in panel.get("value") or [] if value is not None]
        y_values = [float(value) for value in panel.get("residual") or [] if value is not None]
        trend_x = [float(value) for value in panel.get("trend_value") or [] if value is not None]
        trend_y = [float(value) for value in panel.get("trend_residual") or [] if value is not None]

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=escape_plotly_text(panel.get("term") or "Residuals"),
                marker={"size": 6, "color": TEAL, "opacity": 0.55},
                hovertemplate=(
                    f"{escape_plotly_template_text(panel.get('term') or 'Covariate')}<br>Value: %{{x:.3f}}<br>Martingale residual: %{{y:.3f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        if trend_x and trend_y:
            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode="lines",
                    line={"width": 2.5, "color": "rgba(13, 148, 136, 0.95)"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="rgba(90, 103, 118, 0.7)", row=row, col=col)
        robust_range = _diagnostic_residual_axis_range(y_values, trend_y)
        clipped_for_readability = clipped_for_readability or (robust_range is not None)
        fig.update_xaxes(title=escape_plotly_text(panel.get("term") or "Covariate"), row=row, col=col, **_COMMON_AXES)
        fig.update_yaxes(
            title="Martingale residual",
            row=row,
            col=col,
            range=robust_range,
            **_COMMON_AXES,
        )

    top_margin = 72 + ((max_title_lines - 1) * 18)
    if clipped_for_readability:
        subtitle_text, subtitle_lines = _wrap_annotation_text(
            "Screening view only: one or more martingale residual panels were clipped for readability. "
            "Inspect hover values and raw outputs before judging linearity.",
            width=86,
            max_lines=2,
        )
        fig.add_annotation(
            text=subtitle_text,
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.12,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            align="left",
            font={"size": 12, "color": INK, "family": "Sora, sans-serif"},
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=4,
        )
        top_margin += 24 + ((subtitle_lines - 1) * 16)
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 60, "r": 30, "t": top_margin, "b": 68},
        title={"text": ""},
        height=max(360, rows * 280 + ((max_title_lines - 1) * 24)),
    )
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
            hovertemplate=f"{escape_plotly_template_text(variable_name)} = %{{x:.3f}}<br>Chi-square = %{{y:.3f}}<extra></extra>",
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
                    + (f"<br>Selection-adjusted p: {_format_p_value(adjusted_p)}" if adjusted_p is not None else "")
                    + (f"<br>Raw p: {_format_p_value(raw_p)}" if raw_p is not None else "")
                    + "<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=optimal, line_dash="dot", line_color=ACCENT, opacity=0.5)

    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 70, "r": 30, "t": 110, "b": 70},
        title={
            "text": f"Optimal Cutpoint Scan: {escape_plotly_text(variable_name)}",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
    )
    if optimal is not None:
        p_parts = []
        if adjusted_p is not None:
            p_parts.append(_p_value_expression(adjusted_p, "Adj. p"))
        if raw_p is not None:
            p_parts.append(_p_value_expression(raw_p, "Raw p"))
        group_parts = []
        label_below = result.get("label_below_cutpoint")
        label_above = result.get("label_above_cutpoint")
        if label_below is not None:
            group_parts.append(f"&lt;= cutpoint: {escape_plotly_text(label_below)}")
        if label_above is not None:
            group_parts.append(f"&gt; cutpoint: {escape_plotly_text(label_above)}")
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
    fig.update_xaxes(title=escape_plotly_text(variable_name), **_COMMON_AXES)
    fig.update_yaxes(title="Log-rank chi-square statistic", **_COMMON_AXES)
    return figure_to_json(fig)


# ── Feature importance ──────────────────────────────────────────


def build_feature_importance_figure(
    importances: list[dict[str, Any]],
    model_name: str = "Model",
    *,
    title_label: str = "Feature Importance",
) -> dict[str, Any]:
    if not importances:
        return figure_to_json(go.Figure())

    top = importances[:20]
    top = list(reversed(top))
    labels = [row["feature"] for row in top]
    display_labels, axis_layout = _feature_plot_axis_layout(labels)
    values = [row["importance"] for row in top]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            customdata=[escape_plotly_text(label) for label in labels],
            marker={"color": SLATE, "line": {"width": 0}},
            hovertemplate="%{customdata}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={k: v for k, v in axis_layout.items() if k != "height"},
        title={
            "text": f"{model_name} {title_label}",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=axis_layout["height"],
    )
    fig.update_xaxes(title="Importance", **_COMMON_AXES)
    fig.update_yaxes(
        automargin=True,
        tickmode="array",
        tickvals=labels,
        ticktext=display_labels,
        **_COMMON_AXES,
    )
    return figure_to_json(fig)


# ── SHAP ────────────────────────────────────────────────────────


def build_shap_figure(shap_result: dict[str, Any]) -> dict[str, Any]:
    importance = shap_result.get("feature_importance", [])
    if not importance:
        return figure_to_json(go.Figure())
    method = str(shap_result.get("method", "tree"))
    safe_mode = bool(shap_result.get("safe_mode"))
    title_text = "SHAP Feature Importance"
    if method == "kernel":
        title_text = "Approximate SHAP Screening Importance"
    if safe_mode:
        title_text = "Reduced-Feature SHAP Screening Importance"

    subtitle_text = ""
    subtitle_lines = 0
    if safe_mode:
        companion = shap_result.get("companion_model") or {}
        subtitle_text, subtitle_lines = _wrap_annotation_text(
            "SHAP safe mode refit a reduced companion model for explanation only: "
            f"{companion.get('selected_feature_count_raw', 'NA')} raw features / "
            f"{companion.get('selected_feature_count_encoded', 'NA')} encoded features. "
            "Performance metrics above still belong to the original full model fit.",
            width=88,
            max_lines=3,
        )

    top = importance[:15]
    top = list(reversed(top))
    labels = [row["feature"] for row in top]
    display_labels, axis_layout = _feature_plot_axis_layout(labels)
    values = [row["mean_abs_shap"] for row in top]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            customdata=[escape_plotly_text(label) for label in labels],
            marker={"color": ACCENT, "line": {"width": 0}},
            hovertemplate="%{customdata}: mean|SHAP| = %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={
            **{k: v for k, v in axis_layout.items() if k != "height"},
            "t": max(108, int(axis_layout.get("t", 32)) + 72 + ((subtitle_lines - 1) * 18 if subtitle_lines else 0)),
        },
        height=axis_layout["height"],
    )
    fig.add_annotation(
        text=title_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=1.15 + ((subtitle_lines - 1) * 0.03 if subtitle_lines else 0),
        showarrow=False,
        font={"family": "Source Serif 4, serif", "size": 22, "color": INK},
        align="left",
        xanchor="left",
        yanchor="bottom",
    )
    if subtitle_text:
        fig.add_annotation(
            text=subtitle_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=1.04 + ((subtitle_lines - 1) * 0.025),
            showarrow=False,
            font={"size": 12, "color": INK},
            align="left",
            xanchor="left",
            yanchor="bottom",
        )
    fig.update_xaxes(title="Mean |SHAP value|", **_COMMON_AXES)
    fig.update_yaxes(
        automargin=True,
        tickmode="array",
        tickvals=labels,
        ticktext=display_labels,
        **_COMMON_AXES,
    )
    return figure_to_json(fig)


# ── Model comparison ────────────────────────────────────────────


def build_model_comparison_figure(comparison: dict[str, Any]) -> dict[str, Any]:
    table = comparison.get("comparison_table", [])
    if not table:
        return figure_to_json(go.Figure())

    models = [row["model"] for row in table]
    c_indices = [row.get("c_index") for row in table]
    safe_c = [
        (float(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(float(v)) else None)
        for v in c_indices
    ]
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
            cliponaxis=False,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="solid", line_color="rgba(51,65,85,0.8)", line_width=1.5, opacity=0.85)
    fig.add_annotation(
        text="Reference (0.5)",
        xref="paper",
        yref="y",
        x=0.02,
        y=0.5,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 12, "color": "rgba(71,85,105,0.95)"},
        bgcolor="rgba(255,255,255,0.88)",
        borderpad=3,
        yshift=6,
    )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 80, "r": 30, "t": 90, "b": 60},
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


def build_loss_curve_figure(
    loss_history: list[float],
    model_name: str = "Model",
    monitor_loss_history: list[float] | None = None,
    best_monitor_epoch: int | None = None,
    epochs_trained: int | None = None,
    max_epochs_requested: int | None = None,
    stopped_early: bool | None = None,
    monitor_label: str = "Monitor loss",
    monitor_goal: str = "min",
) -> dict[str, Any]:
    if not loss_history:
        return figure_to_json(go.Figure())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(loss_history) + 1)),
            y=loss_history,
            mode="lines",
            line={"width": 2, "color": TEAL},
            hovertemplate="Epoch %{x}: Training loss = %{y:.4f}<extra></extra>",
            name="Training loss",
        )
    )
    if monitor_loss_history:
        if best_monitor_epoch is None and monitor_loss_history:
            best_monitor_epoch = (
                int(np.argmin(np.asarray(monitor_loss_history, dtype=float))) + 1
                if monitor_goal == "min"
                else int(np.argmax(np.asarray(monitor_loss_history, dtype=float))) + 1
            )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(monitor_loss_history) + 1)),
                y=monitor_loss_history,
                mode="lines",
                line={"width": 2, "color": ACCENT},
                hovertemplate=f"Epoch %{{x}}: {monitor_label} = %{{y:.4f}}<extra></extra>",
                name=monitor_label,
            )
        )
        if best_monitor_epoch is not None and best_monitor_epoch >= 1:
            fig.add_vline(
                x=best_monitor_epoch,
                line_dash="dash",
                line_color=GOLD,
                line_width=1.5,
                opacity=0.9,
            )
            fig.add_annotation(
                x=best_monitor_epoch,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"Best monitor epoch: {best_monitor_epoch}",
                showarrow=False,
                yanchor="bottom",
                font={"size": 12, "color": INK},
                bgcolor="rgba(255,255,255,0.88)",
                borderpad=3,
            )
    status_text = None
    if stopped_early and epochs_trained:
        status_text = f"Stopped early at epoch {epochs_trained}"
    elif max_epochs_requested and epochs_trained and epochs_trained >= max_epochs_requested:
        status_text = f"Trained to max epoch ({max_epochs_requested})"
    elif epochs_trained:
        status_text = f"Trained for {epochs_trained} epoch(s)"
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
        title={
            "text": (
                f"{model_name} Training Loss and {monitor_label}"
                if monitor_loss_history
                else f"{model_name} Training Loss"
            ),
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=380,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
    )
    if status_text:
        fig.add_annotation(
            text=status_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font={"size": 12, "color": INK},
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=4,
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
        matrix = [list(row) for row in zip(*matrix, strict=True)]

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
    selected_features = [escape_plotly_text(features[idx]) for idx in selected_idx]
    z = [
        [
            None
            if (not row or feat_idx >= len(row) or row[feat_idx] is None)
            else float(row[feat_idx])
            for row in matrix
        ]
        for feat_idx in selected_idx
    ]

    # Category labels must stay unique; otherwise Plotly merges columns whose
    # times collapse to the same rounded text (for example 1.21 and 1.24).
    time_labels: list[str] = []
    for decimals in range(1, 7):
        time_labels = [f"{float(t):.{decimals}f}" for t in eval_times]
        if len(set(time_labels)) == len(time_labels):
            break
    else:
        time_labels = [f"{float(t):.6g} [{idx + 1}]" for idx, t in enumerate(eval_times)]

    importance_label = str(result.get("importance_label") or "Importance")
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=time_labels,
            y=selected_features,
            colorscale=[[0, SLATE], [1, ACCENT]],
            colorbar={"title": {"text": escape_plotly_text(importance_label)}},
            hovertemplate=(
                "Feature: %{y}<br>Time: %{x}<br>"
                + escape_plotly_template_text(importance_label)
                + ": %{z:.4f}<extra></extra>"
            ),
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
    feature: str = str(pdp_data.get("feature", "Feature"))
    feature_text = escape_plotly_text(feature)
    feature_template = escape_plotly_template_text(feature)
    values: list[Any] = pdp_data.get("values", [])
    mean_risk: list[float] = pdp_data.get("mean_risk", [])
    feature_type = str(pdp_data.get("feature_type", "numeric"))

    if not values or not mean_risk:
        return figure_to_json(go.Figure())

    fig = go.Figure()
    if feature_type == "categorical":
        fig.add_trace(
            go.Bar(
                x=[escape_plotly_text(value) for value in values],
                y=mean_risk,
                marker={"color": SLATE, "line": {"color": INK, "width": 0.4}},
                hovertemplate=f"{feature_template} = %{{x}}<br>Mean risk = %{{y:.4f}}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=values,
                y=mean_risk,
                mode="lines",
                line={"width": 2.5, "color": SLATE},
                hovertemplate=f"{feature_template} = %{{x:.3f}}<br>Mean risk = %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
        title={
            "text": f"Partial Dependence: {feature_text}",
            "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK},
            "x": 0.02,
        },
        height=420,
    )
    fig.update_xaxes(title=feature_text, **_COMMON_AXES)
    fig.update_yaxes(title="Mean predicted risk", **_COMMON_AXES)
    return figure_to_json(fig)


# ── Marker evaluation ─────────────────────────────────────────

MARKER_TIER_COLORS = {
    "robust": SAGE,
    "suggestive": GOLD,
    "marginal only": PLUM,
    "not supported": "rgba(148,163,184,0.75)",
}


def _marker_layout(fig: go.Figure, title: str, *, height: int, left: int = 70) -> None:
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": left, "r": 30, "t": 80, "b": 70},
        title={"text": title, "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK}, "x": 0.02},
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )


def build_marker_stability_figure(result: dict[str, Any]) -> dict[str, Any]:
    """Each marker's selection frequency against its direction consistency over the subsamples (primary
    lens), coloured by tier, with the robust-tier thresholds."""
    primary = str(result.get("primary_lens", "marginal"))
    settings = result.get("settings") or {}
    fig = go.Figure()
    for tier, color in MARKER_TIER_COLORS.items():
        members = [
            row
            for row in result.get("marker_table", [])
            if row.get("tier") == tier
            and isinstance(row.get(primary), dict)
            and row[primary].get("selection_frequency") is not None
            and row[primary].get("direction_consistency") is not None
        ]
        if not members:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row[primary]["selection_frequency"] for row in members],
                y=[row[primary]["direction_consistency"] for row in members],
                mode="markers",
                name=tier,
                marker={"size": 9, "color": color, "line": {"width": 0.6, "color": INK}},
                customdata=[[escape_plotly_text(row["marker"]), escape_plotly_text(_format_p_value(row[primary].get("p_fwer")))] for row in members],
                hovertemplate="%{customdata[0]}<br>Selected in %{x:.0%} of subsamples<br>Same direction in %{y:.0%}<br>Family-wise p = %{customdata[1]}<extra></extra>",
            )
        )
    frequency = float(settings.get("robust_frequency", 0.5))
    direction = float(settings.get("robust_direction", 0.9))
    fig.add_vline(x=frequency, line_dash="dash", line_color=INK, line_width=1, opacity=0.5)
    fig.add_hline(y=direction, line_dash="dash", line_color=INK, line_width=1, opacity=0.5)
    fig.add_annotation(
        text=f"Robust: family-wise p ≤ {float(settings.get('alpha', 0.05)):g}, selected in ≥ {frequency:.0%}, same direction in ≥ {direction:.0%}",
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.02,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"size": 12, "color": INK},
        bgcolor="rgba(255,255,255,0.85)",
        borderpad=4,
    )
    _marker_layout(fig, "Marker Stability Across Subsamples", height=460)
    fig.update_xaxes(title="Selection frequency", range=[-0.02, 1.02], tickformat=".0%", **_COMMON_AXES)
    fig.update_yaxes(title="Direction consistency", range=[-0.02, 1.02], tickformat=".0%", **_COMMON_AXES)
    return figure_to_json(fig)


def build_marker_rank_figure(result: dict[str, Any], *, top: int = 25) -> dict[str, Any]:
    """Median rank and 95% rank interval over the subsamples for the strongest markers (primary lens)."""
    primary = str(result.get("primary_lens", "marginal"))
    rows = [
        row
        for row in result.get("marker_table", [])
        if isinstance(row.get(primary), dict)
        and row[primary].get("median_rank") is not None
        and None not in (row[primary].get("rank_interval") or [None])
    ]
    rows = sorted(rows, key=lambda row: row[primary]["median_rank"])[:top]
    rows.reverse()
    labels = [str(row["marker"]) for row in rows]
    display_labels, axis_layout = _feature_plot_axis_layout(labels, width=30, max_lines=2)
    fig = go.Figure()
    for tier, color in MARKER_TIER_COLORS.items():
        members = [row for row in rows if row.get("tier") == tier]
        if not members:
            continue
        fig.add_trace(
            go.Scatter(
                x=[row[primary]["median_rank"] for row in members],
                y=[str(row["marker"]) for row in members],
                mode="markers",
                name=tier,
                marker={"size": 10, "color": color, "line": {"width": 0.8, "color": INK}},
                error_x={
                    "type": "data",
                    "array": [row[primary]["rank_interval"][1] - row[primary]["median_rank"] for row in members],
                    "arrayminus": [row[primary]["median_rank"] - row[primary]["rank_interval"][0] for row in members],
                    "thickness": 1.5,
                    "width": 0,
                    "color": color,
                },
                customdata=[
                    [escape_plotly_text(row["marker"]), row[primary]["rank_interval"][0], row[primary]["rank_interval"][1]]
                    for row in members
                ],
                hovertemplate="%{customdata[0]}<br>Median rank %{x:.0f} (95%: %{customdata[1]:.0f} to %{customdata[2]:.0f})<extra></extra>",
            )
        )
    if not rows:
        fig.add_annotation(
            text="Rank intervals need at least one subsample.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": INK},
        )
    _marker_layout(fig, "Rank Uncertainty of the Strongest Markers", height=max(420, axis_layout["height"]), left=axis_layout["l"])
    fig.update_xaxes(title="Rank across subsamples (1 = strongest)", **_COMMON_AXES)
    fig.update_yaxes(automargin=True, tickmode="array", tickvals=labels, ticktext=display_labels, **_COMMON_AXES)
    return figure_to_json(fig)


def build_marker_replication_figure(validation: dict[str, Any]) -> dict[str, Any]:
    """External hazard ratio of each locked marker, coloured by whether it replicated."""
    rows = []
    for row in validation.get("markers", []):
        tested = row.get("adjusted") or row.get("marginal")
        if tested and all(isinstance(tested.get(key), (int, float)) and tested[key] > 0 for key in ("hazard_ratio", "ci_lower", "ci_upper")):
            rows.append((row, tested))
    rows.reverse()
    labels = [str(row["marker"]) for row, _ in rows]
    display_labels, axis_layout = _feature_plot_axis_layout(labels, width=30, max_lines=2)
    groups = (
        ("replicated", SAGE, lambda row: row.get("replicated")),
        ("same direction, not significant", GOLD, lambda row: not row.get("replicated") and row.get("same_direction")),
        ("opposite direction", ACCENT, lambda row: not row.get("same_direction")),
    )
    fig = go.Figure()
    fig.add_vline(x=1.0, line_color=INK, line_width=1.5, opacity=0.75)
    for name, color, belongs in groups:
        members = [(row, tested) for row, tested in rows if belongs(row)]
        if not members:
            continue
        fig.add_trace(
            go.Scatter(
                x=[tested["hazard_ratio"] for _, tested in members],
                y=[str(row["marker"]) for row, _ in members],
                mode="markers",
                name=name,
                marker={"size": 11, "color": color, "line": {"width": 1, "color": INK}},
                error_x={
                    "type": "data",
                    "array": [tested["ci_upper"] - tested["hazard_ratio"] for _, tested in members],
                    "arrayminus": [tested["hazard_ratio"] - tested["ci_lower"] for _, tested in members],
                    "thickness": 1.5,
                    "width": 0,
                },
                customdata=[
                    [escape_plotly_text(row["marker"]), escape_plotly_text(_format_p_value(row.get("replication_p_holm")))]
                    for row, _ in members
                ],
                hovertemplate="%{customdata[0]}<br>HR %{x:.3f}<br>Replication p (Holm) = %{customdata[1]}<extra></extra>",
            )
        )
    if not rows:
        fig.add_annotation(
            text="No finite external hazard ratios to plot.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": INK},
        )
    _marker_layout(fig, "Marker Replication in the External Cohort", height=max(380, axis_layout["height"]), left=axis_layout["l"])
    fig.update_xaxes(title="Hazard ratio (log scale)", type="log", **_COMMON_AXES)
    fig.update_yaxes(automargin=True, tickmode="array", tickvals=labels, ticktext=display_labels, **_COMMON_AXES)
    return figure_to_json(fig)

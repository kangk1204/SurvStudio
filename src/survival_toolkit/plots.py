from __future__ import annotations

import json
import re
import textwrap
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from survival_toolkit.reporting import _count, signature_fit_failed, signature_is_clinical_only

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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return "NA"
    p_value = float(value)
    if p_value < 0.0:
        return "NA"
    if p_value < 1e-16:
        return "<1e-16"
    if p_value < 0.001:
        return "<0.001"
    text = f"{p_value:.3f}"
    # Never let rounding push a p-value across the conventional 0.05 threshold: show more
    # digits until the printed value stays below it (0.0496 -> "0.0496", 0.04996 -> "0.04996").
    digits = 3
    while p_value < 0.05 <= float(text):
        digits += 1
        if digits > 17:
            return "<0.05"
        text = f"{p_value:.{digits}f}"
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


def _log_axis_ticks(values: list[Any]) -> dict[str, Any]:
    """Tick values for a hazard-ratio axis on a log scale.

    Plotly labels only the leading digit of minor log ticks ("4 5 6 ... 1 2 3"), which reads badly on a
    forest plot; this picks round values (1-2-5, or finer for narrow ranges) and labels them in full.
    """
    finite = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(value) and value > 0]
    if not finite:
        return {}
    low, high = min(finite), max(finite)
    decades = float(np.log10(high / low))
    if decades > 3:
        mantissas: tuple[float, ...] = (1,)
    elif decades > 1.2:
        mantissas = (1, 2, 5)
    elif decades > 0.4:
        mantissas = (1, 1.5, 2, 3, 5, 7)
    else:
        mantissas = (1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)
    ticks = []
    for exponent in range(int(np.floor(np.log10(low))) - 1, int(np.ceil(np.log10(high))) + 1):
        for mantissa in mantissas:
            value = float(f"{mantissa * 10.0 ** exponent:.6g}")
            if low / 1.25 <= value <= high * 1.25:
                ticks.append(value)
    if sum(low <= tick <= high for tick in ticks) < 3:
        return {}  # a narrow range: Plotly's own ticks are labelled in full there
    return {"tickmode": "array", "tickvals": ticks, "ticktext": [f"{tick:g}" for tick in ticks]}


def _as_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_pairs(x_values: Any, y_values: Any) -> tuple[list[float], list[float]]:
    """The points whose x and y are both finite numbers, kept together so a gap in one list cannot
    shift the other."""
    xs: list[float] = []
    ys: list[float] = []
    for x_value, y_value in zip(x_values or [], y_values or []):
        x_number, y_number = _as_finite_float(x_value), _as_finite_float(y_value)
        if x_number is not None and y_number is not None:
            xs.append(x_number)
            ys.append(y_number)
    return xs, ys


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


_KM_RESERVED_COLORS = {"high": ACCENT, "high risk": ACCENT, "low": SLATE, "low risk": SLATE}


def _km_group_color(label: Any, fallback_index: int) -> str:
    normalized = str(label or "").strip().lower()
    return _KM_RESERVED_COLORS.get(normalized, PALETTE[fallback_index % len(PALETTE)])


def _km_group_colors(labels: list[Any]) -> list[str]:
    """One colour per group: red for High and blue for Low whatever their order, and the other groups
    cycle through the palette without those two colours when a High or Low group is on the plot (so a
    three-group Intermediate curve is not drawn in High's red)."""
    normalized = [str(label or "").strip().lower() for label in labels]
    reserved = {_KM_RESERVED_COLORS[name] for name in normalized if name in _KM_RESERVED_COLORS}
    if not reserved:
        return [_km_group_color(label, index) for index, label in enumerate(labels)]
    others = [color for color in PALETTE if color not in reserved]
    colors = []
    next_other = 0
    for name in normalized:
        if name in _KM_RESERVED_COLORS:
            colors.append(_KM_RESERVED_COLORS[name])
        else:
            colors.append(others[next_other % len(others)])
            next_other += 1
    return colors


def _km_group_dash(label: Any, fallback_index: int) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"high", "high risk", "low", "low risk"}:
        return "solid"
    return ["solid", "dash", "dot", "dashdot"][fallback_index % 4]


# The palette and the four dash patterns give at most 24 colour-dash pairs (four colours beside High and Low), and a
# plot can hold 50 groups: a curve that repeats an earlier curve's pair gets a marker symbol of its own. Twelve
# symbols cover the worst case, 48 groups beside High and Low sharing four pairs.
_KM_REPEAT_SYMBOLS = (
    "circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star", "hexagon", "pentagon", "hourglass", "bowtie",
)


def _km_group_symbols(styles: list[tuple[str, str]]) -> list[str | None]:
    """No marker for the first curve of each colour and dash, then a different symbol for each repeat of that pair."""
    uses: dict[tuple[str, str], int] = {}
    symbols: list[str | None] = []
    for style in styles:
        repeat = uses.get(style, 0)
        uses[style] = repeat + 1
        symbols.append(None if repeat == 0 else _KM_REPEAT_SYMBOLS[(repeat - 1) % len(_KM_REPEAT_SYMBOLS)])
    return symbols


# Names of the weighted tests for results without the analysis's own label (test_p_value_label).
_KM_TEST_LABELS = {
    "logrank": "log-rank",
    "gehan_breslow": "Gehan-Breslow",
    "tarone_ware": "Tarone-Ware",
    "fleming_harrington": "Fleming-Harrington",
}


# ── KM & Cox (existing) ────────────────────────────────────────


def build_km_figure(km_result: dict[str, Any], time_unit_label: str = "Months", show_confidence_bands: bool = True) -> dict[str, Any]:
    fig = go.Figure()
    confidence_level = float(km_result.get("confidence_level", 0.95) or 0.95)
    # :g keeps a 97.5% band from being printed as 98%.
    confidence_percent = f"{confidence_level * 100:g}"
    unit_template = escape_plotly_template_text(time_unit_label)
    curve_colors = _km_group_colors([curve["group"] for curve in km_result["curves"]])
    curve_dashes = [_km_group_dash(curve["group"], index) for index, curve in enumerate(km_result["curves"])]
    curve_symbols = _km_group_symbols(list(zip(curve_colors, curve_dashes)))
    color_by_group = {str(curve["group"]): color for curve, color in zip(km_result["curves"], curve_colors)}
    for idx, curve in enumerate(km_result["curves"]):
        label = curve["group"]
        color = curve_colors[idx]
        symbol = curve_symbols[idx]
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
                mode="lines" if symbol is None else "lines+markers",
                name=display_label,
                line={"shape": "hv", "width": 3, "color": color, "dash": curve_dashes[idx]},
                # A few markers along the curve are enough to tell it apart.
                **({} if symbol is None else {"marker": {"symbol": symbol, "size": 8, "color": color, "maxdisplayed": 12}}),
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

    risk_times, risk_rows = _km_risk_rows(km_result)
    # The numbers at risk sit under the time axis, one row per group, as journals print them. Group
    # labels end left of the time-0 counts, which are centred on the axis origin.
    row_height = 20
    risk_top = 78
    labels = [escape_plotly_text(row["group"]) for row in risk_rows]
    label_shift = 10 + 4 * max((len(str(row["counts"][0])) for row in risk_rows), default=0)
    label_width = 7 * max((len(str(row["group"])) for row in risk_rows), default=0)
    left_margin = max(70, min(240, label_shift + max(label_width, 100) + 8)) if risk_rows else 70
    bottom_margin = risk_top + row_height * len(risk_rows) + 16 if risk_rows else 70
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": left_margin, "r": 30, "t": 80, "b": bottom_margin},
        height=460 + (bottom_margin - 70),
        title={
            "text": "Kaplan-Meier Survival Curve",
            "font": {"family": "Source Serif 4, serif", "size": 24, "color": INK},
            "x": 0.02,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
        hovermode="x unified",
    )
    notes = []
    if km_result.get("test"):
        test = km_result["test"]
        # The analysis names the test it ran, with the Fleming-Harrington weight.
        name = str(test.get("test") or "")
        test_label = str(km_result.get("test_p_value_label") or _KM_TEST_LABELS.get(name, name.replace("_", " ")))
        notes.append(f"{escape_plotly_text(test_label[:1].upper() + test_label[1:])} test: {_p_value_expression(test['p_value'])}")
    elif km_result.get("outcome_informed_group"):
        notes.append("Outcome-informed grouping: fresh raw p-value suppressed")
    if show_confidence_bands:
        notes.append(f"Shaded bands: {confidence_percent}% pointwise CI")
    if notes:
        x, y, xanchor, yanchor = _km_note_position(km_result)
        fig.add_annotation(
            text="<br>".join(notes),
            xref="paper", yref="paper", x=x, y=y,
            showarrow=False, font={"size": 12, "color": INK},
            align="left" if xanchor == "left" else "right", xanchor=xanchor, yanchor=yanchor,
            bgcolor="rgba(255,255,255,0.85)", borderpad=4,
        )
    if risk_rows:
        fig.add_annotation(
            text="<b>Number at risk</b>", xref="paper", yref="paper", x=0, y=0, xanchor="right", yanchor="top",
            xshift=-label_shift, yshift=-(risk_top - row_height), showarrow=False, font={"size": 12, "color": INK}, align="right",
        )
        row_colors = _km_group_colors([row["group"] for row in risk_rows])
        for index, (row, label) in enumerate(zip(risk_rows, labels, strict=True)):
            color = color_by_group.get(str(row["group"]), row_colors[index])
            shift = -(risk_top + row_height * index)
            fig.add_annotation(
                text=label, xref="paper", yref="paper", x=0, y=0, xanchor="right", yanchor="top",
                xshift=-label_shift, yshift=shift, showarrow=False, font={"size": 12, "color": color}, align="right",
            )
            for time, count in zip(risk_times, row["counts"], strict=True):
                fig.add_annotation(
                    text=str(count), xref="x", yref="paper", x=time, y=0, xanchor="center", yanchor="top",
                    yshift=shift, showarrow=False, font={"size": 12, "color": INK},
                )
    axis_ticks = {"tickmode": "array", "tickvals": risk_times, "ticktext": [f"{time:g}" for time in risk_times]} if risk_times else {}
    fig.update_xaxes(title=f"Time ({escape_plotly_text(time_unit_label)})", **_COMMON_AXES, **axis_ticks, range=[0, km_result["display_horizon"]])
    fig.update_yaxes(title="Survival probability", tickformat=".0%", range=[0, 1.02], **_COMMON_AXES)
    return figure_to_json(fig)


# Corners for the test and band notes, tried in this order: (x, y, xanchor, yanchor) and the box the
# notes take there as (x0, x1, y0, y1) in plot fractions.
_KM_NOTE_CORNERS = (
    ((0.01, 0.02, "left", "bottom"), (0.0, 0.34, 0.0, 0.17)),
    ((0.99, 0.98, "right", "top"), (0.66, 1.0, 0.83, 1.0)),
    ((0.99, 0.02, "right", "bottom"), (0.66, 1.0, 0.0, 0.17)),
)


def _km_note_position(km_result: dict[str, Any]) -> tuple[float, float, str, str]:
    """The first corner no survival curve runs through, else above the plot at the right."""
    horizon = float(km_result.get("display_horizon") or 0.0)
    curves = km_result.get("curves") or []
    for position, (x0, x1, y0, y1) in _KM_NOTE_CORNERS:
        if horizon <= 0 or not any(_km_curve_crosses(curve, x0 * horizon, x1 * horizon, y0 * 1.02, y1 * 1.02) for curve in curves):
            return position
    return 1.0, 1.0, "right", "bottom"


def _km_curve_crosses(curve: dict[str, Any], start: float, end: float, low: float, high: float) -> bool:
    """Whether a step curve (drawn with its vertical drops) enters the box [start, end] x [low, high]."""
    times = np.asarray(curve.get("timeline") or [], dtype=float)
    survival = np.asarray(curve.get("survival") or [], dtype=float)
    if times.size == 0 or times.size != survival.size or start > times[-1]:
        return False
    end = min(end, float(times[-1]))
    upper = float(survival[max(np.searchsorted(times, start, side="right") - 1, 0)])
    lower = float(survival[max(np.searchsorted(times, end, side="right") - 1, 0)])
    return lower <= high and upper >= low


def _km_risk_rows(km_result: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    """Tick times and per-group at-risk counts from the KM result's risk table (empty when absent)."""
    table = km_result.get("risk_table") or {}
    times = [float(time) for time in table.get("times") or []]
    columns = list(table.get("columns") or [])[1:]
    if not times or len(columns) != len(times):
        return [], []
    rows = []
    for row in table.get("rows") or []:
        counts = [row.get(column) for column in columns]
        if any(count is None for count in counts):
            return [], []
        rows.append({"group": row.get("Group", ""), "counts": [int(count) for count in counts]})
    return times, rows


def build_cox_forest_figure(cox_result: dict[str, Any]) -> dict[str, Any]:
    raw_rows = list(reversed(cox_result["results_table"]))
    # Finite and positive estimates only (a log axis cannot draw an infinite bound); the others are named in a note.
    rows = [row for row in raw_rows if _drawable_interval(row, ("Hazard ratio", "CI lower", "CI upper"))]
    not_drawn = [str(row.get("Label")) for row in cox_result["results_table"] if not _drawable_interval(row, ("Hazard ratio", "CI lower", "CI upper"))]
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

    note, note_lines = _not_drawn_note(not_drawn, prefix="Not drawn (no finite hazard ratio or interval)", width=90) if not_drawn else ("", 0)
    extra = 18 * note_lines + 12 if note_lines else 0
    fig.update_layout(
        **_COMMON_LAYOUT,
        margin={"l": axis_layout["l"], "r": 40, "t": 28, "b": 70 + extra},
        title={
            "text": "",
            "font": {"family": "Source Serif 4, serif", "size": 24, "color": INK},
            "x": 0.02,
        },
        height=max(420, axis_layout["height"]) + extra,
    )
    if note:
        # The terms left out are named under the axis title.
        fig.add_annotation(
            text=note, xref="paper", yref="paper", x=0.0, y=0.0, xanchor="left", yanchor="top", yshift=-62,
            showarrow=False, align="left", font={"size": 12, "color": INK},
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
    fig.update_xaxes(
        title="Hazard ratio (log scale)",
        type="log",
        **_log_axis_ticks([1.0, *(row["CI lower"] for row in rows), *(row["CI upper"] for row in rows)]),
        **_COMMON_AXES,
    )
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
        x_values, y_values = _finite_pairs(panel.get("log_time"), panel.get("residual"))
        trend_x, trend_y = _finite_pairs(panel.get("trend_log_time"), panel.get("trend_residual"))
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
        x_values, y_values = _finite_pairs(panel.get("value"), panel.get("residual"))
        trend_x, trend_y = _finite_pairs(panel.get("trend_value"), panel.get("trend_residual"))

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
    # find_optimal_cutpoint names it optimal_cutpoint; the "Make groups" derive summary names it cutoff.
    optimal = result.get("optimal_cutpoint", result.get("cutoff"))

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
    if not subtitle_text:
        fig.update_layout(
            **_COMMON_LAYOUT,
            margin={k: v for k, v in axis_layout.items() if k != "height"},
            title={"text": title_text, "font": {"family": "Source Serif 4, serif", "size": 22, "color": INK}, "x": 0.02},
            height=axis_layout["height"],
        )
    else:
        fig.update_layout(
            **_COMMON_LAYOUT,
            margin={
                **{k: v for k, v in axis_layout.items() if k != "height"},
                "t": max(108, int(axis_layout.get("t", 32)) + 72 + (subtitle_lines - 1) * 18),
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
    unranked_modes = [str(row.get("evaluation_mode") or "") for row in table if not row.get("comparable_for_ranking", True)]
    unranked_kinds = []
    if any("apparent" in mode for mode in unranked_modes):
        unranked_kinds.append("apparent-fallback rows")
    if any("apparent" not in mode for mode in unranked_modes):
        unranked_kinds.append("rows without a complete C-index estimate")
    note = (
        f"<br><sup>* {' and '.join(unranked_kinds)} shown for transparency and excluded from rank ordering</sup>"
        if unranked_kinds
        else ""
    )

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
        values = np.asarray(monitor_loss_history, dtype=float)
        if best_monitor_epoch is None and np.isfinite(values).any():
            # A missing (NaN) or infinite monitor value is never the best; with no finite value there is no best epoch.
            worst = np.inf if monitor_goal == "min" else -np.inf
            finite = np.where(np.isfinite(values), values, worst)
            best_monitor_epoch = int(np.argmin(finite) if monitor_goal == "min" else np.argmax(finite)) + 1
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
        status_text = f"Trained for {_count(epochs_trained, 'epoch')}"
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
    # Rows run from the most important feature down, so the first one is drawn at the top.
    fig.update_yaxes(autorange="reversed", **_COMMON_AXES)
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
    scatter = go.Scattergl if len(result.get("marker_table", [])) > 2_000 else go.Scatter
    # Plotly draws later traces on top: the grey "not supported" points go first so they cannot hide a robust or
    # suggestive marker at the same place, and legendrank keeps the legend in tier order.
    for rank, (tier, color) in reversed(list(enumerate(MARKER_TIER_COLORS.items(), start=1))):
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
            scatter(
                x=[row[primary]["selection_frequency"] for row in members],
                y=[row[primary]["direction_consistency"] for row in members],
                mode="markers",
                name=tier,
                legendrank=rank,
                marker={"size": 9, "color": color, "line": {"width": 0.6, "color": INK}},
                customdata=[[escape_plotly_text(row["marker"]), escape_plotly_text(_format_p_value(row[primary].get("p_fwer")))] for row in members],
                hovertemplate="%{customdata[0]}<br>Selected in %{x:.0%} of subsamples<br>Same direction in %{y:.0%}<br>Family-wise p = %{customdata[1]}<extra></extra>",
            )
        )
    frequency = float(settings.get("robust_frequency", 0.5))
    direction = float(settings.get("robust_direction", 0.9))
    fig.add_vline(x=frequency, line_dash="dash", line_color=INK, line_width=1, opacity=0.5)
    fig.add_hline(y=direction, line_dash="dash", line_color=INK, line_width=1, opacity=0.5)
    rule = f"Robust: family-wise p ≤ {float(settings.get('alpha', 0.05)):g}, selected in ≥ {frequency:.0%}, same direction in ≥ {direction:.0%}"
    if not _family_wise_computed(result):
        rule += "<br>No permutations were run, so no marker could be robust."
    fig.add_annotation(
        text=rule,
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
    highest = max((row[primary]["rank_interval"][1] for row in rows), default=1.0)
    title = "Rank across subsamples (1 = strongest)"
    if highest > 200:
        # Ranks of a genome-wide panel run to tens of thousands; a log axis keeps the top ranks apart.
        ticks = [value for value in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000) if value <= highest * 1.5]
        fig.update_xaxes(title=title, type="log", range=[np.log10(0.8), np.log10(highest * 1.25)],
                         tickmode="array", tickvals=ticks, ticktext=[f"{value:,}" for value in ticks], **_COMMON_AXES)
    else:
        fig.update_xaxes(title=title, range=[0.5, highest + 0.5], **_COMMON_AXES)
    # Rows are drawn by rank (strongest at the top), not in the order the tier traces list them.
    fig.update_yaxes(
        automargin=True,
        tickmode="array",
        tickvals=labels,
        ticktext=display_labels,
        categoryorder="array",
        categoryarray=labels,
        **_COMMON_AXES,
    )
    return figure_to_json(fig)


_FUNNEL_GREY = "rgba(148,163,184,0.75)"
_NOT_PERMUTED = "not computed (no permutations)"


def _family_wise_computed(result: dict[str, Any]) -> bool:
    """Whether the run computed family-wise p-values: it ran permutations (results that do not say so, any finite
    family-wise p-value)."""
    n_permutations = (result.get("null") or {}).get("n_permutations")
    if n_permutations is not None:
        return int(n_permutations) > 0
    primary = str(result.get("primary_lens", "marginal"))
    return any(
        _finite_number((row.get(primary) or {}).get("p_fwer")) for row in result.get("marker_table", []) if isinstance(row.get(primary), dict)
    )


def _no_subsample_evaluated(result: dict[str, Any]) -> bool:
    """Whether the result says that no subsample was evaluated, so the robust tier could not be assessed."""
    resampling = result.get("resampling") or {}
    if resampling.get("stability_assessed") is not None:
        return not resampling["stability_assessed"]
    return "n_valid" in resampling and not int(resampling.get("n_valid") or 0)


def marker_evidence_funnel(result: dict[str, Any]) -> list[dict[str, Any]]:
    """How many markers clear each successively stricter bar on the primary lens.

    Each bar is (up to permutation noise) a subset of the one above: a BH q-value is never below its p-value,
    and a robust marker is a family-wise rejection by definition. A bar the run could not compute (family-wise
    p-values and the robust tier without permutations, the robust tier without subsamples) has no count and a
    ``note`` saying why, so it is not read as zero markers.
    """
    primary = str(result.get("primary_lens", "marginal"))
    settings = result.get("settings") or {}
    alpha = float(settings.get("alpha", 0.05))
    cohort = result.get("cohort") or {}
    stats = [row[primary] for row in result.get("marker_table", []) if isinstance(row.get(primary), dict)]

    def passing(key: str, level: float, *, strict: bool = False) -> int:
        values = [item.get(key) for item in stats]
        finite = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(float(value))]
        return sum(1 for value in finite if (value < level if strict else value <= level))

    tested = int(cohort.get("n_markers_evaluated") or len(stats))
    dropped = len(cohort.get("dropped_markers") or [])
    permuted = _family_wise_computed(result)
    if not permuted:
        robust_note = _NOT_PERMUTED
    elif _no_subsample_evaluated(result):
        robust_note = "not assessed (no subsamples)"
    else:
        robust_note = None
    stages = []
    if dropped:
        stages.append({"label": "Supplied", "count": tested + dropped, "color": "rgba(148,163,184,0.35)"})
    stages += [
        {"label": "Tested", "count": tested, "color": _FUNNEL_GREY},
        {"label": f"p < {alpha:g}", "count": passing("p_value", alpha, strict=True), "color": _FUNNEL_GREY},
        {"label": f"FDR q ≤ {alpha:g}", "count": passing("q_bh", alpha), "color": GOLD},
        (
            {"label": f"Family-wise p ≤ {alpha:g}", "count": passing("p_fwer", alpha), "color": PLUM}
            if permuted
            else {"label": f"Family-wise p ≤ {alpha:g}", "count": None, "color": _FUNNEL_GREY, "note": _NOT_PERMUTED}
        ),
        (
            {"label": "Robust", "count": int((result.get("tier_counts") or {}).get("robust", 0)), "color": SAGE}
            if robust_note is None
            else {"label": "Robust", "count": None, "color": _FUNNEL_GREY, "note": robust_note}
        ),
    ]
    return stages


def build_marker_summary_figure(result: dict[str, Any]) -> dict[str, Any]:
    """Marker counts, the full model's apparent and gap-adjusted C, and the repeated selection
    procedure's left-out C and gain over the clinical covariates."""
    added_value = result.get("primary_lens") == "added_value"
    signature = result.get("signature") or {}
    # The model the right panel is about: with no marker selected it holds the clinical covariates only; it can also
    # have failed to fit in the full cohort, or not exist (no marker selected and no clinical covariates).
    if signature_is_clinical_only(signature):
        model_title, empty_note = "Clinical model and procedure C-index", "No C-index is available for the model."
    elif signature_fit_failed(result):
        model_title, empty_note = "C-index (full-cohort model not fitted)", "The model could not be fitted in the full cohort."
    elif signature.get("markers") or signature.get("apparent_c") is not None:
        model_title, empty_note = "Model and procedure C-index", "No C-index is available for the model."
    else:
        model_title, empty_note = "C-index (no marker selected)", "No marker was selected, so there is no model."
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.2,
        subplot_titles=("Markers clearing each bar", model_title),
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 14, "color": INK, "family": "Sora, sans-serif"}
        annotation.yshift = 8

    # A genome-wide panel runs from tens of thousands to a handful, so bar length is then log10(count + 1),
    # which keeps zero at zero; a small panel keeps plain counts.
    stages = marker_evidence_funnel(result)
    labels = [stage["label"] for stage in stages]
    counts = [stage["count"] for stage in stages]
    # A bar that was not computed has no length and says why beside it.
    log_scale = max((count for count in counts if count is not None), default=0) > 50
    lengths = [0.0 if count is None else float(np.log10(count + 1)) if log_scale else float(count) for count in counts]
    fig.add_trace(
        go.Bar(
            x=lengths,
            y=labels,
            orientation="h",
            marker={"color": [stage["color"] for stage in stages], "line": {"width": 0}},
            customdata=[stage.get("note") if stage["count"] is None else f"{stage['count']:,}" for stage in stages],
            hovertemplate=(
                "%{y}: %{customdata} markers<extra></extra>"
                if None not in counts
                else [f"%{{y}}: {'%{customdata}' if count is None else '%{customdata} markers'}<extra></extra>" for count in counts]
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    longest = max(max(lengths, default=1.0), 1.0)
    fig.add_trace(
        go.Scatter(
            x=[length + 0.025 * longest for length in lengths],
            y=labels,
            mode="text",
            text=[stage.get("note") if stage["count"] is None else f"<b>{stage['count']:,}</b>" for stage in stages],
            textposition="middle right",
            textfont={
                "size": 13,
                "color": INK if None not in counts else [INK if count is not None else "rgba(100,116,139,0.95)" for count in counts],
            },
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if log_scale:
        ticks = [value for value in (1, 10, 100, 1_000, 10_000, 100_000) if np.log10(value + 1) <= longest]
        fig.update_xaxes(
            title="Markers (log scale)",
            range=[0, longest * 1.22],
            tickmode="array",
            tickvals=[float(np.log10(value + 1)) for value in ticks],
            ticktext=[f"{value:,}" for value in ticks],
            row=1,
            col=1,
            **_COMMON_AXES,
        )
    else:
        fig.update_xaxes(title="Markers", range=[0, longest * 1.22], dtick=max(1, int(np.ceil(longest / 5))), row=1, col=1, **_COMMON_AXES)
    fig.update_yaxes(autorange="reversed", row=1, col=1, **_COMMON_AXES)

    # Every left-out estimate repeats selection and fitting in the smaller training subsample.
    ladder = [
        ("Apparent", signature.get("apparent_c"), ACCENT),
        ("Subsample<br>gap-adjusted", signature.get("optimism_corrected_c"), SLATE),
        ("Whole procedure<br>(left-out)", signature.get("signature_c_left_out"), SAGE),
    ]
    if added_value:
        ladder.append(("Clinical only<br>(left-out)", signature.get("clinical_c_left_out"), "rgba(100,116,139,0.9)"))
    ladder = [(label, float(value), color) for label, value, color in ladder if isinstance(value, (int, float)) and np.isfinite(float(value))]
    gain_note = _left_out_gain_note(signature) if added_value else None
    if ladder:
        for label, value, color in ladder:
            fig.add_trace(
                go.Scatter(x=[0.5, value], y=[label, label], mode="lines", line={"color": color, "width": 2}, opacity=0.35, hoverinfo="skip", showlegend=False),
                row=1,
                col=2,
            )
        fig.add_trace(
            go.Scatter(
                x=[value for _, value, _ in ladder],
                y=[label for label, _, _ in ladder],
                mode="markers+text",
                marker={"size": 14, "color": [color for _, _, color in ladder], "line": {"width": 1, "color": INK}},
                text=[f"{value:.3f}" for _, value, _ in ladder],
                textposition="middle right",
                textfont={"size": 13, "color": INK},
                cliponaxis=False,
                hovertemplate="%{y}: C = %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        low = min(0.5, *(value for _, value, _ in ladder))
        high = max(value for _, value, _ in ladder)
        fig.add_vline(x=0.5, line_dash="dot", line_color=INK, line_width=1, opacity=0.45, row=1, col=2)
        fig.update_xaxes(title="Harrell's C (0.5 = chance)", range=[low - 0.02, high + 0.045 + 0.1 * (high - low)], row=1, col=2, **_COMMON_AXES)
        fig.update_yaxes(autorange="reversed", row=1, col=2, **_COMMON_AXES)
        if gain_note:
            # Under the ladder's axis title, where the bottom margin grows to hold it.
            fig.add_annotation(
                text=gain_note,
                xref="x2 domain",
                yref="y2 domain",
                x=1.0,
                y=0.0,
                xanchor="right",
                yanchor="top",
                yshift=-56,
                showarrow=False,
                align="right",
                font={"size": 12, "color": INK},
            )
    else:
        fig.add_annotation(
            text=empty_note,
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 13, "color": INK},
        )
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
    noted = bool(ladder and gain_note)
    _marker_layout(fig, "Marker Evaluation at a Glance", height=440 if noted else 400, left=150)
    fig.update_layout(bargap=0.3)
    if noted:
        fig.update_layout(margin={"b": 110})
    return figure_to_json(fig)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and bool(np.isfinite(float(value)))


def _left_out_gain_note(signature: dict[str, Any]) -> str | None:
    """The mean paired gain over the clinical covariates in the patients left out, with its 95% interval when the
    result has one; None without both left-out C-indices."""
    model, clinical = signature.get("signature_c_left_out"), signature.get("clinical_c_left_out")
    if not (_finite_number(model) and _finite_number(clinical)):
        return None
    paired = signature.get("signature_gain_left_out")
    gain = float(paired) if _finite_number(paired) else float(model) - float(clinical)
    interval = signature.get("signature_gain_left_out_ci")
    span = ""
    if _finite_number(paired) and isinstance(interval, (list, tuple)) and len(interval) == 2 and all(_finite_number(value) for value in interval):
        span = f" (95% CI {float(interval[0]):.3f} to {float(interval[1]):.3f})"
    return f"Left-out gain over the clinical covariates:<br><b>{gain:+.3f}</b>{span}"


def _drawable_interval(estimate: Any, keys: tuple[str, ...]) -> bool:
    """Whether an estimate and its interval can be drawn on a log axis: finite and positive (an infinite bound cannot)."""
    return all(_finite_number(estimate.get(key)) and float(estimate[key]) > 0 for key in keys)


def _not_drawn_note(
    entries: list[str], *, prefix: str = "Not drawn", limit: int = 8, width: int = 110, max_lines: int = 3
) -> tuple[str, int]:
    """"Not drawn: a (reason), b (reason)" for a figure note, wrapped, escaped and cut after ``limit`` entries."""
    shown = ", ".join(entries[:limit]) + (f" and {len(entries) - limit} more" if len(entries) > limit else "")
    lines = textwrap.wrap(f"{prefix}: {shown}", width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [_truncate_label_fragment(" ".join(lines[max_lines - 1 :]), width)]
    return "<br>".join(escape_plotly_text(line) for line in lines), len(lines)


_REPLICATION_FIT_LABELS = {"added_value": "with the clinical covariates", "marginal": "marker alone"}


def _replication_fit(row: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """The external fit a locked marker's replication test used, with its lens: with the clinical covariates for
    "added_value", without them for "marginal", and none when ``tested`` is None (that fit was not estimable).
    Results without ``tested`` used the adjusted fit when there was one."""
    lens = row["tested"] if "tested" in row else ("added_value" if row.get("adjusted") else "marginal")
    fit = row.get("adjusted") if lens == "added_value" else row.get("marginal") if lens == "marginal" else None
    return (lens, fit) if isinstance(fit, dict) else (None, None)


def build_marker_replication_figure(validation: dict[str, Any]) -> dict[str, Any]:
    """The locked model in the external cohort: its C-index beside the clinical covariates alone (top), and each
    locked marker's external hazard ratio from the fit its replication test used, coloured by whether it replicated
    (bottom). Markers without a drawable estimate are named in a note under the forest instead."""
    rows = []
    not_drawn: list[str] = []
    for row in validation.get("markers", []):
        lens, tested = _replication_fit(row)
        name = str(row.get("marker"))
        if row.get("absent"):
            not_drawn.append(f"{name} (not measured)")
        elif tested is None:
            not_drawn.append(f"{name} (not estimable)")
        elif _drawable_interval(tested, ("hazard_ratio", "ci_lower", "ci_upper")):
            rows.append((row, {**tested, "lens": lens}))
        elif _drawable_interval(tested, ("hazard_ratio",)):
            not_drawn.append(f"{name} (interval not finite)")
        else:
            not_drawn.append(f"{name} (estimate not finite)")
    rows.reverse()
    labels = [str(row["marker"]) for row, _ in rows]
    display_labels, axis_layout = _feature_plot_axis_layout(labels, width=30, max_lines=2)
    groups = (
        ("replicated", SAGE, lambda row: row.get("replicated")),
        ("same direction, not significant", GOLD, lambda row: not row.get("replicated") and row.get("same_direction")),
        ("opposite direction", ACCENT, lambda row: not row.get("same_direction")),
    )
    metrics = validation.get("metrics") or {}
    ladder = []
    if _finite_number(metrics.get("c_index")):
        ladder.append(("Locked model", float(metrics["c_index"]), metrics.get("c_index_ci") or [None, None], SLATE))
    if _finite_number(metrics.get("clinical_only_c_index")):
        ladder.append(("Clinical covariates alone", float(metrics["clinical_only_c_index"]), [None, None], "rgba(100,116,139,0.9)"))

    # Heights in pixels: a fixed C-index panel, room for its axis and the next title, and a forest that
    # grows with the number of markers; make_subplots takes them as shares of the plotting area.
    forest_height = max(220, axis_layout["height"] - 180)
    top_height, gap = (130, 110) if ladder else (1, 1)
    area = top_height + gap + forest_height
    gain = metrics.get("delta_c_index")
    gain_ci = metrics.get("delta_c_index_ci") or [None, None]
    top_title = "C-index in this cohort"
    if ladder and _finite_number(gain):
        span = f" (95% CI {gain_ci[0]:+.3f} to {gain_ci[1]:+.3f})" if None not in gain_ci else ""
        top_title += f": gain over the clinical covariates {gain:+.3f}{span}"
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[top_height / area, forest_height / area],
        vertical_spacing=gap / area,
        subplot_titles=(top_title if ladder else "", "Hazard ratio of each locked marker"),
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 14, "color": INK, "family": "Sora, sans-serif"}
        annotation.xanchor = "left"
        annotation.x = 0.0
    if ladder:
        for label, value, interval, color in ladder:
            if interval and None not in interval:
                fig.add_trace(go.Scatter(x=list(interval), y=[label, label], mode="lines", line={"color": color, "width": 2.5},
                                         hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=[value for _, value, _, _ in ladder],
                y=[label for label, _, _, _ in ladder],
                mode="markers+text",
                marker={"size": 13, "color": [color for *_, color in ladder], "line": {"width": 1, "color": INK}},
                text=[f"{value:.3f}" for _, value, _, _ in ladder],
                textposition="top center",
                textfont={"size": 12, "color": INK},
                hovertemplate="%{y}: C = %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        bounds = [bound for _, value, interval, _ in ladder for bound in (value, *(item for item in interval if item is not None))]
        low, high = min(0.5, *bounds), max(bounds)
        fig.add_vline(x=0.5, line_dash="dot", line_color=INK, line_width=1, opacity=0.45, row=1, col=1)
        fig.update_xaxes(title="Harrell's C (0.5 = chance)", range=[low - 0.02, high + 0.03], row=1, col=1, **_COMMON_AXES)
        # First row on top, with room above it for the value label.
        fig.update_yaxes(range=[len(ladder) - 0.5, -0.8], row=1, col=1, **_COMMON_AXES)
    else:
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)

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
                    [
                        escape_plotly_text(row["marker"]),
                        escape_plotly_text(_format_p_value(row.get("replication_p_holm"))),
                        _REPLICATION_FIT_LABELS.get(str(tested["lens"]), ""),
                    ]
                    for row, tested in members
                ],
                hovertemplate="%{customdata[0]}<br>HR %{x:.3f} (%{customdata[2]})<br>Replication p (Holm) = %{customdata[1]}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if rows:
        # Added after the markers: Plotly leaves a line out of a subplot that has no traces yet.
        fig.add_vline(x=1.0, line_color=INK, line_width=1.5, opacity=0.75, row=2, col=1)
    else:
        fig.add_annotation(
            text="No finite external hazard ratios to plot.",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": INK},
        )
    # Markers without a drawable estimate are named under the legend, never drawn with another fit's hazard ratio.
    note, note_lines = _not_drawn_note(not_drawn, width=95) if not_drawn else ("", 0)
    bottom = 110 + (18 * note_lines + 24 if note_lines else 0)
    if note:
        fig.add_annotation(
            text=note,
            xref="paper",
            yref="paper",
            x=0.0,
            y=0.0,
            xanchor="left",
            yanchor="top",
            yshift=-122,
            showarrow=False,
            align="left",
            font={"size": 12, "color": INK},
        )
    _marker_layout(fig, "Locked Model in the External Cohort", height=area + 80 + bottom, left=axis_layout["l"])
    fig.update_layout(margin={"b": bottom}, legend={"orientation": "h", "yanchor": "top", "y": -88 / area, "xanchor": "left", "x": 0.0})
    bounds = [tested[key] for _, tested in rows for key in ("ci_lower", "ci_upper")]
    fig.update_xaxes(title="Hazard ratio (log scale)", type="log", **_log_axis_ticks([1.0, *bounds]), row=2, col=1, **_COMMON_AXES)
    # Markers keep the recipe's order (first at the top) instead of being grouped by replication status.
    fig.update_yaxes(
        automargin=True,
        tickmode="array",
        tickvals=labels,
        ticktext=display_labels,
        categoryorder="array",
        categoryarray=labels,
        row=2,
        col=1,
        **_COMMON_AXES,
    )
    return figure_to_json(fig)

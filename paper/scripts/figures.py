"""Figures 1 to 5 and Supplementary Figures S1 to S6 of the software paper, drawn from the files the analysis scripts
write to results/, at print size: BMC prints a figure at most 170 mm wide, so every figure is drawn at its final size,
at most 170 mm wide, reserving 25 mm of BMC's 225-mm combined height for a legend, with a project readability
floor of 7 pt. save() refuses one that is not. Figure 1b is a native screenshot of the Markers summary after
case study I (INTERFACE), which Figure 1 places under the workflow diagram (fig1_workflow).

Needs matplotlib, numpy and pandas (no SurvStudio). Writes PNG (300 dpi; Figure 1, with its screenshot, 600 dpi),
PDF, SVG and a per-figure provenance record to paper/figures/.
Usage: python figures.py [name ...] draws the named figures only (workflow, interface, markers, estimates, models,
comparison, simulation, luad_external, breast_external, breast_er_external, breast_er_sensitivity, tiers); without
names, all of them, the comparison once run_competitors.sh has written its results.

A figure is drawn only from results of one run of the analysis: every file it reads must carry the stamp its
script wrote (results/stamps/), unchanged since, from the same SurvStudio commit and the same analysis code, and
computed from the results files that are there now (common.result_problems). A figure whose files fail, or lack a
value it draws (results of an earlier version of the scripts), is not drawn, the others are, and the run ends with an
error that names the files; rerun the steps that wrote them.

Pooled estimates are drawn with the Hartung-Knapp-Sidik-Jonkman 95% interval and, where there are three or more
cohorts, the 95% prediction interval (common.random_effects).
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from common import STAMP_DTYPES, result_problems  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
FIGURES.mkdir(exist_ok=True)
# Figure 1b: the native summary panel after case study I (TCGA-LUAD, Xena HiSeqV2.gz, age, sex and stage,
# defaults), captured in a 900-pixel-wide window at three device pixels per pixel; its provenance records fonts.
INTERFACE = PAPER / "interface" / "marker_summary_case_study_i.png"
INTERFACE_PROVENANCE = PAPER / "interface" / "marker_summary_case_study_i.provenance.json"
# BMC's width and combined figure/legend height; 25 mm reserved for the legend and our own 7-pt readability floor.
PRINT_WIDTH = 170 / 25.4
PRINT_HEIGHT = 200 / 25.4
MIN_POINTS = 7.0
PAD = 0.02  # inches of white space kept around the drawing
WIDE = PRINT_WIDTH - 0.06  # a full-width figure, its drawing and the white space around it within the print width
INK = "#1f2933"
MUTED = "#6b7785"
ML = "#2f65d9"
DL = "#db7e15"
ROBUST = "#2e8b57"
SUGGESTIVE = "#d4a017"
OPPOSITE = "#c0392b"
LIGHT = "#e8edf2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "survstudio-paper",
})


# The results files the figure being drawn has read.
LOADED: list[str] = []


class MixedResults(Exception):
    """The files a figure reads do not come from one run of the analysis (common.result_problems)."""


class NotPrintable(Exception):
    """A figure beyond the print dimensions or below the project's 7-pt text floor."""


def load(name: str, **options):
    LOADED.append(name)
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if name.endswith(".json") else pd.read_csv(path, **options)


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def print_size(fig) -> tuple[float, float, list[str]]:
    """The figure's width and height as saved (inches), and what keeps it from print: wider than 170 mm, or text
    below 7 pt or more than 200 mm high (leaving 25 mm for the legend)."""
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = box.width + 2 * PAD, box.height + 2 * PAD
    problems = []
    if width > PRINT_WIDTH + 1e-6:
        problems.append(f"{width * 25.4:.1f} mm wide (at most {PRINT_WIDTH * 25.4:.0f} mm)")
    if height > PRINT_HEIGHT + 1e-6:
        problems.append(f"{height * 25.4:.1f} mm high (at most 200 mm, reserving 25 mm for the legend)")
    small = sorted({f"{text.get_text()!r} ({text.get_fontsize():g} pt)" for text in fig.findobj(Text)
                    if text.get_visible() and text.get_text().strip() and text.get_fontsize() < MIN_POINTS - 1e-9})
    if small:
        problems.append(f"text below {MIN_POINTS:g} pt: " + ", ".join(small[:6]) + (" ..." if len(small) > 6 else ""))
    return width, height, problems


def save(fig, name: str, dpi: int = 300) -> None:
    problems = result_problems(LOADED)
    if problems:
        plt.close(fig)
        raise MixedResults(problems)
    width, height, problems = print_size(fig)
    if problems:
        plt.close(fig)
        raise NotPrintable(problems)
    metadata = {"png": {"Software": "SurvStudio publication figures"},
                "pdf": {"Creator": "SurvStudio publication figures", "Title": name, "CreationDate": None, "ModDate": None},
                "svg": {"Creator": "SurvStudio publication figures", "Title": name, "Date": None}}
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight", pad_inches=PAD, metadata=metadata[suffix])
    # Matplotlib's static SVG 1.1 DTD is unnecessary for a portable, inactive export.
    svg = FIGURES / f"{name}.svg"
    svg_text = svg.read_text(encoding="utf-8").replace(
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n', "")
    if "<!DOCTYPE" in svg_text or "<!ENTITY" in svg_text:
        raise ValueError("SVG contains an unexpected DTD or entity declaration")
    svg.write_text(svg_text, encoding="utf-8")
    from common import git_commit, sha256_file
    try:
        head = subprocess.run(["git", "-C", str(PAPER.parent), "rev-parse", "HEAD"], capture_output=True, text=True)
        git_head = head.stdout.strip() if head.returncode == 0 else "unknown"
    except OSError:
        git_head = "unknown"
    provenance = {"schema": "SurvStudio paper figure provenance 1", "figure": name,
                  "rendering_commit": git_commit(PAPER.parent), "rendering_git_head": git_head,
                  "rendering_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "dimensions_mm": [width * 25.4, height * 25.4], "png_dpi": dpi,
                  "inputs": {input_name: json.loads((RESULTS / "stamps" / f"{input_name}.json").read_text())
                             for input_name in dict.fromkeys(LOADED)},
                  "outputs": {suffix: sha256_file(FIGURES / f"{name}.{suffix}") for suffix in metadata}}
    if name == "fig1":
        provenance["interface"] = {"png_sha256": sha256_file(INTERFACE), "provenance_sha256": sha256_file(INTERFACE_PROVENANCE)}
    (FIGURES / f"{name}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"wrote {name} ({width * 25.4:.0f} x {height * 25.4:.0f} mm)")


def interval_bounds(value) -> tuple[float | None, float | None]:
    """An interval SurvStudio reports as [lower, upper] or {"lower": ..., "upper": ...}; (None, None) when absent."""
    if isinstance(value, dict):
        value = [value.get("lower"), value.get("upper")]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(bound is not None for bound in value):
        return float(value[0]), float(value[1])
    return None, None


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def pooled_text(pooled: dict) -> str:
    """A pooled estimate with its HKSJ interval, and the prediction interval when there is one."""
    text = f"{signed(pooled['estimate'])} ({signed(pooled['hksj_ci_lower'])} to {signed(pooled['hksj_ci_upper'])})"
    if pooled.get("pi_lower") is not None:
        text += f"\nPI {signed(pooled['pi_lower'])} to {signed(pooled['pi_upper'])}"
    return text


def span(ax, low: float, high: float, y: float, colour: str, lw: float, zorder: int = 1) -> None:
    """A horizontal interval, cut at the x limits already set, with an arrowhead where it runs on beyond them."""
    left, right = ax.get_xlim()
    ax.plot([max(low, left), min(high, right)], [y, y], color=colour, lw=lw, solid_capstyle="butt", zorder=zorder)
    for end, beyond, marker in ((left, low < left, "<"), (right, high > right, ">")):
        if beyond:
            ax.plot(end, y, marker=marker, color=colour, ms=3.5, mew=0, clip_on=False, zorder=zorder + 1)


def draw_pooled(ax, pooled: dict, y: float, colour: str = INK, half: float = 0.28) -> None:
    """A pooled estimate as a diamond spanning its HKSJ interval, with its prediction interval as a thin line; both are
    cut at the x limits already set, with an arrowhead where they run on."""
    if pooled.get("pi_lower") is not None:
        span(ax, pooled["pi_lower"], pooled["pi_upper"], y, colour, 0.8)
    ax.fill([pooled["hksj_ci_lower"], pooled["estimate"], pooled["hksj_ci_upper"], pooled["estimate"]], [y, y + half, y, y - half],
            color=colour, zorder=2)
    left, right = ax.get_xlim()
    for end, beyond, marker in ((left, pooled["hksj_ci_lower"] < left, "<"), (right, pooled["hksj_ci_upper"] > right, ">")):
        if beyond:
            ax.plot(end, y, marker=marker, color=colour, ms=4.5, mew=0, clip_on=False, zorder=3)


def gain_ticks(ax) -> None:
    """Ticks every 0.05 on a gain axis, every 0.1 when it spans more than 0.3."""
    left, right = ax.get_xlim()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05 if right - left <= 0.3 else 0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05) if right - left > 0.3 else matplotlib.ticker.NullLocator())


def gain_limits(values: list[float], step: float = 0.05) -> tuple[float, float]:
    """x limits for gains in C that hold ``values`` and zero, in whole steps."""
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.floor(min(0.0, *finite) / step) * step), float(np.ceil(max(0.0, *finite) / step) * step)


# ── Figure 1: what SurvStudio checks by default, and the Markers tab ───────────
WORKFLOW_HEIGHT = 3.3  # inches, at the full width: the diagram's 60 units of height with its 7-pt text


def draw_workflow(ax) -> None:
    """The workflow diagram on ``ax``, which should be WIDE by WORKFLOW_HEIGHT inches (the text is sized for that)."""
    ax.set_xlim(-1.0, 102.0)
    ax.set_ylim(0, 60)
    ax.axis("off")
    # Column: (left edge, width, fill, edge colour, title, boxes); boxes: (title, text, body lines), top to bottom.
    columns = {
        "inputs": (0, 25.5, "#dbe7fb", ML, "Inputs", [
            ("Clinical table", "CSV, TSV, Excel or Parquet", 1),
            ("Omics matrix", "CSV, TSV, Parquet, or .gz\nfiles as GEO and Xena\nserve them; TCGA\nbarcodes matched", 4),
            ("External cohort", "same or another platform", 1),
        ]),
        "checks": (29.5, 42.5, "#dcefe4", ROBUST, "Analyses and checks", [
            ("Survival curves and Cox regression", "Kaplan–Meier with numbers at risk, log-rank,\nEfron Cox model, proportional-hazards tests", 2),
            ("Marker evaluation", "score tests for added value over clinical data\n→ multiplicity-adjusted permutation tests\n    (Westfall–Young; marker residuals permuted)\n"
                                  "→ whole screen repeated on subsamples: tiers\n→ clinical-only C gain; approx. 95% interval", 5),
            ("Prediction models", "machine and deep learning on the same splits\n(holdout, repeated CV, locked test); paired\nintervals for the difference from Cox", 3),
        ]),
        "outputs": (76.0, 25.0, "#fbe7d4", DL, "Outputs", [
            ("Figures and tables", "forest plots, marker\ntables, leaderboards\nwith intervals", 3),
            ("Reporting checklists", "REMARK and TRIPOD+AI\nfilled from the run\n(Word, Markdown)", 3),
            ("Locked model", "hashed recipe applied\nunchanged to the\nexternal cohort", 3),
        ]),
    }
    line = 2.3  # one line of 7-pt text, in y units (60 units over 3.3 inches)
    placed: dict[str, list[tuple[float, float, float, float]]] = {}
    for key, (x, width, fill, edge, title, boxes) in columns.items():
        ax.text(x + width / 2, 57.2, title, ha="center", va="center", fontsize=8, fontweight="bold", color=edge)
        heights = [4.6 + lines * line + 1.0 for *_, lines in boxes]
        top, bottom = 54, 7
        gap = (top - bottom - sum(heights)) / (len(boxes) - 1)
        placed[key] = []
        for (name, body, _), height in zip(boxes, heights):
            y = top - height
            ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2,rounding_size=1.2", linewidth=0.8, edgecolor=edge, facecolor=fill))
            ax.text(x + 1.2, y + height - 1.3, name, ha="left", va="top", fontsize=7.5, fontweight="bold")
            ax.text(x + 1.2, y + height - 4.4, body, ha="left", va="top", fontsize=7, linespacing=1.3)
            placed[key].append((x, y, width, height))
            top = y - gap
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.5, mutation_scale=11, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(29.1, 33), xytext=(26.1, 33), arrowprops=arrow)
    ax.annotate("", xy=(75.6, 33), xytext=(72.6, 33), arrowprops=arrow)
    # The external cohort reaches the locked model under the middle column.
    x_e, y_e, w_e, _ = placed["inputs"][2]
    x_l, y_l, w_l, _ = placed["outputs"][2]
    route = dict(color=DL, lw=1.1, ls=(0, (3, 2)))
    ax.plot([x_e + w_e / 2, x_e + w_e / 2], [y_e - 0.2, 2.2], **route)
    ax.plot([x_e + w_e / 2, x_l + w_l / 2], [2.2, 2.2], **route)
    ax.annotate("", xy=(x_l + w_l / 2, y_l - 0.3), xytext=(x_l + w_l / 2, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=DL, lw=1.1, mutation_scale=10, shrinkA=0, shrinkB=0))
    ax.text(50.8, 2.9, "external validation, without refitting", ha="center", va="bottom", fontsize=7, color=DL)


def figure_workflow() -> None:
    fig = plt.figure(figsize=(WIDE, WORKFLOW_HEIGHT))
    draw_workflow(fig.add_axes([0, 0, 1, 1]))
    save(fig, "fig1_workflow")


def figure_interface() -> None:
    """Figure 1: a, the workflow; b, the actual summary panel, with its recorded fonts checked at print size."""
    shot = plt.imread(INTERFACE)
    provenance = json.loads(INTERFACE_PROVENANCE.read_text(encoding="utf-8"))
    from common import sha256_file
    if sha256_file(INTERFACE) != provenance["summary_png_sha256"]:
        raise MixedResults(["interface screenshot changed since its capture record"])
    gap, shot_width = 0.15, WIDE
    effective_points = shot_width * 72 * provenance["summary_min_svg_font_px"] / provenance["summary_css_width"]
    if effective_points < MIN_POINTS:
        raise NotPrintable([f"interface text is only {effective_points:.2f} pt at print size"])
    shot_height = shot_width * shot.shape[0] / shot.shape[1]
    height = WORKFLOW_HEIGHT + gap + shot_height
    fig = plt.figure(figsize=(WIDE, height))
    draw_workflow(fig.add_axes([0, 1 - WORKFLOW_HEIGHT / height, 1, WORKFLOW_HEIGHT / height]))
    ax = fig.add_axes([(WIDE - shot_width) / 2 / WIDE, 0, shot_width / WIDE, shot_height / height])
    ax.imshow(shot, interpolation="lanczos")
    ax.axis("off")
    for letter, top in (("a", 1.0), ("b", shot_height / height)):
        fig.text(0.0, top, letter, fontsize=9, fontweight="bold", va="top", ha="left")
    save(fig, "fig1", dpi=600)


# ── Figure 2: genome-wide markers in TCGA-LUAD ─────────────────────────────────
def figure_markers() -> None:
    summary = load("tcga_markers_summary.json")
    funnel = summary["funnel"]
    table = load("tcga_markers.csv")
    maxima = load("permutation_maxima.csv")
    permutation = load("permutation_maximum_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 3.05), gridspec_kw={"width_ratios": [0.95, 1.05, 1.0], "wspace": 0.62, "left": 0.2,
                                                                         "right": 0.985, "top": 0.93, "bottom": 0.33})

    ax = axes[0]
    stages = [
        ("Genes in the\nXena file", funnel["genes_in_file"]),
        ("Tested (varying in\n≥10% of patients)", funnel["tested"]),
        ("p < 0.05 beyond\nage, sex, stage", funnel["adjusted_p_below_0_05"]),
        ("Benjamini–Hochberg\nq ≤ 0.05", funnel["bh_q_below_0_05"]),
        ("Family-wise p ≤ 0.05\nand stable", funnel["robust"]),
    ]
    positions = np.arange(len(stages))[::-1]
    colours = [LIGHT, "#c9d6e5", "#9fb6d0", SUGGESTIVE, ROBUST]
    ax.barh(positions, [count for _, count in stages], color=colours, edgecolor=MUTED, linewidth=0.5, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e6)
    ax.set_xticks([1, 100, 10_000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(positions, [label for label, _ in stages], linespacing=1.1)
    for position, (_, count) in zip(positions, stages):
        ax.text(count * 1.3, position, f"{count:,}", va="center", fontsize=7)
    ax.set_xlabel("Genes (log scale)")
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a", x=-1.0)

    ax = axes[1]
    bins = np.logspace(np.log10(10), np.log10(max(maxima["all_genes"].max(), 300) * 1.1), 40)
    ax.hist(maxima["all_genes"], bins=bins, color=OPPOSITE, alpha=0.55, label="All non-constant genes")
    ax.hist(maxima["near_constant_removed"], bins=bins, color=ML, alpha=0.7, label="Near-constant genes removed")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    height = ax.get_ylim()[1]
    for key, colour, level in (("all_genes", OPPOSITE, 0.6), ("near_constant_removed", ML, 0.93)):
        threshold = permutation[key]["threshold_95"]
        ax.axvline(threshold, color=colour, lw=1, ls="--")
        ax.text(threshold * 0.92, height * level, f"95%: {threshold:.0f}", color=colour, fontsize=7, ha="right", va="top")
    # The strongest observed gene that passes the near-constant filter (a near-constant gene may score higher).
    top = next(item for item in permutation["top_observed"] if item["mode_share"] <= 0.9)
    ax.axvline(top["chi2"], color=ROBUST, lw=1, label="Strongest gene after the filter")
    ax.text(top["chi2"] * 1.08, height * 0.8, f"{top['marker']}\nχ² = {top['chi2']:.1f}", color=ROBUST, fontsize=7, ha="left", va="top")
    ax.set_xlabel("Largest score χ² in a permutation")
    ax.set_ylabel("Permutations")
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    order = ["All non-constant genes", "Near-constant genes removed", "Strongest gene after the filter"]
    ax.legend([handles[label] for label in order], order, loc="upper left", bbox_to_anchor=(-0.3, -0.25), handlelength=1.2, borderaxespad=0,
              labelspacing=0.3)
    panel_label(ax, "b", x=-0.42, y=1.02)

    ax = axes[2]
    frequency = table["added_value_selection_frequency"]
    direction = table["added_value_direction_consistency"]
    shown = frequency > 0
    for tier, colour, size in (("not supported", "#b8c2cc", 4), ("suggestive", SUGGESTIVE, 7), ("robust", ROBUST, 16)):
        chosen = shown & (table["tier"] == tier)
        ax.scatter(frequency[chosen], direction[chosen], s=size, color=colour, edgecolor="none" if tier != "robust" else INK,
                   linewidth=0.4, label=f"{tier} ({int((table['tier'] == tier).sum()):,})", zorder=3 if tier == "robust" else 2)
    robust = table[table["tier"] == "robust"].sort_values("added_value_selection_frequency", ascending=False)
    for step, (_, row) in enumerate(robust.iterrows()):
        # The robust genes sit close together at 100%; their names step down to the right, each on a leader line.
        ax.annotate(row["marker"], (row["added_value_selection_frequency"], row["added_value_direction_consistency"]),
                    xytext=(0.99, 0.92 - 0.066 * step), textcoords="data", fontsize=7, color=ROBUST, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=ROBUST, lw=0.4, shrinkA=1, shrinkB=2))
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0.9, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Selected in subsamples")
    ax.set_ylabel("Same direction in subsamples")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.35, -0.25), markerscale=1.2, handletextpad=0.3, borderaxespad=0, labelspacing=0.3)
    panel_label(ax, "c", x=-0.5, y=1.02)
    save(fig, "fig2_markers")


# ── Figure 3: internal and external estimates in three settings ───────────────
SETTINGS = [
    {"case": "I", "title": "Lung adenocarcinoma\nTCGA; {k} GEO cohorts", "summary": "tcga_markers_summary.json", "pooled": "external_pooled.json",
     "part": "within_cohort", "cohorts": "GEO cohorts"},
    {"case": "IV", "title": "Breast cancer, survival\nMETABRIC; {k} cohorts", "summary": "breast_markers_summary.json",
     "pooled": "breast_external_pooled.json", "part": None, "cohorts": "cohorts"},
    {"case": "V", "title": "ER+ breast, recurrence\nMETABRIC; {k} cohorts", "summary": "breast_er_markers_summary.json",
     "pooled": "breast_er_external_pooled.json", "part": None, "cohorts": "cohorts"},
]


def figure_estimates() -> None:
    settings = []
    for setting in SETTINGS:
        pooled = load(setting["pooled"])
        settings.append({**setting, "signature": load(setting["summary"])["signature"], "pooled": pooled[setting["part"]] if setting["part"] else pooled})
    held_out = load("breast_er_sensitivity.json")
    fig = plt.figure(figsize=(WIDE, 5.3))
    top = fig.add_gridspec(1, 3, left=0.27, right=0.985, top=0.9, bottom=0.575, wspace=0.14)
    below = fig.add_gridspec(1, 1, left=0.27, right=0.69, top=0.445, bottom=0.14)

    # a: the C-index ladder of each setting, internal (development) rows above the external (pooled) ones.
    rows = ["Apparent", "Subsample gap-\nadjusted", "Selection procedure,\nleft-out patients", "Left-out patients,\nclinical only", "External cohorts,\nmodel",
            "External cohorts,\nclinical only"]
    colours = [OPPOSITE, ROBUST, ML, MUTED, ML, MUTED]
    positions = np.arange(len(rows))[::-1].astype(float)
    ladders = []
    for setting in settings:
        signature, pooled = setting["signature"], setting["pooled"]
        ladders.append([
            (signature["apparent_c"], None), (signature["optimism_corrected_c"], None), (signature["signature_c_left_out"], None),
            (signature["clinical_c_left_out"], None),
            (pooled["model_c"]["estimate"], (pooled["model_c"]["hksj_ci_lower"], pooled["model_c"]["hksj_ci_upper"])),
            (pooled["clinical_c"]["estimate"], (pooled["clinical_c"]["hksj_ci_lower"], pooled["clinical_c"]["hksj_ci_upper"])),
        ])
    bounds = [bound for ladder in ladders for value, interval in ladder for bound in (value, *(interval or ()))]
    limits = (np.floor(min(bounds) * 20) / 20, np.ceil(max(bounds) * 20 + 0.4) / 20)
    for column, (setting, ladder) in enumerate(zip(settings, ladders)):
        ax = fig.add_subplot(top[0, column])
        ax.axhspan(-0.55, 1.5, color=LIGHT, zorder=0, lw=0)
        for position, (value, interval), colour in zip(positions, ladder, colours):
            if interval:
                ax.plot(interval, [position, position], color=colour, lw=1.4, solid_capstyle="butt")
            ax.plot(value, position, "o", color=colour, ms=4, mec=INK, mew=0.3, zorder=3)
            ax.text(value, position + 0.2, f"{value:.3f}", ha="center", va="bottom", fontsize=7, color=colour)
        ax.set_xlim(*limits)
        ax.set_ylim(-0.55, len(rows) - 0.3)
        ax.set_yticks(positions, rows if column == 0 else [""] * len(rows), linespacing=1.05)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([tick for tick in np.arange(0.0, 1.01, 0.1) if limits[0] + 1e-9 < tick <= limits[1] + 1e-9]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.set_title(setting["title"].format(k=setting["pooled"]["delta_c"]["k"]), loc="left", fontsize=7.5, linespacing=1.15)
        if column == 1:
            ax.set_xlabel("C-index (external: pooled, 95% HKSJ interval)")
        if column == 0:
            panel_label(ax, "a", x=-0.95, y=1.16)

    # b: the gain over the clinical covariates, inside (left-out patients) and outside the development data.
    ax = fig.add_subplot(below[0, 0])
    entries = []  # (label, estimate, (lower, upper) or None, prediction interval or None, marker, colour)
    headers = []
    for setting in settings:
        signature, pooled = setting["signature"], setting["pooled"]
        headers.append((len(entries), setting["title"].split("\n")[0]))
        entries.append(("Selection procedure (left-out)", signature["signature_gain_left_out"], interval_bounds(signature.get("signature_gain_left_out_ci")),
                        None, "o", ROBUST))
        if setting["case"] == "V":
            sites = held_out["held_out_gain_pooled"]
            entries.append((f"Held-out METABRIC sites ({sites['k']})", sites["estimate"], (sites["hksj_ci_lower"], sites["hksj_ci_upper"]),
                            (sites["pi_lower"], sites["pi_upper"]), "s", SUGGESTIVE))
        delta = pooled["delta_c"]
        entries.append((f"External {setting['cohorts']} ({delta['k']})", delta["estimate"], (delta["hksj_ci_lower"], delta["hksj_ci_upper"]),
                        (delta["pi_lower"], delta["pi_upper"]), "D", ML))
    # One row above each setting's rows holds its name.
    y, labels, spots, header_rows = 0.0, [], [], {}
    for index, entry in enumerate(entries):
        for start, name in headers:
            if start == index:
                header_rows[name] = y
                y += 1.0
        spots.append(y)
        labels.append(entry[0])
        y += 1.0
    total = y
    # The axis holds every estimate and 95% interval; a prediction interval that runs beyond it ends in an arrowhead.
    core = [bound for _, estimate, interval, *_ in entries for bound in (estimate, *(interval or ())) if bound is not None]
    ax.set_xlim(*gain_limits(core))
    beside = ax.get_yaxis_transform()
    for spot, (_, estimate, interval, prediction, marker, colour) in zip(spots, entries):
        row = total - spot
        if prediction and prediction[0] is not None:
            span(ax, prediction[0], prediction[1], row, colour, 0.7)
        if interval and interval[0] is not None:
            span(ax, interval[0], interval[1], row, colour, 2.0, zorder=2)
        ax.plot(estimate, row, marker, color=colour, ms=4.5, mec=INK, mew=0.4, zorder=4)
        text = signed(estimate) + (f" ({signed(interval[0])} to {signed(interval[1])})" if interval and interval[0] is not None else "")
        ax.text(1.03, row, text, transform=beside, ha="left", va="center", fontsize=7, color=INK)
    at_left = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for name, spot in header_rows.items():
        ax.text(0.012, total - spot, name, transform=at_left, ha="left", va="center", fontsize=7, fontweight="bold")
    ax.set_yticks([total - spot for spot in spots], labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0.4, total + 0.1)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.axvline(0.02, color=MUTED, lw=0.6, ls=":")
    gain_ticks(ax)
    ax.set_xlabel("Gain in C over the clinical covariates")
    ax.text(1.03, total + 0.1, "Gain (95% interval)", transform=beside, ha="left", va="bottom", fontsize=7, color=MUTED)
    handles = [Line2D([], [], color=INK, lw=2.0, label="95% interval (pooled: HKSJ)"), Line2D([], [], color=INK, lw=0.7, label="95% prediction interval"),
               Line2D([], [], color=MUTED, lw=0.6, ls=":", label="a gain of 0.02")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.0), ncol=3, handlelength=1.8, columnspacing=1.5, borderaxespad=0.1)
    panel_label(ax, "b", x=-0.62, y=1.04)
    save(fig, "fig3_estimates")


# ── Figure 4: prediction models on the same test patients ─────────────────────
def figure_models() -> None:
    table = load("model_comparison.csv").sort_values("c")
    summary = load("model_comparison_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.7), sharey=True, gridspec_kw={"wspace": 0.08, "left": 0.215, "right": 0.985, "top": 0.86,
                                                                                    "bottom": 0.15})
    y = np.arange(len(table))
    reference = (table["model"] == "Cox PH").to_numpy()
    colours = np.array([INK if cox else ML if family == "Classical ML" else DL for cox, family in zip(reference, table["family"])])
    ax = axes[0]
    ax.errorbar(table["c"], y, xerr=[table["c"] - table["c_lower"], table["c_upper"] - table["c"]], fmt="none", ecolor=colours, lw=1)
    ax.scatter(table["c"][~reference], y[~reference], c=colours[~reference], s=16, zorder=3, edgecolor=INK, linewidth=0.4)
    ax.scatter(table["c"][reference], y[reference], marker="D", color=INK, s=14, zorder=3)
    ax.axvline(table.loc[reference, "c"].iloc[0], color=INK, lw=0.7, ls="--")
    ax.set_yticks(y, table["model"].replace({"DeepHit": "DeepHit variant"}))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("C-index\n(95% bootstrap interval)")
    panel_label(ax, "a", x=-0.62, y=1.02)

    ax = axes[1]
    others = ~reference
    ax.errorbar(table.loc[others, "delta_vs_cox"], y[others], xerr=[table.loc[others, "delta_vs_cox"] - table.loc[others, "delta_lower"],
                table.loc[others, "delta_upper"] - table.loc[others, "delta_vs_cox"]], fmt="none", ecolor=colours[others], lw=1)
    ax.scatter(table.loc[others, "delta_vs_cox"], y[others], c=colours[others], s=16, zorder=3, edgecolor=INK, linewidth=0.4)
    ax.scatter([0], y[reference], marker="D", color=INK, s=14, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Difference from Cox regression\n(paired 95% interval)")
    ax.text(0.02, 0.99, f"{summary['test_patients']} test patients,\n{summary['test_events']} deaths", transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    panel_label(ax, "b", x=-0.05, y=1.02)
    fig.legend(handles=[Line2D([], [], marker="D", color=INK, lw=0, ms=4, label="Cox regression (reference)"),
                        Patch(color=ML, label="Classical machine learning"), Patch(color=DL, label="Deep learning")],
               loc="lower center", bbox_to_anchor=(0.6, 0.9), ncol=3, handlelength=1.2, columnspacing=1.5, borderaxespad=0)
    save(fig, "fig4_models")


# ── Supplementary Figure S1: plasmode simulation on TCGA-LUAD RNA-seq ─────────
def figure_simulation() -> None:
    summary = load("simulation_summary.csv")
    replicates = load("simulation_replicates.csv", dtype=STAMP_DTYPES)
    names = {
        "null_filter": "Null, filter on", "null_no_filter": "Null, filter off",
        "alt_0.30_filter": "β 0.30, filter on", "alt_0.30_no_filter": "β 0.30, filter off",
        "alt_0.45_filter": "β 0.45, filter on", "alt_0.45_no_filter": "β 0.45, filter off",
    }
    summary = summary[summary["scenario"].isin(names)].set_index("scenario").loc[[name for name in names if name in set(summary["scenario"])]]
    fig = plt.figure(figsize=(WIDE, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=1.15, left=0.2, right=0.99, top=0.88, bottom=0.2)

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(summary))[::-1]
    colours = [MUTED if scenario.startswith("null") else ML for scenario in summary.index]
    # The Monte Carlo interval of a rate, cut at zero; the axis always holds the whole interval.
    lower = np.maximum(summary["fwer"] - 1.96 * summary["fwer_mcse"], 0.0)
    upper = summary["fwer"] + 1.96 * summary["fwer_mcse"]
    ax.errorbar(summary["fwer"], y, xerr=[summary["fwer"] - lower, upper - summary["fwer"]], fmt="none", ecolor=colours, lw=1)
    ax.scatter(summary["fwer"], y, c=colours, s=14, zorder=3, edgecolor=INK, linewidth=0.4)
    ax.axvline(0.05, color=OPPOSITE, lw=0.8, ls="--")
    ax.set_yticks(y, [f"{names[scenario]} ({int(row.replicates)})" for scenario, row in summary.iterrows()])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(0.15, float(np.ceil(upper.max() * 1.05 * 20) / 20)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_ylim(-0.6, len(summary) - 0.4)
    ax.set_xlabel("Any unlinked discoveries\n(95% Monte Carlo interval)")
    panel_label(ax, "a", x=-1.25, y=1.02)

    ax = fig.add_subplot(grid[0, 1])
    alternative = summary[summary["beta"] > 0]
    y = np.arange(len(alternative))[::-1]
    ax.scatter(alternative["power_fwer"], y, marker="o", s=16, color=ML, edgecolor=INK, linewidth=0.4, zorder=3, label="family-wise p ≤ 0.05")
    ax.scatter(alternative["power_robust"], y, marker="D", s=12, color=ROBUST, edgecolor=INK, linewidth=0.4, zorder=3, label="robust tier")
    for position, row in zip(y, alternative.itertuples()):
        ax.plot([row.power_robust, row.power_fwer], [position, position], color=LIGHT, lw=2, zorder=1)
    # Each row names its linked and false discoveries per replicate (family-wise tier).
    labels = [f"{names[scenario]}\nlinked {row.linked_found_per_replicate:.1f}, false {row.false_per_replicate:.2f}"
              for scenario, row in alternative.iterrows()]
    ax.set_yticks(y, labels, linespacing=1.1)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(alternative) - 0.4)
    ax.set_xlabel("Share of the 5 true\nmarkers found")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    panel_label(ax, "b", x=-1.2, y=1.02)

    ax = fig.add_subplot(grid[0, 2])
    scored = replicates[replicates["scenario"].isin(["alt_0.30_filter", "alt_0.45_filter"])].dropna(subset=["new_patients_c"])
    estimates = [("apparent_c", "Apparent", OPPOSITE), ("corrected_c", "Gap-adjusted", ROBUST), ("left_out_c", "Left-out", ML)]
    errors = [scored[column] - scored["new_patients_c"] for column, _, _ in estimates]
    positions = np.arange(len(estimates), 0, -1)
    parts = ax.boxplot(errors, positions=positions, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                       medianprops={"color": INK, "lw": 0.9})
    for patch, (_, _, colour) in zip(parts["boxes"], estimates):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    for key in ("whiskers", "caps"):
        for line in parts[key]:
            line.set_color(INK)
            line.set_linewidth(0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    # Each row names its mean error (bias) against the model's C in 3,000 new patients.
    ax.set_yticks(positions, [f"{label}\n{signed(error.mean())}" for (_, label, _), error in zip(estimates, errors)], linespacing=1.1)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("C-index estimate minus\nC in new patients")
    ax.set_title("β 0.30 and 0.45, filter on", fontsize=7, color=MUTED, pad=4)
    panel_label(ax, "c", x=-0.62, y=1.02)
    save(fig, "figS1_simulation")


# ── Supplementary Figure S5: where the positive control's gain went ───────────
def figure_breast_er_sensitivity() -> None:
    folds = load("breast_er_sensitivity.csv").sort_values("held_out_site")
    summary = load("breast_er_sensitivity.json")
    external = load("breast_er_external_validation.csv")
    fig = plt.figure(figsize=(WIDE, 2.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.72, left=0.2, right=0.985, top=0.82, bottom=0.16)

    ax = fig.add_subplot(grid[0, 0])
    pooled = summary["held_out_gain_pooled"]
    # The axis holds each site's interval and the pooled one; a prediction interval beyond it ends in an arrowhead.
    ax.set_xlim(*gain_limits([*folds["held_out_gain_lower"], *folds["held_out_gain_upper"], *folds["internal_gain"],
                              pooled["hksj_ci_lower"], pooled["hksj_ci_upper"]]))
    y = np.arange(len(folds))[::-1] + 1.5
    for position, row in zip(y, folds.itertuples()):
        ax.plot([row.internal_gain, row.held_out_gain], [position, position], color=LIGHT, lw=2, zorder=1)
        ax.plot([row.held_out_gain_lower, row.held_out_gain_upper], [position - 0.12] * 2, color=ML, lw=1)
    ax.scatter(folds["internal_gain"], y, marker="D", s=14, color=ROBUST, edgecolor=INK, linewidth=0.4, zorder=3, label="inside the development sites")
    ax.scatter(folds["held_out_gain"], y - 0.12, marker="s", s=14, color=ML, edgecolor=INK, linewidth=0.4, zorder=3, label="held-out site")
    draw_pooled(ax, pooled, 0.5)
    ax.scatter([summary["internal_gain_mean"]], [0.5], marker="D", s=14, color=ROBUST, edgecolor=INK, linewidth=0.4, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.7)
    gain_ticks(ax)
    ax.set_yticks([*y, 0.5], [f"Site {row.held_out_site} ({row.held_out_n}, {row.held_out_events})" for row in folds.itertuples()] + ["Pooled"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0, len(folds) + 2)
    ax.set_xlabel("Gain in C over the clinical covariates")
    ax.text(-0.02, 1.0, "METABRIC site held out\n(patients, relapses)", transform=ax.transAxes, ha="right", va="bottom", color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    panel_label(ax, "a", x=-0.7, y=1.2)

    ax = fig.add_subplot(grid[0, 1])
    order = external.sort_values(["endpoint", "cohort"], ascending=[False, True]).reset_index(drop=True)
    by_endpoint = summary["external_by_endpoint"]
    # The axis holds each cohort's interval; a pooled interval of two or three cohorts (t with 1 or 2 degrees of freedom)
    # can run far beyond it and ends in an arrowhead.
    ax.set_xlim(*gain_limits([*order["delta_lower"], *order["delta_upper"], *(by_endpoint[endpoint]["gain"]["estimate"] for endpoint in by_endpoint)]))
    y = np.arange(len(order))[::-1] + 2.5
    colours = [ML if endpoint == "rfs" else SUGGESTIVE for endpoint in order["endpoint"]]
    ax.errorbar(order["delta_c"], y, xerr=[order["delta_c"] - order["delta_lower"], order["delta_upper"] - order["delta_c"]], fmt="none", ecolor=colours, lw=1)
    ax.scatter(order["delta_c"], y, c=colours, marker="s", s=14, edgecolor=INK, linewidth=0.4, zorder=3)
    labels = [f"{row.cohort} ({'RFS' if row.endpoint == 'rfs' else 'DMFS'})" for row in order.itertuples()]
    for position, endpoint, colour in ((1.3, "rfs", ML), (0.5, "dmfs", SUGGESTIVE)):
        draw_pooled(ax, by_endpoint[endpoint]["gain"], position, colour, half=0.25)
    ax.axvline(0, color=MUTED, lw=0.7)
    gain_ticks(ax)
    ax.set_yticks([*y, 1.3, 0.5], labels + ["Pooled RFS", "Pooled DMFS"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0, len(order) + 3)
    ax.set_xlabel("Gain in C over the clinical covariates")
    ax.text(0.0, 1.02, "Pooled: diamond 95% HKSJ interval,\nline 95% prediction interval", transform=ax.transAxes, ha="left", va="bottom", color=MUTED)
    panel_label(ax, "b", x=-0.62, y=1.2)
    save(fig, "figS5_positive_control_sensitivity")


# ── Supplementary Figure S6: evidence tiers against external replication ─────
def figure_tier_replication() -> None:
    summary = load("tier_replication.json")
    names = {"robust": "Robust", "suggestive": "Suggestive", "marginal only": "Marginal only",
             "nominal only": "Nominal p < 0.05\nonly", "no evidence": "No evidence"}
    colours = {"robust": ROBUST, "suggestive": SUGGESTIVE, "marginal only": MUTED, "nominal only": ML, "no evidence": "#9aa5b1"}
    titles = {"I": "Lung adenocarcinoma,\nsurvival", "IV": "Breast cancer,\nsurvival", "V": "ER-positive breast,\nrecurrence"}
    cases = list(summary["cases"].items())
    rows = [group for group in names if any(result["groups"].get(group, {}).get("evaluable") for _, result in cases)]
    position = dict(zip(rows, np.arange(len(rows))[::-1]))
    fig, axes = plt.subplots(1, len(cases), figsize=(WIDE, 2.9), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.22, "left": 0.17, "right": 0.985, "top": 0.78, "bottom": 0.2})
    for ax, letter, (case, result) in zip(np.atleast_1d(axes), "abc", cases):
        for group, value in result["groups"].items():
            if not value["evaluable"]:
                continue
            y = position[group]
            low, high = (100 * bound for bound in value["rate_ci"])
            ax.plot([low, high], [y, y], color=colours[group], lw=1.3, solid_capstyle="butt")
            ax.scatter([100 * value["rate"]], [y], s=20, color=colours[group], edgecolor=INK, linewidth=0.4, zorder=3)
            ax.text(99, y + 0.2, f"{value['replicated']:,} of {value['evaluable']:,}", va="bottom", ha="right", fontsize=7, color=MUTED)
        ax.set_yticks(list(position.values()), [names[group] for group in position], linespacing=1.05)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.grid(axis="x", color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
        patients = sum(cohort["n"] for cohort in result["cohorts"])
        ax.set_title(f"{titles.get(case, result['label'])} ({case})\n{len(result['cohorts'])} cohorts, {patients:,} patients", fontsize=7,
                     loc="left", color=INK, linespacing=1.15)
        ax.set_xlabel("Genes replicated (%)")
        panel_label(ax, letter, x=-0.04, y=1.3)
    fig.text(0.17, 0.025, "Binomial intervals are descriptive; genes can be correlated.", fontsize=7, color=MUTED)
    save(fig, "figS6_tier_replication")


# ── Supplementary Figures S2 to S4: each cohort's gain and each gene's replication ──
def figure_luad_external() -> None:
    cohorts = load("external_validation.csv")
    markers = load("external_markers.csv")
    external_figure(
        cohorts[cohorts["scaling"] == "within_cohort"].reset_index(drop=True), load("external_pooled.json")["within_cohort"],
        markers[markers["scaling"] == "within_cohort"], load("tcga_markers_summary.json"), load("tcga_locked_model.json"),
        gain_label="Gain in C over age, sex and stage", robust_note="* robust in TCGA-LUAD", name="figS2_luad_external",
    )


def figure_breast_external() -> None:
    external_figure(
        load("breast_external_validation.csv"), load("breast_external_pooled.json"), load("breast_external_markers.csv"),
        load("breast_markers_summary.json"), load("breast_locked_model.json"),
        gain_label="Gain in C over age, size, nodes, grade, ER", robust_note="* robust in METABRIC", name="figS3_breast_external",
    )


def figure_breast_er_external() -> None:
    external_figure(
        load("breast_er_external_validation.csv"), load("breast_er_external_pooled.json"), load("breast_er_external_markers.csv"),
        load("breast_er_markers_summary.json"), load("breast_er_locked_model.json"),
        gain_label="Gain in C over age, size, nodes, grade", robust_note="* robust in METABRIC ER+", name="figS4_breast_er_external",
        events="relapses or metastases",
    )


def external_figure(cohorts, pooled, markers, summary, recipe, *, gain_label: str, robust_note: str, name: str, events: str = "deaths") -> None:
    genes = list(recipe["markers"])
    rows = len(cohorts)
    forest, heat = 0.2 * rows + 0.95, 0.19 * rows + 1.75
    height = forest + heat + 0.15
    fig = plt.figure(figsize=(WIDE, height))
    upper = fig.add_gridspec(1, 1, left=0.22, right=0.69, top=1 - 0.42 / height, bottom=1 - (forest - 0.05) / height)
    lower = fig.add_gridspec(1, 1, left=0.22, right=0.69, top=(heat - 0.7) / height, bottom=0.95 / height)

    # a: each cohort's gain with its bootstrap interval, and the pooled gain.
    ax = fig.add_subplot(upper[0, 0])
    delta = pooled["delta_c"]
    ax.set_xlim(*gain_limits([*cohorts["delta_lower"], *cohorts["delta_upper"], delta["hksj_ci_lower"], delta["hksj_ci_upper"]]))
    labels = [f"{row.cohort} ({row.n}, {row.events})" for row in cohorts.itertuples()]
    y = np.arange(rows)[::-1] + 1.5
    ax.errorbar(cohorts["delta_c"], y, xerr=[cohorts["delta_c"] - cohorts["delta_lower"], cohorts["delta_upper"] - cohorts["delta_c"]],
                fmt="s", color=ML, ms=3.2, lw=0.9, capsize=0)
    draw_pooled(ax, delta, 0.5)
    ax.axvline(0, color=MUTED, lw=0.7)
    gain_ticks(ax)
    ax.set_yticks([*y, 0.5], [*labels, "Pooled (random effects)"])
    ax.set_ylim(0, rows + 1.9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(gain_label)
    beside = ax.get_yaxis_transform()
    for position, row in zip(y, cohorts.itertuples()):
        ax.text(1.03, position, f"{signed(row.delta_c)} ({signed(row.delta_lower)} to {signed(row.delta_upper)})", transform=beside,
                ha="left", va="center", fontsize=7)
    ax.text(1.03, 0.5, f"{signed(delta['estimate'])} (HKSJ {signed(delta['hksj_ci_lower'])} to {signed(delta['hksj_ci_upper'])})"
            + (f"\nPI {signed(delta['pi_lower'])} to {signed(delta['pi_upper'])}" if delta.get("pi_lower") is not None else ""),
            transform=beside, ha="left", va="center", fontsize=7, linespacing=1.15)
    ax.text(-0.03, 1.0, f"Cohort (patients,\n{events})", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, linespacing=1.1)
    ax.text(1.03, 1.0, "Gain (95% interval)", transform=ax.transAxes, ha="left", va="bottom", color=MUTED)
    panel_label(ax, "a", x=-0.45, y=1.03)

    # b: each locked gene's replication test in each cohort.
    ax = fig.add_subplot(lower[0, 0])
    codes = {"not measured": 0, "opposite direction": 1, "same direction": 2, "replicated": 3}
    matrix = np.zeros((rows, len(genes)))
    for i, cohort in enumerate(cohorts["cohort"]):
        for j, gene in enumerate(genes):
            row = markers[(markers["marker"] == gene) & (markers["cohort"] == cohort)].iloc[0]
            if not row["measured"]:
                matrix[i, j] = codes["not measured"]
            elif row["replicated"]:
                matrix[i, j] = codes["replicated"]
            elif row["same_direction"]:
                matrix[i, j] = codes["same direction"]
            else:
                matrix[i, j] = codes["opposite direction"]
    colours = matplotlib.colors.ListedColormap([LIGHT, "#f1b8b0", "#cfe7d7", ROBUST])
    ax.imshow(matrix, cmap=colours, vmin=-0.5, vmax=3.5, aspect="auto")
    robust = set(summary["robust"])
    ax.set_xticks(range(len(genes)), [f"{gene}{'*' if gene in robust else ''}" for gene in genes], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(rows), list(cohorts["cohort"]))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(genes)), minor=True)
    ax.set_yticks(np.arange(-0.5, rows), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    replicated = (matrix == codes["replicated"]).sum(axis=0)
    measured = (matrix != codes["not measured"]).sum(axis=0)
    for j in range(len(genes)):
        ax.text(j, -0.62, f"{replicated[j]}/{measured[j]}", ha="center", va="bottom", fontsize=7, color=ROBUST if replicated[j] else MUTED)
    handles = [Patch(color=colour, label=label) for label, colour in
               (("replicated (Holm p ≤ 0.05,\nsame direction)", ROBUST), ("same direction, not significant", "#cfe7d7"),
                ("opposite direction", "#f1b8b0"), ("not measured on the platform", LIGHT))]
    # The notes close the legend, so they stay below it however few cohorts the heatmap has.
    notes = [Patch(alpha=0, label=robust_note), Patch(alpha=0, label="above each gene: cohorts\nreplicated / measured")]
    legend = ax.legend(handles=handles + notes, loc="upper left", bbox_to_anchor=(1.03, 1.0), handlelength=1, labelspacing=0.45, borderaxespad=0)
    for text in legend.get_texts()[len(handles):]:
        text.set_color(MUTED)
    panel_label(ax, "b", x=-0.45, y=1.14)
    save(fig, name)


# ── Figure 5: the pipelines commonly used to publish signatures, against SurvStudio ─
OTHER = "#8a94a3"


def comparison_null_rows(rows: list[dict]) -> list[tuple[str, dict, bool]]:
    """The claims under the null, as (label, row, is SurvStudio), in the figure's order."""
    wanted = [
        ("SurvStudio: a marker declared", lambda row: row["approach"] == "SurvStudio" and row["claim"].startswith("at least one marker"), True),
        ("SurvStudio: gain verdict 'adds'", lambda row: row["approach"] == "SurvStudio" and "gain verdict" in row["claim"], True),
        ("Mime, 3 selection cohorts: C ≥ 0.55", lambda row: row["approach"] == "P1 Mime" and str(row.get("design", "")).startswith("3 selection"), False),
        ("Mime, 7 selection cohorts: C ≥ 0.55", lambda row: row["approach"] == "P1 Mime" and row.get("design") == "all seven", False),
        ("Cox → LASSO: training p < 0.05", lambda row: row["approach"].startswith("P2"), False),
        ("Best cut-off: a gene with p < 0.05", lambda row: row["approach"].startswith("P3"), False),
    ]
    found = []
    for label, match, ours in wanted:
        hits = [row for row in rows if match(row)]
        if len(hits) != 1:
            raise KeyError(f"one null row for {label!r} (found {len(hits)})")
        found.append((label, hits[0], ours))
    return found


def comparison_real_rows(rows: list[dict]) -> list[tuple[str, dict, bool]]:
    """The TCGA-LUAD to GEO rows, as (label, row, is SurvStudio), in the figure's order."""
    wanted = [
        ("SurvStudio", lambda row: row["approach"] == "SurvStudio", True),
        ("SurvStudio, genes all cohorts measure", lambda row: row["approach"].startswith("SurvStudio ("), True),
        ("Mime, 7 selection cohorts", lambda row: row["approach"] == "P1 Mime" and str(row.get("design", "")).startswith("all seven"), False),
        ("Mime, 3 selection + 4 sealed", lambda row: row["approach"] == "P1 Mime" and str(row.get("design", "")).startswith("3 selection"), False),
        ("Cox → LASSO", lambda row: row["approach"].startswith("P2"), False),
    ]
    found = []
    for label, match, ours in wanted:
        hits = [row for row in rows if match(row)]
        if len(hits) != 1:
            raise KeyError(f"one real-data row for {label!r} (found {len(hits)})")
        found.append((label, hits[0], ours))
    return found


def figure_comparison() -> None:
    table = load("competitors_table.json")
    null = comparison_null_rows(table["experiment_2"])
    real = comparison_real_rows(table["experiment_1"])
    fig = plt.figure(figsize=(WIDE, 5.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.05], width_ratios=[1.0, 1.0], hspace=0.68, wspace=0.12,
                            left=0.3, right=0.985, top=0.93, bottom=0.1)

    # a: how often each approach makes its claim when no marker adds anything (script 06's null).
    ax = fig.add_subplot(grid[0, :])
    y = np.arange(len(null))[::-1]
    for position, (label, row, ours) in zip(y, null):
        rate, mcse = 100 * float(row["claim_rate"]), 100 * float(row.get("claim_rate_mcse") or 0.0)
        ax.barh(position, rate, height=0.62, color=ROBUST if ours else OTHER)
        if mcse > 0:
            ax.plot([max(rate - 1.96 * mcse, 0), min(rate + 1.96 * mcse, 100)], [position, position], color=INK, lw=0.8)
        bounds = row.get("claim_rate_failure_bounds")
        if bounds and bounds[1] > bounds[0]:
            ax.plot([100 * bounds[0], 100 * bounds[1]], [position - 0.22, position - 0.22], color=MUTED, lw=3)
        ax.text(min(rate + 1.96 * mcse, 100) + 1.5, position, f"{rate:.1f}%" if rate < 10 else f"{rate:.0f}%", va="center", fontsize=7)
    labels = [label + (f"\n{row['replicates']}/{row['planned_replicates']} completed" if row.get("planned_replicates") else "")
              for label, row, _ in null]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Claim rate among completed fits (%)\nThin: 95% Monte Carlo; thick: bounds including failures")
    ax.set_title("Different claims under a conditional marker null (TCGA-LUAD plasmode)", loc="left", fontsize=7.5)
    panel_label(ax, "a", x=-0.4)

    # b: the C-index each approach would report, against its C-index in the external cohorts.
    ax = fig.add_subplot(grid[1, 0])
    y = np.arange(len(real))[::-1]
    for position, (label, row, ours) in zip(y, real):
        colour = ROBUST if ours else OTHER
        reported, external = float(row["reported_c"]), float(row["external_c"])
        ax.annotate("", xy=(external, position), xytext=(reported, position),
                    arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.0, "shrinkA": 2, "shrinkB": 2, "mutation_scale": 7})
        ax.scatter([reported], [position], s=18, facecolor="white", edgecolor=colour, linewidth=1.0, zorder=3)
        ax.scatter([external], [position], s=18, color=colour, edgecolor=INK, linewidth=0.4, zorder=3)
    clinical = [float(row["external_clinical_c"]) for _, row, _ in real if row.get("external_clinical_c") is not None]
    if clinical:
        ax.axvline(float(np.median(clinical)), color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(y, [label for label, _, _ in real])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.55, 0.8)
    ax.set_xticks([0.55, 0.6, 0.65, 0.7, 0.75])
    if clinical:
        ax.text(float(np.median(clinical)) + 0.003, 0.5, "clinical only", fontsize=7, color=MUTED, va="center")
    ax.set_xlabel("C-index: reported (open) → external (filled)")
    ax.set_title("TCGA-LUAD → seven GEO cohorts", loc="left", fontsize=7.5)
    panel_label(ax, "b", x=-0.72)

    # c: the gain over the clinical covariates in the external cohorts, with SurvStudio's internal estimate above it.
    ax = fig.add_subplot(grid[1, 1], sharey=ax)
    values = []
    for _, row, _ in real:
        values += [row.get("gain_hksj_lower") if row.get("gain_hksj_lower") is not None else row.get("gain_lower"),
                   row.get("gain_hksj_upper") if row.get("gain_hksj_upper") is not None else row.get("gain_upper"),
                   row.get("reported_gain_lower"), row.get("reported_gain_upper")]
    ax.set_xlim(*gain_limits([value for value in values if value is not None]))
    for position, (label, row, ours) in zip(y, real):
        colour = ROBUST if ours else OTHER
        low = row.get("gain_hksj_lower") if row.get("gain_hksj_lower") is not None else row.get("gain_lower")
        high = row.get("gain_hksj_upper") if row.get("gain_hksj_upper") is not None else row.get("gain_upper")
        # The split design gives the median gain and its range over the 35 splits, drawn dashed.
        split = str(row.get("design", "")).startswith("3 selection")
        if low is not None and high is not None:
            left, right = ax.get_xlim()
            ax.plot([max(low, left), min(high, right)], [position - 0.12, position - 0.12], color=colour, lw=1.2, ls="--" if split else "-")
        ax.scatter([row["gain"]], [position - 0.12], s=18, color=colour, edgecolor=INK, linewidth=0.4, zorder=3)
        if row.get("reported_gain_lower") is not None:
            span(ax, row["reported_gain_lower"], row["reported_gain_upper"], position + 0.2, colour, 0.8)
            ax.scatter([row["reported_gain"]], [position + 0.2], s=16, marker="D", facecolor="white", edgecolor=colour, linewidth=1.0, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    gain_ticks(ax)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Gain in C over age, sex and stage")
    ax.legend(handles=[Line2D([], [], marker="D", color=ROBUST, markerfacecolor="white", lw=0.8, ms=4, label="internal (left out)"),
                       Line2D([], [], marker="o", color=OTHER, lw=1.2, ms=4, label="external (pooled)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7, handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
    panel_label(ax, "c", x=-0.05)
    save(fig, "fig5_comparison")


def main(wanted: set[str], *, available_only: bool = False) -> None:
    refused = {}
    if not wanted and not (RESULTS / "competitors_table.json").exists():
        # The comparison's results come from run_competitors.sh, which runs after run_all.sh; draw it once they exist.
        print("not drawn yet: comparison (run run_competitors.sh first)")
    for name, draw in (("workflow", figure_workflow), ("interface", figure_interface), ("markers", figure_markers),
                       ("estimates", figure_estimates), ("models", figure_models), ("comparison", figure_comparison),
                       ("simulation", figure_simulation), ("luad_external", figure_luad_external),
                       ("breast_external", figure_breast_external), ("breast_er_external", figure_breast_er_external),
                       ("breast_er_sensitivity", figure_breast_er_sensitivity), ("tiers", figure_tier_replication)):
        if wanted and name not in wanted:
            continue
        if name == "comparison" and not wanted and not (RESULTS / "competitors_table.json").exists():
            continue
        LOADED.clear()
        try:
            draw()
        except FileNotFoundError as exc:
            plt.close("all")
            if not available_only:
                raise
            # Partial analyses can feed composite figures that also need another case study.
            # Only an absent input is pending; stale or inconsistent existing results still fail.
            print(f"pending: {name} (missing {exc.filename}; run the other contributing steps)")
        except (MixedResults, NotPrintable) as exc:
            refused[name] = exc.args[0]
        except KeyError as exc:
            # A value the figure needs is not in its results: they were written by an earlier version of the scripts.
            plt.close("all")
            refused[name] = [f"no {exc.args[0]!r} in {', '.join(dict.fromkeys(LOADED))}: written by an earlier version of the analysis "
                             "scripts; rerun the steps that write them"]
    if refused:
        for name, problems in refused.items():
            print(f"not drawn: {name}", *(f"  - {problem}" for problem in problems), sep="\n", file=sys.stderr)
        raise SystemExit(f"{len(refused)} figure(s) not drawn, from mixed, stale or older results or not printable: {', '.join(refused)}")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    available_only = "--available" in arguments
    main(set(arguments) - {"--available"}, available_only=available_only)

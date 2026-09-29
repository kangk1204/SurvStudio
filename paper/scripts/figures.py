"""Figures 1 to 6 and Supplementary Figures S1 to S3 of the software paper, drawn from the files the analysis
scripts write to results/.

Needs matplotlib, numpy and pandas (no SurvStudio). Writes PNG (300 dpi) and PDF to paper/figures/.
Usage: python figures.py [name ...] draws the named figures only (workflow, markers, external, models, breast,
breast_er, simulation, breast_er_sensitivity, tiers); without names, all of them.

A figure is drawn only from results of one run of the analysis: every file it reads must carry the stamp its
script wrote (results/stamps/), unchanged since, from the same SurvStudio commit and the same analysis code, and
computed from the results files that are there now (common.result_problems). A figure whose files fail is not
drawn, the others are, and the run ends with an error that names the files; rerun the steps that wrote them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from common import STAMP_DTYPES, result_problems  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
FIGURES.mkdir(exist_ok=True)
WIDE = 6.85  # inches, a two-column figure
INK = "#1f2933"
MUTED = "#6b7785"
ML = "#2f65d9"
DL = "#db7e15"
ROBUST = "#2e8b57"
SUGGESTIVE = "#d4a017"
OPPOSITE = "#c0392b"
LIGHT = "#e8edf2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7,
})


# The results files the figure being drawn has read.
LOADED: list[str] = []


class MixedResults(Exception):
    """The files a figure reads do not come from one run of the analysis (common.result_problems)."""


def load(name: str, **options):
    LOADED.append(name)
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if name.endswith(".json") else pd.read_csv(path, **options)


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str) -> None:
    problems = result_problems(LOADED)
    if problems:
        plt.close(fig)
        raise MixedResults(problems)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ── Figure 1: what SurvStudio checks by default ────────────────────────────────
def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(WIDE, 3.4))
    ax.set_xlim(-1.5, 101.5)
    ax.set_ylim(0, 60)
    ax.axis("off")
    # Column: (left edge, width, fill, edge colour); boxes: (title, text, height), top to bottom.
    columns = {
        "inputs": (0, 26.5, "#dbe7fb", ML, "Inputs", [
            ("Clinical table", "CSV, Excel or Parquet", 9),
            ("Omics matrix", "CSV, TSV or Parquet, or .gz\nfiles as GEO and Xena serve\nthem; TCGA barcodes matched", 13.5),
            ("External cohort", "same or another platform", 9),
        ]),
        "checks": (30, 40, "#dcefe4", ROBUST, "Checks run by default", [
            ("Survival curves and Cox regression", "Kaplan-Meier with numbers at risk, log-rank,\nEfron Cox model, proportional-hazards tests", 11),
            ("Marker evaluation", "score tests for added value over clinical data\n→ family-wise error by permutation\n    (Westfall-Young; marker residuals permuted)\n→ whole screen repeated on subsamples: tiers\n→ corrected C against clinical-only C", 17.5),
            ("Prediction models", "machine and deep learning on the same splits\n(holdout, repeated CV, locked test); paired\nintervals for the difference from Cox", 12.5),
        ]),
        "outputs": (73.5, 26.5, "#fbe7d4", DL, "Outputs", [
            ("Figures and tables", "forest plots, marker tables,\nleaderboards with intervals", 11),
            ("Reporting checklists", "REMARK and TRIPOD+AI filled\nfrom the run (Word, Markdown)", 11),
            ("Locked model", "hashed recipe applied\nunchanged to the external\ncohort", 12.5),
        ]),
    }
    placed: dict[str, list[tuple[float, float, float, float]]] = {}
    for key, (x, width, fill, edge, title, boxes) in columns.items():
        ax.text(x + width / 2, 57.5, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=edge)
        top, bottom = 54, 7
        gap = (top - bottom - sum(height for *_, height in boxes)) / (len(boxes) - 1)
        placed[key] = []
        for name, body, height in boxes:
            y = top - height
            ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2,rounding_size=1.2", linewidth=0.8, edgecolor=edge, facecolor=fill))
            ax.text(x + 1.2, y + height - 1.5, name, ha="left", va="top", fontsize=7.3, fontweight="bold")
            ax.text(x + 1.2, y + height - 4.3, body, ha="left", va="top", fontsize=6.0, linespacing=1.3)
            placed[key].append((x, y, width, height))
            top = y - gap
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=12, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(29.5, 33), xytext=(27.0, 33), arrowprops=arrow)
    ax.annotate("", xy=(73.0, 33), xytext=(70.5, 33), arrowprops=arrow)
    # The external cohort reaches the locked model under the middle column.
    x_e, y_e, w_e, h_e = placed["inputs"][2]
    x_l, y_l, w_l, h_l = placed["outputs"][2]
    route = dict(color=DL, lw=1.1, ls=(0, (3, 2)))
    ax.plot([x_e + w_e / 2, x_e + w_e / 2], [y_e - 0.2, 2.5], **route)
    ax.plot([x_e + w_e / 2, x_l + w_l / 2], [2.5, 2.5], **route)
    ax.annotate("", xy=(x_l + w_l / 2, y_l - 0.3), xytext=(x_l + w_l / 2, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=DL, lw=1.1, mutation_scale=10, shrinkA=0, shrinkB=0))
    ax.text(50, 3.2, "external validation, without refitting", ha="center", va="bottom", fontsize=6.3, color=DL)
    save(fig, "fig1_workflow")


# ── Figure 2: genome-wide markers in TCGA-LUAD ─────────────────────────────────
def figure_markers() -> None:
    summary = load("tcga_markers_summary.json")
    funnel = summary["funnel"]
    table = load("tcga_markers.csv")
    maxima = load("permutation_maxima.csv")
    permutation = load("permutation_maximum_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.55), gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.55})

    ax = axes[0]
    stages = [
        ("Genes in the Xena file", funnel["genes_in_file"]),
        ("Tested (varying in ≥10% of patients)", funnel["tested"]),
        ("p < 0.05 beyond age, sex, stage", funnel["adjusted_p_below_0_05"]),
        ("Benjamini-Hochberg q ≤ 0.05", funnel["bh_q_below_0_05"]),
        ("Family-wise p ≤ 0.05 and stable", funnel["robust"]),
    ]
    positions = np.arange(len(stages))[::-1]
    colours = [LIGHT, "#c9d6e5", "#9fb6d0", SUGGESTIVE, ROBUST]
    ax.barh(positions, [count for _, count in stages], color=colours, edgecolor=MUTED, linewidth=0.5, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1, 2e5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_yticks(positions, [label for label, _ in stages])
    for position, (_, count) in zip(positions, stages):
        ax.text(count * 1.25, position, f"{count:,}", va="center", fontsize=7)
    ax.set_xlabel("Genes (log scale)")
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a", x=-1.05)

    ax = axes[1]
    bins = np.logspace(np.log10(10), np.log10(max(maxima["all_genes"].max(), 300) * 1.1), 40)
    ax.hist(maxima["all_genes"], bins=bins, color=OPPOSITE, alpha=0.55, label="All non-constant genes")
    ax.hist(maxima["near_constant_removed"], bins=bins, color=ML, alpha=0.7, label="Near-constant genes removed")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    height = ax.get_ylim()[1]
    for key, colour, level in (("all_genes", OPPOSITE, 0.62), ("near_constant_removed", ML, 0.93)):
        threshold = permutation[key]["threshold_95"]
        ax.axvline(threshold, color=colour, lw=1, ls="--")
        ax.text(threshold * 0.93, height * level, f"95%: {threshold:.0f}", color=colour, fontsize=6.3, ha="right")
    # The strongest observed gene that passes the near-constant filter (a near-constant gene may score higher).
    top = next(item for item in permutation["top_observed"] if item["mode_share"] <= 0.9)
    ax.axvline(top["chi2"], color=ROBUST, lw=1, label="Strongest gene after the filter")
    ax.text(top["chi2"] * 1.07, height * 0.8, f"{top['marker']}\nχ² = {top['chi2']:.1f}", color=ROBUST, fontsize=6.3, ha="left", va="top")
    ax.set_xlabel("Largest score χ² in a permutation")
    ax.set_ylabel("Permutations")
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    order = ["All non-constant genes", "Near-constant genes removed", "Strongest gene after the filter"]
    ax.legend([handles[label] for label in order], order, loc="lower center", bbox_to_anchor=(0.6, 1.0), fontsize=6, handlelength=1.2,
              borderaxespad=0.2)
    panel_label(ax, "b", x=-0.3)

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
                    xytext=(0.99, 0.93 - 0.055 * step), textcoords="data", fontsize=6, color=ROBUST, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=ROBUST, lw=0.4, shrinkA=1, shrinkB=2))
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0.9, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Selected in subsamples")
    ax.set_ylabel("Same direction in subsamples")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", fontsize=6, markerscale=1.2, handletextpad=0.2)
    panel_label(ax, "c", x=-0.3)
    save(fig, "fig2_markers")


# ── Figure 3: the locked model in seven GEO cohorts ────────────────────────────
def figure_external() -> None:
    cohorts = load("external_validation.csv")
    markers = load("external_markers.csv")
    external_figure(
        cohorts[cohorts["scaling"] == "within_cohort"].reset_index(drop=True),
        load("external_pooled.json")["within_cohort"],
        markers[markers["scaling"] == "within_cohort"],
        load("tcga_markers_summary.json"),
        load("tcga_locked_model.json"),
        gain_label="C gain over age, sex and stage", development="TCGA", external="GEO",
        pooled_note="GEO: pooled over seven cohorts", robust_note="* robust in TCGA-LUAD", name="fig3_external",
    )


# ── Figure 5: breast cancer, METABRIC to the MetaGxBreast cohorts ─────────────
def figure_breast() -> None:
    cohorts = load("breast_external_validation.csv")
    external_figure(
        cohorts, load("breast_external_pooled.json"), load("breast_external_markers.csv"),
        load("breast_markers_summary.json"), load("breast_locked_model.json"),
        gain_label="C gain over age, size, nodes, grade, ER", development="METABRIC", external="External",
        pooled_note=f"External: pooled over {len(cohorts)} cohorts", robust_note="* robust in METABRIC", name="fig5_breast",
    )


# ── Figure 6: positive control, recurrence in ER-positive breast cancer ───────
def figure_breast_er() -> None:
    cohorts = load("breast_er_external_validation.csv")
    external_figure(
        cohorts, load("breast_er_external_pooled.json"), load("breast_er_external_markers.csv"),
        load("breast_er_markers_summary.json"), load("breast_er_locked_model.json"),
        gain_label="C gain over age, size, nodes, grade", development="METABRIC", external="External",
        pooled_note=f"External: pooled over {len(cohorts)} cohorts", robust_note="* robust in METABRIC ER+", name="fig6_breast_er",
        events="relapses or metastases",
    )


def external_figure(cohorts, pooled, markers, summary, recipe, *, gain_label: str, development: str, external: str,
                    pooled_note: str, robust_note: str, name: str, events: str = "deaths") -> None:
    fig = plt.figure(figsize=(WIDE, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.25, 1.0], wspace=0.95, hspace=0.62)

    ax = fig.add_subplot(grid[0, 0])
    labels = [f"{row.cohort} ({row.n}, {row.events})" for row in cohorts.itertuples()]
    y = np.arange(len(cohorts))[::-1] + 1.5
    ax.errorbar(cohorts["delta_c"], y, xerr=[cohorts["delta_c"] - cohorts["delta_lower"], cohorts["delta_upper"] - cohorts["delta_c"]],
                fmt="s", color=ML, ms=3.5, lw=0.9, capsize=0)
    delta = pooled["delta_c"]
    ax.fill([delta["ci_lower"], delta["estimate"], delta["ci_upper"], delta["estimate"]], [0.5, 0.78, 0.5, 0.22], color=INK)
    ax.axvline(0, color=MUTED, lw=0.7)
    # The pooled estimate sits in its row label, clear of the diamond whatever the width of its interval.
    pooled_label = f"Pooled (random effects)\n{delta['estimate']:+.3f} ({delta['ci_lower']:+.3f} to {delta['ci_upper']:+.3f})"
    ax.set_yticks([*y, 0.5], [*labels, pooled_label])
    ax.set_ylim(0, len(cohorts) + 2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(gain_label)
    ax.text(-0.02, 1.0, f"Cohort (patients, {events})", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=MUTED)
    panel_label(ax, "a", x=-0.95, y=1.06)

    ax = fig.add_subplot(grid[0, 1])
    signature = summary["signature"]
    model_c, clinical_c = pooled["model_c"], pooled["clinical_c"]
    # Neither pair of model and clinical rows is read as the gain: the two left-out means may average different
    # subsamples, and the external model and clinical C are each pooled on their own (their own weights and
    # between-cohort variance), so their gap is not the pooled gain. Each pair carries its paired gain instead:
    # SurvStudio's on the same left-out patients, and the pooled per-cohort difference of panel a.
    ladder = [
        (f"{development} apparent", signature["apparent_c"], None, OPPOSITE),
        (f"{development} optimism-corrected", signature["optimism_corrected_c"], None, ROBUST),
        (f"{development} left out, model", signature["signature_c_left_out"], None, ML),
        (f"{development} left out, clinical only", signature["clinical_c_left_out"], None, MUTED),
        (f"{external}, model\n(pooled separately)", model_c["estimate"], (model_c["ci_lower"], model_c["ci_upper"]), ML),
        (f"{external}, clinical only\n(pooled separately)", clinical_c["estimate"], (clinical_c["ci_lower"], clinical_c["ci_upper"]), MUTED),
    ]
    positions = np.arange(len(ladder))[::-1]
    for position, (_, value, interval, colour) in zip(positions, ladder):
        if interval:
            ax.plot(interval, [position, position], color=colour, lw=1)
        ax.plot(value, position, "o", color=colour, ms=4.5)
        ax.text(value, position + 0.25, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3, color=colour)
    ax.set_yticks(positions, [label for label, *_ in ladder])
    ax.tick_params(axis="y", length=0)
    values = [bound for _, value, interval, _ in ladder for bound in (value, *(interval or ()))]
    ax.set_xlim(np.floor(min(values) * 50 - 1) / 50, np.ceil(max(values) * 50 + 1) / 50)
    ax.set_ylim(-0.6, len(ladder) - 0.2)
    ax.set_xlabel("C-index")
    ax.text(0.99, 0.02, pooled_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=MUTED)
    delta = pooled["delta_c"]
    gains = (
        (positions[2], positions[3], f"paired gain\n{signature['signature_gain_left_out']:+.3f}"),
        (positions[4], positions[5], f"pooled paired gain\n{delta['estimate']:+.3f} (panel a)\n({delta['ci_lower']:+.3f} to {delta['ci_upper']:+.3f})"),
    )
    # A bracket to the right of each pair of rows (x in axes fractions, y in rows).
    beside = ax.get_yaxis_transform()
    for top, bottom, text in gains:
        ax.plot([1.02, 1.04, 1.04, 1.02], [top, top, bottom, bottom], color=INK, lw=0.7, transform=beside, clip_on=False)
        ax.text(1.07, (top + bottom) / 2, text, transform=beside, ha="left", va="center", fontsize=6.0, color=INK, linespacing=1.25)
    panel_label(ax, "b", x=-0.85, y=1.06)

    ax = fig.add_subplot(grid[1, :])
    genes = list(recipe["markers"])
    codes = {"not measured": 0, "opposite direction": 1, "same direction": 2, "replicated": 3}
    matrix = np.zeros((len(cohorts), len(genes)))
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
    ax.set_xticks(range(len(genes)), [f"{gene}{'*' if gene in robust else ''}" for gene in genes], rotation=35, ha="right")
    ax.set_yticks(range(len(cohorts)), list(cohorts["cohort"]))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(genes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cohorts)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    replicated = (matrix == codes["replicated"]).sum(axis=0)
    measured = (matrix != codes["not measured"]).sum(axis=0)
    for j in range(len(genes)):
        ax.text(j, -0.75, f"{replicated[j]}/{measured[j]}", ha="center", va="bottom", fontsize=6, color=ROBUST if replicated[j] else MUTED)
    handles = [Patch(color=colour, label=label) for label, colour in
               (("replicated (Holm p ≤ 0.05, same direction)", ROBUST), ("same direction, not significant", "#cfe7d7"),
                ("opposite direction", "#f1b8b0"), ("not measured on the platform", LIGHT))]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6, handlelength=1)
    ax.text(1.01, 0.02, f"{robust_note}\nabove each gene: cohorts\nreplicated / measured", transform=ax.transAxes, fontsize=6, color=MUTED, va="bottom")
    panel_label(ax, "c", x=-0.14, y=1.12)
    save(fig, name)


# ── Figure 4: prediction models on the same test patients ─────────────────────
def figure_models() -> None:
    table = load("model_comparison.csv").sort_values("c")
    summary = load("model_comparison_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.7), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(table))
    colours = [ML if family == "Classical ML" else DL for family in table["family"]]
    ax = axes[0]
    ax.errorbar(table["c"], y, xerr=[table["c"] - table["c_lower"], table["c_upper"] - table["c"]], fmt="none", ecolor=colours, lw=1)
    ax.scatter(table["c"], y, c=colours, s=18, zorder=3, edgecolor=INK, linewidth=0.4)
    cox = table.loc[table["model"] == "Cox PH", "c"].iloc[0]
    ax.axvline(cox, color=ML, lw=0.8, ls="--")
    ax.set_yticks(y, [f"{model} ({'ML' if family == 'Classical ML' else 'DL'})" for model, family in zip(table["model"], table["family"])])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("C-index (95% bootstrap interval)")
    ax.legend(handles=[Patch(color=ML, label="Classical machine learning"), Patch(color=DL, label="Deep learning")], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6, borderaxespad=0.2)
    panel_label(ax, "a", x=-0.62)

    ax = axes[1]
    others = table["model"] != "Cox PH"
    ax.errorbar(table.loc[others, "delta_vs_cox"], y[others], xerr=[table.loc[others, "delta_vs_cox"] - table.loc[others, "delta_lower"],
                table.loc[others, "delta_upper"] - table.loc[others, "delta_vs_cox"]], fmt="none", ecolor=np.array(colours)[others], lw=1)
    ax.scatter(table.loc[others, "delta_vs_cox"], y[others], c=np.array(colours)[others], s=18, zorder=3, edgecolor=INK, linewidth=0.4)
    ax.scatter([0], y[~others], marker="D", color=INK, s=14, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Difference from Cox regression (paired 95% interval)")
    ax.text(0.02, 0.99, f"{summary['test_patients']} test patients,\n{summary['test_events']} deaths", transform=ax.transAxes, fontsize=6.3, color=MUTED, va="top")
    panel_label(ax, "b", x=-0.04)
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
    fig = plt.figure(figsize=(WIDE, 2.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.9], wspace=0.95)

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(summary))[::-1]
    colours = [MUTED if scenario.startswith("null") else ML for scenario in summary.index]
    ax.errorbar(summary["fwer"], y, xerr=1.96 * summary["fwer_mcse"], fmt="none", ecolor=colours, lw=1)
    ax.scatter(summary["fwer"], y, c=colours, s=16, zorder=3, edgecolor=INK, linewidth=0.4)
    ax.axvline(0.05, color=OPPOSITE, lw=0.8, ls="--")
    ax.set_yticks(y, [f"{names[scenario]} ({int(row.replicates)})" for scenario, row in summary.iterrows()])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(0.12, float((summary["fwer"] + 2 * summary["fwer_mcse"]).max())))
    ax.set_ylim(-0.6, len(summary) - 0.4)
    ax.set_xlabel("Family-wise error rate\n(95% Monte Carlo interval)")
    panel_label(ax, "a", x=-0.95)

    ax = fig.add_subplot(grid[0, 1])
    alternative = summary[summary["beta"] > 0]
    y = np.arange(len(alternative))[::-1]
    ax.scatter(alternative["power_fwer"], y, marker="o", s=18, color=ML, edgecolor=INK, linewidth=0.4, zorder=3, label="family-wise p ≤ 0.05")
    ax.scatter(alternative["power_robust"], y, marker="D", s=14, color=ROBUST, edgecolor=INK, linewidth=0.4, zorder=3, label="robust tier")
    for position, row in zip(y, alternative.itertuples()):
        ax.plot([row.power_robust, row.power_fwer], [position, position], color=LIGHT, lw=2, zorder=1)
    # Each row names its linked and false discoveries per replicate (family-wise tier).
    labels = [f"{names[scenario]}\nlinked {row.linked_found_per_replicate:.1f}, false {row.false_per_replicate:.2f}"
              for scenario, row in alternative.iterrows()]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(alternative) - 0.4)
    ax.set_xlabel("Share of the 5 true markers found")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6, handletextpad=0.2, borderaxespad=0.2)
    panel_label(ax, "b", x=-0.95)

    ax = fig.add_subplot(grid[0, 2])
    scored = replicates[replicates["scenario"].isin(["alt_0.30_filter", "alt_0.45_filter"])].dropna(subset=["new_patients_c"])
    estimates = [("apparent_c", "Apparent", OPPOSITE), ("corrected_c", "Corrected", ROBUST), ("left_out_c", "Left-out", ML)]
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
    ax.set_yticks(positions, [f"{label}\n{error.mean():+.3f}" for (_, label, _), error in zip(estimates, errors)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("C-index estimate minus\nC in new patients")
    ax.set_title("β 0.30 and 0.45, filter on", fontsize=6.5, color=MUTED, pad=4)
    panel_label(ax, "c", x=-0.62)
    save(fig, "figS1_simulation")


# ── Supplementary Figure S2: where the positive control's gain went ───────────
def figure_breast_er_sensitivity() -> None:
    folds = load("breast_er_sensitivity.csv").sort_values("held_out_site")
    summary = load("breast_er_sensitivity.json")
    external = load("breast_er_external_validation.csv")
    fig = plt.figure(figsize=(WIDE, 2.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.75)

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(folds))[::-1] + 1.5
    for position, row in zip(y, folds.itertuples()):
        ax.plot([row.internal_gain, row.held_out_gain], [position, position], color=LIGHT, lw=2, zorder=1)
        ax.plot([row.held_out_gain_lower, row.held_out_gain_upper], [position - 0.12] * 2, color=ML, lw=1)
    ax.scatter(folds["internal_gain"], y, marker="D", s=16, color=ROBUST, edgecolor=INK, linewidth=0.4, zorder=3, label="inside the development sites")
    ax.scatter(folds["held_out_gain"], y - 0.12, marker="s", s=16, color=ML, edgecolor=INK, linewidth=0.4, zorder=3, label="held-out site")
    pooled = summary["held_out_gain_pooled"]
    ax.fill([pooled["ci_lower"], pooled["estimate"], pooled["ci_upper"], pooled["estimate"]], [0.5, 0.78, 0.5, 0.22], color=INK)
    ax.scatter([summary["internal_gain_mean"]], [0.5], marker="D", s=16, color=ROBUST, edgecolor=INK, linewidth=0.4, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks([*y, 0.5], [f"Site {row.held_out_site} ({row.held_out_n}, {row.held_out_events})" for row in folds.itertuples()] + ["Pooled"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0, len(folds) + 2)
    ax.set_xlabel("C gain over the clinical covariates")
    ax.text(-0.02, 1.0, "METABRIC site held out\n(patients, relapses)", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=6, handletextpad=0.2, borderaxespad=0.2)
    panel_label(ax, "a", x=-0.85, y=1.06)

    ax = fig.add_subplot(grid[0, 1])
    order = external.sort_values(["endpoint", "cohort"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(order))[::-1] + 2.5
    colours = [ML if endpoint == "rfs" else SUGGESTIVE for endpoint in order["endpoint"]]
    ax.errorbar(order["delta_c"], y, xerr=[order["delta_c"] - order["delta_lower"], order["delta_upper"] - order["delta_c"]], fmt="none", ecolor=colours, lw=1)
    ax.scatter(order["delta_c"], y, c=colours, marker="s", s=16, edgecolor=INK, linewidth=0.4, zorder=3)
    labels = [f"{row.cohort} ({'RFS' if row.endpoint == 'rfs' else 'DMFS'})" for row in order.itertuples()]
    by_endpoint = summary["external_by_endpoint"]
    for position, endpoint, colour in ((1.3, "rfs", ML), (0.5, "dmfs", SUGGESTIVE)):
        gain = by_endpoint[endpoint]["gain"]
        ax.fill([gain["ci_lower"], gain["estimate"], gain["ci_upper"], gain["estimate"]], [position, position + 0.25, position, position - 0.25], color=colour)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks([*y, 1.3, 0.5], labels + ["Pooled RFS", "Pooled DMFS"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0, len(order) + 3)
    ax.set_xlabel("C gain over the clinical covariates")
    panel_label(ax, "b", x=-0.62, y=1.06)
    save(fig, "figS2_positive_control_sensitivity")


# ── Supplementary Figure S3: evidence tiers against external replication ─────
def figure_tier_replication() -> None:
    summary = load("tier_replication.json")
    names = {"robust": "Robust", "suggestive": "Suggestive", "marginal only": "Marginal only",
             "nominal only": "Nominal p < 0.05 only", "no evidence": "No evidence"}
    colours = {"robust": ROBUST, "suggestive": SUGGESTIVE, "marginal only": MUTED, "nominal only": ML, "no evidence": "#9aa5b1"}
    titles = {"I": "Lung adenocarcinoma, survival", "IV": "Breast cancer, survival", "V": "ER-positive breast, recurrence"}
    cases = list(summary["cases"].items())
    rows = [group for group in names if any(result["groups"].get(group, {}).get("evaluable") for _, result in cases)]
    position = dict(zip(rows, np.arange(len(rows))[::-1]))
    fig, axes = plt.subplots(1, len(cases), figsize=(WIDE, 2.45), sharex=True, sharey=True)
    for ax, letter, (case, result) in zip(np.atleast_1d(axes), "abc", cases):
        for group, value in result["groups"].items():
            if not value["evaluable"]:
                continue
            y = position[group]
            low, high = (100 * bound for bound in value["rate_ci"])
            ax.plot([low, high], [y, y], color=colours[group], lw=1.3, solid_capstyle="butt")
            ax.scatter([100 * value["rate"]], [y], s=24, color=colours[group], edgecolor=INK, linewidth=0.4, zorder=3)
            ax.text(99, y + 0.22, f"{value['replicated']:,} of {value['evaluable']:,}", va="bottom", ha="right", fontsize=5.8, color=MUTED)
        ax.set_yticks(list(position.values()), [names[group] for group in position])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(rows) - 0.25)
        ax.grid(axis="x", color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
        patients = sum(cohort["n"] for cohort in result["cohorts"])
        ax.set_title(f"{titles.get(case, result['label'])} ({case})\n{len(result['cohorts'])} cohorts, {patients:,} patients", fontsize=7, loc="left", color=INK)
        ax.set_xlabel("Genes replicated (%)")
        panel_label(ax, letter, x=-0.04, y=1.2)
    fig.subplots_adjust(wspace=0.14)
    save(fig, "figS3_tier_replication")


def main(wanted: set[str]) -> None:
    refused = {}
    for name, draw in (("workflow", figure_workflow), ("markers", figure_markers), ("external", figure_external),
                       ("models", figure_models), ("breast", figure_breast), ("breast_er", figure_breast_er), ("simulation", figure_simulation),
                       ("breast_er_sensitivity", figure_breast_er_sensitivity), ("tiers", figure_tier_replication)):
        if wanted and name not in wanted:
            continue
        LOADED.clear()
        try:
            draw()
        except MixedResults as exc:
            refused[name] = exc.args[0]
    if refused:
        for name, problems in refused.items():
            print(f"not drawn: {name}", *(f"  - {problem}" for problem in problems), sep="\n", file=sys.stderr)
        raise SystemExit(f"{len(refused)} figure(s) not drawn from mixed or stale results: {', '.join(refused)}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))

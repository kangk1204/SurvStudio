"""Recompute and plot the marker-null audit from its per-replicate records.

Usage: python plot_marker_calibration.py --input marker_calibration.json --output figure
The four-scenario global-null audit is not evidence of general strong FWER control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"independent_markers": "Independent markers", "linear_marker_relation": "Linear relation to Z",
          "heteroscedastic_markers": "Heteroscedastic markers", "nonlinear_marker_relation": "Nonlinear relation to Z"}


def wilson(count: int, n: int) -> tuple[float, float]:
    # Closed form, independently of the statsmodels interval used by the simulator.
    z = 1.959963984540054
    p = count / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    rows = raw["replicates"]
    identifiers = [(row["scenario"], row["replicate"]) for row in rows]
    assert len(identifiers) == len(set(identifiers)), "Duplicated scenario/replicate identifiers"
    assert set(row["scenario"] for row in rows) == set(LABELS)
    source = []
    for scenario in LABELS:
        subset = [row for row in rows if row["scenario"] == scenario]
        assert len(subset) == raw["settings"]["replicates"]
        assert all(type(row["fwer_rejected"]) is bool for row in subset)
        assert all(0 <= row["events"] <= raw["settings"]["patients"] for row in subset)
        errors = sum(row["fwer_rejected"] for row in subset)
        lower, upper = wilson(errors, len(subset))
        saved = next(summary for summary in raw["summary"] if summary["scenario"] == scenario)["fwer_rejected"]
        assert errors == saved["count"]
        assert np.isclose(errors / len(subset), saved["rate"], atol=1e-15, rtol=0)
        assert np.allclose([lower, upper], saved["monte_carlo_wilson_95"], atol=1e-14, rtol=0)
        source.append({"scenario": scenario, "replicates": len(subset), "errors": errors, "rate_percent": 100 * errors / len(subset),
                       "lower_percent": 100 * lower, "upper_percent": 100 * upper})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "source_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "svg.fonttype": "none", "svg.hashsalt": "survstudio-marker-audit-20261002", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.3, 3.05))
    for index, row in enumerate(source):
        y = 3 - index
        rate = row["rate_percent"]
        ax.errorbar(rate, y, xerr=[[rate - row["lower_percent"]], [row["upper_percent"] - rate]],
                    fmt="o", capsize=3, markersize=5, linewidth=1.2, color="#0072B2")
        ax.text(10.4, y, f"{row['errors']}/{row['replicates']}   {rate:.1f}%", va="center", ha="left", fontsize=8)
    ax.axvline(5, color="#555555", linestyle="--", linewidth=0.9)
    ax.text(5.12, 3.45, "Nominal 5%", color="#555555", va="center", fontsize=8)
    ax.set_yticks([3, 2, 1, 0], list(LABELS.values()))
    ax.set_ylim(-0.5, 3.75)
    ax.set_xlim(0, 13.4)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Family-wise rejection rate (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Added-value permutation screen under four conditional nulls", loc="left", fontsize=9, pad=12)
    fig.text(0.02, 0.025, "180 patients, 30 markers; 499 permutations per replicate. Bars: 95% Monte Carlo Wilson intervals.", fontsize=7)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.80, bottom=0.23)
    fig.savefig(args.output / "marker_calibration.png", dpi=300, metadata={"Software": "SurvStudio publication audit"})
    fig.savefig(args.output / "marker_calibration.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.output / "marker_calibration.svg", metadata={"Date": None})
    svg = args.output / "marker_calibration.svg"
    # Matplotlib's optional SVG 1.1 DTD is unnecessary and refers to a remote URI.
    svg.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg.read_text(), count=1))
    plt.close(fig)
    (args.output / "scientific_check.json").write_text(json.dumps({"passed": True, "total_replicates": len(rows),
        "checks": ["unique scenario/replicate IDs", "expected counts and scenarios", "event count bounds", "rates recomputed from individual replicates", "Wilson intervals independently recomputed in closed form"],
        "interpretation": "Specified global-null scenarios only; calibration of general strong FWER and robust tiers is not established."}, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""The 95% interval of the left-out gain over the clinical covariates.

The gain is the mean over subsamples of the selected-marker model's C-index minus the clinical-only model's
C-index in the patients left out. Its interval is the corrected resampled t of Nadeau and Bengio (Machine
Learning 2003;52:239-281), which allows for the overlap of the subsamples.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.marker_evaluation as marker_evaluation
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers
from test_frontend_review import _run_page, example_dataset, marker_payloads  # noqa: F401 (pytest fixtures used by name)

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")


# t(0.975, df) for df = 2, 3, 5 and 24.
_T_2, _T_3, _T_5, _T_24 = 4.302652729749462, 3.1824463052837078, 2.5705818356363146, 2.063898561628024


def _hand_interval(values: list[float], sizes: list[tuple[int, int]], t_quantile: float) -> list[float]:
    """mean -/+ t * sqrt((1/J + rho) s^2), written out term by term."""
    j = len(values)
    mean = sum(values) / j
    variance = sum((value - mean) ** 2 for value in values) / (j - 1)
    rho = sum(left_out / subsample for subsample, left_out in sizes) / j
    half_width = t_quantile * math.sqrt((1.0 / j + rho) * variance)
    return [mean - half_width, mean + half_width]


# ── The formula ──────────────────────────────────────────────────


def test_the_interval_is_the_corrected_resampled_t_of_nadeau_and_bengio() -> None:
    gains = [0.012, -0.004, 0.021, 0.003, -0.009, 0.015]
    sizes = [(126, 74), (126, 74), (127, 73), (126, 74), (125, 75), (126, 74)]

    interval = marker_evaluation._corrected_resampled_t_interval(gains, [left_out / subsample for subsample, left_out in sizes])

    assert interval == pytest.approx(_hand_interval(gains, sizes, _T_5), abs=1e-12)
    # sqrt(1 + J rho), about 2.1 times as wide as a t interval that treats the overlapping subsamples as independent.
    rho = sum(left_out / subsample for subsample, left_out in sizes) / len(sizes)
    naive_half_width = _T_5 * float(np.std(gains, ddof=1)) / math.sqrt(len(gains))
    assert (interval[1] - interval[0]) / (2 * naive_half_width) == pytest.approx(math.sqrt(1 + len(gains) * rho))


def test_the_interval_needs_two_values_a_share_for_each_and_a_finite_variance() -> None:
    interval = marker_evaluation._corrected_resampled_t_interval
    assert interval([0.01], [0.6]) is None
    assert interval([], []) is None
    assert interval([0.01, 0.02], [0.6]) is None
    with np.errstate(all="ignore"):
        assert interval([0.01, float("inf")], [0.6, 0.6]) is None
    # Identical values have no spread: the interval is the value itself.
    assert interval([0.0, 0.0, 0.0], [0.6, 0.6, 0.6]) == [0.0, 0.0]


def test_the_optimism_summary_gives_the_gain_its_sd_and_interval_over_the_paired_replicates() -> None:
    # Four scored replicates; the third has no clinical-only C, so the gain pairs the other three.
    c_in = [0.74, 0.71, 0.73, 0.70]
    c_out = [0.66, 0.64, 0.69, 0.65]
    clinical_c_out = [0.65, 0.645, float("nan"), 0.62]
    sizes = [(126, 74), (125, 75), (126, 74), (127, 73)]

    summary = marker_evaluation._optimism_summary(c_in, c_out, clinical_c_out, [], [], sizes)

    gains = [0.66 - 0.65, 0.64 - 0.645, 0.65 - 0.62]
    paired_sizes = [sizes[0], sizes[1], sizes[3]]
    assert summary["n_clinical_replicates"] == 3
    assert summary["signature_gain_left_out"] == pytest.approx(sum(gains) / 3, abs=1e-15)
    assert summary["signature_gain_left_out_sd"] == pytest.approx(math.sqrt(sum((gain - sum(gains) / 3) ** 2 for gain in gains) / 2))
    assert summary["signature_gain_left_out_ci"] == pytest.approx(_hand_interval(gains, paired_sizes, _T_2), abs=1e-12)
    # The left-out C-index gets the same interval over all four replicates.
    assert summary["n_signature_replicates"] == 4
    assert summary["signature_c_left_out_ci"] == pytest.approx(_hand_interval(c_out, sizes, _T_3), abs=1e-12)


def test_the_optimism_summary_gives_no_interval_without_sizes_or_with_one_paired_replicate() -> None:
    c_in, c_out, clinical_c_out = [0.74, 0.71], [0.66, 0.64], [0.65, 0.645]
    sizes = [(126, 74), (125, 75)]

    without_sizes = marker_evaluation._optimism_summary(c_in, c_out, clinical_c_out, [], [])
    one = marker_evaluation._optimism_summary(c_in[:1], c_out[:1], clinical_c_out[:1], [], [], sizes[:1])
    no_clinical = marker_evaluation._optimism_summary(c_in, c_out, [], [], [], sizes)

    assert without_sizes["signature_gain_left_out_ci"] is None and without_sizes["signature_c_left_out_ci"] is None
    assert without_sizes["signature_gain_left_out_sd"] is not None
    assert one["n_clinical_replicates"] == 1 and one["signature_gain_left_out"] == pytest.approx(0.01)
    assert one["signature_gain_left_out_ci"] is None and one["signature_gain_left_out_sd"] is None
    assert no_clinical["signature_gain_left_out"] is None and no_clinical["signature_gain_left_out_ci"] is None
    assert no_clinical["signature_c_left_out_ci"] is not None


# ── The engine ───────────────────────────────────────────────────


def _cohort(seed: int, n: int = 220) -> pd.DataFrame:
    """Age and one of six markers are prognostic."""
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    markers = rng.normal(size=(n, 6))
    event_time = rng.exponential(np.exp(-(0.8 * age + 0.5 * markers[:, 0])))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame({"os_time": np.minimum(event_time, censor_time), "os_event": (event_time <= censor_time).astype(int), "age": age})
    for index in range(markers.shape[1]):
        frame[f"m{index}"] = markers[:, index]
    return frame


_SETTINGS = MarkerSettings(n_permutations=19, n_resamples=25, random_seed=4)
_MARKERS = [f"m{index}" for index in range(6)]


def test_evaluate_markers_returns_the_interval_of_the_left_out_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    original = marker_evaluation._optimism_summary

    def spy(c_in, c_out, clinical_c_out, beta_in, beta_out, *sizes):
        captured.update(c_in=list(c_in), c_out=list(c_out), clinical=list(clinical_c_out), sizes=list(*sizes))
        return original(c_in, c_out, clinical_c_out, beta_in, beta_out, *sizes)

    monkeypatch.setattr(marker_evaluation, "_optimism_summary", spy)
    frame = _cohort(1)

    result = evaluate_markers(frame, time_column="os_time", event_column="os_event", marker_columns=_MARKERS, clinical_columns=["age"], settings=_SETTINGS)

    signature = result["signature"]
    assert "m0" in signature["markers"]
    low, high = signature["signature_gain_left_out_ci"]
    assert low < signature["signature_gain_left_out"] < high
    assert signature["signature_gain_left_out_sd"] > 0
    assert signature["n_clinical_replicates"] == 25
    c_low, c_high = signature["signature_c_left_out_ci"]
    assert c_low < signature["signature_c_left_out"] < c_high
    # Each scored replicate records its subsample and the patients left out of it.
    assert len(captured["sizes"]) == len(captured["c_out"]) == 25
    assert all(subsample + left_out == len(frame) for subsample, left_out in captured["sizes"])
    # The interval is the formula applied to the replicates' paired gains.
    gains = [outside - clinical for outside, clinical in zip(captured["c_out"], captured["clinical"])]
    assert [low, high] == pytest.approx(_hand_interval(gains, captured["sizes"], _T_24), abs=1e-12)
    assert signature["signature_gain_left_out"] == pytest.approx(signature["signature_c_left_out"] - signature["clinical_c_left_out"])
    json.dumps(result, allow_nan=False)


def test_a_marker_evaluation_without_clinical_covariates_has_no_gain_interval() -> None:
    frame = _cohort(2)

    signature = evaluate_markers(frame, time_column="os_time", event_column="os_event", marker_columns=_MARKERS, settings=_SETTINGS)["signature"]

    assert signature["n_clinical_replicates"] == 0
    assert signature["signature_gain_left_out"] is None and signature["signature_gain_left_out_ci"] is None
    assert signature["signature_gain_left_out_sd"] is None


# ── The Markers tab ──────────────────────────────────────────────


_SUMMARIZE = r"""
  const analysis = {
    primary_lens: "added_value",
    tier_counts: { robust: 1, suggestive: 0 },
    cohort: { n: 300, events: 120, n_markers_evaluated: 12 },
    settings: { alpha: 0.05, robust_frequency: 0.5, robust_direction: 0.9 },
    resampling: { n_valid: 40, fraction: 0.632, stability_assessed: true },
    null: { n_permutations: 199, lens2_null: "smith" },
  };
  const selected = { markers: ["GENE_A"], clinical_only: false, apparent_c: 0.71, optimism_corrected_c: 0.69, signature_optimism: 0.01,
    signature_c_left_out: 0.68, clinical_c_left_out: 0.66, n_clinical_replicates: 40 };
  const summarize = (changes) => {
    page.context.__payload = { analysis: { ...analysis, signature: { ...selected, ...changes } } };
    const summary = page.run("markerSummary(__payload)");
    return { strengths: summary.strengths.join(" | "), cautions: summary.cautions.join(" | ") };
  };
"""


@_needs_node
def test_the_verdict_follows_the_interval_of_the_gain(tmp_path: Path) -> None:
    result = _run_page(tmp_path, _SUMMARIZE + r"""
      return {
        little: summarize({ signature_gain_left_out: 0.002, signature_gain_left_out_ci: [-0.011, 0.015] }),
        smallButReal: summarize({ signature_gain_left_out: 0.01, signature_gain_left_out_ci: [0.004, 0.016] }),
        adds: summarize({ signature_gain_left_out: 0.021, signature_gain_left_out_ci: [0.006, 0.036] }),
        uncertain: summarize({ signature_gain_left_out: 0.012, signature_gain_left_out_ci: [-0.009, 0.033] }),
        procedure: summarize({ markers: [], apparent_c: null, optimism_corrected_c: null,
          signature_gain_left_out: 0.012, signature_gain_left_out_ci: [-0.009, 0.033] }),
      };
    """)

    comparison = "In the patients left out of each of 40 subsamples, the selected-marker model reached C 0.68 against 0.66 for the clinical covariates alone"
    # The whole interval below 0.02: little discrimination, a caution, even when the interval lies above 0.
    assert f"{comparison}, a gain of +0.002 (95% CI -0.011 to 0.015). The selected markers add little discrimination beyond the clinical covariates." in result["little"]["cautions"]
    assert "a gain of +0.010 (95% CI 0.004 to 0.016). The selected markers add little discrimination" in result["smallButReal"]["cautions"]
    assert "a gain of" not in result["little"]["strengths"] + result["smallButReal"]["strengths"]
    # The whole interval above 0 (and reaching 0.02): the markers add discrimination, a strength.
    assert f"{comparison}, a gain of +0.021 (95% CI 0.006 to 0.036). The selected markers add discrimination beyond the clinical covariates." in result["adds"]["strengths"]
    assert "a gain of" not in result["adds"]["cautions"]
    # Neither: an uncertain gain, a caution that gives the interval.
    assert (
        f"{comparison}, a gain of +0.012 (95% CI -0.009 to 0.033). The gain is uncertain: its interval includes both no gain and a gain of 0.02 or more."
    ) in result["uncertain"]["cautions"]
    assert "add little" not in result["uncertain"]["cautions"] and "a gain of" not in result["uncertain"]["strengths"]
    # Without a full-cohort model the gain and its interval belong to the whole procedure, with no verdict on selected markers.
    assert "the whole selection procedure reached C 0.68 against 0.66 for the clinical covariates alone, a gain of +0.012 (95% CI -0.009 to 0.033)." in result["procedure"]["cautions"]
    assert "selected markers" not in result["procedure"]["cautions"] and "uncertain" not in result["procedure"]["cautions"]


@_needs_node
def test_a_result_without_the_interval_keeps_the_verdict_on_the_mean_gain(tmp_path: Path) -> None:
    """Results saved before the interval existed, or with a single paired subsample, are judged by the mean gain against 0.02."""
    result = _run_page(tmp_path, _SUMMARIZE + r"""
      return {
        small: summarize({ signature_gain_left_out: 0.004 }),
        large: summarize({ signature_gain_left_out: 0.035 }),
        single: summarize({ signature_gain_left_out: 0.004, signature_gain_left_out_ci: null, n_clinical_replicates: 1 }),
        partial: summarize({ signature_gain_left_out: 0.035, signature_gain_left_out_ci: [null, null] }),
      };
    """)

    comparison = "the selected-marker model reached C 0.68 against 0.66 for the clinical covariates alone"
    assert f"{comparison} (+0.004). The selected markers add little discrimination beyond the clinical covariates." in result["small"]["cautions"]
    assert f"{comparison} (+0.035)." in result["large"]["strengths"] and "add discrimination" not in result["large"]["strengths"]
    assert "In the patients left out of the one subsample that could be scored" in result["single"]["cautions"]
    assert "(+0.004). The selected markers add little" in result["single"]["cautions"]
    assert f"{comparison} (+0.035)." in result["partial"]["strengths"]
    assert "95% CI" not in " ".join(value for case in result.values() for value in case.values())


@_needs_node
def test_the_markers_tab_shows_the_interval_the_server_computed(tmp_path: Path, marker_payloads: dict) -> None:
    signature = marker_payloads["added"]["analysis"]["signature"]
    low, high = signature["signature_gain_left_out_ci"]
    gain = signature["signature_gain_left_out"]

    result = _run_page(tmp_path, r"""
      page.context.__payload = fixtures.markers.added;
      const summary = page.run("markerSummary(__payload)");
      return [...summary.strengths, ...summary.cautions].join(" | ");
    """, markers=marker_payloads)

    assert f"a gain of {gain:+.3f} (95% CI {low:.3f} to {high:.3f})." in result

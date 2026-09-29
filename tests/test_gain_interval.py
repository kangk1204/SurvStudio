"""The 95% interval of the left-out gain over the clinical covariates.

The gain is the mean over subsamples of the selected-marker model's C-index minus the clinical-only model's
C-index in the patients left out. Its interval is the corrected resampled t of Nadeau and Bengio (Machine
Learning 2003;52:239-281), which allows for the overlap of the subsamples.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import survival_toolkit.marker_evaluation as marker_evaluation
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers


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

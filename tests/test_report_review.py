"""Regression tests for findings of the September 2026 code review: report text that has to follow what a run
actually did, and the package entry points."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_toolkit.duplicates import MIN_GAP, MIN_R
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers
from survival_toolkit.reporting import checklist_markdown, remark_checklist, tripod_ai_checklist

_MARKERS = [f"m{index}" for index in range(8)]
_REQUEST = {"time_column": "os_time", "event_column": "os_event", "event_positive_value": 1, "marker_columns": _MARKERS}


def _cohort(seed: int = 5, n: int = 240, marker_effect: float = 0.9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    grade = rng.integers(0, 2, size=n)
    markers = rng.normal(size=(n, len(_MARKERS)))
    linear = 0.7 * age + 0.4 * grade + marker_effect * markers[:, 0] + 0.6 * marker_effect * markers[:, 1]
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    return pd.DataFrame(
        {
            "os_time": np.minimum(event_time, censor_time),
            "os_event": (event_time <= censor_time).astype(int),
            "age": age,
            "grade": np.where(grade == 1, "high", "low"),
            **{name: markers[:, index] for index, name in enumerate(_MARKERS)},
        }
    )


def _evaluate(frame: pd.DataFrame, **settings) -> dict:
    return evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=_MARKERS,
        clinical_columns=["age", "grade"],
        categorical_clinical=["grade"],
        settings=MarkerSettings(shortlist_size=5, random_seed=11, **settings),
    )


def _items(report: dict) -> dict:
    return {entry["item"]: entry for entry in report["items"]}


# ── REMARK ───────────────────────────────────────────────────────


def test_remark_text_without_subsamples_says_stability_and_optimism_were_not_assessed() -> None:
    result = _evaluate(_cohort(), n_permutations=99, n_resamples=0)
    assert result["resampling"]["n_valid"] == 0

    report = remark_checklist(result, request=_REQUEST, dataset={"filename": "c.csv", "n_rows": 240})

    items = _items(report)
    methods, results = report["methods"], report["results"]
    assert "was not repeated on subsamples, so the stability of the selection was not assessed" in methods
    assert "Without subsamples the stability rule could not be applied" in methods
    assert "repeated on 0" not in methods and "0 event-stratified" not in methods
    assert "could not be corrected for optimism because no subsample was available" in methods
    assert "Stability over subsamples was not assessed" in results and "not corrected for optimism" in results
    assert items["10"]["status"] == "partly"
    assert items["18"]["text"].startswith("No internal validation was done: no subsamples were drawn")


def test_remark_text_without_permutations_claims_no_family_wise_control() -> None:
    result = _evaluate(_cohort(), n_permutations=0, n_resamples=6)

    report = remark_checklist(result, request=_REQUEST)

    methods = report["methods"]
    assert "No permutations were run" in methods and "from 0 permutations" not in methods
    assert "no marker could be called robust, suggestive or marginal only" in methods
    assert "A marker was robust when" not in methods
    assert "No permutations were run" in report["results"]
    assert _items(report)["10"]["status"] == "partly"


def _hand_built_result(**changes) -> dict:
    result = {
        "primary_lens": "added_value",
        "marker_table": [],
        "tier_counts": {"robust": 0, "suggestive": 1, "marginal only": 1, "not supported": 1},
        "cohort": {"n": 200, "events": 90, "n_markers_evaluated": 3, "clinical_columns": ["age"], "strata_columns": [], "dropped_markers": []},
        "settings": {"alpha": 0.05, "fdr_level": 0.1, "shortlist_size": 50, "max_signature_markers": 10},
        "null": {"n_permutations": 1000, "lens2_null": "smith"},
        "resampling": {"n_valid": 200, "n_failed": 0, "fraction": 0.632},
        "signature": {"markers": ["m1"], "apparent_c": 0.74, "optimism_corrected_c": 0.71},
        "duplicates": {},
    }
    for key, value in changes.items():
        result[key] = {**result[key], **value} if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def test_remark_text_when_every_subsample_failed_or_the_optimism_is_missing() -> None:
    failed = remark_checklist(_hand_built_result(resampling={"n_valid": 0, "n_failed": 5}), request=_REQUEST)
    assert "run on 5 event-stratified subsamples of 63.2% of the patients, but every one failed" in failed["methods"]
    assert _items(failed)["18"]["text"].startswith("No internal validation was done: all 5 subsamples failed")

    partial = remark_checklist(_hand_built_result(resampling={"n_valid": 180, "n_failed": 20}), request=_REQUEST)
    assert "repeated on 180 event-stratified subsamples of 63.2% of the patients (20 more failed and were left out)" in partial["methods"]

    uncorrected = remark_checklist(_hand_built_result(signature={"optimism_corrected_c": None}), request=_REQUEST)
    assert "could not be corrected for optimism because no subsample gave a model" in uncorrected["methods"]
    assert "(not corrected for optimism)" in uncorrected["results"]
    assert "apparent C-index (0.740) could not be corrected for optimism" in _items(uncorrected)["18"]["text"]
    assert _items(uncorrected)["10"]["status"] == "partly"


def test_remark_text_names_the_permutation_scheme_by_its_setting() -> None:
    for scheme in ("smith", "freedman_lane"):
        methods = remark_checklist(_hand_built_result(null={"lens2_null": scheme}), request=_REQUEST)["methods"]
        assert "the residuals of each marker after regression on the clinical covariates were permuted (Smith method; Winkler et al. 2014)" in methods
    raw = remark_checklist(_hand_built_result(null={"lens2_null": "raw"}), request=_REQUEST)["methods"]
    assert "the marker values themselves were permuted" in raw and "Smith" not in raw


def test_remark_text_for_a_model_without_markers_calls_it_the_clinical_model() -> None:
    frame = _cohort(seed=3, marker_effect=0.0)
    result = _evaluate(frame, n_permutations=99, n_resamples=10)
    assert result["signature"]["markers"] == [] and result["signature"]["apparent_c"] is not None

    report = remark_checklist(result, request=_REQUEST)

    assert "(none)" not in report["results"] and "selected-marker model" not in report["results"]
    assert "No marker was selected, so the final model held the clinical covariates only; its apparent C-index was" in report["results"]
    assert "No marker was selected, so the final Cox model held the clinical covariates only" in report["methods"]
    assert "the clinical model's C-index was corrected from" in _items(report)["18"]["text"]


def test_remark_patient_flow_separates_patients_without_marker_values_from_exclusions() -> None:
    result = _hand_built_result(cohort={"n": 390, "events": 120})
    matrix = {"filename": "expr.tsv", "n_markers": 3, "n_matched": 400, "id_column": "id", "fingerprint": "f"}

    with_matrix = _items(remark_checklist(result, request=_REQUEST, dataset={"filename": "c.csv", "n_rows": 500, "marker_matrix": matrix}))
    plain = _items(remark_checklist(result, request=_REQUEST, dataset={"filename": "c.csv", "n_rows": 391}))

    assert with_matrix["12"]["text"] == (
        "390 patients with 120 events were analysed; 100 of 500 rows had no values in the marker matrix and 10 of the 400 "
        "matched patients were excluded for a missing or invalid outcome, clinical covariate or stratum."
    )
    assert plain["12"]["text"].endswith("; 1 of 391 rows was excluded for a missing or invalid outcome, clinical covariate or stratum.")


def test_remark_left_out_comparison_gives_the_number_of_subsamples() -> None:
    signature = {"markers": ["m1"], "apparent_c": 0.74, "optimism_corrected_c": 0.71, "signature_c_left_out": 0.70, "clinical_c_left_out": 0.66}

    unpaired = remark_checklist(_hand_built_result(signature={**signature, "n_signature_replicates": 40}), request=_REQUEST)
    paired = remark_checklist(
        _hand_built_result(signature={**signature, "n_signature_replicates": 40, "n_paired_replicates": 18, "delta_c_left_out": 0.05}),
        request=_REQUEST,
    )

    assert "In the patients left out of each of 40 subsamples, it reached a mean C-index of 0.700 against 0.660" in unpaired["results"]
    assert "(mean difference +0.040)" in unpaired["results"]
    assert "each of 18 subsamples" in paired["results"] and "(mean difference +0.050)" in paired["results"]


def test_remark_text_uses_singular_nouns_for_one() -> None:
    result = _hand_built_result(tier_counts={"robust": 1, "suggestive": 0, "marginal only": 1}, cohort={"n_markers_evaluated": 1})

    report = remark_checklist(result, request=_REQUEST)
    items = _items(report)

    assert report["results"].startswith("Of 1 marker, 1 was robust, 0 suggestive and 1 marginal only.")
    assert "1 candidate marker;" in items["8"]["text"] and "for 1 candidate marker (" in items["9"]["text"]
    assert items["14"]["text"].startswith("1 marker was marginal only: associated with survival on its own")
    assert "1 markers" not in " ".join(entry["text"] for entry in report["items"])


def test_duplicate_screen_sentence_follows_the_checks_that_ran() -> None:
    near = {"checked": True, "markers_used": 5000, "n_pairs": 0, "n_identical": 0}
    unknown = remark_checklist(_hand_built_result(duplicates=near, cohort={"n_markers_evaluated": 5000}), request=_REQUEST)
    binary = remark_checklist(
        _hand_built_result(duplicates={**near, "identical_checked": False}, cohort={"n_markers_evaluated": 5000}), request=_REQUEST
    )
    small = remark_checklist(_hand_built_result(duplicates={"checked": False, "n_pairs": 0, "n_identical": 0}), request=_REQUEST)

    assert f"correlated at least {MIN_R:g} and stood {MIN_GAP:g} above" in unknown["methods"]
    assert "also flagged when the markers took many distinct values" in unknown["methods"]
    assert "identical values" not in binary["methods"] and "flagged no patients" in binary["results"]
    # Three markers: neither check ran, so nothing is claimed.
    assert "repeated samples" not in small["methods"] and "repeated samples" not in small["results"]


def test_checklist_markdown_escapes_pipes_after_backslashes_and_raw_html() -> None:
    report = {
        "guideline": "REMARK",
        "reference": "r",
        "software": "s",
        "methods": "Covariates: <script>alert(1)</script> & stage",
        "results": "r",
        "items": [{"item": "8", "section": "Study design", "topic": "Candidate variables", "status": "reported", "text": "stage\\|grade | <b>x</b>"}],
    }

    text = checklist_markdown(report)

    assert "Covariates: &lt;script&gt;alert(1)&lt;/script&gt; &amp; stage" in text
    # GFM drops one backslash before each pipe in a table cell, leaving stage\\|grade (a literal backslash and pipe).
    assert text.splitlines()[-1] == "| 8 | Study design | Candidate variables | Filled in by SurvStudio | stage\\\\\\|grade \\| &lt;b&gt;x&lt;/b&gt; |"


# ── TRIPOD+AI ────────────────────────────────────────────────────


def _dl_mixed() -> dict:
    return {
        "family": "dl",
        "evaluation_mode": "mixed_holdout_apparent",
        "n_patients": 600,
        "n_events": 250,
        "evaluation_split_fingerprint": "fp1",
        "comparison_table": [
            {"model": "DeepSurv", "c_index": 0.66, "evaluation_mode": "holdout", "comparable_for_ranking": True, "rank": 1},
            {"model": "DeepHit", "c_index": 0.91, "evaluation_mode": "apparent", "comparable_for_ranking": False, "rank": None},
        ],
        "errors": [],
        "request_config": {"time_column": "t", "event_column": "e", "event_positive_value": 1, "features": ["a", "b"], "categorical_features": [],
                           "epochs": 100, "learning_rate": 0.001, "hidden_layers": [64], "dropout": 0.1, "batch_size": 64},
    }


def _ml_incomplete() -> dict:
    return {
        "family": "ml",
        "evaluation_mode": "repeated_cv_incomplete",
        "cv_folds": 5,
        "cv_repeats": 3,
        "n_patients": 600,
        "n_events": 250,
        "evaluation_split_fingerprint": "fp2",
        "comparison_table": [
            {"model": "Random Survival Forest", "c_index": 0.70, "n_evaluations": 15, "n_failures": 0, "evaluation_mode": "repeated_cv"},
            {"model": "Cox PH", "c_index": 0.69, "n_evaluations": 15, "n_failures": 0, "evaluation_mode": "repeated_cv"},
            {"model": "LASSO-Cox", "c_index": None, "n_evaluations": 14, "n_failures": 1, "evaluation_mode": "repeated_cv_incomplete"},
            {"model": "Gradient Boosted Survival", "c_index": None, "n_evaluations": 0, "n_failures": 15, "evaluation_mode": "repeated_cv_incomplete"},
        ],
        "errors": [{"model": "Gradient Boosted Survival", "repeat": 1, "fold": fold, "error": "x"} for fold in range(15)]
        + [{"model": "LASSO-Cox", "repeat": 2, "fold": 3, "error": "y"}],
        "excluded_models": ["Gradient Boosted Survival", "LASSO-Cox"],
        "request_config": {"time_column": "t", "event_column": "e", "event_positive_value": 1, "features": ["a"], "categorical_features": [],
                           "n_estimators": 100, "max_depth": None, "learning_rate": 0.1},
    }


def test_tripod_describes_an_apparent_fallback_per_model_and_does_not_rank_it() -> None:
    report = tripod_ai_checklist([_dl_mixed()])
    items = _items(report)

    assert items["16"]["text"] == (
        "Performance was estimated by a holdout set stratified by event status for DeepSurv; DeepHit fell back to apparent "
        "performance on the patients used for fitting, which is optimistic, and was not ranked."
    )
    assert report["results"].startswith("The highest C-index was 0.660 (DeepSurv).")
    assert "Not ranked: DeepHit (apparent evaluation" in report["results"] and "0.910" not in report["results"]
    assert "models that fell back to apparent evaluation were not ranked" in report["methods"]
    assert "Choosing the best of" not in items["26"]["text"]


def test_tripod_counts_fitted_models_and_lost_folds_of_an_incomplete_cross_validation() -> None:
    report = tripod_ai_checklist([_ml_incomplete()])
    items = _items(report)

    assert "folds that failed or fell back to apparent evaluation were left out" in items["16"]["text"]
    assert items["21"]["text"] == (
        "3 models were fitted (LASSO-Cox failed in 1 of 15 folds); fitting failed for Gradient Boosted Survival (all 15 folds)."
    )
    assert report["results"] == (
        "The highest C-index was 0.700 (Random Survival Forest). Across the 2 ranked models the C-index ranged from 0.690 to "
        "0.700. Not ranked: LASSO-Cox (failed in 1 of 15 folds); Gradient Boosted Survival (failed in 15 of 15 folds)."
    )
    assert "Choosing the best of 2 models" in items["26"]["text"]


def test_tripod_describes_each_family_when_the_comparisons_differ() -> None:
    ml = {**_ml_incomplete(), "n_locked_test_patients": 120, "n_development_patients": 480, "n_locked_test_events": 50}
    dl = {**_dl_mixed(), "n_patients": 580, "n_events": 240}

    report = tripod_ai_checklist([ml, dl])
    items = _items(report)

    assert items["6"]["text"].startswith(
        "For the classical machine-learning models, 600 patients with a usable outcome were analysed; for the deep-learning "
        "models, 580 patients with a usable outcome were analysed."
    )
    assert items["9"]["text"].startswith(
        "For the classical machine-learning models, 1 predictor: a; categorical: none; for the deep-learning models, 2 predictors: a, b"
    )
    assert "600 patients and 250 events for 1 candidate predictor; for the deep-learning models, 580 patients and 240 events for 2 candidate predictors" in items["10"]["text"]
    assert "locked test set 120 patients; for the deep-learning models, 580 patients and 240 events" in items["20"]["text"]
    assert "(with a locked test set for the classical machine-learning models)" in items["26"]["text"]
    assert "compared with SurvStudio" in report["methods"] and "the deep-learning models, 580 patients (240 events)" in report["methods"]


def test_tripod_preprocessing_names_the_standardisation_of_each_model() -> None:
    ml = _ml_incomplete()
    methods = tripod_ai_checklist([ml, _dl_mixed()])["methods"]
    assert (
        "for Cox PH and LASSO-Cox every encoded column was standardised with the training mean and standard deviation; for neural "
        "networks numeric predictors were standardised with the training mean and standard deviation"
    ) in methods

    trees_only = {**ml, "comparison_table": ml["comparison_table"][:1], "errors": []}
    report = tripod_ai_checklist([trees_only])
    assert "standardised" not in report["methods"] and "standardised" not in _items(report)["7"]["text"]
    assert _items(report)["21"]["text"] == "1 model was fitted."
    assert "Across" not in report["results"] and "1 predictor:" in _items(report)["9"]["text"]


# ── Entry points and packaging ───────────────────────────────────


def test_cli_rejects_ports_outside_the_valid_range() -> None:
    from survival_toolkit.__main__ import _build_parser, _port

    assert _port("8001") == 8001
    for value in ("0", "70000", "-1", "http"):
        with pytest.raises(argparse.ArgumentTypeError):
            _port(value)
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["serve", "--port", "70000"])


@pytest.mark.parametrize(
    ("host", "warns"),
    [("127.0.0.1", False), ("127.0.0.2", False), ("0:0:0:0:0:0:0:1", False), ("[::1]", False), ("LOCALHOST", False),
     ("::", True), ("0.0.0.0", True), ("192.168.1.20", True), ("lab.example", True)],
)
def test_cli_warns_only_when_the_server_is_reachable_beyond_loopback(
    host: str, warns: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from survival_toolkit.__main__ import _warn_if_reachable_from_network

    monkeypatch.delenv("SURVSTUDIO_CONTAINER", raising=False)
    _warn_if_reachable_from_network(host)
    assert ("WARNING" in capsys.readouterr().err) is warns


def test_pyproject_declares_the_web_stack_the_app_imports() -> None:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {entry.split(">=")[0].split("[")[0]: entry for entry in pyproject["project"]["dependencies"]}

    assert dependencies["pydantic"] == "pydantic>=2.0"
    assert "starlette" in dependencies

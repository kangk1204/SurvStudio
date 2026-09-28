from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from survival_toolkit import __version__
from survival_toolkit.marker_evaluation import MarkerSettings, evaluate_markers
from survival_toolkit.reporting import (
    STATUS_LABELS,
    checklist_markdown,
    checklist_rows,
    remark_checklist,
    tripod_ai_checklist,
)

_SETTINGS = MarkerSettings(n_permutations=99, n_resamples=20, shortlist_size=5, random_seed=11)
_MARKERS = [f"m{index}" for index in range(8)]


def _cohort(seed: int = 5, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    grade = rng.integers(0, 2, size=n)
    markers = rng.normal(size=(n, len(_MARKERS)))
    linear = 0.7 * age + 0.4 * grade + 0.7 * markers[:, 0]
    event_time = rng.exponential(np.exp(-linear))
    censor_time = rng.exponential(1.5, size=n)
    frame = pd.DataFrame(
        {
            "os_time": np.minimum(event_time, censor_time),
            "os_event": (event_time <= censor_time).astype(int),
            "age": age,
            "grade": np.where(grade == 1, "high", "low"),
            **{name: markers[:, index] for index, name in enumerate(_MARKERS)},
            "constant": 1.0,
        }
    )
    frame.loc[:4, "age"] = np.nan
    return frame


def _request(markers: list[str]) -> dict:
    return {"time_column": "os_time", "event_column": "os_event", "event_positive_value": 1, "marker_columns": markers}


def test_remark_checklist_fills_in_what_the_run_knows() -> None:
    frame = _cohort()
    markers = [*_MARKERS, "constant"]
    result = evaluate_markers(
        frame,
        time_column="os_time",
        event_column="os_event",
        marker_columns=markers,
        clinical_columns=["age", "grade"],
        categorical_clinical=["grade"],
        settings=_SETTINGS,
    )

    report = remark_checklist(result, request=_request(markers), dataset={"filename": "cohort.csv", "n_rows": 240, "dataset_hash": "abc"})

    items = {entry["item"]: entry for entry in report["items"]}
    assert list(items) == [str(number) for number in range(1, 21)]
    assert {entry["status"] for entry in report["items"]} <= set(STATUS_LABELS)
    assert items["3"]["status"] == items["4"]["status"] == items["19"]["status"] == "author"
    assert items["10"]["text"] == report["methods"] and items["10"]["status"] == "reported"
    assert "(Smith method; Winkler et al. 2014)" in report["methods"] and "99 permutations" in report["methods"]
    assert "Freedman" not in report["methods"] and "fixed before the analysis" not in report["methods"]
    assert "subsamples of 63.2% of the patients" in report["methods"]
    assert "Benjamini-Hochberg q-value was at most 0.05" in report["methods"]
    assert f"SurvStudio {__version__}" in report["methods"]
    assert "constant" in items["8"]["text"]
    assert "cohort.csv" in items["2"]["text"] and "fingerprint abc" in items["2"]["text"]
    assert "5 of 240 rows were excluded" in items["12"]["text"]
    # Exact fits cover the shortlist, the 5 strongest markers on the added-value lens and every supported one, which
    # is smaller than the panel here; every one of them had an estimate.
    table = result["marker_table"]
    strongest = sorted(table, key=lambda row: -(row["added_value"]["chi2"] or 0.0))[:5]
    shortlist = {row["marker"] for row in strongest} | {row["marker"] for row in table if row["tier"] != "not supported"}
    assert {row["marker"] for row in table if row.get("exact")} == shortlist and len(shortlist) < len(_MARKERS)
    assert {row["marker"] for row in table if (row.get("exact") or {}).get("adjusted")} == shortlist
    assert items["17"]["status"] == "reported"
    assert f"given for {len(shortlist)} markers (the 5 strongest and every supported one), whether or not" in items["17"]["text"]
    assert "Unadjusted HR" in items["15"]["text"]
    robust = result["tier_counts"]["robust"]
    assert report["results"].startswith(f"Of {len(_MARKERS)} markers, {robust} {'was' if robust == 1 else 'were'} robust")
    # The selected-marker model is set against the clinical covariates alone in the patients left out, by the mean
    # paired difference evaluate_markers reports.
    signature = result["signature"]
    assert signature["signature_gain_left_out"] is not None
    assert (
        f"against {signature['clinical_c_left_out']:.3f} for the clinical covariates alone "
        f"(mean difference {signature['signature_gain_left_out']:+.3f})"
    ) in report["results"]
    assert re.search(r"In the patients left out of each of \d+ subsamples, it reached a mean C-index", report["results"])
    assert "the selected-marker model reached a mean C-index" in items["18"]["text"]
    assert "more than 90% of patients at one value were excluded" in report["methods"]


def test_remark_checklist_without_clinical_covariates_asks_for_adjusted_effects() -> None:
    frame = _cohort().drop(columns=["constant"])
    result = evaluate_markers(frame, time_column="os_time", event_column="os_event", marker_columns=_MARKERS, settings=_SETTINGS)

    report = remark_checklist(result, request=_request(_MARKERS))

    items = {entry["item"]: entry for entry in report["items"]}
    assert "unadjusted Cox score test" in report["methods"]
    assert "Smith method" not in report["methods"] and "marginal only" not in report["methods"]
    assert items["16"]["status"] == "partly" and items["17"]["status"] == "author"
    assert items["2"]["text"].startswith("The analysed table")
    assert "excluded" not in items["12"]["text"]


def test_checklist_markdown_has_one_row_per_item_and_escapes_table_characters() -> None:
    report = {
        "guideline": "REMARK",
        "reference": "Ref 2005",
        "software": "SurvStudio 9",
        "methods": "Methods text.",
        "results": "Results text.",
        "items": [{"item": "1", "section": "Introduction", "topic": "a | b", "status": "author", "text": "line\nbreak"}],
    }

    text = checklist_markdown(report)

    assert text.startswith("# REMARK checklist\n")
    assert "Generated with SurvStudio 9" in text and "## Methods\n\nMethods text." in text
    assert "| 1 | Introduction | a \\| b | Authors to complete | line break |" in text
    assert checklist_rows(report)[0]["Status"] == "Authors to complete"


def _ml_comparison(**changes) -> dict:
    return {
        "family": "ml",
        "comparison_table": [
            {"model": "Random Survival Forest", "c_index": 0.70, "ibs": 0.18, "brier_skill_score": 0.12, "locked_test_c_index": 0.66},
            {"model": "LASSO-Cox", "c_index": 0.68, "ibs": 0.19, "brier_skill_score": 0.10},
        ],
        "errors": [{"model": "Gradient Boosted Survival", "error": "did not converge"}],
        "n_patients": 600,
        "n_events": 250,
        "evaluation_mode": "repeated_cv",
        "cv_folds": 5,
        "cv_repeats": 3,
        "locked_test_fraction": 0.2,
        "n_development_patients": 480,
        "n_locked_test_patients": 120,
        "n_locked_test_events": 50,
        "evaluation_split_fingerprint": "abc123",
        "request_config": {
            "time_column": "rfs_days",
            "event_column": "event",
            "event_positive_value": 1,
            "features": ["age", "grade", "nodes"],
            "categorical_features": ["grade"],
            "n_estimators": 100,
            "max_depth": None,
            "learning_rate": 0.1,
        },
        **changes,
    }


def _dl_comparison(**changes) -> dict:
    base = _ml_comparison()
    return {
        **base,
        "family": "dl",
        "comparison_table": [{"model": "DeepSurv", "c_index": 0.72}],
        "errors": [],
        "request_config": {
            **{key: base["request_config"][key] for key in ("time_column", "event_column", "event_positive_value", "features", "categorical_features")},
            "epochs": 100,
            "learning_rate": 0.001,
            "hidden_layers": [64, 64],
            "dropout": 0.1,
            "batch_size": 64,
            "early_stopping_patience": 10,
        },
        **changes,
    }


def test_tripod_ai_checklist_describes_shared_splits_and_both_model_families() -> None:
    report = tripod_ai_checklist([_ml_comparison(), _dl_comparison()], dataset={"filename": "gbsg2.csv", "n_rows": 600})

    items = {entry["item"]: entry for entry in report["items"]}
    assert list(items) == [str(number) for number in range(1, 28)]
    assert {entry["status"] for entry in report["items"]} <= set(STATUS_LABELS)
    methods = report["methods"]
    assert "same data partitions (split fingerprint abc123)" in methods
    assert "3 repeats of 5-fold cross-validation stratified by event status on a development set of 480 patients" in methods
    assert "locked test set of 120 patients (50 events; 20% of the cohort)" in methods
    assert "overall accuracy of the classical machine-learning models with the integrated Brier score" in methods
    assert "up to 100 epochs" in methods and "LASSO-Cox penalty was chosen by inner cross-validation" in methods
    assert items["12"]["text"] == methods
    assert items["21"]["text"] == "3 models were fitted; fitting failed for Gradient Boosted Survival."
    assert report["results"].startswith("The highest C-index was 0.720 (DeepSurv).")
    assert "ranged from 0.680 to 0.720" in report["results"]
    assert "3 predictors: age, grade, nodes; categorical: grade" in items["9"]["text"]
    assert "gbsg2.csv" in items["5"]["text"]


def test_tripod_ai_checklist_flags_families_scored_on_different_partitions() -> None:
    report = tripod_ai_checklist([_ml_comparison(), _dl_comparison(evaluation_mode="holdout", evaluation_split_fingerprint="other")])

    assert "different data partitions" in report["methods"]
    items = {entry["item"]: entry for entry in report["items"]}
    assert "Choosing the best of 3 models on the same data makes its C-index optimistic" in items["26"]["text"]
    assert "not a fair comparison" in report["results"]
    assert "For the classical machine-learning models, performance was estimated by 3 repeats" in report["methods"]
    assert "For the deep-learning models, performance was estimated by a holdout set stratified by event status" in report["methods"]


def test_tripod_ai_checklist_reads_a_real_model_comparison() -> None:
    pytest.importorskip("sksurv")
    from survival_toolkit.ml_models import compare_survival_models
    from survival_toolkit.sample_data import make_example_dataset

    features = ["age", "stage", "biomarker_score", "immune_index"]
    result = compare_survival_models(
        make_example_dataset(),
        time_column="os_months",
        event_column="os_event",
        features=features,
        categorical_features=["stage"],
        n_estimators=20,
        random_state=3,
    )
    request = {"time_column": "os_months", "event_column": "os_event", "event_positive_value": 1, "features": features, "categorical_features": ["stage"], "n_estimators": 20, "learning_rate": 0.1}

    report = tripod_ai_checklist([{**result, "family": "ml", "request_config": request}])

    items = {entry["item"]: entry for entry in report["items"]}
    assert f"holdout set of {result['n_evaluation_patients']} patients" in items["16"]["text"]
    assert f"{result['n_patients']} patients" in items["6"]["text"]
    best = max(row["c_index"] for row in result["comparison_table"] if row.get("c_index") is not None)
    assert f"{best:.3f}" in items["23"]["text"]

"""Conservative display rules; fitted models and numerical scores are unchanged."""
from __future__ import annotations

from survival_toolkit.evaluation import MIN_STRATUM_FOR_HOLDOUT

POLICY_VERSION = "prediction-interpretation/1"
# The existing ML summary treats fewer than ten training events as cautionary.
# These are display safeguards, not a sample-size guarantee or study pass criteria.
MIN_TRAIN_EVENTS = 10


def _count(record, keys):
    for key in keys:
        if record.get(key) is not None:
            return float(record[key])
    return None


def interpretation_reasons(mode, train_events=None, test_events=None):
    reasons = []
    if mode not in {"holdout", "repeated_cv"}:
        reasons.append("No complete held-out evaluation; the score is exploratory.")
    if train_events is not None and train_events < MIN_TRAIN_EVENTS:
        reasons.append("Fewer than 10 training events; model ranking is withheld.")
    if test_events is not None and test_events < MIN_STRATUM_FOR_HOLDOUT:
        reasons.append("Fewer than 4 evaluation events; model ranking is withheld.")
    return reasons


def apply_prediction_interpretation(result):
    """Propagate ranking limits to API, summary, comparison CSV and report tables.

    Repeated CV uses the smallest actual fold count, not its rounded average.
    Missing historical counts are not invented. Raw C and fitted risk scores stay
    available, including when the result cannot support a model ranking.
    """
    mode = result.get("evaluation_mode")
    if mode is None:
        return
    rows = result.get("comparison_table")
    records = rows if rows is not None else [result]
    changed = False
    notes = []
    for record in records:
        train = _count(record, ("train_events", "training_events", "n_fit_events"))
        test = _count(record, ("test_events", "evaluation_events", "n_evaluation_events"))
        if train is None:
            train = _count(result, ("n_fit_events", "training_events"))
        if test is None:
            test = _count(result, ("n_evaluation_events", "evaluation_events"))
        folds = [r for r in result.get("fold_results", []) if r.get("model") == record.get("model")]
        for keys, is_train in ((("train_events", "training_events"), True), (("test_events", "evaluation_events"), False)):
            counts = [_count(fold, keys) for fold in folds]
            counts = [value for value in counts if value is not None]
            if counts:
                if is_train:
                    train = min(counts)
                else:
                    test = min(counts)
        reasons = interpretation_reasons(record.get("evaluation_mode", mode), train, test)
        record["performance_interpretation"] = {
            "policy": POLICY_VERSION, "status": "exploratory" if reasons else "evaluation_available",
            "reasons": reasons, "minimum_training_events": train, "minimum_evaluation_events": test,
            "boundary": "Display safeguards do not establish adequate power or clinical validity.",
        }
        if reasons:
            changed = True
            notes.extend(reasons)
            if rows is not None:
                record["rank"] = None
                record["comparable_for_ranking"] = False
    notes = list(dict.fromkeys(notes))
    if not changed:
        return
    if rows is not None:
        eligible = [r for r in rows if r.get("rank") is not None and r.get("comparable_for_ranking") is not False]
        for rank, row in enumerate(eligible, 1):
            row["rank"] = rank
        result["ranking_complete"] = False
        result["excluded_models"] = sorted(set(result.get("excluded_models", [])) | {
            str(row["model"]) for row in rows if row.get("rank") is None})
    for key in ("scientific_summary", "insight_board"):
        summary = result.get(key)
        if not isinstance(summary, dict):
            continue
        previous_headline = str(summary.get("headline", ""))
        existing_no_rank = previous_headline.startswith("No ") and "rank" in previous_headline.lower()
        if summary.get("status") != "review":
            summary["status"] = "caution"
        headline = (
            "Some scores are exploratory and excluded from model ranking. Review the evaluation limits below."
            if rows is not None else "This is an exploratory score; it does not establish predictive performance."
        )
        if not existing_no_rank:
            summary["headline"] = headline
        # Old positive judgments and a selected top model must not survive a hold.
        summary["strengths"] = []
        summary["metrics"] = [m for m in summary.get("metrics", []) if m.get("value") is None or not any(
            word in str(m.get("label", "")).lower() for word in ("top", "best", "rank"))]
        summary.setdefault("cautions", []).extend(n for n in notes if n not in summary.get("cautions", []))
        summary.setdefault("next_steps", []).append("Use more events and held-out evaluation before comparing model performance.")
    tables = result.get("manuscript_tables")
    if isinstance(tables, dict):
        tables.setdefault("table_notes", []).extend(n for n in notes if n not in tables.get("table_notes", []))
        ranks = {row["model"]: row.get("rank") for row in rows or []}
        for value in tables.values():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict) and row.get("Model") in ranks and "Rank" in row:
                        row["Rank"] = ranks[row["Model"]] if ranks[row["Model"]] is not None else "Not ranked"
    result["performance_interpretation_notes"] = notes

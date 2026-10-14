"""Training-only outcome reconstruction screen; findings require provenance review.

These are engineering thresholds, not a hypothesis test or a leakage certificate.
Predicting observed follow-up/censoring is different from predicting survival risk.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROXY_VERSION = "outcome-proxy/2"
MIN_ROWS = 48
STRUCTURAL_MIN_ROWS = 24
STRUCTURAL_EVENT_MIN_PER_LEVEL = 6
MAX_ROWS = 1200
INNER_FOLDS = 3
SCREEN_SEED = 2026101101
LIMITATION = (
    "Patterns alone cannot establish when a variable was measured or prove leakage. "
    "A strong baseline predictor can also be flagged. Undetected proxies may remain."
)


def training_fingerprint(frame: pd.DataFrame, categorical_features: Sequence[str] = ()) -> str:
    digest = hashlib.sha256()
    schema = {
        "columns": [repr(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "category_levels": [
            {"levels": [repr(level) for level in dtype.categories], "ordered": dtype.ordered}
            if isinstance(dtype, pd.CategoricalDtype) else None for dtype in frame.dtypes
        ],
        "categorical_features": list(map(str, categorical_features)),
    }
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8", errors="replace"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _structural_findings(frame, time_column, event_column, features):
    from survival_toolkit.concurrency import raise_if_cancelled

    findings = []
    times = frame[time_column].to_numpy(dtype=float)
    events = frame[event_column].to_numpy(dtype=int)
    for position, feature in enumerate(features):
        if position % 32 == 0:
            raise_if_cancelled()
        # A string category can be a renamed event indicator too. Two levels
        # must map exactly to the two event states, with support in both states.
        # This requests source/timing review; it does not establish leakage.
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            labels = frame[feature].astype("string").str.strip()
            usable = (labels.notna() & labels.ne("")).to_numpy(dtype=bool, na_value=False)
            n_usable = int(usable.sum())
            if n_usable >= STRUCTURAL_MIN_ROWS and labels[usable].nunique() == 2:
                states, counts = np.unique(events[usable], return_counts=True)
                if len(states) == 2 and counts.min() >= STRUCTURAL_EVENT_MIN_PER_LEVEL:
                    groups = pd.DataFrame({"label": labels[usable].to_numpy(), "event": events[usable]})
                    if groups.groupby("label", observed=True)["event"].nunique().eq(1).all():
                        findings.append({"kind": "exact_categorical_event_pattern", "target": "event",
                            "features": [str(feature)], "score": 1.0, "rows_checked": n_usable,
                            "training_rows": int(len(frame)), "missing_predictor_rows": int(len(frame) - n_usable)})
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        n_finite = int(finite.sum())
        if n_finite < STRUCTURAL_MIN_ROWS or np.unique(values[finite]).size < 2:
            continue
        x = values[finite]
        centered = x - x.mean()
        norm = np.linalg.norm(centered)
        if norm == 0 or not np.isfinite(norm):
            continue
        x = centered / norm
        for target_name, target in (("event", events[finite]), ("follow_up", times[finite])):
            if target_name == "event":
                states, counts = np.unique(target, return_counts=True)
                if len(states) != 2 or counts.min() < STRUCTURAL_EVENT_MIN_PER_LEVEL:
                    continue
                if any(f["features"] == [str(feature)] and f["target"] == "event" for f in findings):
                    continue
            y = target - target.mean()
            y_norm = np.linalg.norm(y)
            if y_norm == 0:
                continue
            r2 = float(np.clip(np.dot(x, y / y_norm) ** 2, 0, 1))
            if r2 >= 1 - 1e-12:
                findings.append({"kind": "exact_affine_outcome_pattern", "target": target_name,
                                 "features": [str(feature)], "score": r2, "rows_checked": n_finite,
                                 "training_rows": int(len(frame)), "missing_predictor_rows": int(len(frame) - n_finite)})
        # Rank equality also catches renamed log, square and reciprocal time columns.
        if np.unique(times[finite]).size >= 12:
            ranks_x = pd.Series(values[finite]).rank().to_numpy()
            ranks_t = pd.Series(times[finite]).rank().to_numpy()
            rho = float(np.corrcoef(ranks_x, ranks_t)[0, 1])
            if abs(rho) >= 1 - 1e-12 and not any(
                f["features"] == [str(feature)] and f["target"] == "follow_up" for f in findings
            ):
                findings.append({"kind": "exact_monotone_follow_up_pattern", "target": "follow_up",
                                 "features": [str(feature)], "score": abs(rho), "rows_checked": n_finite,
                                 "training_rows": int(len(frame)), "missing_predictor_rows": int(len(frame) - n_finite)})
    return findings


def _inner_design(train, validation, features, categorical_features):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categories = [column for column in features if column in categorical_features or
                  not pd.api.types.is_numeric_dtype(train[column])]
    numeric = [column for column in features if column not in categories]
    # Conversion does not learn from validation rows; imputation, levels and scale do.
    fit = train[list(features)].copy()
    score = validation[list(features)].copy()
    for column in numeric:
        fit[column] = pd.to_numeric(fit[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        score[column] = pd.to_numeric(score[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in categories:
        fit[column] = fit[column].astype("string").fillna("<missing>").astype(str)
        score[column] = score[column].astype("string").fillna("<missing>").astype(str)
    transformers = []
    if numeric:
        transformers.append(("numeric", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric))
    if categories:
        transformers.append(("category", OneHotEncoder(
            handle_unknown="ignore", max_categories=32, sparse_output=False), categories))
    encoder = ColumnTransformer(transformers)
    return encoder.fit_transform(fit), encoder.transform(score)


def screen_training_outcomes(frame: pd.DataFrame, time_column: str, event_column: str,
                             features: Sequence[str], categorical_features: Sequence[str] = ()) -> dict[str, Any]:
    """Screen one actual training partition without using its evaluation rows.

    Fixed three-fold out-of-fold predictions inside that partition test near-perfect
    reconstruction by a tree ensemble and a small two-hidden-layer neural network.
    No survival model, selected features or reported survival metrics are changed.
    """
    from survival_toolkit.concurrency import raise_if_cancelled

    features = list(features)
    audit = {"version": PROXY_VERSION, "scope": "training_partition_only",
             "qualification": "engineering_screen_not_leakage_certificate", "coverage": "limited",
             "training_rows": int(len(frame)), "training_hash": training_fingerprint(
                 frame[[time_column, event_column, *features]], categorical_features),
             "status": "not_assessed", "findings": [], "scores": [], "notes": [LIMITATION],
             "settings": {"seed": SCREEN_SEED, "inner_folds": INNER_FOLDS, "min_rows": MIN_ROWS,
                          "structural_min_rows": STRUCTURAL_MIN_ROWS,
                          "structural_event_min_per_level": STRUCTURAL_EVENT_MIN_PER_LEVEL,
                          "max_rows": MAX_ROWS, "event_auc": 0.995, "event_balanced_accuracy": 0.98,
                          "log_follow_up_r2": 0.995, "tree_count": 48, "tree_depth": 6,
                          "neural_layers": [16, 8], "neural_max_iter": 150,
                          "neural_max_features": 64, "joint_max_features": 256, "category_budget": 32}}
    findings = _structural_findings(frame, time_column, event_column, features)
    audit["findings"] = findings
    if findings:
        audit["status"] = "review_required"
        audit["notes"].append("An exact outcome pattern was found in the recorded complete training rows; learned screening was not needed.")
        if len(frame) >= MIN_ROWS:
            audit["coverage"] = "screened"
        else:
            audit["notes"].append(f"Structural checks ran, but learned screening needs at least {MIN_ROWS} training rows and was not assessed.")
        return audit
    if len(frame) < MIN_ROWS or not features:
        audit["notes"].append(f"Pattern screening needs at least {MIN_ROWS} training rows and one predictor; it was not assessed.")
        return audit
    import sklearn
    audit["environment"] = {"sklearn": sklearn.__version__, "numpy": np.__version__}
    audit["coverage"] = "screened"
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_selection import f_classif, f_regression
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from threadpoolctl import threadpool_limits

    # Deterministic, outcome-independent subsampling bounds interactive cost.
    work = frame.reset_index(drop=True)
    if len(work) > MAX_ROWS:
        positions = np.random.default_rng(SCREEN_SEED).choice(len(work), MAX_ROWS, replace=False)
        work = work.iloc[np.sort(positions)].reset_index(drop=True)
        audit["notes"].append(f"Learned screening used {MAX_ROWS} training rows; structural checks used all training rows.")
        audit["coverage"] = "limited"
    audit["screened_rows"] = int(len(work))
    categories = [feature for feature in features if feature in categorical_features
                  or not pd.api.types.is_numeric_dtype(frame[feature])]
    budgeted_categories = [str(feature) for feature in categories
                           if frame[feature].nunique(dropna=False) > 32]
    if budgeted_categories:
        audit["coverage"] = "limited"
        audit["category_budget_columns"] = budgeted_categories
        audit["notes"].append("Learned screening groups rare categories beyond the 32-category budget; categorical coverage is limited.")
    event = work[event_column].to_numpy(dtype=int)
    times = np.log1p(work[time_column].to_numpy(dtype=float))
    audit["status"] = "partially_assessed" if audit["coverage"] == "limited" else "no_pattern_detected"
    for target_name, target in (("event", event), ("log_follow_up", times)):
        _, counts = np.unique(target, return_counts=True)
        if target_name == "event" and (len(counts) != 2 or counts.min() < 12):
            audit["status"] = "partially_assessed"
            audit["notes"].append("Event reconstruction was not assessed: fewer than 12 training rows in an event stratum.")
            continue
        if target_name == "log_follow_up" and np.std(target) == 0:
            audit["status"] = "partially_assessed"
            audit["notes"].append("Follow-up reconstruction was not assessed: follow-up is constant.")
            continue
        splitter = (StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SCREEN_SEED)
                    if target_name == "event" else KFold(INNER_FOLDS, shuffle=True, random_state=SCREEN_SEED))
        predictions = {method: np.full(len(work), np.nan) for method in ("tree", "neural")}
        budget_hits = {method: 0 for method in predictions}
        for fold, (fit_rows, score_rows) in enumerate(splitter.split(work, target)):
            raise_if_cancelled()
            x_fit, x_score = _inner_design(work.iloc[fit_rows], work.iloc[score_rows], features, categorical_features)
            # Selection is fitted independently inside each screen-training fold.
            f_scores = (f_classif if target_name == "event" else f_regression)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                relevance = np.nan_to_num(f_scores(x_fit, target[fit_rows])[0], nan=-1, posinf=1e300)
            if x_fit.shape[1] > 256:
                columns = np.argsort(-relevance, kind="stable")[:256]
                x_fit, x_score, relevance = x_fit[:, columns], x_score[:, columns], relevance[columns]
                audit["status"] = "partially_assessed"
            neural_columns = np.argsort(-relevance, kind="stable")[:64]
            if x_fit.shape[1] > 64:
                audit["status"] = "partially_assessed"
            y_fit = target[fit_rows]
            location, scale = (0.0, 1.0) if target_name == "event" else (float(y_fit.mean()), float(y_fit.std()))
            if scale == 0:
                raise ValueError("Follow-up is constant within a screen-training fold.")
            models = {
                "tree": (RandomForestClassifier(n_estimators=48, max_depth=6, min_samples_leaf=2,
                                                 max_features=1.0, n_jobs=1, random_state=SCREEN_SEED + fold)
                         if target_name == "event" else ExtraTreesRegressor(
                             n_estimators=48, max_depth=6, min_samples_leaf=2, n_jobs=1, random_state=SCREEN_SEED + fold)),
                "neural": (MLPClassifier if target_name == "event" else MLPRegressor)(
                    hidden_layer_sizes=(16, 8), activation="tanh", solver="lbfgs", alpha=0.1,
                    max_iter=150, max_fun=5000, random_state=SCREEN_SEED + fold),
            }
            for method, model in models.items():
                raise_if_cancelled()
                fit_x = x_fit[:, neural_columns] if method == "neural" else x_fit
                score_x = x_score[:, neural_columns] if method == "neural" else x_score
                with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(fit_x, (y_fit - location) / scale)
                budget_hits[method] += sum(issubclass(w.category, ConvergenceWarning) for w in recorded)
                prediction = (model.predict_proba(score_x)[:, 1] if target_name == "event"
                              else model.predict(score_x) * scale + location)
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite outcome reconstruction predictions.")
                predictions[method][score_rows] = prediction
        for method, prediction in predictions.items():
            score = {"method": method, "target": target_name, "neural_budget_hits": budget_hits[method],
                     "scope": "out_of_fold_within_training_partition"}
            if target_name == "event":
                score["auc"] = float(roc_auc_score(target, prediction))
                score["balanced_accuracy"] = float(balanced_accuracy_score(target, prediction >= 0.5))
                flagged = score["auc"] >= 0.995 and score["balanced_accuracy"] >= 0.98
            else:
                score["r2"] = float(r2_score(target, prediction))
                flagged = score["r2"] >= 0.995
            audit["scores"].append(score)
            if flagged:
                audit["findings"].append({"kind": "near_perfect_outcome_reconstruction", "target": target_name,
                                          "method": method, "features": list(map(str, features)), "metrics": score})
    if audit["status"] == "partially_assessed":
        audit["coverage"] = "limited"
        audit["notes"].append("Learned coverage is limited by stratum size or feature budgets; no clear result is a leakage certificate.")
    if audit["findings"]:
        audit["status"] = "review_required"
    return audit

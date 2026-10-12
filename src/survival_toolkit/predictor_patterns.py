"""Compare added predictors and describe fixed-model predictions on training support."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from survival_toolkit.analysis import (
    _coerce_survival_time_values, _cohort_frame, _require_dataframe_columns, detect_duplicate_identifier_columns,
)
from survival_toolkit.concurrency import raise_if_cancelled
from survival_toolkit.encoding import canonical_category_values, transform_feature_encoder
from survival_toolkit.errors import InternalAnalysisError, UserInputError, user_input_boundary
from survival_toolkit.evaluation import evaluation_split_fingerprint, stratified_holdout_indices
from survival_toolkit.marker_screen import harrell_c_many


def _paired_intervals(time, event, risks, *, n_bootstrap, seed):
    """Retain all scheduled draws, including undefined concordances, without replacement."""
    points = harrell_c_many(time, event, risks)
    if not np.isfinite(points).all():
        raise UserInputError("The held-out patients have no usable pairs for risk-ranking comparison.")
    draws = np.full((n_bootstrap, 2), np.nan)
    rng = np.random.default_rng(seed)
    for i in range(n_bootstrap):
        raise_if_cancelled()
        rows = rng.integers(0, len(time), size=len(time))
        draws[i] = harrell_c_many(time[rows], event[rows], risks[rows])
    valid = np.isfinite(draws).all(axis=1)
    usable = draws[valid]
    intervals = [[None, None], [None, None]]
    delta_ci = [None, None]
    if len(usable) >= 20:
        intervals = np.quantile(usable, [.025, .975], axis=0).T.tolist()
        delta_ci = np.quantile(usable[:, 1] - usable[:, 0], [.025, .975]).tolist()
    return {
        "rows": [{"feature_set": name, "c_index": float(points[i]), "c_index_ci": intervals[i]}
                 for i, name in enumerate(["Clinical only", "Clinical + added predictors"])],
        "delta_c": float(points[1] - points[0]), "delta_ci": delta_ci,
        "n_bootstrap_requested": n_bootstrap, "n_bootstrap_valid": int(valid.sum()),
        "n_bootstrap_undefined": int((~valid).sum()),
        "interval_scope": "95% paired test-patient bootstrap, conditional on these two fitted models; no retraining uncertainty",
        "_bootstrap": draws,
    }


def _fit(frame, *, time_column, event_column, features, categorical_features,
         model_type, train_idx, test_idx, seed, n_estimators, max_depth, epochs):
    train = frame.iloc[train_idx].copy()
    if model_type == "deepsurv":
        from survival_toolkit.deep_models import (
            _fit_deep_encoder, _transform_deep_frame, train_deepsurv,
        )
        encoder = _fit_deep_encoder(train, features, categorical_features)
        # Explicitly reuse the common split; neither encoder nor early stopping sees test rows.
        ordered = pd.concat([train, frame.iloc[test_idx]], ignore_index=True)
        prepared = _transform_deep_frame(ordered, time_column=time_column,
                                        event_column=event_column, encoder=encoder)
        split = {"train_idx": np.arange(len(train_idx)),
                 "eval_idx": np.arange(len(train_idx), len(frame)),
                 "evaluation_mode": "holdout",
                 "evaluation_note": "Shared 70/30 held-out patients for both feature sets."}
        result = train_deepsurv(
            None, time_column, event_column, features, categorical_features=categorical_features,
            event_positive_value=1, prepared_data=prepared, evaluation_split=split,
            random_seed=seed, epochs=epochs, _return_fitted_model=True,
        )
        model = result.pop("_fitted_model")

        def predict(raw):
            import torch
            encoded = transform_feature_encoder(raw, encoder, output="dataframe")
            tensor = torch.from_numpy(np.ascontiguousarray(encoded.to_numpy(dtype=np.float32)))
            model.eval()
            with torch.inference_mode():
                return model(tensor).detach().cpu().numpy().reshape(-1).astype(float)

        metadata = {"epochs_trained": result["epochs_trained"], "tie_method": "breslow"}
    else:
        from survival_toolkit.ml_models import train_gradient_boosted_survival, train_random_survival_forest
        trainer = train_gradient_boosted_survival if model_type == "gbs" else train_random_survival_forest
        result = trainer(
            frame, time_column, event_column, features, categorical_features=categorical_features,
            event_positive_value=1, random_state=seed, n_estimators=n_estimators,
            max_depth=max_depth, compute_importance=False, compute_brier=False,
        )
        if result["model_stats"]["evaluation_mode"] != "holdout":
            raise UserInputError("A real holdout is required; apparent evaluation is not used here.")
        # Test alignment independently of any metric or rounding convention.
        for actual, positions in [(result["_analysis_train_frame"], train_idx),
                                  (result["_analysis_eval_frame"], test_idx)]:
            expected = frame.iloc[positions].reset_index(drop=True)
            if not actual[[time_column, event_column, *features]].equals(expected[[time_column, event_column, *features]]):
                raise InternalAnalysisError("The fitted model did not retain the shared patient partition.")
        encoder = result["_feature_encoder"]
        columns = list(result["_X_encoded"].columns)
        model = result["_model"]

        def predict(raw):
            encoded = transform_feature_encoder(raw, encoder, output="dataframe").reindex(columns=columns, fill_value=0)
            return np.asarray(model.predict(encoded.to_numpy(dtype=float)), dtype=float).reshape(-1)

        metadata = {"n_estimators": n_estimators, "max_depth": max_depth}
    test_risk = predict(frame.iloc[test_idx])
    if len(test_risk) != len(test_idx) or not np.isfinite(test_risk).all():
        raise InternalAnalysisError("The model did not predict a finite risk for every shared test patient.")
    encoder_hash = hashlib.sha256(json.dumps(encoder, sort_keys=True, default=str).encode()).hexdigest()
    return {"predict": predict, "encoder": encoder, "training": train,
            "test_risk": test_risk, "transform_hash": encoder_hash, "settings": metadata}


def _category_labels(series):
    return canonical_category_values(pd.Series(series))


def _levels(series):
    return sorted(canonical_category_values(series).dropna().unique().tolist())


def describe_pattern(fitted, *, target_feature, group_feature, seed, n_points=25):
    """Common training-background curves, with unsupported combinations explicitly null."""
    train = fitted["training"]
    encoder = fitted["encoder"]
    categorical = set(encoder.get("categorical_features", []))
    if group_feature not in categorical:
        raise UserInputError("The grouping variable must be categorical; mark stage as categorical.")
    levels = _levels(train[group_feature])
    if not 1 <= len(levels) <= 8:
        raise UserInputError("Choose a grouping variable with one to eight training categories.")
    is_categorical = target_feature in categorical
    if is_categorical:
        grid = _levels(train[target_feature])
        if not 1 <= len(grid) <= 12:
            raise UserInputError("The added predictor has too many categories to display.")
    else:
        values = pd.to_numeric(train[target_feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 10 or values.nunique() < 2:
            raise UserInputError("The added predictor needs at least ten training measurements and two values.")
        lo, hi = values.quantile([.05, .95])
        grid = np.linspace(float(lo), float(hi), n_points).tolist()
    # Sample positions without outcomes; the same background is used for every category.
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(train), min(300, len(train)), replace=False))
    background = train.iloc[rows].copy()
    observed_risk = fitted["predict"](background)
    center, scale = float(np.mean(observed_risk)), float(np.std(observed_risk))
    if not np.isfinite(observed_risk).all() or not np.isfinite(scale) or scale <= 1e-12:
        raise UserInputError("The fitted training risk is constant; a risk-pattern curve is not estimable.")
    labels = _category_labels(train[group_feature])
    records, supports = [], []
    for level in levels:
        raise_if_cancelled()
        part = train.loc[labels == str(level)]
        if is_categorical:
            observed = set(_levels(part[target_feature]))
            low = high = None
        else:
            measured = pd.to_numeric(part[target_feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            low, high = (measured.quantile([.05, .95]).tolist() if len(measured) >= 10 else [None, None])
            observed = set()
        supports.append({"group": str(level), "training_patients": len(part), "lower": low, "upper": high})
        for value in grid:
            supported = (len(part) >= 10 and (str(value) in observed if is_categorical else
                         low is not None and float(low) <= float(value) <= float(high)))
            score = None
            if supported:
                variant = background.copy()
                variant[group_feature] = level
                variant[target_feature] = value
                risk = fitted["predict"](variant)
                if not np.isfinite(risk).all():
                    raise UserInputError("A model prediction failed; the pattern is not reported.")
                score = float((np.mean(risk) - center) / scale)
            records.append({"group": str(level), "value": value, "risk_score": score, "supported": supported})
    if not any(r["supported"] for r in records):
        raise UserInputError("No group has enough observed training support for this pattern.")
    return {"target_feature": target_feature, "group_feature": group_feature,
            "target_type": "categorical" if is_categorical else "numeric",
            "rows": records, "support": supports, "background_patients": len(background),
            "risk_scale": "Mean model risk centered and scaled by the common training-background risk (training SD)",
            "scope": "Fixed augmented model; common training background; group-specific 5th–95th percentile support for numeric targets",
            "interpretation": "Model predictions, not observed survival probabilities, causal effects or an interaction test",
            "model_refitted_for_curve": False}


@user_input_boundary
def compare_predictor_sets(
    df: pd.DataFrame, *, time_column: str, event_column: str, event_positive_value: Any,
    clinical_features: Sequence[str], added_features: Sequence[str], categorical_features: Sequence[str],
    target_feature: str, group_feature: str, model_type: str = "gbs", random_seed: int = 42,
    n_estimators: int = 100, max_depth: int | None = 3, epochs: int = 100,
    n_bootstrap: int = 1000, bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    clinical, added = list(clinical_features), list(added_features)
    if not clinical or not added or set(clinical) & set(added):
        raise UserInputError("Choose nonempty, separate clinical and added predictor sets.")
    features = clinical + added
    if target_feature not in added or group_feature not in clinical or target_feature == group_feature:
        raise UserInputError("Choose the pattern predictor from the added set and its group from the clinical set.")
    if model_type not in {"gbs", "rsf", "deepsurv"}:
        raise UserInputError("This comparison supports gradient boosting, survival forest and DeepSurv.")
    _require_dataframe_columns(df, [time_column, event_column, *features])
    if time_column in features or event_column in features:
        raise UserInputError("Outcome columns cannot be used as predictors.")
    raw_times = _coerce_survival_time_values(df[time_column], time_column)
    if (raw_times < 0).any():
        raise UserInputError("Negative follow-up times must be corrected before predictor comparison.")
    if detect_duplicate_identifier_columns(df):
        raise UserInputError("Repeated patient identifiers must be resolved before predictor comparison.")
    frame = _cohort_frame(df, time_column, event_column, event_positive_value=event_positive_value,
                          extra_columns=features, drop_missing_extra_columns=False)
    source_rows = list(frame.attrs.get("source_row_index", frame.index.tolist()))
    frame = frame.reset_index(drop=True)
    frame.attrs = {}
    train_idx, test_idx, mode = stratified_holdout_indices(frame[event_column], random_state=random_seed)
    if mode != "holdout":
        raise UserInputError("Not enough patients or events for a real holdout; no apparent fallback is used.")
    fingerprint = evaluation_split_fingerprint(source_rows, [(train_idx, test_idx)], kind="holdout")
    results = []
    for selected in [clinical, features]:
        raise_if_cancelled()
        results.append(_fit(frame, time_column=time_column, event_column=event_column,
                            features=selected, categorical_features=[c for c in categorical_features if c in selected],
                            model_type=model_type, train_idx=train_idx, test_idx=test_idx, seed=random_seed,
                            n_estimators=n_estimators, max_depth=max_depth, epochs=epochs))
    risk = np.column_stack([r["test_risk"] for r in results])
    time = frame.iloc[test_idx][time_column].to_numpy(dtype=float)
    event = frame.iloc[test_idx][event_column].to_numpy(dtype=int)
    interval_seed = int(bootstrap_seed if bootstrap_seed is not None else (random_seed + 1) % 2**32)
    comparison = _paired_intervals(time, event, risk, n_bootstrap=n_bootstrap, seed=interval_seed)
    draws = comparison.pop("_bootstrap")
    pattern = None
    pattern_error = None
    # An unestimable descriptive curve does not invalidate the separately completed paired comparison.
    try:
        pattern = describe_pattern(results[1], target_feature=target_feature, group_feature=group_feature, seed=random_seed)
    except UserInputError as exc:
        pattern_error = str(exc)
    ci = comparison["delta_ci"]
    interpretation = ("Improvement is uncertain: the paired interval includes zero." if ci[0] is None or ci[0] <= 0 <= ci[1]
                      else "Added predictors improved risk ranking in this fixed holdout." if ci[0] > 0
                      else "Added predictors reduced risk ranking in this fixed holdout.")
    return {"method_version": "predictor-patterns/1", "model_type": model_type,
            "clinical_features": clinical, "added_features": added,
            "evaluation_mode": "holdout", "evaluation_split_fingerprint": fingerprint,
            "training_patients": len(train_idx), "test_patients": len(test_idx), "test_events": int(event.sum()),
            "excluded_missing_outcome_rows": len(df) - len(frame), "random_seed": random_seed,
            "bootstrap_seed": interval_seed, "comparison": comparison, "interpretation": interpretation,
            "transform_hashes": [r["transform_hash"] for r in results],
            "model_settings": [r["settings"] for r in results],
            "pattern": pattern, "pattern_error": pattern_error,
            "scope": "Exploratory added-predictor comparison; repeated choices on these test patients are not independent validation",
            "test_predictions": {"row_ids": [str(source_rows[i]) for i in test_idx], "time": time.tolist(),
                                 "event": event.tolist(), "risks": {"Clinical only": risk[:, 0].tolist(),
                                                                    "Clinical + added predictors": risk[:, 1].tolist()}},
            # Internal-only reproducibility material; stripped from ordinary API responses.
            "_bootstrap": draws, "_training_positions": train_idx.tolist(), "_test_positions": test_idx.tolist()}

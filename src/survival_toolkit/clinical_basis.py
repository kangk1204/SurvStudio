"""Training-only clinical transformations for marker evaluation (method version 1).

The raw spline columns match Hmisc::rcspline.eval(inclx=TRUE, norm=2).
No knot jitter, basis selection from outcomes, or clipping of external values is used.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from survival_toolkit.encoding import fit_feature_encoder, transform_feature_encoder

KNOT_QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)
CLINICAL_BASES = ("linear", "restricted_cubic_spline")
BASIS_METHOD_VERSION = "clinical-basis/1"


class ClinicalBasisError(ValueError):
    """A requested clinical basis cannot be estimated; no alternative is selected."""


def rcs_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Explicit five-knot restricted cubic spline with linear tails, Hmisc norm=2."""
    values = np.asarray(x, dtype=float).reshape(-1)
    k = np.asarray(knots, dtype=float)
    if k.shape != (5,) or not np.isfinite(k).all() or not np.all(np.diff(k) > 0):
        raise ClinicalBasisError("Five finite, strictly increasing spline knots are required.")
    if not np.isfinite(values).all():
        raise ClinicalBasisError("Clinical spline inputs must be finite after training imputation.")
    # Computing in span units avoids overflow in the cubic terms for large raw units.
    span = k[-1] - k[0]
    u = np.minimum((values - k[0]) / span, 1.0)
    t = (k - k[0]) / span
    last = np.maximum(u - t[-1], 0.0) ** 3
    penultimate = np.maximum(u - t[-2], 0.0) ** 3
    nonlinear = [span * (np.maximum(u - tj, 0.0) ** 3
                         + ((t[-2] - tj) * last - (t[-1] - tj) * penultimate)
                         / (t[-1] - t[-2])) for tj in t[:-2]]
    # Evaluate the right tail directly as a line: subtraction of huge cubic terms
    # would lose precision for external values far beyond the development range.
    beyond = values > k[-1]
    for j, tj in enumerate(t[:-2]):
        boundary = (1 - tj) * ((1 - tj) ** 2 - (1 - t[-2]) ** 2)
        slope = 3 * (1 - tj) * (t[-2] - tj)
        nonlinear[j][beyond] = span * boundary + slope * (values[beyond] - k[-1])
    result = np.column_stack([values, *nonlinear])
    if not np.isfinite(result).all():
        raise ClinicalBasisError("The clinical spline transformation is not finite.")
    return result


def fit_clinical_encoder(frame: pd.DataFrame, columns: Sequence[str],
                         categorical: Sequence[str] = (), *, basis: str = "linear") -> dict[str, Any]:
    if basis not in CLINICAL_BASES:
        raise ValueError(f"clinical_basis must be one of {CLINICAL_BASES}.")
    try:
        base = fit_feature_encoder(frame, columns, categorical, standardize_numeric=False)
    except ValueError as exc:
        if "No usable features remain" in str(exc):
            raise ClinicalBasisError(str(exc)) from exc
        raise
    for name in base["numeric_features"]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ClinicalBasisError(f"Clinical feature {name} contains infinite values.")
    encoded = transform_feature_encoder(frame, base, output="dataframe")
    names = list(base["feature_names"])
    specifications: dict[str, Any] = {}
    numeric_kinds: dict[str, str] = {}
    occupied = set(names)
    for name in base["numeric_features"]:
        x = encoded[name].to_numpy(dtype=float)
        observed = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy(dtype=float)
        numeric_kinds[name] = "binary_or_constant" if np.unique(observed).size <= 2 else "continuous"
        if basis == "linear" or numeric_kinds[name] != "continuous":
            continue
        knots = np.quantile(x, KNOT_QUANTILES, method="linear")
        raw = rcs_basis(x, knots)
        terms = [name]
        for j in range(1, 4):
            term = f"{name}__rcs{j}"
            if term in occupied:
                raise ClinicalBasisError(f"Generated spline term {term} conflicts with another clinical term.")
            occupied.add(term)
            terms.append(term)
        # The linear column retains its original units for interpretable linear coefficients.
        means = np.r_[0.0, raw[:, 1:].mean(axis=0)]
        scales = np.r_[1.0, raw[:, 1:].std(axis=0)]
        if np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ClinicalBasisError(f"The spline for {name} has a degenerate column.")
        specifications[name] = {"knots": knots.tolist(), "terms": terms,
                                "means": means.tolist(), "scales": scales.tolist()}
        position = names.index(name)
        names[position:position + 1] = terms
    return {**base, "feature_names": names, "encoded_columns": names,
            "clinical_basis": basis, "basis_method_version": BASIS_METHOD_VERSION,
            "knot_quantiles": list(KNOT_QUANTILES), "spline_specifications": specifications,
            "numeric_kinds": numeric_kinds,
            "base_encoder": base}


def transform_clinical_encoder(frame: pd.DataFrame, encoder: dict[str, Any], *,
                               output: str = "numpy") -> np.ndarray | pd.DataFrame:
    # Recipes v1/v2 use exactly their original encoder and prediction calculation.
    base = encoder.get("base_encoder", encoder)
    result = transform_feature_encoder(frame, base, output="dataframe")
    for name, spec in encoder.get("spline_specifications", {}).items():
        raw = rcs_basis(result[name].to_numpy(dtype=float), spec["knots"])
        raw = (raw - np.asarray(spec["means"])) / np.asarray(spec["scales"])
        for j, term in enumerate(spec["terms"]):
            result[term] = raw[:, j]
    result = result.reindex(columns=encoder["feature_names"])
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ClinicalBasisError("The frozen clinical transformation produced non-finite values.")
    return result if output == "dataframe" else result.to_numpy(dtype=float)


def check_clinical_encoder(encoder: dict[str, Any]) -> None:
    try:
        _check_clinical_encoder(encoder)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("The locked clinical transformation is malformed.") from exc


def _check_clinical_encoder(encoder: dict[str, Any]) -> None:
    """Validate v3 transformation structure in addition to the recipe's integrity hash."""
    if encoder.get("clinical_basis") not in CLINICAL_BASES:
        raise ValueError("The locked clinical basis is invalid.")
    if encoder.get("basis_method_version") != BASIS_METHOD_VERSION:
        raise ValueError("The locked clinical transformation version is unsupported.")
    base = encoder.get("base_encoder")
    if not isinstance(base, dict) or base.get("features") != encoder.get("features"):
        raise ValueError("The locked clinical base encoder is invalid.")
    kinds = encoder.get("numeric_kinds")
    if not isinstance(kinds, dict) or set(kinds) != set(base["numeric_features"]) or any(v not in {"continuous", "binary_or_constant"} for v in kinds.values()):
        raise ValueError("The locked clinical numeric types are invalid.")
    expected = list(base["feature_names"])
    for name, spec in encoder.get("spline_specifications", {}).items():
        if encoder["clinical_basis"] != "restricted_cubic_spline" or name not in base["numeric_features"]:
            raise ValueError("The locked spline is not a numeric clinical feature.")
        rcs_basis(np.zeros(1), spec["knots"])
        for values in (spec["knots"], spec["means"], spec["scales"]):
            if not isinstance(values, list) or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
                raise ValueError("The locked spline needs numeric knot/scaling lists.")
        means, scales = np.asarray(spec["means"], float), np.asarray(spec["scales"], float)
        if means.shape != (4,) or scales.shape != (4,) or not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("The locked spline scaling is invalid.")
        terms = spec["terms"]
        if len(terms) != 4 or terms[0] != name or len(set(terms)) != 4:
            raise ValueError("The locked spline terms are invalid.")
        i = expected.index(name)
        expected[i:i + 1] = terms
    if expected != encoder["feature_names"] or len(set(expected)) != len(expected):
        raise ValueError("The locked clinical term mapping is invalid.")

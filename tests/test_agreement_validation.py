import importlib.util
from pathlib import Path

import numpy as np


_path = Path(__file__).resolve().parents[1] / "validation" / "agreement" / "run_agreement.py"
_spec = importlib.util.spec_from_file_location("agreement_runner", _path)
agreement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agreement)


def test_agreement_check_accepts_tolerance_and_matching_non_estimability():
    cases = [{"name": "fixture", "r": {"coefficient": 1.0, "upper median CI": np.nan},
              "survstudio": {"coefficient": 1.0 + 1e-8, "upper median CI": np.nan}}]
    assert agreement.agreement_failures(cases, atol=1e-6, rtol=1e-7) == []


def test_agreement_check_catches_errors_the_difference_summary_hides():
    cases = [{"name": "fixture", "r": {"missing": 1.0, "lost": 2.0, "wrong": 3.0},
              "survstudio": {"lost": np.nan, "wrong": 4.0}}]
    failures = agreement.agreement_failures(cases, atol=1e-6, rtol=1e-7)
    assert [failure["quantity"] for failure in failures] == ["missing", "lost", "wrong"]
    assert agreement.compare(cases[0]["r"], cases[0]["survstudio"])["missing"] is None


def test_empty_reference_runs_cannot_pass():
    assert agreement.agreement_failures([], atol=1e-6, rtol=1e-7)
    assert agreement.agreement_failures([{"name": "empty", "r": {}, "survstudio": {}}], atol=1e-6, rtol=1e-7)

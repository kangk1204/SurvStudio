"""An adoption report must not pass through incomplete or mixed evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "validation/publication_v3"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    # Existing study tests may install isolated protocol modules in sys.modules.
    for name in ("study", "control"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("confirmation_grid_audit", DIRECTORY / "confirmation_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(gate):
    cfg = gate.protocol()
    freeze = dict(selection_eligible=True, reference_passed=True, candidate=cfg["candidates"][0],
                  source_hashes={"source.py": "a" * 64}, environment={"python": "fixture"},
                  _input_file_sha256="b" * 64)
    summaries = []
    for stage in ("main", "extension", "large", "stress"):
        rows = []
        planned = 0
        for cell in gate.cells(stage):
            for condition in cell["conditions"]:
                n = cell["replicates"]
                planned += n
                for method in cfg["methods"]:
                    allowed = n
                    rows.append(dict(stage=stage, condition=condition, n=cell["n"], p=cell["p"], method=method,
                        planned=n, completed=n, missing=0, failures=0, diagnostic_failures=0, allowed=allowed,
                        allowed_fraction=1., withhold_fraction=0., allowed_lower95=gate.binomial(n,n,"lower"),
                        fwer=0., fwer_lower95=0., fwer_upper95=gate.binomial(0,n,"upper"),
                        conditional_fwer=0., conditional_lower95=0., conditional_upper95=gate.binomial(0,n,"upper"),
                        power_ratio=None))
        configuration = {k: deepcopy(freeze[k]) for k in ("candidate", "source_hashes", "environment")}
        configuration.update(stage=stage, seed=cfg[("extension" if stage == "large" else stage) + "_seed"],
                             freeze_sha256=freeze["_input_file_sha256"])
        summaries.append(dict(stage=stage, complete=True, planned_datasets=planned, completed_datasets=planned,
                              methods=cfg["methods"], configuration=configuration, summaries=rows))
    return summaries, freeze


def test_exact_complete_grid(gate):
    summaries, freeze = evidence(gate)
    result = gate.audit(summaries, freeze)
    assert result["dataset_cells"] == 55 and result["method_records"] == 747500
    assert result["production_promotion"] is False


@pytest.mark.parametrize("defect", ["missing_cell", "duplicate_cell", "wrong_repeats", "wrong_seed",
                                   "wrong_source", "wrong_seal", "wrong_methods", "missing_stage",
                                   "nonfinite", "zero_with_value", "fraction_mismatch", "unknown_cell",
                                   "forged_bound", "nonfinite_power"])
def test_incomplete_mixed_and_invalid_grid_rejected(gate, defect):
    summaries, freeze = evidence(gate)
    row = summaries[0]["summaries"][0]
    if defect == "missing_cell": summaries[0]["summaries"].pop()
    elif defect == "duplicate_cell": summaries[0]["summaries"].append(deepcopy(row))
    elif defect == "wrong_repeats": row["planned"] -= 1
    elif defect == "wrong_seed": summaries[0]["configuration"]["seed"] += 1
    elif defect == "wrong_source": summaries[0]["configuration"]["source_hashes"]["source.py"] = "c" * 64
    elif defect == "wrong_seal": summaries[0]["configuration"]["freeze_sha256"] = "c" * 64
    elif defect == "wrong_methods": summaries[0]["methods"] = summaries[0]["methods"][:-1]
    elif defect == "missing_stage": summaries.pop()
    elif defect == "nonfinite": row["fwer_upper95"] = float("nan")
    elif defect == "zero_with_value": row.update(allowed=0, allowed_fraction=0., withhold_fraction=1.)
    elif defect == "fraction_mismatch": row["allowed_fraction"] = .99
    elif defect == "unknown_cell": row["condition"] = "favorable_unplanned"
    elif defect == "forged_bound": row["fwer_upper95"] = 0.
    elif defect == "nonfinite_power":
        row=next(r for r in summaries[0]["summaries"] if r["condition"]=="partial_weak")
        row["power_ratio"]=dict(point=1.,lower95=float("nan"),mc95=[.9,1.1],undefined_denominator_draws=0)
    with pytest.raises(ValueError): gate.audit(summaries, freeze)


def test_zero_allowed_is_unestimable_not_zero_error(gate):
    summaries, freeze = evidence(gate)
    row = summaries[0]["summaries"][0]
    row.update(allowed=0, allowed_fraction=0., withhold_fraction=1., allowed_lower95=0.,
               conditional_fwer=None, conditional_upper95=None, conditional_lower95=None)
    assert gate.audit(summaries, freeze)["passed"] is True

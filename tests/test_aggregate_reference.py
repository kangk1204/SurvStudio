"""Independent aggregate verification retains unresolved and zero-denominator cases."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1]/"validation/publication_v3/verify_aggregate_reference.py"
    spec = importlib.util.spec_from_file_location("aggregate_reference_fixture", path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_stream_preserves_default_numpy_integer_draws(tmp_path):
    reference = module(); seed = [2026100327, 180, 30, 10, 3]
    path = tmp_path/"stream.i32"
    manifest = reference.row_stream(path, 7, np.random.SeedSequence(seed))
    rng = np.random.default_rng(np.random.SeedSequence(seed)); expected = []
    for start in range(0, 9999, 128):
        expected.append(rng.integers(0, 7, size=(min(128, 9999-start), 7)))
    actual = np.fromfile(path, dtype="<i4").reshape(9999, 7)
    np.testing.assert_array_equal(actual, np.concatenate(expected))
    assert manifest["bytes"] == 9999*7*4 and manifest["sha256"] == reference.digest(path)
    with pytest.raises(FileExistsError): reference.row_stream(path, 7, np.random.SeedSequence(seed))


@pytest.mark.parametrize("actual,expected", [(0, None), (None, 0), (False, 0.), ({}, {"power":None}), ([0], [None])])
def test_comparison_never_conflates_null_zero_or_missing_fields(actual, expected):
    assert not all(c["passed"] for c in module().compare(actual, expected))


def test_raw_fixture_covers_missing_failures_and_zero_denominators(tmp_path):
    reference = module(); paths, summary_path = reference.self_test(tmp_path)
    summary = json.loads(summary_path.read_text()); records, configuration, sources = reference.read_ledgers(paths)
    assert summary["planned_datasets"] == 45 and summary["completed_datasets"] == 33
    assert summary["complete"] is False and configuration["freeze_sha256"] == "synthetic-not-a-seal"
    assert sources[str(paths[0].resolve())] == reference.digest(paths[0])
    rows = summary["summaries"]
    assert any(r["failures"] for r in rows) and any(r["diagnostic_failures"] for r in rows)
    assert any(r["conditional_fwer"] is None for r in rows)
    assert any(r["power_ratio"] and r["power_ratio"]["undefined_denominator_draws"] > 0 for r in rows)
    assert all(r["power_ratio"] is None for r in rows if r["p"] == 4)
    assert all(r["completed"] == 0 and r["paired_difference_from_legacy"]["complete_pairs"] == 0 for r in rows if r["p"] == 5)
    with pytest.raises(ValueError, match="Duplicate"): reference.read_ledgers(paths+paths)
    cell = dict(n=40, p=3, replicates=9)
    records[("independent", 40, 3, 0)][0][0]["allowed"] = 1
    with pytest.raises(ValueError, match="booleans"):
        reference.raw_cell(records, cell, "independent", "v3_linear", reference.study.protocol()["methods"])


def test_wrong_owner_is_rejected(tmp_path):
    reference = module(); paths, _ = reference.self_test(tmp_path)
    with sqlite3.connect(paths[0]) as db:
        cfg = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0]); cfg.update(owners=2, owner=1)
        db.execute("UPDATE metadata SET value=?", (json.dumps(cfg),))
    with pytest.raises(ValueError, match="wrong owner"): reference.read_ledgers(paths)

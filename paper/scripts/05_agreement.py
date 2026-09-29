"""Table 2: the largest difference from R survival per analysis, from SurvStudio's numerical agreement report
(validation/agreement/run_agreement.py writes docs/validation/numerical_agreement.json).

The report is read from the SurvStudio checkout that holds this folder (or from SURVSTUDIO_REPO, which run_all.sh
and run_step.sh set to SURVSTUDIO_SRC). Writes agreement_summary.csv and, in agreement_summary.json, where the report
came from: its date and environment as it records them, its SHA-256, the last SurvStudio commit that changed it (and
whether it has changed since), and the SurvStudio commit read now.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from common import RESULTS, sha256_file, survstudio_version, write_csv_atomic, write_json

REPOSITORY = Path(os.environ.get("SURVSTUDIO_REPO") or Path(__file__).resolve().parents[2])
REPORT = Path("docs") / "validation" / "numerical_agreement.json"


def git(*arguments: str) -> str | None:
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(REPOSITORY), *arguments],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


data = json.loads((REPOSITORY / REPORT).read_text(encoding="utf-8"))
rows = []
for case in data["cases"]:
    reference = {key: value for key, value in case["r"].items() if value is not None}
    row = {"analysis": case["name"], "quantities": len(reference)}
    for package in ("survstudio", "lifelines", "sksurv"):
        differences = [value for value in (case["differences"].get(package) or {}).values() if value is not None]
        row[f"{package}_max_difference"] = max(differences) if differences else None
    rows.append(row)
table = pd.DataFrame(rows)
write_csv_atomic(table, RESULTS / "agreement_summary.csv")
status = git("status", "--porcelain", "--", REPORT.as_posix())
write_json(RESULTS / "agreement_summary.json", {
    "report": REPORT.as_posix(),
    "report_sha256": sha256_file(REPOSITORY / REPORT),
    "report_generated": (data.get("environment") or {}).get("date"),
    "report_environment": data.get("environment"),
    "report_last_commit": git("log", "-1", "--format=%h %cs", "--", REPORT.as_posix()),
    "report_changed_since_commit": None if status is None else bool(status),
    "survstudio": survstudio_version(),
})
print(table.to_string(index=False))

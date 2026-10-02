"""Version 2 confirmation using the unchanged v1 DGP, score, ownership and summary code.

All specification differences are explicit in confirmation_protocol.json. The v1
files and evidence remain intact. The offline F candidate is excluded.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from survival_toolkit.marker_diagnostics import METHOD_VERSION
from development import exception_context

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("confirmation_protocol.json")
spec = importlib.util.spec_from_file_location("_original_guarded_study", ROOT / "validation/guarded_inference/study.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)
kernel.PROTOCOL = PROTOCOL
kernel.SOURCE_FILES = [*kernel.SOURCE_FILES, "validation/diagnostic_followup/confirmation.py",
                       "validation/diagnostic_followup/confirmation_protocol.json",
                       "validation/diagnostic_followup/launch_confirmation.py",
                       "validation/diagnostic_followup/freeze_confirmation.py",
                       "validation/diagnostic_followup/development.py", "validation/diagnostic_followup/protocol.json"]
hashes = kernel.hashes
verify_freeze = kernel.verify_freeze
original_calculation = kernel.calculation


def calculation_with_context(*args, **kwargs):
    try:
        return original_calculation(*args, **kwargs)
    except Exception as exc:
        # Preserve the exception and failed index, adding metadata before its
        # live traceback is discarded by the original append-only runner.
        exc.add_note("Synthetic array context: " + json.dumps(exception_context(exc)["frames"]))
        raise


kernel.calculation = calculation_with_context


if __name__ == "__main__":
    if METHOD_VERSION != "marker-inference/2":
        raise SystemExit("Only the reviewed v2 method can execute this new confirmation")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("run")
    worker.add_argument("--stage", choices=["main", "extension", "large"], required=True)
    worker.add_argument("--condition", required=True)
    worker.add_argument("--n", type=int, default=180)
    worker.add_argument("--p", type=int, default=30)
    worker.add_argument("--permutations", type=int, default=999)
    worker.add_argument("--start", type=int, default=0)
    worker.add_argument("--stop", type=int, required=True)
    worker.add_argument("--workers", type=int, default=1)
    worker.add_argument("--worker", type=int, default=0)
    worker.add_argument("--output", required=True)
    worker.add_argument("--freeze", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("ledgers", nargs="+")
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        kernel.run(args)
    else:
        output = Path(args.output)
        if output.exists():
            raise ValueError("Do not overwrite a v2 confirmation summary")
        kernel.summary(args.ledgers, output)
        result = json.loads(output.read_text())
        if not result["complete"]:
            result["model_release_status"] = {method: "pending" for method in result["model_release_status"]}
        result["note"] = "Complete fixed-index v2 confirmation" if result["complete"] else "Incomplete v2 confirmation; no qualification decision"
        output.write_text(json.dumps(result, indent=2) + "\n")

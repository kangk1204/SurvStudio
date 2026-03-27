from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import uvicorn

from survival_toolkit.analysis import load_dataframe_from_path, profile_dataframe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="survival-toolkit")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI app with Uvicorn.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Load a local file path and print a JSON dataset profile.",
    )
    inspect_parser.add_argument("path", help="Path to a CSV, TSV, XLSX, XLS, or Parquet file.")

    return parser


def _run_inspect(path: str) -> int:
    path_obj = Path(path)
    dataframe = load_dataframe_from_path(path_obj)
    profile = profile_dataframe(dataframe, dataset_id="cli_inspect", filename=path_obj.name)
    print(json.dumps(profile, indent=2, sort_keys=True, default=str))
    return 0


def _run_serve(host: str, port: int, reload: bool) -> int:
    uvicorn.run(
        "survival_toolkit.app:app",
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "inspect":
        try:
            return _run_inspect(args.path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.command in {None, "serve"}:
        return _run_serve(
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8000),
            reload=bool(getattr(args, "reload", False)),
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

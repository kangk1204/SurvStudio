from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import uvicorn

from survival_toolkit.analysis import load_dataframe_from_path, profile_dataframe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="survstudio")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI app with Uvicorn.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        dest="allowed_hosts",
        metavar="HOST",
        help=(
            "Additional Host header value the server should answer (repeatable), for example a LAN name or "
            "reverse-proxy host. Localhost and the --host bind address are always allowed."
        ),
    )

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


_BIND_HOST_ENV_VAR = "SURVSTUDIO_BIND_HOST"
_ALLOWED_HOSTS_ENV_VAR = "SURVSTUDIO_ALLOWED_HOSTS"


def _configure_request_host_guard(host: str, allowed_hosts: Sequence[str] = ()) -> None:
    """Expose the bind host (and extra allowed hosts) to the app's Host/Origin guard.

    Environment variables are used so the setting also reaches `--reload` worker processes.
    """

    os.environ[_BIND_HOST_ENV_VAR] = str(host)
    extra_hosts = [str(item).strip() for item in allowed_hosts if str(item).strip()]
    if extra_hosts:
        existing = [item.strip() for item in os.environ.get(_ALLOWED_HOSTS_ENV_VAR, "").split(",") if item.strip()]
        os.environ[_ALLOWED_HOSTS_ENV_VAR] = ",".join(dict.fromkeys([*existing, *extra_hosts]))


_LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# Set by the Docker image, which has to listen on every interface of the container.
_CONTAINER_ENV_VAR = "SURVSTUDIO_CONTAINER"


def _warn_if_reachable_from_network(host: str) -> None:
    """SurvStudio has no login, so binding beyond loopback exposes it to the network."""

    normalized = str(host).strip().strip("[]").lower()
    if normalized in _LOOPBACK_BIND_HOSTS:
        return
    if os.environ.get(_CONTAINER_ENV_VAR) == "1":
        print(
            f"SurvStudio is listening on {host} inside its container. It has no login, so publish the port on "
            "127.0.0.1 only (docker run -p 127.0.0.1:8000:8000 ...) and open http://localhost:8000.",
            file=sys.stderr,
        )
        return
    print(
        f"WARNING: SurvStudio is listening on {host}. It has no login: anyone who can reach this address can "
        "upload data, run analyses (including long model training), and open datasets whose IDs they know. "
        "Use 127.0.0.1 unless every machine on this network is trusted.",
        file=sys.stderr,
    )


def _run_serve(host: str, port: int, reload: bool, allowed_hosts: Sequence[str] = ()) -> int:
    _configure_request_host_guard(host, allowed_hosts)
    _warn_if_reachable_from_network(host)
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
            allowed_hosts=list(getattr(args, "allowed_hosts", []) or []),
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

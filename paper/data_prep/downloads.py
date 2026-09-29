"""Download public files once and record where each came from."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def fetch(url: str, target: Path) -> Path:
    """Download url to target unless it is already there, and record the URL, the download
    date and the SHA-256 in manifest.json next to the file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        partial.replace(target)
    manifest_path = target.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not all(isinstance(entry, dict) and "sha256" in entry for entry in manifest.values()):
        manifest = {}
    if target.name not in manifest:
        downloaded = dt.date.fromtimestamp(target.stat().st_mtime).isoformat()
        manifest[target.name] = {"url": url, "downloaded": downloaded, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target

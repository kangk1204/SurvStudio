"""Survival analysis toolkit package."""

import pandas as pd

__all__ = ["__version__"]

# Single source of the package version (pyproject.toml reads it via tool.setuptools.dynamic).
__version__ = "0.2.0"

# Prefer copy-on-write semantics for defensive DatasetStore snapshots without
# paying the full deep-copy cost on every access.
try:
    _pandas_major = int(str(pd.__version__).split(".", maxsplit=1)[0])
except (TypeError, ValueError):
    _pandas_major = 0

if _pandas_major < 3:
    pd.options.mode.copy_on_write = True

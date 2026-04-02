"""Survival analysis toolkit package."""

import pandas as pd

__all__ = ["__version__"]

__version__ = "0.1.0"

# Prefer copy-on-write semantics for defensive DatasetStore snapshots without
# paying the full deep-copy cost on every access.
pd.options.mode.copy_on_write = True

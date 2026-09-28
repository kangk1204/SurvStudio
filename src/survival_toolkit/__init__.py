"""Survival analysis toolkit package.

SurvStudio is written for pandas copy-on-write, the only mode in pandas 3. On pandas 2.x, importing this
package therefore switches copy-on-write on for the whole Python session
(``pd.options.mode.copy_on_write = True``): chained assignment such as ``df["a"][0] = 1`` then no longer
changes ``df``. On pandas 3 nothing changes. To keep pandas 2.x defaults in your own code, run SurvStudio in
a separate session or upgrade to pandas 3.
"""

import pandas as pd

__all__ = ["__version__"]

# Single source of the package version (pyproject.toml reads it via tool.setuptools.dynamic).
__version__ = "0.2.0"

# The dataset store hands out defensive snapshots and the analysis code derives frames from them; both
# rely on copy-on-write semantics (no writes through views, lazy copies), which pandas 3 always uses.
try:
    _pandas_major = int(str(pd.__version__).split(".", maxsplit=1)[0])
except (TypeError, ValueError):
    _pandas_major = 0

if _pandas_major < 3:
    pd.options.mode.copy_on_write = True

#!/usr/bin/env bash
# Run one analysis script (with arguments) the way run_all.sh does: run_step.sh <script.py> [args...]
# SURVSTUDIO_SRC, ANALYSIS_PYTHON and SURVSTUDIO_DATA as in run_all.sh.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SURVSTUDIO_SRC="${SURVSTUDIO_SRC:-$(cd "$here/.." && pwd)}"
if [ -z "${ANALYSIS_PYTHON:-}" ]; then
  if [ -x "$SURVSTUDIO_SRC/.venv/bin/python" ]; then ANALYSIS_PYTHON="$SURVSTUDIO_SRC/.venv/bin/python"; else ANALYSIS_PYTHON=python3; fi
fi
# The SurvStudio commit, as run_all.sh stamps it (the redrawn figures do not count as changes).
git=(git -c safe.directory='*' --no-optional-locks -C "$SURVSTUDIO_SRC")
if [ "$("${git[@]}" rev-parse --show-toplevel 2>/dev/null)" = "$(cd "$SURVSTUDIO_SRC" && pwd -P)" ] &&
   SURVSTUDIO_COMMIT="$("${git[@]}" describe --always 2>/dev/null)"; then
  [ -z "$("${git[@]}" status --porcelain --untracked-files=no -- . ':(exclude)paper/figures' 2>/dev/null)" ] || SURVSTUDIO_COMMIT="$SURVSTUDIO_COMMIT-dirty"
else
  SURVSTUDIO_COMMIT=unknown
fi
export SURVSTUDIO_COMMIT
case "$SURVSTUDIO_COMMIT" in
  unknown|*-dirty) echo "warning: SurvStudio in $SURVSTUDIO_SRC is at '$SURVSTUDIO_COMMIT'; the results cannot be tied to a commit" >&2 ;;
esac
export SURVSTUDIO_REPO="$SURVSTUDIO_SRC"
export PYTHONPATH="$SURVSTUDIO_SRC/src"
cd "$here/scripts"
exec "$ANALYSIS_PYTHON" "$@"

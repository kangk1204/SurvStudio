#!/usr/bin/env bash
# Reproduce every number and figure of the software paper.
#   bash run_all.sh           every step (00 to 17 but the duplicate audits 11 and 12), then every figure
#   bash run_all.sh 03 04     the selected steps, then only the figures drawn from their results
# SURVSTUDIO_SRC: the SurvStudio checkout whose src/ is imported, by default the one holding this folder; its commit
# (`git describe --always`, "-dirty" marking uncommitted changes to anything but paper/figures) is recorded in the
# results. ANALYSIS_PYTHON: an environment built from requirements-lock.txt, by default $SURVSTUDIO_SRC/.venv (else
# python3); FIGURE_PYTHON (matplotlib) defaults to the same. SURVSTUDIO_DATA: the data folder, by default paper/data
# (written by prepare_data.sh). SIM_WORKERS: processes for the simulation (06), default 8; SEED_WORKERS: processes for
# the seed variability (16), default 4. figures.py draws a figure only from results of one run (see README.md), so after
# a partial rerun, rerun the steps it names. The comparison with other pipelines (scripts 18) needs its own R
# environment and runs with run_competitors.sh, after this script.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SURVSTUDIO_SRC="${SURVSTUDIO_SRC:-$(cd "$here/.." && pwd)}"
if [ -z "${ANALYSIS_PYTHON:-}" ]; then
  if [ -x "$SURVSTUDIO_SRC/.venv/bin/python" ]; then ANALYSIS_PYTHON="$SURVSTUDIO_SRC/.venv/bin/python"; else ANALYSIS_PYTHON=python3; fi
fi
FIGURE_PYTHON="${FIGURE_PYTHON:-$ANALYSIS_PYTHON}"
SIM_WORKERS="${SIM_WORKERS:-8}"
SEED_WORKERS="${SEED_WORKERS:-4}"
steps=("$@")
# The duplicate audits (11, 12) are run on their own; the analyses read the committed breast_duplicate_pairs.csv.
[ ${#steps[@]} -eq 0 ] && steps=(00 01 02 03 04 05 06 07 08 09 10 13 14 15 15b 16 17)
# The figures each step's results feed (figures.py names); 00, 05, 13, 15b, 16 and 17 feed none.
figures=()
for step in "${steps[@]}"; do
  case "$step" in
    01) figures+=(markers estimates luad_external) ;;
    02) figures+=(markers) ;;
    03) figures+=(estimates luad_external) ;;
    04) figures+=(models) ;;
    06) figures+=(simulation) ;;
    07|08) figures+=(estimates breast_external) ;;
    09) figures+=(estimates breast_er_external) ;;
    10) figures+=(estimates breast_er_external breast_er_sensitivity) ;;
    14) figures+=(estimates breast_er_sensitivity) ;;
    15) figures+=(tiers) ;;
  esac
done
# Fail before any analysis runs, not after it, when the figures cannot be drawn.
if { [ $# -eq 0 ] || [ ${#figures[@]} -gt 0 ]; } && ! "$FIGURE_PYTHON" -c "import matplotlib, numpy, pandas" >/dev/null 2>&1; then
  echo "error: FIGURE_PYTHON ($FIGURE_PYTHON) cannot import matplotlib, numpy and pandas. Point FIGURE_PYTHON (or" >&2
  echo "ANALYSIS_PYTHON) at an environment built from paper/requirements-lock.txt, which includes matplotlib" >&2
  echo "(see README.md)." >&2
  exit 1
fi
# The SurvStudio commit the results are stamped with. The figures are left out of the "-dirty" check: this script
# redraws them, and a rerun after it must not count its own figures as changes to the code.
git=(git -c safe.directory='*' --no-optional-locks -C "$SURVSTUDIO_SRC")
if SURVSTUDIO_COMMIT="$("${git[@]}" describe --always 2>/dev/null)"; then
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
for step in "${steps[@]}"; do
  for script in "${step}"_*.py; do
    echo "== $script (SurvStudio $SURVSTUDIO_COMMIT)"
    if [ "$script" = 06_simulation.py ]; then
      "$ANALYSIS_PYTHON" "$script" "$SIM_WORKERS"
    elif [ "$script" = 16_seed_variability.py ]; then
      "$ANALYSIS_PYTHON" "$script" "$SEED_WORKERS"
    else
      "$ANALYSIS_PYTHON" "$script"
    fi
  done
done
if [ $# -eq 0 ]; then
  "$FIGURE_PYTHON" figures.py
elif [ ${#figures[@]} -gt 0 ]; then
  "$FIGURE_PYTHON" figures.py --available $(printf '%s\n' "${figures[@]}" | sort -u)
fi

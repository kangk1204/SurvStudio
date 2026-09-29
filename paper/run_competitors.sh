#!/usr/bin/env bash
# The comparison with the pipelines commonly used to publish prognostic signatures (see competitors/README.md): P1 Mime,
# P2 univariate Cox -> LASSO -> Cox and P3 KM Plotter's best cut-off, developed on TCGA-LUAD and validated in the seven
# GEO cohorts of case study II (experiment 1), and their claims under script 06's plasmode null (experiment 2).
#   bash run_competitors.sh                      every step: checks data real sensitivity null table
#   bash run_competitors.sh real table           selected steps
# Run it after run_all.sh (it reads case studies I and II and scripts 06 and 15 from results/) and after
# competitors/setup_r_env.sh (the R environment). SURVSTUDIO_SRC, ANALYSIS_PYTHON and SURVSTUDIO_DATA as in
# run_all.sh, and:
#   RSCRIPT          the competitors' Rscript (default competitors/renv/bin/Rscript)
#   WORKERS          parallel processes for the null replicates (default 8)
#   MIME_CORES       cores of the real-data Mime runs (default 6)
#   NULL_REPLICATES  replicates of P2 and P3 (default 200); MIME_REPLICATES, those of P1 (default 50)
# Every R run writes its log next to its results under results/competitors/. A null replicate that finished is not
# run again, so the null step resumes after an interruption.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
RSCRIPT="${RSCRIPT:-$here/competitors/renv/bin/Rscript}"
WORKERS="${WORKERS:-8}"
MIME_CORES="${MIME_CORES:-6}"
NULL_REPLICATES="${NULL_REPLICATES:-200}"
MIME_REPLICATES="${MIME_REPLICATES:-50}"
# The candidate caps of competitors.py (CANDIDATE_CAP, SENSITIVITY_CAP).
MIME_CAP=100
SENSITIVITY_CAP=500
export RSCRIPT MIME_CAP OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(checks data real sensitivity null table)
work="$here/results/competitors"
cohorts=TCGA,GSE13213,GSE30219,GSE31210,GSE41271,GSE50081,GSE68465,GSE72094
python_step() { bash "$here/run_step.sh" "$@"; }
r_real() {  # r_real <script> <output folder name> [arguments...]
  local script="$1" name="$2"
  shift 2
  mkdir -p "$work/real"
  "$RSCRIPT" "$here/competitors/$script" source=real input="$work/input" cohorts="$cohorts" out="$work/real/$name" "$@" > "$work/real/$name.log" 2>&1
}
for step in "${steps[@]}"; do
  echo "== $step $(date -Is)"
  case "$step" in
    checks) python_step 17_competitors_checks.py ;;
    data) python_step 17_competitors_data.py ;;
    real)
      r_real run_p2.R p2
      r_real run_mime.R "mime_cap$MIME_CAP" cap="$MIME_CAP" cores="$MIME_CORES"
      python_step 17_competitors_real.py ;;
    sensitivity)
      # 500 candidates: every model whose first algorithm is not StepCox, through Mime's single and double modes.
      r_real run_mime.R "mime_cap$SENSITIVITY_CAP" cap="$SENSITIVITY_CAP" plan=feasible cores="$MIME_CORES"
      python_step 17_competitors_real.py ;;
    null)
      python_step 17_competitors_null.py generate "$NULL_REPLICATES"
      seq 0 $((NULL_REPLICATES - 1)) | xargs -P "$WORKERS" -I{} bash "$here/competitors/null_replicate.sh" p2 {}
      seq 0 $((MIME_REPLICATES - 1)) | xargs -P "$WORKERS" -I{} bash "$here/competitors/null_replicate.sh" mime {}
      python_step 17_competitors_null.py p3 "$WORKERS"
      python_step 17_competitors_null.py summarise ;;
    table) python_step 17_competitors_table.py ;;
    *) echo "unknown step $step (checks data real sensitivity null table)" >&2; exit 1 ;;
  esac
done

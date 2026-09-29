#!/usr/bin/env bash
# One R pipeline on one null replicate of 17_competitors_null.py (run_competitors.sh runs many in parallel):
#   null_replicate.sh p2|mime <replicate number>
# A replicate whose run.json exists is finished and skipped; a failed run leaves no run.json and is reported.
# RSCRIPT: the competitors' R; MIME_CAP: Mime's candidates (default 100).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
pipeline="$1"
replicate="$(printf '%04d' "$2")"
null="$here/../results/competitors/null"
folder="$null/rep_$replicate"
[ -f "$folder/$pipeline/run.json" ] && exit 0
case "$pipeline" in
  p2) arguments=("$here/run_p2.R") ;;
  mime) arguments=("$here/run_mime.R" "cap=${MIME_CAP:-100}" cores=1) ;;
  *) echo "unknown pipeline $pipeline" >&2; exit 1 ;;
esac
if ! "${RSCRIPT:?}" "${arguments[@]}" source=null input="$folder/design.csv" expression="$null/tcga_expression.csv" out="$folder/$pipeline" \
    > "$folder/$pipeline.log" 2>&1; then
  echo "replicate $2: $pipeline failed (see $folder/$pipeline.log)" >&2
fi
exit 0

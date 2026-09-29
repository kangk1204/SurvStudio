#!/usr/bin/env bash
# Build the data the paper's scripts read, from public downloads and the three files of data_snapshot/:
#   bash prepare_data.sh
# The data go to SURVSTUDIO_DATA, by default paper/data (about 4.5 GB with the downloads and the ExperimentHub cache).
# ANALYSIS_PYTHON: the environment of run_all.sh (by default $SURVSTUDIO_SRC/.venv, else python3). RSCRIPT: R 4.5.2
# with the packages of data_prep/install_packages.R (default Rscript). Downloads already there are not fetched again.
# At the end the data are checked against luad_data_manifest.csv and breast_data_manifest.csv.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SURVSTUDIO_SRC="${SURVSTUDIO_SRC:-$(cd "$here/.." && pwd)}"
if [ -z "${ANALYSIS_PYTHON:-}" ]; then
  if [ -x "$SURVSTUDIO_SRC/.venv/bin/python" ]; then ANALYSIS_PYTHON="$SURVSTUDIO_SRC/.venv/bin/python"; else ANALYSIS_PYTHON=python3; fi
fi
RSCRIPT="${RSCRIPT:-Rscript}"
export SURVSTUDIO_DATA="${SURVSTUDIO_DATA:-$here/data}"
data="$SURVSTUDIO_DATA"
prep="$here/data_prep"
snapshot="$here/data_snapshot"
qc="$data/cohorts/luad/qc"
step() { printf '\n== %s\n' "$*"; }

step "Checking the tools and the snapshot files (data_snapshot/SNAPSHOT.md) against their SHA-256"
(cd "$here/scripts" && "$ANALYSIS_PYTHON" -c 'import common; print("  ok:", ", ".join(common.check_snapshot()))')
"$RSCRIPT" -e 'for (p in c("GEOquery", "MetaGxBreast", "data.table")) if (!requireNamespace(p, quietly = TRUE))
  stop(p, " is not installed: run Rscript data_prep/install_packages.R"); cat("  R", as.character(getRversion()), "\n")'
mkdir -p "$data"
echo "  data folder: $data"

step "1/7 TCGA-LUAD from UCSC Xena: HiSeqV2.gz, the clinical matrix and the TCGA-CDR survival table"
"$ANALYSIS_PYTHON" "$prep/export_tcga_luad.py" "$data"

step "2/7 The seven GEO series (GSE13213, GSE30219, GSE31210, GSE41271, GSE50081, GSE68465, GSE72094) with GEOquery"
"$RSCRIPT" "$prep/export_geo_luad.R" "$data"

step "3/7 The QC decisions the paper applied: data_snapshot/sample_decisions.csv -> cohorts/luad/qc/"
mkdir -p "$qc"
cp "$snapshot/sample_decisions.csv" "$qc/sample_decisions.csv"

step "4/7 The harmonized LUAD clinical table (outcomes as recorded, QC decisions applied): cohorts/luad/harmonized_clinical.csv"
"$ANALYSIS_PYTHON" "$prep/harmonize_luad.py" "$data"

step "5/7 Gene-level expression with the NCBI gene_info snapshot: cohorts/luad/<cohort>/expression_genes.csv.gz"
"$ANALYSIS_PYTHON" "$prep/harmonize_expression.py" "$data"

step "6/7 The QC rerun (downloads the GDC PanCanAtlas sample-quality annotations), compared with the decisions the paper applied"
"$ANALYSIS_PYTHON" "$prep/qc_cohorts.py" "$data" luad > "$qc/qc_cohorts.log"
echo "  report: $qc/report.md (log: $qc/qc_cohorts.log)"
if cmp -s "$qc/sample_decisions.csv" "$snapshot/sample_decisions.csv"; then
  echo "  the rerun reproduces the decisions the paper applied"
else
  mv "$qc/sample_decisions.csv" "$qc/sample_decisions.rerun.csv"
  cp "$snapshot/sample_decisions.csv" "$qc/sample_decisions.csv"
  echo "warning: the QC rerun decides differently (kept as $qc/sample_decisions.rerun.csv); the harmonized table keeps" >&2
  echo "the decisions the paper applied (data_snapshot/sample_decisions.csv)." >&2
fi

step "7/7 The 39 MetaGxBreast cohorts from Bioconductor ExperimentHub: cohorts/breast/"
"$RSCRIPT" "$prep/export_curated_cohorts.R" "$data" breast

step "The downloads against the ones the paper used (data_prep/downloads.sha256; a difference is reported, not fatal)"
if ! (cd "$data" && sha256sum -c "$prep/downloads.sha256"); then
  echo "warning: a download differs from the one the paper used; the checks below tell whether the analysis inputs changed" >&2
fi

step "The analysis inputs: cBioPortal's METABRIC table (the snapshot copy when a download differs), the LUAD and breast data against their manifests"
(cd "$here/scripts" && "$ANALYSIS_PYTHON" - <<'EOF'
import common

common.metabric_cbioportal()
print(f"  cBioPortal METABRIC table: {common.BREAST / 'METABRIC' / 'cbioportal_patients.csv'} is the copy the paper used")
print(f"  LUAD data: {common.check_luad_manifest()} files match {common.LUAD_MANIFEST.name} (the .gz files by their decompressed content)")
print(f"  breast data: {common.check_breast_manifest()} files match {common.BREAST_MANIFEST.name}")
EOF
)
echo
echo "Data ready in $data. Next: bash $here/run_all.sh"

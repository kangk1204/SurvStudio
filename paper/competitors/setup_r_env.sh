#!/usr/bin/env bash
# The R environment of the competitor pipelines (Mime, uni-Cox -> LASSO -> Cox): a conda environment with R 4.4 and
# the packages conda-forge and bioconda carry, then plsRcox, compareC, forestploter and snowfall from a dated CRAN
# snapshot, CoxBoost and Mime from git at the pinned commits (install_packages.R), and the versions it ended up with
# (versions.R). Nothing is installed outside the two folders below.
#   COMPETITORS_R_ENV  the environment's prefix (default paper/competitors/renv)
#   COMPETITORS_SRC    where CoxBoost and Mime are cloned (default paper/competitors/src)
#   MAMBA              mamba or conda (default: mamba on the PATH, else ~/miniforge3/bin/mamba)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
RENV="${COMPETITORS_R_ENV:-$here/renv}"
SRC="${COMPETITORS_SRC:-$here/src}"
MAMBA="${MAMBA:-$(command -v mamba || echo "$HOME/miniforge3/bin/mamba")}"
# Mime (l-magnificence/Mime, package Mime1): the commit of 2025-09-23, the latest when the comparison was run.
MIME_REPOSITORY=https://github.com/l-magnificence/Mime.git
MIME_COMMIT=9a9f6ac89851bf631f9df3868b2fa624bed49df2
# CoxBoost, which Mime's README installs from GitHub (binderh/CoxBoost): the commit of 2026-09-24.
COXBOOST_REPOSITORY=https://github.com/binderh/CoxBoost.git
COXBOOST_COMMIT=50d5fff7d7eebc4a1174dfa32adcfb4b4a69ad13

if [ ! -x "$RENV/bin/Rscript" ]; then
  "$MAMBA" create -y -p "$RENV" -c conda-forge -c bioconda --strict-channel-priority \
    r-base=4.4 r-survival r-randomforestsrc r-glmnet r-gbm r-dplyr r-tibble r-tidyr r-ggplot2 r-data.table r-matrix \
    r-caret r-e1071 r-doparallel r-future r-ggpubr r-gridextra r-magrittr r-meta r-proc r-readr r-recipes r-stringr \
    r-tidyverse r-pbapply r-reshape2 r-upsetr r-scales r-hmisc r-rocr r-remotes r-biocmanager r-jsonlite r-viridis \
    r-kknn r-mixtools r-compositions r-survivalroc r-bart r-misctools r-ggsci r-superpc r-survivalsvm \
    r-ggbreak r-rocit r-aplot r-ckmeans.1d.dp r-boruta r-snowfall \
    bioconductor-complexheatmap bioconductor-mixomics compilers make
fi
mkdir -p "$SRC"
clone() {  # clone <repository> <folder> <commit>
  [ -d "$SRC/$2/.git" ] || git clone -q "$1" "$SRC/$2"
  git -C "$SRC/$2" fetch -q origin
  git -C "$SRC/$2" checkout -q "$3"
  [ "$(git -C "$SRC/$2" rev-parse HEAD)" = "$3" ] || { echo "error: $2 is not at $3" >&2; exit 1; }
}
clone "$MIME_REPOSITORY" Mime "$MIME_COMMIT"
clone "$COXBOOST_REPOSITORY" CoxBoost "$COXBOOST_COMMIT"
# R CMD INSTALL finds the environment's compilers (x86_64-conda-linux-gnu-cc, ...) on the PATH.
export PATH="$RENV/bin:$PATH"
"$RENV/bin/Rscript" "$here/install_packages.R" "$SRC/CoxBoost" "$SRC/Mime"
# The environment's own path is written as "renv" in both records.
MIME_COMMIT="$MIME_COMMIT" COXBOOST_COMMIT="$COXBOOST_COMMIT" "$RENV/bin/Rscript" "$here/versions.R" | sed "s#$RENV#renv#g" > "$here/versions.txt"
"$MAMBA" list -p "$RENV" --explicit | sed "s#$RENV#renv#g" > "$here/renv-conda-explicit.txt"
echo "R environment ready: $RENV (versions in $here/versions.txt)"

# Download the seven lung adenocarcinoma GEO cohorts of case study II.
#   Rscript data_prep/export_geo_luad.R <data_dir> [GSE ...]
#
# For each series this saves the series-matrix sample annotations (clinical.csv),
# the probe annotation (probes.csv) and the expression matrix (expression.csv.gz,
# samples x probes) under <data_dir>/cohorts/luad/<GSE>/. Survival fields differ
# between series and are curated in a separate harmonization step (harmonize_luad.py).
suppressPackageStartupMessages({
  library(GEOquery)
  library(Biobase)
  library(data.table)
})
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript export_geo_luad.R <data_dir> [GSE ...]")
data_dir <- args[[1]]
series <- if (length(args) > 1) args[-1] else c("GSE31210", "GSE72094", "GSE30219", "GSE50081", "GSE68465", "GSE13213", "GSE41271")
cache <- file.path(data_dir, "geo_cache")
dir.create(cache, recursive = TRUE, showWarnings = FALSE)
options(timeout = 3600)

for (accession in series) {
  folder <- file.path(data_dir, "cohorts", "luad", accession)
  if (file.exists(file.path(folder, "expression.csv.gz"))) {
    cat(accession, "already exported\n")
    next
  }
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  sets <- getGEO(accession, GSEMatrix = TRUE, AnnotGPL = FALSE, destdir = cache)
  for (index in seq_along(sets)) {
    eset <- sets[[index]]
    suffix <- if (length(sets) > 1) paste0("_", annotation(eset)) else ""
    pheno <- pData(eset)
    pheno$sample_id <- sampleNames(eset)
    fwrite(pheno, file.path(folder, paste0("clinical", suffix, ".csv")))
    fwrite(fData(eset), file.path(folder, paste0("probes", suffix, ".csv")))
    expression <- t(exprs(eset))
    fwrite(data.table(sample_id = rownames(expression), expression), file.path(folder, paste0("expression", suffix, ".csv.gz")))
    cat(accession, annotation(eset), ncol(eset), "samples,", nrow(eset), "probes\n")
  }
}
cat("GEOquery", as.character(packageVersion("GEOquery")), "\n")

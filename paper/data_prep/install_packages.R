# Install the R packages the data preparation needs (MetaGxBreast from Bioconductor,
# GEOquery and data.table). Run once:  Rscript data_prep/install_packages.R
options(Ncpus = max(1L, parallel::detectCores() - 2L))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
packages <- c("MetaGxBreast", "GEOquery", "data.table")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) BiocManager::install(missing, ask = FALSE, update = FALSE)
cat("Bioconductor", as.character(BiocManager::version()), "\n")
for (package in packages) {
  version <- if (requireNamespace(package, quietly = TRUE)) as.character(packageVersion(package)) else "NOT INSTALLED"
  cat(package, version, "\n")
}

# R packages of the competitor pipelines that conda-forge does not carry, installed into the conda R environment
# that setup_r_env.sh creates: plsRcox, compareC, forestploter and snowfall (and their missing dependencies) from a
# dated CRAN snapshot, CoxBoost and Mime from their git clones at the pinned commits.
# Usage: Rscript install_packages.R <CoxBoost clone> <Mime clone>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript install_packages.R <CoxBoost clone> <Mime clone>")
snapshot <- Sys.getenv("CRAN_SNAPSHOT", "https://packagemanager.posit.co/cran/2026-09-28")
# Source packages of the snapshot, with a generous download timeout (the default of 60 s fails on slow links).
options(repos = c(CRAN = snapshot), Ncpus = as.integer(Sys.getenv("INSTALL_CORES", "8")), timeout = 900)
bioc <- BiocManager::repositories()
repos <- c(CRAN = snapshot, bioc[names(bioc) != "CRAN"])
wanted <- c("plsRcox", "compareC", "forestploter", "snowfall")
for (attempt in 1:3) {
  missing <- wanted[!vapply(wanted, requireNamespace, logical(1), quietly = TRUE)]
  if (!length(missing)) break
  install.packages(missing, repos = repos, dependencies = c("Depends", "Imports", "LinkingTo"))
}
# Mime's RSF-based models call randomForestSRC::var.select, which randomForestSRC removed in 3.4.0 (2025-05-25):
# 3.3.3 (2025-01-15), the last version with it, replaces conda-forge's.
if (!nzchar(system.file(package = "randomForestSRC")) || packageVersion("randomForestSRC") != "3.3.3") {
  install.packages("https://cran.r-project.org/src/contrib/Archive/randomForestSRC/randomForestSRC_3.3.3.tar.gz", repos = NULL, type = "source")
}
for (clone in args) {
  status <- system2(file.path(R.home("bin"), "R"), c("CMD", "INSTALL", "--no-test-load", shQuote(clone)))
  if (status != 0) stop("R CMD INSTALL failed for ", clone)
}
for (package in c(wanted, "CoxBoost", "Mime1")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("not installed: ", package)
}
# Checked in a new R process: this one may hold the namespace of the version installed before.
check <- system2(file.path(R.home("bin"), "Rscript"), c("-e", shQuote("stopifnot(exists('var.select', envir = asNamespace('randomForestSRC')))")))
if (check != 0) stop("randomForestSRC lacks var.select, which Mime calls")
cat("installed:", paste(c(wanted, "CoxBoost", "Mime1"), collapse = ", "), "\n")

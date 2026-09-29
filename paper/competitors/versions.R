# The versions of R and of the packages the competitor pipelines use, written by setup_r_env.sh to versions.txt.
packages <- c("Mime1", "CoxBoost", "glmnet", "survival", "randomForestSRC", "gbm", "plsRcox", "plsRglm", "superpc",
              "survivalsvm", "mixOmics", "compareC", "snowfall", "Matrix", "data.table", "dplyr", "BiocManager")
cat("R:", R.version.string, "\n")
cat("platform:", R.version$platform, "\n")
info <- sessionInfo()
cat("BLAS:", info$BLAS, "\nLAPACK:", info$LAPACK, "\n")
for (package in packages) {
  version <- tryCatch(as.character(packageVersion(package)), error = function(e) "not installed")
  cat(sprintf("%s: %s\n", package, version))
}
cat("Mime commit:", Sys.getenv("MIME_COMMIT", "unknown"), "(github.com/l-magnificence/Mime)\n")
cat("CoxBoost commit:", Sys.getenv("COXBOOST_COMMIT", "unknown"), "(github.com/binderh/CoxBoost)\n")
cat("CRAN snapshot:", Sys.getenv("CRAN_SNAPSHOT", "https://packagemanager.posit.co/cran/2026-09-28"), "\n")

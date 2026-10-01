# Shared R code of the competitor pipelines (run_mime.R, run_p2.R): the cohorts in Mime's input layout (ID,
# OS.time in days, OS, then the genes z-scored within the cohort), the univariate Cox screen, and output helpers.
#
# Two sources of cohorts (cohort_source):
#   real  input=<folder>: the files 18_competitors_data.py writes, TCGA.csv (development) and one CSV per GEO cohort,
#         listed in cohorts=;
#   null  input=<replicate design>: a null replicate of 18_competitors_null.py, whose rows index the TCGA expression
#         matrix (expression=, null/tcga_expression.csv); each cohort is z-scored here, within itself.
# A pipeline screens every gene on the development cohort, then builds every cohort with only the genes it goes on
# with (z-scoring is per gene, so this changes no value), which keeps a null replicate's memory small.
suppressPackageStartupMessages({
  library(survival)
  library(data.table)
})
options(stringsAsFactors = FALSE)

read_table <- function(path) as.data.frame(data.table::fread(path, check.names = FALSE, showProgress = FALSE))

cohort_source <- function(args) {
  if (args$source == "real") {
    names <- strsplit(args$cohorts, ",")[[1]]
    read_cohort <- function(name, genes = NULL) {
      frame <- read_table(file.path(args$input, paste0(name, ".csv")))
      frame$ID <- as.character(frame$ID)
      if (is.null(genes)) frame else frame[, c("ID", "OS.time", "OS", genes), drop = FALSE]
    }
    return(list(
      development = function() read_cohort(names[1]),
      cohorts = function(genes) stats::setNames(lapply(names, read_cohort, genes = genes), names)
    ))
  }
  design <- read_table(args$input)
  expression <- read_table(args$expression)
  all_genes <- setdiff(colnames(expression), "patient_id")
  matrix <- as.matrix(expression[, all_genes, drop = FALSE])
  rm(expression)
  names <- unique(design$cohort)
  # One cohort of the replicate, its genes z-scored within it (R's scale: mean 0, SD with n - 1; a gene constant in
  # the cohort is 0).
  build <- function(name, genes) {
    part <- design[design$cohort == name, ]
    values <- scale(matrix[part$row, genes, drop = FALSE])
    values[is.na(values)] <- 0
    frame <- data.frame(ID = paste0(name, "_", seq_len(nrow(part))), OS.time = part$time_days, OS = part$event, check.names = FALSE)
    frame <- cbind(frame, as.data.frame(values, check.names = FALSE))
    colnames(frame) <- c("ID", "OS.time", "OS", genes)
    frame
  }
  list(
    development = function() build(names[1], all_genes),
    cohorts = function(genes) stats::setNames(lapply(names, build, genes = genes), names)
  )
}

# Mime's univariate Cox screen (SigUnicox: coxph(Surv(OS.time, OS) ~ gene), Efron ties, the Wald p-value) for every
# gene of the development cohort, through coxph.fit, which coxph calls, without the formula machinery.
unicox <- function(frame) {
  genes <- colnames(frame)[-(1:3)]
  y <- Surv(frame$OS.time, frame$OS)
  control <- coxph.control()
  rows <- lapply(genes, function(gene) {
    x <- matrix(as.numeric(frame[[gene]]), ncol = 1)
    fit <- tryCatch(coxph.fit(x, y, strata = NULL, offset = NULL, init = NULL, control = control, weights = NULL,
                              method = "efron", rownames = NULL), error = function(e) NULL)
    if (is.null(fit) || !is.finite(fit$coefficients[1]) || !(fit$var[1, 1] > 0)) return(c(NA, NA, NA))
    z <- fit$coefficients[1] / sqrt(fit$var[1, 1])
    c(fit$coefficients[1], z, 2 * pnorm(-abs(z)))
  })
  table <- data.frame(gene = genes, do.call(rbind, rows), check.names = FALSE)
  colnames(table) <- c("gene", "log_hr", "z", "p")
  table
}

write_csv <- function(frame, path) data.table::fwrite(frame, path)

# Arguments as key=value pairs.
arguments <- function(defaults) {
  given <- commandArgs(trailingOnly = TRUE)
  for (item in given) {
    parts <- strsplit(item, "=", fixed = TRUE)[[1]]
    if (length(parts) != 2 || !(parts[1] %in% names(defaults))) stop("unknown argument: ", item)
    defaults[[parts[1]]] <- parts[2]
  }
  defaults
}

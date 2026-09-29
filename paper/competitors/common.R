# Shared R code of the competitor pipelines (run_mime.R, run_p2.R): the cohorts in Mime's input layout (ID,
# OS.time in days, OS, then the genes z-scored within the cohort), the univariate Cox screen, and output helpers.
#
# Two sources of cohorts:
#   real  <input folder>: the files 17_competitors_data.py writes, TCGA.csv (development) and one CSV per GEO cohort;
#   null  <replicate file>: a null replicate of 17_competitors_null.py, whose rows index the TCGA expression matrix
#         (null/tcga_expression.csv next to the replicate folders); each cohort is z-scored here, within itself.
suppressPackageStartupMessages({
  library(survival)
  library(data.table)
})
options(stringsAsFactors = FALSE)

read_table <- function(path) as.data.frame(data.table::fread(path, check.names = FALSE, showProgress = FALSE))

# The cohorts of the real comparison: TCGA first (development), then the GEO cohorts in the order given.
real_cohorts <- function(folder, names) {
  cohorts <- lapply(names, function(name) {
    frame <- read_table(file.path(folder, paste0(name, ".csv")))
    frame$ID <- as.character(frame$ID)
    frame
  })
  names(cohorts) <- names
  cohorts
}

# The cohorts of a null replicate: rows of the TCGA expression matrix (1-based) with new outcomes, each cohort's
# genes z-scored within it (R's scale: mean 0, SD with n - 1), as 17_competitors_null.py describes.
null_cohorts <- function(replicate_file, expression) {
  design <- read_table(replicate_file)
  genes <- setdiff(colnames(expression), "patient_id")
  expression <- as.matrix(expression[, genes, drop = FALSE])
  cohorts <- list()
  for (name in unique(design$cohort)) {
    part <- design[design$cohort == name, ]
    values <- scale(expression[part$row, , drop = FALSE])
    values[is.na(values)] <- 0
    frame <- data.frame(ID = paste0(name, "_", seq_len(nrow(part))), OS.time = part$time_days, OS = part$event, check.names = FALSE)
    frame <- cbind(frame, as.data.frame(values, check.names = FALSE))
    colnames(frame) <- c("ID", "OS.time", "OS", genes)
    cohorts[[name]] <- frame
  }
  cohorts
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

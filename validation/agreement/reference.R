# R survival reference values for SurvStudio's numerical agreement report.
#
#   Rscript reference.R <work_dir>
#
# Reads the cohorts and settings that run_agreement.py writes to <work_dir> and writes
# <work_dir>/r_values.csv with one row per compared quantity (case, quantity, value).
# The proportional-hazards statistics use the classic Grambsch-Therneau formulas on the
# Schoenfeld residuals with log time (what SurvStudio reports), not the newer cox.zph test.

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
work <- args[[1]]
spec <- fromJSON(file.path(work, "spec.json"), simplifyVector = FALSE)
rows <- list()

emit <- function(case, quantity, value) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, quantity = quantity, value = as.numeric(value))
}

read_cohort <- function(name) {
  # pandas writes missing values as empty fields; R would otherwise read them as a level "".
  read.csv(file.path(work, paste0(name, ".csv")), check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA"))
}

set_factors <- function(data, references) {
  for (column in names(references)) {
    data[[column]] <- relevel(factor(data[[column]]), ref = references[[column]])
  }
  data
}

classic_zph <- function(fit) {
  residuals <- resid(fit, "schoenfeld")
  if (is.null(dim(residuals))) {
    residuals <- matrix(residuals, ncol = 1, dimnames = list(names(residuals), NULL))
  }
  deaths <- nrow(residuals)
  times <- as.numeric(rownames(residuals))
  centred <- log(times) - mean(log(times))
  scaled <- residuals %*% fit$var * deaths
  term <- c((centred %*% scaled)^2 / (diag(fit$var) * deaths * sum(centred^2)))
  global_score <- c(centred %*% residuals)
  global <- c(global_score %*% fit$var %*% global_score) * deaths / sum(centred^2)
  list(term = term, global = global)
}

for (case in spec$km) {
  data <- read_cohort(case$cohort)
  surv <- Surv(data[[case$time]], data[[case$event]])
  groups <- factor(data[[case$group]])
  fit <- survfit(surv ~ groups, conf.type = "log-log")
  at <- summary(fit, times = unlist(case$times), extend = TRUE)
  labels <- sub("^groups=", "", as.character(at$strata))
  for (index in seq_along(at$time)) {
    prefix <- paste0(labels[index], " @ ", at$time[index])
    emit(case$name, paste0(prefix, " survival"), at$surv[index])
    emit(case$name, paste0(prefix, " CI lower"), at$lower[index])
    emit(case$name, paste0(prefix, " CI upper"), at$upper[index])
  }
  table <- summary(fit, rmean = case$tau)$table
  for (index in seq_len(nrow(table))) {
    label <- sub("^groups=", "", rownames(table)[index])
    emit(case$name, paste0(label, " median"), table[index, "median"])
    emit(case$name, paste0(label, " median CI lower"), table[index, "0.95LCL"])
    emit(case$name, paste0(label, " median CI upper"), table[index, "0.95UCL"])
    emit(case$name, paste0(label, " RMST"), table[index, "rmean"])
    emit(case$name, paste0(label, " RMST SE"), table[index, "se(rmean)"])
  }
  test <- survdiff(surv ~ groups)
  emit(case$name, "log-rank chi-square", test$chisq)
}

for (case in spec$cox) {
  data <- set_factors(read_cohort(case$cohort), case$references)
  terms <- unlist(case$covariates)
  formula_text <- paste0("Surv(", case$time, ", ", case$event, ") ~ ", paste(sprintf("`%s`", terms), collapse = " + "))
  if (length(case$strata) > 0) {
    formula_text <- paste0(formula_text, " + ", paste(sprintf("strata(`%s`)", unlist(case$strata)), collapse = " + "))
  }
  fit <- coxph(as.formula(formula_text), data = data, ties = "efron")
  names_r <- names(coef(fit))
  labels <- vapply(names_r, function(name) {
    label <- case$labels[[name]]
    if (is.null(label)) stop("No SurvStudio label for the R coefficient ", name)
    label
  }, character(1))
  for (index in seq_along(names_r)) {
    emit(case$name, paste0(labels[index], " coefficient"), coef(fit)[index])
    emit(case$name, paste0(labels[index], " SE"), sqrt(diag(fit$var))[index])
  }
  emit(case$name, "partial log-likelihood", fit$loglik[2])
  emit(case$name, "likelihood-ratio chi-square", 2 * diff(fit$loglik))
  if (length(case$strata) == 0) {
    emit(case$name, "concordance", concordance(fit, timewt = "n")$concordance)
  }
  zph <- classic_zph(fit)
  for (index in seq_along(names_r)) {
    emit(case$name, paste0(labels[index], " PH chi-square"), zph$term[index])
  }
  emit(case$name, "global PH chi-square", zph$global)
}

for (case in spec$score) {
  data <- read_cohort(case$cohort)
  surv <- Surv(data[[case$time]], data[[case$event]])
  clinical <- as.matrix(data[, unlist(case$clinical_design), drop = FALSE])
  null_fit <- coxph(surv ~ clinical, ties = "efron")
  for (marker in unlist(case$markers)) {
    x <- data[[marker]]
    marginal <- coxph(surv ~ x, init = 0, iter.max = 0, ties = "efron")
    adjusted <- coxph(surv ~ clinical + x, init = c(coef(null_fit), 0), iter.max = 0, ties = "efron")
    emit(case$name, paste0(marker, " marginal score chi-square"), marginal$score)
    emit(case$name, paste0(marker, " added-value score chi-square"), adjusted$score)
  }
}

for (case in spec$fit) {
  data <- read_cohort(case$cohort)
  design <- as.matrix(data[, unlist(case$covariates), drop = FALSE])
  surv <- Surv(data[[case$time]], data[[case$event]])
  strata_codes <- if (length(case$strata) > 0) data[[case$strata[[1]]]] else rep(1, nrow(data))
  for (ties in c("efron", "breslow")) {
    fit <- coxph(surv ~ design + strata(strata_codes), ties = ties)
    for (index in seq_along(case$covariates)) {
      emit(case$name, paste0(case$covariates[[index]], " coefficient (", ties, ")"), coef(fit)[index])
      emit(case$name, paste0(case$covariates[[index]], " SE (", ties, ")"), sqrt(diag(fit$var))[index])
    }
    emit(case$name, paste0("partial log-likelihood (", ties, ")"), fit$loglik[2])
  }
}

write.csv(do.call(rbind, rows), file.path(work, "r_values.csv"), row.names = FALSE)

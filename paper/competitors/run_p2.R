# P2, the most common signature recipe: univariate Cox (p < 0.05) on the development cohort -> LASSO-Cox with 10-fold
# cross-validation (glmnet, lambda.min, fixed seed) -> multivariable Cox of the selected genes, whose linear predictor
# is the risk score in every cohort.
#   Rscript run_p2.R source=real input=<folder> cohorts=TCGA,GSE13213,... out=<folder>
#   Rscript run_p2.R source=null input=<replicate file> expression=<tcga_expression.csv> out=<folder>
# Writes to out/: unicox.csv (every gene), lasso.csv (the genes LASSO keeps and their coefficients), cox.csv (the
# multivariable Cox fit), risk.csv (cohort, ID, OS.time, OS, risk) and run.json.
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
suppressPackageStartupMessages(library(glmnet))
args <- arguments(list(source = "real", input = "", cohorts = "", expression = "", out = "", seed = "5201314", p = "0.05"))
began <- Sys.time()
dir.create(args$out, recursive = TRUE, showWarnings = FALSE)
source_ <- cohort_source(args)
development <- source_$development()
screen <- unicox(development)
genes_screened <- nrow(screen)
rm(development)
write_csv(screen, file.path(args$out, "unicox.csv"))
kept <- screen$gene[!is.na(screen$p) & screen$p < as.numeric(args$p)]
# Every cohort with the genes passing the screen (the development cohort first).
cohorts <- source_$cohorts(kept)
train <- cohorts[[1]]

p2 <- function(train, kept, seed) {
  x <- as.matrix(train[, kept, drop = FALSE])
  y <- Surv(train$OS.time, train$OS)
  set.seed(seed)
  cv <- cv.glmnet(x, y, family = "cox", alpha = 1, nfolds = 10)
  lasso <- as.matrix(coef(cv, s = "lambda.min"))
  selected <- rownames(lasso)[lasso[, 1] != 0]
  list(cv = cv, lasso = data.frame(gene = selected, coefficient = lasso[selected, 1]), selected = selected)
}

result <- list(selected = character(0))
if (length(kept) >= 2) result <- p2(train, kept, as.integer(args$seed))
selected <- result$selected
write_csv(if (length(selected)) result$lasso else data.frame(gene = character(0), coefficient = numeric(0)), file.path(args$out, "lasso.csv"))
beta <- numeric(0)
converged <- NA
if (length(selected) >= 1) {
  x <- as.matrix(train[, selected, drop = FALSE])
  fit <- coxph(Surv(train$OS.time, train$OS) ~ x)
  beta <- coef(fit)
  names(beta) <- selected
  converged <- fit$iter < coxph.control()$iter.max
  s <- summary(fit)$coefficients
  write_csv(data.frame(gene = selected, coefficient = unname(beta), se = s[, "se(coef)"], p = s[, "Pr(>|z|)"]),
            file.path(args$out, "cox.csv"))
  beta[is.na(beta)] <- 0
}
risk <- do.call(rbind, lapply(names(cohorts), function(name) {
  frame <- cohorts[[name]]
  score <- if (length(selected)) as.numeric(as.matrix(frame[, selected, drop = FALSE]) %*% beta) else rep(NA_real_, nrow(frame))
  data.frame(cohort = name, ID = frame$ID, OS.time = frame$OS.time, OS = frame$OS, risk = score)
}))
write_csv(risk, file.path(args$out, "risk.csv"))
info <- list(genes = genes_screened, unicox_kept = length(kept), selected = length(selected),
             lambda_min = if (length(kept) >= 2) result$cv$lambda.min else NA, cox_converged = converged,
             seconds = as.numeric(difftime(Sys.time(), began, units = "secs")), seed = as.integer(args$seed),
             glmnet = as.character(packageVersion("glmnet")), survival = as.character(packageVersion("survival")),
             r = R.version.string)
writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, digits = NA, na = "null"), file.path(args$out, "run.json"))

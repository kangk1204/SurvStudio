# P1 Mime: ML.Dev.Prog.Sig (Mime1, the pinned commit) trained on the development cohort and scored in every cohort.
# Mime's own univariate-Cox candidate filter (p < 0.05) is kept; the candidates are the `cap` genes with the smallest
# univariate p-values among those passing it (Mime then repeats its filter on them, which keeps all of them).
# Every other setting is Mime's documented default (nodesize 5, the example seed 5201314).
#   Rscript run_mime.R source=real input=<folder> cohorts=TCGA,GSE13213,... cap=100 out=<folder> cores=6
#   Rscript run_mime.R source=null input=<replicate file> expression=<tcga_expression.csv> cap=100 out=<folder> cores=1
# plan=all runs mode "all" (every one of Mime's models); plan=feasible runs, through Mime's single and double modes,
# every model whose first algorithm is not StepCox (the sensitivity run with 500 candidates, where StepCox cannot fit
# the full Cox model: more genes than deaths).
# Writes to out/: unicox.csv, candidates.txt, cindex.csv (model, cohort, Mime's C), risk.csv.gz (model, cohort, ID,
# OS.time, OS, RS), genes.csv (model, genes the fitted model uses) and run.json.
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
args <- arguments(list(source = "real", input = "", cohorts = "", expression = "", out = "", cap = "100", seed = "5201314",
                       cores = "6", nodesize = "5", plan = "all", p = "0.05"))
cores <- as.integer(args$cores)
seed <- as.integer(args$seed)
options(rf.cores = cores, mc.cores = cores)
suppressPackageStartupMessages(library(Mime1))
# CoxBoost's optimCoxBoostPenalty(parallel = TRUE), as Mime calls it, runs through snowfall, which CoxBoost's
# documentation asks to initialise first; sequential mode computes the same folds.
suppressPackageStartupMessages(library(snowfall))
snowfall::sfInit(parallel = FALSE)
began <- Sys.time()
dir.create(args$out, recursive = TRUE, showWarnings = FALSE)
if (args$source == "real") {
  cohorts <- real_cohorts(args$input, strsplit(args$cohorts, ",")[[1]])
} else {
  cohorts <- null_cohorts(args$input, read_table(args$expression))
}
train <- cohorts[[1]]
screen <- unicox(train)
write_csv(screen, file.path(args$out, "unicox.csv"))
passing <- screen[!is.na(screen$p) & screen$p < as.numeric(args$p), ]
passing <- passing[order(passing$p), ]
candidates <- head(passing$gene, as.integer(args$cap))
writeLines(candidates, file.path(args$out, "candidates.txt"))

run <- function(...) {
  ML.Dev.Prog.Sig(train_data = train, list_train_vali_Data = cohorts, candidate_genes = candidates,
                  unicox.filter.for.candi = TRUE, unicox_p_cutoff = as.numeric(args$p),
                  nodesize = as.integer(args$nodesize), seed = seed, cores_for_parallel = cores, ...)
}

# The models of mode "all" whose first algorithm is not StepCox, as single and double calls.
feasible_calls <- function() {
  alphas <- seq(0.1, 0.9, 0.1)
  directions <- c("both", "backward", "forward")
  calls <- list(list(mode = "single", single_ml = "RSF"))
  for (alpha in alphas) calls[[length(calls) + 1]] <- list(mode = "single", single_ml = "Enet", alpha_for_Enet = alpha)
  for (ml in c("CoxBoost", "plsRcox", "superpc", "GBM", "survivalsvm", "Ridge", "Lasso")) calls[[length(calls) + 1]] <- list(mode = "single", single_ml = ml)
  second <- list(RSF = c("CoxBoost", "Enet", "GBM", "Lasso", "plsRcox", "Ridge", "StepCox", "superpc", "survivalsvm"),
                 CoxBoost = c("Enet", "GBM", "Lasso", "plsRcox", "Ridge", "StepCox", "superpc", "survivalsvm"),
                 Lasso = c("CoxBoost", "GBM", "plsRcox", "RSF", "StepCox", "superpc", "survivalsvm"))
  for (first in names(second)) for (ml in second[[first]]) {
    if (ml == "Enet") {
      for (alpha in alphas) calls[[length(calls) + 1]] <- list(mode = "double", double_ml1 = first, double_ml2 = ml, alpha_for_Enet = alpha)
    } else if (ml == "StepCox") {
      for (direction in directions) calls[[length(calls) + 1]] <- list(mode = "double", double_ml1 = first, double_ml2 = ml, direction_for_stepcox = direction)
    } else {
      calls[[length(calls) + 1]] <- list(mode = "double", double_ml1 = first, double_ml2 = ml)
    }
  }
  calls
}

if (args$plan == "all") {
  res <- run(mode = "all")
  cindex <- res$Cindex.res
  riskscore <- res$riskscore
  fits <- res$ml.res
} else {
  cindex <- data.frame()
  riskscore <- list()
  fits <- list()
  for (call in feasible_calls()) {
    part <- tryCatch(do.call(run, call), error = function(e) {
      message("failed: ", paste(unlist(call), collapse = " "), ": ", conditionMessage(e))
      NULL
    })
    if (is.null(part) || !is.list(part) || is.null(part$Cindex.res)) next
    cindex <- rbind(cindex, part$Cindex.res)
    riskscore <- c(riskscore, part$riskscore)
    fits <- c(fits, part$ml.res)
  }
}

# The genes a fitted model uses: non-zero coefficients where the model has them, else every input gene.
model_genes <- function(fit) {
  genes <- tryCatch({
    if (inherits(fit, "cv.glmnet")) {
      b <- as.matrix(coef(fit, s = fit$lambda.min)); rownames(b)[b[, 1] != 0]
    } else if (is.list(fit) && inherits(fit$fit, "glmnet") && !is.null(fit$cv.fit)) {
      b <- as.matrix(coef(fit$fit, s = fit$cv.fit$lambda.min)); rownames(b)[b[, 1] != 0]
    } else if (inherits(fit, "CoxBoost")) {
      b <- coef(fit); names(b)[b != 0]
    } else if (inherits(fit, "coxph")) {
      b <- coef(fit); names(b)[!is.na(b)]
    } else if (inherits(fit, "rfsrc")) {
      fit$xvar.names
    } else if (is.list(fit) && inherits(fit$fit, "gbm")) {
      fit$fit$var.names
    } else if (inherits(fit, "survivalsvm")) {
      fit$var.names
    } else if (is.list(fit) && length(fit) == 2 && !is.null(fit[[1]]$feature.scores)) {
      threshold <- fit[[2]]$thresholds[which.max(fit[[2]][["scor"]][1, ])]
      names(fit[[1]]$feature.scores)[abs(fit[[1]]$feature.scores) >= threshold]
    } else if (!is.null(fit$dataX)) {
      colnames(fit$dataX)
    } else {
      NA_character_
    }
  }, error = function(e) NA_character_)
  genes <- genes[!is.na(genes)]
  if (!length(genes)) NA_character_ else genes
}

models <- unique(cindex$Model)
write_csv(data.frame(model = cindex$Model, cohort = cindex$ID, cindex = cindex$Cindex), file.path(args$out, "cindex.csv"))
risk <- do.call(rbind, lapply(names(riskscore), function(model) {
  do.call(rbind, lapply(names(riskscore[[model]]), function(cohort) {
    frame <- riskscore[[model]][[cohort]]
    data.frame(model = model, cohort = cohort, ID = frame$ID, OS.time = frame$OS.time, OS = frame$OS, RS = frame$RS)
  }))
}))
data.table::fwrite(risk, file.path(args$out, "risk.csv.gz"), compress = "gzip")
genes <- do.call(rbind, lapply(names(fits), function(model) {
  used <- model_genes(fits[[model]])
  data.frame(model = model, n_genes = if (all(is.na(used))) NA_integer_ else length(used), genes = paste(used, collapse = ";"))
}))
write_csv(genes, file.path(args$out, "genes.csv"))
info <- list(plan = args$plan, cap = as.integer(args$cap), unicox_passing = nrow(passing), candidates = length(candidates),
             models = length(models), cohorts = names(cohorts), seed = seed, nodesize = as.integer(args$nodesize), cores = cores,
             seconds = as.numeric(difftime(Sys.time(), began, units = "secs")),
             mime = as.character(packageVersion("Mime1")), r = R.version.string)
writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, digits = NA, na = "null"), file.path(args$out, "run.json"))
snowfall::sfStop()

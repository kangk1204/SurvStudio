# P1 Mime: ML.Dev.Prog.Sig (Mime1, the pinned commit) trained on the development cohort and scored in every cohort.
# Mime's own univariate-Cox candidate filter (p < 0.05) is kept; the candidates are the `cap` genes with the smallest
# univariate p-values among those passing it (Mime then repeats its filter on them, which keeps all of them).
# Every other setting is Mime's documented default (nodesize 5, the example seed 5201314).
#   Rscript run_mime.R source=real input=<folder> cohorts=TCGA,GSE13213,... cap=100 out=<folder> cores=6
#   Rscript run_mime.R source=null input=<replicate file> expression=<tcga_expression.csv> cap=100 out=<folder> cores=1
# plan=all runs mode "all" (every one of Mime's models). plan=feasible runs mode "all" without the models whose first
# algorithm is StepCox, for the sensitivity run with 500 candidates, where StepCox cannot fit the full Cox model (more
# genes than deaths): Mime's own source of the pinned commit (mime_source=, checked against the installed function)
# with its section "3.StepCox" cut out. (Mime's single and double modes, separate copies of the code, cannot stand in:
# in double mode, RSF + Enet fits alpha 0.1 whatever alpha it is given.)
# Writes to out/: unicox.csv, candidates.txt, cindex.csv (model, cohort, Mime's C), risk.csv.gz (model, cohort, ID,
# OS.time, OS, RS), genes.csv (model, genes the fitted model uses) and run.json.
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
args <- arguments(list(source = "real", input = "", cohorts = "", expression = "", out = "", cap = "100", seed = "5201314",
                       cores = "6", nodesize = "5", plan = "all", p = "0.05",
                       mime_source = file.path(here, "src", "Mime", "R", "ML.Dev.Prog.Sig.R")))
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
source_ <- cohort_source(args)
development <- source_$development()
screen <- unicox(development)
rm(development)
write_csv(screen, file.path(args$out, "unicox.csv"))
passing <- screen[!is.na(screen$p) & screen$p < as.numeric(args$p), ]
passing <- passing[order(passing$p), ]
candidates <- head(passing$gene, as.integer(args$cap))
writeLines(candidates, file.path(args$out, "candidates.txt"))
# Every cohort with the candidate genes only (the development cohort first): Mime keeps no other column.
cohorts <- source_$cohorts(candidates)
train <- cohorts[[1]]

# Mode "all" without the models whose first algorithm is StepCox: Mime's source with its section "3.StepCox" (from its
# heading to the heading of section 4, CoxBoost) cut out, after checking that the source is the installed function.
without_stepcox_first <- function(path) {
  lines <- readLines(path, encoding = "UTF-8")
  original <- new.env()
  eval(parse(text = lines, encoding = "UTF-8"), envir = original)
  if (!identical(deparse(body(original$ML.Dev.Prog.Sig)), deparse(body(Mime1::ML.Dev.Prog.Sig)))) {
    stop(path, " is not the source of the installed Mime1::ML.Dev.Prog.Sig")
  }
  first <- grep("^[[:space:]]*# 3[.]StepCox -+", lines)
  last <- grep("^[[:space:]]*# # 4[.]CoxBoost -+", lines)
  if (length(first) != 1 || length(last) != 1 || last <= first) stop("cannot find Mime's section 3.StepCox in ", path)
  cut <- new.env()
  eval(parse(text = lines[-(first:(last - 1))], encoding = "UTF-8"), envir = cut)
  cut$ML.Dev.Prog.Sig
}

prog_sig <- if (args$plan == "all") Mime1::ML.Dev.Prog.Sig else without_stepcox_first(args$mime_source)
res <- prog_sig(train_data = train, list_train_vali_Data = cohorts, candidate_genes = candidates,
                unicox.filter.for.candi = TRUE, unicox_p_cutoff = as.numeric(args$p), mode = "all",
                nodesize = as.integer(args$nodesize), seed = seed, cores_for_parallel = cores)
cindex <- res$Cindex.res
riskscore <- res$riskscore
fits <- res$ml.res

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

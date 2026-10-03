#!/usr/bin/env Rscript
# Independent aggregation from raw replicate indicators and shared row streams.
# No Python summaries, p-values, counts, intervals or qualification are read.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) > 1) .libPaths(c(args[2], .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
directory <- args[1]
settings <- fromJSON(file.path(directory, "settings.json"), simplifyVector=FALSE)
stopifnot(settings$draws == 9999L, settings$planned > 0L)
d <- read.csv(file.path(directory, "raw-indicators.csv"), check.names=FALSE)
reasons <- read.csv(file.path(directory, "raw-reasons.csv"), check.names=FALSE)
stopifnot(!anyDuplicated(d$index), all(diff(d$index) > 0),
          all(d$index >= 0 & d$index < settings$planned))
flag.names <- c("failure", "allowed", "null_rejected", "raw_null_rejected",
                "diagnostic_failure", "baseline_failure", "baseline_allowed",
                "baseline_null_rejected")
for (name in flag.names) stopifnot(all(d[[name]] %in% 0:1))
stopifnot(all(d$true_rejected >= 0 & d$true_rejected <= d$n_true),
          all(d$baseline_true_rejected >= 0 & d$baseline_true_rejected <= d$baseline_n_true),
          all(!d$failure | (!d$allowed & !d$null_rejected & d$true_rejected == 0)),
          all(!d$null_rejected | (d$allowed & !d$failure)),
          all(!d$baseline_null_rejected | (d$baseline_allowed & !d$baseline_failure)),
          nrow(d) <= settings$planned)
if (nrow(reasons)) stopifnot(all(reasons$index %in% d$index),
                            !anyDuplicated(reasons[c("index", "reason")]),
                            all(reasons$count > 0 & reasons$count == floor(reasons$count)))
cp <- function(hits, total, side) {
  if (!total) return(NULL)
  if (side == "upper") return(if (hits == total) 1 else qbeta(.95, hits+1, total-hits))
  if (hits == 0) 0 else qbeta(.05, hits, total-hits+1)
}
read.stream <- function(path, columns, consume) {
  if (!columns) return(invisible(NULL))
  connection <- file(path, "rb"); on.exit(close(connection))
  for (begin in seq.int(0L, settings$draws-1L, by=128L)) {
    rows <- min(128L, settings$draws-begin)
    values <- readBin(connection, integer(), n=rows*columns, size=4L, endian="little")
    stopifnot(length(values) == rows*columns, all(values >= 0L & values < columns))
    consume(matrix(values+1L, nrow=rows, byrow=TRUE), begin+1L, rows)
  }
  stopifnot(length(readBin(connection, integer(), n=1L, size=4L, endian="little")) == 0L)
}
planned <- settings$planned; completed <- nrow(d); missing <- planned-completed
failed <- sum(d$failure); uncertain <- failed+missing
allowed <- !d$failure & as.logical(d$allowed)
hits <- sum(d$null_rejected); raw <- sum(d$raw_null_rejected)
conditional <- sum(d$null_rejected[allowed]); count.allowed <- sum(allowed)
reason.counts <- reason.datasets <- list()
for (reason in sort(unique(reasons$reason))) {
  take <- reasons$reason == reason
  reason.counts[[reason]] <- sum(reasons$count[take])
  reason.datasets[[reason]] <- sum(take)
}
ratio <- power <- NULL
if (settings$partial_null) {
  a <- ifelse(d$n_true > 0, d$true_rejected/pmax(d$n_true, 1), 0)
  b <- ifelse(d$baseline_failure, 1,
              ifelse(d$baseline_n_true > 0, d$baseline_true_rejected/pmax(d$baseline_n_true, 1), 0))
  # Match fixed-index order, followed by unresolved planned indices.
  a <- c(a, rep(0, missing)); b <- c(b, rep(1, missing))
  if (mean(b) > 0) {
    power <- mean(a); ratios <- numeric(settings$draws); undefined <- 0L
    read.stream(file.path(directory, "ratio-rows.i32"), planned, function(indices, start, rows) {
      numerator <- rowMeans(matrix(a[indices], nrow=rows))
      denominator <- rowMeans(matrix(b[indices], nrow=rows))
      undefined <<- undefined+sum(denominator == 0)
      ratios[start:(start+rows-1L)] <<- ifelse(denominator > 0, numerator/pmax(denominator, .Machine$double.xmin), 0)
    })
    ratio <- list(point=mean(a)/mean(b), lower95=unname(quantile(ratios, .05, type=7)),
                  mc95=list(unname(quantile(ratios, .025, type=7)),
                            if (undefined) NULL else unname(quantile(ratios, .975, type=7))),
                  undefined_denominator_draws=undefined,
                  zero_denominator_policy=if (undefined) "conservative lower bound; upper interval undefined" else "not needed")
  }
}
pairs <- !d$failure & !d$baseline_failure; pair.count <- sum(pairs)
differences <- list(complete_pairs=pair.count, unresolved_pairs=planned-pair.count, differences=list())
if (pair.count) {
  values <- cbind(d$null_rejected-d$baseline_null_rejected,
                  d$allowed-d$baseline_allowed,
                  ifelse(d$n_true > 0, d$true_rejected/pmax(d$n_true, 1), 0)-
                  ifelse(d$baseline_n_true > 0, d$baseline_true_rejected/pmax(d$baseline_n_true, 1), 0))[pairs,,drop=FALSE]
  resampled <- matrix(0, nrow=settings$draws, ncol=3L)
  read.stream(file.path(directory, "difference-rows.i32"), pair.count, function(indices, start, rows) {
    for (j in 1:3) resampled[start:(start+rows-1L),j] <<- rowMeans(matrix(values[,j][indices], nrow=rows))
  })
  for (j in 1:3) {
    point <- mean(values[,j]); radius <- (planned-pair.count)/planned
    differences$differences[[c("fwer", "allowed_fraction", "power")[j]]] <- list(
      point_complete_pairs=point,
      mc95_complete_pairs=as.list(unname(quantile(resampled[,j], c(.025,.975), type=7))),
      all_planned_failure_bounds=list(max(-1, pair.count/planned*point-radius), min(1, pair.count/planned*point+radius)))
  }
}
result <- list(stage=settings$stage, condition=settings$condition, n=settings$n, p=settings$p,
               method=settings$method, planned=planned, completed=completed, missing=missing, failures=failed,
               diagnostic_failures=sum(d$diagnostic_failure), allowed=count.allowed,
               allowed_fraction=count.allowed/planned, allowed_lower95=cp(count.allowed, planned, "lower"),
               fwer=hits/planned, raw_fwer_bounds=list(raw/planned, (raw+uncertain)/planned),
               fwer_failure_bounds=list(hits/planned, (hits+uncertain)/planned),
               fwer_upper95=cp(hits+uncertain, planned, "upper"), fwer_lower95=cp(hits, planned, "lower"),
               conditional_fwer=if (!count.allowed) NULL else conditional/count.allowed,
               conditional_upper95=cp(conditional, count.allowed, "upper"),
               conditional_lower95=cp(conditional, count.allowed, "lower"),
               withhold_fraction=1-count.allowed/planned, withhold_reasons=reason.counts,
               power=power, power_ratio=ratio, withhold_reason_dataset_counts=reason.datasets,
               withhold_reason_fractions=lapply(reason.datasets, function(x) x/planned),
               healthy_diagnostic_false_alarm_fraction=if (settings$healthy) sum(!d$allowed & !d$failure & !d$diagnostic_failure)/planned else NULL,
               paired_difference_from_legacy=differences,
               mean_diagnostic_seconds=if (completed) mean(d$diagnostic_seconds) else NULL,
               elapsed_p95_seconds=if (completed) unname(quantile(d$elapsed, .95, type=7)) else NULL)
write(toJSON(result, auto_unbox=TRUE, digits=NA, null="null", na="null"), file.path(directory, "r-summary.json"))

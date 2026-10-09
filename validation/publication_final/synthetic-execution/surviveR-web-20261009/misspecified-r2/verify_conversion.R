args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
original <- read.csv(args[1], check.names = FALSE)
converted <- read.csv(args[2], check.names = FALSE)
stopifnot(!anyDuplicated(original$time), !anyDuplicated(converted$time))
# The native download sorts by time; align independently before comparing.
original <- original[order(original$time), , drop = FALSE]
converted <- converted[order(converted$time), , drop = FALSE]
cutoff <- median(original$m0)
expected <- ifelse(original$m0 > cutoff, "HIGH", "LOW")
common_columns <- intersect(names(original), names(converted))
max_difference <- max(abs(as.matrix(original[, common_columns]) -
                          as.matrix(converted[, common_columns])))
checks <- data.frame(
  check = c("rows", "original_columns_retained", "event_values",
            "events", "numeric_content", "all_median_classifications",
            "HIGH_count", "LOW_count", "fixed_median"),
  observed = c(nrow(converted), length(common_columns),
               paste(sort(unique(converted$event)), collapse = ":"),
               sum(converted$event), format(max_difference, digits = 17),
               sum(converted$m0_Median == expected),
               sum(converted$m0_Median == "HIGH"),
               sum(converted$m0_Median == "LOW"),
               format(cutoff, digits = 17)),
  expected = c(600, ncol(original), "0:1", 345, "<=1e-6", 600,
               300, 300, "0.8184982509438479 +/- 1e-12"),
  passed = c(nrow(converted) == 600,
             identical(common_columns, names(original)),
             identical(sort(unique(converted$event)), c(0L, 1L)),
             sum(converted$event) == 345,
             is.finite(max_difference) && max_difference <= 1e-6,
             all(converted$m0_Median == expected),
             sum(converted$m0_Median == "HIGH") == 300,
             sum(converted$m0_Median == "LOW") == 300,
             abs(cutoff - 0.8184982509438479) <= 1e-12))
write.csv(checks, file.path(args[3], "independent-R-conversion-checks.csv"),
          row.names = FALSE)
capture.output(sessionInfo(), file = file.path(args[3], "R-sessionInfo.txt"))
stopifnot(all(checks$passed))
print(checks, row.names = FALSE)

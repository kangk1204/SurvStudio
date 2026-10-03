args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
d <- read.csv(gzfile(file.path(root, "confirmation-v2", "replicate-outcomes.csv.gz")))
rows <- list()
for (condition in c("partial_weak", "partial_strong")) {
  baseline <- d[d$stage=="main" & d$condition==condition & d$method=="legacy_linear", ]
  for (method in c("guarded_linear", "guarded_spline")) {
    guarded <- d[d$stage=="main" & d$condition==condition & d$method==method, ]
    paired <- merge(guarded, baseline, by="index", suffixes=c("_guarded", "_baseline"))
    stopifnot(nrow(paired)==5000, !any(paired$calculation_failed_guarded), !any(paired$calculation_failed_baseline))
    a <- paired$guarded_true_rejected_guarded / paired$n_true_guarded
    b <- paired$guarded_true_rejected_baseline / paired$n_true_baseline
    ratio <- mean(a)/mean(b)
    se <- sqrt((var(a) + ratio^2*var(b) - 2*ratio*cov(a,b))/length(a))/mean(b)
    rows[[length(rows)+1]] <- data.frame(condition=condition,method=method,datasets=length(a),ratio=ratio,
      paired_delta_MC95_lower=ratio-1.96*se,paired_delta_MC95_upper=ratio+1.96*se)
  }
}
write.csv(do.call(rbind,rows),file.path(root,"power-retention-independent-R.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(root,"power-retention-R-session.txt"))

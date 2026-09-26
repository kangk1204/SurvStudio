# Reference values for tests/test_marker_screen.py, from R's survival package (3.8.x).
#
# Run from the repository root:  Rscript tests/r_reference/marker_screen_reference.R
#
# The cohort is the 40-row tied-time table used by the R reference tests in
# tests/test_analysis.py (_R_REFERENCE_ROWS). Three marker columns are derived
# from the row number i = 1..40 so that Python can rebuild them exactly:
#   m1 = sin(i)                (continuous)
#   m2 = (7 * i mod 11) - 5    (integer codes with ties)
#   m3 = 0.6 * x + cos(i)      (correlated with the clinical covariate x)
#
# "univariate" is the Cox score test of one marker at beta = 0 (R's coxph()$score
# with iter.max = 0). "adjusted" is the score test for adding one marker to the
# fitted clinical model x + z: coxph(~ x + z + m, init = c(coef(f0), 0),
# iter.max = 0)$score, the efficient score statistic U^2 / (I_mm - I_mz I_zz^-1 I_zm).

library(survival)

rows <- paste0(
  "2,1,0.034,0,A;2,0,1.36,0,B;1,1,1.225,0,B;6,1,-0.51,1,B;6,0,-0.298,0,B;1,0,-0.527,0,B;",
  "1,1,0.57,0,B;8,1,-0.056,0,B;2,1,0.747,1,A;1,0,-1.847,1,B;1,1,1.567,1,A;4,1,-0.096,0,A;",
  "2,0,0.68,1,B;5,1,-0.137,0,B;2,1,-0.379,0,B;1,1,0.463,1,A;3,1,0.825,1,A;2,0,-0.203,1,A;",
  "1,1,-0.153,1,B;2,1,0.686,0,A;3,0,-0.87,0,A;2,1,-1.514,1,A;5,1,0.395,1,A;2,1,-0.671,0,B;",
  "10,1,-1.92,1,B;1,0,-0.814,0,A;1,0,-0.468,1,A;4,0,-1.193,0,A;7,1,-1.492,0,B;1,0,0.037,0,A;",
  "1,1,0.897,1,A;3,0,-0.233,1,A;4,0,-0.744,1,B;1,1,0.385,0,B;3,0,0.717,0,A;1,0,-0.3,1,B;",
  "2,0,0.545,1,A;2,1,1.043,0,A;8,0,-0.207,0,A;7,1,-0.814,0,A"
)
parts <- strsplit(strsplit(rows, ";")[[1]], ",")
d <- data.frame(
  time = as.numeric(sapply(parts, `[`, 1)),
  event = as.integer(sapply(parts, `[`, 2)),
  x = as.numeric(sapply(parts, `[`, 3)),
  z = as.integer(sapply(parts, `[`, 4)),
  s = sapply(parts, `[`, 5)
)
i <- seq_len(nrow(d))
d$m1 <- sin(i)
d$m2 <- ((7 * i) %% 11) - 5
d$m3 <- 0.6 * d$x + cos(i)
markers <- c("m1", "m2", "m3")

fit_at <- function(formula, init, ties) {
  coxph(formula, data = d, ties = ties, init = init, control = coxph.control(iter.max = 0))
}

fmt <- function(values) paste(sprintf("%.12g", values), collapse = ", ")

for (ties in c("efron", "breslow")) {
  for (stratified in c(FALSE, TRUE)) {
    suffix <- if (stratified) " + strata(s)" else ""
    label <- sprintf("%s, strata=%s", ties, stratified)
    univariate <- sapply(markers, function(m) {
      fit_at(as.formula(paste0("Surv(time, event) ~ ", m, suffix)), 0, ties)$score
    })
    null_loglik <- fit_at(as.formula(paste0("Surv(time, event) ~ m1", suffix)), 0, ties)$loglik[1]
    f0 <- coxph(as.formula(paste0("Surv(time, event) ~ x + z", suffix)), data = d, ties = ties)
    adjusted <- sapply(markers, function(m) {
      fit_at(as.formula(paste0("Surv(time, event) ~ x + z + ", m, suffix)), c(coef(f0), 0), ties)$score
    })
    cat(sprintf("%s | univariate: %s\n", label, fmt(univariate)))
    cat(sprintf("%s | adjusted: %s\n", label, fmt(adjusted)))
    cat(sprintf("%s | loglik beta=0: %s | clinical loglik: %s\n", label, fmt(null_loglik), fmt(f0$loglik[2])))
  }
}

# Exact fits used for the shortlist: clinical + m3, Efron ties, no strata.
f0 <- coxph(Surv(time, event) ~ x + z, data = d, ties = "efron")
f1 <- coxph(Surv(time, event) ~ x + z + m3, data = d, ties = "efron")
cat(sprintf("exact efron | m3 coef: %s | m3 se: %s | nested LR: %s\n",
            fmt(coef(f1)[["m3"]]), fmt(sqrt(diag(vcov(f1)))[3]), fmt(2 * (f1$loglik[2] - f0$loglik[2]))))

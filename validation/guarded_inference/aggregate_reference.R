# Independent base-R aggregation of every published fixed replicate, including failures.
args <- commandArgs(trailingOnly=TRUE);root <- args[1]
d <- read.csv(gzfile(file.path(root,"replicate-outcomes.csv.gz")),na.strings="",stringsAsFactors=FALSE)
keys <- c("stage","condition","n","p","method")
expected <- read.csv(file.path(root,"confirmation-cells.csv"),check.names=FALSE)
records <- list()
emit <- function(r,quantity,value) {
  records[[length(records)+1]] <<- cbind(r[,keys,drop=FALSE],data.frame(quantity=quantity,value=value))
}
cp <- function(h,n) if(n==0) c(NA_real_,NA_real_) else c(if(h==0) 0 else qbeta(.025,h,n-h+1),if(h==n) 1 else qbeta(.975,h+1,n-h))
upper <- function(h,n) if(n==0) NA_real_ else if(h==n) 1 else qbeta(.95,h+1,n-h)
for(i in seq_len(nrow(expected))) {
  r <- expected[i,,drop=FALSE];keep <- rep(TRUE,nrow(d))
  for(k in keys) keep <- keep & d[[k]]==r[[k]]
  x <- d[keep,,drop=FALSE];n <- r$planned;fail <- sum(x$calculation_failed)+(n-nrow(x));allowed <- sum(x$allowed)
  h <- sum(x$guarded_null_rejected);raw <- sum(x$raw_null_rejected,na.rm=TRUE)
  values <- c(completed=nrow(x),failures=sum(x$calculation_failed),diagnostic_exceptions=sum(x$diagnostic_failed),missing=n-nrow(x),
    allowed=allowed,allowed_fraction=allowed/n,fwer=h/n,fwer_raw_lower=raw/n,fwer_raw_upper=(raw+fail)/n,
    fwer_failure_bounds_lower=h/n,fwer_failure_bounds_upper=(h+fail)/n,
    conditional_fwer=if(allowed==0) NA_real_ else h/allowed,conditional_upper95=upper(h,allowed),
    withhold_fraction=(n-allowed)/n,fwer_worst_case_upper95=upper(h+fail,n))
  for(spec in list(list(name="raw_fwer_mc95",h=raw,n=n),list(name="raw_fwer_worst_case_mc95",h=raw+fail,n=n),
                  list(name="fwer_mc95",h=h,n=n),list(name="conditional_mc95",h=h,n=allowed))) {
    z <- cp(spec$h,spec$n);values[paste0(spec$name,c("_lower","_upper"))] <- z
  }
  if(grepl("^partial_",r$condition)) {
    powers <- x$guarded_true_rejected/x$n_true;power <- sum(powers)/n;half <- 1.96*sd(powers)/sqrt(n)
    values[c("power","power_mc95_lower","power_mc95_upper","power_failure_bounds_lower","power_failure_bounds_upper")] <-
      c(power,max(0,power-half),min(1,power+half),power,min(1,power+fail/n))
  }
  if(r$condition %in% c("independent","linear","correlated","z_dependent_censoring"))
    values["healthy_false_alarm_fraction"] <- (nrow(x)-allowed-sum(x$calculation_failed))/n
  for(k in names(values)) emit(r,k,values[k])
}
write.csv(do.call(rbind,records),file.path(root,"independent-R-aggregate-values.csv"),row.names=FALSE)
stopifnot(nrow(d)==256500,length(unique(paste(d$stage,d$condition,d$n,d$p,d$index)))==85500)
capture.output(sessionInfo(),file=file.path(root,"aggregate-R-session.txt"))

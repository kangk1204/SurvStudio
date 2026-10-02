args <- commandArgs(trailingOnly=TRUE)
library(survival)
d <- read.csv(args[1])
x <- cbind(as.numeric(d$grade=="B"),as.numeric(d$grade=="C"),d$z,d$binary,d$m0)
times <- c()
for(i in 1:13) {
  started <- proc.time()["elapsed"]
  fit <- coxph(Surv(d$time,d$event)~x,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14))
  elapsed <- proc.time()["elapsed"]-started
  if(i>3) times <- c(times,elapsed)
}
write.csv(data.frame(run=1:10,seconds=times),args[2],row.names=FALSE)
write.csv(data.frame(coefficient=coef(fit)),paste0(args[2],".coefficients.csv"),row.names=FALSE)

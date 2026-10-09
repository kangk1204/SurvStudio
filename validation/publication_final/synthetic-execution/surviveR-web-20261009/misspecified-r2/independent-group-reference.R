args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L)
library(survival)
d <- read.csv(args[1], check.names=FALSE)
endpoint <- as.numeric(args[3])
stopifnot(nrow(d)==600, sum(d$event)==345, endpoint==60)
dir.create(args[2], recursive=TRUE, showWarnings=FALSE)
# Recreate the locked grouping from raw input, independently of the native CSV.
cutoff <- median(d$m0)
d$group <- factor(ifelse(d$m0>cutoff, 'HIGH', 'LOW'), levels=c('LOW','HIGH'))
d$time_native <- pmin(d$time,endpoint)
d$event_native <- as.integer(d$event==1 & d$time<=endpoint)
fit <- coxph(Surv(time_native,event_native)~group,data=d,ties='efron',x=TRUE)
sm <- summary(fit)
co <- data.frame(group='HIGH',HR=sm$conf.int[,'exp(coef)'],
                inverse_HR=sm$conf.int[,'exp(-coef)'],
                lower=sm$conf.int[,'lower .95'],upper=sm$conf.int[,'upper .95'],
                p=sm$coefficients[,'Pr(>|z|)'])
write.csv(co,file.path(args[2],'cox-60.csv'),row.names=FALSE)
km <- survfit(Surv(time_native,event_native)~group,data=d)
ss <- summary(km,times=c(0,5,10,15,20,30,40,50,60),extend=TRUE)
write.csv(data.frame(group=sub('^group=','',ss$strata),time=ss$time,
                    risk=ss$n.risk,cumulative_events=ave(ss$n.event,ss$strata,FUN=cumsum),
                    survival=ss$surv,lower=ss$lower,upper=ss$upper),
          file.path(args[2],'KM-60.csv'),row.names=FALSE)
lr <- survdiff(Surv(time_native,event_native)~group,data=d)
write.csv(data.frame(first='HIGH',second='LOW',raw_p=pchisq(lr$chisq,1,lower.tail=FALSE),
                    BH_p=pchisq(lr$chisq,1,lower.tail=FALSE)),
          file.path(args[2],'logrank-60.csv'),row.names=FALSE)
write.csv(data.frame(n=nrow(d),events=sum(d$event_native),endpoint=endpoint,
                    median=cutoff,LOW=sum(d$group=='LOW'),HIGH=sum(d$group=='HIGH'),
                    concordance=sm$concordance[1]),
          file.path(args[2],'summary-60.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(args[2],'sessionInfo.txt'))
print(co)

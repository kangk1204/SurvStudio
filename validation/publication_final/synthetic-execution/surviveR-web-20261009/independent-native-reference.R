args <- commandArgs(trailingOnly=TRUE)
library(survival)
dir.create(args[2], recursive=TRUE, showWarnings=FALSE)
d <- read.csv(args[1], check.names=FALSE)
d$grade <- factor(d$grade, levels=c('A','B','C'))
stopifnot(nrow(d)==260, all(d$event_code2==d$event+1))
for (endpoint in c(60, 46, 15, max(d$time))) {
  dd <- d
  dd$event_native <- as.integer(dd$event_code2==2 & dd$time<=endpoint)
  dd$time_native <- pmin(dd$time, endpoint)
  label <- if (endpoint==max(d$time)) 'full' else as.character(endpoint)
  fit <- coxph(Surv(time_native,event_native)~grade, data=dd, ties='efron', x=TRUE)
  sm <- summary(fit)
  co <- data.frame(group=sub('^grade','',rownames(sm$coefficients)),
    HR=sm$conf.int[,'exp(coef)'], inverse_HR=sm$conf.int[,'exp(-coef)'],
    lower=sm$conf.int[,'lower .95'], upper=sm$conf.int[,'upper .95'], p=sm$coefficients[,'Pr(>|z|)'])
  write.csv(co,file.path(args[2],paste0('cox-',label,'.csv')),row.names=FALSE)
  km <- survfit(Surv(time_native,event_native)~grade, data=dd)
  ss <- summary(km, times=c(0,5,10,15,20,30,40,50,60), extend=TRUE)
  write.csv(data.frame(group=sub('^grade=','',ss$strata),time=ss$time,risk=ss$n.risk,
    survival=ss$surv,lower=ss$lower,upper=ss$upper),file.path(args[2],paste0('KM-',label,'.csv')),row.names=FALSE)
  rows <- list(); pp <- c()
  for(pair in list(c('A','B'),c('A','C'),c('B','C'))) {
    zz <- dd[dd$grade %in% pair,]; zz$grade <- droplevels(zz$grade)
    lr <- survdiff(Surv(time_native,event_native)~grade, data=zz)
    pp <- c(pp,pchisq(lr$chisq,1,lower.tail=FALSE))
    rows[[length(rows)+1]] <- data.frame(first=pair[1],second=pair[2],raw_p=tail(pp,1))
  }
  rr <- do.call(rbind,rows);rr$BH_p <- p.adjust(pp,'BH')
  write.csv(rr,file.path(args[2],paste0('logrank-',label,'.csv')),row.names=FALSE)
  overall <- survdiff(Surv(time_native,event_native)~grade,data=dd)
  write.csv(data.frame(n=nrow(dd),events=sum(dd$event_native),concordance=sm$concordance[1],
    overall_logrank_p=pchisq(overall$chisq,2,lower.tail=FALSE)),
    file.path(args[2],paste0('summary-',label,'.csv')),row.names=FALSE)
}
capture.output(sessionInfo(),file=file.path(args[2],'sessionInfo.txt'))

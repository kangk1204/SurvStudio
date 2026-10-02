args <- commandArgs(trailingOnly=TRUE)
out <- args[1]
if (length(args)>1) .libPaths(c(args[2],.libPaths()))
library(survival)
spec <- read.csv(file.path(out,"cases.csv"),stringsAsFactors=FALSE)
states <- list()
for (i in seq_len(nrow(spec))) {
  name <- spec$name[i]
  d <- read.csv(file.path(out,paste0(name,".csv")))
  x <- as.matrix(d[,setdiff(names(d),c("time","event")),drop=FALSE])
  messages <- character()
  fit <- withCallingHandlers(
    coxph(Surv(time,event)~x,data=d,ties="efron",x=TRUE,y=TRUE,
          control=coxph.control(eps=1e-9,toler.chol=1e-14,iter.max=50)),
    warning=function(w) {messages <<- c(messages,conditionMessage(w)); invokeRestart("muffleWarning")})
  write.csv(data.frame(coefficient=coef(fit)),file.path(out,paste0(name,"-r-coefficients.csv")),row.names=FALSE)
  write.csv(vcov(fit),file.path(out,paste0(name,"-r-covariance.csv")),row.names=FALSE)
  write.csv(data.frame(lp=as.vector(x %*% coef(fit))),file.path(out,paste0(name,"-r-lp.csv")),row.names=FALSE)
  states[[i]] <- data.frame(name=name,loglik=fit$loglik[2],
                           separation_warning=any(grepl("may be infinite",messages)),
                           nonconvergence_warning=any(grepl("did not converge",messages)),
                           warnings=paste(messages,collapse=" | "))
}
write.csv(do.call(rbind,states),file.path(out,"r-fit-states.csv"),row.names=FALSE)
# Independent HC3 calculation. F(q,n-k) is an offline approximation, not an exact robust test.
d <- read.csv(file.path(out,"hc3.csv"))
x <- as.matrix(d[,setdiff(names(d),"y"),drop=FALSE])
y <- d$y
bread <- solve(crossprod(x))
beta <- as.vector(bread %*% crossprod(x,y))
res <- y-as.vector(x %*% beta)
h <- rowSums((x %*% bread)*x)
covariance <- bread %*% crossprod(x,x*(res/(1-h))^2) %*% bread
select <- 2:ncol(x)
w <- as.numeric(t(beta[select]) %*% solve(covariance[select,select],beta[select]))
q <- length(select)
write.csv(data.frame(statistic=w,chi_square_p=pchisq(w,q,lower.tail=FALSE),
                     approximate_F_p=pf(w/q,q,nrow(x)-ncol(x),lower.tail=FALSE)),
          file.path(out,"r-hc3.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))

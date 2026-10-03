# Independent public-case oracle. Read raw rows, prespecified settings and shared
# bootstrap row indices only; never read a Python coefficient, prediction or metric.
args <- commandArgs(TRUE)
out <- args[1]
if (length(args)>1) .libPaths(c(args[2],.libPaths()))
library(survival)
library(Hmisc)
options(warn=2) # Numerical reference warnings must not silently certify a fit.
settings <- read.csv(file.path(out,"reference-settings.csv"),stringsAsFactors=FALSE)
development.raw <- read.csv(file.path(out,"rotterdam.csv"),stringsAsFactors=FALSE)
external.raw <- read.csv(file.path(out,"gbsg-sealed.csv"),stringsAsFactors=FALSE)
draws <- as.matrix(read.csv(file.path(out,"bootstrap-rows.csv"),check.names=FALSE))+1L
prepare <- function(d,development,endpoint,limit) {
  if (development) d <- d[d$nodes>0,,drop=FALSE]
  if (any(d$er<0|d$pgr<0,na.rm=TRUE)) stop("Negative receptor measurement")
  d$logER <- log1p(d$er);d$logPGR <- log1p(d$pgr)
  if (!development) d$size <- ifelse(is.na(d$size),NA,ifelse(d$size<=20,"<=20",ifelse(d$size<=50,"20-50",">50")))
  d$size <- unname(c("<=20"="0","20-50"="1",">50"="2")[as.character(d$size)])
  if (development) {
    ignore <- d$recur==0 & d$death==1 & d$rtime<d$dtime
    if (endpoint=="conservative_RFS") {
      d$event <- ifelse(d$recur==1|ignore,d$recur,d$death)
      d$time <- ifelse(d$recur==1|ignore,d$rtime,d$dtime)
    } else {
      d$event <- pmax(d$recur,d$death)
      d$time <- ifelse(d$recur==1,d$rtime,d$dtime)
    }
  } else {d$event<-d$status;d$time<-d$rfstime}
  d$event[d$time>limit]<-0L;d$time<-pmin(d$time,limit)
  d
}
designs <- function(training,external,basis) {
  numeric <- c("age","meno","nodes","logER","hormon")
  levels <- sort(unique(training$size[!is.na(training$size)]))
  x <- matrix(numeric(0),nrow(training),0)
  z <- matrix(numeric(0),nrow(external),0)
  specification <- data.frame(quantity=character(),value=numeric())
  append.spec <- function(name,value) {
    specification <<- rbind(specification,data.frame(quantity=name,value=value))
  }
  for (level in levels[-1]) {
    x <- cbind(x,as.numeric(!is.na(training$size)&training$size==level))
    z <- cbind(z,as.numeric(!is.na(external$size)&external$size==level))
    colnames(x)[ncol(x)] <- colnames(z)[ncol(z)] <- paste0("size_",level)
  }
  if (anyNA(training$size)) {
    x <- cbind(x,size__missing=as.numeric(is.na(training$size)))
    z <- cbind(z,size__missing=as.numeric(is.na(external$size)))
  }
  for (name in numeric) {
    median <- median(training[[name]],na.rm=TRUE)
    if (!is.finite(median)) median<-0
    a <- training[[name]];b <- external[[name]]
    a[is.na(a)]<-median;b[is.na(b)]<-median
    append.spec(paste0(name,"_median"),median)
    if (basis=="restricted_cubic_spline" && length(unique(na.omit(training[[name]])))>2) {
      knots <- quantile(a,c(.05,.275,.5,.725,.95),type=7,names=FALSE)
      if (any(diff(knots)<=0)) stop("Duplicate prespecified knots")
      a <- rcspline.eval(a,knots=knots,inclx=TRUE,norm=2)
      b <- rcspline.eval(b,knots=knots,inclx=TRUE,norm=2)
      means <- c(0,colMeans(a[,-1,drop=FALSE]))
      scales <- c(1,sqrt(colMeans(sweep(a[,-1,drop=FALSE],2,means[-1])^2)))
      for (j in 1:5) append.spec(paste0(name,"_knot",j),knots[j])
      for (j in 1:4) {
        append.spec(paste0(name,"_mean",j),means[j])
        append.spec(paste0(name,"_scale",j),scales[j])
      }
      a <- sweep(sweep(a,2,means),2,scales,"/")
      b <- sweep(sweep(b,2,means),2,scales,"/")
      colnames(a) <- colnames(b) <- c(name,paste0(name,"__rcs",1:3))
    } else {
      a <- matrix(a,ncol=1,dimnames=list(NULL,name))
      b <- matrix(b,ncol=1,dimnames=list(NULL,name))
    }
    x <- cbind(x,a);z <- cbind(z,b)
  }
  median <- median(training$logPGR,na.rm=TRUE)
  append.spec("logPGR_median",median)
  a<-training$logPGR;b<-external$logPGR
  a[is.na(a)]<-median;b[is.na(b)]<-median
  list(clinical=x,external.clinical=z,full=cbind(x,logPGR=a),external.full=cbind(z,logPGR=b),specification=specification)
}
fit.model <- function(d,x) {
  input <- data.frame(time=d$time,event=d$event,x,check.names=FALSE)
  coxph(Surv(time,event)~.,data=input,ties="efron",x=TRUE,y=TRUE,
        control=coxph.control(eps=1e-12,toler.chol=1e-14,iter.max=100))
}
concordance.value <- function(d,lp) {
  if (!any(d$event==1)) return(NA_real_)
  concordance(Surv(time,event)~lp,data=d,reverse=TRUE,timewt="n")$concordance
}
km.risk <- function(d,horizon) {
  if (!nrow(d)) return(NA_real_)
  s <- survfit(Surv(time,event)~1,data=d)
  position <- findInterval(horizon,s$time)
  1-if (position==0) 1 else s$surv[position]
}
for (i in seq_len(nrow(settings))) {
  cfg<-settings[i,];name<-cfg$name
  dev<-prepare(development.raw,TRUE,cfg$endpoint,cfg$limit)
  ext<-prepare(external.raw,FALSE,cfg$endpoint,cfg$limit)
  if (ncol(draws)!=nrow(ext)) stop("Bootstrap stream row count changed")
  write.csv(dev[,c("time","event","size","logER","logPGR")],file.path(out,paste0(name,"-r-development.csv")),row.names=FALSE)
  write.csv(ext[,c("time","event","size","logER","logPGR")],file.path(out,paste0(name,"-r-external.csv")),row.names=FALSE)
  ds<-designs(dev,ext,cfg$basis)
  write.csv(ds$specification,file.path(out,paste0(name,"-r-transform.csv")),row.names=FALSE)
  write.csv(ds$full,file.path(out,paste0(name,"-r-development-design.csv")),row.names=FALSE)
  write.csv(ds$external.full,file.path(out,paste0(name,"-r-external-design.csv")),row.names=FALSE)
  predictions<-list();models<-list();metrics<-list();calibration<-list();bootstrap<-list()
  for (kind in c("clinical_only","clinical_plus_PGR")) {
    x<-if(kind=="clinical_only") ds$clinical else ds$full
    z<-if(kind=="clinical_only") ds$external.clinical else ds$external.full
    fit<-fit.model(dev,x);beta<-coef(fit)
    if (any(!is.finite(beta))) stop("Independent Cox fit is not estimable")
    lp<-as.vector(x%*%beta);external.lp<-as.vector(z%*%beta)
    centre<-mean(lp)
    sf<-survfit(fit,newdata=as.data.frame(as.list(colMeans(x))))
    event.rows<-sf$n.event>0
    baseline<-data.frame(time=sf$time[event.rows],log_cumulative_hazard=log(sf$cumhaz[event.rows]))
    write.csv(data.frame(term=names(beta),coefficient=as.vector(beta)),file.path(out,paste0(name,"-",kind,"-r-coefficients.csv")),row.names=FALSE)
    write.csv(baseline,file.path(out,paste0(name,"-",kind,"-r-baseline.csv")),row.names=FALSE)
    boundaries<-quantile(lp,c(.2,.4,.6,.8),type=7,names=FALSE)
    write.csv(data.frame(boundary=boundaries),file.path(out,paste0(name,"-",kind,"-r-groups.csv")),row.names=FALSE)
    write.csv(data.frame(lp=lp),file.path(out,paste0(name,"-",kind,"-r-development-lp.csv")),row.names=FALSE)
    c.value<-concordance.value(ext,external.lp)
    slope<-coef(coxph(Surv(time,event)~external.lp,data=ext,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14,iter.max=100)))[1]
    for (horizon in c(cfg$primary,cfg$secondary1,cfg$secondary2)) {
      position<-findInterval(horizon,baseline$time)
      hazard<-if(position==0) 0 else exp(baseline$log_cumulative_hazard[position])
      survival<-exp(-hazard*exp(external.lp-centre))
      predictions[[length(predictions)+1]]<-data.frame(model=kind,horizon=horizon,row=seq_len(nrow(ext))-1L,lp=external.lp,survival=survival)
      metrics[[length(metrics)+1]]<-data.frame(model=kind,horizon=horizon,c_index=c.value,calibration_slope=as.numeric(slope),expected_risk=mean(1-survival),lp_center=centre,loglik=fit$loglik[2])
      group<-findInterval(external.lp,boundaries)
      for (g in 0:4) {
        rows<-group==g
        calibration[[length(calibration)+1]]<-data.frame(model=kind,horizon=horizon,group=g,n=sum(rows),predicted_risk=if(any(rows)) mean(1-survival[rows]) else NA_real_,observed_KM_risk=km.risk(ext[rows,,drop=FALSE],horizon))
      }
    }
    bootstrap[[kind]]<-vapply(seq_len(nrow(draws)),function(b) concordance.value(ext[draws[b,],,drop=FALSE],external.lp[draws[b,]]),numeric(1))
  }
  write.csv(do.call(rbind,predictions),file.path(out,paste0(name,"-r-predictions.csv")),row.names=FALSE)
  write.csv(do.call(rbind,metrics),file.path(out,paste0(name,"-r-metrics.csv")),row.names=FALSE)
  write.csv(do.call(rbind,calibration),file.path(out,paste0(name,"-r-calibration.csv")),row.names=FALSE)
  write.csv(data.frame(clinical_only=bootstrap$clinical_only,clinical_plus_PGR=bootstrap$clinical_plus_PGR,delta=bootstrap$clinical_plus_PGR-bootstrap$clinical_only),file.path(out,paste0(name,"-r-bootstrap.csv")),row.names=FALSE)
}
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))

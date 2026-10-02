# Explicit R survival workflow: event recoding, nested training, frozen prediction,
# HC3 misspecification flag and manual input/endpoint checks. These checks are script
# steps; they are not claimed to be automatic defaults of survival::coxph.
args <- commandArgs(trailingOnly=TRUE)
out <- args[1]
if(length(args)>1) .libPaths(c(args[2],.libPaths()))
suppressPackageStartupMessages({library(survival);library(jsonlite)})
a <- read.csv(file.path(out,"A.csv"));b <- read.csv(file.path(out,"B.csv"))
d <- read.csv(file.path(out,"misspecified.csv"));duplicate <- read.csv(file.path(out,"duplicate.csv"))
recipe <- fromJSON(file.path(out,"recipe.json"),simplifyVector=FALSE)
rows <- list()
emit <- function(name,value) rows[[length(rows)+1]] <<- data.frame(quantity=name,value=as.numeric(value))
encode <- function(data,encoder) {
  columns <- list()
  for(name in encoder$categorical_features) {
    mapping <- encoder$categorical_mappings[[name]]
    for(level in mapping$retained_levels) columns[[mapping$level_columns[[level]]]] <- as.numeric(data[[name]]==level)
  }
  for(name in encoder$numeric_features) {
    x <- data[[name]];x[is.na(x)] <- encoder$numeric_impute_values[[name]];columns[[name]] <- x
  }
  as.matrix(as.data.frame(columns)[,unlist(encoder$feature_names),drop=FALSE])
}
encoder <- recipe$clinical$encoder$base_encoder
clinical_a <- encode(a,encoder);clinical_b <- encode(b,encoder)
event <- as.numeric(a$event_code2==2)
emit("recoded_events",sum(event))
groups <- factor(a$grade,levels=c("A","B","C"))
km <- survfit(Surv(a$time,event)~groups,conf.type="log-log")
at <- summary(km,times=c(5,10,15),extend=TRUE)
for(i in seq_along(at$time)) emit(paste0(sub("groups=","",at$strata[i])," @ ",at$time[i]," survival"),at$surv[i])
clinical_fit <- coxph(Surv(a$time,event)~clinical_a+a$m0,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14))
for(i in seq_along(coef(clinical_fit))) emit(paste0("adjusted_coefficient_",i),coef(clinical_fit)[i])
emit("adjusted_loglik",clinical_fit$loglik[2])

# Nested selection uses only the fixed inner training rows, never their held-out outcomes.
inside <- a$inner_train==1;z <- clinical_a[inside,,drop=FALSE];m <- as.matrix(a[inside,paste0("m",0:29)])
null <- coxph(Surv(a$time[inside],event[inside])~z,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14))
scores <- vapply(seq_len(ncol(m)),function(i) {
  x <- m[,i]
  coxph(Surv(a$time[inside],event[inside])~z+x,init=c(coef(null),0),iter.max=0,ties="efron")$score
},numeric(1))
selected <- which.max(scores);x <- m[,selected]
nested <- coxph(Surv(a$time[inside],event[inside])~z+x,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14))
nested_eta <- as.vector(cbind(clinical_a[!inside,,drop=FALSE],a[!inside,paste0("m",selected-1)]) %*% coef(nested))
emit("inner_selected_marker",selected-1)
emit("inner_heldout_C",concordance(Surv(a$time[!inside],event[!inside])~nested_eta,reverse=TRUE,timewt="n")$concordance)

# Frozen coefficients are applied once; there is no external refit or basis selection.
all_b <- cbind(clinical_b,as.matrix(b[,unlist(recipe$markers),drop=FALSE]))
colnames(all_b) <- c(colnames(clinical_b),unlist(recipe$markers))
eta <- as.vector(all_b[,unlist(recipe$model$terms),drop=FALSE] %*% unlist(recipe$model$coefficients))
emit("locked_external_C",concordance(Surv(b$time,b$event)~eta,reverse=TRUE,timewt="n")$concordance)
cal <- coxph(Surv(b$time,b$event)~eta,ties="efron",control=coxph.control(eps=1e-12,toler.chol=1e-14))
emit("locked_external_calibration",coef(cal)[1])
write.csv(data.frame(linear_predictor=eta),file.path(out,"r_locked_prediction.csv"),row.names=FALSE)

# Required residual diagnostics, implemented explicitly in this R workflow.
hc3 <- function(y,design,start) {
  inverse <- solve(crossprod(design));beta <- as.vector(inverse %*% crossprod(design,y))
  res <- y-as.vector(design %*% beta);h <- rowSums((design %*% inverse)*design)
  cov <- inverse %*% crossprod(design,design*(res/(1-h))^2) %*% inverse
  selected <- start:ncol(design)
  wald <- as.numeric(t(beta[selected]) %*% solve(cov[selected,selected],beta[selected]))
  c(statistic=wald,p_value=pchisq(wald,length(selected),lower.tail=FALSE))
}
base <- cbind(1,d$z);inverse <- solve(crossprod(base));hat <- rowSums((base %*% inverse)*base)
p_values <- c()
for(i in 0:29) {
  raw <- d[[paste0("m",i)]];residual <- raw-as.vector(base %*% inverse %*% crossprod(base,raw))
  mean_test <- hc3(residual,cbind(base,d$z^2,d$z^3),3)
  variance_test <- hc3(residual^2/(1-hat),base,2)
  emit(paste0("diagnostic_m",i,"_mean"),mean_test["statistic"])
  emit(paste0("diagnostic_m",i,"_variance"),variance_test["statistic"])
  p_values <- c(p_values,mean_test["p_value"],variance_test["p_value"])
}
emit("misspecified_withheld_by_explicit_rule",as.integer(any(p.adjust(p_values,"holm")<=.01)))
emit("duplicate_ID_rows",sum(duplicated(duplicate$patient_id)))
check_roles <- function(time,event,markers) {
  if(anyDuplicated(c(time,event,markers))) stop("Outcome and marker roles overlap")
  TRUE
}
role_rejected <- tryCatch({check_roles("time","event","event");FALSE},error=function(e) TRUE)
emit("outcome_role_rejected",as.integer(role_rejected))
check_endpoint <- function(time,event) {
  time_family <- sub("_.*$","",time);event_family <- sub("_.*$","",event)
  if(time_family!=event_family) stop("Declared endpoints differ")
  TRUE
}
endpoint_rejected <- tryCatch({check_endpoint("rfs_months","dmfs_event");FALSE},error=function(e) TRUE)
emit("manual_endpoint_mismatch_rejected",as.integer(endpoint_rejected))
write.csv(do.call(rbind,rows),file.path(out,"r_values.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"R_session.txt"))

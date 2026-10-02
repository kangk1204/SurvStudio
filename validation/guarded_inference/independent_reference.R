args <- commandArgs(trailingOnly=TRUE)
out <- args[1]
if (length(args) > 1) .libPaths(c(args[2], .libPaths()))
library(survival)
library(Hmisc)
train <- read.csv(file.path(out, "training.csv"))
external <- read.csv(file.path(out, "external.csv"))
knots <- as.numeric(read.csv(file.path(out, "knots.csv"))$knot)
means <- as.numeric(read.csv(file.path(out, "scaling.csv"))$mean)
scales <- as.numeric(read.csv(file.path(out, "scaling.csv"))$scale)
median_z <- median(train$z, na.rm=TRUE)
train$z[is.na(train$z)] <- median_z
external$z[is.na(external$z)] <- median_z
stopifnot(max(abs(knots - quantile(train$z, c(.05,.275,.5,.725,.95), type=7))) < 1e-12)
transform <- function(d) {
  r <- Hmisc::rcspline.eval(d$z, knots=knots, inclx=TRUE, norm=2)
  r <- sweep(sweep(r, 2, means, "-"), 2, scales, "/")
  cbind(grade_B=as.numeric(d$grade=="B"), grade_C=as.numeric(d$grade=="C"), r, binary=d$binary)
}
x <- transform(train)
y <- transform(external)
write.csv(x, file.path(out,"r_training_basis.csv"), row.names=FALSE)
write.csv(y, file.path(out,"r_external_basis.csv"), row.names=FALSE)
fit <- coxph(Surv(time,event) ~ x + m0 + m1, data=train, ties="efron", x=TRUE, y=TRUE,
             control=coxph.control(eps=1e-12, toler.chol=1e-14, iter.max=100))
coef <- coefficients(fit)
eta <- as.vector(cbind(y, external$m0, external$m1) %*% coef)
write.csv(data.frame(coefficient=coef), file.path(out,"r_coefficients.csv"), row.names=FALSE)
write.csv(data.frame(linear_predictor=eta), file.path(out,"r_external_prediction.csv"), row.names=FALSE)
c_value <- concordance(Surv(external$time,external$event) ~ eta, reverse=TRUE, timewt="n")$concordance
cal <- coxph(Surv(time,event) ~ eta, data=external, ties="efron", control=coxph.control(eps=1e-12,toler.chol=1e-14))
write.csv(data.frame(metric=c("C","calibration_slope","loglik"), value=c(c_value,coef(cal)[1],fit$loglik[2])),
          file.path(out,"r_metrics.csv"), row.names=FALSE)
capture.output(sessionInfo(), file=file.path(out,"R_session.txt"))
clinical <- coxph(Surv(time,event) ~ x, data=train, ties="efron", x=TRUE, y=TRUE,
                  control=coxph.control(eps=1e-12, toler.chol=1e-14, iter.max=100))
schoenfeld <- residuals(clinical, type="schoenfeld")
g <- log(as.numeric(rownames(schoenfeld)))
g <- g - mean(g)
score <- as.vector(crossprod(g, schoenfeld))
ph <- as.numeric(t(score) %*% vcov(clinical) %*% score * nrow(schoenfeld) / sum(g*g))
write.csv(data.frame(statistic=ph, p_value=pchisq(ph,ncol(x),lower.tail=FALSE)),
          file.path(out,"r_classic_ph.csv"), row.names=FALSE)
linear_x <- cbind(grade_B=as.numeric(train$grade=="B"),grade_C=as.numeric(train$grade=="C"),z=train$z,binary=train$binary)
linear_fit <- coxph(Surv(time,event) ~ linear_x, data=train, ties="efron",
                     control=coxph.control(eps=1e-12,toler.chol=1e-14,iter.max=100))
lr <- 2*(clinical$loglik[2]-linear_fit$loglik[2])
write.csv(data.frame(coefficient=coef(linear_fit)),file.path(out,"r_linear_coefficients.csv"),row.names=FALSE)
write.csv(data.frame(statistic=lr,p_value=pchisq(lr,ncol(x)-ncol(linear_x),lower.tail=FALSE)),
          file.path(out,"r_functional_LR.csv"),row.names=FALSE)
# Independent HC3 sandwich and joint Wald calculation, without Python/statsmodels.
d <- cbind(1,train$z,train$z^2)
bread <- solve(crossprod(d))
beta <- as.vector(bread %*% crossprod(d,train$m0))
res <- train$m0-as.vector(d %*% beta)
hat <- rowSums((d %*% bread)*d)
covariance <- bread %*% crossprod(d,d*(res/(1-hat))^2) %*% bread
hc3 <- as.numeric(t(beta[2:3]) %*% solve(covariance[2:3,2:3],beta[2:3]))
write.csv(data.frame(statistic=hc3,p_value=pchisq(hc3,2,lower.tail=FALSE)),file.path(out,"r_HC3.csv"),row.names=FALSE)
